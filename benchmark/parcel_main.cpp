#include <iostream>
#include <vector>
#include <thread>
#include <cassert>
#include <atomic>
#include <chrono>
#include <deque>
#include "mpi.h"
#include "common.hpp"
#include "common_mpi.hpp"

#ifndef MPIX_STREAM_NULL
#define MPIX_Stream int
#define MPIX_STREAM_NULL 0
#define MPIX_Stream_create(...) MPI_ERR_UNKNOWN
#define MPIX_Stream_comm_create(...) MPI_ERR_UNKNOWN
#define MPIX_Stream_free(...) MPI_ERR_UNKNOWN
#define MPIX_Stream_progress(...) MPI_ERR_UNKNOWN
#endif

#define MPI_SAFECALL(x) {      \
        int ret = (x); \
        if (ret != MPI_SUCCESS) { \
            fprintf(stderr, "%s:%d:%s: " \
            "MPI call failed!\n", __FILE__, __LINE__, #x); \
            exit(1); \
        }\
    }

struct config_t {
    enum class comp_type_t {
        REQUEST,
        CONTINUE,
    } comp_type = comp_type_t::CONTINUE;
    enum class progress_type_t {
        SHARED,
    } progress_type = progress_type_t::SHARED;
    int nthreads = 0;
    int use_cont_imm = true;
    int use_cont_forget = true;
    int use_thread_single = true;
    int enable_stream = false;
    int nsteps = 0;
    int nvcis = 1;
};

config_t g_config;
int g_rank, g_nranks;
bench::context_t g_context;
const int MAX_HEADER_SIZE = 4096;
__thread MPI_Request header_req = MPI_REQUEST_NULL;
thread_local std::vector<int> recv_buffer(MAX_HEADER_SIZE);

struct alignas(64) device_t {
    MPIX_Stream stream = MPIX_STREAM_NULL;
    MPI_Comm comm;
    int max_tag;
    device_t() : max_tag(32767) {};
};
std::vector<device_t> g_devices;
__thread device_t *tls_device_p;;
__thread int tls_device_id;
__thread int tls_device_rank;
__thread int tls_device_nranks;
__thread int tls_header_tag;
__thread int tls_next_tag = 0;
__thread int pending_send = 0;
__thread int pending_recv = 0;
detail::comp_manager_base_t *g_comp_manager_p;

std::vector<int> *encode_header(int tag, bench::parcel_t *parcel) {
    auto *header = new std::vector<int>(3 + parcel->chunks.size() +
            (parcel->piggyback.size() + sizeof(int) - 1) / sizeof(int));
    int i = 0;
    (*header)[i++] = tag;
    (*header)[i++] = static_cast<int>(parcel->piggyback.size());
    (*header)[i++] = static_cast<int>(parcel->chunks.size());
    for (const auto& chunk : parcel->chunks) {
        (*header)[i++] = static_cast<int>(chunk.size());
    }
    if (parcel->piggyback.size())
        memcpy(&(*header)[i], parcel->piggyback.data(), parcel->piggyback.size());
    return header;
}

void decode_header(const std::vector<int>& header, int *tag, bench::parcel_t *parcel) {
    int i = 0;
    *tag = header[i++];
    parcel->piggyback.resize(header[i++]);
    parcel->chunks.resize(header[i++]);
    for (auto& chunk : parcel->chunks) {
        chunk.resize(header[i++]);
    }
    assert(header.size() >= 3 + parcel->chunks.size() +
                            (parcel->piggyback.size() + sizeof(int) - 1) / sizeof(int));
    if (parcel->piggyback.size())
        memcpy(parcel->piggyback.data(), &header[i], parcel->piggyback.size());
}

int sender_callback(int error_code, void *user_data) {
    auto *parcel = static_cast<bench::parcel_t*>(user_data);
    auto *header = static_cast<std::vector<int>*>(parcel->local_context);
    delete header;
    int step = *(int*) parcel->piggyback.data();
    if (step == 0) {
        delete parcel;
        bench::context_t::signal_send_comp();
    }
    return MPI_SUCCESS;
}

void send_parcel(bench::parcel_t *parcel) {
    // get the tag
    int tag = (tls_next_tag++ * tls_device_nranks + tls_device_rank) % (tls_device_p->max_tag - 1) + tls_device_nranks;
    auto header = encode_header(tag, parcel);
    parcel->local_context = header;
    std::vector<MPI_Request> requests;
    MPI_Request request;
    MPI_SAFECALL(MPI_Isend(header->data(), header->size(), MPI_INT,
                           parcel->peer_rank, tls_header_tag, tls_device_p->comm, &request));
    requests.push_back(request);
    for (auto & chunk : parcel->chunks) {
        MPI_SAFECALL(MPI_Isend(chunk.data(), chunk.size(), MPI_CHAR,
                               parcel->peer_rank, tag, tls_device_p->comm, &request));
        requests.push_back(request);
    }
    g_comp_manager_p->push(std::move(requests), sender_callback, parcel);
}

int receiver_callback(int error_code, void *user_data) {
    auto *parcel = static_cast<bench::parcel_t*>(user_data);
    int step = *(int*) parcel->piggyback.data();
    if (step == 0) {
        bench::context_t::signal_recv_comp(parcel);
    } else {
        --step;
        *(int*) parcel->piggyback.data() = step;
        send_parcel(parcel);
    }
    return MPI_SUCCESS;
}

void try_send_parcel() {
    auto *parcel = g_context.get_parcel();
    if (parcel) {
        assert(parcel->piggyback.size() > sizeof(int));
        *(int*) parcel->piggyback.data() = g_config.nsteps - 1;
        send_parcel(parcel);
    }
}

void try_receive_parcel() {
    MPI_Status status;
    int flag = 0;
    MPI_SAFECALL(MPI_Test(&header_req, &flag, &status));
    if (flag) {
        // got a new header message
        int tag;
        auto *parcel = new bench::parcel_t;
        decode_header(recv_buffer, &tag, parcel);
        assert(tag > 0);
        // post a new receive for header
        MPI_SAFECALL(MPI_Irecv(recv_buffer.data(), recv_buffer.size(), MPI_INT, MPI_ANY_SOURCE, tls_header_tag, tls_device_p->comm, &header_req));
        // post the follow-up receives
        parcel->peer_rank = status.MPI_SOURCE;
        if (!parcel->chunks.empty()) {
            std::vector<MPI_Request> requests;
            for (auto &chunk: parcel->chunks) {
                MPI_Request request;
                MPI_SAFECALL(MPI_Irecv(chunk.data(), chunk.size(), MPI_CHAR,
                                       status.MPI_SOURCE, tag, tls_device_p->comm, &request));
                requests.push_back(request);
            }
            g_comp_manager_p->push(std::move(requests), receiver_callback, parcel);
        } else {
            receiver_callback(MPI_SUCCESS, parcel);
        }
    }
}

void do_progress() {
    if (tls_device_p->stream != MPIX_STREAM_NULL)
        MPI_SAFECALL(MPIX_Stream_progress(tls_device_p->stream));
    g_comp_manager_p->progress();
}

void worker_thread_fn(int thread_id) {
    int nthreads_per_device = (g_config.nthreads + g_config.nvcis - 1) / g_config.nvcis;
    tls_device_id = thread_id / nthreads_per_device;
    tls_device_rank = thread_id - tls_device_id * nthreads_per_device;
    tls_device_nranks = nthreads_per_device;
    tls_device_p = &g_devices[tls_device_id];
    g_comp_manager_p->init_thread();
    tls_header_tag = 0;
    MPI_SAFECALL(MPI_Irecv(recv_buffer.data(), recv_buffer.size(), MPI_INT, MPI_ANY_SOURCE, tls_header_tag, tls_device_p->comm, &header_req));
    while (!g_context.is_done()) {
        try_receive_parcel();
        try_send_parcel();
        if (g_config.progress_type == config_t::progress_type_t::SHARED)
            do_progress();
    }
    if (header_req != MPI_REQUEST_NULL) {
        MPI_Cancel(&header_req);
    }
    g_comp_manager_p->free_thread();
}

int main(int argc, char *argv[]) {
//    bool wait_for_dbg = true;
//    while (wait_for_dbg) continue;
    bench::args_parser_t args_parser;
    args_parser.add("progress-type", required_argument, (int*)&g_config.progress_type,
                    {{"shared", (int)config_t::progress_type_t::SHARED},});
    args_parser.add("comp-type", required_argument, (int*)&g_config.comp_type,
                    {{"continue", (int)config_t::comp_type_t::CONTINUE},
                     {"request", (int)config_t::comp_type_t::REQUEST},});
    args_parser.add("cont-imm", required_argument, (int*)&g_config.use_cont_imm);
    args_parser.add("cont-forget", required_argument, (int*)&g_config.use_cont_forget);
    args_parser.add("thread-single", required_argument, (int*)&g_config.use_thread_single);
    args_parser.add("enable-stream", required_argument, (int*)&g_config.enable_stream);
    args_parser.add("nvcis", required_argument, (int*)&g_config.nvcis);
    // initialize the benchmark context
    g_context.parse_args(argc, argv, args_parser);
    g_config.nthreads = g_context.get_nthreads();
    g_config.nsteps = g_context.get_nsteps();
    assert(!g_config.enable_stream || g_config.nthreads == g_config.nvcis);
    if (g_config.enable_stream) {
        setenv("MPIR_CVAR_CH4_RESERVE_VCIS", std::to_string(g_config.nvcis).c_str(), true);
    } else {
        setenv("MPIR_CVAR_CH4_NUM_VCIS", std::to_string(g_config.nvcis).c_str(), true);
    }
    // initialize MPI
    if (g_config.nthreads == 1 && g_config.use_thread_single) {
        MPI_SAFECALL(MPI_Init(nullptr, nullptr));
    } else {
        int provided;
        MPI_SAFECALL(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided));
        assert(provided == MPI_THREAD_MULTIPLE);
    }
    MPI_SAFECALL(MPI_Comm_size(MPI_COMM_WORLD, &g_nranks));
    MPI_SAFECALL(MPI_Comm_rank(MPI_COMM_WORLD, &g_rank));
//    if (g_rank == 0) {
//        bool wait_for_dbg = true;
//        while (wait_for_dbg) continue;
//    }
    MPI_Barrier(MPI_COMM_WORLD);
    // Setup problem
    g_context.setup(g_rank, g_nranks);
    // Initialize the completion manager
    switch (g_config.comp_type) {
        case config_t::comp_type_t::REQUEST:
            g_comp_manager_p = new detail::comp_manager_request_t;
            break;
        case config_t::comp_type_t::CONTINUE:
#ifdef MPIX_CONT_DEFER_COMPLETE
            g_comp_manager_p = new detail::comp_manager_continue_t(
                    g_config.use_cont_imm, g_config.use_cont_forget);
#else
            fprintf(stderr, "Cannot use MPIX Continuation with the current MPI installation.\n");
#endif
            break;
    }
    // Initialize devices
    for (int i = 0; i < g_config.nvcis; ++i) {
        device_t device;
        if (g_config.enable_stream) {
            MPIX_Stream_create(MPI_INFO_NULL, &device.stream);
            MPIX_Stream_comm_create(MPI_COMM_WORLD, device.stream, &device.comm);
        } else {
            device.stream = MPIX_STREAM_NULL;
            MPI_Comm_dup(MPI_COMM_WORLD, &device.comm);
        }
        // get MPI max tag
        void* max_tag_p;
        int flag;
        MPI_SAFECALL(MPI_Comm_get_attr(device.comm, MPI_TAG_UB, &max_tag_p, &flag));
        if (flag)
            device.max_tag = *(int*) max_tag_p;
        g_devices.push_back(device);
    }

    MPI_SAFECALL(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    // spawn the worker threads
    std::vector<std::thread> worker_threads;
    for (int i = 1; i < g_config.nthreads; ++i) {
        worker_threads.emplace_back(worker_thread_fn, i);
    }
    worker_thread_fn(0);
    // Wait for the worker threads to join
    for (auto& worker : worker_threads) {
        worker.join();
    }

    MPI_SAFECALL(MPI_Barrier(MPI_COMM_WORLD));
    auto end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    g_context.report(total_time.count());

    // Finalize streams
    for (auto & device : g_devices) {
        MPI_Comm_free(&device.comm);
    if (device.stream != MPIX_STREAM_NULL)
            MPIX_Stream_free(&device.stream);
    }
    delete g_comp_manager_p;
    MPI_SAFECALL(MPI_Finalize());
    return 0;
}