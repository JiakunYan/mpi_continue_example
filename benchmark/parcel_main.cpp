#include <iostream>
#include <vector>
#include <thread>
#include <cassert>
#include <atomic>
#include <chrono>
#include <deque>
#include "mpi.h"
#include "common.hpp"

#ifdef OMPI_MPI_H
#include "mpi-ext.h"
#define MPIX_CONT_IMMEDIATE 0
#endif

#define MPI_SAFECALL(x) {      \
        int ret = (x); \
        if (ret != MPI_SUCCESS) { \
            fprintf(stderr, "%s:%d:%s: " \
            "MPI call failed!\n", __FILE__, __LINE__, #x); \
            exit(1); \
        }\
    }

namespace detail {
    typedef int (cb_fn_t)(int error_code, void *user_data);
    struct pending_parcel_t {
        std::vector<MPI_Request> requests;
        cb_fn_t *cb;
        void *context;
    };
    thread_local std::deque<pending_parcel_t*> pending_parcels;
    void push(std::vector<MPI_Request>&& requests, cb_fn_t *cb, void *context) {
        auto *parcel = new pending_parcel_t;
        parcel->requests = std::move(requests);
        parcel->cb = cb;
        parcel->context = context;
        pending_parcels.push_back(parcel);
    }
    bool progress() {
        if (pending_parcels.empty())
            return false;
        auto *parcel = pending_parcels.front();
        pending_parcels.pop_front();
        int flag;
        MPI_SAFECALL(MPI_Testall(parcel->requests.size(), parcel->requests.data(), &flag, MPI_STATUS_IGNORE));
        if (flag) {
            parcel->cb(MPI_SUCCESS, parcel->context);
            return true;
        } else {
            pending_parcels.push_back(parcel);
            return false;
        }
    }
}

struct config_t {
    enum class comp_type_t {
        REQUEST,
        CONTINUE,
    } comp_type = comp_type_t::CONTINUE;
    enum class progress_type_t {
        SHARED,
    } progress_type = progress_type_t::SHARED;
    int use_cont_imm = true;
    int use_thread_single = true;
} g_config;
int g_rank, g_nranks;
bench::context_t g_context;
int g_max_tag;
alignas(64) std::atomic<int> g_next_tag(0);
const int MAX_HEADER_SIZE = 1024;

thread_local MPI_Request tls_cont_req = MPI_REQUEST_NULL;
thread_local MPI_Request tls_request_header = MPI_REQUEST_NULL;
thread_local std::vector<int> tls_recv_buffer(MAX_HEADER_SIZE);

std::vector<int> *encode_header(int tag, bench::parcel_t *parcel) {
    auto *header = new std::vector<int>;
    header->push_back(tag);
    header->push_back(static_cast<int>(parcel->msgs.size()));
    for (const auto& chunk : parcel->msgs) {
        header->push_back(static_cast<int>(chunk.size()));
    }
    return header;
}

void decode_header(const std::vector<int>& header, int *tag, bench::parcel_t *parcel) {
    *tag = header[0];
    parcel->msgs.resize(header[1]);
    for (int i = 0; i < parcel->msgs.size(); ++i) {
        parcel->msgs[i].resize(header[2 + i]);
    }
}

int sender_callback(int error_code, void *user_data) {
    auto *parcel = static_cast<bench::parcel_t*>(user_data);
    auto *header = static_cast<std::vector<int>*>(parcel->local_context);
    delete header;
    delete parcel;
    g_context.signal_send_comp();
    return MPI_SUCCESS;
}

int receiver_callback(int error_code, void *user_data) {
    auto *parcel = static_cast<bench::parcel_t*>(user_data);
    g_context.signal_recv_comp(parcel);
    return MPI_SUCCESS;
}

void try_send_parcel() {
    auto *parcel = g_context.get_parcel();
    if (parcel) {
        // send the parcel
        int tag = g_next_tag++ % (g_max_tag - 1) + 1;
        auto header = encode_header(tag, parcel);
        parcel->local_context = header;
        std::vector<MPI_Request> requests;
        MPI_Request request;
        // header messages always use tag 0.
        MPI_SAFECALL(MPI_Isend(header->data(), header->size(), MPI_INT,
                               parcel->peer_rank, 0, MPI_COMM_WORLD, &request));
        requests.push_back(request);
        for (auto & chunk : parcel->msgs) {
            MPI_SAFECALL(MPI_Isend(chunk.data(), chunk.size(), MPI_CHAR,
                                   parcel->peer_rank, tag, MPI_COMM_WORLD, &request));
            requests.push_back(request);
        }
        if (g_config.comp_type == config_t::comp_type_t::REQUEST) {
            detail::push(std::move(requests), sender_callback, parcel);
        } else {
            int cont_flag = 0;
            if (g_config.use_cont_imm)
                cont_flag = MPIX_CONT_IMMEDIATE;
            MPIX_Continueall(static_cast<int>(requests.size()), requests.data(), sender_callback,
                             parcel, cont_flag, MPI_STATUSES_IGNORE, tls_cont_req);
        }
    }
}

void try_receive_parcel() {
    if (tls_request_header == MPI_REQUEST_NULL)
        MPI_SAFECALL(MPI_Irecv(tls_recv_buffer.data(), tls_recv_buffer.size(), MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &tls_request_header));

    MPI_Status status;
    int flag = 0;
    MPI_SAFECALL(MPI_Test(&tls_request_header, &flag, &status));
    if (flag) {
        // got a new header message
        int tag;
        auto *parcel = new bench::parcel_t;
        decode_header(tls_recv_buffer, &tag, parcel);
        assert(tag > 0);
        // post a new receive for header
        MPI_SAFECALL(MPI_Irecv(tls_recv_buffer.data(), tls_recv_buffer.size(), MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &tls_request_header));
        // post the follow-up receives
        parcel->peer_rank = status.MPI_SOURCE;
        if (!parcel->msgs.empty()) {
            std::vector<MPI_Request> requests;
            for (auto &chunk: parcel->msgs) {
                MPI_Request request;
                MPI_SAFECALL(MPI_Irecv(chunk.data(), chunk.size(), MPI_CHAR,
                                       status.MPI_SOURCE, tag, MPI_COMM_WORLD, &request));
                requests.push_back(request);
            }
            if (g_config.comp_type == config_t::comp_type_t::REQUEST) {
                detail::push(std::move(requests), receiver_callback, parcel);
            } else {
                int cont_flag = 0;
                if (g_config.use_cont_imm)
                    cont_flag = MPIX_CONT_IMMEDIATE;
                MPIX_Continueall(static_cast<int>(requests.size()), requests.data(), receiver_callback,
                                 parcel, cont_flag, MPI_STATUSES_IGNORE, tls_cont_req);
            }
        } else {
            receiver_callback(MPI_SUCCESS, parcel);
        }
    }
}

void cancel_receive() {
    if (tls_request_header != MPI_REQUEST_NULL) {
        MPI_Cancel(&tls_request_header);
    }
}

void do_progress() {
    if (g_config.comp_type == config_t::comp_type_t::REQUEST) {
        detail::progress();
    } else {
        int flag;
        MPI_SAFECALL(MPI_Test(&tls_cont_req, &flag, MPI_STATUS_IGNORE));
        if (flag) {
            MPI_Start(&tls_cont_req);
        }
    }
}

void worker_thread_fn(int thread_id) {
    if (g_config.progress_type == config_t::progress_type_t::SHARED) {
        MPI_SAFECALL(MPIX_Continue_init(0, 0, MPI_INFO_NULL, &tls_cont_req));
        MPI_SAFECALL(MPI_Start(&tls_cont_req));
    }
    while (!g_context.is_done()) {
        try_receive_parcel();
        try_send_parcel();
        if (g_config.progress_type == config_t::progress_type_t::SHARED)
            do_progress();
    }
    cancel_receive();
    if (g_config.progress_type == config_t::progress_type_t::SHARED) {
        MPI_SAFECALL(MPI_Request_free(&tls_cont_req));
    }
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
    args_parser.add("thread-single", required_argument, (int*)&g_config.use_thread_single);
    // initialize the benchmark context
    g_context.parse_args(argc, argv, args_parser);
    int nthreads = g_context.get_nthreads();
    // initialize MPI
    if (nthreads == 1 && g_config.use_thread_single) {
        MPI_SAFECALL(MPI_Init(nullptr, nullptr));
    } else {
        int provided;
        MPI_SAFECALL(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided));
        assert(provided == MPI_THREAD_MULTIPLE);
    }
    MPI_SAFECALL(MPI_Comm_size(MPI_COMM_WORLD, &g_nranks));
    MPI_SAFECALL(MPI_Comm_rank(MPI_COMM_WORLD, &g_rank));
    // get MPI max tag
    void* max_tag_p;
    int flag;
    MPI_SAFECALL(MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &max_tag_p, &flag));
    if (flag)
        g_max_tag = *(int*) max_tag_p;
    else
        g_max_tag = 32767;
    // Setup problem
    g_context.setup(g_rank, g_nranks);

    // initialize the continuation request
    MPI_SAFECALL(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    // spawn the worker threads
    std::vector<std::thread> worker_threads;
    for (int i = 1; i < nthreads; ++i) {
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

//    MPI_SAFECALL(MPI_Finalize());
    return 0;
}