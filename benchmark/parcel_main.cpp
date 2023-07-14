#include <iostream>
#include <vector>
#include <thread>
#include <cassert>
#include <atomic>
#include <chrono>
#include "mpi.h"
#include "common.hpp"

#define MPI_SAFECALL(x) {      \
        int ret = (x); \
        if (ret != MPI_SUCCESS) { \
            fprintf(stderr, "%s:%d:%s: " \
            "MPI call failed!\n", __FILE__, __LINE__, #x); \
            exit(1); \
        }\
    }

int g_rank, g_nranks;
MPI_Request g_cont_req;
bench::context_t g_context;
int g_max_tag;
alignas(64) std::atomic<int> g_next_tag(0);
alignas(64) std::atomic<bool> g_progress_flag(false);
const int MAX_HEADER_SIZE = 1024;

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
        MPIX_Continueall(static_cast<int>(requests.size()), requests.data(), sender_callback,
                         parcel, 0, MPI_STATUSES_IGNORE, g_cont_req);
    }
}

thread_local MPI_Request request_header = MPI_REQUEST_NULL;
thread_local std::vector<int> recv_buffer(MAX_HEADER_SIZE);

void try_receive_parcel() {
    if (request_header == MPI_REQUEST_NULL)
        MPI_SAFECALL(MPI_Irecv(recv_buffer.data(), recv_buffer.size(), MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &request_header));

    MPI_Status status;
    int flag = 0;
    MPI_SAFECALL(MPI_Test(&request_header, &flag, &status));
    if (flag) {
        // got a new header message
        int tag;
        auto *parcel = new bench::parcel_t;
        decode_header(recv_buffer, &tag, parcel);
        assert(tag > 0);
        // post a new receive for header
        MPI_SAFECALL(MPI_Irecv(recv_buffer.data(), recv_buffer.size(), MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &request_header));
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
            MPIX_Continueall(static_cast<int>(requests.size()), requests.data(), receiver_callback,
                             parcel, 0, MPI_STATUSES_IGNORE, g_cont_req);
        } else {
            receiver_callback(MPI_SUCCESS, parcel);
        }
    }
}

void cancel_receive() {
    if (request_header != MPI_REQUEST_NULL) {
        MPI_Cancel(&request_header);
    }
}

void do_progress() {
    int flag;
    MPI_SAFECALL(MPI_Test(&g_cont_req, &flag, MPI_STATUS_IGNORE));
    if (flag) {
        MPI_Start(&g_cont_req);
    }
}

void worker_thread_fn(int thread_id) {
    // The sender
    bool need_progress = g_context.get_nthreads() == 1;
    while (!g_context.is_done()) {
        try_receive_parcel();
        try_send_parcel();
        if (need_progress)
            do_progress();
    }
    cancel_receive();
}

void progress_thread_fn(int thread_id) {
    while (g_progress_flag) {
        do_progress();
    }
}

int main(int argc, char *argv[]) {
//    bool wait_for_dbg = true;
//    while (wait_for_dbg) continue;
    // initialize the benchmark context
    g_context.parse_args(argc, argv);
    int nthreads = g_context.get_nthreads();
    // initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    assert(provided == MPI_THREAD_MULTIPLE);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    // get MPI max tag
    void* max_tag_p;
    int flag;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &max_tag_p, &flag);
    if (flag)
        g_max_tag = *(int*) max_tag_p;
    else
        g_max_tag = 32767;
    // Setup problem
    g_context.setup(g_rank, g_nranks);

    // initialize the continuation request
    MPIX_Continue_init(0, 0, MPI_INFO_NULL, &g_cont_req);
    MPI_Start(&g_cont_req);
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (nthreads == 1) {
        worker_thread_fn(0);
    } else {
        // spawn the progress thread
        g_progress_flag = true;
        std::thread progress_thread(progress_thread_fn, nthreads - 1);
        // spawn the worker threads
        std::vector<std::thread> worker_threads;
        for (int i = 0; i < nthreads - 1; ++i) {
            worker_threads.emplace_back(worker_thread_fn, i);
        }

        // Wait for the worker threads to join
        for (int i = 0; i < nthreads - 1; ++i) {
            worker_threads[i].join();
        }
        // Wait for the progress threads to join
        g_progress_flag = false;
        progress_thread.join();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    g_context.report(total_time.count());

    MPI_Request_free(&g_cont_req);
    MPI_Finalize();
}