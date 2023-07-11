#include <iostream>
#include <vector>
#include <thread>
#include <cassert>
#include <atomic>
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

struct config_t {
    int nthreads = 8;
} g_config;

int g_rank, g_nranks;
MPI_Request g_cont_req;
bench::context_t g_context;
int g_max_tag;
alignas(64) std::atomic<int> g_next_tag(0);
alignas(64) std::atomic<bool> g_progress_flag(false);

int sender_callback(int error_code, void *user_data) {
    auto *parcel = static_cast<bench::parcel_t*>(user_data);
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
        std::vector<MPI_Request> requests;
        MPI_Request request;
        // header messages always use tag 0.
        MPI_SAFECALL(MPI_Isend(&tag, 1, MPI_INT, parcel->peer_rank, 0, MPI_COMM_WORLD, &request));
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
thread_local int recv_buffer;

void try_receive_parcel() {
    if (request_header == MPI_REQUEST_NULL)
        MPI_SAFECALL(MPI_Irecv(&recv_buffer, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &request_header));

    MPI_Status status;
    int flag;
    MPI_SAFECALL(MPI_Test(&request_header, &flag, &status));
    if (flag) {
        // got a new header message
        int tag = recv_buffer;
        assert(tag > 0);
        // post a new receive for header
        MPI_SAFECALL(MPI_Irecv(&recv_buffer, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &request_header));
        // post the follow-up receives
        // TODO: Ideally we should also tranfer the chunk number/size through the header message.
        auto *parcel = new bench::parcel_t;
        parcel->peer_rank = status.MPI_SOURCE;
        if (g_context.nchunks > 0) {
            parcel->msgs.resize(g_context.nchunks);
            std::vector<MPI_Request> requests;
            for (auto &chunk: parcel->msgs) {
                MPI_Request request;
                chunk.resize(g_context.chunk_size);
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

void worker_thread_fn(int thread_id) {
    // The sender
    while (!g_context.is_done()) {
        try_receive_parcel();
        try_send_parcel();
    }
    cancel_receive();
}

void progress_thread_fn(int thread_id) {
    while (g_progress_flag) {
        int flag;
        MPI_SAFECALL(MPI_Test(&g_cont_req, &flag, MPI_STATUS_IGNORE));
        if (flag) {
            MPI_Start(&g_cont_req);
        }
    }
}

int main(int argc, char *argv[]) {
    // initialize the benchmark context
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
    // initialize MPI
    g_context.setup(g_rank, g_nranks, argc, argv);

    // initialize the continuation request
    MPIX_Continue_init(0, 0, MPI_INFO_NULL, &g_cont_req);
    MPI_Start(&g_cont_req);

    // spawn the progress thread
    g_progress_flag = true;
    std::thread progress_thread(progress_thread_fn, 0);
    // spawn the worker threads
    std::vector<std::thread> worker_threads;
    for (int i = 0; i < g_config.nthreads; ++i) {
        worker_threads.emplace_back(worker_thread_fn, i);
    }

    // Wait for the worker threads to join
    for (int i = 0; i < g_config.nthreads; ++i) {
        worker_threads[i].join();
    }
    // Wait for the progress threads to join
    g_progress_flag = false;
    progress_thread.join();

    MPI_Request_free(&g_cont_req);
    MPI_Finalize();
}