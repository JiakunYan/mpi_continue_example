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
#define MPIX_CONT_FORGET 0

#ifndef OMPI_HAVE_MPI_EXT_CONTINUE
#define MPIX_Continue_init(...) MPI_ERR_UNKNOWN
#define MPIX_Continue(...) MPI_ERR_UNKNOWN
#define MPIX_Continueall(...) MPI_ERR_UNKNOWN
#endif
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
    int nvcis = 1;
};

config_t g_config;
int g_rank, g_nranks;
bench::context_t g_context;
std::vector<MPIX_Stream> g_streams;
std::vector<MPI_Comm> g_comms;
int g_max_tag;
alignas(64) std::atomic<int> g_next_tag(0);
const int MAX_HEADER_SIZE = 1024;
struct tls_context_t {
    MPIX_Stream stream;
    MPI_Comm comm;
    MPI_Request header_req;
    std::vector<int> recv_buffer;
    explicit tls_context_t(int thread_id) : recv_buffer(MAX_HEADER_SIZE), header_req(MPI_REQUEST_NULL) {
        int nthreads_per_vci = (g_config.nthreads + g_config.nvcis - 1) / g_config.nvcis;
        int vci_id = thread_id / nthreads_per_vci;
        stream = g_streams[vci_id];
        comm = g_comms[vci_id];
    }
};
thread_local tls_context_t *tls_context_p;

namespace detail {
    class comp_manager_base_t {
    public:
        typedef int (cb_fn_t)(int error_code, void *user_data);
        virtual ~comp_manager_base_t() = default;
        virtual void push(std::vector<MPI_Request>&& requests, cb_fn_t *cb, void *context) = 0;
        virtual bool progress() = 0;
        virtual void init_thread() {};
        virtual void free_thread() {};
    };
    class comp_manager_request_t : public comp_manager_base_t {
    public:
        struct pending_parcel_t {
            std::vector<MPI_Request> requests;
            cb_fn_t *cb;
            void *context;
        };
        void push(std::vector<MPI_Request>&& requests, cb_fn_t *cb, void *context) override {
            auto *parcel = new pending_parcel_t;
            parcel->requests = std::move(requests);
            parcel->cb = cb;
            parcel->context = context;
            pending_parcels.push_back(parcel);
        }
        bool progress() override {
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
    private:
        static thread_local std::deque<pending_parcel_t*> pending_parcels;
    };
    thread_local std::deque<comp_manager_request_t::pending_parcel_t*> comp_manager_request_t::pending_parcels;

    class comp_manager_continue_t : public comp_manager_base_t {
    public:
        explicit comp_manager_continue_t(config_t config_) : config(config_) {}
        void push(std::vector<MPI_Request>&& requests, cb_fn_t *cb, void *context) override {
            int cont_flag = 0;
            if (config.use_cont_imm)
                cont_flag |= MPIX_CONT_IMMEDIATE;
            if (config.use_thread_single)
                cont_flag |= MPIX_CONT_FORGET;
            MPI_SAFECALL(MPIX_Continueall(static_cast<int>(requests.size()), requests.data(), cb,
                                          context, cont_flag, MPI_STATUSES_IGNORE, tls_cont_req));
        }
        bool progress() override {
            if (tls_context_p->stream != MPIX_STREAM_NULL)
                MPI_SAFECALL(MPIX_Stream_progress(tls_context_p->stream));
            if (config.use_cont_forget) {
//                MPI_SAFECALL(MPIX_Stream_progress(MPIX_STREAM_NULL));
            } else {
                int flag;
                MPI_SAFECALL(MPI_Test(&tls_cont_req, &flag, MPI_STATUS_IGNORE));
                if (flag) {
                    MPI_Start(&tls_cont_req);
                }
            }
            return true;
        }
        void init_thread() override {
            if (config.progress_type == config_t::progress_type_t::SHARED &&
                config.comp_type == config_t::comp_type_t::CONTINUE) {
                MPI_SAFECALL(MPIX_Continue_init(0, 0, MPI_INFO_NULL, &tls_cont_req));
                MPI_SAFECALL(MPI_Start(&tls_cont_req));
            }
        }
        void free_thread() override {
            if (tls_cont_req != MPI_REQUEST_NULL) {
                MPI_SAFECALL(MPI_Request_free(&tls_cont_req));
            }
        }
    private:
        static thread_local MPI_Request tls_cont_req;
        config_t config;
    };
    thread_local MPI_Request comp_manager_continue_t::tls_cont_req = MPI_REQUEST_NULL;
}
detail::comp_manager_base_t *g_comp_manager_p;

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
                               parcel->peer_rank, 0, tls_context_p->comm, &request));
        requests.push_back(request);
        for (auto & chunk : parcel->msgs) {
            MPI_SAFECALL(MPI_Isend(chunk.data(), chunk.size(), MPI_CHAR,
                                   parcel->peer_rank, tag, tls_context_p->comm, &request));
            requests.push_back(request);
        }
        g_comp_manager_p->push(std::move(requests), sender_callback, parcel);
    }
}

void try_receive_parcel() {
    if (tls_context_p->header_req == MPI_REQUEST_NULL)
        MPI_SAFECALL(MPI_Irecv(tls_context_p->recv_buffer.data(), tls_context_p->recv_buffer.size(), MPI_INT, MPI_ANY_SOURCE, 0, tls_context_p->comm, &tls_context_p->header_req));

    MPI_Status status;
    int flag = 0;
    MPI_SAFECALL(MPI_Test(&tls_context_p->header_req, &flag, &status));
    if (flag) {
        // got a new header message
        int tag;
        auto *parcel = new bench::parcel_t;
        decode_header(tls_context_p->recv_buffer, &tag, parcel);
        assert(tag > 0);
        // post a new receive for header
        MPI_SAFECALL(MPI_Irecv(tls_context_p->recv_buffer.data(), tls_context_p->recv_buffer.size(), MPI_INT, MPI_ANY_SOURCE, 0, tls_context_p->comm, &tls_context_p->header_req));
        // post the follow-up receives
        parcel->peer_rank = status.MPI_SOURCE;
        if (!parcel->msgs.empty()) {
            std::vector<MPI_Request> requests;
            for (auto &chunk: parcel->msgs) {
                MPI_Request request;
                MPI_SAFECALL(MPI_Irecv(chunk.data(), chunk.size(), MPI_CHAR,
                                       status.MPI_SOURCE, tag, tls_context_p->comm, &request));
                requests.push_back(request);
            }
            g_comp_manager_p->push(std::move(requests), receiver_callback, parcel);
        } else {
            receiver_callback(MPI_SUCCESS, parcel);
        }
    }
}

void cancel_receive() {
    if (tls_context_p->header_req != MPI_REQUEST_NULL) {
        MPI_Cancel(&tls_context_p->header_req);
    }
}

void do_progress() {
    g_comp_manager_p->progress();
}

void worker_thread_fn(int thread_id) {
    tls_context_p = new tls_context_t(thread_id);
    g_comp_manager_p->init_thread();
    while (!g_context.is_done()) {
        try_receive_parcel();
        try_send_parcel();
        if (g_config.progress_type == config_t::progress_type_t::SHARED)
            do_progress();
    }
    cancel_receive();
    g_comp_manager_p->free_thread();
    delete tls_context_p;
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
    // Initialize the completion manager
    switch (g_config.comp_type) {
        case config_t::comp_type_t::REQUEST:
            g_comp_manager_p = new detail::comp_manager_request_t;
            break;
        case config_t::comp_type_t::CONTINUE:
            g_comp_manager_p = new detail::comp_manager_continue_t(g_config);
            break;
    }
    // Initialize streams
    for (int i = 0; i < g_config.nvcis; ++i) {
        MPI_Comm comm;
        if (g_config.enable_stream) {
            MPIX_Stream stream;
            MPIX_Stream_create(MPI_INFO_NULL, &stream);
            g_streams.push_back(stream);
            MPIX_Stream_comm_create(MPI_COMM_WORLD, stream, &comm);
        } else {
            g_streams.push_back(MPIX_STREAM_NULL);
            MPI_Comm_dup(MPI_COMM_WORLD, &comm);
        }
        g_comms.push_back(comm);
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
//    for (int i = 0; i < g_config.nvcis; ++i) {
//        MPI_Comm_free(&g_comms[i]);
//        /* FIXME: seems to be a bug here */
////        if (g_streams[i] != MPIX_STREAM_NULL)
////            MPIX_Stream_free(&g_streams[i]);
//    }
//    delete g_comp_manager_p;
//    MPI_SAFECALL(MPI_Finalize());
    return 0;
}