#include <iostream>
#include <vector>
#include <thread>
#include <cassert>
#include <atomic>
#include <chrono>
#include <deque>
#include "common.hpp"

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
int g_rank = 0, g_nranks = 1;
bench::context_t g_context;
int g_max_tag = 32767;
__thread int tls_next_tag = 0;
const int MAX_HEADER_SIZE = 4096;

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
    memcpy(parcel->piggyback.data(), &header[i], parcel->piggyback.size());
}

int sender_callback(int error_code, void *user_data) {
    auto *parcel = static_cast<bench::parcel_t*>(user_data);
    auto *header = static_cast<std::vector<int>*>(parcel->local_context);
    delete header;
    delete parcel;
    bench::context_t::signal_send_comp();
    return 0;
}

int receiver_callback(int error_code, void *user_data) {
    auto *parcel = static_cast<bench::parcel_t*>(user_data);
    bench::context_t::signal_recv_comp(parcel);
    return 0;
}

void try_sendrecv_parcel() {
    auto *parcel = g_context.get_parcel();
    if (parcel) {
        // send the parcel
        int tag = tls_next_tag++ % (g_max_tag - 1) + 1;
        auto header = encode_header(tag, parcel);
        parcel->local_context = header;
        auto *recv_parcel = new bench::parcel_t(*parcel);
        sender_callback(0, parcel);
        receiver_callback(0, recv_parcel);
    }
}

void worker_thread_fn(int thread_id) {
    while (!g_context.is_done()) {
        try_sendrecv_parcel();
    }
}

int main(int argc, char *argv[]) {
//    bool wait_for_dbg = true;
//    while (wait_for_dbg) continue;
    bench::args_parser_t args_parser;
    // initialize the benchmark context
    g_context.parse_args(argc, argv, args_parser);
    g_config.nthreads = g_context.get_nthreads();
    // Setup problem
    g_context.setup(g_rank, g_nranks);
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

    auto end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    g_context.report(total_time.count());
    return 0;
}