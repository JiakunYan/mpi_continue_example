#include "common.hpp"
#include "mpi.h"
#include "common_mpi.hpp"

struct config_t {
    int nmsgs = 1000;
    int niters = 100;
};
bench::args_parser_t args_parser;
config_t g_config;
bench::context_t g_context;
int g_rank, g_nranks;

void worker_fn(int thread_id) {
    std::deque<MPI_Request> request_pool;
    std::vector<char> recv_buf(8);
    for (int i = 0; i < g_config.nmsgs; ++i) {
        MPI_Request request;
        MPI_SAFECALL(MPI_Irecv(recv_buf.data(), recv_buf.size(), MPI_BYTE, 0, i, MPI_COMM_WORLD, &request));
        request_pool.push_back(request);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < g_config.niters; ++i) {
        for (int j = 0; j < g_config.nmsgs; ++j) {
            auto request = request_pool.front();
            request_pool.pop_front();
            int flag;
            MPI_SAFECALL(MPI_Test(&request, &flag, MPI_STATUS_IGNORE));
            assert(flag == 0);
            request_pool.push_back(request);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    args_parser.print(false);
    std::cout << "Result: {"
              << "\"Average time per request (us)\": " << total_time * 1e6 / g_config.niters / g_config.nmsgs << ", "
              << "\"Average time per iteration (us)\": " << total_time * 1e6 / g_config.niters << ", "
              << "\"Total time (s)\": " << total_time << " }"
              << std::endl;

    for (int i = 0; i < g_config.nmsgs; ++i) {
        auto request = request_pool.front();
        request_pool.pop_front();
        MPI_SAFECALL(MPI_Cancel(&request));
    }
}

int main(int argc, char *argv[]) {
//    bool wait_for_dbg = true;
//    while (wait_for_dbg) continue;
    args_parser.add("nmsgs", required_argument, (int*)&g_config.nmsgs);
    args_parser.add("niters", required_argument, (int*)&g_config.niters);
    args_parser.parse_args(argc, argv);
    // initialize MPI
    MPI_SAFECALL(MPI_Init(nullptr, nullptr));
    MPI_SAFECALL(MPI_Comm_size(MPI_COMM_WORLD, &g_nranks));
    MPI_SAFECALL(MPI_Comm_rank(MPI_COMM_WORLD, &g_rank));

    worker_fn(0);

    MPI_SAFECALL(MPI_Finalize());
    return 0;
}