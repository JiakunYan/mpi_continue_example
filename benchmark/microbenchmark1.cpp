#include "common.hpp"
#include "mpi.h"
#include "common_mpi.hpp"

struct config_t {
    int niters = 100;
    int nmsgs = 100;
    int msg_size = 8;
    int use_diff_tag = true;
    int use_diff_order = true;
};
config_t g_config;
int g_rank, g_nranks;

void sender_fn() {
    char signal[1];
    std::vector<char> buffer(g_config.msg_size);
    std::vector<MPI_Request> requests(g_config.nmsgs);
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < g_config.niters; ++i) {
        for (int j = 0; j < g_config.nmsgs; ++j) {
            int tag = 0;
            if (g_config.use_diff_tag) {
                tag = j;
            }
            MPI_SAFECALL(MPI_Isend(buffer.data(), buffer.size(), MPI_BYTE, 1 - g_rank, tag, MPI_COMM_WORLD, &requests[j]));
        }
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        MPI_Recv(signal, 1, MPI_BYTE, 1 - g_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "Result: {"
              << "\"Average time per request (us)\": " << total_time * 1e6 / g_config.niters / g_config.nmsgs << ", "
              << "\"Average time per iteration (us)\": " << total_time * 1e6 / g_config.niters << ", "
              << "\"Total time (s)\": " << total_time << " }"
              << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
}

void receiver_fn() {
    char signal[1];
    std::vector<char> buffer(g_config.msg_size);
    std::vector<MPI_Request> requests(g_config.nmsgs);
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < g_config.niters; ++i) {
        for (int j = 0; j < g_config.nmsgs; ++j) {
            int tag = 0;
            if (g_config.use_diff_tag) {
                tag = j;
                if (g_config.use_diff_order) {
                    tag = g_config.nmsgs - j - 1;
                }
            }
            MPI_SAFECALL(MPI_Irecv(buffer.data(), buffer.size(), MPI_BYTE, 1 - g_rank, tag, MPI_COMM_WORLD, &requests[j]));
        }
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
        MPI_Send(signal, 1, MPI_BYTE, 1 - g_rank, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char *argv[]) {
//    bool wait_for_dbg = true;
//    while (wait_for_dbg) continue;
    bench::args_parser_t args_parser;
    args_parser.add("nmsgs", required_argument, (int*)&g_config.nmsgs);
    args_parser.add("msg-size", required_argument, (int*)&g_config.msg_size);
    args_parser.add("niters", required_argument, (int*)&g_config.niters);
    args_parser.add("use-diff-tag", required_argument, (int*)&g_config.use_diff_tag);
    args_parser.add("use-diff-order", required_argument, (int*)&g_config.use_diff_order);
    args_parser.parse_args(argc, argv);
    // initialize MPI
    setenv("MPIR_CVAR_CH4_RESERVE_VCIS", "1", true);
    MPI_SAFECALL(MPI_Init(nullptr, nullptr));
    MPI_SAFECALL(MPI_Comm_size(MPI_COMM_WORLD, &g_nranks));
    MPI_SAFECALL(MPI_Comm_rank(MPI_COMM_WORLD, &g_rank));
    if (g_rank == 0) {
        args_parser.print(false);
        sender_fn();
    } else {
        receiver_fn();
    }


    MPI_SAFECALL(MPI_Finalize());
    return 0;
}