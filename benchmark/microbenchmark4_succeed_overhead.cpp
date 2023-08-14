#include "common.hpp"
#include "mpi.h"
#include "common_mpi.hpp"

#ifdef MPIX_STREAM_NULL

struct config_t {
    int nmsgs = 200000;
    int niters = 10;
    enum class comp_type_t {
        REQUEST,
        CONTINUE,
    } comp_type = comp_type_t::CONTINUE;
};
bench::args_parser_t args_parser;
config_t g_config;
bench::context_t g_context;
int g_rank, g_nranks;

int fake_data;
int fake_callback(int error_code, void *user_data) {
    return MPI_SUCCESS;
}

void worker_fn(int thread_id) {
    std::vector<MPI_Request> recv_requests(g_config.nmsgs);
    std::vector<MPI_Request> send_requests(g_config.nmsgs);
    std::vector<char> send_buf(8);
    std::vector<char> recv_buf(8);
    double send_total_time = 0, recv_total_time = 0;
    auto total_start = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < g_config.niters; ++j) {
        for (int i = 0; i < g_config.nmsgs; ++i) {
            MPI_SAFECALL(
                    MPI_Irecv(recv_buf.data(), recv_buf.size(), MPI_BYTE, 0, 0, MPI_COMM_WORLD, &recv_requests[i]));
            MPI_SAFECALL(
                    MPI_Isend(send_buf.data(), send_buf.size(), MPI_BYTE, 0, 0, MPI_COMM_WORLD, &send_requests[i]));
        }

        MPI_Request request;
        MPI_SAFECALL(MPI_Irecv(recv_buf.data(), recv_buf.size(), MPI_BYTE, 0, 0, MPI_COMM_WORLD, &request));
        MPI_SAFECALL(MPI_Send(send_buf.data(), send_buf.size(), MPI_BYTE, 0, 0, MPI_COMM_WORLD));
        MPI_SAFECALL(MPI_Wait(&request, MPI_STATUSES_IGNORE));

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < g_config.nmsgs; ++i) {
            if (g_config.comp_type == config_t::comp_type_t::REQUEST) {
                int flag;
                MPI_SAFECALL(MPI_Test(&recv_requests[i], &flag, MPI_STATUS_IGNORE));
                assert(flag == 1);
            } else {
                MPI_SAFECALL(MPIX_Continue(&recv_requests[i], fake_callback, &fake_data,
                                           MPIX_CONT_IMMEDIATE | MPIX_CONT_FORGET, MPI_STATUS_IGNORE,
                                           MPI_REQUEST_NULL));
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        recv_total_time += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < g_config.nmsgs; ++i) {
            if (g_config.comp_type == config_t::comp_type_t::REQUEST) {
                int flag;
                MPI_SAFECALL(MPI_Test(&send_requests[i], &flag, MPI_STATUS_IGNORE));
                assert(flag == 1);
            } else {
                MPI_SAFECALL(MPIX_Continue(&send_requests[i], fake_callback, &fake_data,
                                           MPIX_CONT_IMMEDIATE | MPIX_CONT_FORGET, MPI_STATUS_IGNORE,
                                           MPI_REQUEST_NULL));
            }
        }
        end = std::chrono::high_resolution_clock::now();
        send_total_time += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    }
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::duration<double>>(total_end - total_start).count();
    args_parser.print(false);
    std::cout << "Result: {"
              << "\"Total time (s)\": " << total_time << ", "
              << "\"Average time per recv request (us)\": " << recv_total_time * 1e6 / g_config.niters / g_config.nmsgs << ", "
              << "\"Average time per send request (us)\": " << send_total_time * 1e6 / g_config.niters / g_config.nmsgs << " }"
              << std::endl;
}

int main(int argc, char *argv[]) {
//    bool wait_for_dbg = true;
//    while (wait_for_dbg) continue;
    args_parser.add("nmsgs", required_argument, (int*)&g_config.nmsgs);
    args_parser.add("niters", required_argument, (int*)&g_config.niters);
    args_parser.add("comp-type", required_argument, (int*)&g_config.comp_type,
                    {{"continue", (int)config_t::comp_type_t::CONTINUE},
                     {"request", (int)config_t::comp_type_t::REQUEST},});
    args_parser.parse_args(argc, argv);
    // initialize MPI
    MPI_SAFECALL(MPI_Init(nullptr, nullptr));
    MPI_SAFECALL(MPI_Comm_size(MPI_COMM_WORLD, &g_nranks));
    MPI_SAFECALL(MPI_Comm_rank(MPI_COMM_WORLD, &g_rank));
    assert(g_nranks == 1);

    worker_fn(0);

    MPI_SAFECALL(MPI_Finalize());
    return 0;
}

#else
int main() {
    return 0;
}
#endif