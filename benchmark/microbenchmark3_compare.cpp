#include "common.hpp"
#include "mpi.h"
#include "common_mpi.hpp"

#ifdef MPIX_STREAM_NULL

struct config_t {
    enum class comp_type_t {
        REQUEST,
        CONTINUE,
    } comp_type = comp_type_t::CONTINUE;
    int nmsgs = 100;
    int niters = 100;
    int msg_size = 8;
    int use_diff_tag = true;
    int use_diff_order = false;
};
config_t g_config;
detail::comp_manager_base_t *g_comp_manager_p;
int g_rank, g_nranks;

int receiver_callback(int error_code, void *user_data) {
    int *counter = static_cast<int*>(user_data);
    --(*counter);
    return MPI_SUCCESS;
}

void send_fn() {
    int total_test_counter = 0;
    std::vector<char> buffer(g_config.msg_size);
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < g_config.niters; ++i) {
        for (int j = 0; j < g_config.nmsgs; ++j) {
            int tag = 0;
            if (g_config.use_diff_tag) {
                tag = j;
            }
            MPI_SAFECALL(MPI_Send(buffer.data(), buffer.size(), MPI_BYTE, 1 - g_rank, tag, MPI_COMM_WORLD));
        }
        int test_counter;
        MPI_Recv(&test_counter, 1, MPI_INT, 1 - g_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        total_test_counter += test_counter;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "Result: {"
              << "\"Failed test rate\": " << (double)(total_test_counter - g_config.nmsgs * g_config.niters) / total_test_counter << ", "
              << "\"Average time per request (us)\": " << total_time * 1e6 / g_config.niters / g_config.nmsgs << ", "
              << "\"Average time per iter (us)\": " << total_time * 1e6 / g_config.niters << ", "
              << "\"Total time (s)\": " << total_time << " }"
              << std::endl;
}

void recv_fn() {
    std::vector<char> recv_buf(g_config.msg_size);
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < g_config.niters; ++i) {
        int counter = g_config.nmsgs;
        for (int j = 0; j < g_config.nmsgs; ++j) {
            std::vector<MPI_Request> requests(1);
            int tag = 0;
            if (g_config.use_diff_tag) {
                tag = j;
                if (g_config.use_diff_order) {
                    tag = g_config.nmsgs - j - 1;
                }
            }
            MPI_SAFECALL(MPI_Irecv(recv_buf.data(), recv_buf.size(), MPI_BYTE, 1 - g_rank, tag, MPI_COMM_WORLD, &requests[0]));
            g_comp_manager_p->push(std::move(requests), receiver_callback, &counter);
        }
        int test_counter = 0;
        while (counter) {
            if (g_config.comp_type == config_t::comp_type_t::CONTINUE)
                MPIX_Stream_progress(MPIX_STREAM_NULL);
            ++test_counter;
            g_comp_manager_p->progress();
        }
        MPI_Send(&test_counter, 1, MPI_INT, 1 - g_rank, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char *argv[]) {
//    bool wait_for_dbg = true;
//    while (wait_for_dbg) continue;
    bench::args_parser_t args_parser;
    args_parser.add("nmsgs", required_argument, (int*)&g_config.nmsgs);
    args_parser.add("niters", required_argument, (int*)&g_config.niters);
    args_parser.add("msg-size", required_argument, (int*)&g_config.msg_size);
    args_parser.add("comp-type", required_argument, (int*)&g_config.comp_type,
                    {{"continue", (int)config_t::comp_type_t::CONTINUE},
                     {"request", (int)config_t::comp_type_t::REQUEST},});
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
    }
//    if (g_rank == 0) {
//        bool wait_for_dbg = true;
//        while (wait_for_dbg) continue;
//    }
//    MPI_Barrier(MPI_COMM_WORLD);
    switch (g_config.comp_type) {
        case config_t::comp_type_t::REQUEST:
            g_comp_manager_p = new detail::comp_manager_request_t;
            break;
        case config_t::comp_type_t::CONTINUE:
            g_comp_manager_p = new detail::comp_manager_continue_t(1, 1);
            break;
    }

    g_comp_manager_p->init_thread();
    if (g_rank == 0) {
        send_fn();
    } else {
        recv_fn();
    }
    g_comp_manager_p->free_thread();

    MPI_SAFECALL(MPI_Finalize());
    return 0;
}

#else
int main() {
    return 0;
}
#endif