#include "common.hpp"
#include "mpi.h"
#include "common_mpi.hpp"

struct config_t {
    enum class comp_type_t {
        REQUEST,
        CONTINUE,
    } comp_type = comp_type_t::CONTINUE;
    int nmsgs = 10000;
    int msg_size = 8;
    enum class order_t {
        SAME,
        REVERSE,
    };
    order_t order;
};
config_t g_config;
detail::comp_manager_base_t *g_comp_manager_p;
int g_rank, g_nranks;
MPIX_Stream stream;
MPI_Comm comm;

int receiver_callback(int error_code, void *user_data) {
    int *counter = static_cast<int*>(user_data);
    --(*counter);
    return MPI_SUCCESS;
}

void worker_fn(int thread_id) {
    char buffer[1];
    if (g_rank == 0) {
        // pre-post recvs
        int counter = g_config.nmsgs;
        std::vector<char> recv_buf(g_config.msg_size);
        for (int i = 0; i < g_config.nmsgs; ++i) {
            std::vector<MPI_Request> requests(1);
            MPI_SAFECALL(MPI_Irecv(recv_buf.data(), recv_buf.size(), MPI_BYTE, 1 - g_rank, i, comm, &requests[0]));
            assert(requests[0] != MPI_REQUEST_NULL);
            g_comp_manager_p->push(std::move(requests), receiver_callback, &counter);
        }
        MPI_Send(buffer, 1, MPI_BYTE, 1 - g_rank, 0, comm);
        auto start = std::chrono::high_resolution_clock::now();
        while (counter) {
            MPIX_Stream_progress(stream);
            g_comp_manager_p->progress();
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        std::cout << "Result: {"
                  << "\"Average time per request (us)\": " << total_time * 1e6 / g_config.nmsgs << ", "
                  << "\"Total time (s)\": " << total_time << " }"
                  << std::endl;
    } else {
        std::vector<MPI_Request> requests(g_config.nmsgs);
        std::vector<char> send_buf(g_config.msg_size);
        MPI_Recv(buffer, 1, MPI_BYTE, 1 - g_rank, 0, comm, MPI_STATUS_IGNORE);
        for (int i = g_config.nmsgs - 1; i >= 0; --i) {
            MPI_SAFECALL(MPI_Isend(send_buf.data(), send_buf.size(), MPI_BYTE, 1 - g_rank, i, comm, &requests[i]));
        }
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    }
    MPI_Barrier(comm);
}

int main(int argc, char *argv[]) {
//    bool wait_for_dbg = true;
//    while (wait_for_dbg) continue;
    bench::args_parser_t args_parser;
    args_parser.add("nmsgs", required_argument, (int*)&g_config.nmsgs);
    args_parser.add("msg-size", required_argument, (int*)&g_config.msg_size);
    args_parser.add("order", required_argument, (int*)&g_config.order,
                    {{"same", (int)config_t::order_t::SAME},
                     {"reverse", (int)config_t::order_t::REVERSE},});
    args_parser.add("comp-type", required_argument, (int*)&g_config.comp_type,
                    {{"continue", (int)config_t::comp_type_t::CONTINUE},
                     {"request", (int)config_t::comp_type_t::REQUEST},});
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
    MPI_SAFECALL(MPIX_Stream_create(MPI_INFO_NULL, &stream));
    MPIX_Stream_comm_create(MPI_COMM_WORLD, stream, &comm);

    g_comp_manager_p->init_thread();
    worker_fn(0);
    g_comp_manager_p->free_thread();

    MPI_SAFECALL(MPI_Finalize());
    return 0;
}