#ifndef MPI_CONTINUE_EXAMPLE_COMMON_HPP
#define MPI_CONTINUE_EXAMPLE_COMMON_HPP
#include <unistd.h>

namespace bench {
    typedef std::vector<char> msg_t;
    struct parcel_t {
        int peer_rank;
        std::vector<msg_t> msgs;
        void *local_context;
    };
    class context_t {
    public:
        void parse_args(int argc, char *argv[]) {
            nparcels = 10000;
            nchunks = 4;
            chunk_size = 8192;
            nthreads = 8;
            is_verbose = false;
            int opt;
            while ((opt = getopt(argc, argv, "p:c:s:t:v")) != -1) {
                switch (opt) {
                    case 'p': nparcels = std::stoi(optarg); break;
                    case 'c': nchunks = std::stoi(optarg); break;
                    case 's': chunk_size = std::stoi(optarg); break;
                    case 't': nthreads = std::stoi(optarg); break;
                    case 'v': is_verbose = true; break;
                    default:
                        fprintf(stderr, "Usage: %s [-ilw] [file...]\n", argv[0]);
                        exit(EXIT_FAILURE);
                }
            }
        }
        void setup(int rank_, int nranks_) {
            rank = rank_;
            nranks = nranks_;
            send_count = 0;
            send_comp_count = 0;
            recv_comp_count = 0;
            send_done_flag = false;
            recv_done_flag = false;
            assert(nranks == 1 || nranks % 2 == 0);
            if (nranks == 1) {
                send_comp_expected = nparcels;
                recv_comp_expected = nparcels;
                peer_rank = 0;
            } else {
                peer_rank = (1 - rank % 2) + rank / 2 * 2;
                if (rank % 2 == 0) {
                    // the sender
                    send_comp_expected = nparcels;
                    recv_comp_expected = 0;
                    recv_done_flag = true;
                } else {
                    // the receiver
                    send_comp_expected = 0;
                    send_done_flag = true;
                    recv_comp_expected = nparcels;
                }
            }
        }

        int get_nthreads() const {
            return nthreads;
        }

        parcel_t *get_parcel() {
            parcel_t *parcel = nullptr;
            int count = send_count;
            if (count < send_comp_expected) {
                count = ++send_count;
                if (count <= send_comp_expected) {
                    parcel = new parcel_t;
                    parcel->peer_rank = peer_rank;
                    parcel->msgs.resize(nchunks);
                    for (auto &chunk: parcel->msgs) {
                        chunk.resize(chunk_size);
                    }
                }
            }
            return parcel;
        }
        bool signal_send_comp() {
            int count = ++send_comp_count;
            if (count == send_comp_expected) {
                if (!send_done_flag)
                    send_done_flag = true;
                return true;
            } else {
                assert(count < send_comp_expected);
                return false;
            }
        }
        bool signal_recv_comp(parcel_t *parcel) {
            delete parcel;
            int count = ++recv_comp_count;
            if (count == recv_comp_expected) {
                if (!recv_done_flag)
                    recv_done_flag = true;
                return true;
            } else {
                assert(count < recv_comp_expected);
                return false;
            }
        }
        bool is_done() {
            return send_done_flag && recv_done_flag;
        }
        void report(double total_time) {
            if (rank == 0) {
                std::cout
                        << "nparcels: " << nparcels << "\n"
                        << "nchunks: " << nchunks << "\n"
                        << "chunk_size (B): " << chunk_size << "\n"
                        << "nthreads: " << nthreads << "\n"
                        << "nranks: " << nranks << "\n"
                        << "Total time (s): " << total_time << std::endl;
            }
        }
    private:
        int nparcels, nchunks, chunk_size, nthreads;
        bool is_verbose;
        int rank, nranks;
        int send_comp_expected, recv_comp_expected, peer_rank;
        alignas(64) std::atomic<int> send_count;
        alignas(64) std::atomic<int> send_comp_count;
        alignas(64) std::atomic<int> recv_comp_count;
        alignas(64) std::atomic<bool> send_done_flag;
        alignas(64) std::atomic<bool> recv_done_flag;
    };
} // namespace bench

#endif //MPI_CONTINUE_EXAMPLE_COMMON_HPP
