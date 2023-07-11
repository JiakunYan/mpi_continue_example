#ifndef MPI_CONTINUE_EXAMPLE_COMMON_HPP
#define MPI_CONTINUE_EXAMPLE_COMMON_HPP

namespace bench {
    typedef std::vector<char> msg_t;
    struct parcel_t {
        int peer_rank;
        std::vector<msg_t> msgs;
    };
    class context_t {
    public:
        void setup(int rank, int nranks, int argc, char *argv[]) {
            nparcels = 1000;
            nchunks = 0;
            chunk_size = 8192;
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
                } else {
                    // the receiver
                    send_comp_expected = 0;
                    recv_comp_expected = nparcels;
                }
            }
        }

        parcel_t *get_parcel() {
            parcel_t *parcel = nullptr;
            if (++send_count <= nparcels) {
                parcel = new parcel_t;
                parcel->peer_rank = peer_rank;
                parcel->msgs.resize(nchunks);
                for (auto &chunk: parcel->msgs) {
                    chunk.resize(chunk_size);
                }
            }
            return parcel;
        }
        bool signal_send_comp() {
            int count = ++send_comp_count;
            if (count >= send_comp_expected) {
                if (!send_done_flag)
                    send_done_flag = true;
                return true;
            } else {
                return false;
            }
        }
        bool signal_recv_comp(parcel_t *parcel) {
            delete parcel;
            int count = ++recv_comp_count;
            if (count >= recv_comp_expected) {
                if (!recv_done_flag)
                    recv_done_flag = true;
                return true;
            } else {
                return false;
            }
        }
        bool is_done() {
            return send_done_flag && recv_done_flag;
        }
//    private:
        int nparcels, nchunks, chunk_size;
        int send_comp_expected, recv_comp_expected, peer_rank;
        alignas(64) std::atomic<int> send_count;
        alignas(64) std::atomic<int> send_comp_count;
        alignas(64) std::atomic<int> recv_comp_count;
        alignas(64) std::atomic<bool> send_done_flag;
        alignas(64) std::atomic<bool> recv_done_flag;
    };
} // namespace bench

#endif //MPI_CONTINUE_EXAMPLE_COMMON_HPP
