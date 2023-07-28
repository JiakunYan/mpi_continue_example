#ifndef MPI_CONTINUE_EXAMPLE_COMMON_HPP
#define MPI_CONTINUE_EXAMPLE_COMMON_HPP
#include <unistd.h>
#include <getopt.h>
#include <cstring>
#include <utility>

namespace bench {
    class args_parser_t {
    public:
        struct dict_entry_t {
            std::string key;
            int val;
            dict_entry_t(const char *key_, int val_) : key(key_), val(val_) {}
        };
        void add(const std::string& name, int has_arg, int *ptr,
                 const std::vector<dict_entry_t> &dict = {}) {
            args.push_back({name, has_arg, ptr, dict});
        }

        void parse_args(int argc, char *argv[]) {
            int long_flag;
            std::vector<struct option> long_options;
            long_options.reserve(args.size() + 1);
            for (int i = 0; i < args.size(); ++i) {
                long_options.push_back({args[i].name.c_str(), args[i].has_arg, &long_flag, i});
            }
            long_options.push_back({nullptr, 0, nullptr, 0});
            while (getopt_long(argc, argv, "", long_options.data(), nullptr) == 0) {
                int i = long_flag;
                if (args[i].has_arg == no_argument) {
                    *args[i].ptr = 1;
                    continue;
                } else {
                    bool found = false;
                    for (const auto &item: args[i].dict) {
                        if (item.key == optarg) {
                            *args[i].ptr = item.val;
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        *args[i].ptr = std::stoi(optarg);
//                        printf("Assign %d to %s\n", *args[i].ptr, args[i].name.c_str());
                    } else {
//                        printf("Assign %d (%s) to %s\n", *args[i].ptr, optarg, args[i].name.c_str());
                    }
                }
            }
        }

        void print(bool verbose) {
            if (!verbose) {
                std::cout << "ArgsParser: {";
            }
            for (auto& arg : args) {
                if (verbose) {
                    std::string verbose_val;
                    if (!arg.dict.empty()) {
                        for (auto &item : arg.dict) {
                            if (item.val == *arg.ptr) {
                                verbose_val = item.key;
                                break;
                            }
                        }
                    }
                    std::cout << "ArgsParser: " << arg.name << " = " << *arg.ptr;
                    if (!verbose_val.empty()) {
                        std::cout << "(" << verbose_val << ")";
                    }
                    std::cout << std::endl;
                } else {
                    std::cout << "\"" << arg.name << "\": " << *arg.ptr << ", ";
                }
            }
            if (!verbose) {
                std::cout << "}\n";
            }
        }
    private:
        struct arg_t {
            std::string name;
            int has_arg;
            int *ptr;
            std::vector<dict_entry_t> dict;
        };
        std::vector<arg_t> args;
    };

    typedef std::vector<char> msg_t;
    struct parcel_t {
        int peer_rank;
        std::vector<msg_t> msgs;
        void *local_context;
    };
    class context_t {
    public:
        void parse_args(int argc, char *argv[], args_parser_t args_parser_ = args_parser_t()) {
            args_parser = std::move(args_parser_);
            nparcels = 10000;
            nchunks = 4;
            chunk_size = 8192;
            nthreads = 8;
            is_verbose = false;
            args_parser.add("nparcels", required_argument, &nparcels);
            args_parser.add("nchunks", required_argument, &nchunks);
            args_parser.add("chunk-size", required_argument, &chunk_size);
            args_parser.add("nthreads", required_argument, &nthreads);
            args_parser.add("verbose", no_argument, &is_verbose);
            args_parser.parse_args(argc, argv);
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
                } else {
                    // the receiver
                    send_comp_expected = 0;
                    recv_comp_expected = nparcels;
                }
                if (send_comp_expected == 0)
                    send_done_flag = true;
                if (recv_comp_expected == 0)
                    recv_done_flag = true;
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
            double msg_rate = nparcels / total_time;
            double bandwidth = msg_rate * nchunks * chunk_size;
            if (rank == 0) {
                args_parser.print(is_verbose);
                if (is_verbose) {
                    std::cout
                            << "nparcels: " << nparcels << "\n"
                            << "nchunks: " << nchunks << "\n"
                            << "chunk_size (B): " << chunk_size << "\n"
                            << "nthreads: " << nthreads << "\n"
                            << "nranks: " << nranks << "\n"
                            << "Total time (s): " << total_time << "\n"
                            << "Message Rate (K/s): " << msg_rate / 1e3 << "\n"
                            << "Bandwidth (MB/s): " << bandwidth / 1e6
                            << std::endl;
                } else {
                    std::cout << "Result: {"
                            << "\"nparcels\": " << nparcels << ", "
                            << "\"nchunks\": " << nchunks << ", "
                            << "\"chunk_size (B)\": " << chunk_size << ", "
                            << "\"nthreads\": " << nthreads << ", "
                            << "\"nranks\": " << nranks << ", "
                            << "\"Total time (s)\": " << total_time << ", "
                            << "\"Message Rate (K/s)\": " << msg_rate / 1e3 << ", "
                            << "\"Bandwidth (MB/s)\": " << bandwidth / 1e6 << " }"
                            << std::endl;
                }
            }
        }
    private:
        args_parser_t args_parser;
        int nparcels, nchunks, chunk_size, nthreads, is_verbose;
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
