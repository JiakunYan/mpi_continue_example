#ifndef MPI_CONTINUE_EXAMPLE_COMMON_HPP
#define MPI_CONTINUE_EXAMPLE_COMMON_HPP
#include <unistd.h>
#include <getopt.h>
#include <cstring>
#include <utility>
#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <atomic>
#include <cassert>

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
        msg_t piggyback;
        std::vector<msg_t> chunks;
        void *local_context;
        parcel_t() = default;
        parcel_t(int peer_rank_, int piggyback_size, int nchunks, int chunk_size)
            : peer_rank(peer_rank_), piggyback(piggyback_size), chunks(nchunks), local_context(nullptr) {
            for (auto &chunk: chunks) {
                chunk.resize(chunk_size);
            }
        }
    };
    class context_t {
    public:
        void parse_args(int argc, char *argv[], args_parser_t args_parser_ = args_parser_t()) {
            args_parser = std::move(args_parser_);
            nparcels = 10000;
            piggyback_size = 8;
            nchunks = 4;
            chunk_size = 8192;
            nthreads = 8;
            inject_rate = 0;
            nsteps = 1;
            is_verbose = false;
            args_parser.add("nparcels", required_argument, &nparcels);
            args_parser.add("piggyback-size", required_argument, &piggyback_size);
            args_parser.add("nchunks", required_argument, &nchunks);
            args_parser.add("chunk-size", required_argument, &chunk_size);
            args_parser.add("nthreads", required_argument, &nthreads);
            args_parser.add("inject-rate", required_argument, &inject_rate);
            args_parser.add("nsteps", required_argument, &nsteps);
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
                send_count_expected = nparcels;
                send_comp_expected = nparcels;
                recv_comp_expected = nparcels;
                peer_rank = 0;
            } else {
                peer_rank = (1 - rank % 2) + rank / 2 * 2;
                if (rank % 2 == 0) {
                    send_count_expected = nparcels;
                } else {
                    send_count_expected = 0;
                }
                if ((rank + nsteps) % 2 == 0) {
                    send_comp_expected = 0;
                    recv_comp_expected = nparcels;
                } else {
                    send_comp_expected = nparcels;
                    recv_comp_expected = 0;
                }
                if (send_comp_expected == 0)
                    send_done_flag = true;
                if (recv_comp_expected == 0)
                    recv_done_flag = true;
            }
            send_count_batch_size = std::max(send_count_expected / nthreads / send_count_batch_percent, 1);
            start_time = std::chrono::high_resolution_clock::now();
        }

        int get_nthreads() const {
            return nthreads;
        }

        int get_nsteps() const {
            return nsteps;
        }
        
        bool check_inject_rate(int next_count) {
            if (inject_rate) {
                // Check whether we would exceed injection rate.
                double elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::high_resolution_clock::now() - start_time).count();
                double wouldbe_inject_rate = next_count / elapsed_time;
                if (wouldbe_inject_rate > inject_rate)
                    return false;
            }
            return true;
        }

        parcel_t *get_parcel() {
            parcel_t *parcel = nullptr;
            if (tls_context.send_count_reserved == 0) {
                int count = send_count;
                int next_batch = std::min(send_count_expected - count, send_count_batch_size);
                if (count < send_count_expected && check_inject_rate(count + next_batch)) {
                    count = send_count.fetch_add(next_batch, std::memory_order_relaxed);
                    if (count < send_count_expected) {
                        tls_context.send_count_reserved = std::min(next_batch, send_count_expected - count);
                    }
                }
            }
            if (tls_context.send_count_reserved > 0) {
                --tls_context.send_count_reserved;
                parcel = new parcel_t(peer_rank, piggyback_size, nchunks, chunk_size);
            }
            return parcel;
        }
        static void signal_send_comp() {
            ++tls_context.send_comp_count;
        }
        static void signal_recv_comp(parcel_t *parcel) {
            delete parcel;
            ++tls_context.recv_comp_count;
        }
        bool is_done() {
            if (tls_context.send_comp_count == tls_context.last_send_comp_count &&
                tls_context.recv_comp_count == tls_context.last_recv_comp_count) {
                if (++tls_context.poke_count == poke_count_before_flush) {
                    if (tls_context.send_comp_count > 0) {
                        int new_count = send_comp_count.fetch_add(tls_context.send_comp_count) + tls_context.send_comp_count;
                        if (new_count == send_comp_expected) {
                            send_done_flag = true;
                        }
                        tls_context.send_comp_count = 0;
                    }
                    if (tls_context.recv_comp_count > 0) {
                        int new_count = recv_comp_count.fetch_add(tls_context.recv_comp_count) + tls_context.recv_comp_count;
                        if (new_count == recv_comp_expected) {
                            recv_done_flag = true;
                        }
                        tls_context.recv_comp_count = 0;
                    }
                    tls_context.poke_count = 0;
                    tls_context.last_send_comp_count = 0;
                    tls_context.last_recv_comp_count = 0;
                }
            } else {
                tls_context.poke_count = 0;
                tls_context.last_send_comp_count = tls_context.send_comp_count;
                tls_context.last_recv_comp_count = tls_context.recv_comp_count;
            }
            return send_done_flag && recv_done_flag;
        }
        void report(double total_time) {
            double msg_rate = nparcels / total_time;
            double bandwidth = msg_rate * (nchunks * chunk_size + piggyback_size);
            if (rank == 0) {
                args_parser.print(is_verbose);
                if (is_verbose) {
                    std::cout
                            << "nparcels: " << nparcels << "\n"
                            << "nchunks: " << nchunks << "\n"
                            << "chunk_size (B): " << chunk_size << "\n"
                            << "nthreads: " << nthreads << "\n"
                            << "nranks: " << nranks << "\n"
                            << "Injection Rate (K/s): " << inject_rate / 1e3 << "\n"
                            << "Total time (s): " << total_time << "\n"
                            << "Latency (us): " << total_time * 1e6 / nsteps << "\n"
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
                            << "\"Injection Rate (K/s)\": " << inject_rate / 1e3 << ", "
                            << "\"Total time (s)\": " << total_time << ", "
                            << "\"Latency (us)\": " << total_time * 1e6 / nsteps << ", "
                            << "\"Message Rate (K/s)\": " << msg_rate / 1e3 << ", "
                            << "\"Bandwidth (MB/s)\": " << bandwidth / 1e6 << " }"
                            << std::endl;
                }
            }
        }
    private:
        args_parser_t args_parser;
        int nparcels, piggyback_size, nchunks, chunk_size, nthreads, inject_rate, is_verbose, nsteps;
        const int send_count_batch_percent = 1000;
        const int poke_count_before_flush = 1000;
        int send_count_batch_size;
        std::chrono::high_resolution_clock::time_point start_time;
        int rank, nranks;
        int send_count_expected, send_comp_expected, recv_comp_expected, peer_rank;
        alignas(64) std::atomic<int> send_count;
        alignas(64) std::atomic<int> send_comp_count;
        alignas(64) std::atomic<int> recv_comp_count;
        alignas(64) std::atomic<bool> send_done_flag;
        alignas(64) std::atomic<bool> recv_done_flag;
        struct tls_context_t {
            int send_count_reserved = 0;
            int send_comp_count = 0;
            int recv_comp_count = 0;
            int poke_count = 0;
            int last_send_comp_count = 0;
            int last_recv_comp_count = 0;
        };
        static __thread tls_context_t tls_context;
    };
    __thread context_t::tls_context_t context_t::tls_context;
} // namespace bench

#endif //MPI_CONTINUE_EXAMPLE_COMMON_HPP
