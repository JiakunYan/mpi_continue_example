#ifndef MPI_CONTINUE_EXAMPLE_COMMON_MPI_HPP
#define MPI_CONTINUE_EXAMPLE_COMMON_MPI_HPP

#include "mpi.h"
#include <deque>

#define MPI_SAFECALL(x) {      \
        int ret = (x); \
        if (ret != MPI_SUCCESS) { \
            fprintf(stderr, "%s:%d:%s: " \
            "MPI call failed!\n", __FILE__, __LINE__, #x); \
            exit(1); \
        }\
    }

namespace detail {
    class comp_manager_base_t {
    public:
        typedef int (cb_fn_t)(int error_code, void *user_data);
        virtual ~comp_manager_base_t() = default;
        virtual void push(std::vector<MPI_Request>&& requests, cb_fn_t *cb, void *context) = 0;
        virtual bool progress() = 0;
        virtual void init_thread() {};
        virtual void free_thread() {};
    };
    class comp_manager_request_t : public comp_manager_base_t {
    public:
        struct pending_parcel_t {
            std::vector<MPI_Request> requests;
            cb_fn_t *cb;
            void *context;
        };
        void push(std::vector<MPI_Request>&& requests, cb_fn_t *cb, void *context) override {
            int flag;
            MPI_SAFECALL(MPI_Testall(requests.size(), requests.data(), &flag, MPI_STATUS_IGNORE));
            if (flag) {
                cb(MPI_SUCCESS, context);
            } else {
                auto *parcel = new pending_parcel_t;
                parcel->requests = std::move(requests);
                parcel->cb = cb;
                parcel->context = context;
                pending_parcels.push_back(parcel);
            }
        }
        bool progress() override {
            if (pending_parcels.empty())
                return false;
            auto *parcel = pending_parcels.front();
            pending_parcels.pop_front();
            int flag;
            MPI_SAFECALL(MPI_Testall(parcel->requests.size(), parcel->requests.data(), &flag, MPI_STATUS_IGNORE));
            if (flag) {
                parcel->cb(MPI_SUCCESS, parcel->context);
                return true;
            } else {
                pending_parcels.push_back(parcel);
                return false;
            }
        }
    private:
        static thread_local std::deque<pending_parcel_t*> pending_parcels;
    };
    thread_local std::deque<comp_manager_request_t::pending_parcel_t*> comp_manager_request_t::pending_parcels;

#ifdef MPIX_CONT_DEFER_COMPLETE
    class comp_manager_continue_t : public comp_manager_base_t {
    public:
        explicit comp_manager_continue_t(int use_cont_imm, int use_cont_forget) {
            config.use_cont_imm = use_cont_imm;
            config.use_cont_forget = use_cont_forget;
        }
        void push(std::vector<MPI_Request>&& requests, cb_fn_t *cb, void *context) override {
            int cont_flag = 0;
            if (config.use_cont_imm)
                cont_flag |= MPIX_CONT_IMMEDIATE;
            if (config.use_cont_forget)
                cont_flag |= MPIX_CONT_FORGET;
            MPI_SAFECALL(MPIX_Continueall(static_cast<int>(requests.size()), requests.data(), cb,
                                          context, cont_flag, MPI_STATUSES_IGNORE, tls_cont_req));
        }
        bool progress() override {
            if (config.use_cont_forget) {
//                MPI_SAFECALL(MPIX_Stream_progress(MPIX_STREAM_NULL));
            } else {
                int flag;
                MPI_SAFECALL(MPI_Test(&tls_cont_req, &flag, MPI_STATUS_IGNORE));
                if (flag) {
                    MPI_Start(&tls_cont_req);
                }
            }
            return true;
        }
        void init_thread() override {
            MPI_SAFECALL(MPIX_Continue_init(0, 0, MPI_INFO_NULL, &tls_cont_req));
            MPI_SAFECALL(MPI_Start(&tls_cont_req));
        }
        void free_thread() override {
            if (tls_cont_req != MPI_REQUEST_NULL) {
                MPI_SAFECALL(MPI_Request_free(&tls_cont_req));
            }
        }
    private:
        static thread_local MPI_Request tls_cont_req;
        struct config_t {
            int use_cont_imm;
            int use_cont_forget;
        } config;
    };
    thread_local MPI_Request comp_manager_continue_t::tls_cont_req = MPI_REQUEST_NULL;
#endif

}

#endif //MPI_CONTINUE_EXAMPLE_COMMON_MPI_HPP
