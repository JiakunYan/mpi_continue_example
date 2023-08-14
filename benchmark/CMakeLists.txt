add_executable(parcel parcel_main.cpp)
target_link_libraries(parcel PRIVATE MPI::MPI_CXX Threads::Threads)

add_executable(local local_main.cpp)
target_link_libraries(local PRIVATE Threads::Threads)

add_executable(microbenchmark0_failed_polling microbenchmark0_failed_polling.cpp)
target_link_libraries(microbenchmark0_failed_polling PRIVATE MPI::MPI_CXX)

add_executable(microbenchmark1_recv_order microbenchmark1_recv_order.cpp)
target_link_libraries(microbenchmark1_recv_order PRIVATE MPI::MPI_CXX)

add_executable(microbenchmark2 microbenchmark2.cpp)
target_link_libraries(microbenchmark2 PRIVATE MPI::MPI_CXX)

add_executable(microbenchmark3_compare microbenchmark3_compare.cpp)
target_link_libraries(microbenchmark3_compare PRIVATE MPI::MPI_CXX)

add_executable(microbenchmark4_succeed_overhead microbenchmark4_succeed_overhead.cpp)
target_link_libraries(microbenchmark4_succeed_overhead PRIVATE MPI::MPI_CXX)