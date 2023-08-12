add_executable(parcel parcel_main.cpp)
target_link_libraries(parcel PRIVATE MPI::MPI_CXX Threads::Threads)

add_executable(local local_main.cpp)
target_link_libraries(local PRIVATE Threads::Threads)

add_executable(microbenchmark0 microbenchmark0.cpp)
target_link_libraries(microbenchmark0 PRIVATE MPI::MPI_CXX)

add_executable(microbenchmark1 microbenchmark1.cpp)
target_link_libraries(microbenchmark1 PRIVATE MPI::MPI_CXX)

add_executable(microbenchmark2 microbenchmark2.cpp)
target_link_libraries(microbenchmark2 PRIVATE MPI::MPI_CXX)

add_executable(microbenchmark3 microbenchmark3.cpp)
target_link_libraries(microbenchmark3 PRIVATE MPI::MPI_CXX)

add_executable(microbenchmark4 microbenchmark4.cpp)
target_link_libraries(microbenchmark4 PRIVATE MPI::MPI_CXX)