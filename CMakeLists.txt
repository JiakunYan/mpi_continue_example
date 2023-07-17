cmake_minimum_required(VERSION 3.16)
project(mpi_continue_example)

find_package(MPI REQUIRED)

add_executable(hello_world hello_world.cpp)
target_link_libraries(hello_world PRIVATE MPI::MPI_CXX)

add_executable(basic basic.c)
target_link_libraries(basic PRIVATE MPI::MPI_C)

find_package(Threads REQUIRED)
add_executable(basic_mt basic_mt.c)
target_link_libraries(basic_mt PRIVATE MPI::MPI_C Threads::Threads)

add_subdirectory(benchmark)