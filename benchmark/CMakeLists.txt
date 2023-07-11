add_executable(parcel parcel_main.cpp)
target_link_libraries(parcel PRIVATE MPI::MPI_CXX Threads::Threads)