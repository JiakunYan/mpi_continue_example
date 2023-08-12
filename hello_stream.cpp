#include <iostream>
#include <vector>
#include "mpi.h"


int main()
{
#ifdef MPIX_STREAM_NULL
    int rank, nranks;
    MPI_Init(nullptr, nullptr);

    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);

    const int nstreams = 8;
    std::vector<MPIX_Stream> streams;
    std::vector<MPI_Comm> comms;
    for (int i = 0; i < nstreams; ++i) {
        MPIX_Stream stream;
        MPIX_Stream_create(MPI_INFO_NULL, &stream);
        streams.push_back(stream);
        MPI_Comm comm;
        MPIX_Stream_comm_create(MPI_COMM_WORLD, stream, &comm);
        comms.push_back(comm);
    }

    printf("Hello world from %d/%d\n", rank, nranks);

    for (int i = 0; i < nstreams; ++i) {
        MPI_Comm_free(&comms[i]);
        MPIX_Stream_free(&streams[i]);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
#endif
    return 0;
}