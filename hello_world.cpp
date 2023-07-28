#include <iostream>
#include "mpi.h"

int main()
{
    int rank, nranks;
    MPI_Init(nullptr, nullptr);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Hello world from %d/%d\n", rank, nranks);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}