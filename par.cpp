#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <mpi.h>
#include <omp.h>
#include <algorithm>

std::vector<bool> sieve(int N, int start, int stop, int id)
{
    std::vector<bool> primeVec(stop - start + 1, true);
    if (start == 0)
    {
        primeVec[0] = false;
        primeVec[1] = false;
    }

    int k = 2;

    while (pow(k, 2) <= N)
    {

#pragma omp parallel for // this is the part we can parallelize!
        // calculate the first multiple of k within [start, stop] e.g. if id = 1 with processes = 8 and n = 100 and k = 3, than this would return the first multiple of 3 between 100/8 rounded down and 100/8 *2, which would be
        // the first multiple between of 3 between 12 and 24 which is 12. With the same id but processes = 4 it would be the first multiple between 25 and 50, so 27.
        // the -1 in (start + k - 1) ensures that we round correctly as to make the start inclusive, if it were to be removed the first scenario would return a 15 instead of a 12!
        for (int i = std::max(k, (start + k - 1) / k) * k; i <= stop; i += k)
        {
            #pragma omp critical // inconsistent prime count without this because of synchronicity issues
            primeVec[i - start] = false; // start has to be substracted because i am now using local vectors instead of the full vectors like in the seq version.
        }

        if (id == 0)
        {                                                                        // this can still function the same but we want to broadcast the next k value to all processes
            auto it = std::find(primeVec.begin() + k + 1, primeVec.end(), true); // find the iterator pointing to the first true value (after our current k value)
            k = std::distance(primeVec.begin(), it);                             // convert it to the index
        }

        MPI_Bcast(&k, 1, MPI_LONG, 0, MPI_COMM_WORLD); // broadcast the next value of k to all processes
    }

    return primeVec;
}

int main(int argc, char *argv[])
{

    int nProcesses;
    int id;
    int N = 10000;
    int start;
    int stop;
    int global_count = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    double startTime = MPI_Wtime();

    start = (N / nProcesses) * id;
    stop = (N / nProcesses) * (id + 1) - 1;
    if (id == nProcesses - 1)
    {
        stop = N - 1; // last process uses the remaining numbers
    }

    std::vector<bool> primeVec = sieve(N, start, stop, id);

    int localCount = std::count(primeVec.begin(), primeVec.end(), true); // local count of true values

    MPI_Reduce(&localCount, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD); // sum up the counts of prime numbers from all processes

    double duration = MPI_Wtime() - startTime;

    if (id == 0)
    {
        std::cout << "Found total prime numbers: " << global_count << std::endl;
        std::cout << "Total time taken: " << duration << " seconds" << std::endl;
        std::cout << "nProcesses: " << nProcesses << std::endl;
    }

    MPI_Finalize();
}