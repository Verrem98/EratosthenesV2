#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <mpi.h>
#include <algorithm>

std::vector<bool> sieve(int N)
{
    std::vector<bool> primeVec(N, true);
    primeVec[0] = false;
    primeVec[1] = false;

    int k = 2;

    while (pow(k, 2) <= N)
    {
        for (int i = pow(k, 2); i < N; i += k)
        {
            primeVec[i] = false;
        }

        auto it = std::find(primeVec.begin() + k + 1, primeVec.end(), true); // find the iterator pointing to the first true value (after our current k value)
        int index = std::distance(primeVec.begin(), it);                     // convert it to the index

        k = index;
    }

    return primeVec;
}

int main(int argc, char *argv[])
{

    int startTime = -1;
    int totalTime = -1;
    int N = 1000;

    MPI_Init(&argc, &argv);

    startTime = MPI_Wtime();
    std::vector<bool> outputVec = sieve(N);
    totalTime = MPI_Wtime() - startTime;

    int count = std::count(outputVec.begin(), outputVec.end(), true); // count the number of trues in the vector

    std::cout << "Number of trues: " << count << std::endl;

    std::cout << "Total runtime:: " << totalTime << std::endl;

    return 0;
}