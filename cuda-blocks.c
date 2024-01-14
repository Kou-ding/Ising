#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void isingCuda(int* currentCuda, int* nextCuda, int n) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // Calculate the sum of neighboring spins
            int sum = currentCuda[((i - 1 + n) % n)*n + j] +
                    currentCuda[((i + 1) % n)*n + j] +
                    currentCuda[i*n + (j - 1 + n) % n] +
                    currentCuda[i*n + (j + 1) % n] + 
                    currentCuda[i*n + j];
            // Update the next state based on the sum
            nextCuda[i*n + j] = sum > 0 ? 1 : -1;
        }
    }
    // Swap the pointers of current and next arrays
        int* temp = current;
        current = next;
        next = temp;
}

void isingSimulation(int n, int k) {
    // Allocate memory for two arrays
    int* current;
    int* next;
    int* currentCuda;
    int* nextCuda;

    cudaMalloc(&currentCuda, n * n * sizeof(int));
    cudaMalloc(&nextCuda, n * n * sizeof(int));

    cudaMemcpy(currentCuda, current, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(nextCuda, next, n * n * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize the current state with random values
    srand(1);
    //srand(time(NULL));
    for (int i = 0; i < n * n; i++) {
        current[i] = rand() % 2 == 0 ? -1 : 1;
    }

    // Perform k iterations
    for (int iter = 0; iter < k; iter++) {
        // Call the kernel function with n blocks and n threads per block
        isingCuda<<<n, n>>>(currentCuda, nextCuda, n);

        // Wait for all threads to finish
        cudaDeviceSynchronize();
    }

    // Free the memory
    free(current);
    free(next);
    cudaFree(currentCuda);
    cudaFree(nextCuda);
}

int main() {
    int n = 4; // Size of the Ising model
    int k = 5; // Number of iterations

    isingSimulation(n, k);
    
    return 0;
}