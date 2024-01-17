#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void isingCuda(int* currentCuda, int* nextCuda, int n) {
    
    // Calculate the global index of the thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n) {
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

void isingSimulation(int n, int k) {
    // Allocate input vectors h_A and h_B in host memory
    int* current = (int*)malloc(n * n * sizeof(int));
    int* next = (int*)malloc(n * n * sizeof(int));

    // Allocate vectors in device memory
    int* currentCuda;
    cudaMalloc(&currentCuda, n * n * sizeof(int));
    int* nextCuda;
    cudaMalloc(&nextCuda, n * n * sizeof(int));

     // Initialize the current state with random values
     srand(1);
    //srand(time(NULL));
    for (int i = 0; i < n * n; i++) {
        current[i] = rand() % 2 == 0 ? -1 : 1;
    }

    // Copy vectors from host memory to device memory
    cudaMemcpy(currentCuda, current, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(nextCuda, next, n * n * sizeof(int), cudaMemcpyHostToDevice);

   
    // Declare cuda parameters
    int numBlocks = 1; // Number of blocks
    dim3 threadsPerBlock(n, n); // Number of threads per block

    // Perform k iterations
    for (int iter = 0; iter < k; iter++) {
        // Call the kernel function with n blocks and n threads per block
        isingCuda<<<numBlocks, threadsPerBlock>>>(currentCuda, nextCuda, n);

        // Swap the pointers of current and next arrays
        int* temp = currentCuda;
        currentCuda = nextCuda;
        nextCuda = temp;
        
        // Wait for all threads to finish
        cudaDeviceSynchronize();
    }
    cudaMemcpy(current, currentCuda, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(next, nextCuda, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    /* Print the matrix
    printf("|");
    for (int i = 0; i < n * n; i++) {
        if (current[i] == 1){
            printf(" %d|", current[i]);
        } 
        else{
            printf("%d|", current[i]);
        }
        if (((i+1)%n)==0 && i!=0 && i!=n*n-1) printf("\n|");
        if (((i+1)%(n*n))==0 && i!=0) printf("\n");
    }*/
    // Free the memory
    free(current);
    free(next);
    cudaFree(currentCuda);
    cudaFree(nextCuda);
}

int main() {
    int n = 1300000; // Size of the Ising model
    int k = 5; // Number of iterations
    clock_t start, end;
    double cpu_time_used;

    start = clock();

    isingSimulation(n, k);
    
    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time taken: %f seconds\n", cpu_time_used);
    return 0;
}