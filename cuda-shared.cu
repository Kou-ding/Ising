#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void isingCuda(int* currentCuda, int* nextCuda, int n, int elementsPerThread) {
    // Declare shared memory
    extern __shared__ int sharedCurrent[];

    // Calculate the global index of the thread
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread loads multiple elements into shared memory
    for (int offset = 0; offset < elementsPerThread; offset++) {
        int idx = globalIdx * elementsPerThread + offset;
        if (idx < n * n) {
            sharedCurrent[idx] = currentCuda[idx];
        }
    }

    // Synchronize to make sure the data is loaded before computation
    __syncthreads();

    // Each thread processes multiple elements
    for (int offset = 0; offset < elementsPerThread; offset++) {
        int idx = globalIdx * elementsPerThread + offset;
        if (idx < n * n) {
            int i = idx / n;
            int j = idx % n;

            // Calculate the sum of neighboring spins using shared memory
            int sum = sharedCurrent[((i - 1 + n) % n)*n + j] +
                    sharedCurrent[((i + 1) % n)*n + j] +
                    sharedCurrent[i*n + (j - 1 + n) % n] +
                    sharedCurrent[i*n + (j + 1) % n] + 
                    sharedCurrent[i*n + j];
            // Update the next state based on the sum
            nextCuda[i*n + j] = sum > 0 ? 1 : -1;
        }
    }
}

void isingSimulation(int n, int k, int numThreads) {
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
    //dim3 threadsPerBlock(n, n); // Number of threads per block
    // Calculate the number of elements processed by each thread
    int elementsPerThread = (n * n + numThreads - 1) / numThreads;

    // Perform k iterations
    for (int iter = 0; iter < k; iter++) {
        // Call the kernel function with n blocks and n threads per block
        isingCuda<<<numBlocks, numThreads, n * n * sizeof(int)>>>(currentCuda, nextCuda, n, elementsPerThread);

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
        if (((i+1)%(n*n))==0 && i!=0) printf("\n----------------\n");
    }*/
    // Free the memory
    free(current);
    free(next);
    cudaFree(currentCuda);
    cudaFree(nextCuda);
}

int main() {
    int n; // Size of the Ising model
    int k; // Number of iterations
    int numThreads; // Number of threads
    printf("Enter the size of the Ising model: ");
    scanf("%d", &n);
    printf("Enter the number of iterations: ");
    scanf("%d", &k);
    printf("Enter the number of threads: ");
    scanf("%d", &numThreads);
    
    clock_t start, end;
    double cpu_time_used;

    start = clock();

    isingSimulation(n, k, numThreads);
    
    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time taken: %f seconds\n", cpu_time_used);
    return 0;
}