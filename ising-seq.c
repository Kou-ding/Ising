#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void isingSimulation(int n, int k) {
    // Allocate memory for two arrays
    int* current = (int*)malloc(n * n * sizeof(int));
    int* next = (int*)malloc(n * n * sizeof(int));

    // Initialize the current state with random values
    srand(1);
    //srand(time(NULL));
    for (int i = 0; i < n * n; i++) {
        current[i] = rand() % 2 == 0 ? -1 : 1;
    }

    // Perform k iterations
    for (int iter = 0; iter < k; iter++) {
        // Iterate over each cell
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                // Calculate the sum of neighboring spins
                int sum = current[((i - 1 + n) % n)*n + j] +
                          current[((i + 1) % n)*n + j] +
                          current[i*n + (j - 1 + n) % n] +
                          current[i*n + (j + 1) % n] + 
                          current[i*n + j];

                // Update the next state based on the sum
                next[i*n + j] = sum > 0 ? 1 : -1;
            }
        }
        // Print the matrix
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
        }
        
        // Swap the pointers of current and next arrays
        int* temp = current;
        current = next;
        next = temp;
    }
    // Free the memory
    free(current);
    free(next);
}

int main() {
    int n = 4; // Size of the Ising model
    int k = 5; // Number of iterations

    isingSimulation(n, k);
    
    return 0;
}