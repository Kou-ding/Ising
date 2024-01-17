#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void isingSimulation(int n, int k) {
    // Allocate memory for two arrays
    int* current = (int*)malloc(n * n * sizeof(int));
    if (current == NULL) {
        printf("current: Memory not available.\n");
        exit(1);
    }

    int* next = (int*)malloc(n * n * sizeof(int));
    if (next == NULL) {
        printf("next: Memory not available.\n");
        exit(1);
    }
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
        /* Print for debugging
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
        */
        // Swap the pointers of current and next arrays
        int* temp = current;
        current = next;
        next = temp;
    }
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
}

int main() {
    int n; // Size of the Ising model
    int k; // Number of iterations
    printf("Enter the size of the Ising model: ");
    scanf("%d", &n);
    printf("Enter the number of iterations: ");
    scanf("%d", &k);
    
    clock_t start, end;
    double cpu_time_used;

    start = clock();
    isingSimulation(n, k);
    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Time taken: %f seconds\n", cpu_time_used);
    return 0;
}