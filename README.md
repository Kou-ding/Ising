# Ising

### Homework 3

In this assignment we are tasked to code the evolution of an Ising model. The implementations are initially in C and progressively GPU processing elements are added, specifically speaking CUDA C libraries, to make the whole simulation parallel and thus faster.

### Sequential

First we prototype the code in basic C to get a grasp of the foundementals. The Ising model we are going code involves around a matrix in which each element has its own atomic spin. The elements follow Periodic boundary conditions meaning that there are no edge elements. The matrix wraps around both in the vertical and the horizontal directions creating an infinately looping cell of atomic spins. The algorithm we apply takes the spin of an element's neighbors, including itself, and then if the majority is -1 the atom's spin becomes -1 in the next iteration. Also if the majority of the spins is 1 it becomes 1. The program runs for k iterations of this process and the matrix is nxn. For the actual code, we initialize two arrays **int\* current** and **int\* next**. We are going to read the values of the **int\* current** and write them to **int\* next**. Afterwards in the code we are going to be swapping the two to begin the next iteration of the code. Something that we need to pay attention to is the way we use a 1 dimensional pointer to represent the 2d matrix so that we can more easily implement the periodic boundaries and also speed up calculation omitting if then clauses on edge elements which would slow down the code.

#### Periodic boundary conditions

To deal with the edge case we divise this way of checking neighbor spins:
- current[((i - 1 + n) % n)*n + j] 
    - the element that is up
- current[((i + 1) % n)*n + j] 
    - the element that is down
- current[i*n + (j - 1 + n) % n] 
    - the element on the left
- current[i*n + (j + 1) % n] 
    - the element on the right
- current[i*n + j];
    - the element itself

Afterwards we some them all up to get the final atomic spin of the element accoring to the following condition:
```c
next[i*n + j] = sum > 0 ? 1 : -1;
```
Finally after all k iteration finish we free the two pointers and return 0.

### GPU with one thread per moment

### GPU with one thread computing a block of moments

### GPU with multiple thread sharing common input moments

### Tutorial
The code is divided into 4 files:
- ising-seq.c
```bash
make seq
```
- cuda-moment.c
```bash
make mom
```
- cuda-block.c
```bash
make blo
```
- cuda-multithreads.c
```bash
make mul
```