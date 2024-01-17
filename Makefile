# $^ (all prerequisites): refers to whatever is after the ":"
# $@ (target): refers to whatever is before the ":"
# $< (first prerequisite): refers to the first thing that is after the ":"

CC = gcc
CFLAGS = -Wall -std=c99
NVCC = nvcc
CUFLAGS = -O3

# all is the default action in makefiles
all: seq threads blocks shared

# Sequantial
seq: ising-seq.c
	$(CC) $(CFLAGS) $^ -o $@

# GPU with one thread per moment
threads: cuda-threads.cu
	$(NVCC) $(CUFLAGS) $^ -o $@

# GPU with one thread computing a block of moments
blocks: cuda-blocks.cu
	$(NVCC) $(CUFLAGS) $^ -o $@

# GPU with multiple thread sharing common input moments
shared: cuda-shared.cu
	$(NVCC) $(CUFLAGS) $^ -o $@


clean:
	rm -f seq threads blocks shared