# $^ (all prerequisites): refers to whatever is after the ":"
# $@ (target): refers to whatever is before the ":"
# $< (first prerequisite): refers to the first thing that is after the ":"

CC = gcc
CFLAGS = -Wall

# all is the default action in makefiles
all: seq threads blocks shared

# Sequantial
seq: ising-seq.c 
	$(CC) $(CFLAGS) $^ -o $@

# GPU with one thread per moment
threads: cuda-threads.c
	$(CC) $(CFLAGS) $^ -o $@

# GPU with one thread computing a block of moments
blocks: cuda-blocks.c
	$(CC) $(CFLAGS) $^ -o $@

# GPU with multiple thread sharing common input moments
shared: cuda-shared.c
	$(CC) $(CFLAGS) $^ -o $@


clean:
	rm -f seq threads blocks shared