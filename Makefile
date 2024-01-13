# $^ (all prerequisites): refers to whatever is after the ":"
# $@ (target): refers to whatever is before the ":"
# $< (first prerequisite): refers to the first thing that is after the ":"

CC = gcc
CFLAGS = -Wall

# all is the default action in makefiles
all: seq mom blo mul

# Sequantial
seq: ising-seq.c 
	$(CC) $(CFLAGS) $^ -o $@

# GPU with one thread per moment
mom: cuda-moment.c
	$(CC) $(CFLAGS) $^ -o $@

#  GPU with one thread computing a block of moments
blo: cuda-block.c
	$(CC) $(CFLAGS) $^ -o $@

# GPU with multiple thread sharing common input moments
mul: cuda-multithreads.c
	$(CC) $(CFLAGS) $^ -o $@


clean:
	rm -f seq mom blo mul