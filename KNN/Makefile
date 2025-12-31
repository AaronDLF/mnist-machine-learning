CC=g++
MNIST_ML_ROOT := $(shell cd .. && pwd)
SRC := ./src
INCLUDE_DIR := ./include
CFLAGS := -std=c++11 -DEUCLID

all : main

main : $(SRC)/knn.cc
	$(CC) $(CFLAGS) $(SRC)/knn.cc -o main -L$(MNIST_ML_ROOT)/lib -I$(INCLUDE_DIR) \
	-I$(MNIST_ML_ROOT)/include -ldata -Wl,-rpath,$(MNIST_ML_ROOT)/lib

clean :
	rm -f main
