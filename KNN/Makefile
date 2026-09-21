CC=g++
MNIST_ML_ROOT := $(shell cd .. && pwd)
SRC := ./src
INCLUDE_DIR := ./include
ROOT_INCLUDE := $(MNIST_ML_ROOT)/include
ROOT_LIB := $(MNIST_ML_ROOT)/lib
CFLAGS := -std=c++11 -DEUCLID -O2

# Headers of this folder and of the root project: if one changes, main is rebuilt
HEADERS := $(wildcard $(INCLUDE_DIR)/*.hpp) $(wildcard $(ROOT_INCLUDE)/*.hpp)

.PHONY: all clean

all : main

main : $(SRC)/knn.cc $(HEADERS)
	$(CC) $(CFLAGS) $(SRC)/knn.cc -o main -L$(ROOT_LIB) -I$(INCLUDE_DIR) \
	-I$(ROOT_INCLUDE) -ldata -Wl,-rpath,$(ROOT_LIB)

clean :
	rm -f main
