CC=g++
INCLUDE_DIR := ./include
SRC := ./src
CFLAGS := -std=c++11 -g
LIB_DATA := libdata.so

all : $(LIB_DATA)

$(LIB_DATA) : libdir objdir obj/data_handler.o obj/data.o
	$(CC) $(CFLAGS) -o ./lib/$(LIB_DATA) obj/*.o
	rm -r ./obj

libdir :
	mkdir -p ./lib

objdir :
	mkdir -p ./obj

obj/data_handler.o : $(SRC)/data_handler.cc
	$(CC) -fPIC $(CFLAGS) -o obj/data_handler.o -I$(INCLUDE_DIR) -c $(SRC)/data_handler.cc

obj/data.o : $(SRC)/data.cc
	$(CC) -fPIC $(CFLAGS) -o obj/data.o -I$(INCLUDE_DIR) -c $(SRC)/data.cc

clean:
	rm -r ./lib
	rm -r ./obj
