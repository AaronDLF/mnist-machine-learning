CC=g++
INCLUDE_DIR := ./include
SRC := ./src
OBJ_DIR := ./obj
LIB_DIR := ./lib
CFLAGS := -std=c++11 -g -O2
LIB_DATA := libdata.so

# Every .cc listed here becomes obj/<name>.o and goes inside libdata.so
SOURCES := $(SRC)/data_handler.cc $(SRC)/data.cc $(SRC)/common.cc
OBJECTS := $(patsubst $(SRC)/%.cc,$(OBJ_DIR)/%.o,$(SOURCES))

# Every header: if one changes, the objects are rebuilt
HEADERS := $(wildcard $(INCLUDE_DIR)/*.hpp)

.PHONY: all clean

all : $(LIB_DIR)/$(LIB_DATA)

# Link step: all .o files -> one shared library
$(LIB_DIR)/$(LIB_DATA) : $(OBJECTS)
	mkdir -p $(LIB_DIR)
	$(CC) -shared -o $@ $(OBJECTS)

# Compile step: one rule for every obj/X.o built from src/X.cc
$(OBJ_DIR)/%.o : $(SRC)/%.cc $(HEADERS)
	mkdir -p $(OBJ_DIR)
	$(CC) -fPIC $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

clean :
	rm -rf $(LIB_DIR) $(OBJ_DIR)
