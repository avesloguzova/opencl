BIN := bin

OS := $(shell uname)

ifeq ($(OS),Darwin)
CXX=g++
CXX_FLAGS=--std=c++11 -g -framework OpenCL
else
CXX=g++
CXX_FLAGS=--std=c++11 -g -lcl -L/usr/lib/beignet
endif

all: before program

before:
	mkdir -p $(BIN)

program: main.cpp
	$(CXX) $(CXX_FLAGS) $? -o $(BIN)/$@

clean:
	rm -rf $(BIN)
