OS=$(shell uname)

ifeq ($(OS),Darwin)
CXX=g++
CXX_FLAGS=--std=c++11 -framework OpenCL
else
CXX=g++
CXX_FLAGS=--std=c++11 -lcl -L/usr/lib/beignet
endif

all: program

program: main.cpp
	$(CXX) $(CXX_FLAGS) $? -o $@

clean:
	rm program
