# Makefile for MyDelaunayTriangulation
# author: Tomasz Kwasniewicz
# date  : 2023-06-05

SHELL = /bin/sh

# Set up compiler
CC:=g++
CFLAGS:=-W -Wall -Wextra -pedantic-errors -std=c++20 -O3 -g 

all: main 

predicates.o: predicates.c predicates.h
	gcc -c $< -O3

main.o: main.cpp
	$(CC) -c $< $(CFLAGS)

points.o: points.cpp points.h
	$(CC) -c $< $(CFLAGS)

triangulation.o: triangulation.cpp triangulation.h
	$(CC) -c $< $(CFLAGS)

main: main.o points.o triangulation.o predicates.o
	$(CC) -o $@ $^ $(CFLAGS)

.Phony: clean

clean:
	rm -rf *.o *.dat main 
