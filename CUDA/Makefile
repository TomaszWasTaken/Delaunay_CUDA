NVCC:=nvcc
FLAGS:=-O4 -ccbin=/opt/gcc-11.4/bin/g++-11.4 -lstdc++ -gencode arch=compute_75,code=sm_75 --use_fast_math

all: main 

main.o: main.cu
	$(NVCC) -c $< $(FLAGS)

point_insertion.o: point_insertion.cu point_insertion.h
	$(NVCC) -c $< $(FLAGS)

flipping.o: flipping.cu flipping.h
	$(NVCC) -c $< $(FLAGS)
	
morton.o: morton.cu morton.h
	$(NVCC) -c $< $(FLAGS)

setup.o: setup.cu setup.h
	$(NVCC) -c $< $(FLAGS)

triangulation.o: triangulation.cu triangulation.h
	$(NVCC) -c $< $(FLAGS)

main: main.o point_insertion.o flipping.o morton.o setup.o triangulation.o
	$(NVCC) -o $@ $^ $(FLAGS)

.Phony: clean

clean:
	rm -rf *.o *.nsys-rep *.sqlite main
