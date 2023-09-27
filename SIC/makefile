NVCC=nvcc
CUDAFLAGS= -arch=sm_60
OPT= -g -G
RM=/bin/rm -f
all: SIC

main: SIC.o SIC_Random.o SIC_Generate.o SIC_Rayleigh.o SIC_OPA.o
	${NVCC} ${OPT} -o main SIC.o SIC_Random.o SIC_Generate.o SIC_Rayleigh.o SIC_OPA.o

SIC_OPA.o: SIC_header.cuh SIC_OPA.cpp
		${NVCC} ${OPT} ${CUDAFLAGS} -std=c++11 -c SIC_OPA.cpp

SIC_Rayleigh.o: SIC_header.cuh SIC_Rayleigh.cpp
	${NVCC} ${OPT} ${CUDAFLAGS} -std=c++11 -c SIC_Rayleigh.cpp

SIC_Generate.o: SIC_header.cuh SIC_Generate.cpp
	${NVCC} ${OPT} ${CUDAFLAGS} -std=c++11 -c SIC_Generate.cpp

SIC_Random.o: SIC_header.cuh SIC_Random.cpp
	${NVCC} ${OPT} ${CUDAFLAGS} -std=c++11 -c SIC_Random.cpp

SIC.o: SIC_header.cuh SIC.cu
	$(NVCC) ${OPT} $(CUDAFLAGS)	-std=c++11 -c SIC.cu -lcufft

SIC: SIC.o SIC_Random.o SIC_Generate.o SIC_Rayleigh.o SIC_OPA.o
	${NVCC} ${CUDAFLAGS} -o SIC SIC.o SIC_Random.o SIC_Generate.o SIC_Rayleigh.o SIC_OPA.o -lcufft

clean:
	${RM} *.o Main
