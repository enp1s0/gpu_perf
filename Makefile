NVCC=nvcc
NVCCFLAGS=-std=c++14 -lcublas -I./cutf -arch=sm_60
TARGET=gemm_perf

$(TARGET):main.cu
	$(NVCC) $< $(NVCCFLAGS) -o $@

clean:
	rm -f $(TARGET)
