NVCC=nvcc
NVCCFLAGS=-std=c++14 -lcublas -I./cutf
TARGET=gemm_perf

$(TARGET):main.cu
	$(NVCC) $< $(NVCCFLAGS) -o $@

clean:
	rm -f $(TARGET)
