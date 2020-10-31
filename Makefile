NVCC=nvcc
NVCCFLAGS=-std=c++14 -lcublas
TARGET=gemm_perf

$(TARGET):main.cu
	$(NVCC) $< $(NVCCFLAGS) -o $@

clean:
	rm -f $(TARGET)
