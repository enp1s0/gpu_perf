NVCC=nvcc
NVCCFLAGS=-std=c++14 -lcublas -I./cutf/include -arch=sm_80
TARGET=gemm_perf

$(TARGET):main.cu
	$(NVCC) $< $(NVCCFLAGS) -o $@

clean:
	rm -f $(TARGET)
