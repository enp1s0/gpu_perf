NVCC=nvcc
NVCCFLAGS=-std=c++14 -lcublas -I./cutf/include -I./gpu_monitor/include -arch=sm_80
NVCCFLAGS+=-I./gpu_monitor/include -L./gpu_monitor/build -lgpu_monitor -lnvidia-ml
TARGET=gemm_perf

$(TARGET):main.cu
	$(NVCC) $< $(NVCCFLAGS) -o $@

clean:
	rm -f $(TARGET)
