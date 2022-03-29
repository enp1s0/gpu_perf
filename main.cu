#include <iostream>
#include <chrono>
#include <cutf/cublas.hpp>
#include <cutf/memory.hpp>
#include <cutf/type.hpp>
#include <gpu_monitor/gpu_monitor.hpp>

using compute_t = float;

constexpr std::size_t min_log_N = 8;
constexpr std::size_t max_log_N = 14;
constexpr std::size_t C = 1lu << 7;
constexpr std::size_t time_for_measuring_in_sec = 10;

int main() {
	auto mat_a = cutf::memory::get_device_unique_ptr<compute_t>(1lu << (2 * max_log_N));
	auto mat_b = cutf::memory::get_device_unique_ptr<compute_t>(1lu << (2 * max_log_N));
	auto mat_c = cutf::memory::get_device_unique_ptr<compute_t>(1lu << (2 * max_log_N));

	const auto alpha = cutf::type::cast<compute_t>(1);
	const auto beta = cutf::type::cast<compute_t>(1);

	auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();

	std::printf("m,n,k,throughput_in_tflops,avg_power_consumption_in_W,integrated_power_consumption_in_Ws,test_count\n");
	for (unsigned log_N = min_log_N; log_N <= max_log_N; log_N++) {
		const std::size_t N = 1lu << log_N;
		CUTF_CHECK_ERROR(cudaDeviceSynchronize());
		const auto start_clock = std::chrono::system_clock::now();
		for (std::size_t c = 0; c < C; c++) {
			CUTF_CHECK_ERROR(cutf::cublas::gemm(
						*cublas_handle.get(),
						CUBLAS_OP_N, CUBLAS_OP_N,
						N, N, N,
						&alpha,
						mat_a.get(), N,
						mat_b.get(), N,
						&beta,
						mat_c.get(), N
						));
		}
		CUTF_CHECK_ERROR(cudaDeviceSynchronize());
		const auto end_clock = std::chrono::system_clock::now();

		const auto elapsed_time_0 = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6;

		const std::size_t test_count = std::max<std::size_t>(time_for_measuring_in_sec / elapsed_time_0 * C, 5);

		const auto profiling_result = mtk::gpu_monitor::measure_power_consumption(
        [&]() {
            cudaDeviceSynchronize();
						for (std::size_t c = 0; c < test_count; c++) {
							CUTF_CHECK_ERROR(cutf::cublas::gemm(
										*cublas_handle.get(),
										CUBLAS_OP_N, CUBLAS_OP_N,
										N, N, N,
										&alpha,
										mat_a.get(), N,
										mat_b.get(), N,
										&beta,
										mat_c.get(), N
										));
						}
            cudaDeviceSynchronize();
        },
        50
    );
		const auto elapsed_time = mtk::gpu_monitor::get_elapsed_time(profiling_result);
		const auto integrated_power_consumption = mtk::gpu_monitor::get_integrated_power_consumption(profiling_result);
		std::printf("%lu,%lu,%lu,%e,%e,%e,%lu\n",
				N, N, N,
				(2 * N * N * N) / (elapsed_time_0 / C) * 1e-12,
				integrated_power_consumption / elapsed_time,
				integrated_power_consumption / test_count,
				test_count
				);

		std::fflush(stdout);
	}
}
