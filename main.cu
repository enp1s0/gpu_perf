#include <iostream>
#include <chrono>
#include <cutf/cublas.hpp>
#include <cutf/memory.hpp>
#include <cutf/type.hpp>

using compute_t = float;

constexpr std::size_t N = 1lu << 14;
constexpr std::size_t C = 1lu << 5;

int main() {
	auto mat_a = cutf::memory::get_device_unique_ptr<compute_t>(N * N);
	auto mat_b = cutf::memory::get_device_unique_ptr<compute_t>(N * N);
	auto mat_c = cutf::memory::get_device_unique_ptr<compute_t>(N * N);

	const auto alpha = cutf::type::cast<compute_t>(1);
	const auto beta = cutf::type::cast<compute_t>(1);

	auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();

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

	std::printf("%15s : %lu\n", "N", N);
	const auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_clock - start_clock).count() * 1e-3;
	std::printf("%15s : %e [s]\n", "time", elapsed_time);
	const auto complexity = 2 * N * N * N * C;
	std::printf("%15s : %e [TFlop/s]\n", "performance", complexity / elapsed_time / (1lu << 40));
}
