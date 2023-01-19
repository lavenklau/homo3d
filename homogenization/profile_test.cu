#include "cuda_runtime.h"
#include "culib/vector_intrinsic.cuh"
#include "culib/lib.cuh"
#include "device_launch_parameters.h"
#include <chrono>

using namespace culib;

__global__ void test_vadd4_kernel(char4* pchar4, int arr_size, bool use_simd) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= arr_size)  return;
	if (use_simd) {
		char4 c4 = pchar4[tid];
		char4 sum = make_char4(1, 1, 1, 1);
		sum = vadd4(sum, c4);
		pchar4[tid] = sum;
	} else {
		char4 c4 = pchar4[tid];
		char4 sum = make_char4(1, 1, 1, 1);
		sum.x += c4.x;
		sum.y += c4.y;
		sum.z += c4.z;
		sum.w += c4.w;
		pchar4[tid] = sum;
	}
}

__global__ void example1(unsigned int* pchar4) {
	//pchar4[0] = vadd4(pchar4[0], make_char4(11, 11, 11, 11));
	pchar4[0] = __vadd4(pchar4[0], 0x01010101);
}

__global__ void test_alignment_kernel(int arr_size, double* pdata) {
	size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= arr_size) return;
	pdata[tid] = pdata[tid] + 1;
}

__global__ void example2(int* p) {

}

void test_vadd4(void) {
	char4* pchar4;
	int arr_size = 10000;
	cudaMalloc(&pchar4, arr_size);

	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, arr_size, 512);

	// warm up
	test_vadd4_kernel << <grid_size, block_size >> > (pchar4, arr_size, false);
	cudaDeviceSynchronize();
	cuda_error_check;

	auto t0 = std::chrono::high_resolution_clock::now();
	test_vadd4_kernel << <grid_size, block_size >> > (pchar4, arr_size, false);
	cudaDeviceSynchronize();
	cuda_error_check;
	auto t1 = std::chrono::high_resolution_clock::now();

	test_vadd4_kernel << <grid_size, block_size >> > (pchar4, arr_size, true);
	cudaDeviceSynchronize();
	cuda_error_check;
	auto t2 = std::chrono::high_resolution_clock::now();

	std::cout << "Without simd : " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0 << " ms" << std::endl;
	std::cout << "With    simd : " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000.0 << " ms" << std::endl;
}

void warm_up(void) {
	char4* pchar4;
	int arr_size = 10000;
	cudaMalloc(&pchar4, arr_size);

	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, arr_size, 512);

	// warm up
	test_vadd4_kernel << <grid_size, block_size >> > (pchar4, arr_size, false);
	cudaDeviceSynchronize();
	cuda_error_check;

	cudaFree(pchar4);
}

void test_alignment(void) {
	//warm_up();
	double* pdata;
	int arr_size = 10010000;
	cudaMalloc(&pdata, sizeof(double) * arr_size);

	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, 10000000, 512);

	int nonAlignOffset = 7;

	// warm up
	for (int i = 0; i < 1; i++) {
		test_alignment_kernel << <grid_size, block_size >> > (10000000, pdata);
		cudaDeviceSynchronize();
		test_alignment_kernel << <grid_size, block_size >> > (10000000, pdata + nonAlignOffset);
		cudaDeviceSynchronize();
	}

	auto t0 = std::chrono::high_resolution_clock::now();
	test_alignment_kernel << <grid_size, block_size >> > (10000000, pdata );
	cudaDeviceSynchronize();
	cuda_error_check;
	auto t1 = std::chrono::high_resolution_clock::now();

	test_alignment_kernel << <grid_size, block_size >> > (10000000, pdata + nonAlignOffset);
	cudaDeviceSynchronize();
	cuda_error_check;
	auto t2 = std::chrono::high_resolution_clock::now();

	cudaFree(pdata);

	std::cout << "Alignment     time cost : " << std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000. << " ms" << std::endl;
	std::cout << "Non-Alignment time cost : " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() / 1000. << " ms" << std::endl;
}

template<typename Func>
struct LambdaWrapper {
	//auto kerFunc(void) {
	//	return [=] __device__(int tid) {
	//		return tid;
	//	};
	//}
	Func func;
	LambdaWrapper(Func f) : func(f) {}
	__device__ auto operator()(int tid) {
		return func(tid);
	}
};

template<typename Func>
LambdaWrapper<Func> make_wrapper(Func f) {
	return LambdaWrapper<Func>(f);
}

void test_nest_lambda(void) {
	float initval = 1.2;
	auto ker1 = [=] __device__(int tid) {
		return tid + initval;
	};
	//auto ker2 = [=] __device__(int tid) {
	//	return ker1(tid) * 2;
	//};
	auto kernel = make_wrapper(ker1);

	//Lambda lam;
	//auto ker3 = lam.kerFunc();
	float* farray;
	cudaMalloc(&farray, sizeof(float) * 100);
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, 100, 256);
	map <<< grid_size, block_size >> > (farray, 100, kernel);
	cudaDeviceSynchronize();
	cuda_error_check;
	std::vector<float> fhost(100);
	cudaMemcpy(fhost.data(), farray, sizeof(float) * 100, cudaMemcpyDeviceToHost);
	for (int i = 0; i < fhost.size(); i++) {
		printf("%f\n", fhost[i]);
	}
	cudaFree(farray);
}


void cuda_test(void) {
	//test_vadd4();
	//test_alignment();
	//test_nest_lambda();
}
