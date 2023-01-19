
#pragma once
#ifndef __CULIB_CUH_H
#define __CULIB_CUH_H


#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include<nvfunctional>
#include"array"
#include<iostream>
#include <vector>
#include"curand.h"
#include "cub/cub.cuh"
#include "warp_primitive.cuh"
#include <memory>
#include "cuda/std/type_traits"

#define cuda_error_check do{ \
	auto err = cudaPeekAtLastError(); \
	if (err != 0) { \
		printf("\x1b[31mCUDA error occured at line %d in file %s, error type %s \x1b[0m\n", __LINE__,__FILE__, cudaGetErrorName(err));\
	} \
}while(0)

#define CheckErr(funCall) do { \
	auto Err = funCall;\
	if (Err != CUDA_SUCCESS) {\
		printf("\x1b[31mCUDA error occured at line %d in file %s, error type %s \x1b[0m\n", __LINE__, __FILE__, cudaGetErrorName(Err));\
	}\
} while (0)

#define AbortErr() do { \
	auto Err = cudaPeekAtLastError();\
	if (Err != CUDA_SUCCESS) {\
		printf("\x1b[31mCUDA error occured at line %d in file %s, error type %s \x1b[0m\n", __LINE__, __FILE__, cudaGetErrorName(Err));\
		exit(-1);\
	}\
} while (0)


namespace culib {
	extern void lib_test(void);
	extern int get_device_info(void);
	extern void use4Bytesbank(void);
	extern void use8Bytesbank(void);
	extern void init_cuda(void);


	extern void devArray2matlab(const char* pname, float* pdata, size_t len);
	extern void devArray2matlab(const char* pname, double* pdata, size_t len);
	extern void devArray2matlab(const char* pname, int* pdata, size_t len);

	//typedef double Scaler;

	size_t __host__ __device__ inline round(size_t sz, int K) {
		if (sz % K == 0) {
			return sz;
		}
		else {
			return (sz + K - 1) / K * K;
		}
	}

	size_t __host__ __device__ inline ceilDiv(size_t sz, int K) {
		return round(sz, K) / K;
	}

	inline void make_kernel_param(size_t* block_num, size_t* block_size, size_t num_tasks, size_t prefer_block_size) {
		*block_size = prefer_block_size;
		*block_num = (num_tasks + prefer_block_size - 1) / prefer_block_size;
	}

	struct TempBuffer;
	struct ManagedTempBlock;
	struct TempBufferPool;
	struct TempBufferPlace;

	struct TempBuffer {
	private:
		void* pdata;
		size_t siz;
		bool unified = false;
		friend struct TempBufferPool;
		friend struct TempBufferPlace;
		friend struct ManagedTempBlock;
	private:
		template<typename T> T* data(void) { return (T*)pdata; }
		TempBuffer(const TempBuffer&) = delete;
		bool is_unified(void) { return unified; }
		TempBuffer& operator=(const TempBuffer&) = delete;
	public:
		TempBuffer(size_t size, bool unify = false);
		~TempBuffer(void) { cudaFree(pdata); }
	};

	struct TempBufferPlace {
	private:
		TempBufferPool& pool_;
		int bufferid;
		std::unique_ptr<TempBuffer> buffer;
		friend struct TempBufferPool;
		TempBufferPlace(TempBufferPool& pool, int bufid, std::unique_ptr<TempBuffer> tmp);
		TempBufferPlace(TempBufferPlace&& place);

	public:
		template<typename T = void> T* data(void) { return buffer->template data<T>(); }
		~TempBufferPlace(void);
	};

	struct ManagedTempBlock {
		TempBufferPool& pool;
		int startBlock;
		int endBlock;
		friend struct TempBufferPool;
	private:
		ManagedTempBlock(ManagedTempBlock&& other);
		ManagedTempBlock(const ManagedTempBlock& other) = delete;
		ManagedTempBlock(TempBufferPool& pool_, int startBlock_, int endBlock_);
	public:
		template<typename T> T* data(void) {
			return (T*)(pool.unifiedBuffer->template data<char*>() + startBlock * 32);
		}
		template<typename T> T& rdata(void) {
			return *(T*)(pool.unifiedBuffer->template data<char*>() + startBlock * 32);
		}
		~ManagedTempBlock(void);
	};

	struct TempBufferPool {
		friend struct TempBufferPlace;
		friend struct ManagedTempBlock;
		TempBufferPool(void);
		TempBufferPlace getBuffer(size_t requireSize);
		template<typename T>
		ManagedTempBlock getUnifiedBlock(int n = 1) {
			int requiesize = sizeof(T) * n;
			int nblock = round(requiesize, 32) / 32;
			int startBlock = -1;
			int endBlock = -1;
			for (int i = 0; i < blockPlace32.size() - nblock + 1; ) {
				if (blockPlace32[i]) { i++; }
				else {
					bool allFree = true;
					int j = i + 1;
					for (; j < i + nblock; j++) {
						if (blockPlace32[j]) {
							allFree = false; break;
						}
					}
					if (allFree) {
						startBlock = i;
						break;
					}
					else {
						i = j + 1;
					}
				}
			}
			if (startBlock == -1) {
				printf("\033[31mnot enough unified memory or too large blocks\033[0m\n");
				throw std::runtime_error("unexpected error");
			}
			endBlock = startBlock + nblock;
			//for (int i = startBlock; i < endBlock; i++) {
			//	blockPlace32[i] = true;
			//}
			return ManagedTempBlock(*this, startBlock, endBlock);
		}
	private:
		std::vector<std::unique_ptr<TempBuffer>> buffers;
		// only for very very small segments
		std::unique_ptr<TempBuffer> unifiedBuffer;
		std::vector<bool> blockPlace32;
		std::unique_ptr<TempBuffer>& operator[](size_t id);
	};

	TempBufferPool& getTempPool(void);
	TempBufferPlace getTempBuffer(size_t siz);
	//constexpr size_t node_num = 213 * 213 * 213;
	//constexpr size_t node_task_num = 204;

	template<typename DType, int N>
	struct devArray_t {
		DType _data[N];
		__host__ __device__ DType* data(void) { return _data; }
		__host__ __device__ const DType* data(void) const { return _data; }
		__host__ __device__ const DType& operator[](int k) const {
			return _data[k];
		}

		__host__ __device__  DType& operator[](int k) {
			return _data[k];
		}

		__host__ __device__ devArray_t<DType, N> operator-(const devArray_t<DType, N>& arr2) const {
			devArray_t<DType, N> darr;
			for (int i = 0; i < N; i++) {
				darr[i] = _data[i] - arr2[i];
			}
			return darr;
		}

		__host__ __device__ devArray_t<DType, N> operator+(const devArray_t<DType, N>& arr2) const {
			devArray_t<DType, N> arr;
			for (int i = 0; i < N; i++) {
				arr[i] = _data[i] + arr2[i];
			}
			return arr;
		}

		__host__ void destroy(void) {
			if (std::is_pointer<DType>::value) {
				for (int i = 0; i < N; i++) {
					cudaFree(_data[i]);
				}
			}
		}

		__host__ void create(int nelement) {
			if (std::is_pointer<DType>::value) {
				for (int i = 0; i < N; i++) {
					cudaMalloc(&_data[i], sizeof(DType) * nelement);
				}
			}
		}
	};

	template<typename DType, int N>
	__host__ __device__ devArray_t<DType, N> operator*(DType s, const devArray_t<DType, N>& arr2) {
		devArray_t<DType, N> arr;
		for (int i = 0; i < N; i++) {
			arr[i] = s * arr2[i];
		}
		return arr;
	}


	template <int... Ns> struct nArgs { static constexpr int value = nArgs<Ns...>::value + 1; };

	template <int N1, int... Ns> struct nArgs<N1, Ns...> { static constexpr int value = nArgs<Ns...>::value + 1; };

	template<> struct nArgs<> { static constexpr int value = 0; };

	template<int N1, int... Ns> struct MassProduct { static constexpr int value = N1 * MassProduct<Ns...>::value; };
	template <int N1> struct MassProduct<N1> { static constexpr int value = N1; };

	template<int N, int... Ns> struct FirstArg { static constexpr int value = N; };

	template <int N1, int... Ns> struct LastArg { static constexpr int value = LastArg<Ns...>::value; };

	template<int N> struct LastArg<N> { static constexpr int value = N; };


	template<typename T, int... Ns> struct GraftArray;

	template <typename T, int N1, int... Ns>
	struct GraftArray<T, N1, Ns...>
	{
		int _ldd;
		T* _ptr;
		__host__ __device__ GraftArray(T* pdata, int ldd) :_ptr(pdata), _ldd(ldd) {}
		//template<bool> inline GraftArray<T, Ns...> operator[](int i);

		static constexpr bool value = nArgs<Ns...>::value >= 1;
		template <bool Q = value, typename std::enable_if<Q, GraftArray<T, Ns...>>::type* = nullptr>
		__host__ __device__ inline GraftArray<T, Ns...> operator[](int i)
		{
			//typedef typename std::enable_if<Q::value, GraftArray<T, Ns...>>::type retType;
			return GraftArray<T, Ns...>(_ptr + i * _ldd * MassProduct<Ns...>::value, _ldd);
		}

		template <bool Q = value, typename std::enable_if<!Q, GraftArray<T>>::type* = nullptr>
		__host__ __device__ inline GraftArray<T> operator[](int i)
		{
			//typedef typename std::enable_if<Q::value, GraftArray<T>>::type retType;
			return GraftArray<T>(_ptr + i * _ldd);
		}

	};

	template <typename T>
	struct GraftArray<T>
	{
		T* _ptr;
		__host__ __device__ GraftArray(T* pdata) : _ptr(pdata) {}
		__host__ __device__ inline T& operator[](int i) {
			return _ptr[i];
		}
	};

	//Scaler array_norm2(Scaler* dev_data/*, Scaler* host_data*/, int n, bool root = true);
	__host__ void show_cuSolver_version(void);


	template <typename T, unsigned int blockSize>
	__device__ void warpReduce(volatile T* sdata, unsigned int tid) {
		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
		if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
		if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
		if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
	}

	template <typename T, unsigned int blockSize>
	__global__ void reduce(T* g_idata, T* g_odata, unsigned int n) {
		extern __shared__ T sdata[];
		unsigned int tid = threadIdx.x;
		unsigned int i = blockIdx.x * (blockSize * 2) + tid;
		unsigned int gridSize = blockSize * 2 * gridDim.x;
		sdata[tid] = 0;
		while (i < n) { sdata[tid] += g_idata[i] + g_idata[i + blockSize]; i += gridSize; }
		__syncthreads();
		if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
		if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
		if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
		if (tid < 32) warpReduce<blockSize>(sdata, tid);
		if (tid == 0) g_odata[blockIdx.x] = sdata[0];
	}

	template<typename Generator, typename ReduceOp, typename SType, int BlockSize = 256>
	__global__ void blockMapReduce(int len, Generator gen_, ReduceOp red_, SType invalidValue, void* pTemp, size_t szTemp) {
#define USE_SHARED_MEM_REDUCE 1
		size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
		//using T = decltype(red_(gen_(0), gen_(1)));
		using T = SType;
		ReduceOp red = red_;
		Generator gen = gen_;
		int sizeTempT = szTemp / sizeof(T);
		int warpId = threadIdx.x / 32;
		int laneId = threadIdx.x % 32;
		int gridStride = blockDim.x * gridDim.x;
#if USE_SHARED_MEM_REDUCE
		__shared__ volatile T partialRes[BlockSize / 2];
#else
		__shared__ T partialRes[BlockSize / 32];
#endif
		T voidValue = invalidValue;
		// load data / Grid stride reduce
		T x = voidValue;
		int vid = tid;
		// ToDo : reduce two elements in each loop rather than one
		while (vid < len) {
			x = red(x, gen(vid));
			vid += gridStride;
		}
#if USE_SHARED_MEM_REDUCE
		// block reduce sum
		if (threadIdx.x >= BlockSize / 2) partialRes[threadIdx.x - BlockSize / 2] = x;
		__syncthreads();
		if (threadIdx.x < BlockSize / 2) partialRes[threadIdx.x] = red(partialRes[threadIdx.x], x);
		__syncthreads();
		if (BlockSize / 2 >= 128) {
			if (threadIdx.x < BlockSize / 4) {
				partialRes[threadIdx.x] = red(partialRes[threadIdx.x], partialRes[threadIdx.x + BlockSize / 4]);
			}
			__syncthreads();
		}
		if (BlockSize / 2 >= 256) {
			if (threadIdx.x < BlockSize / 8) {
				partialRes[threadIdx.x] = red(partialRes[threadIdx.x], partialRes[threadIdx.x + BlockSize / 8]);
			}
			__syncthreads();
		}
		if (BlockSize / 2 >= 512) {
			if (threadIdx.x < BlockSize / 16) {
				partialRes[threadIdx.x] = red(partialRes[threadIdx.x], partialRes[threadIdx.x + BlockSize / 16]);
			}
			__syncthreads();
		}
		// warp reduce by shulffe
		if (warpId == 0) {
			x = partialRes[laneId] + partialRes[laneId + 32];
#pragma unroll
			for (int offset = 16; offset > 0; offset >>= 1) {
				x = red(x, shfl_down(x, offset));
			}
			if (laneId == 0) {
				if (blockIdx.x < sizeTempT)  ((T*)pTemp)[blockIdx.x] = x;
			}
		}
#else
		// block  reduce via warp
#pragma unroll
		for (int offset = 16; offset > 0; offset /= 2) {
			T y = shfl_down(x, offset);
			x = red(x, y);
		}
		if (laneId == 0) {
			partialRes[warpId] = x;
		}
		__syncthreads();

		if (BlockSize / 32 > 32) printf("\033[31m BlockSize should be less than 1024\033[0m\n");
		if (warpId == 0) {
			x = voidValue;
			if (laneId < BlockSize / 32) {
				x = partialRes[laneId];
			}
#pragma unroll
			for (int offset = 16; offset > 0; offset >>= 2) {
				T y = shfl_down(x, offset);
				x = red(x, y);
			}
			if (laneId == 0) {
				if (blockIdx.x < sizeTempT) {
					((T*)pTemp)[blockIdx.x] = x;
				}
			}
		}
#endif
	}

	template<typename T>
	__global__ void init_array_kernel(T* array, T value, int array_size) {
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid < array_size) {
			array[tid] = value;
		}
	}

	template<typename T>
	void init_array(T* dev_array, T value, int array_size) {
		size_t grid_dim;
		size_t block_dim;
		make_kernel_param(&grid_dim, &block_dim, array_size, 512);
		init_array_kernel << <grid_dim, block_dim >> > (dev_array, value, array_size);
		cudaDeviceSynchronize();
		cuda_error_check;
	}


	template<typename T, typename Scalar, typename Lam>
	__global__ void map(T* g_data, Scalar* dst, int n, Lam func) {
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid < n) {
			dst[tid] = func(g_data[tid]);
		}
	}

	template<typename T, typename Lam>
	__global__ void map(T* dst, int n, Lam func) {
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid < n) {
			dst[tid] = func(tid);
		}
	}

	template<typename Lam>
	__global__ void map(int n, Lam func) {
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid < n) {
			func(tid);
		}
	}

	template<typename T>
	__global__ void array_min(T* in1, T* in2, T* out, size_t n) {
		size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid < n) {
			T s1 = in1[tid];
			T s2 = in2[tid];
			out[tid] = s1 < s2 ? s1 : s2;
		}
	}

	template<typename T>
	__global__ void array_max(T* in1, T* in2, T* out, size_t n) {
		size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid < n) {
			T s1 = in1[tid];
			T s2 = in2[tid];
			out[tid] = s1 > s2 ? s1 : s2;
		}
	}


	template<typename T, typename Lambda>
	__global__ void traverse(T* dst, int num, Lambda func) {
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid < num) {
			dst[tid] = func(tid);
		}
	}

	template<typename Lambda>
	__global__ void traverse_noret(int num, Lambda func) {
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid < num) {
			func(tid);
		}
	}

	template<typename T, typename Lambda>
	__global__ void traverse(T* dst[], int num_array, int array_size, Lambda func) {
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		int array_id = tid / array_size;
		int entry_id = tid % array_size;
		if (tid < array_size * num_array) {
			dst[array_id][entry_id] = func(array_id, entry_id);
		}
	}

	template<typename Lambda>
	__global__ void traverse_noret(int num_array, int array_size, Lambda func) {
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		int array_id = tid / array_size;
		int entry_id = tid % array_size;
		if (tid < array_size * num_array) {
			func(array_id, entry_id);
		}
	}


	template <typename T, unsigned int blockSize>
	__device__ void warpMax(volatile T* sdata, unsigned int tid) {
		if (blockSize >= 64) { T s = sdata[tid + 32]; if (sdata[tid] < s) sdata[tid] = s; };
		if (blockSize >= 32) { T s = sdata[tid + 16]; if (sdata[tid] < s) sdata[tid] = s; };
		if (blockSize >= 16) { T s = sdata[tid + 8]; if (sdata[tid] < s) sdata[tid] = s; };
		if (blockSize >= 8) { T s = sdata[tid + 4]; if (sdata[tid] < s) sdata[tid] = s; };
		if (blockSize >= 4) { T s = sdata[tid + 2]; if (sdata[tid] < s) sdata[tid] = s; };
		if (blockSize >= 2) { T s = sdata[tid + 1]; if (sdata[tid] < s) sdata[tid] = s; };
	}

	template <typename T, unsigned int blockSize>
	__device__ void warpMin(volatile T* sdata, unsigned int tid) {
		if (blockSize >= 64) { T s = sdata[tid + 32]; if (sdata[tid] > s) sdata[tid] = s; };
		if (blockSize >= 32) { T s = sdata[tid + 16]; if (sdata[tid] > s) sdata[tid] = s; };
		if (blockSize >= 16) { T s = sdata[tid + 8]; if (sdata[tid] > s) sdata[tid] = s; };
		if (blockSize >= 8) { T s = sdata[tid + 4]; if (sdata[tid] > s) sdata[tid] = s; };
		if (blockSize >= 4) { T s = sdata[tid + 2]; if (sdata[tid] > s) sdata[tid] = s; };
		if (blockSize >= 2) { T s = sdata[tid + 1]; if (sdata[tid] > s) sdata[tid] = s; };
	}

	// dump_array_sum makes original array dirty, make sure dump is large enough
	template<typename T, int blockSize = 512>
	T dump_array_sum(const T* dump, size_t n) {
		void* d_temp_buf = nullptr;
		size_t temp_size = 0;
		T* out = nullptr;
		auto err = cub::DeviceReduce::Sum(d_temp_buf, temp_size, dump, out, n);
		if (err != CUDA_SUCCESS) {
			printf("\033[31mcub sum failed, error %s\033[0m\n", cudaGetErrorName(err));
		}
		//d_temp_buf = (T*)reserve_buf(temp_size + sizeof(T) * 10);
		auto buffer = getTempPool().getBuffer(temp_size + 32);
		d_temp_buf = buffer.template data<>();
		out = ((T*)d_temp_buf) + (temp_size + sizeof(T) - 1) / sizeof(T);
		err = cub::DeviceReduce::Sum(d_temp_buf, temp_size, dump, out, n);
		if (err != CUDA_SUCCESS) {
			printf("\033[31mcub sum failed, error %s\033[0m\n", cudaGetErrorName(err));
		}
		T sum;
		cudaMemcpy(&sum, out, sizeof(T), cudaMemcpyDeviceToHost);
		return sum;
	}

	//extern double dump_array_sum(float* dump, size_t n);
	template<typename T, typename DT = T, typename Lambda>
	DT dump_map_sum(const T* dump, Lambda func, size_t n, DT voidValue = 0) {
#if 1
		auto gen = [=] __device__(int vid) {
			return func(dump[vid]);
		};
		//std::cout << "T = " << typeid(T).name() << std::endl;
		auto redu = [=] __device__(DT x, DT y) {
			return x + y;
		};
		// not working until func has host copy
		//  instead, using direct template parameter
		//using DT = cuda::std::invoke_result_t<Lambda, T>;
		int szTemp = sizeof(DT) * n / 128;
		auto buffer = getTempPool().getBuffer(szTemp + 1024);
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, n, 256);
		blockMapReduce << <grid_size, block_size >> > (n, gen, redu, voidValue, buffer.template data<>(), szTemp);
		cudaDeviceSynchronize();
		cuda_error_check;
		// This should be enough
		DT* d_tmp = buffer.template data<DT>() + round(grid_size, 1024 / sizeof(DT));
		size_t d_tmp_size = szTemp + 1024 - grid_size * sizeof(DT);
		//auto unifiedBlock = getTempPool().getUnifiedBlock<DT>();
		cudaDeviceSynchronize();
		cuda_error_check;
		//DT* sum = unifiedBlock.template data<DT>();
		DT* devSum;
		size_t szsum;
		//std::cout << "DT = " << typeid(DT).name() << std::endl;
		CheckErr(cub::DeviceReduce::Sum(nullptr, szsum, buffer.template data<DT>(), devSum, grid_size));
		devSum = d_tmp + ceilDiv(szsum, sizeof(DT));
		//printf("cub require size %zu, have %zu \n", szsum, szTemp);
		if (szsum < d_tmp_size) {
			CheckErr(cub::DeviceReduce::Sum(d_tmp, szsum, buffer.template data<DT>(), devSum, grid_size));
		}
		else {
			printf("\033[31mcub require size %zu, provided %zu \033[0m\n", szsum, d_tmp_size);
		}
		cudaDeviceSynchronize();
		cuda_error_check;
		DT ss = 0;
		cudaMemcpy(&ss, devSum, sizeof(DT), cudaMemcpyDeviceToHost);
		return ss;
#else
		size_t szTemp;
		auto resbuf = getTempPool().getUnifiedBlock<DT>();
		auto p_res = resbuf.template data<DT>();
		// NVCC regard the return type of all device lambda to be INT.. :(
		auto redu = [=] __device__(T x, T y) {
			return func(x) + func(y);
			//return x + y;
		};
		CheckErr(cub::DeviceReduce::Reduce(nullptr, szTemp, dump, p_res, n, redu, T(voidValue)));
		auto buffer = getTempPool().getBuffer(szTemp);
		CheckErr(cub::DeviceReduce::Reduce(buffer.template data<>(), szTemp, dump, p_res, n, redu, T(voidValue)));
		//  cudaStreamSynchronize(0); // todo
		cudaDeviceSynchronize();
		cuda_error_check;
		return *p_res;
#endif
	}

	template<typename DT, typename Lambda, int blockSize = 256>
	DT sequence_sum(Lambda gen, size_t n, DT voidValue) {
#if 1
		//std::cout << "T = " << typeid(T).name() << std::endl;
		auto redu = [=] __device__(auto x, auto y) {
			return x + y;
		};
		// not working until func has host copy
		//  instead, using direct template parameter
		//using DT = cuda::std::invoke_result_t<Lambda, T>;
		int szTemp = sizeof(DT) * n / 128;
		auto buffer = getTempPool().getBuffer(szTemp + 1024);
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, n, blockSize);
		blockMapReduce<decltype(gen), decltype(redu), decltype(voidValue), blockSize> << <grid_size, block_size >> > (
			n, gen, redu, voidValue, buffer.template data<>(), szTemp);
		cudaDeviceSynchronize();
		cuda_error_check;
		// This should be enough
		DT* d_tmp = buffer.template data<DT>() + round(grid_size, 1024 / sizeof(DT));
		size_t d_tmp_size = szTemp + 1024 - grid_size * sizeof(DT);
		//auto unifiedBlock = getTempPool().getUnifiedBlock<DT>();
		cudaDeviceSynchronize();
		cuda_error_check;
		//DT* sum = unifiedBlock.template data<DT>();
		DT* psum = nullptr;
		//DT sum(0);
		size_t szsum;
		//std::cout << "DT = " << typeid(DT).name() << std::endl;
		CheckErr(cub::DeviceReduce::Sum(nullptr, szsum, buffer.template data<DT>(), psum, grid_size));
		psum = d_tmp + ceilDiv(szsum, sizeof(DT));
		//printf("cub require size %zu, have %zu \n", szsum, szTemp);
		if (szsum < d_tmp_size) {
			CheckErr(cub::DeviceReduce::Sum(d_tmp, szsum, buffer.template data<DT>(), psum, grid_size));
		}
		else {
			printf("\033[31mcub require size %zu, provided %zu \033[0m\n", szsum, d_tmp_size);
		}
		cudaDeviceSynchronize();
		cuda_error_check;
		DT sum = 0;
		cudaMemcpy(&sum, psum, sizeof(DT), cudaMemcpyDeviceToHost);
		return sum;
#else
		size_t szTemp;
		auto resbuf = getTempPool().getUnifiedBlock<DT>();
		auto p_res = resbuf.template data<DT>();
		// NVCC regard the return type of all device lambda to be INT.. :(
		auto redu = [=] __device__(T x, T y) {
			return func(x) + func(y);
			//return x + y;
		};
		CheckErr(cub::DeviceReduce::Reduce(nullptr, szTemp, dump, p_res, n, redu, T(voidValue)));
		auto buffer = getTempPool().getBuffer(szTemp);
		CheckErr(cub::DeviceReduce::Reduce(buffer.template data<>(), szTemp, dump, p_res, n, redu, T(voidValue)));
		//  cudaStreamSynchronize(0); // todo
		cudaDeviceSynchronize();
		cuda_error_check;
		return *p_res;
#endif
	}

	template<typename DT, typename InputIteratorT, typename Reduce>
	DT sequence_reduce(InputIteratorT iter, Reduce op, size_t n, DT voidValue) {
		DT* d_temp_storage = nullptr;
		size_t temp_storage_bytes;
		DT* d_out = nullptr;
		// get temp storage requirement
		cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, iter, d_out, n, op, voidValue);
		// get temp buffer
		auto buffer = getTempPool().getBuffer(temp_storage_bytes + 1024);
		// get data pointer from buffer
		d_temp_storage = buffer.template data<DT>();
		// relocate d_out
		d_out = d_temp_storage + ceilDiv(temp_storage_bytes, sizeof(DT));
		// perform reduction
		cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, iter, d_out, n, op, voidValue);
		// download result
		DT result;
		cudaMemcpy(&result, d_out, sizeof(DT), cudaMemcpyDeviceToHost);
		cuda_error_check;
		return result;
	}

	template<typename T>
	__device__ T clamp(T value, T low, T upp) {
		if (value < low) return low;
		if (value > upp) return upp;
		return value;
	}

	template<typename T, int BlockSize = 256>
	T norm(T* in_datax, T* in_datay, T* in_dataz, size_t n, T* sum_dst = nullptr) {
#if 1
		auto gen = [=] __device__(int tid) {
			T x = in_datax[tid], y = in_datay[tid], z = in_dataz[tid];
			return x * x + y * y + z * z;
		};

		T sum = sequence_sum<T, decltype(gen), BlockSize>(gen, n, T(0));
		T nrm = sqrt(sum);
		if (sum_dst != nullptr) { cudaMemcpy(sum_dst, &nrm, sizeof(T), cudaMemcpyHostToDevice); }
		return nrm;
#else

		auto gen = [=] __device__(int tid) {
			int vid = tid % n;
			int cord = tid / n;
			T* p = cord == 0 ? in_datax : (cord == 1 ? in_datay : in_dataz);
			T val = p[vid];
			return val * val;
		};
		T sum = sequence_sum(gen, 3 * n, T(0));
		T nrm = sqrt(sum);
		if (sum_dst != nullptr) { cudaMemcpy(sum_dst, &nrm, sizeof(T), cudaMemcpyHostToDevice); }
		return nrm;
#endif
	}

	template<typename T>
	T dot(T* pdatax, T* pdatay, T* pdataz, T* qdatax, T* qdatay, T* qdataz, T* odata, size_t n, T* sum_dst = nullptr) {
		// todo : use flattenned v3 rather than distinct pointer via index transform
		auto gen = [=] __device__(int tid) {
			T x1 = pdatax[tid], y1 = pdatay[tid], z1 = pdataz[tid];
			T x2 = qdatax[tid], y2 = qdatay[tid], z2 = qdataz[tid];
			return x1 * x2 + y1 * y2 + z1 * z2;
		};
		T sum = sequence_sum(gen, n, T(0));
		if (sum_dst != nullptr) { cudaMemcpy(sum_dst, &sum, sizeof(T), cudaMemcpyHostToDevice); }
		return sum;
	}

	template<typename T>
	T dot(const T* indata1, const T* indata2, T* dump_buf, size_t n, T* dot_dst = nullptr) {
		auto gen = [=] __device__(int tid) {
			T p = indata1[tid];
			T q = indata2[tid];
			return p * q;
		};
		T sum = sequence_sum(gen, n, T(0));
		if (dot_dst != nullptr) { cudaMemcpy(dot_dst, &sum, sizeof(T), cudaMemcpyHostToDevice); }
		return sum;
	}

	template<typename T>
	T parallel_max(const T* indata, size_t array_size, T* max_dst = nullptr) {
		//auto resblock = getTempPool().getUnifiedBlock<T>();
		T* pres/* = resblock.template data<T>()*/;
		T* d_temp_buf = nullptr;
		size_t szTemp;
		CheckErr(cub::DeviceReduce::Max(d_temp_buf, szTemp, indata, pres, array_size));
		auto buffer = getTempPool().getBuffer(round(szTemp + 32, 32));
		d_temp_buf = buffer.template data<T>();
		pres = d_temp_buf + ceilDiv(szTemp, sizeof(T));
		CheckErr(cub::DeviceReduce::Max(d_temp_buf, szTemp, indata, pres, array_size));
		cudaDeviceSynchronize();
		cuda_error_check;
		if (max_dst != nullptr) cudaMemcpy(max_dst, pres, sizeof(T), cudaMemcpyDeviceToDevice);
		T res;
		cudaMemcpy(&res, pres, sizeof(T), cudaMemcpyDeviceToHost);
		return res;
	}

	template<typename T>
	struct AbsWrapT {
		T val;
		__host__ __device__ bool operator<(const AbsWrapT<T>& y) const {
			return abs(val) < abs(y.val);
		}
		__host__ __device__ bool operator>(const AbsWrapT<T>& y) const {
			return abs(val) > abs(y.val);
		}
		__host__ __device__ operator T() {
			return val;
		}
		__host__ __device__ AbsWrapT(T val_) : val(val_) {}
		__host__ __device__ AbsWrapT(void) {}
	};

	template<typename T>
	struct MaxOp {
		__device__ __forceinline__ T
			operator()(const T& a, const T& b) const { return (b > a) ? b : a; }
	};

	template<typename T>
	struct MinOp {
		__device__ __forceinline__ T
			operator()(const T& a, const T& b) const { return (b < a) ? b : a; }
	};

	template<typename T>
	struct SumOp {
		__device__ __forceinline__ T
			operator()(const T& a, const T& b) const { return a + b; }
	};

	template<typename T>
	T parallel_maxabs(const T* indata, size_t array_size, T* max_dst = nullptr) {
		const AbsWrapT<T>* wrap = reinterpret_cast<const AbsWrapT<T>*>(indata);
		static_assert(sizeof(AbsWrapT<T>) == sizeof(T), "Wrapped type has different size");
		auto maxabs = parallel_max(wrap, array_size, reinterpret_cast<AbsWrapT<T>*>(max_dst));
		return maxabs;
	}

	template<typename T>
	T parallel_min(const T* indata, size_t array_size, T* min_dst = nullptr) {
		//auto resblock = getTempPool().getUnifiedBlock<T>();
		T* pres /*= resblock.template data<T>()*/;
		T* d_temp_buf = nullptr;
		size_t szTemp;
		CheckErr(cub::DeviceReduce::Min(d_temp_buf, szTemp, indata, pres, array_size));
		auto buffer = getTempPool().getBuffer(round(szTemp + 32, 32));
		d_temp_buf = buffer.template data<T>();
		pres = d_temp_buf + ceilDiv(szTemp, sizeof(T));
		CheckErr(cub::DeviceReduce::Min(d_temp_buf, szTemp, indata, pres, array_size));
		cudaDeviceSynchronize();
		if (min_dst != nullptr) cudaMemcpy(min_dst, pres, sizeof(T), cudaMemcpyDeviceToDevice);
		T res;
		cudaMemcpy(&res, pres, sizeof(T), cudaMemcpyDeviceToHost);
		cuda_error_check;
		return res;
	}

	template<typename T>
	T parallel_sum(const T* indata, size_t array_size, T* sum_dst = nullptr) {
		T s = dump_array_sum(indata, array_size);
		if (sum_dst != nullptr) cudaMemcpy(sum_dst, &s, sizeof(T), cudaMemcpyHostToDevice);
		return s;
	}

	// to increase accuracy of low resolution number
	template<typename T>
	double parallel_sum_d(const T* indata, size_t array_size, T* sum_dst = nullptr) {
		double s = dump_map_sum(indata, [=]__device__(T in) { return in; }, array_size, double(0));
		if (sum_dst != nullptr) cudaMemcpy(sum_dst, &s, sizeof(T), cudaMemcpyHostToDevice);
		return s;
	}

	template<typename T, typename Lambda>
	T parallel_map_sum(const T* indata, size_t array_size, Lambda func, T* sum_dst = nullptr) {
		double s = dump_map_sum(indata, func, array_size);
		if (sum_dst != nullptr) cudaMemcpy(sum_dst, &s, sizeof(T), cudaMemcpyHostToDevice);
		return s;
	}

	template<typename T>
	T parallel_diffdot(int v3size, T* v1[3], T* v2[3], T* v3[3], T* v4[3]) {
		constexpr int blockSize = 512;
		size_t grid_dim, block_dim;
		make_kernel_param(&grid_dim, &block_dim, v3size, blockSize);
		devArray_t<const T*, 3> _v1, _v2, _v3, _v4;
		for (int i = 0; i < 3; i++) {
			_v1[i] = v1[i];
			_v2[i] = v2[i];
			_v3[i] = v3[i];
			_v4[i] = v4[i];
		}
		auto v3diffdot = [=] __device__(int tid) {
			return (_v2[0][tid] - _v1[0][tid]) * (_v4[0][tid] - _v3[0][tid])
				+ (_v2[1][tid] - _v1[1][tid]) * (_v4[1][tid] - _v3[1][tid])
				+ (_v2[2][tid] - _v1[2][tid]) * (_v4[2][tid] - _v3[2][tid]);
		};
		T s = sequence_sum(v3diffdot, v3size, T(0));
		return s;
	}

	template<typename T>
	T parallel_diffdiffdot(int v3size, T* v1[3], T* v2[3], T* v3[3], T* v4[3],
		T* u1[3], T* u2[3], T* u3[3], T* u4[3]) {
		constexpr int blockSize = 512;
		size_t grid_dim, block_dim;
		make_kernel_param(&grid_dim, &block_dim, v3size, blockSize);
		cuda_error_check;
		devArray_t<const T*, 3> _v1, _v2, _v3, _v4, _u1, _u2, _u3, _u4;
		for (int i = 0; i < 3; i++) {
			_v1[i] = v1[i];
			_v2[i] = v2[i];
			_v3[i] = v3[i];
			_v4[i] = v4[i];
			_u1[i] = u1[i];
			_u2[i] = u2[i];
			_u3[i] = u3[i];
			_u4[i] = u4[i];
		}
		auto v3diffdot = [=] __device__(int tid) {
			T s = (_v2[0][tid] - _v1[0][tid] - _v4[0][tid] + _v3[0][tid]) * (_u2[0][tid] - _u1[0][tid] - _u4[0][tid] + _u3[0][tid]) +
				(_v2[1][tid] - _v1[1][tid] - _v4[1][tid] + _v3[1][tid]) * (_u2[1][tid] - _u1[1][tid] - _u4[1][tid] + _u3[1][tid]) +
				(_v2[2][tid] - _v1[2][tid] - _v4[2][tid] + _v3[2][tid]) * (_u2[2][tid] - _u1[2][tid] - _u4[2][tid] + _u3[2][tid]);
			return s;
		};
		T s = sequence_sum(v3diffdot, v3size, T(0));
		return s;
	}

	template<typename T>
	__device__ T vsum_wise(int k, T* p) {
		return p[k];
	}

	template<typename T, typename... Args>
	__device__ T vsum_wise(int k, T* v1, Args*... v) {
		return v1[k] + vsum_wise(k, v...);
	}

	template<typename _TOut, typename T, typename... Args>
	__global__ void vsum_kernel(int n, _TOut* out, T* v1, Args*... args) {
		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= n) return;
		out[tid] = vsum_wise(tid, v1, args...);
	}


	template<typename _Tout, typename T, typename... Args>
	void vsum(int n, _Tout* out, T* v1, Args*... args) {
		size_t grid_dim, block_dim;
		make_kernel_param(&grid_dim, &block_dim, n, 512);
		vsum_kernel << <grid_dim, block_dim >> > (n, out, v1, args...);
		cudaDeviceSynchronize();
		cuda_error_check;
	}


	template<typename T>
	struct array_t {
		T* _ptr;
		size_t _len;
		__host__ __device__ array_t(T* ptr, size_t len) : _ptr(ptr), _len(len) { }
		__host__  array_t(size_t len, T val) {
			cudaMalloc(&_ptr, sizeof(T) * len);
			init_array(_ptr, val, len);
		}
		__host__ void set(size_t offs, T val) {
			cudaMemcpy(_ptr + offs, &val, sizeof(T), cudaMemcpyHostToDevice);
		}

		__host__ void free(void) {
			cudaFree(_ptr);
		}

		__host__ T* data(void) {
			return _ptr;
		}

		template<typename T1>
		__host__ void operator=(const array_t<T1>& ar1) const {
			T* dst = _ptr;
			T1* src = ar1._ptr;
			size_t grid_size, block_size;
			make_kernel_param(&grid_size, &block_size, _len, 512);
			map << <grid_size, block_size >> > (_len, [=] __device__(int eid) {
				dst[eid] = src[eid];
			});
			cudaDeviceSynchronize();
			cuda_error_check;
		}

		// this function cannot pass compilation due to error on type traits
		template<typename F/*, std::enable_if_t<std::is_arithmetic_v<F>, int> = 0*/>
		__host__ array_t& operator/=(F f) {
			T* src = _ptr;
			size_t grid_size, block_size;
			make_kernel_param(&grid_size, &block_size, _len, 512);
			map << <grid_size, block_size >> > (_len, [=] __device__(int eid) {
				src[eid] /= f;
			});
			cudaDeviceSynchronize();
			cuda_error_check;
			return (*this);
		}

		template<typename F>
		__host__ array_t& operator/=(const array_t<F>& f2) {
			T* op1 = _ptr;
			const F* op2 = f2._ptr;
			size_t grid_size, block_size;
			make_kernel_param(&grid_size, &block_size, _len, 512);
			map << <grid_size, block_size >> > (_len, [=] __device__(int eid) {
				op1[eid] /= op2[eid];
			});
			cudaDeviceSynchronize();
			cuda_error_check;
			return *this;
		}

		template<typename F>
		__host__ array_t& operator*=(F f) {
			T* src = _ptr;
			size_t grid_size, block_size;
			make_kernel_param(&grid_size, &block_size, _len, 512);
			map << <grid_size, block_size >> > (_len, [=] __device__(int eid) {
				src[eid] *= f;
			});
			cudaDeviceSynchronize();
			cuda_error_check;
			return (*this);
		}

		template<typename F>
		__host__ array_t& operator+=(F f) {
			T* src = _ptr;
			size_t grid_size, block_size;
			make_kernel_param(&grid_size, &block_size, _len, 512);
			map << <grid_size, block_size >> > (_len, [=] __device__(int eid) {
				src[eid] += f;
			});
			cudaDeviceSynchronize();
			cuda_error_check;
			return (*this);
		}

		template<typename F>
		__host__ array_t& operator-=(F f) {
			T* src = _ptr;
			size_t grid_size, block_size;
			make_kernel_param(&grid_size, &block_size, _len, 512);
			map << <grid_size, block_size >> > (_len, [=] __device__(int eid) {
				src[eid] -= f;
			});
			cudaDeviceSynchronize();
			cuda_error_check;
			return (*this);
		}
	};

	template<typename T>
	struct _randArrayGen {
		void gen(curandGenerator_t& gen, double** dst, int nArray, size_t len) {
			printf("\033[31mUnsupported scalar type!\033[0m\n");
		}
	};

	template<>
	struct _randArrayGen<double> {
		void gen(curandGenerator_t& gen, double** dst, int nArray, size_t len) {
			for (int i = 0; i < nArray; i++) {
				curandGenerateUniformDouble(gen, dst[i], len);
			}
		}
	};

	template<>
	struct _randArrayGen<float> {
		void gen(curandGenerator_t& gen, float** dst, int nArray, size_t len) {
			for (int i = 0; i < nArray; i++) {
				curandGenerateUniform(gen, dst[i], len);
			}
		}
	};

	template<typename T>
	void randArray(T** dst, int nArray, size_t len, T low = T{ 0 }, T upp = T{ 0 }) {
		curandGenerator_t generator;
		curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(generator, (int)time(nullptr));
		_randArrayGen<T>  gen;
		gen.gen(generator, dst, nArray, len);
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, len, 512);
		for (int i = 0; i < nArray; i++) {
			T* pdata = dst[i];
			culib::traverse << <grid_size, block_size >> > (pdata, len, [=] __device__(int tid) {
				T value = pdata[tid];
				value = low + value * (upp - low);
				return  value;
			});
			cudaDeviceSynchronize();
			cuda_error_check;
		}
	}

	template<typename T>
	void check_array_len(const T* _pdata, size_t len) {
		CUdeviceptr pbase, pdata;
		pdata = reinterpret_cast<CUdeviceptr>(_pdata);
		size_t arrlen;
		cuMemGetAddressRange(&pbase, &arrlen, pdata);
		if (len * sizeof(T) > arrlen) {
			printf("\033[31m-- Check length Failed! \033[0m\n");
		}
		cuda_error_check;
	}

	template<typename T>
	__device__ bool read_gbit(const T* bits, size_t id) {
		return bits[id / (sizeof(T) * 8)] & (T{ 1 } << (id % (sizeof(T) * 8)));
	}

	template<typename T, typename std::enable_if<!std::is_pointer<T>::value, int>::type = 0>
	__device__ bool read_gbit(T word, int id) {
		return word & (T{ 1 } << id);
	}

	template<typename T>
	__device__ void set_gbit(T* bits, size_t id) {
		bits[id / (sizeof(T) * 8)] |= (T{ 1 } << (id % (sizeof(T) * 8)));
	}

	template<typename T, typename std::enable_if<!std::is_pointer<T>::value, int>::type = 0>
	__device__ void set_gbit(T word, int id) {
		word |= (T{ 1 } << id);
	}

	template<typename T>
	__device__ void reset_gbit(T* bits, size_t id) {
		bits[id / (sizeof(T) * 8)] &= ~(T{ 1 } << (id % (sizeof(T) * 8)));
	}

	template<typename T, typename std::enable_if<!std::is_pointer<T>::value, int>::type = 0>
	__device__ void reset_gbit(T word, int id) {
		word &= ~(T{ 1 } << id);
	}

	template<typename T>
	__device__ void atomic_set_gbit(T* bits, size_t id) {
		atomicOr(
			((int*)(void*)bits) + id / (sizeof(int) * 8),
			int{ 1 } << (id % (sizeof(int) * 8))
		);
	}

	template<typename T>
	__device__ void atomic_reset_gbit(T* bits, size_t id) {
		atomicAnd(
			((int*)(void*)bits) + id / (sizeof(int) * 8),
			~(int{ 1 } << (id % (sizeof(int) * 8)))
		);
	}

	template<typename T>
	__device__ void initSharedMem(volatile T* pshared, int len, T val = T{ 0 }) {
		for (int id = threadIdx.x; id < len; id += blockDim.x) {
			pshared[id] = val;
		}
		//int base = 0;
		//while (base < len) {
		//	if (base + threadIdx.x < len) {
		//		pshared[base + threadIdx.x] = val;
		//	}
		//	base += blockDim.x;
		//}
	}

	template<typename T/*, bool SelfAllocate = false */>
	struct gBitSAT {
		static constexpr size_t size_mask = sizeof(T) * 8 - 1;
		const T* _bitarray;
		const int* _chunksat;
		template<int N, bool stop = (N == 0)>
		struct firstOne {
			static constexpr int value = 1 + firstOne< (N >> 1), ((N >> 1) == 0)>::value;
		};

		template<int N>
		struct firstOne<N, true> {
			static constexpr int value = -1;
		};

		template<typename intT>
		__host__ __device__ inline int countOne(intT num) const {
#if 0
			int n = 0;
			while (num) {
				num &= (num - 1);
				n++;
			}
			return n;
#else
			return __popc(num);
#endif
		}

		//__host__ ~gBitSAT() {
		//	if (SelfAllocate) {
		//		cudaFree(_bitarray);
		//		cudaFree(_chunksat);
		//	}
		//}

		__host__ void destroy(void) {
			cudaFree(const_cast<T*>(_bitarray));
			cudaFree(const_cast<int*>(_chunksat));
		}

		//template<bool Allocate = SelfAllocate, std::enable_if<!Allocate, void>::type *= nullptr>
		__host__ __device__  gBitSAT(const T* bitarray, const int* chunksat)
			:_bitarray(bitarray), _chunksat(chunksat)
		{ }

		//template<bool Allocate = SelfAllocate, std::enable_if<Allocate, void>::type *= nullptr>
		__host__  gBitSAT(const std::vector<T>& hostbits, const std::vector<int>& hostsat) {
			cudaMalloc(const_cast<T**>(&_bitarray), hostbits.size() * sizeof(T));
			cudaMemcpy(const_cast<T*>(_bitarray), hostbits.data(), sizeof(T) * hostbits.size(), cudaMemcpyHostToDevice);

			cudaMalloc(const_cast<int**>(&_chunksat), hostsat.size() * sizeof(int));
			cudaMemcpy(const_cast<int*>(_chunksat), hostsat.data(), sizeof(int) * hostsat.size(), cudaMemcpyHostToDevice);
		}

		__host__ __device__ gBitSAT(void) = default;

		__host__ __device__ int operator[](size_t id) const {
			int ent = id >> firstOne<sizeof(T) * 8>::value;
			int mod = id & size_mask;
			return _chunksat[ent] + countOne(_bitarray[ent] & ((T{ 1 } << mod) - 1));
		}

		__host__ __device__ int operator()(size_t id) const {
			int ent = id >> firstOne<sizeof(T) * 8>::value;
			int mod = id & size_mask;
			T resword = _bitarray[ent];
			if ((resword & (T{ 1 } << mod)) == 0) {
				return -1;
			}
			else {
				return _chunksat[ent] + countOne(resword & ((T{ 1 } << mod) - 1));
			}
		}
	};

	template<typename Lambda>
	void parallel_do(int ntask, int block_size, Lambda work) {
		size_t grid_size, blockSize;
		make_kernel_param(&grid_size, &blockSize, ntask, block_size);
		traverse_noret << <grid_size, blockSize >> > (ntask, work);
		cudaDeviceSynchronize();
		cuda_error_check;
	}

	template<typename T1, typename T2>
	void type_cast(T2* dst, T1* src, size_t len) {
		auto castker = [=] __device__(int tid) {
			dst[tid] = src[tid];
		};
		parallel_do(len, 256, castker);
	}

	template<typename T1, typename T2>
	void type_cast2D(T2* dst, int dstPitchT, T1* src, int srcPitchT, int xsize, int ysize) {
		auto castker = [=] __device__(int tid) {
			int x = tid % xsize;
			int y = tid / xsize;
			dst[x + y * dstPitchT] = src[x + y * srcPitchT];
		};
		parallel_do(xsize * ysize, 256, castker);
	}

	template<typename Scalar>
	__global__ void remap(int n, Scalar* p, Scalar l_, Scalar h_, Scalar L_, Scalar H_) {
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		Scalar sca = (H_ - L_) / (h_ - l_);
		if (tid < n) {
			p[tid] = (p[tid] - l_) * sca + L_;
		}
	}

	__device__ inline float tanproj(float val, float beta, float tau = 0.5f) {
		const float tbtau = tanhf(beta * tau);
		float newval = (tbtau + tanhf(beta * (val - tau))) / (tbtau + tanhf(beta * (1.f - tau)));
		//if (isnan(newval)) newval = 0.001;
		//if (newval < 0.001) newval = 0.001;
		//if (newval > 1) newval = 1;
		return newval;
	}

	__device__ inline float sigmoid(float val, float k, float center) {
		return 1.f / (1 + expf(-k * (val - center)));
	}
}

namespace cub {
	template <typename T> struct NumericTraits<culib::AbsWrapT<T>> : NumericTraits<T> {
		// overwrite Lowest function
		static __host__ __device__ __forceinline__ T Lowest() { return 0; }
	};
	
}

#endif
