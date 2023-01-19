#pragma once
#include <type_traits>
#include "cuda_runtime.h"

#define FULL_MASK 0xffffffff

#define SHLF_OP_OFFSET(type)  template<typename Ts, int sT = sizeof(Ts), std::enable_if_t<sT==4,int> = 0> \
__device__ Ts shfl_##type (Ts data, int offset) { \
	return __shfl_down_sync(FULL_MASK, data, offset); \
} \
template<typename Ts, int sT = sizeof(Ts), std::enable_if_t<sT==8,int> = 0> \
__device__ Ts shfl_##type (Ts data, int offset) { \
	return __shfl_down_sync(FULL_MASK, data, offset); \
} 

//template<typename Ts, int sT = sizeof(Ts)>
//Ts shfl_down(Ts data, int offset) {
//	return Ts{ 0 };
//}
//
//template<typename Ts>
//Ts shfl_down<Ts, 4>(Ts data, int offset) {
//	return reinterpret_cast<Ts>(
//		__shfl_down(reinterpret_cast<int>(kergrad[i]), offset);
//	)
//}
//
//template<typename Ts>
//Ts shfl_down<Ts, 8>(Ts data, int offset) {
//	long long val = 0;
//	val |= __shfl_down(int(reinterpret_cast<long long>(data)), offset);
//	val |= __shfl_down(int(reinterpret_cast<long long>(data) >> 32), offset) << 32;
//	return reinterpret_cast<Ts>(val);
//}

SHLF_OP_OFFSET(down)
SHLF_OP_OFFSET(up)

//template<typename Ts, int sT = sizeof(Ts), std::enable_if_t<sT == 4, int> = 0>
//__device__ inline Ts shfl_(Ts data, int srcLane, int width = 32) {
//	return reinterpret_cast<Ts>(
//		__shfl(reinterpret_cast<int>(data), srcLane, width)
//	);
//}
//
//template<typename Ts, int sT = sizeof(Ts), std::enable_if_t<sT == 8, int> = 0>
//__device__ inline Ts shfl_(Ts data, int srcLane, int width = 0) {
//	long long val = 0;
//	//val |= __shfl(int(reinterpret_cast<long long>(data)), srcLane, width);
//	//val |= __shfl(int(reinterpret_cast<long long>(data) >> 32), srcLane, width) << 32;
//	return reinterpret_cast<Ts>(val);
//}

