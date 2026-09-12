#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "culib/lib.cuh"
#include "culib/warp_primitive.cuh"
#include <curand_kernel.h>

namespace homo {

// enum Order : int;
// enum TensorSym : int;
//inline int ceilDiv(size_t a, size_t b) {
//	size_t c = a / b;
//	return a % b ? c + 1 : c;
//}

//__global__ void initRandStates(unsigned long long seed, int n, curandState* states) {
//	int idx = threadIdx.x + blockDim.x * blockIdx.x;
//	curand_init(seed, idx, 0, &states[idx]);
//}

template<typename T, typename Tview>
void tensor_rand(Tview dataview, T low, T upp) {
	//curandState* randstates;
	//// malloc states
	//cudaMalloc(&randstates, sizeof(curandState) * dataview.size());
	//// init states
	//
	//cudaFree(randstates);
	T* randarr;
	cudaMalloc(&randarr, sizeof(T) * dataview.size());
	randArray(&randarr, 1, dataview.size(), low, upp);
	cudaMemcpy2D(dataview.data(), dataview.getPitchT() * sizeof(T),
				 randarr, dataview.size(0) * sizeof(T),
				 dataview.size(0) * sizeof(T),
				 dataview.size(1) * dataview.size(2), cudaMemcpyDeviceToDevice);
	cuda_error_check;
	cudaFree(randarr);
}

template<typename Scalar, typename Acc>
__global__ void range_kernel(
	devArray_t<Scalar, 3> start, devArray_t<Scalar, 3> end, devArray_t<int, 3> steps,
	Acc acc, Order axis) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int na[3] = {steps[0] + 1, steps[1] + 1, steps[2] + 1};
	if (tid >= na[0] * na[1] * na[2])
		return;
	Scalar steplen[3] = {
		(end[0] - start[0]) / steps[0],
		(end[1] - start[1]) / steps[1],
		(end[2] - start[2]) / steps[2],
	};
	int id[3] = {
		tid % na[0],
		tid / na[0] % na[1],
		tid / (na[0] * na[1])};
	Scalar pos[3] = {
		id[0] * steplen[0] + start[0],
		id[1] * steplen[1] + start[1],
		id[2] * steplen[2] + start[2]};

	if (axis >= 3)
		return;
	acc(id[0], id[1], id[2]) = pos[axis];
	//printf("[range] %e\n", pos[axis]);
}

template<typename Acc>
__global__ void symetrize_tensor_kernel(Acc acc_, TensorSym sym, bool average) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	auto acc = acc_;
	if (sym == Reflection3) {
		int repReso[3] = {acc.size(0) / 2, acc.size(1) / 2, acc.size(2) / 2};
		int repPos[3] = {
			tid % repReso[0],
			tid / repReso[0] % repReso[1],
			tid / (repReso[0] * repReso[1])};
		if (repPos[2] >= repReso[2])
			return;
		int orbit[8][3];
		for (int i = 0; i < 8; i++) {
			int flip[3] = {i % 2, i / 2 % 2, i / 4};
			for (int j = 0; j < 3; j++) {
				if (flip[j]) {
					orbit[i][j] = acc.size(j) - 1 - repPos[j];
				} else {
					orbit[i][j] = repPos[j];
				}
			}
		}
		auto val = acc(orbit[0][0], orbit[0][1], orbit[0][2]);
		if (average) {
			for (int i = 1; i < 8; i++) {
				val += acc(orbit[i][0], orbit[i][1], orbit[i][2]);
			}
			val /= 8;
		}
		// Note : No write-read conflict due to non-intersection of orbits
		for (int i = 0; i < 8; i++) {
			acc(orbit[i][0], orbit[i][1], orbit[i][2]) = val;
		}
	} else if (sym == Reflection6) {
		int repReso[3] = {acc.size(0) / 2, acc.size(1) / 2, acc.size(2) / 2};
		int repPos[3];
		// z > y > x, for simplicity..
		int vid = tid;
		int x = 0, y = 0, z = 0;
		z = powf(6.f * vid, 0.3333333f) + 1.1f;
		while ((z + 1) * (z + 2) * (z + 3) / 6 > vid) {
			z--;
		}
		if (z + 1 >= repReso[2])
			return;
		vid -= (z + 1) * (z + 2) * (z + 3) / 6;
		y = powf(2.f * vid, 0.5f) + 1.1f;
		while ((y + 1) * (y + 2) / 2 > vid) {
			y--;
		}
		vid -= (y + 1) * (y + 2) / 2;
		x = vid;
		repPos[0] = x;
		repPos[1] = y + 1;
		repPos[2] = z + 1;
		//if (tid < 300) {
		//	printf("[%d] = (%d, %d, %d)\n", (int)tid, repPos[0], repPos[1], repPos[2]);
		//}
		// traverse orbits [S_3] x [Reflect_3]
		using T = typename Acc::Scalar;
		T orbitSum = 0;
#pragma unroll
		for (int i = 0; i < 8; i++) {
			int orbit[3];
			int flip[3] = {i % 2, i / 2 % 2, i / 4};
			for (int j = 0; j < 3; j++) {
				if (flip[j]) {
					orbit[j] = acc.size(j) - 1 - repPos[j];
				} else {
					orbit[j] = repPos[j];
				}
			}
			// [a, b, c]
			orbitSum += acc(orbit[0], orbit[1], orbit[2]);
			// [a, c, b]
			orbitSum += acc(orbit[0], orbit[2], orbit[1]);
			// [c, a, b]
			orbitSum += acc(orbit[2], orbit[0], orbit[1]);
			// [c, b, a]
			orbitSum += acc(orbit[2], orbit[1], orbit[0]);
			// [b, a, c]
			orbitSum += acc(orbit[1], orbit[0], orbit[2]);
			// [b, c, a]
			orbitSum += acc(orbit[1], orbit[2], orbit[0]);
		}
		orbitSum /= 48;
#pragma unroll
		for (int i = 0; i < 8; i++) {
			int orbit[3];
			int flip[3] = {i % 2, i / 2 % 2, i / 4};
			for (int j = 0; j < 3; j++) {
				if (flip[j]) {
					orbit[j] = acc.size(j) - 1 - repPos[j];
				} else {
					orbit[j] = repPos[j];
				}
			}
			// [a, b, c]
			acc(orbit[0], orbit[1], orbit[2]) = orbitSum;
			// [a, c, b]
			acc(orbit[0], orbit[2], orbit[1]) = orbitSum;
			// [c, a, b]
			acc(orbit[2], orbit[0], orbit[1]) = orbitSum;
			// [c, b, a]
			acc(orbit[2], orbit[1], orbit[0]) = orbitSum;
			// [b, a, c]
			acc(orbit[1], orbit[0], orbit[2]) = orbitSum;
			// [b, c, a]
			acc(orbit[1], orbit[2], orbit[0]) = orbitSum;
		}
	} else if (sym == Rotate3) {
		int repReso[3] = {acc.size(0) / 2, acc.size(1) / 2, acc.size(2) / 2};
		int repPos[3];
		// z > y, x
		int vid = tid;
		int x = 0, y = 0, z = 0;
		z = powf(3.f * vid, 0.3333333f) + 1.1f;
		while (z * (z + 1) * (2 * z + 1) / 6 > vid) {
			z--;
		}
		if (z >= repReso[2])
			return;
		vid -= z * (z + 1) * (2 * z + 1) / 6;
		y = vid / (z + 1);
		x = vid % (z + 1);
		repPos[0] = x;
		repPos[1] = y;
		repPos[2] = z;
		//if (z == repReso[2] - 1) {
		//	printf("[%d] = (%d, %d, %d)\n", (int)tid, repPos[0], repPos[1], repPos[2]);
		//}
		// traverse orbits [S_3] x [Reflect_3]
		using T = typename Acc::Scalar;
		T orbitSum = 0;
		const int permu[6][3] = {
			{0, 2, 1}, {2, 1, 0}, {1, 0, 2}, {2, 0, 1}, {1, 2, 0}, {0, 1, 2}};
#if 1
#pragma unroll
		for (int i = 0; i < 6; i++) {
			const int orbit[3] = {repPos[permu[i][0]], repPos[permu[i][1]], repPos[permu[i][2]]};
			int p[3];
			// traverse possible sign
			// odd permutation,  1/3 negative
			if (i < 3) {
				// x
				orbitSum += acc(repReso[0] - 1 - orbit[0], repReso[1] + orbit[1], repReso[2] + orbit[2]);
				// y
				orbitSum += acc(repReso[0] + orbit[0], repReso[1] - 1 - orbit[1], repReso[2] + orbit[2]);
				// z
				orbitSum += acc(repReso[0] + orbit[0], repReso[1] + orbit[1], repReso[2] - 1 - orbit[2]);
				// x y z
				orbitSum += acc(repReso[0] - 1 - orbit[0], repReso[1] - 1 - orbit[1], repReso[2] - 1 - orbit[2]);
			}
			// even permutation, 2 / 0 negative
			else {
				// origin
				orbitSum += acc(repReso[0] + orbit[0], repReso[1] + orbit[1], repReso[2] + orbit[2]);
				// x y
				orbitSum += acc(repReso[0] - 1 - orbit[0], repReso[1] - 1 - orbit[1], repReso[2] + orbit[2]);
				// y z
				orbitSum += acc(repReso[0] + orbit[0], repReso[1] - 1 - orbit[1], repReso[2] - 1 - orbit[2]);
				// z x
				orbitSum += acc(repReso[0] - 1 - orbit[0], repReso[1] + orbit[1], repReso[2] - 1 - orbit[2]);
			}
		}
		// average
		orbitSum /= 24;
		// write back
#pragma unroll
		for (int i = 0; i < 6; i++) {
			const int orbit[3] = {repPos[permu[i][0]], repPos[permu[i][1]], repPos[permu[i][2]]};
			int p[3];
			// traverse possible sign
			// odd permutation,  1/3 negative
			if (i < 3) {
				// x
				acc(repReso[0] - 1 - orbit[0], repReso[1] + orbit[1], repReso[2] + orbit[2]) = orbitSum;
				// y
				acc(repReso[0] + orbit[0], repReso[1] - 1 - orbit[1], repReso[2] + orbit[2]) = orbitSum;
				// z
				acc(repReso[0] + orbit[0], repReso[1] + orbit[1], repReso[2] - 1 - orbit[2]) = orbitSum;
				// x y z
				acc(repReso[0] - 1 - orbit[0], repReso[1] - 1 - orbit[1], repReso[2] - 1 - orbit[2]) = orbitSum;
			}
			// even permutation, 2 / 0 negative
			else {
				// origin
				acc(repReso[0] + orbit[0], repReso[1] + orbit[1], repReso[2] + orbit[2]) = orbitSum;
				// x y
				acc(repReso[0] - 1 - orbit[0], repReso[1] - 1 - orbit[1], repReso[2] + orbit[2]) = orbitSum;
				// y z
				acc(repReso[0] + orbit[0], repReso[1] - 1 - orbit[1], repReso[2] - 1 - orbit[2]) = orbitSum;
				// z x
				acc(repReso[0] - 1 - orbit[0], repReso[1] + orbit[1], repReso[2] - 1 - orbit[2]) = orbitSum;
			}
		}
#endif
	}
}

template<typename AccS, typename AccSdiff, typename Accd, typename Kernel>
__global__ void unarymap_kernel(AccS accs, AccSdiff tempdif, Accd accd, size_t n_tol, Kernel kernel) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= n_tol)
		return;
	using Scalar = typename AccS::Scalar;
	Kernel ker = kernel;
	Scalar diffe;
	Scalar opval = accs(tid);
	Scalar val = ker.eval(opval, diffe);
	accd(tid) = val;
	tempdif(tid) = diffe;
	//printf("[%zd] x = %f  v = %f  d = %f \n", tid, accs(tid), val, diffe);
}

template<typename AccS, typename AccSdiff, typename TempDif, typename AccDdiff, typename Kernel>
__global__ void unarymap_backward(AccS accs, AccSdiff srcdif, TempDif tempdif, AccDdiff dstdif, size_t n_tol, Kernel ker) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= n_tol)
		return;
	auto lastdiff = dstdif(tid);
	//auto srcval = accs[tid];
	srcdif(tid) += tempdif(tid) * lastdiff;
}

template<typename AccS, typename Accd, typename AccTemp, typename Kernel>
__global__ void conv_kernel(int ndst, AccS accs, Accd accd, AccTemp tmpacc, Kernel kernel) {
	// traverse dst tensor
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= ndst)
		return;
	Kernel ker = kernel;
	int i = tid % accd.dim[0];
	int j = tid / accd.dim[0] % accd.dim[1];
	int k = tid / (accd.dim[0] * accd.dim[1]);
	typedef typename Accd::Scalar Td;
	typedef typename AccS::Scalar Ts;
	Td sum(0);
	Td weightSum(0);
	// kernel offset
	for (int neighid = 0; neighid < ker.size(); neighid++) {
		int off[3];
		ker.neigh(neighid, off);
		Td weight = ker.weight(off);
		Ts val = ker.padValue();
		off[0] += i;
		off[1] += j;
		off[2] += k;
		if (off[0] >= 0 && off[0] < accs.dim[0] &&
			off[1] >= 0 && off[1] < accs.dim[1] &&
			off[2] >= 0 && off[2] < accs.dim[2]) {
			val = accs(off[0], off[1], off[2]);
			weightSum += weight;
		} else if (ker.is_period()) {
			// Make Sure the kernel is smaller than domain!!
			off[0] = (off[0] + accs.dim[0]) % accs.dim[0];
			off[1] = (off[1] + accs.dim[1]) % accs.dim[1];
			off[2] = (off[2] + accs.dim[2]) % accs.dim[2];
			val = accs(off[0], off[1], off[2]);
			weightSum += weight;
		}
		//if (tid == 0) {
		//	printf("nei = %d  val = %e  w = %e\n", neighid, val, weight);
		//}
		sum += weight * val;
		//if (tid == 0) {
		//	printf("nei = %d  off = (%d, %d, %d) w = %f  val = %f sum = %f\n", neighid, off[0], off[1], off[2], weight, val, sum);
		//}
	}
	accd(i, j, k) = sum / weightSum;
	tmpacc(i, j, k) = weightSum;
}

template<typename AccS, typename AccTemp, typename Accd, typename Kernel>
__global__ void conv_backward_kernel(int nsrc, AccS accs, Accd accd, AccTemp tmpacc, Kernel kernel) {
	typedef typename Accd::Scalar Td;
	typedef typename AccS::Scalar Ts;
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nsrc)
		return;
	Kernel ker = kernel;
	int i = tid % accs.dim[0];
	int j = tid / accs.dim[0] % accs.dim[1];
	int k = tid / (accs.dim[0] * accs.dim[1]);
	Ts diffsum(0);
	for (int neighid = 0; neighid < ker.size(); neighid++) {
		int off[3];
		ker.neigh(neighid, off);
		Td weight = ker.weight(off);
		//off[0] -= i; off[1] -= j; off[2] -= k;
		int dstp[3] = {i - off[0], j - off[1], k - off[2]};
		if (dstp[0] >= 0 && dstp[0] < accd.dim[0] &&
			dstp[1] >= 0 && dstp[1] < accd.dim[1] &&
			dstp[2] >= 0 && dstp[2] < accd.dim[2]) {
			auto dst_ws = tmpacc(dstp[0], dstp[1], dstp[2]);
			diffsum += weight * accd(dstp[0], dstp[1], dstp[2]) / dst_ws;
		} else if (ker.is_period()) {
			dstp[0] = (dstp[0] + accd.dim[0]) % accd.dim[0];
			dstp[1] = (dstp[1] + accd.dim[1]) % accd.dim[1];
			dstp[2] = (dstp[2] + accd.dim[2]) % accd.dim[2];
			auto dst_ws = tmpacc(dstp[0], dstp[1], dstp[2]);
			diffsum += weight * accd(dstp[0], dstp[1], dstp[2]) / dst_ws;
		}
	}
	accs(i, j, k) = diffsum;
}

// please fix block size as 512
template<int BlockSize, bool omitPreMap, typename AccS, typename Accd, typename ReduceOp,
		 std::enable_if_t<(BlockSize == 512), int>>
__global__ void tensor_reduce(size_t nsrc, AccS accs, Accd accd, ReduceOp op) {
	using Scalar = std::remove_reference_t<decltype(accs(0))>;
	__shared__ Scalar rawSum[BlockSize / 2];
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;
	bool lastReduce = gridDim.x == 1;
	// gather data
	Scalar s = op.Id;
	if (tid < nsrc) {
		s = accs(tid);
		if (!omitPreMap)
			s = op.preMap(s);
	}
	if (threadIdx.x >= BlockSize / 2) {
		rawSum[threadIdx.x - BlockSize / 2] = s;
	}
	__syncthreads();
	if (threadIdx.x < BlockSize / 2) {
		rawSum[threadIdx.x] = op.combine(rawSum[threadIdx.x], s);
	}
	__syncthreads();
	if (threadIdx.x < BlockSize / 4) {
		rawSum[threadIdx.x] = op.combine(rawSum[threadIdx.x + BlockSize / 4], rawSum[threadIdx.x]);
	}
	__syncthreads();
	if (threadIdx.x < BlockSize / 8) {
		rawSum[threadIdx.x] = op.combine(rawSum[threadIdx.x + BlockSize / 8], rawSum[threadIdx.x]);
	}
	__syncthreads();
	// warp reduce
	if (warpId < 1) {
		s = op.combine(rawSum[threadIdx.x], rawSum[threadIdx.x + 32]);
		for (int offset = 16; offset > 0; offset >>= 1) {
			s = op.combine(s, shfl_down(s, offset));
		}
		if (threadIdx.x == 0) {
			if (lastReduce) {
				s = op.postMap(s);
			}
			accd(blockIdx.x) = s;
		}
	}
}

// please fix block size as 512
template<int BlockSize, typename Scalar, typename AccS, typename Accd, typename ReduceOp,
		 std::enable_if_t<(BlockSize == 512), int>>
__global__ void tensor_reduce_gradient(AccS accs, Accd grad, Scalar res, Scalar lastdiff, ReduceOp op) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;
	if (tid >= accs.size())
		return;
	grad(tid) = op.diff(res, accs(tid)) * lastdiff;
	// if(threadIdx.x == 0) {
	// 	printf("grad = %e, res = %f, src = %f, lastdiff = %f\n",grad(tid), res, accs(tid), lastdiff);
	// }
}

} // namespace homo
