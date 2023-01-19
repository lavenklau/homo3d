#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "culib/lib.cuh"
#include "culib/warp_primitive.cuh"
#include <curand_kernel.h>

namespace homo {
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
	}

	template<typename Scalar, typename Acc>
	__global__ void range_kernel(
		devArray_t<Scalar, 3> start, devArray_t<Scalar, 3> end, devArray_t<int, 3> steps,
		Acc acc, Order axis
	) {
		size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
		int na[3] = { steps[0] + 1, steps[1] + 1, steps[2] + 1 };
		if (tid >= na[0] * na[1] * na[2])  return;
		Scalar steplen[3] = {
			(end[0] - start[0]) / steps[0],
			(end[1] - start[1]) / steps[1],
			(end[2] - start[2]) / steps[2],
		};
		int id[3] = {
			tid % na[0] ,
			tid / na[0] % na[1] ,
			tid / (na[0] * na[1])
		};
		Scalar pos[3] = { 
			id[0] * steplen[0] + start[0],
			id[1] * steplen[1] + start[1],
			id[2] * steplen[2] + start[2]
		};

		if (axis >= 3) return;
		acc(id[0], id[1], id[2]) = pos[axis];
		//printf("[range] %e\n", pos[axis]);
	}

	template<typename Acc>
	__global__ void symetrize_tensor_kernel(Acc acc_, TensorSym sym, bool average) {
		size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
		auto acc = acc_;
		if (sym == Reflection3) {
			int repReso[3] = { acc.size(0) / 2, acc.size(1) / 2, acc.size(2) / 2 };
			int repPos[3] = {
				tid % repReso[0],
				tid / repReso[0] % repReso[1],
				tid / (repReso[0] * repReso[1])
			};
			if (repPos[2] >= repReso[2]) return;
			int orbit[8][3];
			for (int i = 0; i < 8; i++) {
				int flip[3] = { i % 2, i / 2 % 2, i / 4 };
				for (int j = 0; j < 3; j++) {
					if (flip[j]) {
						orbit[i][j] = acc.size(j) - 1 - repPos[j];
					}
					else {
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
		}
		else if (sym == Reflection6) {
			int repReso[3] = { acc.size(0) / 2, acc.size(1) / 2, acc.size(2) / 2 };
			int repPos[3];
			// z > y > x, for simplicity..
			int vid = tid;
			int x = 0, y = 0, z = 0;
			z = powf(6.f * vid, 0.3333333f) + 1.1f;
			while ((z + 1) * (z + 2) * (z + 3) / 6 > vid) { z--; }
			if (z + 1 >= repReso[2]) return;
			vid -= (z + 1) * (z + 2) * (z + 3) / 6;
			y = powf(2.f * vid, 0.5f) + 1.1f;
			while ((y + 1) * (y + 2) / 2 > vid) { y--; }
			vid -= (y + 1) * (y + 2) / 2;
			x = vid;
			repPos[0] = x; repPos[1] = y + 1; repPos[2] = z + 1;
			//if (tid < 300) {
			//	printf("[%d] = (%d, %d, %d)\n", (int)tid, repPos[0], repPos[1], repPos[2]);
			//}
			// traverse orbits [S_3] x [Reflect_3] 
			using T = typename Acc::Scalar;
			T orbitSum = 0;
#pragma unroll
			for (int i = 0; i < 8; i++) {
				int orbit[3];
				int flip[3] = { i % 2, i / 2 % 2, i / 4 };
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
				int flip[3] = { i % 2, i / 2 % 2, i / 4 };
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
			int repReso[3] = { acc.size(0) / 2, acc.size(1) / 2, acc.size(2) / 2 };
			int repPos[3];
			// z > y, x
			int vid = tid;
			int x = 0, y = 0, z = 0;
			z = powf(3.f * vid, 0.3333333f) + 1.1f;
			while (z * (z + 1) * (2 * z + 1) / 6 > vid) { z--; }
			if (z >= repReso[2]) return;
			vid -= z * (z + 1) * (2 * z + 1) / 6;
			y = vid / (z + 1);
			x = vid % (z + 1);
			repPos[0] = x; repPos[1] = y; repPos[2] = z;
			//if (z == repReso[2] - 1) {
			//	printf("[%d] = (%d, %d, %d)\n", (int)tid, repPos[0], repPos[1], repPos[2]);
			//}
			// traverse orbits [S_3] x [Reflect_3] 
			using T = typename Acc::Scalar;
			T orbitSum = 0;
			const int permu[6][3] = {
				{0,2,1},{2,1,0},{1,0,2},
				{2,0,1},{1,2,0},{0,1,2}
			};
#if 1
#pragma unroll
			for (int i = 0; i < 6; i++) {
				const int orbit[3] = { repPos[permu[i][0]], repPos[permu[i][1]], repPos[permu[i][2]] };
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
				const int orbit[3] = { repPos[permu[i][0]], repPos[permu[i][1]], repPos[permu[i][2]] };
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
		if (tid >= n_tol) return;
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
		if (tid >= n_tol) return;
		auto lastdiff = dstdif(tid);
		//auto srcval = accs[tid];
		srcdif(tid) += tempdif(tid) * lastdiff;
	}

	template<typename AccS, typename Accd, typename AccTemp, typename Kernel>
	__global__ void conv_kernel(int ndst, AccS accs, Accd accd, AccTemp tmpacc, Kernel kernel) {
		// traverse dst tensor
		size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
		if (tid >= ndst) return;
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
				off[2] >= 0 && off[2] < accs.dim[2]
				) {
				val = accs(off[0], off[1], off[2]);
				weightSum += weight;
			}
			else if (ker.is_period()) {
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
		if (tid >= nsrc) return;
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
			int dstp[3] = { i - off[0], j - off[1], k - off[2] };
			if (dstp[0] >= 0 && dstp[0] < accd.dim[0] &&
				dstp[1] >= 0 && dstp[1] < accd.dim[1] &&
				dstp[2] >= 0 && dstp[2] < accd.dim[2]) {
				auto dst_ws = tmpacc(dstp[0], dstp[1], dstp[2]);
				diffsum += weight * accd(dstp[0], dstp[1], dstp[2]) / dst_ws;
			}
			else if (ker.is_period()) {
				dstp[0] = (dstp[0] + accd.dim[0]) % accd.dim[0];
				dstp[1] = (dstp[1] + accd.dim[1]) % accd.dim[1];
				dstp[2] = (dstp[2] + accd.dim[2]) % accd.dim[2];
				auto dst_ws = tmpacc(dstp[0], dstp[1], dstp[2]);
				diffsum += weight * accd(dstp[0], dstp[1], dstp[2]) / dst_ws;
			}
		}
		accs(i, j, k) = diffsum;
	}

	// map 32 dst elements to 8(x) warps
	// { t_i } -> { f_j } 
	// f_j = Reduce_i ker(i, t_i, p_j)
	template<typename Config, typename AccS, typename Accd, typename Kernel, typename ReduceOpAcc>
	__global__ void fullCon_Reduce(Config config, AccS accs, Accd accd, Kernel ker, ReduceOpAcc red_out) {
		typedef typename Accd::Scalar Td;
		typedef typename AccS::Scalar Ts;
		Config conf = config;
		constexpr int sBatch = Config::szSrcBatch;
		constexpr int nParam = Kernel::szParamGroup;
		constexpr int BlockSize = Config::blockSize;
		constexpr int nThreadPerParam = BlockSize / sBatch;
		constexpr int nDstInBlock = BlockSize / sBatch;
		int n_parampass = ceilDiv(accs.size() / nParam, sBatch);
		__shared__ Ts param[sBatch][nParam];
		__shared__ Td res[sBatch / 2][nDstInBlock < 32 ? 32 : nDstInBlock];
		size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
		//if (!conf.is_valid(tid)) return;

		typename ReduceOpAcc::Scalar redop;

		int warpId = blockIdx.x / 32;
		int laneId = blockIdx.x % 32;

		//int did = 32 * blockIdx.x + laneId;

		//int sgroupid = blockIdx.x * sBatch + warpId;

		//Td partialRes{ 0 };

		int blockParamId = threadIdx.x / nThreadPerParam;
		int blockDstId = threadIdx.x % nThreadPerParam;

		int did = blockDstId + blockIdx.x * nDstInBlock;

		for (int parampass = 0; parampass < n_parampass; parampass++) {
			// load parameter
			for (int loaded = 0; loaded < sBatch * nParam; loaded += BlockSize) {
				int id = threadIdx.x + loaded;
				if (id < sBatch * nParam) {
					param[id / nParam][id % nParam] = accs(id / nParam, id % nParam);
				}
			}
			__syncthreads();
			int batchid = blockParamId + sBatch * parampass;
			int pos[3];
			accd.index(did, pos);
			Td val = ker(batchid, param[batchid], pos);
			redop.accept(val);
			__syncthreads();
		}


		if (warpId >= sBatch / 2) {
			for (int dstid = laneId; dstid < nDstInBlock; dstid += 32) {
				Td val{ 0 };
				val = redop.accepted();
				res[warpId - sBatch / 2][dstid] = val;
			}
		}
		__syncthreads();
		if (warpId < sBatch / 2) {
			for (int dstid = laneId; dstid < nDstInBlock; dstid += 32) {
				res[warpId][dstid] += redop.accepted();
			}
		}
		__syncthreads();
		if constexpr (sBatch / 2 >= 4) {
			constexpr int half = sBatch / 2 / 2;
			if (warpId < half) {
				for (int dstid = laneId; dstid < nDstInBlock; dstid += 32) {
					res[warpId][dstid] += res[warpId + half][dstid];
				}
			}
			__syncthreads();
		}
		if constexpr (sBatch / 2 >= 8) {
			constexpr int half = sBatch / 2 / 2 / 2;
			if (warpId < half) {
				for (int dstid = laneId; dstid < nDstInBlock; dstid += 32) {
					res[warpId][dstid] += res[warpId + half][dstid];
				}
			}
			__syncthreads();
		}
		if constexpr (sBatch / 2 >= 16) {
			constexpr int half = sBatch / 2 / 2 / 2;
			if (warpId < half) {
				for (int dstid = laneId; dstid < nDstInBlock; dstid += 32) {
					res[warpId][dstid] += res[warpId + half][dstid];
				}
			}
			__syncthreads();
		}
		if (warpId < 1) {
			for (int dstid = laneId; dstid < nDstInBlock; dstid += 32) {
				Td result = redop.reduce(res[warpId][dstid] + res[warpId + 1][dstid]);
				accd[did] = result;
				red_out[did] = redop;
			}
		}
	}


	// map 1 block dst elements to 8(x) warps
	// { t_i } -> { f_j } 
	// f_j = Reduce_i ker(i, t_i, p_j)
	template<typename Config, typename AccS, typename Accd, typename Kernel, typename ReduceOpAcc ,typename PartialSrcGradAcc>
	__global__ void fullCon_Reduce_backward_stage(Config config, AccS accs, Accd accd, Kernel ker, ReduceOpAcc red, PartialSrcGradAcc srcGrad) {
		typedef typename Accd::Scalar Td;
		typedef typename AccS::Scalar Ts;
		Config conf = config;
		constexpr int szParamGroup = Kernel::szParamGroup;
		constexpr int sBatch = Config::szSrcBatch;
		constexpr int n_dstpass = Config::n_dstpass;
		constexpr int n_srcpass = Config::n_srcpass;
		constexpr int BlockSize = Config::blockSize;
		int GridSize = gridDim.x * blockDim.x;
		int n_dststride = ceilDiv(accd.size(), n_dstpass);
		int nBlocksInDststride = ceilDiv(n_dststride, BlockSize);
		int n_nBlocksInDststride = ceilDiv(accs.size(), sBatch * n_srcpass);
		int n_dst = conf.n_dst;
		__shared__ Ts param[sBatch][szParamGroup];
		__shared__ Ts kerval[sBatch];
		__shared__ Ts kerGradSum[sBatch];
		__shared__ Td res[sBatch][BlockSize / 32];
		__shared__ Ts grads[szParamGroup][sBatch][BlockSize / 32];

		size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
		
		int warpId = threadIdx.x / 32;
		int laneId = threadIdx.x % 32;

		for (int srcpass = 0; srcpass < n_srcpass; srcpass++) {
			int sid = srcpass * sBatch * n_nBlocksInDststride +
				blockIdx.x / nBlocksInDststride * sBatch;
			if (threadIdx.x < szParamGroup * sBatch) {
				int groupid = sid  + threadIdx.x / szParamGroup;
				param[threadIdx.x / szParamGroup][threadIdx.x % szParamGroup] = accs(groupid, threadIdx.x % szParamGroup);
				if (threadIdx.x % szParamGroup == 0) {
					kerval[threadIdx.x] = ker(groupid, param[threadIdx.x / szParamGroup]);
				}
			}
			__syncthreads();
			for (int dstpass = 0; dstpass < n_dstpass; dstpass++) {
				int did = dstpass * n_dststride + blockIdx.x % nBlocksInDststride * BlockSize + threadIdx.x;
				typename ReduceOpAcc::Scalar op;
				Td dsval = 0;
				bool d_valid = false;
				if (did < accd.size()) {
					dsval = accd[did];
					dsval = op.grad_result(dsval);
					d_valid = true;
				} 	
				//res[threadIdx.x] = dsval;
				//__syncthreads();
				Ts kergrad;
				// warp reduction
				for (int i = 0; i < sBatch; i++) {
					kergrad = op.grad(kerval[i], dsval);
					for (int offset = 32 / 2; offset > 0; offset /= 2) {
						kergrad += shlf_down(kergrad, offset);
					}
					if (laneId == 0) res[i][warpId] = kergrad;
				}
				__syncthreads();
				// block reduction 
				// do not use blocksize > 1024  !!!
				if constexpr (BlockSize / 32 < 32) {
					int baseBatch = 0;
					int batchid = warpId + baseBatch;
					while (batchid < sBatch) {
						if (laneId < BlockSize / 32) {
							kergrad = res[batchid][laneId];
						} else {
							kergrad = 0;
						}
						for (int offset = 32; offset > 0; offset /= 2) {
							kergrad += shlf_down(kergrad, offset);
						}
					
						if (laneId == 0) kerGradSum[batchid] += kergrad;

						batchid += BlockSize / 32;
					}
				}
				__syncthreads();

			}
			if (threadIdx.x < sBatch) {
				srcGrad(sid  + threadIdx.x, blockIdx.x % nBlocksInDststride) = kerGradSum[threadIdx.x];
			}
		}
	}

	template<typename Config, typename AccS, typename Kernel, typename PartialSrcGradAcc, typename SrcGradAcc>
	__global__ void fullCon_Reduce_backward(Config config, AccS accs, Kernel ker, PartialSrcGradAcc srcGradGroup, SrcGradAcc gradacc) {
		typedef typename AccS::Scalar Ts;
		Config conf = config;
		constexpr int szParamGroup = Kernel::szParamGroup;
		constexpr int sBatch = Config::szSrcBatch;
		constexpr int n_dstpass = Config::n_dstpass;
		constexpr int n_srcpass = Config::n_srcpass;
		constexpr int BlockSize = Config::blockSize;
		// for gather
		constexpr int BlockSizeGather = Config::blockSizeGather;
		constexpr int sBatchGather = Config::szSrcBatchGather;
		constexpr int n_parampass = Config::n_parampassGather;

		int sizeGroupGrads = srcGradGroup.size(1);
		int n_gradpass = sizeGroupGrads / BlockSizeGather;
		int nBlocksParam = ceilDiv(accs.size() / szParamGroup, n_parampass * sBatchGather);

		int GridSize = gridDim.x * blockDim.x;
		int n_dststride = ceilDiv(conf.n_dst, n_dstpass);
		int nBlocksInDststride = ceilDiv(n_dststride, BlockSize);
		int n_nBlocksInDststride = ceilDiv(accs.size(), sBatch * n_srcpass);
		//int n_dst = conf.n_dst;
		//int n_srcblocks = conf.n_src / sBatch + (conf.n_src % sBatch == 0 ? 0 : 1);
		// one group contains multigrid grad stride
		//int n_gradstride = conf.n_gradstride;
		//int n_groupgrads = conf.n_groupgrads;
		__shared__ Ts param[sBatch][szParamGroup];
		__shared__ Ts kerval[sBatchGather];
		__shared__ Ts kerGrad[sBatchGather][BlockSizeGather / 32 < 32 ? 32 : (BlockSizeGather / 32)];
		__shared__ Ts grads[sBatchGather][szParamGroup];

		int tid = blockIdx.x * blockDim.x + threadIdx.x;
		int warpId = threadIdx.x / 32;
		int laneId = threadIdx.x % 32;
		

		for (int parampass = 0; parampass < n_parampass; parampass++) {
			for (int batchid = 0; batchid < sBatchGather; batchid++) {
				int sgroudid = nBlocksParam * parampass * batchid + blockIdx.x;

				//Ts param[szParamGroup];
				//conf.gatherParam(accs, sgroudid, param[batchid]);
				for (int k = 0; k < szParamGroup; k++) {
					param[batchid][k] = accs(sgroudid, k);
				}

				Ts kerGradPass{ 0. };
				for (int gradpass = 0; gradpass < n_gradpass; gradpass++) {
					int did = gradpass * BlockSize + threadIdx.x;
					if (did < sizeGroupGrads) kerGradPass += srcGradGroup(sgroudid, did);
				}
				// warp reduce
				for (int offset = 32; offset > 0; offset /= 2) {
					kerGradPass += shlf_down(kerGradPass, offset);
				}
				if constexpr (BlockSize / 32 <= 32) {
					if (laneId == 0) kerGrad[batchid][warpId] = kerGradPass;
				}
			}
			__syncthreads();
			for (int batchoffset = 0; batchoffset < sBatchGather; batchoffset += BlockSize / 32) {
				int batchid = (batchoffset + warpId);
				if (batchid >= sBatchGather) continue;
				int sgroudid = nBlocksParam * parampass * batchid + blockIdx.x;
				Ts kerGradPass{ 0. };
				// reduce block
				// Todo : use other warp deal with other batch, instead of loop on batch
				/*if (warpId == 0)*/ {
					if (laneId < BlockSize / 32) {
						kerGradPass = kerGrad[batchid][laneId];
					}
					else {
						kerGradPass = 0;
					}
					for (int offset = 0; offset < 32; offset++) {
						kerGradPass += shlf_down(kerGradPass, offset);
					}
					// calculate differential on param
					if (laneId == 0) {
						ker.grad(sgroudid, param[batchid], grads[batchid]);
					}
					if constexpr (szParamGroup <= 32) {
						if (threadIdx.x < szParamGroup) {
							//grads[batchid][threadIdx.x] *= shlf_(kerGradPass, 0);
							gradacc(sgroudid, threadIdx.x) = grads[batchid][threadIdx.x] * shlf_(kerGradPass, 0);
						}
					}
				}
			}
			__syncthreads();
		}
	}
}
