#include "../cmdline.h"
#include "homoExpression.h"
#include "utils.h"
#include "AutoDiff/TensorExpression.h"
#include "AutoDiff/TensorExpression.cuh"
#include "culib/gpuVector.cuh"
#include "optimization/OCOptimizer.h"
#include "optimization/mmaOptimizer.h"
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>

using namespace homo;
using namespace culib;

// cos 2 pi kx, k = 1, 2 , 3, ..., n,  n must be less than 100 
template<typename T, int BlockSize = 256>
__global__ void randTribase_cos_kernel(TensorView<T> view, int n_period, float* coeff) {

	__shared__ float gSum[BlockSize / 32 / 2][32];
	__shared__ float cosv[3][100][32];

	constexpr float pi2 = 3.1415926 * 2;

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;
	int vid = blockIdx.x * 32 + laneId;

	if (n_period > 100) n_period = 100;

	int nbasis1st = n_period * 3;
	int nbasis2nd = n_period * n_period * 3;
	int nbasis = nbasis1st + nbasis2nd;

	float dm[3] = { view.size(0),view.size(1),view.size(2) };
	int siz = view.size();
	bool is_ghost = false;
	is_ghost = vid >= siz;
	float s = 0;
	float p[3];
	// initialize cos value
	if (!is_ghost) {
		int posi[3];
		view.index(vid, posi);
		p[0] = posi[0] / (dm[0] - 1);
		p[1] = posi[1] / (dm[1] - 1);
		p[2] = posi[2] / (dm[2] - 1);

		for (int i = warpId; i < n_period; i += BlockSize / 32) {
#pragma unroll
			for (int j = 0; j < 3; j++) cosv[j][i][laneId] = cosf(pi2 * (i + 1) * p[j]);
		}
	} else {
		for (int i = warpId; i < n_period; i += BlockSize / 32) {
#pragma unroll
			for (int j = 0; j < 3; j++) cosv[j][i][laneId] = 0;
		}
	}
	__syncthreads();
	
	if (!is_ghost) {
		for (int i = warpId; i < nbasis; i += BlockSize / 32) {
			if (i < nbasis1st) {
				s += coeff[i] * cosv[i % 3][i / 3][laneId];
			} else if (i < nbasis1st + nbasis2nd) {
				int j = (i - nbasis1st) / (n_period * n_period);
				int k = (i - nbasis1st) % (n_period * n_period);
				float cv1{ 0 };
				float cv2{ 0 };
				if (j == 0) {
					cv1 = cosv[1][k % n_period][laneId];
					cv2 = cosv[2][k / n_period][laneId];
				} else if (j == 1) {
					cv1 = cosv[0][k % n_period][laneId];
					cv2 = cosv[2][k / n_period][laneId];
				} else if (j == 2) {
					cv1 = cosv[0][k % n_period][laneId];
					cv2 = cosv[1][k / n_period][laneId];
				}
				s += coeff[i] * cv1 * cv2;
			} else {
			
			}
		}
	}
	//if (tid == 0) {
	//	printf("cv = (%f, %f, %f)\n", cosv[0][0][0], cosv[1][0][0], cosv[2][0][0]);
	//	printf("coeff = (%f, %f, %f)\n", coeff[0], coeff[1], coeff[2]);
	//}
	if (warpId >= 4) { gSum[warpId - 4][laneId] = s; }
	__syncthreads();
	if (warpId < 4) { gSum[warpId][laneId] += s; }
	__syncthreads();
	if (warpId < 2) { gSum[warpId][laneId] += gSum[warpId + 2][laneId]; }
	__syncthreads();
	if (warpId < 1) {
		s = gSum[warpId][laneId] + gSum[warpId + 1][laneId];
		if (!is_ghost) {
			view(vid) = s;
		}
	}
}

// cos 2 pi kx, k = 1, 2 , 3, ..., n,  n must be less than 50 
template<typename T, int BlockSize = 256>
__global__ void randTribase_sincos_kernel(TensorView<T> view, int n_period, float* coeff, bool centered = false) {

	__shared__ float gSum[BlockSize / 32 / 2][32];
	__shared__ float cosv[3][50][32];
	__shared__ float sinv[3][50][32];
	__shared__ glm::mat3 rot;

	constexpr float pi2 = 3.1415926 * 2;

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;
	int vid = blockIdx.x * 32 + laneId;

	if (n_period > 50) n_period = 50;

	int nbasis1st = n_period * 6;
	int nbasis2nd = n_period * n_period * 36;
	int nbasis = nbasis1st + nbasis2nd;

	if (centered && threadIdx.x == 0) {
		auto q = glm::normalize(glm::quat(coeff[nbasis], coeff[nbasis + 1], coeff[nbasis + 2], coeff[nbasis + 3]));
		rot = glm::toMat3(q);
	}
	__syncthreads();

	float dm[3] = { view.size(0),view.size(1),view.size(2) };
	int siz = view.size();
	bool is_ghost = false;
	is_ghost = vid >= siz;
	float s = 0;
	float p[3];
	// initialize cos and sin value
	if (!is_ghost) {
		int posi[3];
		view.index(vid, posi);
		p[0] = posi[0] / (dm[0] - 1);
		p[1] = posi[1] / (dm[1] - 1);
		p[2] = posi[2] / (dm[2] - 1);

		if (centered) {
			p[0] -= 0.5f; p[1] -= 0.5f; p[2] -= 0.5f; 
			glm::vec3 v(p[0], p[1], p[2]);
			v = rot * v;
			p[0] = v[0]; p[1] = v[1]; p[2] = v[2];
		}

		for (int i = warpId; i < n_period; i += BlockSize / 32) {
#pragma unroll
			for (int j = 0; j < 3; j++) {
				cosv[j][i][laneId] = cosf(pi2 * (i + 1) * p[j]);
				sinv[j][i][laneId] = sinf(pi2 * (i + 1) * p[j]);
			}
		}
	} else {
		for (int i = warpId; i < n_period; i += BlockSize / 32) {
#pragma unroll
			for (int j = 0; j < 3; j++) {
				cosv[j][i][laneId] = 0;
				sinv[j][i][laneId] = 0;
			}
		}
	}
	__syncthreads();
	
	if (!is_ghost) {
		for (int i = warpId; i < nbasis; i += BlockSize / 32) {
			if (i < nbasis1st) {
				if (i < nbasis1st / 2) {
					s += coeff[i] * cosv[i % 3][i / 3][laneId];
				}
				else {
					int inext = i % (nbasis1st / 2);
					s += coeff[i] * sinv[(inext % 3)][inext / 3][laneId];
				}
			} else if (i < nbasis1st + nbasis2nd) {
				// todo
				int j = (i - nbasis1st) / (n_period * 6);
				int k = (i - nbasis1st) % (n_period * 6);
				float cv1{ 0 };
				float cv2{ 0 };
				if (j < n_period * 3) {
					cv1 = cosv[j / n_period][j % n_period][laneId];
				} else {
					cv1 = sinv[j / n_period - 3][j % n_period][laneId];
				}
				if (k < n_period * 3) {
					cv2 = cosv[k / n_period][k % n_period][laneId];
				} else {
					cv2 = sinv[k / n_period - 3][k % n_period][laneId];
				}
				s += coeff[i] * cv1 * cv2;
			} else {
			
			}
		}
	}
	//if (tid == 0) {
	//	printf("cv = (%f, %f, %f)\n", cosv[0][0][0], cosv[1][0][0], cosv[2][0][0]);
	//	printf("coeff = (%f, %f, %f)\n", coeff[0], coeff[1], coeff[2]);
	//}
	if (warpId >= 4) { gSum[warpId - 4][laneId] = s; }
	__syncthreads();
	if (warpId < 4) { gSum[warpId][laneId] += s; }
	__syncthreads();
	if (warpId < 2) { gSum[warpId][laneId] += gSum[warpId + 2][laneId]; }
	__syncthreads();
	if (warpId < 1) {
		s = gSum[warpId][laneId] + gSum[warpId + 1][laneId];
		if (!is_ghost) {
			view(vid) = s;
		}
	}
}

template<typename T>
void symmetrizeField(Tensor<T> field, cfg::Symmetry sym) {
	if (!FLAGS_usesym)
		return;
	if (sym == cfg::Symmetry::reflect3) {
		field.symmetrize(Reflection3);
	} else if (sym == cfg::Symmetry::reflect6) {
		field.symmetrize(Reflection6);
	} else if (sym == cfg::Symmetry::rotate3) {
		field.symmetrize(Rotate3);
	} else if (sym == cfg::Symmetry::NONE) {
	} else {
		printf("\033[31m unsupported symmetry\033[0m\n");
	}
}

template<typename T>
void randTri(Tensor<T> rho, cfg::HomoConfig config) {
	int n_period = config.initperiod;
	float volGoal = config.volRatio;
	auto view = rho.view();
	size_t block_size = 256;
	size_t grid_size = ceilDiv(view.size(), 32);
	int nbasis1st = n_period * 6;
	int nbasis2nd = n_period * n_period * 36;
	int nbasis = nbasis1st + nbasis2nd;
	if (config.winit == cfg::InitWay::randcenter || 
		config.winit == cfg::InitWay::rep_randcenter) {
		nbasis += 4;
	}
	float* coeffs;
	cudaMalloc(&coeffs, sizeof(float) * nbasis);

	if (config.winit == cfg::InitWay::rep_randcenter) {
		std::vector<float> coeffshost[1];
		printf("reading coefficients from %s...", config.inputrho.c_str());
		homoutils::readVectors(config.inputrho, coeffshost);
		printf("  %d read  [%s]\n", int(coeffshost[0].size()),
			(coeffshost[0].size() == nbasis ? "\033[32mMatch\033[0m" : "\033[31mUnmatch\033[0m"));
		cudaMemcpy(coeffs, coeffshost[0].data(), sizeof(float) * coeffshost[0].size(), cudaMemcpyHostToDevice);
		cuda_error_check;
		config.winit = cfg::InitWay::randcenter;
	} else {
		randArray(&coeffs, 1, nbasis, -1.f, 1.f);
	}

	// write coefficient to file
	std::vector<float> coeffvec[1];
	coeffvec[0].resize(nbasis);
	cudaMemcpy(coeffvec->data(), coeffs, sizeof(float) * nbasis, cudaMemcpyDeviceToHost);
	homoutils::writeVectors(getPath("coeff"), coeffvec);
	randTribase_sincos_kernel << <grid_size, block_size >> > (view, n_period, coeffs, config.winit == cfg::InitWay::randcenter);
	cudaDeviceSynchronize();
	cuda_error_check;
	size_t siz = view.size();

	symmetrizeField(rho, config.sym);

	float t_u = n_period * n_period * 37;
	float t_l = -t_u;
	float t = (t_u + t_l) / 2;
	for (int iter = 0; iter < 60; iter++) {
		t = (t_u + t_l) / 2;
		auto ker = [=] __device__(int id) {
			float val = view(id);
			float x = sigmoid(val, 15, t);
			float M = min(volGoal * 1.5f, 1.f);
			//float m = volGoal * 0.01f;
			float m = 0.001f;
			x = x * (M - m) + m;
			return x;
		};
		float s = sequence_sum(ker, siz, 0.f) / siz;
		float rel_err = abs(s - volGoal) / volGoal ;
		if (rel_err < 1e-3)  break;
		if (s > volGoal) t_l = t;
		if (s < volGoal) t_u = t;
		printf("searching level set isovalue t = %4.2e , v = %4.2f%% , it = %d   \n", t, s * 100, iter);
	}
	printf("\n");
	rho.mapInplace([=] __device__(int x, int y, int z, float val) {
		float v = sigmoid(val, 15, t);
		float M = min(volGoal * 1.5f, 1.f);
		//float m = volGoal * 0.01f;
		float m = 0.001f;
		v = v * (M - m) + m;
		return v;
	});
	cudaFree(coeffs);
}

struct ConvergeChecker {
private:
	constexpr static int circleLen = 10;
	double thres = 1e-3;
	double Obj[circleLen];
	int p_cir = 0;
private:
	int cirId(int id) {
		while (id < 0) id += circleLen;
		return id % circleLen;
	}
	double get(int id) { return Obj[cirId(id)]; }
	bool checkSeq(void) {
		bool outThres = false;
		// the change ratio is less than threshold for continous 3 iterations
		for (int i = 0; i < 3; i++) {
			outThres = outThres || abs(get(p_cir - i) - get(p_cir - i - 1)) / abs(get(p_cir - i - 1)) > thres;
		}
		return !outThres;
	}
public:
	ConvergeChecker(double thres_ = 1e-3) : thres(thres_) {}
	bool is_converge(int iter, double obj) {
		int eid = cirId(iter);
		p_cir = eid;
		Obj[eid] = obj;
		if (iter <= 11)
			return false;
		else
			return checkSeq();
	}
};


void initDensity(var_tsexp_t<>& rho, cfg::HomoConfig config);


