#include "OCOptimizer.h"
#include "cuda_runtime.h"
#include "culib/lib.cuh"
#include "AutoDiff/TensorExpression.h"
#include "cmdline.h"


using namespace homo;
using namespace culib;

template<typename T>
__global__ void update_kernel(int ne,
	const T* sens, T g,
	const T* rhoold, T* rhonew,
	T minRho, T stepLimit, T damp) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= ne) return;
	
	T rho = rhoold[tid];

	T B = -sens[tid] / g;
	if (B < 0) B = 0.01f;
	T newrho = powf(B, damp) * rho;
	//if (tid == 0) {
	//	printf("sens = %.4e  g = %.4e  damp = %.4e  rho =%.4e  newrho = %.4e\n",
	//		sens[tid], g, damp, rho, newrho);
	//}

	if (newrho - rho < -stepLimit) newrho = rho - stepLimit;
	if (newrho - rho > stepLimit) newrho = rho + stepLimit;
	if (newrho < minRho) newrho = minRho;
	if (newrho > 1) newrho = 1;
	rhonew[tid] = newrho;
}

template <typename ProjectKernel>
void update_imp(int ne, float minRho, const float *sens, float *rho, float volratio,
 float step_limit, float damp, ProjectKernel projker)
{
	float* newrho;
	cudaMalloc(&newrho, sizeof(float) * ne);
	float maxSens = abs(parallel_maxabs(sens, ne));
	printf("max sens = %f\n", maxSens);
	float minSens = 0;
	for (int itn = 0; itn < 20; itn++) {
		float gSens = (maxSens + minSens) / 2;
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, ne, 256);
		update_kernel << <grid_size, block_size >> > (ne, sens, gSens, rho, newrho,
			minRho, step_limit, damp);
		cudaDeviceSynchronize();
		cuda_error_check;
		auto rhoPhys = [=] __device__ (int tid) {
			var_exp_t<> rhovar(newrho[tid]);
			auto proj = projker;
			//auto rhoexp = proj.template evalExp(rhovar);
			float xphys = proj.eval(rhovar).value();
			return xphys;
		};
		float curVol = sequence_sum(rhoPhys, ne, 0.f) / ne;
		printf("[OC] : g = %.4e   vol = %4.2f%% (Goal %4.2f%%)       \r", gSens, curVol * 100, volratio * 100);
		if (curVol < volratio - 0.0001) {
			maxSens = gSens;
		}
		else if (curVol > volratio + 0.0001) {
			minSens = gSens;
		}
		else {
			break;
		}
	}
	printf("\n");
	cudaMemcpy(rho, newrho, sizeof(float) * ne, cudaMemcpyDeviceToDevice);
	cudaFree(newrho);
}

template<> void OCOptimizer<eye_umker_t<float>>::update(const float* sens, float* rho, float volratio) {
	update_imp<eye_umker_t<float>>(ne, minRho, sens, rho, volratio, step_limit, damp, projKer);
}
template<> void OCOptimizer<sigmoid_umker_t<float>>::update(const float* sens, float* rho, float volratio) {
	update_imp<sigmoid_umker_t<float>>(ne, minRho, sens, rho, volratio, step_limit, damp, projKer);
}

__device__ bool is_bounded(int p[3], int reso[3]) {
	return p[0] >= 0 && p[0] < reso[0] &&
		p[1] >= 0 && p[1] < reso[1] &&
		p[2] >= 0 && p[2] < reso[2];
}

template<typename Kernel>
__global__ void filterSens_kernel(
	int ne, devArray_t<int, 3> reso, size_t pitchT,
	const float* sens, const float* rho, const float* weightSum, float* newsens, Kernel wfunc) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= ne) return;
	int epos[3] = { tid % reso[0], tid / reso[0] % reso[1], tid / (reso[0] * reso[1]) };
	float wsum = 0;
	int ereso[3] = { reso[0],reso[1],reso[2] };
	Kernel ker = wfunc;
	float sum = 0;
	for (int nei = 0; nei < wfunc.size(); nei++) {
		int offset[3];
		ker.neigh(nei, offset);
		float w = ker.weight(offset);
		int neighpos[3] = { epos[0] + offset[0], epos[1] + offset[1], epos[2] + offset[2] };
		if (ker.is_period()) {
			for (int i = 0; i < 3; i++) neighpos[i] = (neighpos[i] + reso[i]) % reso[i];
		}
		if (is_bounded(neighpos, ereso)) {
			int neighid = neighpos[0] + (neighpos[1] + neighpos[2] * ereso[1]) * pitchT;
			//w /= weightSum[neighid];
			sum += sens[neighid] * rho[neighid] * w;
			wsum += w;
		}
	}
	int eid = epos[0] + (epos[1] + epos[2] * ereso[1]) * pitchT;
	sum /= wsum * rho[eid];
	newsens[eid] = sum;
}

template<typename Kernel>
__global__ void weightSum_kernel(int ne, devArray_t<int, 3> reso, size_t pitchT,
	float* weightSum, Kernel wfunc
) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= ne) return;
	int epos[3] = { tid % reso[0], tid / reso[0] % reso[1], tid / (reso[0] * reso[1]) };
	float wsum = 0;
	int ereso[3] = { reso[0],reso[1],reso[2] };
	Kernel ker = wfunc;
	float sum = 0;
	for (int nei = 0; nei < wfunc.size(); nei++) {
		int offset[3];
		ker.neigh(nei, offset);
		float w = ker.weight(offset);
		int neighpos[3] = { epos[0] + offset[0], epos[1] + offset[1], epos[2] + offset[2] };
		if (ker.is_period()) {
			for (int i = 0; i < 3; i++) neighpos[i] = (neighpos[i] + reso[i]) % reso[i];
		}
		if (is_bounded(neighpos, ereso)) {
			int neighid = neighpos[0] + (neighpos[1] + neighpos[2] * ereso[1]) * pitchT;
			wsum += w;
		}
	}
	int eid = epos[0] + (epos[1] + epos[2] * ereso[1]) * pitchT;
	weightSum[eid] = wsum;
}

template <typename ProjectKernel>
void filterSens_imp(int ne, float *sens, const float *rho, size_t pitchT, int reso[3], float radius, ProjectKernel projker)
{
	static float* filterWeightSum = nullptr;
	if (!filterWeightSum) {
		cudaMalloc(&filterWeightSum, sizeof(float) * reso[1] * reso[2] * pitchT);
		init_array(filterWeightSum, float(0), reso[1] * reso[2] * pitchT);
	}
	float* newsens;
	cudaMalloc(&newsens, sizeof(float) * reso[1] * reso[2] * pitchT);
	radial_convker_t<float, Linear> convker(radius, 0, true, FLAGS_periodfilt);
	devArray_t<int, 3> ereso{ reso[0],reso[1],reso[2] };
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, ne, 256);
	filterSens_kernel << <grid_size, block_size >> > (ne, ereso, pitchT, sens, rho, filterWeightSum, newsens, convker);
	cudaDeviceSynchronize();
	cuda_error_check;
	cudaMemcpy(sens, newsens, sizeof(float) * reso[1] * reso[2] * pitchT, cudaMemcpyDeviceToDevice);
	cudaFree(newsens);
}

template<> void OCOptimizer<eye_umker_t<float>>::filterSens(float* sens, const float* rho, size_t pitchT, int reso[3], float radius){
	filterSens_imp<eye_umker_t<float>>(ne, sens, rho, pitchT, reso, radius, projKer);
}
template<> void OCOptimizer<sigmoid_umker_t<float>>::filterSens(float* sens, const float* rho, size_t pitchT, int reso[3], float radius){
	filterSens_imp<sigmoid_umker_t<float>>(ne, sens, rho, pitchT, reso, radius, projKer);
}

template <typename Kernel>
__global__ void filterSens_Tensor_kernel(
	TensorView<float> sens, TensorView<float> rho, TensorView<float> newsens, Kernel wfunc) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ne = rho.size();
	int reso[3] = { rho.size(0),rho.size(1),rho.size(2) };
	if (tid >= ne) return;
	int epos[3] = { tid % reso[0], tid / reso[0] % reso[1], tid / (reso[0] * reso[1]) };
	float wsum = 0;
	Kernel ker = wfunc;
	float sum = 0;
	for (int nei = 0; nei < wfunc.size(); nei++) {
		int offset[3];
		ker.neigh(nei, offset);
		float w = ker.weight(offset);
		int neighpos[3] = { epos[0] + offset[0], epos[1] + offset[1], epos[2] + offset[2] };
		if (ker.is_period()) {
			for (int i = 0; i < 3; i++) neighpos[i] = (neighpos[i] + reso[i]) % reso[i];
		}
		if (is_bounded(neighpos, reso)) {
			//int neighid = neighpos[0] + (neighpos[1] + neighpos[2] * ereso[1]) * pitchT;
			//w /= weightSum[neighid];
			sum += sens(neighpos[0], neighpos[1], neighpos[2]) * rho(neighpos[0], neighpos[1], neighpos[2]) * w;
			wsum += w;
		} 
	}
	//int eid = epos[0] + (epos[1] + epos[2] * ereso[1]) * pitchT;
	sum /= wsum * rho(epos[0], epos[1], epos[2]);
	newsens(epos[0], epos[1], epos[2]) = sum;
}

template <typename ProjKernel>
void filterSens_imp(Tensor<float> sens, Tensor<float> rho, float radius) {
	Tensor<float> newsens(rho.getDim());
	newsens.reset(0);
	radial_convker_t<float, Linear> convker(radius, 0, true, false);
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, rho.size(), 256);
	filterSens_Tensor_kernel << <grid_size, block_size >> > (sens.view(), rho.view(), newsens.view(), convker);
	cudaDeviceSynchronize();
	cuda_error_check;
	sens.copy(newsens);
}

template<> void OCOptimizer<eye_umker_t<float>>::filterSens(Tensor<float> sens, Tensor<float> rho, float radius /*= 2*/) {
	filterSens_imp<eye_umker_t<float>>(sens, rho, radius);
}
template<> void OCOptimizer<sigmoid_umker_t<float>>::filterSens(Tensor<float> sens, Tensor<float> rho, float radius /*= 2*/) {
	filterSens_imp<sigmoid_umker_t<float>>(sens, rho, radius);
}

template<typename T>
__global__ void update_Tensor_kernel(TensorView<T> sens, T g,
	TensorView<T> rhoold, TensorView<T> rhonew,
	T minRho, T stepLimit, T damp) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ne = rhoold.size();
	if (tid >= ne) return;

	T rho = rhoold(tid);

	T B = -sens(tid) / g;
	if (B < 0) B = 0.01f;
	T newrho = powf(B, damp) * rho;

	if (newrho - rho < -stepLimit) newrho = rho - stepLimit;
	if (newrho - rho > stepLimit) newrho = rho + stepLimit;
	if (newrho < minRho) newrho = minRho;
	if (newrho > 1) newrho = 1;
	rhonew(tid) = newrho;
}

template <typename ProjKernel>
void update_imp(Tensor<float> sens, Tensor<float> rho, float volratio, float minRho, float step_limit, float damp, ProjKernel projker) {
	Tensor<float> newrho(rho.getDim());
	newrho.reset(0);
	float maxSens = abs(sens.maxabs());
	printf("max sens = %f\n", maxSens);
	float minSens = 0;
	for (int itn = 0; itn < 20; itn++) {
		float gSens = (maxSens + minSens) / 2;
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, rho.size(), 256);
		update_Tensor_kernel << <grid_size, block_size >> > (sens.view(), gSens, rho.view(), newrho.view(),
			minRho, step_limit, damp);
		cudaDeviceSynchronize();
		cuda_error_check;
		//float curVol = parallel_sum(newrho, ne) / ne;
		auto newrhoAcc=newrho.view();
		auto sumPhys = [=]__device__(int tid){
			var_exp_t<> rhovar(newrhoAcc(tid));
			auto proj = projker;
			return proj.eval(rhovar).value();
		};
		float curVol = sequence_sum(sumPhys, newrhoAcc.size(), 0.f) / newrho.size();
		printf("[OC] : g = %.4e   vol = %4.2f%% (Goal %4.2f%%)       \r", gSens, curVol * 100, volratio * 100);
		if (curVol < volratio - 0.0001) {
			maxSens = gSens;
		}
		else if (curVol > volratio + 0.0001) {
			minSens = gSens;
		}
		else {
			break;
		}
	}
	printf("\n");
	rho.copy(newrho);
}

template<> void OCOptimizer<eye_umker_t<float>>::update(Tensor<float> sens, Tensor<float> rho, float volratio) {
	update_imp<eye_umker_t<float>>(sens, rho, volratio, minRho, step_limit, damp, projKer);
}

template<> void OCOptimizer<sigmoid_umker_t<float>>::update(Tensor<float> sens, Tensor<float> rho, float volratio) {
	update_imp<sigmoid_umker_t<float>>(sens, rho, volratio, minRho, step_limit, damp, projKer);
}

