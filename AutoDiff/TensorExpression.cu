#include "TensorExpression.h"
#include "cuda_runtime.h"
// #include "voxelIO/openvdb_wrapper_t.h"  //openEXR not compatible :<
#include <algorithm>
#include "culib/lib.cuh"

using namespace homo;
using namespace culib;


template<typename T>
__global__ void vdb2tensor_kernel(TensorView<T> tf, cudaTextureObject_t tensorTex) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int reso[3] = { tf.size(0),tf.size(1),tf.size(2) };
	int ne = reso[0] * reso[1] * reso[2];
	if (tid >= ne) return;
	int epos[3] = { tid % reso[0], tid / reso[0] % reso[1], tid / (reso[0] * reso[1]) };

	float p[3] = { float(epos[0]) / reso[0], float(epos[1]) / reso[1], float(epos[2]) / reso[2] };
	float f = tex3D<float>(tensorTex, p[0], p[1], p[2]);
	tf(tid) = f;
}


void homo::vdb2tensor(const std::string& fname, TensorView<float> tf, bool interpolation)
{
	printf("reading density %s...", fname.c_str());
	std::vector<int> pos[3];
	std::vector<float> value;
	// openEXR not compatible :<
	//openvdb_wrapper_t<float>::openVDBfile2grid(fname, pos, value);
	loadvdb(fname, pos, value);
	
	//auto pv = std::tie(pos, value);
	int origin[3];
	int reso[3];
	for (int j = 0; j < 3; j++) {
		origin[j] = *std::min_element(pos[j].begin(), pos[j].end());
		reso[j] = 1 + *std::max_element(pos[j].begin(), pos[j].end()) - origin[j];
	}
	printf(" reso = (%d, %d, %d)\n", reso[0], reso[1], reso[2]);
	int ne = reso[0] * reso[1] * reso[2];
	std::vector<float> newvalues(ne, 0);
	for (int i = 0; i < value.size(); i++) {
		int p[3] = { pos[0][i] - origin[0], pos[1][i] - origin[1], pos[2][i] - origin[2] };
		int lexid = p[0] + p[1] * reso[0] + p[2] * reso[0] * reso[1];
		newvalues[lexid] = value[i];
	}

	// if not interpolate density
	if (!interpolation) {
		cudaMemcpy2D(tf.data(), tf.getPitchT() * sizeof(float),
			newvalues.data(), tf.size(0) * sizeof(float),
			tf.size(0) * sizeof(float),
			tf.size(1) * tf.size(2), cudaMemcpyHostToDevice);
		cuda_error_check;
		return;
	}

	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray* cuArray;
	cudaExtent extent{ reso[0],reso[1],reso[2] };
	//cudaMalloc3DArray(cudaArray_t * array, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int flags __dv(0));
	CheckErr(cudaMalloc3DArray(&cuArray, &channelDesc, extent));
	// Copy to device memory some data located at address h_data
	// in host memory cudaMemcpy3DParms copyParams = {0};
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr(newvalues.data(), reso[0] * sizeof(float), reso[1], reso[2]);
	copyParams.dstArray = cuArray;
	copyParams.extent = extent;
	copyParams.kind = cudaMemcpyHostToDevice;
	CheckErr(cudaMemcpy3D(&copyParams));
	//CheckErr(cudaMemcpyToArray(cuArray, 0, 0, values.data(), values.size(), cudaMemcpyHostToDevice)); // [deprecated]

	// Specify texture
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;
	// Set texture description parameters
	struct cudaTextureDesc tensorTexDesc;
	memset(&tensorTexDesc, 0, sizeof(tensorTexDesc));
	tensorTexDesc.addressMode[0] = cudaAddressModeBorder;
	tensorTexDesc.addressMode[1] = cudaAddressModeBorder;
	tensorTexDesc.addressMode[2] = cudaAddressModeBorder;
	tensorTexDesc.filterMode = cudaFilterModeLinear;
	tensorTexDesc.readMode = cudaReadModeElementType;
	tensorTexDesc.normalizedCoords = 1;
	// create texture object
	cudaTextureObject_t tensorTex = 0;
	CheckErr(cudaCreateTextureObject(&tensorTex, &resDesc, &tensorTexDesc, NULL));

	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, tf.size(), 256);
	vdb2tensor_kernel << <grid_size, block_size >> > (tf, tensorTex);
	cudaDeviceSynchronize();
	cuda_error_check;

	CheckErr(cudaDestroyTextureObject(tensorTex));
	CheckErr(cudaFreeArray(cuArray));
	cuda_error_check;
}

template<typename T>
__global__ void projectTensor_kernel(TensorView<T> tf, float beta, float tau, float a = 1.f, float b = 0.f) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= tf.size()) return;
	int eid = tid;
	float rho = a * tf(eid) + b;
	rho = tanproj(rho, beta, tau);
	if (rho < 0.5) rho = 1e-9;
	if (rho > 0.5) rho = 1;
	tf(eid) = rho;
}

void homo::tensorProject(TensorView<float> tf, float beta, float tau, float a, float b) {
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, tf.size(), 256);
	projectTensor_kernel << <grid_size, block_size >> > (tf, beta, tau, a, b);
	cudaDeviceSynchronize();
	cuda_error_check;
}
