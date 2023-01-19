#include "cuda_runtime.h"
#include "matlab/matlab_utils.h"

namespace culib {
	void devArray2matlab(const char* pname, float* pdata, size_t len) {
		Eigen::Matrix<float, -1, 1> vec(len, 1);
		cudaMemcpy(vec.data(), pdata, sizeof(float) * len, cudaMemcpyDeviceToHost);
		eigen2ConnectedMatlab(pname, vec);
	}
	void devArray2matlab(const char* pname, double* pdata, size_t len) {
		Eigen::Matrix<double, -1, 1> vec(len, 1);
		cudaMemcpy(vec.data(), pdata, sizeof(double) * len, cudaMemcpyDeviceToHost);
		eigen2ConnectedMatlab(pname, vec);
	}
	void devArray2matlab(const char* pname, int* pdata, size_t len) {
		Eigen::Matrix<int, -1, 1> vec(len, 1);
		cudaMemcpy(vec.data(), pdata, sizeof(int) * len, cudaMemcpyDeviceToHost);
		eigen2ConnectedMatlab(pname, vec);
	}

}
