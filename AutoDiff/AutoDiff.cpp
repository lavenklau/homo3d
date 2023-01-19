#include "AutoDiff.h"
#include "cuda_runtime.h"

#ifdef AUTODIFF_WITH_MATLAB

#include "matlab/matlab_utils.h"

void data2matrix_g(const std::string& mtname, float* pdata, int m, int n /*= 1*/, int pitch /*= -1*/)
{
	Eigen::Matrix<float, -1, -1> datamat(m, n);
	if (pitch != -1) {
		cudaMemcpy2D(datamat.data(), m * sizeof(float), pdata, pitch, m * sizeof(float), n, cudaMemcpyDeviceToHost);
	} else {
		cudaMemcpy(datamat.data(), pdata, m * n * sizeof(float), cudaMemcpyDeviceToHost);
	}
	eigen2ConnectedMatlab(mtname, datamat);
}

void data2matrix_g(const std::string& mtname, double* pdata, int m, int n /*= 1*/, int pitch /*= -1*/)
{
	Eigen::Matrix<double, -1, -1> datamat(m, n);
	if (pitch != -1) {
		cudaMemcpy2D(datamat.data(), m * sizeof(double), pdata, pitch, m * sizeof(double), n, cudaMemcpyDeviceToHost);
	} else {
		cudaMemcpy(datamat.data(), pdata, m * n * sizeof(double), cudaMemcpyDeviceToHost);
	}
	eigen2ConnectedMatlab(mtname, datamat);
}

void data2matrix_h(const std::string& mtname, float* pdata, int m, int n /*= 1*/, int pitch /*= -1*/)
{
	Eigen::Matrix<float, -1, -1> datamat(m, n);
	if (pitch != -1) {
		cudaMemcpy2D(datamat.data(), m * sizeof(float), pdata, pitch, m * sizeof(float), n, cudaMemcpyHostToHost);
	} else {
		cudaMemcpy(datamat.data(), pdata, m * n * sizeof(float), cudaMemcpyHostToHost);
	}
	eigen2ConnectedMatlab(mtname, datamat);
}

void data2matrix_h(const std::string& mtname, double* pdata, int m, int n /*= 1*/, int pitch /*= -1*/)
{
	Eigen::Matrix<double, -1, -1> datamat(m, n);
	if (pitch != -1) {
		cudaMemcpy2D(datamat.data(), m * sizeof(double), pdata, pitch, m * sizeof(double), n, cudaMemcpyHostToHost);
	} else {
		cudaMemcpy(datamat.data(), pdata, m * n * sizeof(double), cudaMemcpyHostToHost);
	}
	eigen2ConnectedMatlab(mtname, datamat);
}
#endif
