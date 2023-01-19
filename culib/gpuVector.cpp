#include "cuda_runtime.h"
#include "matlab/matlab_utils.h"
#include "Eigen/Eigen"

void pass_matrix_to_matlab(const char* namestr, float* pdata, int nrows, int ncols, int wordpitch, bool rowmajor = false) {
	Eigen::Matrix<float, -1, -1> mat(nrows, ncols);
	if (rowmajor) {
		mat.resize(ncols, nrows);
		cudaMemcpy2D(mat.data(), ncols * sizeof(float), pdata, wordpitch * sizeof(float),
			ncols * sizeof(float), nrows, cudaMemcpyDeviceToHost);
		mat.transposeInPlace();
	}
	else {
		mat.resize(nrows, ncols);
		cudaMemcpy2D(mat.data(), nrows * sizeof(float), pdata, wordpitch * sizeof(float),
			nrows * sizeof(float), ncols, cudaMemcpyDeviceToHost);
	}
	eigen2ConnectedMatlab(namestr, mat);
}

void pass_matrix_to_matlab(const char* namestr, double* pdata, int nrows, int ncols, int wordpitch, bool rowmajor = false) {
	Eigen::Matrix<double, -1, -1> mat(nrows, ncols);
	if (rowmajor) {
		mat.resize(ncols, nrows);
		cudaMemcpy2D(mat.data(), ncols * sizeof(double), pdata, wordpitch * sizeof(double),
			ncols * sizeof(double), nrows, cudaMemcpyDeviceToHost);
		mat.transposeInPlace();
	}
	else {
		mat.resize(nrows, ncols);
		cudaMemcpy2D(mat.data(), nrows * sizeof(double), pdata, wordpitch * sizeof(double),
			nrows * sizeof(double), ncols, cudaMemcpyDeviceToHost);
	}
	eigen2ConnectedMatlab(namestr, mat);

}

void pass_matrix_to_matlab(const char* namestr, int* pdata, int nrows, int ncols, int wordpitch, bool rowmajor = false) {
	Eigen::Matrix<int, -1, -1> mat(nrows, ncols);
	if (rowmajor) {
		mat.resize(ncols, nrows);
		cudaMemcpy2D(mat.data(), ncols * sizeof(int), pdata, wordpitch * sizeof(int),
			ncols * sizeof(int), nrows, cudaMemcpyDeviceToHost);
		mat.transposeInPlace();
	}
	else {
		mat.resize(nrows, ncols);
		cudaMemcpy2D(mat.data(), nrows * sizeof(int), pdata, wordpitch * sizeof(int),
			nrows * sizeof(int), ncols, cudaMemcpyDeviceToHost);
	}
	eigen2ConnectedMatlab(namestr, mat);
}
void pass_matrix_to_matlab(const char* namestr, bool* pdata, int nrows, int ncols, int wordpitch, bool rowmajor = false) {
	Eigen::Matrix<bool, -1, -1> mat(nrows, ncols);
	if (rowmajor) {
		mat.resize(ncols, nrows);
		cudaMemcpy2D(mat.data(), ncols * sizeof(bool), pdata, wordpitch * sizeof(bool),
			ncols * sizeof(bool), nrows, cudaMemcpyDeviceToHost);
		mat.transposeInPlace();
	}
	else {
		mat.resize(nrows, ncols);
		cudaMemcpy2D(mat.data(), nrows * sizeof(bool), pdata, wordpitch * sizeof(bool),
			nrows * sizeof(bool), ncols, cudaMemcpyDeviceToHost);
	}
	eigen2ConnectedMatlab(namestr, mat);
}
