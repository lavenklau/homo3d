#define _USE_MATH_DEFINES
#include "TensorExpression.h"
#include "matlab/matlab_utils.h"
#include "voxelIO/openvdb_wrapper_t.h"

#define cuda_error_check do{ \
	auto err = cudaGetLastError(); \
	if (err != 0) { \
		printf("\x1b[31mCUDA error occured at line %d in file %s, error type %s \x1b[0m\n", __LINE__,__FILE__, cudaGetErrorName(err));\
	} \
}while(0)

namespace homo {
	void tensor2matlab(const std::string& tname, const TensorView<float>& tf) {
		Eigen::Matrix<float, -1, 1> tfdata(tf.size(), 1);
		cudaMemcpy2D(tfdata.data(), tf.size(0) * sizeof(float),
			tf.data(), tf.getPitchT() * sizeof(float), tf.size(0) * sizeof(float), tf.size(1) * tf.size(2), cudaMemcpyDeviceToHost);
		cuda_error_check;
		eigen2ConnectedMatlab(tname, tfdata);
	}
	void tensor2matlab(const std::string& tname, const TensorView<double>& tf) {
		Eigen::Matrix<double, -1, 1> tfdata(tf.size(), 1);
		cudaMemcpy2D(tfdata.data(), tf.size(0) * sizeof(double),
			tf.data(), tf.getPitchT() * sizeof(double), tf.size(0) * sizeof(double), tf.size(1) * tf.size(2), cudaMemcpyDeviceToHost);
		cuda_error_check;
		eigen2ConnectedMatlab(tname, tfdata);
	}

	void tensor2vdb(const std::string& fname, const TensorView<float>& tf) {
		std::vector<float> vhost(tf.size());
		cudaMemcpy2D(vhost.data(), tf.size(0) * sizeof(float),
			tf.data(), tf.getPitchT() * sizeof(float), tf.size(0) * sizeof(float), tf.size(1) * tf.size(2), cudaMemcpyDeviceToHost);
		int gsize[3] = { tf.size(0),tf.size(1),tf.size(2) };
		openvdb_wrapper_t<float>::lexicalGrid2openVDBfile(fname, gsize, vhost);
	}
	void tensor2vdb(const std::string& fname, const TensorView<double>& tf) {
		std::vector<double> vhost(tf.size());
		cudaMemcpy2D(vhost.data(), tf.size(0) * sizeof(double),
			tf.data(), tf.getPitchT() * sizeof(double), tf.size(0) * sizeof(double), tf.size(1) * tf.size(2), cudaMemcpyDeviceToHost);
		std::vector<float> vhostf(vhost.begin(), vhost.end());
		int gsize[3] = { tf.size(0),tf.size(1),tf.size(2) };
		openvdb_wrapper_t<float>::lexicalGrid2openVDBfile(fname, gsize, vhostf);
	}

	void loadvdb(const std::string& filename, std::vector<int> pos[3], std::vector<float>& gridvalues) {
		openvdb_wrapper_t<float>::openVDBfile2grid(filename, pos, gridvalues);
	}
}
