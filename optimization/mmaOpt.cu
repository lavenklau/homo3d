#include "mmaOpt.h"

#define __CUDACC__
#include "cuda.h"
#include "helper_cuda.h"
#include "helper_math.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_texture_types.h"
#include "texture_types.h"
#include "culib/gpuVector.cuh"
#include "culib/lib.cuh"
#include <tuple>

using namespace culib;
//#define DEBUG_MMA_OPT

//std::vector<int> nlineSearchStep;
//int lsindex = 0;
//int mma_iter;

extern void solveLinearHost(int nconstrain, const double* Alamhost, const double* ahost, double zet, double z, const double* bb, double* xhost);


void test_gVector(void)
{
	char cbuf[100];
#if 0
	gv::gVector<double> v1(20);
	std::vector<double> vhost(20);
	for (int i = 0; i < vhost.size(); i++) vhost[i] = i;
	v1.set(vhost.data());
	//v1.set(v1 * 2 >= 3 && v1 / 2 < 6, 0.333);
	v1.set(v1 * 2 >= 3 || v1 / 2 < 6, 0.333);
	//v1.set(v1 * 2 >= 3 ^ v1 / 2 < 6, 0.333);
	v1 = gv::power(v1, 2) / v1 * 2;
	//constexpr bool truv = gv::is_expression<decltype(v1 > 10)>::value;
	//typedef gv::scalar_t<double, 0> Vd;
	//Vd v(10);
	//constexpr bool truv = gv::is_expression<gv::scalar_t<double>>::value;
	//v1.lambda_test();
	for (int i = 0; i < v1.size(); i++) {
		std::cout << v1[i] << ", ";
	}
	std::cout << std::endl;

	gv::gVector<double> v2(2);
	v2[0] = 1; v2[1] = 2;
	//gv::gVector<double> v2arr = v2.dup(5, 2);
	gv::gVector<double> v2arr = v2.concat(v2).concat(v2).concat(v2).concat(v1);
	for (int i = 0; i < v2arr.size(); i++) {
		std::cout << v2arr[i] << ",";
	}
	std::cout << std::endl;
#elif 0
	gv::gVector<double> v1(100000);
	v1.set(1);
	v1.concat(v1 + 1).concat(v1 + 2).toMatlab("v1s");
	v1.pitchSumInPlace(1, 10000, 20000, true);
	v1.toMatlab("v1");
	gv::gVector<double> v2(1000);
	v1.gatherPitch(v2.data(), 1, 5, 20000, 1);
	v2.toMatlab("v2");
	scanf_s("%s", cbuf);
#elif 0
	double* tmp;
	size_t nbytepitch;
	cudaMallocPitch(&tmp, &nbytepitch, 10000 * sizeof(double), 10);
	std::cout << "nbytepich = " << nbytepitch << std::endl;
	gv::gVector<double> v1;
	v1.move(tmp, (nbytepitch / sizeof(double)) * 10);
	v1.set(1);
	v1.pitchSumInPlace(5, 10000, nbytepitch / sizeof(double));
	v1.toMatlab("v1");
	scanf_s("%s", cbuf);
#elif 1
	gv::gVector<double>::Init(1000);
	double* tmp;
	size_t nbytepitch;
	cudaMallocPitch(&tmp, &nbytepitch, 10000 * sizeof(double), 100);
	int wordpitch = nbytepitch / sizeof(double);
	std::cout << "-- nbyte = " << nbytepitch << ", " << "wordpitch = " << wordpitch << std::endl;
	gv::gVector<double> v1;
	v1.move(tmp, nbytepitch / sizeof(double) * 100);
	v1.set(0.5);
	gv::gVector<double> v2(nbytepitch / sizeof(double));
	v2.set(2.);
	//v2.dup(100, 10000, wordpitch);
	gv::gVector<double> v3 /*= v1 * v2.dup(100, 10000, wordpitch)*/;
	v3 = v1 * 2;
	v3.pitchSumInPlace(4, 10000, wordpitch);
	v3.toMatlab("v3");
	gv::gVector<double> v4(100);
	v3.gatherPitch(v4.data(), 1, 100, wordpitch, 1);
	v4.toMatlab("v4");
#endif
}
template<int N>
__host__ __device__ int round(int n) {
	if (n % N == 0) {
		return n;
	}
	else {
		int rn = (n + (N - 1)) / N * N;
		return rn;
	}
};

template<typename T, unsigned int blockSize>
__device__ void blockReduce(volatile T* sdata) {
	int len = blockSize;
	while (len > 64) {
		len /= 2;
		if (threadIdx.x < len) {
			sdata[threadIdx.x] += sdata[threadIdx.x + len];
		}
		__syncthreads();
	}
	if (threadIdx.x < 32) {
		warpReduce<T, blockSize>(sdata, threadIdx.x);
	}
}



template<typename Scalar, typename WT, int BlockSize, int BatchSize/*, bool RowMajor = true*/>
__global__ void matrix_mult_selfadjoint_rowmajor(
	const Scalar* pdata, int nwordvalid, int nwordpitch,
	int npitch, const WT* weight, Scalar* dst, int ndstwordpitch) {
#if 0
	constexpr int nWarp = BlockSize / 32;
	//__shared__ volatile Scalar sum[BatchSize][matBlockSize];
	//__shared__ volatile Scalar A[nWarp][32];
	//__shared__ volatile Scalar AT[nWarp][32];
	//__shared__ volatile WT W[nWarp][32];
	__shared__ volatile Scalar AAT[BlockSize];

	int nblockStride = lib_details::round<BlockSize>(nwordvalid) / BlockSize;
	int nbatch = lib_details::round<BatchSize>(npitch) / BatchSize;
	int nnbatch = nbatch * nbatch;

	int nblockBatch = nblockStride * nbatch * nblockStride * nbatch;

	int batchid = blockIdx.x / nblockBatch;

	int batchRowid = batchid / nbatch * BatchSize;
	int batchColid = batchid % nbatch * BatchSize;

	int rowid = blockIdx.x % nblockBatch / nblockStride / BatchSize + batchRowid;
	int colid = blockIdx.x % nblockBatch / nblockStride % BatchSize + batchColid;

	int eidoffset = blockIdx.x % nblockBatch % nblockStride * BlockSize;

	int eid = eidoffset + threadIdx.x;

	Scalar aat = 0;
	WT w = 0;
	bool validindex = eid < nwordvalid&& rowid < npitch&& colid < npitch;

	if (validindex) {
		aat = pdata[eid + nwordpitch * rowid] * pdata[eid + nwordpitch * colid] * weight[eid];
	}
	AAT[threadIdx.x] = att;
	__syncthreads();

	lib_details::blockReduce<Scalar, BlockSize>(AAT);

	if (threadIdx.x == 0) {
		dst[(rowid * npitch + colid) * ndstwordpitch + eid / BlockSize];
	}

#else
	constexpr int nWarp = BlockSize / 32;
	//__shared__ volatile Scalar sum[BatchSize][matBlockSize];
	__shared__ volatile Scalar A[BatchSize][nWarp][32];
	__shared__ volatile Scalar AT[BatchSize][nWarp][32];
	__shared__ volatile WT W[nWarp][32];
	__shared__ volatile Scalar AAT[BatchSize][BatchSize][BlockSize / 2];

	int nblockStride = ::round<BlockSize>(nwordvalid) / BlockSize;
	//int nnblockStride = nblockStride * nblockStride;
	int nbatch = ::round<BatchSize>(npitch) / BatchSize;
	//int nnbatch = nbatch * nbatch;
	int eid = blockIdx.x % nblockStride * blockDim.x + threadIdx.x;
	//int batchRowid = blockIdx.x / (nblockStride * nbatch);
	//int batchColid = blockIdx.x % (nblockStride * nbatch);

	int WarpId = threadIdx.x / 32;
	int warpTid = threadIdx.x % 32;

	// load weight
	bool validindex = eid < nwordvalid;
	WT w = 0;
	if (validindex) {
		w = weight[eid];
	}
	W[WarpId][warpTid] = w;
	__syncthreads();

	for (int batchRowoffset = 0; batchRowoffset < nbatch; batchRowoffset += BatchSize) {
		//validindex = rowid < npitch && validindex;
		// load A
		for (int i = 0; i < BatchSize; i++) {
			int rowid = i + batchRowoffset;
			Scalar a = 0;
			if (validindex && rowid < npitch) {
				a = pdata[eid + rowid * nwordpitch];
			}
			A[i][WarpId][warpTid] = a;
		}
		__syncthreads();

		for (int batchColoffset = 0; batchColoffset < nbatch; batchColoffset += BatchSize) {

			//validindex = validindex && colid < npitch;

			// load At matrix block elements
			for (int j = 0; j < BatchSize; j++) {
				int colid = j + batchColoffset;
				Scalar  at = 0;
				if (validindex && colid < npitch) {
					at = pdata[eid + colid * nwordpitch];
				}
				AT[j][WarpId][warpTid] = at;
			}
			__syncthreads();

			// multiplication
			double aat[BatchSize][BatchSize];
			for (int i = 0; i < BatchSize; i++) {
				for (int j = 0; j < BatchSize; j++) {
					aat[i][j] = A[i][WarpId][warpTid] * AT[j][WarpId][warpTid] * W[WarpId][warpTid];
				}
			}

			if (threadIdx.x >= (BlockSize / 2)) {
				for (int i = 0; i < BatchSize; i++) {
					for (int j = 0; j < BatchSize; j++) {
						int halfid = threadIdx.x - (BlockSize / 2);
						AAT[i][j][halfid] = aat[i][j];
					}
				}
			}
			__syncthreads();

			if (threadIdx.x < (BlockSize / 2)) {
				for (int i = 0; i < BatchSize; i++) {
					for (int j = 0; j < BatchSize; j++) {
						int halfid = threadIdx.x + (BlockSize / 2);
						AAT[i][j][threadIdx.x] += aat[i][j];
					}
				}
			}
			__syncthreads();

			if (BlockSize >= 64 * 4) {
				if (threadIdx.x < (BlockSize / 4)) {
					for (int i = 0; i < BatchSize; i++) {
						for (int j = 0; j < BatchSize; j++) {
							AAT[i][j][threadIdx.x] += AAT[i][j][threadIdx.x + BlockSize / 4];
						}
					}
				}
				__syncthreads();
			}

			if (BlockSize >= 64 * 8) {
				if (threadIdx.x < (BlockSize / 8)) {
					for (int i = 0; i < BatchSize; i++) {
						for (int j = 0; j < BatchSize; j++) {
							AAT[i][j][threadIdx.x] += AAT[i][j][threadIdx.x + BlockSize / 8];
						}
					}
				}
				__syncthreads();
			}

			if (BlockSize >= 64 * 16) {
				if (threadIdx.x < (BlockSize / 16)) {
					for (int i = 0; i < BatchSize; i++) {
						for (int j = 0; j < BatchSize; j++) {
							AAT[i][j][threadIdx.x] += AAT[i][j][threadIdx.x + BlockSize / 16];
						}
					}
				}
				__syncthreads();
			}

			// warpReduce
			if (threadIdx.x < 32) {
				for (int i = 0; i < BatchSize; i++) {
					for (int j = 0; j < BatchSize; j++) {
						AAT[i][j][threadIdx.x] += AAT[i][j][threadIdx.x + 32];
						AAT[i][j][threadIdx.x] += AAT[i][j][threadIdx.x + 16];
						AAT[i][j][threadIdx.x] += AAT[i][j][threadIdx.x + 8];
						AAT[i][j][threadIdx.x] += AAT[i][j][threadIdx.x + 4];
						AAT[i][j][threadIdx.x] += AAT[i][j][threadIdx.x + 2];
						AAT[i][j][threadIdx.x] += AAT[i][j][threadIdx.x + 1];
					}
				}
			}

			if (threadIdx.x == 0) {
				for (int i = 0; i < BatchSize; i++) {
					int rowid = i + batchRowoffset;
					for (int j = 0; j < BatchSize; j++) {
						int colid = j + batchColoffset;
						if (validindex && rowid < npitch && colid < npitch) {
							dst[(colid + rowid * npitch) * ndstwordpitch + blockIdx.x % nblockStride] = AAT[i][j][0];
						}
					}
				}
			}
		}
	}
#endif
}


template<typename Scalar, typename WT, int BlockSize, int BatchSize/*, bool RowMajor = true*/>
void matrix_AAt_rowmajor(const Scalar* pdata, int nwordvalid, int nwordpitch, int npitch, const WT* weight, Scalar* dst, int ndstwordpitch) {
	int nblockStride = gv::round<BlockSize>(nwordvalid) / BlockSize;

	int nbatch = gv::round<BatchSize>(npitch) / BatchSize;

	int nblock = nblockStride /** nbatch*/;
	
	size_t ndstpitch;
	Scalar* tmp_data;
	cudaMallocPitch(&tmp_data, &ndstpitch, nblockStride * sizeof(Scalar), npitch * npitch);
	gv::gVector<Scalar> vtmp;
	vtmp.move(tmp_data, ndstpitch / sizeof(Scalar) * npitch * npitch);
	vtmp.set(Scalar(0.));
	cuda_error_check;

	size_t grid_size = nblock, block_size = BlockSize;
	matrix_mult_selfadjoint_rowmajor<Scalar, WT, BlockSize, BatchSize><<<grid_size, block_size>>>(pdata, nwordvalid, nwordpitch, npitch, weight, vtmp.data(), ndstpitch / sizeof(Scalar));
	cudaDeviceSynchronize();
	cuda_error_check;

	vtmp.pitchSumInPlace(4, nblockStride, ndstpitch / sizeof(Scalar));

	auto gather = [=] __device__(int eid) {
		int row = eid / npitch;
		int col = eid % npitch;
		int k = col + row * ndstwordpitch;
		dst[k] = tmp_data[(col + row * npitch) * (ndstpitch / sizeof(Scalar))];
	};	
	parallel_do(npitch * npitch, 512, gather);
}

std::tuple<gv::gVector<double>, double, double> kktcheck(
	int nconstrain, int nvar,
	const gv::gVector<double>& xmma, const gv::gVector<double>& ymma, double z,
	const gv::gVector<double>& lam, const gv::gVector<double>& xsi, const gv::gVector<double>& eta,
	const gv::gVector<double>& mu, double zet, const gv::gVector<double>& s,
	const gv::gVector<double>& xmin, const gv::gVector<double>& xmax,
	const gv::gVector<double>& df0dx, const gv::gVector<double>& gval, const gv::gVector<double>& dgdx,
	int wordPitch,
	double a0, const gv::gVector<double>& a, const gv::gVector<double>& b, const gv::gVector<double>& c, const gv::gVector<double>& d
	)
{
	gv::gVector<double> dgdxtmp(dgdx);
	dgdxtmp.weightedPitchSumInPlace(4, nvar, wordPitch, lam.data(), true);
	gv::gVector<double> rex = df0dx + dgdxtmp.slice(0, nvar) - xsi + eta;
	auto rey = c + d * ymma - mu - lam;
	auto rez = a0 - zet - a.dot(lam);
	auto relam = gval - a * z - ymma + s;
	auto rexsi = xsi * (xmma - xmin);
	auto reeta = eta * (xmax - xmma);
	auto remu = mu * ymma;
	auto rezet = zet * z;
	auto res = lam * s;
	gv::gVector<double> residu = rex.concat(rey, rez, relam, rexsi, reeta, remu, rezet, res);
	double residumax = residu.abs().max();
	double residunorm = residu.norm();
	return std::make_tuple(std::move(residu), residunorm, residumax);
}

void update_pqlambda_gvec(
	int nconstrain, int nvar,
	const gv::gVector<double>& uxinv1, const gv::gVector<double>& xlinv1,
	const gv::gVector<double>& p0, const gv::gVector<double>& q0,
	const gv::gVector<double>& P, const gv::gVector<double> & Q,
	const gv::gVector<double>& lam,
	gv::gVector<double>& plam, gv::gVector<double>& qlam,
	gv::gVector<double>& gmat, gv::gVector<double>& gvec
) {
	cuda_error_check;
	int Pwordpitch = P.size() / nconstrain;
	int Qwordpitch = Q.size() / nconstrain;
	const double* lam_data = lam.data();
	const double* p0data = p0.data();
	const double* q0data = q0.data();
	const double* Pdata = P.data();
	const double* Qdata = Q.data();
	double* plamdata = plam.data();
	double* qlamdata = qlam.data();
	auto kernel = [=] __device__(int eid) {
		double psum = p0data[eid];
		double qsum = q0data[eid];
		for (int i = 0; i < nconstrain; i++) {
			psum += Pdata[i * Pwordpitch + eid] * lam_data[i];
			qsum += Qdata[i * Pwordpitch + eid] * lam_data[i];
		}
		plamdata[eid] = psum;
		qlamdata[eid] = qsum;
	};
	parallel_do(nvar, 512, kernel);
	// update gvector
	//gmat.set(0.);// delete here 
	gmat = P * uxinv1.dup(nconstrain, nvar, Pwordpitch) + Q * xlinv1.dup(nconstrain, nvar, Qwordpitch);
	gmat.pitchSumInPlace(4, nvar, Pwordpitch);
	gmat.gatherPitch(gvec.data(), 1, nconstrain, Pwordpitch, 1);
	cuda_error_check;

}

typedef std::tuple<
	gv::gVector<double>, gv::gVector<double>, double,
	gv::gVector<double>, gv::gVector<double>, gv::gVector<double>,
	gv::gVector<double>, double, gv::gVector<double>
> mma_arg_pack;

	//return std::make_tuple(
	//	std::move(x), std::move(y), std::move(z),
	//	std::move(lam), std::move(xsi), std::move(eta),
	//	std::move(mu), std::move(zet), std::move(s));
mma_arg_pack subsolv_g(int nconstrain, int nvar, double epsimin,
	const gv::gVector<double>& low, const gv::gVector<double>& upp,
	const gv::gVector<double>& alfa, const gv::gVector<double>& beta,
	const gv::gVector<double>& p0, const gv::gVector<double>& q0,
	const gv::gVector<double>& P, const gv::gVector<double>& Q,
	int Pwordpitch, int Qwordpitch,
	double a0, const gv::gVector<double>& a, const gv::gVector<double>& b, const gv::gVector<double>& c, const gv::gVector<double>& d
) {
	typedef gv::gVector<double> gVector;
	gVector x = 0.5 * (alfa + beta);

	double z = 1; // 1
	//double y = 1; // m
	
	gVector lam(nconstrain);
	lam.set(1);
	gVector y(nconstrain);
	y.set(1);

	gVector xsi = (1. / (x - alfa)).max(1);
	gVector eta = (1. / (beta - x)).max(1);

	gVector mu = (0.5 * c).max(1); // m

	double zet = 1; // 1
	//double s = 1;   // m
	gVector s(nconstrain);
	s.set(1);

	size_t itera = 0;

	double epsi = 1;

	gVector ux1, xl1, ux2, xl2, ux3, xl3;
	gVector uxinv1, xlinv1, uxinv2, xlinv2;
	gVector plam(nvar), qlam(nvar);
	gVector gvec(nconstrain);
	gVector dpsidx(nvar);
	gVector GG;

	// residual and step
	gVector rex(nvar), delx(nvar);
	gVector rey(nconstrain), dely(nconstrain);
	double rez = 0, delz = 0;
	gVector relam(nconstrain), dellam(nconstrain);
	gVector rexsi(nvar);
	gVector reeta(nvar);
	gVector remu(nconstrain);
	double rezet = 0;
	gVector res(nconstrain);

	// step vector
	gVector dlam(nconstrain), dy(nconstrain), dmu(nconstrain), ds(nconstrain);
	gVector dx(nvar), dxsi(nvar), deta(nvar);
	double dz,dzet;
	gVector xold(nvar), xsiold(nvar), etaold(nvar);
	gVector lamold(nconstrain), yold(nconstrain), muold(nconstrain), sold(nconstrain);
	double zold, zetold;
	//gVector Puxinv(nconstrain * Pwordpitch), Qxlinv(nconstrain * Qwordpitch);

	// tmp vector
	gVector gmat;
	gVector diagx;
	gVector diagxinv;
	gVector GGdelxdiagx;
	gVector GGxx(nconstrain);
	gVector blam;
	gVector bb;
	std::vector<double> bbhost(nconstrain + 1);
	std::vector<double> xhost(nconstrain + 1);
	std::vector<double> Alam_host(nconstrain * nconstrain);
	gVector Alam(nconstrain * nconstrain);
	std::vector<double> ahost(nconstrain);

#ifdef DEBUG_MMA_OPT
	P.toMatlab("P");
	Q.toMatlab("Q");
#endif

	while (epsi > epsimin) {

#ifdef DEBUG_MMA_OPT
		x.toMatlab("x");
#endif

		ux1 = upp - x;
		xl1 = x - low;
		ux2 = ux1 * ux1;
		xl2 = xl1 * xl1;
		uxinv1 = 1. / ux1;
		xlinv1 = 1. / xl1;

		std::cout << "-- epsi = " << epsi;
		
		cuda_error_check;
		// update plambda, qlambda, gvec
		update_pqlambda_gvec(nconstrain, nvar, uxinv1, xlinv1, p0, q0, P, Q, lam, plam, qlam, gmat, gvec);

#ifdef DEBUG_MMA_OPT
		gvec.toMatlab("gvec");
		plam.toMatlab("plam");
		qlam.toMatlab("qlam");
#endif

		dpsidx = plam / ux2 - qlam / xl2;

#ifdef DEBUG_MMA_OPT
		dpsidx.toMatlab("dpsidx");
#endif

		// residual
		cuda_error_check;
		rex = dpsidx - xsi + eta;
		rey = c + d * y - mu - lam;
		rez = a0 - zet - a.dot(lam);
		relam = gvec - a * z - y + s - b;
		rexsi = xsi * (x - alfa) - epsi;
		reeta = eta * (beta - x) - epsi;
		remu = mu * y - epsi;
		rezet = zet * z - epsi;
		res = lam * s - epsi;

		// DEBUG
		{
			//rex.toMatlab("rex");
			//rey.toMatlab("rey");
			//relam.toMatlab("relam");
			//rexsi.toMatlab("rexsi");
			//reeta.toMatlab("reeta");
			//remu.toMatlab("remu");
			//res.toMatlab("res");
		}

		// residual norm
		auto residu1 = rex.concat(rey).concat(rez);
		auto residu2 = relam.concat(rexsi).concat(reeta).concat(remu).concat(rezet).concat(res);
		auto residu = residu1.concat(residu2);
		double residunorm = residu.norm();
		double residumax = residu.max(-residu).max();

		//std::cout << "-- Log line " << __LINE__ << std::endl;

		int ittt = 0;

		while (residumax > 0.9 * epsi && ittt < 200) {
			ittt++;
			itera++;
			ux1 = upp - x;
			xl1 = x - low;
			ux2 = ux1 * ux1;
			xl2 = xl1 * xl1;
			ux3 = ux1 * ux2;
			xl3 = xl1 * xl2;

			uxinv1 = 1. / ux1;
			xlinv1 = 1. / xl1;
			uxinv2 = 1. / ux2;
			xlinv2 = 1. / xl2;

			update_pqlambda_gvec(nconstrain, nvar, uxinv1, xlinv1, p0, q0, P, Q, lam, plam, qlam, gmat, gvec);

			GG = uxinv2.dup(nconstrain, nvar, Pwordpitch) * P - xlinv2.dup(nconstrain, nvar, Qwordpitch) * Q;

			dpsidx = plam / ux2 - qlam / xl2;

			gv::gVector<double> gvec0(gvec);

			delx = dpsidx - epsi / (x - alfa) + epsi / (beta - x);
			dely = c + d * y - lam - epsi / y;
			delz = a0 - a.dot(lam) - epsi / z;
			dellam = gvec - a * z - y - b + epsi / lam;

#ifdef DEBUG_MMA_OPT
			// DEBUG
			{
				x.toMatlab("x");
				y.toMatlab("y");
				a.toMatlab("a");
				b.toMatlab("b");
				lam.toMatlab("lam");
				xsi.toMatlab("xsi");
				eta.toMatlab("eta");
				mu.toMatlab("mu");
				s.toMatlab("s");
				gvec.toMatlab("gvec");
				delx.toMatlab("delx");
				dely.toMatlab("dely");
				dellam.toMatlab("dellam");
			}
#endif

			diagx = 2 * (plam / ux3 + qlam / xl3) + xsi / (x - alfa) + eta / (beta - x);
			diagxinv = 1. / diagx;
			auto diagy = d + mu / y;
			auto diagyinv = 1. / diagy;
			auto diaglam = s / lam;
			auto diaglamyi = diaglam + diagyinv;

			//std::cout << "-- log " << __LINE__ << std::endl;
			GGdelxdiagx = GG * (delx / diagx).dup(nconstrain, nvar, Pwordpitch);
			GGdelxdiagx.pitchSumInPlace(4, nvar, Pwordpitch);
			//GGxx;
			GGdelxdiagx.gatherPitch(GGxx.data(), 1, nconstrain, Pwordpitch, 1);
			blam = dellam + dely / diagy - GGxx;
			bb = blam.concat(delz);
			//std::cout << "-- log " << __LINE__ << std::endl;
			
			//diagx.toMatlab("diagx");
			//GG.toMatlab("gg");
			//bb.toMatlab("bb");
			//diagy.toMatlab("diagy");
			//diagxinv.toMatlab("diagxinv");
			//diaglamyi.toMatlab("diaglamyi");
			//blam.toMatlab("blam");

			//bbhost(nconstrain + 1);
			bb.get(bbhost.data(), nconstrain + 1);
			//std::vector<double> xhost(nconstrain + 1);
			if (nconstrain < nvar || nconstrain < 1000) {
				//gVector Alam(nconstrain * nconstrain);
				//std::cout << "-- log " << __LINE__ << std::endl;
				matrix_AAt_rowmajor<double, double, 512, 1>(GG.data(), nvar, Pwordpitch, nconstrain, diagxinv.data(), Alam.data(), nconstrain);

				//Alam.toMatlab("Alam");

				double* Alam_data = Alam.data();
				auto add_diag_kernel = [=] __device__(int eid) {
					Alam_data[eid + eid * nconstrain] += diaglamyi.eval(eid);
					return;
				};	
				parallel_do(nconstrain, 512, add_diag_kernel);

				//std::cout << "-- log " << __LINE__ << std::endl;
#ifdef DEBUG_MMA_OPT
				Alam.toMatlab("Alam");
#endif

				//std::vector<double> Alam_host(Alam.size());
				Alam.get(Alam_host.data(), Alam.size());

				//std::vector<double> ahost(nconstrain);
				a.get(ahost.data(), nconstrain);
				solveLinearHost(nconstrain, Alam_host.data(), ahost.data(), zet, z, bbhost.data(), xhost.data());
				dlam.set(xhost.data());
				dz = xhost[nconstrain];
				GGdelxdiagx = GG /*/ diagx.dup(nconstrain, Pwordpitch)*/;
				GGdelxdiagx.weightedPitchSumInPlace(4, nvar, Pwordpitch, dlam.data(), true);
				dx = -delx / diagx - GGdelxdiagx.slice(0, nvar) / diagx;
#ifdef DEBUG_MMA_OPT
				GGdelxdiagx.slice(0, nvar).toMatlab("gg");
				GGdelxdiagx.slice(0, nvar).toMatlab("ggslice");
				dlam.toMatlab("dlam");
				delx.toMatlab("delx");
				diagx.toMatlab("diagx");
#endif
			} else {
				std::cerr << "large number of constrain are not supported" << std::endl;
			}

			// compute step direction
			dy = -dely / diagy + dlam / diagy;
			dxsi = -xsi + epsi / (x - alfa) - (xsi * dx) / (x - alfa);
			deta = -eta + epsi / (beta - x) + (eta * dx) / (beta - x);
			dmu = -mu + epsi / y - (mu * dy) / y;
			dzet = -zet + epsi / z - zet * dz / z;
			ds = -s + epsi / lam - (s * dlam) / lam;
#ifdef DEBUG_MMA_OPT
			dx.toMatlab("dx");
			dy.toMatlab("dy");
			dxsi.toMatlab("dxsi");
			deta.toMatlab("deta");
			dmu.toMatlab("dmu");
			ds.toMatlab("ds");
#endif
			auto xx = y.concat(z).concat(lam).concat(xsi).concat(eta).concat(mu).concat(zet).concat(s);
			auto dxx = dy.concat(dz).concat(dlam).concat(dxsi).concat(deta).concat(dmu).concat(dzet).concat(ds);

			// line search
			auto stepxx = -1.01 * dxx / xx;
			double stmxx = stepxx.max();
			auto stepalfa = -1.01 * dx / (x - alfa);
			double stmalfa = stepalfa.max();
			auto stepbeta = 1.01 * dx / (beta - x);
			double stmbeta = stepbeta.max();
			double stmalbe = (std::max)(stmalfa, stmbeta);
			double stmalbexx = (std::max)(stmalbe, stmxx);
			double stminv = (std::max)(stmalbexx, 1.);
			double steg = 1.0 / stminv;

			if (isnan(steg)) { std::cout << "\033[31m" << "-- NaN step occurred! " << "\033[0m" << std::endl; }

			// - - -
			xold = x;
			yold = y;
			zold = z;
			lamold = lam;
			xsiold = xsi;
			etaold = eta;
			muold = mu;
			zetold = zet;
			sold = s;
			// - - -
			int itto = 0;
			double resinew = 2 * residunorm;
			while (resinew > residunorm && itto < 50) {
				itto = itto + 1;
				x = xold + steg * dx;
				y = yold + steg * dy;
				z = zold + steg * dz;
				mu = muold + steg * dmu;
				lam = lamold + steg * dlam;
				xsi = xsiold + steg * dxsi;
				eta = etaold + steg * deta;
				zet = zetold + steg * dzet;
				s = sold + steg * ds;
				ux1 = upp - x;
				xl1 = x - low;
				uxinv1 = 1 / (upp - x);
				xlinv1 = 1 / (x - low);
				update_pqlambda_gvec(nconstrain, nvar, uxinv1, xlinv1, p0, q0, P, Q, lam, plam, qlam, gmat, gvec);
//#ifdef DEBUG_MMA_OPT
				//x.toMatlab("x");
				//uxinv1.toMatlab("uxinv1");
				//xlinv1.toMatlab("xlinv1");
				//plam.toMatlab("plam");
				//qlam.toMatlab("qlam");
				//gvec.toMatlab("gvec");
//#endif
				dpsidx = plam / (ux1 * ux1) - qlam / (xl1 * xl1);
				
				rex = dpsidx - xsi + eta;
				rey = c + d * y - mu - lam;
				rez = a0 - zet - a.dot(lam);
				relam = gvec - a * z - y + s - b;
				rexsi = xsi * (x - alfa) - epsi;
				reeta = eta * (beta - x) - epsi;
				remu = mu * y - epsi;
				rezet = zet * z - epsi;
				res = lam * s - epsi;
				residu1 = rex.concat(rey).concat(rez);
				residu2 = relam.concat(rexsi, reeta, remu, rezet, res);
				residu = residu1.concat(residu2);
				resinew = residu.norm();
				steg /= 2;

#ifdef DEBUG_MMA_OPT
				residu.toMatlab("residu");
				rex.toMatlab("rex");
				rey.toMatlab("rey");
				relam.toMatlab("relam");
				rexsi.toMatlab("rexsi");
				reeta.toMatlab("reeta");
				remu.toMatlab("remu");
				res.toMatlab("res");
				dpsidx.toMatlab("dpsidx");
#endif
			}

			//if (mma_iter == 1 && (itto >= 50 || itto != nlineSearchStep[lsindex])) {
			//	printf("line search failed\n"); 
			//	std::cout << "-- ittt = " << ittt << ", " << "residunorm = " << residunorm << std::endl;
			//	std::cout << "-- step " << steg << ", initial step = " << 1. / stminv << std::endl;
			//	xold.toMatlab("xold");
			//	yold.toMatlab("yold");
			//	dx.toMatlab("dx");
			//	dy.toMatlab("dy");
			//	GG.toMatlab("GG");
			//	residu.toMatlab("residu");
			//	Alam.toMatlab("Alam");
			//	blam.toMatlab("blam");
			//	delx.toMatlab("delx");
			//	dellam.toMatlab("dellam");
			//	dely.toMatlab("dely");
			//	gvec.toMatlab("gvec");
			//	gvec0.toMatlab("gvec0");
			//	diagx.toMatlab("diagx");
			//	delx.toMatlab("delx");
			//	GGxx.toMatlab("GGxx");
			//	diagy.toMatlab("diagy");
			//	bb.toMatlab("bb");
			//	diaglamyi.toMatlab("diaglamyi");
			//	lamold.toMatlab("lamold");
			//}
			//lsindex++;

			residunorm = resinew;
			residumax = residu.abs().max();
			steg = 2 * steg;
		}

		std::cout << " ittt = " << ittt << std::endl;

		epsi = 0.1 * epsi;
	}

	//lsindex = 0;

	return std::make_tuple(
		std::move(x), std::move(y), std::move(z),
		std::move(lam), std::move(xsi), std::move(eta),
		std::move(mu), std::move(zet), std::move(s));
}


// return updated x, low_g and upp_g are updated in place
gv::gVector<double> mmasub_ker(int nconstrain, int nvar,
	int itn, gv::gVector<double>& xvar_g,
	gv::gVector<double>& xmin_g, gv::gVector<double>& xmax_g,
	gv::gVector<double>& xold1_g, gv::gVector<double>& xold2_g,
	double f0val, gv::gVector<double>& df0dx_g, gv::gVector<double>& gval_g, gv::gVector<double>& dgdx_g,
	gv::gVector<double>& low_g, gv::gVector<double>& upp_g,
	double a0, gv::gVector<double>& a_g, gv::gVector<double>& c_g, gv::gVector<double>& d_g,
	double move, int dgdxvarpitch
) {
	using gVector = gv::gVector<double>;

	size_t nvarpitch = dgdxvarpitch;

	double epsimin = 1e-7;
	double raa0 = 1e-5;
	double albefa = 0.1;
	double asyinit = 0.5;
	double asyincr = 1.2;
	double asydecr = 0.7;

	gVector factor(nvar);
	factor.set(1);

	gVector lowmin(nvar), lowmax(nvar);
	gVector uppmin(nvar), uppmax(nvar);

	if (itn <= 2) {
		low_g = xvar_g - asyinit * (xmax_g - xmin_g);
		upp_g = xvar_g + asyinit * (xmax_g - xmin_g);
	}
	else {
		auto zzz = (xvar_g - xold1_g) * (xold1_g - xold2_g);
		factor.set(zzz > 0, asyincr);
		factor.set(zzz < 0, asydecr);
		low_g = xvar_g - factor * (xold1_g - low_g);
		upp_g = xvar_g + factor * (upp_g - xold1_g);
#ifdef DEBUG_MMA_OPT
		low_g.toMatlab("low");
		upp_g.toMatlab("upp");
#endif
		lowmin = xvar_g - 10 * (xmax_g - xmin_g);
		lowmax = xvar_g - 0.01 * (xmax_g - xmin_g);
		uppmin = xvar_g + 0.01 * (xmax_g - xmin_g);
		uppmax = xvar_g + 10 * (xmax_g - xmin_g);
		low_g.maximize(lowmin);
		low_g.minimize(lowmax);
		upp_g.minimize(uppmax);
		upp_g.maximize(uppmin);
#ifdef DEBUG_MMA_OPT
		factor.toMatlab("factor");
#endif
	}

#ifdef DEBUG_MMA_OPT
	low_g.toMatlab("low");
	upp_g.toMatlab("upp");
	xvar_g.toMatlab("xvar");
#endif

	auto zzz1 = low_g + albefa * (xvar_g - low_g);
	auto zzz2 = xvar_g - move * (xmax_g - xmin_g);
	auto zzz = zzz1.max(zzz2);
	gVector alfa = zzz.max(xmin_g);
	//alfa.toMatlab("alfa");

	auto zzz11 = upp_g - albefa * (upp_g - xvar_g);
	auto zzz22 = xvar_g + move * (xmax_g - xmin_g);
	auto zzzz = zzz11.min(zzz22);
	gVector beta = zzzz.min(xmax_g);
	//beta.toMatlab("beta");

	auto xmami = (xmax_g - xmin_g).max(1e-5);
	auto xmamiinv = 1. / xmami;

	gVector ux1 = upp_g - xvar_g;
	gVector ux2 = ux1 * ux1;
	gVector xl1 = xvar_g - low_g;
	gVector xl2 = xl1 * xl1;
	gVector uxinv = 1. / ux1;
	gVector xlinv = 1. / xl1;

	//auto p0 = df0dx_g.max(0);
	//auto q0 = (-df0dx_g).max(0);
	gVector pq0 = 1e-3 * (df0dx_g.max(0) + (-df0dx_g).max(0)) + raa0 * xmamiinv;
	gVector p0 = (df0dx_g.max(0) + pq0) * ux2;
	gVector q0 = ((-df0dx_g).max(0) + pq0) * xl2;

#ifdef DEBUG_MMA_OPT
	pq0.toMatlab("pq0");
#endif

	gVector P = dgdx_g.max(0);
	gVector Q = (-dgdx_g).max(0);

#ifdef DEBUG_MMA_OPT
	P.toMatlab("P");
	Q.toMatlab("Q");
#endif

	// add PQ 
	{
		double* Pdata = P.data();
		double* Qdata = Q.data();
		double* ux2data = ux2.data();
		double* xl2data = xl2.data();
		int nwordpitch = nvarpitch / sizeof(double);
		auto kernel = [=] __device__(int eid) {
			for (int i = 0; i < nconstrain; i++) {
				double p = Pdata[eid + i * nwordpitch];
				double q = Qdata[eid + i * nwordpitch];
				double pq = 1e-3 * (p + q) + raa0 * xmamiinv.eval(eid);
				Pdata[eid + i * nwordpitch] = (p + pq) * ux2data[eid];
				Qdata[eid + i * nwordpitch] = (q + pq) * xl2data[eid];
			}
		};
		parallel_do(nvar, 512, kernel);
	}

#ifdef DEBUG_MMA_OPT
	uxinv.toMatlab("uxinv");
	xlinv.toMatlab("xlinv");
	P.toMatlab("P");
	Q.toMatlab("Q");
#endif

	gVector bmat(P);
	gVector b(nconstrain);
	int wordpitch;
	{
		double* bmat_data = bmat.data();
		double* Pbdata = P.data();
		double* Qbdata = Q.data();
		double* uxinv_data = uxinv.data();
		double* xlinv_data = xlinv.data();
		wordpitch = nvarpitch / sizeof(double);
		auto kernel = [=] __device__(int eid) {
			for (int i = 0; i < nconstrain; i++) {
				bmat_data[eid + i * wordpitch] = Pbdata[eid + i * wordpitch] * uxinv_data[eid] + Qbdata[eid + i * wordpitch] * xlinv_data[eid];
			}
		};
		parallel_do(nvar, 512, kernel);
		bmat.pitchSumInPlace(4, nvar, nvarpitch / sizeof(double));
		double* bdata = b.data();
		double* gval_data = gval_g.data();
		auto kernel_gather = [=] __device__(int eid) {
			bdata[eid] = bmat_data[eid * wordpitch] - gval_data[eid];
		};
		parallel_do(nconstrain, 512, kernel_gather);
	}

#ifdef DEBUG_MMA_OPT
	b.toMatlab("b");
	p0.toMatlab("p0");
	q0.toMatlab("q0");
	a_g.toMatlab("a");
	c_g.toMatlab("c");
	d_g.toMatlab("d");
#endif
	
	//return std::make_tuple(
	//	std::move(x), std::move(y), std::move(z),
	//	std::move(lam), std::move(xsi), std::move(eta),
	//	std::move(mu), std::move(zet), std::move(s));

	auto[x, y, z, lam, xsi, eta, mu, zet, s] = subsolv_g(
		nconstrain, nvar, epsimin,
		low_g, upp_g,
		alfa, beta,
		p0, q0,
		P, Q,
		wordpitch, wordpitch,
		a0, a_g, b, c_g, d_g);

#ifdef DEBUG_MMA_OPT
	x.toMatlab("xnew");
	y.toMatlab("ynew");
	lam.toMatlab("lamnew");
	xsi.toMatlab("xsinew");
	eta.toMatlab("etanew");
	mu.toMatlab("munew");
	s.toMatlab("snew");
#endif
	
	return x;

}

void mmasub_g(int nconstrain, int nvar,
	int itn, double* xvar, double* xmin, double* xmax, double* xold1, double* xold2,
	double f0val, double* df0dx, double* gval, cudaPitchedPtr dgdx,
	double* low, double* upp,
	double a0, double* a, double* c, double* d,
	double move
) {
	gv::gVector<double>::Init();

	typedef gv::gVector<double> gVector;
	typedef gv::gVectorMap<double> gVectorMap;

	gVectorMap xvar_g(xvar, nvar);

	gVectorMap xmin_g(xmin, nvar), xmax_g(xmax, nvar);

	gVectorMap xold1_g(xold1, nvar), xold2_g(xold2, nvar);

	gVectorMap df0dx_g(df0dx, nvar);

	gVectorMap gval_g(gval, nconstrain);

	gVectorMap dgdx_g((double*)dgdx.ptr, dgdx.pitch / sizeof(double) * nconstrain);
	
	gVectorMap low_g(low, nvar), upp_g(upp, nvar);

	gVectorMap a_g(a, nconstrain), c_g(c, nconstrain), d_g(d, nconstrain);

	auto x = mmasub_ker(nconstrain, nvar, itn,
		xvar_g, xmin_g, xmax_g, xold1_g, xold2_g,
		f0val, df0dx_g, gval_g, dgdx_g,
		low_g, upp_g, a0, a_g, c_g, d_g,
		move, dgdx.pitch);

	cudaMemcpy(xold2, xold1, sizeof(double) * nvar, cudaMemcpyDeviceToDevice);
	cudaMemcpy(xold1, xvar, sizeof(double) * nvar, cudaMemcpyDeviceToDevice);
	cudaMemcpy(xvar, x.data(), x.size() * sizeof(double), cudaMemcpyDeviceToDevice);
	
	// low upp are already mapped
	//low_g.get(low, nvar);
	//upp_g.get(upp, nvar);
}

void mmasub_h(int nconstrain, int nvar,
	int itn, double* xvar, double* xmin, double* xmax, double* xold1, double* xold2,
	double f0val, double* df0dx, double* gval, double* dgdx,
	double* low, double* upp,
	double a0, double* a, double* c, double* d,
	double move
) 
{
	gv::gVector<double>::Init();

	typedef gv::gVector<double> gVector;
	// transfer data from host to device
	gVector xvar_g(nvar);
	xvar_g.set(xvar);

	gVector xmin_g(nvar), xmax_g(nvar);
	xmin_g.set(xmin);
	xmax_g.set(xmax);

	gVector xold1_g(nvar), xold2_g(nvar);
	xold1_g.set(xold1);
	xold2_g.set(xold2);

	gVector df0dx_g(nvar);
	df0dx_g.set(df0dx);

	gVector gval_g(nconstrain);
	gval_g.set(gval);

	double* gx;
	size_t nvarpitch;
	cudaMallocPitch(&gx, &nvarpitch, sizeof(double) * nvar, nconstrain);
	cudaMemcpy2D(gx, nvarpitch, dgdx, nvar * sizeof(double), sizeof(double) * nvar, nconstrain, cudaMemcpyHostToDevice);
	cuda_error_check;
	gVector dgdx_g(gv::gVectorMap<double>(gx, nvarpitch / sizeof(double) * nconstrain));
	

	gVector low_g(nvar), upp_g(nvar);
	low_g.set(low);
	upp_g.set(upp);

	gVector a_g(nconstrain), c_g(nconstrain), d_g(nconstrain);
	a_g.set(a);
	c_g.set(c);
	d_g.set(d);

	auto x = mmasub_ker(nconstrain, nvar, itn,
		xvar_g, xmin_g, xmax_g, xold1_g, xold2_g,
		f0val, df0dx_g, gval_g, dgdx_g,
		low_g, upp_g, a0, a_g, c_g, d_g,
		move, nvarpitch);

	memcpy(xold2, xold1, sizeof(double) * nvar);
	memcpy(xold1, xvar, sizeof(double) * nvar);
	x.get(xvar, nvar);
	low_g.get(low, nvar);
	upp_g.get(upp, nvar);
}




