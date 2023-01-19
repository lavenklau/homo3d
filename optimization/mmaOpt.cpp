#include <iostream>
#include "cuda_runtime.h"
#include "mmaOpt.h"
#include <tuple>
#include "matlab/matlab_utils.h"

//#define EIGEN_USE_MKL
#include "Eigen/Eigen"


//extern std::vector<int> nlineSearchStep;

extern void mmasub_h(int ncontrain, int nvar,
	int itn, double* xvar, double* xmin, double* xmax, double* xold1, double* xold2,
	double f0val, double* df0dx, double* gval, double* dgdx,
	double* low, double* upp,
	double a0, double* a, double* c, double* d,double move);
extern void mmasub_g(int ncontrain, int nvar,
	int itn, double* xvar, double* xmin, double* xmax, double* xold1, double* xold2,
	double f0val, double* df0dx, double* gval, cudaPitchedPtr dgdx,
	double* low, double* upp,
	double a0, double* a, double* c, double* d,double move);

extern void solveLinearHost(int nconstrain, const double* Alamhost, const double* ahost, double zet, double z, const double* bb, double* xhost);
extern void test_gVector(void);

API_MMAOPT void mmasubHost(
	int ncontrain, int nvar,
	int itn, double* xvar, double* xmin, double* xmax, double* xold1, double* xold2,
	double f0val, double* df0dx, double* gval, double* dgdx,
	double* low, double* upp,
	double a0, double* a, double* c, double* d, double move
)
{
	mmasub_h(ncontrain, nvar, itn, xvar, xmin, xmax, xold1, xold2, f0val, df0dx, gval, dgdx, low, upp, a0, a, c, d, move);
}

API_MMAOPT void mmasubDevice(
	int ncontrain, int nvar,
	int itn, double* xvar, double* xmin, double* xmax, double* xold1, double* xold2,
	double f0val, double* df0dx, double* gval, cudaPitchedPtr dgdx, 
	double* low, double* upp,
	double a0, double* a, double* c, double* d, double move
) {
	mmasub_g(ncontrain, nvar, itn, xvar, xmin, xmax, xold1, xold2, f0val, df0dx, gval, dgdx, low, upp, a0, a, c, d, move);
}

void solveLinearHost(int nconstrain, const double* Alamhost, const double* ahost, double zet, double z, const double* bb, double* xhost) {
	Eigen::Matrix<double, -1, -1> A(nconstrain + 1, nconstrain + 1);
	Eigen::Matrix<double, -1, 1> b = Eigen::Matrix<double, -1, 1>::Map(bb, nconstrain + 1);
	for (int i = 0; i < nconstrain; i++) {
		for (int j = 0; j < nconstrain; j++) {
			A(j, i) = Alamhost[i + j * nconstrain];
		}
		A(nconstrain, i) = ahost[i];
	}
	for (int j = 0; j < nconstrain; j++) {
		A(j, nconstrain) = ahost[j];
	}
	A(nconstrain, nconstrain) = -zet / z;


	//Eigen::LDLT<Eigen::Matrix<double, -1, -1>> solver(A);

	Eigen::ColPivHouseholderQR<Eigen::Matrix<double, -1, -1>> solver(A);

	Eigen::Matrix<double, -1, 1> x = solver.solve(b);

	//eigen2ConnectedMatlab("A", A);
	//eigen2ConnectedMatlab("b", b);

	for (int i = 0; i < nconstrain + 1; i++) xhost[i] = x[i];
}

