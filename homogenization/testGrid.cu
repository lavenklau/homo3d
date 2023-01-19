#include "homogenization.h"
#include "homoCommon.cuh"

using namespace homo;

__global__ void enforce_boundary_force_kernel(
	int neboundary, int istrain, devArray_t<double*, 3> fout, CellFlags* eflags, float* rholist) {
	__shared__ float KE[24][24];
	loadTemplateMatrix(KE);

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= neboundary) return;

	int ereso[3] = { gGridCellReso[0],gGridCellReso[1],gGridCellReso[2] };
	int neface[3] = { ereso[1] * ereso[2],ereso[0] * ereso[2],ereso[0] * ereso[1] };
	// normal x : right face
	char3 bound{ 0,0,0 };
	int epos[3] = { -2,-2,-2 };
	double u[8][3];
	// eps_xx or eps_xz
	if (istrain == 0 || istrain == 4) {
		bound.x = 0;
		epos[0] = ereso[0] - 1;
		epos[1] = tid % ereso[1];
		epos[2] = tid / ereso[1];
		if (istrain == 0) {
			for (int i = 0; i < 8; i++) {
				u[i][0] = i % 2 * ereso[0]; u[i][1] = 0; u[i][2] = 0;
			}
		} else {
			for (int i = 0; i < 8; i++) {
				u[i][0] = 0; u[i][1] = 0; u[i][2] = i % 2 * ereso[0];
			}
		}
	}
	// eps_yy  OR  eps_xy
	else if (istrain == 1 || istrain == 5) {
		bound.y = 0;
		epos[0] = tid % ereso[0];
		epos[1] = ereso[1] - 1;
		epos[2] = tid / ereso[0];
		if (istrain == 1) {
			for (int i = 0; i < 8; i++) {
				u[i][0] = 0; u[i][1] = i / 2 % 2 * ereso[1]; u[i][2] = 0;
			}
		} else {
			for (int i = 0; i < 8; i++) {
				u[i][0] = i / 2 % 2 * ereso[1]; u[i][1] = i / 2 % 2 * ereso[1]; u[i][2] = 0;
			}
		}
	}
	// eps_zz OR eps_yz
	else if (istrain == 2 || istrain == 3) {
		bound.z = 0;
		epos[0] = tid % ereso[0];
		epos[1] = tid / ereso[0];
		epos[2] = ereso[2] - 1;
		if (istrain == 2) {
			for (int i = 0; i < 8; i++) {
				u[i][0] = 0; u[i][1] = 0; u[i][2] = i / 4 * ereso[2];
			}
		} else {
			for (int i = 0; i < 8; i++) {
				u[i][0] = 0; u[i][1] = i / 4 * ereso[2]; u[i][2] = 0;
			}
		}
	}

	int eid = lexi2gs(epos, gGsCellReso, gGsCellEnd);

	if (eid == -1) print_exception;

	CellFlags eflag = eflags[eid];

	if (!eflag.is_max_boundary()) { print_exception; }

	float rho = rholist[eid];

	double f[8][3];
	for (int vi = 0; vi < 8; vi++) {
		int virow = vi * 3;
		double frow[3] = {};
		for (int vj = 0; vj < 8; vj++) {
			int vjcol = vj * 3;
			for (int ir = 0; ir < 3; ir++) {
				for (int jc = 0; jc < 3; jc++) {
					frow[ir] += KE[virow + ir][vjcol + jc] * u[vj][jc];
				}
			}
		}
		f[vi][0] = frow[0]; f[vi][1] = frow[1]; f[vi][2] = frow[2];
	}
	
	if (bound.z) {
		
	}
}
