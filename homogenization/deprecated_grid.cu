#include "grid.h"
#include "culib/lib.cuh"
#include "homoCommon.cuh"

// slower than baseline
// map 16 warp to 32 vertices, 1 neight element dispatched to 2 threads 
template<int BlockSize = 32 * 16>
__global__ void gs_relaxation_otf_kernel_test_512(
	int gs_set, float* rholist,
	devArray_t<int, 3> gridCellReso,
	VertexFlags* vflags, CellFlags* eflags,
	// SOR relaxing factor
	double w = 1.f,
	float diag_strength = 0
) {

	__shared__ int gsCellReso[3][8];
	__shared__ int gsVertexReso[3][8];
	__shared__ int gsCellEnd[8];
	__shared__ int gsVertexEnd[8];
	__shared__ double KE[24][24];
	__shared__ double sumKeU[3][8][32];
	__shared__ float sumKs[9][8][32];

	initSharedMem(&sumKeU[0][0][0], sizeof(sumKeU) / sizeof(double));
	initSharedMem(&sumKs[0][0][0], sizeof(sumKs) / sizeof(float));

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;

	// local vertex id in gs set
	int vid = blockIdx.x * 32 + laneId;

	bool fiction = false;

	// load cell and vertex reso
	constant2DToShared(gGsCellReso, gsCellReso);
	constant2DToShared(gGsVertexReso, gsVertexReso);
	constantToShared(gGsCellEnd, gsCellEnd);
	constantToShared(gGsVertexEnd, gsVertexEnd);
	//constantToShared(gU, U);
	loadTemplateMatrix(KE);

	// to global vertex id
	vid = gs_set == 0 ? vid : gsVertexEnd[gs_set - 1] + vid;

	if (vid >= gsVertexEnd[gs_set]) fiction = true;

	GridVertexIndex indexer(gridCellReso[0], gridCellReso[1], gridCellReso[2]);
	VertexFlags vflag;
	if (!fiction) {
		vflag = vflags[vid];
		fiction |= vflag.is_fiction();
		indexer.locate(vid, vflag.get_gscolor(), gsVertexEnd);
	}
	
	double KeU[3] = { 0. };
	float Ks[3][3] = { 0.f };

	//fiction |= vflag.is_max_boundary();

	int NeiCell = warpId % 8;

	if (!fiction && !vflag.is_period_padding()) {
		int elementId = indexer.neighElement(NeiCell, gsCellEnd, gsCellReso).getId();
		int vselfrow = (7 - NeiCell) * 3;
		float rho_penal = 0;
		CellFlags eflag;
		float penal = exp_penal[0];
		if (elementId != -1) {
			eflag = eflags[elementId];
			if (!eflag.is_fiction()) rho_penal = powf(rholist[elementId], penal);
		}

		if (elementId != -1 && !eflag.is_fiction() /*&& !eflag.is_period_padding()*/) {
#pragma unroll
			for (int vj = 0; vj < 4; vj++) {
				int i = vj + warpId / 8 * 4;
				if (i == 7 - NeiCell) continue;
				int vneigh =
					(NeiCell % 2 + i % 2) +
					(NeiCell / 2 % 2 + i / 2 % 2) * 3 +
					(NeiCell / 4 + i / 4) * 9;
				int vneighId = indexer.neighVertex(vneigh, gsVertexEnd, gsVertexReso).getId();
				VertexFlags nvflag;
				if (vneighId != -1) {
					nvflag = vflags[vneighId];
					if (!nvflag.is_fiction()) {
						double u[3] = { gU[0][vneighId], gU[1][vneighId], gU[2][vneighId] };
						if (nvflag.is_dirichlet_boundary()) {
							u[0] = u[1] = u[2] = 0;
						}
						int colsel = i * 3;
						KeU[0] += KE[vselfrow][colsel] * u[0] + KE[vselfrow][colsel + 1] * u[1] + KE[vselfrow][colsel + 2] * u[2];
						vselfrow++;
						KeU[1] += KE[vselfrow][colsel] * u[0] + KE[vselfrow][colsel + 1] * u[1] + KE[vselfrow][colsel + 2] * u[2];
						vselfrow++;
						KeU[2] += KE[vselfrow][colsel] * u[0] + KE[vselfrow][colsel + 1] * u[1] + KE[vselfrow][colsel + 2] * u[2];
					}
				}
			}
			KeU[0] *= rho_penal; KeU[1] *= rho_penal; KeU[2] *= rho_penal;

			if (warpId < 8) {
				for (int k3row = 0; k3row < 3; k3row++) {
					for (int k3col = 0; k3col < 3; k3col++) {
						Ks[k3row][k3col] += float(KE[vselfrow + k3row][vselfrow + k3col]) * rho_penal;
					}
				}
			}
		}
	}

	if (warpId >= 8) {
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				sumKs[i * 3 + j][warpId - 8][laneId] = Ks[i][j];
			}
			sumKeU[i][warpId - 8][laneId] = KeU[i];
		}
	}
	__syncthreads();

	if (warpId < 8) {
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				sumKs[i * 3 + j][warpId][laneId] += Ks[i][j];
			}
			sumKeU[i][warpId][laneId] += KeU[i];
		}
	}
	__syncthreads();

	if (warpId < 4) {
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				sumKs[i * 3 + j][warpId][laneId] += sumKs[i * 3 + j][warpId + 4][laneId];
			}
			sumKeU[i][warpId][laneId] += sumKeU[i][warpId + 4][laneId];
		}
	}
	__syncthreads();

	if (warpId < 2) {
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				 sumKs[i * 3 + j][warpId][laneId] += sumKs[i * 3 + j][warpId + 2][laneId];
			}
			sumKeU[i][warpId][laneId] += sumKeU[i][warpId + 2][laneId];
		}
	}
	__syncthreads();

	if (warpId < 1 && !vflag.is_period_padding() && !fiction) {
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				Ks[i][j] = sumKs[i * 3 + j][warpId][laneId] + sumKs[i * 3 + j][warpId + 1][laneId];
			}
			KeU[i] = sumKeU[i][warpId][laneId] + sumKeU[i][warpId + 1][laneId];
		}

		double u[3] = { gU[0][vid],gU[1][vid],gU[2][vid] };

		// relax
#if !USING_SOR 
		u[0] = (gF[0][vid] - KeU[0] - Ks[0][1] * u[1] - Ks[0][2] * u[2]) / Ks[0][0];
		u[1] = (gF[1][vid] - KeU[1] - Ks[1][0] * u[0] - Ks[1][2] * u[2]) / Ks[1][1];
		u[2] = (gF[2][vid] - KeU[2] - Ks[2][0] * u[0] - Ks[2][1] * u[1]) / Ks[2][2];
#else
		u[0] = w * (gF[0][vid] - KeU[0] - Ks[0][1] * u[1] - Ks[0][2] * u[2]) / Ks[0][0] + (1 - w) * u[0];
		u[1] = w * (gF[1][vid] - KeU[1] - Ks[1][0] * u[0] - Ks[1][2] * u[2]) / Ks[1][1] + (1 - w) * u[1];
		u[2] = w * (gF[2][vid] - KeU[2] - Ks[2][0] * u[0] - Ks[2][1] * u[1]) / Ks[2][2] + (1 - w) * u[2];
#endif

		// if dirichlet boundary;
		if (vflag.is_dirichlet_boundary()) { u[0] = u[1] = u[2] = 0; }
		// update
		gU[0][vid] = u[0];
		gU[1][vid] = u[1];
		gU[2][vid] = u[2];
	}
}

