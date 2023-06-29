#include "grid.h"
#include "homoCommon.cuh"

using namespace homo;

__device__ static bool inStrictBound(int pi[3], int cover[3]) {
	return pi[0] > -cover[0] && pi[0] < cover[0] &&
		pi[1] > -cover[1] && pi[1] < cover[1] &&
		pi[2] > -cover[2] && pi[2] < cover[2];
}

//template<int BlockSize = 256>
__global__ void gs_relaxation_otf_kernel_opt(
	int gs_set, float* rholist,
	devArray_t<int, 3> gridCellReso,
	VertexFlags* vflags, CellFlags* eflags,
	float w = 1.f
) {
	__shared__ float LM[5];
	__shared__ float RHO[8][32];
	__shared__ float sumKeU[3][4][32];
	__shared__ float sumKs[3][3];

	// load lam and mu 
	if (threadIdx.x < 5) {
		LM[threadIdx.x] = gLM[threadIdx.x];
	}

	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;

	int vid = blockIdx.x * 32 + laneId;
	
	bool fiction = false;

	// to global vertex id
	vid = gs_set == 0 ? vid : gGsVertexEnd[gs_set - 1] + vid;

	fiction = vid >= gGsVertexEnd[gs_set];

	GridVertexIndex indexer(gridCellReso[0], gridCellReso[1], gridCellReso[2]);
	VertexFlags vflag;
	if (!fiction) {
		vflag = vflags[vid];
		fiction = fiction || vflag.is_fiction() || vflag.is_period_padding();
		indexer.locate(vid, vflag.get_gscolor(), gGsVertexEnd);
	}

	// load density field
	for (int i = 0; i < 8; i++) { RHO[i][laneId] = 0; }
	if (!fiction) {
		CellFlags eflag;
#pragma unroll
		for (int i = 0; i < 8; i++) {
			int elementId = indexer.neighElement(i, gGsCellEnd, gGsCellReso).getId();
			if (elementId != -1) {
				eflag = eflags[elementId];
			/*	if (!eflag.is_fiction())*/ RHO[i][laneId] = powf(rholist[elementId], exp_penal[0]);
			} 	
		}
	} 
	__syncthreads();

	int ev[27];
	if (!fiction) {
		for (int i = 0; i < 27; i++) {
			VertexFlags nvflag;
			int vneighId = indexer.neighVertex(i, gGsVertexEnd, gGsVertexReso).getId();
			ev[i] = vneighId;
			if (vneighId != -1) {
				nvflag = vflags[vneighId];
			}
		}
	}

	float KeU[3] = { 0. };
	float u[3];
	if (!fiction) {
		if (warpId == 0) {
			u[0] = gU[0][ev[0]]; u[1] = gU[1][ev[0]]; u[2] = gU[2][ev[0]];
			KeU[0] += u[0] * (LM[3] * (-2.f * RHO[0][laneId]));
			KeU[1] += u[0] * (LM[2] * (-3.f * RHO[0][laneId]));
			KeU[2] += u[0] * (LM[2] * (-3.f * RHO[0][laneId]));
			KeU[2] += u[1] * (LM[2] * (-3.f * RHO[0][laneId]));
			KeU[1] += u[1] * (LM[3] * (-2.f * RHO[0][laneId]));
			KeU[0] += u[1] * (LM[2] * (-3.f * RHO[0][laneId]));
			KeU[2] += u[2] * (LM[3] * (-2.f * RHO[0][laneId]));
			KeU[0] += u[2] * (LM[2] * (-3.f * RHO[0][laneId]));
			KeU[1] += u[2] * (LM[2] * (-3.f * RHO[0][laneId]));
			u[0] = gU[0][ev[1]]; u[1] = gU[1][ev[1]]; u[2] = gU[2][ev[1]];
			KeU[0] += u[0] * (LM[0] * (2.f * RHO[0][laneId] + 2.f * RHO[1][laneId]));
			KeU[1] += u[0] * (LM[1] * (3.f * RHO[0][laneId] + -3.f * RHO[1][laneId]));
			KeU[2] += u[0] * (LM[1] * (3.f * RHO[0][laneId] + -3.f * RHO[1][laneId]));
			KeU[2] += u[1] * (LM[2] * (-6.f * RHO[0][laneId] + -6.f * RHO[1][laneId]));
			KeU[0] += u[1] * (LM[1] * (-3.f * RHO[0][laneId] + 3.f * RHO[1][laneId]));
			KeU[1] += u[1] * (LM[4] * (-2.f * RHO[0][laneId] + -2.f * RHO[1][laneId]));
			KeU[1] += u[2] * (LM[2] * (-6.f * RHO[0][laneId] + -6.f * RHO[1][laneId]));
			KeU[2] += u[2] * (LM[4] * (-2.f * RHO[0][laneId] + -2.f * RHO[1][laneId]));
			KeU[0] += u[2] * (LM[1] * (-3.f * RHO[0][laneId] + 3.f * RHO[1][laneId]));
			u[0] = gU[0][ev[2]]; u[1] = gU[1][ev[2]]; u[2] = gU[2][ev[2]];
			KeU[0] += u[0] * (LM[3] * (-2.f * RHO[1][laneId]));
			KeU[1] += u[0] * (LM[2] * (3.f * RHO[1][laneId]));
			KeU[2] += u[0] * (LM[2] * (3.f * RHO[1][laneId]));
			KeU[1] += u[1] * (LM[3] * (-2.f * RHO[1][laneId]));
			KeU[2] += u[1] * (LM[2] * (-3.f * RHO[1][laneId]));
			KeU[0] += u[1] * (LM[2] * (3.f * RHO[1][laneId]));
			KeU[1] += u[2] * (LM[2] * (-3.f * RHO[1][laneId]));
			KeU[2] += u[2] * (LM[3] * (-2.f * RHO[1][laneId]));
			KeU[0] += u[2] * (LM[2] * (3.f * RHO[1][laneId]));
		}
		else if (warpId == 1) {
			u[0] = gU[0][ev[3]]; u[1] = gU[1][ev[3]]; u[2] = gU[2][ev[3]];
			KeU[0] += u[0] * (LM[4] * (-2.f * RHO[0][laneId] + -2.f * RHO[2][laneId]));
			KeU[1] += u[0] * (LM[1] * (-3.f * RHO[0][laneId] + 3.f * RHO[2][laneId]));
			KeU[2] += u[0] * (LM[2] * (-6.f * RHO[0][laneId] + -6.f * RHO[2][laneId]));
			KeU[1] += u[1] * (LM[0] * (2.f * RHO[0][laneId] + 2.f * RHO[2][laneId]));
			KeU[0] += u[1] * (LM[1] * (3.f * RHO[0][laneId] + -3.f * RHO[2][laneId]));
			KeU[2] += u[1] * (LM[1] * (3.f * RHO[0][laneId] + -3.f * RHO[2][laneId]));
			KeU[1] += u[2] * (LM[1] * (-3.f * RHO[0][laneId] + 3.f * RHO[2][laneId]));
			KeU[2] += u[2] * (LM[4] * (-2.f * RHO[0][laneId] + -2.f * RHO[2][laneId]));
			KeU[0] += u[2] * (LM[2] * (-6.f * RHO[0][laneId] + -6.f * RHO[2][laneId]));
			u[0] = gU[0][ev[4]]; u[1] = gU[1][ev[4]]; u[2] = gU[2][ev[4]];
			KeU[2] += u[0] * (LM[1] * (6.f * RHO[0][laneId] + -6.f * RHO[1][laneId] + 6.f * RHO[2][laneId] + -6.f * RHO[3][laneId]));
			KeU[0] += u[0] * (LM[2] * (4.f * RHO[0][laneId] + 4.f * RHO[1][laneId] + 4.f * RHO[2][laneId] + 4.f * RHO[3][laneId]));
			KeU[1] += u[0] * (LM[2] * (3.f * RHO[0][laneId] + -3.f * RHO[1][laneId] + -3.f * RHO[2][laneId] + 3.f * RHO[3][laneId]));
			KeU[2] += u[1] * (LM[1] * (6.f * RHO[0][laneId] + 6.f * RHO[1][laneId] + -6.f * RHO[2][laneId] + -6.f * RHO[3][laneId]));
			KeU[0] += u[1] * (LM[2] * (3.f * RHO[0][laneId] + -3.f * RHO[1][laneId] + -3.f * RHO[2][laneId] + 3.f * RHO[3][laneId]));
			KeU[1] += u[1] * (LM[2] * (4.f * RHO[0][laneId] + 4.f * RHO[1][laneId] + 4.f * RHO[2][laneId] + 4.f * RHO[3][laneId]));
			KeU[0] += u[2] * (LM[1] * (-6.f * RHO[0][laneId] + 6.f * RHO[1][laneId] + -6.f * RHO[2][laneId] + 6.f * RHO[3][laneId]));
			KeU[2] += u[2] * (LM[2] * (-8.f * RHO[0][laneId] + -8.f * RHO[1][laneId] + -8.f * RHO[2][laneId] + -8.f * RHO[3][laneId]));
			KeU[1] += u[2] * (LM[1] * (-6.f * RHO[0][laneId] + -6.f * RHO[1][laneId] + 6.f * RHO[2][laneId] + 6.f * RHO[3][laneId]));
			u[0] = gU[0][ev[5]]; u[1] = gU[1][ev[5]]; u[2] = gU[2][ev[5]];
			KeU[2] += u[0] * (LM[2] * (6.f * RHO[1][laneId] + 6.f * RHO[3][laneId]));
			KeU[1] += u[0] * (LM[1] * (3.f * RHO[1][laneId] + -3.f * RHO[3][laneId]));
			KeU[0] += u[0] * (LM[4] * (-2.f * RHO[1][laneId] + -2.f * RHO[3][laneId]));
			KeU[1] += u[1] * (LM[0] * (2.f * RHO[1][laneId] + 2.f * RHO[3][laneId]));
			KeU[2] += u[1] * (LM[1] * (3.f * RHO[1][laneId] + -3.f * RHO[3][laneId]));
			KeU[0] += u[1] * (LM[1] * (-3.f * RHO[1][laneId] + 3.f * RHO[3][laneId]));
			KeU[2] += u[2] * (LM[4] * (-2.f * RHO[1][laneId] + -2.f * RHO[3][laneId]));
			KeU[1] += u[2] * (LM[1] * (-3.f * RHO[1][laneId] + 3.f * RHO[3][laneId]));
			KeU[0] += u[2] * (LM[2] * (6.f * RHO[1][laneId] + 6.f * RHO[3][laneId]));
		}
		else if (warpId == 2) {
			u[0] = gU[0][ev[6]]; u[1] = gU[1][ev[6]]; u[2] = gU[2][ev[6]];
			KeU[0] += u[0] * (LM[3] * (-2.f * RHO[2][laneId]));
			KeU[1] += u[0] * (LM[2] * (3.f * RHO[2][laneId]));
			KeU[2] += u[0] * (LM[2] * (-3.f * RHO[2][laneId]));
			KeU[0] += u[1] * (LM[2] * (3.f * RHO[2][laneId]));
			KeU[2] += u[1] * (LM[2] * (3.f * RHO[2][laneId]));
			KeU[1] += u[1] * (LM[3] * (-2.f * RHO[2][laneId]));
			KeU[2] += u[2] * (LM[3] * (-2.f * RHO[2][laneId]));
			KeU[0] += u[2] * (LM[2] * (-3.f * RHO[2][laneId]));
			KeU[1] += u[2] * (LM[2] * (3.f * RHO[2][laneId]));
			u[0] = gU[0][ev[7]]; u[1] = gU[1][ev[7]]; u[2] = gU[2][ev[7]];
			KeU[2] += u[0] * (LM[1] * (3.f * RHO[2][laneId] + -3.f * RHO[3][laneId]));
			KeU[1] += u[0] * (LM[1] * (-3.f * RHO[2][laneId] + 3.f * RHO[3][laneId]));
			KeU[0] += u[0] * (LM[0] * (2.f * RHO[2][laneId] + 2.f * RHO[3][laneId]));
			KeU[0] += u[1] * (LM[1] * (3.f * RHO[2][laneId] + -3.f * RHO[3][laneId]));
			KeU[1] += u[1] * (LM[4] * (-2.f * RHO[2][laneId] + -2.f * RHO[3][laneId]));
			KeU[2] += u[1] * (LM[2] * (6.f * RHO[2][laneId] + 6.f * RHO[3][laneId]));
			KeU[1] += u[2] * (LM[2] * (6.f * RHO[2][laneId] + 6.f * RHO[3][laneId]));
			KeU[2] += u[2] * (LM[4] * (-2.f * RHO[2][laneId] + -2.f * RHO[3][laneId]));
			KeU[0] += u[2] * (LM[1] * (-3.f * RHO[2][laneId] + 3.f * RHO[3][laneId]));
			u[0] = gU[0][ev[8]]; u[1] = gU[1][ev[8]]; u[2] = gU[2][ev[8]];
			KeU[1] += u[0] * (LM[2] * (-3.f * RHO[3][laneId]));
			KeU[2] += u[0] * (LM[2] * (3.f * RHO[3][laneId]));
			KeU[0] += u[0] * (LM[3] * (-2.f * RHO[3][laneId]));
			KeU[2] += u[1] * (LM[2] * (3.f * RHO[3][laneId]));
			KeU[1] += u[1] * (LM[3] * (-2.f * RHO[3][laneId]));
			KeU[0] += u[1] * (LM[2] * (-3.f * RHO[3][laneId]));
			KeU[0] += u[2] * (LM[2] * (3.f * RHO[3][laneId]));
			KeU[1] += u[2] * (LM[2] * (3.f * RHO[3][laneId]));
			KeU[2] += u[2] * (LM[3] * (-2.f * RHO[3][laneId]));
		}
		else if (warpId == 3) {
			u[0] = gU[0][ev[9]]; u[1] = gU[1][ev[9]]; u[2] = gU[2][ev[9]];
			KeU[0] += u[0] * (LM[4] * (-2.f * RHO[0][laneId] + -2.f * RHO[4][laneId]));
			KeU[1] += u[0] * (LM[2] * (-6.f * RHO[0][laneId] + -6.f * RHO[4][laneId]));
			KeU[2] += u[0] * (LM[1] * (-3.f * RHO[0][laneId] + 3.f * RHO[4][laneId]));
			KeU[0] += u[1] * (LM[2] * (-6.f * RHO[0][laneId] + -6.f * RHO[4][laneId]));
			KeU[2] += u[1] * (LM[1] * (-3.f * RHO[0][laneId] + 3.f * RHO[4][laneId]));
			KeU[1] += u[1] * (LM[4] * (-2.f * RHO[0][laneId] + -2.f * RHO[4][laneId]));
			KeU[2] += u[2] * (LM[0] * (2.f * RHO[0][laneId] + 2.f * RHO[4][laneId]));
			KeU[1] += u[2] * (LM[1] * (3.f * RHO[0][laneId] + -3.f * RHO[4][laneId]));
			KeU[0] += u[2] * (LM[1] * (3.f * RHO[0][laneId] + -3.f * RHO[4][laneId]));
			u[0] = gU[0][ev[10]]; u[1] = gU[1][ev[10]]; u[2] = gU[2][ev[10]];
			KeU[2] += u[0] * (LM[2] * (3.f * RHO[0][laneId] + -3.f * RHO[1][laneId] + -3.f * RHO[4][laneId] + 3.f * RHO[5][laneId]));
			KeU[0] += u[0] * (LM[2] * (4.f * RHO[0][laneId] + 4.f * RHO[1][laneId] + 4.f * RHO[4][laneId] + 4.f * RHO[5][laneId]));
			KeU[1] += u[0] * (LM[1] * (6.f * RHO[0][laneId] + -6.f * RHO[1][laneId] + 6.f * RHO[4][laneId] + -6.f * RHO[5][laneId]));
			KeU[0] += u[1] * (LM[1] * (-6.f * RHO[0][laneId] + 6.f * RHO[1][laneId] + -6.f * RHO[4][laneId] + 6.f * RHO[5][laneId]));
			KeU[2] += u[1] * (LM[1] * (-6.f * RHO[0][laneId] + -6.f * RHO[1][laneId] + 6.f * RHO[4][laneId] + 6.f * RHO[5][laneId]));
			KeU[1] += u[1] * (LM[2] * (-8.f * RHO[0][laneId] + -8.f * RHO[1][laneId] + -8.f * RHO[4][laneId] + -8.f * RHO[5][laneId]));
			KeU[0] += u[2] * (LM[2] * (3.f * RHO[0][laneId] + -3.f * RHO[1][laneId] + -3.f * RHO[4][laneId] + 3.f * RHO[5][laneId]));
			KeU[2] += u[2] * (LM[2] * (4.f * RHO[0][laneId] + 4.f * RHO[1][laneId] + 4.f * RHO[4][laneId] + 4.f * RHO[5][laneId]));
			KeU[1] += u[2] * (LM[1] * (6.f * RHO[0][laneId] + 6.f * RHO[1][laneId] + -6.f * RHO[4][laneId] + -6.f * RHO[5][laneId]));
			u[0] = gU[0][ev[11]]; u[1] = gU[1][ev[11]]; u[2] = gU[2][ev[11]];
			KeU[0] += u[0] * (LM[4] * (-2.f * RHO[1][laneId] + -2.f * RHO[5][laneId]));
			KeU[1] += u[0] * (LM[2] * (6.f * RHO[1][laneId] + 6.f * RHO[5][laneId]));
			KeU[2] += u[0] * (LM[1] * (3.f * RHO[1][laneId] + -3.f * RHO[5][laneId]));
			KeU[2] += u[1] * (LM[1] * (-3.f * RHO[1][laneId] + 3.f * RHO[5][laneId]));
			KeU[1] += u[1] * (LM[4] * (-2.f * RHO[1][laneId] + -2.f * RHO[5][laneId]));
			KeU[0] += u[1] * (LM[2] * (6.f * RHO[1][laneId] + 6.f * RHO[5][laneId]));
			KeU[2] += u[2] * (LM[0] * (2.f * RHO[1][laneId] + 2.f * RHO[5][laneId]));
			KeU[1] += u[2] * (LM[1] * (3.f * RHO[1][laneId] + -3.f * RHO[5][laneId]));
			KeU[0] += u[2] * (LM[1] * (-3.f * RHO[1][laneId] + 3.f * RHO[5][laneId]));
		}
		else if (warpId == 4) {
			u[0] = gU[0][ev[12]]; u[1] = gU[1][ev[12]]; u[2] = gU[2][ev[12]];
			KeU[2] += u[0] * (LM[1] * (-6.f * RHO[0][laneId] + -6.f * RHO[2][laneId] + 6.f * RHO[4][laneId] + 6.f * RHO[6][laneId]));
			KeU[1] += u[0] * (LM[1] * (-6.f * RHO[0][laneId] + 6.f * RHO[2][laneId] + -6.f * RHO[4][laneId] + 6.f * RHO[6][laneId]));
			KeU[0] += u[0] * (LM[2] * (-8.f * RHO[0][laneId] + -8.f * RHO[2][laneId] + -8.f * RHO[4][laneId] + -8.f * RHO[6][laneId]));
			KeU[0] += u[1] * (LM[1] * (6.f * RHO[0][laneId] + -6.f * RHO[2][laneId] + 6.f * RHO[4][laneId] + -6.f * RHO[6][laneId]));
			KeU[2] += u[1] * (LM[2] * (3.f * RHO[0][laneId] + -3.f * RHO[2][laneId] + -3.f * RHO[4][laneId] + 3.f * RHO[6][laneId]));
			KeU[1] += u[1] * (LM[2] * (4.f * RHO[0][laneId] + 4.f * RHO[2][laneId] + 4.f * RHO[4][laneId] + 4.f * RHO[6][laneId]));
			KeU[0] += u[2] * (LM[1] * (6.f * RHO[0][laneId] + 6.f * RHO[2][laneId] + -6.f * RHO[4][laneId] + -6.f * RHO[6][laneId]));
			KeU[2] += u[2] * (LM[2] * (4.f * RHO[0][laneId] + 4.f * RHO[2][laneId] + 4.f * RHO[4][laneId] + 4.f * RHO[6][laneId]));
			KeU[1] += u[2] * (LM[2] * (3.f * RHO[0][laneId] + -3.f * RHO[2][laneId] + -3.f * RHO[4][laneId] + 3.f * RHO[6][laneId]));
#if 0
			u[0] = gU[0][ev[13]]; u[1] = gU[1][ev[13]]; u[2] = gU[2][ev[13]];
			KeU[2] += u[0] * (LM[2] * (6.f * RHO[0][laneId] + -6.f * RHO[1][laneId] + 6.f * RHO[2][laneId] + -6.f * RHO[3][laneId] + -6.f * RHO[4][laneId] + 6.f * RHO[5][laneId] + -6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]));
			KeU[1] += u[0] * (LM[2] * (6.f * RHO[0][laneId] + -6.f * RHO[1][laneId] + -6.f * RHO[2][laneId] + 6.f * RHO[3][laneId] + 6.f * RHO[4][laneId] + -6.f * RHO[5][laneId] + -6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]));
			KeU[0] += u[0] * (LM[3] * (8.f * RHO[0][laneId] + 8.f * RHO[1][laneId] + 8.f * RHO[2][laneId] + 8.f * RHO[3][laneId] + 8.f * RHO[4][laneId] + 8.f * RHO[5][laneId] + 8.f * RHO[6][laneId] + 8.f * RHO[7][laneId]));
			KeU[1] += u[1] * (LM[3] * (8.f * RHO[0][laneId] + 8.f * RHO[1][laneId] + 8.f * RHO[2][laneId] + 8.f * RHO[3][laneId] + 8.f * RHO[4][laneId] + 8.f * RHO[5][laneId] + 8.f * RHO[6][laneId] + 8.f * RHO[7][laneId]));
			KeU[2] += u[1] * (LM[2] * (6.f * RHO[0][laneId] + 6.f * RHO[1][laneId] + -6.f * RHO[2][laneId] + -6.f * RHO[3][laneId] + -6.f * RHO[4][laneId] + -6.f * RHO[5][laneId] + 6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]));
			KeU[0] += u[1] * (LM[2] * (6.f * RHO[0][laneId] + -6.f * RHO[1][laneId] + -6.f * RHO[2][laneId] + 6.f * RHO[3][laneId] + 6.f * RHO[4][laneId] + -6.f * RHO[5][laneId] + -6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]));
			KeU[1] += u[2] * (LM[2] * (6.f * RHO[0][laneId] + 6.f * RHO[1][laneId] + -6.f * RHO[2][laneId] + -6.f * RHO[3][laneId] + -6.f * RHO[4][laneId] + -6.f * RHO[5][laneId] + 6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]));
			KeU[0] += u[2] * (LM[2] * (6.f * RHO[0][laneId] + -6.f * RHO[1][laneId] + 6.f * RHO[2][laneId] + -6.f * RHO[3][laneId] + -6.f * RHO[4][laneId] + 6.f * RHO[5][laneId] + -6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]));
			KeU[2] += u[2] * (LM[3] * (8.f * RHO[0][laneId] + 8.f * RHO[1][laneId] + 8.f * RHO[2][laneId] + 8.f * RHO[3][laneId] + 8.f * RHO[4][laneId] + 8.f * RHO[5][laneId] + 8.f * RHO[6][laneId] + 8.f * RHO[7][laneId]));
#endif
			u[0] = gU[0][ev[14]]; u[1] = gU[1][ev[14]]; u[2] = gU[2][ev[14]];
			KeU[2] += u[0] * (LM[1] * (6.f * RHO[1][laneId] + 6.f * RHO[3][laneId] + -6.f * RHO[5][laneId] + -6.f * RHO[7][laneId]));
			KeU[0] += u[0] * (LM[2] * (-8.f * RHO[1][laneId] + -8.f * RHO[3][laneId] + -8.f * RHO[5][laneId] + -8.f * RHO[7][laneId]));
			KeU[1] += u[0] * (LM[1] * (6.f * RHO[1][laneId] + -6.f * RHO[3][laneId] + 6.f * RHO[5][laneId] + -6.f * RHO[7][laneId]));
			KeU[0] += u[1] * (LM[1] * (-6.f * RHO[1][laneId] + 6.f * RHO[3][laneId] + -6.f * RHO[5][laneId] + 6.f * RHO[7][laneId]));
			KeU[1] += u[1] * (LM[2] * (4.f * RHO[1][laneId] + 4.f * RHO[3][laneId] + 4.f * RHO[5][laneId] + 4.f * RHO[7][laneId]));
			KeU[2] += u[1] * (LM[2] * (3.f * RHO[1][laneId] + -3.f * RHO[3][laneId] + -3.f * RHO[5][laneId] + 3.f * RHO[7][laneId]));
			KeU[0] += u[2] * (LM[1] * (-6.f * RHO[1][laneId] + -6.f * RHO[3][laneId] + 6.f * RHO[5][laneId] + 6.f * RHO[7][laneId]));
			KeU[1] += u[2] * (LM[2] * (3.f * RHO[1][laneId] + -3.f * RHO[3][laneId] + -3.f * RHO[5][laneId] + 3.f * RHO[7][laneId]));
			KeU[2] += u[2] * (LM[2] * (4.f * RHO[1][laneId] + 4.f * RHO[3][laneId] + 4.f * RHO[5][laneId] + 4.f * RHO[7][laneId]));

			u[0] = gU[0][ev[15]]; u[1] = gU[1][ev[15]]; u[2] = gU[2][ev[15]];
			KeU[0] += u[0] * (LM[4] * (-2.f * RHO[2][laneId] + -2.f * RHO[6][laneId]));
			KeU[2] += u[0] * (LM[1] * (-3.f * RHO[2][laneId] + 3.f * RHO[6][laneId]));
			KeU[1] += u[0] * (LM[2] * (6.f * RHO[2][laneId] + 6.f * RHO[6][laneId]));
			KeU[2] += u[1] * (LM[1] * (3.f * RHO[2][laneId] + -3.f * RHO[6][laneId]));
			KeU[0] += u[1] * (LM[2] * (6.f * RHO[2][laneId] + 6.f * RHO[6][laneId]));
			KeU[1] += u[1] * (LM[4] * (-2.f * RHO[2][laneId] + -2.f * RHO[6][laneId]));
			KeU[2] += u[2] * (LM[0] * (2.f * RHO[2][laneId] + 2.f * RHO[6][laneId]));
			KeU[1] += u[2] * (LM[1] * (-3.f * RHO[2][laneId] + 3.f * RHO[6][laneId]));
			KeU[0] += u[2] * (LM[1] * (3.f * RHO[2][laneId] + -3.f * RHO[6][laneId]));
		}
		else if (warpId == 5) {
			u[0] = gU[0][ev[16]]; u[1] = gU[1][ev[16]]; u[2] = gU[2][ev[16]];
			KeU[1] += u[0] * (LM[1] * (-6.f * RHO[2][laneId] + 6.f * RHO[3][laneId] + -6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]));
			KeU[0] += u[0] * (LM[2] * (4.f * RHO[2][laneId] + 4.f * RHO[3][laneId] + 4.f * RHO[6][laneId] + 4.f * RHO[7][laneId]));
			KeU[2] += u[0] * (LM[2] * (3.f * RHO[2][laneId] + -3.f * RHO[3][laneId] + -3.f * RHO[6][laneId] + 3.f * RHO[7][laneId]));
			KeU[2] += u[1] * (LM[1] * (6.f * RHO[2][laneId] + 6.f * RHO[3][laneId] + -6.f * RHO[6][laneId] + -6.f * RHO[7][laneId]));
			KeU[0] += u[1] * (LM[1] * (6.f * RHO[2][laneId] + -6.f * RHO[3][laneId] + 6.f * RHO[6][laneId] + -6.f * RHO[7][laneId]));
			KeU[1] += u[1] * (LM[2] * (-8.f * RHO[2][laneId] + -8.f * RHO[3][laneId] + -8.f * RHO[6][laneId] + -8.f * RHO[7][laneId]));
			KeU[1] += u[2] * (LM[1] * (-6.f * RHO[2][laneId] + -6.f * RHO[3][laneId] + 6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]));
			KeU[2] += u[2] * (LM[2] * (4.f * RHO[2][laneId] + 4.f * RHO[3][laneId] + 4.f * RHO[6][laneId] + 4.f * RHO[7][laneId]));
			KeU[0] += u[2] * (LM[2] * (3.f * RHO[2][laneId] + -3.f * RHO[3][laneId] + -3.f * RHO[6][laneId] + 3.f * RHO[7][laneId]));
			u[0] = gU[0][ev[17]]; u[1] = gU[1][ev[17]]; u[2] = gU[2][ev[17]];
			KeU[1] += u[0] * (LM[2] * (-6.f * RHO[3][laneId] + -6.f * RHO[7][laneId]));
			KeU[2] += u[0] * (LM[1] * (3.f * RHO[3][laneId] + -3.f * RHO[7][laneId]));
			KeU[0] += u[0] * (LM[4] * (-2.f * RHO[3][laneId] + -2.f * RHO[7][laneId]));
			KeU[2] += u[1] * (LM[1] * (3.f * RHO[3][laneId] + -3.f * RHO[7][laneId]));
			KeU[0] += u[1] * (LM[2] * (-6.f * RHO[3][laneId] + -6.f * RHO[7][laneId]));
			KeU[1] += u[1] * (LM[4] * (-2.f * RHO[3][laneId] + -2.f * RHO[7][laneId]));
			KeU[0] += u[2] * (LM[1] * (-3.f * RHO[3][laneId] + 3.f * RHO[7][laneId]));
			KeU[1] += u[2] * (LM[1] * (-3.f * RHO[3][laneId] + 3.f * RHO[7][laneId]));
			KeU[2] += u[2] * (LM[0] * (2.f * RHO[3][laneId] + 2.f * RHO[7][laneId]));
			u[0] = gU[0][ev[18]]; u[1] = gU[1][ev[18]]; u[2] = gU[2][ev[18]];
			KeU[2] += u[0] * (LM[2] * (3.f * RHO[4][laneId]));
			KeU[0] += u[0] * (LM[3] * (-2.f * RHO[4][laneId]));
			KeU[1] += u[0] * (LM[2] * (-3.f * RHO[4][laneId]));
			KeU[0] += u[1] * (LM[2] * (-3.f * RHO[4][laneId]));
			KeU[2] += u[1] * (LM[2] * (3.f * RHO[4][laneId]));
			KeU[1] += u[1] * (LM[3] * (-2.f * RHO[4][laneId]));
			KeU[0] += u[2] * (LM[2] * (3.f * RHO[4][laneId]));
			KeU[1] += u[2] * (LM[2] * (3.f * RHO[4][laneId]));
			KeU[2] += u[2] * (LM[3] * (-2.f * RHO[4][laneId]));
		}
		else if (warpId == 6) {
			u[0] = gU[0][ev[19]]; u[1] = gU[1][ev[19]]; u[2] = gU[2][ev[19]];
			KeU[2] += u[0] * (LM[1] * (-3.f * RHO[4][laneId] + 3.f * RHO[5][laneId]));
			KeU[1] += u[0] * (LM[1] * (3.f * RHO[4][laneId] + -3.f * RHO[5][laneId]));
			KeU[0] += u[0] * (LM[0] * (2.f * RHO[4][laneId] + 2.f * RHO[5][laneId]));
			KeU[2] += u[1] * (LM[2] * (6.f * RHO[4][laneId] + 6.f * RHO[5][laneId]));
			KeU[0] += u[1] * (LM[1] * (-3.f * RHO[4][laneId] + 3.f * RHO[5][laneId]));
			KeU[1] += u[1] * (LM[4] * (-2.f * RHO[4][laneId] + -2.f * RHO[5][laneId]));
			KeU[0] += u[2] * (LM[1] * (3.f * RHO[4][laneId] + -3.f * RHO[5][laneId]));
			KeU[1] += u[2] * (LM[2] * (6.f * RHO[4][laneId] + 6.f * RHO[5][laneId]));
			KeU[2] += u[2] * (LM[4] * (-2.f * RHO[4][laneId] + -2.f * RHO[5][laneId]));
			u[0] = gU[0][ev[20]]; u[1] = gU[1][ev[20]]; u[2] = gU[2][ev[20]];
			KeU[1] += u[0] * (LM[2] * (3.f * RHO[5][laneId]));
			KeU[0] += u[0] * (LM[3] * (-2.f * RHO[5][laneId]));
			KeU[2] += u[0] * (LM[2] * (-3.f * RHO[5][laneId]));
			KeU[2] += u[1] * (LM[2] * (3.f * RHO[5][laneId]));
			KeU[0] += u[1] * (LM[2] * (3.f * RHO[5][laneId]));
			KeU[1] += u[1] * (LM[3] * (-2.f * RHO[5][laneId]));
			KeU[2] += u[2] * (LM[3] * (-2.f * RHO[5][laneId]));
			KeU[0] += u[2] * (LM[2] * (-3.f * RHO[5][laneId]));
			KeU[1] += u[2] * (LM[2] * (3.f * RHO[5][laneId]));
			u[0] = gU[0][ev[21]]; u[1] = gU[1][ev[21]]; u[2] = gU[2][ev[21]];
			KeU[1] += u[0] * (LM[1] * (-3.f * RHO[4][laneId] + 3.f * RHO[6][laneId]));
			KeU[0] += u[0] * (LM[4] * (-2.f * RHO[4][laneId] + -2.f * RHO[6][laneId]));
			KeU[2] += u[0] * (LM[2] * (6.f * RHO[4][laneId] + 6.f * RHO[6][laneId]));
			KeU[0] += u[1] * (LM[1] * (3.f * RHO[4][laneId] + -3.f * RHO[6][laneId]));
			KeU[2] += u[1] * (LM[1] * (-3.f * RHO[4][laneId] + 3.f * RHO[6][laneId]));
			KeU[1] += u[1] * (LM[0] * (2.f * RHO[4][laneId] + 2.f * RHO[6][laneId]));
			KeU[2] += u[2] * (LM[4] * (-2.f * RHO[4][laneId] + -2.f * RHO[6][laneId]));
			KeU[1] += u[2] * (LM[1] * (3.f * RHO[4][laneId] + -3.f * RHO[6][laneId]));
			KeU[0] += u[2] * (LM[2] * (6.f * RHO[4][laneId] + 6.f * RHO[6][laneId]));
		}
		else if (warpId == 7) {
			u[0] = gU[0][ev[22]]; u[1] = gU[1][ev[22]]; u[2] = gU[2][ev[22]];
			KeU[2] += u[0] * (LM[1] * (-6.f * RHO[4][laneId] + 6.f * RHO[5][laneId] + -6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]));
			KeU[1] += u[0] * (LM[2] * (3.f * RHO[4][laneId] + -3.f * RHO[5][laneId] + -3.f * RHO[6][laneId] + 3.f * RHO[7][laneId]));
			KeU[0] += u[0] * (LM[2] * (4.f * RHO[4][laneId] + 4.f * RHO[5][laneId] + 4.f * RHO[6][laneId] + 4.f * RHO[7][laneId]));
			KeU[1] += u[1] * (LM[2] * (4.f * RHO[4][laneId] + 4.f * RHO[5][laneId] + 4.f * RHO[6][laneId] + 4.f * RHO[7][laneId]));
			KeU[2] += u[1] * (LM[1] * (-6.f * RHO[4][laneId] + -6.f * RHO[5][laneId] + 6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]));
			KeU[0] += u[1] * (LM[2] * (3.f * RHO[4][laneId] + -3.f * RHO[5][laneId] + -3.f * RHO[6][laneId] + 3.f * RHO[7][laneId]));
			KeU[0] += u[2] * (LM[1] * (6.f * RHO[4][laneId] + -6.f * RHO[5][laneId] + 6.f * RHO[6][laneId] + -6.f * RHO[7][laneId]));
			KeU[1] += u[2] * (LM[1] * (6.f * RHO[4][laneId] + 6.f * RHO[5][laneId] + -6.f * RHO[6][laneId] + -6.f * RHO[7][laneId]));
			KeU[2] += u[2] * (LM[2] * (-8.f * RHO[4][laneId] + -8.f * RHO[5][laneId] + -8.f * RHO[6][laneId] + -8.f * RHO[7][laneId]));
			u[0] = gU[0][ev[23]]; u[1] = gU[1][ev[23]]; u[2] = gU[2][ev[23]];
			KeU[1] += u[0] * (LM[1] * (3.f * RHO[5][laneId] + -3.f * RHO[7][laneId]));
			KeU[2] += u[0] * (LM[2] * (-6.f * RHO[5][laneId] + -6.f * RHO[7][laneId]));
			KeU[0] += u[0] * (LM[4] * (-2.f * RHO[5][laneId] + -2.f * RHO[7][laneId]));
			KeU[1] += u[1] * (LM[0] * (2.f * RHO[5][laneId] + 2.f * RHO[7][laneId]));
			KeU[2] += u[1] * (LM[1] * (-3.f * RHO[5][laneId] + 3.f * RHO[7][laneId]));
			KeU[0] += u[1] * (LM[1] * (-3.f * RHO[5][laneId] + 3.f * RHO[7][laneId]));
			KeU[1] += u[2] * (LM[1] * (3.f * RHO[5][laneId] + -3.f * RHO[7][laneId]));
			KeU[0] += u[2] * (LM[2] * (-6.f * RHO[5][laneId] + -6.f * RHO[7][laneId]));
			KeU[2] += u[2] * (LM[4] * (-2.f * RHO[5][laneId] + -2.f * RHO[7][laneId]));
			u[0] = gU[0][ev[24]]; u[1] = gU[1][ev[24]]; u[2] = gU[2][ev[24]];
			KeU[0] += u[0] * (LM[3] * (-2.f * RHO[6][laneId]));
			KeU[2] += u[0] * (LM[2] * (3.f * RHO[6][laneId]));
			KeU[1] += u[0] * (LM[2] * (3.f * RHO[6][laneId]));
			KeU[2] += u[1] * (LM[2] * (-3.f * RHO[6][laneId]));
			KeU[1] += u[1] * (LM[3] * (-2.f * RHO[6][laneId]));
			KeU[0] += u[1] * (LM[2] * (3.f * RHO[6][laneId]));
			KeU[0] += u[2] * (LM[2] * (3.f * RHO[6][laneId]));
			KeU[2] += u[2] * (LM[3] * (-2.f * RHO[6][laneId]));
			KeU[1] += u[2] * (LM[2] * (-3.f * RHO[6][laneId]));
			u[0] = gU[0][ev[25]]; u[1] = gU[1][ev[25]]; u[2] = gU[2][ev[25]];
			KeU[0] += u[0] * (LM[0] * (2.f * RHO[6][laneId] + 2.f * RHO[7][laneId]));
			KeU[1] += u[0] * (LM[1] * (-3.f * RHO[6][laneId] + 3.f * RHO[7][laneId]));
			KeU[2] += u[0] * (LM[1] * (-3.f * RHO[6][laneId] + 3.f * RHO[7][laneId]));
			KeU[2] += u[1] * (LM[2] * (-6.f * RHO[6][laneId] + -6.f * RHO[7][laneId]));
			KeU[0] += u[1] * (LM[1] * (3.f * RHO[6][laneId] + -3.f * RHO[7][laneId]));
			KeU[1] += u[1] * (LM[4] * (-2.f * RHO[6][laneId] + -2.f * RHO[7][laneId]));
			KeU[1] += u[2] * (LM[2] * (-6.f * RHO[6][laneId] + -6.f * RHO[7][laneId]));
			KeU[0] += u[2] * (LM[1] * (3.f * RHO[6][laneId] + -3.f * RHO[7][laneId]));
			KeU[2] += u[2] * (LM[4] * (-2.f * RHO[6][laneId] + -2.f * RHO[7][laneId]));
			u[0] = gU[0][ev[26]]; u[1] = gU[1][ev[26]]; u[2] = gU[2][ev[26]];
			KeU[0] += u[0] * (LM[3] * (-2.f * RHO[7][laneId]));
			KeU[1] += u[0] * (LM[2] * (-3.f * RHO[7][laneId]));
			KeU[2] += u[0] * (LM[2] * (-3.f * RHO[7][laneId]));
			KeU[1] += u[1] * (LM[3] * (-2.f * RHO[7][laneId]));
			KeU[2] += u[1] * (LM[2] * (-3.f * RHO[7][laneId]));
			KeU[0] += u[1] * (LM[2] * (-3.f * RHO[7][laneId]));
			KeU[1] += u[2] * (LM[2] * (-3.f * RHO[7][laneId]));
			KeU[2] += u[2] * (LM[3] * (-2.f * RHO[7][laneId]));
			KeU[0] += u[2] * (LM[2] * (-3.f * RHO[7][laneId]));
		}
	}

#if 1

	if (warpId >= 4) {
		for (int i = 0; i < 3; i++) sumKeU[i][warpId - 4][laneId] = KeU[i];
	}
	__syncthreads();
	if (warpId < 4) {
		for (int i = 0; i < 3; i++) sumKeU[i][warpId][laneId] += KeU[i];
	}
	__syncthreads();
	if (warpId < 2) {
		for (int i = 0; i < 3; i++) sumKeU[i][warpId][laneId] += sumKeU[i][warpId + 2][laneId];
	}
	__syncthreads();
	if (warpId < 1 && !fiction) {
		for (int i = 0; i < 3; i++) KeU[i] = sumKeU[i][warpId][laneId] + sumKeU[i][warpId + 1][laneId];

		//if (ev[13] == 394689) {
		//	printf("ku = (%.4le, %.4le, %.4le)\n", KeU[0], KeU[1], KeU[2]);
		//}
		//double f[3] = { gF[0][ev[13]], gF[1][ev[13]], gF[2][ev[13]] };
		u[0] = gU[0][ev[13]]; u[1] = gU[1][ev[13]]; u[2] = gU[2][ev[13]];
		//KeU[0] =
		//	u[0] * (LM[3] * (8.f * RHO[0][laneId] + 8.f * RHO[1][laneId] + 8.f * RHO[2][laneId] + 8.f * RHO[3][laneId] + 8.f * RHO[4][laneId] + 8.f * RHO[5][laneId] + 8.f * RHO[6][laneId] + 8.f * RHO[7][laneId])) 
		//	+ u[1] * (LM[2] * (6.f * RHO[0][laneId] + -6.f * RHO[1][laneId] + -6.f * RHO[2][laneId] + 6.f * RHO[3][laneId] + 6.f * RHO[4][laneId] + -6.f * RHO[5][laneId] + -6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]))
		//	+ u[2] * (LM[2] * (6.f * RHO[0][laneId] + -6.f * RHO[1][laneId] + 6.f * RHO[2][laneId] + -6.f * RHO[3][laneId] + -6.f * RHO[4][laneId] + 6.f * RHO[5][laneId] + -6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]));
		//KeU[1] += u[0] * (LM[2] * (6.f * RHO[0][laneId] + -6.f * RHO[1][laneId] + -6.f * RHO[2][laneId] + 6.f * RHO[3][laneId] + 6.f * RHO[4][laneId] + -6.f * RHO[5][laneId] + -6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]))
		//	+ u[2] * (LM[2] * (6.f * RHO[0][laneId] + 6.f * RHO[1][laneId] + -6.f * RHO[2][laneId] + -6.f * RHO[3][laneId] + -6.f * RHO[4][laneId] + -6.f * RHO[5][laneId] + 6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]))
		//	+ u[1] * (LM[3] * (8.f * RHO[0][laneId] + 8.f * RHO[1][laneId] + 8.f * RHO[2][laneId] + 8.f * RHO[3][laneId] + 8.f * RHO[4][laneId] + 8.f * RHO[5][laneId] + 8.f * RHO[6][laneId] + 8.f * RHO[7][laneId]));
		//KeU[2] += u[0] * (LM[2] * (6.f * RHO[0][laneId] + -6.f * RHO[1][laneId] + 6.f * RHO[2][laneId] + -6.f * RHO[3][laneId] + -6.f * RHO[4][laneId] + 6.f * RHO[5][laneId] + -6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]))
		//	+ u[1] * (LM[2] * (6.f * RHO[0][laneId] + 6.f * RHO[1][laneId] + -6.f * RHO[2][laneId] + -6.f * RHO[3][laneId] + -6.f * RHO[4][laneId] + -6.f * RHO[5][laneId] + 6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]))
		//	+ u[2] * (LM[3] * (8.f * RHO[0][laneId] + 8.f * RHO[1][laneId] + 8.f * RHO[2][laneId] + 8.f * RHO[3][laneId] + 8.f * RHO[4][laneId] + 8.f * RHO[5][laneId] + 8.f * RHO[6][laneId] + 8.f * RHO[7][laneId]));
		float d = LM[3] * 8.f * (RHO[0][laneId] + RHO[1][laneId] + RHO[2][laneId] + RHO[3][laneId] + RHO[4][laneId] + RHO[5][laneId] + RHO[6][laneId] + RHO[7][laneId]);
		float t1 = LM[2] * (6.f * RHO[0][laneId] + -6.f * RHO[1][laneId] + -6.f * RHO[2][laneId] + 6.f * RHO[3][laneId] + 6.f * RHO[4][laneId] + -6.f * RHO[5][laneId] + -6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]);
		float t2 = LM[2] * (6.f * RHO[0][laneId] + -6.f * RHO[1][laneId] + 6.f * RHO[2][laneId] + -6.f * RHO[3][laneId] + -6.f * RHO[4][laneId] + 6.f * RHO[5][laneId] + -6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]);
		float t3 = LM[2] * (6.f * RHO[0][laneId] + 6.f * RHO[1][laneId] + -6.f * RHO[2][laneId] + -6.f * RHO[3][laneId] + -6.f * RHO[4][laneId] + -6.f * RHO[5][laneId] + 6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]);
		u[0] = w * (gF[0][ev[13]] - KeU[0] - u[1] * t1 - u[2] * t2) / d + (1. - w) * u[0];
		u[1] = w * (gF[1][ev[13]] - KeU[1] - u[0] * t1 - u[2] * t3) / d + (1. - w) * u[1];
		u[2] = w * (gF[2][ev[13]] - KeU[2] - u[0] * t2 - u[1] * t3) / d + (1. - w) * u[2];

		if (vflag.is_dirichlet_boundary()) {
			u[0] = 0; u[1] = 0; u[2] = 0;
		}

		gU[0][ev[13]] = u[0]; gU[1][ev[13]] = u[1]; gU[2][ev[13]] = u[2];
	}
#endif
}

void homo::Grid::gs_relaxation_ex(float w_SOR /*= 1.f*/)
{
	if (!is_root) return;
	// change to 8 bytes bank
	use4Bytesbank();
	useGrid_g();
	devArray_t<int, 3>  gridCellReso{};
	devArray_t<int, 8>  gsCellEnd{};
	devArray_t<int, 8>  gsVertexEnd{};
	for (int i = 0; i < 8; i++) {
		gsCellEnd[i] = gsCellSetEnd[i];
		gsVertexEnd[i] = gsVertexSetEnd[i];
		if (i < 3) gridCellReso[i] = cellReso[i];
	}
	for (int i = 0; i < 8; i++) {
		size_t grid_size, block_size;
		int n_gs = gsVertexEnd[i] - (i == 0 ? 0 : gsVertexEnd[i - 1]);
		make_kernel_param(&grid_size, &block_size, n_gs * 8, 32 * 8);
		gs_relaxation_otf_kernel_opt << <grid_size, block_size >> > (i, rho_g, gridCellReso, vertflag, cellflag, w_SOR);
		cudaDeviceSynchronize();
		cuda_error_check;
		enforce_period_boundary(u_g);
	}
	enforce_period_boundary(u_g);
	//pad_vertex_data(u_g);
	cudaDeviceSynchronize();
	cuda_error_check;

}

__global__ void interpDensityFrom_kernel(float* rholist, cudaTextureObject_t rhoTex) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int reso[3] = { gGridCellReso[0],gGridCellReso[1],gGridCellReso[2] };
	int ne = gGridCellReso[0] * gGridCellReso[1] * gGridCellReso[2];
	if (tid >= ne) return;
	int epos[3] = { tid % reso[0], tid / reso[0] % reso[1], tid / (reso[0] * reso[1]) };

	//lexi2gs(int lexpos[3], int gsreso[3][8], int gsend[8], bool padded = false) 
	int gsid = lexi2gs(epos, gGsCellReso, gGsCellEnd);
	//float3 p{ float(epos[0]) / reso[0], float(epos[1]) / reso[1], float(epos[2]) / reso[2] };
	float p[3] = { float(epos[0]) / reso[0], float(epos[1]) / reso[1], float(epos[2]) / reso[2] };
	float f = tex3D<float>(rhoTex, p[0], p[1], p[2]);
	rholist[gsid] = f;
}

void homo::Grid::interpDensityFrom(const std::string& fname, VoxelIOFormat frmat)
{
	useGrid_g();
	std::vector<float> values;
	int reso[3];
	readDensity(fname, values, reso, frmat);

	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaArray* cuArray;
	cudaExtent extent{ reso[0],reso[1],reso[2] };
	//cudaMalloc3DArray(cudaArray_t * array, const struct cudaChannelFormatDesc* desc, struct cudaExtent extent, unsigned int flags __dv(0));
	CheckErr(cudaMalloc3DArray(&cuArray, &channelDesc, extent));
	// Copy to device memory some data located at address h_data
    // in host memory cudaMemcpy3DParms copyParams = {0};
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr(values.data(), reso[0] * sizeof(float), reso[1], reso[2]);
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
	// Set texture description
	struct cudaTextureDesc rhoTexDesc;
	memset(&rhoTexDesc, 0, sizeof(rhoTexDesc));
	rhoTexDesc.addressMode[0] = cudaAddressModeBorder;
	rhoTexDesc.addressMode[1] = cudaAddressModeBorder;
	rhoTexDesc.addressMode[2] = cudaAddressModeBorder;
	rhoTexDesc.filterMode = cudaFilterModeLinear;
	rhoTexDesc.readMode = cudaReadModeElementType;
	rhoTexDesc.normalizedCoords = 1;
	// create texture object
	cudaTextureObject_t rhoTex = 0;
	CheckErr(cudaCreateTextureObject(&rhoTex, &resDesc, &rhoTexDesc, NULL));
	
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, n_cells(), 256);
	interpDensityFrom_kernel << <grid_size, block_size >> > (rho_g, rhoTex);
	cudaDeviceSynchronize();
	cuda_error_check;

	CheckErr(cudaDestroyTextureObject(rhoTex));
	CheckErr(cudaFreeArray(cuArray));
	cuda_error_check;

	pad_cell_data(rho_g);
}


//template<int BlockSize = 256>
__global__ void update_residual_otf_kernel_opt(
	int nv, float* rholist,
	devArray_t<int, 3> gridCellReso, 
	VertexFlags* vflags, CellFlags* eflags,
	float diag_strength
) {
	__shared__ float LM[5];
	__shared__ float RHO[8][32];
	__shared__ float sumKeU[3][4][32];
	__shared__ float sumKs[3][3];

	// load lam and mu 
	if (threadIdx.x < 5) {
		LM[threadIdx.x] = gLM[threadIdx.x];
	}

	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;

	int vid = blockIdx.x * 32 + laneId;
	
	bool fiction = false;

	fiction = vid >= nv;

	GridVertexIndex indexer(gridCellReso[0], gridCellReso[1], gridCellReso[2]);
	VertexFlags vflag;
	if (!fiction) {
		vflag = vflags[vid];
		fiction = fiction || vflag.is_fiction() || vflag.is_period_padding();
		indexer.locate(vid, vflag.get_gscolor(), gGsVertexEnd);
	}

	// load density field
	for (int i = 0; i < 8; i++) { RHO[i][laneId] = 0; }
	if (!fiction) {
		CellFlags eflag;
#pragma unroll
		for (int i = 0; i < 8; i++) {
			int elementId = indexer.neighElement(i, gGsCellEnd, gGsCellReso).getId();
			if (elementId != -1) {
				eflag = eflags[elementId];
			/*	if (!eflag.is_fiction())*/ RHO[i][laneId] = powf(rholist[elementId], exp_penal[0]);
			} 	
		}
	} 
	__syncthreads();

	int ev[27];
	if (!fiction) {
		for (int i = 0; i < 27; i++) {
			VertexFlags nvflag;
			int vneighId = indexer.neighVertex(i, gGsVertexEnd, gGsVertexReso).getId();
			ev[i] = vneighId;
			if (vneighId != -1) {
				nvflag = vflags[vneighId];
			}
		}
	}

	float KeU[3] = { 0. };
	float u[3];
	if (!fiction) {
		if (warpId == 0) {
			u[0] = gU[0][ev[0]]; u[1] = gU[1][ev[0]]; u[2] = gU[2][ev[0]];
			KeU[0] += u[0] * (LM[3] * (-2.f * RHO[0][laneId]));
			KeU[1] += u[0] * (LM[2] * (-3.f * RHO[0][laneId]));
			KeU[2] += u[0] * (LM[2] * (-3.f * RHO[0][laneId]));
			KeU[2] += u[1] * (LM[2] * (-3.f * RHO[0][laneId]));
			KeU[1] += u[1] * (LM[3] * (-2.f * RHO[0][laneId]));
			KeU[0] += u[1] * (LM[2] * (-3.f * RHO[0][laneId]));
			KeU[2] += u[2] * (LM[3] * (-2.f * RHO[0][laneId]));
			KeU[0] += u[2] * (LM[2] * (-3.f * RHO[0][laneId]));
			KeU[1] += u[2] * (LM[2] * (-3.f * RHO[0][laneId]));
			u[0] = gU[0][ev[1]]; u[1] = gU[1][ev[1]]; u[2] = gU[2][ev[1]];
			KeU[0] += u[0] * (LM[0] * (2.f * RHO[0][laneId] + 2.f * RHO[1][laneId]));
			KeU[1] += u[0] * (LM[1] * (3.f * RHO[0][laneId] + -3.f * RHO[1][laneId]));
			KeU[2] += u[0] * (LM[1] * (3.f * RHO[0][laneId] + -3.f * RHO[1][laneId]));
			KeU[2] += u[1] * (LM[2] * (-6.f * RHO[0][laneId] + -6.f * RHO[1][laneId]));
			KeU[0] += u[1] * (LM[1] * (-3.f * RHO[0][laneId] + 3.f * RHO[1][laneId]));
			KeU[1] += u[1] * (LM[4] * (-2.f * RHO[0][laneId] + -2.f * RHO[1][laneId]));
			KeU[1] += u[2] * (LM[2] * (-6.f * RHO[0][laneId] + -6.f * RHO[1][laneId]));
			KeU[2] += u[2] * (LM[4] * (-2.f * RHO[0][laneId] + -2.f * RHO[1][laneId]));
			KeU[0] += u[2] * (LM[1] * (-3.f * RHO[0][laneId] + 3.f * RHO[1][laneId]));
			u[0] = gU[0][ev[2]]; u[1] = gU[1][ev[2]]; u[2] = gU[2][ev[2]];
			KeU[0] += u[0] * (LM[3] * (-2.f * RHO[1][laneId]));
			KeU[1] += u[0] * (LM[2] * (3.f * RHO[1][laneId]));
			KeU[2] += u[0] * (LM[2] * (3.f * RHO[1][laneId]));
			KeU[1] += u[1] * (LM[3] * (-2.f * RHO[1][laneId]));
			KeU[2] += u[1] * (LM[2] * (-3.f * RHO[1][laneId]));
			KeU[0] += u[1] * (LM[2] * (3.f * RHO[1][laneId]));
			KeU[1] += u[2] * (LM[2] * (-3.f * RHO[1][laneId]));
			KeU[2] += u[2] * (LM[3] * (-2.f * RHO[1][laneId]));
			KeU[0] += u[2] * (LM[2] * (3.f * RHO[1][laneId]));
		}
		else if (warpId == 1) {
			u[0] = gU[0][ev[3]]; u[1] = gU[1][ev[3]]; u[2] = gU[2][ev[3]];
			KeU[0] += u[0] * (LM[4] * (-2.f * RHO[0][laneId] + -2.f * RHO[2][laneId]));
			KeU[1] += u[0] * (LM[1] * (-3.f * RHO[0][laneId] + 3.f * RHO[2][laneId]));
			KeU[2] += u[0] * (LM[2] * (-6.f * RHO[0][laneId] + -6.f * RHO[2][laneId]));
			KeU[1] += u[1] * (LM[0] * (2.f * RHO[0][laneId] + 2.f * RHO[2][laneId]));
			KeU[0] += u[1] * (LM[1] * (3.f * RHO[0][laneId] + -3.f * RHO[2][laneId]));
			KeU[2] += u[1] * (LM[1] * (3.f * RHO[0][laneId] + -3.f * RHO[2][laneId]));
			KeU[1] += u[2] * (LM[1] * (-3.f * RHO[0][laneId] + 3.f * RHO[2][laneId]));
			KeU[2] += u[2] * (LM[4] * (-2.f * RHO[0][laneId] + -2.f * RHO[2][laneId]));
			KeU[0] += u[2] * (LM[2] * (-6.f * RHO[0][laneId] + -6.f * RHO[2][laneId]));
			u[0] = gU[0][ev[4]]; u[1] = gU[1][ev[4]]; u[2] = gU[2][ev[4]];
			KeU[2] += u[0] * (LM[1] * (6.f * RHO[0][laneId] + -6.f * RHO[1][laneId] + 6.f * RHO[2][laneId] + -6.f * RHO[3][laneId]));
			KeU[0] += u[0] * (LM[2] * (4.f * RHO[0][laneId] + 4.f * RHO[1][laneId] + 4.f * RHO[2][laneId] + 4.f * RHO[3][laneId]));
			KeU[1] += u[0] * (LM[2] * (3.f * RHO[0][laneId] + -3.f * RHO[1][laneId] + -3.f * RHO[2][laneId] + 3.f * RHO[3][laneId]));
			KeU[2] += u[1] * (LM[1] * (6.f * RHO[0][laneId] + 6.f * RHO[1][laneId] + -6.f * RHO[2][laneId] + -6.f * RHO[3][laneId]));
			KeU[0] += u[1] * (LM[2] * (3.f * RHO[0][laneId] + -3.f * RHO[1][laneId] + -3.f * RHO[2][laneId] + 3.f * RHO[3][laneId]));
			KeU[1] += u[1] * (LM[2] * (4.f * RHO[0][laneId] + 4.f * RHO[1][laneId] + 4.f * RHO[2][laneId] + 4.f * RHO[3][laneId]));
			KeU[0] += u[2] * (LM[1] * (-6.f * RHO[0][laneId] + 6.f * RHO[1][laneId] + -6.f * RHO[2][laneId] + 6.f * RHO[3][laneId]));
			KeU[2] += u[2] * (LM[2] * (-8.f * RHO[0][laneId] + -8.f * RHO[1][laneId] + -8.f * RHO[2][laneId] + -8.f * RHO[3][laneId]));
			KeU[1] += u[2] * (LM[1] * (-6.f * RHO[0][laneId] + -6.f * RHO[1][laneId] + 6.f * RHO[2][laneId] + 6.f * RHO[3][laneId]));
			u[0] = gU[0][ev[5]]; u[1] = gU[1][ev[5]]; u[2] = gU[2][ev[5]];
			KeU[2] += u[0] * (LM[2] * (6.f * RHO[1][laneId] + 6.f * RHO[3][laneId]));
			KeU[1] += u[0] * (LM[1] * (3.f * RHO[1][laneId] + -3.f * RHO[3][laneId]));
			KeU[0] += u[0] * (LM[4] * (-2.f * RHO[1][laneId] + -2.f * RHO[3][laneId]));
			KeU[1] += u[1] * (LM[0] * (2.f * RHO[1][laneId] + 2.f * RHO[3][laneId]));
			KeU[2] += u[1] * (LM[1] * (3.f * RHO[1][laneId] + -3.f * RHO[3][laneId]));
			KeU[0] += u[1] * (LM[1] * (-3.f * RHO[1][laneId] + 3.f * RHO[3][laneId]));
			KeU[2] += u[2] * (LM[4] * (-2.f * RHO[1][laneId] + -2.f * RHO[3][laneId]));
			KeU[1] += u[2] * (LM[1] * (-3.f * RHO[1][laneId] + 3.f * RHO[3][laneId]));
			KeU[0] += u[2] * (LM[2] * (6.f * RHO[1][laneId] + 6.f * RHO[3][laneId]));
		}
		else if (warpId == 2) {
			u[0] = gU[0][ev[6]]; u[1] = gU[1][ev[6]]; u[2] = gU[2][ev[6]];
			KeU[0] += u[0] * (LM[3] * (-2.f * RHO[2][laneId]));
			KeU[1] += u[0] * (LM[2] * (3.f * RHO[2][laneId]));
			KeU[2] += u[0] * (LM[2] * (-3.f * RHO[2][laneId]));
			KeU[0] += u[1] * (LM[2] * (3.f * RHO[2][laneId]));
			KeU[2] += u[1] * (LM[2] * (3.f * RHO[2][laneId]));
			KeU[1] += u[1] * (LM[3] * (-2.f * RHO[2][laneId]));
			KeU[2] += u[2] * (LM[3] * (-2.f * RHO[2][laneId]));
			KeU[0] += u[2] * (LM[2] * (-3.f * RHO[2][laneId]));
			KeU[1] += u[2] * (LM[2] * (3.f * RHO[2][laneId]));
			u[0] = gU[0][ev[7]]; u[1] = gU[1][ev[7]]; u[2] = gU[2][ev[7]];
			KeU[2] += u[0] * (LM[1] * (3.f * RHO[2][laneId] + -3.f * RHO[3][laneId]));
			KeU[1] += u[0] * (LM[1] * (-3.f * RHO[2][laneId] + 3.f * RHO[3][laneId]));
			KeU[0] += u[0] * (LM[0] * (2.f * RHO[2][laneId] + 2.f * RHO[3][laneId]));
			KeU[0] += u[1] * (LM[1] * (3.f * RHO[2][laneId] + -3.f * RHO[3][laneId]));
			KeU[1] += u[1] * (LM[4] * (-2.f * RHO[2][laneId] + -2.f * RHO[3][laneId]));
			KeU[2] += u[1] * (LM[2] * (6.f * RHO[2][laneId] + 6.f * RHO[3][laneId]));
			KeU[1] += u[2] * (LM[2] * (6.f * RHO[2][laneId] + 6.f * RHO[3][laneId]));
			KeU[2] += u[2] * (LM[4] * (-2.f * RHO[2][laneId] + -2.f * RHO[3][laneId]));
			KeU[0] += u[2] * (LM[1] * (-3.f * RHO[2][laneId] + 3.f * RHO[3][laneId]));
			u[0] = gU[0][ev[8]]; u[1] = gU[1][ev[8]]; u[2] = gU[2][ev[8]];
			KeU[1] += u[0] * (LM[2] * (-3.f * RHO[3][laneId]));
			KeU[2] += u[0] * (LM[2] * (3.f * RHO[3][laneId]));
			KeU[0] += u[0] * (LM[3] * (-2.f * RHO[3][laneId]));
			KeU[2] += u[1] * (LM[2] * (3.f * RHO[3][laneId]));
			KeU[1] += u[1] * (LM[3] * (-2.f * RHO[3][laneId]));
			KeU[0] += u[1] * (LM[2] * (-3.f * RHO[3][laneId]));
			KeU[0] += u[2] * (LM[2] * (3.f * RHO[3][laneId]));
			KeU[1] += u[2] * (LM[2] * (3.f * RHO[3][laneId]));
			KeU[2] += u[2] * (LM[3] * (-2.f * RHO[3][laneId]));
		}
		else if (warpId == 3) {
			u[0] = gU[0][ev[9]]; u[1] = gU[1][ev[9]]; u[2] = gU[2][ev[9]];
			KeU[0] += u[0] * (LM[4] * (-2.f * RHO[0][laneId] + -2.f * RHO[4][laneId]));
			KeU[1] += u[0] * (LM[2] * (-6.f * RHO[0][laneId] + -6.f * RHO[4][laneId]));
			KeU[2] += u[0] * (LM[1] * (-3.f * RHO[0][laneId] + 3.f * RHO[4][laneId]));
			KeU[0] += u[1] * (LM[2] * (-6.f * RHO[0][laneId] + -6.f * RHO[4][laneId]));
			KeU[2] += u[1] * (LM[1] * (-3.f * RHO[0][laneId] + 3.f * RHO[4][laneId]));
			KeU[1] += u[1] * (LM[4] * (-2.f * RHO[0][laneId] + -2.f * RHO[4][laneId]));
			KeU[2] += u[2] * (LM[0] * (2.f * RHO[0][laneId] + 2.f * RHO[4][laneId]));
			KeU[1] += u[2] * (LM[1] * (3.f * RHO[0][laneId] + -3.f * RHO[4][laneId]));
			KeU[0] += u[2] * (LM[1] * (3.f * RHO[0][laneId] + -3.f * RHO[4][laneId]));
			u[0] = gU[0][ev[10]]; u[1] = gU[1][ev[10]]; u[2] = gU[2][ev[10]];
			KeU[2] += u[0] * (LM[2] * (3.f * RHO[0][laneId] + -3.f * RHO[1][laneId] + -3.f * RHO[4][laneId] + 3.f * RHO[5][laneId]));
			KeU[0] += u[0] * (LM[2] * (4.f * RHO[0][laneId] + 4.f * RHO[1][laneId] + 4.f * RHO[4][laneId] + 4.f * RHO[5][laneId]));
			KeU[1] += u[0] * (LM[1] * (6.f * RHO[0][laneId] + -6.f * RHO[1][laneId] + 6.f * RHO[4][laneId] + -6.f * RHO[5][laneId]));
			KeU[0] += u[1] * (LM[1] * (-6.f * RHO[0][laneId] + 6.f * RHO[1][laneId] + -6.f * RHO[4][laneId] + 6.f * RHO[5][laneId]));
			KeU[2] += u[1] * (LM[1] * (-6.f * RHO[0][laneId] + -6.f * RHO[1][laneId] + 6.f * RHO[4][laneId] + 6.f * RHO[5][laneId]));
			KeU[1] += u[1] * (LM[2] * (-8.f * RHO[0][laneId] + -8.f * RHO[1][laneId] + -8.f * RHO[4][laneId] + -8.f * RHO[5][laneId]));
			KeU[0] += u[2] * (LM[2] * (3.f * RHO[0][laneId] + -3.f * RHO[1][laneId] + -3.f * RHO[4][laneId] + 3.f * RHO[5][laneId]));
			KeU[2] += u[2] * (LM[2] * (4.f * RHO[0][laneId] + 4.f * RHO[1][laneId] + 4.f * RHO[4][laneId] + 4.f * RHO[5][laneId]));
			KeU[1] += u[2] * (LM[1] * (6.f * RHO[0][laneId] + 6.f * RHO[1][laneId] + -6.f * RHO[4][laneId] + -6.f * RHO[5][laneId]));
			u[0] = gU[0][ev[11]]; u[1] = gU[1][ev[11]]; u[2] = gU[2][ev[11]];
			KeU[0] += u[0] * (LM[4] * (-2.f * RHO[1][laneId] + -2.f * RHO[5][laneId]));
			KeU[1] += u[0] * (LM[2] * (6.f * RHO[1][laneId] + 6.f * RHO[5][laneId]));
			KeU[2] += u[0] * (LM[1] * (3.f * RHO[1][laneId] + -3.f * RHO[5][laneId]));
			KeU[2] += u[1] * (LM[1] * (-3.f * RHO[1][laneId] + 3.f * RHO[5][laneId]));
			KeU[1] += u[1] * (LM[4] * (-2.f * RHO[1][laneId] + -2.f * RHO[5][laneId]));
			KeU[0] += u[1] * (LM[2] * (6.f * RHO[1][laneId] + 6.f * RHO[5][laneId]));
			KeU[2] += u[2] * (LM[0] * (2.f * RHO[1][laneId] + 2.f * RHO[5][laneId]));
			KeU[1] += u[2] * (LM[1] * (3.f * RHO[1][laneId] + -3.f * RHO[5][laneId]));
			KeU[0] += u[2] * (LM[1] * (-3.f * RHO[1][laneId] + 3.f * RHO[5][laneId]));
		}
		else if (warpId == 4) {
			u[0] = gU[0][ev[12]]; u[1] = gU[1][ev[12]]; u[2] = gU[2][ev[12]];
			KeU[2] += u[0] * (LM[1] * (-6.f * RHO[0][laneId] + -6.f * RHO[2][laneId] + 6.f * RHO[4][laneId] + 6.f * RHO[6][laneId]));
			KeU[1] += u[0] * (LM[1] * (-6.f * RHO[0][laneId] + 6.f * RHO[2][laneId] + -6.f * RHO[4][laneId] + 6.f * RHO[6][laneId]));
			KeU[0] += u[0] * (LM[2] * (-8.f * RHO[0][laneId] + -8.f * RHO[2][laneId] + -8.f * RHO[4][laneId] + -8.f * RHO[6][laneId]));
			KeU[0] += u[1] * (LM[1] * (6.f * RHO[0][laneId] + -6.f * RHO[2][laneId] + 6.f * RHO[4][laneId] + -6.f * RHO[6][laneId]));
			KeU[2] += u[1] * (LM[2] * (3.f * RHO[0][laneId] + -3.f * RHO[2][laneId] + -3.f * RHO[4][laneId] + 3.f * RHO[6][laneId]));
			KeU[1] += u[1] * (LM[2] * (4.f * RHO[0][laneId] + 4.f * RHO[2][laneId] + 4.f * RHO[4][laneId] + 4.f * RHO[6][laneId]));
			KeU[0] += u[2] * (LM[1] * (6.f * RHO[0][laneId] + 6.f * RHO[2][laneId] + -6.f * RHO[4][laneId] + -6.f * RHO[6][laneId]));
			KeU[2] += u[2] * (LM[2] * (4.f * RHO[0][laneId] + 4.f * RHO[2][laneId] + 4.f * RHO[4][laneId] + 4.f * RHO[6][laneId]));
			KeU[1] += u[2] * (LM[2] * (3.f * RHO[0][laneId] + -3.f * RHO[2][laneId] + -3.f * RHO[4][laneId] + 3.f * RHO[6][laneId]));
#if 1
			u[0] = gU[0][ev[13]]; u[1] = gU[1][ev[13]]; u[2] = gU[2][ev[13]];
			KeU[2] += u[0] * (LM[2] * (6.f * RHO[0][laneId] + -6.f * RHO[1][laneId] + 6.f * RHO[2][laneId] + -6.f * RHO[3][laneId] + -6.f * RHO[4][laneId] + 6.f * RHO[5][laneId] + -6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]));
			KeU[1] += u[0] * (LM[2] * (6.f * RHO[0][laneId] + -6.f * RHO[1][laneId] + -6.f * RHO[2][laneId] + 6.f * RHO[3][laneId] + 6.f * RHO[4][laneId] + -6.f * RHO[5][laneId] + -6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]));
			KeU[0] += u[0] * (LM[3] * (8.f * RHO[0][laneId] + 8.f * RHO[1][laneId] + 8.f * RHO[2][laneId] + 8.f * RHO[3][laneId] + 8.f * RHO[4][laneId] + 8.f * RHO[5][laneId] + 8.f * RHO[6][laneId] + 8.f * RHO[7][laneId]));
			KeU[1] += u[1] * (LM[3] * (8.f * RHO[0][laneId] + 8.f * RHO[1][laneId] + 8.f * RHO[2][laneId] + 8.f * RHO[3][laneId] + 8.f * RHO[4][laneId] + 8.f * RHO[5][laneId] + 8.f * RHO[6][laneId] + 8.f * RHO[7][laneId]));
			KeU[2] += u[1] * (LM[2] * (6.f * RHO[0][laneId] + 6.f * RHO[1][laneId] + -6.f * RHO[2][laneId] + -6.f * RHO[3][laneId] + -6.f * RHO[4][laneId] + -6.f * RHO[5][laneId] + 6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]));
			KeU[0] += u[1] * (LM[2] * (6.f * RHO[0][laneId] + -6.f * RHO[1][laneId] + -6.f * RHO[2][laneId] + 6.f * RHO[3][laneId] + 6.f * RHO[4][laneId] + -6.f * RHO[5][laneId] + -6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]));
			KeU[1] += u[2] * (LM[2] * (6.f * RHO[0][laneId] + 6.f * RHO[1][laneId] + -6.f * RHO[2][laneId] + -6.f * RHO[3][laneId] + -6.f * RHO[4][laneId] + -6.f * RHO[5][laneId] + 6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]));
			KeU[0] += u[2] * (LM[2] * (6.f * RHO[0][laneId] + -6.f * RHO[1][laneId] + 6.f * RHO[2][laneId] + -6.f * RHO[3][laneId] + -6.f * RHO[4][laneId] + 6.f * RHO[5][laneId] + -6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]));
			KeU[2] += u[2] * (LM[3] * (8.f * RHO[0][laneId] + 8.f * RHO[1][laneId] + 8.f * RHO[2][laneId] + 8.f * RHO[3][laneId] + 8.f * RHO[4][laneId] + 8.f * RHO[5][laneId] + 8.f * RHO[6][laneId] + 8.f * RHO[7][laneId]));
#endif
			u[0] = gU[0][ev[14]]; u[1] = gU[1][ev[14]]; u[2] = gU[2][ev[14]];
			KeU[2] += u[0] * (LM[1] * (6.f * RHO[1][laneId] + 6.f * RHO[3][laneId] + -6.f * RHO[5][laneId] + -6.f * RHO[7][laneId]));
			KeU[0] += u[0] * (LM[2] * (-8.f * RHO[1][laneId] + -8.f * RHO[3][laneId] + -8.f * RHO[5][laneId] + -8.f * RHO[7][laneId]));
			KeU[1] += u[0] * (LM[1] * (6.f * RHO[1][laneId] + -6.f * RHO[3][laneId] + 6.f * RHO[5][laneId] + -6.f * RHO[7][laneId]));
			KeU[0] += u[1] * (LM[1] * (-6.f * RHO[1][laneId] + 6.f * RHO[3][laneId] + -6.f * RHO[5][laneId] + 6.f * RHO[7][laneId]));
			KeU[1] += u[1] * (LM[2] * (4.f * RHO[1][laneId] + 4.f * RHO[3][laneId] + 4.f * RHO[5][laneId] + 4.f * RHO[7][laneId]));
			KeU[2] += u[1] * (LM[2] * (3.f * RHO[1][laneId] + -3.f * RHO[3][laneId] + -3.f * RHO[5][laneId] + 3.f * RHO[7][laneId]));
			KeU[0] += u[2] * (LM[1] * (-6.f * RHO[1][laneId] + -6.f * RHO[3][laneId] + 6.f * RHO[5][laneId] + 6.f * RHO[7][laneId]));
			KeU[1] += u[2] * (LM[2] * (3.f * RHO[1][laneId] + -3.f * RHO[3][laneId] + -3.f * RHO[5][laneId] + 3.f * RHO[7][laneId]));
			KeU[2] += u[2] * (LM[2] * (4.f * RHO[1][laneId] + 4.f * RHO[3][laneId] + 4.f * RHO[5][laneId] + 4.f * RHO[7][laneId]));
		}
		else if (warpId == 5) {
			u[0] = gU[0][ev[15]]; u[1] = gU[1][ev[15]]; u[2] = gU[2][ev[15]];
			KeU[0] += u[0] * (LM[4] * (-2.f * RHO[2][laneId] + -2.f * RHO[6][laneId]));
			KeU[2] += u[0] * (LM[1] * (-3.f * RHO[2][laneId] + 3.f * RHO[6][laneId]));
			KeU[1] += u[0] * (LM[2] * (6.f * RHO[2][laneId] + 6.f * RHO[6][laneId]));
			KeU[2] += u[1] * (LM[1] * (3.f * RHO[2][laneId] + -3.f * RHO[6][laneId]));
			KeU[0] += u[1] * (LM[2] * (6.f * RHO[2][laneId] + 6.f * RHO[6][laneId]));
			KeU[1] += u[1] * (LM[4] * (-2.f * RHO[2][laneId] + -2.f * RHO[6][laneId]));
			KeU[2] += u[2] * (LM[0] * (2.f * RHO[2][laneId] + 2.f * RHO[6][laneId]));
			KeU[1] += u[2] * (LM[1] * (-3.f * RHO[2][laneId] + 3.f * RHO[6][laneId]));
			KeU[0] += u[2] * (LM[1] * (3.f * RHO[2][laneId] + -3.f * RHO[6][laneId]));

			u[0] = gU[0][ev[16]]; u[1] = gU[1][ev[16]]; u[2] = gU[2][ev[16]];
			KeU[1] += u[0] * (LM[1] * (-6.f * RHO[2][laneId] + 6.f * RHO[3][laneId] + -6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]));
			KeU[0] += u[0] * (LM[2] * (4.f * RHO[2][laneId] + 4.f * RHO[3][laneId] + 4.f * RHO[6][laneId] + 4.f * RHO[7][laneId]));
			KeU[2] += u[0] * (LM[2] * (3.f * RHO[2][laneId] + -3.f * RHO[3][laneId] + -3.f * RHO[6][laneId] + 3.f * RHO[7][laneId]));
			KeU[2] += u[1] * (LM[1] * (6.f * RHO[2][laneId] + 6.f * RHO[3][laneId] + -6.f * RHO[6][laneId] + -6.f * RHO[7][laneId]));
			KeU[0] += u[1] * (LM[1] * (6.f * RHO[2][laneId] + -6.f * RHO[3][laneId] + 6.f * RHO[6][laneId] + -6.f * RHO[7][laneId]));
			KeU[1] += u[1] * (LM[2] * (-8.f * RHO[2][laneId] + -8.f * RHO[3][laneId] + -8.f * RHO[6][laneId] + -8.f * RHO[7][laneId]));
			KeU[1] += u[2] * (LM[1] * (-6.f * RHO[2][laneId] + -6.f * RHO[3][laneId] + 6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]));
			KeU[2] += u[2] * (LM[2] * (4.f * RHO[2][laneId] + 4.f * RHO[3][laneId] + 4.f * RHO[6][laneId] + 4.f * RHO[7][laneId]));
			KeU[0] += u[2] * (LM[2] * (3.f * RHO[2][laneId] + -3.f * RHO[3][laneId] + -3.f * RHO[6][laneId] + 3.f * RHO[7][laneId]));

			u[0] = gU[0][ev[17]]; u[1] = gU[1][ev[17]]; u[2] = gU[2][ev[17]];
			KeU[1] += u[0] * (LM[2] * (-6.f * RHO[3][laneId] + -6.f * RHO[7][laneId]));
			KeU[2] += u[0] * (LM[1] * (3.f * RHO[3][laneId] + -3.f * RHO[7][laneId]));
			KeU[0] += u[0] * (LM[4] * (-2.f * RHO[3][laneId] + -2.f * RHO[7][laneId]));
			KeU[2] += u[1] * (LM[1] * (3.f * RHO[3][laneId] + -3.f * RHO[7][laneId]));
			KeU[0] += u[1] * (LM[2] * (-6.f * RHO[3][laneId] + -6.f * RHO[7][laneId]));
			KeU[1] += u[1] * (LM[4] * (-2.f * RHO[3][laneId] + -2.f * RHO[7][laneId]));
			KeU[0] += u[2] * (LM[1] * (-3.f * RHO[3][laneId] + 3.f * RHO[7][laneId]));
			KeU[1] += u[2] * (LM[1] * (-3.f * RHO[3][laneId] + 3.f * RHO[7][laneId]));
			KeU[2] += u[2] * (LM[0] * (2.f * RHO[3][laneId] + 2.f * RHO[7][laneId]));
			u[0] = gU[0][ev[18]]; u[1] = gU[1][ev[18]]; u[2] = gU[2][ev[18]];
			KeU[2] += u[0] * (LM[2] * (3.f * RHO[4][laneId]));
			KeU[0] += u[0] * (LM[3] * (-2.f * RHO[4][laneId]));
			KeU[1] += u[0] * (LM[2] * (-3.f * RHO[4][laneId]));
			KeU[0] += u[1] * (LM[2] * (-3.f * RHO[4][laneId]));
			KeU[2] += u[1] * (LM[2] * (3.f * RHO[4][laneId]));
			KeU[1] += u[1] * (LM[3] * (-2.f * RHO[4][laneId]));
			KeU[0] += u[2] * (LM[2] * (3.f * RHO[4][laneId]));
			KeU[1] += u[2] * (LM[2] * (3.f * RHO[4][laneId]));
			KeU[2] += u[2] * (LM[3] * (-2.f * RHO[4][laneId]));
		}
		else if (warpId == 6) {
			u[0] = gU[0][ev[19]]; u[1] = gU[1][ev[19]]; u[2] = gU[2][ev[19]];
			KeU[2] += u[0] * (LM[1] * (-3.f * RHO[4][laneId] + 3.f * RHO[5][laneId]));
			KeU[1] += u[0] * (LM[1] * (3.f * RHO[4][laneId] + -3.f * RHO[5][laneId]));
			KeU[0] += u[0] * (LM[0] * (2.f * RHO[4][laneId] + 2.f * RHO[5][laneId]));
			KeU[2] += u[1] * (LM[2] * (6.f * RHO[4][laneId] + 6.f * RHO[5][laneId]));
			KeU[0] += u[1] * (LM[1] * (-3.f * RHO[4][laneId] + 3.f * RHO[5][laneId]));
			KeU[1] += u[1] * (LM[4] * (-2.f * RHO[4][laneId] + -2.f * RHO[5][laneId]));
			KeU[0] += u[2] * (LM[1] * (3.f * RHO[4][laneId] + -3.f * RHO[5][laneId]));
			KeU[1] += u[2] * (LM[2] * (6.f * RHO[4][laneId] + 6.f * RHO[5][laneId]));
			KeU[2] += u[2] * (LM[4] * (-2.f * RHO[4][laneId] + -2.f * RHO[5][laneId]));

			u[0] = gU[0][ev[20]]; u[1] = gU[1][ev[20]]; u[2] = gU[2][ev[20]];
			KeU[1] += u[0] * (LM[2] * (3.f * RHO[5][laneId]));
			KeU[0] += u[0] * (LM[3] * (-2.f * RHO[5][laneId]));
			KeU[2] += u[0] * (LM[2] * (-3.f * RHO[5][laneId]));
			KeU[2] += u[1] * (LM[2] * (3.f * RHO[5][laneId]));
			KeU[0] += u[1] * (LM[2] * (3.f * RHO[5][laneId]));
			KeU[1] += u[1] * (LM[3] * (-2.f * RHO[5][laneId]));
			KeU[2] += u[2] * (LM[3] * (-2.f * RHO[5][laneId]));
			KeU[0] += u[2] * (LM[2] * (-3.f * RHO[5][laneId]));
			KeU[1] += u[2] * (LM[2] * (3.f * RHO[5][laneId]));

			u[0] = gU[0][ev[21]]; u[1] = gU[1][ev[21]]; u[2] = gU[2][ev[21]];
			KeU[1] += u[0] * (LM[1] * (-3.f * RHO[4][laneId] + 3.f * RHO[6][laneId]));
			KeU[0] += u[0] * (LM[4] * (-2.f * RHO[4][laneId] + -2.f * RHO[6][laneId]));
			KeU[2] += u[0] * (LM[2] * (6.f * RHO[4][laneId] + 6.f * RHO[6][laneId]));
			KeU[0] += u[1] * (LM[1] * (3.f * RHO[4][laneId] + -3.f * RHO[6][laneId]));
			KeU[2] += u[1] * (LM[1] * (-3.f * RHO[4][laneId] + 3.f * RHO[6][laneId]));
			KeU[1] += u[1] * (LM[0] * (2.f * RHO[4][laneId] + 2.f * RHO[6][laneId]));
			KeU[2] += u[2] * (LM[4] * (-2.f * RHO[4][laneId] + -2.f * RHO[6][laneId]));
			KeU[1] += u[2] * (LM[1] * (3.f * RHO[4][laneId] + -3.f * RHO[6][laneId]));
			KeU[0] += u[2] * (LM[2] * (6.f * RHO[4][laneId] + 6.f * RHO[6][laneId]));

			u[0] = gU[0][ev[22]]; u[1] = gU[1][ev[22]]; u[2] = gU[2][ev[22]];
			KeU[2] += u[0] * (LM[1] * (-6.f * RHO[4][laneId] + 6.f * RHO[5][laneId] + -6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]));
			KeU[1] += u[0] * (LM[2] * (3.f * RHO[4][laneId] + -3.f * RHO[5][laneId] + -3.f * RHO[6][laneId] + 3.f * RHO[7][laneId]));
			KeU[0] += u[0] * (LM[2] * (4.f * RHO[4][laneId] + 4.f * RHO[5][laneId] + 4.f * RHO[6][laneId] + 4.f * RHO[7][laneId]));
			KeU[1] += u[1] * (LM[2] * (4.f * RHO[4][laneId] + 4.f * RHO[5][laneId] + 4.f * RHO[6][laneId] + 4.f * RHO[7][laneId]));
			KeU[2] += u[1] * (LM[1] * (-6.f * RHO[4][laneId] + -6.f * RHO[5][laneId] + 6.f * RHO[6][laneId] + 6.f * RHO[7][laneId]));
			KeU[0] += u[1] * (LM[2] * (3.f * RHO[4][laneId] + -3.f * RHO[5][laneId] + -3.f * RHO[6][laneId] + 3.f * RHO[7][laneId]));
			KeU[0] += u[2] * (LM[1] * (6.f * RHO[4][laneId] + -6.f * RHO[5][laneId] + 6.f * RHO[6][laneId] + -6.f * RHO[7][laneId]));
			KeU[1] += u[2] * (LM[1] * (6.f * RHO[4][laneId] + 6.f * RHO[5][laneId] + -6.f * RHO[6][laneId] + -6.f * RHO[7][laneId]));
			KeU[2] += u[2] * (LM[2] * (-8.f * RHO[4][laneId] + -8.f * RHO[5][laneId] + -8.f * RHO[6][laneId] + -8.f * RHO[7][laneId]));
		}
		else if (warpId == 7) {
			u[0] = gU[0][ev[23]]; u[1] = gU[1][ev[23]]; u[2] = gU[2][ev[23]];
			KeU[1] += u[0] * (LM[1] * (3.f * RHO[5][laneId] + -3.f * RHO[7][laneId]));
			KeU[2] += u[0] * (LM[2] * (-6.f * RHO[5][laneId] + -6.f * RHO[7][laneId]));
			KeU[0] += u[0] * (LM[4] * (-2.f * RHO[5][laneId] + -2.f * RHO[7][laneId]));
			KeU[1] += u[1] * (LM[0] * (2.f * RHO[5][laneId] + 2.f * RHO[7][laneId]));
			KeU[2] += u[1] * (LM[1] * (-3.f * RHO[5][laneId] + 3.f * RHO[7][laneId]));
			KeU[0] += u[1] * (LM[1] * (-3.f * RHO[5][laneId] + 3.f * RHO[7][laneId]));
			KeU[1] += u[2] * (LM[1] * (3.f * RHO[5][laneId] + -3.f * RHO[7][laneId]));
			KeU[0] += u[2] * (LM[2] * (-6.f * RHO[5][laneId] + -6.f * RHO[7][laneId]));
			KeU[2] += u[2] * (LM[4] * (-2.f * RHO[5][laneId] + -2.f * RHO[7][laneId]));
			u[0] = gU[0][ev[24]]; u[1] = gU[1][ev[24]]; u[2] = gU[2][ev[24]];
			KeU[0] += u[0] * (LM[3] * (-2.f * RHO[6][laneId]));
			KeU[2] += u[0] * (LM[2] * (3.f * RHO[6][laneId]));
			KeU[1] += u[0] * (LM[2] * (3.f * RHO[6][laneId]));
			KeU[2] += u[1] * (LM[2] * (-3.f * RHO[6][laneId]));
			KeU[1] += u[1] * (LM[3] * (-2.f * RHO[6][laneId]));
			KeU[0] += u[1] * (LM[2] * (3.f * RHO[6][laneId]));
			KeU[0] += u[2] * (LM[2] * (3.f * RHO[6][laneId]));
			KeU[2] += u[2] * (LM[3] * (-2.f * RHO[6][laneId]));
			KeU[1] += u[2] * (LM[2] * (-3.f * RHO[6][laneId]));
			u[0] = gU[0][ev[25]]; u[1] = gU[1][ev[25]]; u[2] = gU[2][ev[25]];
			KeU[0] += u[0] * (LM[0] * (2.f * RHO[6][laneId] + 2.f * RHO[7][laneId]));
			KeU[1] += u[0] * (LM[1] * (-3.f * RHO[6][laneId] + 3.f * RHO[7][laneId]));
			KeU[2] += u[0] * (LM[1] * (-3.f * RHO[6][laneId] + 3.f * RHO[7][laneId]));
			KeU[2] += u[1] * (LM[2] * (-6.f * RHO[6][laneId] + -6.f * RHO[7][laneId]));
			KeU[0] += u[1] * (LM[1] * (3.f * RHO[6][laneId] + -3.f * RHO[7][laneId]));
			KeU[1] += u[1] * (LM[4] * (-2.f * RHO[6][laneId] + -2.f * RHO[7][laneId]));
			KeU[1] += u[2] * (LM[2] * (-6.f * RHO[6][laneId] + -6.f * RHO[7][laneId]));
			KeU[0] += u[2] * (LM[1] * (3.f * RHO[6][laneId] + -3.f * RHO[7][laneId]));
			KeU[2] += u[2] * (LM[4] * (-2.f * RHO[6][laneId] + -2.f * RHO[7][laneId]));
			u[0] = gU[0][ev[26]]; u[1] = gU[1][ev[26]]; u[2] = gU[2][ev[26]];
			KeU[0] += u[0] * (LM[3] * (-2.f * RHO[7][laneId]));
			KeU[1] += u[0] * (LM[2] * (-3.f * RHO[7][laneId]));
			KeU[2] += u[0] * (LM[2] * (-3.f * RHO[7][laneId]));
			KeU[1] += u[1] * (LM[3] * (-2.f * RHO[7][laneId]));
			KeU[2] += u[1] * (LM[2] * (-3.f * RHO[7][laneId]));
			KeU[0] += u[1] * (LM[2] * (-3.f * RHO[7][laneId]));
			KeU[1] += u[2] * (LM[2] * (-3.f * RHO[7][laneId]));
			KeU[2] += u[2] * (LM[3] * (-2.f * RHO[7][laneId]));
			KeU[0] += u[2] * (LM[2] * (-3.f * RHO[7][laneId]));
		}
	}

#if 1

	if (warpId >= 4) {
		for (int i = 0; i < 3; i++) sumKeU[i][warpId - 4][laneId] = KeU[i];
	}
	__syncthreads();
	if (warpId < 4) {
		for (int i = 0; i < 3; i++) sumKeU[i][warpId][laneId] += KeU[i];
	}
	__syncthreads();
	if (warpId < 2) {
		for (int i = 0; i < 3; i++) sumKeU[i][warpId][laneId] += sumKeU[i][warpId + 2][laneId];
	}
	__syncthreads();
	if (warpId < 1 && !fiction) {
		for (int i = 0; i < 3; i++) KeU[i] = sumKeU[i][warpId][laneId] + sumKeU[i][warpId + 1][laneId];

		float r[3];
		r[0] = gF[0][ev[13]] - KeU[0]; r[1] = gF[1][ev[13]] - KeU[1]; r[2] = gF[2][ev[13]] - KeU[2];

		if (vflag.is_dirichlet_boundary()) { r[0] = 0; r[1] = 0; r[2] = 0; }

		gR[0][ev[13]] = r[0]; gR[1][ev[13]] = r[1]; gR[2][ev[13]] = r[2];
	}
#endif
}

void homo::Grid::update_residual_ex(void)
{
	useGrid_g();
	devArray_t<int, 3> gridCellReso{ cellReso[0],cellReso[1],cellReso[2] };
	VertexFlags* vflags = vertflag;
	CellFlags* eflags = cellflag;
	if (assemb_otf) {
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, n_gsvertices() * 8, 32 * 8);
		update_residual_otf_kernel_opt << <grid_size, block_size >> > (n_gsvertices(), rho_g, gridCellReso,
			vflags, eflags, diag_strength);
		cudaDeviceSynchronize();
		cuda_error_check;
	}
	pad_vertex_data(r_g);
}

// gather per fine element matrix to coarse stencil, one thread for one coarse vertex 
// stencil was organized in lexico order(No padding), and should be transferred to gs order
//template<int BlockSize = 256>
__global__ void restrict_stencil_otf_aos_kernel_1(
	int nv, float* rholist, CellFlags* eflags, VertexFlags* vflags
) {
	//__shared__ glm::mat<3, 3, double> KE[8][8];
	__shared__ glm::mat3 KE[8][8];
	__shared__ int coarseReso[3];
	__shared__ int fineReso[3];
	
	if (threadIdx.x < 3) { 
		coarseReso[threadIdx.x] = gGridCellReso[threadIdx.x]; 
		fineReso[threadIdx.x] = coarseReso[threadIdx.x] * gUpCoarse[threadIdx.x];
	}
	loadTemplateMatrix(KE);

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int coarseRatio[3] = { gUpCoarse[0], gUpCoarse[1], gUpCoarse[2] };
	int vipos[3] = {
		tid % (coarseReso[0] + 1),
		tid / (coarseReso[0] + 1) % (coarseReso[1] + 1),
		tid / ((coarseReso[0] + 1) * (coarseReso[1] + 1)) };
	//size_t vid = lexi2gs(vipos, gGsVertexReso, gGsVertexEnd);
	size_t vid = tid;

	//bool debug = vid == 63;
	bool debug = false;

	if (vid >= nv) return;

	vipos[0] *= coarseRatio[0]; vipos[1] *= coarseRatio[1]; vipos[2] *= coarseRatio[2];

	float pr = coarseRatio[0] * coarseRatio[1] * coarseRatio[2];

	if (debug) { printf("vipos = (%d, %d, %d)\n", vipos[0], vipos[1], vipos[2]); }

	for (int vj = 0; vj < 27; vj++) {
		int coarse_vj_off[3] = {
			coarseRatio[0] * (vj % 3 -1),
			coarseRatio[1] * (vj / 3 % 3 -1),
			coarseRatio[2] * (vj / 9 - 1)
		};
		//glm::mat<3, 3, double> st(0.f);
		glm::mat3 st(0.f);
		if (debug) { printf("coarse_vj_off = (%d, %d, %d)\n", coarse_vj_off[0], coarse_vj_off[1], coarse_vj_off[2]); }
		for (int xfine_off = -coarseRatio[0]; xfine_off < coarseRatio[0]; xfine_off++) {
			for (int yfine_off = -coarseRatio[1]; yfine_off < coarseRatio[1]; yfine_off++) {
				for (int zfine_off = -coarseRatio[2]; zfine_off < coarseRatio[2]; zfine_off++) {
					int e_fine_off[3] = {
						coarse_vj_off[0] + xfine_off,
						coarse_vj_off[1] + yfine_off,
						coarse_vj_off[2] + zfine_off,
					};
					// exclude elements out of neighborhood
					if (e_fine_off[0] < -coarseRatio[0] || e_fine_off[0] >= coarseRatio[0] ||
						e_fine_off[1] < -coarseRatio[1] || e_fine_off[1] >= coarseRatio[1] ||
						e_fine_off[2] < -coarseRatio[2] || e_fine_off[2] >= coarseRatio[2]) {
						continue;
					};
					if (debug) { printf(" e_fine_off = (%d, %d, %d)\n", e_fine_off[0], e_fine_off[1], e_fine_off[2]); }
					int e_fine_pos[3] = {
						vipos[0] + e_fine_off[0], vipos[1] + e_fine_off[1], vipos[2] + e_fine_off[2]
					};
					// exclude padded element
					if (e_fine_pos[0] < 0 || e_fine_pos[0] >= fineReso[0] ||
						e_fine_pos[1] < 0 || e_fine_pos[1] >= fineReso[1] ||
						e_fine_pos[2] < 0 || e_fine_pos[2] >= fineReso[2]) {
						continue;
					}
					int eid = lexi2gs(e_fine_pos, gGsFineCellReso, gGsFineCellEnd);
					//auto eflag = eflags[eid];
					float rho_penal = powf(rholist[eid], exp_penal[0]);
					if (debug) { printf(" e_fine_pos = (%d, %d, %d), eid = %d, rhopenal = %f\n", e_fine_pos[0], e_fine_pos[1], e_fine_pos[2], eid, rho_penal); }
					for (int e_vi = 0; e_vi < 8; e_vi++) {
						int e_vi_fine_off[3] = {
							e_fine_off[0] + e_vi % 2,
							e_fine_off[1] + e_vi / 2 % 2,
							e_fine_off[2] + e_vi / 4
						};
						if (!inStrictBound(e_vi_fine_off, coarseRatio)) continue;
						float wi = (coarseRatio[0] - abs(e_vi_fine_off[0])) *
							(coarseRatio[1] - abs(e_vi_fine_off[1])) *
							(coarseRatio[2] - abs(e_vi_fine_off[2])) / pr;
						if (debug) printf("   e_vi_off = (%d, %d, %d), wi = %f\n", e_vi_fine_off[0], e_vi_fine_off[1], e_vi_fine_off[2], wi);
						wi *= rho_penal;
						for (int e_vj = 0; e_vj < 8; e_vj++) {
							int vij_off[3] = {
								abs(e_fine_off[0] + e_vj % 2 - coarse_vj_off[0]),
								abs(e_fine_off[1] + e_vj / 2 % 2 - coarse_vj_off[1]),
								abs(e_fine_off[2] + e_vj / 4 - coarse_vj_off[2])
							};
							if (vij_off[0] >= coarseRatio[0] || vij_off[1] >= coarseRatio[1] ||
								vij_off[2] >= coarseRatio[2]) {
								continue;
							}
							float wj = (coarseRatio[0] - vij_off[0]) *
								(coarseRatio[1] - vij_off[1]) *
								(coarseRatio[2] - vij_off[2]) / pr;
							if (debug) printf("    vij_off = (%d, %d, %d), wi = %f\n", vij_off[0], vij_off[1], vij_off[2], wj);
							st += (wi * wj) * KE[e_vi][e_vj];
						}
					}
				}
			}
		}
		rxstencil[vj][vid] = st;
	}
}

// one thread of one coarse vertex
//template<int BlockSize = 256>
__global__ void restrict_stencil_aos_kernel_1(
	int nv_coarse, int nv_fine,
	VertexFlags* vflags,
	VertexFlags* vfineflags
) {
	__shared__ int gsVertexEnd[8];
	__shared__ int gsFineVertexEnd[8];
	__shared__ int gsFineVertexReso[3][8];

	if (threadIdx.x < 24) {
		gsFineVertexReso[threadIdx.x / 8][threadIdx.x % 8] = gGsFineVertexReso[threadIdx.x / 8][threadIdx.x % 8];
	}
	if (threadIdx.x < 8) {
		gsVertexEnd[threadIdx.x] = gGsVertexEnd[threadIdx.x];
		gsFineVertexEnd[threadIdx.x] = gGsFineVertexEnd[threadIdx.x];
	}
	__syncthreads();

	bool fiction = false;
	int laneId = threadIdx.x % 32;
	int warpId = threadIdx.x / 32;
	//size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	//size_t vid = blockIdx.x * 32 + laneId;
	size_t vid = blockIdx.x * blockDim.x + threadIdx.x;

	if (vid >= nv_coarse) fiction = true;

	VertexFlags vflag;
	if (!fiction) { 
		vflag = vflags[vid]; 
		fiction = vflag.is_fiction();
	}

	int coarseRatio[3] = { gUpCoarse[0], gUpCoarse[1], gUpCoarse[2] };

	float pr = coarseRatio[0] * coarseRatio[1] * coarseRatio[2];

	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);
	indexer.locate(vid, vflag.get_gscolor(), gsVertexEnd);

	bool nondyadic = coarseRatio[0] > 2 || coarseRatio[1] > 2 || coarseRatio[2] > 2;

	if (!fiction && !vflag.is_period_padding()) {
		for (int i = 0; i < 27; i++) {
			int coarse_vj_off[3] = {
				coarseRatio[0] * (i % 3 - 1),
				coarseRatio[1] * (i / 3 % 3 - 1),
				coarseRatio[2] * (i / 9 - 1)
			};
			glm::mat3 st(0.f);
			for (int xfine_off = -coarseRatio[0]; xfine_off <= coarseRatio[0]; xfine_off++) {
				for (int yfine_off = -coarseRatio[1]; yfine_off <= coarseRatio[1]; yfine_off++) {
					for (int zfine_off = -coarseRatio[2]; zfine_off <= coarseRatio[2]; zfine_off++) {
						int vi_fine_off[3] = {
							xfine_off + coarse_vj_off[0],
							yfine_off + coarse_vj_off[1],
							zfine_off + coarse_vj_off[2]
						};
						if (!inStrictBound(vi_fine_off, coarseRatio)) continue;
						int vi_neighId;
						if (nondyadic) {
							vi_neighId = indexer.neighFineVertex(vi_fine_off, coarseRatio, gsFineVertexEnd, gsFineVertexReso, true).getId();
						} else {
							vi_neighId = indexer.neighFineVertex(vi_fine_off, coarseRatio, gsFineVertexEnd, gsFineVertexReso, false).getId();
						}
						float wi = (coarseRatio[0] - abs(vi_fine_off[0])) *
							(coarseRatio[1] - abs(vi_fine_off[1])) *
							(coarseRatio[2] - abs(vi_fine_off[2])) / pr;
						for (int vj_offid = 0; vj_offid < 27; vj_offid++) {
							int vij_off[3] = {
								abs(vi_fine_off[0] + vj_offid % 3 - 1 - coarse_vj_off[0]),
								abs(vi_fine_off[1] + vj_offid / 3 % 3 - 1 - coarse_vj_off[1]),
								abs(vi_fine_off[2] + vj_offid / 9 - 1 - coarse_vj_off[2]) 
							};
							if (vij_off[0] >= coarseRatio[0] || vij_off[1] >= coarseRatio[1] || vij_off[2] >= coarseRatio[2]) {
								continue;
							}
							float wj = (coarseRatio[0] - vij_off[0]) * 
								(coarseRatio[1] - vij_off[1]) * 
								(coarseRatio[2] - vij_off[2]) / pr;
							st += wi * wj * rxFineStencil[vj_offid][vi_neighId];
						}
					}
				}
			}
			rxstencil[i][vid] = st;
		}
	}
}
