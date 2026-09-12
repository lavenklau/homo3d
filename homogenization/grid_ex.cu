#include "grid.h"
#include "homoCommon.cuh"

using namespace homo;

__device__ bool inStrictBound(int pi[3], int cover[3]) {
	return pi[0] > -cover[0] && pi[0] < cover[0] &&
		   pi[1] > -cover[1] && pi[1] < cover[1] &&
		   pi[2] > -cover[2] && pi[2] < cover[2];
}

// gather per fine element matrix to coarse stencil, one thread for one coarse vertex
// stencil was organized in lexico order(No padding), and should be transferred to gs order
//template<int BlockSize = 256>
template<typename T>
__global__ void restrict_stencil_otf_aos_kernel_1(
	int nv, T* rholist, CellFlags* eflags, VertexFlags* vflags) {
	//__shared__ glm::mat<3, 3, double> KE[8][8];
	__shared__ glm::mat3 KE[8][8];
	__shared__ int coarseReso[3];
	__shared__ int fineReso[3];
	__shared__ int gsFineCellReso[3][8];
	__shared__ int gsFineCellEnd[8];

	if (threadIdx.x < 3) {
		coarseReso[threadIdx.x] = gGridCellReso[threadIdx.x];
		fineReso[threadIdx.x] = coarseReso[threadIdx.x] * gUpCoarse[threadIdx.x];
	}
	if (threadIdx.x < 8) {
		for (int i = 0; i < 3; i++)
			gsFineCellReso[i][threadIdx.x] = gGsFineCellReso[i][threadIdx.x];
		gsFineCellEnd[threadIdx.x] = gGsFineCellEnd[threadIdx.x];
	}

	loadTemplateMatrix(KE);

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int coarseRatio[3] = {gUpCoarse[0], gUpCoarse[1], gUpCoarse[2]};
	int vipos[3] = {
		tid % (coarseReso[0] + 1),
		tid / (coarseReso[0] + 1) % (coarseReso[1] + 1),
		tid / ((coarseReso[0] + 1) * (coarseReso[1] + 1))};
	//size_t vid = lexi2gs(vipos, gGsVertexReso, gGsVertexEnd);
	size_t vid = tid;

	//bool debug = vid == 63;
	bool debug = false;

	if (vid >= nv)
		return;

	vipos[0] *= coarseRatio[0];
	vipos[1] *= coarseRatio[1];
	vipos[2] *= coarseRatio[2];

	float pr = coarseRatio[0] * coarseRatio[1] * coarseRatio[2];

	if (debug) {
		printf("vipos = (%d, %d, %d)\n", vipos[0], vipos[1], vipos[2]);
	}

	for (int vj = 0; vj < 27; vj++) {
		int coarse_vj_off[3] = {
			coarseRatio[0] * (vj % 3 - 1),
			coarseRatio[1] * (vj / 3 % 3 - 1),
			coarseRatio[2] * (vj / 9 - 1)};
		//glm::mat<3, 3, double> st(0.f);
		glm::mat3 st(0.f);
		if (debug) {
			printf("coarse_vj_off = (%d, %d, %d)\n", coarse_vj_off[0], coarse_vj_off[1], coarse_vj_off[2]);
		}
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
					if (debug) {
						printf(" e_fine_off = (%d, %d, %d)\n", e_fine_off[0], e_fine_off[1], e_fine_off[2]);
					}
					int e_fine_pos[3] = {
						vipos[0] + e_fine_off[0], vipos[1] + e_fine_off[1], vipos[2] + e_fine_off[2]};
					// exclude padded element
					if (e_fine_pos[0] < 0 || e_fine_pos[0] >= fineReso[0] ||
						e_fine_pos[1] < 0 || e_fine_pos[1] >= fineReso[1] ||
						e_fine_pos[2] < 0 || e_fine_pos[2] >= fineReso[2]) {
						continue;
					}
					int eid = lexi2gs(e_fine_pos, gsFineCellReso, gsFineCellEnd);
					//auto eflag = eflags[eid];
					float rho_penal = powf(float(rholist[eid]), exp_penal[0]);
					if (debug) {
						printf(" e_fine_pos = (%d, %d, %d), eid = %d, rhopenal = %f\n", e_fine_pos[0], e_fine_pos[1], e_fine_pos[2], eid, rho_penal);
					}
					for (int e_vi = 0; e_vi < 8; e_vi++) {
						int e_vi_fine_off[3] = {
							e_fine_off[0] + e_vi % 2,
							e_fine_off[1] + e_vi / 2 % 2,
							e_fine_off[2] + e_vi / 4};
						if (!inStrictBound(e_vi_fine_off, coarseRatio))
							continue;
						float wi = (coarseRatio[0] - abs(e_vi_fine_off[0])) *
								   (coarseRatio[1] - abs(e_vi_fine_off[1])) *
								   (coarseRatio[2] - abs(e_vi_fine_off[2])) / pr;
						if (debug)
							printf("   e_vi_off = (%d, %d, %d), wi = %f\n", e_vi_fine_off[0], e_vi_fine_off[1], e_vi_fine_off[2], wi);
						wi *= rho_penal;
						for (int e_vj = 0; e_vj < 8; e_vj++) {
							int vij_off[3] = {
								abs(e_fine_off[0] + e_vj % 2 - coarse_vj_off[0]),
								abs(e_fine_off[1] + e_vj / 2 % 2 - coarse_vj_off[1]),
								abs(e_fine_off[2] + e_vj / 4 - coarse_vj_off[2])};
							if (vij_off[0] >= coarseRatio[0] || vij_off[1] >= coarseRatio[1] ||
								vij_off[2] >= coarseRatio[2]) {
								continue;
							}
							float wj = (coarseRatio[0] - vij_off[0]) *
									   (coarseRatio[1] - vij_off[1]) *
									   (coarseRatio[2] - vij_off[2]) / pr;
							if (debug)
								printf("    vij_off = (%d, %d, %d), wi = %f\n", vij_off[0], vij_off[1], vij_off[2], wj);
							st += (wi * wj) * KE[e_vi][e_vj];
						}
					}
				}
			}
		}
		if (vj == 13) {
			for (int k = 0; k < 3; k++) {
				if (abs(st[k][k]) < 1e-4) {
					st[k][k] = 1e-4;
				}
			}
		}
		rxstencil[vj][vid] = st;
		// if (vj == 13)
		// {
		// 	printf(" [%d] stencil diag %4.2e, %4.2e, %4.2e\n", vid, st[0][0], st[1][1], st[2][2]);
		// }
	}
}

// one thread of one coarse vertex
//template<int BlockSize = 256>
__global__ void restrict_stencil_aos_kernel_1(
	int nv_coarse, int nv_fine,
	VertexFlags* vflags,
	VertexFlags* vfineflags) {
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

	if (vid >= nv_coarse)
		fiction = true;

	VertexFlags vflag;
	if (!fiction) {
		vflag = vflags[vid];
		fiction = vflag.is_fiction();
	}

	int coarseRatio[3] = {gUpCoarse[0], gUpCoarse[1], gUpCoarse[2]};

	float pr = coarseRatio[0] * coarseRatio[1] * coarseRatio[2];

	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);
	indexer.locate(vid, vflag.get_gscolor(), gsVertexEnd);

	bool nondyadic = coarseRatio[0] > 2 || coarseRatio[1] > 2 || coarseRatio[2] > 2;

	if (!fiction && !vflag.is_period_padding()) {
		for (int i = 0; i < 27; i++) {
			int coarse_vj_off[3] = {
				coarseRatio[0] * (i % 3 - 1),
				coarseRatio[1] * (i / 3 % 3 - 1),
				coarseRatio[2] * (i / 9 - 1)};
			glm::mat3 st(0.f);
			for (int xfine_off = -coarseRatio[0]; xfine_off <= coarseRatio[0]; xfine_off++) {
				for (int yfine_off = -coarseRatio[1]; yfine_off <= coarseRatio[1]; yfine_off++) {
					for (int zfine_off = -coarseRatio[2]; zfine_off <= coarseRatio[2]; zfine_off++) {
						int vi_fine_off[3] = {
							xfine_off + coarse_vj_off[0],
							yfine_off + coarse_vj_off[1],
							zfine_off + coarse_vj_off[2]};
						if (!inStrictBound(vi_fine_off, coarseRatio))
							continue;
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
								abs(vi_fine_off[2] + vj_offid / 9 - 1 - coarse_vj_off[2])};
							if (vij_off[0] >= coarseRatio[0] || vij_off[1] >= coarseRatio[1] || vij_off[2] >= coarseRatio[2]) {
								continue;
							}
							float wj = (coarseRatio[0] - vij_off[0]) *
									   (coarseRatio[1] - vij_off[1]) *
									   (coarseRatio[2] - vij_off[2]) / pr;
							st += wi * wj * glm::mat3(rxFineStencil[vj_offid][vi_neighId]);
						}
					}
				}
			}
			if (i == 13) {
				for (int k = 0; k < 3; k++) {
					if (abs(st[k][k]) < 1e-4) {
						st[k][k] = 1e-4;
					}
				}
			}
			rxstencil[i][vid] = st;
		}
	}
}

template __global__ void restrict_stencil_otf_aos_kernel_1<half>(int nv, half* rholist, CellFlags* eflags, VertexFlags* vflags);
template __global__ void restrict_stencil_otf_aos_kernel_1<float>(int nv, float* rholist, CellFlags* eflags, VertexFlags* vflags);