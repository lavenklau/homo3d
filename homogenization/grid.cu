#include "grid.h"
//#define __CUDACC__
#include "culib/lib.cuh"
#include <vector>
#include "utils.h"
#include <fstream>
#include "homoCommon.cuh"

#define DIRICHLET_STRENGTH 1e3

//#define DIAG_STRENGTH 1e6
//#define DIAG_STRENGTH 0

using namespace homo;
using namespace culib;

__constant__ VT* gU[3];
__constant__ VT* gF[3];
__constant__ VT* gR[3];
__constant__ VT* gUfine[3];
__constant__ VT* gFfine[3];
__constant__ VT* gRfine[3];
__constant__ VT* gUcoarse[3];
__constant__ VT* gFcoarse[3];
__constant__ VT* gRcoarse[3];
__constant__ float gKE[24][24];
__constant__ double gKEd[24][24];
__constant__ Lame gKLame[24][24];
__constant__ float gKELame[24][24];
__constant__ float gKEMu[24][24];
//__constant__ float* rxstencil[27][9];
__constant__ glm::hmat3* rxstencil[27];
//__constant__ float* rxCoarseStencil[27][9];
__constant__ glm::hmat3* rxCoarseStencil[27];
//__constant__ float* rxFineStencil[27][9];
__constant__ glm::hmat3* rxFineStencil[27];
__constant__ double gLM[5];

__constant__ int gUpCoarse[3];
__constant__ int gDownCoarse[3];
__constant__ int gGridCellReso[3];
__constant__ int gCoarseGridCellReso[3];
__constant__ int gGsCellReso[3][8];
__constant__ int gGsVertexReso[3][8];
__constant__ int gGsVertexEnd[8];
__constant__ int gGsCellEnd[8];
__constant__ int gGsFineVertexReso[3][8];
__constant__ int gGsCoarseVertexReso[3][8];
__constant__ int gGsFineCellReso[3][8];
__constant__ int gGsCoarseCellReso[3][8];
__constant__ int gGsCoarseVertexEnd[8];
__constant__ int gGsFineVertexEnd[8];
__constant__ int gGsFineCellEnd[8];

//__constant__ double* guchar[6][3];
//__constant__ double* gfchar[6][3];
__constant__ float* guchar[3];

__constant__ float exp_penal[1];
__constant__ float LAM[1];
__constant__ float MU[1];

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

__device__ void gsid2pos(int gsid, int color, int gsreso[3][8], int gsend[8], int pos[3]) {
	int setid = gsid - (color == 0 ? 0 : gsend[color - 1]);
	int gsorg[3] = {color % 2, color / 2 % 2, color / 4};
	int gspos[3] = {setid % gsreso[0][color], setid / gsreso[0][color] % gsreso[1][color], setid / (gsreso[0][color] * gsreso[1][color])};
	for (int i = 0; i < 3; i++) {
		pos[i] = gspos[i] * 2 + gsorg[i] - 1;
	}
}

void homo::Grid::useGrid_g(void) {
	cudaMemcpyToSymbol(gU, u_g, sizeof(gU));
	cudaMemcpyToSymbol(gF, f_g, sizeof(gF));
	cudaMemcpyToSymbol(gR, r_g, sizeof(gR));
	if (fine != nullptr) {
		cudaMemcpyToSymbol(gUfine, fine->u_g, sizeof(gUfine));
		cudaMemcpyToSymbol(gFfine, fine->f_g, sizeof(gFfine));
		cudaMemcpyToSymbol(gRfine, fine->r_g, sizeof(gRfine));

		cudaMemcpyToSymbol(rxFineStencil, fine->stencil_g, sizeof(rxFineStencil));
		cudaMemcpyToSymbol(gUpCoarse, upCoarse.data(), sizeof(gUpCoarse));
		cudaMemcpyToSymbol(gGsFineVertexEnd, fine->gsVertexSetEnd, sizeof(gGsFineVertexEnd));
		cudaMemcpyToSymbol(gGsFineCellEnd, fine->gsCellSetEnd, sizeof(gGsFineCellEnd));
		cudaMemcpyToSymbol(gGsFineVertexReso, fine->gsVertexReso, sizeof(gGsFineVertexReso));
		cudaMemcpyToSymbol(gGsFineCellReso, fine->gsCellReso, sizeof(gGsFineCellReso));
	}
	if (Coarse != nullptr) {
		cudaMemcpyToSymbol(gUcoarse, Coarse->u_g, sizeof(gUcoarse));
		cudaMemcpyToSymbol(gFcoarse, Coarse->f_g, sizeof(gFcoarse));
		cudaMemcpyToSymbol(gRcoarse, Coarse->r_g, sizeof(gRcoarse));

		cudaMemcpyToSymbol(rxCoarseStencil, Coarse->stencil_g, sizeof(rxCoarseStencil));
		cudaMemcpyToSymbol(gDownCoarse, downCoarse.data(), sizeof(gDownCoarse));
		cudaMemcpyToSymbol(gGsCoarseVertexEnd, Coarse->gsVertexSetEnd, sizeof(gGsCoarseVertexEnd));
		cudaMemcpyToSymbol(gGsCoarseVertexReso, Coarse->gsVertexReso, sizeof(gGsCoarseVertexReso));
		cudaMemcpyToSymbol(gCoarseGridCellReso, Coarse->cellReso.data(), sizeof(gCoarseGridCellReso));
		cudaMemcpyToSymbol(gGsCoarseCellReso, Coarse->gsCellReso, sizeof(gGsCoarseCellReso));
	}
	cudaMemcpyToSymbol(rxstencil, stencil_g, sizeof(rxstencil));
	cudaMemcpyToSymbol(gGridCellReso, cellReso.data(), sizeof(gGridCellReso));
	cudaMemcpyToSymbol(gGsVertexEnd, gsVertexSetEnd, sizeof(gGsVertexEnd));
	cudaMemcpyToSymbol(gGsCellEnd, gsCellSetEnd, sizeof(gGsCellEnd));
	cudaMemcpyToSymbol(gGsVertexReso, gsVertexReso, sizeof(gGsVertexReso));

	if (is_root) {
		// cudaMemcpyToSymbol(guchar, uchar_g, sizeof(guchar));
		//cudaMemcpyToSymbol(gfchar, fchar_g, sizeof(gfchar));
		cudaMemcpyToSymbol(gGsCellReso, gsCellReso, sizeof(gGsCellReso));
	}
}

__global__ void setVertexFlags_kernel(
	int nv, VertexFlags* pflag,
	devArray_t<int, 3> cellReso,
	devArray_t<int, 8> vGSend, devArray_t<int, 8> vGSvalid) {
	size_t vid = blockIdx.x * blockDim.x + threadIdx.x;
	if (vid >= nv)
		return;

	VertexFlags flag = pflag[vid];

	int set_id = -1;
	for (int i = 0; i < 8; i++) {
		if (vid < vGSend[i]) {
			set_id = i;
			break;
		}
	}

	if (set_id == -1) {
		flag.set_fiction();
		pflag[vid] = flag;
		return;
	}

	flag.set_gscolor(set_id);

	int gsid = vid - (set_id >= 1 ? vGSend[set_id - 1] : 0);
	do {
		// check if GS padding
		if (gsid >= vGSvalid[set_id]) {
			flag.set_fiction();
			break;
		}

		// check if periodic boundary padding
		int org[3] = {set_id % 2, set_id / 2 % 2, set_id / 4};
		int gsvreso[3] = {};
		for (int i = 0; i < 3; i++)
			gsvreso[i] = (cellReso[i] + 2 - org[i]) / 2 + 1;

		int gspos[3] = {gsid % gsvreso[0], gsid / gsvreso[0] % gsvreso[1], gsid / (gsvreso[0] * gsvreso[1])};
		int pos[3] = {gspos[0] * 2 + org[0], gspos[1] * 2 + org[1], gspos[2] * 2 + org[2]};

		// check if dirichlet boundary
		if ((pos[0] - 1) % cellReso[0] == 0 &&
			(pos[1] - 1) % cellReso[1] == 0 &&
			(pos[2] - 1) % cellReso[2] == 0) {
			flag.set_dirichlet_boundary();
		}

		// is left padding
		if (pos[0] == 0 || pos[1] == 0 || pos[2] == 0) {
			//flag.set_fiction();
			flag.set_period_padding();
		}

		// is right padding
		if (pos[0] == cellReso[0] + 2 || pos[1] == cellReso[1] + 2 || pos[2] == cellReso[2] + 2) {
			//flag.set_fiction();
			flag.set_period_padding();
		}

		// is boundary
		if (pos[0] == 1) {
			flag.set_boundary(LEFT_BOUNDARY);
		}
		if (pos[1] == 1) {
			flag.set_boundary(NEAR_BOUNDARY);
		}
		if (pos[2] == 1) {
			flag.set_boundary(DOWN_BOUNDARY);
		}
		if (pos[0] == cellReso[0] + 1) {
			flag.set_boundary(RIGHT_BOUNDARY);
		}
		if (pos[1] == cellReso[1] + 1) {
			flag.set_boundary(FAR_BOUNDARY);
		}
		if (pos[2] == cellReso[2] + 1) {
			flag.set_boundary(UP_BOUNDARY);
		}
	} while (0);

	pflag[vid] = flag;
}

__global__ void setCellFlags_kernel(
	int nc, CellFlags* pflag,
	devArray_t<int, 3> cellReso,
	devArray_t<int, 8> vGSend, devArray_t<int, 8> vGSvalid) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nc)
		return;

	CellFlags flag = pflag[tid];

	int set_id = -1;
	for (int i = 0; i < 8; i++) {
		if (tid < vGSend[i]) {
			set_id = i;
			break;
		}
	}

	if (set_id == -1)
		return;

	flag.set_gscolor(set_id);

	int gsid = tid - (set_id - 1 >= 0 ? vGSend[set_id - 1] : 0);

	do {
		// check if GS padding
		if (gsid >= vGSvalid[set_id]) {
			flag.set_fiction();
			break;
		}

		// check if periodic boundary padding
		int org[3] = {set_id % 2, set_id / 2 % 2, set_id / 4};
		int gsreso[3] = {};
		for (int i = 0; i < 3; i++)
			gsreso[i] = (cellReso[i] + 1 - org[i]) / 2 + 1;

		int gspos[3] = {gsid % gsreso[0], gsid / gsreso[0] % gsreso[1], gsid / (gsreso[0] * gsreso[1])};
		int pos[3] = {gspos[0] * 2 + org[0], gspos[1] * 2 + org[1], gspos[2] * 2 + org[2]};

		// check if dirichlet boundary
		if (pos[0] == 1 && pos[1] == 1 && pos[2] == 1) {
			flag.set_dirichlet_boundary();
		}

		// is left padding
		if (pos[0] == 0 || pos[1] == 0 || pos[2] == 0) {
			//flag.set_fiction();
			//if ((pos[0] == 0) + (pos[1] == 0) + (pos[2] == 0) == 1) {
			flag.set_period_padding();
			//}
		}

		// is right padding
		if (pos[0] == cellReso[0] + 1 || pos[1] == cellReso[1] + 1 || pos[2] == cellReso[2] + 1) {
			//flag.set_fiction();
			//if ((pos[0] == cellReso[0] + 1) + (pos[1] == cellReso[1] + 1) + (pos[2] == cellReso[1] + 1) == 1) {
			flag.set_period_padding();
			//}
		}

		// is boundary
		if (pos[0] == 1) {
			flag.set_boundary(LEFT_BOUNDARY);
		}
		if (pos[1] == 1) {
			flag.set_boundary(NEAR_BOUNDARY);
		}
		if (pos[2] == 1) {
			flag.set_boundary(DOWN_BOUNDARY);
		}
		if (pos[0] == cellReso[0]) {
			flag.set_boundary(RIGHT_BOUNDARY);
		}
		if (pos[1] == cellReso[1]) {
			flag.set_boundary(FAR_BOUNDARY);
		}
		if (pos[2] == cellReso[2]) {
			flag.set_boundary(UP_BOUNDARY);
		}
	} while (0);

	pflag[tid] = flag;
}

void homo::Grid::setFlags_g(void) {
	VertexFlags* vflag = vertflag;
	CellFlags* eflag = cellflag;
	cudaMemset(vflag, 0, sizeof(VertexFlags) * n_gsvertices());
	cudaMemset(eflag, 0, sizeof(CellFlags) * n_gscells());
	cuda_error_check;
	devArray_t<int, 3> ereso{cellReso[0], cellReso[1], cellReso[2]};
	devArray_t<int, 8> vGsend, vGsvalid;
	devArray_t<int, 8> eGsend, eGsvalid;
	for (int i = 0; i < 8; i++) {
		vGsend[i] = gsVertexSetEnd[i];
		vGsvalid[i] = gsVertexSetValid[i];

		eGsend[i] = gsCellSetEnd[i];
		eGsvalid[i] = gsCellSetValid[i];
	}
	// set vertex flags
	auto cfg = make_kernel_param(n_gsvertices(), 512);
	setVertexFlags_kernel<<<cfg.grid, cfg.block>>>(n_gsvertices(), vflag, ereso, vGsend, vGsvalid);
	cudaDeviceSynchronize();
	cuda_error_check;

	// set cell flags
	cfg = make_kernel_param(n_gscells(), 512);
	setCellFlags_kernel<<<cfg.grid, cfg.block>>>(n_gscells(), eflag, ereso, eGsend, eGsvalid);
	cudaDeviceSynchronize();
	cuda_error_check;
}

// map 32 vertices to 8 warp
template<typename T, int BlockSize = 32 * 8>
__global__ void gs_relaxation_otf_kernel(
	int gs_set, T* rholist,
	devArray_t<int, 3> gridCellReso,
	VertexFlags* vflags, CellFlags* eflags,
	// SOR relaxing factor
	float w = 1.f,
	float diag_strength = 0) {

	__shared__ int gsCellReso[3][8];
	__shared__ int gsVertexReso[3][8];
	__shared__ int gsCellEnd[8];
	__shared__ int gsVertexEnd[8];

	//#if USE_LAME_MATRIX
	//	__shared__ Lame KLAME[24][24];
	//#else
	//__shared__ float KE[24][24];
	__shared__ float KE[24][24];
	//#endif

	__shared__ float sumKeU[3][4][32];
	__shared__ float sumKs[9][4][32];

	//__shared__ double* U[3];

	//__shared__ int NeNv[8][8];

	initSharedMem(&sumKeU[0][0][0], sizeof(sumKeU) / sizeof(float));
	initSharedMem(&sumKs[0][0][0], sizeof(sumKs) / sizeof(float));

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;

	//	if (laneId < 1) {
	//#pragma unroll
	//		for (int i = 0; i < 8; i++) {
	//			NeNv[warpId][i] = (warpId % 2 + i % 2) +
	//				(warpId / 2 % 2 + i / 2 % 2) * 3 +
	//				(warpId / 4 + i / 4) * 9;
	//		}
	//	}

	// local vertex id in gs set
	int vid = blockIdx.x * 32 + laneId;

	bool fiction = false;

	//#if USE_LAME_MATRIX
	//	// load Lame matrix
	//	loadLameMatrix(KLAME);
	//#else
	// load template matrix
	//#endif

	// load cell and vertex reso
	constant2DToShared(gGsCellReso, gsCellReso);
	constant2DToShared(gGsVertexReso, gsVertexReso);
	constantToShared(gGsCellEnd, gsCellEnd);
	constantToShared(gGsVertexEnd, gsVertexEnd);
	//constantToShared(gU, U);
	loadTemplateMatrix(KE);

	// to global vertex id
	vid = gs_set == 0 ? vid : gsVertexEnd[gs_set - 1] + vid;

	if (vid >= gsVertexEnd[gs_set])
		fiction = true;

	GridVertexIndex indexer(gridCellReso[0], gridCellReso[1], gridCellReso[2]);
	VertexFlags vflag;
	if (!fiction) {
		vflag = vflags[vid];
		fiction = fiction || vflag.is_fiction();
		indexer.locate(vid, vflag.get_gscolor(), gsVertexEnd);
	}

	float KeU[3] = {0.};
	float Ks[3][3] = {0.f};

	//fiction |= vflag.is_max_boundary();

	if (!fiction && !vflag.is_period_padding()) {
		int elementId = indexer.neighElement(warpId, gsCellEnd, gsCellReso).getId();
		int vselfrow = (7 - warpId) * 3;
		float rho_penal = 0;
		CellFlags eflag;
		float penal = exp_penal[0];
		if (elementId != -1) {
			eflag = eflags[elementId];
			if (!eflag.is_fiction())
				rho_penal = rhoPenalMin + powf(float(rholist[elementId]), penal);
		}

		if (elementId != -1 && !eflag.is_fiction() /*&& !eflag.is_period_padding()*/) {
#pragma unroll
			for (int i = 0; i < 8; i++) {
				if (i == 7 - warpId)
					continue;
				//#if 0
				int vneigh =
					(warpId % 2 + i % 2) +
					(warpId / 2 % 2 + i / 2 % 2) * 3 +
					(warpId / 4 + i / 4) * 9;
				//#else
				//				int vneigh = NeNv[warpId][i];
				//#endif
				int vneighId = indexer.neighVertex(vneigh, gsVertexEnd, gsVertexReso).getId();
				VertexFlags nvflag;
				if (vneighId != -1) {
					nvflag = vflags[vneighId];
					//{
					//	int pn[3];
					//	auto p = indexer.getPos();
					//	p.x -= 1; p.y -= 1; p.z -= 1;
					//	gsid2pos(vneighId, nvflag.get_gscolor(), gsVertexReso, gsVertexEnd, pn);
					//	if (abs(pn[0] - p.x) >= 2 || abs(pn[1] - p.y) >= 2 || abs(pn[2] - p.z) >= 2) {
					//		printf("E%d, vid = %d, neigh = %d, neighid = %d\n", __LINE__, vid, vneigh, vneighId);
					//	}
					//}
					if (!nvflag.is_fiction()) {
						float u[3] = {gU[0][vneighId], gU[1][vneighId], gU[2][vneighId]};
						if (nvflag.is_dirichlet_boundary()) {
							u[0] = u[1] = u[2] = 0;
						}
						int colsel = i * 3;
						KeU[0] += KE[vselfrow][colsel] * u[0] + KE[vselfrow][colsel + 1] * u[1] + KE[vselfrow][colsel + 2] * u[2];
						KeU[1] += KE[vselfrow + 1][colsel] * u[0] + KE[vselfrow + 1][colsel + 1] * u[1] + KE[vselfrow + 1][colsel + 2] * u[2];
						KeU[2] += KE[vselfrow + 2][colsel] * u[0] + KE[vselfrow + 2][colsel + 1] * u[1] + KE[vselfrow + 2][colsel + 2] * u[2];
					}
				}
			}
			KeU[0] *= rho_penal;
			KeU[1] *= rho_penal;
			KeU[2] *= rho_penal;

			for (int k3row = 0; k3row < 3; k3row++) {
				for (int k3col = 0; k3col < 3; k3col++) {
					Ks[k3row][k3col] += float(KE[vselfrow + k3row][vselfrow + k3col]) * rho_penal;
				}
			}

			//if (diag_strength) {
			//	Ks[0][0] += diag_strength;
			//	Ks[1][1] += diag_strength;
			//	Ks[2][2] += diag_strength;
			//}
		}
	}

	if (warpId >= 4) {
		//#pragma unroll
		for (int i = 0; i < 3; i++) {
			//#pragma unroll
			for (int j = 0; j < 3; j++) {
				sumKs[i * 3 + j][warpId - 4][laneId] = Ks[i][j];
			}
			sumKeU[i][warpId - 4][laneId] = KeU[i];
		}
	}
	__syncthreads();

	if (warpId < 4) {
		//#pragma unroll
		for (int i = 0; i < 3; i++) {
			//#pragma unroll
			for (int j = 0; j < 3; j++) {
				sumKs[i * 3 + j][warpId][laneId] += Ks[i][j];
			}
			sumKeU[i][warpId][laneId] += KeU[i];
		}
	}
	__syncthreads();

	if (warpId < 2) {
		//#pragma unroll
		for (int i = 0; i < 3; i++) {
			//#pragma unroll
			for (int j = 0; j < 3; j++) {
				sumKs[i * 3 + j][warpId][laneId] += sumKs[i * 3 + j][warpId + 2][laneId];
			}
			sumKeU[i][warpId][laneId] += sumKeU[i][warpId + 2][laneId];
		}
	}
	__syncthreads();

	if (warpId < 1 && !vflag.is_period_padding() && !fiction) {
		//#pragma unroll
		for (int i = 0; i < 3; i++) {
			//#pragma unroll
			for (int j = 0; j < 3; j++) {
				Ks[i][j] = sumKs[i * 3 + j][warpId][laneId] + sumKs[i * 3 + j][warpId + 1][laneId];
			}
			KeU[i] = sumKeU[i][warpId][laneId] + sumKeU[i][warpId + 1][laneId];
		}

		// DEBUG
		//if (vid == 394689) {
		//	printf("ku = (%.4le, %.4le, %.4le)\n", KeU[0], KeU[1], KeU[2]);
		//}

		float u[3] = {gU[0][vid], gU[1][vid], gU[2][vid]};

		// relax
		u[0] = w * (float(gF[0][vid]) - KeU[0] - Ks[0][1] * u[1] - Ks[0][2] * u[2]) / Ks[0][0] + (float(1) - w) * u[0];
		u[1] = w * (float(gF[1][vid]) - KeU[1] - Ks[1][0] * u[0] - Ks[1][2] * u[2]) / Ks[1][1] + (float(1) - w) * u[1];
		u[2] = w * (float(gF[2][vid]) - KeU[2] - Ks[2][0] * u[0] - Ks[2][1] * u[1]) / Ks[2][2] + (float(1) - w) * u[2];

		// if dirichlet boundary;
		if (vflag.is_dirichlet_boundary()) {
			u[0] = u[1] = u[2] = 0;
		}
		// update
		gU[0][vid] = u[0];
		gU[1][vid] = u[1];
		gU[2][vid] = u[2];
	}
}

// map 32 vertices to 13 warp
template<int BlockSize = 32 * 13>
__global__ void gs_relaxation_kernel(
	int gs_set,
	VertexFlags* vflags,
	// SOR relaxing factor
	VT w = 1.f) {
	__shared__ float sumAu[3][7][32];
	__shared__ int gsVertexEnd[8];
	__shared__ int gsVertexReso[3][8];

	constantToShared(gGsVertexEnd, gsVertexEnd);
	constant2DToShared(gGsVertexReso, gsVertexReso);
	initSharedMem(&sumAu[0][0][0], sizeof(sumAu) / sizeof(float));
	__syncthreads();

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;

	// local vertex id in gs set
	int vid = blockIdx.x * 32 + laneId;
	// global vertex id
	vid = gs_set == 0 ? vid : gsVertexEnd[gs_set - 1] + vid;

	bool fiction = false;
	if (vid >= gsVertexEnd[gs_set])
		fiction = true;
	VertexFlags vflag;
	if (!fiction)
		vflag = vflags[vid];
	fiction = fiction || vflag.is_fiction();

	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);

	glm::vec<3, float> Au(0.);

	if (!fiction && !vflag.is_period_padding()) {
		indexer.locate(vid, vflag.get_gscolor(), gsVertexEnd);

		for (int noff : {0, 14}) {
			int vneigh = warpId + noff;
			int neighId = indexer.neighVertex(vneigh, gsVertexEnd, gsVertexReso).getId();
			if (neighId == -1)
				continue;
			VertexFlags neiflag = vflags[neighId];
			if (!neiflag.is_fiction()) {
				glm::vec<3, float> u(gU[0][neighId], gU[1][neighId], gU[2][neighId]);
				Au += glm::mat3(rxstencil[vneigh][vid]) * u;
			}
		}
	}

	if (warpId >= 7) {
		for (int i = 0; i < 3; i++) {
			sumAu[i][warpId - 7][laneId] = Au[i];
		}
	}
	__syncthreads();

	if (warpId < 7) {
		if (warpId < 6) {
			for (int i = 0; i < 3; i++) {
				sumAu[i][warpId][laneId] += Au[i];
			}
		} else {
			for (int i = 0; i < 3; i++) {
				sumAu[i][6][laneId] = Au[i];
			}
		}
	}
	__syncthreads();

	if (warpId < 3) {
		for (int i = 0; i < 3; i++) {
			sumAu[i][warpId][laneId] += sumAu[i][warpId + 4][laneId];
		}
	}
	__syncthreads();

	if (warpId < 2) {
		for (int i = 0; i < 3; i++) {
			sumAu[i][warpId][laneId] += sumAu[i][warpId + 2][laneId];
		}
	}
	__syncthreads();

	if (warpId < 1 && !fiction) {
		for (int i = 0; i < 3; i++) {
			Au[i] = sumAu[i][warpId][laneId] + sumAu[i][warpId + 1][laneId];
		}

		//if (vflag.is_dirichlet_boundary()) {
		//	gU[0][vid] = 0; gU[1][vid] = 0; gU[2][vid] = 0;
		//	return;
		//}

		if (!vflag.is_period_padding()) {
			VT u[3] = {gU[0][vid], gU[1][vid], gU[2][vid]};
			// glm::hmat3 st = rxstencil[13][vid];
			glm::mat3 st = rxstencil[13][vid];
			VT f[3] = {gF[0][vid], gF[1][vid], gF[2][vid]};
			// u[0] = w * (f[0] - half(Au[0]) - rxstencil[13][1][vid] * u[1] - rxstencil[13][2][vid] * u[2]) / rxstencil[13][0][vid] + (half(1) - w) * u[0];
			// u[1] = w * (f[1] - half(Au[1]) - rxstencil[13][3][vid] * u[0] - rxstencil[13][5][vid] * u[2]) / rxstencil[13][4][vid] + (half(1) - w) * u[1];
			// u[2] = w * (f[2] - half(Au[2]) - rxstencil[13][6][vid] * u[0] - rxstencil[13][7][vid] * u[1]) / rxstencil[13][8][vid] + (half(1) - w) * u[2];
			u[0] = w * (f[0] - VT(Au[0]) - st[1][0] * u[1] - st[2][0] * u[2]) / st[0][0] + (VT(1) - w) * u[0];
			u[1] = w * (f[1] - VT(Au[1]) - st[0][1] * u[0] - st[2][1] * u[2]) / st[1][1] + (VT(1) - w) * u[1];
			u[2] = w * (f[2] - VT(Au[2]) - st[0][2] * u[0] - st[1][2] * u[1]) / st[2][2] + (VT(1) - w) * u[2];

			//if (rxstencil[13][0][vid] == 0) {
			//	short3 pos = indexer.getPos();
			//	printf("pos = (%d, %d, %d) d = (%e %e %e)\n",
			//		pos.x - 1, pos.y - 1, pos.z - 1,
			//		rxstencil[13][0][vid], rxstencil[13][4][vid], rxstencil[13][8][vid]);
			//}

			gU[0][vid] = u[0];
			gU[1][vid] = u[1];
			gU[2][vid] = u[2];
		}
	}
}

// scatter per fine element matrix to coarse stencil, one thread for one element
// stencil was organized in lexico order(No padding), and should be transferred to gs order

template<int BlockSize = 256>
__global__ void restrict_residual_kernel_1(
	int nv_coarse,
	VertexFlags* vflags,
	VertexFlags* vfineflags,
	devArray_t<int, 8> GsVertexEnd,
	devArray_t<int, 8> GsFineVertexEnd) {
	__shared__ int gsVertexEnd[8];
	__shared__ int gsFineVertexEnd[8];
	__shared__ int gsFineVertexReso[3][8];

	if (threadIdx.x < 24) {
		gsFineVertexReso[threadIdx.x / 8][threadIdx.x % 8] =
			gGsFineVertexReso[threadIdx.x / 8][threadIdx.x % 8];
		if (threadIdx.x < 8) {
			gsVertexEnd[threadIdx.x] = GsVertexEnd[threadIdx.x];
			gsFineVertexEnd[threadIdx.x] = GsFineVertexEnd[threadIdx.x];
		}
	}
	__syncthreads();

	size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= nv_coarse)
		return;

	VertexFlags vflag = vflags[tid];
	bool fiction = vflag.is_fiction();

	int setid = vflag.get_gscolor();

	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);
	indexer.locate(tid, vflag.get_gscolor(), gsVertexEnd);

	int coarseRatio[3] = {gUpCoarse[0], gUpCoarse[1], gUpCoarse[2]};
	float pr = coarseRatio[0] * coarseRatio[1] * coarseRatio[2];

	bool nondyadic = coarseRatio[0] > 2 || coarseRatio[1] > 2 || coarseRatio[2] > 2;

	float r[3] = {0.};

	if (!fiction && !vflag.is_period_padding()) {
		for (int offx = -coarseRatio[0] + 1; offx < coarseRatio[0]; offx++) {
			for (int offy = -coarseRatio[1] + 1; offy < coarseRatio[1]; offy++) {
				for (int offz = -coarseRatio[2] + 1; offz < coarseRatio[2]; offz++) {
					int off[3] = {offx, offy, offz};
					float w = (coarseRatio[0] - abs(offx)) * (coarseRatio[1] - abs(offy)) * (coarseRatio[2] - abs(offz)) / pr;
					int neighVid = -1;
					// DEBUG
					if (nondyadic)
						neighVid = indexer.neighFineVertex(off, coarseRatio, gsFineVertexEnd, gsFineVertexReso, true).getId();
					else
						neighVid = indexer.neighFineVertex(off, coarseRatio, gsFineVertexEnd, gsFineVertexReso, false).getId();

					VertexFlags vfineflag;
					if (neighVid != -1) {
						vfineflag = vfineflags[neighVid];
						if (!vfineflag.is_fiction()) {
							r[0] += float(gRfine[0][neighVid]) * w;
							r[1] += float(gRfine[1][neighVid]) * w;
							r[2] += float(gRfine[2][neighVid]) * w;
							//if (tid == 40911) {
							//	short3 mypos = indexer.getPos();
							//	printf("mypos = (%d, %d, %d)  neioff = [%d](%d, %d, %d) rfine = (%e %e %e)\n",
							//		mypos.x - 1, mypos.y - 1, mypos.z - 1, neighVid, offx, offy, offz,
							//		gRfine[0][neighVid], gRfine[1][neighVid], gRfine[2][neighVid]);
							//}
						}
					}
				}
			}
		}

		//if (vflag.is_dirichlet_boundary()) r[0] = r[1] = r[2] = 0;
		for (int i = 0; i < 3; i++) {
			gF[i][tid] = r[i];
		}
	}
}

// one thread of one coarse vertex

template<int BlockSize = 256>
__global__ void prolongate_correction_kernel_1(
	bool is_root,
	int nv_fine,
	VertexFlags* vflags,
	VertexFlags* vcoarseflags,
	devArray_t<int, 8> GsVertexEnd,
	devArray_t<int, 8> GsCoarseVertexEnd) {
	__shared__ int coarseRatio[3];
	__shared__ int gsCoarseVertexReso[3][8];
	__shared__ int gsCoarseVertexEnd[8];

	if (threadIdx.x < 24) {
		gsCoarseVertexReso[threadIdx.x / 8][threadIdx.x % 8] = gGsCoarseVertexReso[threadIdx.x / 8][threadIdx.x % 8];
	}
	if (threadIdx.x < 8) {
		gsCoarseVertexEnd[threadIdx.x] = GsCoarseVertexEnd[threadIdx.x];
	}
	if (threadIdx.x < 3) {
		coarseRatio[threadIdx.x] = gDownCoarse[threadIdx.x];
	}
	__syncthreads();

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	bool fiction = false;
	if (tid >= nv_fine) {
		return;
	}

	VertexFlags vflag = vflags[tid];
	fiction = fiction || vflag.is_fiction();

	float pr = coarseRatio[0] * coarseRatio[1] * coarseRatio[2];

	bool isRoot = is_root;

	if (!fiction && !vflag.is_period_padding()) {
		GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);
		indexer.locate(tid, vflag.get_gscolor(), GsVertexEnd._data);

		float u[3] = {0.};
		int nvCoarse[8];
		float w[8];
		int remainder[3];
		indexer.neighCoarseVertex(nvCoarse, w, coarseRatio, gsCoarseVertexEnd, gsCoarseVertexReso, remainder);
		for (int i = 0; i < 8; i++) {
			int neighId = nvCoarse[i];
			if (neighId != -1) {
				float uc[3] = {gUcoarse[0][neighId], gUcoarse[1][neighId], gUcoarse[2][neighId]};
				//VertexFlags vcflag = vcoarseflags[neighId];
				//if (vcflag.is_dirichlet_boundary()) uc[0] = uc[1] = uc[2] = 0;
				// DEBUG
				//if (tid == 111) {
				//	printf("uc = (%.4le, %.4le, %.4le) w = %6.4f rem = (%03d, %03d, %03d) pr = %f\n", uc[0], uc[1], uc[2], w[i],
				//		remainder[0], remainder[1], remainder[2], pr);
				//}
				u[0] += uc[0] * w[i];
				u[1] += uc[1] * w[i];
				u[2] += uc[2] * w[i];
			}
		}

		if (isRoot && vflag.is_dirichlet_boundary()) {
			u[0] = u[1] = u[2] = 0;
		}
		gU[0][tid] += VT(u[0]);
		gU[1][tid] += VT(u[1]);
		gU[2][tid] += VT(u[2]);
	}
}

void homo::Grid::gs_relaxation(float w_SOR /*= 1.f*/, int times_ /*= 1*/) {
	AbortErr();
	// change to 8 bytes bank
	use8Bytesbank();
	useGrid_g();
	devArray_t<int, 3> gridCellReso{};
	devArray_t<int, 8> gsCellEnd{};
	devArray_t<int, 8> gsVertexEnd{};
	for (int i = 0; i < 8; i++) {
		gsCellEnd[i] = gsCellSetEnd[i];
		gsVertexEnd[i] = gsVertexSetEnd[i];
		if (i < 3)
			gridCellReso[i] = cellReso[i];
	}
	for (int itn = 0; itn < times_; itn++) {
		for (int i = 0; i < 8; i++) {
			int set_id = i;
			int n_gs = gsVertexEnd[set_id] - (set_id == 0 ? 0 : gsVertexEnd[set_id - 1]);
			if (assemb_otf) {
				auto cfg = make_kernel_param(n_gs * 8, 32 * 8);
				gs_relaxation_otf_kernel<<<cfg.grid, cfg.block>>>(set_id, rho_g, gridCellReso, vertflag, cellflag, w_SOR, diag_strength);
			} else {
				auto cfg = make_kernel_param(n_gs * 13, 32 * 13);
				gs_relaxation_kernel<<<cfg.grid, cfg.block>>>(set_id, vertflag, w_SOR);
			}
			cudaDeviceSynchronize();
			cuda_error_check;
			enforce_period_boundary(u_g);
		}
	}
	enforce_period_boundary(u_g);
	//pad_vertex_data(u_g);
	cudaDeviceSynchronize();
	cuda_error_check;
}

void homo::Grid::prolongate_correction(void) {
	useGrid_g();
	VertexFlags* vflags = vertflag;
	VertexFlags* vcoarseFlags = Coarse->vertflag;
	devArray_t<int, 8> gsVertexEnd{}, gsCoarseVertexEnd{};
	for (int i = 0; i < 8; i++) {
		gsVertexEnd[i] = gsVertexSetEnd[i];
		gsCoarseVertexEnd[i] = Coarse->gsVertexSetEnd[i];
	}
	int nv_fine = n_gsvertices();
	auto cfg = make_kernel_param(nv_fine, 256);
	prolongate_correction_kernel_1<<<cfg.grid, cfg.block>>>(is_root, nv_fine, vflags, vcoarseFlags, gsVertexEnd, gsCoarseVertexEnd);
	cudaDeviceSynchronize();
	cuda_error_check;
	enforce_period_boundary(u_g);
}

void homo::Grid::restrict_residual(void) {
	useGrid_g();
	VertexFlags* vflags = vertflag;
	VertexFlags* vfineflags = fine->vertflag;
	devArray_t<int, 8> gsVertexEnd{}, gsFineVertexEnd{};
	for (int i = 0; i < 8; i++) {
		gsVertexEnd[i] = gsVertexSetEnd[i];
		gsFineVertexEnd[i] = fine->gsVertexSetEnd[i];
	}
	int nv = n_gsvertices();
	auto cfg = make_kernel_param(nv, 256);
	restrict_residual_kernel_1<<<cfg.grid, cfg.block>>>(nv, vflags, vfineflags, gsVertexEnd, gsFineVertexEnd);
	cudaDeviceSynchronize();
	cuda_error_check;
	pad_vertex_data(f_g);
}

// map 32 vertices to 8 warp
template<typename T>
__global__ void update_residual_otf_kernel_1(
	int nv, T* rholist,
	devArray_t<int, 3> gridCellReso,
	VertexFlags* vflags, CellFlags* eflags,
	float diag_strength) {
	__shared__ int gsCellReso[3][8];
	__shared__ int gsVertexReso[3][8];
	__shared__ int gsCellEnd[8];
	__shared__ int gsVertexEnd[8];
	__shared__ float KE[24][24];

	__shared__ float sumKeU[3][4][32];

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;

	// local vertex id in gs set
	int vid = blockIdx.x * 32 + laneId;

	bool fiction = false;

	fiction = fiction || vid >= nv;
	VertexFlags vflag;
	if (!fiction) {
		vflag = vflags[vid];
		fiction = fiction || vflag.is_fiction() || vflag.is_period_padding();
	}
	int set_id = vflag.get_gscolor();

	// load cell and vertex reso
	constant2DToShared(gGsCellReso, gsCellReso);
	constant2DToShared(gGsVertexReso, gsVertexReso);
	constantToShared(gGsCellEnd, gsCellEnd);
	constantToShared(gGsVertexEnd, gsVertexEnd);
	initSharedMem(&sumKeU[0][0][0], sizeof(sumKeU) / sizeof(float));
	// load template matrix
	loadTemplateMatrix(KE);

	GridVertexIndex indexer(gridCellReso[0], gridCellReso[1], gridCellReso[2]);
	if (!fiction) {
		indexer.locate(vid, vflag.get_gscolor(), gsVertexEnd);
	}

	float KeU[3] = {0.};

	// if (!fiction) gR[0][vid] = KE[0][0];

	int elementId = -1;
	if (!fiction)
		elementId = indexer.neighElement(warpId, gsCellEnd, gsCellReso).getId();
	int vselfrow = (7 - warpId) * 3;
	float rhop = 0;
	CellFlags eflag;
	float penal = exp_penal[0];
	if (elementId != -1) {
		eflag = eflags[elementId];
		if (!eflag.is_fiction())
			rhop = rhoPenalMin + powf(float(rholist[elementId]), penal);
	}

	// DEBUG
	//bool debug = false;
	if (elementId != -1 && !eflag.is_fiction() && !fiction) {
#pragma unroll
		for (int i = 0; i < 8; i++) {
			int vneigh =
				(warpId % 2 + i % 2) +
				(warpId / 2 % 2 + i / 2 % 2) * 3 +
				(warpId / 4 + i / 4) * 9;
			int vneighId = indexer.neighVertex(vneigh, gsVertexEnd, gsVertexReso).getId();
			VertexFlags nvflag;
			if (vneighId != -1) {
				nvflag = vflags[vneighId];
				if (!nvflag.is_fiction()) {
					float u[3] = {gU[0][vneighId], gU[1][vneighId], gU[2][vneighId]};
					if (nvflag.is_dirichlet_boundary()) {
						u[0] = u[1] = u[2] = 0;
					}

					int colsel = i * 3;
					KeU[0] += (KE[vselfrow][colsel] * u[0] + KE[vselfrow][colsel + 1] * u[1] + KE[vselfrow][colsel + 2] * u[2]);
					KeU[1] += (KE[vselfrow + 1][colsel] * u[0] + KE[vselfrow + 1][colsel + 1] * u[1] + KE[vselfrow + 1][colsel + 2] * u[2]);
					KeU[2] += (KE[vselfrow + 2][colsel] * u[0] + KE[vselfrow + 2][colsel + 1] * u[1] + KE[vselfrow + 2][colsel + 2] * u[2]);
					//if (diag_strength) {
					//	if (i == 7 - warpId) {
					//		KeU[0] += u[0] * diag_strength;
					//		KeU[1] += u[1] * diag_strength;
					//		KeU[2] += u[2] * diag_strength;
					//		//hasDiag = true;
					//	}
					//}
				}
			}
		}
		KeU[0] *= rhop;
		KeU[1] *= rhop;
		KeU[2] *= rhop;
		//if (!hasDiag)
		//{
		//	auto p = indexer.getPos();
		//	if (p.x == 7 && p.y == 1 && p.z == 1) {
		//		debug = true;
		//		double u[3] = { gU[0][vid], gU[1][vid], gU[2][vid] };
		//		printf("u = (%.4e, %.4e, %.4e) keU = (%.4e, %.4e, %.4e)\n", u[0], u[1], u[2], KeU[0], KeU[1], KeU[2]);
		//	}
		//}
	}
	if (warpId >= 4) {
		for (int i = 0; i < 3; i++) {
			sumKeU[i][warpId - 4][laneId] = KeU[i];
		}
	}
	__syncthreads();

	if (warpId < 4) {
		for (int i = 0; i < 3; i++) {
			sumKeU[i][warpId][laneId] += KeU[i];
		}
	}
	__syncthreads();

	if (warpId < 2) {
		for (int i = 0; i < 3; i++) {
			sumKeU[i][warpId][laneId] += sumKeU[i][warpId + 2][laneId];
		}
	}
	__syncthreads();

	if (warpId < 1 && !fiction && !vflag.is_period_padding()) {
		for (int i = 0; i < 3; i++) {
			KeU[i] = sumKeU[i][warpId][laneId] + sumKeU[i][warpId + 1][laneId];
		}

		float r[3] = {
			float(gF[0][vid]) - KeU[0],
			float(gF[1][vid]) - KeU[1],
			float(gF[2][vid]) - KeU[2]};

		if (vflag.is_dirichlet_boundary()) {
			r[0] = r[1] = r[2] = 0;
		}

		//if (debug) {
		//	printf("sumKu = (%.4e, %.4e, %.4e)\n", KeU[0], KeU[1], KeU[2]);
		//}
		//{
		//	auto p = indexer.getPos();
		//	bool debug = p.x == 7 && p.y == 1 && p.z == 7;
		//	if (debug) {
		//		if (set_id != 7) { print_exception; }
		//		printf("\nresid | f = (%e, %e, %e)   keU = (%e, %e, %e)\n", gF[0][vid], gF[1][vid], gF[2][vid], KeU[0], KeU[1], KeU[2]);

		//		printf("resid | \n");
		//		for (int i = 0; i < 27; i++) {
		//			int neighid = indexer.neighVertex(i, gsVertexEnd, gsVertexReso).getId();
		//			if (neighid == -1) {
		//				printf("   void\n");
		//			}
		//			else {
		//				printf("  [%d]  %e %e %e\n", neighid, gU[0][neighid], gU[1][neighid], gU[2][neighid]);
		//			}
		//		}
		//		printf("\n");
		//	}

		//}

		// relax
		gR[0][vid] = r[0];
		gR[1][vid] = r[1];
		gR[2][vid] = r[2];
	}
}

// map 32 vertices to 9 warp
template<int BlockSize = 32 * 9>
__global__ void update_residual_kernel_1(
	int nv,
	VertexFlags* vflags) {

	__shared__ int gsVertexEnd[8];
	__shared__ int gsVertexReso[3][8];
	__shared__ float sumKu[3][5][32];

	constantToShared(gGsVertexEnd, gsVertexEnd);
	constant2DToShared(gGsVertexReso, gsVertexReso);

	initSharedMem(&sumKu[0][0][0], sizeof(sumKu) / sizeof(float));

	__syncthreads();

	bool fiction = false;

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;
	int vid = blockIdx.x * 32 + laneId;
	if (vid >= nv)
		fiction = true;

	VertexFlags vflag;
	if (!fiction)
		vflag = vflags[vid];
	fiction = fiction || vflag.is_fiction();
	int color = vflag.get_gscolor();

	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);
	indexer.locate(vid, vflag.get_gscolor(), gsVertexEnd);

	glm::vec<3, float> KeU(0.);
	if (!fiction && !vflag.is_period_padding()) {
		for (auto off : {0, 9, 18}) {
			int vneigh = off + warpId;
			int neighId = indexer.neighVertex(vneigh, gsVertexEnd, gsVertexReso).getId();
			if (neighId != -1) {
				VertexFlags neighFlag = vflags[neighId];
				if (!neighFlag.is_fiction()) {
					glm::vec<3, float> u(gU[0][neighId], gU[1][neighId], gU[2][neighId]);
					KeU += rxstencil[vneigh][vid] * u;
				}
			}
		}
	}

	if (warpId >= 4) {
		sumKu[0][warpId - 4][laneId] = KeU[0];
		sumKu[1][warpId - 4][laneId] = KeU[1];
		sumKu[2][warpId - 4][laneId] = KeU[2];
	}
	__syncthreads();

	if (warpId < 4) {
		sumKu[0][warpId][laneId] += KeU[0];
		sumKu[1][warpId][laneId] += KeU[1];
		sumKu[2][warpId][laneId] += KeU[2];
	}
	__syncthreads();

	if (warpId < 2) {
		sumKu[0][warpId][laneId] += sumKu[0][warpId + 2][laneId];
		sumKu[1][warpId][laneId] += sumKu[1][warpId + 2][laneId];
		sumKu[2][warpId][laneId] += sumKu[2][warpId + 2][laneId];
	}
	__syncthreads();

	if (warpId < 1 && !fiction) {
		KeU[0] = sumKu[0][warpId][laneId] + sumKu[0][warpId + 1][laneId] + sumKu[0][4][laneId];
		KeU[1] = sumKu[1][warpId][laneId] + sumKu[1][warpId + 1][laneId] + sumKu[1][4][laneId];
		KeU[2] = sumKu[2][warpId][laneId] + sumKu[2][warpId + 1][laneId] + sumKu[2][4][laneId];

		float r[3] = {gF[0][vid] - VT(KeU[0]), gF[1][vid] - VT(KeU[1]), gF[2][vid] - VT(KeU[2])};
		//if (vflag.is_dirichlet_boundary()) {
		//	r[0] = r[1] = r[2] = 0;
		//}

		gR[0][vid] = r[0];
		gR[1][vid] = r[1];
		gR[2][vid] = r[2];
	}
}

void homo::Grid::update_residual(void) {
	useGrid_g();
	devArray_t<int, 3> gridCellReso{cellReso[0], cellReso[1], cellReso[2]};
	VertexFlags* vflags = vertflag;
	CellFlags* eflags = cellflag;
	if (assemb_otf) {
		auto cfg = make_kernel_param(n_gsvertices() * 8, 32 * 8);
		update_residual_otf_kernel_1<<<cfg.grid, cfg.block>>>(n_gsvertices(), rho_g, gridCellReso,
															  vflags, eflags, diag_strength);
		cudaDeviceSynchronize();
		cuda_error_check;
	} else {
		int nv = n_gsvertices();
		auto cfg = make_kernel_param(n_gsvertices() * 9, 32 * 9);
		update_residual_kernel_1<<<cfg.grid, cfg.block>>>(nv, vflags);
		cudaDeviceSynchronize();
		cuda_error_check;
	}
	pad_vertex_data(r_g);
}

__device__ int gsPos2Id(int pos[3], int* gsEnd, int (*gsReso)[8]) {
	int posRem[3] = {pos[0] % 2, pos[1] % 2, pos[2] % 2};
	int gsPos[3] = {pos[0] / 2, pos[1] / 2, pos[2] / 2};
	int color = posRem[0] + posRem[1] * 2 + posRem[2] * 4;

	int gsid = (color == 0 ? 0 : gsEnd[color - 1]) +
			   gsPos[0] +
			   gsPos[1] * gsReso[0][color] +
			   gsPos[2] * gsReso[0][color] * gsReso[1][color];
	if (gsid >= gsEnd[color]) {
		return -1;
	}
	return gsid;
}

//__global__ void padding_period_vertex_data_kernel(
//	devArray_t<double*, 3> v
//) {
//	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//
//	int n_updown = (gGridCellReso[0] + 1) * (gGridCellReso[1] + 1);
//	int n_leftright = (gGridCellReso[1] + 1) * (gGridCellReso[2] + 1);
//	int n_frontback = (gGridCellReso[0] + 1) * (gGridCellReso[2] + 1);
//
//	int srcpos[3];
//	int dstpos[3];
//	if (tid < 2 * n_updown) {
//		int layeroffset = tid;
//		if (tid >= n_updown)  layeroffset = tid - n_updown;
//		dstpos[0] = layeroffset % (gGridCellReso[0] + 1) + 1;
//		dstpos[1] = layeroffset / (gGridCellReso[0] + 1) + 1;
//		srcpos[0] = srcpos[0];
//		srcpos[1] = srcpos[1];
//		if (tid < n_updown) {
//			dstpos[2] = 0;
//			srcpos[2] = gGridCellReso[2] + 1;
//		} else {
//			dstpos[2] = gGridCellReso[2] + 2;
//			srcpos[2] = 1;
//		}
//		goto __period_copy;
//	}
//	tid -= 2 * n_updown;
//	if (tid < 2 * n_leftright) {
//		int layeroffset = tid;
//		if (tid >= n_leftright) layeroffset = tid - n_leftright;
//		dstpos[1] = tid % (gGridCellReso[1] + 1) + 1;
//		dstpos[2] = tid / (gGridCellReso[1] + 1) + 1;
//		srcpos[1] = dstpos[1];
//		srcpos[2] = dstpos[2];
//		if (tid >= n_leftright) {
//			dstpos[0] = 0;
//			srcpos[0] = gGridCellReso[0] + 1;
//		}
//		else {
//			dstpos[0] = gGridCellReso[0] + 2;
//			srcpos[0] = 1;
//		}
//		goto __period_copy;
//	}
//	tid -= 2 * n_leftright;
//	if (tid < 2 * n_frontback) {
//		int layeroffset = tid;
//		if (tid >= n_frontback) layeroffset = tid - n_frontback;
//		dstpos[0] = layeroffset % (gGridCellReso[0] + 1) + 1;
//		dstpos[2] = layeroffset / (gGridCellReso[0] + 1) + 1;
//		srcpos[0] = dstpos[0];
//		srcpos[2] = dstpos[2];
//		if (tid >= n_frontback) {
//			dstpos[1] = 0;
//			srcpos[1] = gGridCellReso[1] + 1;
//		}
//		else {
//			dstpos[1] = gGridCellReso[1] + 2;
//			srcpos[1] = 1;
//		}
//		goto __period_copy;
//	}
//
//__period_copy:
//	int srcid = gsPos2Id(srcpos, gGsVertexEnd, gGsVertexReso);
//	int dstid = gsPos2Id(dstpos, gGsVertexEnd, gGsVertexReso);
//
//	v[0][dstid] = v[0][srcid];
//	v[1][dstid] = v[1][srcid];
//	v[2][dstid] = v[2][srcid];
//}
//
//void Grid::padding_period_vertex_data(double* v[3])
//{
//	size_t grid_size, block_size;
//	int n_vboundary =
//		(cellReso[0] + 1) * (cellReso[1] + 1) * 2 +
//		(cellReso[1] + 1) * (cellReso[2] + 1) * 2+
//		(cellReso[0] + 1) * (cellReso[2] + 1) * 2;
//	auto cfg = make_kernel_param(n_vboundary, 256);
//	devArray_t<double*, 3> vdata{ v[0],v[1],v[2] };
//	padding_period_vertex_data_kernel <<<cfg.grid, cfg.block>>> (vdata);
//	cudaDeviceSynchronize();
//	cuda_error_check;
//}

//template<typename T>
//__global__ void padding_period_cell_data_kernel(
//	T* u
//) {
//	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//
//	int n_updown = gGridCellReso[0] * gGridCellReso[1];
//	int n_leftright = gGridCellReso[1] * gGridCellReso[2];
//	int n_frontback = gGridCellReso[0] * gGridCellReso[2];
//
//	int srcpos[3];
//	int dstpos[3];
//	if (tid < 2 * n_updown) {
//		int layeroffset = tid;
//		if (tid >= n_updown)  layeroffset = tid - n_updown;
//		dstpos[0] = layeroffset % gGridCellReso[0] + 1;
//		dstpos[1] = layeroffset / gGridCellReso[0] + 1;
//		srcpos[0] = srcpos[0];
//		srcpos[1] = srcpos[1];
//		if (tid < n_updown) {
//			dstpos[2] = 0;
//			srcpos[2] = gGridCellReso[2];
//		}
//		else {
//			dstpos[2] = gGridCellReso[2] + 1;
//			srcpos[2] = 1;
//		}
//		goto __period_copy;
//	}
//	tid -= 2 * n_updown;
//	if (tid < 2 * n_leftright) {
//		int layeroffset = tid;
//		if (tid >= n_leftright) layeroffset = tid - n_leftright;
//		dstpos[1] = tid % gGridCellReso[1] + 1;
//		dstpos[2] = tid / gGridCellReso[1] + 1;
//		srcpos[1] = dstpos[1];
//		srcpos[2] = dstpos[2];
//		if (tid >= n_leftright) {
//			dstpos[0] = 0;
//			srcpos[0] = gGridCellReso[0];
//		}
//		else {
//			dstpos[0] = gGridCellReso[0] + 1;
//			srcpos[0] = 1;
//		}
//		goto __period_copy;
//	}
//	tid -= 2 * n_leftright;
//	if (tid < 2 * n_frontback) {
//		int layeroffset = tid;
//		if (tid >= n_frontback) layeroffset = tid - n_frontback;
//		dstpos[0] = layeroffset % gGridCellReso[0] + 1;
//		dstpos[2] = layeroffset / gGridCellReso[0] + 1;
//		srcpos[0] = dstpos[0];
//		srcpos[2] = dstpos[2];
//		if (tid >= n_frontback) {
//			dstpos[1] = 0;
//			srcpos[1] = gGridCellReso[1];
//		}
//		else {
//			dstpos[1] = gGridCellReso[1] + 1;
//			srcpos[1] = 1;
//		}
//		goto __period_copy;
//	}
//
//__period_copy:
//	int srcid = gsPos2Id(srcpos, gGsCellEnd, gGsCellReso);
//	int dstid = gsPos2Id(dstpos, gGsCellEnd, gGsCellReso);
//
//	u[dstid] = u[srcid];
//}
//
//void Grid::padding_period_element_data(float* rho) {
//	size_t grid_size, block_size;
//	int n_eboundary =
//		cellReso[0] * cellReso[1] * 2 +
//		cellReso[1] * cellReso[2] * 2 +
//		cellReso[0] * cellReso[2] * 2;
//	auto cfg = make_kernel_param(n_eboundary, 256);
//	padding_period_cell_data_kernel <<<cfg.grid, cfg.block>>> (rho);
//	cudaDeviceSynchronize();
//	cuda_error_check;
//
//}

// ToDo : map 32 vertices to 8 warp
template<typename T>
__global__ void enforce_unit_macro_strain_kernel(
	int nv, int istrain, devArray_t<Grid::VT*, 3> fcharlist, VertexFlags* vflags, CellFlags* eflags, T* rholist) {

	__shared__ Lame KLAME[24][24];

	loadLameMatrix(KLAME);
	float lam = LAM[0];
	float mu = MU[0];

	bool fiction = false;
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nv) {
		fiction = true;
		return;
	}

	VertexFlags vflag;
	if (!fiction) {
		vflag = vflags[tid];
		fiction = vflag.is_fiction();
	}

	int vid = tid;

	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);
	indexer.locate(vid, vflag.get_gscolor(), gGsVertexEnd);

	short3 vpos = indexer.getPos();

	int neighVid[27];
	for (int i = 0; i < 27; i++) {
		neighVid[i] = indexer.neighVertex(i, gGsVertexEnd, gGsVertexReso).getId();
	}

	//double fchar[6][3] = { 0. };
	float fchar[3] = {0.};
	do {
		if (vflag.is_period_padding() || vflag.is_fiction())
			break;
		for (int ei = 0; ei < 8; ei++) {
			int neighEid = indexer.neighElement(ei, gGsCellEnd, gGsCellReso).getId();
			if (neighEid == -1)
				continue;
			CellFlags eflag = eflags[neighEid];
			// if (eflag.is_fiction() || eflag.is_period_padding()) continue;
			float rho_penal = rhoPenalMin + powf(rholist[neighEid], exp_penal[0]);
			int kirow = (7 - ei) * 3;
			//float flamchar[6][3] = { 0 };
			//float fmuchar[6][3] = { 0 };
			float flamchar[3] = {0};
			float fmuchar[3] = {0};
			for (int kj = 0; kj < 8; kj++) {
				int kjcol = kj * 3;
				// e_xx
				float uj[3] = {};
				/*for (int istrain = 0; istrain < 6; istrain++)*/ {
					elementMacroDisplacement(kj, istrain, uj);
					flamchar[0] += KLAME[kirow][kjcol].lam() * uj[0] + KLAME[kirow][kjcol + 1].lam() * uj[1] + KLAME[kirow][kjcol + 2].lam() * uj[2];
					flamchar[1] += KLAME[kirow + 1][kjcol].lam() * uj[0] + KLAME[kirow + 1][kjcol + 1].lam() * uj[1] + KLAME[kirow + 1][kjcol + 2].lam() * uj[2];
					flamchar[2] += KLAME[kirow + 2][kjcol].lam() * uj[0] + KLAME[kirow + 2][kjcol + 1].lam() * uj[1] + KLAME[kirow + 2][kjcol + 2].lam() * uj[2];
					fmuchar[0] += KLAME[kirow][kjcol].mu() * uj[0] + KLAME[kirow][kjcol + 1].mu() * uj[1] + KLAME[kirow][kjcol + 2].mu() * uj[2];
					fmuchar[1] += KLAME[kirow + 1][kjcol].mu() * uj[0] + KLAME[kirow + 1][kjcol + 1].mu() * uj[1] + KLAME[kirow + 1][kjcol + 2].mu() * uj[2];
					fmuchar[2] += KLAME[kirow + 2][kjcol].mu() * uj[0] + KLAME[kirow + 2][kjcol + 1].mu() * uj[1] + KLAME[kirow + 2][kjcol + 2].mu() * uj[2];
				}
			}
			/*for (int istrain = 0; istrain < 6; istrain++)*/ {
				fchar[0] += rho_penal * (flamchar[0] * lam + fmuchar[0] * mu);
				fchar[1] += rho_penal * (flamchar[1] * lam + fmuchar[1] * mu);
				fchar[2] += rho_penal * (flamchar[2] * lam + fmuchar[2] * mu);
			}
		}
	} while (0);

	for (int j = 0; j < 3; j++) {
		fcharlist[j][vid] = fchar[j];
	}
}

void homo::Grid::enforce_unit_macro_strain(int istrain) {
	useGrid_g();
	cuda_error_check;
	VertexFlags* vflags = vertflag;
	CellFlags* eflags = cellflag;
	devArray_t<VT*, 3> fcharlist{f_g[0], f_g[1], f_g[2]};
	auto cfg = make_kernel_param(n_gsvertices(), 256);
	enforce_unit_macro_strain_kernel<<<cfg.grid, cfg.block>>>(n_gsvertices(), istrain, fcharlist, vflags, eflags, rho_g);
	cudaDeviceSynchronize();
	cuda_error_check;
}

float Grid::v3_norm(VT* v[3], bool removePeriodDof /*= false*/, int len /*= -1*/) {
	if (len < 0)
		len = n_gsvertices();
	//auto buffer = getTempPool().getBuffer(sizeof(double) * (len / 100));
	if (!removePeriodDof) {
		double nrm = norm(v[0], v[1], v[2], len, (float*)(0));
		cuda_error_check;
		return nrm;
	} else {
		double n2 = v3_dot(v, v, removePeriodDof);
		return sqrt(n2);
	}
}

void homo::Grid::v3_rand(VT* v[3], VT low, VT upp, int len /*= -1*/) {
	if (len == -1)
		len = n_gsvertices();
	randArray(v, 3, len, low, upp);
}

template<typename T, typename Tout, int BlockSize = 256>
__global__ void v3_diffnorm_kernel(devArray_t<T*, 3> vlist, devArray_t<T*, 3> ulist, Tout* p_out, size_t len) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	__shared__ Tout blocksum[BlockSize / 32];

	int base = 0;

	Tout s = 0;

	for (; base + tid < len && tid < len; base += stride) {
		int vid = base + tid;
		T vu[3] = {
			vlist[0][vid] - ulist[0][vid],
			vlist[1][vid] - ulist[1][vid],
			vlist[2][vid] - ulist[2][vid]};
		s += Tout(vu[0] * vu[0] + vu[1] * vu[1] + vu[2] * vu[2]);
	}

	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;

	for (int offset = 16; offset > 0; offset /= 2) {
		s += shfl_down(s, offset);
	}

	if (laneId == 0) {
		blocksum[warpId] = s;
	}

	__syncthreads();

	if (warpId == 0) {
		if (BlockSize / 32 > 32)
			print_exception;
		if (threadIdx.x < BlockSize / 32) {
			s = blocksum[threadIdx.x];
		} else {
			s = 0;
		}

		for (int offset = 16; offset > 0; offset /= 2) {
			s += shfl_down(s, offset);
		}

		if (laneId == 0) {
			p_out[blockIdx.x] = s;
		}
	}
}

float homo::Grid::v3_diffnorm(VT* v[3], VT* u[3], int len /*= -1*/) {
	if (len < 0)
		len = n_gsvertices();
	auto buffer = getTempPool().getBuffer(len * sizeof(float) / 100);
	float* buf = buffer.template data<float>();
	devArray_t<VT*, 3> vlist, ulist;
	vlist[0] = v[0];
	vlist[1] = v[1];
	vlist[2] = v[2];
	ulist[0] = u[0];
	ulist[1] = u[1];
	ulist[2] = u[2];
	auto cfg = make_kernel_param(len, 256);
	v3_diffnorm_kernel<<<cfg.grid, cfg.block>>>(vlist, ulist, buf, len);
	cudaDeviceSynchronize();
	cuda_error_check;

	double s = dump_array_sum(buf, cfg.grid);

	return sqrt(s);
}

void Grid::v3_reset(VT* v[3], int len /*= -1*/) {
	if (len < 0)
		len = n_gsvertices();
	for (int i = 0; i < 3; i++) {
		cudaMemset(v[i], 0, sizeof(VT) * len);
	}
	cudaDeviceSynchronize();
}

void Grid::v3_copy(VT* dst[3], VT* src[3], int len /*= -1*/) {
	if (len < 0)
		len = n_gsvertices();
	for (int i = 0; i < 3; i++) {
		cudaMemcpy(dst[i], src[i], sizeof(VT) * len, cudaMemcpyDeviceToDevice);
	}
	cuda_error_check;
}

template<typename Vec, typename T>
__global__ void compliance_kernel(
	int nv, devArray_t<T*, 3> ug, devArray_t<T*, 3> vg,
	Vec rholist, T* elementCompliance,
	CellFlags* eflags, VertexFlags* vflags
	//bool derivative = false
) {
	__shared__ float KE[24][24];

	loadTemplateMatrix(KE);

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= nv)
		return;

	int vid = tid;

	VertexFlags vflag = vflags[vid];
	if (vflag.is_fiction() || vflag.is_period_padding())
		return;

	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);

	indexer.locate(vid, vflag.get_gscolor(), gGsVertexEnd);

	int elementId = indexer.neighElement(0, gGsCellEnd, gGsCellReso).getId();

	if (elementId != -1) {
		CellFlags eflag = eflags[elementId];
		if (!eflag.is_fiction() && !eflag.is_period_padding()) {
			float prho;
			float pwn = exp_penal[0];
			prho = rhoPenalMin + powf(rholist[elementId], pwn);
			//float prho = rholist[elementId];
			float c = 0;
			//int neighVid[8];
			float u[8][3];
			float v[8][3];
			for (int i = 0; i < 8; i++) {
				int id = i % 2 + i / 2 % 2 * 3 + i / 4 * 9;
				int neighVid = indexer.neighVertex(id, gGsVertexEnd, gGsVertexReso).getId();
				if (neighVid == -1)
					print_exception;
				for (int k = 0; k < 3; k++) {
					u[i][k] = ug[k][neighVid];
					v[i][k] = vg[k][neighVid];
				}
			}
			for (int ki = 0; ki < 8; ki++) {
				int kirow = ki * 3;
				for (int kj = 0; kj < 8; kj++) {
					int kjcol = kj * 3;
					for (int ri = 0; ri < 3; ri++) {
						for (int cj = 0; cj < 3; cj++) {
							c += u[ki][ri] * KE[kirow + ri][kjcol + cj] * v[kj][cj] * prho;
						}
					}
				}
			}
			elementCompliance[elementId] = c;
		}
	}
}

double Grid::compliance(VT* ug[3], VT* vg[3]) {
	useGrid_g();
	auto cfg = make_kernel_param(n_gsvertices(), 256);
	auto Cebuffer = getTempPool().getBuffer(n_gscells() * sizeof(float));
	float* Ce = Cebuffer.template data<float>();
	init_array(Ce, 0.f, n_gscells());
	devArray_t<VT*, 3> u{ug[0], ug[1], ug[2]};
	devArray_t<VT*, 3> v{vg[0], vg[1], vg[2]};
	compliance_kernel<<<cfg.grid, cfg.block>>>(n_gsvertices(), u, v, rho_g, Ce, cellflag, vertflag);
	cudaDeviceSynchronize();
	cuda_error_check;

	{
		//std::vector<float> ce(n_gscells());
		//cudaMemcpy(ce.data(), Ce, sizeof(float) * n_gscells(), cudaMemcpyDeviceToHost);
		//array2matlab("ce", ce.data(), ce.size());
	}

	double C = dump_array_sum(Ce, n_gscells());
	return C;
}

void homo::Grid::v3_linear(VT a1, VT* v1g[3], VT a2, VT* v2g[3], VT* vg[3], int len /* =-1 */) {
	if (len == -1)
		len = n_gsvertices();
	auto cfg = make_kernel_param(len, 512);

	devArray_t<VT*, 3> v1{v1g[0], v1g[1], v1g[2]};
	devArray_t<VT*, 3> v2{v2g[0], v2g[1], v2g[2]};
	devArray_t<VT*, 3> v{vg[0], vg[1], vg[2]};

	auto ker = [=] __device__(int tid) {
		v[0][tid] = a1 * v1[0][tid] + a2 * v2[0][tid];
		v[1][tid] = a1 * v1[1][tid] + a2 * v2[1][tid];
		v[2][tid] = a1 * v1[2][tid] + a2 * v2[2][tid];
	};

	map<<<cfg.grid, cfg.block>>>(len, ker);
	cudaDeviceSynchronize();
	cuda_error_check;
}

//double homo::Grid::elasticTensorElement(int i, int j)
//{
//	//setMacroStrainDisplacement(i, u_g);
//	//v3_linear(1, u_g, -1, uchar_g[i], u_g);
//
//	//setMacroStrainDisplacement(j, f_g);
//	//v3_linear(1, f_g, -1, fchar_g[j], f_g);
//
//	//return compliance(u_g, f_g);
//}

template<typename T>
__global__ void set_macro_strain_displacement_kernel(
	int nv,
	int strain_i,
	VertexFlags* vflags,
	devArray_t<T*, 3> u) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nv)
		return;

	int vid = tid;
	VertexFlags vflag = vflags[vid];

	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);
	indexer.locate(vid, vflag.get_gscolor(), gGsVertexEnd);

	auto pos = indexer.getPos();

	if (!vflag.is_fiction()) {
		switch (strain_i) {
		case 0:
			u[0][vid] = pos.x;
			u[1][vid] = 0;
			u[2][vid] = 0;
			break;
		case 1:
			u[0][vid] = 0;
			u[1][vid] = pos.y;
			u[2][vid] = 0;
			break;
		case 2:
			u[0][vid] = 0;
			u[1][vid] = 0;
			u[2][vid] = pos.z;
			break;
		case 3: // yz
			u[0][vid] = 0;
			u[1][vid] = pos.z / 2.f;
			u[2][vid] = pos.y / 2.f;
			break;
		case 4: // xz
			u[0][vid] = pos.z / 2.f;
			u[1][vid] = 0;
			u[2][vid] = pos.x / 2.f;
			break;
		case 5: // xy
			u[0][vid] = pos.y / 2.f;
			u[1][vid] = pos.x / 2.f;
			u[2][vid] = 0;
			break;
		}
	}
}

void homo::Grid::setMacroStrainDisplacement(int i, VT* u[3]) {
	auto cfg = make_kernel_param(n_gsvertices(), 256);
	devArray_t<VT*, 3> ug{u[0], u[1], u[2]};
	VertexFlags* vflags = vertflag;
	set_macro_strain_displacement_kernel<<<cfg.grid, cfg.block>>>(n_gsvertices(), i, vflags, ug);
	cudaDeviceSynchronize();
	cuda_error_check;
}

template<typename T>
struct constVec {
	T val;
	constVec(T val_)
		: val(val_) {}
	__device__ constVec(const constVec& v2) = default;
	__device__ T operator()(size_t k) {
		return val;
	}
	__device__ T operator[](size_t k) const {
		return val;
	}
};

template<typename T>
__global__ void sensitivity_kernel(int nv,
								   int iStrain, int jStrain,
								   devArray_t<float*, 3> ui, devArray_t<float*, 3> uj,
								   T* rholist, VertexFlags* vflags, CellFlags* eflags,
								   float* elementSens, float volume) {
	__shared__ float KE[24][24];

	loadTemplateMatrix(KE);

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= nv)
		return;

	int vid = tid;

	VertexFlags vflag = vflags[vid];
	if (vflag.is_fiction() || vflag.is_period_padding())
		return;

	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);

	indexer.locate(vid, vflag.get_gscolor(), gGsVertexEnd);

	int elementId = indexer.neighElement(0, gGsCellEnd, gGsCellReso).getId();

	float vol = volume;

	if (elementId != -1) {
		CellFlags eflag = eflags[elementId];
		if (!eflag.is_fiction() && !eflag.is_period_padding()) {
			float pwn = exp_penal[0];
			float prho = pwn * powf(rholist[elementId], pwn - 1);
			//float prho = rholist[elementId];
			float c = 0;
			//int neighVid[8];
			float u[8][3];
			float v[8][3];
			for (int i = 0; i < 8; i++) {
				int id = i % 2 + i / 2 % 2 * 3 + i / 4 * 9;
				int neighVid = indexer.neighVertex(id, gGsVertexEnd, gGsVertexReso).getId();
				if (neighVid == -1)
					print_exception;
				//u[i][0] = i % 2
				elementMacroDisplacement(i, iStrain, u[i]);
				elementMacroDisplacement(i, jStrain, v[i]);
				u[i][0] -= ui[0][neighVid];
				u[i][1] -= ui[1][neighVid];
				u[i][2] -= ui[2][neighVid];
				v[i][0] -= uj[0][neighVid];
				v[i][1] -= uj[1][neighVid];
				v[i][2] -= uj[2][neighVid];
			}
			for (int ki = 0; ki < 8; ki++) {
				int kirow = ki * 3;
				for (int kj = 0; kj < 8; kj++) {
					int kjcol = kj * 3;
					for (int ri = 0; ri < 3; ri++) {
						for (int cj = 0; cj < 3; cj++) {
							c += u[ki][ri] * KE[kirow + ri][kjcol + cj] * v[kj][cj] * prho;
						}
					}
				}
			}
			elementSens[elementId] = c / vol;
		}
	}
}

void homo::Grid::sensitivity(int i, int j, float* sens) {
	// setMacroStrainDisplacement(i, u_g);
	// v3_linear(1, u_g, -1, uchar_g[i], u_g);
	// v3_copy(u_g, uchar_g[i]);

	// setMacroStrainDisplacement(j, f_g);
	// v3_linear(1, f_g, -1, fchar_g[j], f_g);
	// v3_copy(f_g, uchar_g[j]);
	NO_SUPPORT_ERROR;
}

// scatter per fine element matrix to coarse stencil
// stencil was organized in lexico order(No padding), and should be transferred to gs order
//template<int BlockSize = 256>
//__global__ void restrict_stencil_otf_kernel_1(
//	int ne, float* rholist, CellFlags* eflags,
//	devArray_t<int, 8> gsCellEnd, devArray_t<int, 3> CoarseCellReso
void homo::Grid::restrict_stencil(void) {
	if (is_root)
		return;
	if (fine->assemb_otf) {
		// fine->useGrid_g();
		useGrid_g();
		for (int i = 0; i < 27; i++) {
			// for (int j = 0; j < 9; j++) {
			// cudaMemset(stencil_g[i][j], 0, sizeof(float) * n_gsvertices());
			// }
			cudaMemset(stencil_g[i], 0, sizeof(glm::hmat3) * n_gsvertices());
		}
		cudaDeviceSynchronize();
		cuda_error_check;
		// auto cfg = make_kernel_param(fine->n_gscells(), 256);
		// restrict_stencil_otf_kernel_1 <<<cfg.grid, cfg.block>>> (fine->n_gscells(), fine->rho_g, fine->cellflag, fine->vertflag, fine->diag_strength);
		int nv = (cellReso[0] + 1) * (cellReso[1] + 1) * (cellReso[2] + 1);
		auto cfg = make_kernel_param(nv, 256);
		restrict_stencil_otf_aos_kernel_1<<<cfg.grid, cfg.block>>>(nv, fine->rho_g, fine->cellflag, fine->vertflag);
		cudaDeviceSynchronize();
		cuda_error_check;

		//lexistencil2matlab("Kl");

		useGrid_g();
		{
			//float* v[3] = { stencil_g[13][0], stencil_g[13][4], stencil_g[13][8] };
			//v3_toMatlab("st3", v);
		}
		lexiStencil2gsorder();
		//stencil2matlab("Kg");
		enforce_period_stencil(true);
		//stencil2matlab("Kp");
		{
			//float* v[3] = { stencil_g[13][0], stencil_g[13][4], stencil_g[13][8] };
			//v3_toMatlab("st3", v);
			//auto vidmap1 = getVertexLexidMap();
			//array2matlab("vidmap1", vidmap1.data(), vidmap1.size());
		}
	} else {
		useGrid_g();
		cudaDeviceSynchronize();
		cuda_error_check;
		int nvfine = fine->n_gsvertices();
		//printf("--\n");
		auto cfg = make_kernel_param(n_gsvertices(), 256);
		//restrict_stencil_kernel_1 <<<cfg.grid, cfg.block>>> (n_gsvertices(), nvfine, vertflag, fine->vertflag);
		restrict_stencil_aos_kernel_1<<<cfg.grid, cfg.block>>>(n_gsvertices(), nvfine, vertflag, fine->vertflag);
		cudaDeviceSynchronize();
		cuda_error_check;
		//stencil2matlab("Khost");
		enforce_period_stencil(false);
	}
}

void uploadTemplaceMatrix(const double* ke, float penal) {
	float fke[24 * 24];
	for (int i = 0; i < 24 * 24; i++)
		fke[i] = ke[i];
	cudaMemcpyToSymbol(gKE, fke, sizeof(gKE));
	cudaMemcpyToSymbol(gKEd, ke, sizeof(gKEd));
	cudaMemcpyToSymbol(exp_penal, &penal, sizeof(float));
}

void uploadTemplateLameMatrix(const char* kelam72, const char* kemu72, float Lam, float Mu) {
	short2 lammu[24][24];
	for (int i = 0; i < 24; i++) {
		for (int j = 0; j < 24; j++) {
			short2 lm{kelam72[i * 24 + j], kemu72[i * 24 + j]};
			lammu[i][j] = lm;
		}
	}
	if (sizeof(gKLame) != sizeof(lammu))
		print_exception;
	cudaMemcpyToSymbol(gKLame, &lammu[0][0], sizeof(lammu));
	Lam /= 72;
	Mu /= 72;
	cudaMemcpyToSymbol(LAM, &Lam, sizeof(Lam));
	cudaMemcpyToSymbol(MU, &Mu, sizeof(Mu));
	cuda_error_check;

	// LAM Mu set
	double lamu[5];
	lamu[0] = (Lam - 2 * Mu);
	lamu[1] = (Lam - Mu);
	lamu[2] = (Lam + Mu);
	lamu[3] = (Lam + 4 * Mu);
	lamu[4] = (2 * Lam + 5 * Mu);
	cudaMemcpyToSymbol(gLM, lamu, sizeof(gLM));
}

void uploadTemplateLameMatrix(const float* kelam, const float* kemu, float Lam, float Mu) {
	cudaMemcpyToSymbol(gKELame, &kelam[0], sizeof(gKELame));
	cudaMemcpyToSymbol(gKEMu, &kemu[0], sizeof(gKEMu));
	cudaMemcpyToSymbol(LAM, &Lam, sizeof(Lam));
	cudaMemcpyToSymbol(MU, &Mu, sizeof(Mu));
	cuda_error_check;
}

template<typename T>
__global__ void lexi2gsorder_kernel(T* src, T* dst,
									devArray_t<int, 3> srcreso, devArray_t<int, 8> gsEnd,
									bool srcpaded = false) {
	size_t n_src = srcreso[0] * srcreso[1] * srcreso[2];
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= n_src)
		return;
	// padded src pos
	int srcpos[3] = {tid % srcreso[0], tid / srcreso[0] % srcreso[1], tid / (srcreso[0] * srcreso[1])};
	// if not padding, add padding
	if (!srcpaded) {
		srcpos[0] += 1;
		srcpos[1] += 1;
		srcpos[2] += 1;
	}
	int gsorg[3] = {srcpos[0] % 2, srcpos[1] % 2, srcpos[2] % 2};
	int gscolor = gsorg[0] + gsorg[1] * 2 + gsorg[2] * 4;

	int gsreso[3] = {};
	for (int k = 0; k < 3; k++) {
		// last index - org / 2, should padd 1
		gsreso[k] = (srcreso[k] + 1 - gsorg[k]) / 2 + 1;
	}

	int setpos[3] = {srcpos[0] / 2, srcpos[1] / 2, srcpos[2] / 2};
	int setid = setpos[0] + setpos[1] * gsreso[0] + setpos[2] * gsreso[0] * gsreso[1];

	int gsid = setid + (gscolor == 0 ? 0 : gsEnd[gscolor - 1]);

	dst[gsid] = src[tid];
}

template<typename T>
void lexi2gsorder_imp(T* src, T* dst, Grid::LexiType type_,
					  std::array<int, 3> cellReso, int gsVertexSetEnd[8],
					  int gsCellSetEnd[8], bool lexipadded /*= false*/) {
	if (type_ == Grid::VERTEX) {
		devArray_t<int, 3> reso{cellReso[0] + 1, cellReso[1] + 1, cellReso[2] + 1};
		devArray_t<int, 8> gsend;
		int nv = reso[0] * reso[1] * reso[2];
		for (int k = 0; k < 8; k++)
			gsend[k] = gsVertexSetEnd[k];
		auto cfg = make_kernel_param(nv, 256);
		lexi2gsorder_kernel<<<cfg.grid, cfg.block>>>(src, dst, reso, gsend, lexipadded);
		cudaDeviceSynchronize();
		cuda_error_check;
	} else if (type_ == Grid::CELL) {
		devArray_t<int, 3> reso{cellReso[0], cellReso[1], cellReso[2]};
		devArray_t<int, 8> gsend;
		int ne = reso[0] * reso[1] * reso[2];
		for (int k = 0; k < 8; k++)
			gsend[k] = gsCellSetEnd[k];
		auto cfg = make_kernel_param(ne, 256);
		lexi2gsorder_kernel<<<cfg.grid, cfg.block>>>(src, dst, reso, gsend, lexipadded);
		cudaDeviceSynchronize();
		cuda_error_check;
	}
}

void homo::Grid::lexi2gsorder(float* src, float* dst, LexiType type_, bool lexipadded /*= false*/) {
	lexi2gsorder_imp(src, dst, type_, cellReso, gsVertexSetEnd, gsCellSetEnd, lexipadded);
}
void homo::Grid::lexi2gsorder(half* src, half* dst, LexiType type_, bool lexipadded /*= false*/) {
	lexi2gsorder_imp(src, dst, type_, cellReso, gsVertexSetEnd, gsCellSetEnd, lexipadded);
}
void homo::Grid::lexi2gsorder(glm::hmat3* src, glm::hmat3* dst, LexiType type_, bool lexipadded /*= false*/) {
	lexi2gsorder_imp(src, dst, type_, cellReso, gsVertexSetEnd, gsCellSetEnd, lexipadded);
}

void homo::Grid::lexiStencil2gsorder(void) {
	auto tmpname = getMem().addBuffer(n_gsvertices() * sizeof(glm::hmat3));
	glm::hmat3* tmp = getMem().getBuffer(tmpname)->data<glm::hmat3>();
	for (int i = 0; i < 27; i++) {
		cudaMemset(tmp, 0, sizeof(glm::hmat3) * n_gsvertices());
		cudaDeviceSynchronize();
		cuda_error_check;
		lexi2gsorder(stencil_g[i], tmp, VERTEX);
		cudaMemcpy(stencil_g[i], tmp, sizeof(glm::hmat3) * n_gsvertices(), cudaMemcpyDeviceToDevice);
	}
	getMem().deleteBuffer(tmpname);
	cuda_error_check;
}

template<int BlockSize = 256>
__global__ void enforce_period_stencil_subst_kernel(void) {

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int vreso[3] = {gGridCellReso[0] + 1, gGridCellReso[1] + 1, gGridCellReso[2] + 1};

	int gsid_min = -1;
	int gsid_max = -1;

	do {
		// down - up
		int du_end = vreso[0] * vreso[1];
		if (tid < du_end) {
			int vid = tid;
			int pos[3] = {vid % vreso[0], vid / vreso[0], 0};
			gsid_min = lexi2gs(pos, gGsVertexReso, gGsVertexEnd);
			pos[2] = vreso[2] - 1;
			gsid_max = lexi2gs(pos, gGsVertexReso, gGsVertexEnd);
			break;
		}

		// left - right
		int lr_end = du_end + vreso[1] * vreso[2];
		if (tid < lr_end) {
			int vid = tid - du_end;
			int pos[3] = {0, vid % vreso[1], vid / vreso[1]};
			gsid_min = lexi2gs(pos, gGsVertexReso, gGsVertexEnd);
			pos[0] = vreso[0] - 1;
			gsid_max = lexi2gs(pos, gGsVertexReso, gGsVertexEnd);
			break;
		}

		// near - far
		int nf_end = lr_end + vreso[0] * vreso[2];
		if (tid < nf_end) {
			int vid = tid - lr_end;
			int pos[3] = {vid % vreso[0], 0, vid / vreso[0]};
			gsid_min = lexi2gs(pos, gGsVertexReso, gGsVertexEnd);
			pos[1] = vreso[1] - 1;
			gsid_max = lexi2gs(pos, gGsVertexReso, gGsVertexEnd);
			break;
		}
	} while (0);

	if (gsid_min != -1 && gsid_max != -1) {
		for (int i = 0; i < 27; i++) {
			for (int j = 0; j < 9; j++) {
				rxstencil[i][j][gsid_max] = rxstencil[i][j][gsid_min];
			}
		}
	}
}

template<typename T, int N>
__global__ void pad_vertex_data_kernel(int nvfacepad, int nvedgepadd, devArray_t<T*, N> v, VertexFlags* vflags);

template<typename T, int N>
void pad_vertex_data_imp(T** v, std::array<int, 3> cellReso, VertexFlags* vertflag) {
	int nvpadface = (cellReso[0] + 1) * (cellReso[1] + 1) +
					(cellReso[1] + 1) * (cellReso[2] + 1) +
					(cellReso[0] + 1) * (cellReso[2] + 1);
	int nvpadedge = 2 * ((cellReso[0] + 3) * (cellReso[1] + 3) - (cellReso[0] + 1) * (cellReso[1] + 1)) +
					4 * (cellReso[2] + 1);
	devArray_t<T*, N> arr(v);
	auto cfg = make_kernel_param(nvpadface + nvpadedge, 256);
	pad_vertex_data_kernel<<<cfg.grid, cfg.block>>>(nvpadface, nvpadedge, arr, vertflag);
	cudaDeviceSynchronize();
	cuda_error_check;
}

void homo::Grid::enforce_period_stencil(bool additive) {
	useGrid_g();
	for (int i = 0; i < 27; i++) {
		// for (int j = 0; j < 9; j+=3) {
		// 	half* st[3] = { stencil_g[i][j],stencil_g[i][j + 1],stencil_g[i][j + 2] };
		// 	enforce_period_vertex(st, additive);
		// 	pad_vertex_data(st);
		// }
		enforce_period_vertex(stencil_g[i], additive);
	}
	if (fine->is_root) {
		restrict_stencil_arround_dirichelt_boundary();
	}
	pad_vertex_data_imp<glm::hmat3, 27>(stencil_g, cellReso, vertflag);
}

template<typename Flag>
__global__ void gsid2pos_kernel(int n, Flag* flags, devArray_t<int*, 3> pos, int off = -1) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= n) {
		return;
	}
	constexpr bool isVertexId = std::is_same_v<std::decay_t<Flag>, VertexFlags>;
	constexpr bool isCellId = std::is_same_v<std::decay_t<Flag>, CellFlags>;
	if (isVertexId) {
		Flag vflag = flags[tid];
		if (!vflag.is_fiction()) {
			int p[3];
			gsid2pos(tid, vflag.get_gscolor(), gGsVertexReso, gGsVertexEnd, p);
			if (off < 0 || off > 2)
				for (int i = 0; i < 3; i++)
					pos[i][tid] = p[i];
			else
				pos[off][tid] = p[off];
		}
	} // vertex id
	else if (isCellId) {
		Flag eflag = flags[tid];
		if (!eflag.is_fiction()) {
			int p[3];
			gsid2pos(tid, eflag.get_gscolor(), gGsCellReso, gGsCellEnd, p);
			if (off < 0 || off > 2)
				for (int i = 0; i < 3; i++)
					pos[i][tid] = p[i];
			else
				pos[off][tid] = p[off];
		}
	} // element id
	else {
		if (tid == 0) {
			printf("\033[31mno such flags type at grid.cu, line %d\033[0m\n", __LINE__);
		}
	}
}

void homo::Grid::getGsVertexPos(std::vector<int> hostpos[3]) {
	useGrid_g();
	devArray_t<int*, 3> pos;
	auto cfg = make_kernel_param(n_gsvertices(), 256);
	for (int i = 0; i < 3; i++) {
		// auto buffer = getTempBuffer(sizeof(int) * n_gsvertices());
		// pos[i] = buffer.data<int>();
		cudaMallocManaged(&pos[i], sizeof(int) * n_gsvertices());
		init_array(pos[i], -2, n_gsvertices());
		// ...
		gsid2pos_kernel<<<cfg.grid, cfg.block>>>(n_gsvertices(), vertflag, pos, i);
		cudaDeviceSynchronize();
		cuda_error_check;
		hostpos[i].resize(n_gsvertices());
		cudaMemcpy(hostpos[i].data(), pos[i], sizeof(int) * n_gsvertices(), cudaMemcpyDeviceToHost);
		cudaFree(pos[i]);
	}
}

void homo::Grid::getGsElementPos(std::vector<int> hostpos[3]) {
	useGrid_g();
	devArray_t<int*, 3> pos;
	auto cfg = make_kernel_param(n_gscells(), 256);
	for (int i = 0; i < 3; i++) {
		// auto buffer = getTempBuffer(sizeof(int) * n_gscells());
		// pos[i] = buffer.data<int>();
		cudaMallocManaged(&pos[i], sizeof(int) * n_gscells());
		init_array(pos[i], -2, n_gscells());
		// ...
		gsid2pos_kernel<<<cfg.grid, cfg.block>>>(n_gscells(), cellflag, pos, i);
		cudaDeviceSynchronize();
		cuda_error_check;
		hostpos[i].resize(n_gscells());
		cudaMemcpy(hostpos[i].data(), pos[i], sizeof(int) * n_gscells(), cudaMemcpyDeviceToHost);
		cudaFree(pos[i]);
	}
}

void homo::Grid::getDensity(std::vector<float>& rho, bool lexiOrder /*= false*/) {
	rho.resize(n_gscells());
	if (std::is_same_v<float, RhoT>) {
		cudaMemcpy(rho.data(), rho_g, sizeof(float) * rho.size(), cudaMemcpyDeviceToHost);
	} else {
		auto tmpbuf = getTempBuffer(sizeof(float) * n_gscells());
		auto* tmpdata = tmpbuf.data<float>();
		type_cast(tmpdata, rho_g, rho.size());
		cudaMemcpy(rho.data(), tmpdata, sizeof(float) * rho.size(), cudaMemcpyDeviceToHost);
	}
	if (!lexiOrder)
		return;
	else
		throw std::runtime_error("not implemented"); // toDO
}

template<typename T, int N>
__global__ void enforce_period_boundary_vertex_kernel(int siz, devArray_t<T*, N> v, VertexFlags* vflags, bool additive = false) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= siz)
		return;
	int pos[3] = {-2, -2, -2};
	int ereso[3] = {gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]};

	//bool debug = false;

	do {
		if (tid < ereso[0] * ereso[1]) {
			pos[0] = tid % ereso[0];
			pos[1] = tid / ereso[0];
			pos[2] = 0;
			break;
		}
		tid -= ereso[0] * ereso[1];
		if (tid < ereso[1] * (ereso[2] - 1)) {
			pos[0] = 0;
			pos[1] = tid % ereso[1];
			pos[2] = tid / ereso[1] + 1;
			break;
		}
		tid -= ereso[1] * (ereso[2] - 1);
		if (tid < (ereso[0] - 1) * (ereso[2] - 1)) {
			pos[0] = tid % (ereso[0] - 1) + 1;
			pos[1] = 0;
			pos[2] = tid / (ereso[0] - 1) + 1;
			break;
		}
	} while (0);
	if (pos[0] <= -2 || pos[1] <= -2 || pos[2] <= -2)
		return;

	//if (pos[0] == 0 && pos[1] == 7 && pos[2] == 0) debug = true;

	int gsid = lexi2gs(pos, gGsVertexReso, gGsVertexEnd);
	VertexFlags vflag = vflags[gsid];
	T val[N] = {/*v[0][gsid],v[1][gsid],v[2][gsid]*/};
	int op_ids[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
	{
		// sum opposite
		//if (debug) printf("vflag = %04x  gsid = %d   edid = %d\n", vflag.flagbits, gsid, gGsVertexEnd[vflag.get_gscolor()]);

		int op_pos[3];
		for (int i = 0; i < vflag.is_set(LEFT_BOUNDARY) + 1; i++) {
			op_pos[0] = pos[0];
			if (i)
				op_pos[0] += ereso[0];
			for (int j = 0; j < vflag.is_set(NEAR_BOUNDARY) + 1; j++) {
				op_pos[1] = pos[1];
				if (j)
					op_pos[1] += ereso[1];
				for (int k = 0; k < vflag.is_set(DOWN_BOUNDARY) + 1; k++) {
					op_pos[2] = pos[2];
					if (k)
						op_pos[2] += ereso[2];
					int op_id = lexi2gs(op_pos, gGsVertexReso, gGsVertexEnd);
					op_ids[i * 4 + j * 2 + k] = op_id;
					if (additive)
						for (int m = 0; m < N; m++)
							val[m] += v[m][op_id];
				}
			}
		}
	}

	// enforce period boundary
	for (int i = 0; i < 8; i++) {
		if (op_ids[i] != -1) {
			if (additive)
				for (int j = 0; j < N; j++)
					v[j][op_ids[i]] = val[j];
			else
				for (int j = 0; j < N; j++)
					v[j][op_ids[i]] = v[j][gsid];
		}
	}
	if (additive) {
		for (int j = 0; j < N; j++)
			v[j][gsid] = val[j];
	}
}

template<typename T, int N>
void enforce_period_vertex_imp(T** v, std::array<int, 3> cellReso, VertexFlags* vertflag, bool additive /*= false*/) {
	int nvdup = cellReso[0] * cellReso[1] + cellReso[1] * (cellReso[2] - 1) + (cellReso[0] - 1) * (cellReso[2] - 1);
	devArray_t<T*, N> varr(v);
	auto cfg = make_kernel_param(nvdup, 256);
	enforce_period_boundary_vertex_kernel<<<cfg.grid, cfg.block>>>(nvdup, varr, vertflag, additive);
	cudaDeviceSynchronize();
	cuda_error_check;
}

void homo::Grid::enforce_period_vertex(double* v[3], bool additive /*= false*/) {
	enforce_period_vertex_imp<double, 3>(v, cellReso, vertflag, additive);
}

void homo::Grid::enforce_period_vertex(half* v[3], bool additive /*= false*/) {
	enforce_period_vertex_imp<half, 3>(v, cellReso, vertflag, additive);
}

void homo::Grid::enforce_period_vertex(glm::hmat3* v, bool additive /*= false*/) {
	glm::hmat3* varr[1] = {v};
	enforce_period_vertex_imp<glm::hmat3, 1>(varr, cellReso, vertflag, additive);
}

void homo::Grid::enforce_period_vertex(float* v[3], bool additive /*= false*/) {
	enforce_period_vertex_imp<float, 3>(v, cellReso, vertflag, additive);
}

template<typename T, int N>
__global__ void pad_vertex_data_kernel(int nvfacepad, int nvedgepadd, devArray_t<T*, N> v, VertexFlags* vflags) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nvfacepad + nvedgepadd)
		return;
	int ereso[3] = {gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]};

	//bool debug = false;

	int boundaryType = -1;

	if (tid < nvfacepad) {
		int pos[3] = {-2, -2, -2};
		// padd face
		do {
			if (tid < (ereso[0] + 1) * (ereso[1] + 1)) {
				pos[0] = tid % (ereso[0] + 1);
				pos[1] = tid / (ereso[0] + 1);
				pos[2] = 0;
				boundaryType = 0;
				break;
			}
			tid -= (ereso[0] + 1) * (ereso[1] + 1);
			if (tid < (ereso[1] + 1) * (ereso[2] + 1)) {
				pos[0] = 0;
				pos[1] = tid % (ereso[1] + 1);
				pos[2] = tid / (ereso[1] + 1);
				boundaryType = 1;
				break;
			}
			tid -= (ereso[1] + 1) * (ereso[2] + 1);
			if (tid < (ereso[0] + 1) * (ereso[2] + 1)) {
				pos[0] = tid % (ereso[0] + 1);
				pos[1] = 0;
				pos[2] = tid / (ereso[0] + 1);
				boundaryType = 2;
				break;
			}
		} while (0);
		if (pos[0] <= -2 || pos[1] <= -2 || pos[2] <= -2)
			return;

		int gsid = lexi2gs(pos, gGsVertexReso, gGsVertexEnd);
		VertexFlags vflag = vflags[gsid]; // padding
		if (boundaryType == 1) {
			for (int i : {-1, 1}) {
				int p[3] = {pos[0] + i, pos[1], pos[2]};
				int q[3] = {pos[0] + ereso[0] + i, pos[1], pos[2]};
				int pid = lexi2gs(p, gGsVertexReso, gGsVertexEnd);
				int qid = lexi2gs(q, gGsVertexReso, gGsVertexEnd);
				if (i == -1)
					for (int j = 0; j < N; j++)
						v[j][pid] = v[j][qid];
				if (i == 1)
					for (int j = 0; j < N; j++)
						v[j][qid] = v[j][pid];
			}
		}
		if (boundaryType == 2) {
			for (int i : {-1, 1}) {
				int p[3] = {pos[0], pos[1] + i, pos[2]};
				int q[3] = {pos[0], pos[1] + ereso[1] + i, pos[2]};
				int pid = lexi2gs(p, gGsVertexReso, gGsVertexEnd);
				int qid = lexi2gs(q, gGsVertexReso, gGsVertexEnd);
				if (i == -1)
					for (int j = 0; j < N; j++)
						v[j][pid] = v[j][qid];
				if (i == 1)
					for (int j = 0; j < N; j++)
						v[j][qid] = v[j][pid];
			}
		}
		if (boundaryType == 0) {
			for (int i : {-1, 1}) {
				int p[3] = {pos[0], pos[1], pos[2] + i};
				int q[3] = {pos[0], pos[1], pos[2] + ereso[2] + i};
				int pid = lexi2gs(p, gGsVertexReso, gGsVertexEnd);
				int qid = lexi2gs(q, gGsVertexReso, gGsVertexEnd);
				//if (debug) {
				//	printf("i = %d  p = (%d %d %d)  q = (%d %d %d)  pid = %d  qid = %d\n", i, p[0], p[1], p[2], q[0], q[1], q[2], pid, qid);
				//}
				if (i == -1)
					for (int j = 0; j < N; j++)
						v[j][pid] = v[j][qid];
				if (i == 1)
					for (int j = 0; j < N; j++)
						v[j][qid] = v[j][pid];
			}
			//if (debug) {
			//	for (int i : {-1, 1}) {
			//		int p[3] = { pos[0] , pos[1] , pos[2] + i };
			//		int q[3] = { pos[0] , pos[1] , pos[2] + ereso[2] + i };
			//		int pid = lexi2gs(p, gGsVertexReso, gGsVertexEnd);
			//		int qid = lexi2gs(q, gGsVertexReso, gGsVertexEnd);
			//		printf("i = %d  vp = (%e, %e, %e)  vq = (%e, %e, %e)\n", i, v[0][pid], v[1][pid], v[2][pid]
			//			, v[0][qid], v[1][qid], v[2][qid]);
			//	}
			//}
		}
	} else if (tid - nvfacepad < nvedgepadd) {
		bool debug = false;
		// padd edge
		int id = tid - nvfacepad;
		int nv_bot = (ereso[0] + 3) * (ereso[1] + 3) - (ereso[0] + 1) * (ereso[1] + 1);
		int po[3] = {0, 0, 0};
		if (id < 2 * nv_bot) {
			po[2] = id / nv_bot * (ereso[2] + 2);
			id = id % nv_bot;
			if (id < 2 * (ereso[0] + 3)) {
				po[0] = id % (ereso[0] + 3);
				po[1] = id / (ereso[0] + 3) * (ereso[1] + 2);
			} else {
				id -= 2 * (ereso[0] + 3);
				po[0] = id / (ereso[1] + 1) * (ereso[0] + 2);
				po[1] = id % (ereso[1] + 1) + 1;
			}
		} else {
			id -= 2 * nv_bot;
			int hid = id / (ereso[2] + 1);
			int vid = id % (ereso[2] + 1);
			po[0] = hid % 2 * (ereso[0] + 2);
			po[1] = hid / 2 * (ereso[1] + 2);
			po[2] = vid + 1;
		}
		po[0] -= 1;
		po[1] -= 1;
		po[2] -= 1;
		int op_pos[3];
		for (int i = 0; i < 3; i++) {
			op_pos[i] = (po[i] + ereso[i]) % ereso[i];
		}
		int myid = lexi2gs(po, gGsVertexReso, gGsVertexEnd);
		int opid = lexi2gs(op_pos, gGsVertexReso, gGsVertexEnd);
		for (int i = 0; i < N; i++)
			v[i][myid] = v[i][opid];
	}
}

void homo::Grid::pad_vertex_data(double* v[3]) {
	pad_vertex_data_imp<double, 3>(v, cellReso, vertflag);
}

template<typename T>
__global__ void enforce_period_element_kernel(int siz, T* celldata, CellFlags* eflags) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= siz)
		return;

	int ereso[3] = {gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]};

	do {
		if (tid < ereso[0] * ereso[1]) {
			for (int off : {-1, 0}) {
				int p[3] = {tid % ereso[0], tid / ereso[0], off};
				int q[3] = {p[0], p[1], ereso[2] + off};
				int pid = lexi2gs(p, gGsCellReso, gGsCellEnd);
				int qid = lexi2gs(q, gGsCellReso, gGsCellEnd);
				if (off == -1)
					celldata[pid] = celldata[qid];
				else
					celldata[qid] = celldata[pid];
			}
			break;
		}
		tid -= ereso[0] * ereso[1];
		if (tid < ereso[1] * ereso[2]) {
			for (int off : {-1, 0}) {
				int p[3] = {off, tid % ereso[1], tid / ereso[1]};
				int q[3] = {ereso[0] + off, p[1], p[2]};
				int pid = lexi2gs(p, gGsCellReso, gGsCellEnd);
				int qid = lexi2gs(q, gGsCellReso, gGsCellEnd);
				if (off == -1)
					celldata[pid] = celldata[qid];
				else
					celldata[qid] = celldata[pid];
			}
			break;
		}
		tid -= ereso[1] * ereso[2];
		if (tid < ereso[0] * ereso[2]) {
			for (int off : {-1, 0}) {
				int p[3] = {tid % ereso[0], off, tid / ereso[0]};
				int q[3] = {p[0], ereso[1] + off, p[2]};
				int pid = lexi2gs(p, gGsCellReso, gGsCellEnd);
				int qid = lexi2gs(q, gGsCellReso, gGsCellEnd);
				if (off == -1)
					celldata[pid] = celldata[qid];
				else
					celldata[qid] = celldata[pid];
			}
			break;
		}
	} while (0);
}

void homo::Grid::enforce_period_element(float* celldata) {
	int nedup = cellReso[0] * cellReso[1] +
				cellReso[1] * cellReso[2] +
				cellReso[0] * cellReso[2];
	auto cfg = make_kernel_param(nedup, 256);
	enforce_period_element_kernel<<<cfg.grid, cfg.block>>>(nedup, celldata, cellflag);
	cudaDeviceSynchronize();
	cuda_error_check;
}
void homo::Grid::enforce_period_element(half* celldata) {
	int nedup = cellReso[0] * cellReso[1] +
				cellReso[1] * cellReso[2] +
				cellReso[0] * cellReso[2];
	auto cfg = make_kernel_param(nedup, 256);
	enforce_period_element_kernel<<<cfg.grid, cfg.block>>>(nedup, celldata, cellflag);
	cudaDeviceSynchronize();
	cuda_error_check;
}

__global__ void vertexlexid_kernel(int* plexid) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ereso[3] = {gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]};
	int nv = (ereso[0] + 1) * (ereso[1] + 1) * (ereso[2] + 1);
	if (tid >= nv)
		return;
	int pos[3];
	pos[0] = tid % (ereso[0] + 1);
	pos[1] = tid / (ereso[0] + 1) % (ereso[1] + 1);
	pos[2] = tid / (ereso[0] + 1) / (ereso[1] + 1);
	int gsid = lexi2gs(pos, gGsVertexReso, gGsVertexEnd);
	plexid[tid] = gsid;
}

std::vector<int> homo::Grid::getVertexLexidMap(void) {
	useGrid_g();
	auto tmpname = getMem().addBuffer(n_gsvertices() * sizeof(float));
	int* tmp = getMem().getBuffer(tmpname)->data<int>();

	int nv = (cellReso[0] + 1) * (cellReso[1] + 1) * (cellReso[2] + 1);
	auto cfg = make_kernel_param(nv, 256);
	std::vector<int> vidmap(nv);

	vertexlexid_kernel<<<cfg.grid, cfg.block>>>(tmp);
	cudaDeviceSynchronize();
	cuda_error_check;

	cudaMemcpy(vidmap.data(), tmp, sizeof(int) * nv, cudaMemcpyDeviceToHost);

	getMem().deleteBuffer(tmp);

	return vidmap;
}

__global__ void celllexid_kernel(int* plexid) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ereso[3] = {gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]};
	int ne = ereso[0] * ereso[1] * ereso[2];
	if (tid >= ne)
		return;
	int pos[3];
	pos[0] = tid % (ereso[0]);
	pos[1] = tid / (ereso[0]) % (ereso[1]);
	pos[2] = tid / (ereso[0]) / (ereso[1]);
	int gsid = lexi2gs(pos, gGsCellReso, gGsCellEnd);
	plexid[tid] = gsid;
}

std::vector<int> homo::Grid::getCellLexidMap(void) {
	useGrid_g();
	auto tmpname = getMem().addBuffer(n_gscells() * sizeof(float));
	int* tmp = getMem().getBuffer(tmpname)->data<int>();

	int ne = cellReso[0] * cellReso[1] * cellReso[2];
	auto cfg = make_kernel_param(ne, 256);
	std::vector<int> eidmap(ne);

	celllexid_kernel<<<cfg.grid, cfg.block>>>(tmp);
	cudaDeviceSynchronize();
	cuda_error_check;

	cudaMemcpy(eidmap.data(), tmp, sizeof(int) * ne, cudaMemcpyDeviceToHost);

	getMem().deleteBuffer(tmp);

	return eidmap;
}

template<typename T>
void enforce_dirichlet_boundary_imp(T* v[3], std::array<int, 3> cellReso, int gsVertexReso[3][8], int gsVertexSetEnd[8]) {
	cudaDeviceSynchronize();
	cuda_error_check;
	int pos[3];
	for (int i = 0; i < 2; i++) {
		pos[0] = i * cellReso[0];
		for (int j = 0; j < 2; j++) {
			pos[1] = j * cellReso[1];
			for (int k = 0; k < 2; k++) {
				pos[2] = k * cellReso[2];
				int gsid = lexi2gs(pos, gsVertexReso, gsVertexSetEnd);
				//printf("gsid = %d\n", gsid);
				for (int n = 0; n < 3; n++)
					cudaMemset(v[n] + gsid, 0, sizeof(T));
			}
		}
	}
	cudaDeviceSynchronize();
	cuda_error_check;
}

void homo::Grid::enforce_dirichlet_boundary(float* v[3]) {
	enforce_dirichlet_boundary_imp(v, cellReso, gsVertexReso, gsVertexSetEnd);
}

void homo::Grid::enforce_dirichlet_boundary(half* v[3]) {
	enforce_dirichlet_boundary_imp(v, cellReso, gsVertexReso, gsVertexSetEnd);
}

template<typename T, typename Tout, int BlockSize = 256>
__global__ void v3_dot_kernel(int nv,
							  VertexFlags* vflags,
							  devArray_t<T*, 3> vlist, devArray_t<T*, 3> ulist, Tout* p_out, bool removePeriodDof = false) {

	__shared__ T blocksum[BlockSize / 32];

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t stride = blockDim.x * gridDim.x;
	bool fiction = false;
	if (tid >= nv)
		fiction = true;

	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;

	size_t baseId = 0;
	Tout bsum = 0;
	for (; baseId + tid < nv; baseId += stride) {
		int vid = baseId + tid;
		VertexFlags vflag = vflags[vid];
		if (vflag.is_fiction())
			continue;
		if (removePeriodDof && (vflag.is_period_padding() || vflag.is_max_boundary()))
			continue;

		T v[3] = {vlist[0][vid], vlist[1][vid], vlist[2][vid]};
		T u[3] = {ulist[0][vid], ulist[1][vid], ulist[2][vid]};
		Tout uv = v[0] * u[0] + v[1] * u[1] + v[2] * u[2];
		bsum += uv;
	}

	// warp reduce
	for (int offset = 16; offset > 0; offset /= 2) {
		bsum += shfl_down(bsum, offset);
	}

	if (laneId == 0) {
		blocksum[warpId] = bsum;
	}

	__syncthreads();

	// block reduce
	if (warpId == 0) {
		if (BlockSize / 32 > 32) {
			print_exception;
		}

		if (threadIdx.x < BlockSize / 32) {
			bsum = blocksum[threadIdx.x];
		} else {
			bsum = 0;
		}
		for (int offset = 16; offset > 0; offset /= 2) {
			bsum += shfl_down(bsum, offset);
		}
		if (laneId == 0) {
			p_out[blockIdx.x] = bsum;
		}
	}
}

float homo::Grid::v3_dot(VT* v[3], VT* u[3], bool removePeriodDof /*= false*/, int len /*= -1*/) {
	if (len == -1)
		len = n_gsvertices();
	int szTemp = len * sizeof(float) / 100;
	if (!removePeriodDof) {
		auto buffer = getTempPool().getBuffer(szTemp);
		auto pTemp = buffer.template data<float>();
		double result = dot(v[0], v[1], v[2], u[0], u[1], u[2], pTemp, len);
		cuda_error_check;
		return result;
	} else {
		devArray_t<VT*, 3> vlist{v[0], v[1], v[2]};
		devArray_t<VT*, 3> ulist{u[0], u[1], u[2]};
		int nv = n_gsvertices();
		auto buffer = getTempBuffer(nv / 100 * sizeof(float));
		float* p_tmp = buffer.template data<float>();
		int batch = nv;
		auto cfg = make_kernel_param(batch, 256);
		v3_dot_kernel<<<cfg.grid, cfg.block>>>(nv, vertflag, vlist, ulist, p_tmp, removePeriodDof);
		cudaDeviceSynchronize();
		double s = dump_array_sum(p_tmp, cfg.grid);
		cuda_error_check;
		return s;
	}
}

void homo::Grid::pad_vertex_data(float* v[3]) {
	pad_vertex_data_imp<float, 3>(v, cellReso, vertflag);
}

void homo::Grid::pad_vertex_data(half* v[3]) {
	pad_vertex_data_imp<half, 3>(v, cellReso, vertflag);
}

void homo::Grid::pad_vertex_data(glm::hmat3* st) {
	pad_vertex_data_imp<glm::hmat3, 1>(&st, cellReso, vertflag);
}

template<typename T, typename Rho>
__global__ void v3_stencilOnLeft_kernel(
	int nv,
	Rho* rholist,
	devArray_t<T*, 3> v, devArray_t<T*, 3> Kv,
	devArray_t<int, 3> gridCellReso,
	VertexFlags* vflags, CellFlags* eflags) {
	__shared__ int gsCellReso[3][8];
	__shared__ int gsVertexReso[3][8];
	__shared__ int gsCellEnd[8];
	__shared__ int gsVertexEnd[8];
	__shared__ Lame KLAME[24][24];

	__shared__ float sumKeU[3][4][32];

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;

	// local vertex id in gs set
	int vid = blockIdx.x * 32 + laneId;

	bool fiction = false;

	fiction = fiction || vid >= nv;
	VertexFlags vflag;
	if (!fiction) {
		vflag = vflags[vid];
		fiction = fiction || vflag.is_fiction();
	}
	int set_id = vflag.get_gscolor();

	// load lame matrix
	loadLameMatrix(KLAME);
	//float Lam = LAM[0];
	//float Mu = MU[0];

	// load cell and vertex reso
	constant2DToShared(gGsCellReso, gsCellReso);
	constant2DToShared(gGsVertexReso, gsVertexReso);
	constantToShared(gGsCellEnd, gsCellEnd);
	constantToShared(gGsVertexEnd, gsVertexEnd);
	initSharedMem(&sumKeU[0][0][0], sizeof(sumKeU) / sizeof(float));

	__syncthreads();

	GridVertexIndex indexer(gridCellReso[0], gridCellReso[1], gridCellReso[2]);
	if (!fiction) {
		indexer.locate(vid, vflag.get_gscolor(), gsVertexEnd);
	}

	float KlamU[3] = {0.};
	float KmuU[3] = {0.};
	float KeU[3] = {0.};

	int elementId = -1;
	if (!fiction)
		elementId = indexer.neighElement(warpId, gsCellEnd, gsCellReso).getId();
	int vselfrow = (7 - warpId) * 3;
	float rho_penal = 0;
	CellFlags eflag;
	float penal = exp_penal[0];
	if (elementId != -1) {
		eflag = eflags[elementId];
		if (!eflag.is_fiction())
			rho_penal = rhoPenalMin + powf(float(rholist[elementId]), penal);
	}

	if (elementId != -1 && !eflag.is_fiction() && !vflag.is_fiction() && !vflag.is_period_padding()) {
		for (int i = 0; i < 8; i++) {
			int vneigh =
				(warpId % 2 + i % 2) +
				(warpId / 2 % 2 + i / 2 % 2) * 3 +
				(warpId / 4 + i / 4) * 9;
			int vneighId = indexer.neighVertex(vneigh, gsVertexEnd, gsVertexReso).getId();
			VertexFlags nvflag;
			if (vneighId != -1) {
				nvflag = vflags[vneighId];
				if (!nvflag.is_fiction()) {
					float u[3] = {v[0][vneighId], v[1][vneighId], v[2][vneighId]};
					if (nvflag.is_dirichlet_boundary()) {
						u[0] = u[1] = u[2] = 0;
					}
					for (int k3row = 0; k3row < 3; k3row++) {
						for (int k3col = 0; k3col < 3; k3col++) {
							KlamU[k3row] += KLAME[vselfrow + k3row][i * 3 + k3col].lam() * u[k3col];
							KmuU[k3row] += KLAME[vselfrow + k3row][i * 3 + k3col].mu() * u[k3col];
						}
					}
					float lam = LAM[0], mu = MU[0];
					KlamU[0] *= lam;
					KlamU[1] *= lam;
					KlamU[2] *= lam;
					KmuU[0] *= mu;
					KmuU[1] *= mu;
					KmuU[2] *= mu;
					KeU[0] = (KlamU[0] + KmuU[0]) * rho_penal;
					KeU[1] = (KlamU[1] + KmuU[1]) * rho_penal;
					KeU[2] = (KlamU[2] + KmuU[2]) * rho_penal;
				}
			}
		}
	}

	if (warpId >= 4) {
		for (int i = 0; i < 3; i++) {
			sumKeU[i][warpId - 4][laneId] = KeU[i];
		}
	}
	__syncthreads();

	if (warpId < 4) {
		for (int i = 0; i < 3; i++) {
			sumKeU[i][warpId][laneId] += KeU[i];
		}
	}
	__syncthreads();

	if (warpId < 2) {
		for (int i = 0; i < 3; i++) {
			sumKeU[i][warpId][laneId] += sumKeU[i][warpId + 2][laneId];
		}
	}
	__syncthreads();

	if (warpId < 1 && !fiction && !vflag.is_period_padding()) {
		for (int i = 0; i < 3; i++) {
			KeU[i] = sumKeU[i][warpId][laneId] + sumKeU[i][warpId + 1][laneId];
		}

		float kv[3] = {KeU[0], KeU[1], KeU[2]};

		if (vflag.is_dirichlet_boundary()) {
			kv[0] = kv[1] = kv[2] = 0;
		}

		Kv[0][vid] = kv[0];
		Kv[1][vid] = kv[1];
		Kv[2][vid] = kv[2];
	}
}

void homo::Grid::v3_stencilOnLeft(VT* v[3], VT* Kv[3]) {
	if (!is_root) {
		return;
	}
	useGrid_g();
	devArray_t<VT*, 3> varr{v[0], v[1], v[2]};
	devArray_t<VT*, 3> Kvarr{Kv[0], Kv[1], Kv[2]};

	devArray_t<int, 3> gridCellReso{cellReso[0], cellReso[1], cellReso[2]};
	VertexFlags* vflags = vertflag;
	CellFlags* eflags = cellflag;
	auto cfg = make_kernel_param(n_gsvertices() * 8, 32 * 8);
	v3_stencilOnLeft_kernel<<<cfg.grid, cfg.block>>>(n_gsvertices(), rho_g, varr, Kvarr, gridCellReso, vflags, eflags);
	cudaDeviceSynchronize();
	cuda_error_check;
	pad_vertex_data(Kv);
}

std::string homo::Grid::checkDeviceError(void) {
	cudaDeviceSynchronize();
	auto err = cudaGetLastError();
	if (err != 0) {
		return cudaGetErrorName(err);
	} else {
		return "";
	}
}

template<typename T>
__global__ void v3_removeT_kernel(int nv, VertexFlags* vflags, devArray_t<T*, 3> u, devArray_t<T, 3> t) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nv)
		return;
	VertexFlags vflag = vflags[tid];
	if (vflag.is_fiction())
		return;

	u[0][tid] -= t[0];
	u[1][tid] -= t[1];
	u[2][tid] -= t[2];
}

// Todo: ignore period dof
void homo::Grid::v3_removeT(VT* u[3], VT tHost[3]) {
	devArray_t<VT*, 3> uarr{u[0], u[1], u[2]};
	devArray_t<VT, 3> tArr{tHost[0], tHost[1], tHost[2]};
	auto cfg = make_kernel_param(n_gsvertices(), 256);
	int nv = n_gsvertices();
	v3_removeT_kernel<<<cfg.grid, cfg.block>>>(nv, vertflag, uarr, tArr);
	cudaDeviceSynchronize();
	cuda_error_check;
}

template<typename T, typename Tout, int BlockSize = 256>
__global__ void v3_average_kernel(devArray_t<T*, 3> vlist, VertexFlags* vflags, int len, devArray_t<Tout*, 3> outlist,
								  bool removePeriodDof, bool firstReduce = false) {
	__shared__ Tout s[3][BlockSize / 32];
	size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
	Tout v[3] = {0.};

	size_t stride = gridDim.x * blockDim.x;

	int base = 0;
	for (; base + tid < len; base += stride) {
		int vid = base + tid;
		if (firstReduce) {
			VertexFlags vflag = vflags[vid];
			if (vflag.is_fiction())
				continue;
			if ((removePeriodDof && vflag.is_max_boundary()) || vflag.is_period_padding())
				continue;
		}
		v[0] += Tout(vlist[0][vid]);
		v[1] += Tout(vlist[1][vid]);
		v[2] += Tout(vlist[2][vid]);
	}

	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;

	// warp reduce
	for (int offset = 16; offset > 0; offset /= 2) {
		v[0] += shfl_down(v[0], offset);
		v[1] += shfl_down(v[1], offset);
		v[2] += shfl_down(v[2], offset);
	}
	if (laneId == 0) {
		s[0][warpId] = v[0];
		s[1][warpId] = v[1];
		s[2][warpId] = v[2];
	}

	// block reduce, do NOT use 1024 or higher blockSize
	if (BlockSize / 32 > 32) {
		print_exception;
	}

	__syncthreads();

	// block reduce
	if (warpId == 0) {
		if (threadIdx.x < BlockSize / 32) {
			v[0] = s[0][threadIdx.x];
			v[1] = s[1][threadIdx.x];
			v[2] = s[2][threadIdx.x];
		} else {
			v[0] = 0;
			v[1] = 0;
			v[2] = 0;
		}

		for (int offset = 16; offset > 0; offset /= 2) {
			v[0] += shfl_down(v[0], offset);
			v[1] += shfl_down(v[1], offset);
			v[2] += shfl_down(v[2], offset);
		}

		if (laneId == 0) {
			outlist[0][blockIdx.x] = v[0];
			outlist[1][blockIdx.x] = v[1];
			outlist[2][blockIdx.x] = v[2];
			//if (v[0] != 0)printf("b%d = %lf\n", int(blockIdx.x), v[0]);
		}
	}
}

void homo::Grid::v3_average(VT* v[3], VT vMean[3], bool removePeriodDof /*= false*/) {
	int le = (n_gsvertices() / 100 + 511) / 512 * 512;
	auto buffer = getTempBuffer(sizeof(float) * le * 3);
	float* ptmp = (float*)buffer.template data<>();
	devArray_t<float*, 3> v3tmp;
	v3tmp[0] = ptmp;
	v3tmp[1] = v3tmp[0] + le;
	v3tmp[2] = v3tmp[1] + le;

	devArray_t<float*, 3> v3out{v3tmp[0] + le / 2, v3tmp[1] + le / 2, v3tmp[2] + le / 2};

	devArray_t<VT*, 3> vlist{v[0], v[1], v[2]};
	int rest = n_gsvertices();
	auto cfg = make_kernel_param(rest, 256);
	if (le / 2 < cfg.grid)
		print_exception;
	v3_average_kernel<<<cfg.grid, cfg.block>>>(vlist, vertflag, rest, v3tmp, removePeriodDof, true);
	cudaDeviceSynchronize();
	cuda_error_check;
	rest = cfg.grid;

	while (rest > 1) {
		cfg = make_kernel_param(rest, 256);
		if (le / 2 < cfg.grid)
			print_exception;
		v3_average_kernel<<<cfg.grid, cfg.block>>>(v3tmp, vertflag, rest, v3out, removePeriodDof, false);
		cudaDeviceSynchronize();
		for (int i = 0; i < 3; i++)
			std::swap(v3tmp[i], v3out[i]);
		rest = cfg.grid;
	}

	float vMean_f[3];
	for (int i = 0; i < 3; i++)
		cudaMemcpy(&vMean_f[i], v3tmp[i], sizeof(float), cudaMemcpyDeviceToHost);

	int nValid;
	if (removePeriodDof) {
		nValid = cellReso[0] * cellReso[1] * cellReso[2];
	} else {
		nValid = (cellReso[0] + 1) * (cellReso[1] + 1) * (cellReso[2] + 1);
	}
	for (int i = 0; i < 3; i++) {
		vMean[i] = vMean_f[i] / nValid;
	}
	cuda_error_check;
}

template<typename T>
__global__ void v3_const_kernel(int nv, VertexFlags* vflags, devArray_t<T*, 3> u, devArray_t<T, 3> t) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nv)
		return;
	VertexFlags vflag = vflags[tid];
	if (vflag.is_fiction())
		return;

	u[0][tid] = t[0];
	u[1][tid] = t[1];
	u[2][tid] = t[2];
}

void homo::Grid::v3_const(VT* v[3], const VT v_const[3]) {
	devArray_t<VT*, 3> uarr{v[0], v[1], v[2]};
	devArray_t<VT, 3> tArr{v_const[0], v_const[1], v_const[2]};
	auto cfg = make_kernel_param(n_gsvertices(), 256);
	int nv = n_gsvertices();
	v3_const_kernel<<<cfg.grid, cfg.block>>>(nv, vertflag, uarr, tArr);
	cudaDeviceSynchronize();
	cuda_error_check;
}

template<typename T>
__global__ void update_rho_kernel(
	int nv, VertexFlags* vflags, CellFlags* eflags,
	float* srcrho, int srcPitchT, T* dstrho) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nv)
		return;

	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);

	VertexFlags vflag;

	vflag = vflags[tid];

	bool fiction = vflag.is_fiction() || vflag.is_period_padding() || vflag.is_max_boundary();

	if (fiction)
		return;

	indexer.locate(tid, vflag.get_gscolor(), gGsVertexEnd);

	int eid = indexer.neighElement(7, gGsCellEnd, gGsCellReso).getId();

	fiction = fiction || eid == -1;
	if (fiction)
		return;

	CellFlags eflag = eflags[eid];
	if (eflag.is_fiction() || eflag.is_period_padding())
		return;

	auto p = indexer.getPos();
	// to element pos without padding
	p.x -= 1;
	p.y -= 1;
	p.z -= 1;

	int sid;
	if (srcPitchT <= 0)
		sid = p.x + (p.y + p.z * gGridCellReso[1]) * gGridCellReso[0];
	else
		sid = p.x + (p.y + p.z * gGridCellReso[1]) * srcPitchT;

	dstrho[eid] = srcrho[sid];
}

template<typename T, int N, typename Flag>
__global__ void pad_data_kernel(
	int nsrcpadd, devArray_t<T*, N> v, Flag* flags,
	devArray_t<int, 3> resosrcpadd, devArray_t<int, 3> srcbasepos, devArray_t<int, 3> period,
	devArray_t<int, 3> resolist, devArray_t<devArray_t<int, 8>, 3> gsreso, devArray_t<int, 8> gsend) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ int gsReso[3][8];
	__shared__ int gsEnd[8];
	if (threadIdx.x < 3 * 8) {
		int i = threadIdx.x / 8;
		int j = threadIdx.x % 8;
		gsReso[i][j] = gsreso[i][j];
		if (i == 0) {
			gsEnd[j] = gsend[j];
		}
	}
	__syncthreads();

	if (tid >= nsrcpadd)
		return;
	int nf[3] = {
		resosrcpadd[0] * resosrcpadd[1],
		resosrcpadd[1] * resosrcpadd[2],
		resosrcpadd[2] * resosrcpadd[0]};
	int n_min = nf[0] + nf[1] + nf[2];
	if (tid >= 2 * n_min)
		return;

	int min_id = tid % n_min;
	int max_id = tid / n_min;

	int pos[3];

	char3 bound = {};

	if (min_id < nf[0]) {
		pos[0] = min_id % resosrcpadd[0];
		pos[1] = min_id / resosrcpadd[0];
		pos[2] = 0;
		bound.z = 1;
	} else if (min_id < nf[0] + nf[1]) {
		pos[0] = 0;
		pos[1] = (min_id - nf[0]) % resosrcpadd[1];
		pos[2] = (min_id - nf[0]) / resosrcpadd[1];
		bound.x = 1;
	} else if (min_id < nf[0] + nf[1] + nf[2]) {
		pos[0] = (min_id - nf[0] - nf[1]) % resosrcpadd[0];
		pos[1] = 0;
		pos[2] = (min_id - nf[0] - nf[1]) / resosrcpadd[0];
		bound.y = 1;
	}

	if (max_id == 1) {
		if (bound.x) {
			pos[0] += resosrcpadd[0] - 1;
		} else if (bound.y) {
			pos[1] += resosrcpadd[1] - 1;
		} else if (bound.z) {
			pos[2] += resosrcpadd[2] - 1;
		} else {
			print_exception; // DEBUG
		}
	}

	pos[0] += srcbasepos[0];
	pos[1] += srcbasepos[1];
	pos[2] += srcbasepos[2];

	int myid = lexi2gs(pos, gsReso, gsEnd);

	//printf("pos = (%d, %d, %d)\n", pos[0], pos[1], pos[2]);
	// scatter padding data
	int oppos[3];
	for (int offx : {-period[0], 0, period[0]}) {
		oppos[0] = offx + pos[0];
		if (oppos[0] < -1 || oppos[0] > resolist[0])
			continue;

		for (int offy : {-period[1], 0, period[1]}) {
			oppos[1] = offy + pos[1];
			if (oppos[1] < -1 || oppos[1] > resolist[1])
				continue;

			for (int offz : {-period[2], 0, period[2]}) {
				oppos[2] = offz + pos[2];
				if (oppos[2] < -1 || oppos[2] > resolist[2])
					continue;

				if ((oppos[0] == -1 || oppos[0] == resolist[0]) ||
					(oppos[1] == -1 || oppos[1] == resolist[1]) ||
					(oppos[2] == -1 || oppos[2] == resolist[2])) {
					int opid = lexi2gs(oppos, gsReso, gsEnd);
					// debug
					if (!flags[opid].is_period_padding()) {
						printf("\033[31moppos = (%d, %d, %d)\033[0m\n", oppos[0], oppos[1], oppos[2]);
						//print_exception;
					}
					for (int i = 0; i < N; i++) {
						v[i][opid] = v[i][myid];
					}
					//printf("pos = (%d, %d, %d) oppos = (%d, %d, %d)\n",
					//	pos[0], pos[1], pos[2], oppos[0], oppos[1], oppos[2]);
				}
			}
		}
	}
}

template<typename T>
void pad_cell_data_imp(T* e, CellFlags* eflags, std::array<int, 3> cellReso, int gsCellReso[3][8], int gsCellSetEnd[8]) {
	int nsrcpadd = 2 * (cellReso[0] * cellReso[1] + cellReso[1] * cellReso[2] + cellReso[0] * cellReso[2]);
	devArray_t<int, 3> resolist{cellReso[0], cellReso[1], cellReso[2]};
	devArray_t<int, 3> resopad{cellReso[0], cellReso[1], cellReso[2]};
	devArray_t<int, 3> padbase{0, 0, 0};
	devArray_t<int, 3> period{cellReso[0], cellReso[1], cellReso[2]};
	devArray_t<devArray_t<int, 8>, 3> gsreso;
	devArray_t<int, 8> gsend;
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 8; j++)
			gsreso[i][j] = gsCellReso[i][j];
	for (int i = 0; i < 8; i++) {
		gsend[i] = gsCellSetEnd[i];
	}
	devArray_t<T*, 1> arr{e};
	auto cfg = make_kernel_param(nsrcpadd, 256);
	pad_data_kernel<<<cfg.grid, cfg.block>>>(nsrcpadd, arr, eflags, resopad, padbase, period, resolist, gsreso, gsend);
	cudaDeviceSynchronize();
	cuda_error_check;
}

void homo::Grid::pad_cell_data(float* e) {
	pad_cell_data_imp(e, cellflag, cellReso, gsCellReso, gsCellSetEnd);
}

void homo::Grid::pad_cell_data(half* e) {
	pad_cell_data_imp(e, cellflag, cellReso, gsCellReso, gsCellSetEnd);
}

void homo::Grid::update(float* rho, int pitchT, bool lexiOrder /*= true*/) {
	if (!lexiOrder) {
		//cudaMemcpy(rho_g, rho, sizeof(float) * n_gscells(), cudaMemcpyDeviceToDevice);
		type_cast(rho_g, rho, n_gscells());
	} else {
		useGrid_g();
		//{
		//	std::vector<float> hostrho(n_cells());
		//	cudaMemcpy(hostrho.data(), rho, sizeof(float) * n_cells(), cudaMemcpyDeviceToHost);
		//	array2matlab("srcrho", hostrho.data(), hostrho.size());
		//}
		int nv = n_gsvertices();
		auto vflags = vertflag;
		auto eflags = cellflag;
		auto cfg = make_kernel_param(nv, 256);
		update_rho_kernel<<<cfg.grid, cfg.block>>>(nv, vflags, eflags, rho, pitchT, rho_g);
		cudaDeviceSynchronize();
		cuda_error_check;
		//{
		//	std::vector<float> hostrho(n_gscells());
		//	cudaMemcpy(hostrho.data(), rho_g, sizeof(float) * n_gscells(), cudaMemcpyDeviceToHost);
		//	array2matlab("newrho", hostrho.data(), hostrho.size());
		//}
		pad_cell_data(rho_g);
		if (0) {
			// std::vector<float> hostrho(n_gscells());
			// cudaMemcpy(hostrho.data(), rho_g, sizeof(float) * n_gscells(), cudaMemcpyDeviceToHost);
			// array2matlab("padrho", hostrho.data(), hostrho.size());
		}
	}
}

template<typename T>
__global__ void enforceCellSymmetry_kernel(T* edata, SymmetryType sym, bool average) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ereso[3] = {gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]};
	int repReso[3] = {ereso[0] / 2, ereso[1] / 2, ereso[2] / 2};
	int repPos[3] = {
		tid % repReso[0],
		tid / repReso[0] % repReso[1],
		tid / (repReso[0] * repReso[1])};
	if (repPos[2] >= repReso[2])
		return;
	int orbit[8];
	if (sym == Simple3) {
		for (int i = 0; i < 8; i++) {
			int orbitpos[3];
			int flip[3] = {i % 2, i / 2 % 2, i / 4};
			for (int j = 0; j < 3; j++) {
				if (flip[j]) {
					orbitpos[j] = ereso[j] - 1 - repPos[j];
				} else {
					orbitpos[j] = repPos[j];
				}
			}
			orbit[i] = lexi2gs(orbitpos, gGsVertexReso, gGsVertexEnd);
		}
		T val = edata[orbit[0]];
		if (average) {
			for (int i = 1; i < 8; i++) {
				val += edata[orbit[i]];
			}
			val /= 8;
		}
		// Note : No write-read conflict due to non-intersection of orbits
		for (int i = 0; i < 8; i++) {
			edata[orbit[i]] = val;
		}
	} else {
		printf("no implementation");
	}
}

template<typename T>
void enforceCellSymmetry_imp(T* celldata, SymmetryType sym, bool average, std::array<int, 3> cellReso) {
	if (sym == None)
		return;
	int repReso[3] = {cellReso[0] / 2, cellReso[1] / 2, cellReso[2] / 2};
	int n_rep = repReso[0] * repReso[1] * repReso[2];
	auto cfg = make_kernel_param(n_rep, 256);
	enforceCellSymmetry_kernel<<<cfg.grid, cfg.block>>>(celldata, sym, average);
	cudaDeviceSynchronize();
	cuda_error_check;
}

void homo::Grid::enforceCellSymmetry(float* celldata, SymmetryType sym, bool average) {
	useGrid_g();
	enforceCellSymmetry_imp(celldata, sym, average, cellReso);
}

void homo::Grid::enforceCellSymmetry(half* celldata, SymmetryType sym, bool average) {
	useGrid_g();
	enforceCellSymmetry_imp(celldata, sym, average, cellReso);
}

template<typename T, int N>
__global__ void checkPeriodVertex_kernel(int nv, devArray_t<T*, N> v, VertexFlags* vflags) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nv)
		return;
	auto vflag = vflags[tid];
	if (!vflag.is_min_boundary() || vflag.is_period_padding())
		return;
	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);
	int color = vflag.get_gscolor();
	indexer.locate(tid, color, gGsVertexEnd);
	auto pos = indexer.getPos();
	pos.x -= 1;
	pos.y -= 1;
	pos.z -= 1;
	// Note : we only check one period vertex, actually there are more.
	if (pos.x == 0)
		pos.x = gGridCellReso[0];
	if (pos.y == 0)
		pos.y = gGridCellReso[1];
	if (pos.z == 0)
		pos.z = gGridCellReso[2];
	int p[3] = {pos.x, pos.y, pos.z};
	int maxid = lexi2gs(p, gGsVertexReso, gGsVertexEnd);
	bool is_same = true;
	for (int i = 0; i < N; i++) {
		if (v[i][tid] != v[i][maxid]) {
			is_same = false;
			break;
		}
	}
	if (!is_same) {
		// printf("[%04d, %04d, %04d] vmin = (%6.4e, %6.4e, %6.4e) vmax = (%6.4e, %6.4e, %6.4e)\n",
		// 	   p[0], p[1], p[2], v[0][tid], v[1][tid], v[2][tid],
		// 	   v[0][maxid], v[1][maxid], v[2][maxid]);
		printf("[%04d, %04d, %04d] = (vid = %d, maxid = %d)\n", p[0], p[1], p[2], int(tid), maxid);
	}
}

template<typename T, int N>
void checkPeriodVertex_imp(int nv, VertexFlags* vertflag, T* v[N]) {
	auto cfg = make_kernel_param(nv, 256);
	devArray_t<T*, N> varr(v);
	checkPeriodVertex_kernel<<<cfg.grid, cfg.block>>>(nv, varr, vertflag);
	cudaDeviceSynchronize();
	cuda_error_check;
}

void homo::Grid::checkPeriodVertex(VT* v[3]) {
	checkPeriodVertex_imp<VT, 3>(n_gsvertices(), vertflag, v);
}

void homo::Grid::checkPeriodStencil() {
	if (is_root) {
		return;
	};
	for (int i = 0; i < 27; i++) {
		printf("checking period stencil %d ...\n", i);
		glm::hmat3* st[1] = {stencil_g[i]};
		checkPeriodVertex_imp<glm::hmat3, 1>(n_gsvertices(), vertflag, st);
	}
}