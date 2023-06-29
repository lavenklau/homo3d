#include "grid.h"
//#define __CUDACC__
#include "culib/lib.cuh"
#include <vector>
#include "utils.h"
#include <fstream>
#include "homoCommon.cuh"

#undef cuda_error_check

#define cuda_error_check do{ \
	auto err = cudaPeekAtLastError(); \
	if (err != cudaSuccess) { \
		printf("\x1b[31mCUDA error occured at line %d in file %s, error [%d] type %s \x1b[0m\n", __LINE__,__FILE__, int(err), cudaGetErrorName(err));\
		exit(0);\
	} \
}while(0)

using namespace homo;
using namespace culib;

__device__ static bool inStrictBound(int pi[3], int cover[3]) {
	return pi[0] > -cover[0] && pi[0] < cover[0] &&
		pi[1] > -cover[1] && pi[1] < cover[1] &&
		pi[2] > -cover[2] && pi[2] < cover[2];
}


template<int BlockSize = 32 * 8>
__global__ void gs_relaxation_heat_otf_kernel(
	int gs_set, float* rhoheat,
	devArray_t<int, 3> gridCellReso,
	VertexFlags* vflags, CellFlags* eflags,
	// SOR relaxing factor
	float w = 1.f,
	float diag_strength = 0
) {

	__shared__ int gsCellReso[3][8];
	__shared__ int gsVertexReso[3][8];
	__shared__ int gsCellEnd[8];
	__shared__ int gsVertexEnd[8];

	__shared__ float KH[8][8];

	__shared__ float sumKeU[1][4][32];
	__shared__ float sumKs[1][4][32];

	//__shared__ double* U[3];

	//__shared__ int NeNv[8][8];

	initSharedMem(&sumKeU[0][0][0], sizeof(sumKeU) / sizeof(float));
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
	loadHeatTemplateMatrix(KH);

	// to global vertex id
	vid = gs_set == 0 ? vid : gsVertexEnd[gs_set - 1] + vid;

	if (vid >= gsVertexEnd[gs_set]) fiction = true;

	GridVertexIndex indexer(gridCellReso[0], gridCellReso[1], gridCellReso[2]);
	VertexFlags vflag;
	if (!fiction) {
		vflag = vflags[vid];
		fiction = fiction || vflag.is_fiction() || vflag.is_period_padding();
	}
	if(!fiction) {
		indexer.locate(vid, vflag.get_gscolor(), gsVertexEnd);
	}
	
	float KeU = { 0. };
	float Ks = { 0.f };

	if (!fiction && !vflag.is_period_padding()) {
		int elementId = indexer.neighElement(warpId, gsCellEnd, gsCellReso).getId();
		float rho_penal = 0;
		CellFlags eflag;
		float penal = exp_penal[0];
		if (elementId != -1) {
			eflag = eflags[elementId];
			if (!eflag.is_fiction()) rho_penal = rhoheat[elementId];
		}

		if (elementId != -1 && !eflag.is_fiction()) {
            int vi = 7 - warpId;
#pragma unroll
			for (int i = 0; i < 8; i++) {
                int vj = i;
				if (vj == 7 - warpId) continue;

				int vneigh =
					(warpId % 2 + i % 2) +
					(warpId / 2 % 2 + i / 2 % 2) * 3 +
					(warpId / 4 + i / 4) * 9;

				int vneighId = indexer.neighVertex(vneigh, gsVertexEnd, gsVertexReso).getId();
				VertexFlags nvflag;
				if (vneighId != -1) {
					nvflag = vflags[vneighId];
					if (!nvflag.is_fiction()) {
						float u = { gUHeat[vneighId]};
						if (nvflag.is_sink()) { u = 0; }
                        KeU += KH[vi][vj] * u;
                    }
				}
			}
			KeU *= rho_penal;

            Ks += float(KH[vi][vi]) * rho_penal;
        }
	}

	if (warpId >= 4) {
        sumKs[0][warpId - 4][laneId] = Ks;
        sumKeU[0][warpId - 4][laneId] = KeU;
    }
	__syncthreads();

	if (warpId < 4) {
        sumKs[0][warpId][laneId] += Ks;
        sumKeU[0][warpId][laneId] += KeU;
    }
	__syncthreads();

	if (warpId < 2) {
        sumKs[0][warpId][laneId] += sumKs[0][warpId + 2][laneId];
        sumKeU[0][warpId][laneId] += sumKeU[0][warpId + 2][laneId];
    }
	__syncthreads();

	if (warpId < 1 && !vflag.is_period_padding() && !fiction) {
        Ks = sumKs[0][warpId][laneId] + sumKs[0][warpId + 1][laneId];
        KeU = sumKeU[0][warpId][laneId] + sumKeU[0][warpId + 1][laneId];

		float u = { gUHeat[vid]};

		// relax
#if !USING_SOR 
		u = (gFHeat[vid] - KeU) / Ks;
#else
        u = w * (gFHeat[vid] - KeU) / Ks + (1 - w) * u;
#endif
		// if dirichlet boundary;
		if (vflag.is_sink()) {
			u = 0; 
			gFHeat[vid] = 0;
		}
		// update
		gUHeat[vid] = u;
	}
}

// map 32 vertices to 13 warp
template<int BlockSize = 32 * 13>
__global__ void gs_relaxation_heat_kernel(	
	int gs_set,
	VertexFlags* vflags,
	// SOR relaxing factor
	float w = 1.f
) {
	__shared__ float sumAu[1][7][32];
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
	if (vid >= gsVertexEnd[gs_set]) fiction = true;
	VertexFlags vflag;
	if (!fiction) vflag = vflags[vid];
	fiction = fiction || vflag.is_fiction();

	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);

	float Au(0.);

	if (!fiction && !vflag.is_period_padding()) {
		indexer.locate(vid, vflag.get_gscolor(), gsVertexEnd);

		for (int noff : {0, 14}) {
			int vneigh = warpId + noff;
			int neighId = indexer.neighVertex(vneigh, gsVertexEnd, gsVertexReso).getId();
			if (neighId == -1) continue;
			VertexFlags neiflag = vflags[neighId];
			if (!neiflag.is_fiction()) {
				float u(gUHeat[neighId]);
				Au += rxHeatStencil[vneigh][vid] * u;
			}
		}
	}


	if (warpId >= 7) {
        sumAu[0][warpId - 7][laneId] = Au;
    }
	__syncthreads();

	if (warpId < 7) {
		if (warpId < 6) {
            sumAu[0][warpId][laneId] += Au;
        } else {
            sumAu[0][6][laneId] = Au;
        }
	}
	__syncthreads();

	if (warpId < 3) {
        sumAu[0][warpId][laneId] += sumAu[0][warpId + 4][laneId];
    }
	__syncthreads();

	if (warpId < 2) {
        sumAu[0][warpId][laneId] += sumAu[0][warpId + 2][laneId];
    }
	__syncthreads();

	if (warpId < 1 && !fiction) {
        Au = sumAu[0][warpId][laneId] + sumAu[0][warpId + 1][laneId];

        if (!vflag.is_period_padding()) {
			float u = { gUHeat[vid]};
			float st = rxHeatStencil[13][vid];
#if !USING_SOR
			u = (gFHeat[vid] - Au) / st;
#else
			u = w * (gFHeat[vid] - Au) / st + (1 - w) * u;
#endif
			gUHeat[vid] = u;
		}
	}
}


void homo::Grid::gs_relaxation_heat(float w_SOR /*= 1.f*/, int times_ /*= 1*/)
{
	// change to 8 bytes bank
	use8Bytesbank();
	useGrid_g();
	devArray_t<int, 3>  gridCellReso{};
	devArray_t<int, 8>  gsCellEnd{};
	devArray_t<int, 8>  gsVertexEnd{};
	for (int i = 0; i < 8; i++) {
		gsCellEnd[i] = gsCellSetEnd[i];
		gsVertexEnd[i] = gsVertexSetEnd[i];
		if (i < 3) gridCellReso[i] = cellReso[i];
	}
	for (int itn = 0; itn < times_; itn++) {
		for (int i = 0; i < 8; i++) {
			int set_id = i;
			size_t grid_size, block_size;
			int n_gs = gsVertexEnd[set_id] - (set_id == 0 ? 0 : gsVertexEnd[set_id - 1]);
			if (assemb_otf) {
				make_kernel_param(&grid_size, &block_size, n_gs * 8, 32 * 8);
				gs_relaxation_heat_otf_kernel<<<grid_size, block_size>>>(set_id, rhoHeat_g, gridCellReso, vertflag, cellflag, w_SOR, diag_strength);
			}
			else {
				make_kernel_param(&grid_size, &block_size, n_gs * 13, 32 * 13);
				gs_relaxation_heat_kernel<<<grid_size, block_size>>>(set_id, vertflag, w_SOR);
			}
			cudaDeviceSynchronize();
			cuda_error_check;
			enforce_period_boundary(uHeat_g);
		}
	}
	enforce_period_boundary(uHeat_g);
	//pad_vertex_data(u_g);
	cudaDeviceSynchronize();
	cuda_error_check;
}

// map 32 vertices to 8 warp
__global__ void update_residual_heat_otf_kernel_1(
	int nv, float* rhoheat,
	devArray_t<int, 3> gridCellReso, 
	VertexFlags* vflags, CellFlags* eflags,
	float diag_strength
) {
	__shared__ int gsCellReso[3][8];
	__shared__ int gsVertexReso[3][8];
	__shared__ int gsCellEnd[8];
	__shared__ int gsVertexEnd[8];

	__shared__ float KH[8][8];

	__shared__ float sumKeU[1][4][32];

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
	loadHeatTemplateMatrix(KH);

	GridVertexIndex indexer(gridCellReso[0], gridCellReso[1], gridCellReso[2]);
	if (!fiction) {
		indexer.locate(vid, vflag.get_gscolor(), gsVertexEnd);
	}

	float KeU = { 0. };

	int elementId = -1;
	if (!fiction) elementId = indexer.neighElement(warpId, gsCellEnd, gsCellReso).getId();
	float rhop = 0;
	CellFlags eflag;
	float penal = exp_penal[0];
	if (elementId != -1) {
		eflag = eflags[elementId];
		if (!eflag.is_fiction()) rhop = rhoheat[elementId];
	}

	// DEBUG
	//bool debug = false;
	if (elementId != -1 && !eflag.is_fiction() && !fiction) {
        int vi = 7 - warpId;
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
                    float u = {gUHeat[vneighId]};
                    if (nvflag.is_sink()) { u = 0; }
                    int vj = i;
					KeU += KH[vj][vi] * u;
				}
			}
		}
		KeU *= rhop;
	}
	if (warpId >= 4) {
        sumKeU[0][warpId - 4][laneId] = KeU;
    }
	__syncthreads();

	if (warpId < 4) {
        sumKeU[0][warpId][laneId] += KeU;
    }
	__syncthreads();

	if (warpId < 2) {
        sumKeU[0][warpId][laneId] += sumKeU[0][warpId + 2][laneId];
    }
	__syncthreads();

	if (warpId < 1 && !fiction && !vflag.is_period_padding()) {
        KeU = sumKeU[0][warpId][laneId] + sumKeU[0][warpId + 1][laneId];

        float r = {gFHeat[vid] - KeU};

        if (vflag.is_sink()) { r = 0; }

		// relax
		gRHeat[vid] = r;
	}
}

// map 32 vertices to 9 warp
template<int BlockSize = 32 * 9>
__global__ void update_residual_heat_kernel_1(
	int nv,
	VertexFlags* vflags
) {
	__shared__ int gsVertexEnd[8];
	__shared__ int gsVertexReso[3][8];
	__shared__ float sumKu[1][5][32];

	constantToShared(gGsVertexEnd, gsVertexEnd);
	constant2DToShared(gGsVertexReso, gsVertexReso);

	initSharedMem(&sumKu[0][0][0], sizeof(sumKu) / sizeof(float));

	__syncthreads();
	
	bool fiction = false;

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;
	int vid = blockIdx.x * 32 + laneId;
	if (vid >= nv) fiction = true;

	VertexFlags vflag;
	if (!fiction) vflag = vflags[vid];
	fiction = fiction || vflag.is_fiction() || vflag.is_period_padding();
	int color = vflag.get_gscolor();

	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);
	if(!fiction) indexer.locate(vid, vflag.get_gscolor(), gsVertexEnd);

	float KeU(0.);
	if (!fiction && !vflag.is_period_padding()) {
		for (auto off : { 0,9,18 }) {
			int vneigh = off + warpId;
			int neighId = indexer.neighVertex(vneigh, gsVertexEnd, gsVertexReso).getId();
			if (neighId != -1) {
				VertexFlags neighFlag = vflags[neighId];
				if (!neighFlag.is_fiction()) {
					float u(gUHeat[neighId]);
					float st=rxHeatStencil[vneigh][vid];
                    KeU += st * u;
                }
			}
		}
	}

	if (warpId >= 4) {
		sumKu[0][warpId - 4][laneId] = KeU;
	}
	__syncthreads();

	if (warpId < 4) {
		sumKu[0][warpId][laneId] += KeU;
	}
	__syncthreads();

	if (warpId < 2) {
		sumKu[0][warpId][laneId] += sumKu[0][warpId + 2][laneId];
	}
	__syncthreads();

	if (warpId < 1 && !fiction) {
		KeU = sumKu[0][warpId][laneId] + sumKu[0][warpId + 1][laneId] + sumKu[0][4][laneId];
        float r = {gFHeat[vid] - KeU};
        gRHeat[vid] = r;
	}
}

void homo::Grid::update_residual_heat(void)
{
	useGrid_g();
	devArray_t<int, 3> gridCellReso{ cellReso[0],cellReso[1],cellReso[2] };
	VertexFlags* vflags = vertflag;
	CellFlags* eflags = cellflag;
	if (assemb_otf) {
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, n_gsvertices() * 8, 32 * 8);
		update_residual_heat_otf_kernel_1 << <grid_size, block_size >> > (n_gsvertices(), rhoHeat_g, gridCellReso,
			vflags, eflags, diag_strength);
		cudaDeviceSynchronize();
		cuda_error_check;
	}
	else {
		size_t grid_size, block_size;
		int nv = n_gsvertices();
		make_kernel_param(&grid_size, &block_size, n_gsvertices() * 9, 32 * 9);
		update_residual_heat_kernel_1 << <grid_size, block_size >> > (nv, vflags);
		cudaDeviceSynchronize();
		cuda_error_check;
	}
	pad_vertex_data(rHeat_g);
}


template<int BlockSize = 256>
__global__ void prolongate_correction_heat_kernel_1(
	bool is_root, int nv_fine,
	VertexFlags* vflags,
	VertexFlags* vcoarseflags,
	devArray_t<int, 8> GsVertexEnd,
	devArray_t<int, 8> GsCoarseVertexEnd
) {
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
	fiction |= vflag.is_fiction();

	float pr = coarseRatio[0] * coarseRatio[1] * coarseRatio[2];

	bool isRoot = is_root;

	if (!fiction && !vflag.is_period_padding()) {
		GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);
		indexer.locate(tid, vflag.get_gscolor(), GsVertexEnd._data);

		float u = { 0. };
		int nvCoarse[8];
		float w[8];
		int remainder[3];
		indexer.neighCoarseVertex(nvCoarse, w, coarseRatio, gsCoarseVertexEnd, gsCoarseVertexReso, remainder);
		for (int i = 0; i < 8; i++) {
			int neighId = nvCoarse[i];
			if (neighId != -1) {
                float uc = {gUHeatcoarse[neighId]};
                u += uc * w[i];
			}
		}

		if (isRoot && vflag.is_sink()) { u = 0; }

		gUHeat[tid] += u;
	}
}

void homo::Grid::prolongate_correction_heat(void)
{
	useGrid_g();
	VertexFlags* vflags = vertflag;
	VertexFlags* vcoarseFlags = Coarse->vertflag;
	devArray_t<int, 8> gsVertexEnd{}, gsCoarseVertexEnd{};
	for (int i = 0; i < 8; i++) {
		gsVertexEnd[i] = gsVertexSetEnd[i];
		gsCoarseVertexEnd[i] = Coarse->gsVertexSetEnd[i];
	}
	int nv_fine = n_gsvertices();
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, nv_fine, 256);
	prolongate_correction_heat_kernel_1 << <grid_size, block_size >> > (is_root, nv_fine, vflags, vcoarseFlags, gsVertexEnd, gsCoarseVertexEnd);
	cudaDeviceSynchronize();
	cuda_error_check;
	enforce_period_boundary(uHeat_g);
}


__global__ void restrict_stencil_heat_otf_aos_kernel_1(
	int nv, float* rhoheat, CellFlags* eflags, VertexFlags* vflags, float ds = 1
) {
	//__shared__ glm::mat<3, 3, double> KE[8][8];
	__shared__ float KE[8][8];
	__shared__ int coarseReso[3];
	__shared__ int fineReso[3];
	
	if (threadIdx.x < 3) { 
		coarseReso[threadIdx.x] = gGridCellReso[threadIdx.x]; 
		fineReso[threadIdx.x] = coarseReso[threadIdx.x] * gUpCoarse[threadIdx.x];
	}
	loadHeatTemplateMatrix(KE);

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
		float st(0.f);
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
					bool isSink[8];
					for (int ev = 0; ev < 8; ev++) {
						int evpos[3] = {e_fine_pos[0] + ev % 2, e_fine_pos[1] + ev / 2 % 2, e_fine_pos[2] + ev / 4};
						int evid = lexi2gs(evpos, gGsFineVertexReso, gGsFineVertexEnd);
						isSink[ev] = vflags[evid].is_sink();
					}
					//auto eflag = eflags[eid];
					float rho_penal = rhoheat[eid];
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
							if (isSink[e_vi] || isSink[e_vj]) {
								if (e_vi == e_vj) {
									st += (wi * wj) * KE[0][0] * ds;
								}
							}
							else
							{
								st += (wi * wj) * KE[e_vi][e_vj];
							}
						}
					}
				}
			}
		}
		rxHeatStencil[vj][vid] = st;
	}
}

// one thread of one coarse vertex
//template<int BlockSize = 256>
__global__ void restrict_stencil_heat_aos_kernel_1(
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
			float st(0.f);
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
							st += wi * wj * rxHeatFineStencil[vj_offid][vi_neighId];
						}
					}
				}
			}
			rxHeatStencil[i][vid] = st;
		}
	}
}

void homo::Grid::restrict_stencil_heat(void)
{
	if (is_root) return;
	if (fine->assemb_otf) {
		useGrid_g();
		size_t grid_size, block_size;
		for (int i = 0; i < 27; i++) {
			cudaMemset(heatStencil_g[i], 0, sizeof(float) * n_gsvertices());
		}
		cudaDeviceSynchronize();
		cuda_error_check;
		int nv = (cellReso[0] + 1) * (cellReso[1] + 1) * (cellReso[2] + 1);
		make_kernel_param(&grid_size, &block_size, nv, 256);
		restrict_stencil_heat_otf_aos_kernel_1 << <grid_size, block_size >> > (nv, fine->rhoHeat_g, fine->cellflag, fine->vertflag);
		cudaDeviceSynchronize();
		cuda_error_check;
		useGrid_g();
		lexiHeatStencil2gsorder();
		enforce_period_heat_stencil(true);
	}
	else {
		useGrid_g();
		cudaDeviceSynchronize();
		cuda_error_check;
		int nvfine = fine->n_gsvertices();
		//printf("--\n");
		size_t grid_size, block_size;
		make_kernel_param(&grid_size, &block_size, n_gsvertices(), 256);
		//restrict_stencil_kernel_1 << <grid_size, block_size >> > (n_gsvertices(), nvfine, vertflag, fine->vertflag);
		restrict_stencil_heat_aos_kernel_1 << <grid_size, block_size >> > (n_gsvertices(), nvfine, vertflag, fine->vertflag);
		cudaDeviceSynchronize();
		cuda_error_check;
		//stencil2matlab("Khost");
		enforce_period_heat_stencil(false);
	}
}

template<int BlockSize = 256>
__global__ void restrict_residual_kernel_1(
	int nv_coarse,
	VertexFlags* vflags,
	VertexFlags* vfineflags,
	devArray_t<int, 8> GsVertexEnd,
	devArray_t<int, 8> GsFineVertexEnd
) {
    __shared__ int gsVertexEnd[8];
    __shared__ int gsFineVertexEnd[8];
    __shared__ int gsFineVertexReso[3][8];

    if (threadIdx.x < 24)
    {
        gsFineVertexReso[threadIdx.x / 8][threadIdx.x % 8] =
            gGsFineVertexReso[threadIdx.x / 8][threadIdx.x % 8];
        if (threadIdx.x < 8)
        {
            gsVertexEnd[threadIdx.x] = GsVertexEnd[threadIdx.x];
            gsFineVertexEnd[threadIdx.x] = GsFineVertexEnd[threadIdx.x];
        }
    }
    __syncthreads();

    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= nv_coarse)
        return;

    VertexFlags vflag = vflags[tid];
    bool fiction = vflag.is_fiction() || vflag.is_period_padding();

    int setid = vflag.get_gscolor();

    GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);
    if (!fiction)
        indexer.locate(tid, vflag.get_gscolor(), gsVertexEnd);

    int coarseRatio[3] = {gUpCoarse[0], gUpCoarse[1], gUpCoarse[2]};
    float pr = coarseRatio[0] * coarseRatio[1] * coarseRatio[2];

    bool nondyadic = coarseRatio[0] > 2 || coarseRatio[1] > 2 || coarseRatio[2] > 2;

    float r = {0.};

    if (!fiction && !vflag.is_period_padding())
    {
        for (int offx = -coarseRatio[0] + 1; offx < coarseRatio[0]; offx++)
        {
            for (int offy = -coarseRatio[1] + 1; offy < coarseRatio[1]; offy++)
            {
                for (int offz = -coarseRatio[2] + 1; offz < coarseRatio[2]; offz++)
                {
                    int off[3] = {offx, offy, offz};
                    float w = (coarseRatio[0] - abs(offx)) * (coarseRatio[1] - abs(offy)) * (coarseRatio[2] - abs(offz)) / pr;
                    int neighVid = -1;
                    // DEBUG
                    if (nondyadic)
                        neighVid = indexer.neighFineVertex(off, coarseRatio, gsFineVertexEnd, gsFineVertexReso, true).getId();
                    else
                        neighVid = indexer.neighFineVertex(off, coarseRatio, gsFineVertexEnd, gsFineVertexReso, false).getId();

                    VertexFlags vfineflag;
                    if (neighVid != -1)
                    {
                        vfineflag = vfineflags[neighVid];
                        if (!vfineflag.is_fiction())
                        {
                            r += gRHeatfine[neighVid] * w;
                        }
                    }
                }
            }

            gFHeat[tid] = r;
        }
    }
}

void homo::Grid::restrict_residual_heat(void) {
    useGrid_g();
    VertexFlags *vflags = vertflag;
    VertexFlags *vfineflags = fine->vertflag;
    devArray_t<int, 8> gsVertexEnd{}, gsFineVertexEnd{};
    for (int i = 0; i < 8; i++)
    {
        gsVertexEnd[i] = gsVertexSetEnd[i];
        gsFineVertexEnd[i] = fine->gsVertexSetEnd[i];
    }
    int nv = n_gsvertices();
    size_t grid_size, block_size;
    make_kernel_param(&grid_size, &block_size, nv, 256);
    restrict_residual_kernel_1<<<grid_size, block_size>>>(nv, vflags, vfineflags, gsVertexEnd, gsFineVertexEnd);
    cudaDeviceSynchronize();
    cuda_error_check;
    pad_vertex_data(fHeat_g);
}

void homo::Grid::reset_displacement_heat(void) {
    init_array(uHeat_g, float(0), n_gsvertices());
}

void homo::Grid::reset_residual_heat(void){
    init_array(rHeat_g, float(0), n_gsvertices());
}

void homo::Grid::reset_force_heat(void){
    init_array(fHeat_g, float(0), n_gsvertices());
}

double homo::Grid::relative_residual_heat(void) {
	return culib::norm(rHeat_g, n_gsvertices()) / (culib::norm(fHeat_g, n_gsvertices())+1e-30);
}


__global__ void setSinkNodes_kernel(int nv, float* vhint, VertexFlags* vflags) {
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nv) return;
	// if(vflags[tid].is_dirichlet_boundary()) {
	// 	vflags[tid].set_sink();
	// }
	if (vhint[tid] < 0.01) {
		vflags[tid].set_sink();
	}
}

// ToDO: set your own sink nodes
void homo::Grid::setSinkNodes(void) {
	v1_rand(rHeat_g, 0, 1);
	enforce_period_boundary(rHeat_g, false);
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, n_gsvertices(), 512);
	setSinkNodes_kernel<<<grid_size, block_size>>>(n_gsvertices(), rHeat_g, vertflag);
	cudaDeviceSynchronize();
	cuda_error_check;
}