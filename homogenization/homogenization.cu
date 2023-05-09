#include "homogenization.h"
#include "device_launch_parameters.h"
#include "homoCommon.cuh"
#include "tictoc.h"
#include "cuda_fp16.h"
#include "mma.h"

#define USE_LAME_MATRIX 1

using namespace homo;
using namespace culib;

void homo::Homogenization::Sensitivity(int i, int j, float* sens)
{
	grid->sensitivity(i, j, sens);
}

__global__ void elasticMatrix_kernel_wise(
	int nv, int iStrain, int jStrain,
	devArray_t<float*, 3> ui, devArray_t<float*, 3> uj,
	float* rholist, VertexFlags* vflags, CellFlags* eflags,
	float* elementCompliance
) {
#if USE_LAME_MATRIX
	__shared__ Lame KLAME[24][24];
	loadLameMatrix(KLAME);
	float lam = LAM[0];
	float mu = MU[0];
#else
	__shared__ float KE[24][24];
	loadTemplateMatrix(KE);
#endif


	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= nv) return;

	int vid = tid;

	VertexFlags vflag = vflags[vid];
	if (vflag.is_fiction() || vflag.is_period_padding()) return;

	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);

	indexer.locate(vid, vflag.get_gscolor(), gGsVertexEnd);

	int elementId = indexer.neighElement(0, gGsCellEnd, gGsCellReso).getId();

	if (elementId != -1) {
		CellFlags eflag = eflags[elementId];
		if (!eflag.is_fiction() && !eflag.is_period_padding()) {
			float prho = powf(rholist[elementId], exp_penal[0]);
			//float prho = rholist[elementId];
			float c = 0;
			//int neighVid[8];
			float u[8][3];
			float v[8][3];
			for (int i = 0; i < 8; i++) {
				int id = i % 2 + i / 2 % 2 * 3 + i / 4 * 9;
				int neighVid = indexer.neighVertex(id, gGsVertexEnd, gGsVertexReso).getId();
				if (neighVid == -1) print_exception;
				//u[i][0] = i % 2 
				elementMacroDisplacement(i, iStrain, u[i]);
				elementMacroDisplacement(i, jStrain, v[i]);
				for (int k = 0; k < 3; k++) {
					u[i][k] -= ui[k][neighVid];
					v[i][k] -= uj[k][neighVid];
				}
			}
#if USE_LAME_MATRIX
			float celam = 0, cemu = 0;
#else
			float ce = 0;
#endif
			for (int ki = 0; ki < 8; ki++) {
				int kirow = ki * 3;
				for (int kj = 0; kj < 8; kj++) {
					int kjcol = kj * 3;
					for (int ri = 0; ri < 3; ri++) {
						for (int cj = 0; cj < 3; cj++) {
#if USE_LAME_MATRIX
							Lame kelamu = KLAME[kirow + ri][kjcol + cj];
							celam += u[ki][ri] * kelamu.lam() * v[kj][cj];
							cemu += u[ki][ri] * kelamu.mu() * v[kj][cj];
#else
							ce += u[ki][ri] * KE[kirow + ri][kjcol + cj] * v[kj][cj];
#endif
						}
					}
				}
			}
#if USE_LAME_MATRIX
			c = (celam * lam + cemu * mu) * prho;
#else
			c = ce * prho;
#endif
			elementCompliance[elementId] = c;
		}
	}
}

__global__ void elasticMatrix_kernel_wise_opt(
	int nv, int iStrain, int jStrain,
	devArray_t<float*, 3> ui, devArray_t<float*, 3> uj,
	float* rholist, VertexFlags* vflags, CellFlags* eflags,
	float* elementCompliance
) {
#if USE_LAME_MATRIX
	__shared__ Lame KLAME[24][24];
	loadLameMatrix(KLAME);
	float lam = LAM[0];
	float mu = MU[0];
#else
	__shared__ float KE[24][24];
	loadTemplateMatrix(KE);
#endif
	__shared__ float uChi[2][24];
	//__shared__ float uchar[2][24][32];
	__shared__ float gSum[1][128];
	
	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;
	int vlaneid = threadIdx.x % 128;

	if (warpId < 2 && laneId == 0) {
		if (warpId == 0) {
			elementMacroDisplacement(iStrain, uChi[0]); 
		} else if (warpId == 1) {
			elementMacroDisplacement(jStrain, uChi[1]);
		}
	}

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	int vid = blockIdx.x * 128 + vlaneid;

	bool is_ghost = false;

	is_ghost = vid >= nv;

	VertexFlags vflag;
	if (!is_ghost) vflag = vflags[vid];
	is_ghost = is_ghost || vflag.is_fiction() || vflag.is_period_padding();

	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);

	if (!is_ghost) indexer.locate(vid, vflag.get_gscolor(), gGsVertexEnd);

	int elementId = -1;
	CellFlags eflag;
	float prho = 0;
	int ev[8];
	if (!is_ghost) {
		elementId = indexer.neighElement(0, gGsCellEnd, gGsCellReso).getId();
		eflag = eflags[elementId];
		prho = powf(rholist[elementId], exp_penal[0]);
	}
	if (elementId != -1 && !is_ghost) {
		is_ghost = is_ghost || eflag.is_fiction() || eflag.is_period_padding();
		for (int i = 0; i < 8; i++) {
			int id = i % 2 + i / 2 % 2 * 3 + i / 4 * 9;
			int neighVid = indexer.neighVertex(id, gGsVertexEnd, gGsVertexReso).getId();
			ev[i] = neighVid;
		}
	}
	__syncthreads();

	float c = 0;
	if (elementId != -1 && !is_ghost && !eflag.is_fiction() && !eflag.is_period_padding()) {
		float clam = 0, cmu = 0;
#pragma unroll
		for (int ki = 0; ki < 12; ki++) {
			int kibase = threadIdx.x / 128 * 12;
			float klamv = 0, kmuv = 0;
#pragma unroll
			for (int kj = 0; kj < 24; kj++) {
				Lame kelamu = KLAME[kibase + ki][kj];
				float vvchar = uChi[1][kj] - float(uj[kj % 3][ev[kj / 3]]);
				klamv +=  kelamu.lam() * vvchar;
				kmuv += kelamu.mu() * vvchar;
			}
			float uuchar = uChi[0][ki + kibase] - float(ui[ki % 3][ev[(ki + kibase) / 3]]);
			clam += uuchar * klamv;
			cmu += uuchar * kmuv;
		}
		c = clam * lam + cmu * mu;
	}
	// block reduce
	if (warpId >= 4) { gSum[0][vlaneid] = c; }
	__syncthreads();
	if (warpId < 4) { gSum[0][vlaneid] = (gSum[0][vlaneid] + c) * prho; }
	__syncthreads();
	if (warpId < 2) { gSum[0][vlaneid] += gSum[0][vlaneid + 64]; }
	__syncthreads();
	// warp reduce
	if (warpId < 1) {
		c = gSum[0][vlaneid] + gSum[0][vlaneid + 32];
		for (int offset = 16; offset > 0; offset /= 2) {
			c += shfl_down(c, offset);
		}
		if (laneId == 0) {
			elementCompliance[blockIdx.x] = c;
		}
	}
}

template<typename T>
__global__ void elasticMatrix_kernel_opt(
	int nv,
	devArray_t<devArray_t<half2*, 3>, 3> ucharlist,
	T* rholist, VertexFlags* vflags, CellFlags* eflags,
	float* elementCompliance, int pitchT
) {
	__shared__ float KE[24][24];
	loadTemplateMatrix(KE);
	__shared__ float uChi[6][24];
	__shared__ float uchar[6][24][32];
	__shared__ float gSum[21][4][32];

	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;

	if (warpId < 6 && laneId == 0) {
		if (warpId == 0) { elementMacroDisplacement<float, 0>(uChi[warpId]); }
		else if (warpId == 1) { elementMacroDisplacement<float, 1>(uChi[warpId]); }
		else if (warpId == 2) { elementMacroDisplacement<float, 2>(uChi[warpId]); }
		else if (warpId == 3) { elementMacroDisplacement<float, 3>(uChi[warpId]); }
		else if (warpId == 4) { elementMacroDisplacement<float, 4>(uChi[warpId]); }
		else if (warpId == 5) { elementMacroDisplacement<float, 5>(uChi[warpId]); }
	}

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	bool is_ghost = false;

	int vid = blockIdx.x * 32 + laneId;

	is_ghost = vid >= nv;

	VertexFlags vflag;

	if (!is_ghost) { vflag = vflags[vid]; }

	is_ghost = is_ghost || vflag.is_fiction() || vflag.is_period_padding();

	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);

	int elementId = -1;

	if (!is_ghost) {
		indexer.locate(vid, vflag.get_gscolor(), gGsVertexEnd);
		elementId = indexer.neighElement(0, gGsCellEnd, gGsCellReso).getId();
	}

	//float vol_inv = 1.f / volume;
	float prho = 0;

	CellFlags eflag;
	int ev[8];
	if (elementId != -1 && !is_ghost) {
		prho = powf(float(rholist[elementId]), exp_penal[0]);
		eflag = eflags[elementId];
		is_ghost = is_ghost || eflag.is_fiction() || eflag.is_period_padding();
		if (!is_ghost) {
			for (int i = 0; i < 8; i++) {
				int id = i % 2 + i / 2 % 2 * 3 + i / 4 * 9;
				int neighVid = indexer.neighVertex(id, gGsVertexEnd, gGsVertexReso).getId();
				ev[i] = neighVid;
			}
			if (warpId < 3) {
#pragma unroll
				for (int j = 0; j < 3; j++) {
#pragma unroll
					for (int i = 0; i < 8; i++) {
						float2 chipair = __half22float2(ucharlist[warpId][j][ev[i]]);
						uchar[warpId * 2][i * 3 + j][laneId] = uChi[warpId * 2][i * 3 + j] - chipair.x;
						uchar[warpId * 2 + 1][i * 3 + j][laneId] = uChi[warpId * 2 + 1][i * 3 + j] - chipair.y;
					}
				}
			}
		}
	}
	__syncthreads();

	float ce[21] = { 0 };
	if (elementId != -1 && !is_ghost && !eflag.is_fiction() && !eflag.is_period_padding()) {
		//float c = 0;
		int counter = 0;
#pragma unroll
		for (int iStrain = 0; iStrain < 6; iStrain++) {
#pragma unroll
			for (int jStrain = iStrain; jStrain < 6; jStrain++) {
				float kv[3] = { 0. };
#pragma unroll
				for (int ki = 0; ki < 3; ki++) {
					int kibase = warpId * 3;
#pragma unroll
					for (int kj = 0; kj < 24; kj++) {
						kv[ki] += KE[kibase + ki][kj] * uchar[jStrain][kj][laneId];
					}
					kv[ki] *= uchar[iStrain][kibase + ki][laneId];
				}
				ce[counter] += kv[0] + kv[1] + kv[2];
				counter++;
			}
		}
	}

	// block reduction
	if (warpId >= 4) {
		for (int i = 0; i < 21; i++) {
			gSum[i][warpId - 4][laneId] = ce[i] * prho;
		}
	}
	__syncthreads();
	if (warpId < 4) {
		for (int i = 0; i < 21; i++) {
			gSum[i][warpId][laneId] += ce[i] * prho;
		}
	}
	__syncthreads();
	if (warpId < 2) {
		for (int i = 0; i < 21; i++) {
			gSum[i][warpId][laneId] += gSum[i][warpId + 2][laneId];
		}
	}
	__syncthreads();
	if (warpId == 0) {
		for (int i = 0; i < 21; i++) {
			ce[i] = gSum[i][0][laneId] + gSum[i][1][laneId];
		}
		// warp reduce
		for (int offset = 16; offset > 0; offset /= 2) {
#pragma unroll
			for (int i = 0; i < 21; i++) {
				ce[i] += shfl_down(ce[i], offset);
			}
		}
		if (laneId == 0) {
			for (int i = 0; i < 21; i++) {
				elementCompliance[blockIdx.x + i * pitchT] = ce[i];
			}
		}
	}
}

double homo::Homogenization::elasticMatrix(int i, int j)
{
	NO_SUPPORT_ERROR;
#if 0
	grid->useGrid_g();
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, grid->n_gsvertices(), 256);
	auto Cebuffer = getTempPool().getBuffer(grid->n_gscells() * sizeof(float));
	float* Ce = Cebuffer.template data<float>();
	init_array(Ce, 0.f, grid->n_gscells());
	auto rholist = grid->rho_g;
	int nv = grid->n_gsvertices();
	devArray_t<float*, 3> ui{ grid->u_g[0], grid->u_g[1], grid->u_g[2] };
	devArray_t<float*, 3> uj{ grid->uchar_g[0], grid->uchar_g[1], grid->uchar_g[2] };
	grid->v3_upload(ui.data(), grid->uchar_h[i]);
	grid->v3_upload(uj.data(), grid->uchar_h[j]);
	auto vflags = grid->vertflag;
	auto eflags = grid->cellflag;
	elasticMatrix_kernel_wise << <grid_size, block_size >> > (nv, i, j, ui, uj, rholist, vflags, eflags, Ce);
	cudaDeviceSynchronize();
	cuda_error_check;

	// DEBUG
	if (0) {
		char buf[100];
		sprintf_s(buf, "Ce%d%d", i, j);
		std::vector<float> cehost(grid->n_gscells());
		cudaMemcpy(cehost.data(), Ce, sizeof(float) * grid->n_gscells(), cudaMemcpyDeviceToHost);
		grid->array2matlab(buf, cehost.data(), grid->n_gscells());
	}

	float C = dump_array_sum(Ce, grid->n_gscells());
	return C;
#endif
}

// 8 warps
template<int N, int BlockSize = 256>
__global__ void fillHalfVertices_kernel(
	devArray_t<devArray_t<float*, 3>, N> uchar,
	devArray_t<devArray_t<float*, 3>, N> dst,
	devArray_t<int, 8> validGsEnd,
	int whichhalf
) {
	__shared__ int gsVHalfReso[3][8];
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	int ereso[3] = { gGridCellReso[0], gGridCellReso[1], gGridCellReso[2] };
	if (threadIdx.x < 8) {
		int org[3] = { threadIdx.x % 2, threadIdx.x / 2 % 2, threadIdx.x / 4 };
		int orgid = threadIdx.x;
		gsVHalfReso[0][orgid] = (ereso[0] - org[0]) / 2 + 1;
		gsVHalfReso[1][orgid] = (ereso[1] - org[1]) / 2 + 1;
		gsVHalfReso[2][orgid] = (ereso[2] / 2 - org[2]) / 2 + 1;
	}
	__syncthreads();
	if (tid >= validGsEnd[7]) return;
	int setcolor = 0;
	for (int i = 0; i < 8; i++) {
		if (tid < validGsEnd[i]) setcolor = i;
		else break;
	}
	int gscolor = 7 - setcolor;
	int setvid = tid - (setcolor == 0 ? 0 : validGsEnd[setcolor - 1]);
	int vpos[3] = {
		setvid % gsVHalfReso[0][setcolor] * 2 + setcolor % 2,
		setvid / gsVHalfReso[0][setcolor] % gsVHalfReso[1][setcolor] * 2 + setcolor / 2 % 2, 
		setvid / (gsVHalfReso[0][setcolor] * gsVHalfReso[1][setcolor]) * 2 + setcolor / 4
	};
	if (whichhalf == 1) vpos[2] += ereso[2] / 2;
	int lexid = vpos[0] + vpos[1] * (ereso[0] + 1) + vpos[2] * (ereso[0] + 1) * (ereso[1] + 1);
	int gsid = lexi2gs(vpos, gGsVertexReso, gGsVertexEnd);
	int nvhalf = (ereso[0] + 1) * (ereso[1] + 1) * (ereso[0] / 2 + 1);
	// Note : so vertices vector should allocate more memory for this  
	int alignedOffset = round(nvhalf * sizeof(float), 512) / sizeof(float);
	for (int i = 0; i < N; i++) {
		float* base[3] = { dst[i][0], dst[i][1], dst[i][2] };
		if (whichhalf == 1) {
			base[0] += alignedOffset; base[1] += alignedOffset; base[2] += alignedOffset;
		}
		for (int j = 0; j < 3; j++) {
			base[j][lexid] = uchar[i][j][gsid];
		}
	}
}

template <typename T, int BlockSize = 256>
__global__ void fillTotalVertices_kernel(
	int nv, VertexFlags *vflags,
	devArray_t<devArray_t<T *, 3>, 6> uchar,
	devArray_t<devArray_t<half2 *, 3>, 3> dst)
{
	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= nv) return;
	int ereso[3] = { gGridCellReso[0], gGridCellReso[1], gGridCellReso[2] };
	int vid = tid;
	VertexFlags vflag = vflags[vid];
	bool fiction = vflag.is_period_padding() || vflag.is_fiction();

	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);

	if (!fiction) {
		//indexer.locate(vid, vflag.get_gscolor(), gGsVertexEnd);
		//auto pos = indexer.getPos();
		//int p[3] = { pos.x - 1, pos.y - 1, pos.z - 1 };
		//int lexid = p[0] + p[1] * (ereso[0] + 1) + p[2] * (ereso[0] + 1) * (ereso[1] + 1);
		for (int i = 0; i < 3; i++) {
			half2 chis{ half(uchar[i * 2][0][vid]), half(uchar[i * 2 + 1][0][vid]) };
			dst[i][0][vid] = chis;
			chis = half2{ half(uchar[i * 2][1][vid]), half(uchar[i * 2 + 1][1][vid]) };
			dst[i][1][vid] = chis;
			chis = half2{ half(uchar[i * 2][2][vid]), half(uchar[i * 2 + 1][2][vid]) };
			dst[i][2][vid] = chis;
		}
	}
}

void homo::Homogenization::elasticMatrix(double C[6][6])
{
	mg_->reset_displacement();
	//auto vidmap = grid->getVertexLexidMap();
	//auto eidmap = grid->getCellLexidMap();
	//grid->array2matlab("eidmap", eidmap.data(), eidmap.size());
	//grid->array2matlab("vidmap", vidmap.data(), vidmap.size());
	for (int i = 0; i < 6; i++) {
		grid->useFchar(i);
		grid->useUchar(i);
		grid->translateForce(2, grid->u_g);
		mg_->solveEquation(config.femRelThres);
		grid->setUchar(i, grid->getDisplacement());
	}

	float vol = grid->n_cells();

	printf("n_cell = %d\n", grid->n_cells());

	use4Bytesbank();
	grid->useGrid_g();
	if (config.useManagedMemory) {
		devArray_t<devArray_t<half*, 3>, 6> ucharlist;
		for (int i = 0; i < 6; i++) {
			for (int j = 0; j < 3; j++) {
				ucharlist[i][j] = grid->uchar_h[i][j];
			}
		}
		auto rho_g = grid->rho_g;
		VertexFlags* vflags = grid->vertflag;
		CellFlags* eflags = grid->cellflag;
		int nv = grid->n_gsvertices();
		size_t grid_size, block_size;
		// prefecth unified memory data to device memory
		devArray_t<devArray_t<half2*, 3>, 3> dst;
		for (int k = 0; k < 3; k++) {
			dst[0][k] = reinterpret_cast<half2*>(grid->f_g[k]);
			dst[1][k] = reinterpret_cast<half2*>(grid->u_g[k]);
			dst[2][k] = reinterpret_cast<half2*>(grid->r_g[k]);
		}
		make_kernel_param(&grid_size, &block_size, nv, 256);
		fillTotalVertices_kernel << <grid_size, block_size >> > (nv, vflags, ucharlist, dst);
		cudaDeviceSynchronize();
		cuda_error_check;
		// compute element energy and sum
		make_kernel_param(&grid_size, &block_size, nv * 8, 256);
		int pitchT = round(grid_size, 128);
		auto buffer = getTempBuffer(pitchT * 21 * sizeof(float));
		init_array(buffer.template data<float>(), 0.f, pitchT * 21);
		//_TIC("ematopt")
		elasticMatrix_kernel_opt << <grid_size, block_size >> > (nv, dst,
			rho_g, vflags, eflags,
			buffer.template data<float>(), pitchT);
		cudaDeviceSynchronize();
		//_TOC;
		//printf("elasticMatrix_kernel_opt  time = %4.2f ms\n", tictoc::get_record("ematopt"));
		cuda_error_check;
		int counter = 0;
		for (int i = 0; i < 6; i++) {
			for (int j = i; j < 6; j++) {
				C[i][j] = parallel_sum(buffer.template data<float>() + counter * pitchT, pitchT) / vol;
				counter++;
			}
		}
		for (int i = 0; i < 6; i++) { for (int j = 0; j < i; j++) { C[i][j] = C[j][i]; } }

		for (int i = 0; i < 3; i++) {
			cudaMemset(grid->u_g[i], 0, nv * sizeof(float));
			cudaMemset(grid->r_g[i], 0, nv * sizeof(float));
			cudaMemset(grid->f_g[i], 0, nv * sizeof(float));
		}
		cudaDeviceSynchronize();
		cuda_error_check;
	}
	else {
		NO_SUPPORT_ERROR;
		#if 0
		size_t grid_size, block_size;
		int nv = grid->n_gsvertices();
		make_kernel_param(&grid_size, &block_size, nv * 2, 256);
		auto buffer = getTempBuffer(grid_size * sizeof(float));
		float* Ce = buffer.template data<float>();
		init_array(Ce, 0.f, grid_size);
		auto rholist = grid->rho_g;
		auto vflags = grid->vertflag;
		auto eflags = grid->cellflag;
		for (int i = 0; i < 6; i++) {
			devArray_t<float*, 3> ui{ grid->u_g[0], grid->u_g[1], grid->u_g[2] };
			grid->v3_upload(ui.data(), grid->uchar_h[i]);
			for (int j = i; j < 6; j++) {
				devArray_t<float*, 3> uj{ grid->uchar_g[0], grid->uchar_g[1], grid->uchar_g[2] };
				grid->v3_upload(uj.data(), grid->uchar_h[j]);
				//_TIC("ematopt");
				elasticMatrix_kernel_wise_opt << <grid_size, block_size >> > (nv, i, j, ui, uj, rholist, vflags, eflags, Ce);
				cudaDeviceSynchronize();
				//_TOC;
				//printf("elasticMatrix_kernel_wise_opt  time = %4.2f ms\n", tictoc::get_record("ematopt"));
				cuda_error_check;
				C[i][j] = dump_array_sum(Ce, grid_size) / vol;
			}
		}
		for (int i = 0; i < 6; i++) { for (int j = 0; j < i; j++) { C[i][j] = C[j][i]; } }
		#endif
	}

	for (int i = 0; i < 6; i++) {
		for (int j = i + 1; j < 6; j++) {
			C[i][j] = C[j][i];
		}
	}	
	//eigen2ConnectedMatlab("C", Eigen::Matrix<double, 6, 6>::Map(C[0], 6, 6));
}


template<typename T>
__global__ void Sensitivity_kernel_wise_opt_2(
	int nv, VertexFlags* vflags, CellFlags* eflags,
	int iStrain, int jStrain,
	devArray_t<float*, 3> ui, devArray_t<float*, 3> uj,
	T* rholist,
	devArray_t<devArray_t<float, 6>, 6> dc,
	float* sens, float volume,
	int pitchT, bool lexiOrder
) {
	__shared__ float KE[24][24];
	__shared__ float dC[6][6];
	__shared__ float uChi[2][24];
	//__shared__ float uchar[2][24][128];
	__shared__ float gSum[1][128];

	//__shared__ int EVid[8];

	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;

	if (warpId < 2 && laneId == 0) {
		int strainId = warpId == 0 ? iStrain : jStrain;
		elementMacroDisplacement(strainId, uChi[warpId]);
	}

	if (threadIdx.x < 36) {
		dC[threadIdx.x / 6][threadIdx.x % 6] = dc[threadIdx.x / 6][threadIdx.x % 6];
	}

	loadTemplateMatrix(KE);

	bool is_ghost = false;

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

	int vlaneid = threadIdx.x % 128;

	int vid = blockIdx.x * 128 + vlaneid;

	is_ghost = vid >= nv;

	VertexFlags vflag;
	if (!is_ghost) vflag = vflags[vid];
	if (vflag.is_fiction() || vflag.is_period_padding()) is_ghost = true;

	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);

	if (!is_ghost) {
		indexer.locate(vid, vflag.get_gscolor(), gGsVertexEnd);
	}

	int elementId;
	if (!is_ghost) elementId = indexer.neighElement(0, gGsCellEnd, gGsCellReso).getId();

	float vol_inv = 1.f / volume;

	CellFlags eflag;
	int ev[8];
	if (elementId != -1 && !is_ghost) {
		eflag = eflags[elementId];
		is_ghost = is_ghost || eflag.is_fiction() || eflag.is_period_padding();
		if (!is_ghost) {
			for (int i = 0; i < 8; i++) {
				int id = i % 2 + i / 2 % 2 * 3 + i / 4 * 9;
				int neighVid = indexer.neighVertex(id, gGsVertexEnd, gGsVertexReso).getId();
				ev[i] = neighVid;
			}
		}
	}
	__syncthreads();
	float dc_lane{ 0. };
	float prho = 0;
	// 8 warp to 32 vertices
	if (elementId != -1 && !is_ghost) {
		float pwn = exp_penal[0];
		prho = pwn * powf(rholist[elementId], pwn - 1);
		/*for (int iStrain = 0; iStrain < 6; iStrain++)*/ {
			/*for (int jStrain = iStrain; jStrain < 6; jStrain++)*/ {
				float c[12] = { 0. };
#pragma unroll
				for (int ki = 0; ki < 12; ki++) {
					int kibase = threadIdx.x / 128 * 12;
#pragma unroll
					for (int kj = 0; kj < 24; kj++) {
						c[ki] += KE[kibase + ki][kj] * (uChi[1][kj] - float(uj[kj % 3][ev[kj / 3]]));
					}
					c[ki] *= (uChi[0][kibase + ki] - float(ui[(kibase + ki) % 3][ev[(kibase + ki) / 3]]));
				}
				dc_lane = c[0] + c[1] + c[2] + c[3] + c[4] + c[5] + c[6] + c[7] + c[8] + c[9] + c[10] + c[11];
			}
		}
	}
	// block reduce
	if (warpId >= 4) { gSum[0][vlaneid] = dc_lane; }
	__syncthreads();
	if (warpId < 4 && !is_ghost && elementId != -1) {
		float s = 0;
		float vp = vol_inv * prho;
		dc_lane += gSum[0][vlaneid];
		float dclast = dC[iStrain][jStrain];
		if (iStrain != jStrain) dclast += dC[jStrain][iStrain];
		s = dc_lane * dclast * vp;
		if (!lexiOrder) {
			sens[elementId] += s;
		}
		else {
			auto p = indexer.getPos();
			// p -> element pos -> element pos without padding 
			p.x -= 2; p.y -= 2; p.z -= 2;
			if (p.x < 0 || p.y < 0 || p.z < 0) print_exception;
			int lexid = p.x + (p.y + p.z * gGridCellReso[1]) * pitchT;
			sens[lexid] += s;
		}
	}
}


// use vector stored in F(chi_0,chi_1) U(chi_2,chi_3) R(chi_4,chi_5)
template <typename T, int BlockSize = 256>
__global__ void Sensitivity_kernel_opt_2(
	int nv, VertexFlags *vflags, CellFlags *eflags,
	devArray_t<devArray_t<half2 *, 3>, 3> ucharlist,
	T *rholist,
	devArray_t<devArray_t<float, 6>, 6> dc,
	float *sens, float volume,
	int pitchT, bool lexiOrder)
{
	__shared__ float KE[24][24];
	__shared__ float dC[6][6];
	__shared__ float uChi[6][24];
	__shared__ float uchar[6][24][32];
	__shared__ float gSum[6][6][4][32];

	int warpId = threadIdx.x / 32;
	int laneId = threadIdx.x % 32;

	if (warpId < 6 && laneId == 0) {
		if (warpId == 0) {
			elementMacroDisplacement<float, 0>(uChi[warpId]);
		} else if (warpId == 1) {
			elementMacroDisplacement<float, 1>(uChi[warpId]);
		} else if (warpId == 2) {
			elementMacroDisplacement<float, 2>(uChi[warpId]);
		} else if (warpId == 3) {
			elementMacroDisplacement<float, 3>(uChi[warpId]);
		} else if (warpId == 4) {
			elementMacroDisplacement<float, 4>(uChi[warpId]);
		} else if (warpId == 5) {
			elementMacroDisplacement<float, 5>(uChi[warpId]);
		}
	}

	if (threadIdx.x < 36) {
		dC[threadIdx.x / 6][threadIdx.x % 6] = dc[threadIdx.x / 6][threadIdx.x % 6];
	}

	loadTemplateMatrix(KE);

	bool is_ghost = false;

	size_t tid = blockIdx.x * blockDim.x + threadIdx.x;


	int vid = blockIdx.x * 32 + laneId;
	
	is_ghost = vid >= nv;

	VertexFlags vflag;
	if (!is_ghost) vflag = vflags[vid];
	if (vflag.is_fiction() || vflag.is_period_padding()) is_ghost = true;

	GridVertexIndex indexer(gGridCellReso[0], gGridCellReso[1], gGridCellReso[2]);

	if (!is_ghost) {
		indexer.locate(vid, vflag.get_gscolor(), gGsVertexEnd);
	}

	int elementId;
	if (!is_ghost) elementId = indexer.neighElement(0, gGsCellEnd, gGsCellReso).getId();

	float vol_inv = 1.f / volume;

	CellFlags eflag;
	int ev[8];
	if (elementId != -1 && !is_ghost) {
		eflag = eflags[elementId];
		is_ghost = is_ghost || eflag.is_fiction() || eflag.is_period_padding();
		if (!is_ghost) {
			for (int i = 0; i < 8; i++) {
				int id = i % 2 + i / 2 % 2 * 3 + i / 4 * 9;
				int neighVid = indexer.neighVertex(id, gGsVertexEnd, gGsVertexReso).getId();
				ev[i] = neighVid;
			}
			if (warpId < 3) {
#pragma unroll
				for (int j = 0; j < 3; j++) {
#pragma unroll
					for (int i = 0; i < 8; i++) {
						float2 chipair = __half22float2(ucharlist[warpId][j][ev[i]]);
						uchar[warpId * 2][i * 3 + j][laneId] = uChi[warpId * 2][i * 3 + j] - chipair.x;
						uchar[warpId * 2 + 1][i * 3 + j][laneId] = uChi[warpId * 2 + 1][i * 3 + j] - chipair.y;
					}
				}
			}
		}
	}
	__syncthreads();

	float dc_lane[6][6] = { 0. };
	float prho = 0;
	// 8 warp to 32 vertices
	if (elementId != -1 && !is_ghost) {
		float pwn = exp_penal[0];
		prho = pwn * powf(float(rholist[elementId]), pwn - 1);
#pragma unroll
		for (int iStrain = 0; iStrain < 6; iStrain++) {
#pragma unroll
			for (int jStrain = iStrain; jStrain < 6; jStrain++) {
				float c[3] = { 0. };
#pragma unroll
				for (int ki = 0; ki < 3; ki++) {
					int kibase = warpId * 3;
#pragma unroll
					for (int kj = 0; kj < 24; kj++) {
						c[ki] += KE[kibase + ki][kj] * uchar[jStrain][kj][laneId];
					}
					c[ki] *= uchar[iStrain][kibase + ki][laneId];
				}
				dc_lane[iStrain][jStrain] = c[0] + c[1] + c[2];
			}
		}
	}
	// block reduce
	if (warpId >= 4) {
		for (int i = 0; i < 6; i++) {
			for (int j = i; j < 6; j++) {
				gSum[i][j][warpId - 4][laneId] = dc_lane[i][j];
			}
		}
	}
	__syncthreads();
	if (warpId < 4) {
		for (int i = 0; i < 6; i++) {
			for (int j = i; j < 6; j++) {
				gSum[i][j][warpId][laneId] += dc_lane[i][j];
			}
		}
	}
	__syncthreads();
	if (warpId < 2) {
		for (int i = 0; i < 6; i++) {
			for (int j = i; j < 6; j++) {
				gSum[i][j][warpId][laneId] += gSum[i][j][warpId + 2][laneId];
			}
		}
	}
	__syncthreads();
	if (warpId < 1 && !is_ghost && elementId != -1) {
		float s = 0;
		float vp = vol_inv * prho;
		for (int i = 0; i < 6; i++) {
			for (int j = i; j < 6; j++) {
				dc_lane[i][j] = gSum[i][j][0][laneId] + gSum[i][j][1][laneId];
				float dclast = dC[i][j];
				if (i != j) dclast += dC[j][i];
				s += dc_lane[i][j] * dclast * vp;
			}
		}
		if (!lexiOrder) {
			sens[elementId] = s;
		} else {
			auto p = indexer.getPos();
			// p -> element pos -> element pos without padding 
			p.x -= 2; p.y -= 2; p.z -= 2;
			if (p.x < 0 || p.y < 0 || p.z < 0) print_exception;
			int lexid = p.x + (p.y + p.z * gGridCellReso[1]) * pitchT;
			sens[lexid] = s;
		}
	}
}

void homo::Homogenization::Sensitivity(float dC[6][6], float* sens, int pitchT, bool lexiOrder /*= false*/)
{
	grid->useGrid_g();
	devArray_t<devArray_t<float, 6>, 6> dc;
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 6; j++) {
			dc[i][j] = dC[i][j];
		}
	}

	init_array(sens, 0.f, grid->cellReso[1] * grid->cellReso[2] * pitchT);

	int nv = grid->n_gsvertices();
	auto vflags = grid->vertflag;
	auto eflags = grid->cellflag;
	auto rholist = grid->rho_g;
	float volume = grid->n_cells();
	size_t grid_size, block_size;
	make_kernel_param(&grid_size, &block_size, nv, 256);
	if (!config.useManagedMemory) {
		NO_SUPPORT_ERROR;
		#if 0
		for (int iStrain = 0; iStrain < 6; iStrain++) {
			devArray_t<float*, 3> ui{ grid->u_g[0], grid->u_g[1], grid->u_g[2] };
			grid->v3_upload(ui.data(), grid->uchar_h[iStrain]);
			for (int jStrain = iStrain; jStrain < 6; jStrain++) {
				devArray_t<float*, 3> uj{ grid->uchar_g[0], grid->uchar_g[1], grid->uchar_g[2] };
				grid->v3_upload(uj.data(), grid->uchar_h[jStrain]);
#if 0
#else
				use4Bytesbank();
				make_kernel_param(&grid_size, &block_size, nv * 2, 256);
				Sensitivity_kernel_wise_opt_2 << <grid_size, block_size >> > (nv, vflags, eflags,
					iStrain, jStrain, ui, uj,
					rholist, dc, sens, volume, pitchT, lexiOrder);
#endif
				cudaDeviceSynchronize();
				cuda_error_check;
			}
		}
		#endif
	}
	else {
		printf("Sensitivity analysis using managed memory...\n");
		devArray_t<devArray_t<half*, 3>, 6> uchar;
		for (int i = 0; i < 6; i++) {
			for (int j = 0; j < 3; j++) {
				uchar[i][j] = grid->uchar_h[i][j];
			}
		}
		devArray_t<devArray_t<half2*, 3>, 3> dst;
		for (int k = 0; k < 3; k++) {
			dst[0][k] = reinterpret_cast<half2*>(grid->f_g[k]);
			dst[1][k] = reinterpret_cast<half2*>(grid->u_g[k]);
			dst[2][k] = reinterpret_cast<half2*>(grid->r_g[k]);
		}
		make_kernel_param(&grid_size, &block_size, nv, 256);
		fillTotalVertices_kernel << <grid_size, block_size >> > (nv, vflags, uchar, dst);
		cudaDeviceSynchronize();
		cuda_error_check;
		make_kernel_param(&grid_size, &block_size, nv * 8, 256);
		Sensitivity_kernel_opt_2 << <grid_size, block_size >> > (nv, vflags, eflags,
			dst,
			rholist, dc, sens, volume, pitchT, lexiOrder);
		cudaDeviceSynchronize();
		cuda_error_check;
	}
	cuda_error_check;

	for (int i = 0; i < 3; i++) {
		cudaMemset(grid->u_g[i], 0, nv * sizeof(float));
		cudaMemset(grid->r_g[i], 0, nv * sizeof(float));
		cudaMemset(grid->f_g[i], 0, nv * sizeof(float));
	}
	cudaDeviceSynchronize();
	cuda_error_check;

	// DEBUG
	if (0) {
		int slist = grid->cellReso[2] * grid->cellReso[1] * grid->cellReso[2];
		std::vector<float> senslist(slist);
		cudaMemcpy2D(senslist.data(), grid->cellReso[0] * sizeof(float),
			sens, pitchT * sizeof(float),
			grid->cellReso[0] * sizeof(float), grid->cellReso[1] * grid->cellReso[2],
			cudaMemcpyDeviceToHost);
		cuda_error_check;
		grid->array2matlab("senslist", senslist.data(), senslist.size());
	}
}


