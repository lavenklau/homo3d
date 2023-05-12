#include "culib/lib.cuh"
#include "culib/vector_intrinsic.cuh"
#include "culib/warp_primitive.cuh"
#include "cuda_runtime.h"
#include "cub/util_ptx.cuh"
#include <cuda/std/tuple>

using namespace culib;

struct Lame {
	short2 lammu;
	__host__ __device__ float lam() volatile { return lammu.x; }
	__host__ __device__ float mu() volatile { return lammu.y; }
	__host__ __device__ volatile Lame& operator=(const Lame& lame2) volatile{
		lammu.x = lame2.lammu.x;
		lammu.y = lame2.lammu.y;
		return *this;
	}
};

extern __constant__ double* gU[3];
extern __constant__ double* gF[3];
extern __constant__ double* gR[3];
extern __constant__ double* gUfine[3];
extern __constant__ double* gFfine[3];
extern __constant__ double* gRfine[3];
extern __constant__ double* gUcoarse[3];
extern __constant__ double* gFcoarse[3];
extern __constant__ double* gRcoarse[3];
extern __constant__ float gKE[24][24];
extern __constant__ double gKEd[24][24];
extern __constant__ Lame gKLame[24][24];
extern __constant__ double gLM[5];
// USE_DOUBLE_STENCIL
extern __constant__ double* rxstencil[27][9];
extern __constant__ double* rxCoarseStencil[27][9];
extern __constant__ double* rxFineStencil[27][9];

extern __constant__ int gUpCoarse[3];
extern __constant__ int gDownCoarse[3];
extern __constant__ int gGridCellReso[3];
extern __constant__ int gCoarseGridCellReso[3];
extern __constant__ int gGsCellReso[3][8];
extern __constant__ int gGsVertexReso[3][8];
extern __constant__ int gGsVertexEnd[8];
extern __constant__ int gGsCellEnd[8];
extern __constant__ int gGsFineVertexReso[3][8];
extern __constant__ int gGsCoarseVertexReso[3][8];
extern __constant__ int gGsCoarseVertexEnd[8];
extern __constant__ int gGsFineVertexEnd[8];

//__constant__ double* guchar[6][3];
//__constant__ double* gfchar[6][3];
extern __constant__ double* guchar[3];

extern __constant__ float exp_penal[1];
extern __constant__ float LAM[1];
extern __constant__ float MU[1];


#define print_exception printf("\033[31mexception at line %d\n\033[0m",__LINE__)

template<typename T>
__device__ void loadTemplateMatrix(volatile T KE[24][24]) {
	for (int i = threadIdx.x; i < 24 * 24; i += blockDim.x) {
		int ri = i / 24;
		int cj = i % 24;
		if (std::is_same_v<T, float>) {
			KE[ri][cj] = gKE[ri][cj];
		}
		else if (std::is_same_v<T, double>) {
			KE[ri][cj] = gKEd[ri][cj];
		}
		else {
			print_exception;
		}
	}
	__syncthreads();
}

__device__ inline void loadLameMatrix(volatile Lame Klm[24][24]) {
	int base = 0;
	while (threadIdx.x + base < 24 * 24) {
		int id = threadIdx.x + base;
		int i = id / 24, j = id % 24;
		Klm[i][j] = gKLame[i][j];
		base += blockDim.x;
	}
	__syncthreads();
}

struct GridVertexIndex {
	short3 cellReso;
	short set_id = -1;
	char3 org;
	short4 gsVertexReso;
	short4 gsPos;

	__host__ __device__ short3 getPos(void) {
		short3 pos;
		pos.x = org.x + gsPos.x * 2;
		pos.y = org.y + gsPos.y * 2;
		pos.z = org.z + gsPos.z * 2;
		return pos;
	}

	__host__ __device__ GridVertexIndex(int resox, int resoy, int resoz) {
		cellReso.x = resox;
		cellReso.y = resoy;
		cellReso.z = resoz;
	}

	__device__ bool locate(int vid, int color, volatile int gsVertexEnd[8]) {
		set_id = color;

		if (set_id == -1) return false;

		org.x = set_id % 2;
		org.y = set_id / 2 % 2;
		org.z = set_id / 4;

		gsVertexReso.x = (cellReso.x + 2 - org.x) / 2 + 1;
		gsVertexReso.y = (cellReso.y + 2 - org.y) / 2 + 1;
		gsVertexReso.z = (cellReso.z + 2 - org.z) / 2 + 1;

		int gsid = set_id == 0 ? vid : vid - gsVertexEnd[set_id - 1];

		gsPos.x = gsid % gsVertexReso.x;
		gsPos.y = gsid / gsVertexReso.x % gsVertexReso.y;
		gsPos.z = gsid / (gsVertexReso.x * gsVertexReso.y);

		return set_id != -1;
	}

	struct IDColor {
		cuda::std::tuple<int, int> neigh_color;
		__device__ int getId(void) { return cuda::std::get<0>(neigh_color); }
		__device__ int getColor(void) { return cuda::std::get<1>(neigh_color); }
	};

	// id must be a positive number less than 27
	//template<bool debug = false >
	__device__ IDColor neighVertex(int id, volatile int* gsVertexEnd, volatile int gsVertexResoAll[3][8]/*, bool debug = false*/) {
		unsigned int uintNeighoffset = 0;
		cub::BFI(uintNeighoffset, uintNeighoffset, id % 3 - 1, 0, 8);
		cub::BFI(uintNeighoffset, uintNeighoffset, id / 3 % 3 - 1, 8, 8);
		cub::BFI(uintNeighoffset, uintNeighoffset, id / 9 - 1, 16, 8);
		//__vabs4(intNeighoffset);
		unsigned int uintOrg = 0;
		cub::BFI(uintOrg, uintOrg, org.x, 0, 8);
		cub::BFI(uintOrg, uintOrg, org.y, 8, 8);
		cub::BFI(uintOrg, uintOrg, org.z, 16, 8);


		unsigned int uintNeighorg = __vadd4(__vadd4(uintOrg, uintNeighoffset), 0x02020202);
		// mod 2
		uintNeighorg &= 0x01010101;

		unsigned int orgdiff = __vsub4(uintNeighorg, uintOrg);

		unsigned int neiEqdiff = ~__vcmpeq4(uintNeighoffset, orgdiff);

		//if (debug) {
		//	printf("neighoff = %x, orgdiff = %x, neieqdiff = %x\n", uintNeighoffset, orgdiff, neiEqdiff);
		//}

		// reduce to
		unsigned int posDiff = uintNeighoffset ^ orgdiff ^ (orgdiff & neiEqdiff);

		int neigh_setid = cub::BFE(uintNeighorg, 0, 8) + cub::BFE(uintNeighorg, 8, 8) * 2 + cub::BFE(uintNeighorg, 16, 8) * 4;

		if (neigh_setid >= 8 || neigh_setid < 0) { printf("error%d\n", __LINE__); }

		int base = neigh_setid == 0 ? 0 : gsVertexEnd[neigh_setid - 1];

		int neipos[3] = {
			gsPos.x + (signed char)cub::BFE(posDiff,0,8),
			gsPos.y + (signed char)cub::BFE(posDiff,8,8),
			gsPos.z + (signed char)cub::BFE(posDiff,16,8)
		};

		//if (/*neipos[0] == 16 && neipos[1] == 17 && neipos[2] == 8 &&*/ gsPos.x == 18 && gsPos.y == 19 && gsPos.z == 10) {
		//	//printf("error%d : (%d, %d, %d)  (%d, %d, %d)\n", __LINE__, neipos[0], neipos[1], neipos[2], gsPos.x, gsPos.y, gsPos.z);
		//	printf("neighoff = %08x, orgdiff = %08x, neieqdiff = %08x\n", uintNeighoffset, orgdiff, neiEqdiff);
		//}

		if (neipos[0] < 0 || neipos[0] >= gsVertexResoAll[0][neigh_setid] ||
			neipos[1] < 0 || neipos[1] >= gsVertexResoAll[1][neigh_setid] ||
			neipos[2] < 0 || neipos[2] >= gsVertexResoAll[2][neigh_setid]) {
			return { {-1,-1} };
		}

		int neigh_id = base + neipos[0] + neipos[1] * gsVertexResoAll[0][neigh_setid] +
			neipos[2] * gsVertexResoAll[0][neigh_setid] * gsVertexResoAll[1][neigh_setid];

		return { {neigh_id,neigh_setid} };
	}

	// id is in 0~7
	__device__ IDColor neighElement(int id, int gsCellEnd[8], int gsCellReso[3][8]) {
#if 0
		unsigned int vertex_set_org = pack(make_char3(set_id % 2, set_id / 2 % 2, set_id / 4));
		unsigned int neigh_set_offset = __vsub4(pack(make_char3(id % 2, id / 2 % 2, id / 4)), 0x01010101);
		unsigned int neigh_set_org = __vadd4(vertex_set_org, neigh_set_offset);
		neigh_set_org = __vabs4(neigh_set_org);

		int neigh_set_id =
			cub::BFE(neigh_set_org, 0, 8) +
			cub::BFE(neigh_set_org, 8, 8) * 2 +
			cub::BFE(neigh_set_org, 16, 8) * 4;
		
		if (neigh_set_id < 0 || neigh_set_id >= 8) print_exception;

		unsigned int gsPosOffset = (~__vcmpeq4(neigh_set_org, vertex_set_org)) & __vcmpeq4(vertex_set_org, 0);

		//if(0){
		//	short3 vorg{ cub::BFE(vertex_set_org, 0, 8),cub::BFE(vertex_set_org, 8, 8),cub::BFE(vertex_set_org, 16, 8) };
		//	if (vorg.x == 0 && vorg.y == 0 && vorg.z == 0) {
		//		if (gsPos.x == 1 && gsPos.y == 1 && gsPos.z == 1) {
		//			printf("nsorg = %08x   vsetorg = %08x   gpoff = %08x\n", neigh_set_org, vertex_set_org, gsPosOffset);
		//		}
		//	}
		//}

		short3 gsElementPos;
		gsElementPos.x = (signed char)cub::BFE(gsPosOffset, 0, 8) + gsPos.x;
		gsElementPos.y = (signed char)cub::BFE(gsPosOffset, 8, 8) + gsPos.y;
		gsElementPos.z = (signed char)cub::BFE(gsPosOffset, 16, 8) + gsPos.z;

		if (gsElementPos.x < 0 || gsElementPos.x >= gsCellReso[0][neigh_set_id] ||
			gsElementPos.y < 0 || gsElementPos.y >= gsCellReso[1][neigh_set_id] ||
			gsElementPos.z < 0 || gsElementPos.z >= gsCellReso[2][neigh_set_id]
			) {
			return { {-1,-1} };
		}

		int base = neigh_set_id == 0 ? 0 : gsCellEnd[neigh_set_id - 1];

		int neigh_id = base + gsElementPos.x +
			gsElementPos.y * gsCellReso[0][neigh_set_id] +
			gsElementPos.z * gsCellReso[1][neigh_set_id] * gsCellReso[0][neigh_set_id];

		return { {neigh_id,neigh_set_id} };
#else
		int cellpos[3] = {
			gsPos.x * 2 + org.x + id % 2 - 1,
			gsPos.y * 2 + org.y + id / 2 % 2 - 1,
			gsPos.z * 2 + org.z + id / 4 - 1
		};
		int egspos[3] = { cellpos[0] / 2, cellpos[1] / 2, cellpos[2] / 2 };
		int esetid = cellpos[0] % 2 + cellpos[1] % 2 * 2 + cellpos[2] % 2 * 4;
		int base = esetid == 0 ? 0 : gsCellEnd[esetid - 1];
		if (cellpos[0] < 0 || cellpos[0] >= cellReso.x + 2 ||
			cellpos[1] < 0 || cellpos[1] >= cellReso.y + 2 ||
			cellpos[2] < 0 || cellpos[2] >= cellReso.z + 2) {
			return { {-1,-1} };
		}
		int eid = base +
			egspos[0] +
			egspos[1] * gsCellReso[0][esetid] +
			egspos[2] * gsCellReso[0][esetid] * gsCellReso[1][esetid];
		return { {eid, esetid} };
#endif
	}

	__device__ IDColor neighCoarseVertex(int id, int coarseRatio[3], volatile int gsCoarseVertexEnd[8], volatile int gsCoarseVertexReso[3][8], volatile int remainder[3]) {
		short3 pos;
		pos.x = gsPos.x * 2 + org.x - 1;
		pos.y = gsPos.y * 2 + org.y - 1;
		pos.z = gsPos.z * 2 + org.z - 1;
		
		if (pos.x < 0 || pos.y < 0 || pos.z < 0) print_exception;

		//bool debug = false;
		//if (pos.x == 0 && pos.y == 2 && pos.z == 0) {
		//	debug = true;
		//}

		char3 idoffCoarse = { id % 2, id / 2 % 2, id / 4 };

		remainder[0] = pos.x % coarseRatio[0];
		remainder[1] = pos.y % coarseRatio[1];
		remainder[2] = pos.z % coarseRatio[2];

		char3 idOff = { idoffCoarse.x * coarseRatio[0], idoffCoarse.y * coarseRatio[1], idoffCoarse.z * coarseRatio[2] };
		
		//char3 posOff = { pos.x % 2, pos.y % 2, pos.z % 2 };

		if (abs(idOff.x - remainder[0]) < coarseRatio[0] &&
			abs(idOff.y - remainder[1]) < coarseRatio[1] &&
			abs(idOff.z - remainder[2]) < coarseRatio[2]) {
			pos.x = pos.x / coarseRatio[0] + idoffCoarse.x + 1;
			pos.y = pos.y / coarseRatio[1] + idoffCoarse.y + 1;
			pos.z = pos.z / coarseRatio[2] + idoffCoarse.z + 1;
		} else {
			return { {-1,-1} };
		}

		//if (debug) {
		//	printf("id = %d  pos = (%ld, %ld, %ld)\n", id, pos.x, pos.y, pos.z);
		//}

		int color = pos.x % 2 + pos.y % 2 * 2 + pos.z % 2 * 4;
		int base = color == 0 ? 0 : gsCoarseVertexEnd[color - 1];
		int gsid = base +
			pos.x / 2 +
			pos.y / 2 * gsCoarseVertexReso[0][color] +
			pos.z / 2 * gsCoarseVertexReso[0][color] * gsCoarseVertexReso[1][color];
		
		//if (debug) {
		//	printf("gsid = %d\n", gsid);
		//}

		if (gsid >= gsCoarseVertexEnd[color]) {
			gsid = -1;
		}

		return { {gsid, color} };
	}

	__device__ void neighCoarseVertex(int neiVid[8], float w[8], int coarseRatio[3], volatile int gsCoarseVertexEnd[8], volatile int gsCoarseVertexReso[3][8], volatile int remainder[3]) {
		short3 pos;
		pos.x = gsPos.x * 2 + org.x - 1;
		pos.y = gsPos.y * 2 + org.y - 1;
		pos.z = gsPos.z * 2 + org.z - 1;
		
		if (pos.x < 0 || pos.y < 0 || pos.z < 0) print_exception;

		remainder[0] = pos.x % coarseRatio[0];
		remainder[1] = pos.y % coarseRatio[1];
		remainder[2] = pos.z % coarseRatio[2];

		short3 coarseBase{ pos.x / coarseRatio[0], pos.y / coarseRatio[1], pos.z / coarseRatio[2] };

		float wc = coarseRatio[0] * coarseRatio[1] * coarseRatio[2];

#pragma unroll
		for (int id = 0; id < 8; id++) {
			short3 idoffCoarse = { id % 2, id / 2 % 2, id / 4 };
			short3 idOff = { idoffCoarse.x * coarseRatio[0], idoffCoarse.y * coarseRatio[1], idoffCoarse.z * coarseRatio[2] };
			w[id] = (coarseRatio[0] - abs(idOff.x - remainder[0])) *
					(coarseRatio[1] - abs(idOff.y - remainder[1])) *
					(coarseRatio[2] - abs(idOff.z - remainder[2])) / wc;
			pos.x = coarseBase.x + idoffCoarse.x + 1;
			pos.y = coarseBase.y + idoffCoarse.y + 1;
			pos.z = coarseBase.z + idoffCoarse.z + 1;
			int color = pos.x % 2 + pos.y % 2 * 2 + pos.z % 2 * 4;
			int base = color == 0 ? 0 : gsCoarseVertexEnd[color - 1];
			int gsid = base +
				pos.x / 2 +
				pos.y / 2 * gsCoarseVertexReso[0][color] +
				pos.z / 2 * gsCoarseVertexReso[0][color] * gsCoarseVertexReso[1][color];
			if (gsid >= gsCoarseVertexEnd[color]) { gsid = -1; }
			neiVid[id] = gsid;
		}
	}

	__device__ IDColor neighFineVertex(int offset[3], int refineRatio[3], int gsFineVertexEnd[8], int gsFineVertexReso[3][8], bool circleAccess = false) {
		short3 pos;
		pos.x = gsPos.x * 2 + org.x - 1;
		pos.y = gsPos.y * 2 + org.y - 1;
		pos.z = gsPos.z * 2 + org.z - 1;
		
		//bool debug = pos.x == 0 && pos.y == 1 && pos.z == 1 &&
		//	offset[0] == 0 && offset[1] == -1 && offset[2] == 0;

		pos.x = pos.x * refineRatio[0] + offset[0];
		pos.y = pos.y * refineRatio[1] + offset[1];
		pos.z = pos.z * refineRatio[2] + offset[2];
		if (circleAccess) {
			int freso = cellReso.x * refineRatio[0];
			pos.x = (pos.x + freso) % freso;
			freso = cellReso.y * refineRatio[1];
			pos.y = (pos.y + freso) % freso;
			freso = cellReso.z * refineRatio[2];
			pos.z = (pos.z + freso) % freso;
		}
		pos.x += 1; pos.y += 1; pos.z += 1;

		int posColor = pos.x % 2 + pos.y % 2 * 2 + pos.z % 2 * 4;
		int base = posColor == 0 ? 0 : gsFineVertexEnd[posColor - 1];

		int posid = base +
			pos.x / 2 +
			pos.y / 2 * gsFineVertexReso[0][posColor] +
			pos.z / 2 * gsFineVertexReso[0][posColor] * gsFineVertexReso[1][posColor];

		if (posid >= gsFineVertexEnd[posColor]) {
			posid = -1;
		}

		//if (debug) {
		//	printf("posid = %d\n", posid);
		//}

		return { {posid,posColor} };
	}

	//__device__ int neighFineElement(int id, int refineRatio[3], int gsFineVertexEnd[8], int gsFineVertexReso[3][8]) {
	//	
	//}
};

template<typename T, int N>
__device__ void constantToShared(T(&constantSrc)[N], volatile T(&dst)[N]) {
	if (N < 32 && threadIdx.x < N) {
		dst[threadIdx.x] = constantSrc[threadIdx.x];
	}
}

template<typename T, int N>
__device__ void constantToShared(devArray_t<T, N>& constantSrc, volatile T(&dst)[N]) {
	if (N < 32 && threadIdx.x < N) {
		dst[threadIdx.x] = constantSrc[threadIdx.x];
	}
}

template<typename T, int N, int M>
__device__ void constant2DToShared(T(&constantSrc)[N][M], volatile T(&dst)[N][M]) {
	if (N * M < 32 && threadIdx.x < N * M) {
		dst[threadIdx.x / M][threadIdx.x % M] = constantSrc[threadIdx.x / M][threadIdx.x % M];
	}
}

template<typename T>
__device__ void elementMacroDisplacement(int inode, int jStrain, T u[3]) {
	short3 pos{ inode % 2, inode / 2 % 2, inode / 4 };
	switch (jStrain) {
	case 0:
		u[0] = pos.x; u[1] = 0; u[2] = 0;
		break;
	case 1:
		u[0] = 0; u[1] = pos.y; u[2] = 0;
		break;
	case 2:
		u[0] = 0; u[1] = 0; u[2] = pos.z;
		break;
	case 3: // yz
		u[0] = 0; u[1] = pos.z / 2.f; u[2] = pos.y / 2.f;
		break;
	case 4: // xz
		u[0] = pos.z / 2.f; u[1] = 0; u[2] = pos.x / 2.f;
		break;
	case 5: // xy
		u[0] = pos.y / 2.f; u[1] = pos.x / 2.f; u[2] = 0;
		break;
	}
}

template<typename T, int iStrain>
__device__ void elementMacroDisplacement(T uchi[24]) {
#pragma unroll
	for (int i = 0; i < 8; i++) {
		int p[3] = { i % 2, i / 2 % 2, i / 4 };
		if constexpr (iStrain == 0) {
			uchi[i * 3] = p[0]; uchi[i * 3 + 1] = 0; uchi[i * 3 + 2] = 0;
		} else if constexpr (iStrain == 1) {
			uchi[i * 3] = 0; uchi[i * 3 + 1] = p[1]; uchi[i * 3 + 2] = 0;
		} else if constexpr (iStrain == 2) {
			uchi[i * 3] = 0; uchi[i * 3 + 1] = 0; uchi[i * 3 + 2] = p[2];
		} else if constexpr (iStrain == 3) {
			uchi[i * 3] = 0; uchi[i * 3 + 1] = p[2] / 2.f; uchi[i * 3 + 2] = p[1] / 2.f;
		} else if constexpr (iStrain == 4) {
			uchi[i * 3] = p[2] / 2.f; uchi[i * 3 + 1] = 0; uchi[i * 3 + 2] = p[0] / 2.f;
		} else if constexpr (iStrain == 5) {
			uchi[i * 3] = p[1] / 2.f; uchi[i * 3 + 1] = p[0] / 2.f; uchi[i * 3 + 2] = 0;
		}
	}
}

template<typename T>
__device__ void elementMacroDisplacement(int iStrain, T uchi[24]) {
#pragma unroll
	for (int i = 0; i < 8; i++) {
		int p[3] = { i % 2, i / 2 % 2, i / 4 };
		if (iStrain == 0) {
			uchi[i * 3] = p[0]; uchi[i * 3 + 1] = 0; uchi[i * 3 + 2] = 0;
		} else if (iStrain == 1) {
			uchi[i * 3] = 0; uchi[i * 3 + 1] = p[1]; uchi[i * 3 + 2] = 0;
		} else if (iStrain == 2) {
			uchi[i * 3] = 0; uchi[i * 3 + 1] = 0; uchi[i * 3 + 2] = p[2];
		} else if (iStrain == 3) {
			uchi[i * 3] = 0; uchi[i * 3 + 1] = p[2] / 2.f; uchi[i * 3 + 2] = p[1] / 2.f;
		} else if (iStrain == 4) {
			uchi[i * 3] = p[2] / 2.f; uchi[i * 3 + 1] = 0; uchi[i * 3 + 2] = p[0] / 2.f;
		} else if (iStrain == 5) {
			uchi[i * 3] = p[1] / 2.f; uchi[i * 3 + 1] = p[0] / 2.f; uchi[i * 3 + 2] = 0;
		}
	}
}

__host__ __device__ inline int lexi2gs(int lexpos[3], int gsreso[3][8], int gsend[8], bool padded = false) {
	int pos[3] = { lexpos[0], lexpos[1], lexpos[2] };
	if (!padded) { 
		pos[0] += 1; pos[1] += 1; pos[2] += 1;
	}
	int org[3] = { pos[0] % 2, pos[1] % 2, pos[2] % 2 };
	int gscolor = org[0] + org[1] * 2 + org[2] * 4;
	pos[0] /= 2; pos[1] /= 2; pos[2] /= 2;
	int gsid = (gscolor == 0 ? 0 : gsend[gscolor - 1]) +
		pos[0] +
		pos[1] * gsreso[0][gscolor] +
		pos[2] * gsreso[0][gscolor] * gsreso[1][gscolor];
	return gsid;
}

__host__ __device__ inline int lexi2gs(short3 lexpos, int gsreso[3][8], int gsend[8], bool padded = false) {
	int pos[3] = { lexpos.x, lexpos.y, lexpos.z };
	if (!padded) { 
		pos[0] += 1; pos[1] += 1; pos[2] += 1;
	}
	int org[3] = { pos[0] % 2, pos[1] % 2, pos[2] % 2 };
	int gscolor = org[0] + org[1] * 2 + org[2] * 4;
	pos[0] /= 2; pos[1] /= 2; pos[2] /= 2;
	int gsid = (gscolor == 0 ? 0 : gsend[gscolor - 1]) +
		pos[0] +
		pos[1] * gsreso[0][gscolor] +
		pos[2] * gsreso[0][gscolor] * gsreso[1][gscolor];
	return gsid;
}

#define NO_SUPPORT_ERROR                                                               \
	do                                                                                 \
	{                                                                                  \
		printf("Error : No support!\n -> file %s\n -> line %d\n", __FILE__, __LINE__); \
		throw std::runtime_error("no support");                                        \
	} while (0)