#pragma once

#pragma  warning(disable:4819)

#include "platform_spec.h"
#include <memory>
#include <string>
#include <vector>
#include <any>
#include <array>
#include <map>
#include <numeric>
#include <iostream>
#include <stdint.h>
#include "gmem/DeviceBuffer.h"
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include "glm/glm.hpp"
#include "cuda_fp16.h"

namespace glm {
	using hmat3 = mat<3, 3, half>;
	using hvec3 = vec<3, half>;
};

namespace homo {

#ifdef __CUDACC__
#define __host_device_func __host__ __device__
#else
#define __host_device_func 
#endif

enum  FlagBit : uint16_t {
	FICTION_FLAG = 1,
	GS_ID = 0b1110,
	PERIOD_PADDING = 0b10000,

	LEFT_BOUNDARY = 0b100000,
	DOWN_BOUNDARY = 0b1000000,
	NEAR_BOUNDARY = 0b10000000,

	RIGHT_BOUNDARY = 0b100000000,
	UP_BOUNDARY = 0b1000000000,
	FAR_BOUNDARY = 0b10000000000,

	MIN_BOUNDARY_MASK = 0b11100000,
	MAX_BOUNDARY_MASK = 0b11100000000,
	BOUNDARY_MASK = 0b11111100000,

	DIRICHLET_BOUNDARY = 0b100000000000
};

struct FlagBase {
	uint16_t flagbits;
	__host_device_func bool is_boundary(void) { return flagbits & BOUNDARY_MASK; }
	__host_device_func bool is_set(FlagBit flag) { return flagbits & flag; }
	__host_device_func bool is_min_boundary(void) { return flagbits & MIN_BOUNDARY_MASK; }
	__host_device_func bool is_max_boundary(void) { return flagbits & MAX_BOUNDARY_MASK; }
	__host_device_func void set_boundary(FlagBit boundaryFlag) { flagbits |= boundaryFlag; }

	__host_device_func bool is_fiction(void) { return flagbits & FlagBit::FICTION_FLAG; }

	__host_device_func void set_fiction(void) { flagbits |= FICTION_FLAG; }

	__host_device_func int get_gscolor(void) { return (flagbits & FlagBit::GS_ID) >> 1; }

	__host_device_func void set_gscolor(int color) {
		flagbits &= ~GS_ID;
		flagbits |= color << 1;
	}

	__host_device_func void set_period_padding(void) { flagbits |= PERIOD_PADDING; }

	__host_device_func bool is_period_padding(void) { return flagbits & PERIOD_PADDING; }

	__host_device_func bool is_dirichlet_boundary(void) { return flagbits & DIRICHLET_BOUNDARY; }
	__host_device_func bool set_dirichlet_boundary(void) { flagbits |= DIRICHLET_BOUNDARY; }
};

struct VertexFlags : public FlagBase {
};

struct CellFlags : public FlagBase
{
	
};

enum VoxelIOFormat {
	binary,
	openVDB
};

enum SymmetryType {
	None,
	Simple3
};

struct GridConfig {
	bool enableManagedMem = true;
	std::string namePrefix;
};

struct Grid {
	Grid* fine = nullptr;
	Grid* Coarse = nullptr;

	GridConfig gridConfig;

	bool is_root = false;

	bool assemb_otf = false;

	// coarse from finer grid
	std::array<int, 3> upCoarse = {};
	// coarse to coarser grid
	std::array<int, 3> downCoarse = {};

	std::array<int, 3> totalCoarse = {};

	std::array<int, 3> rootCellReso;
	std::array<int, 3> availCoarseReso;

	std::array<int, 3> cellReso;

	glm::hmat3* stencil_g[27];

	half* u_g[3];
	half* f_g[3];
	half* r_g[3];
	//double* uchar_g[6][3];
	//double* fchar_g[6][3];
	// float* uchar_g[3];
	half* uchar_h[6][3];
	half* rho_g;
	VertexFlags* vertflag;
	CellFlags* cellflag;

	float exp_penal = 1;

	float diag_strength = 0;

	int gsVertexReso[3][8];
	int gsCellReso[3][8];
	int gsVertexSetValid[8];
	int gsVertexSetRound[8];
	// start id of next gs set
	int gsVertexSetEnd[8];
	int gsCellSetValid[8];
	int gsCellSetRound[8];
	// start id of next gs set
	int gsCellSetEnd[8];

	std::map<std::string, std::any> cellTraits;
	std::map<std::string, std::any> vertexTraits;

	Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double>> hostBiCGSolver;
	Eigen::SparseMatrix<double> Khost;
	//Eigen::SparseQR<decltype(Khost),Eigen::AMDOrderin>

	template<typename T>
	void requestCellTraits(std::string traitName) {
		cellTraits[traitName] = getMem().addBuffer<T>(getName() + traitName, n_gscells())->template data<T>();
	}

	template<typename T>
	void requestVertexTraits(std::string traitName) {
		cellTraits[traitName] = getMem().addBuffer<T>(getName() + traitName, n_gsvertices())->template data<T>();
	}

	template<typename T>
	T* getCellTraits(std::string traitName) {
		T* pTraits;
		try {
			pTraits = std::any_cast<T*>(cellTraits[traitName]);
		}
		catch (...) {
			std::cerr << "\033[31mTrait type does not match the name\033[0m" << std::endl;
		}
		return pTraits;
	}

	template<typename T>
	T* getVertexTraits(std::string traitName) {
		T* pTraits;
		try {
			pTraits = std::any_cast<T*>(cellTraits[traitName]);
		}
		catch (...) {
			std::cerr << "\033[31mTrait type does not match the name\033[0m" << std::endl;
		}
		return pTraits;
	}

	int n_gsvertices(void) {
		return std::accumulate(gsVertexSetRound, gsVertexSetRound + 8, 0);
	}

	int n_gscells(void) {
		return std::accumulate(gsCellSetRound, gsCellSetRound + 8, 0);
	}

	int n_cells(void) {
		return std::accumulate(cellReso.begin(), cellReso.end(), 1, std::multiplies<int>());
	}

	void update(float* rho, int pitchT = -1, bool lexiOrder = true);

	void buildRoot(int xreso, int yreso, int zreso, GridConfig config);

	std::array<int, 3> getCellReso(void) { return cellReso; }

	void setFlags_g(void);

	void useGrid_g(void);

	std::string getName(void);

	std::shared_ptr<Grid> coarse2(GridConfig config);

	bool solveHostEquation(void);

	void testCoarsestModes(void);

	void assembleHostMatrix(void);

	void gs_relaxation(float w_SOR = 1.f, int times_ = 1);

	void gs_relaxation_ex(float w_SOR = 1.f);

	void update_residual_ex();

	void gs_relaxation_profile(float w_SOR = 1.f);

	void update_residual_profile(void);

	float diagPrecondition(float strength);

	void prolongate_correction(void);

	void restrict_residual(void);

	void restrict_stencil(void);

	void restrict_stencil_arround_dirichelt_boundary(void);

	void update_residual(void);

	void enforce_unit_macro_strain(int istrain);

	//void update_uchar(void);

	void setForce(half* f[3]);

	half** getForce(void);

	half** getDisplacement(void);

	half** getResidual(void);

	//double** getFchar(int k);

	//void setFchar(int k, double** f);

	void useFchar(int k);

	void useUchar(int k);

	void setUchar(int k, half** uchar);

	void reset_displacement(void);

	void reset_residual(void);

	void reset_force(void);

	void translateForce(int type_, half* v[3]); // 1. zero dirichlet force; 2. zero global translation

	void reset_density(float rho);

	void randDensity(void);

	void getDensity(std::vector<float>& rho, bool lexiOrder = false);

	void getGsVertexPos(std::vector<int> pos[3]);

	void getGsElementPos(std::vector<int> pos[3]);

	void writeGsVertexPos(const std::string& fname);

	void writeDensity(const std::string& fname, VoxelIOFormat frmat);

	void writeStencil(const std::string& fname);

	void readDensity(const std::string& fname, VoxelIOFormat frmat);

	void readDensity(const std::string& fname, std::vector<float>& values, int reso[3], VoxelIOFormat frmat);

	float sumDensity(void);

	void projectDensity(float beta = 20, float eta = 0.5, float a = 1.f, float b = 0.f);

	void interpDensityFrom(const std::string& fname, VoxelIOFormat frmat);

	void interpDensityFromSDF(const std::string& fname, VoxelIOFormat frmat);
	//double elasticTensorElement(int i, int j);

	void readDisplacement(const std::string& fname);

	std::string checkDeviceError(void);

	void enforceCellSymmetry(half* celldata, SymmetryType sym, bool average);
	void enforceCellSymmetry(float* celldata, SymmetryType sym, bool average);

	void sensitivity(int i, int j, float* sens);

	std::vector<VertexFlags> getVertexflags(void);

	std::vector<CellFlags> getCellflags(void);

	void setMacroStrainDisplacement(int i, half* u[3]);

	void v3_reset(half* v[3], int len = -1);
	void v3_const(half* v[3], const half v_const[3]);
	void v3_rand(half* v[3], half low, half upp, int len = -1);
	float v3_norm(half* v[3], bool removePeriodDof = false, int len = -1);
	float v3_diffnorm(half* v[3], half* u[3], int len = -1);
	void v3_copy(half* dst[3], half* src[3], int len = -1);
	void v3_upload(half* dev[3], half* hst[3]);
	void v3_download(half* hst[3], half* dev[3]);
	void v3_removeT(half* u[3], half tHost[3]);
	void v3_linear(half a1, half* v1[3], half a2, half* v2[3], half* v[3], int len = -1);
	void v3_toMatlab(const std::string& mname, double* v[3], int len = -1);
	void v3_toMatlab(const std::string& mname, half* v[3], int len = -1);
	void v3_write(const std::string& filename, half* v[3], int len = -1);
	void v3_write(const std::string& filename, half* v[3], bool removePeriodDof = false);
	void v3_read(const std::string& filename, half* v[3]);
	void v3_wave(half* u[3], const std::array<half, 3>& radi);
	void v3_create(half* v[3], int len = -1);
	void v3_destroy(half* v[3]);
	float v3_dot(half* v[3], half* u[3], bool removePeriodDof = false, int len = -1);
	Eigen::Matrix<float, -1, 1> v3_toMatrix(half* u[3], bool removePeriodDof = false);
	void v3_fromMatrix(half* u[3], const Eigen::Matrix<float, -1, 1>& b, bool hasPeriodDof = false);
	void v3_stencilOnLeft(half* v[3], half* Kv[3]);
	void v3_average(half* v[3], half vMean[3], bool removePeriodDof = false);

	void array2matlab(const std::string& matname, int* hostdata, int len);
	void array2matlab(const std::string& matname, double* hostdata, int len);
	void array2matlab(const std::string& matname, float* hostdata, int len);

	double relative_residual(void);
	double residual(void);
	//double compliance(double* displacement[3]);
	double compliance(half* u[3], half* v[3]);

	// map lexid to gsid
	std::vector<int> getVertexLexidMap(void);
	std::vector<int> getCellLexidMap(void);

	void stencil2matlab(const std::string& name);
	void lexistencil2matlab(const std::string& name);

	Eigen::SparseMatrix<double> stencil2matrix(void);

	int vgsid2lexid_h(int gsid, bool removePeriodDof = false);
	void vgsid2lexpos_h(int gsid, int pos[3]);
	void egsid2lexpos_h(int gsid, int pos[3]);
	int vlexpos2vlexid_h(int pos[3], bool removePeriodDof = false);
	int vlexid2gsid(int lexid, bool hasPeriodDof = false);
	int elexid2gsid(int lexid);

	//  lexico order with no padding to period padded GS order 
	enum LexiType { VERTEX, CELL };
	void lexi2gsorder(float* src, float* dst, LexiType type_, bool lexipadded = false);
	void lexi2gsorder(half* src, half* dst, LexiType type_, bool lexipadded = false);
	void lexi2gsorder(glm::hmat3* src, glm::hmat3* dst, LexiType type_, bool lexipadded = false);
	void lexiStencil2gsorder(void);
	void enforce_period_stencil(bool additive);
	//void gather_boundary_force(double* f[3]);
	void enforce_period_boundary(half* v[3], bool additive = false);
	void enforce_dirichlet_boundary(float* v[3]);
	void enforce_dirichlet_boundary(half* v[3]);
	void enforce_dirichlet_stencil(void);
	void enforce_period_vertex(double* v[3], bool additive = false);
	void enforce_period_vertex(half* v[3], bool additive = false);
	void enforce_period_vertex(glm::hmat3* v, bool additive = false);
	void pad_vertex_data(double* v[3]);
	void pad_vertex_data(float* v[3]);
	void pad_vertex_data(half* v[3]);
	void pad_vertex_data(glm::hmat3* v);
	void pad_cell_data(float* e);
	void pad_cell_data(half* e);
	void enforce_period_element(float* data);
	void enforce_period_element(half* data);

	void test(void);
	void testIndexer(void);
	void testVflags(void);
	void test_gs_relaxation(void);
	//void testgsid2pos(void);
private:
	// return nv, ne
	std::pair<int, int> countGS(void);
	size_t allocateBuffer(int nv, int ne);
};

extern std::string getPath(const std::string& str);
extern std::string setPathPrefix(const std::string& str);

}
