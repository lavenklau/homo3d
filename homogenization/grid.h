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
#include "glm/glm.hpp"
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <glm/glm.hpp>

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

	DIRICHLET_BOUNDARY = 0b100000000000,


	SINK_NODES = 0b1000000000000,
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
	__host_device_func bool is_sink(void) {return flagbits & SINK_NODES;}
	__host_device_func void set_sink(void) { flagbits |= SINK_NODES;}
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

	//float* stencil_g[27][9];
	glm::mat3* stencil_g[27];

	float* heatStencil_g[27];
	float *uHeat_g, *fHeat_g, *rHeat_g;
	float *rhoHeat_g;

	float* u_g[3];
	float* f_g[3];
	float* r_g[3];
	//double* uchar_g[6][3];
	//double* fchar_g[6][3];
	// float* uchar_g[3];
	float* uchar_h[6][3];
	float* rho_g;
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
	Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::IncompleteLUT<double>> hostBiCGSolverHeat;
	Eigen::SparseMatrix<double> Khost;
	Eigen::SparseMatrix<double> KHeathost;
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

	bool solveHostEquationHeat(void);

	void setSinkNodes(void);

	void testCoarsestModes(void);

	void assembleHostMatrix(void);

	void assembleHeatHostMatrix(void);

	void gs_relaxation(float w_SOR = 1.f, int times_ = 1);

	void gs_relaxation_heat(float w_SOR = 1.f, int times_ = 1);

	void gs_relaxation_ex(float w_SOR = 1.f);

	void update_residual_ex();

	void gs_relaxation_profile(float w_SOR = 1.f);

	void update_residual_profile(void);

	float diagPrecondition(float strength);

	void prolongate_correction(void);

	void prolongate_correction_heat(void);

	void restrict_residual(void);

	void restrict_residual_heat(void);

	void restrict_stencil(void);

	void restrict_stencil_heat(void);

	void update_residual(void);

	void update_residual_heat(void);

	void enforce_unit_macro_strain(int istrain);

	//void update_uchar(void);

	void setForce(float* f[3]);

	float** getForce(void);

	float** getDisplacement(void);

	float** getResidual(void);

	//double** getFchar(int k);

	//void setFchar(int k, double** f);

	void useFchar(int k);

	void useUchar(int k);

	void setUchar(int k, float** uchar);

	void reset_displacement(void);

	void reset_residual(void);

	void reset_force(void);

	void reset_displacement_heat(void);

	void reset_residual_heat(void);

	void reset_force_heat(void);

	void translateForce(int type_, float* v[3]); // 1. zero dirichlet force; 2. zero global translation

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

	double projectDensityToVolume(float vol, float beta = 20);

	void interpDensityFrom(const std::string& fname, VoxelIOFormat frmat);

	void interpDensityFromSDF(const std::string& fname, VoxelIOFormat frmat);
	//double elasticTensorElement(int i, int j);

	void readDisplacement(const std::string& fname);

	std::string checkDeviceError(void);

	void enforceCellSymmetry(float* celldata, SymmetryType sym, bool average);

	void sensitivity(int i, int j, float* sens);

	void heatForce2matlab(std::string nam);
	void heatResidual2matlab(std::string nam);
	void heatDisplacement2matlab(std::string nam);

	std::vector<VertexFlags> getVertexflags(void);

	std::vector<CellFlags> getCellflags(void);

	void setMacroStrainDisplacement(int i, float* u[3]);

	int elexid2gsid(int lexid);
	void restrict_stencil_arround_dirichelt_boundary(void);


	void v3_reset(float* v[3], int len = -1);
	void v1_reset(float* v, float val = 0, int len = -1);
	void v1_write(std::string vn, float* v, int len = -1);
	void v3_const(float* v[3], const float v_const[3]);
	void v3_rand(float* v[3], float low, float upp, int len = -1);
	void v1_rand(float* v, float low, float upp, int len = -1);
	float v3_norm(float* v[3], bool removePeriodDof = false, int len = -1);
	float v1_norm(float* v, bool removePeriodDof = false, int len = -1);
	float v3_diffnorm(float* v[3], float* u[3], int len = -1);
	void v3_copy(float* dst[3], float* src[3], int len = -1);
	void v3_upload(float* dev[3], float* hst[3]);
	void v3_download(float* hst[3], float* dev[3]);
	void v3_removeT(float* u[3], float tHost[3]);
	void v3_linear(float a1, float* v1[3], float a2, float* v2[3], float* v[3], int len = -1);
	void v3_toMatlab(const std::string& mname, double* v[3], int len = -1);
	void v1_toMatlab(const std::string& mname, float* v, int len = -1);
	void v3_toMatlab(const std::string& mname, float* v[3], int len = -1);
	void v3_write(const std::string& filename, float* v[3], int len = -1);
	void v3_write(const std::string& filename, float* v[3], bool removePeriodDof = false);
	void v3_read(const std::string& filename, float* v[3]);
	void v3_wave(float* u[3], const std::array<float, 3>& radi);
	void v3_create(float* v[3], int len = -1);
	void v3_destroy(float* v[3]);
	float v3_dot(float* v[3], float* u[3], bool removePeriodDof = false, int len = -1);
	float v1_dot(float* v, float* u, bool removePeriodDof = false, int len = -1);
	Eigen::Matrix<float, -1, 1> v3_toMatrix(float* u[3], bool removePeriodDof = false);
	Eigen::Matrix<float, -1, 1> v1_toMatrix(float* u, bool removePeriodDof = false);
	void v3_fromMatrix(float* u[3], const Eigen::Matrix<float, -1, 1>& b, bool hasPeriodDof = false);
	void v1_fromMatrix(float* u, const Eigen::Matrix<float, -1, 1>& b, bool hasPeriodDof = false);
	void v3_stencilOnLeft(float* v[3], float* Kv[3]);
	void v3_average(float* v[3], float vMean[3], bool removePeriodDof = false);

	void array2matlab(const std::string& matname, int* hostdata, int len);
	void array2matlab(const std::string& matname, double* hostdata, int len);
	void array2matlab(const std::string& matname, float* hostdata, int len);

	double relative_residual(void);
	double relative_residual_heat(void);
	double residual(void);
	//double compliance(double* displacement[3]);
	double compliance(float* u[3], float* v[3]);

	// map lexid to gsid
	std::vector<int> getVertexLexidMap(void);
	std::vector<int> getCellLexidMap(void);

	void stencil2matlab(const std::string& name);
	void lexistencil2matlab(const std::string& name);

	Eigen::SparseMatrix<double> stencil2matrix(void);
	Eigen::SparseMatrix<double> heatStencil2matrix(void);

	int vgsid2lexid_h(int gsid, bool removePeriodDof = false);
	void vgsid2lexpos_h(int gsid, int pos[3]);
	void egsid2lexpos_h(int gsid, int pos[3]);
	int vlexpos2vlexid_h(int pos[3], bool removePeriodDof = false);
	int vlexid2gsid(int lexid, bool hasPeriodDof = false);

	//  lexico order with no padding to period padded GS order 
	enum LexiType { VERTEX, CELL };
	void lexi2gsorder(float* src, float* dst, LexiType type_, bool lexipadded = false);
	void lexiStencil2gsorder(void);
	void lexiHeatStencil2gsorder(void);
	void enforce_period_stencil(bool additive);
	void enforce_period_heat_stencil(bool additive);
	//void gather_boundary_force(double* f[3]);
	void enforce_period_boundary(float* v[3], bool additive = false);
	void enforce_period_boundary(float* v, bool additive = false);
	void enforce_dirichlet_boundary(float* v[3]);
	void enforce_dirichlet_stencil(void);
	void enforce_period_vertex(double* v[3], bool additive = false);
	void enforce_period_vertex(float* v, bool additive = false);
	void enforce_period_vertex(float* v[3], bool additive = false);
	void enforce_period_vertex(glm::mat3* v, bool additive = false);
	void pad_vertex_data(double* v[3]);
	void pad_vertex_data(float* v[3]);
	void pad_vertex_data(glm::mat3* v);
	void pad_vertex_data(float* v);
	void pad_cell_data(float* e);
	void enforce_period_element(float* data);

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
