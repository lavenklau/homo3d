#define _USE_MATH_DEFINES
#include "grid.h"
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include "utils.h"
#include "matlab/matlab_utils.h"
#include <exception>
#include <Eigen/Eigen>
#include <Eigen/Eigenvalues>
#include "templateMatrix.h"
#include "voxelIO/openvdb_wrapper_t.h"
#include <map>
#include "cuda_profiler_api.h"
#include "tictoc.h"

using namespace homo;

void Grid::buildRoot(int xreso, int yreso, int zreso, GridConfig config)
{
	gridConfig = config;

	if (xreso > 1024 || yreso > 1024 || zreso > 1024) {
		throw std::runtime_error("axis resolution cannot exceed 1024");
	}

	double xlog2 = log2(xreso);
	double ylog2 = log2(yreso);
	double zlog2 = log2(zreso);

	int baseCoarsestReso = std::pow(2, 2);

	int largestCoarseLevel[3] = {
		(std::max)(std::floor(xlog2) - 2, 0.),
		(std::max)(std::floor(ylog2) - 2, 0.),
		(std::max)(std::floor(zlog2) - 2, 0.),
	};

	std::cout << "Largest coarse level " << largestCoarseLevel[0]
		<< ", " << largestCoarseLevel[1] 
		<< ", " << largestCoarseLevel[2] << std::endl;

	double xc = xlog2 - largestCoarseLevel[0];
	double yc = ylog2 - largestCoarseLevel[1];
	double zc = zlog2 - largestCoarseLevel[2];

	printf("(xc, yc, zc) = (%lf, %lf, %lf)\n", xc, yc, zc);

	int xcreso = std::ceil(pow(2, xc));
	int ycreso = std::ceil(pow(2, yc));
	int zcreso = std::ceil(pow(2, zc));

	availCoarseReso[0] = xcreso;
	availCoarseReso[1] = ycreso;
	availCoarseReso[2] = zcreso;

	// assemble stencil on the fly
	assemb_otf = true;

	// corrected resolution
	xreso = xcreso * pow(2, largestCoarseLevel[0]);
	yreso = ycreso * pow(2, largestCoarseLevel[1]);
	zreso = zcreso * pow(2, largestCoarseLevel[2]);

	rootCellReso[0] = cellReso[0] = xreso;
	rootCellReso[1] = cellReso[1] = yreso;
	rootCellReso[2] = cellReso[2] = zreso;

	is_root = true;

	for (int i = 0; i < 3; i++) {
		upCoarse[i] = 0;
		totalCoarse[i] = 0;
	}

	// compute eight Colored node number
	auto nv_ne = countGS();
	
	// allocate buffer
	size_t totalMem = allocateBuffer(nv_ne.first, nv_ne.second);

	setFlags_g();
}

std::string Grid::getName(void)
{
	char buf[1000];
	sprintf_s(buf, "<%s_Grid_%d_%d_%d>", gridConfig.namePrefix.c_str(), cellReso[0], cellReso[1], cellReso[2]);
	return buf;
}

std::shared_ptr<Grid> Grid::coarse2(GridConfig config)
{
	std::shared_ptr<Grid> coarseGrid(new Grid());
	// ...
	coarseGrid->gridConfig = config;
	coarseGrid->fine = this;
	Coarse = coarseGrid.get();
	coarseGrid->is_root = false;
	coarseGrid->assemb_otf = false;
	// ...
	coarseGrid->availCoarseReso = availCoarseReso;
	coarseGrid->rootCellReso = rootCellReso;
	// determine coarse ratio
	bool has_coarse = false;
	for (int i = 0; i < 3; i++) {
		if (cellReso[i] <= availCoarseReso[i]) {
			coarseGrid->upCoarse[i] = 0;
			downCoarse[i] = 0;
		}
		else if (cellReso[i] > config.max_coarse_reso) {
			has_coarse = true;
			//coarseGrid->upCoarse[i] = 2;
			coarseGrid->upCoarse[i] = 2;
			while (cellReso[i] / coarseGrid->upCoarse[i] > config.max_coarse_reso) {
				coarseGrid->upCoarse[i] *= 2;
			}
			downCoarse[i] = coarseGrid->upCoarse[i];
		}
		// DEBUG
		//else if (is_root) {
		//	has_coarse = true;
		//	coarseGrid->upCoarse[i] = 4;
		//	downCoarse[i] = coarseGrid->upCoarse[i];
		//}
		else {
			has_coarse = true;
			coarseGrid->upCoarse[i] = 2;
			downCoarse[i] = 2;
		}
	}
	if (!has_coarse) {
		Coarse = nullptr;
		return {};
	}

	// coarse grid
	for (int i = 0; i < 3; i++) {
		coarseGrid->cellReso[i] = cellReso[i] / downCoarse[i];
	}
	// determine eight colored GS nodes number
	auto nv_ne = coarseGrid->countGS();
	// allocate buffer
	size_t totalMem = coarseGrid->allocateBuffer(nv_ne.first, nv_ne.second);

	coarseGrid->setFlags_g();

	return coarseGrid;
}

// padding left and right one element for data alignment 
// ** depends on cellReso[3]
std::pair<int, int> Grid::countGS(void)
{
	printf("%s Enumerating GS...\n", getName().c_str());
	printf("cell = [%d, %d, %d]\n", cellReso[0], cellReso[1], cellReso[2]);
	int n_gsvertex[8] = {};
	for (int i = 0; i < 8; i++) {
		int org[3] = { i % 2, i / 2 % 2, i / 4 };
		for (int k = 0; k < 3; k++) {
			gsVertexReso[k][i] = (cellReso[k] - org[k] + 2) / 2 + 1;
		}
		n_gsvertex[i] = gsVertexReso[0][i] * gsVertexReso[1][i] * gsVertexReso[2][i];
		gsVertexSetValid[i] = n_gsvertex[i];
		// ceil to multiple of 32
		n_gsvertex[i] = 32 * (n_gsvertex[i] / 32 + bool(n_gsvertex[i] % 32));
		printf("gv[%d] = %d (%d)\n", i, gsVertexSetValid[i], n_gsvertex[i]);
		gsVertexSetRound[i] = n_gsvertex[i];
		int endid = 0;
		for (int j = 0; j < i + 1; j++) {
			endid += n_gsvertex[j];
		}
		gsVertexSetEnd[i] = endid;
	}
	int nv = std::accumulate(n_gsvertex, n_gsvertex + 8, 0);
	printf("Total rounded vertex %d\n", nv);

	int n_gscell[8] = {};
	for (int i = 0; i < 8; i++) {
		int org[3] = { i % 2, i / 2 % 2, i / 4 };
		for (int k = 0; k < 3; k++) {
			gsCellReso[k][i] = ((cellReso[k] + 1 - org[k]) / 2 + 1);
		}
		n_gscell[i] = gsCellReso[0][i] * gsCellReso[1][i] * gsCellReso[2][i];
		gsCellSetValid[i] = n_gscell[i];
		n_gscell[i] = 32 * (n_gscell[i] / 32 + bool(n_gscell[i] % 32));
		printf("ge[%d] = %d (%d)\n", i, gsCellSetValid[i], n_gscell[i]);
		gsCellSetRound[i] = n_gscell[i];
		int endid = 0;
		for (int j = 0; j < i + 1; j++) endid += n_gscell[j];
		gsCellSetEnd[i] = endid;
	}
	int ne = std::accumulate(n_gscell, n_gscell + 8, 0);
	printf("Total rounded cell %d\n", ne);
	return { nv,ne };
}

size_t Grid::allocateBuffer(int nv, int ne)
{
	size_t total = 0;
	// allocate FEM vectors
	for (int i = 0; i < 3; i++) {
		u_g[i] = getMem().addBuffer(homoutils::formated("%s_u_%d", getName().c_str(), i), nv * sizeof(float) * 2)->data<float>();
		f_g[i] = getMem().addBuffer(homoutils::formated("%s_f_%d", getName().c_str(), i), nv * sizeof(float) * 2)->data<float>();
		r_g[i] = getMem().addBuffer(homoutils::formated("%s_r_%d", getName().c_str(), i), nv * sizeof(float) * 2)->data<float>();
	}
	total += nv * 9 * sizeof(float);
	// allocate stencil buffer
	if (!is_root) {
		for (int i = 0; i < 27; i++) {
			//for (int j = 0; j < 9; j++) {
			//	stencil_g[i][j] = getMem().addBuffer(homoutils::formated("%s_st_%d_%d", getName().c_str(), i, j), nv * sizeof(float))->data<float>();
			//}
			stencil_g[i] = getMem().addBuffer(homoutils::formated("%s_st_%d", getName().c_str(), i), nv * sizeof(glm::mat3))->data<glm::mat3>();
		}
		total += nv * sizeof(float) * 27 * 9;
	}
	// allocate characteristic buffer
	/*for (int i = 0; i < 6; i++) */{
		for (int j = 0; j < 3; j++) {
			//fchar_g[i][j] = getMem().addBuffer(homoutils::formated("%s_fc_%d_%d", getName().c_str(), i, j), nv * sizeof(double))->data<double>();
			//uchar_g[i][j] = getMem().addBuffer(homoutils::formated("%s_uc_%d_%d", getName().c_str(), i, j), nv * sizeof(double))->data<double>();
			uchar_g[j] = getMem().addBuffer(homoutils::formated("%s_uc_%d", getName().c_str(), j), nv * sizeof(float))->data<float>();
		}
	}

	// if enable use of managed memory, then allocate managed memory (better performance);
	// otherwise, use host memory 
	if (gridConfig.enableManagedMem) {
		for (int i = 0; i < 6; i++) {
			for (int j = 0; j < 3; j++) {
				uchar_h[i][j] = getMem().addBuffer(homoutils::formated("%s_uchost_%d_%d", getName().c_str(), i, j), nv * sizeof(float), Managed)->data<float>();
			}
			v3_reset(uchar_h[i], nv);
		}
	} else {
		for (int i = 0; i < 6; i++) {
			for (int j = 0; j < 3; j++) {
				uchar_h[i][j] = getMem().addBuffer(homoutils::formated("%s_uchost_%d_%d", getName().c_str(), i, j), nv * sizeof(float), Hostheap)->data<float>();
				memset(uchar_h[i][j], 0, sizeof(float) * nv);
			}
		}
	}

	total += nv * sizeof(float) * 3;
	// allocate flag buffer
	vertflag = getMem().addBuffer<VertexFlags>(homoutils::formated("%s_vflag", getName().c_str()), nv)->data<VertexFlags>();
	cellflag = getMem().addBuffer<CellFlags>(homoutils::formated("%s_cflag", getName().c_str()), ne)->data<CellFlags>();
	total += nv * sizeof(VertexFlags);
	total += ne * sizeof(CellFlags);
	// allocate element buffer
	rho_g = getMem().addBuffer(homoutils::formated("%s_rho", getName().c_str()), ne * sizeof(float))->data<float>();
	total += ne * sizeof(float);

	printf("%s allocated %zd MB GPU memory\n", getName().c_str(), total / 1024 / 1024);

	return total;
}

//void Grid::update_uchar(void)
//{
//	enforce_unit_macro_strain();
//
//	for (int i = 0; i < 6; i++) {
//		double* f[3]{ fchar_g[i][0], fchar_g[i][1], fchar_g[i][2] };
//		setForce(f);
//		
//	}
//}

void Grid::setForce(float* f[3])
{
	v3_copy(f_g, f);
}

float** Grid::getForce(void)
{
	return f_g;
}

float** Grid::getDisplacement(void)
{
	return u_g;
}

//double** Grid::getFchar(int k)
//{
//	return fchar_g[k];
//}

//void Grid::setFchar(int k, double** f)
//{
//	v3_copy(fchar_g[k], f);
//}

double Grid::relative_residual(void)
{
	return v3_norm(r_g) / (v3_norm(f_g) + 1e-30);
}

double homo::Grid::residual(void)
{
	return v3_norm(r_g);
}

void Grid::useFchar(int k)
{
	useGrid_g();
	enforce_unit_macro_strain(k);
	//setForce(fchar_g[k]);
	//enforce_dirichlet_boundary(f_g);
	enforce_period_vertex(f_g, true);
	pad_vertex_data(f_g);
	// DEBUG
	if (0) {
		char buf[100];
		sprintf_s(buf, "./fchar%d", k);
		v3_write(buf, f_g, true);
	}
}

void Grid::reset_displacement(void)
{
	v3_reset(u_g);
}

void Grid::reset_residual(void)
{
	v3_reset(r_g);
}

void Grid::reset_force(void)
{
	v3_reset(f_g);
}

void Grid::setUchar(int k, float** uchar)
{
	//v3_copy(uchar_g[k], uchar);
	v3_download(uchar_h[k], uchar);
}

static Eigen::Matrix<double, -1, -1> transBase;

bool homo::Grid::solveHostEquation(void)
{
	Eigen::VectorXd b = v3_toMatrix(f_g, true).cast<double>();
#if 1
	// remove translation
	b = b - transBase * (transBase.transpose() * b);
#endif
	eigen2ConnectedMatlab("b", b);
	//Eigen::Matrix<double, 3, 1> bmean(0, 0, 0);
	//Eigen::Matrix<double, 3, 1> bmean = b.reshaped(3, b.rows() / 3).colwise().sum();

	Eigen::Matrix<double, -1, 1> x = hostBiCGSolver.solve(b);
	if (hostBiCGSolver.info() != Eigen::Success) {
		hostBiCGSolver.compute(Khost);
		hostBiCGSolver.solve(b);
		printf("\033[31mhost equation failed to solve, err = %d\033[0m\n", int(hostBiCGSolver.info()));
		eigen2ConnectedMatlab("T", transBase);
		eigen2ConnectedMatlab("Khost", Khost);
		eigen2ConnectedMatlab("x", x);
		return false;
	}

	//Eigen::Matrix<double, 3, 1> xmean = x.reshaped(3, b.rows() / 3).colwise().sum() / (b.rows() / 3);
	x = x - transBase * (transBase.transpose() * x);
	eigen2ConnectedMatlab("x", x);

	v3_fromMatrix(u_g, x.cast<float>(), false);

	return true;
}

void homo::Grid::testCoarsestModes(void)
{
	eigen2ConnectedMatlab("Khost", Khost);
	//Eigen::EigenSolver<Eigen::SparseMatrix<double>> eigsol;
	//eigsol.compute(Khost);
	//auto vidmap = getVertexLexidMap();
	//array2matlab("vidmap", vidmap.data(), vidmap.size());
}

void homo::Grid::enforce_dirichlet_stencil(void) {
	for (int cc = 0; cc < 8; cc++) {
		int d_pos[3] = { (cc % 2) * cellReso[0], (cc / 2 % 2) * cellReso[1], (cc / 4) * cellReso[2] };
		int d_lexid = vlexpos2vlexid_h(d_pos);
		int d_gsid = vlexid2gsid(d_lexid, true);
		for (int offid = 0; offid < 27; offid++) {
			int off[3] = { offid % 3 - 1 , offid / 3 % 3 - 1 , offid / 9 - 1 };
			int nei_pos[3] = { d_pos[0] + off[0], d_pos[1] + off[1], d_pos[2] + off[2] };
			for (int kk = 0; kk < 3; kk++) {
				if (nei_pos[kk]<0 || nei_pos[kk]>cellReso[kk]) {
					nei_pos[kk] = (nei_pos[kk] + cellReso[kk]) % cellReso[kk];
				}
			}
			int nei_lexid = vlexpos2vlexid_h(nei_pos);
			int nei_gsid = vlexid2gsid(nei_lexid, true);
			if (offid != 13) {
				//for (int i = 0; i < 9; i++) {
				//	cudaMemset(stencil_g[offid][i] + d_gsid, 0, sizeof(stencil_g[0][0][0]));
				//	cudaMemset(stencil_g[26 - offid][i] + nei_gsid, 0, sizeof(stencil_g[0][0][0]));
				//}
				cudaMemset(stencil_g[offid] + d_gsid, 0, sizeof(glm::mat3));
				cudaMemset(stencil_g[26 - offid] + nei_gsid, 0, sizeof(glm::mat3));
			}
			else {
				//for (int i = 0; i < 9; i++) {
				//	cudaMemset(stencil_g[offid][i] + d_gsid, 0, sizeof(stencil_g[0][0][0]));
				//}
				//for (int row = 0; row < 3; row++) {
				//	double d = 1;
				//	cudaMemcpy(stencil_g[13][row * 3 + row] + d_gsid, &d, sizeof(double), cudaMemcpyHostToDevice);
				//}
				glm::mat3 id(1.);
				cudaMemcpy(stencil_g[13] + d_gsid, &id, sizeof(id), cudaMemcpyHostToDevice);
			}
		}
	}
	//enforce_period_stencil(false);
}

void homo::Grid::assembleHostMatrix(void)
{
	Khost = stencil2matrix();
	eigen2ConnectedMatlab("Khost", Khost);
	hostBiCGSolver.compute(Khost);
	hostBiCGSolver.setTolerance(1e-10);
	// init translation base
	transBase.resize(Khost.rows(), 6);
	transBase.fill(0);
#if 0
	// add trans
	for (int i = 1; i < Khost.rows() / 3; i++) {
		transBase.block<3, 3>(i * 3, 0).setIdentity();
	}
	// add rot
	for (int i = 1; i < Khost.rows() / 3; i++) {
		int r[3] = { i % cellReso[0], i / cellReso[0] % cellReso[1], i / (cellReso[0] * cellReso[1]) };
		Eigen::Matrix<double, 3, 3> Rcross;
		Rcross <<
			0, -r[2], r[1],
			r[2], 0, -r[0],
			-r[1], r[0], 0;
		transBase.block<3, 3>(i * 3, 3) = Rcross;
	}
	
#else
	Eigen::Matrix<double, -1, -1> fk(Khost);
	Eigen::FullPivLU<Eigen::Matrix<double, -1, -1>> dec;
	dec.setThreshold(5e-2);
	dec.compute(fk);
	transBase = dec.kernel();
#endif
	for (int i = 0; i < transBase.cols(); i++) {
		for (int j = 0; j < i; j++) {
			transBase.col(i) -= transBase.col(i).dot(transBase.col(j)) * transBase.col(j);
		}
		transBase.col(i).normalize();
	}
	printf("Coarse system degenerate rank = %d\n", int(transBase.cols()));
	eigen2ConnectedMatlab("transbase", transBase);

	// DEBUG
	if (1) {
		Eigen::Matrix<double, -1, 1> bhost(Khost.rows(), 1);
		bhost.setRandom();
		bhost = bhost - transBase * (transBase.transpose() * bhost);
		Eigen::VectorXd x = hostBiCGSolver.solve(bhost);
		if (hostBiCGSolver.info() != Eigen::Success) {
			printf("\033[31mSolver test failed, err = %d\033[0m\n", int(hostBiCGSolver.info()));
		}
	}
}

void homo::Grid::gs_relaxation_profile(float w_SOR /*= 1.f*/)
{
	//cudaProfilerStart();
	_TIC("relx")
	gs_relaxation(w_SOR);
	_TOC;
	printf("relaxation time  =   %4.2f ms\n", tictoc::get_record("relx"));
	//cudaProfilerStop();
	cudaDeviceSynchronize();
}

void homo::Grid::update_residual_profile(void)
{
	cudaProfilerStart();
	_TIC("updater")
	update_residual();
	_TOC;
	printf("update res time  =   %4.2f ms\n", tictoc::get_record("updater"));
	cudaProfilerStop();
	cudaDeviceSynchronize();
}

float homo::Grid::diagPrecondition(float strength)
{
	diag_strength = strength;
	return strength;
}

float** homo::Grid::getResidual(void)
{
	return r_g;
}

void homo::Grid::useUchar(int k)
{
	v3_upload(u_g, uchar_h[k]);
}

void homo::Grid::writeGsVertexPos(const std::string& fname)
{
	std::vector<int> pos[3];
	getGsVertexPos(pos);
	homoutils::writeVectors(fname, pos);
}

void homo::Grid::writeDensity(const std::string& fname, VoxelIOFormat frmat)
{
	std::vector<int> pos[3];
	getGsElementPos(pos);
	std::vector<float> rho;
	getDensity(rho);
	if (frmat == homo::binary) {
		std::ofstream ofs(fname, std::ios::binary);
		auto eidmap = getCellLexidMap();
		ofs.write((char*)cellReso.data(), sizeof(cellReso));
		for (int i = 0; i < eidmap.size(); i++) {
			float erho = rho[eidmap[i]];
			ofs.write((char*)&erho, sizeof(erho));
		}
		ofs.close();
	}
	else if (frmat == homo::openVDB) {
		std::vector<float> validrho;
		std::vector<int> validpos[3];
		for (int i = 0; i < rho.size(); i++) {
			if (pos[0][i] < 0 || pos[1][i] < 0 || pos[2][i] < 0 ||
				pos[0][i] >= cellReso[0] || pos[1][i] >= cellReso[1] || pos[2][i] >= cellReso[2])
				continue;
			validrho.emplace_back(rho[i]);
			validpos[0].emplace_back(pos[0][i]);
			validpos[1].emplace_back(pos[1][i]);
			validpos[2].emplace_back(pos[2][i]);
		}
		openvdb_wrapper_t<float>::grid2openVDBfile(fname, validpos, validrho);
	}
}

void homo::Grid::writeStencil(const std::string& fname)
{
	std::vector<float> stencil[27 * 9];
	for (int i = 0; i < 27 * 9; i++) {
		stencil[i].resize(n_gsvertices());
	}
	//for (int i = 0; i < 27 * 9; i++) {
	//	cudaMemcpy(stencil[i].data(), stencil_g[i / 9][i % 9], sizeof(float) * n_gsvertices(), cudaMemcpyDeviceToHost);
	//}
	for (int i = 0; i < 27; i++) {
		std::vector<glm::mat3> stmat(n_gsvertices());
		cudaMemcpy(stmat.data(), stencil_g[i], sizeof(glm::mat3) * n_gsvertices(), cudaMemcpyDeviceToHost);
		for (int j = 0; j < 9; j++) {
			for (int k = 0; k < n_gsvertices(); k++) {
				stencil[i * 9 + j][k] = stmat[k][j % 3][j / 3];
			}
		}
	}
	homoutils::writeVectors(fname, stencil);
}

void homo::Grid::readDensity(const std::string& fname, VoxelIOFormat frmat)
{
	std::vector<int> pos[3];
	getGsElementPos(pos);
	std::vector<float> rho;
	auto eidmap = getCellLexidMap();
	//getDensity(rho);
	if (frmat == homo::binary) {
		std::ifstream ifs(fname, std::ios::binary);
		ifs.seekg(0, ifs.end);
		size_t filesize = ifs.tellg();
		ifs.seekg(0, ifs.beg);
		int reso[3];
		ifs.read((char*)reso, sizeof(reso));
		if (reso[0] == cellReso[0] && reso[1] == cellReso[1] && reso[2] == cellReso[2]) {
			int ne = reso[0] * reso[1] * reso[2];
			rho.resize(n_gscells(), 0);
			std::vector<float> rholex(ne);
			ifs.read((char*)rholex.data(), ne * sizeof(float));
			if (ifs.fail()) { printf("\033[31mvdb file is not complete\033[0m\n"); }
			ifs.close();
			for (int i = 0; i < ne; i++) {
				rho[eidmap[i]] = rholex[i];
			}
			cudaMemcpy(rho_g, rho.data(), sizeof(float) * rho.size(), cudaMemcpyHostToDevice);
			pad_cell_data(rho_g);
		} else {
			printf("\033[31mGrid reso does not match\033[0m\n");
		}
	}
	else if (frmat == homo::openVDB) {
		rho.resize(n_gscells());
		std::vector<int> fpos[3];
		std::vector<float> fvalue;
		openvdb_wrapper_t<float>::openVDBfile2grid(fname, fpos, fvalue);
		for (int i = 0; i < fvalue.size(); i++) {
			if (fpos[0][i] >= 0 && fpos[0][i] < cellReso[0] &&
				fpos[1][i] >= 0 && fpos[1][i] < cellReso[1] &&
				fpos[2][i] >= 0 && fpos[2][i] < cellReso[2]) {
				int lexid = fpos[0][i] + fpos[1][i] * cellReso[0] + fpos[2][i] * cellReso[0] * cellReso[1];
				rho[eidmap[lexid]] = fvalue[i];
			}
		}
		cudaMemcpy(rho_g, rho.data(), sizeof(float) * rho.size(), cudaMemcpyHostToDevice);
		pad_cell_data(rho_g);
	}
}

void homo::Grid::readDensity(const std::string& fname, std::vector<float>& values, int reso[3], VoxelIOFormat frmat)
{
	printf("reading density from file %s...", fname.c_str());
	std::vector<float> rho;
	if (frmat == homo::binary) {
		std::ifstream ifs(fname, std::ios::binary);
		ifs.seekg(0, ifs.end);
		size_t filesize = ifs.tellg();
		ifs.seekg(0, ifs.beg);
		ifs.read((char*)reso, sizeof(reso));
		int ne = reso[0] * reso[1] * reso[2];
		rho.resize(ne, 0);
		ifs.read((char*)rho.data(), ne * sizeof(float));
		if (ifs.fail()) {
			printf("\033[31mvdb file is not complete\033[0m\n"); 
			ifs.close();
			throw std::runtime_error("file exception");
		}
		ifs.close();
	}
	else if (frmat == homo::openVDB) {
		std::vector<int> fpos[3];
		std::vector<float> fvalue;
		openvdb_wrapper_t<float>::openVDBfile2grid(fname, fpos, fvalue);
		// checkout resolution
		for (int i = 0; i < 3; i++) { reso[i] = 1 + *std::max_element(fpos[i].begin(), fpos[i].end()); }
		// sort in lexicorder
		rho.resize(reso[0] * reso[1] * reso[2], 0);
		for (int i = 0; i < fvalue.size(); i++) {
			if (fpos[0][i] >= 0 && fpos[1][i] >= 0 && fpos[2][i] >= 0) {
				int lexid = fpos[0][i] + fpos[1][i] * reso[0] + fpos[2][i] * reso[0] * reso[1];
				rho[lexid] = fvalue[i];
			}
		}
	}
	printf(" reso = (%d, %d, %d)\n", reso[0], reso[1], reso[2]);
	values = std::move(rho);
}

void homo::Grid::interpDensityFromSDF(const std::string& fname, VoxelIOFormat frmat)
{
	interpDensityFrom(fname, frmat);
	projectDensity(30.f, 0.f, -1.f);
}

void homo::Grid::readDisplacement(const std::string& fname)
{
	v3_read(fname, u_g);
}

void homo::Grid::v3_upload(float* dev[3], float* hst[3])
{
	for (int i = 0; i < 3; i++) {
		cudaMemcpy(dev[i], hst[i], sizeof(float) * n_gsvertices(), cudaMemcpyHostToDevice);
	}
}

void homo::Grid::v3_download(float* hst[3], float* dev[3])
{
	for (int i = 0; i < 3; i++) {
		cudaMemcpy(hst[i], dev[i], sizeof(float) * n_gsvertices(), cudaMemcpyDeviceToHost);
	}
}

void homo::Grid::v3_toMatlab(const std::string& mname, double* v[3], int len /*= -1*/, bool removePeriodDof /*= false*/)
{
#ifdef ENABLE_MATLAB
	if (len == -1) len = n_gsvertices();
	Eigen::Matrix<double, -1, 3> vmat(len, 3);
	for (int i = 0; i < 3; i++) {
		cudaMemcpy(vmat.col(i).data(), v[i], sizeof(double) * len, cudaMemcpyDeviceToHost);
	}
	if (removePeriodDof) {
		int nv = cellReso[0] * cellReso[1] * cellReso[2];
		Eigen::Matrix<double, -1, -1> umat(3, nv);
		for (int i = 0; i < nv; i++) {
			int pos[3] = { i % cellReso[0], i / cellReso[0] % cellReso[1], i / (cellReso[0] * cellReso[1]) };
			int gsid = vlexid2gsid(i);
			umat.col(i) = vmat.row(gsid).transpose();
		}
		eigen2ConnectedMatlab(mname, umat);
		return;
	}
	eigen2ConnectedMatlab(mname, vmat);
#endif
}

void homo::Grid::v3_toMatlab(const std::string& mname, float* v[3], int len /*= -1*/)
{
	if (len == -1) len = n_gsvertices();
	Eigen::Matrix<float, -1, 3> vmat(len, 3);
	for (int i = 0; i < 3; i++) {
		cudaMemcpy(vmat.col(i).data(), v[i], sizeof(float) * len, cudaMemcpyDeviceToHost);
	}
	eigen2ConnectedMatlab(mname, vmat);

}

void homo::Grid::v3_write(const std::string& filename, float* v[3], int len /*= -1*/)
{
	if (len == -1) len = n_gsvertices();
	std::vector<float> arr[3];
	for (int i = 0; i < 3; i++) {
		arr[i].resize(len);
		cudaMemcpy(arr[i].data(), v[i], sizeof(float) * len, cudaMemcpyDeviceToHost);
	}
	homoutils::writeVectors(filename, arr);
}

void homo::Grid::v3_write(const std::string& filename, float* v[3], bool removePeriodDof /*= false*/) {
	auto b = v3_toMatrix(v, removePeriodDof);
	std::ofstream ofs(filename, std::ios::binary);
	ofs.write((const char*)b.data(), b.size() * sizeof(float));
	ofs.close();
	return;
}

void homo::Grid::v3_read(const std::string& filename, float* v[3])
{
	std::ifstream ifs(filename, std::ios::binary | std::ios::ate);
	if (!ifs) {
		printf("\033[31mCannot open file %s\033[0m\n", filename.c_str());
		throw std::runtime_error("cannot open file");
	}
	size_t fsize = ifs.tellg();
	if (fsize / 3 / 8 != n_gsvertices()) {
		printf("\033[31mUnmatched displacement file\033[0m\n");
		ifs.close();
		exit(-1);
	}
	ifs.seekg(0);
	std::vector<float> u[3];
	for (int i = 0; i < n_gsvertices(); i++) {
		double u_vert[3];
		ifs.read((char*)u_vert, sizeof(u_vert));
		u[0].emplace_back(u_vert[0]); u[1].emplace_back(u_vert[1]); u[2].emplace_back(u_vert[2]);
	}
	ifs.close();
	for (int i = 0; i < 3; i++) {
		cudaMemcpy(v[i], u[i].data(), sizeof(float) * n_gsvertices(), cudaMemcpyHostToDevice);
	}

}

Eigen::Matrix<float, -1, 1> homo::Grid::v3_toMatrix(float* u[3], bool removePeriodDof /*= false*/)
{
	int nv;
	if (removePeriodDof) {
		nv = cellReso[0] * cellReso[1] * cellReso[2];
	} else {
		nv = (cellReso[0] + 1) * (cellReso[1] + 1) * (cellReso[2] + 1);
	}
	Eigen::Matrix<float, -1, 1> b(nv * 3, 1);
	b.fill(0);
	std::vector<float> vhost(n_gsvertices());
	std::vector<VertexFlags> vflags(n_gsvertices());
	cudaMemcpy(vflags.data(), vertflag, sizeof(VertexFlags) * n_gsvertices(), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 3; i++) {
		cudaMemcpy(vhost.data(), u[i], sizeof(float) * n_gsvertices(), cudaMemcpyDeviceToHost);
		for (int k = 0; k < n_gsvertices(); k++) {
			if (vflags[k].is_fiction() || vflags[k].is_period_padding()) continue;
			//int pos[3];
			int id = vgsid2lexid_h(k, removePeriodDof);
			b[id * 3 + i] = vhost[k];
		}
	}
	return b;
}

void homo::Grid::v3_fromMatrix(float* u[3], const Eigen::Matrix<float, -1, 1>& b, bool hasPeriodDof /*= false*/)
{
	for (int i = 0; i < 3; i++) {
		std::vector<float> ui(n_gsvertices());
		std::fill(ui.begin(), ui.end(), 0.);
		int nvlex;
		if (hasPeriodDof) {
			nvlex = (cellReso[0] + 1) * (cellReso[1] + 1) * (cellReso[2] + 1);
		} else {
			nvlex = cellReso[0] * cellReso[1] * cellReso[2];
		}
		for (int k = 0; k < nvlex; k++) {
			int gsid = vlexid2gsid(k, hasPeriodDof);
			ui[gsid] = b[k * 3 + i];
		}
		cudaMemcpy(u[i], ui.data(), sizeof(float) * n_gsvertices(), cudaMemcpyHostToDevice);
	}
	enforce_period_vertex(u, false);
	pad_vertex_data(u);
}

void homo::Grid::array2matlab(const std::string& matname, int* hostdata, int len)
{
#ifdef ENABLE_MATLAB
	Eigen::Matrix<int, -1, 1> hostvec(len, 1);
	memcpy(hostvec.data(), hostdata, sizeof(int) * len);
	eigen2ConnectedMatlab(matname, hostvec);
#endif
}

void homo::Grid::array2matlab(const std::string& matname, double* hostdata, int len)
{
#ifdef ENABLE_MATLAB
	Eigen::Matrix<double, -1, 1> hostvec(len, 1);
	memcpy(hostvec.data(), hostdata, sizeof(double) * len);
	eigen2ConnectedMatlab(matname, hostvec);
#endif
}

void homo::Grid::array2matlab(const std::string& matname, float* hostdata, int len) {
#ifdef ENABLE_MATLAB
	Eigen::Matrix<float, -1, 1> hostvec(len, 1);
	memcpy(hostvec.data(), hostdata, sizeof(float) * len);
	eigen2ConnectedMatlab(matname, hostvec);
#endif
}

void homo::Grid::stencil2matlab(const std::string& name, bool removePeriodDof /*= true*/)
{
#ifdef ENABLE_MATLAB
	auto k = stencil2matrix(removePeriodDof);
	eigen2ConnectedMatlab(name, k);
#endif
}

void homo::Grid::restrictMatrix2matlab(const std::string& name, Grid& coarseGrid)
{
	std::vector<Eigen::Triplet<double>> triplist;
	auto vflags = coarseGrid.getVertexflags();
	for (int k = 0; k < coarseGrid.n_gsvertices(); k++) {
		if (vflags[k].is_fiction() || vflags[k].is_period_padding() /*|| vflags[k].is_max_boundary()*/) continue;
		int vidCoarse = k;
		int vCoarsePos[3];
		coarseGrid.vgsid2lexpos_h(k, vCoarsePos);
		if (vCoarsePos[0] >= coarseGrid.cellReso[0] ||
			vCoarsePos[1] >= coarseGrid.cellReso[1] || vCoarsePos[2] >= coarseGrid.cellReso[2]) {
			continue;
		}
		for (int kk = 0; kk < 3; kk++) vCoarsePos[kk] = (vCoarsePos[kk] + coarseGrid.cellReso[kk]) % coarseGrid.cellReso[kk];
		int vid = coarseGrid.vlexpos2vlexid_h(vCoarsePos, true);
		int vpos[3] = { vCoarsePos[0] * 2, vCoarsePos[1] * 2, vCoarsePos[2] * 2 };
		for (int i = 0; i < 27; i++) {
			int neioffset[3] = { i % 3 - 1, i / 3 % 3 - 1, i / 9 - 1 };
			double w = (2. - abs(neioffset[0])) * (2. - abs(neioffset[1])) * (2. - abs(neioffset[2])) / 8;
			if (w < 0) printf("negative w = %lf\n", w);
			int vneipos[3] = { neioffset[0] + vpos[0], neioffset[1] + vpos[1], neioffset[2] + vpos[2] };
			for (int kk = 0; kk < 3; kk++) vneipos[kk] = (vneipos[kk] + cellReso[kk]) % cellReso[kk];
			int vjd = vlexpos2vlexid_h(vneipos, true);
			for (int row = 0; row < 3; row++) {
				triplist.emplace_back(vid * 3 + row, vjd * 3 + row, w);
			}
		}
	}
	int nvfine = (cellReso[0]) * (cellReso[1]) * (cellReso[2]);
	int nvcoarse = (coarseGrid.cellReso[0]) * (coarseGrid.cellReso[1]) * (coarseGrid.cellReso[2]);

	Eigen::SparseMatrix<double> R(nvcoarse * 3, nvfine * 3);
	R.setFromTriplets(triplist.begin(), triplist.end());
	eigen2ConnectedMatlab(name, R);
}

void homo::Grid::lexistencil2matlab(const std::string& name)
{
	int n_lexiv = (cellReso[0] + 1) * (cellReso[1] + 1) * (cellReso[2] + 1);
	Eigen::SparseMatrix<double> K(n_lexiv * 3, n_lexiv * 3);
	using trip = Eigen::Triplet<double>;
	std::vector<trip> trips;
	//std::vector<float> kij(n_gsvertices());
	std::vector<glm::mat3> kij(n_gsvertices());
	for (int i = 0; i < 27; i++) {
		int noff[3] = { i % 3 - 1, i / 3 % 3 - 1, i / 9 - 1 };
		cudaMemcpy(kij.data(), stencil_g[i], sizeof(glm::mat3) * n_gsvertices(), cudaMemcpyDeviceToHost);
		for (int j = 0; j < 9; j++) {
			//cudaMemcpy(kij.data(), stencil_g[i][j], sizeof(float) * n_gsvertices(), cudaMemcpyDeviceToHost);
			for (int k = 0; k < n_lexiv; k++) {
				int kpos[3] = {
					k % (cellReso[0] + 1), 
					k / (cellReso[0] + 1) % (cellReso[1] + 1),
					k / ((cellReso[0] + 1) * (cellReso[1] + 1)) 
				};
				int npos[3] = { kpos[0] + noff[0],kpos[1] + noff[1],kpos[2] + noff[2] };
				if (npos[0] < 0 || npos[0] > cellReso[0] || 
					npos[1] < 0 || npos[1] > cellReso[1] ||
					npos[2] < 0 || npos[2] > cellReso[2]) {
					continue;
				}
				int nid = npos[0] + npos[1] * (cellReso[0] + 1) + npos[2] * (cellReso[0] + 1) * (cellReso[1] + 1);
				trips.emplace_back(k * 3 + j / 3, nid * 3 + j % 3, kij[k][j % 3][j / 3]);
			}
		}
	}

	K.setFromTriplets(trips.begin(), trips.end());

	eigen2ConnectedMatlab(name, K);
}

Eigen::SparseMatrix<double> homo::Grid::stencil2matrix(bool removePeriodDof /*= true*/)
{
	Eigen::SparseMatrix<double> K;
	int ndof;
	if (removePeriodDof) {
		ndof = cellReso[0] * cellReso[1] * cellReso[2] * 3;
	} else {
		ndof = (cellReso[0] + 1) * (cellReso[1] + 1) * (cellReso[2] + 1) * 3;
	}
	
	K.resize(ndof, ndof);
	//std::vector<float> kij(n_gsvertices());
	std::vector<glm::mat3> kij(n_gsvertices());
	using trip = Eigen::Triplet<double>;
	std::vector<trip> trips;
	std::vector<VertexFlags> vflags(n_gsvertices());
	std::vector<CellFlags> eflags(n_gscells());
	cudaMemcpy(eflags.data(), cellflag, sizeof(CellFlags) * n_gscells(), cudaMemcpyDeviceToHost);
	cudaMemcpy(vflags.data(), vertflag, sizeof(VertexFlags) * n_gsvertices(), cudaMemcpyDeviceToHost);

	//{
	//	for (int i : {448, 450, 454, 456}) {
	//		int p[3];
	//		vgsid2lexpos_h(i, p);
	//		printf("i = %d  p = (%d, %d, %d)\n", i, p[0], p[1], p[2]);
	//	}
	//}

	if (!is_root) {
		for (int i = 0; i < 27; i++) {
			int noff[3] = { i % 3 - 1, i / 3 % 3 - 1, i / 9 - 1 };
			cudaMemcpy(kij.data(), stencil_g[i], sizeof(glm::mat3) * n_gsvertices(), cudaMemcpyDeviceToHost);
			for (int j = 0; j < 9; j++) {
				//cudaMemcpy(kij.data(), stencil_g[i][j], sizeof(float) * n_gsvertices(), cudaMemcpyDeviceToHost);
				for (int k = 0; k < n_gsvertices(); k++) {
					if (vflags[k].is_fiction() || vflags[k].is_period_padding() /*|| vflags[k].is_max_boundary()*/) continue;
					//int gscolor = vflags[k].get_gscolor();
					int vpos[3];
					vgsid2lexpos_h(k, vpos);
					int oldvpos[3] = { vpos[0],vpos[1],vpos[2] };
					if (removePeriodDof) {
						if (vpos[0] >= cellReso[0] || vpos[1] >= cellReso[1] || vpos[2] >= cellReso[2]) continue;
					} else {
						if (vpos[0] >= cellReso[0] + 1 || vpos[1] >= cellReso[1] + 1 || vpos[2] >= cellReso[2] + 1) continue;
					}
					int vid = vlexpos2vlexid_h(vpos, removePeriodDof);
					if (vid == 0 && i == 20) {
						//printf("k = %d  vid = %d  vpos = (%d, %d, %d)\n", k, vid, vpos[0], vpos[1], vpos[2]);
					}
					vpos[0] += noff[0]; vpos[1] += noff[1]; vpos[2] += noff[2];
					if (removePeriodDof) {
						for (int kk = 0; kk < 3; kk++) { vpos[kk] = (vpos[kk] + cellReso[kk]) % cellReso[kk]; }
					} else {
						bool outBound = false;
						for (int kk = 0; kk < 3; kk++) { outBound = outBound || vpos[kk] < 0 || vpos[kk]>cellReso[kk]; }
						if (outBound) continue;
					}
					int neiid = vlexpos2vlexid_h(vpos, removePeriodDof);
					if (vid == 0) {
						//printf("k = %d neiid = %d[%d]  off = (%d, %d, %d) nei = (%d, %d, %d)  val = %e\n",
						//	k, neiid, i, noff[0], noff[1], noff[2], vpos[0], vpos[1], vpos[2], kij[k]);
					}
					if (vid == 759 && neiid == 34582) {
						printf("vpos = (%d, %d, %d)  noff = (%d, %d, %d)\n",
							oldvpos[0], oldvpos[1], oldvpos[2],
							noff[0], noff[1], noff[2]);
					}
					trips.emplace_back(vid * 3 + j / 3, neiid * 3 + j % 3, kij[k][j % 3][j / 3]);
				}
			}
		}
	} else {
		std::vector<float> rhohost(n_gscells());
		cudaMemcpy(rhohost.data(), rho_g, sizeof(float) * n_gscells(), cudaMemcpyDeviceToHost);
		Eigen::Matrix<float, 24, 24> ke = getTemplateMatrix();
		for (int i = 0; i < eflags.size(); i++) {
			if (eflags[i].is_fiction() || eflags[i].is_period_padding()) continue;
			float rho_p = powf(rhohost[i], 1);
			int epos[3];
			egsid2lexpos_h(i, epos);
			for (int vi = 0; vi < 8; vi++) {
				int vipos[3] = { epos[0] + vi % 2, epos[1] + vi / 2 % 2, epos[2] + vi / 4 };
				// todo check Dirichlet boundary
				int vi_id = vlexpos2vlexid_h(vipos, removePeriodDof);
				for (int vj = 0; vj < 8; vj++) {
					int vjpos[3] = { epos[0] + vj % 2, epos[1] + vj / 2 % 2, epos[2] + vj / 4 };
					// todo check Dirichlet boundary
					int vj_id = vlexpos2vlexid_h(vjpos, removePeriodDof);
					for (int krow = 0; krow < 3; krow++) {
						for (int kcol = 0; kcol < 3; kcol++) {
							int ir = vi_id * 3 + krow;
							int jc = vj_id * 3 + kcol;
							trips.emplace_back(ir, jc, ke(vi * 3 + krow, vj * 3 + kcol) * rho_p);
						}
					}
				}
			}
		}
	}

	K.setFromTriplets(trips.begin(), trips.end());

	return K;
}

int homo::Grid::vgsid2lexid_h(int gsid, bool removePeriodDof /*= false*/)
{
	int lexpos[3];

	vgsid2lexpos_h(gsid, lexpos);

	int lexid = vlexpos2vlexid_h(lexpos, removePeriodDof);

	return lexid;
}

void homo::Grid::vgsid2lexpos_h(int gsid, int pos[3])
{
	int color = -1;
	for (int i = 0; i < 8; i++) {
		if (gsid < gsVertexSetEnd[i]) {
			color = i;
			break;
		}
	}
	if (color == -1) throw std::runtime_error("illegal gsid");
	int setid = color == 0 ? gsid : gsid - gsVertexSetEnd[color - 1];
	int gspos[3] = {
		setid % gsVertexReso[0][color],
		setid / gsVertexReso[0][color] % gsVertexReso[1][color],
		setid / (gsVertexReso[0][color] * gsVertexReso[1][color])
	};

	//printf("color = %d  setid = %d  gsvreso = (%d, %d, %d)  gsend = %d gspos = (%d, %d, %d)\n",
	//	color, setid, gsVertexReso[0][color], gsVertexReso[1][color], gsVertexReso[2][color],
	//	gsVertexSetEnd[color - 1], gspos[0], gspos[1], gspos[2]);

	int lexpos[3] = {
		gspos[0] * 2 + color % 2 - 1,
		gspos[1] * 2 + color / 2 % 2 - 1,
		gspos[2] * 2 + color / 4 - 1
	};

	//for (int i = 0; i < 3; i++) lexpos[i] = (lexpos[i] + cellReso[i]) % cellReso[i];

	for (int i = 0; i < 3; i++) pos[i] = lexpos[i];
}

void homo::Grid::egsid2lexpos_h(int gsid, int pos[3])
{
	int color = -1;
	for (int i = 0; i < 8; i++) {
		if (gsid < gsCellSetEnd[i]) {
			color = i;
			break;
		}
	}
	if (color == -1) throw std::runtime_error("illegal gsid");
	int setid = color == 0 ? gsid : gsid - gsCellSetEnd[color - 1];
	int gspos[3] = {
		setid % gsCellReso[0][color],
		setid / gsCellReso[0][color] % gsCellReso[1][color],
		setid / (gsCellReso[0][color] * gsCellReso[1][color])
	};
	int lexpos[3] = {
		gspos[0] * 2 + color % 2 - 1,
		gspos[1] * 2 + color / 2 % 2 - 1,
		gspos[2] * 2 + color / 4 - 1
	};

	//for (int i = 0; i < 3; i++) lexpos[i] = (lexpos[i] + cellReso[i]) % cellReso[i];

	for (int i = 0; i < 3; i++) pos[i] = lexpos[i];
}

int homo::Grid::vlexpos2vlexid_h(int lexpos[3], bool removePeriodDof/* = false*/)
{
	int vreso[3];
	for (int i = 0; i < 3; i++) lexpos[i] = (lexpos[i] + cellReso[i]) % cellReso[i];
	if (removePeriodDof) {
		for (int i = 0; i < 3; i++) {
			vreso[i] = cellReso[i];
		}
	} else {
		for (int i = 0; i < 3; i++) {
			vreso[i] = cellReso[i] + 1;
		}
	}

	for (int i = 0; i < 3; i++) {
		if (lexpos[i] < 0 || lexpos[i] >= vreso[i])
			throw std::runtime_error("illegal lexpos");
	}

	int lexid =
		lexpos[0] +
		lexpos[1] * vreso[0] +
		lexpos[2] * vreso[0] * vreso[1];

	return lexid;
}

int homo::Grid::vlexid2gsid(int lexid, bool hasPeriodDof /*= false*/)
{
	int pos[3];
	int vreso[3];
	if (hasPeriodDof) {
		for (int i = 0; i < 3; i++) vreso[i] = cellReso[i] + 1;
	} else {
		for (int i = 0; i < 3; i++) vreso[i] = cellReso[i];
	}
	pos[0] = lexid % vreso[0] + 1;
	pos[1] = lexid / vreso[0] % vreso[1] + 1;
	pos[2] = lexid / (vreso[0] * vreso[1]) + 1;
	int gspos[3] = { pos[0] / 2,pos[1] / 2,pos[2] / 2 };
	int color = pos[0] % 2 + pos[1] % 2 * 2 + pos[2] % 2 * 4;
	int setid = gspos[0] +
		gspos[1] * gsVertexReso[0][color] +
		gspos[2] * gsVertexReso[0][color] * gsVertexReso[1][color];
	int base = color == 0 ? 0 : gsVertexSetEnd[color - 1];
	int gsid = base + setid;
	return gsid;
}

void homo::Grid::enforce_period_boundary(float* v[3], bool additive /*= false*/)
{
	//if (additive) { throw std::runtime_error("additive should never be set"); }
	enforce_period_vertex(v, additive);
	pad_vertex_data(v);
}

void homo::Grid::test_gs_relaxation(void)
{
	useGrid_g();
	//v3_wave(f_g, { 10,10,10 });
	//enforce_dirichlet_boundary(f_g);
	//enforce_period_boundary(f_g, true);
	//v3_write(getPath("fperiod"), f_g);
	int itn = 0;
	update_residual();
	double rel_res = relative_residual();
	printf("rel_res = %6.4f%%\n", rel_res * 100);
	reset_displacement();
	v3_toMatlab("f0", f_g);
	v3_toMatlab("u0", u_g);
	v3_toMatlab("r0", r_g);

	while (itn++ < 200) {
		gs_relaxation();
		v3_toMatlab("u1", u_g);
		enforce_period_boundary(u_g, false);
		update_residual();
		v3_toMatlab("f", f_g);
		v3_toMatlab("r", r_g);
		v3_toMatlab("u2", u_g);
		if (itn % 5 == 0) {
			char buf[100];
			sprintf_s(buf, "res%d", itn);
			v3_write(getPath(buf), r_g, true);
		}
		double rel_res = relative_residual();
		printf("rel_res = %6.4f%%\n", rel_res * 100);
	}
	exit(0);
}
void homo::Grid::translateForce(int type_, float* v[3]) {
	float t_f[3];
	if (type_ == 1) {
		int gsid = vlexid2gsid(0, true);
		for (int i = 0; i < 3; i++)
			cudaMemcpy(t_f + i, v[i] + gsid, sizeof(float), cudaMemcpyDeviceToHost);
	} else if (type_ == 2) {
		v3_average(v, t_f, true);
	}

	v3_removeT(v, t_f);
}

