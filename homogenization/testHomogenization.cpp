#include "homogenization.h"
#include "cuda_runtime.h"
#include "homogenization/utils.h"
#include "templateMatrix.h"
#include "cmdline.h"
#include "tictoc.h"
#include <set>
#include <tuple>
#include "cuda_profiler_api.h"
#include "matlab/matlab_utils.h"
#include <filesystem>
#include <string>
#include <regex>

using namespace homo;

extern void cudaTest(void);

int findElement(Grid& grid) {
	auto eflags = grid.getCellflags();
	int cid = grid.n_gscells() / 2;
	for (int i = 0; i < 100; i++) {
		if (!eflags[cid + i].is_fiction() && !eflags[cid + i].is_period_padding()) {
			cid += i;
			break;
		}
	}
	if (eflags[cid].is_fiction() || eflags[cid].is_period_padding()) {
		printf("not suitable");
	}
	else {
		printf("cid = %d  flag = %04x\n", cid, int(eflags[cid].flagbits));
	}
	return cid;
}

void setRho(Grid& grid, int rhoid, float newval) {
	float* p = grid.rho_g + rhoid;
	float newrho = newval;
	cudaMemcpy(p, &newrho, sizeof(float), cudaMemcpyHostToDevice);
	
}

std::map<std::pair<char, char>, int> getLamuKeMap(void) {
	auto lam = getKeLam72();
	auto mu = getKeMu72();
	std::map<std::pair<char, char>, int> kemap;
	for (int i = 0; i < 24; i++) {
		for (int j = i; j < 24; j++) {
			int k = i + j * 24;
			kemap[std::pair<char, char>(lam[k], mu[k])] = k;
		}
	}
	return kemap;
}

// <lam, mu, vn, uxyz, fxyz>
std::map<std::tuple<char, char, char, char, char>, Eigen::Matrix<int, 8, 1>> getLamuSetU(void) {
	auto lam = getKeLam72();
	auto mu = getKeMu72();
	std::map<std::tuple<char, char, char, char, char>, Eigen::Matrix<int, 8, 1>>  lamu;
	for (int e = 0; e < 8; e++) {
		int vi = 7 - e;
		char krow = vi * 3;
		char epos[3] = { e % 2, e / 2 % 2, e / 4 };
		for (int vj = 0; vj < 8; vj++) {
			char vjpos[3] = { vj % 2, vj / 2 % 2, vj / 4 };
			char uj = (epos[0] + vjpos[0]) + (epos[1] + vjpos[1]) * 3 + (epos[2] + vjpos[2]) * 9;
			char kcol = vj * 3;
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					int k = krow + i + (kcol + j) * 24;
					if (lamu.count(std::make_tuple(lam[k], mu[k], uj, j, i))) {
						lamu[std::make_tuple(lam[k], mu[k], uj, j, i)][e]++;
					}
					else {
						lamu[std::make_tuple(lam[k], mu[k], uj, j, i)].fill(0);
						lamu[std::make_tuple(lam[k], mu[k], uj, j, i)][e]++;
					}
				}
			}
		}
	}
	return lamu;
}

void batchForwardMeasure(cfg::HomoConfig config) {
	Homogenization hom(config);
	for(const auto& entry : std::filesystem::recursive_directory_iterator(config.inputrho)) {
		auto fn = entry.path().stem().string();
		if (fn == "rho") {
			auto cmd = entry.path().parent_path() / "cmdline";
			std::cout << "[log] reading cmdline " << cmd << std::endl;
			std::ifstream ifs(cmd);
			if (!ifs) {
				std::cerr << "cannot find file " << cmd << std::endl;
				continue;
			}
			std::string sbuf((std::istream_iterator<char>(ifs)), (std::istream_iterator<char>()));
			std::regex volreg("(.|\n)*--vol=([0-9]?\\.[0-9]*)");
			std::smatch results;
			std::regex_match(sbuf, results, volreg);
			double volRatio = std::stod(results[2].str());
			std::cout << "matched volRatio = " << volRatio << std::endl;
			hom.logger() << "Goal volRatio = " << volRatio << std::endl;

			std::cout << "[log] reading file " << entry.path() << std::endl;

			hom.logger() << "reading file " << entry.path() << std::endl;
			hom.getGrid()->readDensity(entry.path().string(), VoxelIOFormat::openVDB);
			double c = hom.getGrid()->projectDensityToVolume(volRatio, 40);
			auto ereso = hom.getGrid()->cellReso;
			double vol = hom.getGrid()->sumDensity() / (ereso[0] * ereso[1] * ereso[2]);
			hom.logger() << "vol = " << vol << ", c = " << c << std::endl;
			std::cout << " vol = " << vol << ", c = " << c << std::endl;
			hom.mg_->updateStencils();
			hom.mg_->reset_displacement();
			double Ch[6][6];
			std::cout << "evaluating..." << std::endl;
			try {
				hom.elasticMatrix(Ch);
			}
			catch(...){
				std::cerr << "failed at file " << entry.path() << std::endl;
				for (int i = 0; i < 36; i++)
					Ch[i / 6][i % 6] = std::numeric_limits<double>::quiet_NaN();
			}
			std::ostringstream ostr;
			for (int i = 0; i < 36; i++) {
				ostr << Ch[i / 6][i % 6] << " ";
			}
			std::cout << "Ch = \n"
					  << Eigen::Matrix<double, 6, 6>::Map(Ch[0]) << std::endl;
			hom.logger() << "Ch = " << ostr.str() << std::endl;
		}
	}
}

void testHomogenization(cfg::HomoConfig config) {
#if 0
	Homogenization hom(64, 64, 64);
	hom.ConfigDiagPrecondition(0);
	hom.getGrid()->reset_density(1);
	hom.getGrid()->testIndexer();
	hom.getGrid()->readDensity(getPath("rhoerr"), openVDB);
	hom.update();
	double C[6][6];
	hom.elasticMatrix(C);
	//hom.getGrid()->v3_toMatlab("fchar0", hom.getGrid()->getFchar(0));
	for (int i = 0; i < 6; i++) {
		printf("%6.2lf  %6.2lf  %6.2lf  %6.2lf  %6.2lf  %6.2lf\n",
			C[i][0], C[i][1], C[i][2], C[i][3], C[i][4], C[i][5]);
	}
	exit(0);
#elif 0
	// test derivative
	double C_[6][6];
	Homogenization hom(16, 16, 16);
	auto grid = hom.getGrid();
	hom.ConfigDiagPrecondition(1e6);
	float initrho = 0.5;
	hom.getGrid()->reset_density(initrho);
	int cid = findElement(*grid);
	setRho(*grid, cid, initrho + 0.1);

	hom.update();
	hom.elasticMatrix(C_);
	float* sens = getMem().addBuffer("sens", grid->n_gscells() * sizeof(float))->data<float>();
	hom.Sensitivity(0, 0, sens);
	{
		std::vector<float> senshost(grid->n_gscells());
		cudaMemcpy(senshost.data(), sens, sizeof(float) * grid->n_gscells(), cudaMemcpyDeviceToHost);
		grid->array2matlab("senslist", senshost.data(), senshost.size());
	}

	setRho(*grid, cid, initrho + 0.11);
	hom.update();
	hom.elasticMatrix(C_);
#endif
	if (config.testname == "none") {
		printf("\033[32mNo test requiring\033[0m\n");
		return;
	}

	printf("\033[33mTest job %s\033[0m\n", config.testname.c_str());

	if (config.testname == "") {
	
	}
	else if(config.testname=="measure") {
		batchForwardMeasure(config);
	}
	else if (config.testname == "vidmap") {
		Homogenization hom(256, 256, 256);
		std::vector<int> vidmap[1];
		vidmap[0] = hom.getGrid()->getVertexLexidMap();
		homoutils::writeVectors(getPath("vidmap"), vidmap);
	}
	else if (config.testname == "eidmap") {
		Homogenization hom(256, 256, 256);
		std::vector<int> eidmap[1];
		eidmap[0] = hom.getGrid()->getCellLexidMap();
		homoutils::writeVectors(getPath("eidmap"), eidmap);
	}
	else if (config.testname == "vcycle") {
		Homogenization hom(config);
		if (config.inputrho.empty()) {
			hom.getGrid()->randDensity();
		} else {
			hom.getGrid()->readDensity(config.inputrho, VoxelIOFormat::openVDB);
		}
		hom.mg_->updateStencils();
		hom.getGrid()->useFchar(4);
		hom.mg_->test_v_cycle();
	}
	else if (config.testname == "testgs") {
		Homogenization hom(config);
		//hom.getGrid()->reset_density(1);
		hom.getGrid()->randDensity();
		hom.mg_->updateStencils();
		hom.getGrid()->useFchar(0);
		hom.getGrid()->test_gs_relaxation();
	}
	else if (config.testname == "testsor") {
		Homogenization hom(config);
		hom.getGrid()->randDensity();
		hom.mg_->updateStencils();
		hom.getGrid()->useFchar(0);
		std::vector<double> errlist[1];
		for (int i = 0; i < 100; i++) {
			hom.getGrid()->gs_relaxation(1.6);
			hom.getGrid()->update_residual();
			errlist[0].emplace_back(hom.getGrid()->relative_residual());
		}
		homoutils::writeVectors(getPath("errsor"), errlist);
		errlist[0].clear();
		hom.getGrid()->reset_displacement();
		for (int i = 0; i < 100; i++) {
			hom.getGrid()->gs_relaxation(1);
			hom.getGrid()->update_residual();
			errlist[0].emplace_back(hom.getGrid()->relative_residual());
		}
		homoutils::writeVectors(getPath("err"), errlist);
	}
	else if (config.testname == "forward") {
		Homogenization hom(config);
		if (config.inputrho.empty()) {
			//hom.getGrid()->reset_density(1);
			//hom.getGrid()->readDensity(getPath("initrho"), VoxelIOFormat::openVDB);
			hom.getGrid()->randDensity();
		} else {
			hom.getGrid()->readDensity(config.inputrho, VoxelIOFormat::openVDB);
		}
		hom.getGrid()->projectDensity(20, 0.5);
		hom.getGrid()->writeDensity(getPath("projrho"), VoxelIOFormat::openVDB);
		hom.mg_->updateStencils();
		double Ch[6][6];
		hom.elasticMatrix(Ch);
		printf("Ch = \n");
		for (int i = 0; i < 6; i++) {
			printf(" %6.4le  %6.4le  %6.4le  %6.4le  %6.4le  %6.4le\n",
				Ch[i][0], Ch[i][1], Ch[i][2], Ch[i][3], Ch[i][4], Ch[i][5]);
		}
#if 0
		double oldCh[6][6];
		for (int i = 0; i < 6; i++) {
			for (int j = i; j < 6; j++) {
				oldCh[i][j] = hom.elasticMatrix(i, j) / hom.getGrid()->n_cells();
			}
		}
		for (int i = 0; i < 6; i++) { for (int j = 0; j < i; j++) { oldCh[i][j] = oldCh[j][i]; } }
		printf("oldCh = \n");
		for (int i = 0; i < 6; i++) {
			printf(" %6.4le  %6.4le  %6.4le  %6.4le  %6.4le  %6.4le\n",
				oldCh[i][0], oldCh[i][1], oldCh[i][2], oldCh[i][3], oldCh[i][4], oldCh[i][5]);
		}
#endif
	}
	else if (config.testname == "forwardsdf") {
		Homogenization hom(config);
		if (config.inputrho.empty()) {
			hom.getGrid()->randDensity();
		} else {
			hom.getGrid()->interpDensityFromSDF(config.inputrho, VoxelIOFormat::openVDB);
		}
		hom.getGrid()->writeDensity(getPath("projrho"), VoxelIOFormat::openVDB);
		hom.mg_->updateStencils();
		double Ch[6][6];
		hom.elasticMatrix(Ch);
		printf("Ch = \n");
		for (int i = 0; i < 6; i++) {
			printf(" %6.4le  %6.4le  %6.4le  %6.4le  %6.4le  %6.4le\n",
				Ch[i][0], Ch[i][1], Ch[i][2], Ch[i][3], Ch[i][4], Ch[i][5]);
		}
	}
	else if (config.testname == "backwardprofile") {
		Homogenization hom(config);
		if (config.inputrho.empty()) {
			//hom.getGrid()->reset_density(1);
			printf("setting random density field...\n");
			hom.getGrid()->randDensity();
			//hom.getGrid()->writeDensity(getPath("initrho"), VoxelIOFormat::openVDB);
			//hom.getGrid()->readDensity(getPath("initrho"), VoxelIOFormat::openVDB);
		} else {
			//hom.getGrid()->readDensity(config.inputrho, VoxelIOFormat::openVDB);
			hom.getGrid()->interpDensityFrom(config.inputrho, VoxelIOFormat::openVDB);
		}
		_TIC("updatest");
		hom.mg_->updateStencils();
		_TOC;
		printf("update stencil  time  =  %4.2f ms\n", tictoc::get_record("updatest"));
		double Ch[6][6];
		//hom.elasticMatrix(Ch);
		float* sens = getMem().getBuffer(getMem().addBuffer(hom.getGrid()->n_cells() * sizeof(float)))->data<float>();
#if 1
		float dch[6][6] = { 1 };
		//Eigen::Matrix<float, 6, 6>::Map(dch[0]).setRandom();
		//cudaProfilerStart();
		//hom.Sensitivity(dch, sens, config.reso[0], true);
		//cudaProfilerStop();
		_TIC("backward");
		//cudaProfilerStart();
		hom.Sensitivity(dch, sens, config.reso[0], true);
		//cudaProfilerStop();
		_TOC;
		printf("backward sensitivity time  =  %4.2f ms\n", tictoc::get_record("backward"));
#else
		Eigen::Matrix<float, -1, -1> sensmat(hom.getGrid()->n_cells(), 21);
		int counter = 0;
		for (int i = 0; i < 6; i++) {
			for (int j = i; j < 6; j++) {
				float dch[6][6] = { 0 };
				dch[i][j] = 1;
				hom.Sensitivity(dch, sens, config.reso[0], true);
				cudaMemcpy(sensmat.col(counter).data(), sens, sizeof(float) * hom.getGrid()->n_cells(), cudaMemcpyDeviceToHost);
				counter++;
			}
		}
		eigen2ConnectedMatlab("wisesens", sensmat);
#endif
	}
	else if (config.testname == "memcheck") {
		Homogenization hom(config);
		//hom.getGrid()->readDensity(config.inputrho, VoxelIOFormat::openVDB);
		hom.getGrid()->reset_density(1);
		hom.mg_->updateStencils();
		hom.getGrid()->useFchar(0);
		hom.mg_->solveEquation();
		hom.getGrid()->setUchar(0, hom.getGrid()->getDisplacement());
		hom.elasticMatrix(0, 0);
	}
	else if (config.testname == "memusage") {
		Homogenization hom(config);
		size_t memuse = getMem().size("<H.*_st.*");
		std::cout << "= memory usage for stencil = " << memuse / 1024 / 1024 << " MB" << std::endl;
		memuse = getMem().size("<H.*_[ufr]_.*");
		std::cout << "= memory usage for ufr = " << memuse / 1024 / 1024 << " MB" << std::endl;
		memuse = getMem().size("<H.*vflag");
		std::cout << "= memory usage for vertex flag = " << memuse / 1024 / 1024 << " MB" << std::endl;
		memuse = getMem().size("<H.*cflag");
		std::cout << "= memory usage for element flag = " << memuse / 1024 / 1024 << " MB" << std::endl;
		memuse = getMem().size("<H.*_rho");
		std::cout << "= memory usage for rho = " << memuse / 1024 / 1024 << " MB" << std::endl;
		memuse = getMem().size(".*uchost.*");
		std::cout << "= memory usage for uchost  = " << memuse / 1024 / 1024 << " MB" << std::endl;
		memuse = getMem().size();
		std::cout << "= memory usage for total   = " << memuse / 1024 / 1024 << " MB" << std::endl;
	}
	else if (config.testname == "vprofile") {
		Homogenization hom(config);
		hom.getGrid()->reset_density(1);
		hom.mg_->updateStencils();
		hom.getGrid()->useFchar(0);
		// warm up
		hom.mg_->v_cycle();
		// profile
		hom.mg_->v_cycle_profile();
	}
	else if (config.testname == "relxprofile") {
		Homogenization hom(config);
		hom.getGrid()->reset_density(1);
		hom.mg_->updateStencils();
		hom.getGrid()->useFchar(0);
		// warm up
		hom.getGrid()->gs_relaxation();
		// profile
		hom.getGrid()->gs_relaxation_profile();
	}
	else if (config.testname == "relxex") {
		Homogenization hom(config);
		if (config.inputrho.empty()) {
			hom.getGrid()->randDensity();
			hom.getGrid()->writeDensity(getPath("initrho"), VoxelIOFormat::openVDB);
		} else {
			hom.getGrid()->readDensity(getPath("initrho"), VoxelIOFormat::openVDB);
		}
		hom.mg_->updateStencils();
#if 1
		cudaProfilerStart();
		hom.getGrid()->gs_relaxation_ex();
		cudaProfilerStop();
		_TIC("relxex");
		hom.getGrid()->gs_relaxation_ex();
		_TOC;
		printf("relxex  time = %f ms\n", tictoc::get_record("relxex"));
#else
		hom.getGrid()->useFchar(0);
		hom.getGrid()->reset_displacement();
		hom.getGrid()->gs_relaxation_ex(1.f);
		hom.getGrid()->update_residual();
		hom.getGrid()->v3_toMatlab("uex", hom.getGrid()->u_g);
		hom.getGrid()->v3_toMatlab("rex", hom.getGrid()->r_g);
		hom.getGrid()->reset_displacement();
		hom.getGrid()->gs_relaxation(1.f);
		hom.getGrid()->update_residual();
		hom.getGrid()->v3_toMatlab("u", hom.getGrid()->u_g);
		hom.getGrid()->v3_toMatlab("r", hom.getGrid()->r_g);
#endif
	}
	else if (config.testname == "updresprofile") {
		Homogenization hom(config);
		hom.getGrid()->reset_density(1);
		hom.mg_->updateStencils();
		hom.getGrid()->useFchar(0);
		// warm up
		hom.getGrid()->update_residual();
		// profile
		hom.getGrid()->update_residual_profile();
	}
	else if (config.testname == "updex") {
		Homogenization hom(config);
		hom.getGrid()->randDensity();
		hom.mg_->updateStencils();
		hom.getGrid()->useFchar(0);
#if 0
		hom.getGrid()->gs_relaxation_ex();
		//hom.getGrid()->update_residual();
		_TIC("rnew");
		hom.getGrid()->update_residual_ex();
		_TOC;
		printf("rnew time = %4.2f ms\n", tictoc::get_record("rnew"));
		hom.getGrid()->v3_toMatlab("rnew", hom.getGrid()->r_g);
		hom.getGrid()->v3_reset(hom.getGrid()->r_g);
		_TIC("rold");
		hom.getGrid()->update_residual();
		_TOC;
		printf("rold time = %4.2f ms\n", tictoc::get_record("rold"));
		hom.getGrid()->v3_toMatlab("rold", hom.getGrid()->r_g);
#else
		hom.getGrid()->v3_toMatlab("fold", hom.getGrid()->f_g);
		hom.mg_->solveEquation();
		hom.getGrid()->update_residual();
		hom.getGrid()->v3_toMatlab("rold", hom.getGrid()->r_g);
		hom.getGrid()->v3_reset(hom.getGrid()->r_g);
		hom.getGrid()->update_residual_ex();
		hom.getGrid()->v3_toMatlab("rnew", hom.getGrid()->r_g);
#endif
	}
	else if (config.testname == "testindex") {
		Homogenization hom(config);
		hom.getGrid()->testIndexer();
	}
	else if (config.testname == "nondyadic") {
		Homogenization hom(16, 16, 16);
		hom.getGrid()->reset_density(1);
		hom.update();
		auto& grids = hom.mg_->grids;
		auto& lastgrid = *grids.rbegin();
		//lastgrid->writeStencil(getPath("stencil4"));
		std::vector<int> vidmap[1];
		vidmap[0] = lastgrid->getVertexLexidMap();
		//homoutils::writeVectors(getPath("lastvidmap"), vidmap);
		//grids[0]->v3_rand(grids[0]->f_g, 0, 1);
		//grids[0]->enforce_period_vertex(grids[0]->f_g, true);
		//grids[0]->pad_vertex_data(grids[0]->f_g);
		grids[0]->v3_read(getPath("frand"), grids[0]->f_g);
		//grids[0]->v3_write(getPath("frand"), grids[0]->f_g);
		grids[0]->reset_displacement();
		grids[0]->update_residual();
		grids[1]->restrict_residual();
#if 1
		//grids[1]->v3_write(getPath("lastf4"), grids[1]->f_g);
#else
		grids[1]->reset_displacement();
		grids[1]->update_residual();
		grids[2]->restrict_residual();
		grids[2]->v3_write(getPath("lastf2"), grids[2]->f_g);
#endif

		//lastgrid->v3_rand(lastgrid->u_g, 0, 1);
		//lastgrid->enforce_period_boundary(lastgrid->u_g);
		lastgrid->v3_read(getPath("urand"), lastgrid->u_g);
		//lastgrid->v3_write(getPath("urand"), lastgrid->u_g);
		for (int i = grids.size() - 2; i >= 0; i--) {
			grids[i]->reset_displacement();
			grids[i]->prolongate_correction();
		}
		//grids[0]->v3_write(getPath("u2"), grids[0]->u_g);
		grids[0]->v3_write(getPath("u4"), grids[0]->u_g);
	}
	else if (config.testname == "lamuset") {
		auto lam = getKeLam72();
		auto mu = getKeMu72();
		std::set<std::pair<char, char>>  lamu;
		for (int i = 0; i < 24 * 24; i++) {
			lamu.insert(std::pair<char, char>(lam[i], mu[i]));
		}
		std::cout << "::Lam Mu set  " << std::endl;;
		for (auto iter = lamu.begin(); iter != lamu.end(); iter++) {
			printf("%d    :    %d\n", (int)iter->first, (int)iter->second);
		}
	}
	else if (config.testname == "lamusetu") {
		auto lamu = getLamuSetU();
		Eigen::IOFormat frmt(4, 0, ", ", "\n", "[", "]");
		std::cout << "::Lam Mu U set  total size = " << lamu.size() << std::endl;;
		printf("Lam      mu     vn     uxyz    fxyz   counter\n");
		std::map<std::pair<char, char>, std::pair<char, char>> redPair;
		decltype(lamu) redLamu;
		for (auto iter = lamu.begin(); iter != lamu.end(); iter++) {
			printf("%04d    %04d    %04d   %04d   %04d   ",
				(int)std::get<0>(iter->first),
				(int)std::get<1>(iter->first),
				(int)std::get<2>(iter->first),
				(int)std::get<3>(iter->first),
				(int)std::get<4>(iter->first)
			);
			for (int i = 0; i < 8; i++) {
				printf("%04d   ", iter->second[i]);
			}
			printf("\n");
			//std::cout << iter->second.transpose() << std::endl;
			char lm = std::get<0>(iter->first);
			char mu = std::get<1>(iter->first);
			char vn = std::get<2>(iter->first);
			char xyz = std::get<3>(iter->first);
			char fxyz = std::get<4>(iter->first);
			auto divisor = std::gcd(lm, mu);
			if (lm / divisor < 0) divisor *= -1;
			redPair[std::pair<char, char>(lm, mu)] = std::pair<char, char>(lm / divisor, mu / divisor);
			auto newkey = std::make_tuple(lm / divisor, mu / divisor, vn, xyz, fxyz);
			if (redLamu.count(newkey)) {
				redLamu[newkey] = redLamu[newkey] + divisor * iter->second;
			} else {
				redLamu[newkey] = divisor * iter->second;
			}
		}
		// reduce data
		std::vector<std::pair<std::decay_t<decltype(redLamu.begin()->first)>, decltype(redLamu.begin()->second)>> redLamuList;
		printf("\n::reduced total size = %d\n", (int)redLamu.size());
		printf("Lam      mu     vn      uxyz    fxyz    counter\n");
		std::map<std::pair<char, char>, int> lmid;
		for (auto iter = redLamu.begin(); iter != redLamu.end(); iter++) {
			printf("%04d    %04d    %04d   %04d    %04d    ",
				(int)std::get<0>(iter->first),
				(int)std::get<1>(iter->first),
				(int)std::get<2>(iter->first),
				(int)std::get<3>(iter->first),
				(int)std::get<4>(iter->first)
			);
			for (int i = 0; i < 8; i++) {
				printf("%04d   ", iter->second[i]);
			}
			printf("\n");
			//std::cout << iter->second.transpose().format(frmt) << std::endl;
			redLamuList.emplace_back(iter->first, iter->second);
			lmid[std::pair<char, char>(std::get<0>(iter->first), std::get<1>(iter->first))] = 0;
		}
		int counter = 0;
		for (auto iter = lmid.begin(); iter != lmid.end(); iter++) {
			iter->second = counter++;
		}
		// sort them
		std::sort(redLamuList.begin(), redLamuList.end(), [](auto const& v0, auto const& v1) {
			int vj0 = std::get<2>(v0.first);
			int vj1 = std::get<2>(v1.first);
			int x0 = std::get<3>(v0.first);
			int x1 = std::get<3>(v1.first);
			if (vj0 < vj1) return true;
			if (vj0 == vj1 && x0 < x1) return true;
			return false;
			});
		printf("\n::sorted = %d\n", (int)redLamuList.size());
		for (auto iter = lmid.begin(); iter != lmid.end(); iter++) {
			printf("%d  =  (%d, %d)\n", iter->second, iter->first.first, iter->first.second);
		}
		printf("LM     vn     xyz   fxyz   counter\n");
		for (auto iter = redLamuList.begin(); iter != redLamuList.end(); iter++) {
			printf("%04d    %04d   %04d   %04d   ",
				(int)lmid[std::pair<char, char>(std::get<0>(iter->first), std::get<1>(iter->first))],
				(int)std::get<2>(iter->first),
				(int)std::get<3>(iter->first),
				(int)std::get<4>(iter->first)
			);
			for (int i = 0; i < 8; i++) {
				printf("%04d   ", iter->second[i]);
			}
			printf("\n");
		}
	}
	else if (config.testname == "kesetu") {
		auto lamu = getLamuSetU();
		auto kemap = getLamuKeMap();
		std::map<std::tuple<int, char, char>, Eigen::Matrix<int, 8, 1>> kesetu;
		for (auto iter = lamu.begin(); iter != lamu.end(); iter++) {
			char lam = std::get<0>(iter->first);
			char mu = std::get<1>(iter->first);
			char vn = std::get<2>(iter->first);
			char xyz = std::get<3>(iter->first);
			int keid = kemap[std::pair<char, char>(lam, mu)];
			kesetu[std::make_tuple(keid, vn, xyz)] = iter->second;
		}

		std::cout << "::Ke U set  total size = " << kesetu.size() << std::endl;;
		printf("Ke     vn     xyz   counter\n");
		for (auto iter = kesetu.begin(); iter != kesetu.end(); iter++) {
			printf("%04d    %04d    %04d   ",
				(int)std::get<0>(iter->first),
				(int)std::get<1>(iter->first),
				(int)std::get<2>(iter->first)
			);
			for (int i = 0; i < 8; i++) {
				printf("%04d   ", iter->second[i]);
			}
			printf("\n");
			//std::cout << iter->second.transpose() << std::endl;
		}
	}
	else if (config.testname == "interpdensity") {
		Homogenization hom(config);
		if (config.inputrho.empty()) {
			return;
		} else {
			hom.getGrid()->interpDensityFrom(config.inputrho, VoxelIOFormat::openVDB);
			//hom.getGrid()->interpDensityFromSDF("C:/Users/zhangdi/Documents/temp/homo/meshrho", VoxelIOFormat::openVDB);
		}
		hom.getGrid()->writeDensity(getPath("interpRho"), VoxelIOFormat::openVDB);
	}
	else if (config.testname == "eigentest") {
		Eigen::Matrix<double, -1, -1> T(192, 32);
		Eigen::Matrix<double, -1, 1> b(192, 1);
		T.setRandom();
		b.setRandom();
		_TIC("eigen");
		b = b - T * (T.transpose() * b);
		_TOC;
		printf("eigen time usage = %4.2f ms \n", tictoc::get_record("eigen"));
	}
	else if (config.testname == "cudatest") {
		cudaTest();
	}
	else {
		printf("\033[31mNo test names %s\033[0m\n", config.testname.c_str());
	}

	printf("\n= = test finished\n");
	freeMem();
	exit(0);
}

