#include "MG.h"
#include <fstream>
#include "Eigen/IterativeLinearSolvers"
#include "matlab/matlab_utils.h"
#include "tictoc.h"
#include "utils.h"

std::shared_ptr<homo::Grid> homo::MG::getRootGrid(void)
{
	return grids[0];
}

void homo::MG::build(MGConfig config)
{
	mgConfig = config;

	std::shared_ptr<Grid> rootGrid(new Grid());
	if (!rootGrid) { throw std::runtime_error("failed to create root Grid"); }

	GridConfig gcon;
	gcon.enableManagedMem = config.enableManagedMem;
	gcon.namePrefix = config.namePrefix;

	rootGrid->buildRoot(config.reso[0], config.reso[1], config.reso[2], gcon);

	grids.emplace_back(rootGrid);

	// coarse grid until enough
	auto coarseGrid = rootGrid->coarse2(gcon);

	if (!coarseGrid) { throw std::runtime_error("failed to create coarse Grid"); }

	while (coarseGrid) {
		grids.push_back(coarseGrid);
		coarseGrid = coarseGrid->coarse2(gcon);
	}
	
	// reset all vectors
	for (int i = 0; i < grids.size(); i++) {
		grids[i]->reset_residual();
		grids[i]->reset_displacement();
		grids[i]->reset_force();
	}
}

void homo::MG::v_cycle(float w_SOR /*= 1.f*/, int pre /*= 1*/, int post /*= 1*/)
{
	for (int i = 0; i < grids.size(); i++) {
		if (i != 0) {
			grids[i - 1]->update_residual();
			grids[i]->restrict_residual();
			grids[i]->reset_displacement();
		}
		if (i == grids.size() - 1) {
			grids[i]->solveHostEquation();
		} else {
			grids[i]->gs_relaxation(w_SOR);
		}
	}

	for (int i = grids.size() - 2; i >= 0; i--) {
		grids[i]->prolongate_correction();
		grids[i]->gs_relaxation(w_SOR);
	}

	// not necessary here
	//grids[0]->update_residual();

	return /*grids[0]->relative_residual()*/;
}

void homo::MG::v_cycle_verbose(float w_SOR /*= 1.f*/, int pre /*= 1*/, int post /*= 1*/)
{
	int depth = grids.size();
	depth = 2;
	for (int i = 0; i < depth; i++) {
		if (i != 0) {
			grids[i - 1]->update_residual();
			//grids[i - 1]->v3_toMatlab("r", grids[i - 1]->getResidual());
			grids[i]->restrict_residual();
			//grids[i]->v3_toMatlab("f", grids[i]->getForce());
			grids[i]->reset_displacement();
		}
		if (i == grids.size() - 1) {
			grids[i]->solveHostEquation();
		} else {
			grids[i]->gs_relaxation(w_SOR);
		}
		//grids[i]->v3_toMatlab("u", grids[i]->getDisplacement());
	}

	for (int i = depth - 2; i >= 0; i--) {
		//grids[i]->v3_toMatlab("u", grids[i]->getDisplacement());
		grids[i]->prolongate_correction();
		//grids[i]->v3_toMatlab("uc", grids[i]->getDisplacement());
		grids[i]->gs_relaxation(w_SOR);
		//grids[i]->v3_toMatlab("urx", grids[i]->getDisplacement());
	}

	grids[0]->update_residual();

	return /*grids[0]->relative_residual()*/;

}

void homo::MG::reset_displacement(void)
{
	for (int i = 0; i < grids.size(); i++) {
		grids[i]->reset_displacement();
	}
}

double homo::MG::solveEquation(double tol /*= 1e-2*/, bool with_guess /*= true*/)
{
	double rel_res = 1;
	int iter = 0;
	if (!with_guess) { grids[0]->reset_displacement(); }
#if 1
	//double* ftmp[3], * u0[3];
	double fnorm = grids[0]->v3_norm(grids[0]->f_g);
	//grids[0]->v3_create(ftmp);
	//grids[0]->v3_create(u0); grids[0]->v3_reset(u0);
	//grids[0]->v3_copy(ftmp, grids[0]->f_g);
	//grids[0]->v3_toMatlab("f", grids[0]->f_g);
	int overflow_counter = 2;
	bool enable_translate_displacement = false;
	std::vector<double> errlist;
	double uch = 1e-7;
	while ((rel_res > tol || uch > 1e-6) && iter++ < 200) {
#if 1
		v_cycle(1);
		//v_cycle_verbose(1);
#else
		grids[0]->gs_relaxation(1.6);
		grids[0]->update_residual();
		rel_res = grids[0]->relative_residual();
#endif
		if (enable_translate_displacement) grids[0]->translateForce(2, grids[0]->u_g);
		rel_res = grids[0]->residual() / (fnorm + 1e-10);
		if (rel_res > 10 || iter >= 199) {
			//throw std::runtime_error("numerical failure");
			if (rel_res > 10) {
				printf("\033[31m\nnumerical explode, resetting initial guess...\033[0m\n");
				std::cerr << "\033[31m\nnumerical explode, resetting initial guess...\033[0m\n";
			} else {
				printf("\033[31mFailed to converge\033[0m\n");
				std::cerr << "\033[31m\nnumerical explode, resetting initial guess...\033[0m\n";
			}
			overflow_counter--;
			if (overflow_counter > 0) {
			} else {
				printf("\033[31mFailed\033[0m\n");
				throw std::runtime_error("MG numerical explode");
			}
			enable_translate_displacement = true;
			grids[0]->v3_toMatlab("ferr", grids[0]->getForce());
			grids[0]->v3_toMatlab("rerr", grids[0]->getResidual());
			grids[0]->v3_toMatlab("uerr", grids[0]->getDisplacement());
			grids[0]->v3_write(getPath("ferr"), grids[0]->getForce(), true);
			grids[0]->v3_write(getPath("rerr"), grids[0]->getResidual(), true);
			grids[0]->v3_write(getPath("uerr"), grids[0]->getDisplacement(), true);
			auto& gc = *grids.rbegin();
			// write coarsest force
			gc->v3_write(getPath("berr"), gc->f_g, true);
			// write coarsest system matrix
			std::ofstream ofs(getPath("Khosterr")); ofs << gc->Khost; ofs.close();
			// write solved x
			gc->v3_write(getPath("xerr"), gc->u_g, true);
			// write gs pos
			gc->writeGsVertexPos(getPath("poserr"));
			grids[0]->reset_displacement();
			grids[0]->writeDensity(getPath("rhoerr"), VoxelIOFormat::openVDB);
			//for (int itererr = 0; itererr < 200; itererr++) {
			//	v_cycle_verbose(1);
			//	grids[0]->v3_write(getPath("ferr"), grids[0]->getForce());
			//	grids[0]->v3_write(getPath("uerr"), grids[0]->getDisplacement());
			//	grids[0]->v3_write(getPath("rerr"), grids[0]->getResidual());
			//	rel_res = grids[0]->residual() / (fnorm + 1e-10);
			//	printf("rr = %4.2lf%%\n", rel_res * 100);
			//}
			grids[0]->reset_displacement();
		}
		//grids[0]->v3_toMatlab("r", grids[0]->r_g);
		//if (iter % 10 == 0) {
		//	grids[0]->v3_write(getPath("u"), grids[0]->u_g);
		//	grids[0]->v3_write(getPath("r"), grids[0]->r_g);
		//}
		//uch = grids[0]->v3_diffnorm(grids[0]->u_g, u0) / grids[0]->v3_norm(grids[0]->u_g);
		//grids[0]->v3_copy(u0, grids[0]->u_g);
		//grids[0]->v3_linear(1, ftmp, grids[0]->diag_strength, grids[0]->u_g, grids[0]->getForce());
		errlist.emplace_back(rel_res);
		printf("rel_res = %4.2lf%%    It.%d       \r", rel_res * 100, iter);
	}
	printf("\n");
	//grids[0]->v3_destroy(u0);
	//grids[0]->v3_destroy(ftmp);
	grids[0]->array2matlab("errs", errlist.data(), errlist.size());
#else
	rel_res = pcg();
#endif
	return rel_res;
}

double homo::MG::pcg(void)
{
	using VType = decltype(grids[0]->u_g);
	VType r, z, x, p, Ap;
	//grids[0]->v3_create(r);
	grids[0]->v3_create(z); grids[0]->v3_reset(z);
	grids[0]->v3_create(x); grids[0]->v3_reset(x);
	grids[0]->v3_create(p); grids[0]->v3_reset(p);
	grids[0]->v3_create(Ap); grids[0]->v3_reset(Ap);
	// for debug
	VType b;
	grids[0]->v3_create(b); grids[0]->v3_reset(b);
	grids[0]->v3_copy(b, grids[0]->f_g);

	// store r in grids[0]->f_g
	r[0] = grids[0]->f_g[0]; r[1] = grids[0]->f_g[1]; r[2] = grids[0]->f_g[2];

	double alpha = 0, beta = 0, sum_ = 0;

	grids[0]->useGrid_g();
	grids[0]->reset_displacement();
	//grids[0]->v3_copy(r, grids[0]->f_g);
	sum_ = grids[0]->v3_dot(r, r, true);
	printf("sum_ / norm = %e\n", sum_ / pow(grids[0]->v3_norm(r, true), 2));
	double rTr0 = sum_;

	//double fs[3];
	//grids[0]->v3_average(grids[0]->f_g, fs, true);
	//printf("fs = %.4e  %.4e  %.4e\n", fs[0], fs[1], fs[2]);
	//grids[0]->v3_removeT(grids[0]->f_g, fs);

	//grids[0]->stencil2matlab("K16");

	auto precondition = [&](/*double* v[3],*/ VType Mv) {
		//grids[0]->setForce(v);
		grids[0]->reset_displacement();
		double rr = 1;
		//while (rr >= 1)
		v_cycle(1.f);
		grids[0]->update_residual();
		rr = grids[0]->relative_residual();
		grids[0]->v3_copy(Mv, grids[0]->u_g);
		return rr;
	};

	auto update_p = [&](void) {
		grids[0]->v3_linear(1.f, z, beta, p, p);
	};

	auto compute_Ap = [&](VType p_, VType Ap_) {
		grids[0]->v3_stencilOnLeft(p_, Ap_);
	};

	// precondition
	precondition(/*r,*/ z);

	// update p
	update_p();
	
	sum_ = grids[0]->v3_dot(r, z, true);
	double zTr_last = sum_;

	double rel_res = 1;

	std::vector<double> errlist;

	int itn = 0;
	while (itn++ < 1000 && rel_res>1e-2) {
		compute_Ap(p, Ap);

		//grids[0]->v3_toMatlab("p", p);
		//grids[0]->v3_toMatlab("Ap", Ap);
		double pAp = grids[0]->v3_dot(p, Ap, true);

		alpha = zTr_last / pAp;

		//printf("zTr = %e  pAp = %e  alpha = %e\n", zTr_last, pAp, alpha);

		// update x
		grids[0]->v3_linear(1.f, x, alpha, p, x);

		// update r, r is stored in grids[0]->f_g
		grids[0]->v3_linear(1.f, r, -alpha, Ap, r);
		//grids[0]->v3_copy(grids[0]->u_g, x);
		//grids[0]->useFchar(0);
		//grids[0]->update_residual();
		//grids[0]->v3_copy(grids[0]->f_g, grids[0]->r_g);

		printf("pTr = %.4e ", grids[0]->v3_dot(r, p, true));

		//{
		//	// check energy
		//	double xAx = grids[0]->compliance(x, x);
		//	double bx = grids[0]->v3_dot(b, x, true);
		//	printf(" xAx = %.4e  bx = %.4e  E = %.4e ", xAx, bx, xAx - 2 * bx);
		//}

		double rTr = grids[0]->v3_dot(r, r, true);

		rel_res = sqrt(rTr / rTr0);

		errlist.emplace_back(rel_res);

		printf("res = %6.4lf%%\  alpha = %.4e\n", rel_res * 100, alpha);

		if (rel_res < 1e-2) { break; }

		double vres = precondition(z);
		//printf("vvres = %6.4lf%%\n", vres * 100);

		double zTr_new = grids[0]->v3_dot(z, r, true);

		beta = zTr_new / zTr_last;

		//printf("beta = %e\n", beta);

		update_p();

		zTr_last = zTr_new;
	}

	eigen2ConnectedMatlab("pcgerrs", Eigen::Matrix<double, -1, 1>::Map(errlist.data(), errlist.size()));

	grids[0]->v3_copy(grids[0]->u_g, x);
	//grids[0]->useFchar(0);
	//grids[0]->update_residual();
	//printf("final residual = %6.4lf%%\n", grids[0]->v3_norm(grids[0]->r_g, true) / grids[0]->v3_norm(grids[0]->f_g, true) * 100);

	grids[0]->v3_destroy(z);
	grids[0]->v3_destroy(x);
	grids[0]->v3_destroy(p);
	grids[0]->v3_destroy(Ap);

	return rel_res;
}

void homo::MG::updateStencils(void)
{
	for (int i = 1; i < grids.size(); i++) {
		grids[i]->restrict_stencil();
		// DEBUG
		//if (i == 1) {
		//	grids[1]->assembleHostMatrix();
		//	//printf("wait here");
		//}
		if (i == grids.size() - 1) {
			grids[i]->assembleHostMatrix();
		}
	}
}

void homo::MG::test_v_cycle(void)
{
	//grids[0]->v3_wave(grids[0]->f_g, { 1,1,1 });
	//grids[0]->enforce_dirichlet_boundary(grids[0]->f_g);
	//grids[0]->enforce_period_boundary(grids[0]->f_g, true);

	int itn = 0;
	grids[0]->writeDensity("initrho", VoxelIOFormat::openVDB);
#if 1
	// grids[1]->stencil2matlab("K0", true);
	// grids[2]->stencil2matlab("K1", true);
	// grids[1]->restrictMatrix2matlab("R", *grids[2]);
	std::vector<double> errhis;
	std::vector<double> timehis;
	double s_time = 0;
	for (int itn = 0; itn < 50; itn++) {
		_TIC("vc");
		v_cycle();
		// grids[0]->gs_relaxation();
		_TOC;
		grids[0]->update_residual();
		double err = grids[0]->relative_residual();
		errhis.push_back(err);
		printf("iter. %d   r = %e\n", itn, err);
		s_time += tictoc::get_record("vc");
		timehis.emplace_back(s_time);
	}
	array2ConnectedMatlab("errhis", errhis.data(), errhis.size());
	homoutils::writeVector("errhis",errhis);
	homoutils::writeVector("timehis",timehis);
	printf("average time = %.2f\n", s_time / 50);
	printf("=finished\n");
#elif 0
	grids[0]->v3_toMatlab("f", grids[0]->f_g);
	double fnorm = grids[0]->v3_norm(grids[0]->f_g);
	grids[0]->writeDensity(getPath("testrho"), VoxelIOFormat::openVDB);
	while (itn++ < 200) {
		grids[0]->gs_relaxation();
		grids[0]->update_residual();
		//grids[0]->v3_toMatlab("r", grids[0]->r_g);
		double rn = grids[0]->v3_norm(grids[0]->r_g);
		double rr = rn / fnorm;
		printf("rel = %6.4lf%%\n", rr * 100);

		if (rr > 1e4) {
			grids[0]->v3_toMatlab("r", grids[0]->r_g);
		}

		grids[1]->restrict_residual();
		grids[1]->reset_displacement();
		grids[1]->gs_relaxation();

		grids[0]->prolongate_correction();
		grids[0]->gs_relaxation();
	}
#elif 0
	grids[0]->v3_toMatlab("f0", grids[0]->f_g);
	grids[0]->update_residual();
	grids[0]->v3_toMatlab("r0", grids[0]->r_g);
	grids[0]->v3_write(getPath("res0"), grids[0]->r_g);
	printf("rel_res = %6.4lf%%\n", grids[0]->relative_residual() * 100);


	grids[1]->restrict_residual();
	grids[1]->v3_toMatlab("f", grids[1]->f_g);
	grids[1]->v3_toMatlab("u0", grids[1]->u_g);
	grids[1]->update_residual();
	grids[1]->v3_toMatlab("r1", grids[1]->r_g);
	double rel_res = grids[1]->relative_residual();
	printf("rel_res = %6.4lf%%\n", rel_res * 100);
	while (itn++ < 200) {
		grids[1]->gs_relaxation();
		grids[1]->v3_toMatlab("u", grids[1]->u_g);
		grids[1]->update_residual();
		grids[1]->v3_toMatlab("r", grids[1]->r_g);
		double rel_res = grids[1]->relative_residual();
		printf("rel_res = %6.4lf%%\n", rel_res * 100);
	}

#elif 0

	//grids[1]->stencil2matlab("K");

	//grids[grids.size() - 1]->stencil2matlab("K");

	std::vector<double> errlist;

	while (itn++ < 200) {
		for (int i = 0; i < grids.size(); i++) {
			if (i != 0) {
				grids[i - 1]->update_residual();
				//grids[i - 1]->v3_toMatlab("r", grids[i - 1]->r_g);
				grids[i]->restrict_residual();
				grids[i]->reset_displacement();
				//grids[i]->v3_toMatlab("f", grids[i]->f_g);
			}
			if (i != grids.size() - 1) grids[i]->gs_relaxation(/*1.8f*/);
			// coarsest
			if (i == grids.size() - 1) {
				//grids[i]->v3_toMatlab("f", grids[i]->f_g);
				//for (int j = 0; j < 30; j++) {
				//	grids[i]->update_residual();
				//	grids[i]->v3_toMatlab("u", grids[i]->u_g);
				//	grids[i]->v3_toMatlab("r", grids[i]->r_g);
				//	printf("  [C] rel = %6.4lf%%\n", grids[i]->relative_residual() * 100);
				//	grids[i]->gs_relaxation();
				//}
				grids[i]->solveHostEquation();
				//grids[i]->update_residual();
				//printf(" [C] r = %6.4lf%%\n", grids[i]->relative_residual() * 100);
			}
			//grids[i]->v3_toMatlab("u", grids[i]->u_g);
		}

		for (int i = grids.size() - 2; i >= 0; i--) {
			//grids[i]->v3_toMatlab("u", grids[i]->u_g);
			grids[i]->prolongate_correction();
			//grids[i]->v3_toMatlab("uc", grids[i]->u_g);
			grids[i]->gs_relaxation(/*1.8f*/);
		}

		grids[0]->update_residual();

		double rel_res = grids[0]->relative_residual();
		printf("[V] rel_res = %4.2lf%%\n", rel_res * 100);

		errlist.emplace_back(rel_res);
	}

	array2ConnectedMatlab("errs", errlist.data(), errlist.size());
	
#elif 1
	std::vector<double> errlist;
	double fnorm = grids[0]->v3_norm(grids[0]->f_g);
	grids[0]->readDisplacement(R"(C:\Users\zhangdi\Documents\temp\111\ustart)");
	while (itn++ < 200) {
		v_cycle(1);
		double rel_res = grids[0]->v3_norm(grids[0]->getResidual()) / fnorm;
		printf("res = %4.2lf%%\n", rel_res * 100);

		if (rel_res > 1e2) {
			printf("\033[31mNumerical explosion\033[0m\n");
			v_cycle_verbose(1);
			grids[1]->restrict_residual();
			double f1 = grids[1]->v3_norm(grids[1]->f_g);
			for (int iter = 0; iter < 200; iter++) {
				grids[1]->gs_relaxation();
				grids[1]->update_residual();
				//rel_res = grids[0]->v3_norm(grids[0]->getResidual()) / fnorm;
				rel_res = grids[1]->v3_norm(grids[1]->getResidual()) / f1;
				printf("rr = %4.2f%%\n", rel_res * 100);
			}
		}
		errlist.emplace_back(rel_res);
	}
	eigen2ConnectedMatlab("errs", Eigen::Matrix<double, -1, 1>::Map(errlist.data(), errlist.size()));

#elif 0
	std::vector<double> errlist;
	double* ftmp[3], * u0[3], * u1[3];
	grids[0]->v3_create(ftmp);
	grids[0]->v3_create(u0); grids[0]->v3_reset(u0);
	grids[0]->v3_create(u1);
	grids[0]->v3_copy(ftmp, grids[0]->getForce());
	while (itn++ < 50) {
		double rel_res = 1;
		//while (rel_res > 1e-2) {
			rel_res = v_cycle(1);
		//}
		//grids[0]->v3_toMatlab("u", grids[0]->u_g);
		double uch = grids[0]->v3_diffnorm(grids[0]->u_g, u0) / grids[0]->v3_norm(grids[0]->u_g);
		grids[0]->v3_copy(u0, grids[0]->u_g);
		//grids[0]->v3_linear(1, ftmp, 1e6, grids[0]->u_g, grids[0]->getForce());
		printf("res = %4.2lf%%  uch = %.4e\n", rel_res * 100, uch);
		errlist.emplace_back(rel_res);
	}
	eigen2ConnectedMatlab("errs", Eigen::Matrix<double, -1, 1>::Map(errlist.data(), errlist.size()));

#elif 0
	auto botGrid = grids[grids.size() - 1];
	for (int i = 1; i < grids.size(); i++) {
		grids[i - 1]->update_residual();
		grids[i]->restrict_residual();
	}
	botGrid->v3_toMatlab("f", botGrid->f_g);
	auto K = grids[grids.size() - 1]->stencil2matrix();
	eigen2ConnectedMatlab("K", K);
	Eigen::BiCGSTAB<Eigen::SparseMatrix<double>, Eigen::IncompleteCholesky<double>> solver;
	solver.compute(K);
	auto b = botGrid->v3_toMatrix(botGrid->f_g, true);
	eigen2ConnectedMatlab("b", b);
	Eigen::Matrix<double, -1, 1> x = solver.solve(b);
	eigen2ConnectedMatlab("x", x);
	botGrid->v3_fromMatrix(botGrid->u_g, x, false);
	botGrid->update_residual();
	botGrid->v3_toMatlab("u", botGrid->u_g);
	botGrid->v3_toMatlab("r", botGrid->r_g);
	printf("rel_res = %6.4lf%%\n", botGrid->relative_residual() * 100);

	botGrid->gs_relaxation();
	botGrid->v3_toMatlab("u", botGrid->u_g);
	botGrid->update_residual();
	botGrid->v3_toMatlab("r", botGrid->r_g);
	printf("rel_res = %6.4lf%%\n", botGrid->relative_residual() * 100);
#elif 0
	auto botGrid = grids[grids.size() - 1];
	for (int i = 1; i < grids.size(); i++) {
		grids[i - 1]->update_residual();
		grids[i]->restrict_residual();
	}
	botGrid->v3_toMatlab("f", botGrid->f_g);
	auto K = grids[grids.size() - 1]->stencil2matrix();
	eigen2ConnectedMatlab("K", K);
	botGrid->reset_displacement();
	while (itn++ < 200) {
		botGrid->gs_relaxation();
		botGrid->update_residual();
		botGrid->v3_toMatlab("r", botGrid->r_g);
		botGrid->v3_toMatlab("u", botGrid->u_g);
		printf("rel_res = %6.4lf%%\n", botGrid->relative_residual() * 100);
	}
#elif 1
	while (itn++ < 200) {
		v_cycle();
		double rel_res = grids[0]->relative_residual();
		printf("rel_res = %6.4lf%%\n", rel_res * 100);
	}
#endif
}

void homo::MG::test_diag_precondition(void)
{
	using VType = decltype(grids[0]->u_g);
	double d = 1e6;
	grids[0]->diagPrecondition(d);
	updateStencils();

	grids[0]->useFchar(0);
	grids[0]->v3_toMatlab("f0", grids[0]->getForce());
	double fnorm = grids[0]->v3_norm(grids[0]->f_g);
	VType ftmp; grids[0]->v3_create(ftmp); grids[0]->v3_copy(ftmp, grids[0]->getForce());
	VType utmp; grids[0]->v3_create(utmp);

	for (int itn = 0; itn < 1000; itn++) {
		grids[0]->v3_copy(utmp, grids[0]->getDisplacement());
		double unorm = grids[0]->v3_norm(utmp);
		v_cycle();
		double rr = grids[0]->residual() / (fnorm + 1e-10);
		double uch = grids[0]->v3_diffnorm(utmp, grids[0]->getDisplacement()) / (unorm + 1e-10);
		// update f
		//grids[0]->v3_toMatlab("ud", grids[0]->getDisplacement());
		grids[0]->v3_toMatlab("fd", grids[0]->getForce());
		grids[0]->v3_toMatlab("rd", grids[0]->getResidual());
		grids[0]->v3_linear(1.f, ftmp, d * 8, grids[0]->getDisplacement(), grids[0]->getForce());
		printf("* iter %04d rr = %4.2lf%%  uch = %.4le\n", itn, rr * 100, uch);
		if (uch < 1e-6) break;
	}

	grids[0]->update_residual();
	grids[0]->v3_toMatlab("rd", grids[0]->getResidual());
	//grids[0]->reset_force();
	//grids[0]->update_residual();
	//grids[0]->v3_toMatlab("r", grids[0]->getResidual());

	// check
	grids[0]->diagPrecondition(0);
	updateStencils();
	//grids[0]->update_residual();
	//grids[0]->v3_toMatlab("rc", grids[0]->getResidual());
	grids[0]->v3_copy(grids[0]->getForce(), ftmp);
	grids[0]->update_residual();
	grids[0]->v3_toMatlab("rc", grids[0]->getResidual());
	double rr = grids[0]->relative_residual();
	printf("Check :  rr = %4.2lf%%", rr * 100);
}

void homo::MG::v_cycle_profile(void)
{
	printf("\n\033[32m = = = = = = = = starting v-cycle profile = = = = = = = = = = \033[0m\n");
	double w_SOR = 1;
	for (int i = 0; i < grids.size(); i++) {
		printf("::grid %d \n", i);
		if (i != 0) {
			_TIC("update r");
			grids[i - 1]->update_residual();
			_TOC;
			printf("  update r      = %fms\n", tictoc::get_record("update r"));
			_TIC("restrict r");
			grids[i]->restrict_residual();
			_TOC;
			printf("  restrict r    = %fms\n", tictoc::get_record("restrict r"));
			grids[i]->reset_displacement();
		}
		if (i == grids.size() - 1) {
			grids[i]->solveHostEquation();
		}
		else {
			_TIC("gs relx");
			grids[i]->gs_relaxation(w_SOR);
			_TOC;
			printf("  gs relaxation = %fms\n", tictoc::get_record("gs relx"));
		}
	}

	for (int i = grids.size() - 2; i >= 0; i--) {
		printf("::grid %d\n", i);
		_TIC("prolong u");
		grids[i]->prolongate_correction();
		_TOC;
		printf("  prolongate    = %fms\n", tictoc::get_record("prolong u"));
		_TIC("gs relx");
		grids[i]->gs_relaxation(w_SOR);
		_TOC;
		printf("  gs relaxation = %fms\n", tictoc::get_record("gs relx"));
	}

	grids[0]->update_residual();

	return /*grids[0]->relative_residual()*/;
}

void homo::MG::test_sor(void)
{
	grids[0]->useFchar(0);
	updateStencils();
	double fnorm = grids[0]->v3_norm(grids[0]->getForce());
	std::vector<double> errlist;
	for (int itn = 0; itn < 100; itn++) {
		v_cycle(1);
		double rr = grids[0]->v3_norm(grids[0]->getResidual()) / fnorm;
		errlist.emplace_back(rr);
	}
	grids[0]->array2matlab("err", errlist.data(), errlist.size());

	errlist.clear();
	grids[0]->reset_displacement();
	for (int itn = 0; itn < 100; itn++) {
		v_cycle(0.8);
		double rr = grids[0]->v3_norm(grids[0]->getResidual()) / fnorm;
		errlist.emplace_back(rr);
	}
	grids[0]->array2matlab("errsor", errlist.data(), errlist.size());
}

void homo::MG::test_pcg(void)
{
	grids[0]->useFchar(0);
	//grids[0]->enforce_dirichlet_boundary(grids[0]->f_g);
	//grids[0]->enforce_period_boundary(grids[0]->f_g, true);
	
	pcg();
}
