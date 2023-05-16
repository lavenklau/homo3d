//#define _TUPLE_
//#define __CUDACC__
#include "Framework.cuh"

using namespace homo;

//void testExpression(void) {
//	var_exp_t<> v0(2), v1(1);
//	//auto v0r = v0.ref(), v1r = v1.ref();
//	auto p = (v0 + v1 + 3.) * (v0 + v1) * v1 /** (v0 - v1)*/ / (v0 - v1);
//	//auto p = (v0r + v1r + 3.) * (v0r + v1r) * v1r / (v0r - v1r);
//	p.eval();
//	p.backward(1);
//	std::cout << "p = " << p.value() << std::endl;
//	std::cout << "d_v0 = " << v0.diff() << ", d_v1 = " << v1.diff() << std::endl;
//}

template<typename Scalar, typename RhoPhys>
void logIter(int iter, cfg::HomoConfig config, var_tsexp_t<>& rho, Tensor<Scalar> sens, elastic_tensor_t<Scalar, RhoPhys>& Ch, double obj) {
	// fixed log 
	if (iter % 5 == 0) {
		rho.value().toVdb(getPath("rho"));
		rho.diff().graft(sens.data());
		//rho.diff().toVdb(getPath("sens"));
		Ch.writeTo(getPath("C"));
	}
	Ch.domain_.logger() << "finished iteration " << iter << std::endl;

	// optional log
	char namebuf[100];
	if (config.logrho != 0 && iter % config.logrho == 0) {
		sprintf_s(namebuf, "rho_%04d", iter);
		rho.value().toVdb(getPath(namebuf));
	}

	if (config.logc != 0 && iter % config.logc == 0) {
		sprintf_s(namebuf, "Clog");
		//Ch.writeTo(getPath(namebuf));
		auto ch = Ch.data();
		std::ofstream ofs;
		if (iter == 0) {
			ofs.open(getPath(namebuf));
		}
		else {
			ofs.open(getPath(namebuf), std::ios::app);
		}
		ofs << "iter " << iter << " ";
		for (int i = 0; i < 36; i++) { ofs << ch[i] << " "; }
		ofs << std::endl;
		ofs.close();
	}

	if (config.logsens != 0 && iter % config.logsens == 0) {
		sprintf_s(namebuf, "sens_%04d", iter);
		rho.diff().graft(sens.data());
		rho.diff().toVdb(getPath(namebuf));
	}

	if (config.logobj != 0 && iter % config.logobj == 0) {
		sprintf_s(namebuf, "objlog");
		std::ofstream ofs;
		if (iter == 0) {
			ofs.open(getPath(namebuf));
		}
		else {
			ofs.open(getPath(namebuf), std::ios::app);
		}
		ofs << "iter " << iter << " ";
		ofs << "obj = " << obj << std::endl;
		ofs.close();
	}
}

void test_OC(void) {
	int reso = 64;
	// define density expression
	var_tsexp_t<> rho(reso, reso, reso);
	// initialize density
	constexpr float pi = 3.1415926;
	rho.rvalue().setValue([=]__device__(int i, int j, int k) {
		float p[3] = { float(i) / reso, float(j) / reso , float(k) / reso };
		// center init
		float cp[3] = { abs(p[0] - 0.5f),abs(p[1] - 0.5f),abs(p[2] - 0.5f) };
		float val = sqrtf(cp[0] * cp[0] + cp[1] * cp[1] + cp[2] * cp[2]);
		if (val < 0.4) val = 0.1f;
		else val = 0.6f;
		if (val < 0.01f) val = 0.01;
		if (val > 1) val = 1;
		return val;
	});

	rho.value().toMatlab("initRho");
	radial_convker_t<float, Linear> convker(2, 0, true, false);
	auto rhop = rho.pow(3);
	var_tsexp_t<> dv(reso, reso, reso);
	auto dvexp = dv.conv(convker);
	Tensor<float> ones(reso, reso, reso);
	ones.reset(1);
	dvexp.eval(); dvexp.backward(ones);
	dv.diff().toMatlab("dv");
	auto dvflat = dv.diff().flatten();

	// define homogenization 
	Homogenization hom(reso, reso, reso);

	// diagonal precondition
	hom.ConfigDiagPrecondition(0);

	// get elastic tensor expression
	elastic_tensor_t<float, decltype(rhop)> Ch(hom, rhop);

	// define objective function as bulk modulus
	auto objective = -(
		Ch(0, 0) + Ch(1, 1) + Ch(2, 2) +
		(Ch(0, 1) + Ch(0, 2) + Ch(1, 2)) * 2
		) / 3.; // bulk modulus

	objective.eval(); objective.backward(1);
	rhop.value().toMatlab("rhop");
	hom.grid->writeDensity(getPath("initDensity"), VoxelIOFormat::openVDB);

	// OC optimizer
	int ne = pow(reso, 3);
	OCOptimizer oc(ne, 0.001, 0.02, 0.5);
	for (int iter = 0; iter < 1000; iter++) {
		float val = objective.eval();
		printf("\033[32m\n * Iter %d   obj = %.4e\033[0m\n", iter, val);
		objective.backward(1);
		auto sens = rho.diff().flatten();
		auto rhoarray = rho.value().flatten();
#if 0
		array_t<float> g(sens.view().data(), sens.view().size());
		array_t<float> dvarr(dvflat.view().data(), dvflat.view().size());
		//sens.toMatlab("sensold");
		g /= dvarr;
#else
		int ereso[3] = { reso,reso,reso };
		//sens.toMatlab("sensold");
		oc.filterSens(sens.data(), rhoarray.data(), reso, ereso);
		//sens.toMatlab("sensnew");
#endif
		//sens.toMatlab("sensnew");
		oc.update(sens.data(), rhoarray.data(), 0.3);
		rho.value().graft(rhoarray.data());
		if (iter % 5 == 0) {
			rho.value().toMatlab("rho");
			rho.diff().graft(sens.data());
			rho.diff().toMatlab("rhosens");
			rho.diff().toVdb(getPath("sens"));
			rho.value().toVdb(getPath("rho"));
			hom.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);
		}
	}
	
	rhop.value().toMatlab("rhofinal");
	hom.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);
}

void test_MMA(void) {
	int reso = 32;

	int ne = pow(reso, 3);

	// define density expression
	var_tsexp_t<> rho(reso, reso, reso);

	// initialize density
	constexpr float pi = 3.1415926;
	rho.rvalue().setValue([=]__device__(int i, int j, int k) {
		float p[3] = { float(i) / reso, float(j) / reso , float(k) / reso };
		float val = cosf(16 * pi * p[0]) + cosf(16 * pi * p[1]) + cosf(16 * pi * p[2]);
		val = (val + 3) / 6 * 0.3f + 0.3f;
		if (val < 0) val = 0.1;
		if (val > 1) val = 1;
		return val;
	});

	rho.value().toMatlab("initRho");

	auto rhop = rho.conv(radial_convker_t<float, Spline4>(2.3, 0.2)).pow(3);

	// define homogenization 
	Homogenization hom(reso, reso, reso);

	// diagonal precondition
	hom.ConfigDiagPrecondition(0);

	// get elastic tensor expression
	elastic_tensor_t<float, decltype(rhop)> Ch(hom, rhop);

	auto objective = -(
		Ch(0, 0) + Ch(1, 1) + Ch(2, 2) +
		(Ch(0, 1) + Ch(0, 2) + Ch(1, 2)) * 2
		) / 3.; // bulk modulus

	float goalVolRatio = 0.3;

	MMAOptimizer mma(1, ne, 1, 0, 1000, 1);
	mma.setBound(0.001, 1);

	for (int itn = 0; itn < 200; itn++) {
		float f0val = objective.eval();
		//rhop.value().toMatlab("rhop");
		objective.backward(1);
		auto rhoArray = rho.value().flatten();
		auto dfdx = rho.diff().flatten();
		//dfdx.toMatlab("dfdx");
		gv::gVector<float> dvdx(ne);
		dvdx.set(1);
		gv::gVector<float> gval(1);
		float* dgdx = dvdx.data();
		float curVol = gv::gVectorMap(rhoArray.data(), ne).sum();
		gval[0] = curVol - ne * goalVolRatio;
		printf("\033[32m \n* Iter %d  obj = %.4e  vol = %4.2f%%\033[0m\n", itn, f0val, curVol / ne * 100);
		//rhoArray.toMatlab("dfdx");
		mma.update(itn, rhoArray.data(), dfdx.data(), gval.data(), &dgdx);
		rho.rvalue().graft(rhoArray.data());
	}
	rhop.value().toMatlab("rhop");
	hom.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);
}

#include "homoCommon.cuh"


void cudaTest(void) {
#if 0
	Tensor<float> x = Tensor<float>::range(0, 1, 10000);;
	Tensor<float> fx(x.getDim());
	fx.copy(x);
	fx.mapInplace([=] __device__(int x, int y, int z, float val) {
		auto newval = tanproj(val, 20, 0.6);
		return newval;
	});
	fx.toMatlab("fx");
#elif 0
	Tensor<float> x(100, 100, 100);
	int n_period = 10;
	int nbasis1st = n_period * 3;
	int nbasis2nd = n_period * n_period * 3;
	int nbasis = nbasis1st + nbasis2nd;
	float* coeffs;
	cudaMalloc(&coeffs, sizeof(float)* nbasis);
	randArray(&coeffs, 1, nbasis, -1.f, 1.f);
	auto view = x.view();
	size_t block_size = 256;
	size_t grid_size = ceilDiv(view.size(), 32);
	randTribase_cos_kernel << <grid_size, block_size >> > (view, n_period, coeffs);
	cudaDeviceSynchronize();
	cuda_error_check;
	x.toVdb(getPath("x"));
	x.toMatlab("x");
	cudaFree(coeffs);
#elif 0
	Tensor<float> x(32, 32, 32);
	x.fromVdb("C:/Users/zhangdi/Documents/temp/homo/rho");
	x.symmetrize(Rotate3);
	x.toVdb("C:/Users/zhangdi/Documents/temp/homo/rho1");
#else
	Tensor<float> x(128, 128, 128);
	int n_period = 10;
	int nbasis1st = n_period * 6;
	int nbasis2nd = n_period * n_period * 36;
	int nbasis = nbasis1st + nbasis2nd;
	float* coeffs;
	cudaMalloc(&coeffs, sizeof(float)* nbasis);
	randArray(&coeffs, 1, nbasis, -1.f, 1.f);
	size_t block_size = 256;
	size_t grid_size = ceilDiv(x.view().size(), 32);
	randTribase_sincos_kernel << <grid_size, block_size >> > (x.view(), n_period, coeffs);
	cudaDeviceSynchronize();
	cudaFree(coeffs);
	cuda_error_check;
	x.toVdb("C:/Users/zhangdi/Documents/temp/homo/rho0");
	x.symmetrize(Rotate3);
	x.toVdb("C:/Users/zhangdi/Documents/temp/homo/rho1");
#endif
}


template<typename Scalar, typename RhoPhys>
void optiBulk(cfg::HomoConfig config, var_tsexp_t<>& rho, Homogenization& hom, elastic_tensor_t<Scalar, RhoPhys>& Ch) {
	int ereso[3] = { rho.value().size(0),rho.value().size(1),rho.value().size(2) };
	int reso = ereso[0];
	int ne = rho.value().size();
	// create a oc optimizer
	OCOptimizer oc(ne, 0.001, config.designStep, config.dampRatio);
	// define objective expression
	auto objective = -(Ch(0, 0) + Ch(1, 1) + Ch(2, 2) +
		(Ch(0, 1) + Ch(0, 2) + Ch(1, 2)) * 2) / 3.; // bulk modulus
	// record objective value
	std::vector<double> objlist;
	// convergence criteria
	ConvergeChecker criteria(config.finthres);
	// main loop of optimization
	for (int iter = 0; iter < config.max_iter; iter++) {
		// abort when cuda error occurs
		AbortErr();
		float val = objective.eval();
		// record objective value
		objlist.emplace_back(val);
		// compute derivative
		objective.backward(1);
		// output to screen
		printf("\033[32m\n * Iter %d   obj = %.4e\033[0m\n", iter, val);
		// check convergence
		if (criteria.is_converge(iter, val)) { printf("= converged\n"); break; }
		// make sensitivity symmetry
		symmetrizeField(rho.diff(), config.sym);
		// flatten the density and sensitivity tensor to array
		auto sens = rho.diff().flatten();
		auto rhoarray = rho.value().flatten();
		// filtering the sensitivity
		oc.filterSens(sens.data(), rhoarray.data(), reso, ereso, config.filterRadius);

		//sens.toMatlab("sensold");
		// update density
		oc.update(sens.data(), rhoarray.data(), config.volRatio);
		// graft array to tensor
		rho.rvalue().graft(rhoarray.data());
		// make density symmetry
		symmetrizeField(rho.value(), config.sym);
		// output temp results
		logIter(iter, config, rho, sens, Ch, val);
	}
	//rhop.value().toMatlab("rhofinal");
	hom.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);
	hom.grid->array2matlab("objlist", objlist.data(), objlist.size());
	rho.value().toVdb(getPath("rho"));
	Ch.writeTo(getPath("C"));
}

template<typename Scalar, typename RhoPhys>
void optiShear(cfg::HomoConfig config, var_tsexp_t<>& rho, Homogenization& hom, elastic_tensor_t<Scalar, RhoPhys>& Ch) {
	int ereso[3] = { rho.value().size(0),rho.value().size(1),rho.value().size(2) };
	int reso = ereso[0];
	int ne = rho.value().size();
	// create a oc optimizer
	OCOptimizer oc(ne, 0.001, config.designStep, config.dampRatio);
	// define objective expression
	auto objective = -(Ch(3, 3) + Ch(4, 4) + Ch(5, 5)) / 3.; // Shear modulus
	// record objective value
	std::vector<double> objlist;
	// convergence criteria
	ConvergeChecker criteria(config.finthres);
	// main loop of optimization
	for (int iter = 0; iter < config.max_iter; iter++) {
		// abort when cuda error occurs
		AbortErr();
		float val = objective.eval();
		// record objective value
		objlist.emplace_back(val);
		// compute derivative
		objective.backward(1);
		// output to screen
		printf("\033[32m\n * Iter %d   obj = %.4e\033[0m\n", iter, val);
		// check convergence
		if (criteria.is_converge(iter, val)) { printf("= converged\n"); break; }
		// make sensitivity symmetry
		symmetrizeField(rho.diff(), config.sym);
		// flatten the density and sensitivity tensor to array
		auto sens = rho.diff().flatten();
		auto rhoarray = rho.value().flatten();
		// filtering the sensitivity
		oc.filterSens(sens.data(), rhoarray.data(), reso, ereso, config.filterRadius);
		// update density
		oc.update(sens.data(), rhoarray.data(), config.volRatio);
		// graft array to tensor
		rho.rvalue().graft(rhoarray.data());
		// make density symmetry
		symmetrizeField(rho.value(), config.sym);
		// output temp results
		logIter(iter, config, rho, sens, Ch, val);
	}
	//rhop.value().toMatlab("rhofinal");
	hom.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);
	hom.grid->array2matlab("objlist", objlist.data(), objlist.size());
	rho.value().toVdb(getPath("rho"));
	Ch.writeTo(getPath("C"));
}

template<typename Scalar, typename RhoPhys>
void optiNpr(cfg::HomoConfig config, var_tsexp_t<>& rho, Homogenization& hom, elastic_tensor_t<Scalar, RhoPhys>& Ch) {
	int ne = rho.value().size();
	int ereso[3] = { rho.value().size(0),rho.value().size(1),rho.value().size(2) };
	int reso = ereso[0];
	// create a oc optimizer
	OCOptimizer oc(ne, 0.001, config.designStep, config.dampRatio);
	// record objective value
	std::vector<double> objlist;
	// convergence criteria
	ConvergeChecker criteria(config.finthres);
	// main loop of optimization
	for (int iter = 0; iter < config.max_iter; iter++) {
		// abort when cuda error occurs
		AbortErr();
		// define objective expression
		float beta = 0.6; // for relaxed poission ratio objective
#if 1
		// if (iter > 20) beta = 0.1;
		auto objective = Ch(0, 1) + Ch(0, 2) + Ch(1, 2) -
						 (Ch(0, 0) + Ch(1, 1) + Ch(2, 2)) * powf(beta, iter);
#elif 1
		//auto objective = ((Ch(0, 1) + Ch(0, 2) + Ch(1, 2))
		//	/ (Ch(0, 0) + Ch(1, 1) + Ch(2, 2)) * beta + 1).log() - (Ch(0, 0) + Ch(1, 1) + Ch(2, 2)).pow(0.5f) * 1e-3f;
		auto objective = ((Ch(0, 1) + Ch(0, 2) + Ch(1, 2))
			/ (Ch(0, 0) + Ch(1, 1) + Ch(2, 2)) * beta + 1).log() - (Ch(0, 0) + Ch(1, 1) + Ch(2, 2)).pow(0.5f) * 1e-3f;
#elif 0
		auto objective = Ch(0, 1) + Ch(0, 2) + Ch(1, 2) -
			(Ch(0, 0) + Ch(1, 1) + Ch(2, 2)).pow(0.5f); // Shear modulus
#endif
		float val = objective.eval();
		// record objective value
		objlist.emplace_back(val);
		// compute derivative
		objective.backward(1);
		// output to screen
		printf("\033[32m\n * Iter %d   obj = %.4e\033[0m\n", iter, val);
		// check convergence
		if (criteria.is_converge(iter, val)) { printf("= converged\n"); break; }
		// make sensitivity symmetry
		symmetrizeField(rho.diff(), config.sym);
		// flatten the density and sensitivity tensor to array
		auto sens = rho.diff().flatten();
		auto rhoarray = rho.value().flatten();
		// filtering the sensitivity
		oc.filterSens(sens.data(), rhoarray.data(), reso, ereso, config.filterRadius);
		// update density
		oc.update(sens.data(), rhoarray.data(), config.volRatio);
		// graft array to tensor
		rho.rvalue().graft(rhoarray.data());
		// make density symmetry
		symmetrizeField(rho.value(), config.sym);
		// output temp results
		logIter(iter, config, rho, sens, Ch, val);
	}
	//rhop.value().toMatlab("rhofinal");
	hom.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);
	hom.grid->array2matlab("objlist", objlist.data(), objlist.size());
	rho.value().toVdb(getPath("rho"));
	Ch.writeTo(getPath("C"));
}

void optiNpr2(cfg::HomoConfig config) {
	// set output prefix
	setPathPrefix(config.outprefix);
	// create homogenization domain
	Homogenization hom(config);
	// update config resolution
	for (int i = 0; i < 3; i++) config.reso[i] = hom.getGrid()->cellReso[i];
	// number of elements
	int ne = config.reso[0] * config.reso[1] * config.reso[2];
	// define density expression
	var_tsexp_t<> rho(config.reso[0], config.reso[1], config.reso[2]);
	// initialize density
	initDensity(rho, config);
	// output initial density
	rho.value().toVdb(getPath("initRho"));
	// rho physic
	auto rhop = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).pow(3);
	// create elastic tensor expression
	elastic_tensor_t<float, decltype(rhop)> Ch(hom, rhop);
	// record objective value
	std::vector<double> objlist;
	// mma optimizer
	MMAOptimizer mma(1, ne, 1, 0, 1000, 1);
	mma.setBound(0.001, 1);
	// convergence criteria
	ConvergeChecker criteria(config.finthres);
	for (int iter = 0; iter < config.max_iter; iter++) {
		printf("\033[32m*   iter %d\033[0m\n", iter);
#if 0
		auto objective = Ch(0, 1) + Ch(0, 2) + Ch(1, 2);
#else
		auto objective = ((Ch(0, 1) + Ch(0, 2) + Ch(1, 2))
			/ (Ch(0, 0) + Ch(1, 1) + Ch(2, 2)) * 0.6f + 1).log();
#endif
		float val = objective.eval();
		objective.backward(1);
		symmetrizeField(rho.diff(), config.sym);
		auto objdiff = rho.diff().flatten();
		Ch.holdOn();
		auto constrain = -(Ch(0, 0) + Ch(1, 1) + Ch(2, 2)) * 1e-4;
		auto gval = getTempPool().getUnifiedBlock<float>();
		gval.rdata<float>() = 4 + constrain.eval();
		constrain.backward(1);
		symmetrizeField(rho.diff(), config.sym);
		auto constraindiff = rho.diff().flatten();
		Ch.holdOff();
		auto rhoArray = rho.value().flatten();
		float* dgdx = constraindiff.data();
		printf("obj = %f    gval = %f\n", val, gval.rdata<float>());
		mma.update(iter, rhoArray.data(), objdiff.data(), gval.data<float>(), &dgdx);
		rho.rvalue().graft(rhoArray.data());
		symmetrizeField(rho.value(), config.sym);
		// output temp results
		auto sens = rho.diff().flatten();
		logIter(iter, config, rho, sens, Ch, val);
	}
	//rhop.value().toMatlab("rhofinal");
	hom.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);
	hom.grid->array2matlab("objlist", objlist.data(), objlist.size());
	rho.value().toVdb(getPath("rho"));
	Ch.writeTo(getPath("C"));
	freeMem();
}

//template<typename Scalar, typename RhoPhys>
void optiCustom(cfg::HomoConfig config) {
	// set output prefix
	setPathPrefix(config.outprefix);
	// create homogenization domain
	Homogenization hom(config);
	// update config resolution
	for (int i = 0; i < 3; i++) config.reso[i] = hom.getGrid()->cellReso[i];
	// number of elements
	int ne = config.reso[0] * config.reso[1] * config.reso[2];
	// define density expression
	var_tsexp_t<> rho(config.reso[0], config.reso[1], config.reso[2]);
	// initialize density
	initDensity(rho, config);
	// output initial density
	rho.value().toVdb(getPath("initRho"));
	// rho physic
	//auto rhop = rho.conv(radial_convker_t<float, Spline4>(config.filterRadius)).pow(3);
	auto rhop = rho.conv(radial_convker_t<float, Linear>(config.filterRadius)).pow(3);
	// create elastic tensor expression
	elastic_tensor_t<float, decltype(rhop)> Ch(hom, rhop);
	// record objective value
	std::vector<double> objlist;
	// oc optimizer
	OCOptimizer oc(ne, 0.001, config.designStep, config.dampRatio);
	// convergence criteria
	ConvergeChecker criteria(config.finthres);
	for (int iter = 0; iter < config.max_iter; iter++) {
		printf("\033[32m*   iter %d\033[0m\n", iter);
		auto objective = -(Ch(0, 0) + Ch(1, 1) + Ch(2, 2) +
			(Ch(0, 1) + Ch(0, 2) + Ch(1, 2)) * 2) / 9.f; // bulk modulus
		float val = objective.eval();
		// record objective value
		objlist.emplace_back(val);
		// compute derivative
		objective.backward(1);
		// output to screen
		printf("\033[32m\n * Iter %d   obj = %.4e\033[0m\n", iter, val);
		// check convergence
		if (criteria.is_converge(iter, val)) { printf("= converged\n"); break; }
		// make sensitivity symmetry
		symmetrizeField(rho.diff(), config.sym);
		// flatten the density and sensitivity tensor to array
		auto sens = rho.diff().flatten();
		auto rhoarray = rho.value().flatten();
		// update density
		oc.update(sens.data(), rhoarray.data(), config.volRatio);
		rho.rvalue().graft(rhoarray.data());
		// make density symmetry
		symmetrizeField(rho.value(), config.sym);
		// output temp results
		logIter(iter, config, rho, sens, Ch, val);
	}
	//rhop.value().toMatlab("rhofinal");
	hom.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);
	hom.grid->array2matlab("objlist", objlist.data(), objlist.size());
	rho.value().toVdb(getPath("rho"));
	Ch.writeTo(getPath("C"));
	freeMem();
}

extern void runCustom(cfg::HomoConfig config);

void runInstance(cfg::HomoConfig config) {
	if (config.obj == cfg::Objective::custom) {
		//optiCustom(config); return;
		runCustom(config); return;
	}
	// set output prefix
	setPathPrefix(config.outprefix);
	// create homogenization domain
	Homogenization hom(config);
	// update config resolution
	for (int i = 0; i < 3; i++) config.reso[i] = hom.getGrid()->cellReso[i];
	// number of elements
	int ne = config.reso[0] * config.reso[1] * config.reso[2];
	// define density expression
	var_tsexp_t<> rho(config.reso[0], config.reso[1], config.reso[2]);
	// initialize density
	initDensity(rho, config);
	// output initial density
	rho.value().toVdb(getPath("initRho"));
	// define penalty term
	//auto rhop = rho.conv(radial_convker_t<float, Spline4>(3.3, 0.2)).pow(3);
	auto rhop = rho.pow(3);
	// create elastic tensor expression
	elastic_tensor_t<float, decltype(rhop)> Ch(hom, rhop);
	AbortErr();
	if (config.obj == cfg::Objective::bulk) {
		optiBulk(config, rho, hom, Ch);
	}
	else if (config.obj == cfg::Objective::shear) {
		optiShear(config, rho, hom, Ch);
	}
	else if (config.obj == cfg::Objective::npr) {
		optiNpr(config, rho, hom, Ch);
	}

	freeMem();
}
