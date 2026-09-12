#include "homogenization/Framework.cuh"

using namespace homo;
using namespace culib;

template<typename Scalar, typename RhoPhys>
void logIter(int iter, cfg::HomoConfig config, var_tsexp_t<>& rho, Tensor<Scalar> sens, elastic_tensor_t<Scalar, RhoPhys>& Ch, double obj) {
	// fixed log
	if (iter % 5 == 0) {
		rho.value().toVdb(getPath("rho"));
		rho.diff().graft(sens.data());
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
		auto ch = Ch.data();
		std::ofstream ofs;
		if (iter == 0) {
			ofs.open(getPath(namebuf));
		} else {
			ofs.open(getPath(namebuf), std::ios::app);
		}
		ofs << "iter " << iter << " ";
		for (int i = 0; i < 36; i++) {
			ofs << ch[i] << " ";
		}
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
		} else {
			ofs.open(getPath(namebuf), std::ios::app);
		}
		ofs << "iter " << iter << " ";
		ofs << "obj = " << obj << std::endl;
		ofs.close();
	}
}

void initDensity(var_tsexp_t<>& rho, cfg::HomoConfig config) {
	int resox = rho.value().length(0);
	int resoy = rho.value().length(1);
	int resoz = rho.value().length(2);
	constexpr float pi = 3.1415926;
	if (config.winit == cfg::InitWay::random || config.winit == cfg::InitWay::randcenter) {
		randTri(rho.value(), config);
	} else if (config.winit == cfg::InitWay::manual) {
		rho.value().fromVdb(config.inputrho, false);
	} else if (config.winit == cfg::InitWay::interp) {
		rho.value().fromVdb(config.inputrho, true);
	} else if (config.winit == cfg::InitWay::rep_randcenter) {
		randTri(rho.value(), config);
	} else if (config.winit == cfg::InitWay::noise) {
		rho.value().rand(0.f, 1.f);
		symmetrizeField(rho.value(), config.sym);
		rho.value().proj(20.f, 0.5f);
		auto view = rho.value().view();
		auto ker = [=] __device__(int id) { return view(id); };
		float s = config.volRatio / (sequence_sum(ker, view.size(), 0.f) / view.size());
		rho.value().mapInplace([=] __device__(int x, int y, int z, float val) {
			float newval = val * s;
			if (newval < 0.001f)
				newval = 0.001;
			if (newval >= 1.f)
				newval = 1.f;
			return newval;
		});
	} else if (config.winit == cfg::InitWay::P) {
		rho.rvalue().setValue([=] __device__(int i, int j, int k) {
			float p[3] = {float(i) / resox, float(j) / resoy, float(k) / resoz};
			float val = cosf(2 * pi * p[0]) + cosf(2 * pi * p[1]) + cosf(2 * pi * p[2]);
			auto newval = tanproj(-val, 20);
			newval = max(min(newval, 1.f), 0.001f);
			return newval;
		});
	} else if (config.winit == cfg::InitWay::G) {
		rho.rvalue().setValue([=] __device__(int i, int j, int k) {
			float p[3] = {float(i) / resox, float(j) / resoy, float(k) / resoz};
			float s[3], c[3];
			for (int i = 0; i < 3; i++) {
				s[i] = sin(2 * pi * p[i]);
				c[i] = cos(2 * pi * p[i]);
			}
			float val = s[0] * c[1] + s[2] * c[0] + s[1] * c[2];
			auto newval = tanproj(val, 20);
			newval = max(min(newval, 1.f), 0.001f);
			return newval;
		});
	} else if (config.winit == cfg::InitWay::D) {
		rho.rvalue().setValue([=] __device__(int i, int j, int k) {
			float p[3] = {float(i) / resox, float(j) / resoy, float(k) / resoz};
			float x = p[0], y = p[1], z = p[2];
			float val = cos(2 * pi * x) * cos(2 * pi * y) * cos(2 * pi * z) - sin(2 * pi * x) * sin(2 * pi * y) * sin(2 * pi * z);
			float newval = tanproj(val, 20);
			newval = max(min(newval, 1.f), 0.001f);
			return newval;
		});
	} else if (config.winit == cfg::InitWay::IWP) {
		rho.rvalue().setValue([=] __device__(int i, int j, int k) {
			float p[3] = {float(i) / resox, float(j) / resoy, float(k) / resoz};
			float x = p[0], y = p[1], z = p[2];
			float val = 2 * (cos(2 * pi * x) * cos(2 * pi * y) + cos(2 * pi * y) * cos(2 * pi * z) + cos(2 * pi * z) * cos(2 * pi * x)) -
						(cos(2 * 2 * pi * x) + cos(2 * 2 * pi * y) + cos(2 * 2 * pi * z));
			float newval = tanproj(val, 20);
			newval = max(min(newval, 1.f), 0.001f);
			return newval;
		});
	}

	// symmetrize density field
	symmetrizeField(rho.value(), config.sym);

	// clamp density value to [rho_min, 1]
	rho.value().clamp(0.001, 1);
}

template<typename Scalar, typename RhoPhys>
void optiBulk(cfg::HomoConfig config, var_tsexp_t<>& rho, Homogenization& hom, elastic_tensor_t<Scalar, RhoPhys>& Ch) {
	int ereso[3] = {rho.value().size(0), rho.value().size(1), rho.value().size(2)};
	int reso = ereso[0];
	int ne = rho.value().size();
	// create a oc optimizer
	OCOptimizer oc(ne, 0.001, config.designStep, config.dampRatio);
	// define objective expression
	auto objective = -(Ch(0, 0) + Ch(1, 1) + Ch(2, 2) +
					   (Ch(0, 1) + Ch(0, 2) + Ch(1, 2)) * 2) /
					 3.; // bulk modulus
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
		if (criteria.is_converge(iter, val)) {
			printf("= converged\n");
			break;
		}
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
	hom.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);
	rho.value().toVdb(getPath("rho"));
	Ch.writeTo(getPath("C"));
}

template<typename Scalar, typename RhoPhys>
void optiShear(cfg::HomoConfig config, var_tsexp_t<>& rho, Homogenization& hom, elastic_tensor_t<Scalar, RhoPhys>& Ch) {
	int ereso[3] = {rho.value().size(0), rho.value().size(1), rho.value().size(2)};
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
		if (criteria.is_converge(iter, val)) {
			printf("= converged\n");
			break;
		}
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
	hom.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);
	rho.value().toVdb(getPath("rho"));
	Ch.writeTo(getPath("C"));
}

template<typename Scalar, typename RhoPhys>
void optiNpr(cfg::HomoConfig config, var_tsexp_t<>& rho, Homogenization& hom, elastic_tensor_t<Scalar, RhoPhys>& Ch) {
	int ne = rho.value().size();
	int ereso[3] = {rho.value().size(0), rho.value().size(1), rho.value().size(2)};
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
		auto objective = Ch(0, 1) + Ch(0, 2) + Ch(1, 2) -
						 (Ch(0, 0) + Ch(1, 1) + Ch(2, 2)) * powf(beta, iter);
		float val = objective.eval();
		// record objective value
		objlist.emplace_back(val);
		// compute derivative
		objective.backward(1);
		// output to screen
		printf("\033[32m\n * Iter %d   obj = %.4e\033[0m\n", iter, val);
		// check convergence
		if (criteria.is_converge(iter, val)) {
			printf("= converged\n");
			break;
		}
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
	hom.grid->writeDensity(getPath("density"), VoxelIOFormat::openVDB);
	rho.value().toVdb(getPath("rho"));
	Ch.writeTo(getPath("C"));
}

void runCustom(cfg::HomoConfig config);

void runInstance(cfg::HomoConfig config) {
	if (config.obj == cfg::Objective::custom) {
		runCustom(config);
		return;
	}
	// set output prefix
	setPathPrefix(config.outprefix);
	// create homogenization domain
	Homogenization hom(config);
	// update config resolution
	for (int i = 0; i < 3; i++)
		config.reso[i] = hom.getGrid()->cellReso[i];
	// define density expression
	var_tsexp_t<> rho(config.reso[0], config.reso[1], config.reso[2]);
	// initialize density
	initDensity(rho, config);
	// output initial density
	rho.value().toVdb(getPath("initRho"));
	// define penalty term
	auto rhop = rho.pow(3);
	// create elastic tensor expression
	elastic_tensor_t<float, decltype(rhop)> Ch(hom, rhop);
	AbortErr();
	if (config.obj == cfg::Objective::bulk) {
		optiBulk(config, rho, hom, Ch);
	} else if (config.obj == cfg::Objective::shear) {
		optiShear(config, rho, hom, Ch);
	} else if (config.obj == cfg::Objective::npr) {
		optiNpr(config, rho, hom, Ch);
	}
}

void example_yours(cfg::HomoConfig config) {
	// add your routines here ...
}

void runCustom(cfg::HomoConfig config) {
	example_yours(config);
}
