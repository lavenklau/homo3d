#include "cmdline.h"
#include <fstream>

DEFINE_int32(reso, 64, "resolution of domain");
DEFINE_string(obj, "bulk", "objective to be optimized: bulk, shear, npr, [customfile]");
DEFINE_string(init, "randc", "init density type, can be: rand, P, D, G, IWP");
DEFINE_string(sym, "reflect6", "symmetry catogory, can be: simp3, none");
DEFINE_double(vol, 0.3, "volume ratio");
DEFINE_double(E, 1e1, "Young's modulu of base material");
DEFINE_double(mu, 0.3, "Poisson ratio of base material");
DEFINE_string(prefix, "", "output path prefix, the final output path is \'prefix+filename\'");
DEFINE_int32(logrho, 0, "log density field per steps, set 0 to disable log");
DEFINE_int32(logc, 0, "log elastic tensor per x steps, set 0 to disable log");
DEFINE_int32(logsens, 0, "log sensitivity per x steps, set 0 to disable log");
DEFINE_int32(logobj, 0, "log objective value per x steps, set 0 to disable log");
DEFINE_string(test, "none", "test name");
DEFINE_bool(managedmem, true, "enable use of managed memory");
DEFINE_string(in, "", "input density field for test");
DEFINE_int32(N, 300, "maximal iteration of optimization");
DEFINE_int32(initperiod, 10, "maximal period basis for initial density");
DEFINE_double(finthres, 5e-4, "threshold of change ratio used for objective convergence check");
DEFINE_double(filter, 2, "filter radius");
DEFINE_double(step, 0.05, "design step for oc");
DEFINE_double(damp, 0.5, "damp ratio for oc");
DEFINE_double(relthres, 0.01, "relative residual threshold for FEM");
DEFINE_bool(periodfilt, false, "use periodic filter");
DEFINE_bool(usesym, true, "whether to use symmetrization");

void cfg::HomoConfig::parse(int argc, char** argv)
{
	gflags::SetVersionString("1.0");
	gflags::SetUsageMessage("Homo3d option:\n");
	gflags::ParseCommandLineFlags(&argc, &argv, false);
	std::string  cmdline_str = gflags::CommandlineFlagsIntoString();

	// write cmdline config to file
	std::ofstream ofs(FLAGS_prefix + "cmdline"); ofs << cmdline_str; ofs.close();

	// fill resolution field
	reso[0] = reso[1] = reso[2] = FLAGS_reso;

	// fill obj field
	if (FLAGS_obj == "bulk") {
		obj = Objective::bulk;
	}
	else if (FLAGS_obj == "shear") {
		obj = Objective::shear;
	}
	else if (FLAGS_obj == "npr") {
		obj = Objective::npr;
	}
	else if (FLAGS_obj == "custom") {
		obj = Objective::custom;
	}
	else {
		printf("\033[31munrecognized Objective type %s\033[0m\n", FLAGS_obj.c_str());
		exit(-1);
	}

	// fill symmetry type
	if (FLAGS_sym == "reflect3") {
		sym = Symmetry::reflect3;
	}
	else if (FLAGS_sym == "reflect6") {
		sym = Symmetry::reflect6;
	}
	else if (FLAGS_sym == "rotate3") {
		sym = Symmetry::rotate3;
	}
	else if (FLAGS_sym == "none") {
		sym = Symmetry::NONE;
	}
	else {
		printf("\033[31munrecognized symmetry type %s\033[0m\n", FLAGS_sym.c_str());
		exit(-1);
	}

	// fill init density means
	if (FLAGS_init == "rand") {
		winit = InitWay::random;
	}
	else if (FLAGS_init == "randc") {
		winit = InitWay::randcenter;
	}
	else if (FLAGS_init == "reprandc") {
		winit = InitWay::rep_randcenter;
	}
	else if (FLAGS_init == "noise") {
		winit = InitWay::noise;
	}
	else if (FLAGS_init == "interp") {
		winit = InitWay::interp;
	}
	else if (FLAGS_init == "P") {
		winit = InitWay::P;
	}
	else if (FLAGS_init == "G") {
		winit = InitWay::G;
	}
	else if (FLAGS_init == "D") {
		winit = InitWay::D;
	}
	else if (FLAGS_init == "IWP") {
		winit = InitWay::IWP;
	}
	else if (FLAGS_init == "manual") {
		winit = InitWay::manual;
	}
	else {
		printf("\033[31munrecognized density initialize mean %s\033[0m\n", FLAGS_init.c_str());
		exit(-1);
	}

	// fill volume ratio
	volRatio = FLAGS_vol;

	// fill Young's Modulus
	youngsModulu = FLAGS_E;

	// fill Poisson ratio
	poissonRatio = FLAGS_mu;

	// fill out path prefix
	outprefix = FLAGS_prefix;

	// whether to log density field
	logrho = FLAGS_logrho;

	// whether to log CH
	logc = FLAGS_logc;

	// whether to log sensitivity
	logsens = FLAGS_logsens;

	// whether to log objective value
	logobj = FLAGS_logobj;

	// fill testname
	testname = FLAGS_test;

	// fill use managed mem flag
	useManagedMemory = FLAGS_managedmem;

	// input density field
	inputrho = FLAGS_in;

	// max iter
	max_iter = FLAGS_N;

	// initial period
	initperiod = FLAGS_initperiod;

	// input threshold
	finthres = FLAGS_finthres;

	// filter radius
	filterRadius = FLAGS_filter;

	// design step
	designStep = FLAGS_step;

	// damp ratio
	dampRatio = FLAGS_damp;

	// relative residual threshold
	femRelThres = FLAGS_relthres;

	// print parsed configuration
	printf("Configuration : \n");
	printf(" = reso  - - - - - - - - - - - - - - - - - - - - - - - %d\n", FLAGS_reso);
	printf(" = obj   - - - - - - - - - - - - - - - - - - - - - - - %s\n", FLAGS_obj.c_str());
	printf(" = init  - - - - - - - - - - - - - - - - - - - - - - - %s\n", FLAGS_init.c_str());
	printf(" = sym   - - - - - - - - - - - - - - - - - - - - - - - %s\n", FLAGS_sym.c_str());
	printf(" = vol   - - - - - - - - - - - - - - - - - - - - - - - %4.2f\n", float(FLAGS_vol));
	printf(" = E     - - - - - - - - - - - - - - - - - - - - - - - %4.2e\n", float(FLAGS_E));
	printf(" = vu    - - - - - - - - - - - - - - - - - - - - - - - %4.2f\n", float(FLAGS_mu));
	printf(" = output    - - - - - - - - - - - - - - - - - - - - - %s\n", FLAGS_prefix.c_str());
	printf(" = logrho    - - - - - - - - - - - - - - - - - - - - - %s\n", b2s(FLAGS_logrho));
	printf(" = logc    - - - - - - - - - - - - - - - - - - - - - - %s\n", b2s(FLAGS_logc));
	printf(" = logsens   - - - - - - - - - - - - - - - - - - - - - %s\n", b2s(FLAGS_logsens));
	printf(" = test    - - - - - - - - - - - - - - - - - - - - - - %s\n", FLAGS_test.c_str());
	printf(" = useManagedMem   - - - - - - - - - - - - - - - - - - %s\n", b2s(FLAGS_managedmem));
	printf(" = maxIter - - - - - - - - - - - - - - - - - - - - - - %d\n", FLAGS_N);
	printf(" = initPeriod  - - - - - - - - - - - - - - - - - - - - %d\n", FLAGS_initperiod);
	printf(" = finthres    - - - - - - - - - - - - - - - - - - - - %e\n", float(FLAGS_finthres));
	printf(" = filterRadius  - - - - - - - - - - - - - - - - - - - %4.2f\n", float(FLAGS_filter));
	printf(" = dampRatio     - - - - - - - - - - - - - - - - - - - %4.2f\n", float(FLAGS_damp));
	printf(" = designStep    - - - - - - - - - - - - - - - - - - - %4.2f\n", float(FLAGS_step));
	printf(" = femRelThres   - - - - - - - - - - - - - - - - - - - %4.2e\n", float(FLAGS_relthres));
	printf(" = input(optional) - - - - - - - - - - - - - - - - - - %s\n", FLAGS_in.c_str());
}
