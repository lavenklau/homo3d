#include "homogenization.h"
#include "templateMatrix.h"
#include "matlab/matlab_utils.h"
#include <chrono>
#include <iomanip>

extern void uploadTemplaceMatrix(const double* ke, float penal);
void uploadTemplateLameMatrix(const char* kelam72, const char* kemu72, float Lam, float Mu);

std::string outpath = "C:/Users/zhangdi/Documents/temp/homo/";

namespace homo {
	std::string getPath(const std::string& str) {
		return outpath + str;
	}
	std::string setPathPrefix(const std::string& str) {
		outpath = str;
		return outpath;
	}

	size_t uid_t::uid;

}

homo::Homogenization::Homogenization(int xreso, int yreso, int zreso, double youngsModu /*= 1e6*/, double poissonRatio /*= 0.3*/)
{
	config.reso[0] = xreso;
	config.reso[1] = yreso;
	config.reso[2] = zreso;
	config.youngsModulu = youngsModu;
	config.poissonRatio = poissonRatio;
	build(config);
}

homo::Homogenization::Homogenization(cfg::HomoConfig config)
{
	build(config);
}

void homo::Homogenization::build(cfg::HomoConfig homconfig)
{
	config = homconfig;

	printf("[Homo] building domain with resolution [%d, %d, %d], E = %.4le, mu = %.4le\n",
		config.reso[0], config.reso[1], config.reso[2], config.youngsModulu, config.poissonRatio);
	mg_.reset(new MG());
	MGConfig mgconf;
	mgconf.namePrefix = getName();
	std::copy(config.reso, config.reso + 3, mgconf.reso);
	mgconf.enableManagedMem = config.useManagedMemory;
	//mgconf.namePrefix;
	mg_->build(mgconf);
	grid = mg_->getRootGrid();

	grid->test();

	initTemplateMatrix(1, getMem(), config.youngsModulu, config.poissonRatio);

	uploadTemplaceMatrix(getTemplateMatrixFp64().data(), power_penal);

	auto E = config.youngsModulu;
	auto v = config.poissonRatio;
	lamConst = v * E / ((1 + v) * (1 - 2 * v));
	muConst = E / (2 * (1 + v));
	uploadTemplateLameMatrix(getKeLam72(), getKeMu72(), lamConst, muConst);
}

void homo::Homogenization::elasticMatrix(float C[6][6])
{
	double c[6][6];
	elasticMatrix(c);
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < 6; j++) {
			C[i][j] = c[i][j];
		}
	}
}

void homo::Homogenization::update(float* rho, int pitchT)
{
	if (rho != nullptr) { grid->update(rho, pitchT); }
	//grid->enforceCellSymmetry(grid->rho_g, sym, true);
	grid->pad_cell_data(grid->rho_g);
	mg_->updateStencils();
}

double homo::Homogenization::elasticMatrix(float* rho, int i, int j)
{
	printf("\033[31m No Support ! \033[0m\n");
	exit(-1);
	return 0;
#if 0
	auto* rho_g_bk = grid->rho_g;
	grid->rho_g = rho;
	update();
	double c = elasticMatrix(i, j);
	grid->rho_g = rho_g_bk;
	return c;
#endif
}

void homo::Homogenization::Sensitivity(float* rho, int i, int j, float* sens)
{
	printf("\033[31m No Support ! \033[0m\n");
	exit(-1);
#if 0
	float* rho_g_bk = grid->rho_g;
	grid->rho_g = rho;
	grid->sensitivity(i, j, sens);
	grid->rho_g = rho_g_bk;
	return;
#endif
}

std::shared_ptr<homo::Grid> homo::Homogenization::getGrid(void)
{
	return grid;
}

std::ofstream homo::Homogenization::logger()
{
	std::ofstream ofs(getPath("log"), std::ios_base::app);
	auto cur = std::chrono::system_clock::now();
	std::time_t curtime = std::chrono::system_clock::to_time_t(cur);

	char buf[1000];
#ifdef _WIN32
	//ctime_s(buf, sizeof(buf), &curtime);
	tm times;
	auto err = localtime_s(&times, &curtime);
	//sprintf_s(buf, "[  %02d:%02d:%02d]", times.tm_hour, times.tm_min, times.tm_sec);
	strftime(buf, sizeof(buf), "[%d/%m/%Y %H:%M:%S]", &times);
	ofs << buf;
#elif defined(__linux__)
	auto tt = localtime(&curtime);
	strftime(buf, sizeof(buf), "[%d/%m/%Y %H:%M:%S]", tt);
	ofs << buf;
#endif
	return ofs;
}

void homo::Homogenization::ConfigDiagPrecondition(float strength)
{
	diag_strength = strength;
	grid->diagPrecondition(diag_strength);
}

std::string homo::Homogenization::getName(void)
{
	char buf[1000];
	sprintf_s(buf, "H%zu", getUid());
	return buf;
}


