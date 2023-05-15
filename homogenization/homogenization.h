#pragma once
#include "MG.h"
#include "cmdline.h"
#include <fstream>

namespace homo {
	
	struct uid_t {
	private:
		static size_t uid;
	protected:
		static void setUid(void) { uid++; };
		uid_t(void) { setUid(); }
		size_t getUid(void) { return uid; }
	};

	struct Homogenization : public uid_t {
		Homogenization(void) = default;
		Homogenization(int xreso, int yreso, int zreso, double youngsModu = 1e6, double poissonRatio = 0.3);
		Homogenization(cfg::HomoConfig config);
		//void build(int xreso, int yreso, int zreso, double youngsModu = 1e6, double poissonRatio = 0.3);
		void build(cfg::HomoConfig config);
		
		void elasticMatrix(double C[6][6]);

		void elasticMatrix(float C[6][6]);

		void update(float* rho = nullptr, int pitchT = -1);

		double elasticMatrix(int i, int j);

		double elasticMatrix(float* rho, int i, int j);

		void Sensitivity(int i, int j, float* sens);

		void Sensitivity(float* rho, int i, int j, float* sens);

		void Sensitivity(float dC[6][6], float* sens, int pitchT, bool lexiOrder = false);

		void Sensitivity_Without_transfer(float dC[6][6], float* sens, int pitchT, bool lexiOrder = false);

		std::shared_ptr<Grid> getGrid(void);

		std::ofstream logger();
		//std::string getPath(const std::string& str);

		void ConfigDiagPrecondition(float strength);

		std::shared_ptr<Grid> grid;
		std::unique_ptr<MG> mg_;
		float power_penal = 1;
		float diag_strength = 1e6;
		//double youngs_modulus = 1e6;
		//double poisson_ratio = 0.3;
		//bool enable_managed_memory = true;
		cfg::HomoConfig config;
		float lamConst;
		float muConst;
		SymmetryType sym;

	private:
		std::string getName(void);
	};
}
