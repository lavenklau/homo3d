#pragma once

#include "grid.h"
#include <vector>

namespace homo {

	struct MGConfig{
		int reso[3];
		bool enableManagedMem = true;
		std::string namePrefix;
	};

struct MG{
	// fine to coarse
	std::vector<std::shared_ptr<Grid>> grids;

	MGConfig mgConfig;

	std::shared_ptr<Grid> getRootGrid(void);

	void build(MGConfig config);

	void v_cycle(float w_SOR = 1.f, int pre = 1, int post = 1);

	void v_cycle(std::vector<int> gstimes_);

	void v_cycle_verbose(float w_SOR = 1.f, int pre = 1, int post = 1);

	void reset_displacement(void);

	double solveEquation(double tol = 1e-2, bool with_guess = true);

	double pcg(void);

	void updateStencils(void);

	void test_v_cycle(void);

	void test_diag_precondition(void);

	void v_cycle_profile(void);

	void test_sor(void);

	void test_pcg(void);
};



}
