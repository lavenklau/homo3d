#pragma once
#ifndef __MMA_OPTIMIZER_H
#define __MMA_OPTIMIZER_H

#include <memory>
#include "mmaOpt.h"

namespace homo {

	class MMAContext;

	class MMAOptimizer {
	public :
		int n, m;
		double a0;
		double a, c, d;
		~MMAOptimizer(void);
		std::unique_ptr<MMAContext> context;
		MMAOptimizer(int nconstrain, int nvar,
			double a0_, double a_, double c_, double d_)
			: a0(a0_), a(a_), c(c_), d(d_), n(nvar), m(nconstrain) {
			init();
		}

		// create context
		void init(void);

		// destroy context
		void free(void);

		void setBound(float* xmin, float* xmax);
		void setBound(float xmin, float xmax);

		void update(int itn, float* x, float* dfdx, float* gval, float** dgdx);
	};

}
#include "MMAContext.h"

#endif
