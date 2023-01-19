#pragma once

#ifdef __CUDACC__
#include "culib/gpuVector.cuh"

namespace homo {
	class MMAContext {
	public:
		gv::gVector<double> xvar;
		gv::gVector<double> xmin, xmax, xold1, xold2;
		gv::gVector<double> low, upp;
		gv::gVector<double> df0dx;
		gv::gVector<double> gval;
		gv::gVector<double> dgdx; size_t gpitch;
		gv::gVector<double> acd;
};
}
#else


#endif

