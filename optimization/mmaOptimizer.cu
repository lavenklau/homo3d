#include "mmaOptimizer.h"
#include "cuda_runtime.h"
#include "culib/gpuVector.cuh"
#include "culib/lib.cuh"
#include "mmaOpt.h"
#include <vector>

using namespace culib;

namespace homo {

	MMAOptimizer::~MMAOptimizer()
	{

	}


	void MMAOptimizer::init(void)
	{
		context = std::make_unique<MMAContext>();
		context->xvar.resize(n);
		context->xmin.resize(n);
		context->xmax.resize(n);
		context->xold1.resize(n);
		context->xold2.resize(n);
		context->low.resize(n);
		context->upp.resize(n);
		context->df0dx.resize(n);
		context->gval.resize(m);
		context->acd.resize(3 * m);
		std::vector<double> acdhost(3 * m);
		for (int i = 0; i < m; i++) {
			acdhost[i] = a;
			acdhost[i + m] = c;
			acdhost[i + 2 * m] = d;
		}
		context->acd.set(acdhost.data());

		void* dgdxdata;
		cudaMallocPitch(&dgdxdata, &context->gpitch, n * sizeof(double), m);
		context->dgdx.move((double*)dgdxdata, context->gpitch / sizeof(double) * m);
	}

	void MMAOptimizer::setBound(float* xmin, float* xmax)
	{
		type_cast(context->xmin.data(), xmin, n);
		type_cast(context->xmax.data(), xmax, n);
	}

	void MMAOptimizer::setBound(float xmin, float xmax)
	{
		context->xmin.set(xmin);
		context->xmax.set(xmax);
	}

	void MMAOptimizer::free(void)
	{
		
	}

	void MMAOptimizer::update(int itn, float* x, float* dfdx, float* gval, float** dgdx)
	{
		type_cast(context->xvar.data(), x, n);
		type_cast(context->df0dx.data(), dfdx, n);
		for (int i = 0; i < m; i++) {
			type_cast(context->dgdx.data() + i * (context->gpitch / sizeof(double)), dgdx[i], n);
		}
		type_cast(context->gval.data(), gval, m);
		double f0val = 1; // doesn't matter
		double move = 1;
		// optimize
		cudaPitchedPtr dgdxptr{ (void*)context->dgdx.data(), context->gpitch, n * sizeof(double), m };
		mmasubDevice(m, n, itn, context->xvar.data(), context->xmin.data(), context->xmax.data(), context->xold1.data(), context->xold2.data(),
			f0val, context->df0dx.data(), context->gval.data(), dgdxptr, context->low.data(), context->upp.data(),
			a0, context->acd.data(), context->acd.data() + m, context->acd.data() + 2 * m, move);
		type_cast(x, context->xvar.data(), n);
	}


}


