#pragma once
#ifndef __OCOPTIMIZER__HPP
#define __OCOPTIMIZER__HPP
#include "AutoDiff/TensorExpression.h"

namespace homo {
	struct OCOptimizer {
		int ne;
		float step_limit = 0.02;
		float damp = 0.5;
		float minRho = 0.001;
		OCOptimizer(int Ne, float min_density /*= 0.001*/, float stepLimit /*= 0.02*/, float dampExponent /*= 0.5*/)
			:ne(Ne), minRho(min_density), step_limit(stepLimit), damp(dampExponent) {}
		OCOptimizer(float min_density /*= 0.001*/, float stepLimit /*= 0.02*/, float dampExponent /*= 0.5*/)
			:minRho(min_density), step_limit(stepLimit), damp(dampExponent) {}
		void filterSens(float* sens, const float* rho, size_t pitchT, int reso[3],float radius=2);
		void filterSens(Tensor<float> sens, Tensor<float> rho, float radius = 2);
		void update(const float* sens, float* rho, float volratio);
		void update(Tensor<float> sens, Tensor<float>  rho, float volratio);
	};

}


#endif
