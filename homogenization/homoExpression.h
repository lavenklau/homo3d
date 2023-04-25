#pragma once

#include "AutoDiff/AutoDiff.h"
#include <fstream>
#include "homogenization.h"

namespace homo {

	template<typename Scalar, typename DensityExp_t> struct elastic_tensor_t;
		
	template<typename Scalar, typename el_tensor_t>
	struct el_var_t 
		: public var_exp_t<el_var_t<Scalar, el_tensor_t>, Scalar>
	{
		el_tensor_t& el;
		using Base = var_exp_t<el_var_t<Scalar, el_tensor_t>, Scalar>;
		__host_device_func el_var_t(el_tensor_t& el_, Scalar& cij, Scalar& gij) : el(el_), Base(cij, gij) {
			//printf("elvar ctor : el = %p, cij = %p, gij = %p\n", &el, &cij, &gij);
		}
		__host_device_func el_var_t(const el_var_t& el2) : el(el2.el), Base((Base&)el2) {
			//printf("elvar copy ctor [p2] = %p, [p] = %p\n", &el2.vardiff, &this->vardiff);
		}
		//__host_device_func el_var_t& operator=(const el_var_t& el2) {
		//	//printf("assign el_var_t called\n");
		//}
		__host_device_func Scalar eval_imp(void) {
			//printf("el_var = %p  el = %p\n", this, &el);
			el.eval();
			return Base::value();
		}

		__host_device_func void backward_imp(double lastdiff) {
			el.backward();
			return; 
		}
		~el_var_t(void) { el.reset(); }
	};

	struct RefCounter {
		int evalCounter = -1;
		int backwardCounter = -1;
		void reset(void) { evalCounter = -1; backwardCounter = -1; }
		__host_device_func int c_eval(void) { 
			if (evalCounter == -1)
				evalCounter = 1;
			else
				evalCounter++;
			return evalCounter; 
		}
		__host_device_func int c_backward(void) {
			if (backwardCounter == -1)
				backwardCounter = 1;
			else
				backwardCounter++;
			return backwardCounter; 
		}
		__host_device_func bool finishedBackward(void) {
			if (backwardCounter == evalCounter){
				backwardCounter = -1;
				return true;
			}
			else {
				return false;
			}
		}
	};

	template<typename Scalar, typename DensityExp_t>
	struct elastic_tensor_t 
	{
		Scalar C_[6][6];
		Scalar gradC_[6][6];
		DensityExp_t densityField;
		RefCounter counter;
		Homogenization& domain_;
		bool hold_on = false;
		bool expired = true;
		__host_device_func el_var_t<Scalar, elastic_tensor_t> operator()(int i, int j) {
			if (i < 0 || i >= 6 || j < 0 || j >= 6) {
				//throw std::runtime_error("error elastic tensor index out of range");
				printf("\031merror elastic tensor index out of range\033[0m\n");
			}
			//printf("el_proto = %p c%d%d = %p  dc%d%d = %p\n", this, i, j, &C_[i][j], i, j, &gradC_[i][j]);;
			return el_var_t<Scalar, elastic_tensor_t>(*this, C_[i][j], gradC_[i][j]);
		}
		void reset(void) { counter.reset(); }
		void holdOn(void) { hold_on = true; }
		void holdOff(void) { hold_on = false; }
		__host_device_func void eval(void) {
			counter.c_eval();
			if (expired && !hold_on) {
				for (int i = 0; i < 6; i++)
					for (int j = 0; j < 6; j++)
						gradC_[i][j] = 0;
				densityField.eval();
				domain_.update(densityField.value().data(), densityField.value().view().getPitchT());
				domain_.elasticMatrix(C_);
				expired = false;
			}
			if (hold_on) {
				for (int i = 0; i < 6; i++)
					for (int j = 0; j < 6; j++)
						gradC_[i][j] = 0;
			}
		}
		__host_device_func void backward() {
			counter.c_backward();
			if (counter.finishedBackward() && (!expired || hold_on)) {
				expired = true;
				//data2matrix_h("dC", &gradC_[0][0], 6, 6);
				counter.reset();
				// do something
				domain_.Sensitivity(gradC_, densityField.diff().data(), densityField.diff().view().getPitchT(), true);
				densityField.backward(densityField.diff());
				return;
			} else {
				return;
			}
		}

	public:
		//friend struct HomoTraits;
		elastic_tensor_t(Homogenization& dom, DensityExp_t& rho)
			: domain_(dom), densityField(rho), expired(true) {
			if (domain_.grid->getCellReso() != rho.getDim()) {
				throw std::runtime_error("density variable does not match the homogenization domain");
			}
		}

		void writeTo(const std::string& filename) {
			std::ofstream ofs(filename, std::ios::binary);
			ofs.write((const char*)(&C_[0][0]), sizeof(C_));
			ofs.close();
		}
		
		const Scalar* data(void) {
			return &C_[0][0];
		}
	};

	//struct HomoTraits {
	//	Homogenization& domain;
	//};

	// utility
	template<typename Exp> using ElasticMatrix = elastic_tensor_t<float, Exp>;
	template<typename RhoPhsyc>
	auto genCH(Homogenization& hom, RhoPhsyc rhointerp) {
		return elastic_tensor_t<float, RhoPhsyc>(hom, rhointerp);
	}
}
