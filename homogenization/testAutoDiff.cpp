#include "homoExpression.h"
#include <iostream>
#include "AutoDiff/TensorExpression.h"

using namespace homo;

void testAutoDiff(void) {
	var_exp_t<> v0(2), v1(1);

	//constexpr bool v = is_linearterm_v<var_exp_t<>>;
	//auto p = v0 + v1;
	auto p = (v0 + v1 + 3.) * (v0 + v1) * v1 / (v0 - v1);
	//p.terms;
	p.eval();
	p.backward(1);
	std::cout << "p = " << p.value() << std::endl;
	std::cout << "d_v0 = " << v0.diff() << ", d_v1 = " << v1.diff() << std::endl;

#if 0
	// define density expression
	var_tsexp_t<> rho(128, 128, 128);

	// define homogenization 
	Homogenization hom(128, 128, 128);

	// get elastic tensor expression
	elastic_tensor_t<float, decltype(rho)> Ch(hom, rho);

	// define objective function as bulk modulus
	auto objective = -(Ch(0, 0) + Ch(1, 1) + Ch(2, 2)) * 3.;

	// evaluate objective
	objective.eval();

	// compute differential
	objective.backward(1);
#elif 0
	var_tsexp_t<> x(Tensor<float>::range(0, 20, 500));
	auto fx = (x + 1).pow(2) - 2;
	float* pdata = fx.value().view().data();
	fx.eval();
	Tensor<float> ones(fx.value().getDim());
	ones.reset(1);
	fx.backward(ones);
	data2matrix_h("x", x.value().data(), 500);
	data2matrix_h("fx", fx.value().data(), 500);
	data2matrix_h("df", x.diff().data(), 500);
#else
#endif
}
