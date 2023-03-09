#pragma once

#include "platform_spec.h"
#include <memory>
#include <vector>
#include <stdint.h>
#include <type_traits>
#include <initializer_list>
#include "gmem/DeviceBuffer.h"
//#include "homogenization.h"

#ifdef __CUDACC__
#include "thrust/tuple.h"
#include "culib/lib.cuh"
#include <cuda/std/tuple>
//using namespace cuda;
#else
#include <tuple>
#endif

#define AUTODIFF_WITH_MATLAB

#ifndef __CUDACC__
//#define __device__
#define __host_device_func 
#define __device_func
#define _disable_opt 
#else
#define __host_device_func __host__ __device__
#define __device_func __device__
#endif

#ifdef AUTODIFF_WITH_MATLAB
#include <string>
void data2matrix_g(const std::string& mtname, float* pdata, int m, int n = 1, int pitch = -1);
void data2matrix_g(const std::string& mtname, double* pdata, int m, int n = 1, int pitch = -1);
void data2matrix_h(const std::string& mtname, float* pdata, int m, int n = 1, int pitch = -1);
void data2matrix_h(const std::string& mtname, double* pdata, int m, int n = 1, int pitch = -1);
#endif

namespace homo {

#ifdef __CUDACC__
	namespace homostd = cuda::std;
#else
	namespace homostd = std;
#endif

	// declare expressions
	template<typename opExp, typename Scalar> struct LinearTerm;
	template<typename Scalar, typename... opExps_t> struct linear_exp_t;
	template<typename opExp1_t, typename opExp2_t, typename Scalar> struct mul_exp_t;
	template<typename opExp1_t, typename opExp2_t, typename Scalar> struct div_exp_t;
	template<typename opExp_t, typename Scalar> struct pow_exp_t;
	template<typename opExp_t, typename Scalar> struct ln_exp_t;
	template<typename opExp_t, typename Scalar> struct exp_exp_t;
	template<typename Scalar = double> struct scalar_exp_t;
	template<typename subVar = void, typename Scalar = double> struct var_exp_t;
	template<typename Scalar> struct rvar_exp_t;

	namespace details {

		template<typename Arg>
		struct exp_trait { using type = void; };
		template<typename Scalar, typename... Args>
		struct exp_trait<linear_exp_t<Scalar, Args...>> { using type = std::tuple<Args...>; };
		template<typename Scalar, typename opExp>
		struct exp_trait<LinearTerm<opExp, Scalar>> { using type = opExp; };

		template<typename Arg>
		using exp_trait_t = typename exp_trait<Arg>::type;

		template<typename Arg> struct is_linearterm { static constexpr bool value = false; };
		template<typename Arg> struct is_linearexp { static constexpr bool value = false; };
		template<typename... Args>
		struct is_linearterm<LinearTerm<Args...>> { static constexpr bool value = true; };
		template<typename... Args>
		struct is_linearexp<linear_exp_t<Args...>> { static constexpr bool value = true; };
		template<typename Arg> constexpr bool is_linearterm_v = is_linearterm<Arg>::value;
		template<typename Arg> constexpr bool is_linearexp_v = is_linearexp<Arg>::value;
		template<typename Arg> struct LinT {};
		template<typename Exp, typename T> struct LinT<LinearTerm<Exp, T>> { using type = T; };
		template<typename T, typename... Exps> struct LinT<linear_exp_t<T, Exps...>> { using type = T; };

		template<typename Arg> struct is_var { static constexpr bool value = false; };
		template<typename Arg> struct is_rvar { static constexpr bool value = false; };
		template<typename... Args> struct is_var<var_exp_t<Args...>> {
			static constexpr bool value = true;
		};
		template<typename... Args> struct is_rvar<rvar_exp_t<Args...>> {
			static constexpr bool value = true;
		};
		template<typename Arg> constexpr bool is_var_v = is_var<Arg>::value;
		template<typename Arg> constexpr bool is_rvar_v = is_rvar<Arg>::value;
		template<typename Arg> using is_var_i = std::enable_if_t<is_var<Arg>::value, int>;
		template<typename Arg> using is_rvar_i = std::enable_if_t<is_rvar<Arg>::value, int>;


		template<typename T1, typename T2 = void>
		struct res_type {
			using type = decltype(T1{} *T2{});
		};

		template<typename T1, typename T2>
		using res_t = typename res_type<T1, T2>::type;

		template<typename... Args> using tuple = homostd::tuple<Args...>;

		template<typename T, int N>
		struct type_repeater {
			using type = decltype(homostd::tuple_cat(std::tuple<T>(), typename type_repeater<T, N - 1>::type()));
		};
		template<typename T> struct type_repeater<T, 0> { using type = decltype(tuple<>()); };

		//template<typename Exp> struct is_zero { static constexpr bool value = false; };
		//template<typename Scalar> struct is_zero<zero_exp_t<Scalar>> { static constexpr bool value = true; };
		//template<typename Exp> constexpr bool is_zero_v = is_zero<Exp>::value;

		struct exp_base_t {/* int a;*/ };

		template<typename ExpT> struct is_exp { static constexpr bool value = std::is_convertible_v<ExpT, exp_base_t>; };
		template<typename ExpT> constexpr bool is_exp_v = is_exp<ExpT>::value;

		template<typename Exp, typename T> using if_exp_t = std::enable_if_t<is_exp_v<Exp>, T>;

		template<typename opExp, typename Scalar,
			if_exp_t<opExp, int> = 0,
			std::enable_if_t<!is_linearterm_v<opExp> && !is_linearexp_v<opExp>, int> = 0 >
		__host_device_func auto make_linear(opExp&& op, Scalar w/* = 1*/) {
			using opExp_t = std::decay_t<opExp>;
			return linear_exp_t<Scalar, opExp_t>(homostd::make_tuple(LinearTerm<opExp_t, Scalar>(op, w)));
		}

		template<typename opExp,
			if_exp_t<opExp, int> = 0,
			std::enable_if_t<!is_linearterm_v<opExp> && !is_linearexp_v<opExp>, int> = 0 >
		__host_device_func auto make_linear(opExp&& op) {
			using opExp_t = std::decay_t<opExp>;
			using Scalar = typename opExp_t::ScalarType;
			return linear_exp_t<Scalar, opExp_t>(homostd::make_tuple(LinearTerm<opExp_t, Scalar>(op)));
		}

		template<typename opExp, typename Scalar = typename LinT<opExp>::type,
			if_exp_t<opExp, int> = 0,
			std::enable_if_t<is_linearterm_v<opExp>, int> = 0>
		__host_device_func auto make_linear(opExp&& op) {
			using opExp_t = std::decay_t<opExp>;
			return linear_exp_t<Scalar, opExp_t>(homostd::make_tuple(op.template cast<double>()));
		}

		template<typename opExp, typename Scalar = typename LinT<opExp>::type,
			if_exp_t<opExp, int> = 0,
			std::enable_if_t<is_linearexp_v<opExp>, int> = 0>
		__host_device_func auto make_linear(opExp&& op) {
			using opExp_t = std::decay_t<opExp>;
			return op.template cast<Scalar>();
		}

		template<typename... Terms>
		__host_device_func auto make_linear(const tuple<Terms...>& terms) {
			return linear_exp_t<typename LinT<std::decay_t<decltype(homostd::get<0>(terms))>>::type, exp_trait_t<Terms>...>(terms);
		}

		template<typename Scalar>
		__host_device_func Scalar pow_(Scalar v, Scalar k) {
			if (std::is_same_v<Scalar, double>) {
				return pow(v, k);
			}
			else {
				return powf(v, k);
			}
		}

		template<typename Scalar>
		__host_device_func Scalar exp_(Scalar v) {
			if (std::is_same_v<Scalar, double>) {
				return exp(v);
			}
			else {
				return expf(v);
			}
		}

		template<typename Scalar>
		__host_device_func Scalar log_(Scalar v) {
			if (std::is_same_v<Scalar, double>) {
				return log(v);
			} else {
				return logf(v);
			}
		}

		template<typename Scalar>
		__host_device_func Scalar sqrt_(Scalar v) {
			if (std::is_same_v<Scalar, double>) {
				return sqrt(v);
			} else {
				return sqrtf(v);
			}
		}
	}
	using namespace details;

	template<bool HasFlag>
	struct ExpFlagBase {
		__host_device_func bool is_expired(void) { return false; }
	};
	template<>
	struct ExpFlagBase<true> {
		enum  bitflag : int {
			expired = 1
		};
		int flag_;
		__host_device_func bool is_expired(void) { return flag_ & bitflag::expired; }
	};

	template<typename Scalar>
	struct zero_exp_t {
		__host_device_func Scalar eval() { return 0; }
		__host_device_func void backward(Scalar lastdiff) { return; }
	};


	template<typename subExp_t, typename Scalar = double>
	struct exp_data_t {
		Scalar diff_;
		Scalar value_;
		__host_device_func exp_data_t(Scalar val, Scalar dif) : value_(val), diff_(dif) {}
		__host_device_func exp_data_t(void) = default;
		__host_device_func Scalar value(void) const { return value_; }
		__host_device_func Scalar diff() const { return diff_; }
		__host_device_func Scalar& rvalue(void) { return value_; }
		__host_device_func Scalar& rdiff(void) { return diff_; }
		__host_device_func const Scalar& rvalue(void) const { return value_; }
		__host_device_func const Scalar& rdiff(void) const{ return diff_; }
	};

	template<typename subExp_t, typename Scalar = double>
	struct exp_method_t {
		using ScalarType = Scalar;

		// for variable reference 
		template<typename subExp = subExp_t, is_var_i<subExp> = 0>
		__host_device_func auto refer(void) {
			return static_cast<subExp&>(*this).ref();
		}
		template<typename subExp = subExp_t,
			std::enable_if_t<!is_var_v<subExp>, int> = 0>
		__host_device_func auto refer(void) const{
			return static_cast<const subExp&>(*this);
		}

		// <* begin operators
		template<typename opExp2, if_exp_t<opExp2, int> = 0>
		__host_device_func auto operator+(opExp2&& op2) {
			using opExp2_t = std::decay_t<opExp2>;
			using T = res_t<Scalar, typename opExp2_t::ScalarType>;
			return make_linear(homostd::tuple_cat(
				make_linear(static_cast<subExp_t*>(this)->template refer<>()).terms,
				make_linear(op2.template refer<>()).terms));
		}

		template<typename T2, std::enable_if_t<std::is_arithmetic_v<T2>, int> = 0>
		__host_device_func auto operator+(T2 s2) {
			using T = res_t<Scalar, T2>;
			return operator+(scalar_exp_t<T>(T(s2)));
		}

		template<typename opExp2, if_exp_t<opExp2, int> = 0>
		__host_device_func auto operator-(opExp2&& op2) {
			using opExp2_t = std::decay_t<opExp2>;
			using T = res_t<Scalar, typename opExp2_t::ScalarType>;
			auto op2_neg = make_linear(op2.template refer<>());
			op2_neg.scale(-1.);
			return make_linear(homostd::tuple_cat(
				make_linear(static_cast<subExp_t*>(this)->template refer<>()).terms,
				op2_neg.terms));
		}

		template<typename T2, std::enable_if_t<std::is_arithmetic_v<T2>, int> = 0>
		__host_device_func auto operator-(T2 s2) {
			using T = res_t<Scalar, T2>;
			return operator+(scalar_exp_t<T>(-T(s2)));
		}

		__host_device_func auto operator-(void) {
			auto op = make_linear(static_cast<subExp_t*>(this)->template refer<>());
			op.scale(-1.);
			return op;
		}

		template<typename opExp2, if_exp_t<opExp2, int> = 0>
		__host_device_func auto operator*(opExp2&& op2) {
			using opExp2_t = std::decay_t<opExp2>;
			using T = res_t<Scalar, typename opExp2_t::ScalarType>;
			using E1 = std::decay_t<decltype(static_cast<subExp_t*>(this)->template refer<>())>;
			using E2 = std::decay_t<decltype(op2.template refer<>())>;
			return mul_exp_t<E1, E2, T>(
				static_cast<subExp_t*>(this)->template refer<>(),
				op2.template refer<>());
		}

		template<typename T2, typename SubExp = subExp_t,
			std::enable_if_t<std::is_arithmetic_v<T2>, int> = 0,
			std::enable_if_t<!is_linearexp_v<SubExp>, int> = 0 >
		__host_device_func auto operator*(T2 s2) {
			using T = res_t<Scalar, T2>;
			using E1 = std::decay_t<decltype(static_cast<subExp_t*>(this)->template refer<>())>;
			return linear_exp_t<T, E1>(LinearTerm<E1,T>(
				static_cast<subExp_t*>(this)->template refer<>(), T(s2)));
		}

		template<typename T2, typename SubExp = subExp_t,
			std::enable_if_t<std::is_arithmetic_v<T2>, int> = 0,
			std::enable_if_t<is_linearexp_v<SubExp>, int> = 0,
			std::enable_if_t<!std::is_same_v<res_t<Scalar, T2>, Scalar>, int> = 0>
		__host_device_func auto operator*(T2 s2) {
			using T = res_t<Scalar, T2>;
			auto me = static_cast<SubExp*>(this)->template cast<T>();
			me.scale(T(s2));
			return me;
		}

		template<typename T2, typename SubExp = subExp_t,
			std::enable_if_t<std::is_arithmetic_v<T2>, int> = 0,
			std::enable_if_t<is_linearexp_v<SubExp>, int> = 0,
			std::enable_if_t<std::is_same_v<res_t<Scalar, T2>, Scalar>, int> = 0>
		__host_device_func auto operator*(T2 s2) {
			using T = res_t<Scalar, T2>;
			auto me = *static_cast<SubExp*>(this);
			me.scale(T(s2));
			return me;
		}

		template<typename opExp2, if_exp_t<opExp2, int> = 0>
		__host_device_func auto operator/(opExp2&& op2) {
			using opExp2_t = std::decay_t<opExp2>;
			using T = res_t<Scalar, typename opExp2_t::ScalarType>;
			using E1 = std::decay_t<decltype(static_cast<subExp_t*>(this)->template refer<>())>;
			using E2 = std::decay_t<decltype(op2.template refer<>())>;
			return div_exp_t<E1, E2, T>(
				static_cast<subExp_t*>(this)->template refer<>(),
				op2.template refer<>());
		}

		template<typename T2, std::enable_if_t<std::is_arithmetic_v<T2>, int> = 0>
		__host_device_func auto operator/(T2 s2) {
			using T = res_t<Scalar, T2>;
			return operator*(T(1) / s2);
		}
		// *> end operators

		template<typename T2, std::enable_if_t<std::is_arithmetic_v<T2>, int> = 0 >
		__host_device_func auto pow(T2 s) {
			using T = res_t<Scalar, T2>;
			return pow_exp_t<subExp_t, T>(static_cast<subExp_t*>(this)->template refer<>(), T(s));
		}

		__host_device_func auto log(void) {
			return ln_exp_t<subExp_t, Scalar>(static_cast<subExp_t*>(this)->template refer<>());
		}

		__host_device_func auto exp(void) {
			return exp_exp_t<subExp_t, Scalar>(static_cast<subExp_t*>(this)->template refer<>());
		}
	};

	template<typename subExp_t, typename Scalar = double, bool HasStorage = true, bool HasFlag = false>
	struct exp_t : public exp_base_t, public exp_data_t<subExp_t, Scalar>, public exp_method_t<subExp_t, Scalar>
	{
		exp_t(void) = default;
		//template<typename Scalar>
		using Data = exp_data_t<subExp_t, Scalar>;
		//exp_t(void) :Data(0, 0) {}
		__host_device_func exp_t(Scalar v, Scalar dif) : Data(v, dif) {}
		__host_device_func exp_t(const exp_t&) = default;

		__host_device_func Scalar eval() {
			Scalar val = static_cast<subExp_t&>(*this).eval_imp();
			// clear diff
			Data::diff_ = 0;
			Data::value_ = val;
			return val;
		}
		__host_device_func void backward(Scalar lastDiff) {
			Data::rdiff() += lastDiff;
			static_cast<subExp_t&>(*this).backward_imp(lastDiff);
		}
	};

	template<typename subExp_t, typename Scalar, bool HasFlag >
	struct exp_t<subExp_t, Scalar, false, HasFlag>
		: public exp_base_t, public exp_method_t<subExp_t, Scalar>
	{
		__host_device_func exp_t(void) = default;
		__host_device_func exp_t(const exp_t&) = default;

		__host_device_func Scalar eval(void) {
			return static_cast<subExp_t*>(this)->eval();
		}
		__host_device_func void backward(Scalar lastDiff) {
			return static_cast<subExp_t*>(this)->backward(lastDiff);
		}
	};

	// *< define operators
	template<typename T1, typename opExp2, std::enable_if_t<std::is_arithmetic_v<T1>, int> = 0, if_exp_t<opExp2, int> = 0>
	__host_device_func auto operator+(T1 s1,  opExp2&& op2) {
		using opExp2_t = std::decay_t<opExp2>;
		using T = res_t<typename opExp2_t::ScalarType, T1>;
		return scalar_exp_t<T>(T(s1)) + op2;
	}

	template<typename T1, typename opExp2, std::enable_if_t<std::is_arithmetic_v<T1>, int> = 0, if_exp_t<opExp2, int> = 0>
	__host_device_func auto operator-(T1 s1,  opExp2&& op2) {
		using opExp2_t = std::decay_t<opExp2>;
		using T = res_t<typename opExp2_t::ScalarType, T1>;
		return scalar_exp_t<T>(T(s1)) - op2;
	}

	template<typename T1, typename opExp2, std::enable_if_t<std::is_arithmetic_v<T1>, int> = 0, if_exp_t<opExp2, int> = 0>
	__host_device_func auto operator*(T1 s1, opExp2&& op2) {
		using opExp2_t = std::decay_t<opExp2>;
		using T = res_t<typename opExp2_t::ScalarType, T1>;
		return scalar_exp_t<T>(T(s1)) * op2;
	}

	template<typename T1, typename opExp2, std::enable_if_t<std::is_arithmetic_v<T1>, int> = 0, if_exp_t<opExp2, int> = 0>
	__host_device_func auto operator/(T1 s1, opExp2&& op2) {
		using opExp2_t = std::decay_t<opExp2>;
		using T = res_t<typename opExp2_t::ScalarType, T1>;
		return scalar_exp_t<T>(T(s1)) / op2;
	}
	// *> end operators

	struct dynexp_t
		: public exp_t<dynexp_t>
	{
		virtual double eval(void) = 0;
		virtual double diff(void) = 0;
	};

	template<typename opExp, typename Scalar>
	struct LinearTerm : public exp_t<LinearTerm<opExp, Scalar>, Scalar>
	{
		using Base = exp_t<LinearTerm<opExp, Scalar>, Scalar>;
		opExp op;
		Scalar w;
		__host_device_func LinearTerm(const opExp& op_, Scalar w_ = Scalar{ 1 }) :op(op_), w(w_) {}
		__host_device_func LinearTerm(const LinearTerm&) = default;
		template<typename T, std::enable_if_t<std::is_same_v<T, Scalar>, int> = 0>
		__host_device_func LinearTerm<opExp, T> cast() const {
			return *this;
		}
		template<typename T, std::enable_if_t<!std::is_same_v<T, Scalar>, int> = 0>
		__host_device_func LinearTerm<opExp, T> cast() const {
			return LinearTerm<opExp, T>(op, T(w));
		}
		__host_device_func void scale(Scalar s) { w *= s; }
		__host_device_func Scalar eval_imp(void) {
			return w * op.eval();
		}
		__host_device_func void backward_imp(Scalar lastdiff) {
			op.backward(lastdiff * w);
		}
	};

	template<typename Scalar, typename... opExps_t>
	struct linear_exp_t :
		public exp_t<linear_exp_t<Scalar, opExps_t...>, Scalar>
	{
		using Base = exp_t<linear_exp_t<Scalar, opExps_t...>, Scalar>;
		tuple<LinearTerm<opExps_t, Scalar>...> terms;

		__host_device_func linear_exp_t(const decltype(terms)& terms_) : terms(terms_) {}

		template<typename... opExp_t>
		__host_device_func linear_exp_t(const LinearTerm<opExp_t, Scalar>&... terms_) 
			: terms(homostd::make_tuple(terms_...))
		{ }

		template<typename T>
		__host_device_func linear_exp_t<T, opExps_t...> cast(void) const{
			return homostd::apply([](auto&& ... args) {
				return linear_exp_t<T, opExps_t...>(homostd::make_tuple(args.template cast<T>()...)); 
				},
				std::move(terms));
		}

		__host_device_func void scale(Scalar s) {
			homostd::apply([&](auto&... args) {
				(args.scale(s),...);
				}, terms);
		}

		template<typename L1, typename L2,
			std::enable_if_t<is_linearexp_v<L1>, int> = 0,
			std::enable_if_t<is_linearexp_v<L2>, int> = 0 >
		__host_device_func linear_exp_t(const L1& l1, const L2& l2) 
			: terms(homostd::tuple_cat(l1.terms, l2.terms))
		{ }

		__host_device_func Scalar eval_imp(void) {
			Scalar s = homostd::apply([](auto&...args) {
				Scalar sum = 0;
				sum = (args.eval() + ...);
				return sum; }, terms);
			return s;
		}

		__host_device_func void backward_imp(Scalar lastdiff) {
			homostd::apply([=](auto&... args) {
				(args.backward(lastdiff), ...);
				}, terms);
		}
	};

	template<template<typename, typename, typename> class subExp_t, typename opExp1_t, typename opExp2_t, typename Scalar>
	struct  binary_exp_t
		:public exp_t<subExp_t<opExp1_t, opExp2_t, Scalar>, Scalar>
	{
		opExp1_t op1;
		opExp2_t op2;
		__host_device_func binary_exp_t(const opExp1_t& op1_, const opExp2_t& op2_) :op1(op1_), op2(op2_) {}
	};

	template<typename opExp1_t, typename opExp2_t, typename Scalar>
	struct mul_exp_t
		: public binary_exp_t<mul_exp_t, opExp1_t, opExp2_t, Scalar>
	{
		using Base = binary_exp_t<mul_exp_t, opExp1_t, opExp2_t, Scalar>;
		__host_device_func mul_exp_t(const opExp1_t& op1_, const opExp2_t& op2_)
			: Base(op1_, op2_) {}
		__host_device_func Scalar eval_imp(void) { return Base::op1.eval() * Base::op2.eval(); }
		__host_device_func void backward_imp(Scalar lastdiff) {
			Base::op1.backward(lastdiff * Base::op2.value());
			Base::op2.backward(lastdiff * Base::op1.value());
		}
	};

	template<typename opExp1_t, typename opExp2_t, typename Scalar>
	struct div_exp_t
		: public binary_exp_t<div_exp_t, opExp1_t, opExp2_t, Scalar>
	{
		using Base = binary_exp_t<div_exp_t, opExp1_t, opExp2_t, Scalar>;
		__host_device_func div_exp_t(const opExp1_t& op1, const opExp2_t& op2) : Base(op1, op2) {}
		__host_device_func Scalar eval_imp(void) { return Base::op1.eval() / Base::op2.eval(); }
		__host_device_func void backward_imp(Scalar lastdiff) {
			Base::op1.backward(lastdiff / Base::op2.value());
			Base::op2.backward(-lastdiff * Base::op1.value() / (Base::op2.value() * Base::op2.value()));
		}
	};

	template<template<typename, typename> class subExp_t, typename opExp_t, typename Scalar>
	struct unary_exp_t
		: public exp_t<subExp_t<opExp_t, Scalar>, Scalar>
	{
		opExp_t op;
		__host_device_func unary_exp_t(const opExp_t& op_) :op(op_) {}
	};

	template<typename opExp_t, typename Scalar>
	struct pow_exp_t
		: public unary_exp_t<pow_exp_t, opExp_t, Scalar> {
		Scalar expo;
		using Base = unary_exp_t<pow_exp_t, opExp_t, Scalar>;
		__host_device_func pow_exp_t(const opExp_t& op, Scalar expo_)
			: Base(op), expo(expo_) {}
		__host_device_func Scalar eval_imp(void) { return pow_(Base::op.eval(), expo); }
		__host_device_func void backward_imp(Scalar lastdiff) {
			Base::op.backward(lastdiff * expo * pow_(Base::op.value(), expo - 1));
		}
	};

	template<typename opExp_t, typename Scalar>
	struct exp_exp_t
		: public unary_exp_t<exp_exp_t, opExp_t, Scalar> {
		using Base = unary_exp_t<exp_exp_t, opExp_t, Scalar>;
		__host_device_func exp_exp_t(const opExp_t& op) : unary_exp_t<exp_exp_t, opExp_t, Scalar>(op) {}
		__host_device_func Scalar eval_imp(void) { return exp_(Base::op.eval()); }
		__host_device_func void backward_imp(Scalar lastdiff) {
			Base::op.backward(lastdiff * Base::value());
		}
	};

	template<typename opExp_t, typename Scalar>
	struct ln_exp_t
		: public unary_exp_t<ln_exp_t, opExp_t, Scalar>
	{
		using Base = unary_exp_t<ln_exp_t, opExp_t, Scalar>;
		ln_exp_t(const opExp_t& op) : unary_exp_t<ln_exp_t, opExp_t, Scalar>(op) {}
		__host_device_func Scalar eval_imp(void) {
			return log_(Base::op.eval());
		}
		__host_device_func void backward_imp(double lastdiff) {
			Base::op.backward(lastdiff / Base::op.value());
		}
	};

	// Heaviside function
	template<typename opExp_t, typename Scalar>
	struct hvs_exp_t
		:public unary_exp_t<hvs_exp_t, opExp_t, Scalar>
	{
		using Base = unary_exp_t<hvs_exp_t, opExp_t, Scalar>;
		Scalar eta, eps;
		__host_device_func hvs_exp_t(const opExp_t& op, Scalar eta_ = 10, Scalar eps_ = 1e-3)
			: Base(op),
			eta(eta_), eps(eps_)
		{}

		__host_device_func Scalar eval_imp(void) {
			Scalar val = Base::op.eval();
			if (val < -eta) {
				return eps;
			}
			else if (val < eta) {
				return 0.75 * (val / eta - pow(val, 3) / (3 * pow(eta, 3))) + 0.5;
			}
			else {
				return 1;
			}
		}

		__host_device_func void backward_imp(Scalar lastdiff) {
			Scalar val = Base::op.value();
			if (val < -eta || val > eta) {
				Base::op.backward(0);
			}
			else if (val < eta) {
				Base::op.backward(1. / eta - pow(val, 2) / pow(eta, 3));
			}
		}
	};

	// sigmoid function
	template<typename opExp_t, typename Scalar>
	struct sgm_exp_t
		: public unary_exp_t<sgm_exp_t, opExp_t, Scalar>
	{
		using Base = unary_exp_t<sgm_exp_t, opExp_t, Scalar>;
		Scalar s = 1;
		__host_device_func sgm_exp_t(const opExp_t& op, Scalar s_ = 1)
			: Base(op), s(s_)
		{ }

		__host_device_func Scalar eval_imp(void) {
			return 1. / (1 + exp(-s * Base::op.eval()));
		}

		__host_device_func void backward_imp(Scalar lastdiff) {
			Base::op.backward(lastdiff * s * Base::value() * (1 - Base::value()));
		}
	};


	template<typename subOp, typename Scalar>
	struct ReduceBase {
		__host_device_func double reduce(const std::vector<Scalar>& values) {
			static_cast<subOp*>(this)->reduce(values);
		}
		__host_device_func void grad(std::vector<Scalar>& grads) {
			static_cast<subOp*>(this)->grad(grads);
		}
	};

	template<typename Scalar>
	struct LinearReduce
		: public ReduceBase<LinearReduce<Scalar>, Scalar>
	{
		std::vector<Scalar> weights;
		__host_device_func LinearReduce(const std::vector<Scalar>& weights_) :weights(weights_) {}
		__host_device_func LinearReduce(int N, Scalar w = 1) : weights(N, w) {}
		__host_device_func LinearReduce(void) = default;
		__host_device_func Scalar reduce(const std::vector<Scalar>& values_) {
			Scalar sum = 0;
			for (int i = 0; i < values_.size(); i++) {
				sum += values_[i] * weights[i];
			}
		}
		__host_device_func void grad(std::vector<Scalar>& grads_) {
			grads_.resize(weights.size());
		}
	};

	template<typename subExp_t, typename reduceOp, typename Scalar>
	struct reduce_exp_t
		: public exp_t<reduce_exp_t<subExp_t, reduceOp, Scalar>, Scalar>
	{
		std::vector<std::shared_ptr<dynexp_t>> exps;
		reduceOp op;
		__host_device_func reduce_exp_t(const reduceOp& op_) : op(op_) {}
		__host_device_func Scalar eval_imp(void) {
			std::vector<Scalar> values;
			for (int i = 0; i < exps.size(); i++) {
				values.emplace_back(exps[i]->eval());
			}
			return op.reduce(values);
		}

		__host_device_func void backward_imp(Scalar lastdiff) {
			std::vector<Scalar> grads;
			op.grad(grads);
			for (int i = 0; i < exps.size(); i++) {
				exps[i]->backward(lastdiff * grads[i]);
			}
		}
	};

	template<typename Scalar>
	struct SoftMaxReduce
		: public ReduceBase<SoftMaxReduce<Scalar>, Scalar>
	{
		std::vector<double> weights;
		std::vector<double> values;
		double alpha;
		double maxValue = 0;
		__host_device_func SoftMaxReduce(double alpha_) :alpha(alpha_) {}
		__host_device_func SoftMaxReduce(SoftMaxReduce&& red) = default;
		__host_device_func double reduce(const std::vector<double>& values_) {
			values = values_;
			weights.resize(values_.size(), 0);
			double sum = 0;
			for (int i = 0; i < values_.size(); i++) {
				weights[i] = exp(alpha * values_[i]);
				sum += weights[i];
			}
			maxValue = 0;
			for (int i = 0; i < values_.size(); i++) {
				weights[i] /= sum;
				maxValue += weights[i] * values_[i];
			}
			return maxValue;
		}

		__host_device_func void grad(std::vector<double>& grads) {
			for (int i = 0; i < values.size(); i++) {
				grads[i] = weights[i] * (1 + alpha * (values[i] - maxValue));
			}
		}
	};

	template<typename Scalar>
	struct linear_dynexp_t
		: public reduce_exp_t<linear_dynexp_t<Scalar>, LinearReduce<Scalar>, Scalar>
	{
		__host_device_func linear_dynexp_t() : reduce_exp_t<linear_dynexp_t<Scalar>, LinearReduce<Scalar>, Scalar>(LinearReduce<Scalar>()) {}
	};

	template<typename Scalar>
	struct softmax_dynexp_t
		: public reduce_exp_t<softmax_dynexp_t<Scalar>, SoftMaxReduce<Scalar>, Scalar>
	{
		__host_device_func softmax_dynexp_t(double alpha) :reduce_exp_t<softmax_dynexp_t<Scalar>, SoftMaxReduce<Scalar>, Scalar>(SoftMaxReduce<Scalar>(alpha)) {}
	};

	template<typename Scalar /*= double*/>
	struct scalar_exp_t
		:public exp_t<scalar_exp_t<Scalar>, Scalar>
	{
		using Base = exp_t<scalar_exp_t<Scalar>, Scalar>;
		__host_device_func scalar_exp_t(Scalar s) : exp_t<scalar_exp_t<Scalar>, Scalar>(s, 0.) { }
		__host_device_func Scalar eval_imp(void) { return Base::value(); }
		__host_device_func void backward_imp(Scalar lastdiff) { }
	};

	template<typename subVar /*= void*/, typename Scalar /*= double*/>
	struct var_exp_t
		: //public exp_base_t
		public exp_t<var_exp_t<subVar, Scalar>, Scalar, false>
	{
		Scalar& var;
		Scalar& vardiff;
		__host_device_func var_exp_t(Scalar& var_, Scalar& vardiff_) :var(var_), vardiff(vardiff_) {
			//printf("var ctor       [v] = %p  [d] = %p\n", &var, &vardiff);
		}
		__host_device_func var_exp_t(const var_exp_t& vari2) : var(vari2.var), vardiff(vari2.vardiff) {
			//printf("var copy ctor  [v] = %p  [d] = %p\n", &vari2.var, &vari2.vardiff);
		};
		//__host_device_func var_exp_t(const var_exp_t& vari2) = default;
		__host_device_func Scalar value(void) { return var; }
		__host_device_func Scalar diff(void) { return vardiff; }
		__host_device_func Scalar& rvalue(void) { return var; }
		__host_device_func Scalar& rdiff(void) { return vardiff; }
		__host_device_func void update(Scalar var_) { var = var_; vardiff = 0; }
		__host_device_func Scalar eval(void) {
			//printf("var = %p subvar = %p\n", this, static_cast<subVar*>(this));
			static_cast<subVar*>(this)->eval_imp();
			vardiff = 0;
			//return static_cast<subVar*>(this)->eval_imp();
			return var;
		}
		__host_device_func void backward(Scalar lastdiff) {
			vardiff += lastdiff;
			static_cast<subVar*>(this)->backward_imp(lastdiff);
		}
		__host_device_func auto ref(void) {
			//printf("var ref [v] = %p  [d] = %p\n", &var, &vardiff);
			return *static_cast<subVar*>(this);
		}
	};

	template<typename Scalar>
	struct rvar_exp_t 
		: public var_exp_t<rvar_exp_t<Scalar>, Scalar>
	{
		using Base = var_exp_t<rvar_exp_t<Scalar>, Scalar>;
		__host_device_func rvar_exp_t(Scalar& var_, Scalar& vardiff_) : Base(var_, vardiff_) {}
		__host_device_func rvar_exp_t(const rvar_exp_t& r2) : Base((const Base&)r2) {};
		__host_device_func Scalar eval_imp(void) {
			//Base::rdiff() = 0;
			//printf("vardiff = %e\n", Base::diff());
			return Base::value();
		}
		__host_device_func void backward_imp(Scalar lastdiff) {
			//printf("vardiff = %e\n", Base::diff());
			//Base::rdiff() += lastdiff;
		}
	};

	template<typename Scalar>
	struct var_exp_t<void, Scalar>
		: //exp_base_t
		public exp_t<var_exp_t<void, Scalar>, Scalar, true>
	{
		using Base = exp_t<var_exp_t<void, Scalar>, Scalar, true>;
		__host_device_func var_exp_t(Scalar var_) :Base(var_, 0) {}
		__host_device_func var_exp_t(const var_exp_t& vari2) = default;
		__host_device_func void update(Scalar var_) { Base::rvalue() = var_; Base::rdiff() = 0; }
		__host_device_func void backward_imp(Scalar lastdiff) { }
		__host_device_func Scalar eval_imp(void) {
			return Base::value();
		}
		__host_device_func auto ref(void) {
			return rvar_exp_t<Scalar>(Base::rvalue(), Base::rdiff());
		}
	};

	template<typename T>
	struct DeviceAddressProxy {
		T* p_dev;
		DeviceAddressProxy(T* p) : p_dev(p) {}
		auto& operator=(T value) {
			cudaMemcpy(p_dev, &value, sizeof(T), cudaMemcpyHostToDevice);
			return *this;
		}
		auto& operator+=(T val) {
			T old;
			cudaMemcpy(&old, p_dev, sizeof(T), cudaMemcpyDeviceToHost);
			old += val;
			cudaMemcpy(p_dev, &old, sizeof(T), cudaMemcpyHostToDevice);
			return *this;
		}
		auto& operator-=(T val) {
			return operator+=(val);
		}
		operator T(void) const {
			T val;
			cudaMemcpy(&val, p_dev, sizeof(T), cudaMemcpyDeviceToHost);
			return val;
		}
	};

}
