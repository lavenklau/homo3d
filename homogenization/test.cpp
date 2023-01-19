#include <iostream>
#include <algorithm>
#include <tuple>
#include <type_traits>

#define ios_out std::cout  <<

template<typename T, typename S>
struct Term {
	T t;
	T eval() { return t; }
	void boo(void) {}
	static int doo(void) {}
	Term(T val) :t(val) {}
};

struct KKK : public Term<int, int> {
	using x = decltype(((Term<int, int>*)nullptr)->boo());
};

template<typename... Args>
using TermList = std::tuple<Term<Args, double>...>;


void foo(int i, int k = 1) {}

void foo(int i) {}


void test(void) {
	double val=1.5;
	double mval = -1;
	double Mval = 2;
	double cval = std::clamp(val, mval, Mval);
	ios_out "string";

	TermList<double, float, int> ts(1., 2, 3);
	double s = std::apply([](auto&... args) {
		double sm = 0;
		//sm = args + ...;
		return (args.eval() + ... + 1);
		}, ts);

	std::apply([&](auto&... args) {
		(args.boo(),...);
		}, ts);

	foo(1, 2);

	std::cout << "size termlist = " << sizeof(TermList<double, float, int>) << std::endl;
}