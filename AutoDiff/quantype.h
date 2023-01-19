#pragma once

#ifndef __CUDACC__
#define __host_device_func 
#else
#define __host_device_func __host__ __device__
#endif

#include <stdint.h>
#include <type_traits>
#include <numeric>

template<typename HostType, int Range, int Divide,
	std::enable_if_t<std::is_integral_v<HostType>, int> = 0,
	std::enable_if_t<std::numeric_limits<HostType>::max() >= Divide, int > = 0>
struct Quantype {
	HostType bits;
	constexpr static float unitf = float(Range) / Divide;
	constexpr static double unitd = double(Range) / Divide;
	__host_device_func inline float f32(void) { return bits * unitf; }
	__host_device_func Quantype(float f) : bits(f / unitf) {
		if (f > Range) printf("\033[31m[Quantype] exception out of range\033[0m\n");
	}
	__host_device_func Quantype(double d) : bits(d / unitd) {
		if (d > Range) printf("\033[31m[Quantype] exception out of range\033[0m\n");
	}
	__host_device_func inline double f64(void) { return bits * unitd; }
	__host_device_func inline operator float() { return f32(); }
	__host_device_func inline operator double() { return f64(); }
	__host_device_func inline Quantype<HostType, Range, Divide> powf(float p) {
		return Quantype<HostType, Range, Divide>(::powf(f32(), p));
	}
	__host_device_func inline Quantype<HostType, Range, Divide> pow(double p) {
		return Quantype<HostType, Range, Divide>(::pow(f64(), p));
	}
};

