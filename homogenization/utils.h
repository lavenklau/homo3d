#pragma once

#include "platform_spec.h"
#include <string>
#include <fstream>
#include <vector>

namespace homoutils {

template<typename... Args>
std::string formated(const char* frmstr, Args... args) {
	char buf[1000];
	sprintf_s(buf, frmstr, args...);
	return buf;
}

template<int modulu = 0>
struct Zp {
	int operator[](int n) {
		n -= modulu * (n / modulu);
		n += modulu;
		return n % modulu;
	}
};

template<>
struct Zp<0> {
	int _modulu;
	Zp(int m) :_modulu(m) {}
	int operator[](int n) {
		n -= _modulu * (n / _modulu);
		n += _modulu;
		return n % _modulu;
	}
};

template<int N>
inline size_t Round(size_t n) {
	size_t rmod = n % N;
	if (!rmod)
		return n;
	else {
		return (n / N + 1) * N;
	}
}

template<typename T, int N = 1>
void writeVectors(const std::string& str, const std::vector<T> (&vecs)[N]){
	std::ofstream ofs(str, std::ios::binary);
	if (!ofs) {
		printf("\033[31mopen file %s failed\033[0m\n", str.c_str());
		return;
	}
	for (int j = 0; j < vecs->size(); j++) {
		for (int i = 0; i < N; i++) {
			ofs.write((char*)&vecs[i][j], sizeof(T));
		}
	}
	return;
}

template<typename T, int N = 1>
void readVectors(const std::string& str, std::vector<T>(&vecs)[N]) {
	std::ifstream ifs(str, std::ios::binary | std::ios::ate);
	if (!ifs) {
		printf("\033[31mopen file %s failed\033[0m\n", str.c_str());
		return;
	}
	size_t fsize = ifs.tellg();
	int vsize = fsize / (N * sizeof(T));
	for (int i = 0; i < N; i++) vecs[i].resize(vsize);
	ifs.seekg(0, std::ios::beg);
	for (int j = 0; j < vsize; j++) {
		for (int i = 0; i < N; i++) {
			ifs.read((char*)&vecs[i][j], sizeof(T));
		}
	}
	return;
}
}
