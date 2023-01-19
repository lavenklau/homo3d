#include "cuda_runtime.h"
#include <type_traits>
#include "cub/util_ptx.cuh"

template<typename T1, typename T2, std::enable_if_t<sizeof(T1) == 4 && sizeof(T2) == 4, int> = 0>
__device__ inline char4 vadd4(T1 i1, T2 i2) {
	unsigned int res = __vadd4(reinterpret_cast<unsigned int&>(i1), reinterpret_cast<unsigned int&>(i2));
	char4 ret = reinterpret_cast<char4&>(res);
	return ret;
}

__device__ inline char4 vsub4(char4 i1, char4 i2) {
	unsigned int res = __vsub4(reinterpret_cast<unsigned int&>(i1), reinterpret_cast<unsigned int&>(i2));
	char4 ret = reinterpret_cast<char4&>(res);
	return ret;
}

__device__ inline unsigned int pack(char3 c3) {
	char4 c4{ c3.x,c3.y,c3.z,0 };
	unsigned int ui = reinterpret_cast<unsigned int&>(c4);
	return  ui;
}

__device__ inline unsigned int pack(short x, short y) {
	unsigned int ui = 0;
	cub::BFI(ui, ui, x, 0, 16);
	cub::BFI(ui, ui, y, 16, 16);
	return ui;
}

