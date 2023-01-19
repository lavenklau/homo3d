#include "vector_intrinsic.cuh"

//__device__ char4 vadd4(char4 i1, char4 i2) {
//	unsigned int res = __vadd4(reinterpret_cast<unsigned int&>(i1), reinterpret_cast<unsigned int&>(i2));
//	char4 ret = reinterpret_cast<char4&>(res);
//	return ret;
//}
