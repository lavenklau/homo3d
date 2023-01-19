#pragma once
#include <stdio.h>

#ifdef __linux__
template<int N, typename... Args>
void sprintf_s(char(&_Buffer)[N], Args... args) { snprintf(_Buffer, N, args...); }
#elif defined(_WIN32)
#else
#error Only Windows and Linux are supported!
#endif
