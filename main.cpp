// homo3d.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include "cmdline.h"

extern void cuda_test(void);
extern void testAutoDiff(void);
extern void testAutoDiff_cu(void);
extern void test_MMA(void);
extern void test_OC(void);
extern void testHomogenization(cfg::HomoConfig config);
extern void test_BulkModulus(void);
extern void test_ShearModulus(void);
extern void test_NegativePoisson(void);
extern void runInstance(cfg::HomoConfig);

namespace homo {
	extern std::string setPathPrefix(const std::string& fprefix);
}

int main(int argc, char** argv)
{
	cfg::HomoConfig config;
	config.parse(argc, argv);

    std::cout << "Hello World!\n";
	cuda_test();
	//testAutoDiff();
	//testAutoDiff_cu();
	//test_MMA();
	//test_OC();
	//test_BulkModulus();
	//test_ShearModulus();
	//test_NegativePoisson();
	homo::setPathPrefix(config.outprefix);
	try {
		testHomogenization(config);
		runInstance(config);
	}
	catch (std::runtime_error e) {
		std::cout << "\033[31m" << "Exception occurred: " << std::endl << e.what() << std::endl << ", aborting..." << "\033[0m" << std::endl;
		exit(-1);
	} catch (...) {
		std::cout << "\033[31m" << "Unhandled Exception occurred, aborting..." << "\033[0m" << std::endl;
		exit(-1);
	}
}


