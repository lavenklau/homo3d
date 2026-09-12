// homo3d entry point.

#include <iostream>
#include "cmdline.h"

extern void runInstance(cfg::HomoConfig);

namespace homo {
extern void freeMem(void);
}

namespace culib {
extern void freeTempPool(void);
}

int main(int argc, char** argv) {
	cfg::HomoConfig config;
	config.parse(argc, argv);

	std::cout << "Hello World!\n";
	try {
		runInstance(config);
	} catch (std::runtime_error e) {
		std::cout << "\033[31m"
				  << "Exception occurred: " << std::endl
				  << e.what() << std::endl
				  << ", aborting..."
				  << "\033[0m" << std::endl;
		exit(-1);
	} catch (...) {
		std::cout << "\033[31m"
				  << "Unhandled Exception occurred, aborting..."
				  << "\033[0m" << std::endl;
		exit(-1);
	}

	homo::freeMem();
	culib::freeTempPool();
	return 0;
}
