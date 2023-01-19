#include <Eigen/Dense>
using namespace Eigen;

#include "matlab_utils.h"
#include<iostream> 

#ifdef ENABLE_MATLAB

MatlabEngine gMatEngine;

// Matlab engine
#pragma comment(lib, "libeng.lib")
#pragma comment(lib, "libmx.lib")

bool MatlabEngine::connect(const std::string &dir, bool closeAll)
{
    if (eng)
        return true;

    if (0) {
		// 0 = success
		// -2 = error - second argument must be NULL
		// -3 = error - engOpenSingleUse failed
		int retstatus;
		if ( !(eng = engOpenSingleUse("\0", NULL, &retstatus)) ) {
            fprintf(stderr, "Can't start MATLAB engine: %d\n", retstatus);
			return false;
		}
    }
    else {
        printf_s( "Starting MATLAB engine ... " );
		//std::cout << "Starting MATLAB engine ... " << std::endl;
        if (!(eng = engOpen("\0"))) {
            fprintf(stderr, "Failed!\n");
			//std::cerr << "Failed!" << std::endl;
			return false;
		}
        printf_s( "Succeed!\n");
		//std::cout << "Succeed!" << std::endl;
	}

	engBuffer[lenEngBuffer-1] = '\0';
	engOutputBuffer(eng, engBuffer, lenEngBuffer); 

	// set current path
    engEvalString(eng, ("cd " + dir).c_str());

	//engEvalString(eng, "clear all; clc; rehash;"); 
    if (closeAll)
		engEvalString(eng, "close all;"); 

	return true;
}

void MatlabEngine::eval(const std::string &cmd)
{
    ensure(connected(), "Not connected to Matlab!");

    if (consoleOutput) 
        engOutputBuffer(eng, engBuffer, lenEngBuffer);
    else
        engOutputBuffer(eng, nullptr, 0);

    //engEvalString(eng, "dbclear all;");
    engEvalString(eng, "if ~isempty(dbstatus), warning('Clear all breakpoint, Matlab cannot be debugged from C++!'); dbclear all; end"); // debug will cause matlab to crash
    if (consoleOutput && *engBuffer) fprintf(stderr, "%s\n", engBuffer);

    int r = engEvalString(eng, cmd.c_str());
    if (r != 0) {
        if (r == 1) fprintf(stderr, "Engine session no longer running!\n");

        fprintf(stderr, "engEvalString error!\n");
    }

    if (consoleOutput && *engBuffer) {
        fprintf(stdout, "--------------\n%s\n--------------\n%s\n", cmd.c_str(), engBuffer);
	}
}


void MatlabEngine::close()
{
    printf( "Shutting down MATLAB engine ... " );
	engClose(eng);
	printf( "done!\n" );
	eng = nullptr;
}

MatlabEngine& getMatEngine() 
{	
	return gMatEngine; 
}


#endif

