#pragma once
#include "gflags/gflags.h"
#include "stdint.h"


DECLARE_int32(reso);
DECLARE_string(obj);
DECLARE_string(init);
DECLARE_string(sym);
DECLARE_double(vol);
DECLARE_double(E);
DECLARE_double(mu);
DECLARE_string(prefix);
DECLARE_int32(logrho);
DECLARE_int32(logc);
DECLARE_int32(logsens);
DECLARE_int32(logobj);
DECLARE_string(test);
DECLARE_bool(managedmem);
DECLARE_string(in);
DECLARE_int32(N);
DECLARE_int32(initperiod);
DECLARE_double(finthres);
DECLARE_double(filter);
DECLARE_double(step);
DECLARE_double(damp);
DECLARE_double(relthres);
DECLARE_bool(periodfilt);
DECLARE_bool(usesym);


#define b2s(boolValue) (boolValue?"Yes":"No")

namespace cfg {
	enum class Objective : uint8_t { bulk, shear, npr, custom };
	enum class Symmetry : uint8_t { reflect3, reflect6, rotate3, NONE };
	enum class InitWay : uint8_t { random, randcenter, noise, manual, interp, rep_randcenter, P, G, D, IWP };

	struct HomoConfig {
		Objective obj;
		Symmetry sym;
		InitWay winit;
		int reso[3];
		double volRatio;
		double youngsModulu;
		double poissonRatio;
		double finthres;
		double filterRadius;
		double designStep;
		double dampRatio;
		double femRelThres;
		std::string outprefix;
		std::string testname;
		std::string inputrho;
		bool useManagedMemory = true;
		int logrho, logc, logsens, logobj;
		int max_iter = 200;
		int initperiod;
		void parse(int argc, char** argv);
	};
}


