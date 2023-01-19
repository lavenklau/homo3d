#pragma once

#ifndef _MY_TICTOC_H
#define _MY_TICTOC_H

#include "map"
#include "string"
#include "chrono"
#include "iostream"
#include "sstream"

namespace tictoc {

	class Record;

	extern Record _record;

	class Record {
		static std::map<std::string, float> _table;
		friend float get_record(const std::string& rec_name);
		friend std::map<std::string, float> clear_record(void);
		friend class live;
	};

	class void_buf :public std::stringbuf {
		virtual int sync() { return 0; }
	};

	extern void_buf _voidBuf;

	extern std::ostream _silent_stream;

	//void func(void) {
	//}

	class live {
		const std::string _name;
		std::chrono::steady_clock::time_point _born;
		std::ostream& tout;
	public:
		live(const std::string& sname, std::ostream& _out = _silent_stream) :tout(_out), _name(sname) {
			_born = std::chrono::steady_clock::now();
		}
		~live() {
			auto _die = std::chrono::steady_clock::now();
			std::chrono::microseconds _age = std::chrono::duration_cast<std::chrono::microseconds>(_die - _born);
			float _cost = float(_age.count()) / 1000;
			tout << "[*] time cost on " << _name <<" : "<<_cost << " ms" << std::endl;
			_record._table[_name] = _cost;
		}
	};

	std::chrono::steady_clock::time_point getTag(void);

	enum time_unit {
		ms,
		s
	};
	template<time_unit tu>
	static float Duration(std::chrono::steady_clock::time_point sta, std::chrono::steady_clock::time_point en) {
		auto _age = std::chrono::duration_cast<std::chrono::microseconds>(en - sta);
		float _cost = 0;
		switch (tu) {
		case ms:
			_cost = float(_age.count()) / 1000;
			break;
		case s:
			_cost = float(_age.count()) / 1000000;
		}
		return _cost;
	}

	float get_record(const std::string& rec_name);

	std::map<std::string, float> clear_record(void);
};

#define _TIC(name) { tictoc::live _anonymous_timer(name);

#define _TOC };

#endif

