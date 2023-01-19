#include "tictoc.h"

std::map<std::string, float> tictoc::Record::_table;


tictoc::void_buf tictoc::_voidBuf;

std::ostream  tictoc::_silent_stream(&_voidBuf);

std::chrono::steady_clock::time_point tictoc::getTag(void)
{
	return std::chrono::steady_clock::now();
}

float tictoc::get_record(const std::string& rec_name)
{
	auto it = _record._table.find(rec_name);
	if (it != _record._table.end()) {
		return it->second;
	}
	else {
		return 0;
	}
}

std::map<std::string, float> tictoc::clear_record(void)
{
	std::map<std::string, float> oldmap = _record._table;
	_record._table.clear();
	return oldmap;
}
