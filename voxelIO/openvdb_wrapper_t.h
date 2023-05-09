#pragma once

#ifndef __OPENVDB_WRAPPER_H
#define __OPENVDB_WRAPPER_H

#include "vector"
#include "type_traits"
#include "string"
#include "glm/glm.hpp"

template<typename Scalar>
class openvdb_wrapper_t {
public:
	static void lexicalGrid2openVDBfile(
		const std::string &filename, int gridSize[3], const std::vector<Scalar> &gridvalues);

	static void grid2openVDBfile(const std::string &filename, std::vector<int> pos[3], const std::vector<Scalar> &gridvalues);

	static void openVDBfile2grid(const std::string &filename, std::vector<int> pos[3], std::vector<Scalar> &gridvalues);

	static void meshFromFile(
		const std::string &filename, std::vector<glm::vec3> &points,
		std::vector<glm::vec<3, int>> &trias, std::vector<glm::vec<4, int>> &quads,
		double isovalue, bool relaxDisorientTri);

	static void meshFromFile(
		const std::string &filename, std::vector<glm::vec3> &points,
		std::vector<glm::vec<4, int>> &quads, double isovalue);
};

// template<> class openvdb_wrapper_t<float>;

#endif

