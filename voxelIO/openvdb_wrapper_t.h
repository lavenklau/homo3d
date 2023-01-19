#pragma once

#ifndef __OPENVDB_WRAPPER_H
#define __OPENVDB_WRAPPER_H

#include <openvdb/openvdb.h>
#include "vector"
#include "type_traits"
#include "string"
#include "openvdb/tools/VolumeToMesh.h"

template<typename Scalar>
class openvdb_wrapper_t {
public:

	static void lexicalGrid2openVDBfile(const std::string& filename, int gridSize[3], const std::vector<Scalar>& gridvalues) {

		if (gridvalues.size() != gridSize[0] * gridSize[1] * gridSize[2]) {
			throw std::string("size of value list does not match given grid size!");
		}

		openvdb::initialize();

		typedef typename std::conditional<std::is_same<Scalar, float>::value, openvdb::FloatGrid, openvdb::DoubleGrid>::type Grid;

		typename Grid::Ptr grid = Grid::create();

		typename Grid::Accessor acc = grid->getAccessor();

		for (int i = 0; i < gridSize[2]; i++) {
			for (int j = 0; j < gridSize[1]; j++) {
				for (int k = 0; k < gridSize[0]; k++) {
					int id = k + j * gridSize[0] + i * gridSize[0] * gridSize[1];
					Scalar val = gridvalues[id];
					openvdb::Coord xyz(k, j, i);
					acc.setValue(xyz, val);
				}
			}
		}

		openvdb::io::File file(filename);

		openvdb::GridPtrVec grids;
		grids.push_back(grid);

		file.write(grids);
		file.close();
	}

	static void grid2openVDBfile(const std::string& filename, std::vector<int> pos[3], const std::vector<Scalar>& gridvalues) {
		if (gridvalues.size() != pos[0].size() || gridvalues.size() != pos[1].size() || gridvalues.size() != pos[2].size()) {
			printf("\033[31msize of value list does not match given grid size!\033[0m\n");
			throw std::string("size of value list does not match given grid size!");
		}

		openvdb::initialize();

		typedef typename std::conditional<std::is_same<Scalar, float>::value, openvdb::FloatGrid, openvdb::DoubleGrid>::type Grid;

		typename Grid::Ptr grid = Grid::create();

		typename Grid::Accessor acc = grid->getAccessor();

		for (int i = 0; i < pos->size(); i++) {
			openvdb::Coord xyz(pos[0][i], pos[1][i], pos[2][i]);
			Scalar val = gridvalues[i];
			acc.setValue(xyz, val);
		}

		grid->setGridClass(openvdb::GRID_FOG_VOLUME);

		openvdb::io::File file(filename);

		openvdb::GridPtrVec grids;
		grids.push_back(grid);

		file.write(grids);

		file.close();
	}

	static void openVDBfile2grid(const std::string& filename, std::vector<int> pos[3], std::vector<Scalar>& gridvalues) {
		openvdb::initialize();
		typedef typename std::conditional<std::is_same<Scalar, float>::value, openvdb::FloatGrid, openvdb::DoubleGrid>::type Grid;
		typedef typename Grid::Ptr grid;
		openvdb::io::File file(filename);

		bool suc = file.open(); 
		if (!suc) {
			printf("\033[31mfailed to open file %s\033[0m\n", filename.c_str());
			throw std::runtime_error("failed open fail");
		}
		// Loop over all grids in the file and retrieve a shared pointer
		// to the one named "LevelSetSphere".  (This can also be done
		// more simply by calling file.readGrid("LevelSetSphere").)
		typename openvdb::GridBase::Ptr baseGrid;
		for (openvdb::io::File::NameIterator nameIter = file.beginName();
			nameIter != file.endName(); ++nameIter)
		{
			baseGrid = file.readGrid(nameIter.gridName());
		}

		pos[0].clear(); pos[1].clear(); pos[2].clear();

		grid gridptr = openvdb::gridPtrCast<Grid>(baseGrid);
		for (auto iter = gridptr->beginValueOn(); iter.test(); ++iter) {
			auto xyz = iter.getCoord();
			Scalar val = iter.getValue();
			pos[0].emplace_back(xyz.x());
			pos[1].emplace_back(xyz.y());
			pos[2].emplace_back(xyz.z());
			gridvalues.emplace_back(val);
		}

		file.close();
	}

	static void meshFromFile(const std::string& filename, std::vector<openvdb::Vec3s>& points, std::vector<openvdb::Vec3I>& trias, std::vector<openvdb::Vec4I>& quads, double isovalue, bool relaxDisorientTri) {
		openvdb::initialize();
		typedef typename std::conditional<std::is_same<Scalar, float>::value, openvdb::FloatGrid, openvdb::DoubleGrid>::type Grid;
		typedef typename Grid::Ptr grid;
		openvdb::io::File file(filename);

		bool suc = file.open();
		if (!suc) {
			printf("\033[31mfailed to open file %s\033[0m\n", filename.c_str());
		}

		openvdb::GridBase::Ptr baseGrid;
		for (openvdb::io::File::NameIterator nameIter = file.beginName();
			nameIter != file.endName(); ++nameIter)
		{
			baseGrid = file.readGrid(nameIter.gridName());
		}

		grid gridptr = openvdb::gridPtrCast<Grid>(baseGrid);

		openvdb::tools::volumeToMesh(*gridptr, points, trias, quads, isovalue, relaxDisorientTri);
	}

	static void meshFromFile(const std::string& filename, std::vector<openvdb::Vec3s>& points, std::vector<openvdb::Vec4I>& quads, double isovalue) {
		openvdb::initialize();
		typedef typename std::conditional<std::is_same<Scalar, float>::value, openvdb::FloatGrid, openvdb::DoubleGrid>::type Grid;
		typedef typename Grid::Ptr grid;
		openvdb::io::File file(filename);

		bool suc = file.open();
		if (!suc) {
			printf("\033[31mfailed to open file %s\033[0m\n", filename.c_str());
		}

		openvdb::GridBase::Ptr baseGrid;
		for (openvdb::io::File::NameIterator nameIter = file.beginName();
			nameIter != file.endName(); ++nameIter)
		{
			baseGrid = file.readGrid(nameIter.gridName());
		}

		grid gridptr = openvdb::gridPtrCast<Grid>(baseGrid);

		openvdb::tools::volumeToMesh(*gridptr, points, quads, isovalue);
	}
};

#endif

