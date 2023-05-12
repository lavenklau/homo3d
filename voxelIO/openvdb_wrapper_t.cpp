#define _USE_MATH_DEFINES
#include "openvdb_wrapper_t.h"
#include <openvdb/openvdb.h>
#include "openvdb/tools/VolumeToMesh.h"

// class openvdb_wrapper_t<float>;

template<>
void openvdb_wrapper_t<float>::lexicalGrid2openVDBfile(const std::string &filename,
 int gridSize[3], const std::vector<float> &gridvalues)
{
    using Scalar = float;
    if (gridvalues.size() != gridSize[0] * gridSize[1] * gridSize[2])
    {
        throw std::string("size of value list does not match given grid size!");
    }

    openvdb::initialize();

    typedef typename std::conditional<std::is_same<Scalar, float>::value, openvdb::FloatGrid, openvdb::DoubleGrid>::type Grid;

    typename Grid::Ptr grid = Grid::create();

    typename Grid::Accessor acc = grid->getAccessor();

    for (int i = 0; i < gridSize[2]; i++)
    {
        for (int j = 0; j < gridSize[1]; j++)
        {
            for (int k = 0; k < gridSize[0]; k++)
            {
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

template<>
void openvdb_wrapper_t<float>::grid2openVDBfile(
    const std::string &filename, std::vector<int> pos[3], const std::vector<float> &gridvalues)
{
    using Scalar = float;
    if (gridvalues.size() != pos[0].size() || gridvalues.size() != pos[1].size() || gridvalues.size() != pos[2].size())
    {
        printf("\033[31msize of value list does not match given grid size!\033[0m\n");
        throw std::string("size of value list does not match given grid size!");
    }

    openvdb::initialize();

    typedef typename std::conditional<std::is_same<Scalar, float>::value, openvdb::FloatGrid, openvdb::DoubleGrid>::type Grid;

    typename Grid::Ptr grid = Grid::create();

    typename Grid::Accessor acc = grid->getAccessor();

    for (int i = 0; i < pos->size(); i++)
    {
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

template<>
void openvdb_wrapper_t<float>::openVDBfile2grid(const std::string& filename, std::vector<int> pos[3], std::vector<float>& gridvalues) {
    using Scalar = float;
    openvdb::initialize();
    typedef typename std::conditional<std::is_same<Scalar, float>::value, openvdb::FloatGrid, openvdb::DoubleGrid>::type Grid;
    typedef typename Grid::Ptr grid;
    openvdb::io::File file(filename);

    bool suc = file.open();
    if (!suc)
    {
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

    pos[0].clear();
    pos[1].clear();
    pos[2].clear();

    grid gridptr = openvdb::gridPtrCast<Grid>(baseGrid);
    for (auto iter = gridptr->beginValueOn(); iter.test(); ++iter)
    {
        auto xyz = iter.getCoord();
        Scalar val = iter.getValue();
        pos[0].emplace_back(xyz.x());
        pos[1].emplace_back(xyz.y());
        pos[2].emplace_back(xyz.z());
        gridvalues.emplace_back(val);
    }

    file.close();
}

template<>
void openvdb_wrapper_t<float>::meshFromFile(
    const std::string &filename, std::vector<glm::vec3> &points,
    std::vector<glm::vec<3, int>> &trias, std::vector<glm::vec<4, int>> &quads, double isovalue, bool relaxDisorientTri)
{
    using Scalar = float;
    openvdb::initialize();
    typedef typename std::conditional<std::is_same<Scalar, float>::value, openvdb::FloatGrid, openvdb::DoubleGrid>::type Grid;
    typedef typename Grid::Ptr grid;
    openvdb::io::File file(filename);

    bool suc = file.open();
    if (!suc)
    {
        printf("\033[31mfailed to open file %s\033[0m\n", filename.c_str());
    }

    openvdb::GridBase::Ptr baseGrid;
    for (openvdb::io::File::NameIterator nameIter = file.beginName();
         nameIter != file.endName(); ++nameIter)
    {
        baseGrid = file.readGrid(nameIter.gridName());
    }

    grid gridptr = openvdb::gridPtrCast<Grid>(baseGrid);

    std::vector<openvdb::Vec3s> points_vdb;
    std::vector<openvdb::Vec3I> tris_vdb;
    std::vector<openvdb::Vec4I> quads_vdb;

    openvdb::tools::volumeToMesh(*gridptr, points_vdb, tris_vdb, quads_vdb, isovalue, relaxDisorientTri);

    for (int i = 0; i < points_vdb.size(); i++) {
        points.emplace_back(points_vdb[i][0], points_vdb[i][1], points_vdb[i][2]);
    }
    for (int i = 0; i < tris_vdb.size(); i++) {
        trias.emplace_back(tris_vdb[i][0], tris_vdb[i][1], tris_vdb[i][2]);
    }
    for (int i = 0; i < quads_vdb.size(); i++) {
        quads.emplace_back(quads_vdb[i][0], quads_vdb[i][1], quads_vdb[i][2], quads_vdb[i][3]);
    }
}

template<>
void openvdb_wrapper_t<float>::meshFromFile(
    const std::string &filename, std::vector<glm::vec3> &points,
    std::vector<glm::vec<4, int>> &quads, double isovalue)
{
    using Scalar = float;
    openvdb::initialize();
    typedef typename std::conditional<std::is_same<Scalar, float>::value, openvdb::FloatGrid, openvdb::DoubleGrid>::type Grid;
    typedef typename Grid::Ptr grid;
    openvdb::io::File file(filename);

    bool suc = file.open();
    if (!suc)
    {
        printf("\033[31mfailed to open file %s\033[0m\n", filename.c_str());
    }

    openvdb::GridBase::Ptr baseGrid;
    for (openvdb::io::File::NameIterator nameIter = file.beginName();
         nameIter != file.endName(); ++nameIter)
    {
        baseGrid = file.readGrid(nameIter.gridName());
    }

    grid gridptr = openvdb::gridPtrCast<Grid>(baseGrid);

    std::vector<openvdb::Vec3s> points_vdb;
    std::vector<openvdb::Vec3I> tris_vdb;
    std::vector<openvdb::Vec4I> quads_vdb;

    openvdb::tools::volumeToMesh(*gridptr, points_vdb, quads_vdb, isovalue);

    for (int i = 0; i < points_vdb.size(); i++) {
        points.emplace_back(points_vdb[i][0], points_vdb[i][1], points_vdb[i][2]);
    }
    for (int i = 0; i < quads_vdb.size(); i++) {
        quads.emplace_back(quads_vdb[i][0], quads_vdb[i][1], quads_vdb[i][2], quads_vdb[i][3]);
    }
}