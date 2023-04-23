cmake_minimum_required(VERSION 3.14)

message(STATUS "cmake version ${CMAKE_VERSION}")

project(homo3d)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set (CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")
set(CMAKE_IGNORE_PATH "/mnt/c/Program Files/vcpkg/installed/x64-windows/share/gflags")
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# disable all warning for clean output
add_definitions(-w)

find_package(Eigen3 REQUIRED)
message(STATUS "Eigen directory ${EIGEN3_INCLUDE_DIRS}")
find_package(OpenVDB REQUIRED)
message(STATUS "OpenVDB include dir ${OpenVDB_INCLUDE_DIRS}")
find_package(gflags REQUIRED)
find_package(glm REQUIRED)


include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/cuda_helper)

enable_language(CUDA)

find_package(CUDA REQUIRED)

# Do what the new package does
find_library(CUDA_DRIVER_LIBRARY
             NAMES cuda_driver cuda
             HINTS ${CUDA_TOOLKIT_ROOT_DIR}
                   ENV CUDA_PATH
             PATH_SUFFIXES nvidia/current lib64 lib/x64 lib)
if (NOT CUDA_DRIVER_LIBRARY)
    # Don't try any stub directories until we have exhausted all other search locations.
    find_library(CUDA_DRIVER_LIBRARY
                 NAMES cuda_driver cuda
                 HINTS ${CUDA_TOOLKIT_ROOT_DIR}
                       ENV CUDA_PATH
                 PATH_SUFFIXES lib64/stubs lib/x64/stubs lib/stubs stubs)
endif ()

include(CheckLanguage)
check_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
	# set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

message(STATUS "CUDA libs ${CUDA_LIBRARIES}")
message(STATUS "CUDA lib  ${CUDA_DRIVER_LIBRARY}")

file(GLOB cudafiles 
            *.cu
            AutoDiff/*.cu
            culib/*.cu 
            gmem/*.cu 
            homogenization/*.cu 
            optimization/*.cu)
file(GLOB deprefiles
            */deprecated*.cu)
list(REMOVE_ITEM cudafiles "${deprefiles}")
# file(GLOB cudafiles homogenization/profile_test.cu culib/lib.cu culib/vector_intrinsic.cu)
message(STATUS "CUDA sources       : ${cudafiles}")
message(STATUS "deprecated sources : ${deprefiles}")

add_library(cubin STATIC ${cudafiles})
set_target_properties(cubin PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(cubin PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(cubin PROPERTIES CUDA_ARCHITECTURES "50;52;61;75;86")
# set_target_properties(cubin PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(cubin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
	--relocatable-device-code=true
    -dc
	--extended-lambda
	--std=c++17
	--expt-relaxed-constexpr
	--compile
    # -gencode arch=compute_80,code=sm_80
	>)
target_link_libraries(cubin PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(cubin PRIVATE ${CUDA_DRIVER_LIBRARY})
target_link_libraries(cubin PRIVATE glm::glm)

file(GLOB cppfiles 
        *.cpp
        AutoDiff/*.cpp
        culib/*.cpp
        gmem/*.cpp
        homogenization/*.cpp
        matlab/*.cpp
        optimization/*.cpp
    )
# file(GLOB cppfiles *.cpp)
add_executable(homo3d ${cppfiles})
target_include_directories(homo3d PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(homo3d PRIVATE cubin)
target_link_libraries(homo3d PRIVATE OpenVDB::openvdb)
target_link_libraries(homo3d PRIVATE -lHalf)
target_link_libraries(homo3d PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(homo3d PRIVATE ${CUDA_DRIVER_LIBRARY})
target_link_libraries(homo3d PRIVATE ${CUDA_curand_LIBRARY})
target_link_libraries(homo3d PRIVATE ${CUDA_cusolver_LIBRARY})
target_link_libraries(homo3d PRIVATE ${CUDA_cusparse_LIBRARY})
target_link_libraries(homo3d PRIVATE ${CUDA_CUBLAS_LIBRARIES})
target_link_libraries(homo3d PRIVATE ${GFLAGS_LIBRARIES})