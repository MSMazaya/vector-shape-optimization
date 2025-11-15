cmake_minimum_required(VERSION 3.13)
project(VectorShapeOpt LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_CMAKE_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(MLIRConfig)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LLVM_CXX_FLAGS}")

include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${MLIR_LIBRARY_DIRS})

add_definitions(${LLVM_DEFINITIONS})
add_definitions(${MLIR_DEFINITIONS})

add_subdirectory(lib)
add_subdirectory(tools)
