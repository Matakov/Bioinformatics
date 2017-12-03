# Authors: Franjo Matkovic

cmake_minimum_required (VERSION 2.6)
project (Bioinformatics)

# packages                                                                                                                                                                                                 
find_package(CUDA REQUIRED)

# set so a c++11 compiler is used
set (CMAKE_CXX_STANDARD 11)

# nvcc flags                                                                                                                                                                                               
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20)

# The version number.
set (Tutorial_VERSION_MAJOR 1)
set (Tutorial_VERSION_MINOR 0)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/main.h.in"
  "${PROJECT_BINARY_DIR}/main.h"
  #"${PROJECT_SOURCE_DIR}/utility.h.in"
  #"${PROJECT_BINARY_DIR}/utility.h"
  )

# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")

cuda_compile(HELPER_O SW.cu)

# add the executable
cuda_add_executable(program main.cpp utility.cpp ${HELPER_O})
