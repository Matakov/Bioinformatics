# Authors: Franjo Matkovic

cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project (Bioinformatics)

find_package(CUDA QUIET REQUIRED)

set (CMAKE_CXX_STANDARD 11)

# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories("${PROJECT_BINARY_DIR}")

cuda_compile(HELPER_O SW.cu)

# add the executable
cuda_add_executable(program main.cpp utility.cpp ${HELPER_O})
