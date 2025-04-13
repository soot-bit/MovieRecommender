cmake_minimum_required(VERSION 3.18)
project(recsys)

set(EIGEN_INCLUDE_DIR /home/soot/Code/Projects/MovieRecommender/src/external/libegen)


find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenMP REQUIRED)

# config backend module
pybind11_add_module(cppEngine src/cppEngine.cpp)

#  most  optimal
target_compile_options(cppEngine PRIVATE -O3 -march=native) #those LLVMs
target_include_directories(cppEngine PRIVATE
    ${EIGEN_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/src/external/EigenRand
    ${PROJECT_SOURCE_DIR}/src/external/indicators/include
)

target_link_libraries(cppEngine PRIVATE OpenMP::OpenMP_CXX pybind11::module)
