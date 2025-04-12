cmake_minimum_required(VERSION 3.18)
project(recsys)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# config backend module
pybind11_add_module(cppEngine src/cppEngine.cpp)

#  compiler opt
target_compile_options(cppEngine PRIVATE -O3 -march=native) #those LLVMs
target_include_directories(cppEngine PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${xtensor_INCLUDE_DIR}
    ${xsimd_INCLUDE_DIR}
)

#link
target_link_libraries(cppEngine PRIVATE
    Eigen3::Eigen
    xtensor
    xsimd
)