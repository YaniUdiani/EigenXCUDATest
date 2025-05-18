cmake_minimum_required(VERSION 3.22)
project(EigenXCUDATest LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

include_directories(
        $ENV{MYLOCAL}/include
        $ENV{HOMEBREW_PREFIX}/include
        $ENV{HOMEBREW_PREFIX}/include/eigen3/
        /usr/local/include
        ./include
        ./src
)

add_executable(EigenXCUDATest main.cu)

set_target_properties(EigenXCUDATest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
