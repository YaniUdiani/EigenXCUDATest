cmake_minimum_required(VERSION 3.22)
project(EigenXCUDATest LANGUAGES C CXX Fortran CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

include_directories(
        $ENV{MYLOCAL}/include
        $ENV{HOMEBREW_PREFIX}/include
        $ENV{HOMEBREW_PREFIX}/include/eigen3/
        /usr/local/include
        ./include
        ./src
)
find_path(EIGEN3_INCLUDE_DIR Eigen/Dense REQUIRED)
find_package(OpenBLAS)
if(OpenBLAS_FOUND)
    message(STATUS "OpenBLAS was found.")
    set(SCKY_AVAILABLE_BLAS_LIBS "openblas")
else()
    message(STATUS "OpenBLAS was not found. Looking for BLAS instead...")
    find_package(BLAS REQUIRED)
    set(SCKY_AVAILABLE_BLAS_LIBS "${BLAS_LIBRARIES}")
    message(STATUS "Found BLAS_LIBRARIES: ${SCKY_AVAILABLE_BLAS_LIBS}")
endif()
find_package(CUDAToolkit REQUIRED)
add_executable(EigenXCUDATest main.cu)
target_compile_options(EigenXCUDATest
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3 --extra-device-vectorization>
        )
set(CMAKE_CUDA_ARCHITECTURES "native")

target_link_libraries(EigenXCUDATest PRIVATE ${SCKY_AVAILABLE_BLAS_LIBS})
target_link_libraries(EigenXCUDATest
        PRIVATE
        CUDA::cublas
        CUDA::cudart
        )
target_compile_definitions(EigenXCUDATest PRIVATE EIGEN_NO_DEBUG)
target_compile_definitions(EigenXCUDATest PRIVATE EIGEN_USE_BLAS)

#set_target_properties(EigenXCUDATest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
