#include "iostream"
#include <chrono>
#include "Eigen/Dense"
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define CUDAErrorCheck(cudaStatus, customMSG) (CUDAErrorCheck(cudaStatus, __FILE__, __LINE__, customMSG))
#define CUBLASErrorCheck(cublasStatus, customMSG) (CUBLASErrorCheck(cublasStatus, __FILE__, __LINE__, customMSG))

typedef Eigen::MatrixXcd MatriX;
void CUBLASZgemm(const Eigen::Ref<const MatriX> &A, const Eigen::Ref<const MatriX> &B, Eigen::Ref<MatriX> C,
                 const MatriX::Scalar &alpha = 1.0, const MatriX::Scalar &beta = 0.0){
    static_assert(std::is_same<MatriX::Scalar, std::complex<double>>::value);
    static_assert(!MatriX::IsRowMajor, "CUDA::CUBLASZgemm(A, B, C,...) requires column major storage of A, B, and C.");

    // CUDA and cuBLAS status and handle
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    // Host (CPU) pointers of A, B, and C typed in CUDA's custom implementation of std::complex double
    const cuDoubleComplex *pAHost = nullptr;
    const cuDoubleComplex *pBHost = nullptr;
    cuDoubleComplex *pCHost = nullptr;

    // Device (GPU) pointers of A, B, and C
    const cuDoubleComplex *devPtrA = nullptr;
    const cuDoubleComplex *devPtrB = nullptr;
    cuDoubleComplex *devPtrC = nullptr;
    std::complex<double> *devPtrCInSTDComplexDouble = nullptr;

    // Clean up
    auto BurnItAll = [&devPtrA, &devPtrB, &devPtrC, &handle] (){
        cudaFree(devPtrA);
        cudaFree(devPtrB);
        cudaFree(devPtrC);
        cublasDestroy(handle);
    };

    // Grab dimensions of A, B, and C
    const Eigen::Index rowsOfA = A.rows();
    const Eigen::Index colsOfA = A.cols();
    const Eigen::Index sizeOfA = A.size();
    const Eigen::Index &ldA = rowsOfA; // Column major storage enforced above

    const Eigen::Index rowsOfB = B.rows();
    const Eigen::Index colsOfB = B.cols();
    const Eigen::Index sizeOfB = B.size();
    const Eigen::Index &ldB = rowsOfB; // Column major storage enforced above

    const Eigen::Index rowsOfC = C.rows();
    const Eigen::Index colsOfC = C.cols();
    const Eigen::Index sizeOfC = C.size();
    const Eigen::Index &ldC = rowsOfC; // Column major storage enforced above

    assert(colsOfA == rowsOfB);
    assert(rowsOfA == rowsOfC);
    assert(colsOfB == colsOfC);

    // Retrieve underlying data from Eigen, then cast std::complex double into cuDoubleComplex
    pAHost = reinterpret_cast<const cuDoubleComplex*>(A.data());
    pBHost = reinterpret_cast<const cuDoubleComplex*>(B.data());
    pCHost = reinterpret_cast<cuDoubleComplex*>(C.data());

    // Cast alpha and beta into cuDoubleComplex
    const cuDoubleComplex *pAlpha = reinterpret_cast<cuDoubleComplex*>(&alpha);
    const cuDoubleComplex *pBeta = reinterpret_cast<cuDoubleComplex*>(&beta);

    // Destroy allocated
    BurnItAll();
}

int main(){
    //  Initializing Eigen Matrices
    int N = 500;
    MatriX a_E = MatriX::Identity(N,N);
    MatriX b_E = MatriX::Ones(N,N);
    MatriX c_E = MatriX::Zero(N,N);

    auto startDot = std::chrono::high_resolution_clock::now();
    c_E.noalias() = a_E * b_E;
    auto stopDot = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationDot = stopDot - startDot;
    std::cout<<"duration: " <<durationDot.count()<<'\n';
    std::cout<<"||c_E||: " <<c_E.norm()<<'\n';

    MultiplyUsingCUBLAS(a_E, b_E, c_E);

    //std::cout<<"a_E: " <<a_E<<'\n';
    //std::cout<<"b_E: " <<b_E<<'\n';
    //unsigned a = 4u;
    //unsigned b = 4u;
    //Eigen::MatrixXcd hi = Eigen::MatrixXcd::Zero(a,b);
}
