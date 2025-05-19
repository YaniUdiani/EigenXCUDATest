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

    // To time CUDA calls within this function
    cudaEvent_t start;
    cudaEvent_t stop;

    // Default uninitialized cuBLAS handle
    cublasHandle_t handle;

    // Host (CPU) pointers of A and B typed in CUDA's custom implementation of complex double
    const cuDoubleComplex *pAHost = nullptr;
    const cuDoubleComplex *pBHost = nullptr;
    std::complex<double> *pCHost = C.data(); // Will store result on the CPU

    // Device (GPU) pointers of A, B, and C
    cuDoubleComplex *devPtrA = nullptr;
    cuDoubleComplex *devPtrB = nullptr;
    cuDoubleComplex *devPtrC = nullptr;
    std::complex<double> *devPtrCInSTDComplexDouble = nullptr; // Will be used to cast out of cuDoubleComplex

    // Clean up and error lambdas
    auto BurnItAll = [&devPtrA, &devPtrB, &devPtrC, &handle, &start, &stop] (){
        cudaFree(devPtrA);
        cudaFree(devPtrB);
        cudaFree(devPtrC);
        cublasDestroy(handle);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    };
    auto CUDAErrorCheck = [&devPtrA, &devPtrB, &devPtrC, &handle, &BurnItAll, &start, &stop]
            (const cudaError_t &cudaStatus, const char *file, int line, const std::string &description){
        if (cudaStatus != cudaSuccess) {
            std::cerr << "\n[ERROR!] in CUBLASZgemm(A, B, C,...) of file " << file << "(line " << line << ")\n";
            std::cerr <<"Failure when doing: " << description << '\n';
            std::cerr << cudaGetErrorString(cudaStatus) << "\n";
            BurnItAll(); // exit(EXIT_FAILURE) doesn't call destructors
            exit(EXIT_FAILURE);
        }
    };
    auto CUBLASErrorCheck = [&devPtrA, &devPtrB, &devPtrC, &handle, &BurnItAll, &start, &stop]
            (const cublasStatus_t &cublasStatus, const char *file, int line, const std::string &description){
        if (cublasStatus != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "\n[ERROR!] in CUBLASZgemm(A, B, C,...) of file " << file << "(line " << line << ")\n";
            std::cerr <<"Failure when doing: " << description << '\n';
            std::cerr << cublasGetStatusString(cublasStatus) << "\n";
            BurnItAll(); // exit(EXIT_FAILURE) doesn't call destructors
            exit(EXIT_FAILURE);
        }
    };

    // Time CUDA calls within this function
    CUDAErrorCheck(cudaEventCreate(&start), "Creation of start timer for the device");
    CUDAErrorCheck(cudaEventCreate(&stop), "Creation of stop timer for the device");
    CUDAErrorCheck(cudaEventRecord(start, 0), "Recording with start timer for the device");

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

    // Retrieve underlying data pointer from Eigen, then cast std::complex<double> into cuDoubleComplex
    pAHost = reinterpret_cast<const cuDoubleComplex*>(A.data());
    pBHost = reinterpret_cast<const cuDoubleComplex*>(B.data());

    // Cast alpha and beta into cuDoubleComplex
    const cuDoubleComplex *pAlpha = reinterpret_cast<const cuDoubleComplex*>(&alpha);
    const cuDoubleComplex *pBeta = reinterpret_cast<const cuDoubleComplex*>(&beta);

    // Allocate A, B, and C on the device
    CUDAErrorCheck(cudaMalloc((void**)&devPtrA, sizeOfA * sizeof(cuDoubleComplex)),
                   "Device memory allocation for input matrix A.");
    CUDAErrorCheck(cudaMalloc((void**)&devPtrB, sizeOfB * sizeof(cuDoubleComplex)),
                   "Device memory allocation for input matrix B.");
    CUDAErrorCheck(cudaMalloc((void**)&devPtrC, sizeOfC * sizeof(cuDoubleComplex)),
                   "Device memory allocation for resultant matrix C.");

    // Copy over A, and B from host to device (no need to do so for C)
    CUDAErrorCheck(cudaMemcpy(devPtrA, pAHost, sizeOfA * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice),
                   "Copy of input matrix A from host to device.");
    CUDAErrorCheck(cudaMemcpy(devPtrB, pBHost, sizeOfB * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice),
                   "Copy of input matrix B from host to device.");
    CUDAErrorCheck(cudaMemset(devPtrC, 42.0, sizeOfC * sizeof(cuDoubleComplex)),
                   "Initialization of resultant matrix C on the device.");

    // Initialize cuBLAS handler, then call Zgemm()
    CUBLASErrorCheck(cublasCreate(&handle), "cuBLAS initialization.");
    CUBLASErrorCheck(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rowsOfA, colsOfB, rowsOfB, pAlpha, devPtrA, ldA,
                                 devPtrB, ldB, pBeta, devPtrC, ldC), "Matrix product of A and B on the device.");

    // Cast away cuDoubleComplex so we can output to host
    devPtrCInSTDComplexDouble = reinterpret_cast<std::complex<double>*>(devPtrC);

    // Copy resultant matrix C from device to host
    CUDAErrorCheck(cudaMemcpy(pCHost, devPtrCInSTDComplexDouble, sizeOfC * sizeof(std::complex<double>), cudaMemcpyDeviceToHost),
                   "Copy of resultant matrix C from device to host.");

    // Stop timing CUDA function
    CUDAErrorCheck(cudaEventRecord(stop, 0), "Stop timer recording for the device.");
    CUDAErrorCheck(cudaEventSynchronize(stop), "Synchronization of CPU with stop event.");

    float minutes = 0.0f;
    CUDAErrorCheck(cudaEventElapsedTime(&minutes, start, stop), "Retrieval of elapsed time on the device");
    std::cout << "CUBLASZgemm() took " << minutes / 60000 << " minutes" << '\n';

    // Destroy unneeded memory
    BurnItAll(); // Yay! :)
}

int main(void){
    //  Initializing Eigen Matrices
    int n = 700;
    int m = 700;
    int k = 700;
    MatriX A = MatriX::Random(n,k);
    MatriX B = MatriX::Random(k,m);
    MatriX C = MatriX::Random(n,m);

    auto startDot = std::chrono::high_resolution_clock::now();
    C.noalias() = A * B;
    auto stopDot = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationDot = stopDot - startDot;
    MatriX AxB = C;
    std::cout<<"CPU computed ||C||: " <<C.norm()<<'\n';
    std::cout<<"CPU duration: " <<durationDot.count()<<'\n';

    auto startGPUDot = std::chrono::high_resolution_clock::now();
    CUBLASZgemm(A, B, C);
    cudaDeviceSynchronize();
    auto stopGPUDot = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationGPUDot = stopGPUDot - startGPUDot;
    std::cout<<"||C_gpu-C_cpu||: " <<(C-AxB).norm()<<'\n';
    std::cout<<"GPU duration using chrono: " <<durationGPUDot.count()<<'\n';
}
