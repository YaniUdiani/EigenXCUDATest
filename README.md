# EigenXCUDATest
Eigen currently doesn't work with NVIDIA's CUDA. This little project passes raw underlying data from Eigen matrices
to cuBLAS, then compares matrix products on CPU vs GPU. How to build project on ICER:

# How to build project on MSU's ICER
```bash
module purge
module load NVHPC
module load OpenBLAS
module load CMake
module load Eigen
cmake -DCMAKE_BUILD_TYPE=Release CMakeLists.txt
make -j 16
./EigenXCUDATest 4000 4000 4000
```
