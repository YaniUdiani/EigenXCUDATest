# EigenXCUDATest
Eigen compatibility with NVIDIA's CUDA is currently limited. This little project passes raw underlying data from Eigen matrices
to cuBLAS, then compares matrix products on the CPU vs GPU.

# How to build project on MSU's ICER
```bash
source build.sh
./EigenXCUDATest 4000 4000 4000
```
