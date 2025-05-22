# EigenXCUDATest
Eigen currently doesn't work with NVIDIA's CUDA. This little project passes raw underlying data from Eigen matrices
to cuBLAS, then compares matrix products on the CPU vs GPU.

# How to build project on MSU's ICER
```bash
source build.sh
./EigenXCUDATest 4000 4000 4000
```
