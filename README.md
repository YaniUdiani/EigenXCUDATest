# EigenXCUDATest
Eigen currently doesn't work with NVIDIA's CUDA. This little project passes raw underlying data from Eigen matrices
to cuBLAS, then compares matrix products on CPU vs GPU.
