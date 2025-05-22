module purge
module load NVHPC
module load OpenBLAS
module load CMake
module load Eigen
cmake -DCMAKE_BUILD_TYPE=Release CMakeLists.txt
make -j 16