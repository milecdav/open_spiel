#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --partition=amdfast
#SBATCH --mem=64G

ml Clang/16.0.6-GCCcore-12.3.0-CUDA-12.3.0
ml CMake/3.26.3-GCCcore-12.3.0
ml Python/3.11.3-GCCcore-12.3.0
ml git/2.41.0-GCCcore-12.3.0-nodocs
ml PyYAML/6.0-GCCcore-12.3.0
ml Python-bundle-PyPI/2023.06-GCCcore-12.3.0
ml protobuf-python/4.24.0-GCCcore-12.3.0
ml IPython/8.14.0-GCCcore-12.3.0
ml flax/0.8.2-foss-2023a-CUDA-12.3.0
ml SciPy-bundle/2023.07-gfbf-2023a
ml Abseil/20230125.3-GCCcore-12.3.0
ml pybind11/2.11.1-GCCcore-12.3.0

cd build

cmake .
make

./examples/battleships_mavs_example