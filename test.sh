#!/bin/bash
#
#modules:
module unload gpu
# module load cpe/23.03
module load PrgEnv-gnu
# module load gcc/11.2.0
module load cmake
module load cudatoolkit/12.2
# module load cudatoolkit/11.7
# avoid bug in cray-libsci/21.08.1.2
# module load cray-libsci/22.11.1.2
# module load cray-libsci/23.02.1.1
module load cray-libsci/23.12.5
ulimit -s unlimited
#MPI settings:
export MPICH_GPU_SUPPORT_ENABLED=0
# export CRAY_ACCEL_TARGET=nvidia80
# echo MPICH_GPU_SUPPORT_ENABLED=$MPICH_GPU_SUPPORT_ENABLED
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:$LD_LIBRARY_PATH
#SUPERLU settings:

SUPERLU_HOME=/global/cfs/cdirs/m2957/tianyi/superlu_dist/build/
export LD_LIBRARY_PATH=$SUPERLU_HOME/lib:$LD_LIBRARY_PATH

export MAX_BUFFER_SIZE=50000000


# export SUPERLU_LBS=GD  
export SUPERLU_ACC_OFFLOAD=0 # this can be 0 to do CPU tests on GPU nodes
# export GPU3DVERSION=0
# export ANC25D=0
# export NEW3DSOLVE=1    
# export NEW3DSOLVETREECOMM=1
# export SUPERLU_BIND_MPI_GPU=1 # assign GPU based on the MPI rank, assuming one MPI per GPU
# export SUPERLU_ACC_SOLVE=1

export SUPERLU_MAXSUP=256 # max supernode size
export SUPERLU_RELAX=64  # upper bound for relaxed supernode size
export SUPERLU_MAX_BUFFER_SIZE=10000000 ## 500000000 # buffer size in words on GPU
export SUPERLU_NUM_LOOKAHEADS=2   ##4, must be at least 2, see 'lookahead winSize'
# export SUPERLU_NUM_GPU_STREAMS=1
# export SUPERLU_N_GEMM=6000 # FLOPS threshold divide workload between CPU and GPU
# nmpipergpu=1
# export SUPERLU_MPI_PROCESS_PER_GPU=$nmpipergpu # 2: this can better saturate GPU

# ##NVSHMEM settings:
# NVSHMEM_HOME=/global/cfs/cdirs/m3894/lib/PrgEnv-gnu/nvshmem_src_2.8.0-3/build/
# export NVSHMEM_USE_GDRCOPY=1
# export NVSHMEM_MPI_SUPPORT=1
# export MPI_HOME=${MPICH_DIR}
# export NVSHMEM_LIBFABRIC_SUPPORT=1
# export LIBFABRIC_HOME=/opt/cray/libfabric/1.15.2.0
# export LD_LIBRARY_PATH=$NVSHMEM_HOME/lib:$LD_LIBRARY_PATH
# export NVSHMEM_DISABLE_CUDA_VMM=1
# export FI_CXI_OPTIMIZED_MRS=false
# export NVSHMEM_BOOTSTRAP_TWO_STAGE=1
# export NVSHMEM_BOOTSTRAP=MPI
# export NVSHMEM_REMOTE_TRANSPORT=libfabric

if [[ $NERSC_HOST == edison ]]; then
  CORES_PER_NODE=24
  THREADS_PER_NODE=48
elif [[ $NERSC_HOST == cori ]]; then
  CORES_PER_NODE=32
  THREADS_PER_NODE=64
  # This does not take hyperthreading into account
elif [[ $NERSC_HOST == perlmutter ]]; then
  CORES_PER_NODE=64
  THREADS_PER_NODE=128
  # GPUS_PER_NODE=4
else
  # Host unknown; exiting
  exit $EXIT_HOST
fi

### ADI matrix test
# srun -n 8 ./adi_mat -r 2 -c 2 -w 2 -v 2 -f .dat /global/cfs/cdirs/m2957/tianyi/matrix/ADI_test/
# srun -n 2 ./adi_mat -r 1 -c 1 -w 1 -v 1 -f .dat /global/cfs/cdirs/m2957/tianyi/matrix/ADI_test/

### ADI matrix generating shifts test
# srun -n 8 ./adi_mat_shifts -r 2 -c 2 -w 2 -v 2 -f .dat /global/cfs/cdirs/m2957/tianyi/matrix/ADI_test/

### fADI matrix test
# srun -n 8 ./fadi_mat -r 2 -c 2 -w 2 -v 2 -f .dat /global/cfs/cdirs/m2957/tianyi/matrix/fADI_test/
# srun -n 2 ./fadi_mat -r 1 -c 1 -w 1 -v 1 -f .dat /global/cfs/cdirs/m2957/tianyi/matrix/fADI_test/

### fADI TTSVD 3D test
# srun -n 3 ./fadi_ttsvd_3d -r 1 -c 1 -w 1 -v 1 -m 1 -n 1 -f .dat /global/cfs/cdirs/m2957/tianyi/matrix/fADI_TTSVD_3D_test_small/
# srun -n 12 ./fadi_ttsvd_3d -r 2 -c 2 -w 2 -v 2 -m 2 -n 2 -f .dat /global/cfs/cdirs/m2957/tianyi/matrix/fADI_TTSVD_3D_test/
# srun -n 2 ./fadi_ttsvd_3d_2grids -r 1 -c 1 -w 1 -v 1 -f .dat /global/cfs/cdirs/m2957/tianyi/matrix/fADI_TTSVD_3D_test_small/
# srun -n 8 ./fadi_ttsvd_3d_2grids -r 2 -c 2 -w 2 -v 2 -f .dat /global/cfs/cdirs/m2957/tianyi/matrix/fADI_TTSVD_3D_test/

### fADI parallel TTSVD 3D test
# srun -n 3 ./fadi_para_ttsvd_3d -r 1 -c 1 -w 1 -v 1 -m 1 -n 1 -f .dat /global/cfs/cdirs/m2957/tianyi/matrix/fADI_para_TTSVD_3D_test_small/
# srun -n 12 ./fadi_para_ttsvd_3d -r 2 -c 2 -w 2 -v 2 -m 2 -n 2 -f .dat /global/cfs/cdirs/m2957/tianyi/matrix/fADI_para_TTSVD_3D_test/

### fADI TTSVD 4D test
# srun -n 2 ./fadi_ttsvd_md -r 1 -c 1 -w 1 -v 1 -f .dat /global/cfs/cdirs/m2957/tianyi/matrix/fADI_TTSVD_4D_test_small/
# srun -n 8 ./fadi_ttsvd_md -r 2 -c 2 -w 2 -v 2 -f .dat /global/cfs/cdirs/m2957/tianyi/matrix/fADI_TTSVD_4D_test/