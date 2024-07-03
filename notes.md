
# TPS - BTE integration
## 0D Batched BTE solver
* Single GPU
  * Solve for series of batched 0d Boltzmann on a single GPU (steady & transient cases) [Done]
* Extend Single GPU version to Multi-GPU and Parla Integration [Done]
* Finally we can move to spatial coupling if we decided to do so 
  
## TPS code
* Docker documentation: https://github.com/pecos/tps/blob/boltzmann-integration/docker/test-gpu/README.md
* Running the TPS code sequentially

```
cd tps
echo $PWD
source /etc/profile.d/lmod.sh
ml list
#./bootstrap
#mkdir build-cpu && cd build-cpu && ../configure --enable-pybind11 && cd ..
cd build-cpu && make all -j16 && cd ../
cp src/*.py build-cpu/src/.

cd tps-inputs/axisymmetric/argon/lowP/single-rxn
echo "launching tps + Boltzmann"
./../../../../../build-cpu/src/tps-time-loop.py -run plasma.reacting.tps2boltzmann.ini


```
* Running the code with MPI
```
module purge
module load gcc/11.2.0 mvapich2/2.3.7 tacc-apptainer/1.1.8
ml list

export CONTAINER_LD_LIB_PATH=/opt/ohpc/pub/libs/gnu9/gsl/2.7/lib:/opt/ohpc/pub/utils/valgrind/3.19.0/lib:/opt/ohpc/pub/libs/gnu9/mvapich2/boost/1.76.0/lib:/opt/ohpc/pub/libs/gnu9/openblas/0.3.7/lib:/opt/ohpc/pub/libs/gnu9/mvapich2/hdf5/1.10.8/lib:/opt/ohpc/pub/libs/gnu9/metis/5.1.0/lib:/opt/ohpc/pub/mpi/mvapich2-gnu9/2.3.6/lib:/opt/ohpc/pub/compiler/gcc/9.4.0/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.singularity.d/libs

pwd
date
MV2_SMP_USE_CMA=0 ibrun apptainer exec --nv ls6-tps_cpu_env_parla_latest.sif  env LD_LIBRARY_PATH=$CONTAINER_LD_LIB_PATH:$LD_LIBRARY_PATH python3 -c "from mpi4py import MPI; import cupy; import parla; import sympy; comm = MPI.COMM_WORLD; rank = comm.Get_rank(); size=comm.Get_size(); print(rank,size);"
cd tps/tps-inputs/axisymmetric/argon/lowP/single-rxn
#MV2_SMP_USE_CMA=0 ibrun apptainer exec --cleanenv --nv ../../../../../../ls6-tps_cpu_env_parla_latest.sif env LD_LIBRARY_PATH=$CONTAINER_LD_LIB_PATH:$LD_LIBRARY_PATH python3  ./../../../../../build-cpu/src/tps-time-loop.py -run plasma.reacting.tps2boltzmann.ini
MV2_SMP_USE_CMA=0 ibrun apptainer exec --nv ../../../../../../ls6-tps_cpu_env_parla_latest.sif env LD_LIBRARY_PATH=$CONTAINER_LD_LIB_PATH:$LD_LIBRARY_PATH python3  ./../../../../../build-cpu/src/tps-time-loop.py -run plasma.reacting.tps2boltzmann.ini
#MV2_SMP_USE_CMA=0 ibrun apptainer exec --cleanenv --nv  ls6-tps_cpu_env_parla_latest.sif ./run.sh

```

## SC24
* Abstract : MAR 26, 2024, paper due: APR 2, 2024
* We need to develop two-way coupled based solver, (i.e., trivial implementation)
* Optimization 1: Perform Hierarchical clustering of the spatial points, with 0-order implementation, (this should resolve the load imbalance issue, that we see in the clustering problem)
* Optimization 2: Use all MPI ranks, this would require scatter and gather of the BTE input fields, and QoIs computed. 
* Optimization 3: Switch for iterative solves, instead of direct solve, pre-conditioning via domain decomposition (we need to do this for the normalized distribution function evolution)




# 1D-BTE
* Perform standalone comparison with PIC DSMC code, with enforced E-field, 
  * Time harmonic spatially homogenous E-field
  * Time harmonic E-field

# 1D-Glow discharge with BTE
## Fluid code
* For fluid approximation need to fix the electron density $n_e$ BC
* Ability to handle tabulated electron kinetic coefficients
* https://github.com/pecos/lxcat-review/blob/master/bolsig/Nominal_3Species.py

## BTE code
* Initial comparison with PIC DSMC with PDE code
* **Implement hybrid, fluid & BTE split scheme to reach the steady state faster**
* Further improvements for time integration
  * Fully-implicit scheme to eliminate operator split ? Matrix free RHS and Jacobian action

  


