#!/bin/bash

#SBATCH --time=72:00:00 # walltime, abbreviated by -t
#SBATCH --nodes=1 # number of cluster nodes, abbreviated by -N
#SBATCH -o /scratch/kingspeak/serial/u1011531/.boltzmann-%j.out-%N # name of the stdout, using the job number (%j) and the first node (%N)
#SBATCH --ntasks=40 # number of MPI tasks, abbreviated by -n # additional information for allocated clusters
#SBATCH --account=soc-np # account - abbreviated by -A
#SBATCH --partition=soc-np # partition, abbreviated by -p # # set data and working directories


#cd /uufs/chpc.utah.edu/common/home/u1011531/Research/Dendro-5.0/build
#make bssnSolver -j28
#cp BSSN_GR/bssnSolver /scratch/kingspeak/serial/u1011531/bssn3/r1/run26_rit/.
#cp rit.par.json /scratch/kingspeak/serial/u1011531/bssn3/r1/run26_rit/.
#cd /scratch/kingspeak/serial/u1011531/bssn3/r1/run26_rit/

#make nlsmSolverNUTS -j4

#mpirun -np 224 ./NLSigma/nlsmSolverNUTS nlsmA_non_linear_d8.par.json 3
#mpirun -np 224 ./NLSigma/nlsmSolverNUTS nlsmA_non_linear_d10.par.json 3
#mpirun -np $SLURM_NTASKS ./NLSigma/nlsmSolverNUTS nlsmA_nuts.par.json 0

pwd
date
module list

#mpirun -np 16 python3 collision_driver_spherical.py -Te 1e-6 -o /scratch/kingspeak/serial/u1011531/cs_analytical_1ev -c g0
mpirun -np 5 python3 collision_driver_spherical.py -Te 1e-6 -o /scratch/kingspeak/serial/u1011531/dat_1ev_no_proj -c g0
