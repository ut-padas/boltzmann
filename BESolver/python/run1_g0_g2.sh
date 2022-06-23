#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Frontera CLX nodes
#
#   *** Serial Job in Small Queue***
# 
# Last revised: 22 June 2021
#
# Notes:
#
#  -- Copy/edit this script as desired.  Launch by executing
#     "sbatch clx.serial.slurm" on a Frontera login node.
#
#  -- Serial codes run on a single node (upper case N = 1).
#       A serial code ignores the value of lower case n,
#       but slurm needs a plausible value to schedule the job.
#
#  -- Use TACC's launcher utility to run multiple serial 
#       executables at the same time, execute "module load launcher" 
#       followed by "module help launcher".
#----------------------------------------------------

#SBATCH -J boltzmann        # Job name
#SBATCH -o .bte.o%j         # Name of stdout output file
#SBATCH -e .bte.e%j         # Name of stderr error file
#SBATCH -p small     # Queue (partition) name
#SBATCH -N 1                # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 24:00:00         # Run time (hh:mm:ss)
##SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A PHY21005         # Project/Allocation name (req'd if you have more than 1)
##SBATCH --mail-user=username@tacc.utexas.edu

# Any other commands must follow all #SBATCH directives...
module list
pwd
date

python3 collision_driver_spherical_eulerian.py -E 1.0e2 -radial_poly bspline -sp_order 1 -spline_qpts 2  -sweep_values 64 128 256 512 1024 2048 -dt 1e-6 -T 1e-4 -q_vt 4 -q_vp 4 -q_st 4 -q_sp 4

# python3 collision_driver_spherical_eulerian.py -E 1.0e0 -radial_poly bspline -sp_order 1 -spline_qpts 2  -sweep_values 32 64 128 -dt 1e-6 -T 4e-4 -q_vt 4 -q_vp 4 -q_st 4 -q_sp 4
# python3 collision_driver_spherical_eulerian.py -E 1.0e1 -radial_poly bspline -sp_order 1 -spline_qpts 2  -sweep_values 32 64 128 -dt 1e-6 -T 4e-4 -q_vt 4 -q_vp 4 -q_st 4 -q_sp 4
# python3 collision_driver_spherical_eulerian.py -E 1.0e2 -radial_poly bspline -sp_order 1 -spline_qpts 2  -sweep_values 32 64 128 -dt 1e-6 -T 1e-4 -q_vt 4 -q_vp 4 -q_st 4 -q_sp 4
# python3 collision_driver_spherical_eulerian.py -E 5.0e2 -radial_poly bspline -sp_order 1 -spline_qpts 2  -sweep_values 32 64 128 -dt 1e-6 -T 1e-4 -q_vt 4 -q_vp 4 -q_st 4 -q_sp 4
# python3 collision_driver_spherical_eulerian.py -E 1.0e3 -radial_poly bspline -sp_order 1 -spline_qpts 2  -sweep_values 32 64 128 -dt 1e-6 -T 1e-4 -q_vt 4 -q_vp 4 -q_st 4 -q_sp 4
# python3 collision_driver_spherical_eulerian.py -E 5.0e3 -radial_poly bspline -sp_order 1 -spline_qpts 2  -sweep_values 32 64 128 -dt 1e-6 -T 1e-4 -q_vt 4 -q_vp 4 -q_st 4 -q_sp 4
# python3 collision_driver_spherical_eulerian.py -E 1.0e4 -radial_poly bspline -sp_order 1 -spline_qpts 2  -sweep_values 32 64 128 -dt 1e-6 -T 1e-4 -q_vt 4 -q_vp 4 -q_st 4 -q_sp 4
