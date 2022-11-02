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

#SBATCH -J 1d3v_glow           # Job name
#SBATCH -o .glow_1d3v.o%j       # Name of stdout output file
#SBATCH -e .glow_1d3v.e%j       # Name of stderr error file
#SBATCH -p rtx            # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 24:00:00        # Run time (hh:mm:ss)
##SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A PHY21005       # Project/Allocation name (req'd if you have more than 1)
##SBATCH --mail-user=username@tacc.utexas.edu

# Any other commands must follow all #SBATCH directives...
module list
pwd
date

T=3.687315634218289e-07
v0_z0=100
num_z=400

# python3 bte_glow_discharge_1d.py -T $T -dt 1e-12 -c g0 g2 -E 0 -ev 20 -Nr 128 -num_z $num_z -q_vt 3 -v0_z0 $v0_z0 -ne 1e0 -gpu 1 -benchmark 0 -ne_fac 8e16 -eta_z 0 -l_max 1 -vx_max 11.11 -device_id 0 &
# python3 bte_glow_discharge_1d.py -T $T -dt 1e-12 -c g0 g2 -E 0 -ev 20 -Nr 128 -num_z $num_z -q_vt 4 -v0_z0 $v0_z0 -ne 1e0 -gpu 1 -benchmark 0 -ne_fac 8e16 -eta_z 0 -l_max 2 -vx_max 11.11 -device_id 1 &
# python3 bte_glow_discharge_1d.py -T $T -dt 1e-12 -c g0 g2 -E 0 -ev 20 -Nr 128 -num_z $num_z -q_vt 5 -v0_z0 $v0_z0 -ne 1e0 -gpu 1 -benchmark 0 -ne_fac 8e16 -eta_z 0 -l_max 3 -vx_max 11.11 -device_id 2 &
# python3 bte_glow_discharge_1d.py -T $T -dt 1e-12 -c g0 g2 -E 0 -ev 20 -Nr 128 -num_z $num_z -q_vt 6 -v0_z0 $v0_z0 -ne 1e0 -gpu 1 -benchmark 0 -ne_fac 8e16 -eta_z 0 -l_max 4 -vx_max 11.11 -device_id 3 

# wait

# num_z=800
# python3 bte_glow_discharge_1d.py -T $T -dt 1e-12 -c g0 g2 -E 0 -ev 20 -Nr 128 -num_z $num_z -q_vt 3 -v0_z0 $v0_z0 -ne 1e0 -gpu 1 -benchmark 0 -ne_fac 8e16 -eta_z 0 -l_max 1 -vx_max 11.11 -device_id 0 &
# python3 bte_glow_discharge_1d.py -T $T -dt 1e-12 -c g0 g2 -E 0 -ev 20 -Nr 128 -num_z $num_z -q_vt 4 -v0_z0 $v0_z0 -ne 1e0 -gpu 1 -benchmark 0 -ne_fac 8e16 -eta_z 0 -l_max 2 -vx_max 11.11 -device_id 1 &
# python3 bte_glow_discharge_1d.py -T $T -dt 1e-12 -c g0 g2 -E 0 -ev 20 -Nr 128 -num_z $num_z -q_vt 5 -v0_z0 $v0_z0 -ne 1e0 -gpu 1 -benchmark 0 -ne_fac 8e16 -eta_z 0 -l_max 3 -vx_max 11.11 -device_id 2 &
# python3 bte_glow_discharge_1d.py -T $T -dt 1e-12 -c g0 g2 -E 0 -ev 20 -Nr 128 -num_z $num_z -q_vt 6 -v0_z0 $v0_z0 -ne 1e0 -gpu 1 -benchmark 0 -ne_fac 8e16 -eta_z 0 -l_max 4 -vx_max 11.11 -device_id 3 

# wait


python3 bte_glow_discharge_1d.py -T $T -dt 1e-12 -c g0 g2 -E 0 -ev 20 -Nr 128 -num_z 400 -q_vt 7 -v0_z0 100 -ne 1e0 -gpu 1 -benchmark 0 -ne_fac 8e16 -eta_z 0 -l_max 5 -vx_max 11.11 -device_id 0 &
python3 bte_glow_discharge_1d.py -T $T -dt 1e-12 -c g0 g2 -E 0 -ev 20 -Nr 128 -num_z 400 -q_vt 8 -v0_z0 100 -ne 1e0 -gpu 1 -benchmark 0 -ne_fac 8e16 -eta_z 0 -l_max 6 -vx_max 11.11 -device_id 1 &
python3 bte_glow_discharge_1d.py -T $T -dt 1e-12 -c g0 g2 -E 0 -ev 20 -Nr 128 -num_z 800 -q_vt 7 -v0_z0 100 -ne 1e0 -gpu 1 -benchmark 0 -ne_fac 8e16 -eta_z 0 -l_max 5 -vx_max 11.11 -device_id 2 &
python3 bte_glow_discharge_1d.py -T $T -dt 1e-12 -c g0 g2 -E 0 -ev 20 -Nr 128 -num_z 800 -q_vt 8 -v0_z0 100 -ne 1e0 -gpu 1 -benchmark 0 -ne_fac 8e16 -eta_z 0 -l_max 6 -vx_max 11.11 -device_id 3 

wait