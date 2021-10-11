"""
@package : simple convergence plots
"""
import matplotlib.pyplot as plt
import spec_spherical
import argparse
import numpy as np
import scipy.constants 
import utils as BEUtils
import collisions

TIME_INDEX=0
MASS_INDEX=1
TEMP_INDEX=3
C000_INDEX=4

f = plt.figure()
f.set_figwidth(10)
f.set_figheight(10)

file_names=["dat/g01_dt_1e-8_Nr_4.dat", "dat/g01_dt_1e-9_Nr_4.dat", "dat/g01_dt_1e-8_Nr_8.dat", "dat/g01_dt_1e-9_Nr_8.dat","dat/g01_dt_1e-8_Nr_16.dat","dat/g01_dt_1e-9_Nr_16.dat"]
plot_label=["G0,G1 (Nr,Nt,Np)=(4,2,1) dt=1e-8","G0,G1 (Nr,Nt,Np)=(4,2,1) dt=1e-9","G0,G1 (Nr,Nt,Np)=(8,2,1) dt=1e-8","G0,G1 (Nr,Nt,Np)=(8,2,1) dt=1e-9","G0,G1 (Nr,Nt,Np)=(16,2,1) dt=1e-8","G0,G1 (Nr,Nt,Np)=(16,2,1) dt=1e-9"]
markers   =[',', '+', '-', '.', 'o', '*']

for i, f in enumerate(file_names):
    data = np.loadtxt(f,skiprows=1)
    plt.plot(data[:,TIME_INDEX],data[:,TEMP_INDEX]*scipy.constants.Boltzmann/collisions.ELECTRON_VOLT,label=plot_label[i],markersize=2,marker='o')

plt.xlabel("time (s)")
plt.ylabel("temperature (eV)")
plt.grid()
plt.legend()
plt.show()



file_names=["dat/g0_dt_1e-8_Nr_8.dat", "dat/g1_dt_1e-8_Nr_8.dat", "dat/g2_dt_1e-8_Nr_8.dat"]
plot_label=["G0 (Nr,Nt,Np)=(8,2,1) dt=1e-8","G1 (Nr,Nt,Np)=(8,2,1) dt=1e-8","G2 (Nr,Nt,Np)=(8,2,1) dt=1e-8"]

for i, f in enumerate(file_names):
    data = np.loadtxt(f,skiprows=1)
    plt.plot(data[:,TIME_INDEX],data[:,TEMP_INDEX]*scipy.constants.Boltzmann/collisions.ELECTRON_VOLT,label=plot_label[i],markersize=2,marker='o')

plt.xlabel("time (s)")
plt.ylabel("temperature (eV)")
plt.grid()
plt.legend()
plt.show()
plt.close()


for i, f in enumerate(file_names):
    data = np.loadtxt(f,skiprows=1)
    plt.plot(data[:,TIME_INDEX],data[:,MASS_INDEX],label=plot_label[i],markersize=2,marker='o')

plt.xlabel("time (s)")
plt.ylabel("number of electrons")
plt.grid()
plt.legend()
plt.show()
plt.close()