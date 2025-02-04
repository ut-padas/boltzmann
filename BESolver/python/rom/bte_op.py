"""
@brief attempt to compress the BTE operators. 
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.constants
import os
import scipy.interpolate

import sys
sys.path.append("../.")
sys.path.append("../plot_scripts")
import basis
import cross_section
import utils as bte_utils
import spec_spherical as sp
import collisions
import plot_utils
import rom_utils

folder_name = "../1dbte_rom/ex2"
args        = rom_utils.load_run_args("%s/1d_bte_args.txt"%(folder_name))

Cvmat        = np.load("%s/1d_bte_bte_cmat.npy"%(folder_name))
Avmat        = np.load("%s/1d_bte_bte_emat.npy"%(folder_name))
Axmat        = np.load("%s/1d_bte_bte_xmat.npy"%(folder_name))

svd_v        = [np.linalg.svd(M) for M in [Cvmat, Avmat]]

plt.figure(figsize=(4, 4), dpi=200)
plt.semilogy(svd_v[0][1]/svd_v[0][1][0], label=r"$C_v$")
plt.semilogy(svd_v[1][1]/svd_v[1][1][0], label=r"$A_v$")
plt.ylabel(r"singular value")
plt.xlabel(r"singular id")
plt.legend()
plt.grid(visible=True)
plt.tight_layout()
#plt.show()
plt.savefig("v-op-svd.png")
plt.close()