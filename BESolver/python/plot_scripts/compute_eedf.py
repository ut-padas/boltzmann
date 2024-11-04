import numpy as np
import plot_utils
import sys
import matplotlib.pyplot as plt
import h5py

# which folder to analyze
print("args: ", sys.argv)    
folder_name = sys.argv[1] 
file_idx    = int(sys.argv[2])

data = plot_utils.load_data_bte(folder_name, [file_idx],None, False, use_ionization=1) 

args   = data[0]
u      = data[1]
v      = data[2]
bte_op = data[3]

# ev_grid           = np.logspace(-3, 2, 512)
ev_grid           = np.linspace(0, 50, 1000)
spec_sp, col_list = plot_utils.gen_spec_sp(args)
ff_r              = plot_utils.compute_radial_components(args, bte_op, spec_sp, ev_grid, v)


Np=u.shape[1]
checb = plot_utils.op(Np)

with h5py.File("%s/eedf.h5"%(folder_name), 'w') as F: 
    F.create_dataset("x[-1,1]"             , data = checb.xp)
    F.create_dataset("ev[eV]"              , data = ev_grid)
    F.create_dataset("f0[eV^-1.5]"         , data = ff_r[0][:, 0, :])
    F.create_dataset("f1[eV^-1.5]"         , data = ff_r[0][:, 1, :])
    F.create_dataset("f2[eV^-1.5]"         , data = ff_r[0][:, 2, :])
    F.close()
    
# F = h5py.File("%s/eedf.h5"%(folder_name), 'r')
# plt.semilogy(F["ev[eV]"][()], F["f0[eV^-1.5]"][()][0::30].T)
# plt.semilogy(F["ev[eV]"][()], np.abs(F["f1[eV^-1.5]"][()][0::30].T))
# plt.semilogy(F["ev[eV]"][()], np.abs(F["f2[eV^-1.5]"][()][0::30].T))
# plt.grid()
# plt.show()



