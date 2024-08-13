import numpy as np
import matplotlib.pyplot as plt
import plot_utils
from itertools import cycle


idx_range   = range(0, 11, 2)
d1          = plot_utils.load_data_bte("./../1dglow_hybrid/Ewt_10K_Nx100_Nr127_Nvt16_l2_dt_5e-4"  , idx_range , None, read_cycle_avg=True)
d2          = plot_utils.load_data_bte("./../1dglow_hybrid/Ewt_10K_Nx200_Nr255_Nvt32_l4_dt_2e-4"  , idx_range , None, read_cycle_avg=True)
d3          = plot_utils.load_data_bte("./../1dglow_hybrid/Ewt_10K_Nx200_Nr511_Nvt64_l8_dt_1e-4"  , idx_range , None, read_cycle_avg=True)

data        = [d1, d2, d3]
lbl         = [r"$N_r$=128,$N_\theta$=16,$l_{max}=2,dt=5e-4$", r"$N_r$=256,$N_\theta$=32,$l_{max}=4,dt=2e-4$", r"$N_r$=512,$N_\theta$=64,$l_{max}=8,dt=1e-4$"]

fnames      = ["../1dglow_hybrid/Ewt_10K_Nx100_Nr127_l2_dt_5e-4", 
               "../1dglow_hybrid/Ewt_10K_Nx200_Nr255_l4_dt_2e-4", 
               "../1dglow_hybrid/Ewt_10K_Nx200_Nr511_l8_dt_1e-4"]

#xx_grid    = np.linspace(-1, 1, 100)
xx_grid     = plot_utils.op(200).xp

macro_xx    = list()
for d in data:
    u   = d[1]
    Np  = u.shape[1]
    
    op  = plot_utils.op(Np)
    v0  = np.polynomial.chebyshev.chebvander(xx_grid, Np-1)
    p0  = np.dot(v0, op.V0pinv)
    
    macro_xx.append(np.einsum("ik,akc->aic", p0, u))

for idx, cycle_id in enumerate(idx_range):
    plt.figure(figsize=(10, 4), dpi=200)
    for i, u in enumerate(macro_xx):
        ne = u[idx, : , 0]
        ni = u[idx, : , 1]
        Te = u[idx, : , 2]
        
        ne_h = macro_xx[-1][idx, : , 0]
        ni_h = macro_xx[-1][idx, : , 1]
        Te_h = macro_xx[-1][idx, : , 2]
    
        plt.subplot(1, 2, 1)
        plt.plot(xx_grid, ne, label=lbl[i])
        plt.grid(visible=True)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(xx_grid, ne * Te, label=lbl[i])
        plt.grid(visible=True)
        plt.legend()
        
        np.savetxt("%s_%02d.csv"%(fnames[i], cycle_id), np.array([xx_grid, ne, Te]).T, header="x\tne\tTe", delimiter="\t", comments="")
        np.savetxt("%s_rel_error_%02d.csv"%(fnames[i], cycle_id), np.array([xx_grid, np.abs(ne-ne_h)/np.max(np.abs(ne_h)), np.abs(Te-Te_h)/np.max(np.abs(Te_h))]).T, header="x\tne\tTe", delimiter="\t", comments="")
    
    
    plt.show()
    plt.close()


xx_grid = plot_utils.op(200).xp
ev_grid = np.linspace(1e-2, 100, 200)
plt.figure(figsize=(16, 16), dpi=200)
xidx = 10

for idx, d in enumerate(data):
    u   = d[1]
    Np  = u.shape[1]
    
    op  = plot_utils.op(Np)
    v0  = np.polynomial.chebyshev.chebvander(xx_grid, Np-1)
    p0  = np.dot(v0, op.V0pinv)
    
    v   = np.einsum("ik,abk->abi", p0, d[2])
    u   = np.einsum("ik,akc->aic", p0, d[1])
    
    spec_sp, col_list = plot_utils.gen_spec_sp(d[0])
    rr                = plot_utils.compute_radial_components(d[0], d[3], spec_sp, ev_grid, v)
    
    for tidx, cycle_id in enumerate(idx_range):
        np.savetxt("%s_fl_x%.3f_%02d.csv"%(fnames[idx], xx_grid[xidx], cycle_id), np.array([ev_grid, rr[tidx, xidx, 0], rr[tidx, xidx, 1], rr[tidx, xidx, 2]]).T, header="x\tf0\tf1\tf2", delimiter="\t", comments="")
    
    
    for l in range(3):
        plt.subplot(3, 3, 3 * l + 1)    
        plt.semilogy(ev_grid, np.abs(rr[1, xidx, l]), label=lbl[idx])
        
        plt.subplot(3, 3, 3 * l + 2)
        plt.semilogy(ev_grid, np.abs(rr[2, xidx, l]), label=lbl[idx])
        
        plt.subplot(3, 3, 3 * l + 3)
        plt.semilogy(ev_grid, np.abs(rr[4, xidx, l]), label=lbl[idx])
        
    

for l in range(3):    
    plt.subplot(3, 3, 3 * l +1)
    plt.legend()
    plt.grid(visible=True)
    plt.xlabel(r"eV")
    plt.ylabel(r"$f_%d [eV^{-3/2}]$"%(l))
    
    plt.subplot(3, 3, 3 * l +2)
    plt.legend()
    plt.grid(visible=True)
    plt.xlabel(r"eV")
    plt.ylabel(r"$f_%d [eV^{-3/2}]$"%(l))
    
    plt.subplot(3, 3, 3 * l +3)
    plt.legend()
    plt.grid(visible=True)
    plt.xlabel(r"eV")
    plt.ylabel(r"$f_%d [eV^{-3/2}]$"%(l))


plt.show()







