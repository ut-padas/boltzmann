import numpy as np
import matplotlib.pyplot as plt
import plot_utils
import h5py
from matplotlib.colors import LogNorm
from matplotlib.colors import Colormap
import os
import sys
sys.path.append("../.")
import basis
import cross_section
import utils as bte_utils
import spec_spherical as sp
import collisions
import scipy.constants

def make_dir(dir_name):
    # Check whether the specified path exists or not
    isExist = os.path.exists(dir_name)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(dir_name)
       print("directory %s is created!"%(dir_name))

def load_run_args(fname):
    args   = dict()
    
    f  = open(fname)
    st = f.read().strip()
    st = st.split(":")[1].strip()
    st = st.split("(")[1]
    st = st.split(")")[0].strip()
    st = st.split(",")

    for s in st:
        kv = s.split("=")
        if len(kv)!=2 or kv[0].strip()=="collisions":
            continue
        args[kv[0].strip()]=kv[1].strip().replace("'", "")
    return args

def compute_radial_components(args, bte_op, spec_sp, ev: np.array, ff):
    t_pts    = ff.shape[0]
    
    ff_lm    = ff#np.array([np.dot(bte_op["po2sh"], ff[idx]) for idx in range(t_pts)])
    Te       = (float)(args["Te"])
    vth      = collisions.electron_thermal_velocity(Te * (scipy.constants.elementary_charge / scipy.constants.Boltzmann))
    c_gamma  = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))
    
    vth      = vth
    spec_sp  = spec_sp
    
    vr       = np.sqrt(ev) * c_gamma / vth
    num_p    = spec_sp._p +1 
    num_sh   = len(spec_sp._sph_harm_lm)
    n_pts    = ff.shape[2]
    
    output   = np.zeros((t_pts, n_pts, num_sh, len(vr)))
    Vqr      = spec_sp.Vq_r(vr,0,1)
    
    
    mm_op    = bte_op["mass"]
    mm_fac   = np.sqrt(4 * np.pi) 
    
    #scale    = np.array([np.dot(mm_op / mm_fac, ff_lm[idx]) * (2 * (vth/c_gamma)**3) for idx in range(t_pts)])
    ff_lm_n  = ff_lm #np.array([ff_lm[idx]/scale[idx] for idx in range(t_pts)])
    
    #print(scale)

    for idx in range(t_pts):
        ff_lm_T  = ff_lm_n[idx].T
        for l_idx, lm in enumerate(spec_sp._sph_harm_lm):
            output[idx, :, l_idx, :] = np.dot(ff_lm_T[:,l_idx::num_sh], Vqr)

    return output

def fft_reconstruct(qoi, axis, tt, dt, k, fprefix, is_real=True):
    assert is_real == True
    fqoi           = np.fft.rfft(qoi, axis=axis)
    freq           = np.fft.rfftfreq(len(tt), d=dt)
    

    plt.figure(figsize=(8, 2), dpi=300)
    plt.xlabel(r"frequency")
    plt.ylabel(r"magnitude")
    plt.semilogy(freq, np.linalg.norm(np.real(fqoi), axis=1), 'o-'  , markersize=1.8, label=r"$cos(\omega_k t)$")
    plt.semilogy(freq, np.linalg.norm(np.imag(fqoi), axis=1), 'o--' , markersize=1.8, label=r"$sin(\omega_k t)$")
    plt.grid(visible=True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s_fft.png"%(fprefix))
    plt.close()

    plt.figure(figsize=(12, 8), dpi=300)
    for i in range(1, k):
        plt.subplot(3, 4, i)
        sf = 1
        if i == 1:
            sf = 1/len(tt)
        else:
            sf = 2/len(tt)


        plt.plot(xx, sf * np.real(fqoi[i-1,:]), label=r"cos($\omega_{%d}$ t)"%(i-1))
        plt.plot(xx, sf * np.imag(fqoi[i-1,:]), label=r"sin($\omega_{%d}$ t)"%(i-1))
        plt.legend()
        plt.grid(visible=True)
        plt.title(r"freq = %.4E"%(freq[i-1]))
        plt.ylabel(r"E [V/m]")
        plt.xlabel(r"$\hat{x}$")
    
    plt.tight_layout()
    plt.savefig("%s_modes.png"%(fprefix))
    plt.close()

    fqoi[k:] = 0.0
    qoi_r    = np.fft.irfft(fqoi, axis=axis)

    #print(qoi_r.shape, qoi.shape)

    dt            = 0.5 * (tt[1] - tt[0])
    dx            = 0.5 * (xx[1] - xx[0])
    extent        = [xx[0]-dx , xx[-1] + dx, tt[0]-dt , tt[-1] + dt]


    plt.figure(figsize=(12, 4), dpi=300)
    plt.subplot(1, 3, 1)
    plt.imshow(qoi, aspect='auto', extent=extent)
    plt.colorbar()
    plt.xlabel(r"$\hat{x}$")
    plt.ylabel(r"$time [T]$")
    plt.title(r"E(x,t)")

    plt.subplot(1, 3, 2)
    plt.imshow(qoi_r, aspect='auto', extent=extent)
    plt.colorbar()
    plt.xlabel(r"$\hat{x}$")
    plt.ylabel(r"$time [T]$")
    plt.title(r"$\sum_{k=0}^{10} a_k(x) cos(k \omega t) + b_k(x) sin(k \omega t)$")

    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(1 - qoi_r/qoi), aspect='auto', extent=extent, norm=LogNorm(vmin=1e-10, vmax=1))
    plt.colorbar()
    plt.xlabel(r"$\hat{x}$")
    plt.ylabel(r"$time [T]$")
    plt.title(r"relative error")

    plt.tight_layout()
    #plt.show()
    plt.savefig("%s_fft_recons.png"%(fprefix))
    plt.close()

def fft_edf(qoi, axis, tt, dt, k, ev, fprefix):
    fqoi           = np.fft.rfft(qoi, axis=axis)
    freq           = np.fft.rfftfreq(len(tt), d=dt)
    

    plt.figure(figsize=(8, 2), dpi=300)
    plt.xlabel(r"frequency")
    plt.ylabel(r"magnitude")
    plt.semilogy(freq, np.linalg.norm(np.real(fqoi), axis=(1, 2)), 'o-'  , markersize=1.8, label=r"$cos(\omega_k t)$")
    plt.semilogy(freq, np.linalg.norm(np.imag(fqoi), axis=(1, 2)), 'o--' , markersize=1.8, label=r"$sin(\omega_k t)$")
    plt.grid(visible=True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("%s_fft.png"%(fprefix))
    plt.close()

    a = compute_radial_components(args, bte_op, spec_sp, ev, np.real(fqoi))
    b = compute_radial_components(args, bte_op, spec_sp, ev, np.imag(fqoi))

    print(a.shape)

    plt.figure(figsize=(12, 8), dpi=300)
    for i in range(1, k):
        plt.subplot(3, 4, i)
        sf = 1
        if i == 1:
            sf = 1/len(tt)
        else:
            sf = 2/len(tt)


        

        plt.semilogy(ev, np.abs(a[i-1, 50, 0, :].T), label=r"cos($\omega_{%d}$ t)"%(i-1))
        plt.semilogy(ev, np.abs(b[i-1, 50, 0, :].T), label=r"sin($\omega_{%d}$ t)"%(i-1))

        # plt.plot(xx, sf * np.real(fqoi[i-1,:]), label=r"cos($\omega_{%d}$ t)"%(i-1))
        # plt.plot(xx, sf * np.imag(fqoi[i-1,:]), label=r"sin($\omega_{%d}$ t)"%(i-1))
        plt.legend()
        plt.grid(visible=True)
        plt.title(r"freq = %.4E"%(freq[i-1]))
        plt.ylabel(r"E [V/m]")
        plt.xlabel(r"$\hat{x}$")
    
    plt.tight_layout()
    plt.savefig("%s_modes.png"%(fprefix))
    plt.close()




    

    
    

folder_name_ops = "../1dglow_hybrid_Nx400/1Torr300K_100V_Ar_3sp2r"
folder_name = "../1dglow_hybrid_Nx400/1Torr300K_100V_Ar_3sp2r_cycle"
args        = load_run_args("%s/1d_glow_args.txt"%(folder_name))
ff          = h5py.File("%s/macro.h5"%(folder_name), 'r')

mop         = np.load("%s/1d_glow_bte_mass_op.npy"%(folder_name_ops))
Po          = np.load("%s/1d_glow_bte_psh2o.npy"%(folder_name_ops))
Ps          = np.load("%s/1d_glow_bte_po2sh.npy"%(folder_name_ops))
g0          = np.load("%s/1d_glow_bte_op_g0.npy"%(folder_name_ops))
g2          = np.load("%s/1d_glow_bte_op_g2.npy"%(folder_name_ops))
temp_op     = np.load("%s/1d_glow_bte_temp_op.npy"%(folder_name_ops))
bte_op      = {"mass": mop , "po2sh": Ps, "psh2o": Po}

spec_sp, col_list = plot_utils.gen_spec_sp(args)

#print(ff.keys())
Et          = np.array(ff["E[Vm^-1]"][()])
xx          = np.array(ff["x[-1,1]"][()])
tt          = np.array(ff["time[T]"][()])
ne          = np.array(ff["ne[m^-3]"][()])
Te          = np.array(ff["Te[eV]"][()])
fvtx        = np.array(ff["Ftvx"][()])

make_dir("%s/fft"%(folder_name))
fft_reconstruct(Et[:-1, :], 0, tt[:-1], (1e-3)/13.56e6, 13, "%s/fft/Et"%(folder_name))
fft_reconstruct(ne[:-1, :], 0, tt[:-1], (1e-3)/13.56e6, 13, "%s/fft/ne"%(folder_name))
fft_reconstruct(Te[:-1, :], 0, tt[:-1], (1e-3)/13.56e6, 13, "%s/fft/Te"%(folder_name))


fft_edf(fvtx[:-1], 0, tt[:-1], (1e-3)/13.56e6, 13, np.linspace(0, 80, 1024), "%s/fft/edf"%(folder_name))








    





    













# def glowfluid():
#     ut         = np.load("ut.npy")
#     tt         = np.linspace(0, 1, ut.shape[-1])
#     ut         = ut.reshape((glow_1d.Np, glow_1d.Nv, ut.shape[-1]))
    
#     plt.figure(figsize=(10, 8), dpi=300)
#     for i in range(40, 161, 20):
#         plt.plot(tt, ut[i, 0, :]     ,      label=r"$n_e$, x=%.4E"%(glow_1d.xp[i]))
#         plt.plot(tt, ut[i, 1, :],'--',      label=r"$n_i$, x=%.4E"%(glow_1d.xp[i]))
#         plt.grid(visible=True)
#         plt.xlabel(r"time")
#         plt.legend()
    
#     plt.savefig("test1.png")
#     plt.close()
    
#     plt.figure(figsize=(10, 8), dpi=300)
#     for i in range(0, 40, 5):
#         plt.semilogy(tt, ut[i, 0, :],      label=r"$n_e$, x=%.4E"%(glow_1d.xp[i]))
#         plt.semilogy(tt, ut[i, 1, :],'--', label=r"$n_i$, x=%.4E"%(glow_1d.xp[i]))
#         plt.grid(visible=True)
#         plt.xlabel(r"time")
#         plt.legend()
        
#     plt.savefig("test2.png")
#     plt.close()
    
    
#     plt.figure(figsize=(10, 8), dpi=300)
#     for i in range(40, 161, 20):
#         plt.plot(tt, ut[i, 2, :] / ut[i, 0, :],      label=r"$T_e$, x=%.4E"%(glow_1d.xp[i]))
#         plt.grid(visible=True)
#         plt.xlabel(r"time")
#         plt.legend()
        
#     plt.savefig("test3.png")
#     plt.close()
    
#     plt.figure(figsize=(10, 8), dpi=300)
#     for i in range(0, 40, 5):
#         plt.plot(tt, ut[i, 2, :] / ut[i, 0, :],      label=r"$T_e$, x=%.4E"%(glow_1d.xp[i]))
#         plt.grid(visible=True)
#         plt.xlabel(r"time")
#         plt.legend()
        
#     plt.savefig("test4.png")
#     plt.close()
    
#     from matplotlib.colors import LogNorm
#     freq    = np.fft.fftfreq(ut.shape[-1], d=1e-3 * glow_1d.param.tau)
#     idx     = freq.argsort()
#     ele_fft = np.fft.fft(ut[:, 0])[:, idx]          #np.array([np.fft.fft(ut[i,0])[idx] for i in range(glow_1d.Np)]).reshape((glow_1d.Np, -1))
#     ion_fft = np.fft.fft(ut[:, 1])[:, idx]          #np.array([np.fft.fft(ut[i,1])[idx] for i in range(glow_1d.Np)]).reshape((glow_1d.Np, -1))
#     Te_fft  = np.fft.fft(ut[:, 2]/ut[:, 0])[:, idx] #np.array([np.fft.fft(ut[i,2]/ut[i,0])[idx] for i in range(glow_1d.Np)]).reshape((glow_1d.Np, -1))
    
#     plt.figure(figsize=(12,12), dpi=100)
#     plt.subplot(3, 1, 1)
#     plt.imshow(np.abs(ele_fft), norm=LogNorm(vmin=1e-6, vmax=1e3), cmap='jet', aspect='auto', interpolation='none', extent=[np.min(freq),np.max(freq), -1,1])
#     plt.colorbar()
#     plt.ylabel(r"x")
#     plt.xlabel(r"freq")
#     plt.title(r"$n_e$")
    
#     plt.subplot(3, 1, 2)
#     plt.imshow(np.abs(ion_fft), norm=LogNorm(vmin=1e-6, vmax=1e3), cmap='jet', aspect='auto', interpolation='none', extent=[np.min(freq),np.max(freq), -1,1])
#     plt.colorbar()
#     plt.ylabel(r"x")
#     plt.xlabel(r"freq")
#     plt.title(r"$n_i$")
    
#     plt.subplot(3, 1, 3)
#     plt.imshow(np.abs(Te_fft), norm=LogNorm(vmin=1e-6, vmax=1e3), cmap='jet', aspect='auto', interpolation='none', extent=[np.min(freq),np.max(freq), -1,1])
#     plt.colorbar()
#     plt.ylabel(r"x")
#     plt.xlabel(r"freq")
#     plt.title(r"$T_e$")
#     plt.tight_layout()
#     plt.savefig("test5.png")
#     plt.close()
    
#     xx=glow_1d.xp
#     for i in range(0, 4):
#         plt.semilogy(freq[idx], np.abs(ele_fft[i]), label=r"x=%.4E"%(xx[i]))
        
#     plt.xlabel(r"freq")
#     plt.ylabel(r"mag")
#     plt.legend()
#     plt.savefig("test.png")
#     plt.close()