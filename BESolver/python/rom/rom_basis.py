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

def eval_fl(spec_sp: sp.SpectralExpansionSpherical, fl:np.array, vr:np.array):
    Vqr = spec_sp.Vq_r(vr, l=0, scale=1)
    fvr = np.dot(Vqr.T, fl)
    return fvr

def compute_radial_comp(args, spec_sp: sp.SpectralExpansionSpherical, mm_op, ff: np.array, ev:np.array):
    Te       = (float)(args["Te"])
    vth      = collisions.electron_thermal_velocity(Te * (scipy.constants.elementary_charge / scipy.constants.Boltzmann))
    c_gamma  = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))
    
    vth      = vth
    spec_sp  = spec_sp
    
    vr       = np.sqrt(ev) * c_gamma / vth
    num_p    = spec_sp._p +1 
    num_sh   = len(spec_sp._sph_harm_lm)
    n_pts    = ff.shape[2]
    t_pts    = ff.shape[0]
    
    output   = np.zeros((t_pts, n_pts, num_sh, len(vr)))
    Vqr      = spec_sp.Vq_r(vr,0,1)
    
    mm_fac   = np.sqrt(4 * np.pi) 
    
    scale    = np.array([np.dot(mm_op / mm_fac, ff[idx]) * (2 * (vth/c_gamma)**3) for idx in range(t_pts)])
    ff_lm_n  = np.array([ff[idx]/scale[idx] for idx in range(t_pts)])
    
    for idx in range(t_pts):
        ff_lm_T  = ff_lm_n[idx].T
        for l_idx, lm in enumerate(spec_sp._sph_harm_lm):
            output[idx, :, l_idx, :] = np.dot(ff_lm_T[:,l_idx::num_sh], Vqr)

    return output

def construct_rom_basis(fl, eps_x, eps_v):
    num_t, num_p, num_x = fl.shape

    Ux, Sx, Vhx = np.linalg.svd(fl.reshape(num_t * num_p, -1)) # Nt Nr x Nx
    Uv, Sv, Vhv = np.linalg.svd(np.swapaxes(fl, 0, 1).reshape((num_p, num_t * num_x))) # Nr x Nt Nx

    Uv  = Uv [:, Sv > Sv[0] * eps_v]
    Vx  = Vhx[Sx > Sx[0] * eps_x, :].T
    
    # #plt.figure(figsize=(4, 4), dpi=300)
    # plt.semilogy(Sx/Sx[0], label=r"$\sigma \text { of } N_t N_r  \times N_x$")
    # plt.semilogy(Sv/Sv[0], label=r"$\sigma \text { of } N_r  \times N_t N_x$")
    # plt.title(r"$N_t = 101 \ N_x = 400 \ N_r =256$")
    # plt.grid(visible=True)
    # plt.ylabel(r"singular value")
    # plt.legend()
    # plt.show()
    # plt.close()
    return Uv, Vx, (Sv , Sx)

folder_name = "../1dglow_hybrid_Nx400/1Torr300K_100V_Ar_3sp2r_cycle"
args        = rom_utils.load_run_args("%s/1d_glow_args.txt"%(folder_name))
ff          = h5py.File("%s/macro.h5"%(folder_name), 'r')
print("args", args, "ff", ff.keys())
c_gamma     = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))
vth         = (float) (args["Te"])**0.5 * c_gamma
num_sh      = (int) (args["l_max"]) + 1
spec_sp,_   = plot_utils.gen_spec_sp(args)
mm_op       = bte_utils.mass_op(spec_sp, scale=1)
ev          = np.linspace(0, 80, 1000)
vr          = np.sqrt(ev) * c_gamma / vth
mm_fac      = np.sqrt(4 * np.pi)
eedf_const  = 2 * (vth/c_gamma)**3  / mm_fac  
np0         = 8e16
ftvx        = np.array(ff["Ftvx"][()]) 
ne          = np.array(ff["ne[m^-3]"][()]) / np0
ftvx        = np.einsum("ik,ivk->ivk", ne, ftvx * eedf_const)
xx          = np.array(ff["x[-1,1]"][()])
tt          = np.array(ff["time[T]"][()])
eps_x       = 1e-4
eps_v       = 1e-6
ttk         = 4

Uv = list()
Vx = list()
S  = list()

for l in range(num_sh):
    u, v, s = construct_rom_basis(ftvx[0::ttk, l::num_sh, :], eps_x, eps_v)
    Uv.append(u)
    Vx.append(v)
    S.append(s)

dofs_lm  = np.array([Uv[l].shape[1] * Vx[l].shape[1] for l in range(num_sh)])
dofs_rom = np.sum(dofs_lm)

num_domains = 3
xsplit      = np.array([-1, -0.75 , 0.75, 1.0 + 1e-8]) #np.linspace(-1, 1 + 1e-6, num_domains+1)
xidx        = [np.arange(len(xx))[np.logical_and(xx>=xsplit[i], xx < xsplit[i+1])] for i in range(num_domains)]


Uv_md = list()
Vx_md = list()
S_md  = list()

for i in range(num_domains):
    Uv_mdl = list()
    Vx_mdl = list()
    S_mdl  = list()

    for l in range(num_sh):
        u, v, s = construct_rom_basis(ftvx[0::ttk, l::num_sh, xidx[i]], eps_x, eps_v)
        Uv_mdl.append(u)
        Vx_mdl.append(v)
        S_mdl .append(s)

    Uv_md.append(Uv_mdl)
    Vx_md.append(Vx_mdl)
    S_md .append(S_mdl)


dofs_lm_md  = np.array([Uv_md[i][l].shape[1] * Vx_md[i][l].shape[1] for i in range(num_domains) for l in range(num_sh)])
dofs_rom_md = np.sum(dofs_lm_md)
dofs_fom    = (spec_sp._p + 1) * num_sh * len(xx)


plt.figure(figsize=(13, 8), dpi=200)

for l in range(num_sh):
    plt.subplot(2, num_sh, l+1)
    plt.semilogy(S[l][0]/S[l][0][0], label=r"full-$\sigma_v$")
    
    for i in range(num_domains):
        plt.semilogy(S_md[i][l][0]/S_md[i][l][0][0], '--' ,label=r"$\sigma_v(x\in[%.4f, %.4f])$"%(xsplit[i], xsplit[i+1]))
    
    plt.legend()
    plt.grid(visible=True)
    plt.ylabel(r"singular value")
    plt.xlabel(r"singular id")
    plt.title(r"$f_%d$ mode"%(l))

for l in range(num_sh):
    plt.subplot(2, num_sh, num_sh + l+1)
    plt.semilogy(S[l][1]/S[l][1][0], label=r"full-$\sigma_x$")

    for i in range(num_domains):
        plt.semilogy(S_md[i][l][1]/S_md[i][l][1][0], '--' ,label=r"$\sigma_x(x\in[%.4f, %.4f])$"%(xsplit[i], xsplit[i+1]))

    
    plt.legend()
    plt.grid(visible=True)
    plt.ylabel(r"singular value")
    plt.xlabel(r"singular id")
    plt.title(r"$f_%d$ mode"%(l))

plt.suptitle(r"%s"%(folder_name))
plt.tight_layout()
#plt.show()
plt.savefig("multi-domain_vs_full_domain.png")
plt.close()

print("dofs_rom / dof_fom = %.4E | dofs_rom_md / dof_fom = %.4E " %((dofs_rom/dofs_fom), (dofs_rom_md/dofs_fom)))







# fidx        = [0, 10, 11]
# data        = [h5py.File("%s/1d_bte_rom_%02d.h5"%(folder_name,i), 'r') for i in fidx]
# #data[1]     = h5py.File("%s/1d_bte_rom_%02d_1.h5"%(folder_name,11), 'r')
# def get_basis_x(i, l):
#     return np.array(data[i]["Vx_%02d"%(l)][()])

# def get_basis_v(i, l):
#     return np.array(data[i]["Uv_%02d"%(l)][()])

# # assert (np.array(data[0]["Uv_00"][()])==np.array(data[1]["Uv_00"][()])).all() == True
# # assert (np.array(data[0]["Uv_01"][()])==np.array(data[1]["Uv_01"][()])).all() == True
# # assert (np.array(data[0]["Vx_00"][()])==np.array(data[1]["Vx_00"][()])).all() == True
# # assert (np.array(data[0]["Vx_01"][()])==np.array(data[1]["Vx_01"][()])).all() == True

# # plt.subplot(1, 2, 1)
# # plt.semilogy(np.abs(get_basis_v(1, 1)))
# # plt.subplot(1, 2, 2)
# # plt.semilogy(np.abs(get_basis_v(2, 1)))
# # plt.show()




# c_gamma   = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))
# vth       = (float) (args["Te"])**0.5 * c_gamma
# num_sh    = (int) (args["l_max"]) + 1
# spec_sp,_ = plot_utils.gen_spec_sp(args)
# mm_op     = bte_utils.mass_op(spec_sp, scale=1)
# ev        = np.linspace(0, 80, 1000)
# vr        = np.sqrt(ev) * c_gamma / vth

# fs        = [np.load("%s/1d_bte_v_all_lm_tb_0.00E+00_to_te_1.00E+00.npy"%(folder_name)),
#              np.load("%s/1d_bte_v_all_lm_tb_1.00E+01_to_te_1.10E+01.npy"%(folder_name))]

# fs_eedf   = [np.abs(compute_radial_comp(args, spec_sp, mm_op, f, ev)) for f in fs]
# fvr_1     = [eval_fl(spec_sp, get_basis_v(0, l), vr) for l in range(num_sh)] 

# # xidx      = 0
# # plt.figure(figsize=(10, 4), dpi=200)
# # plt.subplot(2, 2, 1)
# # plt.semilogy(ev, fs_eedf[0][0::3, xidx, 0].T)
# # plt.grid(visible=True)
# # plt.xlabel(r"ev")
# # plt.title(r"f_0 (old)")

# # plt.subplot(2, 2, 2)
# # plt.semilogy(ev, fs_eedf[0][0::3, xidx, 1].T)
# # plt.grid(visible=True)
# # plt.xlabel(r"ev")
# # plt.title(r"f_1 (old)")

# # plt.subplot(2, 2, 3)
# # plt.semilogy(ev, fs_eedf[1][0::3, xidx, 0].T)
# # plt.grid(visible=True)
# # plt.xlabel(r"ev")
# # plt.title(r"$f_0$ (new)")

# # plt.subplot(2, 2, 4)
# # plt.semilogy(ev, fs_eedf[1][0::3, xidx, 1].T)
# # plt.grid(visible=True)
# # plt.xlabel(r"ev")
# # plt.title(r"$f_1$ (new)")

# # plt.tight_layout()
# # plt.show()

# # sys.exit(0)

# # fvr_1     = [eval_fl(spec_sp, get_basis_v(0, l), vr) for l in range(num_sh)] 
# # fvr_2     = [eval_fl(spec_sp, get_basis_v(2, l), vr) for l in range(num_sh)]

# # kidx      = list(range(0, 4))

# # plt.subplot(1, 2, 1)
# # for k in kidx:
# #     plt.semilogy(ev, np.abs(fvr_1[0][:, k]), label="old (k=%d)"%(k))
# #     plt.semilogy(ev, np.abs(fvr_2[0][:, k]),'--', label="new (k=%d)"%(k))
# # plt.grid(visible=True)
# # plt.legend()

# # plt.subplot(1, 2, 2)
# # for k in kidx:
# #     plt.semilogy(ev, np.abs(fvr_1[1][:, k]), label="old (k=%d)"%(k))
# #     plt.semilogy(ev, np.abs(fvr_2[1][:, k]),'--', label="new (k=%d)"%(k))

# # plt.legend()
# # plt.grid(visible=True)
# # plt.tight_layout()
# # plt.show()







