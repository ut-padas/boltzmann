"""
Module to handle different closure models. 
"""
import numpy as np
import cross_section
import spec_spherical as sp
import basis
import utils as bte_utils
import plot_scripts.plot_utils as plot_utils
import scipy.constants
import matplotlib.pyplot as plt
import h5py
import os
from closure_models import max_entropy_reconstruction, bte_0d_closure, relative_entropy
try: 
    import cupy as cp
except:
    raise ModuleNotFoundError

def make_dir(dir):
    if os.path.exists(dir):
        pass
        #print("run directory exists, data will be overwritten")
        #sys.exit(0)
    else:
        os.makedirs(dir)
        #print("directory %s created"%(dir))

def time_average(qoi, tt):
    """
    computes the time average on grids
    """
    # check if tt is uniform
    nT   = len(tt)
    T    = (tt[-1]-tt[0])
    dt   = T/(nT-1)
    
    assert abs(tt[1] -tt[0] - dt) < 1e-10
    
    tw    = np.ones_like(tt) * dt
    tw[0] = 0.5 * tw[0]; tw[-1] = 0.5 * tw[-1];
    
    assert (T-np.sum(tw)) < 1e-12
    
    return np.einsum("i,ivx->vx",tw, qoi)

def compute_radial_components(spec_sp, ff, ev: np.array, Te0):
    
    t_pts    = ff.shape[0]
    
    ff_lm    = ff
    c_gamma  = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))
    vth      = np.sqrt(Te0) * c_gamma
    
    vr       = np.sqrt(ev) * c_gamma / vth
    num_p    = spec_sp._p +1 
    num_sh   = len(spec_sp._sph_harm_lm)
    n_pts    = ff.shape[2]
    
    output   = np.zeros((t_pts, n_pts, num_sh, len(vr)))
    Vqr      = spec_sp.Vq_r(vr,0,1)
    
    mm_fac   = np.sqrt(4 * np.pi) 
    
    scale    = np.array([1 / mm_fac  for idx in range(t_pts)])
    ff_lm_n  = np.array([ff_lm[idx]/scale[idx] for idx in range(t_pts)])
    
    for idx in range(t_pts):
        ff_lm_T  = ff_lm_n[idx].T
        for l_idx, lm in enumerate(spec_sp._sph_harm_lm):
            output[idx, :, l_idx, :] = np.dot(ff_lm_T[:,l_idx::num_sh], Vqr)


    if spec_sp.get_radial_basis_type() == basis.BasisType.LAGUERRE:
        output *= (1/np.pi**1.5) * np.exp(-vr**2)
        
    return output

def max_entropy_closure(args, data, tidx):
    args                      = data[0]
    spec_bspline, col_list    = plot_utils.gen_spec_sp(args)
    
    Nr                        = 128
    num_qvr, num_qvt, num_qvp = 120, 16, 8
    l_max                     = len(spec_bspline._sph_harm_lm)

    bb                        = basis.BSpline(spec_bspline._basis_p._kdomain,
                                              spec_bspline._basis_p._sp_order,
                                              Nr+1,
                                              sig_pts=spec_bspline._basis_p._sig_pts, knots_vec=None, 
                                              dg_splines=False, verbose = 0, extend_domain_with_log=True)

    sph_lm                    = [[l,0] for l in range(l_max+1)]

    # Lp                        = basis.gLaguerre(alpha=0.5)
    # spec_sp                   = sp.SpectralExpansionSpherical(Nr-1, Lp, sph_lm)
    # nscale                    = (1/np.pi**1.5)/2

    spec_sp                   = sp.SpectralExpansionSpherical(Nr, bb, sph_lm)
    num_qvr                   = spec_sp._basis_p._num_knot_intervals * 5
    nscale                    = 1

    v                         = data[2]
    bte_op                    = data[3]
    #vsh                      = np.einsum("lo,tox->tlx", bte_op["po2sh"], v)
    Te0                       = float(args["Te"])
    c_gamma                   = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))

    mfuncs                    = [lambda vr, vt, vp: np.ones_like(vr),
                                lambda vr, vt, vp : vr * np.cos(vt),
                                lambda vr, vt, vp : vr **2,
                                lambda vr, vt, vp : vr * np.einsum("k,l,m->klm", col_list[1].total_cross_section(np.unique(vr) **2 * Te0), np.unique(vt), np.unique(vp)) * 3.22e22
                                ]

    mnt_ops                   = bte_utils.assemble_moment_ops(spec_sp, num_qvr, num_qvt, num_qvp, mfuncs, scale = nscale)
    mnt_ops_bspline           = bte_utils.assemble_moment_ops(spec_bspline, spec_bspline._basis_p._num_knot_intervals * 5,
                                                            num_qvt, num_qvp, mfuncs, scale = 1.0)


    vsh         = bte_op["po2sh"] @ v[tidx]
    mnt_rhs     = mnt_ops_bspline @ vsh
    mnt_rhs     = mnt_rhs[:, 0::10]
    x, fsh      = max_entropy_reconstruction(spec_sp, np.zeros_like(mnt_rhs), num_qvr, num_qvt, num_qvp, mnt_rhs, mfuncs, np, atol=1e-20, rtol=1e-16, iter_max=1000)
    rel_entropy = relative_entropy(spec_bspline, vsh[:, 0::10], spec_sp, fsh, (num_qvr, num_qvt, num_qvp), np, "KL")

    fsh         = np.einsum("vx,x->vx", fsh, 1/(mnt_ops[0] @ fsh))
    return fsh, rel_entropy, spec_sp

data_folder               = "1dglow_hybrid_Nx400/1Torr300K_100V_Ar_3sp2r_cycle"
ff                        = h5py.File("%s/macro.h5"%(data_folder), 'r')
tt                        = np.array(ff["time[T]"][()])
xx                        = np.array(ff["x[-1,1]"][()])
Ext                       = np.array(ff["E[Vm^-1]"][()])

Ek                        = np.fft.rfftfreq(len(tt)-1, d= 1/(len(tt)-1))
Ex                        = np.fft.rfft(Ext[:-1, ], axis=0)
sf                        = np.ones_like(Ek) * (2/(len(tt)-1))
sf[0]                     = (1/(len(tt)-1))
Ex                        = sf[:, np.newaxis] * Ex

Ex                        = np.concatenate(((np.real(Ex))[:, :, np.newaxis],(np.imag(Ex))[:, :, np.newaxis]), axis=2)
Ex                        = np.swapaxes(Ex, 1, 2)

data                      = plot_utils.load_data_bte(data_folder, range(0, 101), eedf_idx=None, read_cycle_avg=False, use_ionization=1)
args                      = data[0]
spec_bspline, col_list    = plot_utils.gen_spec_sp(args)
Te0                       = float(args["Te"])

out_folder_name = "%s/closure_models"%(data_folder)
make_dir(out_folder_name)
closure_type = "max-entropy"


if (closure_type == "bte_0d"):
    out_file_name = "%s/bte_0d"%(out_folder_name)
    fsh           = bte_0d_closure(args, (Ex[0:20, :, :], Ek[0:20]), fprefix=out_file_name)
    spec_sp       = spec_bspline
elif(closure_type == "max-entropy"):
    # Max entropy closure. 
    out_file_name = "%s/max_entropy"%(out_folder_name)
    fsh, rel_entropy, spec_sp = max_entropy_closure(args, data, tidx=0)


v                         = data[2]
bte_op                    = data[3]
vsh                       = cp.asnumpy(cp.einsum("lo,tox->tlx", cp.asarray(bte_op["po2sh"]), cp.asarray(v)))
vsh_n                     = np.einsum("tvx,tx->tvx", vsh , (1/ np.einsum("l,tlx->tx", bte_op["mass"], vsh)))
vsh_avg                   = time_average(vsh_n, tt)
ev                        = np.linspace(0, 40, 512)


if (closure_type == "bte_0d"):
    xloc = list(range(0, 201, 50))
    xx_c = xx[xloc]
    vsh_n_rc                  = compute_radial_components(spec_bspline, vsh_avg[np.newaxis, :, :] , ev, Te0)
    fsh_n_rc                  = compute_radial_components(spec_sp     , fsh    [np.newaxis, :, :] , ev, Te0)
    lbl                       = "0D-BTE"
elif (closure_type == "max-entropy"):
    xloc = list(range(0, 40, 10))
    xx_c = xx[0::10]
    vsh_n_rc                  = compute_radial_components(spec_bspline, vsh_n  [:,      :, 0::10] , ev, Te0)
    fsh_n_rc                  = compute_radial_components(spec_sp     , fsh    [np.newaxis, :, :] , ev, Te0)
    lbl                       = "max-entropy"


plt.figure(figsize=(16, 4), dpi=300)
plt.subplot(1, 3, 1)
for idx, xidx in enumerate(xloc):
    plt.semilogy(ev, np.abs(vsh_n_rc[0, xidx, 0].T), '-' , color="C%d"%(idx), label=r"1D-BTE (x=%.2f)"%(xx_c[idx]))
    plt.semilogy(ev, np.abs(fsh_n_rc[0, xidx, 0].T), '--', color="C%d"%(idx), label=r"%s (x=%.2f)"%(lbl, xx_c[idx]))
plt.xlabel(r"energy [eV]")
plt.ylabel(r"$f_0$[$eV^3/2$]")
plt.grid(visible=True)

plt.subplot(1, 3, 2)
for idx, xidx in enumerate(xloc):
    plt.semilogy(ev, np.abs(vsh_n_rc[0, xidx, 1].T), '-' , color="C%d"%(idx), label=r"1D-BTE (x=%.2f)"%(xx_c[idx]))
    plt.semilogy(ev, np.abs(fsh_n_rc[0, xidx, 1].T), '--', color="C%d"%(idx), label=r"%s (x=%.2f)"%(lbl, xx_c[idx]))
plt.xlabel(r"energy [eV]")
plt.ylabel(r"$f_1$[$eV^3/2$]")
plt.grid(visible=True)
plt.legend()

plt.subplot(1, 3, 3)
for idx, xidx in enumerate(xloc):
    plt.semilogy(ev, np.abs(vsh_n_rc[0, xidx, 2].T), '-' , color="C%d"%(idx), label=r"1D-BTE (x=%.2f)"%(xx_c[idx]))
    plt.semilogy(ev, np.abs(fsh_n_rc[0, xidx, 2].T), '--', color="C%d"%(idx), label=r"%s (x=%.2f)"%(lbl, xx_c[idx]))
plt.xlabel(r"energy [eV]")
plt.ylabel(r"$f_2$[$eV^3/2$]")
plt.grid(visible=True)
plt.tight_layout()
plt.savefig("%s_eedfs.png"%(out_file_name))
plt.close()