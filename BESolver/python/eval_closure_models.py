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
import glow1d_utils
import enum

class closure_type(enum.IntEnum):
    MAX_ENTROPY = 0
    BTE_0D      = 1


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

def evaluate_closure(ctype, args, data, tt, xx, Ext, fprefix, 
                     ts_k=10, nx_k=50):

    args                      = data[0]
    spec_sp, col_list         = plot_utils.gen_spec_sp(args)
    
    num_vr                    = spec_sp._basis_p._num_knot_intervals * 5
    num_vt, num_vp            = int(args["Nvt"]), 1
    l_max                     = len(spec_sp._sph_harm_lm)
    nscale                    = 1

    bte_op                    = data[3]
    num_p                     = spec_sp._p + 1
    num_sh                    = spec_sp._sph_harm_lm

    assert num_vt % 2         == 0 
    gx, gw                    = basis.Legendre().Gauss_Pn(num_vt//2)
    gx_m1_0 , gw_m1_0         = 0.5 * gx - 0.5, 0.5 * gw
    gx_0_p1 , gw_0_p1         = 0.5 * gx + 0.5, 0.5 * gw
    xp_vt                     = np.append(np.arccos(gx_m1_0), np.arccos(gx_0_p1)) 
    xp_vt_qw                  = np.append(gw_m1_0, gw_0_p1)
      
    # self.xp_vt, self.xp_vt_qw = basis.Legendre().Gauss_Pn(self.Nvt)
    # self.xp_vt                = np.arccos(self.xp_vt) 
      
    xp_cos_vt                 = np.cos(xp_vt)
    xp_vt_l                   = np.array([i * num_vt + j for i in range(num_p) for j in list(np.where(xp_vt <= 0.5 * np.pi)[0])])
    xp_vt_r                   = np.array([i * num_vt + j for i in range(num_p) for j in list(np.where(xp_vt > 0.5 * np.pi)[0])])

    v                         = data[2]
    bte_op                    = data[3]
    
    Cmat                      = bte_op["Cmat"]
    Emat                      = bte_op["Emat"]
    Ps                        = bte_op["po2sh"]
    Po                        = bte_op["psh2o"]

    #vsh                      = np.einsum("lo,tox->tlx", bte_op["po2sh"], v)
    Te0                       = float(args["Te"])
    c_gamma                   = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))
    vth                       = np.sqrt(Te0) * c_gamma
    N0                        = 3.22e22


    mfuncs                    = [ lambda vr, vt, vp : np.ones_like(vr),
                                  lambda vr, vt, vp : vth    * vr * np.cos(vt),
                                  lambda vr, vt, vp : vth**2 * vr **2 * (2/3/c_gamma**2),
                                  #lambda vr, vt, vp : N0 * vth * vr * np.einsum("k,l,m->klm", col_list[0].total_cross_section(np.unique(vr) **2 * Te0), np.ones_like(np.unique(vt)), np.ones_like(np.unique(vp))),
                                  #lambda vr, vt, vp : N0 * vth * vr * np.einsum("k,l,m->klm", col_list[1].total_cross_section(np.unique(vr) **2 * Te0), np.ones_like(np.unique(vt)), np.ones_like(np.unique(vp))),
                                  #lambda vr, vt, vp : np.einsum("k,l,m->klm", uvr, uvt_0, uvp),
                                  #lambda vr, vt, vp : np.einsum("k,l,m->klm", uvr, uvt_1, uvp),
                                  #lambda vr, vt, vp : vth    * vr * np.cos(vt)**2 ,
                                  #lambda vr, vt, vp : vth**3 * vr**3 * (2/3/c_gamma**3),
                                  #lambda vr, vt, vp : vth**4 * vr**4 * (2/3/c_gamma**4),
                                  #lambda vr, vt, vp : vth**5 * vr**5 * (2/3/c_gamma**5),
                                  #lambda vr, vt, vp : q.n0 * q.np0  * vth * vr * np.einsum("k,l,m->klm", col_list[0].total_cross_section(np.unique(vr) **2 * Te0), np.unique(vt), xp.unique(vp)),
                                  
                             
                            ]
        
    mfuncs_Jx                 = [  lambda vr, vt, vp:   vth * vr * np.cos(vt)  * np.ones_like(vr), 
                                   lambda vr, vt, vp:   vth * vr * np.cos(vt)  * vth    * vr * np.cos(vt),
                                   lambda vr, vt, vp:   vth * vr * np.cos(vt)  * vth**2 * vr **2 * (2/3/c_gamma**2),
                                   #lambda vr, vt, vp :  vth * vr * np.cos(vt)  * N0 * vth * vr * np.einsum("k,l,m->klm", col_list[0].total_cross_section(np.unique(vr) **2 * Te0), np.ones_like(np.unique(vt)), np.ones_like(np.unique(vp))),
                                   #lambda vr, vt, vp :  vth * vr * np.cos(vt)  * N0 * vth * vr * np.einsum("k,l,m->klm", col_list[1].total_cross_section(np.unique(vr) **2 * Te0), np.ones_like(np.unique(vt)), np.ones_like(np.unique(vp))),
                                   #lambda vr, vt, vp : vth * vr * np.cos(vt)  * np.einsum("k,l,m->klm", uvr, uvt_0, uvp),
                                   #lambda vr, vt, vp : vth * vr * np.cos(vt)  * np.einsum("k,l,m->klm", uvr, uvt_1, uvp),
                                   #lambda vr, vt, vp : vth * vr * np.cos(vt)  * vth    * vr * np.cos(vt)**2,
                                   #lambda vr, vt, vp : vth * vr * np.cos(vt)  * vth**3 * vr**3 * (2/3/c_gamma**3),
                                   #lambda vr, vt, vp : vth * vr * np.cos(vt)  * vth**4 * vr**4 * (2/3/c_gamma**4),
                                   #lambda vr, vt, vp : vth * vr * np.cos(vt)  * vth**5 * vr**5 * (2/3/c_gamma**5),
                                   #lambda vr, vt, vp : vth * vr * np.cos(vt)  * q.n0 * q.np0  * vth * vr * np.einsum("k,l,m->klm", col_list[0].total_cross_section(np.unique(vr) **2 * Te0), np.unique(vt), xp.unique(vp)),
                                   
                                  ]
    
    mfuncs_qoi                = [r"$n_e$", 
                                 r"$u_z$", 
                                 r"$T_e$",
                                 #r"$\sigma_{\text{elastic}}$",
                                 #r"$\sigma_{\text{ion}}$"
                                 ]
    
    mfuncs_ops                = bte_utils.assemble_moment_ops(spec_sp, num_vr, num_vt, num_vp, mfuncs   , scale=1.0)
    mfuncs_Jx_ops             = bte_utils.assemble_moment_ops(spec_sp, num_vr, num_vt, num_vp, mfuncs_Jx, scale=1.0)

    mfuncs_ops_ords           = bte_utils.assemble_moment_ops_ords(spec_sp, xp_vt, xp_vt_qw, num_vr, num_vp, mfuncs, scale=1.0)
    mfuncs_Jx_ops_ords        = bte_utils.assemble_moment_ops_ords(spec_sp, xp_vt, xp_vt_qw, num_vr, num_vp, mfuncs_Jx, scale=1.0)

    print("moment op assembly end")

    def eval_moments(v, E):
        
        vsh                         = np.einsum("kl,tlx->tkx", Ps, v)
        m                           = np.einsum("il,tlx->tix", mfuncs_ops_ords, v)
        Jxm                         = np.einsum("il,tlx->tix", mfuncs_Jx_ops_ords, v) 

        Jvm                         = N0 * np.einsum("kl,tlx->tkx", Cmat, vsh) + np.einsum("tx,kl,tlx->tkx", E, Emat, vsh)
        Jvm                         = np.einsum("kl,tlx->tkx", Po, Jvm)
        Jvm                         = np.einsum("il,tlx->tix", mfuncs_ops_ords, Jvm) 
        return m, Jxm, Jvm
    
    if (ctype == closure_type.MAX_ENTROPY):
        v                               = v  [0::ts_k, : ,0::nx_k]
        E                               = Ext[0::ts_k,    0::nx_k]
        fo_c                            = np.zeros_like(v)
        m_bte, Jx_bte, Jv_bte           = eval_moments(v, E)

        for tidx, t in enumerate(list(tt[0::ts_k])):
            x, fs, fo                   = max_entropy_reconstruction(spec_sp, np.zeros_like(m_bte[tidx]), num_vr, num_vt, num_vp, m_bte[tidx]/m_bte[tidx, 0], mfuncs, np, atol=1e-20, rtol=1e-16, iter_max=300)
            fs, fo                      = m_bte[tidx,0] * fs, m_bte[tidx,0] * fo
            fo_c[tidx,:,:]              = fo
    
    
        m_c, Jx_c, Jv_c = eval_moments(fo_c, E)

        xx_c   = xx[0::nx_k]
        tt_c   = tt[0::ts_k] 

        clbl   = "MEA"

    elif (ctype == closure_type.BTE_0D):
        xx_c                            = xx
        tt_c                            = tt[0::ts_k] 

        v                               = v  [0::ts_k, : ,0::nx_k]
        E                               = Ext[0::ts_k,    0::nx_k]
        
        m_bte, Jx_bte, Jv_bte           = eval_moments(v, E)

        Ek                              = np.fft.rfftfreq(len(tt)-1, d= 1/(len(tt)-1))
        Ex                              = np.fft.rfft(Ext[:-1, ], axis=0)
        sf                              = np.ones_like(Ek) * (2/(len(tt)-1))
        sf[0]                           = (1/(len(tt)-1))
        Ex                              = sf[:, np.newaxis] * Ex
        Ex                              = np.concatenate(((np.real(Ex))[:, :, np.newaxis],(np.imag(Ex))[:, :, np.newaxis]), axis=2)
        Ex                              = np.swapaxes(Ex, 1, 2)
    
        # truncate the freq. spectrum
        num_freq_modes            = 20
        Ex, Ek                    = Ex[0:num_freq_modes, :, :], Ek[0:num_freq_modes]
        args["use_gpu"]           = '1'
        args["cfl"]               = 1e-2
        if (int(args["use_gpu"])==1):
            import cupy as cp
            xp = cp
        else:
            xp = np

        Ex                        = xp.asarray(Ex)
        Ek                        = xp.asarray(Ek)
        
        def Et(t):
            at = xp.array([[xp.cos(2 * xp.pi * Ek[i] * t ), -xp.sin(2 * xp.pi * Ek[i] * t)] for i in range(len(Ek))])
            y  = xp.sum(Ex * at[:, :, xp.newaxis], axis=(0, 1))
            return y


        
        Ext_a          = xp.array([Et(tt[i]) for i in range(len(tt))])
        v0d, v0d_avg   = bte_0d_closure(args, Et, fprefix=out_file_name, tau=glow_pars.tau, max_cycles=5, sfreq=(tt_c[1]-tt_c[0]))
        v0d            = xp.einsum("vl,tlx->tvx", Po, v0d)
        v0d_avg        = xp.einsum("vl,lx->vx",   Po, v0d_avg)

        if xp == cp:
            v0d     = cp.einsum("tvx,tx->tvx", v0d, cp.asarray(m_bte[:, 0 , :]))
            v0d     = cp.asnumpy(v0d)
            v0d_avg = cp.asnumpy(v0d_avg)
            Ext_a   = cp.asnumpy(Ext_a)
            

        fo_c            = v0d
        m_c, Jx_c, Jv_c = eval_moments(fo_c, E)

        clbl   = "BTE_0d3v"

        xloc   = np.linspace(-1, 0, 10)
        idx    = np.array([np.argmin(np.abs(xx_c-xloc[i])) for i in range(len(xloc))])
        #import matplotlib.colors as mcolors
        #mcolors.CSS4_COLORS
        clrs   =  ["blue", "red", "magenta", "orange", "lightblue", "green",
                "darkblue", "darkred", "brown", "yellowgreen", "skyblue", "navy",
                    "purple", "indigo", "darkorange", "darkgreen", "limegreen", "maroon" ] 

        plt.figure(figsize=(12, 12), dpi=200)
        for ki, k in enumerate(idx):
            plt.semilogy(tt, np.abs(Ext  [:, k]), "--", color=clrs[ki], label=r"BTE_1d3v (x=%.2f)"%(xx_c[k]))
            plt.semilogy(tt, np.abs(Ext_a[:, k]), "-o", color=clrs[ki], label=r"%s (x=%.2f)"%(clbl, xx_c[k]))
    
        plt.xlabel(r"time [T]")
        plt.ylabel(r"|E| [V/m]")
        plt.legend()
        plt.grid(visible=True)
        plt.savefig("%s_efield_vs_time.png"%(fprefix))
        plt.close()
        


    ev    = np.linspace(1e-3, 40, 512)
    v_rc  = compute_radial_components(spec_sp, np.einsum("kl,ilx->ikx", Ps, np.einsum("tvx,tx->tvx", v     , (1/ m_bte[:, 0 , :]))), ev, Te0=Te0)
    f_rc  = compute_radial_components(spec_sp, np.einsum("kl,ilx->ikx", Ps, np.einsum("tvx,tx->tvx", fo_c  , (1/ m_bte[:, 0 , :]))), ev, Te0=Te0)
    num_m = len(mfuncs)
    
    
    xloc   = np.linspace(-1, 0, 10)
    idx    = np.array([np.argmin(np.abs(xx_c-xloc[i])) for i in range(len(xloc))])
    #import matplotlib.colors as mcolors
    #mcolors.CSS4_COLORS
    clrs   =  ["blue", "red", "magenta", "orange", "lightblue", "green",
               "darkblue", "darkred", "brown", "yellowgreen", "skyblue", "navy",
                "purple", "indigo", "darkorange", "darkgreen", "limegreen", "maroon" ] 

    plt.figure(figsize=(18, 18), dpi=200)
    cnt = 1
    
    for i in range(num_m):
        plt.subplot(num_m, 3, cnt)
        for ki, k in enumerate(idx):
            plt.semilogy(tt_c, np.abs(m_bte[:, i, k] / m_bte[:, 0, k]), "--", color=clrs[ki], label=r"BTE_1d3v (x=%.2f)"%(xx_c[k]))
            plt.semilogy(tt_c, np.abs(m_c  [:, i, k] / m_c  [:, 0, k]), "-" , color=clrs[ki], label=r"%s (x=%.2f)"%(clbl, xx_c[k]))

        plt.title(r"%s/$n_e$"%(mfuncs_qoi[i]))
        plt.xlabel(r"time [T]")
        plt.ylabel(r"m")
        plt.grid(visible=True)
        if (cnt == 1):
            plt.legend()
        cnt+=1
        
        plt.subplot(num_m, 3, cnt)
        for ki, k in enumerate(idx):
            plt.semilogy(tt_c, np.abs(Jx_bte[:, i, k] / m_bte[:, 0, k]), "--", color=clrs[ki], label=r"BTE_1d3v (x=%.2f)"%(xx_c[k]))
            plt.semilogy(tt_c, np.abs(Jx_c  [:, i, k] / m_c  [:, 0, k]), "-" , color=clrs[ki], label=r"%s (x=%.2f)"%(clbl, xx_c[k]))

        plt.title(r"Jx(%s)/$n_e$"%(mfuncs_qoi[i]))
        plt.xlabel(r"time [T]")
        plt.ylabel(r"abs(J_x)")
        plt.grid(visible=True)
        cnt+=1

        plt.subplot(num_m, 3, cnt)
        
        for ki, k in enumerate(idx):
            plt.semilogy(tt_c, np.abs(Jv_bte[:, i, k] / m_bte[:, 0, k]), "--", color=clrs[ki], label=r"BTE_1d3v (x=%.2f)"%(xx_c[k]))
            plt.semilogy(tt_c, np.abs(Jv_c  [:, i, k] / m_c  [:, 0, k]), "-" , color=clrs[ki], label=r"%s (x=%.2f)"%(clbl, xx_c[k]))
        
        plt.title(r"Sv(%s)/$n_e$"%(mfuncs_qoi[i]))
        plt.xlabel(r"time [T]")
        plt.ylabel(r"abs(S_v)")
        plt.grid(visible=True)
        cnt+=1

    
    #plt.suptitle(r"t= %.4E [T]"%(t))
    plt.tight_layout()
    #plt.show()
    plt.savefig("%s_vs_time.png"%(fprefix))
    plt.close()


    for tidx, t in enumerate(list(tt_c)):
        print(tidx, t)
        plt.figure(figsize=(12, 16), dpi=200)
        plt.subplot(num_m + 1, 3, 1)
        for ki, k in enumerate(idx):
            plt.semilogy(ev, np.abs(v_rc[tidx, k, 0]), "--", color=clrs[ki], label=r"BTE_1d3v (x=%.2f)"%(xx_c[k]))
            plt.semilogy(ev, np.abs(f_rc[tidx, k, 0]), "-" , color=clrs[ki], label=r"%s (x=%.2f)"%(clbl, xx_c[k]))

        plt.legend(fontsize=6)
        plt.xlabel(r"energy [eV]")
        plt.ylabel(r"$f_0$")
        plt.grid(visible=True)

        plt.subplot(num_m+1, 3, 2)
        for ki, k in enumerate(idx):
            plt.semilogy(ev, np.abs(v_rc[tidx, k, 1]), "--", color=clrs[ki], label=r"BTE_1d3v (x=%.2f)"%(xx_c[k]))
            plt.semilogy(ev, np.abs(f_rc[tidx, k, 1]), "-" , color=clrs[ki], label=r"%s (x=%.2f)"%(clbl, xx_c[k]))

        #plt.legend(fontsize=6)
        plt.xlabel(r"energy [eV]")
        plt.ylabel(r"$f_1$")
        plt.grid(visible=True)

        plt.subplot(num_m+1, 3, 3)
        for ki, k in enumerate(idx):
            plt.semilogy(ev, np.abs(v_rc[tidx, k, 2]), "--", color=clrs[ki], label=r"BTE_1d3v (x=%.2f)"%(xx_c[k]))
            plt.semilogy(ev, np.abs(f_rc[tidx, k, 2]), "-" , color=clrs[ki], label=r"%s (x=%.2f)"%(clbl, xx_c[k]))

        #plt.legend(fontsize=6)
        plt.xlabel(r"energy [eV]")
        plt.ylabel(r"$f_2$")
        plt.grid(visible=True)

        cnt = 4
        for i in range(num_m):
            plt.subplot(num_m+1, 3, cnt)
            plt.semilogy(xx_c, np.abs(1-m_c[tidx, i]/m_bte[tidx, i]), "b-", label=r"BTE_1d3v vs %s"%(clbl))
            #plt.plot(xx_c, m_c[tidx, 0], "r--", label=r"MEA")
            #plt.legend(fontsize=6)
            plt.xlabel(r"$\hat{x}$")
            plt.ylabel(r"relative error")
            plt.title(mfuncs_qoi[i])
            plt.grid(visible=True)
            cnt+=1

            plt.subplot(num_m+1, 3, cnt)
            plt.semilogy(xx_c, np.abs(1-Jx_c[tidx, i]/Jx_bte[tidx, i]), "b-", label=r"BTE vs MEA")
            #plt.legend(fontsize=6)
            plt.xlabel(r"$\hat{x}$")
            plt.ylabel(r"relative error")
            plt.title(r"Jx (" + mfuncs_qoi[i] + r")")
            plt.grid(visible=True)

            cnt+=1
            plt.subplot(num_m+1, 3, cnt)
            plt.semilogy(xx_c, np.abs(1-Jv_c[tidx, i]/Jv_bte[tidx, i]), "b-", label=r"BTE vs MEA")
            #plt.legend(fontsize=6)
            plt.xlabel(r"$\hat{x}$")
            plt.ylabel(r"relative error")
            plt.title(r"Sv (" + mfuncs_qoi[i] + r")")
            plt.grid(visible=True)
            cnt+=1
        
        plt.suptitle(r"t= %.4E [T]"%(t))
        plt.tight_layout()
        #plt.show()
        plt.savefig("%s_tidx_%04d.png"%(fprefix, tidx))
        plt.close()

    return 

data_folder               = "1dglow_hybrid_Nx400/1Torr300K_100V_Ar_3sp2r_cycle"
ff                        = h5py.File("%s/macro.h5"%(data_folder), 'r')
tt                        = np.array(ff["time[T]"][()])
xx                        = np.array(ff["x[-1,1]"][()])
Ext                       = np.array(ff["E[Vm^-1]"][()])

data                      = plot_utils.load_data_bte(data_folder, range(0, 101), eedf_idx=None, read_cycle_avg=False, use_ionization=1)
args                      = data[0]
spec_bspline, col_list    = plot_utils.gen_spec_sp(args)
Te0                       = float(args["Te"])

class A:
    par_file = args["par_file"].replace("'", "")

glow_pars       = glow1d_utils.parameters(A())
out_folder_name = "%s/closure_models"%(data_folder)
make_dir(out_folder_name)

ctype     = closure_type.MAX_ENTROPY

if(ctype == closure_type.MAX_ENTROPY):
    out_file_name  = "%s/mea_3m"%(out_folder_name)
    evaluate_closure(ctype,args, data, tt, xx, Ext, out_file_name, ts_k=10, nx_k=50)
elif(ctype == closure_type.BTE_0D):
    out_file_name  = "%s/bte_0d"%(out_folder_name)
    evaluate_closure(ctype,args, data, tt, xx, Ext, out_file_name, ts_k=10, nx_k=1)
else:
    raise NotImplementedError




# if (closure_type == "bte_0d"):
#     out_file_name = "%s/bte_0d"%(out_folder_name)
#     fsh           = bte_0d_closure(args, (Ex[0:20, :, :], Ek[0:20]), fprefix=out_file_name)
#     spec_sp       = spec_bspline
# elif(closure_type == "max-entropy"):
#     # Max entropy closure. 
#     out_file_name = "%s/max_entropy"%(out_folder_name)
#     fsh, rel_entropy, spec_sp = max_entropy_closure(args, data, tidx=0)


# v                         = data[2]
# bte_op                    = data[3]
# vsh                       = cp.asnumpy(cp.einsum("lo,tox->tlx", cp.asarray(bte_op["po2sh"]), cp.asarray(v)))
# vsh_n                     = np.einsum("tvx,tx->tvx", vsh , (1/ np.einsum("l,tlx->tx", bte_op["mass"], vsh)))
# vsh_avg                   = time_average(vsh_n, tt)
# ev                        = np.linspace(0, 40, 512)


# if (closure_type == "bte_0d"):
#     xloc = list(range(0, 201, 50))
#     xx_c = xx[xloc]
#     vsh_n_rc                  = compute_radial_components(spec_bspline, vsh_avg[np.newaxis, :, :] , ev, Te0)
#     fsh_n_rc                  = compute_radial_components(spec_sp     , fsh    [np.newaxis, :, :] , ev, Te0)
#     lbl                       = "0D-BTE"
# elif (closure_type == "max-entropy"):
#     xloc = list(range(0, 40, 10))
#     xx_c = xx[0::10]
#     vsh_n_rc                  = compute_radial_components(spec_bspline, vsh_n  [:,      :, 0::10] , ev, Te0)
#     fsh_n_rc                  = compute_radial_components(spec_sp     , fsh    [np.newaxis, :, :] , ev, Te0)
#     lbl                       = "max-entropy"


# plt.figure(figsize=(16, 4), dpi=300)
# plt.subplot(1, 3, 1)
# for idx, xidx in enumerate(xloc):
#     plt.semilogy(ev, np.abs(vsh_n_rc[0, xidx, 0].T), '-' , color="C%d"%(idx), label=r"1D-BTE (x=%.2f)"%(xx_c[idx]))
#     plt.semilogy(ev, np.abs(fsh_n_rc[0, xidx, 0].T), '--', color="C%d"%(idx), label=r"%s (x=%.2f)"%(lbl, xx_c[idx]))
# plt.xlabel(r"energy [eV]")
# plt.ylabel(r"$f_0$[$eV^3/2$]")
# plt.grid(visible=True)

# plt.subplot(1, 3, 2)
# for idx, xidx in enumerate(xloc):
#     plt.semilogy(ev, np.abs(vsh_n_rc[0, xidx, 1].T), '-' , color="C%d"%(idx), label=r"1D-BTE (x=%.2f)"%(xx_c[idx]))
#     plt.semilogy(ev, np.abs(fsh_n_rc[0, xidx, 1].T), '--', color="C%d"%(idx), label=r"%s (x=%.2f)"%(lbl, xx_c[idx]))
# plt.xlabel(r"energy [eV]")
# plt.ylabel(r"$f_1$[$eV^3/2$]")
# plt.grid(visible=True)
# plt.legend()

# plt.subplot(1, 3, 3)
# for idx, xidx in enumerate(xloc):
#     plt.semilogy(ev, np.abs(vsh_n_rc[0, xidx, 2].T), '-' , color="C%d"%(idx), label=r"1D-BTE (x=%.2f)"%(xx_c[idx]))
#     plt.semilogy(ev, np.abs(fsh_n_rc[0, xidx, 2].T), '--', color="C%d"%(idx), label=r"%s (x=%.2f)"%(lbl, xx_c[idx]))
# plt.xlabel(r"energy [eV]")
# plt.ylabel(r"$f_2$[$eV^3/2$]")
# plt.grid(visible=True)
# plt.tight_layout()
# plt.savefig("%s_eedfs.png"%(out_file_name))
# plt.close()