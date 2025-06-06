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
import cupy as cp
import matplotlib as mpl
from matplotlib.pyplot import cm
import itertools
#import os
#os.environ["OMP_NUM_THREADS"] = "8"

def asnumpy(x):
    if (xp == cp):
        return xp.asnumpy(x)
    else:
        return x

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

def compute_radial_components(args, bte_op, spec_sp, ev: np.array, ff, normalize=False):
    t_pts    = ff.shape[0]
    ff_lm    = ff #np.array([np.dot(bte_op["po2sh"], ff[idx]) for idx in range(t_pts)])
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
    
    if normalize:
        scale    = np.array([np.dot(mm_op / mm_fac, ff_lm[idx]) * (2 * (vth/c_gamma)**3) for idx in range(t_pts)])
        ff_lm_n  = np.array([ff_lm[idx]/scale[idx] for idx in range(t_pts)])
    else:
        ff_lm_n  = ff_lm   
    
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
    plt.xlabel(r"frequency (Hz)")
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
        plt.title(r"freq = %.4E (Hz)"%(freq[i-1]))
        #plt.ylabel(r"E [V/m]")
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
    #plt.title(r"E(x,t)")

    plt.subplot(1, 3, 2)
    plt.imshow(qoi_r, aspect='auto', extent=extent)
    plt.colorbar()
    plt.xlabel(r"$\hat{x}$")
    plt.ylabel(r"$time [T]$")
    plt.title(r"$\sum_{k=0}^{10} a_k(x) cos(k \omega t) + b_k(x) sin(k \omega t)$")

    plt.subplot(1, 3, 3)
    #plt.imshow(np.abs(1 - qoi_r/qoi), aspect='auto', extent=extent, norm=LogNorm(vmin=1e-10, vmax=1))
    plt.imshow(np.abs((qoi - qoi_r)/np.max(np.abs(qoi))), aspect='auto', extent=extent, norm=LogNorm(vmin=1e-10, vmax=1))
    plt.colorbar()
    plt.xlabel(r"$\hat{x}$")
    plt.ylabel(r"$time [T]$")
    plt.title(r"relative error")

    plt.tight_layout()
    #plt.show()
    plt.savefig("%s_fft_recons.png"%(fprefix))
    plt.close()

def fft_edf(qoi, axis, tt, dt, k, ev, spec_sp, bte_op, fprefix, normalize=False):
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

    a = compute_radial_components(args, bte_op, spec_sp, ev, np.real(fqoi), normalize=normalize)
    b = compute_radial_components(args, bte_op, spec_sp, ev, np.imag(fqoi), normalize=normalize)

    print(a.shape)

    xloc = list(range(0, len(xx), len(xx)//5))
    xloc.append(len(xx)-1)
    #print(a)
    for xidx in xloc:
        print(xidx)
        plt.figure(figsize=(12, 8), dpi=300)
        for i in range(1, k):
            plt.subplot(3, 4, i)
            sf = 1
            if i == 1:
                sf = 1/len(tt)
            else:
                sf = 2/len(tt)


            plt.semilogy(ev, np.abs(a[i-1, xidx, 0, :].T), label=r"$f_0$ -- cos($\omega_{%d}$ t)"%(i-1))
            plt.semilogy(ev, np.abs(b[i-1, xidx, 0, :].T), label=r"$f_0$ -- sin($\omega_{%d}$ t)"%(i-1))

            plt.semilogy(ev, np.abs(a[i-1, xidx, 1, :].T), label=r"$f_1$ -- cos($\omega_{%d}$ t)"%(i-1))
            plt.semilogy(ev, np.abs(b[i-1, xidx, 1, :].T), label=r"$f_1$ -- sin($\omega_{%d}$ t)"%(i-1))

            plt.semilogy(ev, np.abs(a[i-1, xidx, 2, :].T), label=r"$f_2$ -- cos($\omega_{%d}$ t)"%(i-1))
            plt.semilogy(ev, np.abs(b[i-1, xidx, 2, :].T), label=r"$f_2$ -- sin($\omega_{%d}$ t)"%(i-1))

            # plt.plot(xx, sf * np.real(fqoi[i-1,:]), label=r"cos($\omega_{%d}$ t)"%(i-1))
            # plt.plot(xx, sf * np.imag(fqoi[i-1,:]), label=r"sin($\omega_{%d}$ t)"%(i-1))
            plt.legend(fontsize=6)
            plt.grid(visible=True)
            plt.title(r"freq = %.4E"%(freq[i-1]))
            plt.ylabel(r"radial component [$ev^{-3/2}$]")
            plt.xlabel(r"energy [eV]")
        
        plt.tight_layout()
        plt.savefig("%s_modes_xloc_%04d.png"%(fprefix, xidx))
        plt.close()

def fft_rom_analysis(fs, gs, tt, dt, xx, bte_op, fprefix):
    mop               = bte_op["mass"]
    ne_f              = np.einsum("tlx,l->tx", fs, mop)
    ne_g              = np.einsum("tlx,l->tx", gs, mop)

    n_fs              = np.einsum("tvx,tx->tvx", fs, (1/ne_f))
    n_gs              = np.einsum("tvx,tx->tvx", gs, (1/ne_g))

    freq              = np.fft.rfftfreq(len(tt), d=dt)
    n_fs_fft          = np.fft.rfft(n_fs, axis=0)
    n_gs_fft          = np.fft.rfft(n_gs, axis=0)

    fs_fft            = np.fft.rfft(fs, axis=0)
    gs_fft            = np.fft.rfft(gs, axis=0)


    plt.figure(figsize=(10, 4), dpi=200)
    # plt.semilogy(freq, np.linalg.norm(n_fs_fft, axis=(1, 2)) / np.linalg.norm(n_fs_fft, axis=(1, 2))[0] , 'o-',     label=r"$fft(\hat{f}=\frac{f_s}{n_e})$")
    # plt.semilogy(freq, np.linalg.norm(n_gs_fft, axis=(1, 2)) / np.linalg.norm(n_gs_fft, axis=(1, 2))[0] , 'o-',     label=r"$fft(\hat{g}=\frac{g}{n_e})$")

    plt.semilogy(freq, np.linalg.norm(fs_fft, axis=(1, 2))   / np.linalg.norm(fs_fft, axis=(1, 2))[0] , 'o-',     label=r"$fft(f_s)$")
    plt.semilogy(freq, np.linalg.norm(gs_fft, axis=(1, 2))   / np.linalg.norm(gs_fft, axis=(1, 2))[0] , 'o-',     label=r"$fft(g)$")

    plt.legend()
    plt.xlabel("frequency (Hz)")
    plt.ylabel("magnitude")
    plt.grid(visible=True)
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig("%s_fft.png"%(fprefix))

    #rtol  = 1e-6
    nfreq      = 6
    num_sh     = int(args["l_max"]) + 1

    if 1:
        fs_fft_recons = np.zeros(fs_fft.shape, dtype=fs_fft.dtype)
        for fidx in range(nfreq):
            for lidx in range(num_sh):
                u_re, s_re, vt_re = np.linalg.svd(np.real(gs_fft[fidx, lidx::num_sh]))
                u_im, s_im, vt_im = np.linalg.svd(np.imag(gs_fft[fidx, lidx::num_sh]))
                v_re              = vt_re.T
                v_im              = vt_im.T
                rank_k            = 150

                p_re              = u_re[:, 0:rank_k]
                q_re              = v_re[:, 0:rank_k]
                p_im              = u_im[:, 0:rank_k]
                q_im              = v_im[:, 0:rank_k]

                Pp_re             = p_re @ p_re.T
                Pq_re             = q_re @ q_re.T
                Pp_im             = p_im @ p_im.T
                Pq_im             = q_im @ q_im.T
                
                fs_fft_recons[fidx, lidx::num_sh]  = Pp_re @ np.real(fs_fft[fidx, lidx::num_sh]) @ Pq_re + (1j) * Pp_im @ np.imag(fs_fft[fidx, lidx::num_sh]) @ Pq_im
                

            ev         = np.linspace(1e-3, 80, 1024)
            fs_fft_re  = compute_radial_components(args, bte_op, spec_sp, ev, np.real(fs_fft[fidx])       [np.newaxis, :, :]           , normalize=False)
            fs_fft_im  = compute_radial_components(args, bte_op, spec_sp, ev, np.imag(fs_fft[fidx])       [np.newaxis, :, :]           , normalize=False)
            fs_fft_re0 = compute_radial_components(args, bte_op, spec_sp, ev, np.real(fs_fft_recons[fidx])[np.newaxis, :, :]           , normalize=False)
            fs_fft_im0 = compute_radial_components(args, bte_op, spec_sp, ev, np.imag(fs_fft_recons[fidx])[np.newaxis, :, :]           , normalize=False)

            # Set the default color cycle
            #mpl.rcParams['axes.prop_cycle'] =  
            #print(fs_fft_re.shape)
            #cycle   = itertools.cycle(cm.viridis(np.linspace(0, 1, 12)))
            cycle   = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
            # xloc    = list(range(0, 400, 50))
            # xloc.append(len(xx)-1)
            xloc    = [0, 20, 50, 100, 200]

            for idx, (fs, fs_r) in enumerate(list([(fs_fft_re, fs_fft_re0), (fs_fft_im, fs_fft_im0)])):
                plt.figure(figsize=(12, 4), dpi=300)
                for xidx in xloc:
                    clr = next(cycle)
                    plt.subplot(1, 3, 1)
                    plt.title(r"$f_0$")
                    
                    plt.semilogy(ev, np.abs(fs  [0, xidx, 0]), '-'  , color=clr, label = r"FOM F($\epsilon$,$x=%.2f$)"%(xx[xidx]))
                    plt.semilogy(ev, np.abs(fs_r[0, xidx, 0]), 'o--', color=clr, label = r"ROM F($\epsilon$,$x=%.2f$)"%(xx[xidx]), markersize=2.4)
                    
                    plt.legend(fontsize=8)
                    
                    plt.xlabel(r"energy [eV]")
                    plt.grid(visible=True)

                    plt.subplot(1, 3, 2)
                    plt.title(r"$f_1$")

                    plt.semilogy(ev, np.abs(fs  [0, xidx, 1]), '-'  , color=clr, label = r"FOM F($\epsilon$,$x=%.2f$)"%(xx[xidx]))
                    plt.semilogy(ev, np.abs(fs_r[0, xidx, 1]), 'o--', color=clr, label = r"ROM F($\epsilon$,$x=%.2f$)"%(xx[xidx]), markersize=2.4)

                    plt.legend(fontsize=8)

                    plt.xlabel(r"energy [eV]")
                    plt.grid(visible=True)

                    plt.subplot(1, 3, 3)
                    plt.title(r"$f_2$")
                    
                    plt.semilogy(ev, np.abs(fs  [0, xidx, 2]), '-'  , color=clr, label = r"FOM F($\epsilon$,$x=%.2f$)"%(xx[xidx]))
                    plt.semilogy(ev, np.abs(fs_r[0, xidx, 2]), 'o--', color=clr, label = r"ROM F($\epsilon$,$x=%.2f$)"%(xx[xidx]), markersize=2.4)
                    plt.legend(fontsize=8)

                    plt.xlabel(r"energy [eV]")
                    plt.grid(visible=True)

                if idx == 0 : plt.savefig("%s_fft_modes_re_fidx_%04d.png"%(fprefix, fidx))
                if idx == 1 : plt.savefig("%s_fft_modes_im_fidx_%04d.png"%(fprefix, fidx))

    
    kr         = list(range(10, 251, 50))
    nkr        = len(kr)

    rel_errors    = np.zeros((nfreq, 2, nkr))
    fs_fft_recons = np.zeros(fs_fft.shape, dtype=fs_fft.dtype)
    for ik, k in enumerate(kr):
        
        for fidx in range(nfreq):
            for lidx in range(num_sh):
                print("rank = ", k, " processing freq: ", fidx, " l = ", lidx)

                u_re, s_re, vt_re = np.linalg.svd(np.real(gs_fft[fidx, lidx::num_sh]))
                u_im, s_im, vt_im = np.linalg.svd(np.imag(gs_fft[fidx, lidx::num_sh]))

                v_re = vt_re.T
                v_im = vt_im.T

                p_re = u_re[:, 0:k]
                q_re = v_re[:, 0:k]

                p_im = u_im[:, 0:k]
                q_im = v_im[:, 0:k]

                Pp_re = p_re @ p_re.T
                Pq_re = q_re @ q_re.T

                Pp_im = p_im @ p_im.T
                Pq_im = q_im @ q_im.T
        
                fs_fft_recons[fidx, lidx::num_sh]  = Pp_re @ np.real(fs_fft[fidx, lidx::num_sh]) @ Pq_re + (1j) * Pp_im @ np.imag(fs_fft[fidx, lidx::num_sh]) @ Pq_im

            rel_errors[fidx, 0, ik] = np.linalg.norm(np.real(fs_fft[fidx]) - np.real(fs_fft_recons[fidx])) / np.linalg.norm(np.real(fs_fft[fidx]))
            rel_errors[fidx, 1, ik] = np.linalg.norm(np.imag(fs_fft[fidx]) - np.imag(fs_fft_recons[fidx])) / np.linalg.norm(np.imag(fs_fft[fidx]))
            
    plt.figure(figsize=(12, 6), dpi=200)
    for fidx in range(nfreq):
        plt.semilogy(np.array(kr), rel_errors[fidx, 0], 'o-'     , label=r"$\cos(\omega_%d t)$"%(fidx))
        if(fidx>0):
            plt.semilogy(np.array(kr), rel_errors[fidx, 1], 'o--', label=r"$\sin(\omega_%d t)$"%(fidx))

    plt.xlabel(r"number of svd modes used")
    plt.ylabel(r"$||I- P \text{fft}(f_s)||/||\text{fft}(f_s)||$")
    plt.legend()
    plt.grid(visible=True)
    plt.tight_layout()
    #plt.show()
    plt.savefig("%s_fft_rom_projection_error.png"%(fprefix))
    plt.close()

def plot_Et_fft_comp(Et_bte, Et_fluid, tt, dt, k, fprefix):
    freq              = np.fft.rfftfreq(len(tt), 1/len(tt)/13.56e6)
    Et_fluid_fft      = np.fft.rfft(Et_fluid, axis=0)
    Et_fft            = np.fft.rfft(Et_bte, axis=0)

    plt.figure(figsize=(12, 4), dpi=300)
    plt.semilogy(freq, np.linalg.norm(Et_fft, axis=1)       / np.linalg.norm(Et_fft, axis=1)[0]       , 'o-', label=r"$E_{bte}$")
    plt.semilogy(freq, np.linalg.norm(Et_fluid_fft, axis=1) / np.linalg.norm(Et_fluid_fft, axis=1)[0] , 'o--', label=r"$E_{fluid}$")
    plt.xlabel(r"frequency [Hz]")
    plt.ylabel(r"magnitude")
    plt.legend()
    plt.grid(visible=True)
    plt.savefig("%s_fft.png"%(fprefix))

    plt.figure(figsize=(20, 6), dpi=300)
    for i in range(1, k):
        plt.subplot(2, 4, i)
        sf = 1
        if i == 1:
            sf = 1/len(tt)
        else:
            sf = 2/len(tt)


        plt.plot(xx, sf * np.real(Et_fluid_fft[i-1,:]), '--', label=r"$E_{fluid}$ cos($\omega_{%d}$ t)"%(i-1))
        plt.plot(xx, sf * np.imag(Et_fluid_fft[i-1,:]), '--', label=r"$E_{fluid}$ sin($\omega_{%d}$ t)"%(i-1))

        plt.plot(xx, sf * np.real(Et_fft[i-1,:]), label=r"$E_{bte}$ cos($\omega_{%d}$ t)"%(i-1))
        plt.plot(xx, sf * np.imag(Et_fft[i-1,:]), label=r"$E_{bte}$ sin($\omega_{%d}$ t)"%(i-1))

        plt.legend(fontsize=10)
        plt.grid(visible=True)
        plt.title(r"freq = %.4E (Hz)"%(freq[i-1]))
        plt.ylabel(r"E [V/m]")
        plt.xlabel(r"$\hat{x}$")
    
    plt.tight_layout()
    plt.savefig("%s_modes.png"%(fprefix))
    plt.close()

def multi_domain_svd(spec_sp:sp.SpectralExpansionSpherical, fl_tvx, xx : np.array, tt : np.array, domain_bdy:np.array, nr, nc, fprefix, xp):
    num_sh = len(spec_sp._sph_harm_lm)
    num_p  = spec_sp._p+1

    num_pts     = len(domain_bdy)
    num_domains = len(domain_bdy)-1
    assert num_domains > 0
    
    plt.figure(figsize=(4 * nc, 4 * nr), dpi=300)
    plt_cnt = 1
    #xidx   = list()
    xx_idx = np.arange(len(xx))
    for i in range(num_domains):
        xidx     = xx_idx[np.logical_and(xx>=domain_bdy[i], xx< domain_bdy[i+1])]
        fl_tvx_d = fl_tvx[:, :, xidx]
        
        num_t    = fl_tvx_d.shape[0]
        num_x    = fl_tvx_d.shape[2]
        
        sx_l     = list()
        sv_l     = list()
        for l in range(num_sh):
            fl_d = fl_tvx_d[:, l::num_sh, :]
            Ux, Sx, Vhx = xp.linalg.svd(fl_d.reshape(num_t * num_p, -1)) # Nt Nr x Nx
            Uv, Sv, Vhv = xp.linalg.svd(np.swapaxes(fl_d, 0, 1).reshape((num_p, num_t * num_x))) # Nr x Nt Nx

            if (xp == cp):
                Sx = asnumpy(Sx)
                Sv = asnumpy(Sv)

            sx_l.append(Sx)
            sv_l.append(Sv)
    
    
        plt.subplot(nr, nc, plt_cnt)
        for l in range(num_sh):
            plt.semilogy(sx_l[l]/sx_l[l][0], label=r"l=%d $\sigma_x$"%(l))
            plt.semilogy(sv_l[l]/sv_l[l][0], label=r"l=%d $\sigma_{v_r}$"%(l))

        plt.title(r"domain = (%.4E, %.4E)"%(domain_bdy[i], domain_bdy[i+1]))    
        plt.legend()
        plt.grid(visible=True)
        plt_cnt+=1

    #plt.suptitle(io_out)
    plt.tight_layout()
    plt.savefig("%s_svd_domains.png"%(fprefix))


xp                = cp
folder_name_ops   = "../1dglow_hybrid_Nx400/1Torr300K_100V_Ar_3sp2r"
folder_name       = "../1dglow_hybrid_Nx400/1Torr300K_100V_Ar_3sp2r_cycle"
folder_name_fluid = "../1dglow_fluid_Nx400/1Torr300K_100V_Ar_3sp2r_tab_cycle"
oprefix           = "1Torr300K100V"

# folder_name_ops   = "../1dglow_hybrid_Nx400/0.5Torr300K_100V_Ar_3sp2r"
# folder_name       = "../1dglow_hybrid_Nx400/0.5Torr300K_100V_Ar_3sp2r_cycle"
# folder_name_fluid = "../1dglow_fluid_Nx400/0.5Torr300K_100V_Ar_3sp2r_tab_cycle"
# oprefix           = "0.5Torr300K100V"

fvtx              = np.array([np.load("%s/1d_glow_%04d_v.npy"%(folder_name, i)) for i in range(101)])

gvtx_s            = np.load("../1dbte_rom/1Torr300K_100V_Ar_3sp2r/tb_0.00E+00_te_1.00E+00/x__tb_0.00E+00_te_1.00E+00_dt_5.00E-05_255x16x2x400.npy")
#gvtx_s           = np.load("../1dbte_rom/1Torr300K_100V_Ar_3sp2r/tb_0.00E+00_te_1.00E+00_ic100T/x__tb_0.00E+00_te_1.00E+00_dt_5.00E-04_255x16x2x400.npy")

mop               = np.load("%s/1d_glow_bte_mass_op.npy"%(folder_name_ops))
Po                = np.load("%s/1d_glow_bte_psh2o.npy"%(folder_name_ops))
Ps                = np.load("%s/1d_glow_bte_po2sh.npy"%(folder_name_ops))
g0                = np.load("%s/1d_glow_bte_op_g0.npy"%(folder_name_ops))
g2                = np.load("%s/1d_glow_bte_op_g2.npy"%(folder_name_ops))
temp_op           = np.load("%s/1d_glow_bte_temp_op.npy"%(folder_name_ops))
mu_op             = np.load("%s/1d_glow_bte_mobility.npy"%(folder_name_ops))
De_op             = np.load("%s/1d_glow_bte_diffusion.npy"%(folder_name_ops))
bte_op            = {"mass": mop , "po2sh": Ps, "psh2o": Po, "g0": g0, "g2": g2, "mobility": mu_op, "De": De_op}

args              = load_run_args("%s/1d_glow_args.txt"%(folder_name))
ff                = h5py.File("%s/macro.h5"%(folder_name), 'r')
Et                = np.array(ff["E[Vm^-1]"][()])
ne                = np.array(ff["ne[m^-3]"][()])
Te                = np.array(ff["Te[eV]"][()])
xx                = np.array(ff["x[-1,1]"][()])
tt                = np.array(ff["time[T]"][()])
uz                = np.array(ff["uz[ms^{-1}]"][()])
ki                = np.array(ff["ki[m^3s^{-1}]"][()])
mue               = plot_utils.compute_mobility(args, bte_op, fvtx, EbyN=1.0)

ev                = np.linspace(0, 80, 1024)
spec_sp, col_list = plot_utils.gen_spec_sp(args)

ff_fluid          = h5py.File("%s/macro.h5"%(folder_name_fluid))
Et_fluid          = np.array(ff_fluid["E[Vm^-1]"][()])

freq              = np.fft.rfftfreq(len(tt[:-1]), 1/len(tt[:-1])/13.56e6)
Et_fluid_fft      = np.fft.rfft(Et_fluid, axis=0)
Et_fft            = np.fft.rfft(Et, axis=0)

fvtx_s            = asnumpy(xp.einsum("tlx,vl->tvx", xp.array(fvtx), xp.array(Ps)))
ne_f              = np.einsum("tlx,l->tx", fvtx_s, mop)
ne_g              = np.einsum("tlx,l->tx", gvtx_s, mop)
glm               = compute_radial_components(args,  bte_op, spec_sp, ev, gvtx_s, normalize=True)
flm               = compute_radial_components(args,  bte_op, spec_sp, ev, fvtx_s, normalize=True)

ofolder_name      = "glow_rom_analysis_plots"
make_dir("%s"%(ofolder_name))

# multi_domain_svd(spec_sp, xp.array(fvtx_s), np.array(xx), tt, np.array([-1, -0.75, -0.5, 0.5, 0.75, 1]), 2, 3, "%s/%s_md"%(ofolder_name, oprefix), cp)
# multi_domain_svd(spec_sp, xp.array(fvtx_s), np.array(xx), tt, np.array([-1, 1]), 1, 1, "%s/%s_sd"%(ofolder_name, oprefix), cp)
#plot_Et_fft_comp(Et[:-1, :], Et_fluid[:-1, :], tt[:-1], 1/len(tt[:-1])/13.56e6, 9, "%s/E%s"%(ofolder_name, oprefix))

#fft_rom_analysis(fvtx_s, gvtx_s, tt, (1e-3)/13.56e6, xx, bte_op, "%s/%s_fd_rom"%(ofolder_name, oprefix))


gvtx    = asnumpy(xp.einsum("tlx,vl->tvx", xp.array(gvtx_s), xp.array(Po)))
rr_g    = plot_utils.compute_rate_coefficients(args, bte_op, gvtx, ["g0", "g2"])
mue_g   = plot_utils.compute_mobility(args, bte_op, gvtx, EbyN=1.0)



#fvtx             = np.array(ff["Ftvx"][()])

# make_dir("%s/fft"%(folder_name))
fft_reconstruct(Et[:-1, :], 0, tt[:-1], 1/len(tt[:-1])/13.56e6, 13, "%s/%s_Et"%(ofolder_name, oprefix))
fft_reconstruct(ne[:-1, :], 0, tt[:-1], 1/len(tt[:-1])/13.56e6, 13, "%s/%s_ne"%(ofolder_name, oprefix))
fft_reconstruct(ki[:-1, :], 0, tt[:-1], 1/len(tt[:-1])/13.56e6, 13, "%s/%s_ki"%(ofolder_name, oprefix))
fft_reconstruct(mue[:-1, :], 0, tt[:-1], 1/len(tt[:-1])/13.56e6, 13, "%s/%s_mue"%(ofolder_name, oprefix))

fft_reconstruct(rr_g[1][:-1, :], 0, tt[:-1], 1/len(tt[:-1])/13.56e6, 13, "%s/%s_ki_g"%(ofolder_name, oprefix))
fft_reconstruct(mue_g  [:-1, :], 0, tt[:-1], 1/len(tt[:-1])/13.56e6, 13, "%s/%s_mue_g"%(ofolder_name, oprefix))

# fft_reconstruct(ne[:-1, :], 0, tt[:-1], (1e-3)/13.56e6, 13, "%s/fft/ne"%(folder_name))
# fft_reconstruct(Te[:-1, :], 0, tt[:-1], (1e-3)/13.56e6, 13, "%s/fft/Te"%(folder_name))


fft_edf(fvtx_s[:-1], 0, tt[:-1], (1e-3)/13.56e6, 13, ev, spec_sp, bte_op, "%s/%s_edf_f"%(ofolder_name, oprefix))
# fft_edf(gvtx_s[:-1], 0, tt[:-1], (1e-3)/13.56e6, 13, ev, spec_sp, bte_op, "%s/fft/edf_g"%(folder_name))
# fft_edf(np.einsum("tvx,tx->tvx", gvtx_s, (1/ne_g))[:-1],  0, tt[:-1], (1e-3)/13.56e6, 13, ev, spec_sp, bte_op, "%s/fft/edf_hat_g"%(folder_name))
fft_edf(np.einsum("tvx,tx->tvx", fvtx_s, (1/ne_f))[:-1],  0, tt[:-1], (1e-3)/13.56e6, 13, ev, spec_sp, bte_op, "%s/%s_edf_f_hat"%(ofolder_name, oprefix))




#fft_rom_analysis(fvtx_s, gvtx_s, tt, (1e-3)/13.56e6, xx, bte_op)

    





    













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
