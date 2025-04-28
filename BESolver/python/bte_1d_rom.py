"""
@brief: BTE reduce order modeling
"""
try:
    import cupy as cp
    import cupyx.scipy.sparse.linalg
except ImportError:
    print("Please install CuPy for GPU use")

import matplotlib.pyplot as plt
import numpy as np
import glow1d_utils
import sys
from enum import Enum
import h5py
from profile_t import profile_t
from matplotlib.colors import TABLEAU_COLORS, same_color
from glowdischarge_boltzmann_1d import glow1d_boltzmann, args_parse
import scipy
import scipy.constants
import scipy.sparse.linalg
import scipy.optimize
import os
import rom_utils
import argparse

def make_dir(dir_name):
    # Check whether the specified path exists or not
    isExist = os.path.exists(dir_name)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(dir_name)
       print("directory %s is created!"%(dir_name))

class ROM_TYPE(Enum):
    POD    = 0 # POD - orthogonal    (SVD based)
    DLR    = 1 # POD - dynamical low rank
    POD_ID = 2 # POD - Non-orthogonal (ID based)

class TIMER:
    ROM_X = 0
    ROM_V = 1

    FOM_X = 2
    FOM_V = 3

    ROM_VX= 4

    LAST  = 5

def plot_solution(bte : glow1d_boltzmann, bte_rom, F0, F1, fname, time, p_F0=True, p_F1=True):
    """
    F0 : FOM
    F1 : ROM
    """
    xp      = bte.xp_module
    param   = bte.param
    args    = bte.args
    mass_op = bte.op_mass
    temp_op = bte.op_temp
    Ps      = bte.op_po2sh
    Po      = bte.op_psh2o

    spec_sp = bte.op_spec_sp
    num_p   = spec_sp._p + 1
    num_sh  = len(spec_sp._sph_harm_lm)
    num_x   = len(bte.xp)

    F0_lm   = xp.dot(Ps, F0)
    F1_lm   = xp.dot(Ps, F1)
    
    rom_lm  = bte_rom.rom_modes
    G0_lm   = xp.zeros((num_p* num_sh, num_x))
    if rom_lm is not None:
        PUv     = [xp.dot(bte_rom.Uv[l], bte_rom.Uv[l].T) for l in range(rom_lm)]
        PVx     = [xp.dot(bte_rom.Vx[l], bte_rom.Vx[l].T) for l in range(rom_lm)]

        for l in range(rom_lm):
            G0_lm[l::num_sh] = xp.dot(PUv[l], xp.dot(F0_lm[0::num_sh], PVx[l]))
    
    
    F0_lm_n = bte.bte_eedf_normalization(F0_lm)
    F1_lm_n = bte.bte_eedf_normalization(F1_lm)
    G0_lm_n = bte.bte_eedf_normalization(G0_lm)

    ev_range  = (bte.ev_lim[0], bte.ev_lim[1])
    ev_grid   = np.linspace(ev_range[0], ev_range[1], 1024)

    F0_rc     = bte.compute_radial_components(ev_grid, xp.asnumpy(F0_lm_n))
    F1_rc     = bte.compute_radial_components(ev_grid, xp.asnumpy(F1_lm_n))


    ne0     = xp.dot(mass_op, F0_lm)
    ne1     = xp.dot(mass_op, F1_lm)
    ne2     = xp.dot(mass_op, G0_lm)

    Te0     = xp.dot(temp_op, F0_lm) / ne0
    Te1     = xp.dot(temp_op, F1_lm) / ne1
    Te2     = xp.dot(temp_op, G0_lm) / ne2

    g0_0    = xp.dot(bte.op_rate[0], F0_lm_n[0::num_sh])
    g0_1    = xp.dot(bte.op_rate[0], F1_lm_n[0::num_sh])
    g0_2    = xp.dot(bte.op_rate[0], G0_lm_n[0::num_sh])

    g2_0    = xp.dot(bte.op_rate[1], F0_lm_n[0::num_sh])
    g2_1    = xp.dot(bte.op_rate[1], F1_lm_n[0::num_sh])
    g2_2    = xp.dot(bte.op_rate[1], G0_lm_n[0::num_sh])


    xx      = bte.xp

    if xp == cp:
        ne0  = xp.asnumpy(ne0)
        ne1  = xp.asnumpy(ne1)
        ne2  = xp.asnumpy(ne2)

        Te0  = xp.asnumpy(Te0)
        Te1  = xp.asnumpy(Te1)
        Te2  = xp.asnumpy(Te2)

        g0_0 = xp.asnumpy(g0_0)
        g0_1 = xp.asnumpy(g0_1)
        g0_2 = xp.asnumpy(g0_2)

        g2_0 = xp.asnumpy(g2_0)
        g2_1 = xp.asnumpy(g2_1)
        g2_2 = xp.asnumpy(g2_2)

        xx   = xp.asnumpy(xx)


    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plt.figure(figsize=(16, 8), dpi=200)
    plt.subplot(2, 4, 1)
    citer = iter(cycle)
    if(p_F0): plt.semilogy(xx, ne0 * param.np0,      label=r"FOM", color=next(citer))
    if(p_F1): plt.semilogy(xx, ne1 * param.np0,'--', label=r"ROM", color=next(citer))
    #plt.semilogy(xx, ne2 * param.np0, label=r"$P_{U}$ FOM $P_{V}$")
    plt.xlabel(r"$\hat{x}$")
    plt.ylabel(r"$n_e$ $[m^{-3}]$")
    plt.grid(visible=True)
    plt.legend()

    plt.subplot(2, 4, 2)
    citer = iter(cycle)
    if(p_F0): plt.plot(xx, Te0,      label=r"FOM", color=next(citer))
    if(p_F1): plt.plot(xx, Te1,'--', label=r"ROM", color=next(citer))
    #plt.plot(xx, Te2, label=r"$P_{U}$ FOM $P_{V}$")
    plt.xlabel(r"$\hat{x}$")
    plt.ylabel(r"$T_e$ $[eV]$")
    plt.grid(visible=True)
    plt.legend()

    plt.subplot(2, 4, 3)
    citer = iter(cycle)
    if(p_F0): plt.semilogy(xx, g0_0      , label=r"FOM", color=next(citer))
    if(p_F1): plt.semilogy(xx, g0_1, '--', label=r"ROM", color=next(citer))
    #plt.semilogy(xx, g0_2, label=r"$P_{U}$ FOM $P_{V}$")
    plt.xlabel(r"$\hat{x}$")
    plt.ylabel(r"rate coefficient $[m^{3}s^{-1}]$")
    plt.title(r"momentum transfer")
    plt.grid(visible=True)
    plt.legend()

    plt.subplot(2, 4, 4)
    citer = iter(cycle)
    if(p_F0): plt.semilogy(xx, g2_0,       label=r"FOM", color=next(citer))
    if(p_F1): plt.semilogy(xx, g2_1, '--' ,label=r"ROM", color=next(citer))
    #plt.semilogy(xx, g2_2, label=r"$P_{U}$ FOM $P_{V}$")
    plt.xlabel(r"$\hat{x}$")
    plt.ylabel(r"rate coefficient $[m^{3}s^{-1}]$")
    plt.title(r"ionization")
    plt.grid(visible=True)
    plt.legend()


    plt.subplot(2, 4, 5)
    citer = iter(cycle)
    for xidx in range(0, len(xx), len(xx)//3):
        clr = next(citer)
        if(p_F0): plt.semilogy(ev_grid, np.abs(F0_rc[xidx, 0])      , color=clr, label=r"FOM (x=%.2f)"%(xx[xidx]))
        if(p_F1): plt.semilogy(ev_grid, np.abs(F1_rc[xidx, 0]),'--' , color=clr, label=r"ROM (x=%.2f)"%(xx[xidx]))

    plt.grid(visible=True)
    plt.legend()
    plt.xlabel(r"energy [eV]")
    plt.ylabel(r"$f_0$  [$eV^{-3/2}$]")

    plt.subplot(2, 4, 6)
    citer = iter(cycle)
    for xidx in range(0, len(xx), len(xx)//3):
        clr = next(citer)
        if(p_F0): plt.semilogy(ev_grid, np.abs(F0_rc[xidx, 1])      , color=clr, label=r"FOM (x=%.2f)"%(xx[xidx]))
        if(p_F1): plt.semilogy(ev_grid, np.abs(F1_rc[xidx, 1]),'--' , color=clr, label=r"ROM (x=%.2f)"%(xx[xidx]))

    plt.grid(visible=True)
    plt.legend()
    plt.xlabel(r"energy [eV]")
    plt.ylabel(r"$f_1$  [$eV^{-3/2}$]")

    plt.suptitle(r"time = %.4E [T]"%(time))
    plt.tight_layout()

    plt.savefig("%s"%(fname))
    plt.close()
    return

class boltzmann_1d_rom():

    def __init__(self, bte_solver : glow1d_boltzmann):
        self.bte_solver = bte_solver
        self.args       = self.bte_solver.args
        self.rom_type   = ROM_TYPE.POD
        self.pinv_eps   = 1e-10
        self.rom_modes  = None
        self.profile    = True
        self.timer      = [profile_t("") for i in range(TIMER.LAST)]

        
        spec_sp         = self.bte_solver.op_spec_sp
        self.num_p      = spec_sp._p + 1
        self.num_vt     = len(self.bte_solver.xp_vt)
        self.num_sh     = len(spec_sp._sph_harm_lm)
        self.num_x      = len(self.bte_solver.xp)
        
        return 
    
    def init(self):
        bte     = self.bte_solver
        param   = bte.param
        xp      = bte.xp_module
        Cen     = bte.op_col_en
        Ctg     = bte.op_col_gT

        Cop     = bte.param.np0 * bte.param.n0 * (bte.op_col_en + bte.args.Tg * Ctg)
        Av      = bte.op_adv_v
        spec_sp = bte.op_spec_sp
        mm      = xp.asarray(spec_sp.compute_mass_matrix())
        mm_inv  = xp.asarray(spec_sp.inverse_mass_mat())

        num_p   = spec_sp._p + 1
        num_vt  = len(bte.xp_vt)
        num_sh  = len(spec_sp._sph_harm_lm)

        Mvr     = mm[0::num_sh, 0::num_sh]
        Mv_inv  = mm_inv

        Ps      = bte.op_po2sh
        Po      = bte.op_psh2o

        Ax      = xp.kron(bte.op_adv_x, xp.diag(bte.xp_cos_vt))
        Gx      = xp.dot(Ps, xp.dot(Ax, Po))

        Dx      = bte.Dp

        # f0       = xp.random.rand(num_p * num_vt)
        # y1       = xp.dot(Ax, f0).reshape((num_p, num_vt))
        # y0       = xp.einsum("al,lc->ac", bte.op_adv_x, f0.reshape((num_p, num_vt)))
        # y0       = xp.einsum("i,ai->ai" , bte.xp_cos_vt, y0)
        # print("yy = ", xp.linalg.norm(y0 - y1) / xp.linalg.norm(y0))

        self.Ax  = Ax
        self.Gx  = Gx
        self.Dx  = Dx
        self.Cop = Cop
        self.Av  = Av

        return

    def generate_initial_data(self, ne_scale: np.array, Te: np.array):
        bte = self.bte_solver
        xp  = bte.xp_module

        spec_sp     = bte.op_spec_sp
        mmat        = spec_sp.compute_mass_matrix()
        mmat_inv    = spec_sp.inverse_mass_mat(Mmat = mmat)
        vth         = bte.bs_vth
        
        Ps          = bte.op_po2sh
        Po          = bte.op_psh2o
        PoPs        = Po @ Ps

        mass_op     = bte.op_mass
        temp_op     = bte.op_temp
          
        gmx,gmw     = spec_sp._basis_p.Gauss_Pn(spec_sp._num_q_radial)
        Vqr_gmx     = spec_sp.Vq_r(gmx, 0, 1)
          
        num_p       = spec_sp._p +1
        num_sh      = len(spec_sp._sph_harm_lm)
        num_x       = len(bte.xp)
        num_vt      = len(bte.xp_cos_vt)
        h_init      = xp.zeros((len(Te), num_p * num_sh))

        gmx, gmw    = xp.asarray(gmx), xp.asarray(gmw)
        Vqr_gmx     = xp.asarray(Vqr_gmx)
        mmat_inv    = xp.asarray(mmat_inv)

        ns          = len(Te) * len(ne_scale)
          
        ev_max_ext        = (spec_sp._basis_p._t_unique[-1] * vth/bte.c_gamma)**2
        print("v-grid max = %.4E (eV) extended to = %.4E (eV)" %(bte.ev_lim[1], ev_max_ext))

        for i in range(len(Te)):
            v_ratio              = (bte.c_gamma * xp.sqrt(Te[i])/vth)
            hv                   = lambda v : (1/xp.sqrt(np.pi)**3) * xp.exp(-((v/v_ratio)**2)) / v_ratio**3
            h_init[i, 0::num_sh] = xp.sqrt(4 * xp.pi) * xp.dot(mmat_inv[0::num_sh,0::num_sh], xp.dot(Vqr_gmx * hv(gmx) * gmx**2, gmw))
            m0                   = xp.dot(mass_op, h_init[i])
            h_init[i]            = h_init[i]/m0

        xx      = bte.param.L * (bte.xp + 1)
        ne_base = 1e6 * (1e7 + 1e9 * (1-0.5 * xx/bte.param.L)**2 * (0.5 * xx/bte.param.L)**2) / bte.param.np0

        ne      = np.array([ne_base * ne_scale[i] for i in range(len(ne_scale))])

        x_init  = xp.einsum("il,vl->iv"  , h_init, Po)
        x_init  = xp.einsum("iv,jx->jivx", x_init, ne)

        print(x_init.shape)

        x_init_s  = xp.einsum("ijlx,vl->ijvx", x_init, Ps)
        x_init_m  = xp.einsum("ijlx,l->ijx", x_init_s, mass_op)
        x_init_te = xp.einsum("ijlx,l->ijx", xp.einsum("ijvx,ijx->ijvx", x_init_s, (1/x_init_m)), temp_op)

        print(xp.max(x_init_m , axis=2) * bte.param.np0)
        print(xp.max(x_init_te, axis=2))

        return x_init.reshape((len(ne_scale) * len(Te), -1))
    
    def sample_fom(self, Et, v0, tb, te, dt, n_samples, fprefix="", load_from_file=False):
        bte     = self.bte_solver
        tt      = tb
        xp      = bte.xp_module

        folder_name = "%s/tb_%.2E_te_%.2E"%(self.args.dir, tb, te)
        vfname      = "%s/x_%s_tb_%.2E_te_%.2E.npy"%(folder_name, fprefix, tb, te)
        make_dir(folder_name)

        if (load_from_file == True):
            v_all = xp.load(vfname)
            print("read - v_all.shape = ", v_all.shape)
            return v_all.reshape((n_samples, -1)).T

        steps   = int(np.ceil((te-tb)/dt))
        io_freq = steps // (n_samples-1)
        steps   = io_freq * (n_samples-1)

        Ps      = bte.op_po2sh
        Po      = bte.op_psh2o

        v0      = v0.reshape((self.num_p * self.num_vt, self.num_x))
        v       = xp.copy(v0)
        #v_all   = xp.zeros(tuple([n_samples]) + v0.shape)
        v_all   = xp.zeros(tuple([n_samples]) + (self.num_p * self.num_sh, self.num_x))
        ts      = xp.zeros(n_samples)

        assert (Et(0) == Et(0.1)).all() == True
        self.flow_map_fom_op_v(Et(0), dt)
        print("!!!using prefactored v-solve")
        
        for iter in range(steps+1):
            if (iter % io_freq == 0):
                print(iter//io_freq, v_all.shape, type(v))
                #v_all[int(iter//io_freq), :, :] = v[:, :]
                v_all[int(iter//io_freq), :, :] = Ps @ v
                ts   [int(iter//io_freq)]       = tt

                if (iter % (io_freq) == 0):
                    plot_solution(bte, self, v, v, "%s/tidx_%s_%d.png"%(folder_name, fprefix, int(iter//io_freq)), tt, p_F0=True, p_F1=False)

            # bte.bs_E            = Et(tt)
            # v                   = bte.step_bte_x(v, tt, dt * 0.5)
            # v                   = bte.step_bte_v(v, None, tt, dt, ts_type="BE", verbose=1)
            # v                   = bte.step_bte_x(v, tt + 0.5 * dt, dt * 0.5)
            v                     = self.step_fom_op_split(Et(tt), v, tt, dt, verbose = 0, prefactored_vsolve=True)
            tt                   += dt
        
        xp.save(vfname,v_all)
        print("v_all.shape = ", v_all.shape)
        print("ts = ", ts)

        

        return v_all.reshape((n_samples, -1)).T

    def init_rom_basis_from_snapshots(self, xt, rank_vx):
        """
        xt - (num_p * num_vt * num_x, num_snapshots)
        """
        bte     = self.bte_solver
        xp      = bte.xp_module
        Ps      = bte.op_po2sh
        Po      = bte.op_psh2o

        ns      = xt.shape[1]
        assert xp == cp

        # assert xt.shape[0] == self.num_p * self.num_vt * self.num_x
        # yt      = xt.reshape((self.num_p, self.num_vt, self.num_x, ns))
        # yt      = (Ps @ yt.reshape((self.num_p * self.num_vt, self.num_x * ns))).reshape((self.num_p, self.num_sh, self.num_x, ns))
        # zt      = (yt.reshape((self.num_p * self.num_sh * self.num_x, ns)).T).reshape((ns, self.num_p * self.num_sh, self.num_x))

        yt      = xt.reshape((self.num_p, self.num_sh, self.num_x, ns))
        zt      = (yt.reshape((self.num_p * self.num_sh * self.num_x, ns)).T).reshape((ns, self.num_p * self.num_sh, self.num_x))

        assert (zt[0, 0::self.num_sh, :] == yt[:, 0, :, 0]).all()==True
        
        # svd based basis
        def svd(fl, eps_x, eps_v, xp):
            num_t, num_p, num_x = fl.shape
            assert num_p == self.num_p and num_x == self.num_x

            Ux, Sx, Vhx         = xp.linalg.svd(fl.reshape(num_t * num_p, -1)) # Nt Nr x Nx
            Uv, Sv, Vhv         = xp.linalg.svd(xp.swapaxes(fl, 0, 1).reshape((num_p, num_t * num_x))) # Nr x Nt Nx

            kr  = len(Sv[Sv > Sv[0] * eps_v])
            kx  = len(Sx[Sx > Sx[0] * eps_x])

            # Vx  = Vhx[0:rank_vx[1],         :   ].T
            # Uv  = Uv [:           , 0:rank_vx[0]]

            Vx  = Vhx[0:kx,         :   ].T
            Uv  = Uv [:           , 0:kr]

            return Uv, Vx, (Sx, Sv)
        
        # interpolative decomposition
        def id_decomp(fl, eps_x, eps_v, xp):
            if xp == cp:
                fl = cp.asnumpy(fl)

            num_t, num_p, num_x = fl.shape
            Fx                  = fl.reshape(num_t * num_p, -1).T
            Fv                  = xp.swapaxes(fl, 0, 1).reshape((num_p, num_t * num_x))
            
            Vx, _               = rom_utils.adaptive_id_decomp(Fx, eps_x, k_init=10, use_sampling=False)
            Uv, _               = rom_utils.adaptive_id_decomp(Fv, eps_v, k_init=10, use_sampling=False)

            if xp == cp:
                Uv = cp.asarray(Uv)
                Vx = cp.asarray(Vx)

            return Uv, Vx


        Uv  = list()
        Vx  = list()

        svalues=list()
        for l in range(self.num_sh):
            fl = zt[:, l :: self.num_sh, :]
            if (self.rom_type == ROM_TYPE.POD):
                uv, vx , s       = svd(fl, rank_vx[1], rank_vx[0], xp=xp)
                svalues.append(s)
            elif(self.rom_type == ROM_TYPE.POD_ID):
                uv, vx           = id_decomp(fl, eps_x, eps_v, xp=xp)
            elif (self.rom_type == ROM_TYPE.DLR):
                Fs       = xp.dot(Ps, v0)
                u, s, vt = xp.linalg.svd(Fs[l::num_sh])
                rr       = max(100, len(s[s>=s[0] * threshold[l]]))
                uv       = u   [:, 0:rr]
                vx       = vt.T[:, 0:rr]
            else:
                raise NotImplementedError

            Uv.append(uv)
            Vx.append(vx)

        self.Uv = Uv
        self.Vx = Vx
        self.rom_modes = self.num_sh

        if (self.rom_type == ROM_TYPE.POD):
            plt.figure(figsize=(6,6), dpi=200)
            plt.semilogy(xp.asnumpy(svalues[0][0]/svalues[0][0][0]), label=r"$\sigma_x^{l=%d}$"%(0))
            plt.semilogy(xp.asnumpy(svalues[0][1]/svalues[0][1][0]), label=r"$\sigma_v^{l=%d}$"%(0))
            plt.semilogy(xp.asnumpy(svalues[1][0]/svalues[1][0][0]), label=r"$\sigma_x^{l=%d}$"%(1))
            plt.semilogy(xp.asnumpy(svalues[1][1]/svalues[1][1][0]), label=r"$\sigma_v^{l=%d}$"%(1))
            plt.grid(visible=True)
            plt.legend()
            plt.title(r"$N_r$=%d $N_l$=%d $N_x$=%d"%(self.num_p, self.num_sh, self.num_x))
            plt.ylabel(r"normalized singular value")
            plt.tight_layout()
            plt.savefig("%s_svd.png"%(bte.args.fname))
            plt.close()
        

        spec_sp  = bte.op_spec_sp
        num_p    = spec_sp._p + 1
        num_vt   = len(bte.xp_vt)
        num_sh   = len(spec_sp._sph_harm_lm)
        num_x    = len(bte.xp)

        Ps       = bte.op_po2sh
        Po       = bte.op_psh2o

        self.vec_kr_len    = np.array([self.Uv[l].shape[1] for l in range(self.rom_modes)], dtype=np.int32)
        self.vec_kx_len    = np.array([self.Vx[l].shape[1] for l in range(self.rom_modes)], dtype=np.int32)
        
        self.vec_shape     = [(self.Uv[l].shape[1], self.Vx[l].shape[1]) for l in range(self.rom_modes)]
        self.vec_len       = np.array([self.Uv[l].shape[1] * self.Vx[l].shape[1] for l in range(self.rom_modes)], dtype=np.int32)
        self.vec_offset    = np.array([0 for l in range(self.rom_modes)], dtype=np.int32)
        self.vec_kr_offset = np.array([0 for l in range(self.rom_modes)], dtype=np.int32)
        self.vec_kx_offset = np.array([0 for l in range(self.rom_modes)], dtype=np.int32)

        for l in range(1, self.rom_modes):
            self.vec_offset[l]    = self.vec_offset[l-1]     + self.vec_len[l-1]
            self.vec_kr_offset[l] = self.vec_kr_offset[l-1] + self.vec_kr_len[l-1]
            self.vec_kx_offset[l] = self.vec_kx_offset[l-1] + self.vec_kx_len[l-1]

        self.vec_idx   = [xp.arange(self.vec_offset[l], self.vec_offset[l] + self.vec_len[l]) for l in range(self.rom_modes)]
        self.dof_rom   = np.sum(np.array(self.vec_len))
        print(str(self.vec_shape))
        # with open("%s_rom_size.txt"%(bte.args.fname), 'a') as f:
        #     f.write("%s : "%(basis_id) + str(self.vec_shape)+"\n")
        #     f.close()
        
        return 
    
    def get_rom_lm(self, Fr, l, m=0):
        return Fr.reshape((-1))[self.vec_idx[l]].reshape(self.vec_shape[l])

    def create_rom_vec(self):
        bte              = self.bte_solver
        xp               = bte.xp_module
        return xp.zeros(self.dof_rom)

    def append_vec(self, Fr_lm):
        bte   = self.bte_solver
        xp    = bte.xp_module
        Fr    = xp.array([])

        for l in range(len(Fr_lm)):
            Fr = xp.append(Fr, Fr_lm[l].reshape((-1)))

        return Fr

    def encode(self, F):
        rom_modes    = self.rom_modes
        bte          = self.bte_solver
        Uv           = self.Uv
        Vx           = self.Vx

        xp           = bte.xp_module
        Ps           = bte.op_po2sh
        spec_sp      = bte.op_spec_sp
        num_p        = spec_sp._p + 1
        num_sh       = len(spec_sp._sph_harm_lm)

        F_lm         = xp.dot(Ps, F)

        if (self.rom_type  == ROM_TYPE.POD or self.rom_type == ROM_TYPE.DLR):
            Fr_lm        = [xp.dot(Uv[i].T, xp.dot(F_lm[i::num_sh, :], Vx[i])) for i in range(rom_modes)]
        elif(self.rom_type == ROM_TYPE.POD_ID):
            Fr_lm        = [xp.dot(self.Uv_pinv[i], xp.dot(F_lm[i::num_sh, :], self.Vx[i])) for i in range(rom_modes)]
        else:
            raise NotImplementedError
        
        Fr               = xp.empty(self.dof_rom) #xp.array(Fr_lm[0].reshape((-1)))
        for l in range(0, rom_modes):
            Fr[self.vec_idx[l]] = Fr_lm[l].reshape((-1))

        return Fr

    def decode(self, Fr):
        bte     = self.bte_solver
        spec_sp = bte.op_spec_sp
        xp      = bte.xp_module
        Po      = bte.op_psh2o
        Uv      = self.Uv
        Vx      = self.Vx

        num_p   = spec_sp._p + 1
        num_sh  = len(spec_sp._sph_harm_lm)
        num_x   = len(bte.xp)
        F       = xp.zeros((num_p * num_sh, num_x))

        if (self.rom_type  == ROM_TYPE.POD or self.rom_type == ROM_TYPE.DLR):
            for l in range(self.rom_modes):
                Frl              = self.get_rom_lm(Fr, l, m=0)
                F[l::num_sh, : ] = xp.dot(Uv[l], xp.dot(Frl, Vx[l].T))
        elif(self.rom_type == ROM_TYPE.POD_ID):
            for l in range(self.rom_modes):
                Frl              = self.get_rom_lm(Fr, l, m=0)
                F[l::num_sh, : ] = xp.dot(Uv[l], xp.dot(Frl, self.Vx_pinv[l]))
        else:
            raise NotImplementedError
        
        return xp.dot(Po, F)

    def assemble_encode_op(self):
        bte     = self.bte_solver
        xp      = bte.xp_module
        
        dof_rom = self.dof_rom
        dof_fom = self.num_p * self.num_vt * self.num_x
        
        L       = cupyx.scipy.sparse.linalg.LinearOperator((dof_rom, dof_fom), matvec = lambda x : self.encode(x.reshape(self.num_p * self.num_vt, self.num_x)), dtype=xp.float64)
        Lop     = rom_utils.assemble_mat((dof_rom, dof_fom), L, xp=xp)

        x       = xp.random.rand(dof_fom)
        y       = self.encode(x.reshape(self.num_p * self.num_vt, self.num_x))
        y1      = Lop @ x
        
        a1      = xp.linalg.norm(y - y1)/xp.linalg.norm(y)
        #print(y.shape, y1.shape, a1)
        assert a1 < 1e-14
        return Lop 

    def assemble_decode_op(self):
        bte     = self.bte_solver
        xp      = bte.xp_module
        
        dof_rom = self.dof_rom
        dof_fom = self.num_p * self.num_vt * self.num_x
        
        L       = cupyx.scipy.sparse.linalg.LinearOperator((dof_fom, dof_rom), matvec = lambda x : self.decode(x).reshape(-1), dtype=xp.float64)
        Lop     = rom_utils.assemble_mat((dof_fom, dof_rom), L, xp=xp)

        x       = xp.random.rand(dof_rom)
        y       = self.decode(x).reshape((-1))
        y1      = Lop @ x

        a1      = xp.linalg.norm(y - y1)/xp.linalg.norm(y)
        assert a1 < 1e-14
        #print(y.shape, y1.shape, a1)
        return Lop 

    def assemble_rom_op(self, Ef):
        """
        performs matassembly for the VX rom-space op.
        """

        bte     = self.bte_solver
        xp      = bte.xp_module
        param   = bte.param

        spec_sp = bte.op_spec_sp
        num_p   = spec_sp._p + 1
        num_sh  = len(spec_sp._sph_harm_lm)
        num_x   = len(bte.xp)
        num_vt  = len(bte.xp_vt)

        DxT     = self.Dx.T
        po2sh   = bte.op_po2sh
        psh2o   = bte.op_psh2o
        xp_vt_l = bte.xp_vt_l
        xp_vt_r = bte.xp_vt_r

        def rom_vx(Xr, Ef, xp=xp):
            Fo   = self.decode(Xr)
            Fs   = xp.dot(po2sh, Fo)
            Qv   = param.tau * (xp.dot(self.Cop, Fs) + Ef * xp.dot(self.Av, Fs))
            Qv   = xp.dot(psh2o, Qv)
            Qv   = (-xp.dot(self.Ax, xp.dot(Fo, DxT)) +  Qv)

            Qv[xp_vt_l,  0] = 0.0
            Qv[xp_vt_r, -1] = 0.0

            return self.encode(Qv)

        Ndof = self.dof_rom
        Lop  = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec = lambda x: rom_vx(x, Ef, xp=xp), dtype=xp.float64)

        Lop_mat = rom_utils.assemble_mat((self.dof_rom, self.dof_rom),Lop, xp=xp)
        return Lop_mat

    def assemble_rom_v_flow_map(self, Ef, dt):
        """
        Assembles the flow map for the vx space. 
        """

        bte     = self.bte_solver
        xp      = bte.xp_module
        param   = bte.param

        spec_sp = bte.op_spec_sp
        num_p   = spec_sp._p + 1
        num_sh  = len(spec_sp._sph_harm_lm)
        num_x   = len(bte.xp)
        num_vt  = len(bte.xp_vt)

        DxT     = self.Dx.T
        po2sh   = bte.op_po2sh
        psh2o   = bte.op_psh2o
        xp_vt_l = bte.xp_vt_l
        xp_vt_r = bte.xp_vt_r

        def flmap_v(Xr, Ef, dt, xp=xp):
            Fo                  = self.decode(Xr)
            tt                  = 0.0
            bte.bs_E            = Ef
            v                   = bte.step_bte_v(Fo, None, tt, dt, ts_type="BE", verbose=0)
            return self.encode(v)

        Ndof = self.dof_rom
        Lop  = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec = lambda x: flmap_v(x, Ef, dt, xp=xp), dtype=xp.float64)
        Lop_mat = rom_utils.assemble_mat((self.dof_rom, self.dof_rom),Lop, xp=xp)
        return Lop_mat

    def assemble_rom_x_op(self):
        bte     = self.bte_solver
        xp      = bte.xp_module
        param   = bte.param

        DxT     = self.Dx.T
        # po2sh   = bte.op_po2sh
        # psh2o   = bte.op_psh2o
        xp_vt_l = bte.xp_vt_l
        xp_vt_r = bte.xp_vt_r

        def rom_x(Xr, xp=xp):
            Fo   = self.decode(Xr)
            Qv   = (-xp.dot(self.Ax, xp.dot(Fo, DxT)))

            Qv[xp_vt_l,  0] = 0.0
            Qv[xp_vt_r, -1] = 0.0
            return self.encode(Qv)
        
        Ndof    = self.dof_rom
        Lop     = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec = lambda x: rom_x(x, xp=xp), dtype=xp.float64)
        Lop_mat = rom_utils.assemble_mat((self.dof_rom, self.dof_rom),Lop, xp=xp)
        return Lop_mat
    
    def assemble_rom_v_op(self, Ef):
        """
        performs mat-assembly for the v-space rom-space op.
        """

        bte     = self.bte_solver
        xp      = bte.xp_module
        param   = bte.param

        spec_sp = bte.op_spec_sp
        num_p   = spec_sp._p + 1
        num_sh  = len(spec_sp._sph_harm_lm)
        num_x   = len(bte.xp)
        num_vt  = len(bte.xp_vt)

        po2sh   = bte.op_po2sh
        psh2o   = bte.op_psh2o

        def rom_v(Xr, Ef, xp=xp):
            Fo   = self.decode(Xr)
            Fs   = xp.dot(po2sh, Fo)
            Qv   = param.tau * (xp.dot(self.Cop, Fs) + Ef * xp.dot(self.Av, Fs))
            Qv   = xp.dot(psh2o, Qv)
            
            return self.encode(Qv)

        Ndof = self.dof_rom
        Lop  = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec = lambda x: rom_v(x, Ef, xp=xp), dtype=xp.float64)

        Lop_mat = rom_utils.assemble_mat((self.dof_rom, self.dof_rom),Lop, xp=xp)
        return Lop_mat

    def rhs_rom_v(self, Xr, Ef, time, dt, type):
        bte     = self.bte_solver
        spec_sp = bte.op_spec_sp
        param   = bte.param
        xp      = bte.xp_module

        num_p   = spec_sp._p + 1
        num_sh  = len(spec_sp._sph_harm_lm)
        num_x   = len(bte.xp)

        rom_lm  = self.rom_modes
        Ps      = bte.op_po2sh
        Po      = bte.op_psh2o
        E       = xp.diag(Ef)

        if (type == "BE"):
            Xr  = Xr.reshape((-1))
            rhs = [None for i in range(rom_lm)]

            for i in range(rom_lm):
                Xri    = self.get_rom_lm(Xr, i)
                EVi    = xp.dot(E, self.Vx[i])
                #!!! Note: when there is Coulomb collisions, Collision operator is not block diagonal in the l-modes
                rhs[i] = Xri -dt * param.tau * (xp.dot(self.Cvr[i][i], Xri))
                for j in range(rom_lm):
                    Xrj     = self.get_rom_lm(Xr, j)
                    rhs[i] += -dt *param.tau * xp.dot(self.Avr[i][j], xp.dot(Xrj, xp.dot(self.Vx[j].T, EVi)))
            res = self.append_vec(rhs)

            # Xr0    = self.get_rom_lm(Xr, 0)
            # Xr1    = self.get_rom_lm(Xr, 1)
            # EV0    = xp.dot(E, self.Vx[0])
            # EV1    = xp.dot(E, self.Vx[1])

            # rhs[0] = Xr0 -dt * param.tau * (xp.dot(self.Cvr[0][0], Xr0)) - dt * param.tau * (
            #                                 xp.dot(self.Avr[0][0], xp.dot(Xr0, xp.dot(self.Vx[0].T, EV0))) +
            #                                 xp.dot(self.Avr[0][1], xp.dot(Xr1, xp.dot(self.Vx[1].T, EV0)))
            #                                 )

            # rhs[1] = Xr1 -dt * param.tau * (xp.dot(self.Cvr[1][1], Xr1)) - dt * param.tau * (
            #                                 xp.dot(self.Avr[1][0], xp.dot(Xr0, xp.dot(self.Vx[0].T, EV1))) +
            #                                 xp.dot(self.Avr[1][1], xp.dot(Xr1, xp.dot(self.Vx[1].T, EV1)))
            #                                 )

            # res = self.append_vec(rhs)
            return res

    def step_rom_v(self, Ef, Fr, time, dt, type, atol=1e-20, rtol=1e-8, gmres_rst=20, gmres_iter =10, verbose=1):

        if (self.rom_type == ROM_TYPE.DLR):
            return self.step_rom_v_dlr(Ef, Fr, time, dt, type, atol, rtol, gmres_rst, gmres_iter, verbose)

        bte        = self.bte_solver
        xp         = bte.xp_module

        rhs        = xp.copy(Fr).reshape((-1))
        norm_b     = xp.linalg.norm(rhs)
        Ndof       = rhs.shape[0]
        x0         = xp.copy(Fr.reshape((-1)))
        rom_lm     = self.rom_modes

        pc_idx     = self.vspace_pc_setup(Ef, time, dt)

        def Ax(x):
            return self.rhs_rom_v(x, Ef, time, dt, type)

        def Px(x):
            #return x
            ## note - following precond. makes the code slow.
            xp  = self.bte_solver.xp_module
            x0  = xp.dot(self.get_rom_lm(x, 0), self.Vx[0].T)
            x1  = xp.dot(self.get_rom_lm(x, 1), self.Vx[1].T)

            y0  = xp.empty_like(x0)
            y1  = xp.empty_like(x1)

            for idx in pc_idx:
                y0[:, idx[1]] = xp.dot(self.Pvr[idx[0]][0][0], x0[:, idx[1]]) + xp.dot(self.Pvr[idx[0]][0][1], x1[:, idx[1]])
                y1[:, idx[1]] = xp.dot(self.Pvr[idx[0]][1][0], x0[:, idx[1]]) + xp.dot(self.Pvr[idx[0]][1][1], x1[:, idx[1]])

                # y0[:, idx[1]] = xp.dot(self.Pvr[idx[0]][0][0], x0[:, idx[1]])
                # y1[:, idx[1]] = xp.dot(self.Pvr[idx[0]][1][1], x1[:, idx[1]])

            y                  = xp.zeros(self.dof_rom)

            y[self.vec_idx[0]] = xp.dot(y0, self.Vx[0]).reshape((-1))
            y[self.vec_idx[1]] = xp.dot(y1, self.Vx[1]).reshape((-1))

            return y

        if xp == cp:
            Amat_op       = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Ax)
            x0            = Px(x0)
            Pmat_op       = None#cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Px)
            gmres_c       = glow1d_utils.gmres_counter(disp=False)
            v, status     = cupyx.scipy.sparse.linalg.gmres(Amat_op, rhs.reshape((-1)), x0=x0.reshape((-1)), tol=rtol, atol=atol, M=Pmat_op, restart=gmres_rst, maxiter= gmres_rst * gmres_iter, callback=gmres_c)

            norm_res_abs  = xp.linalg.norm(Ax(v) -  rhs.reshape((-1)))
            norm_res_rel  = xp.linalg.norm(Ax(v) -  rhs.reshape((-1))) / norm_b

            if (status !=0) :
                print("time = %.8E T GMRES solver failed! iterations =%d  ||res|| = %.4E ||res||/||b|| = %.4E"%(time, status, norm_res_abs, norm_res_rel))
                sys.exit(-1)
            else:
                if (verbose == 1):
                    #print("[ROM v-space]  time = %.8E T GMRES  iterations =%d  ||res|| = %.4E ||res||/||b|| = %.4E"%(time, gmres_c.niter * gmres_rst, norm_res_abs, norm_res_rel))
                    print("ROM      Boltzmann (v-space) step time = %.6E ||res||=%.12E ||res||/||b||=%.12E gmres iter = %04d"%(time, norm_res_abs, norm_res_rel, gmres_c.niter * gmres_rst))

                return v.reshape(Fr.shape)
        else:
            raise NotImplementedError

    def step_rom_v_dlr(self, Ef, Fr, time, dt, type, atol=1e-20, rtol=1e-8, gmres_rst=20, gmres_iter =10, verbose=1):
        bte              = self.bte_solver
        xp               = bte.xp_module
        spec_sp          = bte.op_spec_sp
        param            = bte.param
        rom_lm           = self.rom_modes

        num_p            = spec_sp._p + 1
        num_sh           = len(spec_sp._sph_harm_lm)
        num_x            = len(bte.xp)

        ## K = UFr
        ## L = VFr^T
        rank_v           = np.array([self.Uv[l].shape[1] for l in range(rom_lm)])
        rank_x           = np.array([self.Vx[l].shape[1] for l in range(rom_lm)])

        #print(rank_x, rank_v, rank_v[0], rank_v[1])
        K_counts         = xp.array([self.Uv[l].shape[0] * self.Vx[l].shape[1] for l in range(rom_lm)])
        L_counts         = xp.array([self.Vx[l].shape[0] * self.Uv[l].shape[1] for l in range(rom_lm)])

        K_offset         = xp.zeros_like(K_counts)
        L_offset         = xp.zeros_like(L_counts)

        for l in range(1, rom_lm):
            K_offset[l] = K_offset[l-1] + K_counts[l-1]
            L_offset[l] = L_offset[l-1] + L_counts[l-1]

        # print(K_counts, K_offset)
        # print(L_counts, L_offset)
        K_mode           = lambda Kv, l : Kv[K_offset[l] : (K_offset[l] + K_counts[l])].reshape((num_p, Vx10[l].shape[1]))
        L_mode           = lambda Lv, l : Lv[L_offset[l] : (L_offset[l] + L_counts[l])].reshape((num_x, Uv10[l].shape[1]))

        Uv00, Vx00, Fr00 = self.Uv, self.Vx, Fr
        dF0              = xp.zeros_like(Fr00)

        def resF(dF, time, dt):
            F                = Fr00 + dF
            Fs               = xp.zeros((num_p * num_sh, num_x))

            for l in range(rom_lm):
                Fs[l::num_sh, :] = xp.dot(Uv00[l], xp.dot(self.get_rom_lm(F, l), Vx00[l].T))

            fom_rhs = param.tau * xp.dot(self.Cop, Fs) + param.tau * Ef * xp.dot(self.Av, Fs)
            rhs     = self.append_vec([xp.dot(Uv00[l].T, xp.dot(fom_rhs[l::num_sh], Vx00[l])) for l in range(rom_lm)])
            return (dF + dt * rhs).reshape((-1))

        def jacF(dF, time, dt):
            Fs               = xp.zeros((num_p * num_sh, num_x))

            for l in range(rom_lm):
                Fs[l::num_sh, :] = xp.dot(Uv00[l], xp.dot(self.get_rom_lm(dF, l), Vx00[l].T))

            fom_rhs = param.tau * xp.dot(self.Cop, Fs) + param.tau * Ef * xp.dot(self.Av, Fs)
            rhs     = self.append_vec([xp.dot(Uv00[l].T, xp.dot(fom_rhs[l::num_sh], Vx00[l])) for l in range(rom_lm)])
            return (dF + dt * rhs).reshape((-1))

        if xp == cp:
            Ndof      = Fr00.shape[0]
            RF_op     = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=lambda x: resF(x, time, dt))
            JF_op     = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=lambda x: jacF(x, time, dt))
            Fs        = glow1d_utils.newton_solver_matfree(dF0, RF_op, JF_op, lambda x: x, atol, rtol, atol, rtol, gmres_rst, gmres_iter * gmres_rst, xp)

            if (verbose == 1):
                print("[ROM basis - Fr solve] solve \
                    simulation time = %.4E | Newton iter = %04d |  GMRES iter = %04d | ||res|| = %.4E | ||res||/||res0|| = %.4E | alpha = %.4E"%(time, Fs["iter"], Fs["iter_gmres"], Fs["atol"], Fs["rtol"], Fs["alpha"]))

            assert Fs["status"] == True

        Fr01             = Fr00 + Fs["x"]
        Uv01, Vx01       = Uv00, Vx00 # basis does not change


        Uv10, Vx10, Fr10 = Uv01, Vx01, Fr01
        K10              = self.append_vec([xp.dot(Uv10[l], self.get_rom_lm(Fr10, l)) for l in range(rom_lm)])
        dK1              = xp.zeros_like(K10)

        def resK(dK, time, dt):
            K                = K10 + dK
            Fs               = xp.zeros((num_p * num_sh, num_x))

            for l in range(rom_lm):
                Fs[l::num_sh, :] = xp.dot(K_mode(K, l), Vx10[l].T)

            fom_rhs = param.tau * xp.dot(self.Cop, Fs) + param.tau * Ef * xp.dot(self.Av, Fs)
            rhs     = self.append_vec([xp.dot(fom_rhs[l::num_sh], Vx10[l]) for l in range(rom_lm)])
            return (dK - dt * rhs).reshape((-1))

        def jacK(dK, time, dt):
            Fs               = xp.zeros((num_p * num_sh, num_x))

            for l in range(rom_lm):
                Fs[l::num_sh, :] = xp.dot(K_mode(dK, l), Vx10[l].T)

            fom_rhs = param.tau * xp.dot(self.Cop, Fs) + param.tau * Ef * xp.dot(self.Av, Fs)
            rhs     = self.append_vec([xp.dot(fom_rhs[l::num_sh], Vx10[l]) for l in range(rom_lm)])
            return (dK - dt * rhs).reshape((-1))

        if xp == cp:
            Ndof      = K10.shape[0]
            RK_op     = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=lambda x: resK(x, time, dt))
            JK_op     = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=lambda x: jacK(x, time, dt))
            Ks        = glow1d_utils.newton_solver_matfree(dK1, RK_op, JK_op, lambda x: x, atol, rtol, atol, rtol, gmres_rst, gmres_iter * gmres_rst, xp)

            if (verbose == 1):
                print("[ROM basis - U solve] solve \
                    simulation time = %.4E | Newton iter = %04d |  GMRES iter = %04d | ||res|| = %.4E | ||res||/||res0|| = %.4E | alpha = %.4E"%(time, Ks["iter"], Ks["iter_gmres"], Ks["atol"], Ks["rtol"], Ks["alpha"]))

            assert Ks["status"] == True

        K11        = K10 + Ks["x"]

        # K11_svd    = [xp.linalg.svd(K_mode(K11, l)) for l in range(rom_lm)]
        # Uv11       = [K11_svd[l][0][:,0:rank_v[l]]  for l in range(rom_lm)]
        # Fr11       = self.append_vec([xp.dot(xp.diag(K11_svd[l][1][0:rank_v[l]]), K11_svd[l][2][0:rank_v[l], :])  for l in range(rom_lm)])

        K11_qr     = [xp.linalg.qr(K_mode(K11, l)) for l in range(rom_lm)]
        Uv11       = [K11_qr[l][0]  for l in range(rom_lm)]
        Fr11       = self.append_vec([K11_qr[l][1]  for l in range(rom_lm)])
        Vx11       = Vx10

        Fr20       = Fr11
        Uv20       = Uv11
        Vx20       = Vx11

        L20        = self.append_vec([xp.dot(Vx20[l], self.get_rom_lm(Fr20, l).T) for l in range(rom_lm)])
        dL2        = xp.zeros_like(L20)

        def resL(dL, time, dt):
            L   = L20 + dL
            Fs  = xp.zeros((num_p * num_sh, num_x))

            for l in range(rom_lm):
                Fs[l::num_sh, :] = xp.dot(Uv20[l], L_mode(L, l).T)

            fom_rhs = param.tau * xp.dot(self.Cop, Fs) + param.tau * Ef * xp.dot(self.Av, Fs)
            rhs     = self.append_vec([xp.dot(fom_rhs[l::num_sh].T, Uv20[l]) for l in range(rom_lm)])
            return (dL - dt * rhs).reshape((-1))

        def jacL(dL, time, dt):
            Fs  = xp.zeros((num_p * num_sh, num_x))

            for l in range(rom_lm):
                Fs[l::num_sh, :] = xp.dot(Uv20[l], L_mode(dL, l).T)

            fom_rhs = param.tau * xp.dot(self.Cop, Fs) + param.tau * Ef * xp.dot(self.Av, Fs)
            rhs     = self.append_vec([xp.dot(fom_rhs[l::num_sh].T, Uv20[l]) for l in range(rom_lm)])
            return (dL - dt * rhs).reshape((-1))


        if xp == cp:
            Ndof      = L20.shape[0]
            RL_op     = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=lambda x: resL(x, time, dt))
            JL_op     = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=lambda x: jacL(x, time, dt))
            Ls        = glow1d_utils.newton_solver_matfree(dL2, RL_op, JL_op, lambda x: x, atol, rtol, atol, rtol * 1e-1, gmres_rst, gmres_iter * gmres_rst, xp)

            if (verbose == 1):
                print("[ROM basis - V solve] solve \
                    simulation time = %.4E | Newton iter = %04d |  GMRES iter = %04d | ||res|| = %.4E | ||res||/||res0|| = %.4E | alpha = %.4E"%(time, Ks["iter"], Ls["iter_gmres"], Ls["atol"], Ls["rtol"], Ls["alpha"]))

            assert Ls["status"] == True


        L21        = L20 + Ls["x"]

        L21_qr     = [xp.linalg.qr(L_mode(L21, l)) for l in range(rom_lm)]
        Vx21       = [L21_qr[l][0]  for l in range(rom_lm)]
        Uv21       = Uv20
        Fr21       = self.append_vec([L21_qr[l][1].T  for l in range(rom_lm)])

        # L21_svd    = [xp.linalg.svd(L_mode(L21, l)) for l in range(rom_lm)]

        # Uv21       = Uv20
        # Vx21       = [L21_svd[l][0][:, 0:rank_x[l]]   for l in range(rom_lm)]
        # #Fr21       = self.append_vec([xp.dot(xp.diag(L21_svd[l][1]), L21_svd[l][2]).T for l in range(rom_lm)])
        # Fr21        = [None for l in range(rom_lm)]

        # for l in range(rom_lm):
        #     kr = rank_v[l]
        #     kx = rank_x[l]
        #     assert kx>=kr

        #     W  = xp.zeros((kx, kr))
        #     W[0:kr,0:kr] = xp.dot(xp.diag(L21_svd[l][1]), L21_svd[l][2])
        #     Fr21[l] = W.T

        # Fr21        = self.append_vec(Fr21)


        # for l in range(rom_lm):
        #     print(Uv21[l].shape, Vx21[l].shape, self.Uv[l].shape, self.Vx[l].shape)

        print("||I - U^t U|| = %.8E "%xp.linalg.norm(xp.dot(Uv21[0].T, Uv21[0]) - xp.eye(Uv21[0].shape[1])))
        print("||I - U^t U|| = %.8E "%xp.linalg.norm(xp.dot(Uv21[1].T, Uv21[1]) - xp.eye(Uv21[1].shape[1])))

        print("||I - V^t V|| = %.8E "%xp.linalg.norm(xp.dot(Vx21[0].T, Vx21[0]) - xp.eye(Vx21[0].shape[1])))
        print("||I - V^t V|| = %.8E "%xp.linalg.norm(xp.dot(Vx21[1].T, Vx21[1]) - xp.eye(Vx21[1].shape[1])))

        self.Uv=Uv21
        self.Vx=Vx21
        return Fr21

    def step_rom_x(self, Ef, Fr, time, dt, type, atol=1e-20, rtol=1e-8, verbose=1):
        bte        = self.bte_solver
        xp         = bte.xp_module
        Ps         = bte.op_po2sh
        Po         = bte.op_psh2o
        rom_lm     = self.rom_modes

        if (type == "FULL-STEP"):
            """
            performs advection in the full order-space
            """
            Fo = self.decode(Fr)
            Fo = bte.step_bte_x(Fo, time, dt * 0.5)
            Fo = bte.step_bte_x(Fo, time + 0.5 * dt, dt * 0.5)
            Fr = self.encode(Fo)
            return Fr.reshape((-1))

        elif(type == "HALF-STEP"):
            Fr  = self.encode(bte.step_bte_x(self.decode(Fr), time, dt * 0.5))
            # brhs                   = self.decode(Fr)
            # brhs[bte.xp_vt_l, 0]   = 0.0
            # brhs[bte.xp_vt_r, -1]  = 0.0
            # brhs                   = self.encode(brhs)
            # assert dt * 0.5 == bte.adv_setup_dt
            # Fr                     = xp.dot(self.Axr_inv, brhs)
            
            # # explicit enforce BC
            # Fo                     = self.decode(Fr)
            # Fo[bte.xp_vt_l, 0 ]    = 0.0 
            # Fo[bte.xp_vt_r, -1]    = 0.0 
            # Fr                     = self.encode(Fo)

            return Fr.reshape((-1))

        else:
            raise NotImplementedError

    def step_rom_op_split(self, Ef, Fr, time, dt, type, atol=1e-20, rtol=1e-10, verbose=1):
        bte            = self.bte_solver
        xp             = bte.xp_module
        gmres_rst      = 8
        gmres_iter     = 30

        #Frh            = self.step_rom_x(Ef, Fr, time, dt, "FOM-ADV-FULL-STEP", atol, rtol, verbose)

        #Frh           = self.step_rom_v(Et, Fr, time,            dt, "BE", atol, rtol, verbose)
        #Frh           = self.step_rom_v_dlr(Ef, Fr, time,            dt, "BE", atol, rtol, verbose)

        if (self.profile):
            cp.cuda.runtime.deviceSynchronize()
            self.timer[TIMER.ROM_X].reset()
            self.timer[TIMER.ROM_X].start()

        Frh            = self.step_rom_x(Ef,  Fr, time,            dt, "HALF-STEP", atol, rtol, verbose)
        #Frh = Fr

        if (self.profile):
            cp.cuda.runtime.deviceSynchronize()
            self.timer[TIMER.ROM_X].stop()

        if (self.profile):
            cp.cuda.runtime.deviceSynchronize()
            self.timer[TIMER.ROM_V].reset()
            self.timer[TIMER.ROM_V].start()

        #Frh            = self.step_rom_v(Ef, Frh, time,            dt, "BE", atol, rtol, gmres_rst, gmres_iter, verbose)
        Frh            = xp.dot(self.Lv_inv, Frh)
        #Frh             = self.encode(xp.dot(bte.op_psh2o, xp.dot(self.Lv_inv , xp.dot(bte.op_po2sh, self.decode(Frh)))))

        
        if (self.profile):
            cp.cuda.runtime.deviceSynchronize()
            self.timer[TIMER.ROM_V].stop()

        if (self.profile):
            cp.cuda.runtime.deviceSynchronize()
            self.timer[TIMER.ROM_X].start()

        Frh            = self.step_rom_x(Ef, Frh, time + 0.5 * dt, dt, "HALF-STEP", atol, rtol, verbose)

        if (self.profile):
            cp.cuda.runtime.deviceSynchronize()
            self.timer[TIMER.ROM_X].stop()

        if (self.profile):
            t1 = self.timer[TIMER.ROM_X].seconds
            t2 = self.timer[TIMER.ROM_V].seconds

            if (verbose==1):
                print("ROM x-solve runtime = %.4E v-solve runtime =%.4E total = %.4E"%(t1, t2, t1+t2))
                print("")
        return Frh

    def step_rom_vx(self, Ef, Fr, time, dt, atol=1e-20, rtol=1e-10, gmres_rst=20, gmres_iter =10, verbose=1):
        bte     = self.bte_solver
        xp      = bte.xp_module
        param   = bte.param

        if (self.profile):
            cp.cuda.runtime.deviceSynchronize()
            self.timer[TIMER.ROM_VX].reset()
            self.timer[TIMER.ROM_VX].start()

        spec_sp = bte.op_spec_sp
        num_p   = spec_sp._p + 1
        num_sh  = len(spec_sp._sph_harm_lm)
        num_x   = len(bte.xp)
        num_vt  = len(bte.xp_vt)
        xp_vt_l = bte.xp_vt_l
        xp_vt_r = bte.xp_vt_r

        # Fp              = self.decode(Fr)
        # #enforce BCs
        # Fp[xp_vt_l,  0] = 0.0
        # Fp[xp_vt_r, -1] = 0.0
        # Fp              = self.encode(Fp)
        # Fp              = xp.dot(self.Lvx_inv, Fp)

        Fp                = xp.dot(self.Lvx_inv, Fr)

        Fp                = self.decode(Fp)
        
        #enforce BCs
        Fp[xp_vt_l,  0]   = 0.0
        Fp[xp_vt_r, -1]   = 0.0
        Fp                = self.encode(Fp)

        if (self.profile):
            cp.cuda.runtime.deviceSynchronize()
            self.timer[TIMER.ROM_VX].stop()
            t1 = self.timer[TIMER.ROM_VX].seconds
            if(verbose==1):
                print("ROM vx-solve runtime = %.4E "%(t1))
                print("")
        return Fp

    def step_rom_eigen_solve_vx(self, Ef, Fr, time, dt, verbose=1):
        bte     = self.bte_solver
        xp      = bte.xp_module
        param   = bte.param

        if (time == 0):
            print("ic updated")
            self.Lvx_C = xp.dot(self.Lvx_Qinv, Fr)
            self.Lvx_C[np.real(self.Lvx_M)>0] = 0.0 
            print(self.Lvx_C)
        
        #print(time, (self.Lvx_C * xp.exp(self.Lvx_M * time)) [np.real(self.Lvx_M)>0])
        Yr = xp.dot(self.Lvx_Q, self.Lvx_C * xp.exp(self.Lvx_M * time))
        Yr = xp.real(Yr)
        Yr = self.decode(Yr)
        Yr[bte.xp_vt_l,  0] = 0.0
        Yr[bte.xp_vt_r, -1] = 0.0
        Yr = self.encode(Yr)
        return Yr

    def compute_history_terms(self, Ef, Lr, ts, dt, verbose=1):
        bte     = self.bte_solver
        xp      = bte.xp_module
        param   = bte.param
        Ir      = xp.eye(Lr.shape[0])

        assert Ir.shape[0] == self.dof_rom
        M       = xp.zeros((len(ts), self.dof_rom, self.dof_rom))
        Xr      = xp.zeros((len(ts), self.dof_rom, self.dof_rom))

        for ridx in range(self.dof_rom):
            Yr                  = self.decode(Ir[:, ridx])
            # Yr[bte.xp_vt_l, 0 ] = 0.0
            # Yr[bte.xp_vt_r, -1] = 0.0
            Xr[0, :, ridx]      = self.encode(Yr)
        
        a1 = xp.linalg.norm(Xr[0] - Ir)/xp.linalg.norm(Ir)
        if (verbose == 1):
            print("||Xr[0] - Ir||/||Ir|| = %.8E"%(a1))


        ImLr    = (xp.eye(Lr.shape[0]) - dt * Lr)
        Fmr     = xp.linalg.inv(ImLr)
        dtsqrd  = 1/dt**2

        MXr     = 0
        tt      = 0

        for ridx in range(self.dof_rom):
            F   = self.decode(Ir[:, ridx])
            tt  = 0 
            #print(ridx)
            for tidx in range(1, len(ts)):
                F                 = self.step_fom_op_split(Ef, F, tt, dt, verbose=0)
                Xr[tidx, :, ridx] = self.encode(F)
                tt               += dt

        mem_norm = xp.zeros(len(ts))
        for tidx in range(1, len(ts)):
            
            # for ridx in range(self.dof_rom):
            #     Yr                  = self.decode(Xr[tidx-1, :, ridx])
            #     Xr[tidx, :, ridx]   = self.encode(self.step_fom_op_split(Ef, Yr, tt, dt, verbose=0))
            
            W1        = xp.linalg.inv(Xr[tidx-1])
            M[tidx-1] = (dtsqrd * ( (ImLr @ Xr[tidx]) - Xr[tidx-1] ) - MXr ) @ W1
            MXr      += M[tidx-1] @ Xr[tidx-1]

            y1 = Xr[tidx]
            y2 = Fmr @ (Xr[tidx-1] + (dt **2) * (MXr))

            k1 = xp.linalg.norm(dt**2 * (M[tidx-1] @ Xr[tidx-1]))
            if (verbose == 1):
                l1 = Fmr @ Xr[tidx-1]
                l2 = Fmr @ ((dt **2) * (MXr))
                mem_norm[tidx] = xp.linalg.norm(l2)/xp.linalg.norm(y1)
                print("time = %.4E relative norm of the memory term = %.8E"%(ts[tidx], mem_norm[tidx]))
                #print("||Mr[%04d dt] @ Xr[%04d]||    = %.8E ||Mr[%04d dt] @ Xr[%04d]||/||Xr[%04d]|| = %.8E"%(tidx-1, tidx-1, k1, tidx-1, tidx-1, tidx-1, k1/xp.linalg.norm(y1)))
                # print("||Ir - Xr^{-1} Xr || / ||Ir|| = %.8E"%(xp.linalg.norm(Ir-W1 @ Xr[tidx-1]) / xp.linalg.norm(Ir)))
                # print("||Ir - Xr Xr^{-1} || / ||Ir|| = %.8E"%(xp.linalg.norm(Ir-Xr[tidx-1]@ W1 ) / xp.linalg.norm(Ir)))
                print("||y2-y1||/||y1||=%.8E"%(xp.linalg.norm(y2[:, 10] - y1[:, 10])/xp.linalg.norm(y1[:, 10])))

            #print(tidx-1, xp.linalg.norm(M[tidx-1] @ Xr[tidx-1, 10]), xp.linalg.norm(Xr[tidx-1, 10]), xp.linalg.norm(Fmr), xp.linalg.norm(Xr[tidx-1, 10] + (dt **2) * (MXr[:, 10])))
            print(tidx-1, Xr[tidx-1, :, 10], y2[:, 10], y1[:, 10])
            tt  +=dt

            # if(tidx == 1):
            #     break

        if verbose==1:
            plt.figure(figsize=(4, 4), dpi=200)
            plt.xlabel(r"time")
            plt.ylabel(r"memory term norm")
            
            plt.semilogy(xp.asnumpy(ts), xp.asnumpy(mem_norm))
            plt.grid(visible=True)
            
            plt.tight_layout()
            plt.savefig("%s_mem_term_learning.png"%(self.bte_solver.args.fname))

        return M

    #############################################################################################
    ############################## FOM routines below ###########################################
    #############################################################################################

    def fom_mv(self, Ef, x):
        bte     = self.bte_solver
        xp      = bte.xp_module
        param   = bte.param

        num_p   = self.num_p
        num_vt  = self.num_vt
        num_sh  = self.num_sh
        num_x   = self.num_x
        
        Ps      = bte.op_po2sh
        Po      = bte.op_psh2o
        DxT     = self.Dx.T
        
        Ndof    = num_p * num_vt * num_x

        x       = x.reshape((num_p * num_vt, num_x))
        xs      = xp.dot(Ps, x) 
        y       = param.tau * xp.dot(Po, xp.dot(self.Cop, xs) + Ef * xp.dot(self.Av, xs)) - xp.dot(self.Ax, xp.dot(x, DxT))

        # bc 
        y[bte.xp_vt_l, 0 ] = 0.0
        y[bte.xp_vt_r, -1] = 0.0

        return y.reshape((-1))

    def fom_adj_mv(self, Ef, x):
        
        bte     = self.bte_solver
        xp      = bte.xp_module
        param   = bte.param

        num_p   = self.num_p
        num_vt  = self.num_vt
        num_sh  = self.num_sh
        num_x   = self.num_x

        PsT     = bte.op_po2sh.T
        PoT     = bte.op_psh2o.T
        CopT    = self.Cop.T
        AvT     = self.Av.T
        AxT     = self.Ax.T
        
        x                  = xp.copy(x) 
        x                  = x.reshape((num_p * num_vt, num_x))
        x[bte.xp_vt_l, 0 ] = 0.0
        x[bte.xp_vt_r, -1] = 0.0

        xs = xp.dot(PoT, x)
        y  = param.tau * np.dot(PsT, xp.dot(CopT, xs) + Ef * xp.dot(AvT, xs)) - xp.dot(np.dot(AxT, x) , self.Dx)
        return y.reshape((-1))

    def step_fom_op_split(self, Ef, F, time, dt, prefactored_vsolve=False, verbose=1):
        """
        full order model timestep
        """
        bte                 = self.bte_solver
        tt                  = time
        bte.bs_E            = Ef
        xp                  = bte.xp_module

        #v                  = bte.step_bte_v(F, None, tt, dt, ts_type="BE", verbose=1)

        # v                 = bte.step_bte_x(F, tt, dt * 0.5)
        # v                 = bte.step_bte_x(v, tt + 0.5 * dt, dt * 0.5)

        if (self.profile):
            cp.cuda.runtime.deviceSynchronize()
            self.timer[TIMER.FOM_X].reset()
            self.timer[TIMER.FOM_X].start()

        v                   = bte.step_bte_x(F, tt, dt * 0.5)
        #v=F
        if (self.profile):
            cp.cuda.runtime.deviceSynchronize()
            self.timer[TIMER.FOM_X].stop()

        if (self.profile):
            cp.cuda.runtime.deviceSynchronize()
            self.timer[TIMER.FOM_V].reset()
            self.timer[TIMER.FOM_V].start()

        if (prefactored_vsolve==True):
            v                    = xp.dot(self.Lv_inv, v)
        else:
            v                    = bte.step_bte_v(v, None, tt, dt, ts_type="BE", verbose=verbose)

        if (self.profile):
            cp.cuda.runtime.deviceSynchronize()
            self.timer[TIMER.FOM_V].stop()

        if (self.profile):
            cp.cuda.runtime.deviceSynchronize()
            self.timer[TIMER.FOM_X].start()

        v                   = bte.step_bte_x(v, tt + 0.5 * dt, dt * 0.5)

        if (self.profile):
            cp.cuda.runtime.deviceSynchronize()
            self.timer[TIMER.FOM_X].stop()

        if (self.profile):
            t1 = self.timer[TIMER.FOM_X].seconds
            t2 = self.timer[TIMER.FOM_V].seconds

            if (verbose==1):
                print("FOM x-solve runtime = %.4E v-solve runtime =%.4E total = %.4E"%(t1, t2, t1+t2))
                print("")

        return v

    def step_fom_vx(self, Ef, F, time, dt, atol=1e-20, rtol=1e-8, gmres_rst=20, gmres_iter =10, pc_type=0, verbose=1):
        bte     = self.bte_solver
        xp      = bte.xp_module
        param   = bte.param

        spec_sp = bte.op_spec_sp
        num_p   = spec_sp._p + 1
        num_sh  = len(spec_sp._sph_harm_lm)
        num_x   = len(bte.xp)
        num_vt  = len(bte.xp_vt)

        DxT     = self.Dx.T
        po2sh   = bte.op_po2sh
        psh2o   = bte.op_psh2o
        xp_vt_l = bte.xp_vt_l
        xp_vt_r = bte.xp_vt_r

        def rhs_op(F, time, dt):
            F                 = F.reshape((num_p * num_vt, num_x))
            Fs                = xp.dot(po2sh, F)
            rhs               = (-xp.dot(self.Ax, xp.dot(F, DxT)) +  param.tau * xp.dot(psh2o, (xp.dot(self.Cop, Fs) + Ef * xp.dot(self.Av, Fs))))

            rhs[xp_vt_l,  0 ] = 0.0
            rhs[xp_vt_r, -1 ] = 0.0

            return rhs.reshape((-1))

        def precond(F, time, dt):
            if (pc_type == 1):
                brhs              = xp.copy(F).reshape((num_p * num_vt, num_x))
                brhs[xp_vt_l,  0] = 0.0
                brhs[xp_vt_r, -1] = 0.0
                
                return self.decode(xp.dot(self.Lvx_inv, self.encode(brhs)))
            else:
                return F


        if xp == cp:
            Ndof          = num_p * num_x * num_vt
            Amat_op       = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec = lambda x: x - dt * rhs_op(x, time, dt))

            brhs          = xp.copy(F)
            #enforce BCs
            brhs[xp_vt_l,  0] = 0.0
            brhs[xp_vt_r, -1] = 0.0
            norm_b        = xp.linalg.norm(brhs)

            F_init        = precond(F, time, dt)
            Pmat_op       = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec= lambda x: precond(x, time, dt))
            gmres_c       = glow1d_utils.gmres_counter(disp=True)
            Fp, status    = cupyx.scipy.sparse.linalg.gmres(Amat_op, brhs.reshape((-1)), x0=F_init.reshape((-1)), tol=rtol, atol=atol, M=Pmat_op, restart=gmres_rst, maxiter= gmres_rst * gmres_iter, callback=gmres_c)

            norm_res_abs  = xp.linalg.norm(Amat_op(Fp) -  brhs.reshape((-1)))
            norm_res_rel  = xp.linalg.norm(Amat_op(Fp) -  brhs.reshape((-1))) / norm_b

            if (status !=0) :
                print("time = %.8E T GMRES solver failed! iterations =%d  ||res|| = %.4E ||res||/||b|| = %.4E"%(time, status, norm_res_abs, norm_res_rel))
                #sys.exit(-1)
                Fp = Fp.reshape(F.shape)
                return Fp
            else:
                if (verbose == 1):
                    print("FOM      Boltzmann (vx-space) step time = %.6E ||res||=%.12E ||res||/||b||=%.12E gmres iter = %04d"%(time, norm_res_abs, norm_res_rel, gmres_c.niter * gmres_rst))

                Fp = Fp.reshape(F.shape)
                # print("l\n", Fp[xp_vt_l, 0])
                # print("r\n", Fp[xp_vt_r,-1])
                return Fp
        else:
            raise NotImplementedError

    def flow_map_fom_op_v(self, Ef, dt):
        bte     = self.bte_solver
        param   = bte.param
        xp      = bte.xp_module

        Ps      = bte.op_po2sh
        Po      = bte.op_psh2o

        assert (Ef[0] == Ef).all() == True

        #Lmat    = xp.eye(self.num_p * self.num_vt) - dt * param.tau * xp.dot(Po, xp.dot(self.Cop, Ps) + xp.dot(Ef[0] * self.Av, Ps))
        Lmat     = xp.eye(self.num_p * self.num_sh) - dt * param.tau * (self.Cop + Ef[0] * self.Av)
        Lmat     = xp.linalg.inv(Lmat)
        
        self.Lv_inv = Po @ Lmat @ Ps
        return self.Lv_inv

    def assemble_fom_op(self, Ef):
        bte     = self.bte_solver
        param   = bte.param
        xp      = bte.xp_module

        num_p   = self.num_p
        num_vt  = self.num_vt 
        num_x   = self.num_x 

        # Cop     = xp.asnumpy(self.Cop)
        # Av      = xp.asnumpy(self.Av)
        # Ax      = xp.asnumpy(self.Ax)
        # Dx      = xp.asnumpy(self.Dx)

        # Ps      = xp.asnumpy(bte.op_po2sh)
        # Po      = xp.asnumpy(bte.op_psh2o)
        # Ef      = xp.asnumpy(Ef)

        # Ndof    = self.num_p * self.num_vt * self.num_x
        # L1      = np.zeros((Ndof, Ndof))
        # L2      = np.zeros((Ndof, Ndof))

        # print("op assembly begin")

        # for i in range(num_p * num_vt):
        #     L1[i * num_x : (i+1) * num_x, i * num_x : (i+1) * num_x] = Dx

        # Cop_o = Po @ Cop @ Ps
        # Av_o  = Po @ Av @ Ps

        # Lv    = np.zeros((Ndof, Ndof))

        # for i in range(num_p * num_vt):
        #     for j in range(num_p * num_vt):
        #         for k in range(num_x):
        #             L2 [ i * num_x + k, j * num_x +  k ] = Ax[i, j]
        #             Lv [ i * num_x + k, j * num_x +  k]  = param.tau * (Cop_o[i,j] + Ef[k] * Av_o[i,j])

        Cop   = self.Cop
        Av    = self.Av
        Ax    = self.Ax
        Dx    = self.Dx

        Ps    = bte.op_po2sh
        Po    = bte.op_psh2o
        Ef    = Ef

        Ndof  = self.num_p * self.num_vt * self.num_x

        Cop_o = Po @ Cop @ Ps
        Av_o  = Po @ Av @ Ps

        L1  = xp.kron(xp.eye(num_p * num_vt), Dx)
        L2  = xp.kron(Ax, xp.eye(num_x))

        Lv  = param.tau * (xp.kron(Cop_o, xp.eye(num_x)) + xp.kron(Av_o, xp.diag(Ef))) 

        Lx  = -np.dot(L2, L1)
        L   = Lv + Lx

        L[bte.xp_vt_l * num_x + 0, :]        = 0.0
        L[bte.xp_vt_r * num_x + num_x-1, : ] = 0.0

        L   = xp.asarray(L) 
        Ef  = xp.asarray(Ef)
        x   = xp.random.rand(Ndof)
        Lx  = xp.dot(L, x)
        LTx = xp.dot(L.T, x) 

        y1  = self.fom_mv(Ef, x)
        y2  = self.fom_adj_mv(Ef, x)

        a1  = xp.linalg.norm(Lx  - y1) / xp.linalg.norm(y1)
        a2  = xp.linalg.norm(LTx - y2) / xp.linalg.norm(y2)

        assert a1 < 1e-14
        assert a2 < 1e-14

        print("op assembly end")

        return L 
    
    ########################### FOM OP END #####################################################

    def save_checkpoint(self, F, Fr, time, fname):
        bte = self.bte_solver
        xp  = bte.xp_module

        assert xp == cp

        with h5py.File("%s"%(fname), "w") as ff:
            ff.create_dataset("time[T]"      , data = np.array([time]))
            ff.create_dataset("rom_lm"       , data = np.array([self.rom_modes]))
            ff.create_dataset("F"            , data = xp.asnumpy(F))
            for l in range(self.rom_modes):
                ff.create_dataset("Fr_%02d"%(l), data = xp.asnumpy(self.get_rom_lm(Fr, l)))
                ff.create_dataset("Uv_%02d"%(l), data = xp.asnumpy(self.Uv[l]))
                ff.create_dataset("Vx_%02d"%(l), data = xp.asnumpy(self.Vx[l]))

            ff.close()
        return

    def restore_checkpoint(self, fname):
        bte = self.bte_solver
        xp  = bte.xp_module

        assert xp == cp

        with h5py.File("%s"%(fname), "r") as ff:
            time           = np.array(ff["time[T]"][()])[0]
            self.rom_modes = np.array(ff["rom_lm"][()])[0]
            F              = xp.array(ff["F"][()])

            Fr             = self.append_vec([xp.array(ff["Fr_%02d"%(l)][()]) for l in range(self.rom_modes)])
            self.Uv        = [xp.array(ff["Uv_%02d"%(l)][()]) for l in range(self.rom_modes)]
            self.Vx        = [xp.array(ff["Vx_%02d"%(l)][()]) for l in range(self.rom_modes)]

            ff.close()

        self.init()
        return F, Fr, time

    def vspace_pc_setup(self, Ef, time, dt):
        bte         = self.bte_solver
        xp          = bte.xp_module

        E           = Ef
        pcEmat      = bte.PmatE
        pcEval      = bte.Evals.reshape((-1,1))
        E           = E.reshape((-1, 1))

        dist        = xp.linalg.norm(E[:, None, :] - pcEval[None, :, :], axis=2)
        c_memship   = xp.argmin(dist, axis=1)
        c_idx       = xp.arange(pcEval.shape[0])
        p_idx       = xp.arange(len(E))
        mask        = c_memship == c_idx[:, None]

        pc_emat_idx = list()
        for i in range(len(pcEval)):
            pc_emat_idx.append((i, p_idx[mask[i, :]]))

        # idx_set  = xp.array([],dtype=xp.int32)
        # for idx_id, idx in enumerate(pc_emat_idx):
        #   idx_set = xp.append(idx_set, idx[1])
        # assert (idx_set.shape[0]==bte.Np), "!!! Error: preconditioner partitioning does not match the domain size"

        return pc_emat_idx

def parse_rom_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-eps_v"        , "--eps_v"         , help="rom v-space svd truncation"    , type=float, default=1e-8)
    parser.add_argument("-eps_x"        , "--eps_x"         , help="rom x-space svd truncation"    , type=float, default=1e-4)
    parser.add_argument("-train_cycles" , "--train_cycles"  , help="number of train cycles"        , type=float, default=1.0 )
    parser.add_argument("-num_samples"  , "--num_samples"   , help="number of solution snapshots"  , type=int  , default=31  )
    parser.add_argument("-rom_step_freq", "--rom_step_freq" , help="rom step frequency"            , type=int  , default=5   )
    parser.add_argument("-par_file"     , "--par_file"      , help="redundant"                     , type=str  , default=""  )

    args_rom  = parser.parse_args()

    assert args.par_file == args_rom.par_file 

    import toml
    tp  = toml.load(args.par_file)
    
    tp0                     = tp["bte-rom"] 
    args_rom.eps_v          = tp0["eps_v"]
    args_rom.eps_x          = tp0["eps_x"]
    args_rom.train_cycles   = tp0["train_cycles"]
    args_rom.num_samples    = tp0["num_samples"]
    args_rom.rom_step_freq  = tp0["rom_step_freq"]

    with open("%s_args.txt"%(args.fname), "a") as ff:
        ff.write("\nrom-args: %s"%(args_rom))
        ff.close()

    return args_rom

def static_efield_driver(E0=1e3):
    from glowdischarge_boltzmann_1d import glow1d_boltzmann, args_parse
    args         = args_parse()

    bte_fom      = glow1d_boltzmann(args)
    u, v         = bte_fom.initialize()
    args         = bte_fom.args

    args_rom     = parse_rom_args(args)


    dt           = args.cfl
    io_cycle     = args.io_cycle_freq
    cp_cycle     = args.cp_cycle_freq

    io_freq      = int(np.round(io_cycle/dt))
    cp_freq      = int(np.round(cp_cycle/dt))
    cycle_freq   = int(np.round(1/dt))
    tio_freq     = 20
    uv_freq      = int(np.round(10/dt))

    rom_eps_x    = args_rom.eps_x
    rom_eps_v    = args_rom.eps_v
    restore      = 0
    rs_idx       = 0
    train_cycles = args_rom.train_cycles
    num_samples  = args_rom.num_samples
    num_history  = 40
    is_rom       = 1
    rom_steps    = args_rom.rom_step_freq
    
    if args.use_gpu==1:
        gpu_device = cp.cuda.Device(args.gpu_device_id)
        gpu_device.use()

    u, v    = bte_fom.step_init(u, v, dt)
    xp      = bte_fom.xp_module

    spec_sp = bte_fom.op_spec_sp
    num_p   = spec_sp._p + 1
    num_sh  = len(spec_sp._sph_harm_lm)

    bte_rom = boltzmann_1d_rom(bte_fom)
    bte_rom.init()
    xxg     = xp.asarray(bte_fom.xp)
    Et      = lambda t: xp.ones_like(bte_fom.xp) * E0 

    Po      = bte_rom.bte_solver.op_psh2o
    Ps      = bte_rom.bte_solver.op_po2sh
    
    ## construct vspace direct solve (I-dtLv)^{-1}
    bte_rom.flow_map_fom_op_v(Et(0), dt)

    recompute   = False
    ## 1. Sample from a fixed initial conditon
    #v_sp = bte_rom.sample_fom(Et, xp.copy(v), 0, 100, dt, num_samples, load_from_file=True)
    #bte_rom.init_rom_basis_from_snapshots(v_sp[:, 0::3], (rom_eps_v, rom_eps_x))

    ## 2. Sample from a series of initial conditions
    v0_ic       = bte_rom.generate_initial_data(np.logspace(-10, 0, 10, base=10), np.linspace(2, 4, 4))
    
    Tend        = 10
    num_samples = 11
    v_sp   = xp.zeros((v0_ic.shape[0], v0_ic.shape[1], num_samples))
    for k in range(v0_ic.shape[0]):
        v_sp[k] = bte_rom.sample_fom(Et, v0_ic[k], 0, Tend, dt, num_samples, fprefix="%03d"%(k), load_from_file=(not recompute))
    
    v_sp  = v_sp[0::4, :, :]
    v_sp  = xp.swapaxes(v_sp, 0, 1).reshape((-1, num_samples * v_sp.shape[0]))

    bte_rom.init_rom_basis_from_snapshots(v_sp[:, 0::1], (rom_eps_v, rom_eps_x))
    Lop_r = bte_rom.assemble_rom_op(Et(0))
    
    ### Eigen solve

    # Lop   = bte_rom.assemble_fom_op(Et(0))

    # Pe    = bte_rom.assemble_encode_op()
    # Pd    = bte_rom.assemble_decode_op()

    # Ir    = xp.eye(Lop_r.shape[0])
    # I     = xp.eye(Lop.shape[0])

    # ImLr  = Ir - dt * Lop_r
    # ImL   = I  - dt * Lop

    # Qr    = xp.linalg.inv(ImLr)
    # Q     = xp.linalg.inv(ImL)

    # print("a1 = %.8E"%(xp.linalg.norm(Ir - Qr @ ImLr) / xp.linalg.norm(Ir)))
    # print("a2 = %.8E"%(xp.linalg.norm(I  - Q  @ ImL)  / xp.linalg.norm(I)))

    # rom_utils.eigen_plot([xp.asnumpy(Lop_r), xp.asnumpy(Lop)], 
    #                      labels = [r"$L_r$", r"$L$"], 
    #                      fname  = "%s_eig_plot_L_%d%d_%d.png"%(args.fname, np.log10(1/rom_eps_v), np.log10(1/rom_eps_x), bte_rom.dof_rom))

    # rom_utils.eigen_plot([xp.asnumpy(Qr), xp.asnumpy(Q)], 
    #                       labels = [r"$(I_r-dt L_r)^{-1}$", r"$(I-dt L)^{-1}$"], 
    #                       fname  = "%s_eig_plot_ImL_%d%d_%d.png"%(args.fname, np.log10(1/rom_eps_v), np.log10(1/rom_eps_x), bte_rom.dof_rom))


    # v    = v_sp[:, 0].reshape((bte_rom.num_p * bte_rom.num_vt , bte_rom.num_x))
    # v[bte_fom.xp_vt_l,  0] = 0.0
    # v[bte_fom.xp_vt_r, -1] = 0.0


    # sys.exit(0)

    ## rom flow map
    Lop_r = xp.linalg.inv(xp.eye(Lop_r.shape[0]) - dt * Lop_r)

    v    = v_sp[:, -1].reshape((bte_rom.num_p * bte_rom.num_vt , bte_rom.num_x))
    v[bte_fom.xp_vt_l,  0] = 0.0
    v[bte_fom.xp_vt_r, -1] = 0.0

    tt        = 0
    if (restore==1):
        F, Fr, tt = bte_rom.restore_checkpoint("%s_rom_%02d.h5"%(bte_fom.args.fname, rs_idx))
        F         = bte_rom.decode(Fr)
        idx       = (rs_idx) * cp_freq
        print("checkpoint restored time = %.4E (s) idx = %d"%(tt, idx))
    else:
        F       = xp.copy(v)
        Fo      = xp.copy(v)
        Fs      = xp.dot(bte_fom.op_po2sh, Fo)
        Fr      = bte_rom.encode(Fo).reshape((-1))
        F1      = bte_rom.decode(Fr)
        idx     = 0
        tt      = 0
    

    print("BTE fom advection step size = %.4E"%(bte_fom.adv_setup_dt))
    print("io freq = %d cycle_freq = %d cp_freq = %d uv_freq = %d" %(io_freq, cycle_freq, cp_freq, uv_freq))
    tT        = args.cycles
    
    F0  = xp.copy(F)
    Fr0 = xp.copy(Fr)
    Fr2 = xp.copy(Fr)

    while tt < tT:
        Ef     = Et(tt)

        if (idx > 0 and idx % cycle_freq== 0):
            ne_r0 = xp.dot(bte_fom.op_mass, xp.dot(bte_fom.op_po2sh, bte_rom.decode(Fr0)))
            ne_r  = xp.dot(bte_fom.op_mass, xp.dot(bte_fom.op_po2sh, bte_rom.decode(Fr)))

            ne_0  = xp.dot(bte_fom.op_mass, xp.dot(bte_fom.op_po2sh, F0))
            ne    = xp.dot(bte_fom.op_mass, xp.dot(bte_fom.op_po2sh, F))

            a1    = xp.linalg.norm((ne_r0 - ne_r) / xp.max(ne_r))
            a2    = xp.linalg.norm((ne_0  - ne)   / xp.max(ne))

            print("time = %.4E [T] ROM : ||ne(T) - ne(0)||/||ne(0)|| = %.4E FOM : ||ne(T) - ne(0)||/||ne(0)|| = %.4E"%(tt, a1, a2))

            F0  = xp.copy(F)
            Fr0 = xp.copy(Fr)

        if (idx % cp_freq == 0):
            print("checkpoint time = %.4E (s)"%(tt))
            bte_rom.save_checkpoint(F, Fr, tt, "%s_rom_%02d.h5"%(bte_fom.args.fname, (idx//cp_freq)))

        if (idx % io_freq == 0):
            print("io output time = %.4E (s)"%(tt))
            F1 = bte_rom.decode(Fr)
            plot_solution(bte_fom, bte_rom, F, F1, fname="%s_rom_%04d.png"%(args.fname, idx//io_freq), time = tt)

        if (idx % rom_steps == 0):
            is_rom   = 0
            
        if(is_rom == 1):
            Fr                                  = bte_rom.decode(Fr)
            Fr[bte_rom.bte_solver.xp_vt_l,   0] = 0.0
            Fr[bte_rom.bte_solver.xp_vt_r,  -1] = 0.0
            Fr                                  = xp.dot(Lop_r, bte_rom.encode(Fr))

        else:
            Fr                                  = bte_rom.decode(Fr)
            Fr                                  = bte_rom.encode(bte_rom.step_fom_op_split(Ef, Fr, tt, dt, verbose = 0))
            is_rom                              = 1
            
        
        
        F       = bte_rom.step_fom_op_split(Ef, F, tt, dt, verbose = 0, prefactored_vsolve=True)

        tt  += dt
        idx +=1

def efield_driver_time_harmonic(E0=1e3):
    from glowdischarge_boltzmann_1d import glow1d_boltzmann, args_parse
    args         = args_parse()

    bte_fom      = glow1d_boltzmann(args)
    u, v         = bte_fom.initialize()
    args         = bte_fom.args

    args_rom     = parse_rom_args(args)


    dt           = args.cfl
    io_cycle     = args.io_cycle_freq
    cp_cycle     = args.cp_cycle_freq

    io_freq      = int(np.round(io_cycle/dt))
    cp_freq      = int(np.round(cp_cycle/dt))
    cycle_freq   = int(np.round(1/dt))
    tio_freq     = 20
    uv_freq      = int(np.round(10/dt))

    rom_eps_x    = args_rom.eps_x
    rom_eps_v    = args_rom.eps_v
    restore      = 0
    rs_idx       = 0
    train_cycles = args_rom.train_cycles
    num_samples  = args_rom.num_samples
    num_history  = 40
    is_rom       = 1
    rom_steps    = args_rom.rom_step_freq
    rom_use_gmres= True
    
    if args.use_gpu==1:
        gpu_device = cp.cuda.Device(args.gpu_device_id)
        gpu_device.use()

    u, v    = bte_fom.step_init(u, v, dt)
    xp      = bte_fom.xp_module

    spec_sp = bte_fom.op_spec_sp
    num_p   = spec_sp._p + 1
    num_sh  = len(spec_sp._sph_harm_lm)

    bte_rom = boltzmann_1d_rom(bte_fom)
    bte_rom.init()
    xxg     = xp.asarray(bte_fom.xp)
    Ex      = xp.ones_like(bte_fom.xp) * E0 
    Et      = lambda t: Ex * xp.sin(2 * np.pi * t)

    Po      = bte_rom.bte_solver.op_psh2o
    Ps      = bte_rom.bte_solver.op_po2sh
    

    # # 1. Sample from a fixed initial conditon
    # recompute   = False
    # v_sp = bte_rom.sample_fom(lambda t : Ex , xp.copy(v), 0, 1, dt, num_samples, load_from_file=(not recompute))
    # bte_rom.init_rom_basis_from_snapshots(v_sp[:, 0::1], (rom_eps_v, rom_eps_x))

    # #2. Sample from a series of initial conditions
    recompute   = False
    nsTe        = 2
    nsne        = 10
    v0_ic       = bte_rom.generate_initial_data(np.logspace(-5, 0, nsne, base=10), np.linspace(2, 4, nsTe))
    
    Tend        = 1
    num_samples = 11
    v_sp        = xp.zeros((v0_ic.shape[0], bte_rom.num_p * bte_rom.num_sh * bte_rom.num_x, num_samples))
    for k in range(v0_ic.shape[0]):
        v_sp[k] = bte_rom.sample_fom(lambda t: Ex, v0_ic[k], 0, Tend, dt, num_samples, fprefix="%03d"%(k), load_from_file=(not recompute))
        
    
    # v_sp  = v_sp[0::1, :, -1]
    # v_sp  = xp.swapaxes(v_sp, 0, 1).reshape((-1,  v_sp.shape[0]))

    v_sp  = v_sp[0::1, :, 0::1]
    v_sp  = xp.swapaxes(v_sp, 0, 1).reshape((-1,  (v_sp.shape[2]) * v_sp.shape[0]))

    bte_rom.init_rom_basis_from_snapshots(v_sp[:, :], (rom_eps_v, rom_eps_x))
    
    Ls_r = bte_rom.assemble_rom_op(Et(0) * 0.0)
    # Eop  = bte_rom.assemble_encode_op()
    # Dop  = bte_rom.assemble_decode_op()

    num_p  = bte_rom.num_p
    num_vt = bte_rom.num_vt
    num_x  = bte_rom.num_x 
    
    def rom_vx(Xr, Ex, xp=xp):
        Fo   = bte_rom.decode(Xr)
        Fs   = xp.dot(Ps, Fo)
        Qv   = bte_rom.bte_solver.param.tau * (Ex * xp.dot(bte_rom.Av, Fs))
        Qv   = xp.dot(Po, Qv)
        Qv   = bte_rom.encode(Qv).reshape((-1))
        return Qv

    dof_fom = bte_rom.num_p * bte_rom.num_vt * bte_rom.num_x
    dof_rom = bte_rom.dof_rom
    Lop     = cupyx.scipy.sparse.linalg.LinearOperator((dof_rom, dof_rom), matvec = lambda x: rom_vx(x, Ex, xp=xp), dtype=xp.float64)
    Lt_r    = rom_utils.assemble_mat((dof_rom, dof_rom), Lop, xp=xp)

    v       = v_sp[:, 0].reshape((bte_rom.num_p * bte_rom.num_sh , bte_rom.num_x))
    v       = Po @ v
    # v[bte_fom.xp_vt_l,  0] = 0.0
    # v[bte_fom.xp_vt_r, -1] = 0.0

    tt        = 0
    if (restore==1):
        F, Fr, tt = bte_rom.restore_checkpoint("%s_rom_%02d.h5"%(bte_fom.args.fname, rs_idx))
        F         = bte_rom.decode(Fr)
        idx       = (rs_idx) * cp_freq
        print("checkpoint restored time = %.4E (s) idx = %d"%(tt, idx))
    else:
        F       = xp.copy(v)
        Fo      = xp.copy(v)
        Fs      = xp.dot(bte_fom.op_po2sh, Fo)
        Fr      = bte_rom.encode(Fo).reshape((-1))
        F1      = bte_rom.decode(Fr)
        idx     = 0
        tt      = 0
    

    print("BTE fom advection step size = %.4E"%(bte_fom.adv_setup_dt))
    print("io freq = %d cycle_freq = %d cp_freq = %d uv_freq = %d" %(io_freq, cycle_freq, cp_freq, uv_freq))
    tT        = args.cycles
    
    F0  = xp.copy(F)
    Fr0 = xp.copy(Fr)
    Fr2 = xp.copy(Fr)


    Ir   = xp.eye(dof_rom)
    ttc  = np.linspace(0, 1, 11)
    Fm_r = xp.array([xp.linalg.inv(Ir - dt * (Ls_r + xp.sin(2 * xp.pi * ttc[i]) * Lt_r)) for i in range(len(ttc))]).reshape((len(ttc), Ir.shape[0], Ir.shape[1]))
    Em_r = xp.array([xp.sin(2 * xp.pi * ttc[i]) for i in range(len(ttc))])

    verbose = 0
    while tt < tT:
        Ef     = Et(tt)

        verbose = (idx % 100 == 0)
        
        if (idx > 0 and idx % cycle_freq== 0):
            ne_r0 = xp.dot(bte_fom.op_mass, xp.dot(bte_fom.op_po2sh, bte_rom.decode(Fr0)))
            ne_r  = xp.dot(bte_fom.op_mass, xp.dot(bte_fom.op_po2sh, bte_rom.decode(Fr)))

            ne_0  = xp.dot(bte_fom.op_mass, xp.dot(bte_fom.op_po2sh, F0))
            ne    = xp.dot(bte_fom.op_mass, xp.dot(bte_fom.op_po2sh, F))

            a1    = xp.linalg.norm((ne_r0 - ne_r) / xp.max(ne_r))
            a2    = xp.linalg.norm((ne_0  - ne)   / xp.max(ne))

            print("time = %.4E [T] ROM : ||ne(T) - ne(0)||/||ne(0)|| = %.4E FOM : ||ne(T) - ne(0)||/||ne(0)|| = %.4E"%(tt, a1, a2))

            F0  = xp.copy(F)
            Fr0 = xp.copy(Fr)

        if (idx % cp_freq == 0):
            print("checkpoint time = %.4E (s)"%(tt))
            bte_rom.save_checkpoint(F, Fr, tt, "%s_rom_%02d.h5"%(bte_fom.args.fname, (idx//cp_freq)))

        if (idx % io_freq == 0):
            print("io output time = %.4E (s)"%(tt))
            F1 = bte_rom.decode(Fr)
            plot_solution(bte_fom, bte_rom, F, F1, fname="%s_rom_%04d.png"%(args.fname, idx//io_freq), time = tt)

        if (idx % rom_steps == 0):
            is_rom   = 0
            
        if(is_rom == 1):
            Fr                                  = bte_rom.decode(Fr)
            Fr[bte_rom.bte_solver.xp_vt_l,   0] = 0.0
            Fr[bte_rom.bte_solver.xp_vt_r,  -1] = 0.0
            Fr                                  = bte_rom.encode(Fr)

            Lop_r                               = Ls_r + xp.sin(2 * xp.pi * tt) * Lt_r

            if rom_use_gmres == True:
                args            = bte_rom.bte_solver.args
                rtol            = args.rtol
                atol            = args.atol
                iter_max        = args.max_iter
                pidx            = xp.argmin(xp.abs(Em_r - xp.sin(2 * xp.pi * tt)))


                def Lmvec(x):
                    return x - dt * (Lop_r @ x)

                def Pmvec(x):
                    return Fm_r[pidx] @ x

                b         = Fr.reshape((-1))
                norm_b    = xp.linalg.norm(b.reshape((-1)))
                Ndof      = Ir.shape[0]
                Lmat_op   = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Lmvec)
                Pmat_op   = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Pmvec)

                gmres_c   = glow1d_utils.gmres_counter(disp=False)
                x, status = cupyx.scipy.sparse.linalg.gmres(Lmat_op, b.reshape((-1)), x0=b.reshape((-1)), tol=rtol, atol=atol, M=Pmat_op, restart=args.gmres_rsrt, maxiter=args.gmres_rsrt * 50, callback=gmres_c)
          
                norm_res_abs  = xp.linalg.norm(Lmvec(x) -  b.reshape((-1)))
                norm_res_rel  = norm_res_abs / norm_b
          
                if (status !=0) :
                    print("t = %08d ROM GMRES solver failed! iterations =%d  ||res|| = %.4E ||res||/||b|| = %.4E"%(tt, status, norm_res_abs, norm_res_rel))
                    sys.exit(-1)
                else:
                    if (verbose==1):
                        print("t = %08d ROM GMRES solver iterations =%d  ||res|| = %.4E ||res||/||b|| = %.4E"%(tt, gmres_c.niter * args.gmres_rsrt, norm_res_abs, norm_res_rel))
                
                Fr = x

                # enforce BCs. 
                Fr                                  = bte_rom.decode(Fr)
                Fr[bte_rom.bte_solver.xp_vt_l,   0] = 0.0
                Fr[bte_rom.bte_solver.xp_vt_r,  -1] = 0.0
                Fr                                  = bte_rom.encode(Fr)

            else:
                # direct solve (LU factorization every timestep)
                Fr                                  = xp.linalg.solve(Ir - dt * Lop_r, bte_rom.encode(Fr))
        else:
            Fr                                  = bte_rom.decode(Fr)
            bte_rom.flow_map_fom_op_v(Ef, dt)
            Fr                                  = bte_rom.encode(bte_rom.step_fom_op_split(Ef, Fr, tt, dt, verbose = 0, prefactored_vsolve=True))
            is_rom                              = 1
            
        
        bte_rom.flow_map_fom_op_v(Ef, dt)
        F       = bte_rom.step_fom_op_split(Ef, F, tt, dt, verbose = verbose, prefactored_vsolve=True)

        tt  += dt
        idx +=1


if __name__ == "__main__":
    #static_efield_driver()
    efield_driver_time_harmonic()
    
    
    














