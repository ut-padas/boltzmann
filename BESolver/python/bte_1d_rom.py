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

def make_dir(dir_name):
    # Check whether the specified path exists or not
    isExist = os.path.exists(dir_name)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(dir_name)
       print("directory %s is created!"%(dir_name))

class ROM_TYPE(Enum):
    POD = 0
    DLR = 1

class TIMER:
    ROM_X = 0
    ROM_V = 1

    FOM_X = 2
    FOM_V = 3

    ROM_VX= 4

    LAST  = 5

def assemble_mat(op_dim:tuple, Lop:scipy.sparse.linalg.LinearOperator, xp=np):
    assert len(op_dim) == 2
    Imat = xp.eye(op_dim[1])

    Aop  = Lop.matmat(Imat)

    assert Aop.shape == op_dim
    return Aop

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
        self.rom_modes  = None
        self.profile    = True
        self.timer      = [profile_t("") for i in range(TIMER.LAST)]

        self.Uv         = [None for l in range(self.args.l_max+1)]
        self.Vx         = [None for l in range(self.args.l_max+1)]
        pass

    def construct_rom_basis(self, Et, v0, tb, te, dt, n_samples, eps_x, eps_v):
        bte     = self.bte_solver
        tt      = tb
        xp      = bte.xp_module

        steps   = int(np.ceil((te-tb)/dt))
        io_freq = steps // (n_samples-1)
        steps   = io_freq * (n_samples-1)

        Ps      = bte.op_po2sh
        Po      = bte.op_psh2o

        v       = xp.copy(v0)
        v_all   = xp.zeros(tuple([n_samples]) + v0.shape)
        ts      = xp.zeros(n_samples)
        
        folder_name = "%s/tb_%.2E_to_te_%.2E"%(self.args.dir, tb, te)
        make_dir(folder_name)
        print(folder_name)
        for iter in range(steps+1):
            if (iter % io_freq == 0):
                print(iter//io_freq, v_all.shape, type(v))
                v_all[int(iter//io_freq), :, :] = v[:, :]
                ts   [int(iter//io_freq)]       = tt
                plot_solution(bte, self, v, v, "%s/tidx_%d.png"%(folder_name,int(iter//io_freq)), tt, p_F0=True, p_F1=False)

            bte.bs_E            = Et(tt)
            v                   = bte.step_bte_x(v, tt, dt * 0.5)
            v                   = bte.step_bte_v(v, None, tt, dt, ts_type="BE", verbose=1)
            v                   = bte.step_bte_x(v, tt + 0.5 * dt, dt * 0.5)
            tt                 += dt

        v_all_lm               = xp.einsum("al,ilx->iax", bte.op_po2sh, v_all)
        # ne_all                 = xp.einsum("l,ilx->ix", bte.op_mass , v_all_lm)
        # v_all_lm               = xp.einsum("ix,ivx->ivx", (1/ne_all), v_all_lm)
        xp.save("%s_v_all_lm_tb_%.2E_to_te_%.2E.npy"%(bte.args.fname, tb, te),v_all_lm)
        print("v_all_lm.shape = ", v_all_lm.shape)
        print("ts = ", ts)

        v_all_lm = xp.load("%s_v_all_lm_tb_%.2E_to_te_%.2E.npy"%(bte.args.fname, tb, te))
        print("read - v_all_lm.shape = ", v_all_lm.shape)

        spec_sp                = bte.op_spec_sp
        num_p                  = spec_sp._p + 1
        num_sh                 = len(spec_sp._sph_harm_lm)
        rom_modes              = num_sh

        #assert rom_modes == 2, "ROM assumes two-term approximation"
        fl=list()

        for l in range(rom_modes):
            fl.append(v_all_lm[:, l::num_sh, :])

        def svd(fl, Uv_old, Vx_old, eps_x, eps_v, xp):
            num_t, num_p, num_x = fl.shape
            print(num_t, num_p, num_x)
            Ux, Sx, Vhx = xp.linalg.svd(fl.reshape(num_t * num_p, -1)) # Nt Nr x Nx
            Uv, Sv, Vhv = xp.linalg.svd(np.swapaxes(fl, 0, 1).reshape((num_p, num_t * num_x))) # Nr x Nt Nx

            # kr  = Uv[:, Sv > Sv[0] * eps_v].shape[1]
            # kx  = Vhx[Sx > Sx[0] * eps_x, :].T.shape[1]
            kr  = len(Sv[Sv > Sv[0] * eps_v])
            kx  = len(Sx[Sx > Sx[0] * eps_x])

            Vx  = Vhx[0:kx, :   ].T
            Uv  = Uv [:   , 0:kr]

            # if (Uv_old is not None):
            #     a1 = xp.linalg.norm(xp.dot(xp.dot(Uv_old, Uv_old.T), Uv) - Uv, axis=0) / xp.linalg.norm(Uv, axis=0)
            #     print("v", a1)
            #     Uv = xp.append(Uv_old, Uv[:, a1>eps_v], axis=1)

            # if (Vx_old is not None):
            #     a1 = xp.linalg.norm(xp.dot(xp.dot(Vx_old, Vx_old.T), Vx) - Vx, axis=0) / xp.linalg.norm(Vx, axis=0)
            #     print("x", a1)
            #     Vx = xp.append(Vx_old, Vx[:, a1>eps_v], axis=1)


            # Vx  = Vhx[0:kx, :  ].T
            # Uv  = Uv [:  , 0:kx]

            # Vx  = Vhx[Sx > Sx[0] * threshold, :].T
            # Uv  = Uv [:, Sv > Sv[0] * threshold]

            return Uv, Vx, (Sx, Sv)

        Uv  = list()
        Vx  = list()

        svalues=list()
        for l in range(rom_modes):
            if (self.rom_type == ROM_TYPE.POD):
                uv, vx , s       = svd(fl[l]             , self.Uv[l], self.Vx[l], eps_x, eps_v, xp=xp)
                
                # uv_l, vx_l , s_l = svd(fl[l][:, :,  0:1] , self.Uv[l], self.Vx[l], eps_x, eps_v, xp=xp)
                # uv_r, vx_r , s_r = svd(fl[l][:, :, -1: ] , self.Uv[l], self.Vx[l], eps_x, eps_v, xp=xp)

                # print(uv.shape, uv_l.shape, uv_r.shape)
                # uv   = xp.append(uv, uv_l, axis=1)
                # uv   = xp.append(uv, uv_r, axis=1)
                # uv,_ = xp.linalg.qr(uv)

                # vx = xp.append(vx, vx_l, axis=1)
                # vx = xp.append(vx, vx_r, axis=1)


            elif (self.rom_type == ROM_TYPE.DLR):
                Fs       = xp.dot(Ps, v0)
                u, s, vt = xp.linalg.svd(Fs[l::num_sh])
                rr       = max(100, len(s[s>=s[0] * threshold[l]]))
                print(l, u.shape, s.shape, vt.T.shape, rr)
                uv       = u   [:, 0:rr]
                vx       = vt.T[:, 0:rr]
            else:
                raise NotImplementedError

            # print("l ", l, "Uv: ", uv.shape, "Vx: ", vx.shape)
            Uv.append(uv)
            Vx.append(vx)
            svalues.append(s)
            #Uv.append(xp.eye(num_p))
            #Vx.append(xp.eye(len(bte.xp)))

        self.Uv = Uv
        self.Vx = Vx
        self.rom_modes = rom_modes

        if (self.rom_type == ROM_TYPE.POD):
            plt.figure(figsize=(6,6), dpi=200)
            plt.semilogy(xp.asnumpy(svalues[0][0]/svalues[0][0][0]), label=r"$\sigma_x^{l=%d}$"%(0))
            plt.semilogy(xp.asnumpy(svalues[0][1]/svalues[0][1][0]), label=r"$\sigma_v^{l=%d}$"%(0))
            plt.semilogy(xp.asnumpy(svalues[1][0]/svalues[1][0][0]), label=r"$\sigma_x^{l=%d}$"%(1))
            plt.semilogy(xp.asnumpy(svalues[1][1]/svalues[1][1][0]), label=r"$\sigma_v^{l=%d}$"%(1))
            plt.grid(visible=True)
            plt.legend()
            plt.title(r"$N_r$=%d $N_l$=%d $N_x$=%d"%(num_p, num_sh, num_p))
            plt.ylabel(r"normalized singular value")
            plt.tight_layout()
            plt.savefig("%s_svd_tb_%.2E_te_%.2E.png"%(bte.args.fname, tb, te))
            plt.close()

        # self.PUv = [xp.dot(self.Uv[l], self.Uv[l].T) for l in range(self.rom_modes)]
        # self.PVx = [xp.dot(self.Vx[l], self.Vx[l].T) for l in range(self.rom_modes)]
        # self.ImPUv = [xp.eye(Uv[l].shape[0]) - xp.dot(self.Uv[l], self.Uv[l].T) for l in range(self.rom_modes)]
        # self.ImPVx = [xp.eye(Vx[l].shape[0]) - xp.dot(self.Vx[l], self.Vx[l].T) for l in range(self.rom_modes)]

        return

    def init(self, basis_id:str):
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

        self.update_basis(self.Uv, self.Vx, basis_id=basis_id)
        return

    def update_basis(self, Uv, Vx, basis_id="0"):
        bte      = self.bte_solver
        param    = bte.param
        xp       = bte.xp_module
        rom_lm   = self.rom_modes

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

        assert num_sh == rom_lm
        for l in range(1, self.rom_modes):
            self.vec_offset[l]    = self.vec_offset[l-1]     + self.vec_len[l-1]
            self.vec_kr_offset[l] = self.vec_kr_offset[l-1] + self.vec_kr_len[l-1]
            self.vec_kx_offset[l] = self.vec_kx_offset[l-1] + self.vec_kx_len[l-1]

        self.vec_idx   = [xp.arange(self.vec_offset[l], self.vec_offset[l] + self.vec_len[l]) for l in range(self.rom_modes)]
        self.dof_rom   = np.sum(np.array(self.vec_len))

        # self.Uv_op     = xp.zeros((num_p * num_sh, np.sum(self.vec_kr_len)))
        # self.Vx_op     = xp.zeros((num_x * num_sh, np.sum(self.vec_kx_len)))

        # for l in range(self.rom_modes):
        #     self.Uv_op[l * num_p: (l+1) * num_p  , self.vec_kr_offset[l] : self.vec_kr_offset[l] + self.vec_kr_len[l]] = self.Uv[l]
        #     self.Vx_op[l * num_x: (l+1) * num_x  , self.vec_kx_offset[l] : self.vec_kx_offset[l] + self.vec_kx_len[l]] = self.Vx[l]

        # UU = xp.dot(self.Uv_op.T, self.Uv_op)
        # VV = xp.dot(self.Vx_op.T, self.Vx_op)

        # for lx in range(rom_lm):
        #     for ly in range(rom_lm):
                
        #         Bu = UU[self.vec_kr_offset[lx] : self.vec_kr_offset[lx] + self.vec_kr_len[lx], 
        #                    self.vec_kr_offset[ly] : self.vec_kr_offset[ly] + self.vec_kr_len[ly]]

        #         Bv = VV[self.vec_kx_offset[lx] : self.vec_kx_offset[lx] + self.vec_kx_len[lx], 
        #                    self.vec_kx_offset[ly] : self.vec_kx_offset[ly] + self.vec_kx_len[ly]]

        #         if (lx==ly):
        #             Iu = xp.eye(self.vec_kr_len[lx])
        #             Iv = xp.eye(self.vec_kx_len[lx])
        #             a1 = xp.linalg.norm(Iu-Bu)/xp.linalg.norm(Iu)
        #             a2 = xp.linalg.norm(Iv-Bv)/xp.linalg.norm(Iv)
        #             print("(%d, %d) = %.8E %.8E"%(lx, ly, a1, a2))
        #         else:
        #             a1 = xp.linalg.norm(Bu)
        #             a2 = xp.linalg.norm(Bv)
        #             print("(%d, %d) = %.8E %.8E"%(lx, ly, a1, a2))


        #sys.exit(0)
        # print(self.Uv_op.shape, self.Vx_op.shape)
        #for l in range(self.rom_modes):
        #     print(xp.dot(self.Uv_op.T, self.Uv_op)[self.vec_kr_offset[l] : self.vec_kr_offset[l] + self.vec_kr_len[l], self.vec_kr_offset[l] : self.vec_kr_offset[l] + self.vec_kr_len[l]])

        # print("basis updated")
        # print("basis shape", self.vec_shape)
        # print("basis ", self.vec_len)
        # print("basis shape", self.vec_offset)
        with open("%s_rom_size.txt"%(bte.args.fname), 'a') as f:
            f.write("%s : "%(basis_id) + str(self.vec_shape)+"\n")
            f.close()

        # #space advection op. diagonalization (Nr x Nr) and its inverse
        # Qx    = bte.op_adv_x_q
        # invQx = bte.op_adv_x_qinv
        # #advection op. inverted solve
        # Lx    = bte.bte_x_shift

        # invQPo   = xp.einsum("al,lbcd->abcd", invQx, Po.reshape((num_p, num_vt, num_p, num_sh))).reshape((num_p * num_vt, num_p * num_sh))
        # PsQ      = xp.einsum("abld,lc->abcd", Ps.reshape((num_p, num_sh, num_p, num_vt)), Qx).reshape((num_p * num_sh, num_p * num_vt))


        self.Uv  = Uv
        self.Vx  = Vx

        Uv0, Uv1 = self.Uv[0], self.Uv[1]
        Vx0, Vx1 = self.Vx[0], self.Vx[1]

        print("||I - U^t U|| = %.8E "%xp.linalg.norm(xp.dot(Uv0.T, Uv0) - xp.eye(Uv0.shape[1])))
        print("||I - U^t U|| = %.8E "%xp.linalg.norm(xp.dot(Uv1.T, Uv1) - xp.eye(Uv1.shape[1])))

        print("||I - V^t V|| = %.8E "%xp.linalg.norm(xp.dot(Vx0.T, Vx0) - xp.eye(Vx0.shape[1])))
        print("||I - V^t V|| = %.8E "%xp.linalg.norm(xp.dot(Vx1.T, Vx1) - xp.eye(Vx1.shape[1])))

        self.Gxr     = [[None for j in range(rom_lm)] for i in range(rom_lm)]
        self.Dxr     = [[None for j in range(rom_lm)] for i in range(rom_lm)]
        self.Cvr     = [[None for j in range(rom_lm)] for i in range(rom_lm)]
        self.Avr     = [[None for j in range(rom_lm)] for i in range(rom_lm)]


        for i in range(rom_lm):
            for j in range(rom_lm):
                self.Gxr[i][j] = xp.dot(self.Uv[i].T, xp.dot(self.Gx [i::num_sh, j::num_sh], self.Uv[j]))
                self.Cvr[i][j] = xp.dot(self.Uv[i].T, xp.dot(self.Cop[i::num_sh, j::num_sh], self.Uv[j]))
                self.Avr[i][j] = xp.dot(self.Uv[i].T, xp.dot(self.Av [i::num_sh, j::num_sh], self.Uv[j]))
                self.Dxr[i][j] = xp.dot(self.Vx[j].T, xp.dot(self.Dx.T, self.Vx[i]))

        num_pc_evals = len(self.bte_solver.Evals)
        self.Pvr     = [[[None for j in range(rom_lm)] for i in range(rom_lm)] for k in range(num_pc_evals)]
        for pc_idx in range(num_pc_evals):
            for i in range(rom_lm):
                for j in range(rom_lm):
                    #print(pc_idx, i, j, self.Uv[i].T.shape, self.bte_solver.PmatE[pc_idx].shape, self.Uv[j].shape, type(self.bte_solver.PmatE[pc_idx]))
                    self.Pvr[pc_idx][i][j] = xp.dot(self.Uv[i].T, xp.dot(self.bte_solver.PmatE[pc_idx][i::num_sh , j::num_sh], self.Uv[j]))


        ### ROM VX solve
        self.pCvr = [xp.dot(xp.dot(Po, self.Cop)[:, l::num_sh], self.Uv[l]) for l in range(rom_lm)]
        self.pAvr = [xp.dot(xp.dot(Po, self.Av) [:, l::num_sh], self.Uv[l]) for l in range(rom_lm)]
        self.pAxr = [xp.dot(xp.dot(self.Ax, Po) [:, l::num_sh], self.Uv[l]) for l in range(rom_lm)]


        print("ROM dof ", self.dof_rom)


        Ef               = Et(0)
        Im               = xp.eye(self.dof_rom)
        
        #self.Lvx_inv     = self.assemble_rom_vx_op(xp.ones_like(bte.xp) * Ef)

        # Lvx              = xp.asnumpy(self.Lvx_inv)
        # M , Q            = np.linalg.eig(Lvx)
        # M , Q            = xp.asarray(M), xp.asarray(Q)
        # Qinv             = xp.linalg.inv(Q)
        # Lvx              = xp.dot(Q * M , Qinv)

        # print("||I -   Q Qinv||   = %.8E "%(xp.linalg.norm(Im - xp.dot(Q, Qinv))/xp.linalg.norm(Im)))
        # print("||I -   Qinv Q||   = %.8E "%(xp.linalg.norm(Im - xp.dot(Qinv, Q))/xp.linalg.norm(Im)))
        # print("||L - Q M Qinv||   = %.8E "%(xp.linalg.norm(self.Lvx_inv - Lvx)/xp.linalg.norm(self.Lvx_inv)))

        # self.Lvx_M    = M
        # self.Lvx_Q    = Q
        # self.Lvx_Qinv = Qinv

        #L1           = Im - bte.args.cfl * self.Lvx_inv
        #self.Lvx_inv = xp.linalg.inv(L1)

        #print("||I -   Lvx Lvx_inv||   = %.8E "%(xp.linalg.norm(Im - xp.dot(L1, self.Lvx_inv))/xp.linalg.norm(Im)))
        #print("||I -   Lvx_inv Lvx||   = %.8E "%(xp.linalg.norm(Im - xp.dot(self.Lvx_inv, L1))/xp.linalg.norm(Im)))

        self.Lv_inv  = self.assemble_rom_v_op (Ef)
        L1           = Im - bte.args.cfl * self.Lv_inv
        self.Lv_inv  = xp.linalg.inv(L1)

        print("||I -   Lv Lv_inv||   = %.8E "%(xp.linalg.norm(Im - xp.dot(L1, self.Lv_inv))/xp.linalg.norm(Im)))
        print("||I -   Lv_inv Lv||   = %.8E "%(xp.linalg.norm(Im - xp.dot(self.Lv_inv, L1))/xp.linalg.norm(Im)))

        #self.flow_map_v    = self.assemble_rom_v_flow_map(Ef, bte.args.cfl)
        #self.flow_map_vx  = xp.dot(self.flow_map_x, xp.dot(self.flow_map_v, self.flow_map_x))
        
        # Lv_inv          = xp.linalg.pinv(xp.eye(self.Cop.shape[0]) - dt * param.tau * (self.Cop + Ef[0] * self.Av), rcond=1e-3) #xp.linalg.inv(xp.eye(self.Cop.shape[0]) - dt * param.tau * (self.Cop + Ef[0] * self.Av))
        # #print(Lv_inv.shape, self.Uv_op.shape)
        # self.Lv_inv     = Lv_inv
        # #self.flow_map_v = xp.dot(self.Uv_op.T, xp.dot(Lv_inv, self.Uv_op))
        
        return

    def get_rom_lm(self, Fr, l, m=0):
        return Fr.reshape((-1))[self.vec_idx[l]].reshape(self.vec_shape[l])

    def rom_vec(self):
        bte              = self.bte_solver
        xp               = bte.xp_module
        return xp.zeros(self.dof_rom)

    def matricized_rom_vec(self, Xr):
        Xrp = xp.zeros((np.sum(self.vec_kr_len), np.sum(self.vec_kx_len)))
        assert Xr.shape[0] == self.dof_rom
        for l in range(self.rom_modes):
            Xrp[self.vec_kr_offset[l] : self.vec_kr_offset[l] + self.vec_kr_len[l], 
                self.vec_kx_offset[l] : self.vec_kx_offset[l] + self.vec_kx_len[l]] = self.get_rom_lm(Xr,l)
        return Xrp
    
    def vectorized_rom_mat(self, Xr):
        assert Xr.shape == (np.sum(self.vec_kr_len), np.sum(self.vec_kx_len))
        Xrp = xp.zeros(self.dof_rom)
        for l in range(self.rom_modes):
            Xrp[self.vec_idx[l]] = Xr[self.vec_kr_offset[l] : self.vec_kr_offset[l] + self.vec_kr_len[l], 
                self.vec_kx_offset[l] : self.vec_kx_offset[l] + self.vec_kx_len[l]].reshape((-1))
        
        return Xrp

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
        Fr_lm        = [xp.dot(Uv[i].T, xp.dot(F_lm[i::num_sh, :], Vx[i])) for i in range(rom_modes)]
        Fr           = xp.array(Fr_lm[0].reshape((-1)))

        for l in range(1, rom_modes):
            Fr = xp.append(Fr, Fr_lm[l].reshape((-1)))

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

        for l in range(self.rom_modes):
            Frl              = self.get_rom_lm(Fr, l, m=0)
            F[l::num_sh, : ] = xp.dot(Uv[l], xp.dot(Frl, Vx[l].T))

        return xp.dot(Po, F)

    def assemble_rom_vx_op(self, Ef):
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

        Lop_mat = assemble_mat((self.dof_rom, self.dof_rom),Lop, xp=xp)
        return Lop_mat

    """
    def assemble_rom_vx_flow_map(self, Ef, dt):
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

        def flmap_vx(Xr, Ef, dt, xp=xp):
            Fo                  = self.decode(Xr)
            tt                  = 0.0
            bte.bs_E            = Ef

            v                   = bte.step_bte_x(Fo, tt, dt * 0.5)
            v                   = bte.step_bte_v(v, None, tt, dt, ts_type="BE", verbose=0)
            v                   = bte.step_bte_x(v, tt + 0.5 * dt, dt * 0.5)
            
            return self.encode(v)

        Ndof = self.dof_rom
        Lop  = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec = lambda x: flmap_vx(x, Ef, dt, xp=xp), dtype=xp.float64)
        Lop_mat = assemble_mat((self.dof_rom, self.dof_rom),Lop, xp=xp)
        return Lop_mat

    def assemble_rom_x_flow_map(self, dt):
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

        def flmap_x(Xr, dt, xp=xp):
            Fo                  = self.decode(Xr)
            tt                  = 0.0
            v                   = bte.step_bte_x(Fo, tt, dt * 0.5)
            return self.encode(v)

        Ndof = self.dof_rom
        Lop  = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec = lambda x: flmap_x(x, dt, xp=xp), dtype=xp.float64)
        Lop_mat = assemble_mat((self.dof_rom, self.dof_rom),Lop, xp=xp)
        return Lop_mat
    """

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
        Lop_mat = assemble_mat((self.dof_rom, self.dof_rom),Lop, xp=xp)
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
        Lop_mat = assemble_mat((self.dof_rom, self.dof_rom),Lop, xp=xp)
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

        Lop_mat = assemble_mat((self.dof_rom, self.dof_rom),Lop, xp=xp)
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
        #Frh            = xp.dot(self.Lv_inv, Frh)
        # a1              = xp.zeros((np.sum(self.vec_kr_len), np.sum(self.vec_kx_len)))
        # for l in range(self.rom_modes):
        #     a1[self.vec_kr_offset[l] : self.vec_kr_offset[l] + self.vec_kr_len[l],
        #        self.vec_kx_offset[l] : self.vec_kx_offset[l] + self.vec_kx_len[l]] = self.get_rom_lm(Frh, l)
        # a1              = xp.dot(self.flow_map_v, a1)
        # for l in range(self.rom_modes):
        #     Frh[self.vec_idx[l]] = a1[self.vec_kr_offset[l] : self.vec_kr_offset[l] + self.vec_kr_len[l],
        #        self.vec_kx_offset[l] : self.vec_kx_offset[l] + self.vec_kx_len[l]].reshape((-1))

        # Frh = xp.dot(self.Uv_op.T,xp.dot(xp.dot(self.Lv_inv, xp.dot(self.Uv_op, xp.dot(self.matricized_rom_vec(Frh), self.Vx_op.T))), self.Vx_op))
        # #Frh  = xp.dot(xp.dot(self.flow_map_v, xp.dot(self.matricized_rom_vec(Frh), self.Vx_op.T)), self.Vx_op)
        # Frh  = self.vectorized_rom_mat(Frh)

        #bte.bs_E        = Ef
        #Frh             = self.encode(bte.step_bte_v(self.decode(Frh), None, tt, dt, ts_type="BE", verbose=verbose))
        #Frh             = self.encode(bte.step_bte_v(xp.dot(bte.op_po2sh, self.decode(Frh)), None, tt, dt, ts_type="BE", verbose=verbose))
        # Frh              = xp.dot(bte.op_po2sh, self.decode(Frh))
        # Frh              = xp.dot(bte.op_psh2o, xp.dot(self.Lv_inv, Frh))
        # Frh              = self.encode(Frh)

        Frh            = xp.dot(self.Lv_inv, Frh)

        
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
    
    #############################################################################################
    ############################## FOM routines below ###########################################
    #############################################################################################

    def step_fom_op_split(self, Ef, F, time, dt, verbose=1):
        """
        full order model timestep
        """
        bte                 = self.bte_solver
        tt                  = time
        bte.bs_E            = Ef

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

        v                   = bte.step_bte_v(v, None, tt, dt, ts_type="BE", verbose=verbose)
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

        self.init(basis_id="cp: t=%.4E"%(time))
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

if __name__ == "__main__":
    from glowdischarge_boltzmann_1d import glow1d_boltzmann, args_parse
    args         = args_parse()

    bte_fom      = glow1d_boltzmann(args)
    u, v         = bte_fom.initialize()
    args         = bte_fom.args


    dt           = args.cfl
    io_cycle     = args.io_cycle_freq
    cp_cycle     = args.cp_cycle_freq

    io_freq      = int(np.round(io_cycle/dt))
    cp_freq      = int(np.round(cp_cycle/dt))
    cycle_freq   = int(np.round(1/dt))
    tio_freq     = 20
    uv_freq      = int(np.round(10/dt))

    rom_eps_x    = 1e-5
    rom_eps_v    = 1e-6
    restore      = 1
    rs_idx       = 6
    train_cycles = 1
    num_samples  = 31
    psteps       = 1


    if args.use_gpu==1:
        gpu_device = cp.cuda.Device(args.gpu_device_id)
        gpu_device.use()

    u, v    = bte_fom.step_init(u, v, dt)
    xp      = bte_fom.xp_module

    spec_sp = bte_fom.op_spec_sp
    num_p   = spec_sp._p + 1
    num_sh  = len(spec_sp._sph_harm_lm)

    bte_rom = boltzmann_1d_rom(bte_fom)
    xxg     = xp.asarray(bte_fom.xp)
    Et      = lambda t: xp.ones_like(bte_fom.xp) * 1e3 #* xp.sin(2 * xp.pi * t)
    #Et      = lambda t: (xxg**3) * 1e4 * xp.sin(2 * xp.pi * t)

    tt        = 0
    if (restore==1):
        F, Fr, tt = bte_rom.restore_checkpoint("%s_rom_%02d.h5"%(bte_fom.args.fname, rs_idx))
        idx       = (rs_idx) * cp_freq
        print("checkpoint restored time = %.4E (s) idx = %d"%(tt, idx))
    else:
        bte_rom.construct_rom_basis(Et, v, 0, train_cycles, dt, num_samples, eps_x = rom_eps_x, eps_v = rom_eps_v)
        bte_rom.init("t=%.4E"%(0))

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
    # flm_error = [list(), list()]
    # ts        = list()

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

        if(idx > 0 and (idx % uv_freq) == 0):
            print("-----------------sampling adjust the rom basis------------------------")
            Fr_old = xp.copy(Fr)
            v_fom  = bte_rom.decode(Fr)
            bte_rom.construct_rom_basis(Et, v_fom, tt, tt + train_cycles, dt, num_samples, eps_x = rom_eps_x, eps_v = rom_eps_v)
            bte_rom.init(basis_id="t=%.4E"%(tt))
            Fr     = bte_rom.encode(v_fom)
            v_fom1 = bte_rom.decode(Fr)
            plot_solution(bte_fom, bte_rom, v_fom, v_fom1, fname="%s_rom_%04d_bc.png"%(args.fname, idx//io_freq), time = tt)
            print("---------------------------------------------------------------------")


        #Fr     = bte_rom.step_rom_eigen_solve_vx(Ef, Fr, tt, dt)
        #Fr2     = bte_rom.step_rom_vx(Ef, Fr2, tt, dt, atol=bte_fom.args.atol, rtol=bte_fom.args.rtol, gmres_rst=20, gmres_iter= 20, verbose = (idx % tio_freq))
        # F      = bte_rom.decode(Fr2)
        #Fr      = Fr2


        if (idx % tio_freq == 1):
            Yr1     = bte_rom.encode(F)   #xp.dot(xp.dot(bte_rom.Uv_op.T, Fs), bte_rom.Vx_op)
            Yr2     = bte_rom.decode(Yr1) #xp.dot(xp.dot(bte_rom.Uv_op, Yr1), bte_rom.Vx_op.T)

            print("||Y-PY|| = %.8E"%(xp.linalg.norm(F-Yr2)/xp.linalg.norm(F)))

        Fr      = bte_rom.step_rom_op_split(Ef, Fr, tt, dt, type="BE", atol=bte_fom.args.atol, rtol=bte_fom.args.rtol, verbose=(idx % tio_freq))
        #Fr     = xp.dot(bte_rom.flow_map_vx, Fr)
        #Fo     = bte_rom.step_fom_ord(Et, Fo, tt, dt, type="BE", atol=1e-20, rtol=1e-3)
        #Fs     = bte_rom.step_fom_sph(Et, Fs, tt, dt, type="BE", atol=1e-20, rtol=1e-2)
        F       = bte_rom.step_fom_op_split(Ef, F, tt, dt, verbose = (idx % tio_freq))
        #F1     = bte_rom.step_fom_vx(Ef, F, tt, psteps * dt, atol=bte_fom.args.atol,  rtol=1e-3, gmres_rst=1000, gmres_iter=1, pc_type=1, verbose = (idx % tio_freq))
        #F2     = bte_rom.step_fom_vx(Ef, F, tt, psteps * dt, atol=bte_fom.args.atol,  rtol=1e-3, gmres_rst=1000, gmres_iter=1, pc_type=0, verbose = (idx % tio_freq))
        #F      = F1
        
        

        tt  += dt
        idx +=1
    
    














