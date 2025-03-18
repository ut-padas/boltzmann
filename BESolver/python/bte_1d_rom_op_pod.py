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

def make_dir(dir_name):
    # Check whether the specified path exists or not
    isExist = os.path.exists(dir_name)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(dir_name)
       print("directory %s is created!"%(dir_name))

class ROM_TYPE(Enum):
    OP_RSVD = 0 

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
    G0_lm   = F1_lm
    
    # rom_lm  = bte_rom.rom_modes
    # G0_lm   = xp.zeros((num_p* num_sh, num_x))
    # if rom_lm is not None:
    #     PUv     = [xp.dot(bte_rom.Uv[l], bte_rom.Uv[l].T) for l in range(rom_lm)]
    #     PVx     = [xp.dot(bte_rom.Vx[l], bte_rom.Vx[l].T) for l in range(rom_lm)]

    #     for l in range(rom_lm):
    #         G0_lm[l::num_sh] = xp.dot(PUv[l], xp.dot(F0_lm[0::num_sh], PVx[l]))
    
    
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
        xp              = self.bte_solver.xp_module
        assert xp == cp , "BTE rom assumes CuPy backend"
        self.args       = self.bte_solver.args
        self.rom_type   = ROM_TYPE.OP_RSVD
        self.profile    = True
        self.timer      = [profile_t("") for i in range(TIMER.LAST)]

        spec_sp         = self.bte_solver.op_spec_sp
        self.num_p      = spec_sp._p + 1
        self.num_vt     = len(self.bte_solver.xp_vt)
        self.num_sh     = len(spec_sp._sph_harm_lm)
        self.num_x      = len(self.bte_solver.xp)
        

        pass

    def init(self):
        bte     = self.bte_solver
        param   = bte.param
        xp      = bte.xp_module
        Cen     = bte.op_col_en
        Ctg     = bte.op_col_gT

        Cop     = bte.param.np0 * bte.param.n0 * (bte.op_col_en + bte.args.Tg * Ctg)
        Av      = bte.op_adv_v
        
        num_p   = self.num_p
        num_vt  = self.num_vt
        num_sh  = self.num_sh
        

        # mm      = xp.asarray(spec_sp.compute_mass_matrix())
        # mm_inv  = xp.asarray(spec_sp.inverse_mass_mat())
        # Mvr     = mm[0::num_sh, 0::num_sh]
        # Mv_inv  = mm_inv

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

    def sample_fom(self, Et, v0, tb, te, dt, n_samples, load_from_file=False):
        bte     = self.bte_solver
        tt      = tb
        xp      = bte.xp_module

        folder_name = "%s/tb_%.2E_to_te_%.2E"%(self.args.dir, tb, te)
        vfname      = "%s/v_all_tb_%.2E_to_te_%.2E.npy"%(folder_name, tb, te)
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
        v_all   = xp.zeros(tuple([n_samples]) + v0.shape)
        ts      = xp.zeros(n_samples)
        
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
        
        xp.save(vfname,v_all)
        print("v_all.shape = ", v_all.shape)
        print("ts = ", ts)

        

        return v_all.reshape((n_samples, -1)).T

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

    def assemble_fom_op(self, Ef):
        bte     = self.bte_solver
        param   = bte.param
        xp      = bte.xp_module

        num_p   = self.num_p
        num_vt  = self.num_vt 
        num_x   = self.num_x 

        Cop     = xp.asnumpy(self.Cop)
        Av      = xp.asnumpy(self.Av)
        Ax      = xp.asnumpy(self.Ax)
        Dx      = xp.asnumpy(self.Dx)

        Ps      = xp.asnumpy(bte.op_po2sh)
        Po      = xp.asnumpy(bte.op_psh2o)
        Ef      = xp.asnumpy(Ef)

        Ndof    = self.num_p * self.num_vt * self.num_x
        L1      = np.zeros((Ndof, Ndof))
        L2      = np.zeros((Ndof, Ndof))

        print("op assembly begin")

        for i in range(num_p * num_vt):
            L1[i * num_x : (i+1) * num_x, i * num_x : (i+1) * num_x] = Dx

        Cop_o = Po @ Cop @ Ps
        Av_o  = Po @ Av @ Ps

        Lv    = np.zeros((Ndof, Ndof))

        for i in range(num_p * num_vt):
            for j in range(num_p * num_vt):
                for k in range(num_x):
                    L2 [ i * num_x + k, j * num_x +  k ] = Ax[i, j]
                    Lv [ i * num_x + k, j * num_x +  k]  = param.tau * (Cop_o[i,j] + Ef[k] * Av_o[i,j])


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
    
    def flow_map_fom_op_v(self, Ef, dt):
        bte     = self.bte_solver
        param   = bte.param
        xp      = bte.xp_module

        Ps      = bte.op_po2sh
        Po      = bte.op_psh2o

        assert (Ef[0] == Ef).all() == True

        Lmat    = xp.eye(self.num_p * self.num_vt) - dt * param.tau * xp.dot(Po, xp.dot(self.Cop, Ps) + xp.dot(Ef[0] * self.Av, Ps))
        Lmat    = xp.linalg.inv(Lmat)
        
        self.Lv_inv = Lmat
        return self.Lv_inv

    def construct_rom_basis(self, Ef, rom_rank, Omega = None):
        bte     = self.bte_solver
        xp      = bte.xp_module

        
        Ndof    = self.num_p * self.num_vt * self.num_x
        if (xp == cp):
            Lop      = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec  = lambda x : self.fom_mv(Ef, x)
                                                                            , rmatvec = lambda x : self.fom_adj_mv(Ef, x))
            #u, s, vt = cupyx.scipy.sparse.linalg.svds(Lop, k=rom_rank)
            u, s, vt =rom_utils.rsvd(Lop, rom_rank, power_iter=5, Omega = Omega, xp=xp)

        else:
            raise NotImplementedError

        self.U = u
        self.S = s
        self.V = vt.T
        print(s/s[0])
        # print(xp.linalg.norm(self.U))
        # print(xp.linalg.norm(self.V))
        # print(u.T @ u)
        # print(vt  @ vt.T)

        x  = xp.random.rand(Ndof)
        y1 = self.fom_mv(Ef, x)
        rr = np.linalg.norm(y1 - xp.dot(u*s, xp.dot(vt, x)))/xp.linalg.norm(y1)
        print("rr = %.8E"%rr)
        # #sys.exit(0)

        assert self.U.shape[1] == self.V.shape[1]
        self.dof_rom   = self.U.shape[1]
        self.Ik        = xp.eye(self.U.shape[1])
        
        return

    def encode(self,  x):
        xp = self.bte_solver.xp_module
        #return xp.dot(self.V.T, x.reshape((-1)))
        return x.reshape((-1))
    
    def decode(self, x):
        xp = self.bte_solver.xp_module
        #assert x.shape[0] == self.dof_rom
        #return xp.dot(self.V, x).reshape((self.num_p * self.num_vt, self.num_x))
        return x.reshape((self.num_p * self.num_vt, self.num_x))

    # def assemble_rom_op(self):
    #     xp = self.bte_solver.xp_module
    #     return xp.dot(self.V.T, self.U * self.S)

    def step_rom_vx(self, x, dt):
        xp = self.bte_solver.xp_module
        y  = xp.copy(x).reshape((self.num_p * self.num_vt, self.num_x))
        
        y[self.bte_solver.xp_vt_l, 0]  = 0.0
        y[self.bte_solver.xp_vt_r, -1] = 0.0

        y = y.reshape((-1))

        Ut = -dt * self.U * self.S
        Vt = self.V.T

        #print("x: %.8E  y: %.8E VTy: %.8E"%(xp.linalg.norm(x), xp.linalg.norm(y), xp.linalg.norm(Vt @ y)))

        y  = y - xp.dot(Ut, xp.dot(xp.linalg.inv(self.Ik + Vt @ Ut) , xp.dot(Vt, y)))
        return y 

    def direct_steady_state(self, Ef, x0, dt, Lop = None):
        bte  = self.bte_solver
        xp   = bte.xp_module
        if Lop is None:
            Lop  = self.assemble_fom_op(Ef)

        num_x = self.num_x

        Ibv                                    = xp.eye(Lop.shape[0])
        Ibv[bte.xp_vt_l * num_x + 0, :]        = 0.0
        Ibv[bte.xp_vt_r * num_x + num_x-1, : ] = 0.0

        Lop  = xp.linalg.solve(xp.eye(Lop.shape[0]) - dt * Lop, Ibv)

        #Lop  = xp.linalg.matrix_power(Lop, int(1/dt))


        # for idx in range(10000):
        #     x1 = xp.dot(Lop, x0)
        #     print("idx : %04d rr = %.8E"%(idx, (xp.linalg.norm(x1-x0)/xp.linalg.norm(x0))))
        #     x0 = x1

        return Lop

    def plot_memory_term(self, Ef, x0, dt, rank_k, Lop=None, Lop_svd = None):
        bte  = self.bte_solver
        xp   = bte.xp_module
        if Lop is None:
            Lop  = self.assemble_fom_op(Ef)

        if Lop_svd is None:
            u, s, vt = xp.linalg.svd(Lop)
            v        = vt.T
        else:
            u  = Lop_svd[0]
            s  = Lop_svd[1]
            vt = Lop_svd[2]
            v  = vt.T

        folder_name = "%s/op_Nr%d_Nvt_%d_Nx%d"%(bte.args.dir, self.num_p, self.num_vt, self.num_x)
        make_dir(folder_name)

        # choice      = np.random.choice(range(Lop.shape[0]), size=(rank_k,), replace=False)    
        # ind         = xp.zeros(Lop.shape[0], dtype=bool)
        # ind[choice] = True
        
        # ur = u[:, 0:rank_k]
        # sr = s[   0:rank_k]
        # vr = v[:, 0:rank_k]

        # ud = u[:, rank_k:]
        # sd = s[   rank_k:]
        # vd = v[:, rank_k:]

        ur = u[:, len(s)-rank_k:]
        sr = s[   len(s)-rank_k:]
        vr = v[:, len(s)-rank_k:]

        ud = u[:, 0:len(s)-rank_k]
        sd = s[   0:len(s)-rank_k]
        vd = v[:, 0:len(s)-rank_k]

        #x0 = xp.dot(vr, xp.dot(vr.T, x0))

        # ur = u[:, ind]
        # sr = s[   ind]
        # vr = v[:, ind]

        # ud = u[:, ~ind]
        # sd = s[   ~ind]
        # vd = v[:, ~ind]

        L1   = xp.dot(ur * sr , vr.T) + xp.dot(ud * sd, vd.T)
        a1   = xp.linalg.norm(Lop-L1)/xp.linalg.norm(Lop)

        print("||Lop - (Lr + Lu)||/||Lop|| = %.8E"%(a1))

        Lrr = xp.dot(vr.T, ur * sr)
        Lru = xp.dot(vr.T, ud * sd)

        Lur = xp.dot(vd.T, ur * sr)
        Luu = xp.dot(vd.T, ud * sd)

        

        num_x                                  = self.num_x
        Ibv                                    = xp.eye(Lop.shape[0])
        Ibv[bte.xp_vt_l * num_x + 0, :]        = 0.0
        Ibv[bte.xp_vt_r * num_x + num_x-1, : ] = 0.0
        
        # y0                                     = Ibv @ x0.reshape((-1))
        # y1                                     = xp.copy(x0).reshape((self.num_p * self. num_vt, self.num_x))
        # y1[self.bte_solver.xp_vt_l,  0 ]       = 0.0
        # y1[self.bte_solver.xp_vt_r, -1 ]       = 0.0
        # y1                                     = y1.reshape((-1))
        # assert xp.linalg.norm(y0-y1)/xp.linalg.norm(y1) < 1e-14
        Ibv                                    = xp.linalg.solve(xp.eye(Lop.shape[0]) - dt * Lop, Ibv)

        xr  = xp.dot(vr.T, x0)
        xu  = xp.dot(vd.T, x0)

        a1  = xp.linalg.norm(x0 - xp.dot(vr, xr) - xp.dot(vd,xu))/xp.linalg.norm(x0)
        print("||x0 - (vr @ xr + vd @ xu)||/||x0|| = %.8E"%(a1))

        T    = 100 
        tt   = xp.linspace(0, T, 1001)
        dt_m = tt[1] - tt[0] 

        assert dt_m > dt and int(dt_m/dt) * dt == dt_m

        Ibv  = xp.linalg.matrix_power(Ibv, int(dt_m/dt))
        #cupyx.scipy.linalg.expm(Luu * (T - tt[i]))
        #b0   = cupyx.scipy.linalg.expm(Luu * T) @ xu
        
        Xr        = xp.zeros((len(x0), len(tt)))
        Xu        = xp.zeros((len(x0), len(tt)))
        
        Xr[:, 0]  = xp.dot(vr,xr)
        Xu[:, 0]  = xp.dot(vd,xu)

        y0_l2     = xp.zeros(len(tt))

        y0        = xp.copy(x0)
        y0_l2[0]  = xp.linalg.norm(y0)
        print("time = %.4E ||y0|| = %.8E "%(tt[0], xp.linalg.norm(y0)))

        F         = y0.reshape((self.num_p * self.num_vt, self.num_x))
        Fr        = xp.dot(vr,xp.dot(vr.T , y0)).reshape((self.num_p * self.num_vt, self.num_x))
        plot_solution(self.bte_solver, self, F, Fr, fname="%s/sol_k_%06d_%04d.png"%(folder_name,rank_k, 0), time = tt[0], p_F1=True)

        for tidx in range(1, len(tt)):
            y0          = xp.dot(Ibv , y0)
            y0_l2[tidx] = xp.linalg.norm(y0)
            #print(y0.shape, vr.shape, xp.dot(vr, xp.dot(vr.T, y0)).shape)
            Xr[:, tidx] = y0 - xp.dot(vr, xp.dot(vr.T, y0))
            Xu[:, tidx] = y0 - xp.dot(vd, xp.dot(vd.T, y0))

            if(tidx % 250 == 0):
                F         = y0.reshape((self.num_p * self.num_vt, self.num_x))
                Fr        = xp.dot(vr,xp.dot(vr.T , y0)).reshape((self.num_p * self.num_vt, self.num_x))
                plot_solution(self.bte_solver, self, F, Fr, fname="%s/sol_k_%06d_%04d.png"%(folder_name, rank_k, tidx), time = tt[tidx], p_F1=True)
                print("time = %.4E ||y0|| = %.8E "%(tt[tidx], xp.linalg.norm(y0)))

        #print(xp.asnumpy(xp.linalg.norm(Xr.T, axis=1)))
        plt.figure(figsize=(8, 4), dpi=300)

        plt.subplot(1, 2, 1)
        plt.title(r"$L = L_x + L_v$")
        plt.xlabel(r"k")
        plt.ylabel(r"$\sigma_k$")
        plt.grid(visible=True)
        plt.semilogy(xp.asnumpy(s/s[0]))
        plt.grid(visible=True)

        plt.subplot(1, 2, 2)
        plt.title(r"rank = %d"%(rank_k))
        plt.xlabel(r"time [T]")
        plt.ylabel(r"||x||")
        plt.grid(visible=True)

        #print(xp.linalg.norm(Xr.T, axis=1), xp.linalg.norm(Xr.T, axis=1).shape)
        #print(y0_l2, y0_l2.shape)

        plt.semilogy(xp.asnumpy(tt), xp.asnumpy(xp.linalg.norm(Xr.T, axis=1))/xp.asnumpy(y0_l2), label=r"$||f - f_r||/||f||$")
        plt.semilogy(xp.asnumpy(tt), xp.asnumpy(xp.linalg.norm(Xu.T, axis=1))/xp.asnumpy(y0_l2), label=r"$||f - f_u||/||f||$")
        #plt.semilogy(xp.asnumpy(tt), xp.asnumpy(xp.linalg.norm(Xu.T, axis=1)), label=r"$||f_u||$")
        #plt.semilogy(xp.asnumpy(tt), xp.asnumpy(y0_l2)                       , label=r"$||f||$")
        plt.legend()
        plt.tight_layout()

        plt.savefig("%s/rom_k%06d.png"%(folder_name, rank_k))

    def plot_memory_term_eig(self, Ef, x0, dt, rank_k, Lop=None, Lop_eig = None):
        bte  = self.bte_solver
        xp   = bte.xp_module
        if Lop is None:
            Lop  = self.assemble_fom_op(Ef)

        if Lop_eig is None:
            s, u     = np.linalg.eig(xp.asnumpy(Lop))
            u        = xp.asarray(u)
            s        = xp.asarray(s)
            vt       = xp.linalg.inv(u)
            v        = vt.T
        else:
            u        = Lop_eig[0]
            s        = Lop_eig[1]
            vt       = Lop_eig[2]
            v        = vt.T
            
        ur = u[:, 0:rank_k]
        sr = s[   0:rank_k]
        vr = v[:, 0:rank_k]

        ud = u[:, rank_k:]
        sd = s[   rank_k:]
        vd = v[:, rank_k:]

        L1   = xp.real(xp.dot(ur * sr , vr.T) + xp.dot(ud * sd, vd.T))
        a1   = xp.linalg.norm(Lop-L1)/xp.linalg.norm(Lop)

        I    = xp.eye(Lop.shape[0])

        print("||Lop - (Lr + Lu)||/||Lop|| = %.8E"%(a1))
        print("||I   - Q Q^{-1}|| /||I||   = %.8E"%(xp.linalg.norm(I - xp.dot(u,vt))/xp.linalg.norm(I)))
        print("||I   - Q^{-1} Q|| /||I||   = %.8E"%(xp.linalg.norm(I - xp.dot(vt,u))/xp.linalg.norm(I)))

        Lrr = xp.dot(vr.T, ur * sr)
        Lru = xp.dot(vr.T, ud * sd)

        Lur = xp.dot(vd.T, ur * sr)
        Luu = xp.dot(vd.T, ud * sd)
       

        num_x                                  = self.num_x
        Ibv                                    = xp.eye(Lop.shape[0])
        Ibv[bte.xp_vt_l * num_x + 0, :]        = 0.0
        Ibv[bte.xp_vt_r * num_x + num_x-1, : ] = 0.0
        Ibv                                    = xp.linalg.solve(xp.eye(Lop.shape[0]) - dt * Lop, Ibv)

        xr  = xp.dot(vr.T, x0)
        xu  = xp.dot(vd.T, x0)

        a1  = xp.linalg.norm(x0 - np.real(xp.dot(ur, xr) + xp.dot(ud,xu)))/xp.linalg.norm(x0)
        print("||x0 - (vr @ xr + vd @ xu)||/||x0|| = %.8E"%(a1))

        T    = 10 
        tt   = xp.linspace(0, T, 1001)
        dt_m = tt[1] - tt[0] 

        assert dt_m > dt and int(dt_m/dt) * dt == dt_m

        Ibv  = xp.linalg.matrix_power(Ibv, int(dt_m/dt))
        #cupyx.scipy.linalg.expm(Luu * (T - tt[i]))
        #b0   = cupyx.scipy.linalg.expm(Luu * T) @ xu
        
        Xr        = xp.zeros((len(xr), len(tt)), dtype=xp.complex128)
        Xu        = xp.zeros((len(xu), len(tt)), dtype=xp.complex128)
        
        Xr[:, 0]  = xr
        Xu[:, 0]  = xu

        y0_l2     = xp.zeros(len(tt))

        y0        = xp.copy(x0)
        y0_l2[0]  = xp.linalg.norm(y0)
        for tidx in range(1, len(tt)):
            y0          = xp.dot(Ibv , y0)
            y0_l2[tidx] = xp.linalg.norm(y0)
            #print(tidx, y0_l2[tidx])
            Xr[:, tidx] = xp.dot(vr.T, y0) 
            Xu[:, tidx] = xp.dot(vd.T, y0)

        plt.figure(figsize=(8, 4), dpi=300)

        plt.subplot(1, 2, 1)
        plt.title(r"$L = L_x + L_v$")
        plt.xlabel(r"k")
        plt.ylabel(r"$\Lambda_k$")
        plt.grid(visible=True)
        plt.plot(xp.asnumpy(xp.real(s)), 'o', markersize=0.4, label="eig-real")
        plt.plot(xp.asnumpy(xp.imag(s)), 'x', markersize=0.4, label="eig-img")
        plt.grid(visible=True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.title(r"rank = %d"%(rank_k))
        plt.xlabel(r"time [T]")
        plt.ylabel(r"||x||")
        plt.grid(visible=True)

        plt.semilogy(xp.asnumpy(tt), xp.asnumpy(xp.linalg.norm(xp.real(Xr.T), axis=1)), label=r"$||Re(f_r)||$")
        plt.semilogy(xp.asnumpy(tt), xp.asnumpy(xp.linalg.norm(xp.imag(Xr.T), axis=1)), label=r"$||Im(f_r)||$")

        plt.semilogy(xp.asnumpy(tt), xp.asnumpy(xp.linalg.norm(xp.real(Xu.T), axis=1)), label=r"$||Re(f_u)||$")
        plt.semilogy(xp.asnumpy(tt), xp.asnumpy(xp.linalg.norm(xp.imag(Xu.T), axis=1)), label=r"$||Im(f_u)||$")
        
        plt.legend()
        plt.tight_layout()

        folder_name = "%s/op_Nr%d_Nvt_%d_Nx%d_eig"%(bte.args.dir, self.num_p, self.num_vt, self.num_x)
        make_dir(folder_name)
        plt.savefig("%s/rom_k%06d.png"%(folder_name, rank_k))

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

        #v                   = bte.step_bte_v(v, None, tt, dt, ts_type="BE", verbose=verbose)
        v                    = xp.dot(self.Lv_inv, v)
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
            # if (pc_type == 1):
            #     brhs              = xp.copy(F).reshape((num_p * num_vt, num_x))
            #     brhs[xp_vt_l,  0] = 0.0
            #     brhs[xp_vt_r, -1] = 0.0
                
            #     return self.decode(xp.dot(self.Lvx_inv, self.encode(brhs)))
            # else:
            #     return F

            return self.step_rom_vx(F, dt)


        if xp == cp:
            Ndof          = num_p * num_x * num_vt
            Amat_op       = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec = lambda x: x - dt * rhs_op(x, time, dt))

            brhs          = xp.copy(F)
            #enforce BCs
            brhs[xp_vt_l,  0] = 0.0
            brhs[xp_vt_r, -1] = 0.0
            norm_b        = xp.linalg.norm(brhs)

            F_init        = precond(F.reshape((-1)), time, dt)
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

    def fom_static_solve(self, Et, F):
        bte     = self.bte_solver
        xp      = bte.xp_module
        param   = bte.param

        Ef      = Et(0)
        assert (Ef == Et(0.1)).all() and (Ef[0] == Ef).all()

        Ndof    = self.num_p * self.num_vt * self.num_x
        km      = int(1e-3 * Ndof) // 2
        tol     = 1e-10

        if (xp == cp):
            LTxL      = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec  = lambda x : self.fom_adj_mv(Ef, self.fom_mv(Ef, x)))
            LxLT      = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec  = lambda x : self.fom_mv(Ef, self.fom_adj_mv(Ef, x)))
            
            #u, s, vt = cupyx.scipy.sparse.linalg.svds(Lop, k=km, which='SA')

            d1, u_sa   = cupyx.scipy.sparse.linalg.eigsh(LxLT, k=km, which="SA", tol=tol)
            d2, v_sa   = cupyx.scipy.sparse.linalg.eigsh(LTxL, k=km, which="SA", tol=tol)

            d3, u_la   = cupyx.scipy.sparse.linalg.eigsh(LxLT, k=km, which="LA", tol=tol)
            d4, v_la   = cupyx.scipy.sparse.linalg.eigsh(LTxL, k=km, which="LA", tol=tol)
        else:
            raise NotImplementedError

        print("SA eigen error : %.8E  u_la ^T u_sa : %.8E"%(xp.linalg.norm(d1-d2)/xp.linalg.norm(d1), xp.linalg.norm(xp.dot(u_la.T, u_sa))))
        print("LA eigen error : %.8E  v_la ^T v_sa : %.8E"%(xp.linalg.norm(d3-d4)/xp.linalg.norm(d2), xp.linalg.norm(xp.dot(v_la.T, v_sa))))

        Lop      = bte_rom.assemble_fom_op(Et(0))
        u, s, vt = xp.linalg.svd(Lop)    

        s_sa     = xp.zeros_like(s)
        s_la     = xp.zeros_like(s)

        s_sa[len(s)-km:] = xp.sqrt(0.5 * (d1 + d2))
        s_la[0:km]       = xp.sqrt(0.5 * (d3 + d4))

        plt.figure(figsize=(6, 6), dpi=300)
        plt.semilogy(xp.asnumpy(s)        , label=r"svd")
        plt.semilogy(xp.asnumpy(s_sa), 'x', markersize=.9, label=r"$sqrt(\lambda)$-Lanczos-SA")
        plt.semilogy(xp.asnumpy(s_la), 'x', markersize=.9, label=r"$sqrt(\lambda)$-Lanczos-LA")
        plt.grid(visible=True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("%s_op_spectrum.png"%(self.bte_solver.args.fname))
        plt.close()

        sys.exit(0)
        # Ps       = bte.op_po2sh
        # Po       = bte.op_psh2o
        # Lmat     = param.tau * xp.dot(Po, xp.dot(self.Cop, Ps) + xp.dot(Ef[0] * self.Av, Ps))
        
        # alpha    = 1e-3
        # Lmat_inv = xp.linalg.inv( Lmat)

        # it  = 0
        # x0  = xp.copy(F).reshape((self.num_p * self.num_vt, self.num_x))
        # DxT = self.Dx.T

        # while(it < 100):
        #     y1                  = xp.dot(self.Ax, xp.dot(x0, DxT))
        #     y1[bte.xp_vt_l, 0 ] = 0.0
        #     y1[bte.xp_vt_r, -1] = 0.0

        #     x1 = x0 - alpha * xp.dot(Lmat_inv, y1)
        #     print(xp.linalg.norm(x1-x0))
        #     x0 = x1
        #     it+=1
        


    ########################### FOM OP END #####################################################

    def save_checkpoint(self, F, Fr, time, fname):
        bte = self.bte_solver
        xp  = bte.xp_module

        assert xp == cp

        with h5py.File("%s"%(fname), "w") as ff:
            ff.create_dataset("time[T]"      , data = np.array([time]))
            ff.create_dataset("F"            , data = xp.asnumpy(F))
            ff.create_dataset("Fr"           , data = xp.asnumpy(Fr))
            ff.create_dataset("U"            , data = xp.asnumpy(self.U))
            ff.create_dataset("S"            , data = xp.asnumpy(self.S))
            ff.create_dataset("V"            , data = xp.asnumpy(self.V))
            ff.close()
        return

    def restore_checkpoint(self, fname):
        bte = self.bte_solver
        xp  = bte.xp_module

        assert xp == cp

        with h5py.File("%s"%(fname), "r") as ff:
            time           = np.array(ff["time[T]"][()])[0]
            F              = xp.array(ff["F"][()])
            Fr             = xp.array(ff["Fr"][()])
            
            self.U         = xp.array(ff["U"][()])
            self.S         = xp.array(ff["S"][()])
            self.V         = xp.array(ff["V"][()])
            
            ff.close()

        self.init(basis_id="[checkpoint]: t=%.4E"%(time))
        return F, Fr, time

    
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

    rom_rank     = 1000
    restore      = 0
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
    bte_rom.init()
    xxg     = xp.asarray(bte_fom.xp)
    Et      = lambda t: xp.ones_like(bte_fom.xp) * 1e3 #* xp.sin(2 * xp.pi * t)
    tt      = 0
    
    bte_rom.flow_map_fom_op_v(Et(0), dt)

    if (restore==1):
        F, Fr, tt = bte_rom.restore_checkpoint("%s_rom_%02d.h5"%(bte_fom.args.fname, rs_idx))
        idx       = (rs_idx) * cp_freq
        print("checkpoint restored time = %.4E (s) idx = %d"%(tt, idx))
    else:
        # bte_rom.fom_static_solve(Et, v)
        # sys.exit(0)
        #bte_rom.flow_map_fom_op_v(Et(0), dt)
        #Omega    = bte_rom.sample_fom(Et, xp.copy(v).reshape((-1)), 0, 1, dt, rom_rank, load_from_file=False)
        # bte_rom.construct_rom_basis(Et(0), rom_rank, Omega)

        #Lop      = bte_rom.assemble_fom_op(Et(0))
        
        # Lop_h    = xp.asnumpy(Lop)
        # s, u     = np.linalg.eig(Lop_h)
        # uinv     = np.linalg.inv(u)
        
        # z0      = v#Omega[:, 0]
        # Lop_eig = (xp.asarray(u), xp.asarray(s), xp.asarray(uinv))
        # for rr in range(1000, Lop.shape[0], 2000):
        #     bte_rom.plot_memory_term_eig(Et(0), xp.copy(z0).reshape((-1)), dt, rank_k=rr, Lop=Lop, Lop_eig = Lop_eig)

        # bte_rom.plot_memory_term_eig(Et(0), xp.copy(z0).reshape((-1)), dt, rank_k=Lop.shape[0]-500, Lop=Lop, Lop_eig = Lop_eig)
        # bte_rom.plot_memory_term_eig(Et(0), xp.copy(z0).reshape((-1)), dt, rank_k=Lop.shape[0]-100, Lop=Lop, Lop_eig = Lop_eig)
        # bte_rom.plot_memory_term_eig(Et(0), xp.copy(z0).reshape((-1)), dt, rank_k=Lop.shape[0]-1, Lop=Lop, Lop_eig = Lop_eig)

        Lop      = bte_rom.assemble_fom_op(Et(0))
        Lop_svd  = xp.linalg.svd(Lop)
        
        z0       = v
        for rr in range(100, Lop.shape[0], 500):
            bte_rom.plot_memory_term(Et(0), xp.copy(z0).reshape((-1)), dt, rank_k=rr, Lop=Lop, Lop_svd = Lop_svd)
        
        sys.exit(0)
        
        #bte_rom.construct_rom_basis(Et(0), rom_rank, Omega=None)
        F       = xp.copy(v)
        Fr      = bte_rom.encode(F)
        
        idx     = 0
        tt      = 0
    

    print("BTE fom advection step size = %.4E"%(bte_fom.adv_setup_dt))
    print("io freq = %d cycle_freq = %d cp_freq = %d uv_freq = %d" %(io_freq, cycle_freq, cp_freq, uv_freq))
    tT        = args.cycles
    # Lr_inv  = xp.linalg.inv(xp.eye(bte_rom.dof_rom) - dt * bte_rom.assemble_rom_op())

    F0        = xp.copy(F)
    Fr0       = xp.copy(Fr)
    
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

        # if (idx % cp_freq == 0):
        #     print("checkpoint time = %.4E (s)"%(tt))
        #     bte_rom.save_checkpoint(F, Fr, tt, "%s_rom_%02d.h5"%(bte_fom.args.fname, (idx//cp_freq)))

        if (idx % io_freq == 0):
            print("io output time = %.4E (s)"%(tt))
            F1 = bte_rom.decode(Fr)
            plot_solution(bte_fom, bte_rom, F, F1, fname="%s_rom_%04d.png"%(args.fname, idx//io_freq), time = tt, p_F1=True)

        # if(idx > 0 and (idx % uv_freq) == 0):
        #     print("-----------------sampling adjust the rom basis------------------------")
        #     Fr_old = xp.copy(Fr)
        #     v_fom  = bte_rom.decode(Fr)
        #     bte_rom.construct_rom_basis(Et, v_fom, tt, tt + train_cycles, dt, num_samples, eps_x = rom_eps_x, eps_v = rom_eps_v)
        #     bte_rom.init(basis_id="t=%.4E"%(tt))
        #     Fr     = bte_rom.encode(v_fom)
        #     v_fom1 = bte_rom.decode(Fr)
        #     plot_solution(bte_fom, bte_rom, v_fom, v_fom1, fname="%s_rom_%04d_bc.png"%(args.fname, idx//io_freq), time = tt)
        #     print("---------------------------------------------------------------------")


        # x       = F.reshape((-1))
        # rr_u    = xp.linalg.norm(x - bte_rom.U @ (bte_rom.U.T @ x))/xp.linalg.norm(x)
        # rr_v    = xp.linalg.norm(x - bte_rom.V @ (bte_rom.V.T @ x))/xp.linalg.norm(x)

        # print("time = %.4E rr_u = %.4E rr_v = %.4E"%(tt, rr_u, rr_v))

        #Fr      = xp.dot(Lr_inv, Fr) #bte_rom.step_rom_op_split(Ef, Fr, tt, dt, type="BE", atol=bte_fom.args.atol, rtol=bte_fom.args.rtol, verbose=(idx % tio_freq))
        Fr       = bte_rom.step_rom_vx(Fr, dt)
        #Fr       = xp.dot(Lop, F.reshape((-1)))
        


        # rr       = xp.linalg.norm(Fr1 - Fr)/np.linalg.norm(Fr)
        # print(rr)
        # Fr      = Fr1
        #Fr     = xp.dot(bte_rom.flow_map_vx, Fr)
        #Fo     = bte_rom.step_fom_ord(Et, Fo, tt, dt, type="BE", atol=1e-20, rtol=1e-3)
        #Fs     = bte_rom.step_fom_sph(Et, Fs, tt, dt, type="BE", atol=1e-20, rtol=1e-2)
        #F      = bte_rom.step_fom_op_split(Ef, F, tt, dt, verbose = (idx % tio_freq))
        #F      = xp.dot(Lop, F.reshape((-1))).reshape(F.shape)
        F       = (Ibv @ (F.reshape((-1)))).reshape(F.shape)
        

        # y       = xp.copy(F)
        # y[bte_fom.xp_vt_l,  0] = 0
        # y[bte_fom.xp_vt_r, -1] = 0

        #F       = bte_rom.step_fom_vx(Ef, F, tt, dt, atol=bte_fom.args.atol,  rtol=1e-6, gmres_rst=30, gmres_iter=10, pc_type=0, verbose = (idx % tio_freq))
        #F2     = bte_rom.step_fom_vx(Ef, F, tt, psteps * dt, atol=bte_fom.args.atol,  rtol=1e-3, gmres_rst=1000, gmres_iter=1, pc_type=0, verbose = (idx % tio_freq))
        #F      = F1
        
        

        tt  += dt
        idx +=1
    
    














