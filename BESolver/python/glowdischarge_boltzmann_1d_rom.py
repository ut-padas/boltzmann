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
from glowdischarge_boltzmann_1d import glow1d_boltzmann, args_parse
import glow1d_utils
import sys
from enum import Enum
import h5py
from profile_t import profile_t

class ROM_TYPE(Enum):
    POD = 0 
    DLR = 1

class TIMER:
    ROM_X = 0
    ROM_V = 1

    FOM_X = 2 
    FOM_V = 3

    LAST  = 4



class boltzmann_1d_rom():

    def __init__(self, bte_solver : glow1d_boltzmann):
        self.bte_solver = bte_solver
        self.args       = self.bte_solver.args
        self.rom_type   = ROM_TYPE.POD
        self.profile    = True
        self.timer      = [profile_t("") for i in range(TIMER.LAST)]


        pass
    
    def construct_rom_basis(self, Et, v0, tb, te, dt, n_samples, threshold):
        bte     = self.bte_solver
        tt      = tb
        xp      = bte.xp_module

        steps   = int(np.ceil((te-tb)/dt))
        io_freq = steps//n_samples
        steps   = io_freq * n_samples

        Ps      = bte.op_po2sh
        Po      = bte.op_psh2o

        v       = xp.copy(v0) 
        v_all   = xp.zeros(tuple([n_samples]) + v0.shape)

        for iter in range(steps):
            if (iter % io_freq == 0):
                print(iter//io_freq, v_all.shape, type(v))
                v_all[int(iter//io_freq), :, :] = v[:, :]

            bte.bs_E            = Et(tt)
            v                   = bte.step_bte_x(v, tt, dt * 0.5)
            v                   = bte.step_bte_v(v, None, tt, dt, ts_type="BE", verbose=1)
            v                   = bte.step_bte_x(v, tt + 0.5 * dt, dt * 0.5)
            tt                 += dt
       
        v_all_lm               = xp.einsum("al,ilx->iax", bte.op_po2sh, v_all)
        xp.save("%s_v_all_lm_tb_%.2E_to_te_%.2E.npy"%(bte.args.fname, tb, te),v_all_lm)
        print("v_all_lm.shape = ", v_all_lm.shape)

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
        
        def svd(fl, threshold, xp):
            num_t, num_p, num_x = fl.shape
            Ux, Sx, Vhx = xp.linalg.svd(fl.reshape(num_t * num_p, -1)) # Nt Nr x Nx
            Uv, Sv, Vhv = xp.linalg.svd(np.swapaxes(fl, 0, 1).reshape((num_p, num_t * num_x))) # Nr x Nt Nx

            kr  = Uv[:, Sv > Sv[0] * threshold].shape[1]
            kx  = Vhx[Sx > Sx[0] * threshold, :].T.shape[1]

            Vx  = Vhx[0:kx, :  ].T
            Uv  = Uv [:  , 0:kx]

            # Vx  = Vhx[Sx > Sx[0] * threshold, :].T
            # Uv  = Uv [:, Sv > Sv[0] * threshold]

            return Uv, Vx, (Sx, Sv)
        
        Uv  = list()
        Vx  = list()

        svalues=list()
        for l in range(rom_modes):
            if (self.rom_type == ROM_TYPE.POD):
                uv, vx , s = svd(fl[l], threshold=threshold[l], xp=xp)
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
            plt.savefig(r"%s_svd.png"%(bte.args.fname))
            plt.close()

        # self.PUv = [xp.dot(self.Uv[l], self.Uv[l].T) for l in range(self.rom_modes)]
        # self.PVx = [xp.dot(self.Vx[l], self.Vx[l].T) for l in range(self.rom_modes)]
        # self.ImPUv = [xp.eye(Uv[l].shape[0]) - xp.dot(self.Uv[l], self.Uv[l].T) for l in range(self.rom_modes)]
        # self.ImPVx = [xp.eye(Vx[l].shape[0]) - xp.dot(self.Vx[l], self.Vx[l].T) for l in range(self.rom_modes)]

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

        self.update_basis(self.Uv, self.Vx)
        return

    def update_basis(self, Uv, Vx):
        bte      = self.bte_solver
        param    = bte.param
        xp       = bte.xp_module
        rom_lm   = self.rom_modes

        spec_sp  = bte.op_spec_sp
        num_p    = spec_sp._p + 1
        num_vt   = len(bte.xp_vt)
        num_sh   = len(spec_sp._sph_harm_lm)

        Ps       = bte.op_po2sh
        Po       = bte.op_psh2o
        
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
        self.Cvr     = [[None for j in range(rom_lm)] for i in range(rom_lm)]
        self.Avr     = [[None for j in range(rom_lm)] for i in range(rom_lm)]
        self.Dxr     = [[None for j in range(rom_lm)] for i in range(rom_lm)]

        ## attemp to do the spatial advection on the projected rom operators- this will require
        ## writing a loop on ordinates, less performative than doing rom-fom-adv-rom projection

        # self.iQPor   = [xp.dot(invQPo[i::num_sh], self.Uv[i]) for i in range(rom_lm)]
        # self.PsQ     = [xp.dot(self.Uv[i].T, PsQ[i::num_sh])  for i in range(rom_lm)]
        # self.Vx_bdy  = [None for i in range(rom_lm)]

        # for i in range(rom_lm):
        #     vx        = xp.copy(self.Vx[i])
        #     vx[0,  :] = 0
        #     vx[-1, :] = 0

        #     self.Vx_bdy[i] = vx
        
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

        self.vec_shape = [(self.Uv[l].shape[1], self.Vx[l].shape[1]) for l in range(self.rom_modes)]
        self.vec_len   = [self.Uv[l].shape[1] * self.Vx[l].shape[1] for l in range(self.rom_modes)] 
        self.vec_offset= [0 for l in range(self.rom_modes)]
        
        for l in range(1, self.rom_modes):
            self.vec_offset[l] = self.vec_offset[l-1] + self.vec_len[l-1]

        self.vec_idx   = [xp.arange(self.vec_offset[l], self.vec_offset[l] + self.vec_len[l]) for l in range(self.rom_modes)]
        self.dof_rom   = np.sum(np.array(self.vec_len))
        
        print("basis updated")
        # print("basis shape", self.vec_shape)
        # print("basis ", self.vec_len)
        # print("basis shape", self.vec_offset)
        return

    def get_rom_lm(self, Fr, l, m=0):
        return Fr.reshape((-1))[self.vec_idx[l]].reshape(self.vec_shape[l])

    def rom_vec(self):
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

    def rhs_fom_sph(self, Fs, Ef, time, dt, type):
        bte     = self.bte_solver
        spec_sp = bte.op_spec_sp
        param   = bte.param
        xp      = bte.xp_module
        Ps      = bte.op_po2sh
        Po      = bte.op_psh2o

        num_p   = spec_sp._p + 1
        num_sh  = len(spec_sp._sph_harm_lm)
        num_x   = len(bte.xp)

        if (type == "BE"):
            Fs               = Fs.reshape((num_p * num_sh, num_x))
            rhs              = Fs + dt * xp.dot(self.Gx, xp.dot(Fs, self.Dx.T)) - dt * param.tau * xp.dot(self.Cop, Fs) - dt * param.tau * Ef * xp.dot(self.Av, Fs)  
            Flo              = xp.dot(Po, Fs[: , 0])
            Fro              = xp.dot(Po, Fs[:, -1])
            
            Flo[bte.xp_vt_l] = 0.0
            Fro[bte.xp_vt_r] = 0.0

            rhs[:,  0]       = Fs[:, 0 ] - xp.dot(Ps, Flo)
            rhs[:, -1]       = Fs[:, -1] - xp.dot(Ps, Fro) 
            return rhs.reshape((-1))
    
    def rhs_fom_ord(self, Fo, Ef, time, dt, type):
        bte     = self.bte_solver
        spec_sp = bte.op_spec_sp
        param   = bte.param
        xp      = bte.xp_module
        Ps      = bte.op_po2sh
        Po      = bte.op_psh2o

        num_p   = spec_sp._p + 1
        num_vt  = len(bte.xp_vt)
        num_sh  = len(spec_sp._sph_harm_lm)
        num_x   = len(bte.xp)

        if (type == "BE"):
            Fo                         = Fo.reshape((num_p * num_vt, num_x))
            rhs                        = Fo + dt * xp.dot(self.Ax, xp.dot(Fo, self.Dx.T)) #- dt * param.tau * xp.dot(self.Cop, Fs) - dt * param.tau * Ef * xp.dot(self.Av, Fs)  
            
            rhs[bte.xp_vt_l,  0]       = Fo[bte.xp_vt_l, 0 ] 
            rhs[bte.xp_vt_r, -1]       = Fo[bte.xp_vt_r, -1] 
            return rhs.reshape((-1))

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

            # for i in range(rom_lm):
            #     Xri    = self.get_rom_lm(Xr, i)
            #     EVi    = xp.dot(E, self.Vx[i])
            #     #!!! Note: when there is Coulomb collisions, Collision operator is not block diagonal in the l-modes
            #     rhs[i] = Xri -dt * param.tau * (xp.dot(self.Cvr[i][i], Xri))
            #     for j in range(rom_lm):
            #         Xrj     = self.get_rom_lm(Xr, j)
            #         rhs[i] += -dt *param.tau * xp.dot(self.Avr[i][j], xp.dot(Xrj, xp.dot(self.Vx[j].T, EVi)))
            # res = self.append_vec(rhs)

            Xr0    = self.get_rom_lm(Xr, 0)
            Xr1    = self.get_rom_lm(Xr, 1)
            EV0    = xp.dot(E, self.Vx[0])
            EV1    = xp.dot(E, self.Vx[1])

            rhs[0] = Xr0 -dt * param.tau * (xp.dot(self.Cvr[0][0], Xr0)) - dt * param.tau * (
                                            xp.dot(self.Avr[0][0], xp.dot(Xr0, xp.dot(self.Vx[0].T, EV0))) + 
                                            xp.dot(self.Avr[0][1], xp.dot(Xr1, xp.dot(self.Vx[1].T, EV0))) 
                                            )

            rhs[1] = Xr1 -dt * param.tau * (xp.dot(self.Cvr[1][1], Xr1)) - dt * param.tau * (
                                            xp.dot(self.Avr[1][0], xp.dot(Xr0, xp.dot(self.Vx[0].T, EV1))) + 
                                            xp.dot(self.Avr[1][1], xp.dot(Xr1, xp.dot(self.Vx[1].T, EV1)))  
                                            )

            res = self.append_vec(rhs)
            return res

    def rhs_rom_x(sefl, Xr, Ef, time, dt, type):
        pass
    
    def rhs_rom_vx(self, Xr, Ef, time, dt, type):
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


        if (type == "BE"):
            Xr  = Xr.reshape((-1))
            Xr0 = self.get_rom_lm(Xr, 0)
            Xr1 = self.get_rom_lm(Xr, 1)

            Uv0 = self.Uv[0]
            Uv1 = self.Uv[1]

            Vx0 = self.Vx[0]
            Vx1 = self.Vx[1]

            E   = Ef
            E00 = xp.dot(Vx0.T, xp.dot(xp.diag(E), Vx0))
            E01 = xp.dot(Vx1.T, xp.dot(xp.diag(E), Vx0))
            E10 = xp.dot(Vx0.T, xp.dot(xp.diag(E), Vx1))
            E11 = xp.dot(Vx1.T, xp.dot(xp.diag(E), Vx1))

            R0  = Xr0 + dt * (xp.dot(self.Gx00, xp.dot(Xr0, self.Dx00)) + xp.dot(self.Gx01, xp.dot(Xr1, self.Dx01))
                              -param.tau * xp.dot(self.Cv00, Xr0) 
                              -param.tau * xp.dot(self.Av00, xp.dot(Xr0, E00)) - param.tau * xp.dot(self.Av01, xp.dot(Xr1, E01)))
            
            R1  = Xr1 + dt * (xp.dot(self.Gx10, xp.dot(Xr0, self.Dx10)) + xp.dot(self.Gx11, xp.dot(Xr1, self.Dx11))
                              -param.tau * xp.dot(self.Cv11, Xr1) 
                              -param.tau * xp.dot(self.Av10, xp.dot(Xr0, E10)) - param.tau * xp.dot(self.Av11, xp.dot(Xr1, E11)))
            

            Fl               = xp.zeros((num_p * num_sh))
            Fr               = xp.zeros((num_p * num_sh))

            Fl[0::num_sh]    = xp.dot(Uv0, Xr0[:, 0])
            Fl[1::num_sh]    = xp.dot(Uv1, Xr1[:, 0])

            Fr[0::num_sh]    = xp.dot(Uv0, Xr0[:, -1])
            Fr[1::num_sh]    = xp.dot(Uv1, Xr1[:, -1])

            Flo              = xp.dot(Po, Fl)
            Fro              = xp.dot(Po, Fr)

            Flo[bte.xp_vt_l] = 0.0
            Fro[bte.xp_vt_r] = 0.0

            Fl               = xp.dot(Ps, Flo)
            Fr               = xp.dot(Ps, Fro)
            
            R0[:, 0]         = Xr0[:, 0]  - xp.dot(Uv0.T, Fl[0::num_sh])
            R1[:, 0]         = Xr1[:, 0]  - xp.dot(Uv1.T, Fl[1::num_sh])

            R0[:, -1]        = Xr0[:, -1] - xp.dot(Uv0.T, Fr[0::num_sh])
            R1[:, -1]        = Xr1[:, -1] - xp.dot(Uv1.T, Fr[1::num_sh])

            res = xp.append(R0.reshape((-1)), R1.reshape((-1)))
            return res
        else:
            raise NotImplementedError
    
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
            Fo = self.decode(Fr)
            Fo = bte.step_bte_x(Fo, time, dt * 0.5)

            Fr = self.encode(Fo)
            return Fr.reshape((-1))
            
        else:
            raise NotImplementedError

    def step_rom_op_split(self, Ef, Fr, time, dt, type, atol=1e-20, rtol=1e-10, verbose=1):
        bte            = self.bte_solver
        xp             = bte.xp_module
        gmres_rst      = 8
        gmres_iter     = 10 
        
        #Frh            = self.step_rom_x(Ef, Fr, time, dt, "FOM-ADV-FULL-STEP", atol, rtol, verbose)

        #Frh           = self.step_rom_v(Et, Fr, time,            dt, "BE", atol, rtol, verbose)
        #Frh           = self.step_rom_v_dlr(Ef, Fr, time,            dt, "BE", atol, rtol, verbose)

        if (self.profile):
            cp.cuda.runtime.deviceSynchronize()
            self.timer[TIMER.ROM_X].reset()
            self.timer[TIMER.ROM_X].start()

        Frh            = self.step_rom_x(Ef,  Fr, time,            dt, "HALF-STEP", atol, rtol, verbose)

        if (self.profile):
            cp.cuda.runtime.deviceSynchronize()
            self.timer[TIMER.ROM_X].stop()

        if (self.profile):
            cp.cuda.runtime.deviceSynchronize()
            self.timer[TIMER.ROM_V].reset()
            self.timer[TIMER.ROM_V].start()

        Frh            = self.step_rom_v(Ef, Frh, time,            dt, "BE", atol, rtol, gmres_rst, gmres_iter, verbose)
        
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

        if (self.profile and verbose==1):
            t1 = self.timer[TIMER.ROM_X].seconds
            t2 = self.timer[TIMER.ROM_V].seconds
            print("ROM x-solve runtime = %.4E v-solve runtime =%.4E total = %.4E"%(t1, t2, t1+t2))
            print("")
        return Frh

    def step_rom_vx(self, Ef, Fr, time, dt, type, atol=1e-20, rtol=1e-10, verbose=1):
        bte          = self.bte_solver
        xp           = bte.xp_module

        rhs          = xp.copy(Fr)
        rhs0         = self.get_rom_lm(rhs, 0)
        rhs1         = self.get_rom_lm(rhs, 1)

        rhs0[:,  0]  = 0 
        rhs0[:, -1]  = 0 
        rhs1[:,  0]  = 0 
        rhs1[:, -1]  = 0 

        x0           = Fr
        rhs          = xp.append(rhs0.reshape((-1)), rhs1.reshape((-1)))
        norm_b       = xp.linalg.norm(rhs.reshape((-1)))
        Ndof         = rhs.shape[0]
        
        def Ax(x):
            return self.rhs(x, Ef, time, dt, type)
        
        def Px(x):
            return x

        if xp == cp:
            Amat_op       = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Ax)
            Pmat_op       = None # cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Px)
            gmres_c       = glow1d_utils.gmres_counter(disp=True)
            gmres_rst     = 30
            gmres_iter    = 60
            v, status     = cupyx.scipy.sparse.linalg.gmres(Amat_op, rhs.reshape((-1)), x0=x0.reshape((-1)), tol=rtol, atol=atol, M=Pmat_op, restart=gmres_rst, maxiter= gmres_rst * gmres_iter, callback=gmres_c)

            norm_res_abs  = xp.linalg.norm(Ax(v) -  rhs.reshape((-1)))
            norm_res_rel  = xp.linalg.norm(Ax(v) -  rhs.reshape((-1))) / norm_b
          
            if (status !=0) :
                print("time = %.8E T GMRES solver failed! iterations =%d  ||res|| = %.4E ||res||/||b|| = %.4E"%(time, status, norm_res_abs, norm_res_rel))
                sys.exit(-1)

            else:
                if (verbose==1):
                    print("[ROM-SPH]  time = %.8E T GMRES  iterations =%d  ||res|| = %.4E ||res||/||b|| = %.4E"%(time, gmres_c.niter * gmres_rst, norm_res_abs, norm_res_rel))
                return v.reshape((Fr.shape))

        else:
            raise NotImplementedError

    def step_fom_sph(self, Ef, Fs, time, dt, type, atol=1e-20, rtol=1e-10, verbose=1):
        bte        = self.bte_solver
        xp         = bte.xp_module
        Ps         = bte.op_po2sh
        Po         = bte.op_psh2o

        rhs        = xp.copy(Fs)
        rhs[:,  0] = 0
        rhs[:, -1] = 0

        norm_b     = xp.linalg.norm(rhs.reshape((-1)))
        Ndof       = rhs.reshape((-1)).shape[0]
        
        x0         = Fs#xp.dot(Ps, self.step_fom(Et, xp.dot(Po, Fs), time, dt))
        #return Fs0

        def Ax(x):
            return self.rhs_fom_sph(x, Ef, time, dt, type)

        def Px(x):
            return x#self.step_fom(Et, x, time, dt).reshape((-1))

        if xp == cp:
            Amat_op    = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Ax)
            Pmat_op    = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Px)
            gmres_c    = glow1d_utils.gmres_counter(disp=True)
            gmres_rst  = 40
            gmres_iter = 60
            v, status  = cupyx.scipy.sparse.linalg.gmres(Amat_op, rhs.reshape((-1)), x0=x0.reshape((-1)), tol=rtol, atol=atol, M=Pmat_op, restart=gmres_rst, maxiter= gmres_rst * gmres_iter, callback=gmres_c)

            norm_res_abs  = xp.linalg.norm(Ax(v) -  rhs.reshape((-1)))
            norm_res_rel  = xp.linalg.norm(Ax(v) -  rhs.reshape((-1))) / norm_b
          
            if (status !=0) :
                print("time = %.8E T GMRES solver failed! iterations =%d  ||res|| = %.4E ||res||/||b|| = %.4E"%(time, status, norm_res_abs, norm_res_rel))
                sys.exit(-1)

            else:

                if (verbose == 1):
                    print("[FOM-SPH]  time = %.8E T GMRES  iterations =%d  ||res|| = %.4E ||res||/||b|| = %.4E"%(time, gmres_c.niter * gmres_rst, norm_res_abs, norm_res_rel))
                
                return v.reshape(Fs.shape)

        else:
            raise NotImplementedError

    def step_fom_ord(self, Ef, Fo, time, dt, type, atol=1e-20, rtol=1e-10, verbose=1):
        bte        = self.bte_solver
        xp         = bte.xp_module
        Ps         = bte.op_po2sh
        Po         = bte.op_psh2o

        rhs                  = xp.copy(Fo) #Fo + 0.5 * dt * xp.dot(self.Ax, xp.dot(Fo, self.Dx.T))
        rhs[bte.xp_vt_l,  0] = 0
        rhs[bte.xp_vt_r, -1] = 0

        norm_b     = xp.linalg.norm(rhs.reshape((-1)))
        Ndof       = rhs.reshape((-1)).shape[0]
        
        x0         = self.step_fom(Et, Fo, time, dt).reshape((-1))

        def Ax(x):
            return self.rhs_fom_ord(x, Ef, time, dt, type)

        def Px(x):
            return x#self.step_fom(Et, x, time, dt).reshape((-1))

        if xp == cp:
            Amat_op   = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Ax)
            Pmat_op   = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Px)
            gmres_c   = glow1d_utils.gmres_counter(disp=True)
            gmres_rst = 80
            gmres_iter= 30
            v, status = cupyx.scipy.sparse.linalg.gmres(Amat_op, rhs.reshape((-1)), x0=x0.reshape((-1)), tol=rtol, atol=atol, M=Pmat_op, restart=gmres_rst, maxiter= gmres_rst * gmres_iter, callback=gmres_c)

            norm_res_abs  = xp.linalg.norm(Ax(v) -  rhs.reshape((-1)))
            norm_res_rel  = xp.linalg.norm(Ax(v) -  rhs.reshape((-1))) / norm_b
          
            if (status !=0) :
                print("time = %.8E T GMRES solver failed! iterations =%d  ||res|| = %.4E ||res||/||b|| = %.4E"%(time, status, norm_res_abs, norm_res_rel))
                #return v.reshape(Fs.shape)
                sys.exit(-1)

            else:

                if (verbose == 1):
                    print("[FOM-ORD]  time = %.8E T GMRES  iterations =%d  ||res|| = %.4E ||res||/||b|| = %.4E"%(time, gmres_c.niter * gmres_rst, norm_res_abs, norm_res_rel))

                return v.reshape(Fo.shape)

        else:
            raise NotImplementedError

    def step_fom(self, Ef, F, time, dt, verbose=1):
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

        if (self.profile and verbose==1):
            t1 = self.timer[TIMER.FOM_X].seconds
            t2 = self.timer[TIMER.FOM_V].seconds
            print("FOM x-solve runtime = %.4E v-solve runtime =%.4E total = %.4E"%(t1, t2, t1+t2))
            print("")

        return v

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



def plot_solution(bte : glow1d_boltzmann, bte_rom: boltzmann_1d_rom, F0, F1, fprefix, time):
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
    PUv     = [xp.dot(bte_rom.Uv[l], bte_rom.Uv[l].T) for l in range(rom_lm)]
    PVx     = [xp.dot(bte_rom.Vx[l], bte_rom.Vx[l].T) for l in range(rom_lm)]

    G0_lm            = xp.zeros((num_p* num_sh, num_x))
    for l in range(rom_lm):
        G0_lm[l::num_sh] = xp.dot(PUv[l], xp.dot(F0_lm[0::num_sh], PVx[l]))

    F0_lm_n = bte.bte_eedf_normalization(F0_lm)
    F1_lm_n = bte.bte_eedf_normalization(F1_lm)
    G0_lm_n = bte.bte_eedf_normalization(G0_lm)
    
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

        
    
    plt.figure(figsize=(16, 4), dpi=200)
    plt.subplot(1, 4, 1)
    plt.semilogy(xx, ne0 * param.np0, label=r"FOM")
    plt.semilogy(xx, ne1 * param.np0, label=r"ROM")
    #plt.semilogy(xx, ne2 * param.np0, label=r"$P_{U}$ FOM $P_{V}$")
    plt.xlabel(r"$\hat{x}$")
    plt.ylabel(r"$n_e$ $[m^{-3}]$")
    plt.grid(visible=True)
    plt.legend()
    
    plt.subplot(1, 4, 2)
    plt.plot(xx, Te0, label=r"FOM")
    plt.plot(xx, Te1, label=r"ROM")
    #plt.plot(xx, Te2, label=r"$P_{U}$ FOM $P_{V}$")
    plt.xlabel(r"$\hat{x}$")
    plt.ylabel(r"$T_e$ $[eV]$")
    plt.grid(visible=True)
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.semilogy(xx, g0_0, label=r"FOM")
    plt.semilogy(xx, g0_1, label=r"ROM")
    #plt.semilogy(xx, g0_2, label=r"$P_{U}$ FOM $P_{V}$")
    plt.xlabel(r"$\hat{x}$")
    plt.ylabel(r"rate coefficient $[m^{3}s^{-1}]$")
    plt.title(r"momentum transfer")
    plt.grid(visible=True)
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.semilogy(xx, g2_0, label=r"FOM")
    plt.semilogy(xx, g2_1, label=r"ROM")
    #plt.semilogy(xx, g2_2, label=r"$P_{U}$ FOM $P_{V}$")
    plt.xlabel(r"$\hat{x}$")
    plt.ylabel(r"rate coefficient $[m^{3}s^{-1}]$")
    plt.title(r"ionization")
    plt.grid(visible=True)
    plt.legend()

    plt.suptitle(r"time = %.4E [T]"%(time))
    plt.tight_layout()

    plt.savefig("%s_rom_%s.png"%(args.fname, fprefix))
    plt.close()
    return

if __name__ == "__main__":
    args = args_parse()

    bte_fom = glow1d_boltzmann(args)
    u, v    = bte_fom.initialize()

    if args.use_gpu==1:
        gpu_device = cp.cuda.Device(args.gpu_device_id)
        gpu_device.use()
    
    dt      = args.cfl
    u, v    = bte_fom.step_init(u, v, dt)
    xp      = bte_fom.xp_module

    spec_sp = bte_fom.op_spec_sp
    num_p   = spec_sp._p + 1
    num_sh  = len(spec_sp._sph_harm_lm)

    bte_rom = boltzmann_1d_rom(bte_fom)
    xxg     = xp.asarray(bte_fom.xp)
    #Et     = lambda t: xp.ones_like(bte_fom.xp) * 1e3 * xp.sin(2 * xp.pi * t)
    Et      = lambda t: (xxg**3) * 1e4 * xp.sin(2 * xp.pi * t)
    
    bte_rom.construct_rom_basis(Et, v, 0, 1, dt, 30, [1e-4 for l in range(num_sh)])
    bte_rom.init()

    with open("%s_rom_size.txt"%(bte_fom.args.fname), 'w') as f:
        f.write(str(bte_rom.vec_shape)+"\n")
    f.close()

    F       = xp.copy(v)
    Fo      = xp.copy(v)
    Fs      = xp.dot(bte_fom.op_po2sh, Fo)
    Fr      = bte_rom.encode(Fo).reshape((-1))
    F1      = bte_rom.decode(Fr)
    
    args       = bte_fom.args

    io_cycle   = args.io_cycle_freq
    cp_cycle   = args.cp_cycle_freq

    io_freq    = int(np.round(io_cycle/dt))
    cp_freq    = int(np.round(cp_cycle/dt))
    cycle_freq = int(np.round(1/dt))
    tio_freq   = 20
    uv_freq    = int(np.round(30/dt))

      
    print("io freq = %d cycle_freq = %d cp_freq = %d uv_freq = %d" %(io_freq, cycle_freq, cp_freq, uv_freq))
    tT        = args.cycles
    idx       = 0
    flm_error = [list(), list()]
    ts        = list()

    tt        = 0
    restore   = 0
    rs_idx    = 0
    if (restore==1):
        F, Fr, tt = bte_rom.restore_checkpoint("%s_rom_%02d.h5"%(bte_fom.args.fname, rs_idx))
        print("checkpoint restored time = %.4E (s)"%(tt))

    F0  = xp.copy(F)
    Fr0 = xp.copy(Fr)

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
            bte_rom.save_checkpoint(F, Fr, tt, "%s_rom_%02d.h5"%(bte_fom.args.fname, (idx//cp_freq) % 2))

        if (idx % io_freq == 0):
            print("io output time = %.4E (s)"%(tt))
            F1 = bte_rom.decode(Fr)
            #F1  = xp.dot(bte_fom.op_psh2o, Fs)
            #F1  = Fo
            plot_solution(bte_fom, bte_rom, F, F1, fprefix="%04d"%(idx//io_freq), time = tt)

        if(idx > 0 and (idx % uv_freq) == 0):
            print("-----------------sampling adjust the rom basis------------------------")
            v_fom = bte_rom.decode(Fr)
            bte_rom.construct_rom_basis(Et, v_fom, tt, tt + 1, dt, 30, [1e-4 for l in range(num_sh)])
            bte_rom.init()
            Fr    = bte_rom.encode(v_fom)
            print("---------------------------------------------------------------------") 

        Fr   = bte_rom.step_rom_op_split(Ef, Fr, tt, dt, type="BE", atol=bte_fom.args.atol, rtol=bte_fom.args.rtol, verbose=(idx % tio_freq))
        #Fo   = bte_rom.step_fom_ord(Et, Fo, tt, dt, type="BE", atol=1e-20, rtol=1e-3)
        #Fs   = bte_rom.step_fom_sph(Et, Fs, tt, dt, type="BE", atol=1e-20, rtol=1e-2)
        F    = bte_rom.step_fom(Ef, F, tt, dt, verbose = (idx % tio_freq))
        
        tt  += dt
        idx +=1
    
    
    
   
   

   






   
