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


class boltzmann_1d_rom():

    def __init__(self, bte_solver : glow1d_boltzmann):
        self.bte_solver = bte_solver
        self.args       = self.bte_solver.args
        pass
    
    def construct_rom_basis(self, Et, v0, tb, te, dt, n_samples, threshold):
        bte     = self.bte_solver
        tt      = tb
        xp      = bte.xp_module

        steps   = int(np.ceil((te-tb)/dt))
        io_freq = steps//n_samples
        steps   = io_freq * n_samples

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

        spec_sp                = bte.op_spec_sp
        num_p                  = spec_sp._p + 1
        num_sh                 = len(spec_sp._sph_harm_lm)
        rom_lm_modes           = 2
        assert rom_lm_modes  == 2, "ROM assumes two-term approximation"
        fl=list()
           
        for l in range(rom_lm_modes):
            fl.append(v_all_lm[:, l::num_sh, :])
        
        def svd(fl, threshold, xp):
            num_t, num_p, num_x = fl.shape
            Ux, Sx, Vhx = xp.linalg.svd(fl.reshape(num_t * num_p, -1)) # Nt Nr x Nx
            Uv, Sv, Vhv = xp.linalg.svd(np.swapaxes(fl, 0, 1).reshape((num_p, num_t * num_x))) # Nr x Nt Nx

            Vx  = Vhx[Sx > Sx[0] * threshold, :].T
            Uv  = Uv [:, Sv > Sv[0] * threshold]

            return Uv, Vx
        
        Uv  = list()
        Vx  = list()

        for l in range(rom_lm_modes):
            uv, vx = svd(fl[l], threshold=threshold, xp=xp)
            Uv.append(uv)
            # Vx.append(vx)
            #Uv.append(xp.eye(num_p))
            Vx.append(xp.eye(len(bte.xp)))
        
        self.Uv = Uv
        self.Vx = Vx
        self.rom_lm_modes = rom_lm_modes

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
        num_sh  = len(spec_sp._sph_harm_lm)
        
        Mvr     = mm[0::num_sh, 0::num_sh]
        Mv_inv  = mm_inv

        Ps      = bte.op_po2sh
        Po      = bte.op_psh2o

        Ax      = xp.dot(Mvr, bte.op_adv_x)
        Ax      = xp.kron(xp.diag(bte.xp_cos_vt), Ax)
        Gx      = xp.dot(Mv_inv, xp.dot(Ps, xp.dot(Ax, Po)))

        U0, U1   = self.Uv[0], self.Uv[1]
        Vx0, Vx1 = self.Vx[0], self.Vx[1]
        Dx       = bte.Dp

        print("||I - U^t U|| = %.8E "%xp.linalg.norm(xp.dot(U0.T, U0) - xp.eye(U0.shape[1])))
        print("||I - U^t U|| = %.8E "%xp.linalg.norm(xp.dot(U1.T, U1) - xp.eye(U1.shape[1])))
        
        print("||I - V^t V|| = %.8E "%xp.linalg.norm(xp.dot(Vx0.T, Vx0) - xp.eye(Vx0.shape[1])))
        print("||I - V^t V|| = %.8E "%xp.linalg.norm(xp.dot(Vx1.T, Vx1) - xp.eye(Vx1.shape[1])))
       
        self.Gx00    = xp.dot(U0.T, xp.dot(Gx[0::num_sh, 0::num_sh], U0))
        self.Gx01    = xp.dot(U0.T, xp.dot(Gx[0::num_sh, 1::num_sh], U1))
        self.Gx10    = xp.dot(U1.T, xp.dot(Gx[1::num_sh, 0::num_sh], U0))
        self.Gx11    = xp.dot(U1.T, xp.dot(Gx[1::num_sh, 1::num_sh], U1))

        self.Av00    = xp.dot(U0.T, xp.dot(Av[0::num_sh, 0::num_sh], U0))
        self.Av01    = xp.dot(U0.T, xp.dot(Av[0::num_sh, 1::num_sh], U1))
        self.Av10    = xp.dot(U1.T, xp.dot(Av[1::num_sh, 0::num_sh], U0))
        self.Av11    = xp.dot(U1.T, xp.dot(Av[1::num_sh, 1::num_sh], U1))

        self.Cv00    = xp.dot(U0.T, xp.dot(Cop[0::num_sh, 0::num_sh], U0))
        self.Cv01    = xp.dot(U0.T, xp.dot(Cop[0::num_sh, 1::num_sh], U1))
        self.Cv10    = xp.dot(U1.T, xp.dot(Cop[1::num_sh, 0::num_sh], U0))
        self.Cv11    = xp.dot(U1.T, xp.dot(Cop[1::num_sh, 1::num_sh], U1))


        self.Dx00    = 0 * xp.dot(Vx0.T, xp.dot(Dx.T, Vx0))
        self.Dx01    = 0 * xp.dot(Vx1.T, xp.dot(Dx.T, Vx0))
        self.Dx10    = 0 * xp.dot(Vx0.T, xp.dot(Dx.T, Vx1))
        self.Dx11    = 0 * xp.dot(Vx1.T, xp.dot(Dx.T, Vx1))

        self.l0_shape = (U0.shape[1], Vx0.shape[1])
        self.l1_shape = (U1.shape[1], Vx1.shape[1])

        self.len_l0   = U0.shape[1] * Vx0.shape[1]
        self.len_l1   = U1.shape[1] * Vx1.shape[1]
        
        return

    def encode(self, F):
        rom_lm_modes = self.rom_lm_modes
        bte          = self.bte_solver
        xp           = bte.xp_module
        Ps           = bte.op_po2sh

        spec_sp      = bte.op_spec_sp
        num_p        = spec_sp._p + 1
        num_sh       = len(spec_sp._sph_harm_lm)

        # print(Uv[0].shape, VxT[0].shape)
        # print(Uv[1].shape, VxT[1].shape)
        Uv    = self.Uv
        Vx    = self.Vx

        F_lm  = xp.dot(Ps, F)
        Fr_lm = [xp.dot(Uv[i].T, xp.dot(F_lm[i::num_sh, :], Vx[i])) for i in range(rom_lm_modes)]
        Fr    = xp.array(Fr_lm[0])
        
        for l in range(1, rom_lm_modes):
            Fr = xp.append(Fr, Fr_lm[l]) 
        
        #print(Fr.shape)
        #print(Fr_lm[0].shape, Fr_lm[1].shape)

        return Fr
        
    def decode(self, Fr):
        bte     = self.bte_solver
        spec_sp = bte.op_spec_sp
        xp      = bte.xp_module
        Po      = bte.op_psh2o
        
        num_p   = spec_sp._p + 1
        num_sh  = len(spec_sp._sph_harm_lm)
        num_x   = len(bte.xp)
        F       = xp.zeros((num_p * num_sh, num_x))

        Fr0     = Fr[0:self.len_l0].reshape(self.l0_shape)
        Fr1     = Fr[self.len_l0:].reshape(self.l1_shape)
        
        Uv      = self.Uv
        Vx      = self.Vx

        F[0::num_sh, : ] = xp.dot(Uv[0], xp.dot(Fr0, Vx[0].T))
        F[1::num_sh, : ] = xp.dot(Uv[1], xp.dot(Fr1, Vx[1].T))

        return xp.dot(Po, F)

    def rhs(self, Xr, Ef, time, dt, type):
        bte     = self.bte_solver
        param   = bte.param
        xp      = bte.xp_module

        if (type == "BE"):
            Xr0 = Xr[ 0  : self.len_l0].reshape(self.l0_shape)
            Xr1 = Xr[self.len_l0 : ].reshape(self.l1_shape)

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
            
            # R0[:, 0]  = Xr0[:, 0]
            # R1[:, 0]  = Xr1[:, 0]

            # R0[:, -1] = Xr0[:, -1]
            # R1[:, -1] = Xr1[:, -1]


            res = xp.append(R0.reshape((-1)), R1.reshape((-1)))
            return res
        else:
            raise NotImplementedError
    
    """
    def jac(self, Xr, Ef, time, dt, type):
        bte     = self.bte_solver
        xp      = bte.xp_module
        
        if (type == "BE"):
            Xr0 = Xr[0]
            Xr1 = Xr[1]

            Vx0 = self.VxT[0].T
            Vx1 = self.VxT[1].T

            E   = Ef
            E00 = xp.dot(Vx0.T, xp.dot(xp.diag(E), Vx0))
            E01 = xp.dot(Vx1.T, xp.dot(xp.diag(E), Vx0))
            E10 = xp.dot(Vx0.T, xp.dot(xp.diag(E), Vx1))
            E11 = xp.dot(Vx1.T, xp.dot(xp.diag(E), Vx1))

            dR0_Xr0  = Xr0 + dt * ( -xp.dot(self.Gx00, xp.dot(Xr0, self.Dx00)) 
                                    +xp.dot(self.Cv00, Xr0) 
                                    +xp.dot(self.Av00, xp.dot(Xr0, E00)))
            
            dR0_Xr1  =  dt * ( -xp.dot(self.Gx01, xp.dot(Xr1, self.Dx01))
                                +xp.dot(self.Av01, xp.dot(Xr1, E01)))
            
            dR1_Xr0  =  dt * ( -xp.dot(self.Gx10, xp.dot(Xr0, self.Dx10)) 
                                +xp.dot(self.Av10, xp.dot(Xr0, E10)))
            
            dR1_Xr1  = Xr1 + dt * ( - xp.dot(self.Gx11, xp.dot(Xr1, self.Dx11))
                                    + xp.dot(self.Cv11, Xr1) 
                                    + xp.dot(self.Av11, xp.dot(Xr1, E11)))
            
            dR0 = dR0_Xr0 + dR0_Xr1
            dR1 = dR1_Xr0 + dR1_Xr1

            JXr = xp.append(dR0.reshape((-1)), dR1.reshape((-1)))
            return JXr
        else:
           raise NotImplementedError

    """
    def step(self, Et, Fr, time, dt, type, atol=1e-20, rtol=1e-10):
        bte      = self.bte_solver
        xp       = bte.xp_module

        x0       = xp.copy(Fr)

        # F                  = self.decode(Fr)
        # F[bte.xp_vt_l, 0]  = 0
        # F[bte.xp_vt_r, -1] = 0
        # Gr                 = self.encode(F)
        
        # Gr0                = Gr[0:self.len_l0].reshape(self.l0_shape)
        # Gr1                = Gr[self.len_l0: ].reshape(self.l1_shape)

        # Fr0                = x0[0:self.len_l0].reshape(self.l0_shape)
        # Fr1                = x0[self.len_l0: ].reshape(self.l1_shape)


        # Fr0[:, 0]          = Gr0[:, 0]
        # Fr1[:, -1]         = Gr1[:, -1]

        # x0                 = xp.append(Fr0.reshape((-1)), Fr1.reshape((-1)))

        norm_b     = xp.linalg.norm(x0.reshape((-1)))
        Ndof       = self.len_l0 + self.len_l1
        Ef         = Et(time)

        def Ax(x):
            return self.rhs(x, Ef, time, dt, type)

        if xp == cp:
            Amat_op   = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Ax)
            Pmat_op   = None
            gmres_c   = glow1d_utils.gmres_counter(disp=False)
            v, status = cupyx.scipy.sparse.linalg.gmres(Amat_op, x0.reshape((-1)), x0=x0.reshape((-1)), tol=rtol, atol=atol, M=Pmat_op, restart=20, maxiter= 20 * 50, callback=gmres_c)

            norm_res_abs  = xp.linalg.norm(Ax(v) -  x0.reshape((-1)))
            norm_res_rel  = xp.linalg.norm(Ax(v) -  x0.reshape((-1))) / norm_b
          
            if (status !=0) :
                print("time = %.8E T GMRES solver failed! iterations =%d  ||res|| = %.4E ||res||/||b|| = %.4E"%(time, status, norm_res_abs, norm_res_rel))
                sys.exit(-1)
            else:
                return v

        else:
            raise NotImplementedError
        

def plot_solution(bte : glow1d_boltzmann, bte_rom: boltzmann_1d_rom, F0, F1, fprefix):
    xp      = bte.xp_module
    param   = bte.param
    args    = bte.args
    mass_op = bte.op_mass
    temp_op = bte.op_temp
    Ps      = bte.op_po2sh
    Po      = bte.op_psh2o

    F0_lm   = xp.dot(Ps, F0)
    F1_lm   = xp.dot(Ps, F1)
    
    ne0     = xp.dot(mass_op, F0_lm)
    ne1     = xp.dot(mass_op, F1_lm)
    xx      = bte.xp

    if xp == cp:
        ne0 = xp.asnumpy(ne0)
        ne1 = xp.asnumpy(ne1)
        xx  = xp.asnumpy(xx)
    
    plt.figure(figsize=(10, 4), dpi=200)
    plt.subplot(1, 4, 1)

    plt.semilogy(xx, ne0 * param.np0, label=r"FOM (1D-BTE)")
    plt.semilogy(xx, ne1 * param.np0, label=r"ROM")
    plt.xlabel(r"$\hat{x}$")
    plt.ylabel(r"$n_e$ $[m^{-3}]$")
    plt.grid(visible=True)
    plt.legend()

    plt.savefig("%s_rom_%s.png"%(args.fname, fprefix))
    plt.close()
    return

if __name__ == "__main__":
    args = args_parse()

    glow_1d = glow1d_boltzmann(args)
    u, v    = glow_1d.initialize()

    if args.use_gpu==1:
        gpu_device = cp.cuda.Device(args.gpu_device_id)
        gpu_device.use()
    
    dt      = args.cfl
    u, v    = glow_1d.step_init(u, v, dt)
    xp      = glow_1d.xp_module

    bte_rom = boltzmann_1d_rom(glow_1d)
    Et      = lambda t: xp.ones_like(glow_1d.xp) * 1e3 * xp.sin(2 * xp.pi * t)
    
    bte_rom.construct_rom_basis(Et, v, 0, 1.0, dt, 30, 1e-4)
    bte_rom.init()

    F       = xp.copy(v)
    Fr      = bte_rom.encode(F)
    F1      = bte_rom.decode(Fr)

    

    #print((F==F1).all())
    #print(xp.linalg.norm(F1-F)/np.linalg.norm(F))

    tt      = 0
    tT      = 2
    dt      = 1e-3
    idx     = 0
    io_freq = 100
    while tt < tT:
        if (idx % io_freq == 0):
            print("time = %.4E (s)"%(tt))
            F1 = bte_rom.decode(Fr)
            plot_solution(glow_1d, bte_rom, F, F1, fprefix="%04d"%(idx//io_freq))

        Fr   = bte_rom.step(Et, Fr, tt, dt, type="BE", atol=1e-20, rtol=1e-8)
        tt  += dt
        idx +=1
    
   

   






   
