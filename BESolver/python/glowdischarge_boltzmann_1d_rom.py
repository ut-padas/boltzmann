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
        rom_modes              = 2
        
        assert rom_modes == 2, "ROM assumes two-term approximation"
        fl=list()
           
        for l in range(rom_modes):
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

        for l in range(rom_modes):
            uv, vx = svd(fl[l], threshold=threshold, xp=xp)
            Uv.append(uv)
            Vx.append(vx)
            #Uv.append(xp.eye(num_p))
            #Vx.append(xp.eye(len(bte.xp)))
        
        self.Uv = Uv
        self.Vx = Vx
        self.rom_modes = rom_modes

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

        U0, U1   = self.Uv[0], self.Uv[1]
        Vx0, Vx1 = self.Vx[0], self.Vx[1]
        Dx       = bte.Dp

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


        self.Dx00    = xp.dot(Vx0.T, xp.dot(Dx.T, Vx0))
        self.Dx01    = xp.dot(Vx1.T, xp.dot(Dx.T, Vx0))
        self.Dx10    = xp.dot(Vx0.T, xp.dot(Dx.T, Vx1))
        self.Dx11    = xp.dot(Vx1.T, xp.dot(Dx.T, Vx1))

        # self.xp_vt   = xp.asarray(bte.xp_vt)
        # self.xp_vt_qw= xp.asarray(bte.xp_vt_qw)

        self.vec_shape = [(self.Uv[l].shape[1], self.Vx[l].shape[1]) for l in range(self.rom_modes)]
        self.vec_len   = [self.Uv[l].shape[1] * self.Vx[l].shape[1] for l in range(self.rom_modes)] 
        self.vec_offset= [0 for l in range(self.rom_modes)]
        
        for l in range(1, self.rom_modes):
            self.vec_offset[l] = self.vec_offset[l-1] + self.vec_len[l-1]

        self.vec_idx   = [xp.arange(self.vec_offset[l], self.vec_offset[l] + self.vec_len[l]) for l in range(self.rom_modes)]
        print(self.vec_shape)
        print(self.vec_len)
        print(self.vec_offset)
        return

    def get_rom_lm(self, Fr, l, m=0):
        return Fr.reshape((-1))[self.vec_idx[l]].reshape(self.vec_shape[l])

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

            R0  = Xr0 + dt * param.tau * (-xp.dot(self.Cv00, Xr0) - xp.dot(self.Av00, xp.dot(Xr0, E00)) - xp.dot(self.Av01, xp.dot(Xr1, E01)))
            R1  = Xr1 + dt * param.tau * (-xp.dot(self.Cv11, Xr1) - xp.dot(self.Av10, xp.dot(Xr0, E10)) - xp.dot(self.Av11, xp.dot(Xr1, E11)))

            res = xp.append(R0.reshape((-1)), R1.reshape((-1)))
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

    def step_rom_v(self, Et, Fr, time, dt, type, atol=1e-20, rtol=1e-8, verbose=1):
        bte        = self.bte_solver
        xp         = bte.xp_module

        rhs        = xp.copy(Fr).reshape((-1))
        norm_b     = xp.linalg.norm(rhs)
        Ndof       = rhs.shape[0]
        Ef         = Et(time)
        x0         = Fr.reshape((-1))
        
        def Ax(x):
            return self.rhs_rom_v(x, Ef, time, dt, type)

        def Px(x):
            return x

        if xp == cp:
            Amat_op       = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Ax)
            Pmat_op       = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Px)
            gmres_c       = glow1d_utils.gmres_counter(disp=True)
            gmres_rst     = 10
            gmres_iter    = 10
            v, status     = cupyx.scipy.sparse.linalg.gmres(Amat_op, rhs.reshape((-1)), x0=x0.reshape((-1)), tol=rtol, atol=atol, M=Pmat_op, restart=gmres_rst, maxiter= gmres_rst * gmres_iter, callback=gmres_c)

            norm_res_abs  = xp.linalg.norm(Ax(v) -  rhs.reshape((-1)))
            norm_res_rel  = xp.linalg.norm(Ax(v) -  rhs.reshape((-1))) / norm_b
          
            if (status !=0) :
                print("time = %.8E T GMRES solver failed! iterations =%d  ||res|| = %.4E ||res||/||b|| = %.4E"%(time, status, norm_res_abs, norm_res_rel))
                sys.exit(-1)

            else:
                if (verbose == 1):
                    print("[ROM v-space]  time = %.8E T GMRES  iterations =%d  ||res|| = %.4E ||res||/||b|| = %.4E"%(time, gmres_c.niter * gmres_rst, norm_res_abs, norm_res_rel))

                return v.reshape(Fr.shape)
        else:
            raise NotImplementedError

    def step_rom_x(self, Et, Fr, time, dt, type, atol=1e-20, rtol=1e-8, verbose=1):
        bte        = self.bte_solver
        xp         = bte.xp_module

        if (type == "FOM-ADV-FULL-STEP"):
            """
            performs advection in the full order-space
            """
            Fo = self.decode(Fr)
            Fo = bte.step_bte_x(Fo, time, dt * 0.5)
            Fo = bte.step_bte_x(Fo, time + 0.5 * dt, dt * 0.5)
            Fr = self.encode(Fo)
            return Fr.reshape((-1))
        elif(type == "FOM-ADV-HALF-STEP"):
            Fo = self.decode(Fr)
            Fo = bte.step_bte_x(Fo, time, dt * 0.5)
            Fr = self.encode(Fo)
            return Fr.reshape((-1))
        else:
            raise NotImplementedError

    def step_rom_op_split(self, Et, Fr, time, dt, type, atol=1e-20, rtol=1e-10, verbose=1):
        bte          = self.bte_solver
        xp           = bte.xp_module

        #Frh          = self.step_rom_v(Et, Fr, time,            dt, "BE", atol, rtol, verbose)

        Frh          = self.step_rom_x(Et,  Fr, time,            dt, "FOM-ADV-HALF-STEP", atol, rtol, verbose)
        Frh          = self.step_rom_v(Et, Frh, time,            dt, "BE", atol, rtol, verbose)
        Frh          = self.step_rom_x(Et, Frh, time + 0.5 * dt, dt, "FOM-ADV-HALF-STEP", atol, rtol, verbose)


        return Frh.reshape((-1))

    def step_rom_vx(self, Et, Fr, time, dt, type, atol=1e-20, rtol=1e-10, verbose=1):
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
        Ef           = Et(time)

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

    def step_fom_sph(self, Et, Fs, time, dt, type, atol=1e-20, rtol=1e-10, verbose=1):
        bte        = self.bte_solver
        xp         = bte.xp_module
        Ps         = bte.op_po2sh
        Po         = bte.op_psh2o

        rhs        = xp.copy(Fs)
        rhs[:,  0] = 0
        rhs[:, -1] = 0

        norm_b     = xp.linalg.norm(rhs.reshape((-1)))
        Ndof       = rhs.reshape((-1)).shape[0]
        Ef         = Et(time)

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

    def step_fom_ord(self, Et, Fo, time, dt, type, atol=1e-20, rtol=1e-10, verbose=1):
        bte        = self.bte_solver
        xp         = bte.xp_module
        Ps         = bte.op_po2sh
        Po         = bte.op_psh2o

        rhs                  = xp.copy(Fo) #Fo + 0.5 * dt * xp.dot(self.Ax, xp.dot(Fo, self.Dx.T))
        rhs[bte.xp_vt_l,  0] = 0
        rhs[bte.xp_vt_r, -1] = 0

        norm_b     = xp.linalg.norm(rhs.reshape((-1)))
        Ndof       = rhs.reshape((-1)).shape[0]
        Ef         = Et(time)

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

    def step_fom(self, Et, F, time, dt):
        """
        full order model timestep
        """
        bte                 = self.bte_solver
        tt                  = time
        bte.bs_E            = Et(tt)

        #v                   = bte.step_bte_v(F, None, tt, dt, ts_type="BE", verbose=1)

        # v                   = bte.step_bte_x(F, tt, dt * 0.5)
        # v                   = bte.step_bte_x(v, tt + 0.5 * dt, dt * 0.5)

        v                   = bte.step_bte_x(F, tt, dt * 0.5)
        v                   = bte.step_bte_v(v, None, tt, dt, ts_type="BE", verbose=1)
        v                   = bte.step_bte_x(v, tt + 0.5 * dt, dt * 0.5)
        return v


def plot_solution(bte : glow1d_boltzmann, bte_rom: boltzmann_1d_rom, F0, F1, fprefix, time):
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
    plt.suptitle(r"time/T = %.4E"%(time))

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
    
    bte_rom.construct_rom_basis(Et, v, 0, 1, dt, 30, 1e-4)
    bte_rom.init()

    F       = xp.copy(v)
    Fo      = xp.copy(v)
    Fs      = xp.dot(glow_1d.op_po2sh, Fo)
    Fr      = bte_rom.encode(Fo).reshape((-1))
    F1      = bte_rom.decode(Fr)

    tt      = 0
    tT      = 2
    idx     = 0
    io_freq = 100
    while tt < tT:
        if (idx % io_freq == 0):
            print("time = %.4E (s)"%(tt))
            F1 = bte_rom.decode(Fr)
            #F1  = xp.dot(glow_1d.op_psh2o, Fs)
            #F1  = Fo
            plot_solution(glow_1d, bte_rom, F, F1, fprefix="%04d"%(idx//io_freq), time = tt)

        Fr   = bte_rom.step_rom_op_split(Et, Fr, tt, dt, type="BE", atol=1e-20, rtol=1e-3)
        #Fo   = bte_rom.step_fom_ord(Et, Fo, tt, dt, type="BE", atol=1e-20, rtol=1e-3)
        #Fs   = bte_rom.step_fom_sph(Et, Fs, tt, dt, type="BE", atol=1e-20, rtol=1e-2)
        F    = bte_rom.step_fom(Et, F, tt, dt)
        tt  += dt
        idx +=1
    
   

   






   
