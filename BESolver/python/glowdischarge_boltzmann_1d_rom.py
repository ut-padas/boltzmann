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


class boltzmann_1d_rom():
    
    def __init__(self, bte_solver : glow1d_boltzmann):
        self.bte_solver = bte_solver
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
                v_all[iter//io_freq, : ] = v

            bte.bs_E            = Et(tt)
            v                   = bte.step_bte_x(v, tt, dt * 0.5)
            v                   = bte.step_bte_v(v, None, tt, dt, ts_type="BE", verbose=1)
            v                   = bte.step_bte_x(v, tt + 0.5 * dt, dt * 0.5)
       
        v_all_lm               = xp.einsum("al,ilx->iax", bte.op_po2sh, v_all)

        spec_sp                = bte.op_spec_sp
        num_p                  = spec_sp._p + 1
        num_sh                 = len(spec_sp._sph_harm_lm)
        rom_lm_modes           = 2
        fl=list()
        for l in range(rom_lm_modes):
           fl.append(v_all_lm[:, l::num_sh, :])
        
        def svd(fl, threshold, xp):
           num_t, num_p, num_x = fl.shape
           Ux, Sx, Vhx = xp.linalg.svd(fl.reshape(num_t * num_p, -1)) # Nt Nr x Nx
           Uv, Sv, Vhv = xp.linalg.svd(np.swapaxes(fl, 0, 1).reshape((num_p, num_t * num_x))) # Nr x Nt Nx

           Vx  = Vhx[Sx > Sx[0] * threshold, :].T
           Uv  = Uv [:, Sv > Sv[0] * threshold]

           return Ux, Vx.T
        
        Uv  = list()
        VxT = list()

        for l in range(rom_lm_modes):
           uv, vxT = svd(fl[l], threshold=threshold, xp=xp)
           Uv.append(uv)
           VxT.append(vxT)
           
    def init(self, Uv, VxT, rom_lm_modes):
       bte     = self.bte_solver
       xp      = bte.xp_module

       Cen     = bte.op_col_en
       Ctg     = bte.op_col_gT
       
       Cop     = bte.op_col_en + bte.args.Tg * Ctg
       Av      = bte.op_adv_v
       
       Mvr     = bte.op_mm
       Mv      = bte.op_mm_full
       Mvr_inv = bte.op_inv_mm
       Mv_inv  = bte.op_inv_mm 
       Ps      = bte.op_po2sh
       Po      = bte.op_psh2o

       Ax      = xp.dot(Mvr_inv, bte.op_adv_x)
       Ax      = xp.kron(xp.diag(bte.xp_cos_vt), Ax)
       Gx      = xp.dot(Mv_inv, xp.dot(Ps, xp.dot(Ax, Po)))

       spec_sp = bte.op_spec_sp
       num_p   = spec_sp._p + 1
       num_sh  = len(spec_sp._sph_harm_lm)

       U0, U1   = Uv[0], Uv[1]
       Vx0, Vx1 = VxT[0].T, VxT[1].T
       Dx       = bte.Dp
       
       self.Uv  = Uv
       self.VxT = VxT

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


       self.len_Fr0 = U0.shape[1] * Vx0.shape[1]
       self.len_Fr1 = U1.shape[1] * Vx1.shape[1]
       return
       
    def rhs(self, Fr, Et, time, dt, type):
       bte     = self.bte_solver
       xp      = bte.xp_module

       if (type == "BE"):
          Fr0 = Fr[0]
          Fr1 = Fr[1]

          Vx0 = self.VxT[0].T
          Vx1 = self.VxT[1].T

          E   = Et(time)
          E00 = xp.dot(Vx0.T, xp.dot(xp.diag(E), Vx0))
          E01 = xp.dot(Vx1.T, xp.dot(xp.diag(E), Vx0))
          E10 = xp.dot(Vx0.T, xp.dot(xp.diag(E), Vx1))
          E11 = xp.dot(Vx1.T, xp.dot(xp.diag(E), Vx1))

          R0  = Fr0 + dt * ( -xp.dot(self.Gx00, xp.dot(Fr0, self.Dx00)) - xp.dot(self.Gx01, xp.dot(Fr1, self.Dx01))
                             +xp.dot(self.Cv00, Fr0) 
                             +xp.dot(self.Av00, xp.dot(Fr0, E00)) +  xp.dot(self.Av01, xp.dot(Fr1, E01)))
          
          R1  = Fr1 + dt * ( -xp.dot(self.Gx10, xp.dot(Fr0, self.Dx10)) - xp.dot(self.Gx11, xp.dot(Fr1, self.Dx11))
                             +xp.dot(self.Cv11, Fr1) 
                             +xp.dot(self.Av10, xp.dot(Fr0, E10)) +  xp.dot(self.Av11, xp.dot(Fr1, E11)))
          

          res = xp.append(R0.reshape((-1)), R1.reshape((-1)))
          return res
       
    def jac(self, Xr, Et, time, dt, type):
       bte     = self.bte_solver
       xp      = bte.xp_module
       if (type == "BE"):
          Xr0 = Xr[0]
          Xr1 = Xr[1]

          Vx0 = self.VxT[0].T
          Vx1 = self.VxT[1].T

          E   = Et(time)
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

    def evolve(self, f0, dt):
       pass
       

if __name__ == "__main__":
   args = args_parse()

   glow_1d = glow1d_boltzmann(args)
   u, v    = glow_1d.initialize()

   if args.use_gpu==1:
    gpu_device = cp.cuda.Device(args.gpu_device_id)
    gpu_device.use()
   
   
