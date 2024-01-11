"""
macroscopic/microscopic modeling of the 1d glow discharge problem
1). We use Gauss-Chebyshev-Lobatto co-location method with implicit time integration. 
"""
import numpy as np
import scipy.constants 
import argparse
import matplotlib.pyplot as plt
import sys
import glow1d_utils
import basis
import collisions
import utils as bte_utils
import basis
import spec_spherical as sp
import collision_operator_spherical as collOpSp
import scipy.constants
import os
import cupyx.scipy.sparse.linalg
import scipy.sparse.linalg
from time import perf_counter, sleep

CUDA_NUM_DEVICES      = 0
PROFILE_SOLVERS       = 0
try:
  import cupy as cp
  #CUDA_NUM_DEVICES=cp.cuda.runtime.getDeviceCount()
except ImportError:
  print("Please install CuPy for GPU use")
  #sys.exit(0)
except:
  print("CUDA not configured properly !!!")
  sys.exit(0)


class gmres_counter(object):
  def __init__(self, disp=True):
      self._disp = disp
      self.niter = 0
  def __call__(self, rk=None):
      self.niter += 1
      if self._disp:
          print('iter %3i\trk = %s' % (self.niter, str(rk)))

class glow1d_boltzmann():
    """
    perform glow discharge simulation with electron Boltzmann solver
    """
    def __init__(self, args) -> None:
      
      self.ts_type_fluid       = "BE"
      self.ts_type_bte_v       = "BE"
      self.ts_op_split_factor  = 1/1
       
      dir            = args.dir
      if os.path.exists(dir):
        print("run directory exists, data will be overwritten")
        #sys.exit(0)
      else:
        os.makedirs(dir)
        print("directory %s created"%(dir))
      
      args.fname=str(dir)+"/"+args.fname
      
      with open("%s_args.txt"%(args.fname), "w") as ff:
        ff.write("args: %s"%(args))
        ff.close()
      
      self.args      = args
      self.param     = glow1d_utils.parameters()
      
      self.Ns  = self.args.Ns                   # Number of species
      self.NT  = self.args.NT                   # Number of temperatures
      self.Nv  = self.args.Ns + self.args.NT    # Total number of 'state' variables
      
      self.Nr    = self.args.Nr + 1             # e-boltzmann number of radial dof  
      self.Nvt   = self.args.Nvt                # e-boltzmann number of ordinates
      
      # currently we assume azimuthal symmetry in the v-space
      self.dof_v = self.Nr * (self.args.l_max+1)
      
      
      self.deg = self.args.Np-1                 # degree of Chebyshev polys we use
      self.Np  = self.args.Np                   # Number of points used to define state in space
      self.Nc  = self.args.Np-2                 # number of collocation pts (Np-2 b/c BCs)
      
      self.ele_idx = 0
      self.ion_idx = 1
      self.Te_idx  = self.Ns
      
      self.fluid_idx = [self.ion_idx]
      
      self.kB   = scipy.constants.Boltzmann
      
      # charge number
      self.Zp    = np.zeros(self.Ns)
      self.Zp[0] = -1 # electrons are always -1
      self.Zp[1] =  1 # ions are always 1
      
      # reaction rate coefficients
      self.r_rates = np.zeros((self.Np , self.Ns))
      
      # mobility
      self.mu = np.zeros((self.Np , self.Ns))
      # diffusivity
      self.D  = np.zeros((self.Np , self.Ns))
      
      self.xp    = -np.cos(np.pi*np.linspace(0,self.deg,self.Np)/self.deg)
      
      #self.xp = np.linspace(-1,1, self.Np)
      from numpy.polynomial import chebyshev as cheb
      # Operators
      ident = np.identity(self.Np)

      # V0p: Coefficients to values at xp
      self.V0p = np.polynomial.chebyshev.chebvander(self.xp, self.deg)

      # V0pinv: xp values to coefficients
      self.V0pinv = np.linalg.solve(self.V0p, ident)

      # V1p: coefficients to derivatives at xp
      self.V1p = np.zeros((self.Np,self.Np))
      for i in range(0,self.Np):
          self.V1p[:,i] = cheb.chebval(self.xp, cheb.chebder(ident[i,:], m=1))

      # Dp: values at xp to derivatives at xp
      self.Dp  = self.V1p @ self.V0pinv
      self.DpT = np.transpose(self.Dp)
      
      # V2p: coefficients to 2nd derivatives at xp
      self.V2p = np.zeros((self.Np,self.Np))
      for i in range(0,self.Np):
          self.V2p[:,i] = cheb.chebval(self.xp, cheb.chebder(ident[i,:], m=2))

      # Lp: values at xp to 2nd derivatives at xp
      self.Lp = self.V2p @ self.V0pinv
      
      # self.xp = np.linspace(-1,1,self.Np)
      # self.Dp = np.eye(self.Np)
      # self.Lp = np.eye(self.Np)
      
      # LpD: values at xp to 2nd derivatives at xc, with identity
      # for top and bottom row (for Dirichlet BCs)
      self.LpD = np.identity(self.Np)
      self.LpD[1:-1,:] = self.Lp[1:-1,:]
      
      self.LpD_inv     = np.linalg.solve(self.LpD, np.eye(self.Np)) 
      
      
      Imat      = np.eye(self.Np)
      Imat[0,0] = Imat[-1,-1] = 0

      self.phi_ni =  np.linalg.solve(self.LpD, -self.param.alpha*Imat)
      self.phi_ne = -self.phi_ni
      
      self.E_ni    = -np.dot(self.Dp, self.phi_ni)
      self.E_ne    = -np.dot(self.Dp, self.phi_ne)
      
      Imat[0,0] = Imat[-1,-1] = 1.0
      self.I_Nx          = Imat
      self.I_Nv          = np.eye(self.dof_v)
      self.I_Nxv_stacked = np.eye(self.dof_v)#np.einsum('i,kl->ikl',np.ones_like(self.xp), self.I_Nv) 
      
      # setting up the Boltzmann grid params
      
      self.qe            = scipy.constants.e
      self.me            = scipy.constants.electron_mass
      self.kB            = scipy.constants.Boltzmann
      self.c_gamma       = np.sqrt(2 * (self.qe/ self.me))
      
      self.Nvt           = self.args.Nvt
      
      assert self.Nvt%2 == 0 
      gx, gw             = basis.Legendre().Gauss_Pn(self.Nvt//2)
      gx_m1_0 , gw_m1_0  = 0.5 * gx - 0.5, 0.5 * gw
      gx_0_p1 , gw_0_p1  = 0.5 * gx + 0.5, 0.5 * gw
      self.xp_vt         = np.append(np.arccos(gx_m1_0), np.arccos(gx_0_p1)) 
      self.xp_vt_qw      = np.append(gw_m1_0, gw_0_p1)
      
      # print(self.xp_vt, self.xp_vt_qw) 
      
      # self.xp_vt, self.xp_vt_qw = basis.Legendre().Gauss_Pn(self.Nvt)
      # self.xp_vt                = np.arccos(self.xp_vt) 
      
      #print(self.xp_vt, self.xp_vt_qw) 

      self.xp_cos_vt     = np.cos(self.xp_vt)
      self.xp_vt_l       = np.array([i * self.Nvt + j for i in range(self.Nr) for j in list(np.where(self.xp_vt <= 0.5 * np.pi)[0])])
      self.xp_vt_r       = np.array([i * self.Nvt + j for i in range(self.Nr) for j in list(np.where(self.xp_vt > 0.5 * np.pi)[0])])
      
      self.bs_Te         = args.Te 
      self.bs_nr         = args.Nr
      self.bs_lm         = [[l,0] for l in range(self.args.l_max+1)]
      self.bs_vth        = collisions.electron_thermal_velocity(self.bs_Te * (self.qe / self.kB))
      self.bs_ev_range   = None
      
      self.op_col_en     = None
      self.op_col_gT     = None
      self.op_col_ee     = None
      self.op_adv_v      = None
      self.op_adv_x      = None
      self.op_qmat       = None 
      self.op_rmat       = None 
      self.op_diag_dg    = None 
      self.op_sigma_m    = None
      self.op_mm         = None
      self.op_inv_mm     = None
      
      self.op_mm_full    = None
      self.op_inv_mm_full= None
      
      self.op_mass       = None
      self.op_temp       = None
      self.op_diffusion  = None
      self.op_mobility   = None
      self.op_rate       = None
      self.op_spec_sp    = None
      self.op_po2sh      = None
      self.op_psh2o      = None
      
      self.bs_coll_list  = list()
      self.xp_module     = np
      
      self.weak_bc_ni    = True
      self.weak_bc_bte   = False
      
      
      #self.vth_fac       = 100
      self.ev_lim         = (0,self.args.ev_max)
      
      self.initialize_boltzmann()
    
    def initialize_boltzmann(self):
      
      for col_idx, col in enumerate(self.args.collisions):
          if "g0NoLoss" == col:
              g  = collisions.eAr_G0_NoEnergyLoss()
              g.reset_scattering_direction_sp_mat()
          elif "g0ConstNoLoss" == col:
              g  = collisions.eAr_G0_NoEnergyLoss(cross_section="g0Const")
              g.reset_scattering_direction_sp_mat()
          elif "g0" in str(col):
              g = collisions.eAr_G0(cross_section=col)
              g.reset_scattering_direction_sp_mat()
          elif "g1" in str(col):
              g = collisions.eAr_G1(cross_section=col)
              g.reset_scattering_direction_sp_mat()
          elif "g2" in str(col):
              g = collisions.eAr_G2(cross_section=col)
              g.reset_scattering_direction_sp_mat()
          else:
              print("unknown collision %s"%(col))
              sys.exit(0)

          self.bs_coll_list.append(g)
          
      args      = self.args
      sig_pts   =  list()
      for col_idx, col in enumerate(self.bs_coll_list):
          g  = self.bs_coll_list[col_idx]
          if g._reaction_threshold >0:
              sig_pts.append(g._reaction_threshold)
      
      vth                    = self.bs_vth
      maxwellian             = bte_utils.get_maxwellian_3d(vth, 1.0)
      
      dg_nodes               = np.sqrt(np.array(sig_pts)) * self.c_gamma / vth
      ev_range               = (self.ev_lim[0], self.ev_lim[1])#((0 * vth /self.c_gamma)**2, (self.vth_fac * vth /self.c_gamma)**2)
      k_domain               = (np.sqrt(ev_range[0]) * self.c_gamma / vth, np.sqrt(ev_range[1]) * self.c_gamma / vth)
      use_ee                 = args.ee_collisions
      self.bs_use_dg         = 1
      
      if use_ee==1:
          self.bs_use_dg=0
      else:
          self.bs_use_dg=0
          
      if self.bs_use_dg==1:
        print("DG boltzmann grid v-space ev=", ev_range, " v/vth=",k_domain)
        print("DG points \n", sig_pts)
      else:
        print("CG boltzmann grid v-space ev=", ev_range, " v/vth=",k_domain)
      
      
      # construct the spectral class 
      bb                     = basis.BSpline(k_domain, self.args.sp_order, self.bs_nr + 1, sig_pts=dg_nodes, knots_vec=None, dg_splines=self.bs_use_dg, verbose = args.verbose, extend_domain_with_log=True)
      spec_sp                = sp.SpectralExpansionSpherical(self.bs_nr, bb, self.bs_lm)
      spec_sp._num_q_radial  = bb._num_knot_intervals * self.args.spline_qpts
      collision_op           = collOpSp.CollisionOpSP(spec_sp)
      self.op_spec_sp        = spec_sp
      
      num_p                  = spec_sp._p + 1
      num_sh                 = len(self.bs_lm)
      num_vt                 = self.Nvt
      
      # ordinates to spherical lm computation (num_sh, self.Nvt)
      # vt_pts                 = self.xp_vt
      # glx, glw               = basis.Legendre().Gauss_Pn(len(vt_pts))
      # vq_theta               = np.arccos(glx)
      # assert (vq_theta == vt_pts).all(), "collocation points does not match with the theta quadrature points"
      tmp1                    = np.matmul(spec_sp.Vq_sph(self.xp_vt, np.zeros_like(self.xp_vt)), np.diag(self.xp_vt_qw) ) * 2 * np.pi
      #tmp1                     = np.linalg.inv(np.transpose(spec_sp.Vq_sph(self.xp_vt, np.zeros_like(self.xp_vt))))
      
      #print(tmp)
      self.op_po2sh = np.zeros((num_p * num_sh, num_p* num_vt))
      for ii in range(num_p):
        for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
          for vt_idx in range(num_vt):
            self.op_po2sh[ii * num_sh + lm_idx, ii * num_vt + vt_idx] = tmp1[lm_idx, vt_idx]
      
      #print(self.op_po2sh)
      
      # spherical lm to ordinates computation (self.Nvt, num_sh)
      tmp2                    = np.transpose(spec_sp.Vq_sph(self.xp_vt, np.zeros_like(self.xp_vt))) 
      self.op_psh2o           = np.zeros((num_p * num_vt, num_p * num_sh)) 
      for ii in range(num_p):
        for vt_idx in range(num_vt):
          for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
            self.op_psh2o[ii * num_vt + vt_idx, ii * num_sh + lm_idx] = tmp2[vt_idx, lm_idx]
      
      # to check if the oridinates to spherical and spherical to ordinates projection is true (weak test not sufficient but necessary condition)
      assert np.allclose(np.dot(tmp2, np.dot(tmp1, 1 + np.cos(self.xp_vt))), 1 + np.cos(self.xp_vt))
      
      self.par_ev_range       = ev_range  
      mm_mat                  = spec_sp.compute_mass_matrix()
      inv_mm_mat              = spec_sp.inverse_mass_mat(Mmat = mm_mat)
      mm_mat                  = mm_mat[0::num_sh, 0::num_sh]
      inv_mm_mat              = inv_mm_mat[0::num_sh, 0::num_sh]
      
      I_Nvt                   = np.eye(self.Nvt)
      # at the moment no need to store the mass-mat itself, what we need is the inverse of mass matrix. 
      #self.op_mm_full        = np.zeros((num_p * num_vt, num_p * num_vt))
      self.op_inv_mm_full     = np.zeros((num_p * num_vt, num_p * num_vt))
      self.op_inv_mm          = inv_mm_mat
      
      for vt_idx in range(num_vt):
        #self.op_mm_full[vt_idx :: num_vt, vt_idx::num_vt] = mm_mat
        self.op_inv_mm_full[vt_idx :: num_vt, vt_idx::num_vt] = inv_mm_mat
        # for i in range(self.Nr):
        #   for j in range(self.Nr):
        #     self.op_inv_mm_full[i * num_vt + vt_idx, j* num_vt + vt_idx] = inv_mm_mat[i,j]
          
      
      gx, gw                  = spec_sp._basis_p.Gauss_Pn(spec_sp._num_q_radial)
      sigma_m                 = np.zeros(len(gx))
      c_gamma                 = self.c_gamma
      gx_ev                   = (gx * vth / c_gamma)**2

      FOp                     = 0
      sigma_m                 = 0
      FOp_g                   = 0
      
      
      def compute_spatial_advection_op():
        spec_sp = self.op_spec_sp
        if spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:
            num_p  = spec_sp._p+1
            num_sh = len(spec_sp._sph_harm_lm)

            k_vec    = spec_sp._basis_p._t
            dg_idx   = spec_sp._basis_p._dg_idx
            sp_order = spec_sp._basis_p._sp_order
    
            [gx, gw] = spec_sp._basis_p.Gauss_Pn((sp_order + 8) * spec_sp._basis_p._num_knot_intervals)
            mm=np.zeros((num_p*num_sh, num_p*num_sh))
            
            for e_id in range(0,len(dg_idx),2):
                ib=dg_idx[e_id]
                ie=dg_idx[e_id+1]

                xb=k_vec[ib]
                xe=k_vec[ie+sp_order+1]
                
                idx_set     = np.logical_and(gx>=xb, gx <=xe)
                gx_e , gw_e = gx[idx_set],gw[idx_set]

                mm_l = np.zeros((num_p,num_p))
                for p in range(ib, ie+1):
                    k_min   = k_vec[p]
                    k_max   = k_vec[p + sp_order + 1]
                    qx_idx  = np.logical_and(gx_e >= k_min, gx_e <= k_max)
                    gmx     = gx_e[qx_idx]
                    gmw     = gw_e[qx_idx]
                    b_p     = spec_sp.basis_eval_radial(gmx, p, 0)  

                    for k in range(max(ib, p - (sp_order+3) ), min(ie+1, p + (sp_order+3))):
                        b_k       = spec_sp.basis_eval_radial(gmx, k, 0)
                        mm_l[p,k] = np.dot((gmx**3) * b_p * b_k, gmw)
            return mm_l
      
      t1 = perf_counter()
      print("assembling collision operators")
      for col_idx in range(len(self.bs_coll_list)):
          g = self.bs_coll_list[col_idx]
          g.reset_scattering_direction_sp_mat()
          col = g._col_name
          
          if args.verbose==1:
              print("collision %d included %s"%(col_idx, col))

          if "g0NoLoss" == col:
              FOp       = FOp + collision_op.assemble_mat(g, maxwellian, vth, tgK=0.0, mp_pool_sz=args.threads)
              FOp_g     = collision_op.electron_gas_temperature(g, maxwellian, vth, mp_pool_sz=args.threads)
              sigma_m  += g.total_cross_section(gx_ev)
          elif "g0ConstNoLoss" == col:
              FOp       = FOp + collision_op.assemble_mat(g, maxwellian, vth, tgK=0.0, mp_pool_sz=args.threads)
              FOp_g     = collision_op.electron_gas_temperature(g, maxwellian, vth, mp_pool_sz=args.threads)
              sigma_m  += g.total_cross_section(gx_ev)
          elif "g0" in col:
              FOp       = FOp + collision_op.assemble_mat(g, maxwellian, vth, tgK=0.0, mp_pool_sz=args.threads)
              FOp_g     = collision_op.electron_gas_temperature(g, maxwellian, vth, mp_pool_sz=args.threads)
              sigma_m  += g.total_cross_section(gx_ev)
          elif "g1" in col:
              FOp       = FOp + collision_op.assemble_mat(g, maxwellian, vth, tgK=0.0, mp_pool_sz=args.threads)
              sigma_m  += g.total_cross_section(gx_ev)
          elif "g2" in col:
              FOp       = FOp + collision_op.assemble_mat(g, maxwellian, vth, tgK=0.0, mp_pool_sz=args.threads)
              sigma_m  += g.total_cross_section(gx_ev)
          else:
              print("%s unknown collision"%(col))
              sys.exit(0)
      t2 = perf_counter()
      print("assembly = %.4E"%(t2-t1))
      
      t1 = perf_counter()
      print("bte qoi op assembly")    
      self.op_sigma_m   = sigma_m
      
      self.op_mass      = bte_utils.mass_op(spec_sp, 1) #* maxwellian(0) * vth**3
      self.op_temp      = bte_utils.temp_op(spec_sp, 1) * (vth**2) * 0.5 * scipy.constants.electron_mass * (2/3/scipy.constants.Boltzmann) / collisions.TEMP_K_1EV
      
      # note non-dimentionalized electron mobility and diffusion,  ## not used at the moment, but needs to check.          
      self.op_mobility  = bte_utils.mobility_op(spec_sp, maxwellian, vth) * self.param.V0 * self.param.tau/self.param.L**2
      self.op_diffusion = bte_utils.diffusion_op(spec_sp, self.bs_coll_list, maxwellian, vth) * self.param.tau/self.param.L**2
          
      rr_op  = [None] * len(self.bs_coll_list)
      for col_idx, g in enumerate(self.bs_coll_list):
          rr_op[col_idx] = bte_utils.reaction_rates_op(spec_sp, [g], maxwellian, vth) 
          
      self.op_rate      = rr_op
      t2 = perf_counter()
      print("assembly = %.4E"%(t2-t1))
      
      t1 = perf_counter()
      print("assembling v-space advection op")
      if self.bs_use_dg == 1 : 
          adv_mat_v, eA, qA = spec_sp.compute_advection_matix_dg(advection_dir=-1.0)
          qA              = np.kron(np.eye(spec_sp.get_num_radial_domains()), np.kron(np.eye(num_p), qA))
      else:
          # cg advection
          adv_mat_v       = spec_sp.compute_advection_matix()
          qA              = np.eye(adv_mat_v.shape[0])
          
      self.op_diag_dg         = qA
      FOp                     = np.matmul(np.transpose(qA), np.matmul(FOp, qA))
      FOp_g                   = np.matmul(np.transpose(qA), np.matmul(FOp_g, qA))
      
      def psh2o_C_po2sh(opM):
          return opM
          #return np.dot(self.op_psh2o, np.dot(opM,self.op_po2sh))
      
      adv_mat_v                *= (1 / vth) * collisions.ELECTRON_CHARGE_MASS_RATIO
      adv_mat_v                 = psh2o_C_po2sh(adv_mat_v)
      
      FOp                       = psh2o_C_po2sh(FOp)
      FOp_g                     = psh2o_C_po2sh(FOp_g)
      
      adv_x                     = np.dot(self.op_inv_mm,  vth  * (self.param.tau/self.param.L) * compute_spatial_advection_op())
      #print(" cond number = %.8E \n"%np.linalg.cond(adv_x))
      adv_x_d, adv_x_q          = np.linalg.eig(adv_x)
      self.op_adv_x_d           = adv_x_d
      self.op_adv_x_q           = adv_x_q
      self.op_adv_x_qinv        = np.linalg.inv(adv_x_q)
      
      t2 = perf_counter()
      print("assembly = %.4E"%(t2-t1))
      #print(self.op_adv_x_d)
      #plt.plot(self.op_adv_x_d)
      #plt.show()
      #print("x-space advection diagonalization rel error = %.8E"%(np.linalg.norm(np.dot(self.op_adv_x_d * self.op_adv_x_q, self.op_adv_x_qinv) - adv_x)/np.linalg.norm(adv_x)))
      # adv_mat_x                 = np.zeros((num_p * num_vt, num_p * num_vt))
      # for vt_idx in range(num_vt):
      #   adv_mat_x[vt_idx::num_vt, vt_idx :: num_vt] = adv_x * np.cos(self.xp_vt[vt_idx])
      
      
      self.op_inv_mm_full       = np.zeros((num_p * num_sh, num_p * num_sh))
      for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
        self.op_inv_mm_full[lm_idx::num_sh, lm_idx::num_sh] = inv_mm_mat
      
      FOp                       = np.dot(self.op_inv_mm_full, FOp)
      FOp_g                     = np.dot(self.op_inv_mm_full, FOp_g)
      adv_mat_v                 = np.dot(self.op_inv_mm_full, adv_mat_v)
      
      self.op_adv_x             = adv_x
      self.op_adv_v             = adv_mat_v
      self.op_col_en            = FOp
      self.op_col_gT            = FOp_g
      
      xp=np
      xp.save("%s_bte_mass_op.npy"%(args.fname), self.op_mass)
      xp.save("%s_bte_temp_op.npy"%(args.fname), self.op_temp)
      xp.save("%s_bte_po2sh.npy"  %(args.fname), self.op_po2sh)
      xp.save("%s_bte_psh2o.npy"  %(args.fname), self.op_psh2o)
      xp.save("%s_bte_op_g0.npy"  %(args.fname), self.op_rate[0])
      
      if (len(self.op_rate) > 1):
        xp.save("%s_bte_op_g2.npy"  %(args.fname), self.op_rate[1])
      
          
      if(use_ee == 1):
          print("e-e collision assembly begin")
          
          hl_op, gl_op         = collision_op.compute_rosenbluth_potentials_op(maxwellian, vth, 1, mmat_inv, mp_pool_sz=args.threads)
          cc_op_a, cc_op_b     = collision_op.coulomb_collision_op_assembly(maxwellian, vth, mp_pool_sz=args.threads)
          
          xp                   = self.xp_module
          qA                   = self.op_diag_dg
          mmat_inv             = self.op_inv_mm_full
          
          cc_op                = xp.dot(cc_op_a, hl_op) + xp.dot(cc_op_b, gl_op)
          cc_op                = xp.dot(cc_op,qA)
          cc_op                = xp.dot(xp.swapaxes(cc_op,1,2),qA)
          cc_op                = xp.swapaxes(cc_op,1,2)
          cc_op                = xp.dot(xp.transpose(qA), cc_op.reshape((num_p*num_sh,-1))).reshape((num_p * num_sh, num_p * num_sh, num_p * num_sh))
          cc_op                = xp.dot(mmat_inv, cc_op.reshape((num_p*num_sh,-1))).reshape((num_p * num_sh, num_p * num_sh, num_p * num_sh))
          
          self.op_col_ee       = cc_op
          
          print("e-e collision assembly end")
      
    def initialize(self,type=0):
      
      xp      = self.xp_module
      ele_idx = self.ele_idx
      ion_idx = self.ion_idx
      Te_idx  = self.Te_idx
      Uin     = xp.zeros((self.Np, self.Nv))
      Vin     = xp.zeros((self.Nr * self.Nvt, self.Np))
      
      if type==0:
        if self.args.restore==1:
          print("~~~restoring solver from %s.npy"%(args.fname))
          Uin = xp.load("%s_%04d_u.npy"%(args.fname, args.rs_idx))
          Vin = xp.load("%s_%04d_v.npy"%(args.fname, args.rs_idx))
        else:
          xx = self.param.L * (self.xp + 1)
          Uin[:, ele_idx] = 1e6 * (1e7 + 1e9 * (1-0.5 * xx/self.param.L)**2 * (0.5 * xx/self.param.L)**2) / self.param.np0
          Uin[:, ion_idx] = 1e6 * (1e7 + 1e9 * (1-0.5 * xx/self.param.L)**2 * (0.5 * xx/self.param.L)**2) / self.param.np0
          Uin[:, Te_idx]  = self.param.Teb
          
          spec_sp     = self.op_spec_sp
          mmat        = spec_sp.compute_mass_matrix()
          mmat_inv    = spec_sp.inverse_mass_mat(Mmat = mmat)
          vth         = self.bs_vth
          mw          = bte_utils.get_maxwellian_3d(vth, 1)

          mass_op     = self.op_mass
          temp_op     = self.op_temp

          v_ratio = 1.0 #np.sqrt(1.0/args.basis_scale)
          hv      = lambda v,vt,vp : (1/np.sqrt(np.pi)**3) * np.exp(-((v/v_ratio)**2)) / v_ratio**3
          h_init  = bte_utils.function_to_basis(spec_sp,hv,mw, spec_sp._num_q_radial, 2, 2, Minv=mmat_inv)
          m0      = xp.dot(mass_op, h_init)
          for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
            if lm[0]>0:
              h_init[lm_idx::len(spec_sp._sph_harm_lm)]=0.0
          
          h_init = h_init/m0
          print("boltzmann initial conditions, mass and temperature (eV)", m0, xp.dot(temp_op, h_init)/m0)
          
          h_init    = xp.dot(self.op_psh2o, h_init)
          h_init_l  = xp.copy(h_init)
          h_init_r  = xp.copy(h_init)
          
          # h_init_l[self.xp_vt_l] = 0
          # h_init_r[self.xp_vt_r] = 0
          
          # h_init_l               = xp.dot(self.op_po2sh,h_init_l)
          # m0                     = xp.dot(mass_op, h_init_l)
          # h_init_l               = h_init_l/m0
          # h_init_l               = xp.dot(self.op_psh2o, h_init_l)
          # h_init_l[self.xp_vt_l] = 0
          
          # hh                     = xp.dot(self.op_po2sh,h_init_l)
          # m0                     = xp.dot(mass_op, hh)
          # print("x at left bdy, mass and temperature (eV)", m0, xp.dot(temp_op, hh)/m0)
          
          
          # h_init_r               = xp.dot(self.op_po2sh,h_init_r)
          # m0                     = xp.dot(mass_op, h_init_r)
          # h_init_r               = h_init_r/m0
          # h_init_r               = xp.dot(self.op_psh2o, h_init_r)
          # h_init_r[self.xp_vt_r] = 0
          
          # hh                     = xp.dot(self.op_po2sh,h_init_r)
          # m0                     = xp.dot(mass_op, hh)
          # print("x at right bdy, mass and temperature (eV)", m0, xp.dot(temp_op, hh)/m0)
          
          for i in range(1, self.Np-1):
            Vin[:,i] = h_init
          
          Vin[:,0]  = h_init_l
          Vin[:,-1] = h_init_r
          
          
          # scale functions to have ne, at initial timestep
          Vin = Vin * Uin[:,ele_idx]
          
          #Vin_lm = self.bte_eedf_normalization(xp.dot(self.op_po2sh, Vin))
          #print(Vin_lm)
          #print(xp.dot(self.op_rate[1] , Vin_lm[0::2]))
        
        self.mu[:, ele_idx] = self.param.mu_e
        self.D[: , ele_idx] = self.param.De
        
        self.mu[:, ion_idx] = self.param.mu_i
        self.D[: , ion_idx] = self.param.Di
        
        
          
      else:
        raise NotImplementedError
      
      return Uin, Vin
    
    def initialize_bte_adv_x(self, dt):
      """initialize spatial advection operator"""
      xp = self.xp_module
      #assert xp == np
      
      self.bte_x_shift      = xp.zeros((self.Nr, self.Nvt, self.Np, self.Np))
      #self.bte_x_shift_rmat = xp.zeros((self.Nr, self.Nvt, self.Np, self.Np))
      DpL = xp.zeros((self.Np, self.Np))
      DpR = xp.zeros((self.Np, self.Np))

      DpL[1:,:]  = self.Dp[1:,:]
      DpR[:-1,:] = self.Dp[:-1,:]
      
      f1 = 1.0
      f2 = 1-f1

      for j in range(self.Nvt):
        if (self.xp_vt[j] <= 0.5 * xp.pi):
          for i in range(self.Nr):
            self.bte_x_shift[i, j, : , :]       = self.I_Nx + f1 * dt * self.op_adv_x_d[i] * xp.cos(self.xp_vt[j]) * DpL
            #self.bte_x_shift_rmat[i,j,:,:]      = -f2 * dt * self.op_adv_x_d[i] * xp.cos(self.xp_vt[j]) * DpL
        else:
          for i in range(self.Nr):
            self.bte_x_shift[i, j, : , :]       = self.I_Nx + f1 * dt * self.op_adv_x_d[i] * xp.cos(self.xp_vt[j]) * DpR
            #self.bte_x_shift_rmat[i, j, : , :]  = -f2 * dt * self.op_adv_x_d[i] * xp.cos(self.xp_vt[j]) * DpR
        
      self.bte_x_shift = xp.linalg.inv(self.bte_x_shift)
      #self.bte_x_shift = cp.asnumpy(cp.linalg.inv(cp.asarray(self.bte_x_shift)))
      
      adv_mat = cp.asnumpy(self.bte_x_shift)
      v1 = -self.xp**2 + 1.2
      plt.figure(figsize=(10,4), dpi=300)
      plt.subplot(1, 2 , 1)
      plt.semilogy(self.xp, v1,"-b",label="t=0")
      y1 = np.copy(v1)
      for i in range(100):
        #y1 = y1 + xp.dot(self.bte_x_shift_rmat[-1,-1],y1)
        y1[0]=0
        y1=xp.dot(adv_mat[-1,-1],y1)
      plt.semilogy(self.xp, y1,"-r", label="left to right")
      y1 = np.copy(v1)
      for i in range(100):
        #y1 = y1 + xp.dot(self.bte_x_shift_rmat[-1, 0],y1)
        y1[-1]=0
        y1=xp.dot(adv_mat[-1,0],y1)
      plt.semilogy(self.xp, y1,"-y", label="right to left ")
      plt.legend()
      plt.grid()
      
      plt.subplot(1, 2 , 2)
      plt.plot(self.xp, v1,"-b",label="t=0")
      y1 = np.copy(v1)
      for i in range(100):
        #y1 = y1 + xp.dot(self.bte_x_shift_rmat[-1,-1],y1)
        y1[0]=0
        y1=xp.dot(adv_mat[-1,-1],y1)
      plt.plot(self.xp, y1,"-r", label="left to right")
      y1 = np.copy(v1)
      for i in range(100):
        #y1 = y1 + xp.dot(self.bte_x_shift_rmat[-1, 0],y1)
        y1[-1]=0
        y1=xp.dot(adv_mat[-1,0],y1)
      plt.plot(self.xp, y1,"-y", label="right to left ")
      plt.legend()
      plt.grid()
      
      #plt.show()
      plt.savefig("test.png")
      plt.close()
      return
          
    def initialize_bte_adv_x1(self, dt):
      """intialize boltzmann space advection operator"""
      xp = self.xp_module
      assert xp == np
      self.bte_x_shift = xp.zeros((self.Nr, self.Nvt, self.Np, self.Np))
      
      sp_order    = 10
      q_per_knot  = 32
      bb          = basis.BSpline((self.xp[0], self.xp[-1]), sp_order, 4 * self.Np , q_per_knot, sig_pts=[], verbose=True, extend_domain=False)
      bgx, bgw    = bb.Gauss_Pn(bb._num_knot_intervals * q_per_knot)
      
      # # chebyshev collocation points to B-spline quadrature points interpolation
      P1          = np.dot(np.polynomial.chebyshev.chebvander(bgx, self.deg), self.V0pinv)
      bb_mm       = bb.mass_mat()
      bb_mm_inv   = bte_utils.choloskey_inv(bb_mm)
      
      
      b_splines   = [bb.Pn(i) for i in range(bb._num_p)] 
      
      def bspline_vander(x):
        Vq_b        = np.empty((bb._num_p, len(x)))
        for i in range(bb._num_p):
          Vq_b[i,:]  = b_splines[i](x, 0)

        return Vq_b
        
      Vq_b  = bspline_vander(bgx)
      
      # b-spline projection coefficient matrix
      P2          = np.dot(bb_mm_inv, np.dot(Vq_b * bgw, P1))
      
      #args.cfl = min(, args.cfl)
      print("time step recomended for x-advection ", (1/np.max(self.op_adv_x_d)))
      
      for i in range(self.Nr):
        for j in range(self.Nvt):
          xx = self.xp - self.op_adv_x_d[i] * np.cos(self.xp_vt[j]) * dt 
          self.bte_x_shift[i,j :, : ] = xp.dot(bspline_vander(xx).T, P2)
          
          # # Chebyshev based interpolation gives ocillations on the domain
          # self.bte_x_shift[i,j :, : ]     = np.polynomial.chebyshev.chebvander(xx, self.deg)
          # self.bte_x_shift[i,j :, : ]     = xp.dot(self.bte_x_shift[i,j], self.V0pinv)
          
          # if (xx<-1).any()==True:
          #   assert((xx>1).all()==False)
          #   self.bte_x_shift[i,j, xx<-1,:] = 0.0 
            
          # if (xx>1).any()==True:
          #   assert((xx<-1).all()==False)
          #   self.bte_x_shift[i,j, xx>1, :] = 0.0 
      
      v1 = -self.xp**2 + 1.2
      plt.figure(figsize=(4,4), dpi=300)
      plt.plot(self.xp, v1,"-b",label="t=0")
      
      y1 = np.copy(v1)
      for i in range(1):
        y1=xp.dot(self.bte_x_shift[-1,-1],y1)
      plt.plot(self.xp, y1,"-r", label="left to right")
      
      y1 = np.copy(v1)
      for i in range(1):
        y1=xp.dot(self.bte_x_shift[-1,0],y1)
      
      plt.plot(self.xp, y1,"-y", label="right to left ")
      
      plt.legend()
      plt.grid()
      #plt.show()
      plt.savefig("test.png")
      plt.close()
      
      return
    
    def copy_operators_H2D(self, dev_id):
      
      if self.args.use_gpu==0:
        return
      
      with cp.cuda.Device(dev_id):
        
        self.xp_cos_vt      = cp.asarray(self.xp_cos_vt)
        self.I_Nx           = cp.asarray(self.I_Nx)
        self.I_Nv           = cp.asarray(self.I_Nv)
        self.I_Nxv_stacked  = cp.asarray(self.I_Nxv_stacked)
        
        self.Dp             = cp.asarray(self.Dp)
        self.LpD            = cp.asarray(self.LpD)
        self.Lp             = cp.asarray(self.Lp)
        self.LpD_inv        = cp.asarray(self.LpD_inv)
        self.Zp             = cp.asarray(self.Zp)
        self.E_ne           = cp.asarray(self.E_ne)
        self.E_ni           = cp.asarray(self.E_ni)
        
        self.r_rates        = cp.asarray(self.r_rates)
        self.mu             = cp.asarray(self.mu)
        self.D              = cp.asarray(self.D)
        
        
        
        self.op_adv_v       = cp.asarray(self.op_adv_v)
        self.op_adv_x       = cp.asarray(self.op_adv_x)
        self.op_adv_x_d     = cp.asarray(self.op_adv_x_d)
        self.op_adv_x_q     = cp.asarray(self.op_adv_x_q)
        self.op_adv_x_qinv  = cp.asarray(self.op_adv_x_qinv)
        
        
        self.op_psh2o       = cp.asarray(self.op_psh2o)
        self.op_po2sh       = cp.asarray(self.op_po2sh)
        self.op_col_en      = cp.asarray(self.op_col_en)
        self.op_col_gT      = cp.asarray(self.op_col_gT)
        
        if self.args.ee_collisions==1:
          self.op_col_ee   = cp.asarray(self.op_col_ee)
          
        self.op_mass        = cp.asarray(self.op_mass)
        self.op_temp        = cp.asarray(self.op_temp)
        self.op_rate        = [cp.asarray(self.op_rate[i]) for i in range(len(self.op_rate))]
        
      return
    
    def copy_operators_D2H(self, dev_id):
      
      if self.args.use_gpu==0:
        return
      
      with cp.cuda.Device(dev_id):
        self.xp_cos_vt      = cp.asnumpy(self.xp_cos_vt)
        self.I_Nx           = cp.asnumpy(self.I_Nx)
        self.I_Nv           = cp.asnumpy(self.I_Nv)
        self.I_Nxv_stacked  = cp.asnumpy(self.I_Nxv_stacked)
        
        self.Dp             = cp.asnumpy(self.Dp)
        self.LpD_inv        = cp.asnumpy(self.DpT)
        self.Zp             = cp.asnumpy(self.Zp)
        self.E_ne           = cp.asnumpy(self.E_ne)
        self.E_ni           = cp.asnumpy(self.E_ni)
        
        self.r_rates        = cp.asnumpy(self.r_rates)
        self.mu             = cp.asnumpy(self.mu)
        self.D              = cp.asnumpy(self.D)
        
        self.op_adv_v       = cp.asnumpy(self.op_adv_v)
        self.op_adv_x       = cp.asnumpy(self.op_adv_x)
        self.op_adv_x_d     = cp.asnumpy(self.op_adv_x_d)
        self.op_adv_x_q     = cp.asnumpy(self.op_adv_x_q)
        self.op_adv_x_qinv  = cp.asnumpy(self.op_adv_x_qinv)
        
        self.op_psh2o       = cp.asnumpy(self.op_psh2o)
        self.op_po2sh       = cp.asnumpy(self.op_po2sh)
        self.op_col_en      = cp.asnumpy(self.op_col_en)
        self.op_col_gT      = cp.asnumpy(self.op_col_gT)
        
        if self.args.ee_collisions==1:
          self.op_col_ee   = cp.asnumpy(self.op_col_ee)
        
        self.op_mass        = cp.asnumpy(self.op_mass)
        self.op_temp        = cp.asnumpy(self.op_temp)
        
        self.op_rate        = [cp.asnumpy(self.op_rate[i]) for i in range(len(self.op_rate))]
      
      return  
    
    def solve_poisson(self, ne, ni,time):
        """Solve Gauss' law for the electric potential.

        Inputs:
          ne   : Values of electron density at xp
          ni   : Values of ion density at xp
          time : Current time

        Outputs: None (sets self.phi to computed potential)
        """
        xp    = self.xp_module
        r     = - self.param.alpha * (ni-ne)
        r[0]  = xp.sin(2 * xp.pi * time) #+ self.params.verticalShift
        r[-1] = 0.0
        return xp.dot(self.LpD_inv, r)
    
    def bte_eedf_normalization(self, v_lm):
      xp = self.xp_module
      # normalization of the distribution function before computing the reaction rates
      mm_fac                   = self.op_spec_sp._sph_harm_real(0, 0, 0, 0) * 4 * np.pi
      mm_op                    = self.op_mass
      c_gamma                  = self.c_gamma
      vth                      = self.bs_vth
      
      scale                    = xp.dot(mm_op / mm_fac, v_lm) * (2 * (vth/c_gamma)**3)
      return v_lm / scale
    
    def bte_to_fluid(self, u, v, time, dt):
      """
      boltzmann to fluid push
      """
      xp      = self.xp_module
      
      ele_idx = self.ele_idx
      ion_idx = self.ion_idx
      Te_idx  = self.Te_idx
      
      num_p   = self.op_spec_sp._p + 1
      num_sh  = len(self.op_spec_sp._sph_harm_lm)
      
      v_lm                   = xp.dot(self.op_po2sh, v)
      u[:, ele_idx]          = xp.dot(self.op_mass[0::num_sh], v_lm[0::num_sh,:])
      u[:, Te_idx]           = xp.dot(self.op_temp[0::num_sh], v_lm[0::num_sh,:])/u[:,ele_idx]
      
      v_lm1                  = self.bte_eedf_normalization(v_lm)
      
      if (len(self.op_rate) > 1):
        self.r_rates[:, ion_idx] = xp.dot(self.op_rate[1], v_lm1[0::num_sh,:]) * self.param.np0 * self.param.tau
      else:
        self.r_rates[:, ion_idx] = 0.0
      
      # treat the negative values. 
      # self.r_rates[self.r_rates[:, ion_idx] <0, ion_idx]  = 0.0
      # Uin[ Uin[:, ele_idx]< 0, ele_idx]                   = 0.0
      # Uin[ Uin[:, Te_idx] < 0, Te_idx]                    = 0.0
      
      return
    
    def fluid_to_bte(self, u, v, time, dt):
      xp      = self.xp_module
      ele_idx = self.ele_idx
      ion_idx = self.ion_idx
      
      phi       = self.solve_poisson(u[:,ele_idx], u[:, ion_idx], time)
      E         = -xp.dot(self.Dp, phi)
      self.bs_E = (E  * self.param.V0 / self.param.L)
      return
    
    def rhs_fluid(self, u : np.array, time, dt):
      """Evaluates the residual.

      Inputs:
        Uin  : Current state
        time : Current time
        dt   : Time step

      Outputs:
        returns residual vector

      Notes:
        This function currently assumes that Ns=2 and NT=1
      """
      xp      = self.xp_module
      
      ele_idx = self.ele_idx
      ion_idx = self.ion_idx
      Te_idx  = self.Te_idx
      
      ni      = u[: , ion_idx]
      ne      = u[: , ele_idx]
      
      mu_i    = self.mu[:, ion_idx]
      mu_e    = self.mu[:, ele_idx]
      
      De      = self.D[:, ele_idx]
      Di      = self.D[:, ion_idx]
      
      phi     = self.solve_poisson(u[:,ele_idx], u[:, ion_idx], time)
      E       = -xp.dot(self.Dp, phi)
      ki      = self.r_rates[:, ion_idx]
      
      nf_var  = len(self.fluid_idx)   #number of fluid variables
      
      Us_x    = xp.dot(self.Dp, u[: , self.fluid_idx])
      fluxJ   = xp.empty((self.Np, nf_var))
      FUin    = xp.empty((self.Np, nf_var))
      
      for idx, sp_idx  in enumerate(self.fluid_idx):
        fluxJ[:, idx] = self.Zp[sp_idx] * self.mu[: , sp_idx] * u[: , sp_idx] * E - self.D[: , sp_idx] * Us_x[:, idx]
      
      assert self.fluid_idx[0] == ion_idx
      
      if self.weak_bc_ni:  
        fluxJ[0 , 0] = self.mu[0 , ion_idx] * ni[0]  * E[0]
        fluxJ[-1, 0] = self.mu[-1, ion_idx] * ni[-1] * E[-1]
      
      fluxJ_x      = xp.dot(self.Dp, fluxJ)
      FUin[:,0]    = ki * self.param.n0 * ne - fluxJ_x[:,0]
      
      strong_bc    = xp.zeros((2,self.Nv))
      if self.args.ts_type=="BE":
        if not self.weak_bc_ni:
          strong_bc[0,  ion_idx] = (fluxJ[0 , 0] - self.mu[0 , ion_idx] * ni[0]  * E[0] )
          strong_bc[-1, ion_idx] = (fluxJ[-1, 0] - self.mu[-1, ion_idx] * ni[-1] * E[-1] )
              
      return FUin, strong_bc
      
    def rhs_fluid_jacobian(self, u : np.array, time, dt):
      xp  = self.xp_module
      dof = self.Nv * self.Np
      jac = xp.zeros((dof, dof))
      
      ele_idx = self.ele_idx
      ion_idx = self.ion_idx
      Te_idx  = self.Te_idx
      
      ne      = u[: , ele_idx]
      ni      = u[: , ion_idx]
      Te      = u[: , Te_idx]
      
      mu_i    = self.mu[:, ion_idx]
      mu_e    = self.mu[:, ele_idx]
      
      De      = self.D[:, ele_idx]
      Di      = self.D[:, ion_idx]

      ki      = self.r_rates[:, ion_idx]
      
      Np      = self.Np
      Nv      = self.Nv
      
      phi_ni  = self.phi_ni
      #phi_ne  = self.phi_ni
      
      E_ni    = self.E_ni
      #E_ne    = self.E_ne
      
      Imat    = self.I_Nx
      
      phi     = self.solve_poisson(ne, ni, time)
      E       = -xp.dot(self.Dp, phi)
      
      nf_vars = len(self.fluid_idx)
      Js_nk    = xp.zeros((nf_vars, self.Np, self.Np))
      
      for idx, i in enumerate(self.fluid_idx):
        if i == ion_idx:
          Js_nk[idx] = self.Zp[i] * self.mu[:,i] * (E * Imat + u[:,i] * E_ni) - self.D[:,i] * self.Dp
        else:
          Js_nk[idx] = self.Zp[i] * self.mu[:,i] * (E * Imat) - self.D[:,i] * self.Dp
          
      assert self.fluid_idx[0] == ion_idx
      Ji_ni        = Js_nk[0]
      if self.weak_bc_ni:
        Ji_ni[0 ,:] = mu_i[0]  * (ni[0]  * E_ni[0 ,:] + E[0]  * Imat[0 ,:])
        Ji_ni[-1,:] = mu_i[-1] * (ni[-1] * E_ni[-1,:] + E[-1] * Imat[-1,:])
      
      Ji_x_ni = xp.dot(self.Dp, Ji_ni)
      
      Rni_ni  = -Ji_x_ni
      
      jac_bc = xp.zeros((2, self.Nv, self.Np * nf_vars))
      if self.args.ts_type=="BE":
        if not self.weak_bc_ni:
          jac_bc[0, ion_idx , 0::nf_vars]  = Ji_ni[0  ,:] - (mu_i[0]  * (ni[0]  * E_ni[0   , :] + E[0]  * ni[0]  * Imat[0 ,:]))
          jac_bc[1, ion_idx , 0::nf_vars]  = Ji_ni[-1 ,:] - (mu_i[-1] * (ni[-1] * E_ni[-1  , :] + E[-1] * ni[-1] * Imat[-1,:]))
      
      return Rni_ni, jac_bc
    
    def step_bte_x(self, v, time, dt):
      "perform the bte x-advection analytically"
      xp        = self.xp_module
      
      if PROFILE_SOLVERS==1:
        if xp == cp:
          cp.cuda.runtime.deviceSynchronize()
        t1 = perf_counter()
      
      Vin       = v.reshape((self.Nr, self.Nvt , self.Np)).reshape(self.Nr, self.Nvt * self.Np)
      Vin       = xp.dot(self.op_adv_x_qinv, Vin).reshape((self.Nr, self.Nvt, self.Np)).reshape((self.Nr * self.Nvt , self.Np))
      
      # enforce rhs BCs
      Vin[self.xp_vt_l, 0]  = 0.0
      Vin[self.xp_vt_r, -1] = 0.0
      
      Vin_adv_x = xp.einsum("ijkl,ijl->ijk",self.bte_x_shift, Vin.reshape((self.Nr, self.Nvt, self.Np)))
      Vin_adv_x  = Vin_adv_x.reshape((self.Nr, self.Nvt *  self.Np))
      Vin_adv_x  = xp.dot(self.op_adv_x_q, Vin_adv_x).reshape((self.Nr , self.Nvt, self.Np)).reshape((self.Nr * self.Nvt, self.Np))
      
      # print((Vin_adv_x[self.xp_vt_l , 0] ==0).all())
      # print((Vin_adv_x[self.xp_vt_r , -1]==0).all())
      
      if PROFILE_SOLVERS==1:
        if xp == cp:
          cp.cuda.runtime.deviceSynchronize()
        t2 = perf_counter()
        print("BTE x-advection time = %.4E (s)" %(t2-t1), flush=True)
      Vin_adv_x[Vin_adv_x<0]=0.0
      return Vin_adv_x
      
    def rhs_bte_v(self, v_lm: np.array, time, dt):
      """
      compute the rhs for the 1d2v boltzmann equation.
      Uin : (Np, Ns)
      Vin : (num_p * num_vt, Np) # boltzmann rhs vector
      """
      xp        = self.xp_module
      E         = self.bs_E
      FVin      = self.param.tau * (self.param.n0 * self.param.np0 * (xp.dot(self.op_col_en, v_lm) + self.param.Tg * xp.dot(self.op_col_gT, v_lm))  + E * xp.dot(self.op_adv_v, v_lm))
      strong_bc = None
      
      return FVin, strong_bc
      
    def rhs_bte_v_jacobian(self, v_lm: np.array, time, dt):
      
      xp      = self.xp_module
      ele_idx = self.ele_idx
      ion_idx = self.ion_idx
      
      dof_v   = self.dof_v
      
      E       = self.bs_E
      
      j1      = self.param.tau * self.param.n0 * self.param.np0 * (self.op_col_en + self.param.Tg * self.op_col_gT)
      jac     = xp.einsum('i,kl->ikl', self.param.tau * E, self.op_adv_v) + j1
      
      jac_bc = None
      return jac, jac_bc
    
    def step_fluid(self, u, du, time, dt, ts_type, verbose=0):
      xp      = self.xp_module
      if PROFILE_SOLVERS==1:
        if xp == cp:
          cp.cuda.runtime.deviceSynchronize()
        t1 = perf_counter()
        
      rhs     = self.rhs_fluid 
      if ts_type == "RK2":
        k1 = dt * rhs(u, time, dt)[0]
        k2 = dt * rhs(u + 0.5 * k1, time + 0.5 * dt, dt)[0]
        return u + k2
      elif ts_type == "RK4":
        k1 = dt * rhs(u           , time           , dt)[0]
        k2 = dt * rhs(u + 0.5 * k1, time + 0.5 * dt, dt)[0]
        k3 = dt * rhs(u + 0.5 * k2, time + 0.5 * dt, dt)[0]
        k4 = dt * rhs(u +  k3     , time + dt      , dt)[0]
        return u + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
      elif ts_type == "BE":
        rtol            = self.args.rtol
        atol            = self.args.atol
        iter_max        = self.args.max_iter
        Imat            = self.I_Nx
        
        assert self.fluid_idx[0] == self.ion_idx
        u1  = xp.zeros_like(u)
        def residual(du):
          u1[:,:]                =  u[:,:]
          u1[:, self.fluid_idx] += du
          
          rhs, bc  = self.rhs_fluid(u1, time + dt, dt) 
          res      = du - dt * rhs
          
          if not self.weak_bc_ni: 
              res[0  , 0] = bc[0  , self.ion_idx]
              res[-1 , 0] = bc[-1 , self.ion_idx]
          
          return res
        
        def jacobian(du):
          rhs_j, j_bc = self.rhs_fluid_jacobian(u, time, dt)
          jac         = Imat - dt * rhs_j
          
          if not self.weak_bc_ni:              
            jac[0 * len(self.fluid_idx) + 0, :]             = j_bc[0, self.ion_idx, :]
            jac[(self.Np-1) * len(self.fluid_idx)   + 0, :] = j_bc[1, self.ion_idx, :]
          
          return jac
        
        ns_info  = glow1d_utils.newton_solver(du[:,self.fluid_idx], residual, jacobian, atol, rtol, iter_max, xp)
        du_fluid = ns_info["x"]
        
        if(verbose==1):
          print("Fluid step time = %.2E "%(time))
          print("  Newton iter {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(ns_info["iter"], ns_info["atol"], ns_info["rtol"]))
        
        if (ns_info["status"]==False):
          print("Fluid step non-linear solver step FAILED!!! try with smaller time step size or increase max iterations")
          print("time = %.2E "%(time))
          print("  Newton iter {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(ns_info["iter"], ns_info["atol"], ns_info["rtol"]))
          sys.exit(-1)
          return u
        
        u1[:,:] = u[:,:]
        u1[:,self.fluid_idx] += du_fluid
        
      elif (ts_type == "IMEX"):
        
        ele_idx = self.ele_idx
        ion_idx = self.ion_idx
        Te_idx  = self.Te_idx
        
        ne = u[:, ele_idx]
        ni = u[:, ion_idx]
        ki = self.r_rates[:, ion_idx]
        ki[ki<0] = 0
        
        # E solve
        phi = self.solve_poisson(ne, ni, time)
        E   = -xp.dot(self.Dp, phi)
        
        
        # ns solve (for heavy)
        mu_i    = self.mu[:, ion_idx]
        Di      = self.D[:, ion_idx]
        
        nsLMat        = self.I_Nx + dt * (mu_i * E * self.Dp + mu_i * xp.dot(self.Dp, E) * self.I_Nx - Di * self.Lp) 
        nsLMat[0,:]   = self.I_Nx[0,:]
        nsLMat[-1,:]  = self.I_Nx[-1,:]
        
        nsRhs         = ni + dt * (ki * self.param.n0 * ne)
        nsRhs[0]      = -xp.dot(self.Dp[0 , 1:]  , ni[1:])   / self.Dp[0 , 0]
        nsRhs[-1]     = -xp.dot(self.Dp[-1, 0:-1], ni[0:-1]) / self.Dp[-1,-1]
        
        ni            = xp.linalg.solve(nsLMat, nsRhs)
        #ni            = xp.dot(self.ns_imex_lmat_inv, nsRhs)
        
        u1            = xp.empty_like(u)
        u1[:,ion_idx] = ni
        u1[:,ele_idx] = u[:,ele_idx]
        u1[:,Te_idx]  = u[:,Te_idx]
        
      if PROFILE_SOLVERS==1:
        if xp == cp:
          cp.cuda.runtime.deviceSynchronize()
        t2 = perf_counter()
        print("fluid solver step time = %.4E (s)" %(t2-t1), flush=True)
        
      return u1
      
    def step_bte_v(self, u, du, time, dt, ts_type, verbose=0):
      xp      = self.xp_module
      
      if PROFILE_SOLVERS==1:
        if xp == cp:
          cp.cuda.runtime.deviceSynchronize()
        t1 = perf_counter()
      
      time    = time
      dt      = dt
      
      rhs     = self.rhs_bte_v
      
      if ts_type == "RK2":
        k1 = dt * rhs(u, time, dt)[0]
        k2 = dt * rhs(u + 0.5 * k1, time + 0.5 * dt, dt)[0]
        return u + k2
      elif ts_type == "RK4":
        k1 = dt * rhs(u           , time           , dt)[0]
        k2 = dt * rhs(u + 0.5 * k1, time + 0.5 * dt, dt)[0]
        k3 = dt * rhs(u + 0.5 * k2, time + 0.5 * dt, dt)[0]
        k4 = dt * rhs(u +  k3     , time + dt      , dt)[0]
        return u + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
      elif ts_type == "BE":
        rtol            = self.args.rtol
        atol            = self.args.atol
        iter_max        = self.args.max_iter
        use_gmres       = True
        
        dof_v           = self.dof_v
        Imat            = self.I_Nxv_stacked
        
        # if (time==0.0):
        #   self.bte_precond_mat = self.bte_assemble_precond(dt)
        
        cp.cuda.runtime.deviceSynchronize()
        a_t1 = perf_counter()
        rhs_j , bc_j    = self.rhs_bte_v_jacobian(u, time, dt)
        Lmat            = Imat - dt * rhs_j
        
        steps_cycle     = int(1/dt)
        pmat_freq       = steps_cycle//50
        step            = int(time/dt)
        
        if use_gmres == True:
          if (step % pmat_freq ==0):
            print("resetting the precond mat")
            self.bte_pmat = xp.linalg.inv(Lmat)
          Pmat = self.bte_pmat   
          
        cp.cuda.runtime.deviceSynchronize()
        a_t2 = perf_counter()
        
        cp.cuda.runtime.deviceSynchronize()
        s_t1 = perf_counter()
        if use_gmres == True:
          uT              = xp.transpose(u)
          def Lmat_mvec(x):
            y = xp.einsum("ijk,ik->ij", Lmat, x.reshape((Lmat.shape[0], Lmat.shape[1])))
            return y.reshape((-1))
          
          def Mmat_mvec(x):
            y = xp.einsum("ijk,ik->ij", Pmat, x.reshape((Pmat.shape[0], Pmat.shape[1])))
            return y.reshape((-1))
          
          Ndof      = Lmat.shape[0] * Lmat.shape[1]
          Lmat_op   = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Lmat_mvec)
          Mmat_op   = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Mmat_mvec)
          v, status = cupyx.scipy.sparse.linalg.gmres(Lmat_op, uT.reshape((-1)), x0=uT.reshape((-1)), tol=1e-10, atol=1e-32, M=Mmat_op, maxiter=100)
          
          norm_b    = xp.linalg.norm(uT.reshape((-1)))
          norm_res  = xp.linalg.norm(Lmat_mvec(v) -  uT.reshape((-1))) / norm_b
          
          if (status !=0) :
            print("GMRES solver failed, using LU factored inverse")
            self.bte_pmat = xp.linalg.inv(Lmat)
            v             = xp.einsum("ijk,ki->ji", self.bte_pmat, u)
            norm_res      = xp.linalg.norm(Lmat_mvec(xp.transpose(v).reshape((-1))) -  uT.reshape((-1))) / norm_b
          else:
            v         = xp.transpose(v.reshape((Lmat.shape[0], Lmat.shape[1])))
        else:
          v        = xp.einsum("ijk,ki->ji", xp.linalg.inv(Lmat), u)
          norm_res = xp.linalg.norm(xp.einsum("ijk,ki->ji", Lmat, v) - u) / xp.linalg.norm(u.reshape(-1))
        
        cp.cuda.runtime.deviceSynchronize()
        s_t2 = perf_counter()
        print("%08d Boltzmann step time = %.6E op. assembly =%.6E solve = %.6E res=%.12E"%(step, time, (a_t2-a_t1), (s_t2-s_t1), norm_res))
        return v 
        
        
        # return 
        
        
        
        # rhs_j , bc_j    = self.rhs_bte_v_jacobian(u, time, dt)
        # A               = Imat - dt * rhs_j
        # A               = xp.linalg.inv(A)
        # v               = xp.einsum("ijk,ki->ji", A, u)
        # return v 
        
        # #rhs, bc = self.rhs_bte_v(u, time, dt)
        
        
        # counter=gmres_counter()
        # v = xp.zeros_like(u)
        # v[:,0] , exitcode = cupyx.scipy.sparse.linalg.gmres(A[0], u[:,0], tol=1e-10, callback=counter)
        # print("spatial loc=%d exitcode=%d"%(0, exitcode))
        # for i in range(1, self.Np):
        #   counter=gmres_counter()
        #   v[:,i], exitcode = cupyx.scipy.sparse.linalg.gmres(A[i], u[:,i], x0=v[:,i-1], tol=1e-10, callback=counter)
        #   print("spatial loc=%d exitcode=%d"%(i, exitcode))
        # #v = xp.linalg.solve(A, rhs)
        # A = xp.linalg.inv(A)
        # v = xp.einsum("ijk,ki->ji", A, rhs)
        
        # return v          
        # def residual(du):
        #   rhs, bc = self.rhs_bte_v(u + du, time + dt, dt)
        #   res     = du - dt * rhs
          
        #   return res
        
        # def jacobian(du):
        #   rhs_j , bc_j = self.rhs_bte_v_jacobian(u, time, dt)
        #   jac          = Imat - dt * rhs_j
        #   return jac
        
        # ns_info = glow1d_utils.newton_solver_batched(du, self.Np, residual, jacobian, atol, rtol, iter_max, self.args.threads, xp)
        # du = ns_info["x"]
        
        # if(verbose==1):
        #   print("Boltzmann step time = %.2E "%(time))
        #   print("  Newton iter {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(ns_info["iter"], ns_info["atol"], ns_info["rtol"]))
        
        # if (ns_info["status"]==False):
        #   print("Boltzmann step non-linear solver step FAILED!!! try with smaller time step size or increase max iterations")
        #   print("time = %.2E "%(time))
        #   print("  Newton iter {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(ns_info["iter"], ns_info["atol"], ns_info["rtol"]))
        #   return u
        
        if PROFILE_SOLVERS==1:
          if xp == cp:
            cp.cuda.runtime.deviceSynchronize()
          t2 = perf_counter()
          print("BTE v-advection time = %.4E (s)" %(t2-t1), flush=True)
        
        return u + du
      elif ts_type == "IMEX":
        rhs_explicit  =  u + dt * self.param.tau * self.bs_E * xp.dot(self.op_adv_v, u)
        return xp.dot(self.bte_imex_lmat_inv, rhs_explicit)
    
    def step_bte_v1(self, u, du, time, dt, ts_type, verbose=0):
      """
      specialized case to handle spatially homogenous E fields. 
      """
      assert ts_type=="BE"
      xp      = self.xp_module
      
      if PROFILE_SOLVERS==1:
        if xp == cp:
          cp.cuda.runtime.deviceSynchronize()
        t1 = perf_counter()
      
      time    = time
      dt      = dt
      rhs     = self.rhs_bte_v
      
      rtol            = self.args.rtol
      atol            = self.args.atol
      iter_max        = self.args.max_iter
      use_gmres       = True
      
      steps_cycle     = int(1/dt)
      pmat_freq       = steps_cycle//50
      step            = int(time/dt)
      
      cp.cuda.runtime.deviceSynchronize()
      a_t1 = perf_counter()
      
      E               = self.bs_E
      assert (E[0]==E).all()==True, "E field is not spatially homogenous"
      E               = self.bs_E[0]
      
      dof_v           = self.dof_v
      Imat            = self.I_Nv
      
      Jmat            = self.param.tau * self.param.n0 * self.param.np0 * (self.op_col_en + self.param.Tg * self.op_col_gT) + self.param.tau * E * self.op_adv_v
      Lmat            = Imat -dt * Jmat
      
      cp.cuda.runtime.deviceSynchronize()
      a_t2 = perf_counter()
      
      cp.cuda.runtime.deviceSynchronize()
      s_t1 = perf_counter()
      v = xp.dot(xp.linalg.inv(Lmat), u)
      cp.cuda.runtime.deviceSynchronize()
      s_t2 = perf_counter()
      norm_res = xp.linalg.norm(xp.dot(Lmat, v) - u) / xp.linalg.norm(u.reshape(-1))
      print("%08d Boltzmann step time = %.6E op. assembly =%.6E solve = %.6E res=%.12E"%(step, time, (a_t2-a_t1), (s_t2-s_t1), norm_res))
      return v
    
    def step_bte(self, u, du, time, dt, ts_type, verbose=0):
      xp     = self.xp_module
      tt_bte = time
      dt_bte = dt
      v      = u
      
      # # First order splitting
      # v       = self.step_bte_x(v, tt_bte, dt_bte)
      # v_lm    = xp.dot(self.op_po2sh,v)
      # v_lm    = self.step_bte_v(v_lm, None, tt_bte, dt_bte, self.ts_type_bte_v , verbose)
      # v       = xp.dot(self.op_psh2o,v_lm)
      # v[v<0]  = 0
      
      # Strang-Splitting
      v       = self.step_bte_x(v, tt_bte, dt_bte * 0.5)
      v_lm    = xp.dot(self.op_po2sh,v)
      v_lm    = self.step_bte_v(v_lm, None, tt_bte, dt_bte, self.ts_type_bte_v , verbose)
      v       = xp.dot(self.op_psh2o,v_lm)
      v       = self.step_bte_x(v, tt_bte + 0.5 * dt_bte, dt_bte * 0.5)
      return v
    
    def solve(self, Uin, Vin):
      tT              = self.args.cycles
      tt              = 0
      
      dt              = self.args.cfl 
      # dt_bte          = self.args.cfl 
      # dt_fluid        = self.args.cfl 
      
      # bte_steps       = int(dt/dt_bte)
      # fluid_steps     = int(dt/dt_fluid)
      steps           = max(1,int(tT/dt))
      print("T = %.4E RF cycles = %.1E dt = %.4E steps = %d atol = %.2E rtol = %.2E max_iter=%d"%(tT, self.args.cycles, dt, steps, self.args.atol, self.args.rtol, self.args.max_iter))
      
      if self.args.use_gpu == 1: 
        Uin1 = cp.asarray(Uin)
        Vin1 = cp.asarray(Vin)
      else:
        Uin1 = Uin
        Vin1 = Vin
      
      if self.args.use_gpu==1:
        self.copy_operators_H2D(args.gpu_device_id)
        self.xp_module = cp
      else:
        self.xp_module = np
        
      xp             = self.xp_module
      u = xp.copy(Uin1)
      v = xp.copy(Vin1)
      
      du    = xp.zeros_like(u)
      dv    = xp.zeros_like(v)
      dv_lm = xp.zeros((self.dof_v, self.Np))
      
      io_freq  = int(1.00/dt)
      
      dg_qmat  = self.op_diag_dg
      dg_qmatT = xp.transpose(self.op_diag_dg)
      
      ts_idx_b  = 0 
      if args.restore==1:
        ts_idx_b = int(args.rs_idx * io_freq)
        tt       = ts_idx_b * dt
        print("restoring solver from ts_idx = ", int(args.rs_idx * io_freq), "at time = ",tt)
        
      if(self.ts_type_bte_v == "IMEX"):
        lmat = self.I_Nv - dt * self.param.tau * self.param.n0 * self.param.np0 * (self.op_col_en + self.param.Tg * self.op_col_gT)
        self.bte_imex_lmat_inv = xp.linalg.inv(lmat)
      else:
        self.bte_imex_lmat_inv = None
      
      if(self.ts_type_fluid == "IMEX"):
        lmat = self.I_Nx - dt * self.Lp * self.param.Di
        lmat[0, :]  = self.I_Nx[0,:]
        lmat[-1, :] = self.I_Nx[-1,:]
        self.ns_imex_lmat_inv = xp.linalg.inv(lmat)
      else:
        self.ns_imex_lmat_inv = None
      
      self.initialize_bte_adv_x(dt * 0.5)
      for ts_idx in range(ts_idx_b, steps):
        du[:,:]=0
        dv[:,:]=0
        
        if (ts_idx % io_freq == 0):
          print("time = %.2E step=%d/%d"%(tt, ts_idx, steps))
          
        if (ts_idx % io_freq == 0):
          self.plot(u, v, "%s_%04d.png"%(args.fname, ts_idx//io_freq), tt)
          xp.save("%s_%04d_u.npy"%(args.fname, ts_idx//io_freq), u)
          xp.save("%s_%04d_v.npy"%(args.fname, ts_idx//io_freq), v)
          
          Vin_lm   = xp.dot(self.op_po2sh, v)
          Vin_lm1  = self.bte_eedf_normalization(Vin_lm)
          Vin_lm1  = xp.asnumpy(Vin_lm1)
      
          vth       = self.bs_vth
          kx_max    = self.op_spec_sp._basis_p._t_unique[-1]
          ev_range  = (self.ev_lim[0] + 1e-6, (kx_max * vth/self.c_gamma)**2 - 1e-6)
          ev_grid   = np.linspace(ev_range[0], ev_range[1], 1024)    
          ff_v      = self.compute_radial_components(ev_grid, Vin_lm1)
          xp.save("%s_%04d_v_eedf.npy"%(args.fname, ts_idx//io_freq), ff_v)
        
        
        # second-order split scheme
        u = self.step_fluid(u, du, tt, dt * 0.5,  self.ts_type_fluid, int(ts_idx % io_freq == 0))
        self.fluid_to_bte(u, v, tt, dt)
        v = self.step_bte(v, dv, tt, dt, None, int(ts_idx % io_freq == 0))
        self.bte_to_fluid(u, v, tt + 0.5 * dt, dt)
        u = self.step_fluid(u, du, tt + 0.5 * dt, dt * 0.5,  self.ts_type_fluid, int(ts_idx % io_freq == 0))
        
        # # first order split scheme
        # u = self.step_fluid(u, du, tt, dt, self.ts_type_fluid, int(ts_idx % io_freq == 0))
        # self.fluid_to_bte(u, v, tt, dt)
        # v = self.step_bte(v, dv, tt, dt, None, int(ts_idx % io_freq == 0))
        # self.bte_to_fluid(u, v, tt + dt, dt)         # bte to fluid
        tt+= dt
        
        
      return u, v
    
    def solve_unit_test1(self, Uin, Vin):
      """_summary_ This will test the advection effects in the full phase space, no ions involved
      E can be specified to be constant or oscillatory field. 

      Args:
          Uin (_type_): _description_
          Vin (_type_): _description_
      """
      dt              = self.args.cfl 
      tT              = self.args.cycles
      tt              = 0
      steps           = max(1,int(tT/dt))
      
      print("++++ Using backward Euler ++++")
      print("T = %.4E RF cycles = %.1E dt = %.4E steps = %d atol = %.2E rtol = %.2E max_iter=%d"%(tT, self.args.cycles, dt, steps, self.args.atol, self.args.rtol, self.args.max_iter))
      
      if self.args.use_gpu == 1: 
        Uin1 = cp.asarray(Uin)
        Vin1 = cp.asarray(Vin)
      else:
        Uin1 = Uin
        Vin1 = Vin
      
      if self.args.use_gpu==1:
        self.copy_operators_H2D(args.gpu_device_id)
        self.xp_module = cp
      else:
        self.xp_module = np
        
      xp             = self.xp_module
      
      u = xp.copy(Uin1)
      v = xp.copy(Vin1)
      
      du    = xp.zeros_like(u)
      dv    = xp.zeros_like(v)
      dv_lm = xp.zeros((self.dof_v, self.Np))
      
      io_freq  = 1000#int(1/dt)
      
      
      
      dg_qmat  = self.op_diag_dg
      dg_qmatT = xp.transpose(self.op_diag_dg)
      
      self.bs_E = xp.zeros_like(self.xp)
      ts_idx_b  = 0 
      if args.restore==1:
        ts_idx_b = int(args.rs_idx * io_freq)
        tt  = ts_idx_b * dt
        print("restoring solver from ts_idx = ", int(args.rs_idx * io_freq), "at time = ",tt)
      
      self.initialize_bte_adv_x(dt)
      for ts_idx in range(ts_idx_b,steps):
        du[:,:]=0
        dv[:,:]=0
        
        if (ts_idx % io_freq == 0):
          print("time = %.2E step=%d/%d"%(tt, ts_idx, steps))
          
        if (ts_idx % io_freq == 0):
          self.plot_unit_test1(u, v, "%s_%04d.png"%(args.fname, ts_idx//io_freq), tt, self.bs_E)
          np.save("%s_%04d_u.npy"%(args.fname, ts_idx//io_freq), u)
          np.save("%s_%04d_v.npy"%(args.fname, ts_idx//io_freq), v)
        
        v   = self.step_bte_x(v, tt, dt)
        tt+= dt
      return u, v
    
    def solve_unit_test2(self, Uin, Vin):
      dt              = self.args.cfl
      dt_bte          = self.args.cfl #* self.ts_op_split_factor
      tT              = self.args.cycles
      tt              = 0
      bte_steps       = int(dt/dt_bte)
      steps           = max(1,int(tT/dt))
      
      print("++++ Using backward Euler ++++")
      print("T = %.4E RF cycles = %.1E dt = %.4E steps = %d atol = %.2E rtol = %.2E max_iter=%d"%(tT, self.args.cycles, dt, steps, self.args.atol, self.args.rtol, self.args.max_iter))
      
      if self.args.use_gpu == 1: 
        Uin1 = cp.asarray(Uin)
        Vin1 = cp.asarray(Vin)
      else:
        Uin1 = Uin
        Vin1 = Vin
      
      if self.args.use_gpu==1:
        self.copy_operators_H2D(args.gpu_device_id)
        self.xp_module = cp
      else:
        self.xp_module = np
        
      xp             = self.xp_module
      
      u = xp.copy(Uin1)
      v = xp.copy(Vin1)
      
      du    = xp.zeros_like(u)
      dv    = xp.zeros_like(v)
      dv_lm = xp.zeros((self.dof_v, self.Np))
      
      io_cycle = 0.1
      io_freq  = int(io_cycle/dt)
      
      # dg_qmat  = self.op_diag_dg
      # dg_qmatT = xp.transpose(self.op_diag_dg)
      
      #self.bs_E       = 400 * xp.sin(2 * xp.pi * xp.asarray(self.xp)) #-xp.ones(len(self.xp)) * 400
      Emax             = 1e4
      Ex               = Emax * xp.ones_like(self.xp) #xp.asarray(self.xp)**7
      # self.bs_E       = xp.ones(len(self.xp)) * Emax
      
      # Et = xp.zeros((1000, self.Np))
      # for i in range(1000):
      #   #print(" loading " +  "ss_1d2v_bte/1d_glow_%04d.npy"%(i))
      #   uf    = xp.load("1d2v_bte/ss_1d2v_bte/1d_glow_%04d.npy"%(i))
      #   phi   = self.solve_poisson(uf[:, 0], uf[:,1], i * dt)
      #   Et[i] = -xp.dot(self.Dp, phi) * (self.param.V0 / self.param.L)
      
      ts_idx_b        = 0
      if args.restore==1:
        ts_idx_b = int(args.rs_idx * io_freq)
        tt       = ts_idx_b * dt
        print("restoring solver from ts_idx = ", int(args.rs_idx * io_freq), "at time = ",tt)
        
      if(self.ts_type_bte_v == "IMEX"):
        lmat = self.I_Nv - dt_bte * self.param.tau * self.param.n0 * self.param.np0 * (self.op_col_en + self.param.Tg * self.op_col_gT)
        self.bte_imex_lmat_inv = xp.linalg.inv(lmat)
      else:
        self.bte_imex_lmat_inv = None
      
      if(self.ts_type_fluid == "IMEX"):
        lmat = self.I_Nx - dt * self.Lp * self.param.Di
        lmat[0, :]  = self.I_Nx[0,:]
        lmat[-1, :] = self.I_Nx[-1,:]
        self.ns_imex_lmat_inv = xp.linalg.inv(lmat)
      else:
        self.ns_imex_lmat_inv = None
        
      self.initialize_bte_adv_x(dt_bte * 0.5)
      cycle_avg_u=xp.zeros_like(u)
      cycle_avg_v=xp.zeros_like(v)
      
      self.bte_to_fluid(u, v, tt, dt)
      for ts_idx in range(ts_idx_b, steps):
        du[:,:]=0
        dv[:,:]=0
        #self.bs_E       = Et[ts_idx % 1000]  #xp.ones(len(self.xp)) * Emax * xp.sin(2* xp.pi * tt)
        self.bs_E        = Ex * xp.sin(2* xp.pi * tt)
        #cycle_avg_u += u
        cycle_avg_v += v
          
        if (ts_idx % io_freq == 0):
          print("time = %.2E step=%d/%d"%(tt, ts_idx, steps))
          self.plot_unit_test1(u, v,           "%s_%04d.png"%(args.fname, ts_idx//io_freq), tt, (self.bs_E * self.param.L /self.param.V0), plot_ionization=True)
          xp.save("%s_%04d_u.npy"%(args.fname, ts_idx//io_freq), u)
          xp.save("%s_%04d_v.npy"%(args.fname, ts_idx//io_freq), v)
          
          if ts_idx>ts_idx_b:
            cycle_avg_v *= 0.5 * dt / io_cycle
            self.bte_to_fluid(cycle_avg_u, cycle_avg_v, tt, dt)
            self.plot_unit_test1(cycle_avg_u, cycle_avg_v, "%s_avg_%04d.png"%(args.fname, ts_idx//io_freq), tt, (self.bs_E * self.param.L /self.param.V0), plot_ionization=True)
            xp.save("%s_%04d_u_avg.npy"%(args.fname, ts_idx//io_freq), cycle_avg_u)
            xp.save("%s_%04d_v_avg.npy"%(args.fname, ts_idx//io_freq), cycle_avg_v)
            
            cycle_avg_u[:,:] = 0
            cycle_avg_v[:,:] = 0
          else:
            xp.save("%s_%04d_u_avg.npy"%(args.fname, ts_idx//io_freq), u)
            xp.save("%s_%04d_v_avg.npy"%(args.fname, ts_idx//io_freq), v)
            
          
          Vin_lm   = xp.dot(self.op_po2sh, v)
          Vin_lm1  = self.bte_eedf_normalization(Vin_lm)
          Vin_lm1  = xp.asnumpy(Vin_lm1)
      
          vth       = self.bs_vth
          #ev_range  = (self.ev_lim[0] + 0.1, self.ev_lim[1]) #((1e-1 * vth /self.c_gamma)**2, (self.vth_fac * vth /self.c_gamma)**2)
          kx_max    = self.op_spec_sp._basis_p._t_unique[-1]
          #ev_range  = (self.ev_lim[0] + 1e-6, (kx_max * vth/self.c_gamma)**2 - 1e-6) #((1e-1 * vth /self.c_gamma)**2, (self.vth_fac * vth /self.c_gamma)**2)
          ev_range  = (self.ev_lim[0], self.ev_lim[1] * 4) #((1e-1 * vth /self.c_gamma)**2, (self.vth_fac * vth /self.c_gamma)**2)
          ev_grid   = np.linspace(ev_range[0], ev_range[1], 1024)    
          ff_v      = self.compute_radial_components(ev_grid, Vin_lm1)
          xp.save("%s_%04d_v_eedf.npy"%(args.fname, ts_idx//io_freq), ff_v)
        
        #v            = self.step_bte(v, dv, tt, dt, None, 0)
        
        # # First order splitting
        # v       = self.step_bte_x(v, tt_bte, dt_bte)
        # v_lm    = xp.dot(self.op_po2sh,v)
        # v_lm    = self.step_bte_v(v_lm, None, tt_bte, dt_bte, self.ts_type_bte_v , verbose)
        # v       = xp.dot(self.op_psh2o,v_lm)
        # v[v<0]  = 0
        
        # Strang-Splitting
        v       = self.step_bte_x(v, tt, dt_bte * 0.5)
        v_lm    = xp.dot(self.op_po2sh,v)
        v_lm    = self.step_bte_v1(v_lm, None, tt, dt_bte, self.ts_type_bte_v , 0)
        v       = xp.dot(self.op_psh2o,v_lm)
        v       = self.step_bte_x(v, tt + 0.5 * dt_bte, dt_bte * 0.5)
        
        cycle_avg_v += v
        tt+= dt
        self.bte_to_fluid(u, v, tt, dt)
        
      return u, v
    
    def compute_radial_components(self, ev: np.array, ff):
        ff_cpu = ff
        
        ff_cpu   = np.transpose(ff_cpu)
        vth      = self.bs_vth
        spec_sp  = self.op_spec_sp
        
        vr       = np.sqrt(ev) * self.c_gamma/ vth
        num_p    = spec_sp._p +1 
        num_sh   = len(spec_sp._sph_harm_lm)
        n_pts    = ff.shape[1]
        
        output   = np.zeros((n_pts, num_sh, len(vr)))
        Vqr      = spec_sp.Vq_r(vr,0,1)
        
        for l_idx, lm in enumerate(spec_sp._sph_harm_lm):
                output[:, l_idx, :] = np.dot(ff_cpu[:,l_idx::num_sh], Vqr)

        return output
    
    def plot_unit_test1(self, Uin, Vin, fname, time, E, plot_ionization=True):
      xp  = self.xp_module
      fig = plt.figure(figsize=(26,10), dpi=300)
      
      ele_idx = self.ele_idx
      ion_idx = self.ion_idx
      Te_idx  = self.Te_idx
      
      num_p   = self.op_spec_sp._p + 1
      num_sh  = len(self.op_spec_sp._sph_harm_lm)
      Vin_lm  = xp.dot(self.op_po2sh, Vin)
      
      def asnumpy(a):
        if self.xp_module == cp:
          return cp.asnumpy(a)
        else:
          return a
      
      ne = asnumpy(Uin[:, ele_idx])
      ni = asnumpy(Uin[:, ion_idx])
      Te = asnumpy(Uin[:, Te_idx])
      E  = asnumpy(E)
      
      plt.subplot(2, 4, 1)
      plt.plot(self.xp, self.param.np0 * ne, 'b', label=r"$n_e$")
      #plt.plot(self.xp, self.param.np0 * ni, '--r', label=r"$n_i$")
      plt.xlabel(r"x/L")
      plt.ylabel(r"$density (m^{-3})$")
      plt.grid(visible=True)
      plt.legend()
      
      plt.subplot(2, 4, 2)
      plt.plot(self.xp, self.param.np0 * (ni-ne), 'b')
      plt.xlabel(r"x/L")
      plt.ylabel(r"$(n_i -n_e)(m^{-3})$")
      plt.grid(visible=True)
      
      plt.subplot(2, 4, 3)
      plt.plot(self.xp, Te, 'b')
      plt.xlabel(r"x/L")
      plt.ylabel(r"$T_e (eV)$")
      plt.grid(visible=True)
      
      plt.subplot(2, 4, 4)
      phi = - (E) * self.xp #np.zeros_like(self.xp)   
      #E   = np.zeros_like(self.xp)  
      plt.plot(self.xp, E * self.param.V0/self.param.L, 'b')
      plt.xlabel(r"x/L")
      plt.ylabel(r"$E (V/m)$")
      plt.grid(visible=True)
      
      plt.subplot(2, 4, 5)
      plt.plot(self.xp, phi * self.param.V0, 'b')
      plt.xlabel(r"x/L")
      plt.ylabel(r"$\phi (V)$")
      plt.grid(visible=True)
      
      
      num_p   = self.op_spec_sp._p + 1
      num_sh  = len(self.op_spec_sp._sph_harm_lm)
      
      plt.subplot(2, 4, 6)
      Vin_lm1 = self.bte_eedf_normalization(Vin_lm)
      
      r_elastic    = asnumpy(xp.dot(self.op_rate[0], Vin_lm1[0::num_sh,:])) 
      plt.semilogy(self.xp, r_elastic    , 'b', label="elastic")
      
      if plot_ionization:
        if (len(self.op_rate) > 1):
          r_ionization = asnumpy(xp.dot(self.op_rate[1], Vin_lm1[0::num_sh,:]))
          plt.semilogy(self.xp, r_ionization , 'r', label="ionization")
      
      plt.xlabel(r"x/L")
      plt.ylabel(r"rate coefficients ($m^3 s^{-1}$)")
      plt.legend()
      plt.grid(visible=True)
      
      Vin_lm1   = asnumpy(Vin_lm1)
      vth       = self.bs_vth
      #ev_range  = (self.ev_lim[0] + 0.1, self.ev_lim[1]) #((1e-1 * vth /self.c_gamma)**2, (self.vth_fac * vth /self.c_gamma)**2)
      kx_max    = self.op_spec_sp._basis_p._t_unique[-1]
      #ev_range  = (self.ev_lim[0] + 1e-6, (kx_max * vth/self.c_gamma)**2 - 1e-6) #((1e-1 * vth /self.c_gamma)**2, (self.vth_fac * vth /self.c_gamma)**2)
      ev_range  = (self.ev_lim[0], self.ev_lim[1] * 4) #((1e-1 * vth /self.c_gamma)**2, (self.vth_fac * vth /self.c_gamma)**2)
      print(ev_range)
      ev_grid   = np.linspace(ev_range[0], ev_range[1], 1024)
      
      ff_v      = self.compute_radial_components(ev_grid, Vin_lm1)
      
      pts = 2
      plt.subplot(2, 4, 7)
      for i in range(pts):
        plt.semilogy(ev_grid, np.abs(ff_v[i][0]),'-', label="x=%.4f"%(self.xp[i]))
      for i in range(pts):
        plt.semilogy(ev_grid, np.abs(ff_v[-1-i][0]) , '--', label="x=%.4f"%(self.xp[-1-i]))
      
      idx = (np.abs(self.xp - 0)).argmin()
      plt.semilogy(ev_grid, np.abs(ff_v[idx][0]), '-', label="x=%.4f"%(self.xp[idx]))
      plt.semilogy(ev_grid, np.abs(ff_v[idx+1][0]), '-', label="x=%.4f"%(self.xp[idx+1]))
      plt.xlabel(r"energy (eV)")
      plt.ylabel(r"abs(f0) $eV^{-3/2}$")
      plt.legend()
      plt.grid(visible=True)
      
      
      plt.subplot(2, 4, 8)
      for i in range(pts):
        plt.semilogy(ev_grid, np.abs(ff_v[i][1])    ,'-', label="x=%.4f"%(self.xp[i]))
      for i in range(pts):
        plt.semilogy(ev_grid, np.abs(ff_v[-1-i][1]) ,'--', label="x=%.4f"%(self.xp[-1-i]))
        
      plt.semilogy(ev_grid, np.abs(ff_v[idx][1]), '-', label="x=%.4f"%(self.xp[idx]))
      plt.semilogy(ev_grid, np.abs(ff_v[idx+1][1]), '-', label="x=%.4f"%(self.xp[idx+1]))
      plt.xlabel(r"energy (eV)")
      plt.ylabel(r"abs(f1) $eV^{-3/2}$")
      plt.legend()
      plt.grid(visible=True)
      
      
      plt.tight_layout()
      plt.suptitle("T=%.4f cycles"%(time))
      fig.savefig(fname)
      plt.close()
    
    def plot(self, Uin, Vin, fname, time):
      xp = self.xp_module
      fig= plt.figure(figsize=(26,10), dpi=300)
      
      def asnumpy(a):
        if self.xp_module == cp:
          return cp.asnumpy(a)
        else:
          return a
      
      ne = asnumpy(Uin[:, self.ele_idx])
      ni = asnumpy(Uin[:, self.ion_idx])
      Te = asnumpy(Uin[:, self.Te_idx])
      
      plt.subplot(2, 4, 1)
      plt.plot(self.xp, self.param.np0 * ne, 'b', label=r"$n_e$")
      plt.plot(self.xp, self.param.np0 * ni, '--r', label=r"$n_i$")
      plt.xlabel(r"x/L")
      plt.ylabel(r"$density (m^{-3})$")
      plt.grid(visible=True)
      plt.legend()
      
      plt.subplot(2, 4, 2)
      plt.plot(self.xp, self.param.np0 * (ni-ne), 'b')
      plt.xlabel(r"x/L")
      plt.ylabel(r"$(n_i -n_e)(m^{-3})$")
      plt.grid(visible=True)
      
      plt.subplot(2, 4, 3)
      plt.plot(self.xp, Te, 'b')
      plt.xlabel(r"x/L")
      plt.ylabel(r"$T_e (eV)$")
      plt.grid(visible=True)
      
      plt.subplot(2, 4, 4)
      phi = self.solve_poisson(Uin[:,0], Uin[:,1], time)
      E = -xp.dot(self.Dp, phi)
      
      phi = asnumpy(phi)
      E   = asnumpy(E) 
      
      plt.plot(self.xp, E * self.param.V0/self.param.L, 'b')
      plt.xlabel(r"x/L")
      plt.ylabel(r"$E (V/m)$")
      plt.grid(visible=True)
      
      plt.subplot(2, 4, 5)
      plt.plot(self.xp, phi * self.param.V0, 'b')
      plt.xlabel(r"x/L")
      plt.ylabel(r"$\phi (V)$")
      plt.grid(visible=True)
      
      
      num_p   = self.op_spec_sp._p + 1
      num_sh  = len(self.op_spec_sp._sph_harm_lm)
      
      Vin_lm  = xp.dot(self.op_po2sh, Vin)
      
      plt.subplot(2, 4, 6)
      Vin_lm1      = self.bte_eedf_normalization(Vin_lm)
      r_elastic    = asnumpy(xp.dot(self.op_rate[0], Vin_lm1[0::num_sh,:]))
      plt.semilogy(self.xp, r_elastic    , 'b', label="elastic")
      
      if (len(self.op_rate) > 1):
        r_ionization = asnumpy(xp.dot(self.op_rate[1], Vin_lm1[0::num_sh,:]))
        plt.semilogy(self.xp, r_ionization , 'r', label="ionization")
      
      plt.xlabel(r"x/L")
      plt.ylabel(r"rate coefficients ($m^3 s^{-1}$)")
      plt.legend()
      plt.grid(visible=True)
      
      vth       = self.bs_vth
      kx_max    = self.op_spec_sp._basis_p._t_unique[-1]
      ev_range  = (self.ev_lim[0] + 1e-6, (kx_max * vth/self.c_gamma)**2 - 1e-6) #((1e-1 * vth /self.c_gamma)**2, (self.vth_fac * vth /self.c_gamma)**2)
      #ev_range  = (self.ev_lim[0] + 0.1, self.ev_lim[1]-0.1) #((1e-1 * vth /self.c_gamma)**2, (self.vth_fac * vth /self.c_gamma)**2)
      
      ev_grid   = np.linspace(ev_range[0], ev_range[1], 500)
      
      ff_v      = self.compute_radial_components(ev_grid, asnumpy(Vin_lm1))
      
      pts = 2
      plt.subplot(2, 4, 7)
      for i in range(pts):
        plt.semilogy(ev_grid, np.abs(ff_v[i][0]),'-', label="x=%.4f"%(self.xp[i]))
      for i in range(pts):
        plt.semilogy(ev_grid, np.abs(ff_v[-1-i][0]) , '--', label="x=%.4f"%(self.xp[-1-i]))
      
      idx = (np.abs(self.xp - 0)).argmin()
      plt.semilogy(ev_grid, np.abs(ff_v[idx][0]), '-', label="x=%.4f"%(self.xp[idx]))
      plt.semilogy(ev_grid, np.abs(ff_v[idx+1][0]), '-', label="x=%.4f"%(self.xp[idx+1]))
      plt.xlabel(r"energy (eV)")
      plt.ylabel(r"abs(f0) $eV^{-3/2}$")
      plt.legend()
      plt.grid(visible=True)
      
      
      plt.subplot(2, 4, 8)
      for i in range(pts):
        plt.semilogy(ev_grid, np.abs(ff_v[i][1])    ,'-', label="x=%.4f"%(self.xp[i]))
      for i in range(pts):
        plt.semilogy(ev_grid, np.abs(ff_v[-1-i][1]) ,'--', label="x=%.4f"%(self.xp[-1-i]))
        
      plt.semilogy(ev_grid, np.abs(ff_v[idx][1]), '-', label="x=%.4f"%(self.xp[idx]))
      plt.semilogy(ev_grid, np.abs(ff_v[idx+1][1]), '-', label="x=%.4f"%(self.xp[idx+1]))
      plt.xlabel(r"energy (eV)")
      plt.ylabel(r"abs(f1) $eV^{-3/2}$")
      plt.legend()
      plt.grid(visible=True)
      
      
      plt.tight_layout()
      plt.suptitle("T=%.4f cycles"%(time))
      fig.savefig(fname)
      plt.close()
    
    def rhs_bte_vx(self, vd: np.array, time, dt, ts_type):
      xp         = self.xp_module
      E          = self.bs_E
      
      # v  - (self.Nr * self.Nvt, self.Np)
      # vd - diagonalized Vin
      vd    = vd.reshape((self.Nr, self.Nvt , self.Np))
      
      # vd    = vd.reshape((self.Nr * self.Nvt, self.Np))
      # vd[self.xp_vt_l, 0]  = 0.0
      # vd[self.xp_vt_r, -1] = 0.0
      # vd    = vd.reshape((self.Nr, self.Nvt , self.Np))
      
      vd_x  = xp.einsum('ia,jka->jki', self.Dp, vd)
      Fv1   = xp.einsum('i,j,ijk->ijk', self.op_adv_x_d, self.xp_cos_vt, vd_x)
      
      v     = xp.einsum('ia,ajk->ijk', self.op_adv_x_q, vd).reshape((self.Nr * self.Nvt, self.Np))
      v_lm  = xp.dot(self.op_po2sh, v)
      
      Fv2   = self.param.tau * (self.param.n0 * self.param.np0 * (xp.dot(self.op_col_en, v_lm) + self.param.Tg * xp.dot(self.op_col_gT, v_lm))  + E * xp.dot(self.op_adv_v, v_lm))
      Fv2   = xp.dot(self.op_psh2o, Fv2)
      Fv2   = xp.einsum('ia,ajk->ijk', self.op_adv_x_qinv, Fv2.reshape((self.Nr, self.Nvt, self.Np)))
      
      Fv    = Fv1 - Fv2
      
      Fv    = Fv.reshape((self.Nr * self.Nvt, self.Np))
      Fv[self.xp_vt_l, 0]  = 0.0
      Fv[self.xp_vt_r, -1] = 0.0
      return  Fv.reshape((-1))
    
    def precond_bte_vx(self, vd:np.array, Mv_mat, time, dt, ts_type):
      xp        = self.xp_module
      tt_bte    = time
      dt_bte    = dt
      
      vd        = xp.copy(vd)
      vd        = vd.reshape((self.Nr * self.Nvt, self.Np))
      
      # vd[self.xp_vt_l, 0]  = 0.0
      # vd[self.xp_vt_r, -1] = 0.0
      # vd                   = xp.einsum('ijkl,ijl->ijk',self.bte_x_shift, vd.reshape((self.Nr, self.Nvt, self.Np)))
      
      vd                   = xp.einsum('il,ljk->ijk', self.op_adv_x_q, vd.reshape((self.Nr, self.Nvt, self.Np))).reshape((self.Nr * self.Nvt, self.Np))
      v_lm                 = xp.dot(self.op_po2sh,vd)
      v_lm                 = xp.einsum('ijk,ki->ji', Mv_mat, v_lm) #self.step_bte_v(v_lm, None, tt_bte, dt_bte, self.ts_type_bte_v , 0)
      vd                   = xp.dot(self.op_psh2o,v_lm).reshape((self.Nr, self.Nvt, self.Np))
      vd                   = xp.einsum('il,ljk->ijk', self.op_adv_x_qinv, vd).reshape((self.Nr * self.Nvt, self.Np))
      
      vd = vd.reshape((self.Nr *  self.Nvt, self.Np))
      vd[self.xp_vt_l, 0]  = 0.0
      vd[self.xp_vt_r, -1] = 0.0
      vd                   = xp.einsum('ijkl,ijl->ijk',self.bte_x_shift, vd.reshape((self.Nr, self.Nvt, self.Np)))
      
      return vd.reshape((-1))
    
    def step_bte_implicit(self, u, du, time, dt, ts_type, verbose):
      xp        = self.xp_module
      Ndof      = self.Nr * self.Nvt * self.Np
      
      Lmat_v     = self.I_Nxv_stacked - dt * self.rhs_bte_v_jacobian(None, time, dt)[0]
      Lmat_v_inv = xp.linalg.inv(Lmat_v)
      
      Lmat_mvec = lambda x : x + dt * self.rhs_bte_vx(x,  time, dt, None)
      Mmat_mvec = lambda x : self.precond_bte_vx(x, Lmat_v_inv, time, dt, None)
      
      Lmat_op   = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Lmat_mvec)
      Mmat_op   = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Mmat_mvec)
      
      u[self.xp_vt_l, 0]  = 0.0
      u[self.xp_vt_r, -1] = 0.0
      v, status = cupyx.scipy.sparse.linalg.gmres(Lmat_op, u.reshape((-1)), x0=u.reshape((-1)), tol=1e-8, atol=1e-32, M=Mmat_op)
      # v1 = xp.einsum('ijkl,ijl->ijk', self.bte_x_shift, u.reshape((self.Nr, self.Nvt, self.Np))).reshape((self.Nr * self.Nvt, self.Np))
      # v  =  v.reshape((self.Nr * self.0Nvt, self.Np))
      # v1 = v1.reshape((self.Nr * self.Nvt, self.Np))
      
      res0 = xp.linalg.norm(Lmat_op(v.reshape((-1))) - u.reshape((-1)))/xp.linalg.norm(u.reshape((-1)))
      # res1 = xp.linalg.norm(Lmat_op(v1.reshape((-1))) - u.reshape((-1)))/xp.linalg.norm(u.reshape((-1)))
      print(res0, status)
      # print(xp.linalg.norm(v-v1))
      
      v=v.reshape((self.Nr * self.Nvt, self.Np))
      # print(v[self.xp_vt_l, 0],status)
      # print(v[self.xp_vt_r,-1])
      # plt.figure(figsize=(10,10), dpi=300)
      # plt.semilogy(np.abs(cp.asnumpy(1-v[:,0]/v1[:,0])),'x',label="")
      # #plt.semilogy(np.abs(cp.asnumpy(v1[:,0])),'.',label="gmres")
      # plt.grid()
      # plt.legend()
      # plt.savefig("./1d2v_bte/junk/t_%4E.png"%(time))
      # plt.close()
      if (status > 0) :
        # res_vec = np.abs(cp.asnumpy(Lmat_op(v.reshape((-1))) - u.reshape((-1)))).reshape((self.Nr , self.Nvt, self.Np))
        # from matplotlib.colors import LogNorm
        
        # plt.figure(figsize=(10,10), dpi=200)
        # plt.imshow(res_vec[:,:,0],norm=LogNorm())
        # # plt.semilogy(res_vec[:,0]   , 'x')
        # # plt.semilogy(res_vec[:,100] , '.')
        # # plt.semilogy(res_vec[:,-1]  , 'o')
        # # plt.grid()
        # plt.colorbar()
        # plt.savefig("residual_x0.png")
        
        # plt.figure(figsize=(10,10), dpi=200)
        # plt.imshow(res_vec[:,:,100],norm=LogNorm())
        # # plt.semilogy(res_vec[:,0]   , 'x')
        # # plt.semilogy(res_vec[:,100] , '.')
        # # plt.semilogy(res_vec[:,-1]  , 'o')
        # # plt.grid()
        # plt.colorbar()
        # plt.savefig("residual_x100.png")
        
        # plt.figure(figsize=(10,10), dpi=200)
        # plt.imshow(res_vec[:,:,-1],norm=LogNorm())
        # # plt.semilogy(res_vec[:,0]   , 'x')
        # # plt.semilogy(res_vec[:,100] , '.')
        # # plt.semilogy(res_vec[:,-1]  , 'o')
        # # plt.grid()
        # plt.colorbar()
        # plt.savefig("residual_xL.png")
        
        print("GMRES solver failed, using LU factored inverse")
        #raise RuntimeError
        #rr=Lmat_op(v.reshape((-1))) - u.reshape((-1))
        #return rr.reshape((self.Nr * self.Nvt, self.Np))
      return v.reshape((self.Nr * self.Nvt, self.Np))
    
    def solve_unit_test3(self, Uin, Vin):
      dt              = self.args.cfl
      dt_bte          = self.args.cfl #* self.ts_op_split_factor
      tT              = self.args.cycles
      tt              = 0
      bte_steps       = int(dt/dt_bte)
      steps           = max(1,int(tT/dt))
      
      print("++++ Using backward Euler ++++")
      print("T = %.4E RF cycles = %.1E dt = %.4E steps = %d atol = %.2E rtol = %.2E max_iter=%d"%(tT, self.args.cycles, dt, steps, self.args.atol, self.args.rtol, self.args.max_iter))
      
      if self.args.use_gpu == 1: 
        Uin1 = cp.asarray(Uin)
        Vin1 = cp.asarray(Vin)
      else:
        Uin1 = Uin
        Vin1 = Vin
      
      if self.args.use_gpu==1:
        self.copy_operators_H2D(args.gpu_device_id)
        self.xp_module = cp
      else:
        self.xp_module = np
        
      xp             = self.xp_module
      
      u = xp.copy(Uin1)
      v = xp.copy(Vin1)
      
      du    = xp.zeros_like(u)
      dv    = xp.zeros_like(v)
      dv_lm = xp.zeros((self.dof_v, self.Np))
      
      io_freq  = int(0.1/dt)
      
      # dg_qmat  = self.op_diag_dg
      # dg_qmatT = xp.transpose(self.op_diag_dg)
      
      #self.bs_E       = 400 * xp.sin(2 * xp.pi * xp.asarray(self.xp)) #-xp.ones(len(self.xp)) * 400
      Emax            = 1000
      self.bs_E       = xp.ones(len(self.xp)) * Emax
      
      # Et = xp.zeros((1000, self.Np))
      # for i in range(1000):
      #   #print(" loading " +  "ss_1d2v_bte/1d_glow_%04d.npy"%(i))
      #   uf    = xp.load("1d2v_bte/ss_1d2v_bte/1d_glow_%04d.npy"%(i))
      #   phi   = self.solve_poisson(uf[:, 0], uf[:,1], i * dt)
      #   Et[i] = -xp.dot(self.Dp, phi) * (self.param.V0 / self.param.L)
      
      ts_idx_b        = 0
      if args.restore==1:
        ts_idx_b = int(args.rs_idx * io_freq)
        tt       = ts_idx_b * dt
        print("restoring solver from ts_idx = ", int(args.rs_idx * io_freq), "at time = ",tt)
        
      if(self.ts_type_bte_v == "IMEX"):
        lmat = self.I_Nv - dt_bte * self.param.tau * self.param.n0 * self.param.np0 * (self.op_col_en + self.param.Tg * self.op_col_gT)
        self.bte_imex_lmat_inv = xp.linalg.inv(lmat)
      else:
        self.bte_imex_lmat_inv = None
      
      if(self.ts_type_fluid == "IMEX"):
        lmat = self.I_Nx - dt * self.Lp * self.param.Di
        lmat[0, :]  = self.I_Nx[0,:]
        lmat[-1, :] = self.I_Nx[-1,:]
        self.ns_imex_lmat_inv = xp.linalg.inv(lmat)
      else:
        self.ns_imex_lmat_inv = None
        
      self.initialize_bte_adv_x(dt)
      vd = xp.einsum('ia,ajk->ijk', self.op_adv_x_qinv, v.reshape((self.Nr, self.Nvt, self.Np))).reshape((self.Nr * self.Nvt, self.Np))
      for ts_idx in range(ts_idx_b, steps):
        du[:,:]=0
        dv[:,:]=0
        #self.bs_E       = Et[ts_idx % 1000]  #xp.ones(len(self.xp)) * Emax * xp.sin(2* xp.pi * tt)
        #self.bs_E        = Emax * xp.sin(2* xp.pi * tt) * xp.ones(len(self.xp))
        if (ts_idx % io_freq == 0):
          print("time = %.2E step=%d/%d"%(tt, ts_idx, steps))
          
        if (ts_idx % io_freq == 0):
          v  = xp.einsum('ia,ajk->ijk', self.op_adv_x_q, vd.reshape((self.Nr, self.Nvt, self.Np))).reshape((self.Nr * self.Nvt, self.Np))
          self.plot_unit_test1(u, v, "%s_%04d.png"%(args.fname, ts_idx//io_freq), tt, (self.bs_E * self.param.L /self.param.V0), plot_ionization=True)
          #self.plot_unit_test1(u, v, "%s_%04d.png"%(args.fname, ts_idx//io_freq), tt, (self.bs_E * self.param.L /self.param.V0), plot_ionization=False)
          xp.save("%s_%04d_u.npy"%(args.fname, ts_idx//io_freq), u)
          xp.save("%s_%04d_v.npy"%(args.fname, ts_idx//io_freq), v)
          #vd = xp.einsum('ia,ajk->ijk', self.op_adv_x_qinv, v.reshape((self.Nr, self.Nvt, self.Np))).reshape((self.Nr * self.Nvt, self.Np))
        
        vd = self.step_bte_implicit(vd, None, tt, dt, None, 0)
        tt+= dt
        
        
        
      return u, v
    
    
parser = argparse.ArgumentParser()
parser.add_argument("-threads", "--threads"                       , help="number of cpu threads (boltzmann operator assembly)", type=int, default=4)
parser.add_argument("-out_fname", "--out_fname"                   , help="output file name for the qois", type=str, default="bte_glow1d")
parser.add_argument("-l_max", "--l_max"                           , help="max polar modes in SH expansion", type=int, default=1)
parser.add_argument("-c", "--collisions"                          , help="collisions model",nargs='+', type=str, default=["g0","g2"])
parser.add_argument("-ev_max", "--ev_max"                         , help="energy max v-space grid (eV)" , type=float, default=50)
parser.add_argument("-sp_order", "--sp_order"                     , help="b-spline order", type=int, default=3)
parser.add_argument("-spline_qpts", "--spline_qpts"               , help="q points per knots", type=int, default=5)
parser.add_argument("-steady", "--steady_state"                   , help="steady state or transient", type=int, default=1)
parser.add_argument("-Te", "--Te"                                 , help="approximate electron temperature (eV)"  , type=float, default=0.5)
parser.add_argument("-Tg", "--Tg"                                 , help="approximate gas temperature (eV)"       , type=float, default=0.0)
parser.add_argument("-Nr", "--Nr"                                 , help="radial refinement"  , type=int, default=128)
parser.add_argument("-Nvt", "--Nvt"                               , help="number of ordinates", type=int, default=3)
parser.add_argument("-Ns", "--Ns"                                 , help="number of species (fluid solver)", type=int, default=2)
parser.add_argument("-NT", "--NT"                                 , help="number of temperatures"          , type=int, default=1)
parser.add_argument("-Np", "--Np"                                 , help="number of collocation points"    , type=int, default=100)
parser.add_argument("-cfl", "--cfl"                               , help="CFL factor (only used in explicit integrations)" , type=float, default=1e-1)
parser.add_argument("-cycles", "--cycles"                         , help="number of cycles to run" , type=float, default=10)
parser.add_argument("-ts_type", "--ts_type"                       , help="ts mode" , type=str, default="BE")
parser.add_argument("-atol", "--atol"                             , help="abs. tolerance" , type=float, default=1e-10)
parser.add_argument("-rtol", "--rtol"                             , help="rel. tolerance" , type=float, default=1e-10)
parser.add_argument("-max_iter", "--max_iter"                     , help="max iterations for Newton solver" , type=int, default=1000)

parser.add_argument("-profile", "--profile"                       , help="profile", type=int, default=0)
parser.add_argument("-warm_up", "--warm_up"                       , help="warm up", type=int, default=5)
parser.add_argument("-runs", "--runs"                             , help="runs "  , type=int, default=10)
parser.add_argument("-store_eedf", "--store_eedf"                 , help="store EEDF"          , type=int, default=0)
parser.add_argument("-use_gpu", "--use_gpu"                       , help="use GPUs"            , type=int, default=0)
parser.add_argument("-gpu_device_id", "--gpu_device_id"           , help="GPU device id to use", type=int, default=0)
parser.add_argument("-store_csv", "--store_csv"                   , help="store csv format of QoI comparisons", type=int, default=1)
parser.add_argument("-plot_data", "--plot_data"                   , help="plot data", type=int, default=1)
parser.add_argument("-ee_collisions", "--ee_collisions"           , help="enable electron-electron collisions", type=float, default=0)
parser.add_argument("-verbose", "--verbose"                       , help="verbose with debug information", type=int, default=0)
parser.add_argument("-restore", "--restore"                       , help="restore the solver" , type=int, default=0)
parser.add_argument("-rs_idx",  "--rs_idx"                        , help="restore file_idx"   , type=int, default=0)

parser.add_argument("-fname", "--fname"                           , help="file name to store the solution" , type=str, default="1d_glow")
parser.add_argument("-dir"  , "--dir"                             , help="file name to store the solution" , type=str, default="glow1d_dir")

args = parser.parse_args()
glow_1d = glow1d_boltzmann(args)
u, v    = glow_1d.initialize()

if args.use_gpu==1:
  gpu_device = cp.cuda.Device(args.gpu_device_id)
  gpu_device.use()

uu,vv   = glow_1d.solve(u, v)
#uu,vv   = glow_1d.solve_unit_test2(u, v)
#uu,vv   = glow_1d.solve_unit_test3(u, v)

