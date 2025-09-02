"""
macroscopic/microscopic modeling of the 1d glow discharge problem
1). We use Gauss-Chebyshev-Lobatto co-location method with implicit time integration. 
"""
import numpy as np
import scipy.constants 
import argparse
import matplotlib.pyplot as plt
import sys
import math

import scipy.interpolate
import glow1d_utils
import basis
import collisions
import utils as bte_utils
import basis
import spec_spherical as sp
import collision_operator_spherical as collOpSp
import scipy.constants
import os
import scipy.sparse.linalg
import cross_section
from time import perf_counter, sleep
import glowdischarge_1d
import mesh

CUDA_NUM_DEVICES      = 0
PROFILE_SOLVERS       = 0
try:
  import cupy as cp
  import cupyx.scipy.sparse.linalg
  #CUDA_NUM_DEVICES=cp.cuda.runtime.getDeviceCount()
except ImportError:
  print("Please install CuPy for GPU use")
  #sys.exit(0)
except:
  print("CUDA not configured properly !!!")
  #sys.exit(0)


class gmres_counter(object):
  def __init__(self, disp=True):
      self._disp = disp
      self.niter = 0
  def __call__(self, rk=None):
      self.niter += 1
      if self._disp:
          print('iter %3i\trk = %s' % (self.niter, str(rk)))

class glow1d_fluid_args():
  def __init__(self, args) -> None:
    self.Ns             = args.Ns
    self.NT             = args.NT
    self.Np             = args.Np
    self.cfl            = args.cfl
    self.cycles         = 30
    self.ts_type        = "BE"
    self.atol           = args.atol
    self.rtol           = args.rtol
    self.fname          = args.fname+"_f_"
    self.restore        = 0
    self.rs_idx         = 0
    self.checkpoint     = 0
    self.max_iter       = 400
    self.dir            = ""
    self.use_tab_data   = 0
    self.bc_dirichlet_e = 0
    self.use_gpu        = 0#args.use_gpu
    self.gpu_device_id  = args.gpu_device_id

class bte_xspace_adv_type():
    USE_BE_CHEB   = 0 
    USE_BE_UPW_FD = 1

class glow1d_boltzmann():
    """
    perform glow discharge simulation with electron Boltzmann solver
    """
    def __init__(self, args) -> None:
      
      self.ts_type_fluid       = "BE"
      self.ts_type_bte_v       = "BE"
      self.ts_op_split_factor  = 1/1
      self.heavies_freeze_E    = True
      self.xspace_adv_type     = args.xadv_type #bte_xspace_adv_type.USE_BE_UPW_FD
       
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
      self.param     = glow1d_utils.parameters(self.args)
      
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
      
      self.op            = mesh.mesh([self.Np], 1, mesh.grid_type.CHEBYSHEV_COLLOC)
      self.xp            = self.op.xcoord[0]

      self.Dp            = self.op.D1[0]
      self.DpT           = self.Dp.T
      self.Lp            = self.op.D2[0]

      self.LpD           = np.identity(self.Np)
      self.LpD[1:-1,:]   = self.Lp[1:-1,:]
      
      self.LpD_inv       = np.linalg.solve(self.LpD, np.eye(self.Np))

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
      
      # self.xp_vt, self.xp_vt_qw = basis.Legendre().Gauss_Pn(self.Nvt)
      # self.xp_vt                = np.arccos(self.xp_vt) 

      LpDvt              = mesh.central_dxx(self.xp_vt)
      I_Nvt              = np.eye(len(self.xp_vt))
      LpDvt              = I_Nvt - self.args.vtDe * LpDvt
      
      #dirichlet BCs
      LpDvt[0]           = I_Nvt[0]
      LpDvt[-1]          = I_Nvt[-1]
      
      # # neumann BCs
      # LpDvt[0]         = mesh.upwinded_dx(self.xp_vt, "LtoR")[0]  
      # LpDvt[-1]        = mesh.upwinded_dx(self.xp_vt, "LtoR")[-1] 
      self.LpDvt         = np.linalg.inv(LpDvt)
      
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
      
      self.collision_names              = list()
      self.coll_list                    = list()

      self.avail_species                = cross_section.read_available_species(self.args.collisions)
      cross_section.CROSS_SECTION_DATA  = cross_section.read_cross_section_data(self.args.collisions)
      self.cross_section_data           = cross_section.CROSS_SECTION_DATA
      
      print("==========read collissions=====================")
      collision_count = 0
      for col_str, col_data in self.cross_section_data.items():
          print(col_str, col_data["type"])
          g = collisions.electron_heavy_binary_collision(col_str, collision_type=col_data["type"])
          g.reset_scattering_direction_sp_mat()
          self.coll_list.append(g)
          self.collision_names.append("C%d"%(collision_count)) 
          collision_count+=1
      print("===============================================")
      print("number of total collisions = %d " %(len(self.coll_list)))
      self.num_collisions = len(self.coll_list)
      self.bs_coll_list = self.coll_list
      
      assert (len(self.avail_species)==1), "currently supports for all reactions from 1 common state"
      
      self.initialize_boltzmann()
    
    def initialize_boltzmann(self):
      args                = self.args

      sig_pts             =  list()
      for col_idx, g in enumerate(self.bs_coll_list):
          g  = self.bs_coll_list[col_idx]
          if g._reaction_threshold != None and g._reaction_threshold >0:
              sig_pts.append(g._reaction_threshold)
      
      self._sig_pts          = np.sort(np.array(list(set(sig_pts))))
      sig_pts                = self._sig_pts
        
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
      if (args.ev_extend==0):
        bb                     = basis.BSpline(k_domain, self.args.sp_order, self.bs_nr + 1, sig_pts=dg_nodes, knots_vec=None, dg_splines=self.bs_use_dg, verbose = args.verbose)
      elif (args.ev_extend==1):
        print("using uniform grid extention")
        bb                     = basis.BSpline(k_domain, self.args.sp_order, self.bs_nr + 1, sig_pts=dg_nodes, knots_vec=None, dg_splines=self.bs_use_dg, verbose = args.verbose, extend_domain=True)
      else:
        assert args.ev_extend==2
        print("using log spaced grid extention")
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
      self.op_po2sh             = np.matmul(spec_sp.Vq_sph(self.xp_vt, np.zeros_like(self.xp_vt)), np.diag(self.xp_vt_qw) ) * 2 * np.pi
      #tmp1                     = np.linalg.inv(np.transpose(spec_sp.Vq_sph(self.xp_vt, np.zeros_like(self.xp_vt))))
      
      # spherical lm to ordinates computation (self.Nvt, num_sh)
      self.op_psh2o             = np.transpose(spec_sp.Vq_sph(self.xp_vt, np.zeros_like(self.xp_vt))) 
      
      # to check if the oridinates to spherical and spherical to ordinates projection is true (weak test not sufficient but necessary condition)
      assert np.allclose(np.dot(self.op_psh2o, np.dot(self.op_po2sh, 1 + np.cos(self.xp_vt))), 1 + np.cos(self.xp_vt))
      
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
      for col_idx, (col_str, col_data) in enumerate(self.cross_section_data.items()):
          g = self.bs_coll_list[col_idx]
          g.reset_scattering_direction_sp_mat()
          col = g._col_name
          
          if args.verbose==1:
              print("collision %d  %s %s"%(col_idx, col, col_data["type"]))
              
          if col_data["type"] == "ELASTIC":
            FOp_g     = collision_op.electron_gas_temperature(g, maxwellian, vth, mp_pool_sz=args.threads)
          
          FOp         = FOp + collision_op.assemble_mat(g, maxwellian, vth, tgK=0.0, mp_pool_sz=args.threads)
          sigma_m    += g.total_cross_section(gx_ev)
          
      t2 = perf_counter()
      print("assembly = %.4E"%(t2-t1))
      
      t1 = perf_counter()
      print("bte qoi op assembly")    
      self.op_sigma_m   = sigma_m
      
      self.op_mass      = bte_utils.mass_op(spec_sp, 1) #* maxwellian(0) * vth**3
      self.op_temp      = bte_utils.temp_op(spec_sp, 1) * (vth**2) * 0.5 * scipy.constants.electron_mass * (2/3/scipy.constants.Boltzmann) / collisions.TEMP_K_1EV
      
      # note non-dimentionalized electron mobility and diffusion,  ## not used at the moment, but needs to check.          
      self.op_mobility  = bte_utils.mobility_op(spec_sp, maxwellian, vth) #* self.param.V0 * self.param.tau/self.param.L**2
      self.op_diffusion = bte_utils.diffusion_op(spec_sp, self.bs_coll_list, maxwellian, vth) #* self.param.tau/self.param.L**2
          
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
      
      # sanity check for eigen decomposition
      eig_rtol                  = np.linalg.norm(self.op_adv_x - np.dot(self.op_adv_x_q * self.op_adv_x_d, self.op_adv_x_qinv)) / np.linalg.norm(self.op_adv_x)
      print("Adv_x : ||A - Q D Q^{-1}||/ ||A|| =%.8E"%(eig_rtol))
      
      xp=np
      xp.save("%s_bte_mass_op.npy"   %(args.fname), self.op_mass)
      xp.save("%s_bte_temp_op.npy"   %(args.fname), self.op_temp)
      xp.save("%s_bte_po2sh.npy"     %(args.fname), self.op_po2sh)
      xp.save("%s_bte_psh2o.npy"     %(args.fname), self.op_psh2o)
      xp.save("%s_bte_mobility.npy"  %(args.fname), self.op_mobility)
      xp.save("%s_bte_diffusion.npy" %(args.fname), self.op_diffusion)
      xp.save("%s_bte_op_g0.npy"     %(args.fname), self.op_rate[0])
      
      if (len(self.op_rate) > 1):
        xp.save("%s_bte_op_g2.npy"  %(args.fname), self.op_rate[1])
        
      save_bte_mat = True
      if save_bte_mat == True:
        xp.save("%s_bte_cmat.npy"   %(args.fname), self.op_col_en + self.param.Tg * self.op_col_gT)
        xp.save("%s_bte_emat.npy"   %(args.fname), self.op_adv_v)
        xp.save("%s_bte_xmat.npy"   %(args.fname), self.op_adv_x)
          
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
      
      if self.args.restore == 1:
        
        print("~~~restoring solver from %s.npy"%(args.fname))
        Uin = xp.load("%s_%04d_u.npy"%(args.fname, args.rs_idx))
        Vin = xp.load("%s_%04d_v.npy"%(args.fname, args.rs_idx))
        
        if Uin.shape[0] != len(self.xp):
          print("Using Chebyshev interpolation for the restore checkpoint")
          npts            = Uin.shape[0]
          xx1             = -np.cos(np.pi*np.linspace(0,npts-1, npts)/(npts-1))
          v0pinv          = np.linalg.solve(np.polynomial.chebyshev.chebvander(xx1, npts-1), np.eye(npts))
          P1              = np.dot(np.polynomial.chebyshev.chebvander(self.xp, npts-1), v0pinv)
          Uin             = np.dot(P1, Uin)
          Vin             = np.dot(P1, Vin.T).T
        
      else:
        if (type == 0):
          
          xx = self.param.L * (self.xp + 1)
          read_from_file   = (self.args.ic_file != "")
          if read_from_file==True:
            fname = self.args.ic_file
            print("loading initial conditoin from ", fname)
            u1              = xp.load(fname)
            npts            = u1.shape[0]
            xx1             = -np.cos(np.pi*np.linspace(0,npts-1, npts)/(npts-1))
            # v0pinv          = np.linalg.solve(np.polynomial.chebyshev.chebvander(xx1, npts-1), np.eye(npts))
            # P1              = np.dot(np.polynomial.chebyshev.chebvander(self.xp, npts-1), v0pinv)
            # u1              = np.dot(P1, u1)
            
            u2                  = np.zeros((len(self.xp), self.Nv))
            u2[:, self.ele_idx] = scipy.interpolate.interp1d(xx1, u1[:, ele_idx], kind="linear", bounds_error="True") (self.xp)
            u2[:, self.ion_idx] = scipy.interpolate.interp1d(xx1, u1[:, ion_idx], kind="linear", bounds_error="True") (self.xp)
            u2[:, self.Te_idx]  = scipy.interpolate.interp1d(xx1, u1[:, Te_idx] , kind="linear", bounds_error="True") (self.xp)
            u1                  = u2
            
            Uin[:, ele_idx] = u1[:, ele_idx] 
            Uin[:, ion_idx] = u1[:, ion_idx]
            if (self.args.ic_neTe == 1):
              print("!!!! resotring from glow fluid code")
              Uin[:, Te_idx]  = u1[:, Te_idx] / u1[:, ele_idx] 
            else:
              Uin[:, Te_idx]  = u1[:, Te_idx]
            
          else:
            Uin[:, ele_idx] = 1e6 * (1e7 + 1e9 * (1-0.5 * xx/self.param.L)**2 * (0.5 * xx/self.param.L)**2) / self.param.np0
            Uin[:, ion_idx] = 1e6 * (1e7 + 1e9 * (1-0.5 * xx/self.param.L)**2 * (0.5 * xx/self.param.L)**2) / self.param.np0
            Uin[:, Te_idx]  = self.args.Te
            
          spec_sp     = self.op_spec_sp
          mmat        = spec_sp.compute_mass_matrix()
          mmat_inv    = spec_sp.inverse_mass_mat(Mmat = mmat)
          vth         = self.bs_vth
          mw          = bte_utils.get_maxwellian_3d(vth, 1)

          mass_op     = self.op_mass
          temp_op     = self.op_temp
          
          [gmx,gmw]   = spec_sp._basis_p.Gauss_Pn(spec_sp._num_q_radial)
          Vqr_gmx     = spec_sp.Vq_r(gmx, 0, 1)
          
          num_p       = spec_sp._p +1
          num_sh      = len(spec_sp._sph_harm_lm)
          h_init      = xp.zeros(num_p * num_sh)
          
          ev_max_ext        = (spec_sp._basis_p._t_unique[-1] * self.bs_vth/self.c_gamma)**2
          print("v-grid max = %.4E (eV) extended to = %.4E (eV)" %(self.ev_lim[1], ev_max_ext))
          for i in range(self.Np):
            v_ratio           = (self.c_gamma * xp.sqrt(Uin[i, Te_idx])/vth)
            hv                = lambda v : (1/np.sqrt(np.pi)**3) * np.exp(-((v/v_ratio)**2)) / v_ratio**3
            h_init[0::num_sh] = xp.sqrt(4 * xp.pi) * xp.dot(mmat_inv[0::num_sh,0::num_sh], xp.dot(Vqr_gmx * hv(gmx) * gmx**2, gmw))
            m0                = xp.dot(mass_op, h_init)
            
            h_init            = h_init/m0
            
            hh1               = self.bte_eedf_normalization(h_init)
            num_sh            = len(spec_sp._sph_harm_lm)
            
            print("BTE idx=%d x_i=%.2E Te=%.8E mass=%.8E temp(eV)=%.8E "%(i, self.xp[i], Uin[i, self.Te_idx], m0, (xp.dot(temp_op, h_init)/m0)), end='')
            print(" k_elastic [m^3s^{-1}] = %.8E " %(xp.dot(self.op_rate[0], hh1[0::num_sh])), end='')
            if (len(self.op_rate) > 1):
              self.r_rates[i, self.ion_idx] = xp.dot(self.op_rate[1], hh1[0::num_sh]) * self.param.np0 * self.param.tau
              print("k_ionization [m^3s^{-1}] = %.8E " %(xp.dot(self.op_rate[1], hh1[0::num_sh])))
          
            Vin[:, i] = xp.einsum("il,rl->ri", self.op_psh2o, h_init.reshape((num_p, num_sh))).reshape((num_p * len(self.xp_vt)))
              
          Vin = Vin * Uin[:,ele_idx]
        else:
          raise NotImplementedError    
        
        
        enforce_bc = False
        if (enforce_bc == True):
          fl  = Vin[:,0]
          fr  = Vin[:,-1]
          
          fl[self.xp_vt_l]=0.0
          fr[self.xp_vt_r]=0.0
        
        
      self.mu[:, ele_idx] = self.param._mu_e
      self.D[: , ele_idx] = self.param._De
    
      self.mu[:, ion_idx] = self.param.mu_i
      self.D[: , ion_idx] = self.param.Di
        
        
          
      return Uin, Vin
    
    def initialize_maxwellian_eedf(self, ne, Te):
      xp          = self.xp_module
      
      temp_op     = self.op_temp
      mass_op     = self.op_mass
      op_rate     = self.op_rate
      psh2o       = self.op_psh2o
      mm_fac      = self.op_spec_sp._sph_harm_real(0, 0, 0, 0) * 4 * np.pi
      
      if xp==cp:
        temp_op     = xp.asnumpy(self.op_temp)
        mass_op     = xp.asnumpy(self.op_mass)
        op_rate     = [xp.asnumpy(self.op_rate[i]) for i in range(len(self.op_rate))]
        psh2o       = xp.asnumpy(self.op_psh2o)
      
      rates       = np.zeros_like(self.r_rates)
      Vin         = np.zeros((self.Nr * self.Nvt, self.Np))
      spec_sp     = self.op_spec_sp
      mmat        = spec_sp.compute_mass_matrix()
      mmat_inv    = spec_sp.inverse_mass_mat(Mmat = mmat)
      vth         = self.bs_vth
      mw          = bte_utils.get_maxwellian_3d(vth, 1)

      [gmx,gmw]   = spec_sp._basis_p.Gauss_Pn(spec_sp._num_q_radial)
      Vqr_gmx     = spec_sp.Vq_r(gmx, 0, 1)
      
      num_p       = spec_sp._p +1
      num_sh      = len(spec_sp._sph_harm_lm)
      num_x       = len(self.xp)
      num_vt      = len(self.xp_vt)
      h_init      = np.zeros(num_p * num_sh)
      
      ev_max_ext        = (spec_sp._basis_p._t_unique[-1] * self.bs_vth/self.c_gamma)**2
      print("v-grid max = %.4E (eV) extended to = %.4E (eV)" %(self.ev_lim[1], ev_max_ext))
      for i in range(self.Np):
        v_ratio           = (self.c_gamma * np.sqrt(Te[i])/vth)
        hv                = lambda v : (1/np.sqrt(np.pi)**3) * np.exp(-((v/v_ratio)**2)) / v_ratio**3
        h_init[0::num_sh] = np.sqrt(4 * np.pi) * np.dot(mmat_inv[0::num_sh,0::num_sh], np.dot(Vqr_gmx * hv(gmx) * gmx**2, gmw))
        m0                = np.dot(mass_op, h_init)
        
        h_init            = h_init/m0
        
        scale             = np.dot(mass_op / mm_fac, h_init) * (2 * (vth / self.c_gamma)**3)
        hh1               = h_init/scale
        num_sh            = len(spec_sp._sph_harm_lm)
        
        print("BTE idx=%d x_i=%.2E Te=%.8E mass=%.8E temp(eV)=%.8E "%(i, self.xp[i], Te[i], m0, (np.dot(temp_op, h_init)/m0)), end='')
        print(" k_elastic [m^3s^{-1}] = %.8E " %(np.dot(op_rate[0], hh1[0::num_sh])), end='')
        if (len(op_rate) > 1):
          rates[i, self.ion_idx] = np.dot(op_rate[1], hh1[0::num_sh]) * self.param.np0 * self.param.tau
          print("k_ionization [m^3s^{-1}] = %.8E " %(rates[i, self.ion_idx] / self.param.np0 / self.param.tau))
        
        Vin[:, i] = np.einsum("il,rlx->rix", psh2o, h_init.reshape((num_p , num_sh, num_x))).reshape((num_p * num_vt, num_x))
        
      # scale functions to have ne, at initial timestep
      Vin = Vin * ne
      return Vin
      
    def initialize_bte_adv_x(self, dt):
      """initialize spatial advection operator"""
      xp                = self.xp_module
      self.adv_setup_dt = dt 
      #assert xp == np
      
      self.bte_x_shift      = xp.zeros((self.Nr, self.Nvt, self.Np, self.Np))
      #self.bte_x_shift_rmat = xp.zeros((self.Nr, self.Nvt, self.Np, self.Np))
      
      if (self.xspace_adv_type == bte_xspace_adv_type.USE_BE_CHEB):
        DpL        = xp.zeros((self.Np, self.Np))
        DpR        = xp.zeros((self.Np, self.Np))
        
        DpL[1:,:]  = self.Dp[1:,:]
        DpR[:-1,:] = self.Dp[:-1,:]

      elif (self.xspace_adv_type == bte_xspace_adv_type.USE_BE_UPW_FD):

        DpL        = xp.zeros((self.Np, self.Np))
        DpR        = xp.zeros((self.Np, self.Np))

        DpL[1:,:]  = xp.array(mesh.upwinded_dx(self.xp, "LtoR"))[1:,:]
        DpR[:-1,:] = xp.array(mesh.upwinded_dx(self.xp, "RtoL"))[:-1,:]
        
      else:
        raise NotImplementedError

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
      
      #self.tmp1        = self.bte_x_shift   
      self.bte_x_shift = xp.linalg.inv(self.bte_x_shift)
      #self.bte_x_shift = cp.asnumpy(cp.linalg.inv(cp.asarray(self.bte_x_shift)))
      
      # adv_mat = cp.asnumpy(self.bte_x_shift)
      # v1 = -self.xp**2 + 1.2
      # plt.figure(figsize=(10,4), dpi=300)
      # plt.subplot(1, 2 , 1)
      # plt.semilogy(self.xp, v1,"-b",label="t=0")
      # y1 = np.copy(v1)
      # for i in range(50):
      #   #y1 = y1 + xp.dot(self.bte_x_shift_rmat[-1,-1],y1)
      #   y1[0]=0
      #   y1=xp.dot(adv_mat[-1,-1],y1)
      # plt.semilogy(self.xp, y1,"-r", label="left to right")
      # y1 = np.copy(v1)
      # for i in range(50):
      #   #y1 = y1 + xp.dot(self.bte_x_shift_rmat[-1, 0],y1)
      #   y1[-1]=0
      #   y1=xp.dot(adv_mat[-1,0],y1)
      # plt.semilogy(self.xp, y1,"-y", label="right to left ")
      # plt.legend()
      # plt.grid()
      
      # plt.subplot(1, 2 , 2)
      # plt.plot(self.xp, v1,"-b",label="t=0")
      # y1 = np.copy(v1)
      # for i in range(50):
      #   #y1 = y1 + xp.dot(self.bte_x_shift_rmat[-1,-1],y1)
      #   y1[0]=0
      #   y1=xp.dot(adv_mat[-1,-1],y1)
      #   # print(y1<0)
      #   # print("L to R", y1[0])
      #   y1[y1<0]=1e-16
        
      # plt.plot(self.xp, y1,"-r", label="left to right")
      # y1 = np.copy(v1)
      # for i in range(50):
      #   #y1 = y1 + xp.dot(self.bte_x_shift_rmat[-1, 0],y1)
      #   y1[-1]=0
      #   y1=xp.dot(adv_mat[-1,0],y1)
      #   # print("R to L", y1[-1])
      #   # y1[y1<0]=1e-16
      # plt.plot(self.xp, y1,"-y", label="right to left ")
      # plt.legend()
      # plt.grid()
      
      # #plt.show()
      # plt.savefig("test.png")
      # plt.close()
      # #sys.exit(0)
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
        self.DpT            = cp.asarray(self.DpT)
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
        self.op_mobility    = cp.asarray(self.op_mobility)
        self.op_diffusion   = cp.asarray(self.op_diffusion)

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
        self.DpT            = cp.asnumpy(self.DpT)
        self.LpD_inv        = cp.asnumpy(self.LpD_inv)
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
        self.op_mobility    = cp.asnumpy(self.op_mobility)
        self.op_diffusion   = cp.asnumpy(self.op_diffusion)
        
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
    
    def ords_to_sph(self, x):
      num_p   = self.op_spec_sp._p + 1
      num_sh  = len(self.op_spec_sp._sph_harm_lm)
      num_vt  = len(self.xp_vt)
      num_x   = len(self.xp)
      xp      = self.xp_module

      return xp.einsum("li,rix->rlx", self.op_po2sh, x.reshape((num_p, num_vt, num_x))).reshape((num_p * num_sh, num_x))
    
    def sph_to_ords(self, x):
      num_p   = self.op_spec_sp._p + 1
      num_sh  = len(self.op_spec_sp._sph_harm_lm)
      num_vt  = len(self.xp_vt)
      num_x   = len(self.xp)
      xp      = self.xp_module

      return xp.einsum("li,rix->rlx", self.op_psh2o, x.reshape((num_p, num_sh, num_x))).reshape((num_p * num_vt, num_x))

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
      num_vt  = len(self.xp_vt)
      num_x   = len(self.xp)
      
      v_lm                   = self.ords_to_sph(v)
      u[:, ele_idx]          = xp.dot(self.op_mass[0::num_sh], v_lm[0::num_sh,:])
      u[:, Te_idx]           = xp.dot(self.op_temp[0::num_sh], v_lm[0::num_sh,:])/u[:,ele_idx]
      
      v_lm1                  = self.bte_eedf_normalization(v_lm)
      
      if (len(self.op_rate) > 1):
        self.r_rates[:, ion_idx] = xp.dot(self.op_rate[1], v_lm1[0::num_sh,:]) * self.param.np0 * self.param.tau
      else:
        self.r_rates[:, ion_idx] = 0.0
      
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
      
      if(self.heavies_freeze_E == False):
        phi     = self.solve_poisson(u[:,ele_idx], u[:, ion_idx], time)
        E       = -xp.dot(self.Dp, phi)
      else:
        E       = self.bs_E
        
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
      
      if(self.heavies_freeze_E == False):
        phi     = self.solve_poisson(ne, ni, time)
        E       = -xp.dot(self.Dp, phi)
        E_ni    = self.E_ni
        #E_ne   = self.E_ne
      else:
        E       = self.bs_E
        E_ni    = 0 * self.E_ni
        #E_ne   = 0 * self.E_ne
      
      
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
      assert self.adv_setup_dt == dt

      Nr        = self.op_spec_sp._p + 1
      Nvt       = len(self.xp_vt)
      Nx        = len(self.xp)
      
      if PROFILE_SOLVERS==1:
        if xp == cp:
          cp.cuda.runtime.deviceSynchronize()
        t1 = perf_counter()
      
      Vin       = v.reshape((self.Nr, self.Nvt , self.Np)).reshape(self.Nr, self.Nvt * self.Np)
      Vin       = xp.dot(self.op_adv_x_qinv, Vin).reshape((self.Nr, self.Nvt, self.Np)).reshape((self.Nr * self.Nvt , self.Np))
      
      #Vin       += xp.einsum("ijkl,ijl->ijk",self.bte_x_shift_rmat, Vin.reshape((self.Nr, self.Nvt, self.Np))).reshape((self.Nr*self.Nvt, self.Np)) 
      # enforce rhs BCs
      Vin[self.xp_vt_l,  0] = 0.0
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
        print("BTE x-advection cost = %.4E (s)" %(t2-t1), flush=True)
      
      #Vin_adv_x[Vin_adv_x<0] = 1e-16
      # if (self.args.vtDe > 0):
      #   Vin_adv_x = xp.einsum("il,rlx->rix", self.LpDvt, Vin_adv_x.reshape((Nr, Nvt, Nx))).reshape((Nr * Nvt, Nx))

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
        print("fluid solver cost = %.4E (s)" %(t2-t1), flush=True)
        
      return u1
    
    def vspace_pc_setup(self):
      xp          = self.xp_module
      E           = self.bs_E
      pcEmat      = self.PmatE
      pcEval      = self.Evals.reshape((-1,1))
      E           = E.reshape((-1, 1))
      
      dist        = xp.linalg.norm(E[:, None, :] - pcEval[None, :, :], axis=2)
      c_memship   = xp.argmin(dist, axis=1)
      c_idx       = xp.arange(pcEval.shape[0])
      p_idx       = xp.arange(len(E))
      mask        = c_memship == c_idx[:, None]
      
      pc_emat_idx = list()
      for i in range(len(pcEval)):
        pc_emat_idx.append((i, p_idx[mask[i, :]]))
        
      
      
      # xp          = self.xp_module
      # E           = self.bs_E
      # pcEmat      = self.PmatE
      # pcEval      = self.Evals
      # pc_emat_idx = list()
      # idx         = xp.where(E<pcEval[0])[0]
      
      # if (len(idx)>0):
      #   pc_emat_idx.append((0, idx))
        
      # for i in range(1, len(pcEval)):
      #   idx  = xp.where((pcEval[i-1]<=E) & (E<pcEval[i]))[0]
      #   if (len(idx)>0):
      #     pc_emat_idx.append((i, idx))
      
      # idx = xp.where(pcEval[-1]<=E)[0]
      # if (len(idx)>0):
      #   #print(E[idx], pcEval[-1])
      #   pc_emat_idx.append((len(pcEval), idx))
      
      idx_set  = xp.array([],dtype=xp.int32)
      for idx_id, idx in enumerate(pc_emat_idx):
        idx_set = xp.append(idx_set, idx[1])
        
      assert (idx_set.shape[0]==self.Np), "!!! Error: preconditioner partitioning does not match the domain size"
      
      return pc_emat_idx
      
    def step_bte_v(self, v, dv, time, dt, ts_type, verbose=0):
      xp      = self.xp_module
      
      if PROFILE_SOLVERS==1:
        if xp == cp:
          cp.cuda.runtime.deviceSynchronize()
        t1 = perf_counter()
      
      time    = time
      dt      = dt
      
      rhs     = self.rhs_bte_v
      u       = self.ords_to_sph(v)
      
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
        E               = self.bs_E
            
        dof_v           = self.dof_v
        
        steps_cycle     = int(1/dt)
        pmat_freq       = steps_cycle//50
        step            = int(time/dt)
        
        if use_gmres == True:
          
          pcEmat      = self.PmatE
          pcEval      = self.Evals
          pc_emat_idx = self.vspace_pc_setup()
          
          def Lmat_mvec(x):
            x      = x.reshape((self.dof_v, self.Np))
            y      = self.param.tau * (self.param.n0 * self.param.np0 * (xp.dot(self.op_col_en, x) + self.param.Tg * xp.dot(self.op_col_gT, x))  + E * xp.dot(self.op_adv_v, x))
            y      = x - dt * y
            return y.reshape((-1))
          
          def Mmat_mvec(x):
            x      = x.reshape((self.dof_v, self.Np))
            y      = xp.copy(x)
            
            for idx_id, idx in enumerate(pc_emat_idx):
              y[:,idx[1]] = xp.dot(pcEmat[idx[0]], y[:, idx[1]])
            
            return y.reshape((-1))
          
          norm_b    = xp.linalg.norm(u.reshape((-1)))
          Ndof      = self.dof_v * self.Np
          if (self.args.use_gpu == 1):
            Lmat_op   = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Lmat_mvec)
            Mmat_op   = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Mmat_mvec)
            gmres_c   = glow1d_utils.gmres_counter(disp=False)
            v, status = cupyx.scipy.sparse.linalg.gmres(Lmat_op, u.reshape((-1)), x0=u.reshape((-1)), tol=rtol, atol=atol, M=Mmat_op, restart=self.args.gmres_rsrt, maxiter=self.args.gmres_rsrt * 50, callback=gmres_c)
          else:
            Lmat_op   = scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Lmat_mvec)
            Mmat_op   = scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Mmat_mvec)
            gmres_c   = glow1d_utils.gmres_counter(disp=False)
            v, status = scipy.sparse.linalg.gmres(Lmat_op, u.reshape((-1)), x0=u.reshape((-1)), rtol=rtol, atol=atol, M=Mmat_op, restart=self.args.gmres_rsrt, maxiter=self.args.gmres_rsrt * 50, callback=gmres_c)

          
          
          norm_res_abs  = xp.linalg.norm(Lmat_mvec(v) -  u.reshape((-1)))
          norm_res_rel  = xp.linalg.norm(Lmat_mvec(v) -  u.reshape((-1))) / norm_b
          
          if (status !=0) :
            print("%08d GMRES solver failed! iterations =%d  ||res|| = %.4E ||res||/||b|| = %.4E"%(step, status, norm_res_abs, norm_res_rel))
            sys.exit(-1)
            # self.bte_pmat = xp.linalg.inv(Lmat)
            # v             = xp.einsum("ijk,ki->ji", self.bte_pmat, u)
            # norm_res      = xp.linalg.norm(Lmat_mvec(xp.transpose(v).reshape((-1))) -  uT.reshape((-1))) / norm_b
          else:
            v               = v.reshape((self.dof_v, self.Np))
        else:
          raise NotImplementedError
        
        v       = self.sph_to_ords(v)
        
        if PROFILE_SOLVERS==1:
          if xp == cp:
            cp.cuda.runtime.deviceSynchronize()
          t2 = perf_counter()
          if (verbose==1):  
            print("BTE (v-space) solve cost = %.6E " %(t2-t1), end=" ")
          
        if (verbose==1):
          print("%08d Boltzmann (v-space) step time = %.6E ||res||=%.12E ||res||/||b||=%.12E gmres iter = %04d"%(step, time, norm_res_abs, norm_res_rel, gmres_c.niter * self.args.gmres_rsrt))
        
        
        return v 
        
      elif ts_type == "IMEX":
        rhs_explicit  =  u + dt * self.param.tau * self.bs_E * xp.dot(self.op_adv_v, u)
        return xp.dot(self.bte_imex_lmat_inv, rhs_explicit)
    
    def step_bte_v1(self, v, dv, time, dt, ts_type, verbose=0):
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
      
      u       = self.ords_to_sph(v)
      
      rtol            = self.args.rtol
      atol            = self.args.atol
      iter_max        = self.args.max_iter
      use_gmres       = True
      
      steps_cycle     = int(1/dt)
      pmat_freq       = steps_cycle//50
      step            = int(time/dt)
      
      if xp == cp:
        cp.cuda.runtime.deviceSynchronize()
      a_t1 = perf_counter()
      
      E               = self.bs_E
      assert (E[0]==E).all()==True, "E field is not spatially homogenous"
      E               = self.bs_E[0]
      
      dof_v           = self.dof_v
      Imat            = self.I_Nv
      
      Jmat            = self.param.tau * self.param.n0 * self.param.np0 * (self.op_col_en + self.param.Tg * self.op_col_gT) + self.param.tau * E * self.op_adv_v
      Lmat            = Imat -dt * Jmat
      
      if xp == cp:
        cp.cuda.runtime.deviceSynchronize()
      a_t2 = perf_counter()
      
      if xp == cp:
        cp.cuda.runtime.deviceSynchronize()
      
      s_t1 = perf_counter()
      v = xp.dot(xp.linalg.inv(Lmat), u)
      
      if xp == cp:
        cp.cuda.runtime.deviceSynchronize()

      s_t2 = perf_counter()
      norm_res = xp.linalg.norm(xp.dot(Lmat, v) - u) / xp.linalg.norm(u.reshape(-1))
      print("%08d Boltzmann step time = %.6E op. assembly =%.6E solve = %.6E res=%.12E"%(step, time, (a_t2-a_t1), (s_t2-s_t1), norm_res))
      v       = self.sph_to_ords(v)
      return v
    
    def step_bte_v2(self, Uin, Vin, time, dt, verbose=0):
      
      """
      v-space BTE solve coupled with E field
      """
      xp      = self.xp_module
      if xp == cp:
        cp.cuda.runtime.deviceSynchronize()

      s_t1    = perf_counter()
      
      gmres_rs    = self.args.gmres_rsrt
      gmres_rtol  = self.args.gmres_rtol
      gmres_atol  = self.args.gmres_atol
      
      
      newton_atol = self.args.atol
      newton_rtol = self.args.rtol
      
      num_p   = self.op_spec_sp._p + 1
      num_sh  = len(self.op_spec_sp._sph_harm_lm)
      
      
      u       = Uin
      v       = Vin
      
      ni      = u[:, self.ion_idx]
      v       = v.reshape((self.Nr * self.Nvt, self.Np)) 
      v_lm    = self.ords_to_sph(v)
      
      ne0     = xp.dot(self.op_mass[0::num_sh], v_lm[0::num_sh])
      E0      = -xp.dot(self.Dp, self.solve_poisson(ne0, ni, time + dt)) * (self.param.V0/self.param.L)
      
      def residual(dv_lm):
        dv_lm = dv_lm.reshape((num_p * num_sh, self.Np))
        vp    = v_lm + dv_lm
        ne    = xp.dot(self.op_mass[0::num_sh], vp[0::num_sh])
        E     = -xp.dot(self.Dp, self.solve_poisson(ne, ni, time + dt)) * (self.param.V0/self.param.L)
        res   = dv_lm  - dt * self.param.tau * (self.param.n0 * self.param.np0 * (xp.dot(self.op_col_en, vp) + self.param.Tg * xp.dot(self.op_col_gT, vp))  + E * xp.dot(self.op_adv_v, vp))
        return res.reshape((-1))
      
      def jacobian(dv_lm):
        dv_lm     = dv_lm.reshape((num_p * num_sh, self.Np))
        phi       = self.solve_poisson(xp.dot(self.op_mass[0::num_sh], dv_lm[0::num_sh]), 0 * ni,  0.0)
        dEdF      = -xp.dot(self.Dp, phi) * (self.param.V0/self.param.L) 
        
        j1        = (self.param.n0 * self.param.np0 * (xp.dot(self.op_col_en, dv_lm) + self.param.Tg * xp.dot(self.op_col_gT, dv_lm))  + E0 * xp.dot(self.op_adv_v, dv_lm))
        j1       += dEdF * xp.dot(self.op_adv_v, v_lm)
        jac       = dv_lm - dt * self.param.tau * j1
        return jac.reshape((-1))
      
      # def precond(dv_lm):
      #   return dv_lm
      
      pcEmat      = self.PmatE
      pcEval      = self.Evals
      self.bs_E   = E0
      pc_emat_idx = self.vspace_pc_setup()
      
      def precond(dv_lm):
        dv_lm       = dv_lm.reshape((num_p * num_sh, self.Np))
        y           = xp.copy(dv_lm)

        for idx_id, idx in enumerate(pc_emat_idx):
          y[:,idx[1]] = xp.dot(pcEmat[idx[0]], y[:, idx[1]])
            
        return y.reshape((-1))
        
      
      dv_lm   = xp.zeros_like(v_lm)
      ns_info = glow1d_utils.newton_solver_matfree(dv_lm.reshape((-1)), residual, jacobian, precond, newton_atol, newton_rtol, gmres_atol, gmres_rtol, gmres_rs, gmres_rs * 50, xp)
      dv_lm   = ns_info["x"].reshape((num_p * num_sh, self.Np))
      
      if xp == cp:
        cp.cuda.runtime.deviceSynchronize()

      s_t2    = perf_counter()
      
      if(verbose==1):
        print("[BTE (v-space)] solve time = %.4E simulation time = %.4E | Newton iter = %04d |  GMRES iter = %04d | ||res|| = %.4E | ||res||/||res0|| = %.4E | alpha=%.4E"%((s_t2 - s_t1), time, ns_info["iter"], ns_info["iter_gmres"], ns_info["atol"], ns_info["rtol"], ns_info["alpha"]))
        
        
      if (ns_info["status"]==False):
        if(verbose==0):
          print("BTE (v-space) time = %.4E | Newton iter = %04d |  GMRES iter = %04d | ||res|| = %.4E | ||res||/||res0|| = %.4E | alpha=%.4E"%(time, ns_info["iter"], ns_info["iter_gmres"], ns_info["atol"], ns_info["rtol"], ns_info["alpha"]))
        print("BTE (v-space) step non-linear solver step FAILED!!! try with smaller time step size or increase max iterations")
        sys.exit(-1)
        return u, v
      
      dv = self.sph_to_ords(dv_lm)
      return u, v + dv
    
    def step_bte_vx_with_E(self, Uin, Vin, time, dt, verbose=0):
      """
      vx-space BTE solve coupled with E field
      """
      xp      = self.xp_module
      
      if xp == cp:
        cp.cuda.runtime.deviceSynchronize()

      s_t1    = perf_counter()
      
      num_p   = self.op_spec_sp._p + 1
      num_sh  = len(self.op_spec_sp._sph_harm_lm)
      num_vt  = self.Nvt
      
      
      u       = Uin
      v       = Vin
      
      ni      = u[:, self.ion_idx]
      v       = v.reshape((self.Nr * self.Nvt, self.Np)) 
      v_lm    = self.ords_to_sph(v)
      
      ne0     = xp.dot(self.op_mass[0::num_sh], v_lm[0::num_sh])
      E0      = -xp.dot(self.Dp, self.solve_poisson(ne0, ni, time + dt)) * (self.param.V0/self.param.L)
      
      gmres_rs    = self.args.gmres_rsrt
      gmres_rtol  = self.args.gmres_rtol
      gmres_atol  = self.args.gmres_atol
      
      
      newton_atol = self.args.atol
      newton_rtol = self.args.rtol
      
      
      def adv_x(v0, time, dt): 
        y   = v0.reshape((self.Nr, self.Nvt , self.Np)).reshape(self.Nr, self.Nvt * self.Np)
        y   = xp.dot(self.op_adv_x_qinv, y).reshape((self.Nr, self.Nvt, self.Np)).reshape((self.Nr * self.Nvt , self.Np))

        # enforce rhs BCs
        y[self.xp_vt_l, 0]  = 0.0
        y[self.xp_vt_r, -1] = 0.0

        y = xp.einsum("ijkl,ijl->ijk",self.bte_x_shift, y.reshape((self.Nr, self.Nvt, self.Np)))
        y  = y.reshape((self.Nr, self.Nvt *  self.Np))
        y  = xp.dot(self.op_adv_x_q, y).reshape((self.Nr , self.Nvt, self.Np)).reshape((self.Nr * self.Nvt, self.Np))
        return y
      
      #return u, adv_x(v, time, dt)
      
      def adv_x_q_inv_vec(x):
        y  = x.reshape((self.Nr, self.Nvt , self.Np)).reshape(self.Nr, self.Nvt * self.Np)
        y  = xp.dot(self.op_adv_x_qinv, y).reshape((self.Nr, self.Nvt, self.Np)).reshape((self.Nr * self.Nvt , self.Np))
        return y
      
      def adv_x_q_vec(x):
        y  = x.reshape((self.Nr, self.Nvt , self.Np)).reshape(self.Nr, self.Nvt * self.Np)
        y  = xp.dot(self.op_adv_x_q, y).reshape((self.Nr, self.Nvt, self.Np)).reshape((self.Nr * self.Nvt , self.Np))
        return y
      
      vd   = adv_x_q_inv_vec(v)
      
      def residual(dv):
        dv    = dv.reshape((self.Nr * self.Nvt, self.Np))
        vp    = vd + dv
        
        Fx    = xp.dot(vp, self.DpT).reshape((self.Nr, self.Nvt, self.Np))
        Fx    = xp.einsum("i,j,ijk->ijk", self.op_adv_x_d, self.xp_cos_vt, Fx).reshape((self.Nr * self.Nvt, self.Np))
        
        yp    = vp.reshape((self.Nr, self.Nvt , self.Np)).reshape(self.Nr, self.Nvt * self.Np)
        yp    = xp.dot(self.op_adv_x_q, yp).reshape((self.Nr, self.Nvt, self.Np)).reshape((self.Nr * self.Nvt , self.Np))
        yp    = self.ords_to_sph(yp)
        
        ne    = xp.dot(self.op_mass[0::num_sh], yp[0::num_sh])
        E     = -xp.dot(self.Dp, self.solve_poisson(ne, ni, time + dt)) * (self.param.V0/self.param.L)
        
        Fv    = self.param.tau * (self.param.n0 * self.param.np0 * (xp.dot(self.op_col_en, yp) + self.param.Tg * xp.dot(self.op_col_gT, yp))  + E * xp.dot(self.op_adv_v, yp))
        Fv    = self.sph_to_ords(Fv)
        
        Fv    = Fv.reshape((self.Nr, self.Nvt , self.Np)).reshape(self.Nr, self.Nvt * self.Np)
        Fv    = xp.dot(self.op_adv_x_qinv, Fv).reshape((self.Nr, self.Nvt, self.Np)).reshape(self.Nr * self.Nvt, self.Np)
        
        rhs                    = Fv - Fx
        res                    = dv - dt * rhs
        
        res[self.xp_vt_l, 0 ]  = (vp[self.xp_vt_l, 0 ] - 0.0)
        res[self.xp_vt_r, -1 ] = (vp[self.xp_vt_r, -1] - 0.0)
        
        res = self.sph_to_ords(self.ords_to_sph(res))
        
        return res.reshape((-1))
      
      def jacobian(dv):
        
        dv    = dv.reshape((self.Nr * self.Nvt, self.Np))
        
        Jx    = xp.dot(dv, self.DpT).reshape((self.Nr, self.Nvt, self.Np))
        Jx    = xp.einsum("i,j,ijk->ijk", self.op_adv_x_d, self.xp_cos_vt, Jx).reshape((self.Nr * self.Nvt, self.Np))
        
        yp    = dv.reshape((self.Nr, self.Nvt , self.Np)).reshape(self.Nr, self.Nvt * self.Np)
        yp    = xp.dot(self.op_adv_x_q, yp).reshape((self.Nr, self.Nvt, self.Np)).reshape((self.Nr * self.Nvt , self.Np))
        yp    = self.ords_to_sph(yp)
        
        ne    = xp.dot(self.op_mass[0::num_sh], yp[0::num_sh])
        dE    = -xp.dot(self.Dp, self.solve_poisson(ne, 0* ni, 0.0)) * (self.param.V0/self.param.L)
        
        Jv    = self.param.tau * (self.param.n0 * self.param.np0 * (xp.dot(self.op_col_en, yp) + self.param.Tg * xp.dot(self.op_col_gT, yp))  + E0 * xp.dot(self.op_adv_v, yp) + dE * xp.dot(self.op_adv_v, v_lm))
        Jv    = self.sph_to_ords(Jv)
        
        Jv    = Jv.reshape((self.Nr, self.Nvt , self.Np)).reshape(self.Nr, self.Nvt * self.Np)
        Jv    = xp.dot(self.op_adv_x_qinv, Jv).reshape((self.Nr, self.Nvt, self.Np)).reshape(self.Nr * self.Nvt, self.Np)
        
        rhsJ  = Jv - Jx 
        jac   = dv - dt * rhsJ
        
        jac[self.xp_vt_l, 0 ]  = dv[self.xp_vt_l, 0 ]
        jac[self.xp_vt_r, -1 ] = dv[self.xp_vt_r, -1]
        
        jac = self.sph_to_ords(self.ords_to_sph(jac))
        
        return jac.reshape((-1))
      
      def op_split(x):
        x = x.reshape((self.Nr * self.Nvt, self.Np))
        y = xp.copy(x)
        
        y[self.xp_vt_l, 0 ]  = 0.0
        y[self.xp_vt_r, -1 ] = 0.0
        y                    = xp.einsum("ijkl,ijl->ijk",self.bte_x_shift, y.reshape((self.Nr, self.Nvt, self.Np))).reshape((self.Nr * self.Nvt, self.Np))
        
        y = y.reshape((self.Nr, self.Nvt, self.Np)).reshape((self.Nr, self.Nvt * self.Np))
        y = xp.dot(self.op_adv_x_q, y).reshape((self.Nr, self.Nvt, self.Np)).reshape((self.Nr * self.Nvt, self.Np))
        
        y           = self.ords_to_sph(y)
        ne          = xp.dot(self.op_mass[0::num_sh], y[0::num_sh])
        E           = -xp.dot(self.Dp, self.solve_poisson(ne, ni, time)) * (self.param.V0/self.param.L)
        
        pcEmat      = self.PmatE
        pcEval      = self.Evals
        self.bs_E   = E
        pc_emat_idx = self.vspace_pc_setup()

        for idx_id, idx in enumerate(pc_emat_idx):
          y[:,idx[1]] = xp.dot(pcEmat[idx[0]], y[:, idx[1]])
        
        y = self.sph_to_ords(y)

        y = y.reshape((self.Nr, self.Nvt, self.Np)).reshape((self.Nr, self.Nvt * self.Np))
        y = xp.dot(self.op_adv_x_qinv, y).reshape((self.Nr, self.Nvt, self.Np)).reshape((self.Nr * self.Nvt, self.Np))
        
        y[self.xp_vt_l, 0 ]  = 0.0
        y[self.xp_vt_r, -1 ] = 0.0
        y                    = xp.einsum("ijkl,ijl->ijk",self.bte_x_shift, y.reshape((self.Nr, self.Nvt, self.Np))).reshape((self.Nr * self.Nvt, self.Np))
        
        return y
      
      pcEmat      = self.PmatE
      pcEval      = self.Evals
      self.bs_E   = E0
      pc_emat_idx = self.vspace_pc_setup()
      
      
      def precond(x):
        #return x.reshape((-1))
        x = x.reshape((self.Nr * self.Nvt, self.Np))
        y = xp.copy(x)
        
        y[self.xp_vt_l, 0 ]  = x[self.xp_vt_l,  0]
        y[self.xp_vt_r, -1 ] = x[self.xp_vt_r, -1]
        y                    = xp.einsum("ijkl,ijl->ijk",self.bte_x_shift, y.reshape((self.Nr, self.Nvt, self.Np))).reshape((self.Nr * self.Nvt, self.Np))
        
        y = y.reshape((self.Nr, self.Nvt, self.Np)).reshape((self.Nr, self.Nvt * self.Np))
        y = xp.dot(self.op_adv_x_q, y).reshape((self.Nr, self.Nvt, self.Np)).reshape((self.Nr * self.Nvt, self.Np))
        
        y           = self.ords_to_sph(y)
        for idx_id, idx in enumerate(pc_emat_idx):
          y[:,idx[1]] = xp.dot(pcEmat[idx[0]], y[:, idx[1]])
        
        y = self.sph_to_ords(y)
        y = y.reshape((self.Nr, self.Nvt, self.Np)).reshape((self.Nr, self.Nvt * self.Np))
        y = xp.dot(self.op_adv_x_qinv, y).reshape((self.Nr, self.Nvt, self.Np)).reshape((self.Nr * self.Nvt, self.Np))
        
        y[self.xp_vt_l, 0 ]  = x[self.xp_vt_l,  0]
        y[self.xp_vt_r, -1 ] = x[self.xp_vt_r, -1]
        y                    = xp.einsum("ijkl,ijl->ijk",self.bte_x_shift, y.reshape((self.Nr, self.Nvt, self.Np))).reshape((self.Nr * self.Nvt, self.Np))
        
        y = self.sph_to_ords(self.ords_to_sph(y))
        
        return y.reshape((-1))
      
      
      lagg_E = False
      if (lagg_E == True):
        # linear scheme with lagging E term;
        # 1). This does not allow us to take larger timesteps, because E lagging. 
        
        def precond(x):
          #return x.reshape((-1))
          x = x.reshape((self.Nr * self.Nvt, self.Np))
          y = xp.copy(x)
          
          y = y.reshape((self.Nr, self.Nvt, self.Np)).reshape((self.Nr, self.Nvt * self.Np))
          y = xp.dot(self.op_adv_x_q, y).reshape((self.Nr, self.Nvt, self.Np)).reshape((self.Nr * self.Nvt, self.Np))
          
          y = self.ords_to_sph(y)
          for idx_id, idx in enumerate(pc_emat_idx):
            y[:,idx[1]] = xp.dot(pcEmat[idx[0]], y[:, idx[1]])
          
          y = self.sph_to_ords(y)
          
          y = y.reshape((self.Nr, self.Nvt, self.Np)).reshape((self.Nr, self.Nvt * self.Np))
          y = xp.dot(self.op_adv_x_qinv, y).reshape((self.Nr, self.Nvt, self.Np)).reshape((self.Nr * self.Nvt, self.Np))
          
          y[self.xp_vt_l, 0 ]  = 0 * x[self.xp_vt_l,  0]
          y[self.xp_vt_r, -1 ] = 0 * x[self.xp_vt_r, -1]
          y                    = xp.einsum("ijkl,ijl->ijk",self.bte_x_shift, y.reshape((self.Nr, self.Nvt, self.Np))).reshape((self.Nr * self.Nvt, self.Np))
          
          return y.reshape((-1))
        
        bd = xp.copy(vd) - 0.5 * residual(xp.zeros_like(vd)).reshape((self.Nr * self.Nvt, self.Np))
        
        bd[self.xp_vt_l, 0]  = 0.0
        bd[self.xp_vt_r, -1] = 0.0
        
        #bd = -residual(xp.zeros_like(vd)).reshape((self.Nr * self.Nvt, self.Np)) + jacobian(vd).reshape((self.Nr * self.Nvt, self.Np))
        
        Ndof       = self.Nr * self.Nvt * self.Np
        Lmat_op    = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=jacobian)
        Mmat_op    = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=precond)
        
        gmres_restart       = 5
        gmres_maxiter       = 1000
        gmres_ctx           = glow1d_utils.gmres_counter(disp=False)
        
        v1, status = cupyx.scipy.sparse.linalg.gmres(Lmat_op, bd.reshape((-1)), x0=vd.reshape((-1)), tol=self.args.rtol, atol=self.args.atol, M=Mmat_op, callback=gmres_ctx, restart=gmres_restart, maxiter=gmres_maxiter)
        
        res           = jacobian(v1) -  bd.reshape((-1))
        norm_b        = xp.linalg.norm(bd.reshape((-1)))
        norm_res_abs  = xp.linalg.norm(res)
        norm_res_rel  = norm_res_abs / norm_b
        
        
        v1         = adv_x_q_vec(v1.reshape((self.Nr * self.Nvt, self.Np)))
        
        if xp == cp:
          cp.cuda.runtime.deviceSynchronize()

        s_t2    = perf_counter()
        print("time = %.2E ||Ax-b|| = %.4E ||Ax-b|| / ||b|| = %.4E solve time = %.4E gmres iterations: %d " %(time, norm_res_abs, norm_res_rel, (s_t2-s_t1), gmres_ctx.niter * gmres_restart))
        return u, v1
      
      #dv       = xp.zeros_like(v)
      dv       = (op_split(vd).reshape((self.Nr * self.Nvt, self.Np)) - vd)
      
      
      ns_info  = glow1d_utils.newton_solver_matfree_v1(dv.reshape((-1)), residual, jacobian, precond, newton_atol, newton_rtol, gmres_atol, gmres_rtol, gmres_rs, gmres_rs * 50, xp)
      dv       = ns_info["x"].reshape((self.Nr * self.Nvt, self.Np))
      
      dv       = adv_x_q_vec(dv)
      
      v1       = v + dv
      
      ## filtering to denoise the ordinates to spherical projections
      v1       = self.sph_to_ords(self.ords_to_sph(v1))
      v1[v1<0] = 1e-16
      
      if xp == cp:
        cp.cuda.runtime.deviceSynchronize()

      s_t2    = perf_counter()
      
      if(verbose==1):
        print("[BTE (vx-space)] solve time = %.4E simulation time = %.4E | Newton iter = %04d |  GMRES iter = %04d | ||res|| = %.4E | ||res||/||res0|| = %.4E | alpha=%.4E"%((s_t2 - s_t1), time, ns_info["iter"], ns_info["iter_gmres"], ns_info["atol"], ns_info["rtol"], ns_info["alpha"]))
        
        
      if (ns_info["status"]==False):
        if(verbose==0):
          print("BTE (vx-space) time = %.4E | Newton iter = %04d |  GMRES iter = %04d | ||res|| = %.4E | ||res||/||res0|| = %.4E | alpha=%.4E"%(time, ns_info["iter"], ns_info["iter_gmres"], ns_info["atol"], ns_info["rtol"], ns_info["alpha"]))
        print("BTE (vx-space) step non-linear solver step FAILED!!! try with smaller time step size or increase max iterations")
        sys.exit(-1)
        return u, v
      
      return u, v1
      
    def step_bte(self, u, v, du, dv, time, dt, ts_type, verbose=0):
      xp     = self.xp_module
      
      #if PROFILE_SOLVERS==1:
      if xp == cp:
        cp.cuda.runtime.deviceSynchronize()
      t1 = perf_counter()
        
      tt_bte = time
      dt_bte = dt
      
      num_sh = len(self.op_spec_sp._sph_harm_lm)
      ni     = u[:, self.ion_idx]
      
      # # First order splitting
      # v       = self.step_bte_x(v, tt_bte, dt_bte)
      # v_lm    = self.ords_to_sph(v)
      # v_lm    = self.step_bte_v(v_lm, None, tt_bte, dt_bte, self.ts_type_bte_v , verbose)
      # v       = self.sph_to_ords(v_lm)
      # v[v<0]  = 0
      
      # Strang-Splitting
      v           = self.step_bte_x(v, tt_bte, dt_bte * 0.5)
      # ne          = xp.dot(self.op_mass[0::num_sh], self.ords_to_sph(v)[0::num_sh])
      # self.bs_E   = -xp.dot(self.Dp, self.solve_poisson(ne, ni, time)) * (self.param.V0/self.param.L)
      
      v           = self.step_bte_v(v, None, tt_bte, dt_bte, self.ts_type_bte_v , verbose)
      
      # # below is coupling E only to the v-space in non-linear solve, problem, this will cause E-field over prediction due to 
      # # not accounting the electron loss due to spatial advection. 
      # u, v    = self.step_bte_v2(u, v, tt_bte, dt_bte, verbose=1)
      
      v       = self.step_bte_x(v, tt_bte + 0.5 * dt_bte, dt_bte * 0.5)
      
      #if PROFILE_SOLVERS==1:
      if xp == cp:
        cp.cuda.runtime.deviceSynchronize()
      t2 = perf_counter()
      if (verbose==1):
        print("BTE(vx-op-split) solver cost = %.4E (s)" %(t2-t1), flush=True)
        
      return v
    
    def step_init(self, Uin, Vin, dt):
      if self.args.use_gpu == 1: 
        Uin1 = cp.asarray(Uin)
        Vin1 = cp.asarray(Vin)
      else:
        Uin1 = Uin
        Vin1 = Vin
      
      if self.args.use_gpu==1:
        self.copy_operators_H2D(self.args.gpu_device_id)
        self.xp_module = cp
      else:
        self.xp_module = np
      
      if (self.args.glow_op_split_scheme == 0 or True):
        xp            = self.xp_module
        num_pc_evals  = 30
        ep            = xp.logspace(xp.log10(1e2), xp.log10(6e5), num_pc_evals//2, base=10)
        self.Evals    = -xp.flip(ep)
        self.Evals    = xp.append(self.Evals,ep)
        vmat          = self.param.n0 * self.param.np0 * (self.op_col_en + self.param.Tg * self.op_col_gT)
        self.PmatC    = xp.linalg.inv(self.I_Nv - 1.0 * dt * self.param.tau * vmat)
        self.PmatE    = list()
        
        num_sh        = len(self.op_spec_sp._sph_harm_lm)
        
        pmat          = self.PmatC
        emat          = xp.linalg.inv(self.I_Nv - dt * self.param.tau * self.Evals[0] * self.op_adv_v - dt * self.param.tau * vmat)
        ep_mat        = emat 
        self.PmatE.append(ep_mat)
        
        for i in range(1, num_pc_evals):
          emat          = xp.linalg.inv(self.I_Nv - dt * self.param.tau * 0.5 * (self.Evals[i-1] + self.Evals[i]) * self.op_adv_v - dt * self.param.tau * vmat)
          ep_mat        = emat 
          self.PmatE.append(ep_mat)
        
        emat          = xp.linalg.inv(self.I_Nv - dt * self.param.tau * self.Evals[-1] * self.op_adv_v - dt * self.param.tau * vmat)
        ep_mat        = emat 
        self.PmatE.append(ep_mat)
        self.PmatE    = xp.array(self.PmatE)
        
        assert len(self.PmatE) == num_pc_evals + 1
        assert len(self.Evals) == num_pc_evals
        print("v-space advection mat preconditioner gird : \n", self.Evals)
        self.initialize_bte_adv_x(0.5 * dt)
        print("================================================================")
        print("                  adv_x initialization with 0.5 * dt                 ")
        print("================================================================")
        
        
        # self.initialize_bte_adv_x(dt)
        # print("================================================================")
        # print("                  adv_x initialization with 1dt                 ")
        # print("================================================================")
        
        #self.op_precon_adv_x = xp.linalg.inv(xp.eye(self.Nr) + dt * self.op_adv_x)
        
        # DpL        = xp.zeros((self.Np, self.Np))
        # DpR        = xp.zeros((self.Np, self.Np))

        # DpL[1:,:]  = self.Dp[1:,:]
        # DpR[:-1,:] = self.Dp[:-1,:]
        
        # adv_x_inv  = xp.empty((self.Nvt, self.Nr * self.Np, self.Nr * self.Np))
        # I1         = xp.eye(self.Nr * self.Np)
        
        # self.xp_vt_L = self.xp_vt <= 0.5 * xp.pi
        # self.xp_vt_R = self.xp_vt  > 0.5 * xp.pi
        
        # print(self.xp_vt[self.xp_vt_L])
        # print(self.xp_vt[self.xp_vt_R])
        
        # tmp  = xp.empty((self.Nr * self.Np, self.Nr * self.Np))
        # for i in range(self.Nr):
        #     for j in range(self.Nr):
        #       tmp[(i * self.Np) : ((i + 1) * self.Np)  , (j * self.Np) : ((j + 1) * self.Np)] = self.op_adv_x[i, j] * self.Dp
        
        # self.op_precon_adv_x  = tmp
        
        # A = xp.eye(self.Np) - dt * self.Dp
        # Ainv = xp.linalg.inv(A)
        # print("||I - A A^{-1}|| = ", xp.allclose(xp.dot(Ainv, A), xp.eye(self.Np), rtol=1e-10, atol=1e-12))
        # print(xp.dot(Ainv, A))
        
        # self.op_precon_adv_xL = I1 + dt * tmp
        # #self.op_precon_adv_xR = I1 - dt * tmp
        # del tmp
        
        # v  = xp.copy(Vin1)
        
        # v1 = xp.einsum("ak,ijk->ija", self.Dp, v.reshape((self.Nr, self.Nvt, self.Np)))
        # v1 = xp.einsum("al,ljk->ajk", self.op_adv_x, v1)
        # v1 = xp.einsum("j,ijk->ijk" , xp.ones_like(self.xp_cos_vt), v1) 
        # v1 = dt * v1 + v.reshape((self.Nr, self.Nvt, self.Np))
        # v1 = v1.reshape((self.Nr * self.Nvt, self.Np))
        
        # #v1[self.xp_vt_l, 0 ] = v[self.xp_vt_l, 0]
        # #v1[self.xp_vt_r, -1] = v[self.xp_vt_r, -1]
        
        
        # v2 = xp.swapaxes(v.reshape((self.Nr, self.Nvt, self.Np)), 1, 2).reshape((self.Nr * self.Np, self.Nvt))
        # v2 = xp.dot(self.op_precon_adv_xL, v2)
        # v2 = xp.swapaxes(v2.reshape((self.Nr, self.Np, self.Nvt)), 1, 2).reshape((self.Nr * self.Nvt, self.Np))
        
        
        # print("v2 vs. v1 ", xp.linalg.norm(v2-v1)/xp.linalg.norm(v1))
        
        
        # A1 = xp.linalg.inv(self.op_precon_adv_xL)
        # print("||I -A A^{-1}|| / ||I|| ", xp.linalg.norm(xp.dot(A1, self.op_precon_adv_xL) - I1)/xp.linalg.norm(I1))
        # print(xp.dot(A1, self.op_precon_adv_xL))
        # # self.op_precon_adv_xR = xp.linalg.inv(self.op_precon_adv_xR)
        
        # v3 = xp.swapaxes(v2.reshape((self.Nr, self.Nvt, self.Np)), 1, 2).reshape((self.Nr * self.Np, self.Nvt))
        # v3 = xp.dot(self.op_precon_adv_xL, v3)
        # v3 = xp.swapaxes(v2.reshape((self.Nr, self.Np, self.Nvt)), 1, 2).reshape((self.Nr * self.Nvt, self.Np))
        
        # print("v2 vs. v3 ", xp.linalg.norm(v2-v3)/xp.linalg.norm(v3))
        # sys.exit(0)
        
        #print("adv mat cond : ", np.linalg.cond(xp.asnumpy(tmp)))
        
        # for theta_idx in range(self.Nvt):
        #   for i in range(self.Nr):
        #     for j in range(self.Nr):
        #       adv_x_inv[theta_idx, (i * self.Np) : ((i + 1) * self.Np)  , (j * self.Np) : ((j + 1) * self.Np)] = dt * self.xp_cos_vt[theta_idx] * self.op_adv_x[i, j] * self.Dp
          
        #   adv_x_inv[theta_idx]+=I1
        
        
        
        # #adv_x_inv = xp.linalg.inv(adv_x_inv)
              
        # self.op_precon_adv_x_l = xp.linalg.inv(xp.eye(self.Nr * self.Np) + 0.5 * dt * self.op_precon_adv_x)
        # self.op_precon_adv_x_r = xp.linalg.inv(xp.eye(self.Nr * self.Np) - 0.5 * dt * self.op_precon_adv_x)
      
      self.bte_to_fluid(Uin1, Vin1, 0.0, dt)
      self.fluid_to_bte(Uin1, Vin1, 0.0, dt)
      
      return Uin1, Vin1
    
    def step(self, u, v, du, dv , time, dt, scheme="strang-splitting", verbose=0):
      xp      = self.xp_module
      tt      = time
      
      if xp == cp:
        cp.cuda.runtime.deviceSynchronize()

      s_t1    = perf_counter()
      
      if scheme=="strang-splitting":
        # second-order split scheme
        
        if (self.heavies_freeze_E):
          self.bs_E = xp.dot(self.Dp, -self.solve_poisson(u[:, self.ele_idx], u[:, self.ion_idx], tt))
          
        u = self.step_fluid(u, du, tt, dt * 0.5,  self.ts_type_fluid, verbose=verbose)
        self.fluid_to_bte(u, v, tt, dt)
        
        if (self.args.glow_op_split_scheme==0):
          v = self.step_bte(u, v, du, dv, tt, dt, None, verbose=verbose)
        elif (self.args.glow_op_split_scheme==1):          
          u, v = self.step_bte_vx_with_E(u, v, time, dt, verbose=verbose)
        
        self.bte_to_fluid(u, v, tt + 0.5 * dt, dt)
      
        if (self.heavies_freeze_E):
          self.bs_E = xp.dot(self.Dp, -self.solve_poisson(u[:, self.ele_idx], u[:, self.ion_idx], tt + 0.5 * dt))
        
        u = self.step_fluid(u, du, tt + 0.5 * dt, dt * 0.5,  self.ts_type_fluid, verbose=verbose)
        
      elif scheme=="first-order":
      
        # first order split scheme
        if (self.heavies_freeze_E):
          self.bs_E = xp.dot(self.Dp, -self.solve_poisson(u[:, self.ele_idx], u[:, self.ion_idx], tt))
          
        u = self.step_fluid(u, du, tt, dt, self.ts_type_fluid, verbose=verbose)
        self.fluid_to_bte(u, v, tt, dt)
        
        if (self.args.glow_op_split_scheme==0):
          v = self.step_bte(u, v, du, dv, tt, dt, None, verbose=verbose)
        elif (self.args.glow_op_split_scheme==1):          
          u, v = self.step_bte_vx_with_E(u, v, time, dt, verbose=verbose)
          
        
        self.bte_to_fluid(u, v, tt + dt, dt)         # bte to fluid
      
      if xp == cp:
        cp.cuda.runtime.deviceSynchronize()

      s_t2    = perf_counter()
      
      if(verbose==1):
        print("[glow step] time = %.4E T solve time %.4E (s)"%(tt, s_t2-s_t1))
      
      return u, v
        
    def solve(self, Uin, Vin, output_cycle_averaged_qois=False):
      tT              = self.args.cycles
      tt              = 0
      
      dt              = self.args.cfl 
      steps           = int(max(1,np.round(tT/dt)))
      
      print("T = %.4E RF cycles = %.1E dt = %.4E steps = %d atol = %.2E rtol = %.2E max_iter=%d"%(tT, self.args.cycles, dt, steps, self.args.atol, self.args.rtol, self.args.max_iter))
      Uin1, Vin1     = self.step_init(Uin, Vin, dt)
      xp             = self.xp_module
      u              = xp.copy(Uin1)
      v              = xp.copy(Vin1)
      
      du    = xp.zeros_like(u)
      dv    = xp.zeros_like(v)
      
      io_cycle = self.args.io_cycle_freq
      io_freq  = int(np.round(io_cycle/dt))
      
      cycle_freq = int(np.round(1/dt))
      
      cp_cycle = self.args.cp_cycle_freq
      cp_freq  = int(np.round(cp_cycle/dt))
      
      print("io freq = %d cycle_freq = %d cp_freq = %d" %(io_freq, cycle_freq, cp_freq))
      
      ts_idx_b  = 0 
      if args.restore==1:
        ts_idx_b = int(args.rs_idx * io_freq)
        tt       = ts_idx_b * dt
        print("restoring solver from ts_idx = ", int(args.rs_idx * io_freq), "at time = ",tt)
        
      if (output_cycle_averaged_qois == True):
        cycle_avg_u       = xp.zeros_like(u)
        cycle_avg_v       = xp.zeros_like(v)
      
      ele_idx           = self.ele_idx
      ion_idx           = self.ion_idx
      Te_idx            = self.Te_idx
      num_p             = self.op_spec_sp._p + 1
      num_sh            = len(self.op_spec_sp._sph_harm_lm)
      
      u0                = xp.copy(u)
      
      # 0  ,    1       ,  2          , 3          , 4 
      # ne , 1.5 neTe me,  mue E ne   , k_ela n0 ne, k_ion n0 ne  
      num_mc_qoi        = 5
      self.macro_qoi    = xp.zeros((num_mc_qoi, self.args.Np))
      output_macro_qoi  = bool(self.args.ca_macro_qoi)
      
      for ts_idx in range(ts_idx_b, steps):
        du[:,:]=0
        dv[:,:]=0
        
        if (ts_idx % cycle_freq == 0):
          u1 = xp.copy(u)
          
          a1 = xp.max(xp.abs(u1[:,0] -u0[:, 0]))
          a2 = xp.max(xp.abs((u1[:,0]-u0[:,0]) / xp.max(xp.abs(u0[:,0]))))
          print("||u(t+T) - u(t)|| = %.8E and ||u(t+T) - u(t)||/||u(t)|| = %.8E"% (a1, a2))
          
          u0=u1
          if output_macro_qoi == True:
            # macro_qoi 
            self.macro_qoi = 0.5 * dt  * self.macro_qoi / 1.0
            np.save("%s_macro_qoi_cycle_%04d.npy"%(args.fname, ts_idx//cycle_freq), self.macro_qoi)
            self.macro_qoi[:, :] = 0.0
          
        
        if (ts_idx % io_freq == 0):
          print("time = %.2E step=%d/%d"%(tt, ts_idx, steps))
          self.plot(u, v, "%s_%04d.png"%(args.fname, ts_idx//io_freq), tt)
          
          if(ts_idx % cp_freq == 0):
            xp.save("%s_%04d_u.npy"%(args.fname, ts_idx//io_freq), u)
            xp.save("%s_%04d_v.npy"%(args.fname, ts_idx//io_freq), v)
          
          if (output_cycle_averaged_qois == True):
            if ts_idx>ts_idx_b:
              cycle_avg_u       *= 0.5 * dt / io_cycle
              cycle_avg_v       *= 0.5 * dt / io_cycle
              #print(np.abs(1-cycle_avg_u[:, ion_idx]/u[:,ion_idx]))
              self.plot(cycle_avg_u, cycle_avg_v, "%s_avg_%04d.png"%(args.fname, ts_idx//io_freq), tt)
              
              if(ts_idx % cp_freq == 0):
                xp.save("%s_%04d_u_avg.npy"%(args.fname, ts_idx//io_freq), cycle_avg_u)
                xp.save("%s_%04d_v_avg.npy"%(args.fname, ts_idx//io_freq), cycle_avg_v)
                
              cycle_avg_u[:,:]       = 0
              cycle_avg_v[:,:]       = 0
              
            else:
              self.plot(u, v, "%s_avg_%04d.png"%(args.fname, ts_idx//io_freq), tt)
              
              if(ts_idx % cp_freq == 0):
                xp.save("%s_%04d_u_avg.npy"%(args.fname, ts_idx//io_freq), u)
                xp.save("%s_%04d_v_avg.npy"%(args.fname, ts_idx//io_freq), v)
          
        if (output_cycle_averaged_qois == True):
          cycle_avg_u     += u
          cycle_avg_v     += (v/u[ : , ele_idx])
          
          
          if output_macro_qoi == True:
            qoi = self.compute_qoi(u, v, tt)
            self.macro_qoi[0] += u[:, 0]   * self.param.np0
            self.macro_qoi[1] += u[:, 0]   * u[:, 2]   * scipy.constants.electron_mass * 1.5 * self.param.np0
            self.macro_qoi[2] += qoi["mu"] * u[:, 0] * self.param.np0 
            
            self.macro_qoi[3] += qoi["rates"][:, 0]  * u[:, 0] * self.param.np0 * self.param.n0 * self.param.np0
            self.macro_qoi[4] += qoi["rates"][:, 1]  * u[:, 0] * self.param.np0 * self.param.n0 * self.param.np0
          
          
        u , v  = self.step(u, v, du, dv, tt, dt, verbose=int(ts_idx % 100 == 0))
        
        if (output_cycle_averaged_qois == True):
          cycle_avg_u     += u
          cycle_avg_v     += (v/u[ : , ele_idx])
          
          if output_macro_qoi == True:
            qoi = self.compute_qoi(u, v, tt)
            self.macro_qoi[0] += u[:, 0]   * self.param.np0
            self.macro_qoi[1] += u[:, 0]   * u[:, 2] * scipy.constants.electron_mass * 1.5 * self.param.np0
            self.macro_qoi[2] += qoi["mu"] * u[:, 0] * self.param.np0 
            
            self.macro_qoi[3] += qoi["rates"][:, 0]  * u[:, 0] * self.param.np0 * self.param.n0 * self.param.np0
            self.macro_qoi[4] += qoi["rates"][:, 1]  * u[:, 0] * self.param.np0 * self.param.n0 * self.param.np0
        
        tt+= dt
        
        
      return u, v
    
    def evolve_1dbte_given_E(self, Uin, Vin, Et, output_cycle_averaged_qois=False):
      """
      Emode = 0 : static E
      Emode = 1 : time-periodic E
      """
      
      tT              = self.args.cycles
      tt              = 0
      
      dt              = self.args.cfl 
      steps           = int(max(1,np.round(tT/dt)))
      
      print("T = %.4E RF cycles = %.1E dt = %.4E steps = %d atol = %.2E rtol = %.2E max_iter=%d"%(tT, self.args.cycles, dt, steps, self.args.atol, self.args.rtol, self.args.max_iter))
      Uin1, Vin1     = self.step_init(Uin, Vin, dt)
      xp             = self.xp_module
      u              = xp.copy(Uin1)
      v              = xp.copy(Vin1)
      
      du             = xp.zeros_like(u)
      dv             = xp.zeros_like(v)
      
      io_cycle       = self.args.io_cycle_freq
      io_freq        = int(np.round(io_cycle/dt))
      
      cp_cycle       = self.args.cp_cycle_freq
      cp_freq        = int(np.round(cp_cycle/dt))
      
      ts_idx_b  = 0 
      if args.restore==1:
        ts_idx_b = int(args.rs_idx * io_freq)
        tt       = ts_idx_b * dt
        print("restoring solver from ts_idx = ", int(args.rs_idx * io_freq), "at time = ",tt)
        
      if (output_cycle_averaged_qois == True):
        cycle_avg_u       = xp.zeros_like(u)
        cycle_avg_v       = xp.zeros_like(v)
      
      ele_idx           = self.ele_idx
      ion_idx           = self.ion_idx
      Te_idx            = self.Te_idx
      num_p             = self.op_spec_sp._p + 1
      num_sh            = len(self.op_spec_sp._sph_harm_lm)
      
      for ts_idx in range(ts_idx_b, steps):
        du[:,:]=0
        dv[:,:]=0
        
        if (ts_idx % io_freq == 0):
          print("time = %.2E step=%d/%d"%(tt, ts_idx, steps))
          self.plot_unit_test1(u, v, "%s_%04d.png"%(args.fname, ts_idx//io_freq), tt, Et(tt))
          xp.save("%s_%04d_u.npy"%(args.fname, ts_idx//io_freq), u)
          xp.save("%s_%04d_v.npy"%(args.fname, ts_idx//io_freq), v)
          
          if (output_cycle_averaged_qois == True):
            if ts_idx>ts_idx_b:
              cycle_avg_u       *= 0.5 * dt / io_cycle
              cycle_avg_v       *= 0.5 * dt / io_cycle
              #print(np.abs(1-cycle_avg_u[:, ion_idx]/u[:,ion_idx]))
              self.plot_unit_test1(cycle_avg_u, cycle_avg_v, "%s_avg_%04d.png"%(args.fname, ts_idx//io_freq), tt, Et(tt))
              
              if (ts_idx % cp_freq == 0):
                xp.save("%s_%04d_u_avg.npy"%(args.fname, ts_idx//io_freq), cycle_avg_u)
                xp.save("%s_%04d_v_avg.npy"%(args.fname, ts_idx//io_freq), cycle_avg_v)
              
              cycle_avg_u[:,:]       = 0
              cycle_avg_v[:,:]       = 0
            else:
              self.plot_unit_test1(u, v, "%s_avg_%04d.png"%(args.fname, ts_idx//io_freq), tt, Et(tt))
              
              
              if (ts_idx % cp_freq == 0):
                xp.save("%s_%04d_u_avg.npy"%(args.fname, ts_idx//io_freq), u)
                xp.save("%s_%04d_v_avg.npy"%(args.fname, ts_idx//io_freq), v)
          
          
        
        if (output_cycle_averaged_qois == True):
          cycle_avg_u     += u
          cycle_avg_v     += (v/u[ : , ele_idx])
        
        v           = self.step_bte_x(v, tt, dt * 0.5)
        #self.bs_E   = Et(tt + 0.5 * dt)
        self.bs_E   = Et(tt)
        v           = self.step_bte_v(v, None, tt, dt, self.ts_type_bte_v , verbose=0)
        v           = self.step_bte_x(v, tt + 0.5 * dt, dt * 0.5)
        
        #v, a1, a2  = self.step_bte_vx_imp(v,None, tt, dt, None, 1)
        self.bte_to_fluid(u, v, tt + dt, dt)         # bte to fluid
        
        
        if (output_cycle_averaged_qois == True):
          cycle_avg_u     += u
          cycle_avg_v     += (v/u[ : , ele_idx])
        
        tt+= dt
        
      return u, v
    
    def evolve_1dbte_adv_x(self, Uin, Vin, output_cycle_averaged_qois=False):
      """
      Emode = 0 : static E
      Emode = 1 : time-periodic E
      """
      
      tT              = self.args.cycles
      tt              = 0
      
      dt              = self.args.cfl 
      steps           = int(max(1,np.round(tT/dt)))
      
      print("T = %.4E RF cycles = %.1E dt = %.4E steps = %d atol = %.2E rtol = %.2E max_iter=%d"%(tT, self.args.cycles, dt, steps, self.args.atol, self.args.rtol, self.args.max_iter))
      Uin1, Vin1     = self.step_init(Uin, Vin, dt)
      xp             = self.xp_module
      u              = xp.copy(Uin1)
      v              = xp.copy(Vin1)
      
      du             = xp.zeros_like(u)
      dv             = xp.zeros_like(v)
      
      io_cycle       = self.args.io_cycle_freq
      io_freq        = int(np.round(io_cycle/dt))
      
      cp_cycle       = self.args.cp_cycle_freq
      cp_freq        = int(np.round(cp_cycle/dt))
      
      ts_idx_b  = 0 
      if args.restore==1:
        ts_idx_b = int(args.rs_idx * io_freq)
        tt       = ts_idx_b * dt
        print("restoring solver from ts_idx = ", int(args.rs_idx * io_freq), "at time = ",tt)
        
      if (output_cycle_averaged_qois == True):
        cycle_avg_u       = xp.zeros_like(u)
        cycle_avg_v       = xp.zeros_like(v)
      
      ele_idx           = self.ele_idx
      ion_idx           = self.ion_idx
      Te_idx            = self.Te_idx
      num_p             = self.op_spec_sp._p + 1
      num_sh            = len(self.op_spec_sp._sph_harm_lm)
      Et                = lambda t : xp.zeros(len(self.xp))
      for ts_idx in range(ts_idx_b, steps):
        du[:,:]=0
        dv[:,:]=0
        
        if (ts_idx % io_freq == 0):
          print("time = %.2E step=%d/%d"%(tt, ts_idx, steps))
          self.plot_unit_test1(u, v, "%s_%04d.png"%(args.fname, ts_idx//io_freq), tt, Et(tt))
          xp.save("%s_%04d_u.npy"%(args.fname, ts_idx//io_freq), u)
          xp.save("%s_%04d_v.npy"%(args.fname, ts_idx//io_freq), v)
          
          if (output_cycle_averaged_qois == True):
            if ts_idx>ts_idx_b:
              cycle_avg_u       *= 0.5 * dt / io_cycle
              cycle_avg_v       *= 0.5 * dt / io_cycle
              #print(np.abs(1-cycle_avg_u[:, ion_idx]/u[:,ion_idx]))
              self.plot_unit_test1(cycle_avg_u, cycle_avg_v, "%s_avg_%04d.png"%(args.fname, ts_idx//io_freq), tt, Et(tt))
              
              if (ts_idx % cp_freq == 0):
                xp.save("%s_%04d_u_avg.npy"%(args.fname, ts_idx//io_freq), cycle_avg_u)
                xp.save("%s_%04d_v_avg.npy"%(args.fname, ts_idx//io_freq), cycle_avg_v)
              
              cycle_avg_u[:,:]       = 0
              cycle_avg_v[:,:]       = 0
            else:
              self.plot_unit_test1(u, v, "%s_avg_%04d.png"%(args.fname, ts_idx//io_freq), tt, Et(tt))
              
              
              if (ts_idx % cp_freq == 0):
                xp.save("%s_%04d_u_avg.npy"%(args.fname, ts_idx//io_freq), u)
                xp.save("%s_%04d_v_avg.npy"%(args.fname, ts_idx//io_freq), v)
          
          
        
        if (output_cycle_averaged_qois == True):
          cycle_avg_u     += u
          cycle_avg_v     += (v/u[ : , ele_idx])
        
        v           = self.step_bte_x(v, tt, dt * 0.5)
        v           = self.step_bte_x(v, tt + 0.5 * dt, dt * 0.5)
        
        #v, a1, a2  = self.step_bte_vx_imp(v,None, tt, dt, None, 1)
        self.bte_to_fluid(u, v, tt + dt, dt)         # bte to fluid
        
        
        if (output_cycle_averaged_qois == True):
          cycle_avg_u     += u
          cycle_avg_v     += (v/u[ : , ele_idx])
        
        tt+= dt
        
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
      
      E   = E / (self.param.V0/self.param.L)
      
      ele_idx = self.ele_idx
      ion_idx = self.ion_idx
      Te_idx  = self.Te_idx
      
      num_p   = self.op_spec_sp._p + 1
      num_sh  = len(self.op_spec_sp._sph_harm_lm)
      Vin_lm  = self.ords_to_sph(Vin)
      
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
      plt.semilogy(self.xp, self.param.np0 * ne, 'b',   label=r"$n_e$")
      plt.semilogy(self.xp, self.param.np0 * ni, '--r', label=r"$n_i$")
      plt.xlabel(r"x/L")
      plt.ylabel(r"$density (m^{-3})$")
      plt.grid(visible=True)
      plt.legend()
      
      plt.subplot(2, 4, 2)
      plt.plot(self.xp, self.param.np0 * ne, 'b',  label=r"$n_e$")
      plt.plot(self.xp, self.param.np0 * ni, '--r',label=r"$n_i$")
      plt.xlabel(r"x/L")
      #plt.ylabel(r"$(n_i -n_e)(m^{-3})$")
      plt.ylabel(r"$density (m^{-3})$")
      plt.legend()
      plt.grid(visible=True)
      
      plt.subplot(2, 4, 3)
      # plt.plot(self.xp, ne * Te * self.param.np0, 'b')
      # plt.ylabel(r"$n_e T_e (eV m^{-3})$")
      
      plt.plot(self.xp, Te, 'b')
      plt.ylabel(r"$T_e (eV)$")
      
      plt.xlabel(r"x/L")
      
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
      
      Vin_lm  = self.ords_to_sph(Vin)
      
      plt.subplot(2, 4, 6)
      Vin_lm1      = self.bte_eedf_normalization(Vin_lm)
      #r_elastic    = ne * self.param.np0 * asnumpy(xp.dot(self.op_rate[0], Vin_lm1[0::num_sh,:]))
      r_elastic    = asnumpy(xp.dot(self.op_rate[0], Vin_lm1[0::num_sh,:]))
      plt.semilogy(self.xp, r_elastic    , 'b', label="elastic")
      
      if (len(self.op_rate) > 1):
        #r_ionization = ne * self.param.np0 * asnumpy(xp.dot(self.op_rate[1], Vin_lm1[0::num_sh,:]))
        r_ionization = asnumpy(xp.dot(self.op_rate[1], Vin_lm1[0::num_sh,:]))
        plt.semilogy(self.xp, r_ionization , 'r', label="ionization")
      
      plt.xlabel(r"x/L")
      plt.ylabel(r"rate coefficients ($m^3 s^{-1}$)")
      #plt.ylabel(r"rate coefficients ($s^{-1}$)")
      plt.legend()
      plt.grid(visible=True)
      
      vth       = self.bs_vth
      kx_max    = self.op_spec_sp._basis_p._t_unique[-1]
      #ev_range  = (self.ev_lim[0] + 1e-6, (kx_max * vth/self.c_gamma)**2 - 1e-6) #((1e-1 * vth /self.c_gamma)**2, (self.vth_fac * vth /self.c_gamma)**2)
      #ev_range  = (self.ev_lim[0], self.ev_lim[1] * 4) #((1e-1 * vth /self.c_gamma)**2, (self.vth_fac * vth /self.c_gamma)**2)
      ev_range  = (self.ev_lim[0], self.ev_lim[1])
      
      ev_grid   = np.linspace(ev_range[0], ev_range[1], 1024)
      
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
    
    def rhs_bte_vx(self, v: np.array, time, dt):
      xp    = self.xp_module
      E     = self.bs_E
      Tg    = self.param.Tg
      n0    = self.param.n0
      
      v     = v.reshape((self.Nr, self.Nvt, self.Np))
      
      Fv1   = xp.einsum('pq,ijq->ijp', self.Dp,          v)
      Fv1   = xp.einsum('ik,kjq->ijq', self.op_adv_x,  Fv1)
      Fv1   = xp.einsum('ijq,j->ijq' , Fv1, self.xp_cos_vt).reshape((-1))
      
      v_lm  = self.ords_to_sph(v.reshape((self.Nr * self.Nvt, self.Np)))
    
      Fv2   = self.param.tau * (n0 * self.param.np0 * (xp.dot(self.op_col_en, v_lm) + Tg * xp.dot(self.op_col_gT, v_lm))  + E * xp.dot(self.op_adv_v, v_lm))
      Fv2   = self.sph_to_ords(Fv2).reshape((-1))
      
      Fv    = Fv2 - Fv1
    
      Fv    = Fv.reshape((self.Nr * self.Nvt, self.Np))
      return  Fv
    
    def step_bte_vx_imp(self, v0, dv, time, dt, ts_type, verbose):
      
      xp      = self.xp_module

      if xp == cp:
        cp.cuda.runtime.deviceSynchronize()

      s_t1    = perf_counter()
      
      Ndof      = self.Nr * self.Nvt * self.Np
      
      atol      = self.args.atol
      rtol      = self.args.rtol
      
      gmres_restart  = 5
      gmres_maxiter  = 1000
      
      
      def Lmat_mvec(x):
        y                   = self.rhs_bte_vx(x, time, dt)
        
        y[self.xp_vt_l,  0] = 0.0 
        y[self.xp_vt_r, -1] = 0.0 
        
        y                   = y.reshape((-1))
        y                   = x - dt * y
        return y.reshape((-1))
      
      def adv_x(v, time, dt): 
        y   = v.reshape((self.Nr, self.Nvt , self.Np)).reshape(self.Nr, self.Nvt * self.Np)
        y   = xp.dot(self.op_adv_x_qinv, y).reshape((self.Nr, self.Nvt, self.Np)).reshape((self.Nr * self.Nvt , self.Np))

        # enforce rhs BCs
        y[self.xp_vt_l, 0]  = 0.0
        y[self.xp_vt_r, -1] = 0.0

        y = xp.einsum("ijkl,ijl->ijk",self.bte_x_shift, y.reshape((self.Nr, self.Nvt, self.Np)))
        y  = y.reshape((self.Nr, self.Nvt *  self.Np))
        y  = xp.dot(self.op_adv_x_q, y).reshape((self.Nr , self.Nvt, self.Np)).reshape((self.Nr * self.Nvt, self.Np))
        return y
      
      pcEmat      = self.PmatE
      pcEval      = self.Evals
      pc_emat_idx = self.vspace_pc_setup()
      spec_sp     = self.op_spec_sp
      num_p       = spec_sp._p + 1
      num_sh      = len(spec_sp._sph_harm_lm)
      
      def Mmat_mvec(x):
        #@mfernando : I am not sure why but the order of the operator split scheme does matter here, 
        # for some reason, doing v-space step followed by the advection step tend to be more accurate
        xp   = self.xp_module
        #y    = self.step_bte_x(x, time, 0.5 * dt)
        
        y    = x.reshape((self.Nr * self.Nvt, self.Np))
        
        # y_lm = self.ords_to_sph(y)
        # y_lm = xp.dot(self.PmatC, y_lm)
        # y    = self.sph_to_ords(y_lm)
        
        y_lm = self.ords_to_sph(y)
        for idx_id, idx in enumerate(pc_emat_idx):
          y_lm[:,idx[1]] = xp.dot(pcEmat[idx[0]], y_lm[:, idx[1]])
        y    = self.sph_to_ords(y_lm)
        
        # x-space operators. 
        y    = adv_x(y, time, dt)
        return y.reshape((-1))
        
      Lmat_op    = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Lmat_mvec)
      Mmat_op    = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Mmat_mvec)
      
      b                   = xp.copy(v0).reshape((self.Nr * self.Nvt, self.Np))
      b[self.xp_vt_l, 0]  = 0.0
      b[self.xp_vt_r, -1] = 0.0
      b                   = b.reshape((-1))
      norm_b              = xp.linalg.norm(b)
      
      gmres_ctx           = glow1d_utils.gmres_counter(disp=False)
      v1, status          = cupyx.scipy.sparse.linalg.gmres(Lmat_op, b.reshape((-1)), x0=v0.reshape((-1)), tol=rtol, atol=atol, M=Mmat_op, callback=gmres_ctx, restart=gmres_restart, maxiter=gmres_maxiter)
      #v1          = Mmat_mvec(b); status=0;
      
      res           = Lmat_mvec(v1) -  b
      norm_res_abs  = xp.linalg.norm(res)
      norm_res_rel  = norm_res_abs / norm_b
      # if (status !=0) :
      #     print("GMRES solver failed! iterations =%d  ||res|| = %.4E ||res||/||b|| = %.4E"%(status, norm_res_abs, norm_res_rel))
      
      v1         = v1.reshape((self.Nr * self.Nvt, self.Np))
      v_lm       = self.ords_to_sph(v1)
      v1         = self.sph_to_ords(v_lm)
      
      if xp == cp:
        cp.cuda.runtime.deviceSynchronize()

      s_t2    = perf_counter()
      
      print("time = %.2E ||Ax-b|| = %.4E ||Ax-b|| / ||b|| = %.4E solve time = %.4E gmres iterations: %d " %(time, norm_res_abs, norm_res_rel, (s_t2-s_t1), gmres_ctx.niter * gmres_restart))
      return v1.reshape((self.Nr * self.Nvt, self.Np)), norm_res_abs, norm_res_abs
    
    def solve_unit_test3(self, Uin, Vin):
      dt              = self.args.cfl
      dt_bte          = self.args.cfl #* self.ts_op_split_factor
      tT              = self.args.cycles
      tt              = 0
      bte_steps       = int(dt/dt_bte)
      steps           = max(1,int(tT/dt))
      
      print("++++ Using backward Euler ++++")
      print("T = %.4E RF cycles = %.1E dt = %.4E steps = %d atol = %.2E rtol = %.2E max_iter=%d"%(tT, self.args.cycles, dt, steps, self.args.atol, self.args.rtol, self.args.max_iter))
      Uin1, Vin1     = self.step_init(Uin, Vin, dt)
      xp             = self.xp_module
      
      u = xp.copy(Uin1)
      v = xp.copy(Vin1)
      
      du    = xp.zeros_like(u)
      dv    = xp.zeros_like(v)
      dv_lm = xp.zeros((self.dof_v, self.Np))
      
      io_freq  = 1#int(0.1/dt)
      
      #self.bs_E       = 400 * xp.sin(2 * xp.pi * xp.asarray(self.xp)) #-xp.ones(len(self.xp)) * 400
      Emax            = 100
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
      
      
      #self.PmatC = xp.linalg.inv(self.I_Nv -  dt * self.param.tau * self.param.n0 * self.param.np0 * (self.op_col_en + self.param.Tg * self.op_col_gT) - dt * self.param.tau * Emax * self.op_adv_v)        
      self.initialize_bte_adv_x(dt)
      for ts_idx in range(ts_idx_b, steps):
        du[:,:]=0
        dv[:,:]=0
        
        self.bte_to_fluid(u, v, tt, dt)
        #self.bs_E       = Et[ts_idx % 1000]  #xp.ones(len(self.xp)) * Emax * xp.sin(2* xp.pi * tt)
        #self.bs_E       = Emax * xp.sin(2* xp.pi * tt) * xp.ones(len(self.xp))
        
        if (ts_idx % io_freq == 0):
          self.plot_unit_test1(u, v, "%s_%04d.png"%(args.fname, ts_idx//io_freq), tt, (self.bs_E * self.param.L /self.param.V0), plot_ionization=True)
          #xp.save("%s_%04d_u.npy"%(args.fname, ts_idx//io_freq), u)
          #xp.save("%s_%04d_v.npy"%(args.fname, ts_idx//io_freq), v)
          
        v, _, _ = self.step_bte_vx_imp(v, None, tt, dt, None, 0)
        
        v       = v.reshape((self.Nr * self.Nvt, self.Np))
        v_lm    = self.ords_to_sph(v)
        v       = self.sph_to_ords(v_lm)
        tt+= dt
        
        
        
      return u, v
    
    def compute_qoi(self, Uin, Vin, time):
      xp       = self.xp_module
      
      # phi      = self.solve_poisson(u[:,self.ele_idx], u[:, self.ion_idx], time)
      # E        = -xp.dot(self.Dp, phi) * self.param.V0/self.param.L
      # EbyN     = E/self.param.n0/self.param.np0
      
      u        = Uin
      v        = Vin
      
      mm_fac   = np.sqrt(4 * np.pi) 
      c_gamma  = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))
      vth      = self.bs_vth

      v_lm     = self.ords_to_sph(v)
      
      m0       = xp.dot(self.op_mass, v_lm)
      Te       = xp.dot(self.op_temp,v_lm) / m0
      scale    = xp.dot(self.op_mass / mm_fac, v_lm) * (2 * (vth/c_gamma)**3)
      v_lm_n   = v_lm/scale
      num_sh   = len(self.op_spec_sp._sph_harm_lm)
      
      n0       = self.param.np0 * self.param.n0
      num_collisions = len(self.bs_coll_list)
      rr_rates = xp.array([xp.dot(self.op_rate[col_idx], v_lm_n[0::num_sh, :]) for col_idx in range(num_collisions)]).reshape((num_collisions, self.Np)).T
      
      # these are computed from SI units qoi/n0
      D_e      = xp.dot(self.op_diffusion, v_lm_n[0::num_sh]) * (c_gamma / 3.) / n0 
      mu_e     = xp.dot(self.op_mobility, v_lm_n[1::num_sh])  * ((c_gamma / (3 * (1 / n0)))) /n0
      
      return {"rates": rr_rates, "mu": mu_e, "D": D_e}
      
    def solve_hybrid(self, Uin, Vin,  dt_bte, dt_fluid):
      """
      Hybrid solver to reach steady-state
      Uin: macroscopic vars
      Vin: BTE for electrons
      
      dt_bte  : glow with BTE timestep size
      dt_fluid: glow with fluid model timestep size 
      """
      
      fluid_args            = glow1d_fluid_args(self.args)
      glow1d_macro_model    = glowdischarge_1d.glow1d_fluid(fluid_args)
      glow1d_macro_model.initialize()
      glow1d_macro_model.initialize_kinetic_coefficients(mode="tabulated")
      
      if (fluid_args.use_gpu==1):
        assert fluid_args.use_tab_data == 0
        glow1d_macro_model.copy_operators_H2D(fluid_args.gpu_device_id)
        glow1d_macro_model.xp_module = cp
        glow1d_macro_model.initialize_kinetic_coefficients(mode="fixed-0")
        
      
      u, v                  = self.step_init(Uin, Vin, dt_bte)
      bte_steps_per_cycle   = (int)(1/dt_bte)
      bte_cyles             = 1
      
      fluid_steps_per_cycle = (int)(1/dt_fluid)
      fluid_cyles           = 100
      
      
      while(1):
        
        glow1d_macro_model.initialize_kinetic_coefficients(mode="tabulated")
        xp                    = self.xp_module
        
        u                     = xp.asarray(u)
        v                     = xp.asarray(v)
        
        u0                    = xp.copy(u)
        v0                    = xp.copy(v)

        du                    = xp.zeros_like(u)
        dv                    = xp.zeros_like(v)

        cycle_avg_u           = xp.zeros_like(u)
        cycle_avg_v           = xp.zeros_like(v)
        io_cycle              = 1.00
        
        num_t_pts             = 100
        mobility_t            = xp.zeros((self.Np, num_t_pts + 1))
        diffusion_t           = xp.zeros((self.Np, num_t_pts + 1))
        rates_t               = xp.zeros((self.Np, len(self.bs_coll_list), num_t_pts + 1))
        kinetic_freq          = bte_steps_per_cycle // num_t_pts
        tt_grid               = np.linspace(0, 1, num_t_pts+1)         
        
        for ts_idx in range(bte_cyles * bte_steps_per_cycle + 1):
          tt = ts_idx * dt_bte
          
          if (ts_idx % kinetic_freq == 0):
            ele_kinetics          = self.compute_qoi(u, v, tt)
            idx                   = ts_idx//kinetic_freq
            mobility_t [:, idx]   = ele_kinetics["mu"]
            diffusion_t[:, idx]   = ele_kinetics["D"]
            rates_t    [:,:, idx] = ele_kinetics["rates"]
          
          if (ts_idx> 0 and ts_idx%bte_steps_per_cycle==0):
            a1 = xp.linalg.norm(u-u0)
            r1 = a1/xp.linalg.norm(u0)
            print("GLOW-BTE time = %.4E T ||u1-u0|| = %.4E ||u1-u0||/||u0|| = %.4E "%(ts_idx * dt_bte, a1, r1))
            
            cycle_avg_u       *= 0.5 * dt_bte / io_cycle
            cycle_avg_v       *= 0.5 * dt_bte / io_cycle
            
            self.plot(u, v,                     "%s_bte_%04d.png"%(self.args.fname, (int)(tt))    , tt)
            self.plot(cycle_avg_u, cycle_avg_v, "%s_bte_avg_%04d.png"%(self.args.fname, (int)(tt)), tt)
            
            u0 = xp.copy(u)
            v0 = xp.copy(v)

          if (ts_idx == bte_cyles * bte_steps_per_cycle):
            break
          
          cycle_avg_u     += u
          cycle_avg_v     += (v/u[ : , self.ele_idx])
          
          u, v = self.step(u, v, du, dv, tt, dt_bte, scheme="strang-splitting", verbose=0)
          
          cycle_avg_u     += u
          cycle_avg_v     += (v/u[ : , self.ele_idx])
          
        pp                = glow1d_macro_model.param
        xp                = glow1d_macro_model.xp_module
        
        if (fluid_args.use_gpu==0):
          u                     = cp.asnumpy(u)
          # ele_kinetics["rates"] = cp.asnumpy(ele_kinetics["rates"])
          # ele_kinetics["mu"]    = cp.asnumpy(ele_kinetics["mu"])
          # ele_kinetics["D"]     = cp.asnumpy(ele_kinetics["D"])
          
          mobility_t  = cp.asnumpy(mobility_t)
          diffusion_t = cp.asnumpy(diffusion_t)
          rates_t     = cp.asnumpy(rates_t)
          
          mobility_t  = scipy.interpolate.interp1d(tt_grid, mobility_t)
          diffusion_t = scipy.interpolate.interp1d(tt_grid, diffusion_t)
          rates_t     = scipy.interpolate.interp1d(tt_grid, rates_t)
          
        nTe     = u[:, self.ele_idx] * u[:, self.Te_idx]
        ne      = u[:, self.ele_idx]
        Te      = nTe/ne
        tt_inp  = np.linspace(0, 1, 10)
        
        plt.figure(figsize=(21, 8), dpi=200)
        plt.subplot(1, 3, 1)
        plt.semilogy(self.xp, cp.asnumpy(pp.ki(nTe, ne)), 'k--', label=r"0D-BTE")
        
        for i in range(len(tt_inp)):
          plt.semilogy(self.xp, rates_t(tt_inp[i])[:,1] * pp.tau * pp.np0, label=r"ki @ (t=%.2E)"%(tt_inp[i]))

        plt.grid(visible=True)
        plt.title(r"ki")
        plt.xlabel(r"x")
        plt.legend()
        
        
        plt.subplot(1, 3, 2)
        plt.semilogy(self.xp, cp.asnumpy(pp.mu_e(nTe, ne)), 'k--', label="0D-BTE")
        for i in range(len(tt_inp)):
          plt.semilogy(self.xp, mobility_t(tt_inp[i]) * (1.0/pp.n0/pp.np0) * (pp.V0 * pp.tau/(pp.L**2)), label=r"$\mu_e$ @ (t=%.2E)"%(tt_inp[i]))
        
        plt.grid(visible=True)
        plt.title(r"$\mu_e$")
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.semilogy(self.xp, cp.asnumpy(pp.De(nTe, ne)), 'k--', label="0D-BTE")
        for i in range(len(tt_inp)):
          plt.semilogy(self.xp, diffusion_t(tt_inp[i]) * (1.0/pp.n0/pp.np0) * (pp.tau/(pp.L**2)), label=r"$D_e$ @ (t=%.2E)"%(tt_inp[i]))
          
        plt.grid(visible=True)
        plt.title(r"$D_e$")
        plt.legend()
        
        plt.savefig("%s_kinetics.png"%(self.args.fname))
        plt.close()
        
        u[:, self.Te_idx] = u[:, self.ele_idx] * u[:, self.Te_idx]
        du1 = xp.zeros_like(u)
        u0  = xp.copy(u)
        
        for ts_idx in range(fluid_cyles * fluid_steps_per_cycle + 1):
          tt                = ts_idx * dt_fluid
          tt_inp            = (ts_idx % fluid_steps_per_cycle) * dt_fluid 
          
          pp.ki             = lambda nTe, ne : rates_t(tt_inp)[:,1] * pp.tau * pp.np0
          pp.ki_nTe         = lambda nTe, ne : 0 * ne
          pp.ki_ne          = lambda nTe, ne : 0 * ne
          
          pp.mu_e           = lambda nTe, ne : (mobility_t(tt_inp)/pp.n0/pp.np0) * (pp.V0 * pp.tau/(pp.L**2)) 
          pp.mu_e_nTe       = lambda nTe, ne : 0 * ne
          pp.mu_e_ne        = lambda nTe, ne : 0 * ne
          
          pp.De             = lambda nTe, ne : (diffusion_t(tt_inp)/pp.n0/pp.np0) * (pp.tau/(pp.L**2))
          pp.De_nTe         = lambda nTe, ne : 0 * ne
          pp.De_ne          = lambda nTe, ne : 0 * ne
          
          
          if (ts_idx> 0 and ts_idx%fluid_steps_per_cycle==0):
            
            a1 = xp.linalg.norm(u-u0)
            r1 = a1/xp.linalg.norm(u0)
            
            print("GLOW-FLUID time = %.4E T ||u1-u0|| = %.4E ||u1-u0||/||u0|| = %.4E "%(ts_idx * dt_fluid, a1, r1))
            
            glow1d_macro_model.plot(u   , tt, "%s_fluid_%04d.png"%(glow1d_macro_model.args.fname, (int)(tt)))
            #glow1d_macro_model.plot(u-u0, tt, "%s_fluid_diff.png"%(glow1d_macro_model.args.fname))
            u0 = xp.copy(u)
          
          if (ts_idx == fluid_cyles * fluid_steps_per_cycle):
            break
            
          v, _ = glow1d_macro_model.solve_step(u, du1, tt, dt_fluid, fluid_args.atol, fluid_args.rtol, fluid_args.max_iter)
          du1  = v-u
          u    = v
          # Te   = u[:, self.Te_idx]/u[:, self.ele_idx]
          # Te[Te<0] = 1.5
          # u[:, self.Te_idx] = Te * u[:, self.ele_idx]
          
          
        
        u[:, self.Te_idx] = u[:, self.Te_idx]/u[:, self.ele_idx]
        v  = self.initialize_maxwellian_eedf(u[:, self.ele_idx], u[:, self.Te_idx])
    
    def fft_analysis(self, Uin, Vin, dt, atol, rtol, max_iter):
      xp              = self.xp_module
      tT              = 1
      steps           = int(max(1,np.round(tT/dt)))
      
      print("T = %.4E RF cycles = %.1E dt = %.4E steps = %d atol = %.2E rtol = %.2E max_iter=%d"%(tT, self.args.cycles, dt, steps, self.args.atol, self.args.rtol, self.args.max_iter))
      Uin1, Vin1     = self.step_init(Uin, Vin, dt)
      xp             = self.xp_module
      u              = xp.copy(Uin1)
      v              = xp.copy(Vin1)
      
      du             = xp.zeros_like(u)
      dv             = xp.zeros_like(v)
      
      io_cycle       = 1.00
      io_freq        = int(np.round(io_cycle/dt))
      
      ele_idx        = self.ele_idx
      ion_idx        = self.ion_idx
      Te_idx         = self.Te_idx
      num_p          = self.op_spec_sp._p + 1
      num_sh         = len(self.op_spec_sp._sph_harm_lm)
      
      extract_freq   = io_freq//io_freq
      x_idx          = list(range(0, self.Np, 10))
      ut             = xp.zeros(tuple(list(u.shape) + [steps//extract_freq +1]))
      ut1            = xp.zeros(tuple(list(u.shape) + [steps//extract_freq +1]))
      vt             = xp.zeros(tuple([num_p * num_sh, len(x_idx)] + [steps//extract_freq +1]))
      tt             = 0
      for ts_idx in range(0, steps+1):
        
        if (ts_idx % extract_freq == 0):
          vlm                    = self.ords_to_sph(v)
          uu                     = xp.copy(u)
          uu[:, ele_idx]         = xp.dot(self.op_mass[0::num_sh], vlm[0::num_sh,:])
          uu[:, Te_idx]          = xp.dot(self.op_temp[0::num_sh], vlm[0::num_sh,:])/u[:,ele_idx]
          
          ut [:,:, ts_idx//extract_freq] = u
          ut1[:,:, ts_idx//extract_freq] = uu
          vt [:,:, ts_idx//extract_freq] = vlm[:, x_idx]
          
        
        if (ts_idx % io_freq == 0):
          print("time = %.2E step=%d/%d"%(tt, ts_idx, steps))
          self.plot(u, v, "%s_%04d.png"%(args.fname, ts_idx//io_freq), tt)
          
        if(ts_idx==steps):
          break
        
        u , v  = self.step(u, v, du, dv, tt, dt)
        # v      = v.reshape((self.Nr * self.Nvt, self.Np))
        # v      = self.sph_to_ords(self.ords_to_sph(v))
        tt    += dt
      
      return ut, ut1, vt
    
    def svd_analysis(self, Uin, Vin, dt, atol, rtol, max_iter):
      xp              = self.xp_module
      tT              = 10
      steps           = int(max(1,np.round(tT/dt)))
      
      print("T = %.4E RF cycles = %.1E dt = %.4E steps = %d atol = %.2E rtol = %.2E max_iter=%d"%(tT, self.args.cycles, dt, steps, self.args.atol, self.args.rtol, self.args.max_iter))
      Uin1, Vin1     = self.step_init(Uin, Vin, dt)
      xp             = self.xp_module
      u              = xp.copy(Uin1)
      v              = xp.copy(Vin1)
      
      du             = xp.zeros_like(u)
      dv             = xp.zeros_like(v)
      
      io_cycle       = 1.00
      io_freq        = int(np.round(io_cycle/dt))
      
      ele_idx        = self.ele_idx
      ion_idx        = self.ion_idx
      Te_idx         = self.Te_idx
      num_p          = self.op_spec_sp._p + 1
      num_sh         = len(self.op_spec_sp._sph_harm_lm)
      
      extract_freq   = io_freq//20
      ut             = xp.zeros(tuple(list(u.shape) + [steps//extract_freq +1]))
      x_idx          = list(range(0, self.Np, 1))
      vt             = xp.zeros(tuple([num_p * num_sh, len(x_idx)] + [steps//extract_freq +1]))
      tt             = 0
      for ts_idx in range(0, steps+1):
        
        if (ts_idx % extract_freq == 0):
          vlm = self.ords_to_sph(v)
          
          ut[:,:, ts_idx//extract_freq] = u
          vt[:,:, ts_idx//extract_freq] = vlm[:, x_idx]
        
        if (ts_idx % io_freq == 0):
          print("time = %.2E step=%d/%d"%(tt, ts_idx, steps))
          self.plot(u, v, "%s_%04d.png"%(args.fname, ts_idx//io_freq), tt)
          
        if(ts_idx==steps):
          break
        
        u , v  = self.step(u, v, du, dv, tt, dt)
        tt    += dt
      
      return ut, vt
    

def args_parse():
  parser = argparse.ArgumentParser()
  parser.add_argument("-threads", "--threads"                       , help="number of cpu threads (boltzmann operator assembly)", type=int, default=4)
  parser.add_argument("-out_fname", "--out_fname"                   , help="output file name for the qois", type=str, default="bte_glow1d")
  parser.add_argument("-l_max", "--l_max"                           , help="max polar modes in SH expansion", type=int, default=1)
  parser.add_argument("-c", "--collisions"                          , help="collisions model", type=str, default="lxcat_data/eAr_crs.Biagi.3sp2r")
  parser.add_argument("-ev_max", "--ev_max"                         , help="energy max v-space grid (eV)" , type=float, default=50)
  parser.add_argument("-ev_extend", "--ev_extend"                   , help="energy max boundary extenstion (0 = no extention, 1= 1.2 ev_max, 2 = 25ev_max)", type=int, default=2)
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
  parser.add_argument("-gmres_atol", "--gmres_atol"                 , help="abs. tolerance for gmres-solve" , type=float, default=1e-20)
  parser.add_argument("-gmres_rtol", "--gmres_rtol"                 , help="rel. tolerance for gmres-solve" , type=float, default=1e-1)
  parser.add_argument("-gmres_rsrt", "--gmres_rsrt"                 , help="gmres restarts", type=int, default=10)

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

  parser.add_argument("-ic_neTe"  , "--ic_neTe"                     , help="ic file written with neTe" , type=int, default=0)
  parser.add_argument("-ca_macro_qoi"  , "--ca_macro_qoi"           , help="use Te based tabulated electron kinetic coefficients" , type=int, default=0)

  parser.add_argument("-fname", "--fname"                           , help="file name to store the solution" , type=str, default="1d_glow")
  parser.add_argument("-dir"  , "--dir"                             , help="file name to store the solution" , type=str, default="glow1d_dir")
  parser.add_argument("-glow_op_split_scheme"  , "--glow_op_split_scheme" , help="glow op. split scheme, 0- E field is frozen while BTE solve,1-E field is solved with BTE solve" , type=int, default=0)

  parser.add_argument("-par_file"        , "--par_file"                   , help="toml par file to specify run parameters" , type=str, default="")
  parser.add_argument("-ic_file"         , "--ic_file"                    , help="initial condition file"                  , type=str, default="")
  parser.add_argument("-io_cycle_freq"   , "--io_cycle_freq"              , help="io output every k-th cycle"              , type=float, default=1e0)
  parser.add_argument("-cp_cycle_freq"   , "--cp_cycle_freq"              , help="checkpoint output every k-th cycle"      , type=float, default=1e1)
  parser.add_argument("-bte_with_E"      , "--bte_with_E"                 , help="only do 1D-bte with E"                   , type=int, default=0)
  parser.add_argument("-vtDe"            , "--vtDe"                       , help="vtheta smoothing parameter"              , type=float, default=0)
  parser.add_argument("-xadv_type"       , "--xadv_type"                  , help="x-adv type 0-cheb, 1-upwinded FD"        , type=int, default=0)
  args  = parser.parse_args()

  if args.par_file != "":
    import toml
    tp  = toml.load(args.par_file)
    
    tp0                 = tp["bte"] 
    args.Np             = tp0["Np"]
    args.l_max          = tp0["lmax"]
    args.collisions     = tp0["collisions"]
    args.ev_max         = tp0["ev_max"]
    args.ev_extend      = tp0["ev_extend"]
    args.sp_order       = tp0["sp_order"]
    args.spline_qpts    = tp0["spline_qpts"]
    args.Te             = tp0["Te"]
    args.Nr             = tp0["Nr"]
    args.Nvt            = tp0["Nvt"]
    args.Np             = tp0["Np"]
    args.cfl            = tp0["dt"]
    args.cycles         = tp0["cycles"]
    args.vtDe           = tp0["vtDe"]
    args.xadv_type      = tp0["xadv_type"]

    tp0                 = tp["glow_1d"] 
    args.Tg             = tp0["Tg"]
    
    tp0                 = tp["solver"]
    args.atol           = tp0["atol"]
    args.rtol           = tp0["rtol"]
    args.gmres_atol     = tp0["gmres_atol"]
    args.gmres_rtol     = tp0["gmres_rtol"]
    args.gmres_rsrt     = tp0["gmres_rsrt"]
    args.max_iter       = tp0["max_iter"]
    args.use_gpu        = tp0["use_gpu"]
    args.gpu_device_id  = tp0["gpu_device_id"]
    args.restore        = tp0["restore"]
    args.rs_idx         = tp0["rs_idx"]
    args.fname          = tp0["fname"]
    args.dir            = tp0["dir"]
    args.ic_file        = tp0["ic_file"]
    args.io_cycle_freq  = tp0["io_cycle"]
    args.cp_cycle_freq  = tp0["cp_cycle"]
    
    args.glow_op_split_scheme = tp0["split_scheme"]
    
    tp0                 = tp["chemistry"]
    args.Ns             = tp0["Ns"]
    args.NT             = tp0["NT"]

    print("intiailized solver using %s"%args.par_file)


  return args  

if __name__ == "__main__":
  args    = args_parse()  
  glow_1d = glow1d_boltzmann(args)
  u, v    = glow_1d.initialize()

  if args.use_gpu==1:
    gpu_device = cp.cuda.Device(args.gpu_device_id)
    gpu_device.use()

  if(args.bte_with_E == 1):
    xp        = cp #glow_1d.xp_module 
    a0        = xp.ones(glow_1d.Np) * 1e4  #xp.asarray(glow_1d.xp)**7 * 8e4  #xp.ones(glow_1d.Np) * 5e4
    Et        = lambda tt : a0 * xp.sin(2 * np.pi * tt)
    uu,vv     = glow_1d.evolve_1dbte_given_E(u, v, Et, output_cycle_averaged_qois=True)
  elif(args.bte_with_E == 2):
    xp        = cp #glow_1d.xp_module 
    a0        = xp.ones(glow_1d.Np) * 1e4  * xp.asarray(glow_1d.xp)**7 #* 8e4  #xp.ones(glow_1d.Np) * 5e4
    Et        = lambda tt : a0 * xp.sin(2 * np.pi * tt)
    uu,vv     = glow_1d.evolve_1dbte_given_E(u, v, Et, output_cycle_averaged_qois=True)
  elif(args.bte_with_E == 3):
    xp        = cp #glow_1d.xp_module 
    a0        = xp.ones(glow_1d.Np) * 0.0  #* 8e4  #xp.ones(glow_1d.Np) * 5e4
    Et        = lambda tt : a0 * xp.sin(2 * np.pi * tt)
    uu,vv     = glow_1d.evolve_1dbte_given_E(u, v, Et, output_cycle_averaged_qois=True)
  elif(args.bte_with_E == 4):
    uu,vv     = glow_1d.evolve_1dbte_adv_x(u, v, output_cycle_averaged_qois=True)
  else:
    uu,vv     = glow_1d.solve(u, v, output_cycle_averaged_qois=True)
    
  # ut, ut1, vt = glow_1d.fft_analysis(u, v, args.cfl, args.atol, args.rtol, args.max_iter)
  # xp.save("%s/ut_bte.npy"%(args.dir) , ut)
  # xp.save("%s/ut1_bte.npy"%(args.dir), ut1)
  # xp.save("%s/vt_bte.npy"%(args.dir) , vt)

  # ut, vt   = glow_1d.svd_analysis(u, v, args.cfl, args.atol, args.rtol, args.max_iter)
  # xp.save("%s/ut_bte_svd.npy"%(args.dir), ut)
  # xp.save("%s/vt_bte_svd.npy"%(args.dir), vt)

  # xp.save("ut_bte_svd.npy", ut)
  # xp.save("vt_bte_svd.npy", vt)

  #uu,vv   = glow_1d.solve_hybrid(u, v, args.cfl, 5e-3)
  #uu,vv   = glow_1d.solve_unit_test2(u, v, 1)
  #uu,vv   = glow_1d.solve_unit_test3(u, v)

