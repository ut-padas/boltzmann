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
import cross_section
from time import perf_counter, sleep

CUDA_NUM_DEVICES      = 0
PROFILE_SOLVERS       = 0

try:
  import cupy as cp
  #CUDA_NUM_DEVICES=cp.cuda.runtime.getDeviceCount()
except ImportError:
  print("Please install CuPy for GPU use")
  sys.exit(0)
except:
  print("CUDA not configured properly !!!")
  sys.exit(0)


parser = argparse.ArgumentParser()
parser.add_argument("-threads", "--threads"                       , help="number of cpu threads (boltzmann operator assembly)", type=int, default=4)
parser.add_argument("-out_fname", "--out_fname"                   , help="output file name for the qois", type=str, default="bte_glow1d")
parser.add_argument("-l_max", "--l_max"                           , help="max polar modes in SH expansion", type=int, default=1)
parser.add_argument("-c", "--collisions"                          , help="collisions model", type=str, default="lxcat_data/eAr_crs.Biagi.3sp2r")
parser.add_argument("-ev_max", "--ev_max"                         , help="energy max v-space grid (eV)" , type=float, default=50)
parser.add_argument("-ev_extend", "--ev_extend"                   , help="energy max boundary extenstion (0 = no extention, 1= 1.2 ev_max, 2 = 25ev_max)", type=int, default=2)
parser.add_argument("-sp_order", "--sp_order"                     , help="b-spline order", type=int, default=3)
parser.add_argument("-spline_qpts", "--spline_qpts"               , help="q points per knots", type=int, default=5)
parser.add_argument("-Te", "--Te"                                 , help="approximate electron temperature (eV) to  compute thermal velocity"  , type=float, default=0.5)
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


class glow1d_boltzmann():
    """
    perform glow discharge simulation with electron Boltzmann solver
    """
    def __init__(self, args) -> None:
      
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
      
      self.Ns        = self.args.Ns                    # Number of species
      self.NT        = self.args.NT                    # Number of temperatures
      self.Nv        = self.args.Ns + self.args.NT     # Total number of 'state' variables
      
      self.Nr        = self.args.Nr + 1                # e-boltzmann number of radial dof  
      self.Nvt       = self.args.Nvt                   # e-boltzmann number of ordinates
      self.dof_v     = self.Nr * (self.args.l_max+1)   # currently we assume azimuthal symmetry in the v-space
      
      self.deg       = self.args.Np-1                 # degree of Chebyshev polys we use
      self.Np        = self.args.Np                   # Number of points used to define state in space
      self.Nc        = self.args.Np-2                 # number of collocation pts (Np-2 b/c BCs)
      
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
      self.LpD         = np.identity(self.Np)
      self.LpD[1:-1,:] = self.Lp[1:-1,:]
      self.LpD_inv     = np.linalg.solve(self.LpD, np.eye(self.Np)) 
      
      
      Imat               = np.eye(self.Np)
      Imat[0,0]          = Imat[-1,-1] = 0
      self.phi_ni        =  np.linalg.solve(self.LpD, -self.param.alpha*Imat)
      self.phi_ne        = -self.phi_ni
      self.E_ni          = -np.dot(self.Dp, self.phi_ni)
      self.E_ne          = -np.dot(self.Dp, self.phi_ne)
      
      Imat[0,0]          = Imat[-1,-1] = 1.0
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
      
      P0 = np.dot(self.op_po2sh, self.op_psh2o)
      P1 = np.dot(self.op_psh2o, self.op_po2sh)
      
      I0 = np.eye(P0.shape[0])
      I1 = np.eye(P1.shape[0])
      
      print("||I0 - P0|| = %.8E \n" %(np.linalg.norm(I0-P0)/np.linalg.norm(I0)))
      print("||I1 - P1|| = %.8E \n" %(np.linalg.norm(I1-P1)/np.linalg.norm(I1)))
      
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
      
      if type==0:
        if self.args.restore==1:
          print("~~~restoring solver from %s.npy"%(args.fname))
          Uin = xp.load("%s_%04d_u.npy"%(args.fname, args.rs_idx))
          Vin = xp.load("%s_%04d_v.npy"%(args.fname, args.rs_idx))
        else:
          xx = self.param.L * (self.xp + 1)
          read_from_file   = False 
          if read_from_file==True:
            fname = "1dglow/1d_glow_1000_fluid.npy"
            print("loading initial conditoin from ", fname)
            fluid_U         = xp.load(fname)
            Uin[:, ele_idx] = fluid_U[:, ele_idx] 
            Uin[:, ion_idx] = fluid_U[:, ion_idx] 
            Uin[:, Te_idx]  = fluid_U[:, Te_idx]  / Uin[:, ele_idx]
          else:
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
              self.r_rates[:, self.ion_idx] = xp.dot(self.op_rate[1], hh1[0::num_sh]) * self.param.np0 * self.param.tau
              print("k_ionization [m^3s^{-1}] = %.8E " %(xp.dot(self.op_rate[1], hh1[0::num_sh])))
            
            Vin[:, i] = xp.dot(self.op_psh2o, h_init)
            
          # scale functions to have ne, at initial timestep
          Vin = Vin * Uin[:,ele_idx]
        
        self.mu[:, ele_idx] = self.param._mu_e
        self.D[: , ele_idx] = self.param._De
        
        self.mu[:, ion_idx] = self.param.mu_i
        self.D[: , ion_idx] = self.param.Di
        
        
          
      else:
        raise NotImplementedError
      
      return Uin, Vin
    
    def bte_eedf_normalization(self, v_lm):
      xp = self.xp_module
      # normalization of the distribution function before computing the reaction rates
      mm_fac                   = self.op_spec_sp._sph_harm_real(0, 0, 0, 0) * 4 * np.pi
      mm_op                    = self.op_mass
      c_gamma                  = self.c_gamma
      vth                      = self.bs_vth
      
      scale                    = xp.dot(mm_op / mm_fac, v_lm) * (2 * (vth/c_gamma)**3)
      return v_lm / scale
  
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
    
    def filter_ordinates(self, v:np.array):
      xp    = self.xp_module
      v     = v.reshape((self.Nr * self.Nvt, self.Np))
      v_lm  = xp.dot(self.op_po2sh, v)
      return xp.dot(self.op_psh2o, v_lm).reshape((-1))
    
    def rhs_bte(self, v:np.array, Q:dict, time, dt):
      xp    = self.xp_module
      
      E     = Q["E"]
      n0    = Q["n0"]
      Tg    = Q["Tg"]
      
      v     = v.reshape((self.Nr, self.Nvt, self.Np))
      
      Fv1   = xp.einsum('pq,ijq->ijp', self.Dp,          v)
      Fv1   = xp.einsum('ik,kjq->ijq', self.op_adv_x,  Fv1)
      Fv1   = xp.einsum('ijq,j->ijq' , Fv1, self.xp_cos_vt).reshape((-1))
      
      v_lm  = xp.dot(self.op_po2sh, v.reshape((self.Nr * self.Nvt, self.Np)))
    
      Fv2   = self.param.tau * (n0 * self.param.np0 * (xp.dot(self.op_col_en, v_lm) + Tg * xp.dot(self.op_col_gT, v_lm))  + E * xp.dot(self.op_adv_v, v_lm))
      Fv2   = xp.dot(self.op_psh2o, Fv2).reshape((-1))
      
      Fv    = Fv2 - Fv1
    
      Fv    = Fv.reshape((self.Nr * self.Nvt, self.Np))
      return  Fv.reshape((-1))
    
    def pc_setup(self, Q, dt):
      xp = self.xp_module
      
      E     = Q["E"]
      n0    = Q["n0"]
      Tg    = Q["Tg"]
      
      bte_x_shift      = xp.zeros((self.Nr, self.Nvt, self.Np, self.Np))
      #self.bte_x_shift_rmat = xp.zeros((self.Nr, self.Nvt, self.Np, self.Np))
      DpL = xp.zeros((self.Np, self.Np))
      DpR = xp.zeros((self.Np, self.Np))

      DpL[1:,:]  = self.Dp[1:,:]
      DpR[:-1,:] = self.Dp[:-1,:]
    
      f1 = 1.0
      #f2 = 1-f1

      for j in range(self.Nvt):
          if (self.xp_vt[j] <= 0.5 * xp.pi):
              for i in range(self.Nr):
                  bte_x_shift[i, j, : , :]       = self.I_Nx + f1 * dt * self.op_adv_x_d[i] * xp.cos(self.xp_vt[j]) * DpL
          else:
              for i in range(self.Nr):
                  bte_x_shift[i, j, : , :]       = self.I_Nx + f1 * dt * self.op_adv_x_d[i] * xp.cos(self.xp_vt[j]) * DpR
              
      bte_x_shift   = xp.linalg.inv(bte_x_shift)
      self.pc_mat_x = bte_x_shift
      self.pc_mat_v = xp.linalg.inv(self.I_Nv - dt * self.param.tau * n0 * self.param.np0 * (self.op_col_en + Tg * self.op_col_gT) - dt * self.param.tau * E * self.op_adv_v)
      
      return 
    
    def bte_step(self, v0, time, dt, Q, atol, rtol):
      xp = self.xp_module
      
      def Lmat_mvec(x):
        y                   = self.rhs_bte(x, Q, time, dt)
        y                   = y.reshape((self.Nr * self.Nvt, self.Np))
        
        y[self.xp_vt_l,  0] = 0.0 
        y[self.xp_vt_r, -1] = 0.0 
        
        y                   = y.reshape((-1))
        y                   = x - dt * y
        return y.reshape((-1))
          
      def Mmat_mvec(x):
        #@mfernando : I am not sure why but the order of the operator split scheme does matter here, 
        # for some reason, doing v-space step followed by the advection step tend to be more accurate
        x     = x.reshape((self.Nr * self.Nvt, self.Np))
        x_lm  = xp.dot(self.op_po2sh, x)
        x_lm  = xp.dot(self.pc_mat_v, x_lm)
        x     = xp.dot(self.op_psh2o, x_lm)
        
        x      = x.reshape((self.Nr, self.Nvt , self.Np)).reshape(self.Nr, self.Nvt * self.Np)
        x      = xp.dot(self.op_adv_x_qinv, x).reshape((self.Nr, self.Nvt, self.Np)).reshape((self.Nr * self.Nvt , self.Np)) # diagonalization
  
        x[self.xp_vt_l,  0] = 0.0 # BC enforce 
        x[self.xp_vt_r, -1] = 0.0 # BC enforce
  
        x     = xp.einsum("ijkl,ijl->ijk", self.pc_mat_x, x.reshape((self.Nr, self.Nvt, self.Np))).reshape((self.Nr, self.Nvt *  self.Np))
        x     = xp.dot(self.op_adv_x_q, x).reshape((self.Nr , self.Nvt, self.Np))
        x     = x.reshape((self.Nr * self.Nvt, self.Np))
        
        return x.reshape((-1))
      
      
      # Lv0 = Lmat_mvec(v0)
      # Mv0 = Mmat_mvec(v0)
      
      # print(Lv0)
      # print(Mv0)
      
      # x0     = v0.reshape((self.Nr * self.Nvt, self.Np))
      # x_lm   = xp.dot(self.op_po2sh, x0)
      # #x_lm  = xp.dot(self.pc_mat_v, x_lm)
      # x1     = xp.dot(self.op_psh2o, x_lm)
      # Perror = xp.array([xp.linalg.norm(x1[:,i]-x0[:,i])/xp.linalg.norm(x0[:,i]) for i in range(self.Np)])
      # print(Perror)
      
      norm_b     = xp.linalg.norm(v0.reshape((-1)))
      Ndof       = self.Nr * self.Nvt * self.Np
      
      Lmat_op    = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Lmat_mvec)
      Mmat_op    = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Mmat_mvec)
      
      b                   = xp.copy(v0).reshape((self.Nr * self.Nvt, self.Np))
      b[self.xp_vt_l, 0]  = 0.0
      b[self.xp_vt_r, -1] = 0.0
      b                   = b.reshape((-1))
      
      v1, status = cupyx.scipy.sparse.linalg.gmres(Lmat_op, b.reshape((-1)), x0=v0.reshape((-1)), tol=rtol/norm_b, atol=atol, M=Mmat_op, maxiter=1000)
      
      res           = Lmat_mvec(v1) -  b
      #np.save("res.npy", res)
      
                
      norm_res_abs  = xp.linalg.norm(res)
      norm_res_rel  = norm_res_abs / norm_b
      if (status !=0) :
          print("GMRES solver failed! iterations =%d  ||res|| = %.4E ||res||/||b|| = %.4E"%(status, norm_res_abs, norm_res_rel))
      
      print("time = %.2E ||Ax-b|| = %.4E ||Ax-b|| / ||b|| = %.4E" %(time, norm_res_abs, norm_res_rel))
      return v1, norm_res_abs, norm_res_abs



glow_1d = glow1d_boltzmann(args)
u, v    = glow_1d.initialize()

if args.use_gpu==1:
    gpu_device = cp.cuda.Device(args.gpu_device_id)
    gpu_device.use()
    
    glow_1d.copy_operators_H2D(args.gpu_device_id)
    glow_1d.xp_module = cp
    u  = cp.asarray(u)
    v  = cp.asarray(v).reshape((-1))
    Q  = dict()
    
    Q["E"]  = 100
    Q["n0"] = glow_1d.param.n0
    Q["Tg"] = glow_1d.param.Tg
    
    tt = 0
    dt = args.cfl
    
    glow_1d.pc_setup(Q, dt)
    for t_idx in range(10):
        v, _ , _   = glow_1d.bte_step(v, tt, dt, Q, args.atol, args.rtol)
        v          = glow_1d.filter_ordinates(v) 
        tt += dt
      
      
  
  


        
        
        
        