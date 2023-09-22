"""
macroscopic/microscopic modeling of the 1d glow discharge problem
1). We use Gauss-Chebyshev-Lobatto co-location method with implicit time integration. 
"""
import numpy as np
import scipy.constants 
import argparse
import matplotlib.pyplot as plt
import sys
import boltzmann_op
import glow1d_utils
import basis
import collisions
import utils as bte_utils
import basis
import spec_spherical as sp
import collision_operator_spherical as collOpSp
import scipy.constants


class glow1d_boltzmann():
    """
    perform glow discharge simulation with electron Boltzmann solver
    """
    def __init__(self, args) -> None:
      self.args      = args
      self.param     = glow1d_utils.parameters()
      
      self.Ns  = self.args.Ns                   # Number of species
      self.NT  = self.args.NT                   # Number of temperatures
      self.Nv  = self.args.Ns + self.args.NT    # Total number of 'state' variables
      
      self.Nr    = self.args.Nr + 1               # e-boltzmann number of radial dof  
      self.Nvt   = self.args.Nvt                  # e-boltzmann number of ordinates
      self.dof_v = self.Nr * self.Nvt
      
      self.deg = self.args.Np-1                 # degree of Chebyshev polys we use
      self.Np  = self.args.Np                   # Number of points used to define state in space
      self.Nc  = self.args.Np-2                 # number of collocation pts (Np-2 b/c BCs)
      
      self.ele_idx = 0
      self.ion_idx = 1
      self.Te_idx  = self.Ns
      
      self.dof_vx = self.Np * self.dof_v
      
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
      self.I_Np = Imat
      
      # setting up the Boltzmann grid params
      
      self.qe            = scipy.constants.e
      self.me            = scipy.constants.electron_mass
      self.kB            = scipy.constants.Boltzmann
      self.c_gamma       = np.sqrt(2 * (self.qe/ self.me))
      
      self.Nvt           = self.args.Nvt
      self.xp_vt         = basis.Legendre().Gauss_Pn(self.Nvt)[0]
      self.xp_vt         = np.arccos(self.xp_vt)
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
      ev_range               = ((0 * vth /self.c_gamma)**2, (6 * vth /self.c_gamma)**2)
      k_domain               = (np.sqrt(ev_range[0]) * self.c_gamma / vth, np.sqrt(ev_range[1]) * self.c_gamma / vth)
      use_ee                 = args.ee_collisions
      use_dg                 = 1
      
      if use_ee==1:
          use_dg=0
      else:
          use_dg=0
      
      print("boltzmann grid v-space ev=", ev_range, " v/vth=",k_domain)
      
      # construct the spectral class 
      bb                     = basis.BSpline(k_domain, self.args.sp_order, self.bs_nr + 1, sig_pts=dg_nodes, knots_vec=None, dg_splines=use_dg, verbose = args.verbose)
      spec_sp                = sp.SpectralExpansionSpherical(self.bs_nr, bb, self.bs_lm)
      spec_sp._num_q_radial  = bb._num_knot_intervals * self.args.spline_qpts
      collision_op           = collOpSp.CollisionOpSP(spec_sp)
      self.op_spec_sp        = spec_sp
      
      num_p                  = spec_sp._p + 1
      num_sh                 = len(self.bs_lm)
      num_vt                 = self.Nvt
      
      # ordinates to spherical lm computation (num_sh, self.Nvt)
      vt_pts                 = self.xp_vt
      glx, glw               = basis.Legendre().Gauss_Pn(len(vt_pts))
      vq_theta               = np.arccos(glx)
      assert (vq_theta == vt_pts).all(), "collocation points does not match with the theta quadrature points"
      tmp                    = np.matmul(spec_sp.Vq_sph(vt_pts, np.zeros_like(vt_pts)), np.diag(glw) ) * 2 * np.pi
      
      #print(tmp)
      self.op_po2sh = np.zeros((num_p * num_sh, num_p* num_vt))
      for ii in range(num_p):
        for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
          for vt_idx in range(num_vt):
            self.op_po2sh[ii * num_sh + lm_idx, ii * num_vt + vt_idx] = tmp[lm_idx, vt_idx]
      
      #print(self.op_po2sh)
      
      # spherical lm to ordinates computation (self.Nvt, num_sh)
      tmp                     = np.transpose(spec_sp.Vq_sph(vt_pts, np.zeros_like(vt_pts))) 
      self.op_psh2o           = np.zeros((num_p * num_vt, num_p * num_sh)) 
      for ii in range(num_p):
        for vt_idx in range(num_vt):
          for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
            self.op_psh2o[ii * num_vt + vt_idx, ii * num_sh + lm_idx] = tmp[vt_idx, lm_idx]
      
      
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
    
            [gx, gw] = spec_sp._basis_p.Gauss_Pn((sp_order + 4) * spec_sp._basis_p._num_knot_intervals)
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
          
      self.op_sigma_m   = sigma_m
      
      self.op_mass      = bte_utils.mass_op(spec_sp, 1) #* maxwellian(0) * vth**3
      self.op_temp      = bte_utils.temp_op(spec_sp, 1) * (vth**2) * 0.5 * scipy.constants.electron_mass * (2/3/scipy.constants.Boltzmann) / collisions.TEMP_K_1EV
      
      # note non-dimentionalized electron mobility and diffusion,  ## not used at the moment, but needs to check.          
      self.op_mobility  = bte_utils.mobility_op(spec_sp, maxwellian, vth) * self.param.V0 * self.param.tau/self.param.L**2
      self.op_diffusion = bte_utils.diffusion_op(spec_sp, self.bs_coll_list, maxwellian, vth) * self.param.tau/self.param.L**2
          
      rr_op  = [None] * len(self.bs_coll_list)
      for col_idx, g in enumerate(self.bs_coll_list):
          rr_op[col_idx] = bte_utils.reaction_rates_op(spec_sp, [g], maxwellian, vth) * self.param.tau * self.param.np0 # note : non-dimensionalized reaction rates 
          
      self.op_rate      = rr_op
          
      if use_dg == 1 : 
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
              return np.dot(self.op_psh2o, np.dot(opM,self.op_po2sh))
              # opM                 = opM.reshape((num_p, num_sh, num_p, num_sh))
              # opM                 = np.tensordot(opM, self.op_po2sh,axes=[[3],[0]])
              # opM                 = np.tensordot(opM, self.op_psh2o,axes=[[1],[1]])
              # opM                 = np.swapaxes(opM,1,2).reshape((num_p * self.Nvt, num_p * self.Nvt))            
              # return opM
          
          
          adv_mat_v                *= (1 / vth) * collisions.ELECTRON_CHARGE_MASS_RATIO
          adv_mat_v                 = psh2o_C_po2sh(adv_mat_v)
          
          FOp                       = psh2o_C_po2sh(FOp)
          FOp_g                     = psh2o_C_po2sh(FOp_g)
          
          adv_x                     = np.dot(self.op_inv_mm,  vth * compute_spatial_advection_op())
          adv_x_d, adv_x_q          = np.linalg.eig(adv_x)
          self.op_adv_x_d           = adv_x_d
          self.op_adv_x_q           = adv_x_q
          self.op_adv_x_qinv        = np.linalg.inv(adv_x_q)
          #print(self.op_adv_x_d)
          #plt.plot(self.op_adv_x_d)
          #plt.show()
          #print("x-space advection diagonalization rel error = %.8E"%(np.linalg.norm(np.dot(self.op_adv_x_d * self.op_adv_x_q, self.op_adv_x_qinv) - adv_x)/np.linalg.norm(adv_x)))
          # adv_mat_x                 = np.zeros((num_p * num_vt, num_p * num_vt))
          # for vt_idx in range(num_vt):
          #   adv_mat_x[vt_idx::num_vt, vt_idx :: num_vt] = adv_x * np.cos(self.xp_vt[vt_idx])
          
          FOp                       = np.dot(self.op_inv_mm_full, FOp)
          FOp_g                     = np.dot(self.op_inv_mm_full, FOp_g)
          adv_mat_v                 = np.dot(self.op_inv_mm_full, adv_mat_v)
          
          self.op_adv_x             = adv_x
          self.op_adv_v             = adv_mat_v
          self.op_col_en            = FOp
          self.op_col_gT            = FOp_g
          
      # if(use_ee == 1):
      #     print("e-e collision assembly begin")
          
      #     hl_op, gl_op         = collision_op.compute_rosenbluth_potentials_op(maxwellian, vth, 1, mmat_inv, mp_pool_sz=args.threads)
      #     cc_op_a, cc_op_b     = collision_op.coulomb_collision_op_assembly(maxwellian, vth, mp_pool_sz=args.threads)
          
      #     xp                   = self.xp_module
      #     hl_op                = xp.asarray(hl_op)
      #     gl_op                = xp.asarray(gl_op) 
      #     cc_op_a              = xp.asarray(cc_op_a)
      #     cc_op_b              = xp.asarray(cc_op_b)
      #     qA                   = xp.asarray(qA)
      #     mmat_inv             = xp.asarray(mmat_inv)
          
      #     cc_op                = xp.dot(cc_op_a, hl_op) + xp.dot(cc_op_b, gl_op)
      #     cc_op                = xp.dot(cc_op,qA)
      #     cc_op                = xp.dot(xp.swapaxes(cc_op,1,2),qA)
      #     cc_op                = xp.swapaxes(cc_op,1,2)
      #     cc_op                = xp.dot(xp.transpose(qA), cc_op.reshape((num_p*num_sh,-1))).reshape((num_p * num_sh, num_p * num_sh, num_p * num_sh))
      #     cc_op                = xp.dot(mmat_inv, cc_op.reshape((num_p*num_sh,-1))).reshape((num_p * num_sh, num_p * num_sh, num_p * num_sh))
      #     cc_op                = xp.asnumpy(cc_op)
      #     self.op_col_ee[idx] = cc_op
          
      #     print("e-e collision assembly end")
      
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
          Uin = xp.load("%s_u.npy"%(args.fname))
          Vin = xp.load("%s_v.npy"%(args.fname))
        else:
          xx = self.param.L * (self.xp + 1)
          Uin[:, ele_idx] = 1e6 * (1e7 + 1e9 * (1-0.5 * xx/self.param.L)**2 * (0.5 * xx/self.param.L)**2) / self.param.np0
          Uin[:, ion_idx] = 1e6 * (1e7 + 1e9 * (1-0.5 * xx/self.param.L)**2 * (0.5 * xx/self.param.L)**2) / self.param.np0
          Uin[:, Te_idx]  = self.param.Teb
        
        self.mu[:, ele_idx] = self.param.mu_e
        self.D[: , ele_idx] = self.param.De
        
        self.mu[:, ion_idx] = self.param.mu_i
        self.D[: , ion_idx] = self.param.Di
        
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
        print("boltzmann initial conditions, mass and temperature (eV)", m0, xp.dot(temp_op, h_init)/m0)
        
        h_init  = xp.dot(self.op_psh2o, h_init)
        
        for i in range(self.Np):
          Vin[:,i] = h_init
          
        # scale functions to have ne, at initial timestep
        Vin = Vin * Uin[:,ele_idx]
        #print(Vin.shape)
          
      else:
        raise NotImplementedError
      
      return Uin, Vin
    
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
        r[0]  = 0.0
        r[-1] = xp.sin(2 * xp.pi * time) #+ self.params.verticalShift
        return xp.dot(self.LpD_inv, r)
    
    def push(self, Uin, Vin, time, dt):
      """
      boltzmann to fluid push
      """
      xp      = self.xp_module
      
      ele_idx = self.ele_idx
      ion_idx = self.ion_idx
      Te_idx  = self.Te_idx
      
      num_p   = self.op_spec_sp._p + 1
      num_sh  = len(self.op_spec_sp._sph_harm_lm)
      
      Vin_lm                   = xp.dot(self.op_po2sh, Vin)
      Uin[:, ele_idx]          = xp.dot(self.op_mass[0::num_sh], Vin_lm[0::num_sh,:])
      Uin[:, Te_idx]           = xp.dot(self.op_temp[0::num_sh], Vin_lm[0::num_sh,:])/Uin[:,ele_idx]
      self.r_rates[:, ion_idx] = xp.abs(xp.dot(self.op_rate[1], Vin_lm[0::num_sh,:]))
      
      return
    
    def pull(self, Uin, Vin, time, dt):
      xp      = self.xp_module
      ele_idx = self.ele_idx
      ion_idx = self.ion_idx
      
      phi       = self.solve_poisson(Uin[:,ele_idx], Uin[:, ion_idx], time)
      E         = -xp.dot(self.Dp, phi)
      self.bs_E = E * self.param.V0 / self.param.L
      return
    
    def rhs_fluid(self, Uin : np.array, time, dt):
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
      
      ni      = Uin[: , ion_idx]
      ne      = Uin[: , ele_idx]
      
      mu_i    = self.mu[:, ion_idx]
      mu_e    = self.mu[:, ele_idx]
      
      De      = self.D[:, ele_idx]
      Di      = self.D[:, ion_idx]
      
      phi     = self.solve_poisson(Uin[:,ele_idx], Uin[:, ion_idx], time)
      E       = -xp.dot(self.Dp, phi)
      ki      = self.r_rates[:, ion_idx]
      
      nf_var  = len(self.fluid_idx)   #number of fluid variables
      
      Us_x    = xp.dot(self.Dp, Uin[: , self.fluid_idx])
      fluxJ   = xp.empty((self.Np, nf_var))
      FUin    = xp.empty((self.Np, nf_var))
      
      for idx, sp_idx  in enumerate(self.fluid_idx):
        fluxJ[:, idx] = self.Zp[sp_idx] * self.mu[: , sp_idx] * Uin[: , sp_idx] * E - self.D[: , sp_idx] * Us_x[:, idx]
      
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
      
    def rhs_fluid_jacobian(self, Uin : np.array, time, dt):
      xp  = self.xp_module
      dof = self.Nv * self.Np
      jac = xp.zeros((dof, dof))
      
      ele_idx = self.ele_idx
      ion_idx = self.ion_idx
      Te_idx  = self.Te_idx
      
      ne      = Uin[: , ele_idx]
      ni      = Uin[: , ion_idx]
      Te      = Uin[: , Te_idx]
      
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
      
      Imat    = self.I_Np 
      
      phi     = self.solve_poisson(ne, ni, time)
      E       = -xp.dot(self.Dp, phi)
      
      nf_vars = len(self.fluid_idx)
      Js_nk    = xp.zeros((nf_vars, self.Np, self.Np))
      
      for idx, i in enumerate(self.fluid_idx):
        if i == ion_idx:
          Js_nk[idx] = self.Zp[i] * self.mu[:,i] * (E * Imat + Uin[:,i] * E_ni) - self.D[:,i] * self.Dp
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
    
    def step_bte_x(self, Vin, time, dt):
      "perform the bte x-advection analytically"
      xp        = self.xp_module
      
      Vin       = Vin.reshape((self.Nr, self.Nvt , self.Np)).reshape(self.Nr, self.Nvt * self.Np)
      Vin       = xp.dot(self.op_adv_x_qinv, Vin).reshape((self.Nr, self.Nvt, self.Np))
      
      Vin_adv_x = xp.empty_like(Vin)
      for i in range(self.Nr):
        for j in range(self.Nvt):
          Vin_adv_x[i,j] = xp.dot(self.bte_x_shift[i,j], Vin[i,j])
          
      Vin_adv_x  = Vin_adv_x.reshape((self.Nr, self.Nvt *  self.Np))
      Vin_adv_x  = xp.dot(self.op_adv_x_q, Vin_adv_x).reshape((self.Nr , self.Nvt, self.Np)).reshape((self.Nr * self.Nvt, self.Np))
      
      return Vin_adv_x
      
    def rhs_bte_v(self, Vin: np.array, time, dt):
      """
      compute the rhs for the 1d2v boltzmann equation.
      Uin : (Np, Ns)
      Vin : (num_p * num_vt, Np) # boltzmann rhs vector
      """
      xp        = self.xp_module
      E         = self.bs_E
      FVin      = self.param.tau * self.param.n0 * self.param.np0 * (xp.dot(self.op_col_en, Vin) + self.param.Tg * xp.dot(self.op_col_gT, Vin))  + E * self.param.tau * xp.dot(self.op_adv_v, Vin)
      strong_bc = None
      
      return FVin, strong_bc
      
    def rhs_bte_v_jacobian(self, Vin: np.array, time, dt):
      
      xp      = self.xp_module
      ele_idx = self.ele_idx
      ion_idx = self.ion_idx
      
      E       = self.bs_E
      jac     = xp.zeros((self.Np, self.dof_v, self.dof_v))
      
      for i in range(self.Np):
        jac[i,:] = self.param.tau * self.param.n0 * self.param.np0 * (self.op_col_en + self.param.Tg * self.op_col_gT) + E[i] * self.param.tau * self.op_adv_v
      
      jac_bc = None
      return jac, jac_bc

    def step_fluid(self, u, du, time, dt, verbose=0):
      
      xp      = self.xp_module
      ts_type = self.args.ts_type
      
      rhs     = self.rhs_fluid 
      
      if ts_type == "RK2":
        k1 = dt * rhs(Uin, time, dt)
        k2 = dt * rhs(Uin + 0.5 * k1, time + 0.5 * dt, dt)
        return Uin + k2
      elif ts_type == "RK4":
        k1 = dt * rhs(Uin           , time           , dt)
        k2 = dt * rhs(Uin + 0.5 * k1, time + 0.5 * dt, dt)
        k3 = dt * rhs(Uin + 0.5 * k2, time + 0.5 * dt, dt)
        k4 = dt * rhs(Uin +  k3     , time + dt      , dt)
        return Uin + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
      elif ts_type == "BE":
        rtol            = self.args.rtol
        atol            = self.args.atol
        iter_max        = self.args.max_iter
        
        Imat            = xp.eye(self.Np)
        
        assert self.fluid_idx[0] == self.ion_idx
    
        def residual(du):
          u1       = u + du
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
        
        ns_info = glow1d_utils.newton_solver(du, residual, jacobian, atol, rtol, iter_max, xp)
        du = ns_info["x"]
        
        if(verbose==1):
          print("Fluid step time = %.2E "%(time))
          print("  Newton iter {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(ns_info["iter"], ns_info["atol"], ns_info["rtol"]))
        
        if (ns_info["status"]==False):
          print("Fluid step non-linear solver step FAILED!!! try with smaller time step size or increase max iterations")
          print("time = %.2E "%(time))
          print("  Newton iter {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(ns_info["iter"], ns_info["atol"], ns_info["rtol"]))
          return u
        
        return u + du
      
    def step_boltzmann(self, u, du, time, dt, verbose=0):
      xp      = self.xp_module
      ts_type = self.args.ts_type
      
      time    = time
      dt      = dt
      
      rhs     = self.rhs_bte_v
      
      if ts_type == "RK2":
        k1 = dt * rhs(u, time, dt)
        k2 = dt * rhs(u + 0.5 * k1, time + 0.5 * dt, dt)
        return u + k2
      elif ts_type == "RK4":
        k1 = dt * rhs(u           , time           , dt)
        k2 = dt * rhs(u + 0.5 * k1, time + 0.5 * dt, dt)
        k3 = dt * rhs(u + 0.5 * k2, time + 0.5 * dt, dt)
        k4 = dt * rhs(u +  k3     , time + dt      , dt)
        return u + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
      elif ts_type == "BE":
        rtol            = self.args.rtol
        atol            = self.args.atol
        iter_max        = self.args.max_iter
        
        Imat1           = xp.eye(self.Nr * self.Nvt)
        Imat            = xp.empty((self.Np, self.Nr * self.Nvt, self.Nr * self.Nvt))
        for i in range(self.Np):
          Imat[i,:,:]   = Imat1[:,:]
    
        def residual(du):
          rhs, bc = self.rhs_bte_v(u + du, time + dt, dt)
          res     = du - dt * rhs
          
          return res
        
        def jacobian(du):
          rhs_j , bc_j = self.rhs_bte_v_jacobian(u, time, dt)
          jac          = Imat - dt * rhs_j
          return jac
        
        ns_info = glow1d_utils.newton_solver_batched(du, self.Np, residual, jacobian, atol, rtol, iter_max, xp)
        du = ns_info["x"]
        
        if(verbose==1):
          print("Boltzmann step time = %.2E "%(time))
          print("  Newton iter {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(ns_info["iter"], ns_info["atol"], ns_info["rtol"]))
        
        if (ns_info["status"]==False):
          print("Boltzmann step non-linear solver step FAILED!!! try with smaller time step size or increase max iterations")
          print("time = %.2E "%(time))
          print("  Newton iter {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(ns_info["iter"], ns_info["atol"], ns_info["rtol"]))
          return u
        
        return u + du
      
    def solve(self, Uin, Vin):
      xp              = self.xp_module
      dt              = self.args.cfl 
      tT              = self.args.cycles
      tt              = 0
      steps           = max(1,int(tT/dt))
      
      print("++++ Using backward Euler ++++")
      print("T = %.4E RF cycles = %.1E dt = %.4E steps = %d atol = %.2E rtol = %.2E max_iter=%d"%(tT, self.args.cycles, dt, steps, self.args.atol, self.args.rtol, self.args.max_iter))
      
      u = xp.copy(Uin)
      v = xp.copy(Vin)
      
      du = xp.zeros_like(u)
      dv = xp.zeros_like(v)
      
      io_freq = 1
      
      # setting the Chebyshev operators for x-space advection for the analytical solution.
      self.bte_x_shift = xp.zeros((self.Nr, self.Nvt, self.Np, self.Np))
      for i in range(self.Nr):
        for j in range(self.Nvt):
          xx = self.xp - self.op_adv_x_d[i] * np.cos(self.xp_vt[j]) * dt * self.param.tau
          self.bte_x_shift[i,j :, : ]     = np.polynomial.chebyshev.chebvander(xx, self.deg)
          # bdy condition 
          self.bte_x_shift[i,j, xx<-1, :] = 0.0 
          self.bte_x_shift[i,j, xx> 1, :] = 0.0 
          
          self.bte_x_shift[i,j :, : ] = xp.dot(self.bte_x_shift[i,j], self.V0pinv)
          
          
      # plt.plot(self.xp, xp.dot(self.op_mass,xp.dot(self.op_po2sh, v)),label="0")
      # v1 = self.step_bte_x(v,  0, 0)
      # v1 = self.step_bte_x(v1, 0, 0)
      # plt.plot(self.xp, xp.dot(self.op_mass, xp.dot(self.op_po2sh, v1)),label="1")
      # plt.grid()
      # plt.legend()
      # plt.show()
      
      for ts_idx in range(steps):
        if (ts_idx % io_freq == 0):
          print("time = %.2E step=%d/%d"%(tt, ts_idx, steps))
          
        self.push(u, v, tt, dt)
        u = self.step_fluid(u, du, tt, dt, int(ts_idx % io_freq == 0))
        self.pull(u, v, tt + dt, dt)
        
        v          = self.step_bte_x(v, tt, dt)
        v          = self.step_boltzmann(v, dv, tt, dt, int(ts_idx % io_freq == 0))
        tt+= dt
      
      return u, v
      
    def plot(self, Uin):
      fig= plt.figure(figsize=(18,8), dpi=300)
      
      ne = np.abs(Uin[:, self.ele_idx])
      ni = np.abs(Uin[:, self.ion_idx])
      Te = np.abs(Uin[:, self.Te_idx])
      
      plt.subplot(2, 3, 1)
      plt.plot(self.xp, self.param.np0 * ne, 'b')
      plt.xlabel(r"x/L")
      plt.ylabel(r"$n_e (m^{-3})$")
      plt.grid(visible=True)
      
      plt.subplot(2, 3, 2)
      plt.plot(self.xp, self.param.np0 * ni, 'b')
      plt.xlabel(r"x/L")
      plt.ylabel(r"$n_i (m^{-3})$")
      plt.grid(visible=True)
      
      plt.subplot(2, 3, 3)
      plt.plot(self.xp, Te, 'b')
      plt.xlabel(r"x/L")
      plt.ylabel(r"$T_e (eV)$")
      plt.grid(visible=True)
      
      
      plt.subplot(2, 3, 4)
      phi = self.solve_poisson(Uin[:,0], Uin[:,1], 0) 
      E = -np.dot(self.Dp, phi)
      plt.plot(self.xp, E * self.param.V0/self.param.L, 'b')
      plt.xlabel(r"x/L")
      plt.ylabel(r"$E (V/m)$")
      plt.grid(visible=True)
      
      plt.subplot(2, 3, 5)
      plt.plot(self.xp, phi * self.param.V0, 'b')
      plt.xlabel(r"x/L")
      plt.ylabel(r"$\phi (V)$")
      plt.grid(visible=True)
      
      plt.tight_layout()
      
      fig.savefig("1d_glow.png")
    
    
parser = argparse.ArgumentParser()
parser.add_argument("-threads", "--threads"                       , help="number of cpu threads (boltzmann operator assembly)", type=int, default=4)
parser.add_argument("-out_fname", "--out_fname"                   , help="output file name for the qois", type=str, default="bte_glow1d")
parser.add_argument("-l_max", "--l_max"                           , help="max polar modes in SH expansion", type=int, default=1)
parser.add_argument("-c", "--collisions"                          , help="collisions model",nargs='+', type=str, default=["g0","g2"])
parser.add_argument("-sp_order", "--sp_order"                     , help="b-spline order", type=int, default=3)
parser.add_argument("-spline_qpts", "--spline_qpts"               , help="q points per knots", type=int, default=5)
parser.add_argument("-steady", "--steady_state"                   , help="steady state or transient", type=int, default=1)
#parser.add_argument("-max_iter", "--max_iter"                     , help="max number of iterations for newton solve", type=int, default=300)
parser.add_argument("-Te", "--Te"                                 , help="approximate electron temperature (eV)"  , type=float, default=2.0)
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
parser.add_argument("-store_csv", "--store_csv"                   , help="store csv format of QoI comparisons", type=int, default=1)
parser.add_argument("-plot_data", "--plot_data"                   , help="plot data", type=int, default=1)
parser.add_argument("-ee_collisions", "--ee_collisions"           , help="enable electron-electron collisions", type=float, default=0)
parser.add_argument("-verbose", "--verbose"                       , help="verbose with debug information", type=int, default=0)
parser.add_argument("-restore", "--restore"                       , help="restore the solver" , type=int, default=0)
parser.add_argument("-fname", "--fname"                 , help="file name to store the solution" , type=str, default="1d_glow")

args = parser.parse_args()
glow_1d = glow1d_boltzmann(args)
u, v    = glow_1d.initialize()
uu,vv   = glow_1d.solve(u, v)

glow_1d.push(uu, vv, 0, 0)
#v       = glow_1d.solve(u, ts_type=args.ts_mode)
glow_1d.plot(uu)

np.save("%s_u.npy"%(args.fname), uu)
np.save("%s_v.npy"%(args.fname), vv)



