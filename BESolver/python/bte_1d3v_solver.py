"""
standalone 1D3V BTE solver
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import math
import scipy.interpolate
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
import spec_spherical
import numpy as np
import mesh
import toml
import argparse
import h5py
import scipy.optimize

PROFILE_SOLVERS=1

try:
    import cupy as cp
    import cupyx.scipy.sparse.linalg
except:
    print("Cupy module not found !")
    #raise ModuleNotFoundError

class adv_xtype():
    CHEB   = 0 
    UPW_FD = 1

class params():
    def __init__(self, par_file):
        xp         = np
        try:
            tp_all = toml.load(par_file)
        except:
            print("Error while loading the parameter file")
        
        tp         = tp_all["bte"]
        self.L     = 0.5 * tp["L"]             # m 
        self.V0    = tp["V0"]                  # V
        self.f     = tp["freq"]                # Hz
        self.tau   = (1/self.f)                # s
        self.qe    = scipy.constants.e         # C
        self.eps0  = scipy.constants.epsilon_0 # eps_0 
        self.kB    = scipy.constants.Boltzmann # J/K
        self.ev_to_K = scipy.constants.electron_volt / scipy.constants.Boltzmann
        self.me    = scipy.constants.electron_mass
        self.Tg    = tp["Tg"]  #K
        self.p0    = tp["p0"]  #Torr

        if self.Tg !=0: 
            self.n0    = self.p0 * scipy.constants.torr / (scipy.constants.Boltzmann * self.Tg) #3.22e22                   #m^{-3}
        else:
            self.n0    = 3.22e22                   #m^{-3}
        
        self.np0   = 8e16                          #"nominal" electron density [1/m^3]
        self.n0    = self.n0/self.np0


        self.sp_order      = tp["sp_order"]
        self.collisions    = tp["collisions"]
        self.sp_order      = tp["sp_order"]
        self.spline_qpts   = tp["spline_qpts"]
        self.Nr            = tp["Nr"]
        self.l_max         = tp["lmax"]
        self.Np            = tp["Np"]
        self.Nvt           = tp["Nvt"]

        self.io_freq       = tp["io_cycle"]
        self.cp_freq       = tp["cp_cycle"]
        self.dt            = tp["dt"]
        self.T             = tp["cycles"]
        self.Te            = tp["Te"]
        self.ev_max        = tp["ev_max"]
        self.ev_extend     = tp["ev_extend"]
        self.dir           = tp["dir"]
        self.fname         = tp["fname"]
        self.use_gpu       = tp["use_gpu"]
        self.dev_id        = tp["gpu_device_id"]

        self.atol          = tp["atol"]
        self.rtol          = tp["rtol"]
        self.max_iter      = tp["max_iter"]
        self.gmres_rsrt    = tp["gmres_rsrt"]

        self.pcEmin        = tp["abs_efield_min"]
        self.pcEmax        = tp["abs_efield_max"]
        self.vts_type      = tp["vspace_ts_type"]

        self.restore       = tp["restore"]
        self.rs_idx        = tp["rs_idx"] 
        self.xgrid_type    = "chebyshev-collocation"
        self.dim           = 1
        self.ee_collisions = 0
        self.verbose       = tp["verbose"]
        self.threads       = tp["threads"]

        dir                = self.dir
        if os.path.exists(dir):
            print("run directory exists, data will be overwritten")
            #sys.exit(0)
        else:
            os.makedirs(dir)
            print("directory %s created"%(dir))
      
        self.fname=str(dir)+"/"+self.fname

        self.vtDe      = tp["vtDe"] #0.25
        self.xadv_type = tp["xadv_type"]
        
        

class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))

class bte_1d3v():

    def __init__(self, args):
        self.params        = params(args.par_file)
        self.qe            = scipy.constants.e
        self.me            = scipy.constants.electron_mass
        self.kB            = scipy.constants.Boltzmann
        self.c_gamma       = np.sqrt(2 * (self.qe/ self.me))
        
        Nvt                = self.params.Nvt
        if (self.params.xgrid_type == "chebyshev-collocation"):
            self.mesh          = mesh.mesh([self.params.Np], self.params.dim, mesh.grid_type.CHEBYSHEV_COLLOC)
        elif(self.params.xgrid_type == "uniform"):
            self.mesh          = mesh.mesh([self.params.Np], self.params.dim, mesh.grid_type.REGULAR_GRID)
        else:
            raise NotImplementedError
        
        self.xp            = self.mesh.xcoord[0]
        self.Dp            = self.mesh.D1[0]
        self.DpT           = self.mesh.D1[0].T

        dof_v              = (self.params.Nr + 1) * (self.params.l_max+1)
        Imat               = np.eye(self.params.Np)
        self.I_Nx          = Imat

        self.I_Nv          = np.eye(dof_v)
        self.I_Nxv_stacked = np.eye(dof_v)
        
        assert Nvt%2 == 0 
        gx, gw             = basis.Legendre().Gauss_Pn(Nvt//2)
        gx_m1_0 , gw_m1_0  = 0.5 * gx - 0.5, 0.5 * gw
        gx_0_p1 , gw_0_p1  = 0.5 * gx + 0.5, 0.5 * gw
        self.xp_vt         = np.append(np.arccos(gx_m1_0), np.arccos(gx_0_p1)) 
        self.xp_vt_qw      = np.append(gw_m1_0, gw_0_p1)
        
        # self.xp_vt, self.xp_vt_qw = basis.Legendre().Gauss_Pn(Nvt)
        # self.xp_vt                = np.arccos(self.xp_vt) 
        
        self.xp_cos_vt     = np.cos(self.xp_vt)
        self.xp_vt_l       = np.array([i * Nvt + j for i in range(self.params.Nr + 1) for j in list(np.where(self.xp_vt <= 0.5 * np.pi)[0])])
        self.xp_vt_r       = np.array([i * Nvt + j for i in range(self.params.Nr + 1) for j in list(np.where(self.xp_vt > 0.5 * np.pi)[0])])

        LpDvt              = mesh.central_dxx(self.xp_vt)
        I_Nvt              = np.eye(len(self.xp_vt))
        LpDvt              = I_Nvt - self.params.vtDe * LpDvt
        
        #dirichlet BCs
        LpDvt[0]           = I_Nvt[0]
        LpDvt[-1]          = I_Nvt[-1]
        
        # # neumann BCs
        # LpDvt[0]           = mesh.upwinded_dx(self.xp_vt, "LtoR")[0]  
        # LpDvt[-1]          = mesh.upwinded_dx(self.xp_vt, "LtoR")[-1] 
        self.LpDvt         = np.linalg.inv(LpDvt)

        
        self.bs_Te         = self.params.Te 
        self.bs_nr         = self.params.Nr
        self.bs_lm         = [[l,0] for l in range(self.params.l_max+1)]
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
        self.ev_lim         = (0,self.params.ev_max)
        
        self.collision_names              = list()
        self.coll_list                    = list()

        self.avail_species                = cross_section.read_available_species(self.params.collisions)
        cross_section.CROSS_SECTION_DATA  = cross_section.read_cross_section_data(self.params.collisions)
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
        use_ee                 = self.params.ee_collisions
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
        if (self.params.ev_extend==0):
            bb                     = basis.BSpline(k_domain, self.params.sp_order, self.bs_nr + 1, sig_pts=dg_nodes, knots_vec=None, dg_splines=self.bs_use_dg, verbose = self.params.verbose)
        elif (self.params.ev_extend==1):
            print("using uniform grid extention")
            bb                     = basis.BSpline(k_domain, self.params.sp_order, self.bs_nr + 1, sig_pts=dg_nodes, knots_vec=None, dg_splines=self.bs_use_dg, verbose = self.params.verbose, extend_domain=True)
        else:
            assert self.params.ev_extend==2
            print("using log spaced grid extention")
            bb                     = basis.BSpline(k_domain, self.params.sp_order, self.bs_nr + 1, sig_pts=dg_nodes, knots_vec=None, dg_splines=self.bs_use_dg, verbose = self.params.verbose, extend_domain_with_log=True)
        
        spec_sp                = sp.SpectralExpansionSpherical(self.bs_nr, bb, self.bs_lm)
        spec_sp._num_q_radial  = bb._num_knot_intervals * self.params.spline_qpts
        collision_op           = collOpSp.CollisionOpSP(spec_sp)
        self.op_spec_sp        = spec_sp
      
        num_p                  = spec_sp._p + 1
        num_sh                 = len(self.bs_lm)
        num_vt                 = self.params.Nvt

        assert num_p == self.params.Nr + 1
      
        # ordinates to spherical lm computation (num_sh, Nvt)
        # vt_pts                 = self.xp_vt
        # glx, glw               = basis.Legendre().Gauss_Pn(len(vt_pts))
        # vq_theta               = np.arccos(glx)
        # assert (vq_theta == vt_pts).all(), "collocation points does not match with the theta quadrature points"
        self.op_po2sh             = np.matmul(spec_sp.Vq_sph(self.xp_vt, np.zeros_like(self.xp_vt)), np.diag(self.xp_vt_qw) ) * 2 * np.pi
        #print(self.op_po2sh)
        
        # spherical lm to ordinates computation (Nvt, num_sh)
        self.op_psh2o           = np.transpose(spec_sp.Vq_sph(self.xp_vt, np.zeros_like(self.xp_vt))) 
        
        # to check if the oridinates to spherical and spherical to ordinates projection is true (weak test not sufficient but necessary condition)
        assert np.allclose(np.dot(self.op_psh2o, np.dot(self.op_po2sh, 1 + np.cos(self.xp_vt))), 1 + np.cos(self.xp_vt))
      
        self.par_ev_range       = ev_range  
        mm_mat                  = spec_sp.compute_mass_matrix()
        inv_mm_mat              = spec_sp.inverse_mass_mat(Mmat = mm_mat)
        mm_mat                  = mm_mat[0::num_sh, 0::num_sh]
        inv_mm_mat              = inv_mm_mat[0::num_sh, 0::num_sh]
        
        I_Nvt                   = np.eye(Nvt)
        # at the moment no need to store the mass-mat itself, what we need is the inverse of mass matrix. 
        #self.op_mm_full        = np.zeros((num_p * num_vt, num_p * num_vt))
        self.op_inv_mm_full     = np.zeros((num_p * num_vt, num_p * num_vt))
        self.op_inv_mm          = inv_mm_mat
        
        for vt_idx in range(num_vt):
            #self.op_mm_full[vt_idx :: num_vt, vt_idx::num_vt] = mm_mat
            self.op_inv_mm_full[vt_idx :: num_vt, vt_idx::num_vt] = inv_mm_mat
            # for i in range(self.params.Nr+1):
            #   for j in range(self.params.Nr+1):
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
            
            if self.params.verbose==1:
                print("collision %d  %s %s"%(col_idx, col, col_data["type"]))
                
            if col_data["type"] == "ELASTIC":
                FOp_g     = collision_op.electron_gas_temperature(g, maxwellian, vth, mp_pool_sz=self.params.threads)
            
            FOp         = FOp + collision_op.assemble_mat(g, maxwellian, vth, tgK=0.0, mp_pool_sz=self.params.threads)
            sigma_m    += g.total_cross_section(gx_ev)
            
        t2 = perf_counter()
        print("assembly = %.4E"%(t2-t1))
      
        t1 = perf_counter()
        print("bte qoi op assembly")    
        self.op_sigma_m   = sigma_m
        
        self.op_mass      = bte_utils.mass_op(spec_sp, 1) #* maxwellian(0) * vth**3
        self.op_temp      = bte_utils.temp_op(spec_sp, 1) * (vth**2) * 0.5 * scipy.constants.electron_mass * (2/3/scipy.constants.Boltzmann) / collisions.TEMP_K_1EV
        
        # note non-dimentionalized electron mobility and diffusion,  ## not used at the moment, but needs to check.          
        self.op_mobility  = bte_utils.mobility_op(spec_sp, maxwellian, vth) #* self.params.V0 * self.params.tau/self.params.L**2
        self.op_diffusion = bte_utils.diffusion_op(spec_sp, self.bs_coll_list, maxwellian, vth) #* self.params.tau/self.params.L**2
            
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
        
        adv_x                     = np.dot(self.op_inv_mm,  vth  * (self.params.tau/self.params.L) * compute_spatial_advection_op())
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
        xp.save("%s_bte_mass_op.npy"   %(self.params.fname), self.op_mass)
        xp.save("%s_bte_temp_op.npy"   %(self.params.fname), self.op_temp)
        xp.save("%s_bte_po2sh.npy"     %(self.params.fname), self.op_po2sh)
        xp.save("%s_bte_psh2o.npy"     %(self.params.fname), self.op_psh2o)
        xp.save("%s_bte_mobility.npy"  %(self.params.fname), self.op_mobility)
        xp.save("%s_bte_diffusion.npy" %(self.params.fname), self.op_diffusion)
        xp.save("%s_bte_op_g0.npy"     %(self.params.fname), self.op_rate[0])
        
        if (len(self.op_rate) > 1):
            xp.save("%s_bte_op_g2.npy"  %(self.params.fname), self.op_rate[1])
            
        save_bte_mat = True
        if save_bte_mat == True:
            xp.save("%s_bte_cmat.npy"   %(self.params.fname), self.op_col_en + self.params.Tg * self.op_col_gT)
            xp.save("%s_bte_emat.npy"   %(self.params.fname), self.op_adv_v)
            xp.save("%s_bte_xmat.npy"   %(self.params.fname), self.op_adv_x)
            
        if(use_ee == 1):
            print("e-e collision assembly begin")
            
            hl_op, gl_op         = collision_op.compute_rosenbluth_potentials_op(maxwellian, vth, 1, mmat_inv, mp_pool_sz=self.params.threads)
            cc_op_a, cc_op_b     = collision_op.coulomb_collision_op_assembly(maxwellian, vth, mp_pool_sz=self.params.threads)
            
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

    def copy_operators_H2D(self, dev_id):
        
        if self.params.use_gpu == 0:
            return
        
        with cp.cuda.Device(dev_id):
            self.xp_cos_vt      = cp.asarray(self.xp_cos_vt)
            self.I_Nx           = cp.asarray(self.I_Nx)
            self.I_Nv           = cp.asarray(self.I_Nv)
            
            self.Dp             = cp.asarray(self.Dp)
            self.DpT            = cp.asarray(self.DpT)
            # self.LpD            = cp.asarray(self.LpD)
            # self.Lp             = cp.asarray(self.Lp)
            # self.LpD_inv        = cp.asarray(self.LpD_inv)
            
            self.op_adv_v       = cp.asarray(self.op_adv_v)
            self.op_adv_x       = cp.asarray(self.op_adv_x)
            self.op_adv_x_d     = cp.asarray(self.op_adv_x_d)
            self.op_adv_x_q     = cp.asarray(self.op_adv_x_q)
            self.op_adv_x_qinv  = cp.asarray(self.op_adv_x_qinv)
            
            self.op_psh2o       = cp.asarray(self.op_psh2o)
            self.op_po2sh       = cp.asarray(self.op_po2sh)
            self.op_col_en      = cp.asarray(self.op_col_en)
            self.op_col_gT      = cp.asarray(self.op_col_gT)
            
            if self.params.ee_collisions==1:
                self.op_col_ee    = cp.asarray(self.op_col_ee)
            
            self.op_mass        = cp.asarray(self.op_mass)
            self.op_temp        = cp.asarray(self.op_temp)
            self.op_rate        = [cp.asarray(self.op_rate[i]) for i in range(len(self.op_rate))]
            self.op_mobility    = cp.asarray(self.op_mobility)
            self.op_diffusion   = cp.asarray(self.op_diffusion)
            self.LpDvt          = cp.asarray(self.LpDvt)

        return
    
    def copy_operators_D2H(self, dev_id):
        if self.params.use_gpu==0:
            return
      
        with cp.cuda.Device(dev_id):
            self.xp_cos_vt      = cp.asnumpy(self.xp_cos_vt)
            self.I_Nx           = cp.asnumpy(self.I_Nx)
            self.I_Nv           = cp.asnumpy(self.I_Nv)
            
            self.Dp             = cp.asnumpy(self.Dp)
            self.DpT            = cp.asnumpy(self.DpT)
            #self.LpD_inv        = cp.asnumpy(self.LpD_inv)
            
            self.op_adv_v       = cp.asnumpy(self.op_adv_v)
            self.op_adv_x       = cp.asnumpy(self.op_adv_x)
            self.op_adv_x_d     = cp.asnumpy(self.op_adv_x_d)
            self.op_adv_x_q     = cp.asnumpy(self.op_adv_x_q)
            self.op_adv_x_qinv  = cp.asnumpy(self.op_adv_x_qinv)
            
            self.op_psh2o       = cp.asnumpy(self.op_psh2o)
            self.op_po2sh       = cp.asnumpy(self.op_po2sh)
            self.op_col_en      = cp.asnumpy(self.op_col_en)
            self.op_col_gT      = cp.asnumpy(self.op_col_gT)
            
            if self.params.ee_collisions==1:
                self.op_col_ee   = cp.asnumpy(self.op_col_ee)
            
            self.op_mass        = cp.asnumpy(self.op_mass)
            self.op_temp        = cp.asnumpy(self.op_temp)
            
            self.op_rate        = [cp.asnumpy(self.op_rate[i]) for i in range(len(self.op_rate))]
            self.op_mobility    = cp.asnumpy(self.op_mobility)
            self.op_diffusion   = cp.asnumpy(self.op_diffusion)
            self.LpDvt          = cp.asnumpy(self.LpDvt)
            
        return  
    
    def __initialize_bte_adv_x__(self, dt):
        
        """initialize spatial advection operator"""
        xp                = self.xp_module
        self.adv_setup_dt = dt 
        #assert xp == np
        
        self.bte_x_shift      = xp.zeros((self.params.Nr+1, self.params.Nvt, self.params.Np, self.params.Np))
        #self.bte_x_shift_rmat = xp.zeros((self.params.Nr+1, self.params.Nvt, self.params.Np, self.params.Np))

        if (self.params.xadv_type == adv_xtype.CHEB):
            DpL        = xp.zeros((self.params.Np, self.params.Np))
            DpR        = xp.zeros((self.params.Np, self.params.Np))

            DpL[1:,:]  = self.Dp[1:,:]
            DpR[:-1,:] = self.Dp[:-1,:]

            print("using chebyshev pseudo-spectral")

        elif (self.params.xadv_type == adv_xtype.UPW_FD):
            DpL         = xp.zeros((self.params.Np, self.params.Np))
            DpR         = xp.zeros((self.params.Np, self.params.Np))

            DpL[1:,:]   = xp.array(mesh.upwinded_dx(self.xp, "LtoR"))[1: ,:]
            DpR[:-1, :] = xp.array(mesh.upwinded_dx(self.xp, "RtoL"))[:-1,:]

            print("using upwinded finite differences")

        else:
            raise NotImplementedError

        f1 = 1.0
        f2 = 1-f1

        for j in range(self.params.Nvt):
            if (self.xp_vt[j] <= 0.5 * xp.pi):
                for i in range(self.params.Nr+1):
                    self.bte_x_shift[i, j, : , :]       = self.I_Nx + f1 * dt * self.op_adv_x_d[i] * xp.cos(self.xp_vt[j]) * DpL
                    #self.bte_x_shift_rmat[i,j,:,:]      = -f2 * dt * self.op_adv_x_d[i] * xp.cos(self.xp_vt[j]) * DpL
            else:
                for i in range(self.params.Nr+1):
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
    
    def vspace_pc_setup(self, E):
        xp          = self.xp_module
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
        
        idx_set  = xp.array([],dtype=xp.int32)
        for idx_id, idx in enumerate(pc_emat_idx):
            idx_set = xp.append(idx_set, idx[1])
        
        assert (idx_set.shape[0]==self.params.Np), "!!! Error: preconditioner partitioning does not match the domain size"
        return pc_emat_idx
    
    def step_bte_x(self, v, time, dt, verbose=0):
        "perform the bte x-advection analytically"
        xp        = self.xp_module
        assert self.adv_setup_dt == dt
        
        if PROFILE_SOLVERS==1:
            if xp == cp:
                cp.cuda.runtime.deviceSynchronize()
            t1 = perf_counter()
        
        Nr        = self.params.Nr + 1
        Nvt       = self.params.Nvt
        Nx        = self.params.Np

        Vin       = v.reshape((Nr, self.params.Nvt , self.params.Np)).reshape(Nr, self.params.Nvt * self.params.Np)
        Vin       = xp.dot(self.op_adv_x_qinv, Vin).reshape((Nr, self.params.Nvt, self.params.Np)).reshape((Nr * self.params.Nvt , self.params.Np))
        
        #Vin       += xp.einsum("ijkl,ijl->ijk",self.bte_x_shift_rmat, Vin.reshape((Nr, self.params.Nvt, self.params.Np))).reshape((Nr*self.params.Nvt, self.params.Np)) 
        # enforce rhs BCs
        Vin[self.xp_vt_l, 0]  = 0.0
        Vin[self.xp_vt_r, -1] = 0.0
        
        Vin_adv_x = xp.einsum("ijkl,ijl->ijk",self.bte_x_shift, Vin.reshape((Nr, self.params.Nvt, self.params.Np)))
        Vin_adv_x  = Vin_adv_x.reshape((Nr, self.params.Nvt *  self.params.Np))
        Vin_adv_x  = xp.dot(self.op_adv_x_q, Vin_adv_x).reshape((Nr , self.params.Nvt, self.params.Np)).reshape((Nr * self.params.Nvt, self.params.Np))
        
        # print((Vin_adv_x[self.xp_vt_l , 0] ==0).all())
        # print((Vin_adv_x[self.xp_vt_r , -1]==0).all())
        
        if PROFILE_SOLVERS==1:
            if xp == cp:
                cp.cuda.runtime.deviceSynchronize()
            t2 = perf_counter()
            if (verbose):
                print("time: [%.4E T] -- BTE x-advection cost = %.4E (s)" %(time, t2-t1), flush=True)
        
        #Vin_adv_x[Vin_adv_x<0] = 1e-16
        if (self.params.vtDe > 0):
            Vin_adv_x = xp.einsum("il,rlx->rix", self.LpDvt, Vin_adv_x.reshape((Nr, Nvt, Nx))).reshape((Nr * Nvt, Nx))
        
        return Vin_adv_x
    
    def step_bte_v(self, E, v, dv, time, dt, verbose=0):
        xp      = self.xp_module
      
        if PROFILE_SOLVERS==1:
            if xp == cp:
                cp.cuda.runtime.deviceSynchronize()
            t1 = perf_counter()
        
        time    = time
        dt      = dt
        u       = self.ords_to_sph(v)
        if self.params.vts_type == "BE":
            rtol            = self.params.rtol
            atol            = self.params.atol
            iter_max        = self.params.max_iter
            use_gmres       = not((E[0] == E).all() == True)
            dof_v           = self.op_col_en.shape[0] 
            
            steps_cycle     = int(1/dt)
            pmat_freq       = steps_cycle//50
            step            = int(time/dt)
            
            if use_gmres == True:
                pcEmat      = self.PmatE
                pcEval      = self.Evals
                pc_emat_idx = self.vspace_pc_setup(E)
                
                def Lmat_mvec(x):
                    x      = x.reshape((dof_v, self.params.Np))
                    y      = self.params.tau * (self.params.n0 * self.params.np0 * (xp.dot(self.op_col_en, x) + self.params.Tg * xp.dot(self.op_col_gT, x))  + E * xp.dot(self.op_adv_v, x))
                    y      = x - dt * y
                    return y.reshape((-1))
                
                def Mmat_mvec(x):
                    x      = x.reshape((dof_v, self.params.Np))
                    y      = xp.copy(x)
                    
                    for idx_id, idx in enumerate(pc_emat_idx):
                        y[:,idx[1]] = xp.dot(pcEmat[idx[0]], y[:, idx[1]])
                        
                    return y.reshape((-1))
                
                norm_b    = xp.linalg.norm(u.reshape((-1)))
                Ndof      = dof_v * self.params.Np
                
                if (self.params.use_gpu == 1):
                    Lmat_op   = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Lmat_mvec)
                    Mmat_op   = cupyx.scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Mmat_mvec)
                    gmres_c   = gmres_counter(disp=False)
                    v, status = cupyx.scipy.sparse.linalg.gmres(Lmat_op, u.reshape((-1)), x0=u.reshape((-1)), tol=rtol, atol=atol, M=Mmat_op, restart=self.params.gmres_rsrt, maxiter=self.params.gmres_rsrt * 50, callback=gmres_c)
                else:
                    Lmat_op   = scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Lmat_mvec)
                    Mmat_op   = scipy.sparse.linalg.LinearOperator((Ndof, Ndof), matvec=Mmat_mvec)
                    gmres_c   = gmres_counter(disp=False)
                    v, status = scipy.sparse.linalg.gmres(Lmat_op, u.reshape((-1)), x0=u.reshape((-1)), rtol=rtol, atol=atol, M=Mmat_op, restart=self.params.gmres_rsrt, maxiter=self.params.gmres_rsrt * 50, callback=gmres_c)

                
                norm_res_abs  = xp.linalg.norm(Lmat_mvec(v) -  u.reshape((-1)))
                norm_res_rel  = xp.linalg.norm(Lmat_mvec(v) -  u.reshape((-1))) / norm_b
                
                if (status !=0) :
                    print("%08d GMRES solver failed! iterations =%d  ||res|| = %.4E ||res||/||b|| = %.4E"%(step, status, norm_res_abs, norm_res_rel))
                    sys.exit(-1)
                    # self.bte_pmat = xp.linalg.inv(Lmat)
                    # v             = xp.einsum("ijk,ki->ji", self.bte_pmat, u)
                    # norm_res      = xp.linalg.norm(Lmat_mvec(xp.transpose(v).reshape((-1))) -  uT.reshape((-1))) / norm_b
                else:
                    v               = v.reshape((dof_v, self.params.Np))
        
            else:
                vmat          = self.params.n0 * self.params.np0 * (self.op_col_en + self.params.Tg * self.op_col_gT)
                Lop           = self.I_Nv - dt * self.params.tau * E[0] * self.op_adv_v - dt * self.params.tau * vmat
                gmres_c       = gmres_counter(disp=False)
                v             = xp.linalg.solve(Lop, u)
                norm_b        = xp.linalg.norm(u)
                norm_res_abs  = xp.linalg.norm(Lop @ v -  u)
                norm_res_rel  = norm_res_abs / norm_b

                #raise NotImplementedError
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
            print("%08d Boltzmann (v-space) step time = %.6E ||res||=%.12E ||res||/||b||=%.12E gmres iter = %04d"%(step, time, norm_res_abs, norm_res_rel, gmres_c.niter * self.params.gmres_rsrt))

        return v 
      
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
    
    def compute_qoi(self, v, time, dt):
        xp       = self.xp_module
        mm_fac   = np.sqrt(4 * np.pi) 
        c_gamma  = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))
        vth      = self.bs_vth

        v_lm     = self.ords_to_sph(v)
        
        m0       = xp.dot(self.op_mass, v_lm)
        Te       = xp.dot(self.op_temp,v_lm) / m0
        scale    = xp.dot(self.op_mass / mm_fac, v_lm) * (2 * (vth/c_gamma)**3)
        v_lm_n   = v_lm/scale
        num_sh   = len(self.op_spec_sp._sph_harm_lm)
        
        n0       = self.params.np0 * self.params.n0
        num_collisions = len(self.bs_coll_list)
        rr_rates = xp.array([xp.dot(self.op_rate[col_idx], v_lm_n[0::num_sh, :]) for col_idx in range(num_collisions)]).reshape((num_collisions, self.params.Np)).T
        
        # these are computed from SI units qoi/n0
        D_e      = xp.dot(self.op_diffusion, v_lm_n[0::num_sh]) * (c_gamma / 3.) / n0 
        mu_e     = xp.dot(self.op_mobility, v_lm_n[1::num_sh])  * ((c_gamma / (3 * ( 1 / n0)))) /n0
        
        return {"rates": rr_rates, "mu": mu_e, "D": D_e}
    
    def maxwellian_eedf(self, ne, Te):
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
        
        #rates       = np.zeros((len(op_rate), self.params.Np))
        num_p       = self.op_spec_sp._p + 1
        Vin         = np.zeros((num_p * self.params.Nvt, self.params.Np))
        spec_sp     = self.op_spec_sp
        mmat        = spec_sp.compute_mass_matrix()
        mmat_inv    = spec_sp.inverse_mass_mat(Mmat = mmat)
        vth         = self.bs_vth
        mw          = bte_utils.get_maxwellian_3d(vth, 1)

        [gmx,gmw]   = spec_sp._basis_p.Gauss_Pn(spec_sp._num_q_radial)
        Vqr_gmx     = spec_sp.Vq_r(gmx, 0, 1)
        
        num_p       = spec_sp._p +1
        num_sh      = len(spec_sp._sph_harm_lm)
        h_init      = np.zeros(num_p * num_sh)
        
        ev_max_ext        = (spec_sp._basis_p._t_unique[-1] * self.bs_vth/self.c_gamma)**2
        print("v-grid max = %.4E (eV) extended to = %.4E (eV)" %(self.ev_lim[1], ev_max_ext))
        for i in range(self.params.Np):
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
                ki = np.dot(op_rate[1], hh1[0::num_sh]) * self.params.np0 * self.params.tau
            print("k_ionization [m^3s^{-1}] = %.8E " %( ki / self.params.np0 / self.params.tau))
            
            Vin[:, i] = np.einsum("il,rl->ri", psh2o, h_init.reshape((num_p , num_sh))).reshape((num_p * len(self.xp_vt)))
            
        # scale functions to have ne, at initial timestep
        Vin = Vin * ne
        return xp.array(Vin)
    
    def initial_condition(self, type=0):
       xp   = np #self.xp_module
       assert xp == np

       xx = self.params.L * (self.xp + 1)
       if (type == 0):
          ne = 1e6 * (1e7 + 1e9 * (1-0.5 * xx/self.params.L)**2 * (0.5 * xx/self.params.L)**2) / self.params.np0
          Te = xp.ones_like(ne) * self.params.Te
          v  = self.maxwellian_eedf(ne, Te)
          return self.xp_module.array(v)
       else:
          raise NotImplementedError

    def init(self, dt):
        xp            = self.xp_module
        num_pc_evals  = 30
        ep            = xp.logspace(xp.log10(self.params.pcEmin), xp.log10(self.params.pcEmax), num_pc_evals//2, base=10)
        self.Evals    = -xp.flip(ep)
        self.Evals    = xp.append(self.Evals,ep)
        vmat          = self.params.n0 * self.params.np0 * (self.op_col_en + self.params.Tg * self.op_col_gT)
        #self.PmatC   = xp.linalg.inv(self.I_Nv - 1.0 * dt * self.params.tau * vmat)
        self.PmatE    = list()
        self.PmatE    = xp.array([xp.linalg.inv(self.I_Nv - dt * self.params.tau * self.Evals[i] * self.op_adv_v - dt * self.params.tau * vmat) for i in range(num_pc_evals)])
        
        print("v-space advection mat preconditioner gird : \n", self.Evals)
        self.__initialize_bte_adv_x__(0.5 * dt)
        
        return 
    
    def step(self, E, v, dv, time, dt, verbose=0):
        xp     = self.xp_module
        
        #if PROFILE_SOLVERS==1:
        if xp == cp:
            cp.cuda.runtime.deviceSynchronize()
        t1     = perf_counter()
        num_sh = len(self.op_spec_sp._sph_harm_lm)
        # Strang-Splitting
        v           = self.step_bte_x(v, time, dt * 0.5, verbose)
        v           = self.step_bte_v(E, v, None, time, dt, verbose)
        v           = self.step_bte_x(v, time + 0.5 * dt, dt * 0.5, verbose)
        
        #if PROFILE_SOLVERS==1:
        if xp == cp:
            cp.cuda.runtime.deviceSynchronize()
        
        t2 = perf_counter()
        
        if (verbose==1):
            print("BTE(vx-op-split) solver cost = %.4E (s)" %(t2-t1), flush=True)
            
        return v
    
    def normalize_edf(self, v):
        xp    = self.xp_module
        v_lm  = self.ords_to_sph(v)
        
        # normalization of the distribution function before computing the reaction rates
        mm_fac                   = self.op_spec_sp._sph_harm_real(0, 0, 0, 0) * 4 * np.pi
        mm_op                    = self.op_mass
        c_gamma                  = self.c_gamma
        vth                      = self.bs_vth
        
        scale                    = xp.dot(mm_op / mm_fac, v_lm) * (2 * (vth/c_gamma)**3)
        return v / scale
    
    def normalize_edf_lm(self, v_lm):
        xp    = self.xp_module
      
        # normalization of the distribution function before computing the reaction rates
        mm_fac                   = self.op_spec_sp._sph_harm_real(0, 0, 0, 0) * 4 * np.pi
        mm_op                    = self.op_mass
        c_gamma                  = self.c_gamma
        vth                      = self.bs_vth
        
        scale                    = xp.dot(mm_op / mm_fac, v_lm) * (2 * (vth/c_gamma)**3)
        return v_lm / scale
    
    def store_checkpoint(self, v, time, dt, fprefix):

        xp  = self.xp_module
        vsh = self.ords_to_sph(v)

        ne  = self.op_mass @ vsh
        Te  = (self.op_temp @ vsh) / ne

        try:
            with h5py.File("%s.h5"%(fprefix), 'w') as F:
                F.create_dataset("time[T]"      , data = np.array([time]))
                F.create_dataset("dt[T]"        , data = np.array([dt]))
                F.create_dataset("edf"          , data = self.asnumpy(v))
                F.create_dataset("ne[m^-3]"     , data = self.params.np0 * self.asnumpy(ne))
                F.create_dataset("Te[eV]"       , data = self.asnumpy(Te))

                F.close()
        except:
           print("checkpoint file write failed at time = %.4E"%(time), " : ", "%s.h5"%(fprefix) )
        
        return

    def restore_checkpoint(self, fprefix):
        xp = self.xp_module
        try:
            with h5py.File("%s.h5"%(fprefix), 'r') as F:
                time = xp.array(F["time[T]"][()])[0]
                dt   = xp.array(F["dt[T]"][()])[0]
                v    = xp.array(F["edf"][()])
                F.close()
        except:
           print("Error while reading the checkpoint file", " : ", "%s.h5"%(fprefix) )

        return time, dt, v
             
    def plot(self, E, v, fname, time):
        xp  = self.xp_module
        fig = plt.figure(figsize=(26,10), dpi=300)
      
        def asnumpy(a):
            if self.xp_module == cp:
                return cp.asnumpy(a)
            else:
                return a
      
        vsh = self.ords_to_sph(v)
        ne  = asnumpy(self.op_mass @ vsh)
        Te  = asnumpy(self.op_temp @ vsh)/ne
        
        plt.subplot(2, 4, 1)
        plt.semilogy(self.xp, self.params.np0 * ne, 'b',   label=r"$n_e$")
        #plt.semilogy(self.xp, self.params.np0 * ni, '--r', label=r"$n_i$")
        plt.xlabel(r"x/L")
        plt.ylabel(r"$density (m^{-3})$")
        plt.grid(visible=True)
        plt.legend()
      
        plt.subplot(2, 4, 2)
        plt.plot(self.xp, self.params.np0 * ne, 'b',  label=r"$n_e$")
        #plt.plot(self.xp, self.params.np0 * ni, '--r',label=r"$n_i$")
        plt.xlabel(r"x/L")
        #plt.ylabel(r"$(n_i -n_e)(m^{-3})$")
        plt.ylabel(r"$density (m^{-3})$")
        plt.legend()
        plt.grid(visible=True)
      
        plt.subplot(2, 4, 3)
        # plt.plot(self.xp, ne * Te * self.params.np0, 'b')
        # plt.ylabel(r"$n_e T_e (eV m^{-3})$")
        
        plt.plot(self.xp, Te, 'b')
        plt.ylabel(r"$T_e (eV)$")
        
        plt.xlabel(r"x/L")
        
        plt.grid(visible=True)
        
        plt.subplot(2, 4, 4)
        E   = asnumpy(E) 
        
        plt.plot(self.xp, E, 'b')
        plt.xlabel(r"x/L")
        plt.ylabel(r"$E (V/m)$")
        plt.grid(visible=True)
      
        # plt.subplot(2, 4, 5)
        # plt.plot(self.xp, phi * self.params.V0, 'b')
        # plt.xlabel(r"x/L")
        # plt.ylabel(r"$\phi (V)$")
        # plt.grid(visible=True)
        
        
        num_p   = self.op_spec_sp._p + 1
        num_sh  = len(self.op_spec_sp._sph_harm_lm)
        Vin_lm  = vsh
        
        plt.subplot(2, 4, 5)
        Vin_lm1      = self.normalize_edf_lm(Vin_lm)
        #r_elastic    = ne * self.params.np0 * asnumpy(xp.dot(self.op_rate[0], Vin_lm1[0::num_sh,:]))
        r_elastic    = asnumpy(xp.dot(self.op_rate[0], Vin_lm1[0::num_sh,:]))
        plt.semilogy(self.xp, r_elastic    , 'b', label="elastic")
        
        if (len(self.op_rate) > 1):
            #r_ionization = ne * self.params.np0 * asnumpy(xp.dot(self.op_rate[1], Vin_lm1[0::num_sh,:]))
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
        
        ev_grid   = np.linspace(max(ev_range[0],1e-2), ev_range[1], 1024)
        
        ff_v      = self.compute_radial_components(ev_grid, asnumpy(Vin_lm1))
        
        pts = 2
        plt.subplot(2, 4, 6)
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
        
        
        plt.subplot(2, 4, 7)
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


        plt.subplot(2, 4, 8)
        for i in range(pts):
            plt.semilogy(ev_grid, np.abs(ff_v[i][2])    ,'-', label="x=%.4f"%(self.xp[i]))
        for i in range(pts):
            plt.semilogy(ev_grid, np.abs(ff_v[-1-i][2]) ,'--', label="x=%.4f"%(self.xp[-1-i]))
            
        plt.semilogy(ev_grid, np.abs(ff_v[idx][2]), '-', label="x=%.4f"%(self.xp[idx]))
        plt.semilogy(ev_grid, np.abs(ff_v[idx+1][2]), '-', label="x=%.4f"%(self.xp[idx+1]))
        plt.xlabel(r"energy (eV)")
        plt.ylabel(r"abs(f2) $eV^{-3/2}$")
        plt.legend()
        plt.grid(visible=True)
        
        
        plt.tight_layout()
        plt.suptitle("T=%.4f cycles"%(time))
        fig.savefig(fname)
        plt.close()
    
    def asnumpy(self, x):
       if type(x) == cp.ndarray:
          return cp.asnumpy(x)
       else:
          assert type(x) == np.ndarray
          return x
    
    

        

            
        
       





    
    

