"""
@brief Perform batched spatially decoupled 0d3v bte solves where each 0d3v is parameterized by 
    1). n0 - heavy density
    2). E  - Electric field (can have oscillatory field)
    3). ni - ion density
    4). collisions model
        a). electron-heavy    collisions
        b). electron-electron collisions
"""

import utils as bte_utils
import spec_spherical as sp
import numpy as np
import collision_operator_spherical as collOpSp
import collisions 
import parameters as params
import basis
import argparse
import bolsig
import sys
import scipy.interpolate
import scipy.constants
import cross_section
from multiprocessing.pool import ThreadPool as WorkerPool
import enum
from os import environ
from profile_t import profile_t

from mpi4py import MPI

try:
    import cupy as cp
    import cupyx.scipy.sparse
except:
    print("cupy module not found")

NVTX_FLAGS=0

class pp(enum.IntEnum):
    ALL           = 0
    SETUP         = 1
    C_CS_SETUP    = 2
    C_EN_SETUP    = 3
    C_EE_SETUP    = 4
    ADV_SETUP     = 5
    INIT_COND     = 6
    SOLVE         = 7
    RHS_EVAL      = 8
    JAC_EVAL      = 9
    JAC_LA_SOL    = 10
    H2D           = 11
    D2H           = 12
    LAST          = 13

class vspace_grid_extension(enum.IntEnum):
    NONE   = 0, # no v-space extension is applied
    LINEAR = 1, # ev truncation energy is k * ev_max
    LOG    = 2, # log space ev extension


def newton_solver_batched(x, n_pts, residual, jacobian, jacobian_inv, atol, rtol, iter_max, xp=np):
    jac      = jacobian(x)
    assert jac.shape[0] == n_pts
    jac_inv  = jacobian_inv(jac) #xp.linalg.inv(jac)
    
    ns_info  = dict()
    alpha    = xp.ones(n_pts)
    x0       = x
    r0       = residual(x0)
    norm_rr  = norm_r0  = xp.linalg.norm(r0, axis=0)
    
    count        = 0
    cond1        = (norm_rr/norm_r0) < rtol
    cond2        = norm_rr < atol
    converged    = (cond1.all() or cond2.all())
    #print("  {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(count, xp.max(norm_rr), xp.max(norm_rr/norm_r0)))
    
    while( not converged and (count < iter_max) ):
        rr        = residual(x)
        norm_rr   = xp.linalg.norm(rr, axis=0)
        
        cond1     = (norm_rr/norm_r0) < rtol
        cond2     = norm_rr < atol
        converged = (cond1.all() or cond2.all())
        
        #print("  {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(count, xp.max(norm_rr), xp.max(norm_rr/norm_r0)))
        
        p         = xp.einsum("ijk,ki->ji", jac_inv, -rr)
        x         = x + alpha * p
        #print("p", xp.linalg.norm(p, axis=0))
        # xk        = x + alpha * p
        # rk        = residual(xk)
        # norm_rk   = xp.linalg.norm(rk, axis=0)
        # idx       = norm_rk > norm_rr
        # print("idx", (idx.any() == True))

        # while((idx.any() == True) and (alpha[idx]>rtol).all()):
        #     alpha[idx] *= 0.5
        #     xk          = x + alpha * p
        #     rk          = residual(xk)
        #     norm_rk     = xp.linalg.norm(rk, axis=0)
        #     idx         = norm_rk > norm_rr
        # x        = x + alpha * p
        
        count    += 1
    
    if (not converged):
        # solver failed !!!
        print("  {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(count, xp.max(norm_rr), xp.max(norm_rr/norm_r0)))
        print("non-linear solver step FAILED!!! try with smaller time step size or increase max iterations")
        #print(rr)
        print("norm_r0 \n",norm_r0)
        print("norm_rr \n",norm_rr)
        ns_info["status"] = converged
        ns_info["x"]      = x
        ns_info["atol"]   = xp.max(norm_rr)
        ns_info["rtol"]   = xp.max(norm_rr/norm_r0)
        ns_info["alpha"]  = alpha
        ns_info["iter"]   = count
        return ns_info
    
    ns_info["status"] = converged
    ns_info["x"]      = x
    ns_info["atol"]   = xp.max(norm_rr)
    ns_info["rtol"]   = xp.max(norm_rr/norm_r0)
    ns_info["alpha"]  = alpha
    ns_info["iter"]   = count
    return ns_info
    
def set_os_envthreads(threads):
    N_THREADS = str(threads)
    environ['OMP_NUM_THREADS'] = N_THREADS
    environ['OPENBLAS_NUM_THREADS'] = N_THREADS
    environ['MKL_NUM_THREADS'] = N_THREADS
    environ['VECLIB_MAXIMUM_THREADS'] = N_THREADS
    environ['NUMEXPR_NUM_THREADS'] = N_THREADS

class bte_0d3v_batched():
    
    def __init__(self, args, ev_max:np.array, Te: np.array, nr: np.array, lm: list, n_vspace_grids : int, collision_model: list) -> None:
        """
        ev_max          : maximum grid energy (eV)
        Te              : approximate electron temperature (eV)
        nr              : radial B-splines to use
        lm              : lm modes for each grid
        n_vspace_grids  : number of v-space grids to use
        collision model : collision string detailing which collisions to perform
        """
        
        comm = MPI.COMM_WORLD
        rank_ = comm.Get_rank()

        self.profile_nn     = ["all","setup", "e-n c_op", "e-e c_op", "adv op", "initialize", "solve", "rhs", "jac", "jac_solve", "H2D", "D2H", "last"]
        self.profile_tt_all = list()
        for i in range(n_vspace_grids):
            profile_tt          = [profile_t(self.profile_nn[i]) for i in range(int(pp.LAST))]
            self.profile_tt_all.append(profile_tt)
        
        self._par_nvgrids    = n_vspace_grids
        self._args           = args # common arguments for all cases
        self._par_ap_Te      = Te 
        self._par_vth        = collisions.electron_thermal_velocity(self._par_ap_Te * collisions.TEMP_K_1EV)
        self._par_nr         = nr
        self._par_ev_range   = [(0, ev_max[i]) for i in range(self._par_nvgrids)]
        self._par_lm         = [lm[i] for i in range(self._par_nvgrids)]    
        self._par_col_model  = collision_model
        
        self._c_gamma        = np.sqrt(2*collisions.ELECTRON_CHARGE_MASS_RATIO)
        
        self._par_bte_params  = [dict() for i in range(self._par_nvgrids)] 
        self._op_col_en       = [None for i in range(self._par_nvgrids)]
        self._op_col_gT       = [None for i in range(self._par_nvgrids)]
        self._op_col_ee       = [None for i in range(self._par_nvgrids)]
        self._op_advection    = [None for i in range(self._par_nvgrids)]
        self._op_mass_mat     = [None for i in range(self._par_nvgrids)]
        
        self._op_qmat         = [None for i in range(self._par_nvgrids)]
        self._op_rmat         = [None for i in range(self._par_nvgrids)]
        self._op_diag_dg      = [None for i in range(self._par_nvgrids)]
        self._op_sigma_m      = [None for i in range(self._par_nvgrids)]
        
        self._op_mass         = [None for i in range(self._par_nvgrids)]
        self._op_temp         = [None for i in range(self._par_nvgrids)]
        self._op_diffusion    = [None for i in range(self._par_nvgrids)]
        self._op_mobility     = [None for i in range(self._par_nvgrids)]
        self._op_rate         = [None for i in range(self._par_nvgrids)]
        self._op_spec_sp      = [None for i in range(self._par_nvgrids)]
        self._op_imat_vx      = [None for i in range(self._par_nvgrids)]
        
        self.xp_module        = np
        self._par_dof         = np.array([(self._par_nr[i]+1) * len(lm) for i in range(self._par_nvgrids)] , dtype=np.int32)
        
        
        profile_tt = self.profile_tt_all[0]
        profile_tt[pp.C_CS_SETUP].start()

        self._collision_names              = list()
        self._coll_list                    = list()
        self._avail_species                = list()
        self._cross_section_data           = list()

        if (len(self._par_col_model)==1):
            avail_species = cross_section.read_available_species(self._par_col_model[0])
            cs_data_all   = cross_section.read_cross_section_data(self._par_col_model[0])
            
            cross_section.CROSS_SECTION_DATA = cs_data_all
            coll_names     = list()
            coll_list      = list()
            
            if rank_ == 0:
                print("==========read collissions===========")
            collision_count = 0
            for col_str, col_data in cs_data_all.items():
                if rank_ == 0:
                    print(col_str, col_data["type"])
                g = collisions.electron_heavy_binary_collision(col_str, collision_type=col_data["type"])
                coll_list.append(g)
                coll_names.append("C%d"%(collision_count)) 
                collision_count+=1
            if rank_ == 0:
                print("=====================================")
                print("number of total collisions = %d " %(len(coll_list)))
            self.num_collisions = len(coll_list)
            
            for i in range(self._par_nvgrids):
                self._avail_species.append(avail_species)
                self._cross_section_data.append(cs_data_all)
                self._coll_list.append(coll_list)
                self._collision_names.append(coll_names)


        else:
            for i in range(self._par_nvgrids):
                #print(i, self._par_col_model[i])
                avail_species = cross_section.read_available_species(self._par_col_model[i])
                cs_data_all   = cross_section.read_cross_section_data(self._par_col_model[i])
                cross_section.CROSS_SECTION_DATA = cs_data_all
                self._avail_species.append(avail_species)
                self._cross_section_data.append(cs_data_all)

                collision_count = 0
                coll_names      = list()
                coll_list       = list()
                for col_str, col_data in cs_data_all.items():
                    if rank_ == 0:
                        print(col_str, col_data["type"])
                    g = collisions.electron_heavy_binary_collision(col_str, collision_type=col_data["type"])
                    coll_list.append(g)
                    coll_names.append("C%d"%(collision_count)) 
                    collision_count+=1
                if rank_ == 0:
                    print("number of total collisions = %d " %(len(coll_list)))
                self.num_collisions = len(coll_list)
                self._coll_list.append(coll_list)
                self._collision_names.append(coll_names)
        
        egrid     = np.logspace(-3, 3, 100, base=10)
        sigma_crs = np.zeros((self._par_nvgrids, len(self._coll_list[0]), len(egrid)))
        for i in range(self._par_nvgrids):
            for col_idx, g in enumerate(self._coll_list[i]):
                sigma_crs[i, col_idx] = g.total_cross_section(egrid)
        
        import matplotlib.pyplot as plt
        col_process_str = list()
        for col_str, col_data in self._cross_section_data[0].items():
            col_process_str.append(col_str)

        # for col_idx in range(sigma_crs.shape[1]):
        #     plt.figure(figsize=(8,8), dpi=200)
            
        #     for i in range(self._par_nvgrids):
        #         plt.loglog(egrid, sigma_crs[i, col_idx], label=r"$T_e = %.4E [K]$"%(self._par_ap_Te[i] * collisions.TEMP_K_1EV))
        
        #     plt.xlabel(r"energy [eV]")
        #     plt.ylabel(r"cross section [$m^2$]")
        #     plt.legend()
        #     plt.title("%s"%(col_process_str[col_idx]))
        #     plt.tight_layout()
        #     plt.savefig("%s_crs_%04d.png"%(self._args.out_fname, col_idx))
        #     #print("file saved in ", "%s_crs.png"%(self._args.out_fname))
        #     plt.close()
            
            
        
        # self._collision_names              = list()
        # self._coll_list                    = list()
        # self._avail_species                = cross_section.read_available_species(self._args.collisions)
        # cross_section.CROSS_SECTION_DATA   = cross_section.read_cross_section_data(self._args.collisions)
        # self._cross_section_data           = cross_section.CROSS_SECTION_DATA
        # print("==========read collissions===========")
        # collision_count = 0
        # for col_str, col_data in self._cross_section_data.items():
        #     print(col_str, col_data["type"])
        #     g = collisions.electron_heavy_binary_collision(col_str, collision_type=col_data["type"])
        #     self._coll_list.append(g)
        #     self._collision_names.append("C%d"%(collision_count)) 
        #     collision_count+=1
        # print("=====================================")
        # print("number of total collisions = %d " %(len(self._coll_list)))
        # self.num_collisions = len(self._coll_list)
        
        # species       = cross_section.read_available_species(self._args.collisions)
        # mole_fraction = np.array(self._args.ns_by_n0)[0:len(species)]
        # assert np.allclose(np.sum(mole_fraction),1), "mole fractions does not add up to 1.0"
        
        profile_tt[pp.C_CS_SETUP].stop()        
        return
    
    def get_collision_list(self):
        return self._coll_list    
    
    def get_collision_names(self):
        return self._collision_names
    
    def get_cross_section_data(self):
        return self._cross_section_data
                
    def assemble_operators(self, grid_idx:int, vspace_ext=vspace_grid_extension.LINEAR):
        """
        perform the operator setup for grid_idx
        """

        comm = MPI.COMM_WORLD
        rank_ = comm.Get_rank()

        # print("Rank = ", rank_, ", starting operator assembly for BTE")

        idx             = grid_idx
        args            = self._args
        collision_model = self._par_col_model
        lm              = self._par_lm[idx]
        profile_tt      = self.profile_tt_all[grid_idx]
        profile_tt[pp.SETUP].start()
        
        vth                    = self._par_vth[idx]
        maxwellian             = bte_utils.get_maxwellian_3d(vth, 1.0)
        
        coll_list              = self._coll_list[idx]
        sig_pts                =  list()
        for col_idx, g in enumerate(coll_list):
            g  = coll_list[col_idx]
            if g._reaction_threshold != None and g._reaction_threshold >0:
                sig_pts.append(g._reaction_threshold)
        
        self._sig_pts          = np.sort(np.array(list(set(sig_pts))))
        sig_pts                = self._sig_pts
        
        dg_nodes               = np.sqrt(np.array(sig_pts)) * self._c_gamma / vth
        ev_range               = self._par_ev_range[idx]    #(0, self._args.ev_max) #((0 * vth /self._c_gamma)**2, (6 * vth /self._c_gamma)**2)
        k_domain               = (np.sqrt(ev_range[0]) * self._c_gamma / vth, np.sqrt(ev_range[1]) * self._c_gamma / vth)
        use_ee                 = args.ee_collisions
        
        # at the moment we don't have flux conditions for the elliptic operator that depends on the Gas temperature, hence disable DG for now. 
        # the DG code should at the limit of advection dominated Tg is low, and E is large. 
        use_dg                 = 0
        self._args.use_dg      = use_dg
        if (vspace_ext == vspace_grid_extension.NONE):
            bb                     = basis.BSpline(k_domain, self._args.sp_order, self._par_nr[idx] + 1, sig_pts=dg_nodes, knots_vec=None, dg_splines=use_dg, verbose = args.verbose, extend_domain=False, extend_domain_with_log=False)
        elif(vspace_ext == vspace_grid_extension.LINEAR):
            bb                     = basis.BSpline(k_domain, self._args.sp_order, self._par_nr[idx] + 1, sig_pts=dg_nodes, knots_vec=None, dg_splines=use_dg, verbose = args.verbose, extend_domain=True, extend_domain_with_log=False)
        elif(vspace_ext == vspace_grid_extension.LOG):
            bb                     = basis.BSpline(k_domain, self._args.sp_order, self._par_nr[idx] + 1, sig_pts=dg_nodes, knots_vec=None, dg_splines=use_dg, verbose = args.verbose, extend_domain=False, extend_domain_with_log=True)
        else:
            raise NotImplementedError

        #bb                    = basis.BSpline(k_domain, self._args.sp_order, self._par_nr[idx] + 1, sig_pts=dg_nodes, knots_vec=None, dg_splines=use_dg, verbose = args.verbose, extend_domain_with_log=True)
        print("Rank = ", rank_, "grid idx: ", idx, " ev=", ev_range, " v/vth=",k_domain, "extended domain (ev) = ", (bb._t[-1] * vth/ self._c_gamma)**2 , "v/vth ext = ", bb._t[-1])
        
        spec_sp                = sp.SpectralExpansionSpherical(self._par_nr[idx], bb, self._par_lm[idx])
        spec_sp._num_q_radial  = bb._num_knot_intervals * self._args.spline_qpts
        collision_op           = collOpSp.CollisionOpSP(spec_sp)
        self._op_spec_sp[idx]  = spec_sp
        
        ## -- below we compute the all the operators for 0d3v bte solve
        self._par_ev_range[idx] = ev_range  
        self._op_mass_mat[idx]  = spec_sp.compute_mass_matrix()
        mmat_inv                = spec_sp.inverse_mass_mat(Mmat = self._op_mass_mat[idx])
        gx, gw                  = spec_sp._basis_p.Gauss_Pn(spec_sp._num_q_radial)
        sigma_m                 = np.zeros(len(gx))
        c_gamma                 = np.sqrt(2*collisions.ELECTRON_CHARGE_MASS_RATIO)
        gx_ev                   = (gx * vth / c_gamma)**2

        sigma_m                 = 0
        FOp                     = [0 for i in range(len(self._avail_species[idx]))]
        FOp_g                   = 0

        profile_tt[pp.C_EN_SETUP].start()
        
        for col_idx, (col_str, col_data) in enumerate(self._cross_section_data[idx].items()):
            g   =  self._coll_list[idx][col_idx]
            g.reset_scattering_direction_sp_mat()

            mol_idx       = self._avail_species[idx].index(col_data["species"])
            FOp[mol_idx] += collision_op.assemble_mat(g, maxwellian, vth, tgK=0.0, mp_pool_sz=args.threads)
            
            if col_data["type"] == "ELASTIC":
                FOp_g  += collision_op.electron_gas_temperature(g, maxwellian, vth, mp_pool_sz=args.threads)
            
            sigma_m  += g.total_cross_section(gx_ev)
        
        profile_tt[pp.C_EN_SETUP].stop()

        self._op_sigma_m[idx]   = sigma_m
        self._op_mass[idx]      = bte_utils.mass_op(spec_sp, 1)
        self._op_temp[idx]      = bte_utils.temp_op(spec_sp, 1) * 0.5 * scipy.constants.electron_mass * (2/3/scipy.constants.Boltzmann) / collisions.TEMP_K_1EV
        
        self._op_mobility[idx]  = bte_utils.mobility_op(spec_sp, maxwellian, vth)
        self._op_diffusion[idx] = bte_utils.diffusion_op(spec_sp, self._coll_list[idx], maxwellian, vth)
        
        rr_op  = [None] * len(self._coll_list[idx])
        for col_idx, g in enumerate(self._coll_list[idx]):
            rr_op[col_idx] = bte_utils.reaction_rates_op(spec_sp, [g], maxwellian, vth)
            
        self._op_rate[idx] = rr_op
        
        num_p  = spec_sp._p + 1
        num_sh = len(lm)
        
        profile_tt[pp.ADV_SETUP].start()
        
        if use_dg == 1 : 
            adv_mat, eA, qA = spec_sp.compute_advection_matix_dg(advection_dir=-1.0)
            qA              = np.kron(np.eye(spec_sp.get_num_radial_domains()), np.kron(np.eye(num_p), qA))
        else:
            # cg advection
            adv_mat         = spec_sp.compute_advection_matix()
            qA              = np.eye(adv_mat.shape[0])
        
        profile_tt[pp.ADV_SETUP].stop()    
        
        self._op_diag_dg[idx]   = qA
        FOp                     = [ np.matmul(np.transpose(qA), np.matmul(FOp[m_idx], qA)) for m_idx in range(len(self._avail_species[idx]))   ]
        FOp_g                   = np.matmul(np.transpose(qA), np.matmul(FOp_g, qA))
        
        self._op_advection[idx] = np.dot(mmat_inv, adv_mat) * (1 / vth) * collisions.ELECTRON_CHARGE_MASS_RATIO
        self._op_col_en[idx]    = np.array([np.dot(mmat_inv, FOp[m_idx])   for m_idx in range(len(self._avail_species[idx]))])
        self._op_col_gT[idx]    = np.dot(mmat_inv, FOp_g)
        
        mm_op   = self._op_mass[idx] * maxwellian(0) * vth**3
        u       = mm_op
        u       = np.dot(np.transpose(mm_op),qA)
        p_vec   = u.reshape((u.shape[0], 1)) / np.sqrt(np.dot(u, u))

        Imat    = np.eye(self._op_col_en[idx][0].shape[0])
        Impp    = (Imat - np.outer(p_vec, p_vec))
        Qm,Rm   = np.linalg.qr(Impp)
        
        if use_dg == 1 : 
            Qmat        = np.delete(Qm,(num_p-1) * num_sh + num_sh-1, axis=1)
            Rmat        = np.delete(Rm,(num_p-1) * num_sh + num_sh-1, axis=0)
        else:
            Qmat        = np.delete(Qm,(num_p-1) * num_sh + 0, axis=1)
            Rmat        = np.delete(Rm,(num_p-1) * num_sh + 0, axis=0)
        
        self._op_qmat[idx] = Qmat
        self._op_rmat[idx] = Rmat

        # print("Rank = ", rank_, "about to begin e-e collision assembly")
        
        if(use_ee == 1):
            if rank_ == 0:
                print("e-e collision assembly begin")
            profile_tt[pp.C_EE_SETUP].start()
            
            hl_op, gl_op         = collision_op.compute_rosenbluth_potentials_op(maxwellian, vth, 1, mmat_inv, mp_pool_sz=args.threads)
            cc_op_a, cc_op_b     = collision_op.coulomb_collision_op_assembly(maxwellian, vth, mp_pool_sz=args.threads)
            
            if (args.use_gpu==1):
                xp                   = cp
            else:
                xp                   = np
            
            hl_op                = xp.asarray(hl_op)
            gl_op                = xp.asarray(gl_op) 
            cc_op_a              = xp.asarray(cc_op_a)
            cc_op_b              = xp.asarray(cc_op_b)
            qA                   = xp.asarray(qA)
            mmat_inv             = xp.asarray(mmat_inv)
            
            cc_op                = xp.dot(cc_op_a, hl_op) + xp.dot(cc_op_b, gl_op)
            cc_op                = xp.dot(cc_op,qA)
            cc_op                = xp.dot(xp.swapaxes(cc_op,1,2),qA)
            cc_op                = xp.swapaxes(cc_op,1,2)
            cc_op                = xp.dot(xp.transpose(qA), cc_op.reshape((num_p*num_sh,-1))).reshape((num_p * num_sh, num_p * num_sh, num_p * num_sh))
            cc_op                = xp.dot(mmat_inv, cc_op.reshape((num_p*num_sh,-1))).reshape((num_p * num_sh, num_p * num_sh, num_p * num_sh))

            if xp == cp:
                cc_op                = xp.asnumpy(cc_op)
                xp._default_memory_pool.free_all_blocks()
                
            self._op_col_ee[idx] = cc_op
            profile_tt[pp.C_EE_SETUP].stop()
            if rank_ == 0:
                print("e-e collision assembly end")
        
        profile_tt[pp.SETUP].stop()
        return
    
    def initialize(self, grid_idx, n_pts, init_type = "maxwellian"):
        """
        Initialize the grid to Maxwell-Boltzmann distribution
        """
        profile_tt  = self.profile_tt_all[grid_idx]
        profile_tt[pp.INIT_COND].start()
        spec_sp     = self._op_spec_sp[grid_idx]
        mmat        = self._op_mass_mat[grid_idx]
        mmat_inv    = spec_sp.inverse_mass_mat(Mmat = mmat)
        vth         = self._par_vth[grid_idx]
        mw          = bte_utils.get_maxwellian_3d(vth, 1)

        mass_op     = self._op_mass[grid_idx]
        temp_op     = self._op_temp[grid_idx]

        if init_type == "maxwellian":
            v_ratio = 1.0 #np.sqrt(1.0/args.basis_scale)
            hv      = lambda v,vt,vp : (1/np.sqrt(np.pi)**3) * np.exp(-((v/v_ratio)**2)) / v_ratio**3
            h_init  = bte_utils.function_to_basis(spec_sp,hv,mw, spec_sp._num_q_radial, 2, 2, Minv=mmat_inv)
        elif init_type == "anisotropic":
            v_ratio = 1.0 #np.sqrt(1.0/args.basis_scale)
            hv      = lambda v,vt,vp : (1/np.sqrt(np.pi)**3 ) * (np.exp(-((v/v_ratio)**2)) / v_ratio**3) * (1 + np.cos(vt))
            h_init  = bte_utils.function_to_basis(spec_sp,hv,mw, spec_sp._num_q_radial, 4, 2, Minv=mmat_inv)
        else:
            raise NotImplementedError
        
        m0 = np.dot(mass_op,h_init) 
        t0 = np.dot(temp_op,h_init) * vth**2 /m0
        print("grid idx = %d -- initial data mass = %.8E temp (eV) = %.8E"%(grid_idx, m0, t0))
        f0 = np.zeros((len(h_init), n_pts))
        
        for i in range(n_pts):
            f0[:,i] = h_init

        Qmat                                    = self._op_qmat[grid_idx]
        INr                                     = np.eye(Qmat.shape[1])
        self._op_imat_vx[grid_idx]              = np.einsum("i,jk->ijk",np.ones(n_pts), INr)

        profile_tt[pp.INIT_COND].stop()
        return f0
    
    def set_boltzmann_parameter(self, grid_idx, key:str, value:np.array):
        self._par_bte_params[grid_idx][key] = value
        return
    
    def get_boltzmann_parameter(self, grid_idx, key:str):
        return self._par_bte_params[grid_idx][key]
    
    def host_to_device_setup(self, *args):
        """
        host_to_device_setup(dev_id, grid_idx) will put grid_idx operators on device dev_id
        """
        dev_id = args[0]
        idx    = args[1]
        
        profile_tt = self.profile_tt_all[idx]
        profile_tt[pp.H2D].start()
        
        with cp.cuda.Device(dev_id):
            self._op_mass_mat[idx]  = cp.asarray(self._op_mass_mat[idx])
            self._op_advection[idx] = cp.asarray(self._op_advection[idx])
            self._op_col_en[idx]    = cp.asarray(self._op_col_en[idx])
            self._op_col_gT[idx]    = cp.asarray(self._op_col_gT[idx])
            self._op_qmat[idx]      = cp.asarray(self._op_qmat[idx])
            self._op_rmat[idx]      = cp.asarray(self._op_rmat[idx])
            self._op_mass[idx]      = cp.asarray(self._op_mass[idx])
            self._op_temp[idx]      = cp.asarray(self._op_temp[idx])
            self._op_mobility[idx]  = cp.asarray(self._op_mobility[idx])
            self._op_diffusion[idx] = cp.asarray(self._op_diffusion[idx])
            self._op_diag_dg[idx]   = cp.asarray(self._op_diag_dg[idx])
            self._op_imat_vx[idx]   = cp.asarray(self._op_imat_vx[idx])
            
            for k, v in self._par_bte_params[idx].items():
                self._par_bte_params[idx][k] = cp.asarray(v)
            
            for col_idx, col in enumerate(self._coll_list[idx]):
                self._op_rate[idx][col_idx] = cp.asarray(self._op_rate[idx][col_idx])
                    
        if self._args.ee_collisions==1:
            with cp.cuda.Device(dev_id):
                self._op_col_ee[idx]   = cp.asarray(self._op_col_ee[idx])
        
        self.xp_module = cp
        profile_tt[pp.H2D].stop()
        return
    
    def device_to_host_setup(self, *args):
        dev_id   = args[0]
        idx      = args[1]
        
        profile_tt = self.profile_tt_all[idx]
        profile_tt[pp.D2H].start()
         
        with cp.cuda.Device(dev_id):
            self._op_mass_mat[idx]  = cp.asnumpy(self._op_mass_mat[idx])
            self._op_advection[idx] = cp.asnumpy(self._op_advection[idx])
            self._op_col_en[idx]    = cp.asnumpy(self._op_col_en[idx])
            self._op_col_gT[idx]    = cp.asnumpy(self._op_col_gT[idx])
            self._op_qmat[idx]      = cp.asnumpy(self._op_qmat[idx])
            self._op_rmat[idx]      = cp.asnumpy(self._op_rmat[idx])
            self._op_mass[idx]      = cp.asnumpy(self._op_mass[idx])
            self._op_temp[idx]      = cp.asnumpy(self._op_temp[idx])
            self._op_mobility[idx]  = cp.asnumpy(self._op_mobility[idx])
            self._op_diffusion[idx] = cp.asnumpy(self._op_diffusion[idx])
            self._op_diag_dg[idx]   = cp.asnumpy(self._op_diag_dg[idx])
            self._op_imat_vx[idx]   = cp.asnumpy(self._op_imat_vx[idx])
            
            for k, v in self._par_bte_params[idx].items():
                self._par_bte_params[idx][k] = cp.asnumpy(v)
                
            for col_idx, col in enumerate(self._coll_list[idx]):
                self._op_rate[idx][col_idx] = cp.asnumpy(self._op_rate[idx][col_idx])
                    
        if self._args.ee_collisions==1:
            with cp.cuda.Device(dev_id):
                self._op_col_ee[idx]   = cp.asnumpy(self._op_col_ee[idx])

        self.xp_module = np
        profile_tt[pp.D2H].stop()
        return
    
    def get_rhs_and_jacobian(self, grid_idx: int, n_pts:int):
        xp           = self.xp_module
        profile_tt   = self.profile_tt_all[grid_idx]
        args         = self._args
        eps_0        = scipy.constants.epsilon_0
        me           = scipy.constants.electron_mass
        qe           = scipy.constants.e
        
        vth          = self._par_vth[grid_idx]
        mw           = bte_utils.get_maxwellian_3d(vth, 1)
        Mop          = self._op_mass[grid_idx].reshape((1,-1))
        Top          = self._op_temp[grid_idx].reshape((1,-1)) * collisions.TEMP_K_1EV

        Qmat         = self._op_qmat[grid_idx]
        Rmat         = self._op_rmat[grid_idx]
        QTmat        = xp.transpose(self._op_qmat[grid_idx])
        
        c_en         = self._op_col_en[grid_idx]
        c_gT         = self._op_col_gT[grid_idx]
        c_ee         = self._op_col_ee[grid_idx]
        adv_mat      = self._op_advection[grid_idx]
        qA           = self._op_diag_dg[grid_idx]
        
        mm_op        = self._op_mass[grid_idx] * mw(0) * vth**3
        u            = mm_op
        u            = xp.dot(xp.transpose(mm_op),qA)
        
        """
        following tensor index notation used. 
        s           - mole fraction (should be number of species involved, X, X*, X** in the collisional process)
        v, u, a, b  - index in v-space
        x, y, z     - index in x-space
        """
        
        Wmat         = xp.einsum("v,svu->su",u, c_en)
        
        QT_Cen       = xp.einsum("mu,suv->smv", QTmat, c_en) #xp.dot(QTmat, c_en)
        QT_Cgt       = xp.dot(QTmat, c_gT)
        QT_A         = xp.dot(QTmat, adv_mat)
        
        QT_Cen_Q     = xp.einsum("av,svu->sau",QTmat, xp.dot(c_en, Qmat)) #xp.dot(QTmat, xp.dot(c_en, Qmat))
        QT_Cgt_Q     = xp.dot(QTmat, xp.dot(c_gT, Qmat))
        QT_A_Q       = xp.dot(QTmat, xp.dot(adv_mat, Qmat))
        Imat         = xp.eye(QTmat.shape[0])
        
        if args.ee_collisions==1:
            assert self._args.use_dg==0, "Coulombic collisoins are not permited in the DG formulation"
            cc_op_l1     = c_ee
            cc_op_l2     = xp.swapaxes(c_ee,1,2)
            
            def gamma_a(fb):
                m0           = mw(0) *  xp.dot(Mop, fb) * vth**3 
                kT           = mw(0) * (xp.dot(Top, fb) / m0) * vth**5 * scipy.constants.Boltzmann 
                kT           = xp.abs(kT).reshape((-1,))
                
                ne           = self._par_bte_params[grid_idx]["ne"]
            
                c_lambda     = ((12 * np.pi * (eps_0 * kT)**(1.5))/(qe**3 * xp.sqrt(ne)))
                gamma_a      = (xp.log(c_lambda) * (qe**4)) / (4 * np.pi * (eps_0 * me)**2) / (vth)**3
                return gamma_a
            
            def res_func(x, time, dt):
                if xp == cp:
                    xp.cuda.runtime.deviceSynchronize()
                
                if(NVTX_FLAGS==1):
                    cp.cuda.nvtx.RangePush("bteRHS")
                    
                profile_tt[pp.RHS_EVAL].start()
                
                n0           = self._par_bte_params[grid_idx]["n0"]
                ne           = self._par_bte_params[grid_idx]["ne"]
                Tg           = self._par_bte_params[grid_idx]["Tg"]
                E            = self._par_bte_params[grid_idx]["E"]
                ns_by_n0     = self._par_bte_params[grid_idx]["ns_by_n0"]
                
                c_ee_x       = xp.dot(c_ee, x)
                c_ee_xx      = xp.einsum("abc,bc->ac", c_ee_x, x)
                
                y1           = (n0 * ( xp.einsum("sx,svx->vx", ns_by_n0, xp.dot(QT_Cen, x))
                                + Tg * xp.dot(QT_Cgt, x) ) 
                                + E * xp.dot(QT_A, x) 
                                - n0 * xp.einsum("sx,sx->x", ns_by_n0, xp.dot(Wmat, x)) * xp.dot(QTmat, x) 
                                + ne * gamma_a(x) * xp.dot(QTmat, c_ee_xx))
                if xp == cp:
                    xp.cuda.runtime.deviceSynchronize()
                profile_tt[pp.RHS_EVAL].stop()
                
                if(NVTX_FLAGS==1):
                    cp.cuda.nvtx.RangePop()
                return y1
            
            def jac_func(x, time, dt):
                if xp==cp:
                    xp.cuda.runtime.deviceSynchronize()
                
                if(NVTX_FLAGS==1):
                    cp.cuda.nvtx.RangePush("bteJac")
                    
                profile_tt[pp.JAC_EVAL].start()
                n0           = self._par_bte_params[grid_idx]["n0"]
                ne           = self._par_bte_params[grid_idx]["ne"]
                Tg           = self._par_bte_params[grid_idx]["Tg"]
                E            = self._par_bte_params[grid_idx]["E"]
                ns_by_n0     = self._par_bte_params[grid_idx]["ns_by_n0"]
                ns           = n0 * ns_by_n0

                #Lmat_pre    = xp.einsum("a,bc->abc", n0, QT_Cen_Q) + xp.einsum("a,bc->abc", n0 * Tg, QT_Cgt_Q)
                Lmat_pre     = xp.einsum("sx,svu->xvu", ns, QT_Cen_Q) + xp.einsum("a,bc->abc", n0 * Tg, QT_Cgt_Q)
                
                cc1_x_p_cc2x = (xp.dot(cc_op_l1, x) + xp.dot(cc_op_l2, x)) 
                mu           = xp.einsum("sx,sx->x", ns, xp.dot(Wmat, x)) 
                cc1_x_p_cc2x = xp.swapaxes(xp.swapaxes(cc1_x_p_cc2x, 0, 2), 1, 2)
                
                ccQ          = xp.einsum("abc,cd->abd", cc1_x_p_cc2x, Qmat)
                QTccQ        = xp.einsum("be,aec->abc", QTmat       , ccQ )
                
                Lmat         = Lmat_pre + xp.einsum("a,bc->abc", E, QT_A_Q) + xp.einsum("a,abc->abc", ne * gamma_a(x), QTccQ)  -xp.einsum("a,bc->abc", mu, Imat)
                
                if xp==cp:
                    xp.cuda.runtime.deviceSynchronize()
                profile_tt[pp.JAC_EVAL].stop()
                
                if(NVTX_FLAGS==1):
                    cp.cuda.nvtx.RangePop()
                    
                return Lmat
                    
        else:
            
            def res_func(x, time, dt):
                if xp == cp:
                    xp.cuda.runtime.deviceSynchronize()
                
                if(NVTX_FLAGS==1):
                    cp.cuda.nvtx.RangePush("bteRHS")
                    
                profile_tt[pp.RHS_EVAL].start()
                n0           = self._par_bte_params[grid_idx]["n0"]
                ne           = self._par_bte_params[grid_idx]["ne"]
                Tg           = self._par_bte_params[grid_idx]["Tg"]
                E            = self._par_bte_params[grid_idx]["E"]
                ns_by_n0     = self._par_bte_params[grid_idx]["ns_by_n0"]
                
                y1           = (n0 * ( xp.einsum("sx,svx->vx", ns_by_n0, xp.dot(QT_Cen, x))
                                + Tg * xp.dot(QT_Cgt, x) ) 
                                + E * xp.dot(QT_A, x) 
                                - n0 * xp.einsum("sx,sx->x", ns_by_n0, xp.dot(Wmat, x)) * xp.dot(QTmat, x) 
                                )
                if xp == cp:
                    xp.cuda.runtime.deviceSynchronize()
                profile_tt[pp.RHS_EVAL].stop()
                
                if(NVTX_FLAGS==1):
                    cp.cuda.nvtx.RangePop()
                    
                return y1
            
            def jac_func(x, time, dt):
                if xp==cp:
                    xp.cuda.runtime.deviceSynchronize()
                
                if(NVTX_FLAGS==1):
                    cp.cuda.nvtx.RangePush("bteJac")
                    
                profile_tt[pp.JAC_EVAL].start()
                n0           = self._par_bte_params[grid_idx]["n0"]
                ne           = self._par_bte_params[grid_idx]["ne"]
                Tg           = self._par_bte_params[grid_idx]["Tg"]
                E            = self._par_bte_params[grid_idx]["E"]
                ns_by_n0     = self._par_bte_params[grid_idx]["ns_by_n0"]
                ns           = n0 * ns_by_n0
                
                Lmat_pre     = xp.einsum("sx,svu->xvu", ns, QT_Cen_Q) + xp.einsum("a,bc->abc", n0 * Tg, QT_Cgt_Q)
                mu           = xp.einsum("sx,sx->x", ns, xp.dot(Wmat, x)) 
                Lmat         = Lmat_pre + xp.einsum("a,bc->abc", E, QT_A_Q) - xp.einsum("a,bc->abc", mu, Imat)
                
                if xp==cp:
                    xp.cuda.runtime.deviceSynchronize()
                profile_tt[pp.JAC_EVAL].stop()
                
                if(NVTX_FLAGS==1):
                    cp.cuda.nvtx.RangePop()
                    
                return Lmat
            
        return res_func, jac_func
    
    def rhs_and_jac_flops(self):
        args   = self._args
        s      = len(self._avail_species[0])
        
        def res_flops(n, p):
            m0 = p * (2 * n -1) * 2 + (1 + 1 + 1) * p                       # ne * gamma_a(x)
            m1 = (n-1) * p * (2 * n -1) + (2 * s -1) * (n-1) * p            # xp.einsum("sx,svx->vx", ns_by_n0, xp.dot(QT_Cen, x))
            m2 = (n-1) * p * (2 * n -1) + (n-1) * p                         # Tg * xp.dot(QT_Cgt, x)
            m3 = (n-1) * p * 3                                              # n0 * ( xp.dot(QT_Cen,x) + Tg * xp.dot(QT_Cgt, x) )
            m4 = (n-1) * p * (2 * n -1) + (n-1) * p                         # ef * xp.dot(QT_A,x)
            m5 = p * (2 * n -1) + p + (n-1) * p * (2 * n - 1)  + (n-1) * p  # n0 * xp.dot(Wmat, x) *  xp.dot(QTmat, x)
            m6 = n * n * p * (2 * n -1)                                     # c_ee_x  = xp.dot(c_ee, x) 
            m7 = n * p * (2 * n -1)  + (n-1) * p * (2 * n - 1) + (n-1) * p  # ne * gamma_a(x) * xp.dot(QTmat, xp.dot(c_ee_x[:,:,ii], x[:,ii]))
            if args.ee_collisions==1:
                return m0 + m1 + m2 + m3 + m4 + m5 + m6 + m7
            else:
                return m1 + m2 + m3 + m4 + m5 
        
        def jac_flops(n,p):
            m0 = p * (2 * n -1) * 2 + (1 + 1 + 1) * p                       # ne * gamma_a(x)
            m1 = n * n * p * (2 * n - 1) * 2 + n * n * p                    # cc1_x_p_cc2x = xp.dot(cc_op_l1, x) + xp.dot(cc_op_l2, x)
            m2 = p * (2 *n -1) + p                                          # mu           = n0 * xp.dot(Wmat, x)
            m3 = (n-1) * (n-1) * p * (2 * s -1) + p * (n-1) * (n-1) * 2     # xp.einsum("sx,svu->xvu", ns, QT_Cen_Q) + xp.einsum("a,bc->abc", n0 * Tg, QT_Cgt_Q) + xp.einsum("a,bc->abc", E, QT_A_Q)
            m4 = n * p * (n-1) * (2 * n -1) + (n-1) * (n-1) * p * (2 * n -1) + (n-1) * (n-1) * p #ga[ii] * xp.dot(QTmat, xp.dot(cc1_x_p_cc2x[:,:,ii], Qmat))
            m5 = p * (n-1) * (n-1)                                          # Imat * mu[ii]
            
            if args.ee_collisions==1:
                return m0 + m1 + m2 + m3 + m4 + m5
            else:
                return m2 + m3 + m5
        
        return res_flops, jac_flops
    
    def batched_inv(self, grid_idx, Jmat):
        """
        computes the batched inverse of array of dense matrices
        """
        xp           = self.xp_module
        profile_tt   = self.profile_tt_all[grid_idx]
        
        if xp==cp:
            xp.cuda.runtime.deviceSynchronize()
        
        if(NVTX_FLAGS==1):
            cp.cuda.nvtx.RangePush("bteJacSolve")
            
        profile_tt[pp.JAC_LA_SOL].start()
        Jmat_inv = xp.linalg.inv(Jmat)
        
        if xp==cp:
            xp.cuda.runtime.deviceSynchronize()
            
        profile_tt[pp.JAC_LA_SOL].stop()
        
        if(NVTX_FLAGS==1):
            cp.cuda.nvtx.RangePop()
            
        return Jmat_inv
            
    def steady_state_solve(self, grid_idx : int, f0 : np.array, atol, rtol, max_iter):
        comm = MPI.COMM_WORLD
        rank_ = comm.Get_rank()

        xp           = self.xp_module
        profile_tt   = self.profile_tt_all[grid_idx]
        
        if xp==cp:
            xp.cuda.runtime.deviceSynchronize()
        
        profile_tt[pp.SOLVE].start()
        
        n_pts        = f0.shape[1]
        vth          = self._par_vth[grid_idx]
        mw           = bte_utils.get_maxwellian_3d(vth, 1)
        
        c_en         = self._op_col_en[grid_idx]
        qA           = self._op_diag_dg[grid_idx]
        mm_op        = self._op_mass[grid_idx] * mw(0) * vth**3
        u            = xp.dot(xp.transpose(mm_op),qA)
        Wmat         = xp.dot(u, c_en)
        
        Qmat         = self._op_qmat[grid_idx]
        Rmat         = self._op_rmat[grid_idx]
        QTmat        = xp.transpose(self._op_qmat[grid_idx])
        
        res_func, jac_func = self.get_rhs_and_jacobian(grid_idx, n_pts)
        
        abs_error       = np.ones(n_pts)
        rel_error       = np.ones(n_pts) 
        iteration_steps = 0

        f0       = f0 / xp.dot(u,f0)
        f0       = np.dot(qA.T, f0)
        
        fb_prev  = xp.dot(Rmat, f0)
        f1       = u  / xp.dot(u, u)
        f1p      = xp.zeros((len(f1), n_pts))
        
        for ii in range(n_pts):
            f1p[:,ii] = f1
            
        h_prev   = f1p + xp.dot(Qmat,fb_prev)
        
        while ((abs_error > atol).any() and (rel_error > rtol).any() and iteration_steps < max_iter):
            Lmat      =  jac_func(h_prev, 0, 0)
            rhs_vec   =  res_func(h_prev, 0, 0)
            abs_error =  xp.linalg.norm(rhs_vec, axis=0)
            
            Lmat_inv  =  self.batched_inv(grid_idx, Lmat)
            
            pp_mat    =  xp.einsum("abc,ca->ba",Lmat_inv, -rhs_vec)
            p         =  xp.dot(Qmat,pp_mat)

            alpha  = xp.ones(n_pts)
            is_diverged = False
            rf_new = xp.linalg.norm(res_func(h_prev + alpha * p, 0, 0), axis=0)
            while ((rf_new  >  abs_error).any()):
                rc = rf_new  >  abs_error
                alpha[rc]*=0.5
                
                if (alpha < 1e-16).all():
                    is_diverged = True
                    break
                
                rf_new = xp.linalg.norm(res_func(h_prev + alpha * p, 0, 0), axis=0)
                
            
            if(is_diverged):
                print("Rank = ", rank_, ", grid_idx ", grid_idx, " [steady-state] iteration ", iteration_steps, ": Residual =", abs_error, "line search step size becomes too small")
                break
            
            h_curr      = h_prev + alpha * p
            
            if iteration_steps % 10 == 0:
                rel_error = xp.linalg.norm(h_prev-h_curr, axis=0)/xp.linalg.norm(h_curr, axis=0)
                print("Rank = ", rank_, ", grid_idx ", grid_idx, " [steady-state] iteration ", iteration_steps, ": abs residual = %.8E rel residual=%.8E mass =%.8E"%(xp.max(abs_error), xp.max(rel_error), xp.max(xp.dot(u, h_prev))))
            
            #fb_prev      = np.dot(Rmat,h_curr)
            h_prev       = h_curr #f1p + np.dot(Qmat,fb_prev)
            iteration_steps+=1

        print("Rank = ", rank_, ", grid_idx ", grid_idx, " [steady-state] nonlinear solver (1) atol=%.8E , rtol=%.8E"%(xp.max(abs_error), xp.max(rel_error)))
        if xp==cp:
            xp.cuda.runtime.deviceSynchronize()
        profile_tt[pp.SOLVE].stop()
        
        h_curr = xp.dot(qA, h_curr)
        h_curr = self.normalized_distribution(grid_idx, h_curr)
        qoi    = self.compute_QoIs(grid_idx, h_curr, effective_mobility=True)
        return h_curr, qoi
    
    def step(self, grid_idx:int, f0:np.array, atol, rtol, max_iter, time, delta_t):
        """
        perform a single step of the transient 0d batched solve. 
        """
        xp           = self.xp_module
        profile_tt   = self.profile_tt_all[grid_idx]
        
        if xp==cp:
            xp.cuda.runtime.deviceSynchronize()
        profile_tt[pp.SOLVE].start()
        
        Qmat         = self._op_qmat[grid_idx]
        Rmat         = self._op_rmat[grid_idx]
        QTmat        = xp.transpose(self._op_qmat[grid_idx])
        qA           = self._op_diag_dg[grid_idx]
        tau          = 1/(self._args.Efreq)
        
        n_pts        = f0.shape[1]
        Imat         = self._op_imat_vx[grid_idx]
        
        rhs , rhs_u  = self.get_rhs_and_jacobian(grid_idx, f0.shape[1])
        tt           = time
        dt           = delta_t
        
        def residual(du):
            return du - dt * rhs(u + xp.dot(Qmat, du), tt + dt, dt)
            #return du  - 0.5 * dt * tau * rhs(u + xp.dot(Qmat, du), (tt + dt) * tau , tau * dt) - 0.5 * dt * tau * rhs(u, tt * tau, dt * tau)

        def jacobian(du):
            return (Imat - dt * rhs_u(u, tt, dt))
            #return Imat - 0.5 * dt * tau * rhs_u(u, tt * tau, dt * tau)
        
        def jacobian_inv(J):
            return self.batched_inv(grid_idx, J) 
        
        u       = f0            
        du      = xp.zeros((Qmat.shape[1],n_pts))
        ns_info = newton_solver_batched(du, n_pts, residual, jacobian, jacobian_inv, atol, rtol, max_iter, xp)
        
        if ns_info["status"]==False:
            print("At time = %.2E "%(time), end='')
            print("non-linear solver step FAILED!!! try with smaller time step size or increase max iterations")
            print("  Newton iter {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(ns_info["iter"], xp.max(ns_info["atol"]), xp.max(ns_info["rtol"])))
            sys.exit(-1)
        
        du          = ns_info["x"]
        v           = u + xp.dot(Qmat, du)
        
        if xp==cp:
            xp.cuda.runtime.deviceSynchronize()
        profile_tt[pp.SOLVE].stop()
        return v
        
    def solve(self, grid_idx:int, f0:np.array, atol, rtol, max_iter:int, solver_type:str):
        xp           = self.xp_module
        profile_tt   = self.profile_tt_all[grid_idx]
        
        Qmat         = self._op_qmat[grid_idx]
        Rmat         = self._op_rmat[grid_idx]
        QTmat        = xp.transpose(self._op_qmat[grid_idx])
        qA           = self._op_diag_dg[grid_idx]
        vth          = self._par_vth[grid_idx]
        mw           = bte_utils.get_maxwellian_3d(vth, 1)
        mm_op        = self._op_mass[grid_idx] * mw(0) * vth**3
        
        if (solver_type == "steady-state"):
            return self.steady_state_solve(grid_idx, f0, atol, rtol, max_iter)
        elif(solver_type== "transient"):
            dt              = self._args.dt     
            tT              = self._args.cycles
            tau             = 1e-7 if self._args.Efreq==0 else (1/self._args.Efreq)
            
            io_cycle        = 1.00
            io_freq         = int(io_cycle/dt)
            steps_total     = int(tT/dt)
            rhs , rhs_u     = self.get_rhs_and_jacobian(grid_idx, f0.shape[1])
            
            
            u               = f0 / xp.dot(mm_op, f0)
            u               = np.dot(qA.T, u)
            tt              = 0
            n_pts           = f0.shape[1]
            INr             = xp.eye(Qmat.shape[1])
            Imat            = xp.einsum("i,jk->ijk",xp.ones(n_pts), INr) 
            
            a1              = a2 = 1.0
            du              = xp.zeros((Qmat.shape[1],n_pts))
            
            u0              = xp.zeros_like(u)
            #u1             = xp.zeros_like(u)
            cycle_avg_u     = xp.zeros_like(u)
            cycle_avg_v     = xp.zeros_like(u)
            
            v_qoi           = xp.zeros((3 + len(self._coll_list[grid_idx]), n_pts))
            u0              = xp.copy(u)
            
            if xp==cp:
                xp.cuda.runtime.deviceSynchronize()
            
            profile_tt[pp.SOLVE].start()
            for ts_idx in range(steps_total+1):
                if (self._args.Efreq>0):
                    eRe = self.get_boltzmann_parameter(grid_idx, "eRe")
                    eIm = self.get_boltzmann_parameter(grid_idx, "eIm")
                    Et  = eRe * xp.cos(2 * xp.pi * (tt + dt)) + eIm * xp.sin(2 * xp.pi * (tt + dt))
                    self.set_boltzmann_parameter(grid_idx, "E", Et)

                def residual(du):
                    return du - dt * tau * rhs(u + xp.dot(Qmat, du), (tt + dt) * tau, dt * tau)
                    #return du  - 0.5 * dt * tau * rhs(u + xp.dot(Qmat, du), (tt + dt) * tau , tau * dt) - 0.5 * dt * tau * rhs(u, tt * tau, dt * tau)
        
                def jacobian(du):
                    return (Imat - dt * tau * rhs_u(u, tt * tau, dt * tau))
                    #return Imat - 0.5 * dt * tau * rhs_u(u, tt * tau, dt * tau)
                    
                def jacobian_inv(J):
                    return self.batched_inv(grid_idx, J) 
                
                ns_info = newton_solver_batched(du, n_pts, residual, jacobian, jacobian_inv, atol, rtol, max_iter, xp)
                
                if ns_info["status"]==False:
                    print("At time = %.2E "%(tt), end='')
                    print("non-linear solver step FAILED!!! try with smaller time step size or increase max iterations")
                    print("  Newton iter {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(ns_info["iter"], xp.max(ns_info["atol"]), xp.max(ns_info["rtol"])))
                    sys.exit(-1)
                    #return u0
                
                if(ts_idx > 0 and ts_idx % io_freq ==0):
                    cycle_avg_u   *= 0.5 * dt / io_cycle
                    
                    h_curr         = xp.dot(qA, cycle_avg_u)
                    h_curr         = self.normalized_distribution(grid_idx, h_curr)
                    qoi            = self.compute_QoIs(grid_idx, h_curr, effective_mobility=False)
                    
                    v_qoi[0]    = qoi["energy"]
                    v_qoi[1]    = qoi["mobility"]
                    v_qoi[2]    = qoi["diffusion"]
                    rates       = qoi["rates"]
                    
                    for col_idx, g in enumerate(self._coll_list[grid_idx]):
                        v_qoi[3 + col_idx] = rates[col_idx]
                    
                    a1 = xp.linalg.norm(u-u0)
                    a2 = a1/ xp.linalg.norm(u0)
                    
                    print("time = %.2E "%(tt), end='')
                    print("  Newton iter {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(ns_info["iter"], xp.max(ns_info["atol"]), xp.max(ns_info["rtol"])))
                    print("||u(t+T) - u(t)|| = %.8E and ||u(t+T) - u(t)||/||u(t)|| = %.8E"% (a1, a2))
                    
                    u0                   = np.copy(u)
                    cycle_avg_v[: , :]   = cycle_avg_u
                    cycle_avg_u[: , :]   = 0
                    
                    if (ts_idx == steps_total):
                        break
                    
                
                cycle_avg_u +=u
                du          = ns_info["x"]
                u           = u + xp.dot(Qmat, du)
                cycle_avg_u +=u
                tt += dt
            
            if xp==cp:
                xp.cuda.runtime.deviceSynchronize()
                                
            profile_tt[pp.SOLVE].stop()    
                        
            h_curr = xp.dot(qA, cycle_avg_v)
            h_curr = self.normalized_distribution(grid_idx, h_curr)
            qoi    = self.compute_QoIs(grid_idx, h_curr, effective_mobility=False)
            
            # xp.save("%s_avg_qoi_%04d.npy"%(self._args.out_fname, grid_idx), v_qoi)
            # xp.save("%s_avg_f_%04d.npy"%(self._args.out_fname, grid_idx)  , cycle_avg_v)
            # xp.save("%s_f_%04d.npy"%(self._args.out_fname, grid_idx)      , u)
            return u, qoi
            
        elif(solver_type=="BE_L"):
            pass
        else:
            raise NotImplementedError    
            
    def compute_QoIs(self, grid_idx,  ff, effective_mobility=True):
        args     = self._args
        spec_sp  = self._op_spec_sp[grid_idx]
        vth      = self._par_vth[grid_idx]
        mw       = bte_utils.get_maxwellian_3d(vth, 1)
        
        n0       = self._par_bte_params[grid_idx]["n0"]
        # ne       = self._par_bte_params[grid_idx]["ne"]
        # ni       = self._par_bte_params[grid_idx]["ni"]
        # Tg       = self._par_bte_params[grid_idx]["Tg"]
        E       = self._par_bte_params[grid_idx]["E"]
        
        num_p   = spec_sp._p + 1
        num_sh  = len(spec_sp._sph_harm_lm)
        
        c_gamma   = self._c_gamma
        eavg_to_K = (2/(3*scipy.constants.Boltzmann))

        #print(tgrid/1e-5)
        #print(Ef(tgrid))
        
        xp  = cp.get_array_module(ff) 

        mm  = xp.dot(self._op_mass[grid_idx], ff) * mw(0) * vth**3
        mu  = 1.5 * xp.dot(self._op_temp[grid_idx], ff) * mw(0) * vth**5  / mm

        if effective_mobility:
            M   = xp.dot(self._op_mobility[grid_idx], xp.sqrt(3) * ff[1::num_sh, :])  * (-(c_gamma / (3 * ( E / n0))))
        else:
            M   = xp.dot(self._op_mobility[grid_idx], xp.sqrt(3) * ff[1::num_sh, :])  * (-(c_gamma / (3 * ( 1 / n0))))
        
        D   = xp.dot(self._op_diffusion[grid_idx],  ff[0::num_sh,:]) * (c_gamma / 3.)
        
        rr  = list()
        for col_idx, g in enumerate(self._coll_list[grid_idx]):
            reaction_rate = xp.dot(self._op_rate[grid_idx][col_idx], ff[0::num_sh, :])
            rr.append(reaction_rate)
            
        rr = xp.array(rr)
        return {"energy":mu, "mobility":M, "diffusion": D, "rates": rr}            

    def normalized_distribution(self, grid_idx: int, ff):
        c_gamma      = np.sqrt(2*scipy.constants.e / scipy.constants.m_e)
        spec_sp      = self._op_spec_sp[grid_idx]
        vth          = self._par_vth[grid_idx]
        mass_op      = self._op_mass[grid_idx]
    
        num_p        = spec_sp._p +1
        num_sh       = len(spec_sp._sph_harm_lm)
        xp           = cp.get_array_module(ff)
    
        mm_fac       = spec_sp._sph_harm_real(0, 0, 0, 0) * 4 * xp.pi
        scale        = xp.dot(mass_op / mm_fac, ff) * (2 * (vth/c_gamma)**3)
        #scale         = np.dot(f_vec,mm_op) * maxwellian(0) * vth**3
        return ff/scale
        
    def compute_radial_components(self, grid_idx : int, ev: np.array, ff):
        ff_cpu = ff
        if cp.get_array_module(ff)==cp:
            ff_cpu = cp.asnumpy(ff)
            
        ff_cpu   = np.transpose(ff_cpu)
        vth      = self._par_vth[grid_idx]
        spec_sp  = self._op_spec_sp[grid_idx]
        
        vr       = np.sqrt(ev) * self._c_gamma / vth
        num_p    = spec_sp._p +1 
        num_sh   = len(spec_sp._sph_harm_lm)
        n_pts    = ff.shape[1]
        
        output   = np.zeros((n_pts, num_sh, len(vr)))
        Vqr      = spec_sp.Vq_r(vr,0,1)
        
        for l_idx, lm in enumerate(spec_sp._sph_harm_lm):
                output[:, l_idx, :] = np.dot(ff_cpu[:,l_idx::num_sh], Vqr)

        return output
    
    def profile_reset(self):
        args = self._args
        for profile_tt in self.profile_tt_all:
            for i in range(pp.LAST):
                profile_tt[i].reset()
        return
    
    def profile_stats(self, fname=""):
        args = self._args
        dt              = args.dt     
        tT              = args.cycles
        tau             = 1e-7 if self._args.Efreq==0 else (1/args.Efreq)
        io_cycle        = 1.00
        io_freq         = int(io_cycle/dt)
        steps_total     = int(tT/dt)
        
        solve_s1        = 1
        if args.solver_type=="transient":
            solve_s1    = 1/steps_total
        
        res_flops, jac_flops = self.rhs_and_jac_flops()
        n                    = (args.l_max + 1) * (args.Nr + 1) 
        p                    = args.n_pts
        profile_tt           = self.profile_tt_all[0]
        
        t_setup              = profile_tt[pp.SETUP].seconds
        t_solve              = profile_tt[pp.SOLVE].seconds * solve_s1
        
        t_rhs                = profile_tt[pp.RHS_EVAL].seconds
        t_jac                = profile_tt[pp.JAC_EVAL].seconds
        t_jac_solve          = profile_tt[pp.JAC_LA_SOL].seconds
        
        
        
        print("--setup\t %.4Es"%(t_setup))
        print("   |electon-X \t %.4Es"%(profile_tt[pp.C_EN_SETUP].seconds))
        print("   |coulombic \t %.4Es"%(profile_tt[pp.C_EE_SETUP].seconds))
        print("   |advection \t %.4Es"%(profile_tt[pp.ADV_SETUP].seconds))
        print("--solve \t %.4Es"%(t_solve))
        
        t1         = profile_tt[pp.RHS_EVAL].seconds/profile_tt[pp.RHS_EVAL].iter
        flops_rhs  = res_flops(n, p) / t1
        
        print("   |rhs \t %.4Es # iter = %d"%(profile_tt[pp.RHS_EVAL].seconds, profile_tt[pp.RHS_EVAL].iter))
        print("   |rhs/call \t %.4Es flops = %.4E"%(t1, flops_rhs))
        
        t1        = profile_tt[pp.JAC_EVAL].seconds/profile_tt[pp.JAC_EVAL].iter
        flops_jac = jac_flops(n, p) / t1
        
        print("   |jacobian \t %.4Es # iter = %d"%(profile_tt[pp.JAC_EVAL].seconds, profile_tt[pp.JAC_EVAL].iter))
        print("   |jacobian/call \t %.4Es flops = %.4E"%(t1, flops_jac))
        
        t1               = profile_tt[pp.JAC_LA_SOL].seconds/profile_tt[pp.JAC_LA_SOL].iter
        flops_jac_solve  = p*((n-1)**3 + 2 * (n-1)**2)/t1
        print("   |jacobian solve \t %.4Es # iter = %d"%(profile_tt[pp.JAC_LA_SOL].seconds, profile_tt[pp.JAC_LA_SOL].iter))
        print("   |jacobian solve /call \t %.4Es flops = %.4E"%(t1, flops_jac_solve))
        
        if fname!="":
            with open(fname, "a") as f:
                header = ["Nv", "Nx", "total", "setup", "solve", "rhs_calls", "rhs_per_call", "rhs_per_call_flops/s",  "jac_calls", "jac_per_call", "jac_per_call_flops/s",  "jac_solve_per_call", "jac_solve_per_call_flops/s"]
                data   = [n, p,
                          t_setup + t_solve, t_setup, t_solve, 
                          profile_tt[pp.RHS_EVAL].iter,
                          t_rhs/profile_tt[pp.RHS_EVAL].iter,
                          flops_rhs,
                          profile_tt[pp.JAC_EVAL].iter,
                          t_jac/profile_tt[pp.JAC_EVAL].iter,
                          flops_jac,
                          profile_tt[pp.JAC_LA_SOL].seconds/profile_tt[pp.JAC_EVAL].iter,
                          flops_jac_solve
                          ]
                data_str= ["%.4E"%d for d in data]
                f.write(",".join(header)+"\n")
                f.write(",".join(data_str)+"\n")
                f.close()
            
                
                
        
        
        
        
