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
from multiprocessing.pool import ThreadPool as WorkerPool

import cupy as cp
import cupyx.scipy.sparse
import enum
from os import environ
from profile_t import profile_t

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

profile_tt  = [None] * int(pp.LAST)
profile_nn  = ["all","setup", "e-n c_op", "e-e c_op", "adv op", "initialize", "solve", "rhs", "jac", "jac_solve", "H2D", "D2H", "last"]
for i in range(pp.LAST):
    profile_tt[i] = profile_t(profile_nn[i])


def newton_solver_batched(x, n_pts, residual, jacobian, atol, rtol, iter_max, xp=np):
    jac      = jacobian(x)
    assert jac.shape[0] == n_pts
    jac_inv  = xp.linalg.inv(jac)
  
    ns_info  = dict()
    alpha    = xp.ones(n_pts)
    
    while((alpha > 1e-10).any()):
        count     = 0
        r0        = residual(x)
        norm_rr   = norm_r0 = xp.linalg.norm(r0, axis=0)
        converged = ((norm_rr/norm_r0 < rtol).all() or (norm_rr < atol).all())
        
        while( not converged and (count < iter_max) ):
            rr        = residual(x)
            norm_rr   = xp.linalg.norm(rr, axis=0)
            converged = ((norm_rr/norm_r0 < rtol).all() or (norm_rr < atol).all())
            
            x         = x + alpha * xp.einsum("ijk,ki->ji", jac_inv, -rr)
            count    += 1
            
        if (not converged):
            alpha *= 0.25
        else:
            #print("  Newton iter {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(count, norm_rr, norm_rr/norm_r0))
            break
    
    if (not converged):
        # solver failed !!!
        print("  {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(count, xp.max(norm_rr), xp.max(norm_rr/norm_r0)))
        print("non-linear solver step FAILED!!! try with smaller time step size or increase max iterations")
        #print(rr)
        print(alpha)
        print(norm_r0)
        print(norm_rr)
        print(norm_rr/norm_r0)
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
    
    def __init__(self, args, Te: np.array, nr: np.array, lm: list, n_vspace_grids : int, collision_model: list) -> None:
        
        #set_os_envthreads(1)
        
        profile_tt[pp.SETUP].start()
        
        self._par_ap_Te      = Te
        self._par_nr         = nr
        self._par_lm         = lm
        self._par_nvgrids    = n_vspace_grids
        self._args           = args # common arguments for all cases
        
        self._c_gamma        = np.sqrt(2*collisions.ELECTRON_CHARGE_MASS_RATIO)
        
        self._par_bte_params  = [None] * self._par_nvgrids
        self._par_ev_range    = [None] * self._par_nvgrids
        
        self._op_col_en       = [None] * self._par_nvgrids
        self._op_col_gT       = [None] * self._par_nvgrids
        self._op_col_ee       = [None] * self._par_nvgrids
        self._op_advection    = [None] * self._par_nvgrids
        self._op_mass_mat     = [None] * self._par_nvgrids
        
        self._op_qmat         = [None] * self._par_nvgrids 
        self._op_rmat         = [None] * self._par_nvgrids 
        self._op_diag_dg      = [None] * self._par_nvgrids 
        self._op_sigma_m      = [None] * self._par_nvgrids
        
        self._op_mass         = [None] * self._par_nvgrids
        self._op_temp         = [None] * self._par_nvgrids
        self._op_diffusion    = [None] * self._par_nvgrids
        self._op_mobility     = [None] * self._par_nvgrids
        self._op_rate         = [None] * self._par_nvgrids
        self._op_spec_sp      = [None] * self._par_nvgrids
        
        self.xp_module        = np
        self._par_dof         = np.array([(self._par_nr[i]+1) * len(lm) for i in range(self._par_nvgrids)] , dtype=np.int32)
        
        profile_tt[pp.C_CS_SETUP].start()
        
        self._coll_list      = list()
        for col_idx, col in enumerate(collision_model):
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

            self._coll_list.append(g)
            
        self._par_vth        = collisions.electron_thermal_velocity(self._par_ap_Te)
        
        sig_pts   =  list()
        for col_idx, col in enumerate(self._coll_list):
            g  = self._coll_list[col_idx]
            if g._reaction_threshold >0:
                sig_pts.append(g._reaction_threshold)
        
        profile_tt[pp.C_CS_SETUP].stop()
        
        def assemble_operators(thread_id):
            
            ib        = 0#(thread_id * self._par_nvgrids ) // args.threads
            ie        = 1#((thread_id + 1)* self._par_nvgrids) // args.threads
            
            
            for idx in range(ib,ie):
                vth                    = self._par_vth[idx]
                maxwellian             = bte_utils.get_maxwellian_3d(vth, 1.0)
                
                dg_nodes               = np.sqrt(np.array(sig_pts)) * self._c_gamma / vth
                ev_range               = (0, self._args.ev_max) #((0 * vth /self._c_gamma)**2, (6 * vth /self._c_gamma)**2)
                k_domain               = (np.sqrt(ev_range[0]) * self._c_gamma / vth, np.sqrt(ev_range[1]) * self._c_gamma / vth)
                use_ee                 = args.ee_collisions
                use_dg                 = 1
                
                if use_ee==1:
                    use_dg=0
                else:
                    use_dg=0
                
                print("grid idx: ", idx, " ev=", ev_range, " v/vth=",k_domain)
                
                bb                     = basis.BSpline(k_domain, self._args.sp_order, self._par_nr[idx] + 1, sig_pts=dg_nodes, knots_vec=None, dg_splines=use_dg, verbose = args.verbose, extend_domain=True)
                spec_sp                = sp.SpectralExpansionSpherical(self._par_nr[idx], bb, self._par_lm)
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

                FOp                     = 0
                sigma_m                 = 0
                FOp_g                   = 0

                profile_tt[pp.C_EN_SETUP].start()
                
                for col_idx, col in enumerate(collision_model):
                    g = self._coll_list[col_idx]
                    g.reset_scattering_direction_sp_mat()
                    assert col == g._col_name, "[error]: collision model inconsistency !!!"
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
                
                profile_tt[pp.C_EN_SETUP].stop()

                self._op_sigma_m[idx] = sigma_m
                
                self._op_mass[idx]      = bte_utils.mass_op(spec_sp, 1)
                self._op_temp[idx]      = bte_utils.temp_op(spec_sp, 1) * 0.5 * scipy.constants.electron_mass * (2/3/scipy.constants.Boltzmann) / collisions.TEMP_K_1EV
                
                self._op_mobility[idx]  = bte_utils.mobility_op(spec_sp, maxwellian, vth)
                self._op_diffusion[idx] = bte_utils.diffusion_op(spec_sp, self._coll_list, maxwellian, vth)
                
                rr_op  = [None] * len(self._coll_list)
                for col_idx, g in enumerate(self._coll_list):
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
                FOp                     = np.matmul(np.transpose(qA), np.matmul(FOp, qA))
                FOp_g                   = np.matmul(np.transpose(qA), np.matmul(FOp_g, qA))
                
                self._op_advection[idx] = np.dot(mmat_inv, adv_mat) * (1 / vth) * collisions.ELECTRON_CHARGE_MASS_RATIO
                self._op_col_en[idx]    = np.dot(mmat_inv, FOp)
                self._op_col_gT[idx]    = np.dot(mmat_inv, FOp_g)
                
                mm_op   = self._op_mass[idx] * maxwellian(0) * vth**3
                u       = mm_op
                u       = np.dot(np.transpose(mm_op),qA)
                p_vec   = u.reshape((u.shape[0], 1)) / np.sqrt(np.dot(u, u))

                Imat    = np.eye(self._op_col_en[idx].shape[0])
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
                
                if(use_ee == 1):
                    print("e-e collision assembly begin")
                    profile_tt[pp.C_EE_SETUP].start()
                    
                    hl_op, gl_op         = collision_op.compute_rosenbluth_potentials_op(maxwellian, vth, 1, mmat_inv, mp_pool_sz=args.threads)
                    cc_op_a, cc_op_b     = collision_op.coulomb_collision_op_assembly(maxwellian, vth, mp_pool_sz=args.threads)
                    
                    xp                   = self.xp_module
                    
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
                    print("e-e collision assembly end")
                    
            return
        
        assemble_operators(0)
        # pool = WorkerPool(self._args.threads)    
        # pool.map(assemble_operators, [i for i in range(self._args.threads)])
        # pool.close()
        # pool.join()
        profile_tt[pp.SETUP].stop()
        return
    
    def initialize(self, grid_idx, n_pts, init_type = "maxwellian"):
        """
        Initialize the grid to Maxwell-Boltzmann distribution
        """
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
            
        profile_tt[pp.INIT_COND].stop()
        return f0
    
    def set_boltzmann_parameters(self, grid_idx: int, n0 : np.array, ne : np.array, ni : np.array, ef:np.array, Tg : np.array, solver_type:str):
        """
        sets the BTE parameters for each v-space grid
        n0        : heavy density in [1/m^3]
        ne        : electron density in [1/m^3]
        ni        : ion density [1/m^3]
        ef        : electric field [V/m]
        ef_period : 
        Tg        : Gas temperature [K]
        """
        
        xp    = self.xp_module
        self._par_bte_params[grid_idx] = {"n0": n0, "ne": ne, "ni": ne, "Tg": Tg, "E":ef}
        
    def host_to_device_setup(self, *args):
        profile_tt[pp.H2D].start()
        dev_id = args[0]
        with cp.cuda.Device(dev_id):
            for idx in range(self._par_nvgrids):
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
                
                for k, v in self._par_bte_params[idx].items():
                    self._par_bte_params[idx][k] = cp.asarray(v)
                
                for col_idx, col in enumerate(self._coll_list):
                    self._op_rate[idx][col_idx] = cp.asarray(self._op_rate[idx][col_idx])
                    
        if self._args.ee_collisions==1:
            with cp.cuda.Device(dev_id):
                for idx in range(self._par_nvgrids):
                    self._op_col_ee[idx]   = cp.asarray(self._op_col_ee[idx])
        
        self.xp_module = cp
        profile_tt[pp.H2D].stop()
        return
    
    def device_to_host_setup(self, *args):
        profile_tt[pp.D2H].start()
        dev_id = args[0]
        with cp.cuda.Device(dev_id):
            for idx in range(self._par_nvgrids):
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
                
                for k, v in self._par_bte_params[idx].items():
                    self._par_bte_params[idx][k] = cp.asnumpy(v)
                    
                for col_idx, col in enumerate(self._coll_list):
                    self._op_rate[idx][col_idx] = cp.asnumpy(self._op_rate[idx][col_idx])
                    
        if self._args.ee_collisions==1:
            with cp.cuda.Device(dev_id):
                for idx in range(self._par_nvgrids):
                    self._op_col_ee[idx]   = cp.asnumpy(self._op_col_ee[idx])

        self.xp_module = np
        profile_tt[pp.D2H].stop()
        return
    
    def get_rhs_and_jacobian(self, grid_idx: int, n_pts:int):
        xp           = self.xp_module
        args         = self._args
        eps_0        = scipy.constants.epsilon_0
        me           = scipy.constants.electron_mass
        qe           = scipy.constants.e
        
        n0           = self._par_bte_params[grid_idx]["n0"]
        ne           = self._par_bte_params[grid_idx]["ne"]
        ni           = self._par_bte_params[grid_idx]["ni"]
        Tg           = self._par_bte_params[grid_idx]["Tg"]
        ef           = self._par_bte_params[grid_idx]["E"]
        
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
        
        Wmat         = xp.dot(u, c_en)
        
        if self._args.Efreq==0.0:
            Etx = lambda t : self._par_bte_params[grid_idx]["E"]  
        else:
            Etx = lambda t : self._par_bte_params[grid_idx]["E"] * xp.sin(2 * xp.pi * t) 
            
        if args.ee_collisions==1:
            cc_op_l1     = c_ee
            cc_op_l2     = xp.swapaxes(c_ee,1,2)
            
            def gamma_a(fb):
                m0           = mw(0) *  xp.dot(Mop, fb) * vth**3 
                kT           = mw(0) * (xp.dot(Top, fb) / m0) * vth**5 * scipy.constants.Boltzmann 
                kT           = xp.abs(kT).reshape((-1,))
            
                c_lambda     = ((12 * np.pi * (eps_0 * kT)**(1.5))/(qe**3 * xp.sqrt(ne)))
                gamma_a      = (xp.log(c_lambda) * (qe**4)) / (4 * np.pi * (eps_0 * me)**2) / (vth)**3
                return gamma_a
            
            QT_Cen     = xp.dot(QTmat, c_en)
            QT_Cgt     = xp.dot(QTmat, c_gT)
            QT_A       = xp.dot(QTmat, adv_mat)
            
            def res_func(x, time, dt):
                if xp == cp:
                    xp.cuda.runtime.deviceSynchronize()
                    
                profile_tt[pp.RHS_EVAL].start()
                E       = Etx(time)
                c_ee_x  = xp.dot(c_ee, x)
                c_ee_xx = xp.einsum("abc,bc->ac", c_ee_x, x)
                
                y1      = n0 * (xp.dot(QT_Cen, x) + Tg * xp.dot(QT_Cgt, x)) + E * xp.dot(QT_A, x) - n0 * xp.dot(Wmat, x) * xp.dot(QTmat, x) + ne * gamma_a(x) * xp.dot(QTmat, c_ee_xx)
                if xp == cp:
                    xp.cuda.runtime.deviceSynchronize()
                profile_tt[pp.RHS_EVAL].stop()
                return y1
            
            QT_Cen_Q     = xp.dot(QTmat, xp.dot(c_en, Qmat))
            QT_Cgt_Q     = xp.dot(QTmat, xp.dot(c_gT, Qmat))
            QT_A_Q       = xp.dot(QTmat, xp.dot(adv_mat, Qmat))
            Imat         = xp.eye(QTmat.shape[0])
            
            Lmat_pre     = xp.einsum("a,bc->abc", n0, QT_Cen_Q) + xp.einsum("a,bc->abc", Tg, QT_Cgt_Q)
            
            def jac_func(x, time, dt):
                if xp==cp:
                    xp.cuda.runtime.deviceSynchronize()
                    
                profile_tt[pp.JAC_EVAL].start()
                E       = Etx(time)
                cc1_x_p_cc2x = ( xp.dot(cc_op_l1, x) + xp.dot(cc_op_l2, x) ) 
                mu           = n0 * xp.dot(Wmat, x)
                cc1_x_p_cc2x = xp.swapaxes(xp.swapaxes(cc1_x_p_cc2x, 0, 2), 1, 2)
                
                ccQ          = xp.einsum("abc,cd->abd", cc1_x_p_cc2x, Qmat)
                QTccQ        = xp.einsum("be,aec->abc", QTmat       , ccQ )
                
                Lmat         = Lmat_pre + xp.einsum("a,bc->abc", E, QT_A_Q) + xp.einsum("a,abc->abc", ne * gamma_a(x), QTccQ)  -xp.einsum("a,bc->abc", mu, Imat)
                
                if xp==cp:
                    xp.cuda.runtime.deviceSynchronize()
                profile_tt[pp.JAC_EVAL].stop()
                return Lmat
                    
        else:
            
            QT_Cen     = xp.dot(QTmat, c_en)
            QT_Cgt     = xp.dot(QTmat, c_gT)
            QT_A       = xp.dot(QTmat, adv_mat)
            
            def res_func(x, time, dt):
                if xp == cp:
                    xp.cuda.runtime.deviceSynchronize()
                    
                profile_tt[pp.RHS_EVAL].start()
                E       = Etx(time)
                
                y1      = n0 * (xp.dot(QT_Cen, x) + Tg * xp.dot(QT_Cgt, x)) + E * xp.dot(QT_A, x) - n0 * xp.dot(Wmat, x) * xp.dot(QTmat, x) 
                if xp == cp:
                    xp.cuda.runtime.deviceSynchronize()
                profile_tt[pp.RHS_EVAL].stop()
                return y1
            
            QT_Cen_Q     = xp.dot(QTmat, xp.dot(c_en, Qmat))
            QT_Cgt_Q     = xp.dot(QTmat, xp.dot(c_gT, Qmat))
            QT_A_Q       = xp.dot(QTmat, xp.dot(adv_mat, Qmat))
            Imat         = xp.eye(QTmat.shape[0])
            
            Lmat_pre     = xp.einsum("a,bc->abc", n0, QT_Cen_Q) + xp.einsum("a,bc->abc", Tg, QT_Cgt_Q)
            
            def jac_func(x, time, dt):
                if xp==cp:
                    xp.cuda.runtime.deviceSynchronize()
                    
                profile_tt[pp.JAC_EVAL].start()
                E            = Etx(time)
                mu           = n0 * xp.dot(Wmat, x)
                Lmat         = Lmat_pre + xp.einsum("a,bc->abc", E, QT_A_Q) - xp.einsum("a,bc->abc", mu, Imat)
                
                if xp==cp:
                    xp.cuda.runtime.deviceSynchronize()
                profile_tt[pp.JAC_EVAL].stop()
                return Lmat
            
        return res_func, jac_func
    
    def rhs_and_jac_flops(self):
        args = self._args
        if args.ee_collisions==1:
            def res_flops(n, p):
                m0 = p * (2 * n -1) * 2 + p                                     # ne * gamma_a(x)
                m1 = (n-1) * p * (2 * n -1)                                     # xp.dot(QT_Cen,x)
                m2 = (n-1) * p * (2 * n -1) + (n-1) * p                         # Tg * xp.dot(QT_Cgt, x)
                m3 = (n-1) * p * 2                                              # n0 * ( xp.dot(QT_Cen,x) + Tg * xp.dot(QT_Cgt, x) )
                m4 = (n-1) * p * (2 * n -1) + (n-1) * p                         # ef * xp.dot(QT_A,x)
                m5 = p * (2 * n -1) + p + (n-1) * p * (2 * n - 1)  + (n-1) * p  # n0 * xp.dot(Wmat, x) *  xp.dot(QTmat, x)
                m6 = n * n * p * (2 * n -1)                                     # c_ee_x  = xp.dot(c_ee, x) 
                m7 = p * (n * (2 * n -1) + (n-1) * (2 * n - 1)  + (n-1))        # ga[ii] * xp.dot(QTmat, xp.dot(c_ee_x[:,:,ii], x[:,ii]))
                return m0 + m1 + m2 + m3 + m4 + m5 + m6 + m7
            
            def jac_flops(n,p):
                m0 = p * (2 * n -1) * 2 + p # ne * gamma_a(x)
                m1 = n * n * p * (2 * n - 1) * 2 + n * n * p #cc1_x_p_cc2x = xp.dot(cc_op_l1, x) + xp.dot(cc_op_l2, x)
                m2 = p * (2 *n -1) + p # mu           = n0 * xp.dot(Wmat, x)
                m3 = p * ((n-1) * (n-1) * 6)  # n0[ii] * (QT_Cen_Q + Tg[ii] * QT_Cgt_Q) + ef[ii] * QT_A_Q +
                m4 = p * ( 2 * n * (n-1) * (2 * n -1)  + (n-1) * (n-1))  # ga[ii] * xp.dot(QTmat, xp.dot(cc1_x_p_cc2x[:,:,ii], Qmat))
                m5 = p * (n-1) * (n-1) # Imat * mu[ii]
                return m0 + m1 + m2 + m3 + m4 + m5
        else:
            def res_flops(n, p):
                m1 = (n-1) * p * (2 * n -1) # xp.dot(QT_Cen,x)
                m2 = (n-1) * p * (2 * n -1) + (n-1) * p # Tg * xp.dot(QT_Cgt, x)
                m3 = (n-1) * p # n0 * ( xp.dot(QT_Cen,x) + Tg * xp.dot(QT_Cgt, x) )
                m4 = (n-1) * p * (2 * n -1) + (n-1) * p   # ef * xp.dot(QT_A,x)
                m5 = p * (2 * n -1) + p + (n-1) * p * (2 * n - 1) + (n-1) * p # n0 * xp.dot(Wmat, x) *  xp.dot(QTmat, x)
                return m1 + m2 + m3 + m4 + m5
            
            def jac_flops(n,p):
                m2 = p * (2 *n -1) + p # mu           = n0 * xp.dot(Wmat, x)
                m3 = p * ((n-1) * (n-1) * 6)  # n0[ii] * (QT_Cen_Q + Tg[ii] * QT_Cgt_Q) + ef[ii] * QT_A_Q +
                m5 = p * (n-1) * (n-1) # Imat * mu[ii]
                return m2 + m3 + m5
        
        return res_flops, jac_flops
            
    def steady_state_solve(self, grid_idx : int, f0 : np.array, atol, rtol, max_iter):
        xp           = self.xp_module
        
        if xp==cp:
            xp.cuda.runtime.deviceSynchronize()
        
        profile_tt[pp.SOLVE].start()
        
        n_pts        = f0.shape[1]
        vth          = self._par_vth[grid_idx]
        mw           = bte_utils.get_maxwellian_3d(vth, 1)
        
        c_en         = self._op_col_en[grid_idx]
        qA           = self._op_diag_dg[grid_idx]
        mm_op        = self._op_mass[grid_idx] * mw(0) * vth**3
        u            = mm_op
        u            = xp.dot(xp.transpose(mm_op),qA)
        Wmat         = xp.dot(u, c_en)
        
        Qmat         = self._op_qmat[grid_idx]
        Rmat         = self._op_rmat[grid_idx]
        QTmat        = xp.transpose(self._op_qmat[grid_idx])
        
        res_func, jac_func = self.get_rhs_and_jacobian(grid_idx, n_pts)
        
        abs_error       = np.ones(n_pts)
        rel_error       = np.ones(n_pts) 
        iteration_steps = 0        

        fb_prev  = xp.dot(Rmat, f0)
        f1       = u / xp.dot(u, u)
        f1p      = xp.zeros((len(f1), n_pts))
        
        for ii in range(n_pts):
            f1p[:,ii] = f1
            
        h_prev   = f1p + xp.dot(Qmat,fb_prev)
        
        while ((abs_error > atol).any() and (rel_error > rtol).any() and iteration_steps < max_iter):
            Lmat      =  jac_func(h_prev, 0, 0)
            rhs_vec   =  res_func(h_prev, 0, 0)
            abs_error =  xp.linalg.norm(rhs_vec, axis=0)
            Lmat_inv  =  xp.linalg.inv(Lmat)
            
            pp_mat    = xp.einsum("abc,ca->ba",Lmat_inv, -rhs_vec)
            p         = xp.dot(Qmat,pp_mat)

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
                print("Iteration ", iteration_steps, ": Residual =", abs_error, "line search step size becomes too small")
                break
            
            h_curr      = h_prev + alpha * p
            
            if iteration_steps % 10 == 0:
                rel_error = xp.linalg.norm(h_prev-h_curr, axis=0)/xp.linalg.norm(h_curr, axis=0)
                print("Iteration ", iteration_steps, ": abs residual = %.8E rel residual=%.8E mass =%.8E"%(xp.max(abs_error), xp.max(rel_error), xp.max(xp.dot(u, h_prev))))
            
            #fb_prev      = np.dot(Rmat,h_curr)
            h_prev       = h_curr #f1p + np.dot(Qmat,fb_prev)
            iteration_steps+=1

        print("Nonlinear solver (1) atol=%.8E , rtol=%.8E"%(xp.max(abs_error), xp.max(rel_error)))
        h_curr = xp.dot(qA, h_curr)
        h_curr = self.normalized_distribution(grid_idx, h_curr)
        qoi    = self.compute_QoIs(grid_idx, h_curr, effective_mobility=False)
        profile_tt[pp.SOLVE].stop()
        return h_curr, qoi
    
    def solve(self, grid_idx:int, f0:np.array, atol, rtol, max_iter:int, solver_type:str):
        xp           = self.xp_module
        Qmat         = self._op_qmat[grid_idx]
        Rmat         = self._op_rmat[grid_idx]
        QTmat        = xp.transpose(self._op_qmat[grid_idx])
        qA           = self._op_diag_dg[grid_idx]
        
        if (solver_type == "steady-state"):
            # need to check for fourier modes for the solver. 
            return self.steady_state_solve(grid_idx, f0, atol, rtol, max_iter)
        elif(solver_type=="BE"):
            use_spsolve     = True
            tau             = 1/(self._args.Efreq)
            dt              = self._args.dt * tau
            tT              = self._args.cycles * tau
            rhs , rhs_u     = self.get_rhs_and_jacobian(grid_idx, f0.shape[1])
            
            
            u               = f0
            tt              = 0
            n_pts           = f0.shape[1]
            Imat            = xp.zeros((n_pts, Qmat.shape[1], Qmat.shape[1]))
            tmp             = xp.eye(Qmat.shape[1])
            
            for i in range(n_pts):
                Imat[i,:,:] = tmp[:,:]
            
            a1 = a2 = 1.0
            du              = xp.zeros((Qmat.shape[1],n_pts))
            #print(tt < tT and (a1 > atol or a2 > rtol), tT, tt)
            while(tt < tT and (a1 > atol or a2 > rtol)):
                u0 = u
                for ts_idx in range(int(1/self._args.dt)):
                    def residual(du):
                        return du - dt * rhs(u + xp.dot(Qmat, du), tt + dt, dt)
           
                    def jacobian(du):
                        if use_spsolve==False:
                            return (Imat - dt * rhs_u(u, tt, dt)).toarray()
                        else:
                            return (Imat - dt * rhs_u(u, tt, dt))
                    
                    ns_info = newton_solver_batched(du, n_pts, residual, jacobian, atol, rtol, max_iter, xp)
                    
                    if ns_info["status"]==False:
                        print("At time = %.2E "%(tt/tau), end='')
                        print("non-linear solver step FAILED!!! try with smaller time step size or increase max iterations")
                        print("  Newton iter {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(ns_info["iter"], xp.max(ns_info["atol"]), xp.max(ns_info["rtol"])))
                        return u0

                    du = ns_info["x"]
                    u  = u + xp.dot(Qmat, du)
                    tt += dt
                    
                    print("time = %.2E "%(tt/tau), end='')
                    print("  Newton iter {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(ns_info["iter"], xp.max(ns_info["atol"]), xp.max(ns_info["rtol"])))
                    
                u1=u
                print("time = %.8E "%(tt/tau), end='')
                print("  Newton iter {0:d}: ||res|| = {1:.6e}, ||res||/||res0|| = {2:.6e}".format(ns_info["iter"], xp.max(ns_info["atol"]), xp.max(ns_info["rtol"])))
                a1 = xp.linalg.norm(u1-u0)
                a2 = a1/ xp.linalg.norm(u0)
                print("||u(t+T) - u(t)|| = %.8E and ||u(t+T) - u(t)||/||u(t)|| = %.8E"% (a1, a2))
            
            h_curr = xp.dot(qA, u)
            h_curr = self.normalized_distribution(grid_idx, h_curr)
            qoi    = self.compute_QoIs(grid_idx, h_curr, effective_mobility=False)
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
        ne       = self._par_bte_params[grid_idx]["ne"]
        ni       = self._par_bte_params[grid_idx]["ni"]
        Tg       = self._par_bte_params[grid_idx]["Tg"]
        ef       = self._par_bte_params[grid_idx]["E"]
        
        num_p   = spec_sp._p + 1
        num_sh  = len(spec_sp._sph_harm_lm)
        
        c_gamma   = self._c_gamma
        eavg_to_K = (2/(3*scipy.constants.Boltzmann))

        #print(tgrid/1e-5)
        #print(Ef(tgrid))
        
        xp  = cp.get_array_module(ff) 

        mm  = xp.dot(self._op_mass[grid_idx], ff) * mw(0) * vth**3
        mu  = xp.dot(self._op_temp[grid_idx], ff) * mw(0) * vth**5  / mm

        if effective_mobility:
            M   = xp.dot(self._op_mobility[grid_idx], xp.sqrt(3) * ff[1::num_sh, :])  * (-(c_gamma / (3 * ( ef / n0))))
        else:
            M   = xp.dot(self._op_mobility[grid_idx], xp.sqrt(3) * ff[1::num_sh, :])  * (-(c_gamma / (3 * ( 1 / n0))))
        
        D   = xp.dot(self._op_diffusion[grid_idx],  ff[0::num_sh,:]) * (c_gamma / 3.)
        
        rr  = list()
        for col_idx, g in enumerate(self._coll_list):
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
        for i in range(pp.LAST):
            profile_tt[i].reset()
        
        return
    
    def profile_stats(self):
        args = self._args
        res_flops, jac_flops = self.rhs_and_jac_flops()
        n                    = (args.l_max + 1) * (args.Nr + 1) 
        p                    = args.n_pts
        
        
        print("--setup\t %.4Es"%(profile_tt[pp.SETUP].seconds))
        print("   |electon-X \t %.4Es"%(profile_tt[pp.C_EN_SETUP].seconds))
        print("   |coulombic \t %.4Es"%(profile_tt[pp.C_EE_SETUP].seconds))
        print("   |advection \t %.4Es"%(profile_tt[pp.ADV_SETUP].seconds))
        print("--solve \t %.4Es"%(profile_tt[pp.SOLVE].seconds))
        print("   |rhs \t %.4Es"%(profile_tt[pp.RHS_EVAL].seconds))
        
        t1      = profile_tt[pp.RHS_EVAL].seconds/profile_tt[pp.RHS_EVAL].iter
        flops   = res_flops(n, p) / t1
        print("   |rhs/call \t %.4Es flops = %.4E"%(t1, flops))
        t1      = profile_tt[pp.JAC_EVAL].seconds/profile_tt[pp.JAC_EVAL].iter
        flops   = res_flops(n, p) / t1
        print("   |jacobian \t %.4Es"%(profile_tt[pp.JAC_EVAL].seconds))
        print("   |jacobian/call \t %.4Es flops = %.4E"%(t1, flops))