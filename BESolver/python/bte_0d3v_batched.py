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
import cupyx



class bte_0d3v_batched():
    
    def __init__(self, args, Te: np.array, nr: np.array, lm: list, n_vspace_grids : int, collision_model: list) -> None:
        
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
        
        
        self._par_dof        = np.array([(self._par_nr[i]+1) * len(lm) for i in range(self._par_nvgrids)] , dtype=np.int32)
        
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
        
        
        def assemble_operators(thread_id):
            
            ib        = (thread_id * self._par_nvgrids ) // args.threads
            ie        = ((thread_id + 1)* self._par_nvgrids) // args.threads
            
            
            for idx in range(ib,ie):
                vth                    = self._par_vth[idx]
                maxwellian             = bte_utils.get_maxwellian_3d(vth, 1.0)
                
                dg_nodes               = np.sqrt(np.array(sig_pts)) * self._c_gamma / vth
                ev_range               = ((0 * vth /self._c_gamma)**2, (6 * vth /self._c_gamma)**2)
                k_domain               = (np.sqrt(ev_range[0]) * self._c_gamma / vth, np.sqrt(ev_range[1]) * self._c_gamma / vth)
                use_ee                 = 0
                use_dg                 = 1
                
                if use_ee==1:
                    use_dg=0
                else:
                    use_dg=1
                
                bb                     = basis.BSpline(k_domain, self._args.sp_order, self._par_nr[idx] + 1, sig_pts=dg_nodes, knots_vec=None, dg_splines=use_dg, verbose = args.verbose)
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
        
                for col_idx, col in enumerate(collision_model):
                    g = self._coll_list[col_idx]
                    g.reset_scattering_direction_sp_mat()
                    assert col == g._col_name, "[error]: collision model inconsistency !!!"
                    if args.verbose==1:
                        print("collision %d included %s"%(col_idx, col))

                    if "g0NoLoss" == col:
                        FOp       = FOp + collision_op.assemble_mat(g, maxwellian, vth, tgK=0.0)
                        FOp_g     = collision_op.electron_gas_temperature(g, maxwellian, vth)
                        sigma_m  += g.total_cross_section(gx_ev)
                    elif "g0ConstNoLoss" == col:
                        FOp       = FOp + collision_op.assemble_mat(g, maxwellian, vth, tgK=0.0)
                        FOp_g     = collision_op.electron_gas_temperature(g, maxwellian, vth)
                        sigma_m  += g.total_cross_section(gx_ev)
                    elif "g0" in col:
                        FOp       = FOp + collision_op.assemble_mat(g, maxwellian, vth, tgK=0.0)
                        FOp_g     = collision_op.electron_gas_temperature(g, maxwellian, vth)
                        sigma_m  += g.total_cross_section(gx_ev)
                    elif "g1" in col:
                        FOp       = FOp + collision_op.assemble_mat(g, maxwellian, vth, tgK=0.0)
                        sigma_m  += g.total_cross_section(gx_ev)
                    elif "g2" in col:
                        FOp       = FOp + collision_op.assemble_mat(g, maxwellian, vth, tgK=0.0)
                        sigma_m  += g.total_cross_section(gx_ev)
                    else:
                        print("%s unknown collision"%(col))
                        sys.exit(0)

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
                
                if use_dg == 1 : 
                    adv_mat, eA, qA = spec_sp.compute_advection_matix_dg(advection_dir=-1.0)
                    qA              = np.kron(np.eye(spec_sp.get_num_radial_domains()), np.kron(np.eye(num_p), qA))
                else:
                    # cg advection
                    adv_mat         = spec_sp.compute_advection_matix()
                    qA              = np.eye(adv_mat.shape[0])
                    
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
                    hl_op, gl_op         = collision_op.compute_rosenbluth_potentials_op(maxwellian, vth, 1, mmat_inv)
                    cc_op_a, cc_op_b     = collision_op.coulomb_collision_op_assembly(maxwellian, vth)
                    
                    cc_op                = np.dot(cc_op_a, hl_op) + np.dot(cc_op_b, gl_op)
                    cc_op                = np.dot(cc_op,qA)
                    cc_op                = np.dot(np.swapaxes(cc_op,1,2),qA)
                    cc_op                = np.swapaxes(cc_op,1,2)
                    cc_op                = np.dot(np.transpose(qA), cc_op.reshape((num_p*num_sh,-1))).reshape((num_p * num_sh, num_p * num_sh, num_p * num_sh))
                    cc_op                = np.dot(mmat_inv, cc_op.reshape((num_p*num_sh,-1))).reshape((num_p * num_sh, num_p * num_sh, num_p * num_sh))
                    
                    self._op_col_ee[idx] = cc_op
                
                
            return
        
        pool = WorkerPool(self._args.threads)    
        pool.map(assemble_operators, [i for i in range(self._args.threads)])
        pool.close()
        pool.join()
        
        return
    
    def initialize(self, grid_idx, n_pts, init_type = "maxwellian"):
        """
        Initialize the grid to Maxwell-Boltzmann distribution
        """
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
        
        return f0
    
    def set_boltzmann_parameters(self, grid_idx: int, n0 : np.array, ne : np.array, ni : np.array, ef : np.array, Tg : np.array):
        self._par_bte_params[grid_idx] = {"n0": n0, "ne": ne, "ni": ne, "ef" : ef, "Tg": Tg}
        
    def host_to_device_setup(self, *args):
        dev_id = args[0]
        with cp.cuda.Device(dev_id):
            for idx in range(self._par_nvgrids):
                self._op_mass_mat[idx]  = cp.asarray(self._op_mass_mat[idx])
                self._op_advection[idx] = cp.asarray(self._op_advection[idx])
                self._op_col_en[idx]    = cp.asarray(self._op_col_en[idx])
                self._op_col_gT[idx]    = cp.asarray(self._op_col_gT[idx])
                #self._op_col_ee[idx]   = cp.asarray(self._op_col_ee[idx])
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
                
        return
    
    def device_to_host_setup(self, *args):
        dev_id = args[0]

        with cp.cuda.Device(dev_id):
            for idx in range(self._par_nvgrids):
                self._op_mass_mat[idx]  = cp.asnumpy(self._op_mass_mat[idx])
                self._op_advection[idx] = cp.asnumpy(self._op_advection[idx])
                self._op_col_en[idx]    = cp.asnumpy(self._op_col_en[idx])
                self._op_col_gT[idx]    = cp.asnumpy(self._op_col_gT[idx])
                
                #self._op_col_ee[idx]   = cp.asnumpy(self._op_col_ee[idx])
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
        return
    
    def steady_state_solve(self,grid_idx : int, f0 : np.array, rtol, atol, max_iter):
        
        n_pts        = f0.shape[1]
        
        eps_0        = scipy.constants.epsilon_0
        me           = scipy.constants.electron_mass
        qe           = scipy.constants.e
        
        xp           = cp.get_array_module(f0)
        
        n0           = self._par_bte_params[grid_idx]["n0"]
        ne           = self._par_bte_params[grid_idx]["ne"]
        ni           = self._par_bte_params[grid_idx]["ni"]
        Tg           = self._par_bte_params[grid_idx]["Tg"]
        ef           = self._par_bte_params[grid_idx]["ef"]
        
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
        
        #cc_op_l1     = c_ee
        #cc_op_l2     = xp.swapaxes(c_ee,1,2)
        
        mm_op        = self._op_mass[grid_idx] * mw(0) * vth**3
        u            = mm_op
        u            = xp.dot(xp.transpose(mm_op),qA)
        
        Wmat         = xp.dot(u, c_en)
        
        
        def gamma_a(fb):
            m0           = mw(0) *  xp.dot(Mop, fb) * vth**3 
            kT           = mw(0) * (xp.dot(Top, fb) / m0) * vth**5 * scipy.constants.Boltzmann 
            kT           = xp.abs(kT).reshape((-1,))
        
            c_lambda     = ((12 * np.pi * (eps_0 * kT)**(1.5))/(qe**3 * xp.sqrt(ne)))
            gamma_a      = (xp.log(c_lambda) * (qe**4)) / (4 * np.pi * (eps_0 * me)**2) / (vth)**3
            #print("mass=%.8E\t Coulomb logarithm %.8E \t gamma_a %.8E \t gamma_a * ne %.8E  \t kT=%.8E temp(ev)=%.8E temp (K)=%.8E " %(m0, np.log(c_lambda) , gamma_a, n0 * ion_deg * gamma_a, kT, kT/scipy.constants.electron_volt, kT/scipy.constants.Boltzmann))
            return gamma_a
        
        def res_func(x):
            Cen_p_Emat_x  = n0 * (xp.dot(c_en,x) +  Tg * xp.dot(c_gT, x))  + ef * xp.dot(adv_mat, x)
            y             = xp.dot(QTmat, Cen_p_Emat_x)   - n0 * xp.dot(Wmat, x) *  xp.dot(QTmat, x)
            return y
        
        def jac_func(x):
            Lmat = xp.outer(xp.dot(QTmat, xp.dot(c_en, Qmat)), n0) + xp.outer(xp.dot(QTmat, xp.dot(c_gT, Qmat)), Tg * n0) + xp.outer(xp.dot(QTmat, xp.dot(adv_mat, Qmat)), ef) - xp.outer(xp.eye(QTmat.shape[0]), n0 * xp.dot(Wmat, x))
            return Lmat.reshape((QTmat.shape[0], QTmat.shape[0] , n_pts)) 
        
        
        abs_error       = np.ones(n_pts)
        rel_error       = np.ones(n_pts) 
        iteration_steps = 0        

        fb_prev  = xp.dot(Rmat, f0)
        f1       = u / xp.dot(u, u)
        f1p      = xp.zeros((len(f1), n_pts))
        
        for ii in range(n_pts):
            f1p[:,ii] = f1
            
        h_prev   = f1p + xp.dot(Qmat,fb_prev)
        
        pp_mat   = xp.zeros((QTmat.shape[0], n_pts))        
        
        while ((abs_error > atol).any() and (rel_error > rtol).any() and iteration_steps < max_iter):
            Lmat      =   jac_func(h_prev)
            rhs_vec   =  -res_func(h_prev)
            abs_error =  xp.linalg.norm(rhs_vec, axis=0)
            
            for ii in range(n_pts):
                #pp_mat[:,ii]  = xp.linalg.lstsq(Lmat[:,:,ii], rhs_vec[:,ii], rcond=1e-16 /xp.linalg.cond(Lmat[:,:,ii]))[0]
                #pp_mat[:,ii]  = xp.linalg.lstsq(Lmat[:,:,ii], rhs_vec[:,ii], rcond=1e-40)[0]
                pp_mat[:,ii]   = xp.linalg.solve(Lmat[:,:,ii], rhs_vec[:,ii])
                
            p         = xp.dot(Qmat,pp_mat)
            #print(p)
            #print(p.shape)

            alpha  = 1e0
            is_diverged = False

            while ((xp.linalg.norm(res_func(h_prev + alpha * p), axis=0)  >  abs_error).any()):
                alpha*=0.5
                if alpha < 1e-30:
                    is_diverged = True
                    break
            
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
        qoi    = self.compute_QoIs(grid_idx, h_curr)
        return h_curr, qoi
            
    def compute_QoIs(self, grid_idx,  ff, effective_mobility=True):
        args     = self._args
        spec_sp  = self._op_spec_sp[grid_idx]
        vth      = self._par_vth[grid_idx]
        mw       = bte_utils.get_maxwellian_3d(vth, 1)
        
        n0       = self._par_bte_params[grid_idx]["n0"]
        ne       = self._par_bte_params[grid_idx]["ne"]
        ni       = self._par_bte_params[grid_idx]["ni"]
        Tg       = self._par_bte_params[grid_idx]["Tg"]
        ef       = self._par_bte_params[grid_idx]["ef"]
        
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
    


