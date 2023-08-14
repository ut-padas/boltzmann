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
                
                self._op_mass_mat[idx] = spec_sp.compute_mass_matrix()
                mmat_inv               = spec_sp.inverse_mass_mat(Mmat = self._op_mass_mat[idx])
                gx, gw                 = spec_sp._basis_p.Gauss_Pn(spec_sp._num_q_radial)
                sigma_m                = np.zeros(len(gx))
                c_gamma                = np.sqrt(2*collisions.ELECTRON_CHARGE_MASS_RATIO)
                gx_ev                  = (gx * vth / c_gamma)**2

                FOp                    = 0
                sigma_m                = 0
                FOp_g                  = 0
        
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
    
    def host_to_device_setup(self, *args):
        
        dev_id = args[0]
        # ib     = args[1]
        # ie     = args[2]
        
        op_list  = [self._op_advection, self._op_col_en, self._op_col_gT, self._op_col_ee,  self._op_qmat,  self._op_mass]
        
        with cp.cuda.Device(dev_id):
            for idx in range(self._par_nvgrids):
                for op in op_list :
                    op[idx]=cp.asarray(op[idx])
            
        return
    
    def device_to_host_setup(self, *args):
        dev_id = args[0]
        # ib     = args[1]
        # ie     = args[2]
        op_list  = [self._op_advection, self._op_col_en, self._op_col_gT, self._op_col_ee,  self._op_qmat,  self._op_mass]
        with cp.cuda.Device(dev_id):
            for idx in range(self._par_nvgrids):
                for op in op_list :
                    op[idx]=cp.asnumpy(op[idx])
            
        return
    
    def initialize(self, grid_idx, n_pts, init_type = "maxwellian"):
        """
        Initialize the grid to Maxwell-Boltzmann distribution
        """
        mmat        = self._op_mass_mat[grid_idx]
        mmat_inv    = spec_sp.inverse_mass_mat(Mmat = mmat)
        spec_sp     = self._op_spec_sp[grid_idx]
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
        print("--initial data mass = %.8E temp (eV) = %.8E"%(m0, t0))
        f0 = np.zeros((len(h_init), n_pts))
        
        for i in range(n_pts):
            f0[:,i] = h_init
        
        return f0
    
    def steady_state_solve(self, f0 : np.array, n0 : np.array, ne : np.array, ni : np.array, ef : np.array, Tg : np.array, grid_idx: int, rtol, atol, max_iter):
        
        n_pts        = n0.shape[0]
        
        eps_0        = scipy.constants.epsilon_0
        me           = scipy.constants.electron_mass
        qe           = scipy.constants.e
        
        f0           = cp.asarray(f0)
        n0           = cp.asarray(n0)
        ne           = cp.asarray(ne)
        ef           = cp.asarray(ef)
        Tg           = cp.asarray(Tg)
        
        vth          = self._par_vth[grid_idx]
        mw           = bte_utils.get_maxwellian_3d(vth, 1)
        Mop          = self._op_mass[grid_idx].reshape((1,-1))
        Top          = self._op_temp[grid_idx].reshape((1,-1)) * collisions.TEMP_K_1EV

        Qmat         = self._op_qmat[grid_idx]
        Rmat         = self._op_rmat[grid_idx]
        QTmat        = cp.transpose(self._op_qmat[grid_idx])
        
        c_en         = self._op_col_en[grid_idx]
        c_gT         = self._op_col_gT[grid_idx]
        c_ee         = self._op_col_ee[grid_idx]
        adv_mat      = self._op_advection[grid_idx]
        qA           = self._op_diag_dg[grid_idx]
        
        cc_op_l1     = c_ee
        cc_op_l2     = cp.swapaxes(c_ee,1,2)
        
        mm_op        = self._op_mass[grid_idx] * mw(0) * vth**3
        u            = mm_op
        u            = cp.dot(cp.transpose(mm_op),qA)
        
        Wmat         = cp.dot(u, c_en)
        
        
        def gamma_a(fb):
            m0           = mw(0) *  cp.dot(Mop, fb) * vth**3 
            kT           = mw(0) * (cp.dot(Top, fb) / m0) * vth**5 * scipy.constants.Boltzmann 
            kT           = cp.abs(kT).reshape((-1,))
        
            c_lambda     = ((12 * np.pi * (eps_0 * kT)**(1.5))/(qe**3 * np.sqrt(ne)))
            gamma_a      = (np.log(c_lambda) * (qe**4)) / (4 * np.pi * (eps_0 * me)**2) / (vth)**3
            #print("mass=%.8E\t Coulomb logarithm %.8E \t gamma_a %.8E \t gamma_a * ne %.8E  \t kT=%.8E temp(ev)=%.8E temp (K)=%.8E " %(m0, np.log(c_lambda) , gamma_a, n0 * ion_deg * gamma_a, kT, kT/scipy.constants.electron_volt, kT/scipy.constants.Boltzmann))
            return gamma_a
        
        def res_func(x):
            Cen_p_Emat_x  = n0 * cp.dot(c_en,x) + n0 * Tg * cp.dot(c_gT, x) + ef * cp.dot(adv_mat, x)
            y             = cp.dot(QTmat, Cen_p_Emat_x)   - n0 * cp.dot(Wmat, x) *  cp.dot(QTmat, x)
            return y
        
        def jac_func(x):
            Lmat = cp.outer(cp.dot(QTmat, cp.dot(c_en, Qmat)), n0) + cp.outer(cp.dot(QTmat, cp.dot(c_gT, Qmat)), Tg * n0) + cp.outer(cp.dot(QTmat, cp.dot(adv_mat, Qmat)), ef) - cp.outer(cp.eye(QTmat.shape[0]), n0 * cp.dot(Wmat, x))
            return Lmat 
        
        
        abs_error       = 1.0
        rel_error       = 1.0 
        iteration_steps = 0        

        fb_prev  = cp.dot(Rmat, f0)
        f1p      = u / np.dot(u, u)
        h_prev   = f1p + np.dot(Qmat,fb_prev)

        while ((rel_error> rtol and abs_error > atol) and iteration_steps < max_iter):
            Lmat      =   jac_func(h_prev)
            rhs_vec   =  -res_func(h_prev)
            abs_error =  cp.linalg.norm(rhs_vec, axis=0)
                        
            p         = cp.linalg.lstsq(Lmat, rhs_vec, rcond=1e-16 /np.linalg.cond(Lmat))[0]
            p         = cp.dot(Qmat,p)

            alpha  = 1e0
            is_diverged = False

            while (cp.linalg.norm(res_func(h_prev + alpha * p), axis=0)  >  abs_error):
                alpha*=0.5
                if alpha < 1e-30:
                    is_diverged = True
                    break
            
            if(is_diverged):
                print("Iteration ", iteration_steps, ": Residual =", abs_error, "line search step size becomes too small")
                break
            
            h_curr      = h_prev + alpha * p
            
            if iteration_steps % 10 == 0:
                rel_error = cp.linalg.norm(h_prev-h_curr, axis=0)/cp.linalg.norm(h_curr, axis=0)
                print("Iteration ", iteration_steps, ": abs residual = %.8E rel residual=%.8E mass =%.8E"%(abs_error, rel_error, np.dot(u, h_prev)))
            
            #fb_prev      = np.dot(Rmat,h_curr)
            h_prev       = h_curr #f1p + np.dot(Qmat,fb_prev)
            iteration_steps+=1

        print("Nonlinear solver (1) atol=%.8E , rtol=%.8E"%(abs_error, rel_error))
        h_curr = cp.dot(qA, h_curr)
        h_curr = cp.asnumpy(h_curr)    
        
        print(h_curr)        
        return    
            
            
        
    
    


