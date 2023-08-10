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



class bte_0d3v_batched():
    
    def __init__(self, args, ef: np.array, n0: np.array, ni: np.array, Te: np.array, nr: np.array, lm: list, batch_sz : int, collision_model: list) -> None:
        
        self._par_ef         = ef
        self._par_n0         = n0
        self._par_ni         = ni
        self._par_ap_Te      = Te
        self._par_nr         = nr
        self._par_lm         = lm
        self._par_batch_sz   = batch_sz
        self._args           = args # common arguments for all cases
        
        self._c_gamma        = np.sqrt(2*collisions.ELECTRON_CHARGE_MASS_RATIO)
        
        self._op_col_en      = [None] * self._par_batch_sz
        self._op_col_ee      = [None] * self._par_batch_sz
        self._op_advection   = [None] * self._par_batch_sz
        self._op_mass_mat    = [None] * self._par_batch_sz
        
        self._op_qmat        = [None] * self._par_batch_sz 
        self._op_diag_dg     = [None] * self._par_batch_sz 
        self._op_sigma_m     = [None] * self._par_batch_sz
        
        self._op_mass        = [None] * self._par_batch_sz
        self._op_temp        = [None] * self._par_batch_sz
        self._op_diffusion   = [None] * self._par_batch_sz
        self._op_mobility    = [None] * self._par_batch_sz
        self._op_rate        = [None] * self._par_batch_sz
        self._op_spec_sp     = [None] * self._par_batch_sz
        
        
        self._par_dof        = np.array([(self._par_nr[i]+1) * len(lm) for i in range(self._par_batch_sz)] , dtype=np.int32)
        
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
        ion_deg              = self._par_ni / self._par_n0
        
        sig_pts   =  list()
        for col_idx, col in enumerate(self._coll_list):
            g  = self._coll_list[col_idx]
            if g._reaction_threshold >0:
                sig_pts.append(g._reaction_threshold)
        
        print(self._coll_list)
        print(sig_pts)
        
        def assemble_operators(thread_id):
            
            ib        = (thread_id * batch_sz) // args.threads
            ie        = ((thread_id + 1)* batch_sz) // args.threads
            
            
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
        
                for col_idx, col in enumerate(collision_model):
                    g = self._coll_list[col_idx]
                    g.reset_scattering_direction_sp_mat()
                    assert col == g._col_name, "[error]: collision model inconsistency !!!"
                    if args.verbose==1:
                        print("collision %d included %s"%(col_idx, col))

                    if "g0NoLoss" == col:
                        FOp       = FOp + n0[idx] * collision_op.assemble_mat(g, maxwellian, vth, tgK=args.Tg)
                        sigma_m  += g.total_cross_section(gx_ev)
                    elif "g0ConstNoLoss" == col:
                        FOp       = FOp + n0[idx] * collision_op.assemble_mat(g, maxwellian, vth, tgK=args.Tg)
                        sigma_m  += g.total_cross_section(gx_ev)
                    elif "g0" in col:
                        FOp       = FOp + n0[idx] * collision_op.assemble_mat(g, maxwellian, vth, tgK=args.Tg)
                        sigma_m  += g.total_cross_section(gx_ev)
                    elif "g1" in col:
                        FOp       = FOp + n0[idx] * collision_op.assemble_mat(g, maxwellian, vth, tgK=args.Tg)
                        sigma_m  += g.total_cross_section(gx_ev)
                    elif "g2" in col:
                        FOp       = FOp + n0[idx] * collision_op.assemble_mat(g, maxwellian, vth, tgK=args.Tg)
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
                
                self._op_advection[idx] = np.dot(mmat_inv, adv_mat)
                self._op_col_en[idx]    = np.dot(mmat_inv, FOp)
                
                if(use_ee == 1):
                    hl_op, gl_op         = collision_op.compute_rosenbluth_potentials_op(maxwellian, vth, 1, mmat_inv)
                    cc_op_a, cc_op_b     = collision_op.coulomb_collision_op_assembly(maxwellian, vth)
                    
                    cc_op                = np.dot(cc_op_a, hl_op) + np.dot(cc_op_b, gl_op)
                    cc_op                = np.dot(cc_op,qA)
                    cc_op                = np.dot(np.swapaxes(cc_op,1,2),qA)
                    cc_op                = np.swapaxes(cc_op,1,2)
                    cc_op                = np.dot(np.transpose(qA), cc_op.reshape((num_p*num_sh,-1))).reshape((num_p * num_sh, num_p * num_sh, num_p * num_sh))
                    
                    self._op_col_ee[idx] = cc_op
                
                
            return
        
        pool = WorkerPool(self._args.threads)    
        pool.map(assemble_operators, [i for i in range(self._args.threads)])
        pool.close()
        pool.join()
        
        return
    
    def steady_state_solve(self):
        pass
    
    


