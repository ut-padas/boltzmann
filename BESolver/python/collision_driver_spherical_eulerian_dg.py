"""
@package Boltzmann collision operator solver. 
"""

# from ast import While
# from cProfile import run
# from cmath import sqrt
# from dataclasses import replace
# from math import ceil
# import string
import os
os.environ["OMP_NUM_THREADS"] = "8"
import enum
import scipy
import scipy.optimize
import scipy.interpolate
from   sympy import rad
from   maxpoly import maxpolyserieseval
import basis
import spec_spherical as sp
import numpy as np
import collision_operator_spherical as colOpSp
import collisions 
import parameters as params
import os
from   time import perf_counter as time, sleep
import utils as BEUtils
import argparse
import scipy.integrate
from   scipy.integrate import ode
from   advection_operator_spherical_polys import *
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib as mpl
from   scipy.interpolate import interp1d
import sys
import bolsig
import csv
from   datetime import datetime

plt.rcParams.update({
    "text.usetex": False,
    "font.size": 14,
    "ytick.major.size": 3
    #"font.family": "Helvetica",
    #"lines.linewidth":2.0
})

col_names = {"g0":"elastic", "g2": "ionization"}


def deriv_fd(x, y):
    mid = (y[1:len(x)]-y[0:len(x)-1])/(x[1:len(x)]-x[0:len(x)-1])
    d = np.zeros_like(x)
    d[0] = mid[0]
    d[-1] = mid[-1]
    d[1:len(x)-1] = .5*(mid[1:len(x)-1]+mid[0:len(x)-2])
    return d


class CollissionMode(enum.Enum):
    ELASTIC_ONLY=0
    ELASTIC_W_EXCITATION=1
    ELASTIC_W_IONIZATION=2
    ELASTIC_W_EXCITATION_W_IONIZATION=3

def solve_collop_dg(steady_state, collOp : colOpSp.CollisionOpSP, maxwellian, vth, E_field, t_end, dt,t_tol, collisions_included):

    spec_sp = collOp._spec

    Mmat = spec_sp.compute_mass_matrix()
    Minv = spec_sp.inverse_mass_mat(Mmat=Mmat)
    mm_inv_error=np.linalg.norm(np.matmul(Mmat,Minv)-np.eye(Mmat.shape[0]))/np.linalg.norm(np.eye(Mmat.shape[0]))
    print("cond(M) = %.4E"%np.linalg.cond(Mmat))
    print("|I-M M^{-1}| error = %.12E"%(mm_inv_error))
    
    MVTH  = vth
    MNE   = maxwellian(0) * (np.sqrt(np.pi)**3) * (vth**3)
    MTEMP = collisions.electron_temperature(MVTH)
    print("==========================================================================")
    vratio = np.sqrt(1.0/args.basis_scale)

    if (args.radial_poly == "maxwell" or args.radial_poly == "laguerre"):
        hv    = lambda v,vt,vp : np.exp((v**2)*(1.-1./(vratio**2)))/vratio**3
    elif (args.radial_poly == "maxwell_energy"):
        hv    = lambda v,vt,vp : np.exp((v**2)*(1.-1./(vratio**2)))/vratio**3
    elif(args.radial_poly == "chebyshev"):
        hv    = lambda v,vt,vp : np.exp(-((v/vratio)**2)) / vratio**3
    elif (args.radial_poly == "bspline"):
        hv    = lambda v,vt,vp : np.exp(-((v/vratio)**2)) / vratio**3
    else:
        raise NotImplementedError

    h_init    = BEUtils.function_to_basis(spec_sp,hv,maxwellian,None,None,None,Minv=Minv)
    h_t       = np.array(h_init)
    
    ne_t      = MNE
    mw_vth    = BEUtils.get_maxwellian_3d(vth,ne_t)
    m0_t0     = mw_vth(0) * np.dot(mass_op, h_t) * vth**3 
    temp_t0   = mw_vth(0) * np.dot(temp_op, h_t) * vth**5 * 0.5 * scipy.constants.electron_mass * (2./ (3 * collisions.BOLTZMANN_CONST)) / m0_t0 
    vth_curr  = vth 
    
    print("Initial temp (eV) = %.12E "   %(temp_t0 * collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT))
    print("Initial mass (1/m^3) = %.12E" %(m0_t0))


    f0_cf = interp1d(ev, bolsig_f0, kind='cubic', bounds_error=False, fill_value=(bolsig_f0[0],bolsig_f0[-1]))
    fa_cf = interp1d(ev, bolsig_a, kind='cubic', bounds_error=False, fill_value=(bolsig_a[0],bolsig_a[-1]))
    ftestt = lambda v,vt,vp : f0_cf(.5*(v*VTH)**2/collisions.ELECTRON_CHARGE_MASS_RATIO)*(1. + fa_cf(.5*(v*VTH)**2/collisions.ELECTRON_CHARGE_MASS_RATIO)*np.cos(vt))
    
    htest                        =  BEUtils.function_to_basis(spec_sp,ftestt,maxwellian,None,None,None,Minv=Minv)
    radial_projection[-1, :, :]  =  BEUtils.compute_radial_components(ev, spec_sp, htest, maxwellian, VTH, 1)
    scale                        =  1./( np.trapz(radial_projection[-1, 0, :]*np.sqrt(ev),x=ev) )
    htest                        =  htest*scale
    radial_projection[-1, :, :]  *= scale
    coeffs_projection.append(htest)

    # plt.subplot(1,2,1)
    # plt.semilogy(ev, radial_projection[-1, 0 , :])
    # plt.grid()
    # plt.title("f0")

    # plt.subplot(1,2,2)
    # plt.semilogy(ev, radial_projection[-1, 1 , :])
    # plt.title("f1")
    # plt.grid()
    # plt.show()

    # solution_vector = np.zeros((2,h_init.shape[0]))
    # return solution_vector

    gx, gw   = spec_sp._basis_p.Gauss_Pn(spec_sp._num_q_radial)
    sigma_m  = np.zeros(len(gx))
    c_gamma  = np.sqrt(2*collisions.ELECTRON_CHARGE_MASS_RATIO)
    gx_ev    = (gx * vth / c_gamma)**2

    FOp = 0
    t1=time()
    if "g0" in collisions_included:
        g0  = collisions.eAr_G0()
        g0.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g0, mw_vth, vth_curr)
        sigma_m += collisions.Collisions.synthetic_tcs(gx_ev, g0._analytic_cross_section_type)

    if "g0ConstNoLoss" in collisions_included:
        g0noloss  = collisions.eAr_G0_NoEnergyLoss(cross_section="g0Const")
        g0noloss.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g0noloss, mw_vth, vth_curr)
        sigma_m += collisions.Collisions.synthetic_tcs(gx_ev, g0noloss._analytic_cross_section_type)
    
    if "g0NoLoss" in collisions_included:
        g0noloss  = collisions.eAr_G0_NoEnergyLoss()
        g0noloss.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g0noloss, mw_vth, vth_curr)
        sigma_m += collisions.Collisions.synthetic_tcs(gx_ev, g0noloss._analytic_cross_section_type)

    if "g0Const" in collisions_included:
        g0const  = collisions.eAr_G0(cross_section="g0Const")
        g0const.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g0const, mw_vth, vth_curr)
        sigma_m += collisions.Collisions.synthetic_tcs(gx_ev, g0const._analytic_cross_section_type)

    if "g2" in collisions_included:
        g2  = collisions.eAr_G2()
        g2.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g2, mw_vth, vth_curr)
        sigma_m += collisions.Collisions.synthetic_tcs(gx_ev, g2._analytic_cross_section_type)

    if "g2Smooth" in collisions_included:
        g2Smooth  = collisions.eAr_G2(cross_section="g2Smooth")
        g2Smooth.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g2Smooth, mw_vth, vth_curr)
        sigma_m += collisions.Collisions.synthetic_tcs(gx_ev, g2Smooth._analytic_cross_section_type)

    if "g2Regul" in collisions_included:
        g2Regul  = collisions.eAr_G2(cross_section="g2Regul")
        g2Regul.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g2Regul, mw_vth, vth_curr)
        sigma_m += collisions.Collisions.synthetic_tcs(gx_ev, g2Regul._analytic_cross_section_type)

    if "g2Const" in collisions_included:
        g2  = collisions.eAr_G2(cross_section="g2Const")
        g2.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g2, mw_vth, vth_curr)
        sigma_m += collisions.Collisions.synthetic_tcs(gx_ev, g2._analytic_cross_section_type)

    if "g2step" in collisions_included:
        g2  = collisions.eAr_G2(cross_section="g2step")
        g2.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g2, mw_vth, vth_curr)
        sigma_m += collisions.Collisions.synthetic_tcs(gx_ev, g2._analytic_cross_section_type)

    if "g2smoothstep" in collisions_included:
        g2  = collisions.eAr_G2(cross_section="g2smoothstep")
        g2.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g2, mw_vth, vth_curr)
        sigma_m += collisions.Collisions.synthetic_tcs(gx_ev, g2._analytic_cross_section_type)

    t2=time()
    print("Assembled the collision op. for Vth : ", vth_curr)
    print("Collision Operator assembly time (s): ",(t2-t1))

    num_p   = spec_sp._p + 1
    num_sh  = len(spec_sp._sph_harm_lm)

    MVTH  = vth
    MNE   = maxwellian(0) * (np.sqrt(np.pi)**3) * (vth**3)
    MTEMP = collisions.electron_temperature(MVTH)
    
    # advmat, eA, qA = spec_sp.compute_advection_matix_dg(advection_dir=-1.0)
    # #eA     = np.kron(np.eye(spec_sp.get_num_radial_domains()), np.kron(np.eye(num_p), eA))
    # qA     = np.kron(np.eye(spec_sp.get_num_radial_domains()), np.kron(np.eye(num_p), qA))
    
    advmat = spec_sp.compute_advection_matix()
    qA     = np.eye(advmat.shape[0])

    # np.set_printoptions(precision=1)
    # print(FOp[0::num_sh, 0::num_sh])
    # print(FOp[1::num_sh, 1::num_sh])
    
    FOp    = np.matmul(np.transpose(qA), np.matmul(FOp, qA))
    h_init = np.dot(np.transpose(qA),h_init)

    # Cmat = np.matmul(Minv, FOp)
    # Emat = (E_field/MVTH) * collisions.ELECTRON_CHARGE_MASS_RATIO * np.matmul(Minv, advmat)

    Cmat  = FOp 
    Emat  = (E_field/ MVTH) * collisions.ELECTRON_CHARGE_MASS_RATIO *advmat 
    Cmat1 = np.matmul(Minv, Cmat + Emat)
    collOp.setup_coulombic_collisions()

    if steady_state:
        u        = mass_op * mw_vth(0) * vth**3 
        for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
            if lm[0] > 0:
                u[lm_idx::num_sh] = 0.0 
        
        u        = np.matmul(np.transpose(u), qA)
        h_prev   = np.copy(h_init)
        h_prev   = h_prev / (np.dot(u, h_prev))

        def residual_func(x, cc_ee, eval_jacobian = True):
            #Cmat1                         = np.matmul(Minv, Cmat + Emat + cc_ee)
            y                             = np.matmul(Cmat + Emat + cc_ee, x)
            y                             = -np.matmul(Mmat, np.dot(u, np.matmul(Cmat1,x)) * x)   + y
            res                           = np.append(y, (np.dot(u, x)-1))

            if eval_jacobian:
                Ji = -2 * Mmat * np.dot(u, np.matmul(Cmat1,x)) + (Cmat + Emat + cc_ee)
                Ji = np.vstack((Ji , u))
                return res, Ji
            else:
                #print(res)
                return res
        
        
        def solver_0(h_prev, rtol = 1e-8, atol=1e-6, max_iter=1000):
            h_curr          = np.copy(h_prev)
            abs_error       = 1
            rel_error       = 1
            iteration_steps = 0

            def low_rank_approx(A=None, r=1):
                """
                Computes an r-rank approximation of a matrix
                given the component u, s, and v of it's SVD
                Requires: numpy
                """
                u, s, v = np.linalg.svd(A, full_matrices=True)

                # Ar = np.zeros((len(u), len(v)))
                # for i in range(r):
                #     Ar += s[i] * np.outer(u.T[i], v[i])
                
                Ar = np.dot(u[:,0:r] * s[0:r], v[0:r,:])
                return Ar

            nn            = Cmat.shape[0]
            Ji            = np.zeros((nn+1,nn))
            Rf            = np.zeros(nn+1)
            ion_deg       = args.ion_deg
            rcond_cuttoff = 1e-12

            

            while (rel_error> rtol and abs_error > atol and iteration_steps < max_iter):
                if args.ee_collisions:
                    CC_ee           = collisions.AR_NEUTRAL_N * ion_deg * collOp.coulomb_collision_mat(1.0, ion_deg, collisions.AR_NEUTRAL_N, h_prev, maxwellian, vth, sigma_m, full_assembly=True)
                else:
                    CC_ee           = 0

                Rf , Ji     =  residual_func(h_prev, CC_ee) # !!! h_prev or h_curr

                p           = np.matmul(np.linalg.pinv(Ji, rcond=rcond_cuttoff) , -Rf)
                abs_error   = np.linalg.norm(Rf)
                #print("|p|", np.linalg.norm(p))
                #alpha       *= 2 #1e2#1/np.linalg.norm(p)
                alpha  = 1e0
                is_diverged = False

                # while (np.linalg.norm(residual_func(h_prev + alpha * p , CC_ee,  eval_jacobian=False))  <  np.linalg.norm(Rf)):
                #     alpha*=2
                #     if alpha > 1e2:
                #         #is_diverged = True
                #         break

                while (np.linalg.norm(residual_func(h_prev + alpha * p , CC_ee,  eval_jacobian=False))  >  np.linalg.norm(Rf)):
                    alpha*=0.5
                    if alpha < 1e-10:
                        is_diverged = True
                        break

                
                if(is_diverged):
                    print("Iteration ", iteration_steps, ": Residual =", np.linalg.norm(Rf))
                    print("line search step size becomes too small")
                    break
                
                h_curr          = h_prev + p*alpha
                rel_error       = np.linalg.norm(h_prev-h_curr)/np.linalg.norm(h_curr)

                if(iteration_steps%10==0):
                    print("Iteration ", iteration_steps, "step size = %.8E abs error = %.8E rel error = %.8E"%(alpha, abs_error, rel_error))

                
                h_prev          = h_curr

                iteration_steps = iteration_steps + 1
            

            print("Nonlinear solver (0) atol=%.8E , rtol=%.8E"%(abs_error, rel_error))
            return h_curr, abs_error, rel_error

        def solver_1(h_prev, rtol = 1e-5 , atol=1e-5, max_iter=1000):
            h_curr          = np.copy(h_prev)
            
            abs_error       = 1.0
            rel_error       = 1.0 
            iteration_steps = 0

            ion_deg         = args.ion_deg
            Cmat_p_Emat     = Cmat + Emat
            Cmat1           = np.matmul(Minv, Cmat_p_Emat)
            full_coulomb = True

            Lmat      = Cmat_p_Emat 
            Lmat_inv  = np.linalg.pinv(Lmat, rcond=1e-30)

            while (rel_error> rtol and abs_error > atol and iteration_steps < max_iter):
                if args.ee_collisions:
                    CC_ee           = collisions.AR_NEUTRAL_N * ion_deg * collOp.coulomb_collision_mat(1.0, ion_deg, collisions.AR_NEUTRAL_N, h_prev, maxwellian, vth, sigma_m, full_assembly=full_coulomb)

                    Lmat      = Cmat_p_Emat + CC_ee
                    Lmat_inv  = np.linalg.pinv(Lmat, rcond=1e-14/np.linalg.cond(Lmat))
                else:
                    CC_ee           = 0

                abs_error = np.linalg.norm(residual_func(h_prev, CC_ee, eval_jacobian=False))

                #Cmat1       = np.matmul(Minv, Cmat + Emat + CC_ee)
                #h_curr     = np.matmul(Lmat_inv, -np.dot(CC_ee, h_prev) + np.matmul(Mmat, np.dot(u, np.matmul(Cmat1,h_prev)) * h_prev))
                h_curr      = np.matmul(Lmat_inv, np.matmul(Mmat, np.dot(u, np.matmul(Cmat1,h_prev)) * h_prev))
                h_curr      = h_curr / (np.dot(u, h_curr))

                if iteration_steps % 10 == 0:
                    rel_error = np.linalg.norm(h_prev-h_curr)/np.linalg.norm(h_curr)
                    print("Iteration ", iteration_steps, ": abs residual = %.8E rel residual=%.8E"%(abs_error, rel_error))
                

                h_prev      = h_curr
                iteration_steps+=1

            print("Nonlinear solver (1) atol=%.8E , rtol=%.8E"%(abs_error, rel_error))
            return h_curr, abs_error, rel_error

        def solver_2(h_prev, rtol = 1e-8, atol=1e-6, max_iter=1000):
            
            h_curr          = np.copy(h_prev)
            
            abs_error       = 1
            rel_error       = 1
            iteration_steps = 0

            nn       = Cmat.shape[0]
            ion_deg  = args.ion_deg

            C_00      = Cmat[0::num_sh, 0::num_sh]
            C_11      = Cmat[1::num_sh, 1::num_sh]

            E_01      = Emat[0::num_sh, 1::num_sh]
            E_10      = Emat[1::num_sh, 0::num_sh]

            Cmat_p_Emat  = Cmat + Emat
            Cmat1        = np.matmul(Minv, Cmat + Emat)
            WW           = lambda ff : np.matmul(Mmat, np.dot(u, np.matmul(Cmat1,ff)) * ff)
            
            full_coulomb = True
            

            ll = 10
            ion_deg_r = ion_deg /(1<<ll)

            for l in range(ll+1):
                ne_by_n0 = (1<<l) * ion_deg_r
                if l < ll:
                    max_iter=100
                else:
                    max_iter=500

                iteration_steps = 0
                print("solve for ion_deg = %.8E"%(ne_by_n0))
                while (rel_error> rtol and abs_error > atol and iteration_steps < max_iter):
                    if args.ee_collisions:
                        CC_ee           = collisions.AR_NEUTRAL_N * ne_by_n0 * collOp.coulomb_collision_mat(1.0, ne_by_n0, collisions.AR_NEUTRAL_N, h_prev, maxwellian, vth, sigma_m, full_assembly=True)
                        CC_00           = CC_ee[0::num_sh, 0::num_sh]
                        CC_10           = CC_ee[1::num_sh, 0::num_sh]
                        CC_11           = CC_ee[1::num_sh, 1::num_sh]
                    else:
                        CC_ee           = 0
                        
                        CC_00           = 0
                        CC_10           = 0
                        CC_11           = 0


                    abs_error = np.linalg.norm(residual_func(h_prev, CC_ee, eval_jacobian=False))
                    Lmat      = Cmat_p_Emat + CC_ee
                    Lmat_inv  = np.linalg.pinv(Lmat, rcond=1e-20/np.linalg.cond(Lmat))
                    h_curr    = np.matmul(Lmat_inv, WW(h_prev))
                    # print(WW(h_prev)[1::num_sh])

                    #print("after solve : ", np.linalg.norm(np.dot(Lmat,h_curr) - WW(h_prev)))
                    #print("after solve : , ", np.linalg.norm(residual_func(h_curr, CC_ee, eval_jacobian=False)))

                    # CC_ee           = collisions.AR_NEUTRAL_N * ion_deg * collOp.coulomb_collision_mat(1.0, ion_deg, collisions.AR_NEUTRAL_N, h_curr, maxwellian, vth, sigma_m, full_assembly=full_coulomb)
                    # CC_00           = CC_ee[0::num_sh, 0::num_sh]
                    # CC_10           = CC_ee[1::num_sh, 0::num_sh]
                    # CC_11           = CC_ee[1::num_sh, 1::num_sh]

                    # Lmat                = C_11 + CC_11
                    # Lmat_inv            = np.linalg.pinv(Lmat, rcond=1e-14/np.linalg.cond(Lmat))
                    # h_curr[1::num_sh]   = np.dot(Lmat_inv, WW(h_prev)[1::num_sh] - np.dot(E_10 + CC_10, h_prev[0::num_sh]))
                    
                    # h_curr[0::num_sh]   = h_prev[0::num_sh]
                    # CC_ee               = collisions.AR_NEUTRAL_N * ion_deg * collOp.coulomb_collision_mat(1.0, ion_deg, collisions.AR_NEUTRAL_N, h_curr, maxwellian, vth, sigma_m, full_assembly=True)
                    # CC_10               = CC_ee[1::num_sh, 0::num_sh]
                    # CC_11               = CC_ee[1::num_sh, 1::num_sh]

                    #h_curr[0::num_sh]   = h_prev[0::num_sh]
                    # Lmat                = C_00 + CC_00
                    # Lmat_inv            = np.linalg.pinv(Lmat, rcond=1e-14/np.linalg.cond(Lmat))
                    # h_curr[0::num_sh]   = np.dot(Lmat_inv, WW(h_prev)[0::num_sh] - 0 * np.dot(E_01, h_prev[1::num_sh]))


                    h_curr              = h_curr / (np.dot(u, h_curr))

                    if iteration_steps % 10 == 0:
                        rel_error = np.linalg.norm(h_prev-h_curr)/np.linalg.norm(h_curr)
                        print("Iteration ", iteration_steps, ": abs residual = %.8E rel residual=%.8E"%(abs_error, rel_error))
                
                    h_prev = h_curr
                    iteration_steps+=1

            print("Nonlinear solver (1) atol=%.8E , rtol=%.8E"%(abs_error, rel_error))
            return h_curr, abs_error, rel_error

        def solver_3(h_prev, rtol = 1e-5 , atol=1e-5, max_iter=1000):
            """
            We use generalized Fokker-Plank to compute the cc terms. 
            """
            h_curr          = np.copy(h_prev)
            
            abs_error       = 1.0
            rel_error       = 1.0 
            iteration_steps = 0

            ion_deg         = args.ion_deg
            Cmat_p_Emat     = Cmat + Emat
            Cmat1           = np.matmul(Minv, Cmat_p_Emat)
            full_coulomb = True

            Lmat      = Cmat_p_Emat 
            Lmat_inv  = np.linalg.pinv(Lmat, rcond=1e-30)

            t1 = time()
            hl_op, gl_op     =  collOp.compute_rosenbluth_potentials_op(maxwellian, vth, 1, Minv)
            cc_op_a, cc_op_b =  collOp.coulomb_collision_op_assembly(maxwellian, vth)
            cc_op            =  np.dot(cc_op_a, hl_op) + np.dot(cc_op_b, gl_op)
            t2 = time()
            print("Coulomb collision Op. assembly %.8E"%(t2-t1))
            
            

            #cc_op = np.swapaxes(cc_op,1,2)
            

            # print("cc_a cond %.8E"%(np.linalg.cond(cc_a)))
            # print("cc_b cond %.8E"%(np.linalg.cond(cc_b)))

            # cc_a = np.matmul(Minv, np.dot(np.swapaxes(cc_op_a,1,2),h_prev))
            # cc_b = np.matmul(Minv, np.dot(np.swapaxes(cc_op_b,1,2),h_prev))

            # print("cc_a cond %.8E"%(np.linalg.cond(cc_a)))
            # print("cc_b cond %.8E"%(np.linalg.cond(cc_b)))

            h_init = np.copy(h_prev)
            while (rel_error> rtol and abs_error > atol and iteration_steps < max_iter):
                if args.ee_collisions:
                    gamma_a     = collOp.gamma_a(h_prev, maxwellian, vth, collisions.AR_NEUTRAL_N, ion_deg)
                    CC_ee       = gamma_a * collisions.AR_NEUTRAL_N * ion_deg * np.dot(cc_op, h_prev)
                    
                    Lmat        = Cmat_p_Emat + CC_ee
                    Lmat_inv    = np.linalg.pinv(Lmat, rcond=1e-14/np.linalg.cond(Lmat))
                else:
                    CC_ee       = 0

                abs_error = np.linalg.norm(residual_func(h_prev, CC_ee, eval_jacobian=False))

                h_curr      = np.matmul(Lmat_inv, np.matmul(Mmat, np.dot(u, np.matmul(Cmat1,h_prev)) * h_prev))
                h_curr      = h_curr / (np.dot(u, h_curr))

                if iteration_steps % 10 == 0:
                    rel_error = np.linalg.norm(h_prev-h_curr)/np.linalg.norm(h_curr)
                    print("Iteration ", iteration_steps, ": abs residual = %.8E rel residual=%.8E"%(abs_error, rel_error))
                

                h_prev      = h_curr
                iteration_steps+=1

            print("Nonlinear solver (1) atol=%.8E , rtol=%.8E"%(abs_error, rel_error))

            # cc_1 = np.matmul(Minv, np.dot(cc_op, h_init))
            # cc_2 = np.matmul(Minv, np.dot(np.swapaxes(cc_op,1,2), h_init))

            # print("f maxwellian condition number on fixed h,g %.4E  on fixed f %.4E"%(np.linalg.cond(cc_1), np.linalg.cond(cc_2)))

            # U1, S1, V1 = np.linalg.svd(cc_1)
            # U2, S2, V2 = np.linalg.svd(cc_2)
            # plt.semilogy(S1, label='fixed on h,g = f_init')
            # plt.semilogy(S2, label='fixed on f = f_init')
            
            # cc_1 = np.matmul(Minv, np.dot(cc_op, h_curr))
            # cc_2 = np.matmul(Minv, np.dot(np.swapaxes(cc_op,1,2), h_curr))

            # print("f at steady state condition number on fixed h,g %.4E  on fixed f %.4E"%(np.linalg.cond(cc_1), np.linalg.cond(cc_2)))

            # U1, S1, V1 = np.linalg.svd(cc_1)
            # U2, S2, V2 = np.linalg.svd(cc_2)
            # plt.semilogy(S1, label='fixed on h,g = f_final')
            # plt.semilogy(S2, label='fixed on f = f_final')
            # plt.grid()
            # plt.legend()
            # plt.show()
            



            return h_curr, abs_error, rel_error


        #h_curr  , atol, rtol  = solver_1(h_prev, rtol = 1e-8, atol=1e-8, max_iter=200)
        h_curr  , atol, rtol  = solver_3(h_prev, rtol = 1e-8, atol=1e-8, max_iter=1000)
        
        solution_vector      = np.zeros((2,h_init.shape[0]))
        solution_vector[0,:] = np.matmul(qA, h_init)
        solution_vector[1,:] = np.matmul(qA, h_curr)
        return {'sol':solution_vector, 'htest': htest, 'atol': atol, 'rtol':rtol, 'tgrid':None}

    else:
        u        = mass_op * mw_vth(0) * vth**3 
        for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
            if lm[0] > 0:
                u[lm_idx::num_sh] = 0.0 
        
        u               = np.matmul(np.transpose(u), qA)
        p_vec           = u.reshape((u.shape[0], 1)) / np.sqrt(np.dot(u, u))
        
        ion_deg         = args.ion_deg
        Cmat_p_Emat     = Cmat + Emat
        Cmat_p_Emat     = np.matmul(Minv, Cmat_p_Emat)
        Wmat            = Cmat_p_Emat
        Imat            = np.eye(Wmat.shape[0])
        Impp            = (Imat - np.matmul(p_vec , np.transpose(p_vec)))

        atol   = 1.0
        rtol   = 1.0

        if args.ee_collisions:
            t1 = time()
            hl_op, gl_op     =  collOp.compute_rosenbluth_potentials_op(maxwellian, vth, 1, Minv)
            cc_op_a, cc_op_b =  collOp.coulomb_collision_op_assembly(maxwellian, vth)
            cc_op            =  np.dot(cc_op_a, hl_op) + np.dot(cc_op_b, gl_op)
            t2 = time()
            print("Coulomb collision Op. assembly %.8E"%(t2-t1))
        else:
            Pmat      = Imat - dt * np.matmul(Impp , (Cmat_p_Emat))
            Pmat_inv  = np.linalg.pinv(Pmat, rcond=1e-14/np.linalg.cond(Pmat))

        
        num_time_samples = 5
        tgrid            = np.append(np.array([0]), np.logspace(np.log10(dt), np.log10(t_end), num_time_samples-1, base=10)) #np.linspace(0, t_end, num_time_samples)
        tgrid_idx        = np.int64(np.floor(tgrid / dt))
        #print(tgrid_idx, tgrid)

        h_prev   = np.copy(h_init)
        for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
            if lm[0]>0:
                h_prev[lm_idx::num_sh] = h_prev[0::num_sh]

        h_prev   = h_prev / (np.dot(u, h_prev))
        
        rtol_desired = 1e-8
        atol_desired = 1e-4
        t_curr       = 0.0

        solution_vector = np.zeros((num_time_samples,h_init.shape[0]))

        sample_idx = 1
        t_step     = 0

        solution_vector[0,:] = h_prev

        f1      = u / np.dot(u, u)
        fb_prev = np.matmul(Impp, h_prev)
        #print(np.dot(u,f1), np.dot(u,fb_prev), np.dot(u, f1+fb_prev))
        cfl_factor = 0.2
        steady_state_tol = 1e-7
        if args.ee_collisions:

            def f_rhs(t,y, cc_ee):
                return np.dot(Impp, np.dot(Cmat_p_Emat + cc_ee , f1 + y) - np.dot(u, np.dot(Wmat,(f1+y))) * (f1+y))
            
            ode_int  = scipy.integrate.ode(f_rhs).set_integrator('vode', method='bdf', atol=1e-10, rtol=1e-10, order=5)
            gamma_a  = collOp.gamma_a(h_prev, maxwellian, vth, collisions.AR_NEUTRAL_N, ion_deg)
            # dt_min   = min(cfl_factor * (E_field/ MVTH) * collisions.ELECTRON_CHARGE_MASS_RATIO, gamma_a * collisions.AR_NEUTRAL_N * ion_deg * (1/np.linalg.cond(np.dot(cc_op,h_prev))))
            # # if dt > dt_min:
            # #     print("dt is too large, < %.2E"%(dt_min))
            # #     dt = dt_min
            # dt = dt_min

            while t_curr < t_end:
                if sample_idx < num_time_samples and t_step == tgrid_idx[sample_idx]:
                    solution_vector[sample_idx,:] = h_prev
                    sample_idx+=1

                gamma_a     = collOp.gamma_a(h_prev, maxwellian, vth, collisions.AR_NEUTRAL_N, ion_deg)
                CC_ee       = gamma_a * collisions.AR_NEUTRAL_N * ion_deg * np.dot(cc_op, h_prev)
                CC_ee       = np.matmul(Minv, CC_ee)
                

                # Pmat        = Imat - dt * np.matmul(Impp , (Cmat_p_Emat + CC_ee))
                # Pmat_inv    = np.linalg.pinv(Pmat, rcond=1e-14/np.linalg.cond(Pmat))
                # rhs_vec     = fb_prev + dt * np.matmul(Impp, np.dot(Cmat_p_Emat + CC_ee, f1) - np.dot(u,np.dot(Wmat, h_prev)) * h_prev)
                # fb_curr     = np.dot(Pmat_inv, rhs_vec)
                # h_curr      = f1 + fb_curr

                
                ode_int.set_f_params(CC_ee)
                ode_int.set_initial_value(fb_prev, t_curr)
                
                fb_curr = ode_int.integrate(t_curr + dt)
                h_curr  = f1 + fb_curr

                rtol= (np.linalg.norm(h_prev - h_curr))/np.linalg.norm(h_curr)
                atol= (np.linalg.norm(h_prev - h_curr))

                t_curr+= dt
                t_step+=1
                h_prev  = h_curr
                fb_prev = fb_curr

                #if t_step%100 == 0: 
                print("time = %.3E solution convergence atol = %.8E rtol = %.8E mass %.10E"%(t_curr, atol, rtol, np.dot(u,h_curr)))
        else:
            while t_curr < t_end:
                if sample_idx < num_time_samples and t_step == tgrid_idx[sample_idx]:
                    solution_vector[sample_idx,:] = h_prev
                    sample_idx+=1

                rhs_vec = fb_prev + dt * np.matmul(Impp, np.dot(Cmat_p_Emat,f1) - np.dot(u,np.dot(Wmat, h_prev)) * h_prev)

                fb_curr = np.dot(Pmat_inv,rhs_vec)
                h_curr  = fb_curr + f1

                rtol= (np.linalg.norm(h_prev - h_curr))/np.linalg.norm(h_curr)
                atol= (np.linalg.norm(h_prev - h_curr))

                t_curr+= dt
                t_step+=1
                h_prev  = h_curr
                fb_prev = fb_curr

                if t_step%100 == 0: 
                    print("time = %.3E solution convergence atol = %.8E rtol = %.8E mass %.10E"%(t_curr, atol, rtol, np.dot(u,h_curr)))

        solution_vector[-1,:] = h_curr
        return {'sol':solution_vector, 'htest': htest, 'atol': atol, 'rtol':rtol, 'tgrid':tgrid}

def solve_bte(steady_state, collOp, maxwellian, vth, E_field, t_end, dt,t_tol, collisions_included):

    spec_sp : sp.SpectralExpansionSpherical   = collOp._spec
    num_p        = spec_sp._p +1
    num_sh       = len(spec_sp._sph_harm_lm)
    
    if spec_sp.get_radial_basis_type() != basis.BasisType.SPLINES:
        raise NotImplementedError

    Mmat         = spec_sp.compute_mass_matrix()
    Minv         = spec_sp.inverse_mass_mat(Mmat=Mmat)
    mm_inv_error = np.linalg.norm(np.matmul(Mmat,Minv)-np.eye(Mmat.shape[0]))/np.linalg.norm(np.eye(Mmat.shape[0]))
    
    print("cond(M) = %.4E"%np.linalg.cond(Mmat))
    print("|I-M M^{-1}| error = %.12E"%(mm_inv_error))
    
    MVTH  = vth
    MNE   = maxwellian(0) * (np.sqrt(np.pi)**3) * (vth**3)
    MTEMP = collisions.electron_temperature(MVTH)
    
    print("==========================================================================")
    vratio = np.sqrt(1.0/args.basis_scale)

    if (args.radial_poly == "maxwell" or args.radial_poly == "laguerre"):
        hv    = lambda v,vt,vp : np.exp((v**2)*(1.-1./(vratio**2)))/vratio**3
    elif (args.radial_poly == "maxwell_energy"):
        hv    = lambda v,vt,vp : np.exp((v**2)*(1.-1./(vratio**2)))/vratio**3
    elif(args.radial_poly == "chebyshev"):
        hv    = lambda v,vt,vp : np.exp(-((v/vratio)**2)) / vratio**3
    elif (args.radial_poly == "bspline"):
        hv    = lambda v,vt,vp : np.exp(-((v/vratio)**2)) / vratio**3
    else:
        raise NotImplementedError

    h_init    = BEUtils.function_to_basis(spec_sp,hv,maxwellian,None,None,None,Minv=Minv)
    h_t       = np.array(h_init)
    
    ne_t      = MNE
    mw_vth    = BEUtils.get_maxwellian_3d(vth,ne_t)
    m0_t0     = mw_vth(0) * np.dot(mass_op, h_t) * vth**3
    temp_t0   = mw_vth(0) * np.dot(temp_op, h_t) * vth**5 * 0.5 * scipy.constants.electron_mass * (2./ (3 * collisions.BOLTZMANN_CONST)) / m0_t0 
    vth_curr  = vth 
    print("Initial temp (eV) = %.12E "   %(temp_t0 * collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT))
    print("Initial mass (1/m^3) = %.12E" %(m0_t0))


    gx, gw   = spec_sp._basis_p.Gauss_Pn(spec_sp._num_q_radial)
    c_gamma  = np.sqrt(2*collisions.ELECTRON_CHARGE_MASS_RATIO)
    
    f0_cf    = interp1d(ev, bolsig_f0, kind='cubic', bounds_error=False, fill_value=(bolsig_f0[0],bolsig_f0[-1]))
    fa_cf    = interp1d(ev, bolsig_a, kind='cubic', bounds_error=False, fill_value=(bolsig_a[0],bolsig_a[-1]))
    ftestt   = lambda v,vt,vp : f0_cf(.5*(v*VTH)**2/collisions.ELECTRON_CHARGE_MASS_RATIO)*(1. + fa_cf(.5*(v*VTH)**2/collisions.ELECTRON_CHARGE_MASS_RATIO)*np.cos(vt))
    
    htest                       = BEUtils.function_to_basis(spec_sp,ftestt,maxwellian,None,None,None,Minv=Minv)
    radial_projection[i, :, :]  = BEUtils.compute_radial_components(ev, spec_sp, htest, maxwellian, VTH, 1)
    scale                       = 1./( np.trapz(radial_projection[i, 0, :]*np.sqrt(ev),x=ev) )
    radial_projection[i, :, :] *= scale
    coeffs_projection.append(htest)

    FOp      = 0
    g_rate   = np.zeros(num_p * num_sh)
    

    k_vec    = spec_sp._basis_p._t
    dg_idx   = spec_sp._basis_p._dg_idx
    sp_order = spec_sp._basis_p._sp_order

    sigma_m  = np.zeros(len(gx)+1)
    gx_ev    = (gx * MVTH / c_gamma)**2
    gx_ev    = np.append(gx_ev, (k_vec[-1] * MVTH / c_gamma)**2)

    t1=time()
    if "g0" in collisions_included:
        g0  = collisions.eAr_G0()
        g0.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g0, mw_vth, vth_curr)
        sigma_m += collisions.Collisions.synthetic_tcs(gx_ev, g0._analytic_cross_section_type)

    if "g0ConstNoLoss" in collisions_included:
        g0  = collisions.eAr_G0_NoEnergyLoss(cross_section="g0Const")
        g0.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g0, mw_vth, vth_curr)
        sigma_m += collisions.Collisions.synthetic_tcs(gx_ev, g0._analytic_cross_section_type)
    
    if "g0NoLoss" in collisions_included:
        g0  = collisions.eAr_G0_NoEnergyLoss()
        g0.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g0, mw_vth, vth_curr)
        sigma_m += collisions.Collisions.synthetic_tcs(gx_ev, g0._analytic_cross_section_type)

    if "g0Const" in collisions_included:
        g0  = collisions.eAr_G0(cross_section="g0Const")
        g0.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g0, mw_vth, vth_curr)
        sigma_m += collisions.Collisions.synthetic_tcs(gx_ev, g0._analytic_cross_section_type)

    if "g2" in collisions_included:
        g2  = collisions.eAr_G2()
        g2.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g2, mw_vth, vth_curr)
        sigma_m += collisions.Collisions.synthetic_tcs(gx_ev, g2._analytic_cross_section_type)
        

    if "g2Smooth" in collisions_included:
        g2  = collisions.eAr_G2(cross_section="g2Smooth")
        g2.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g2, mw_vth, vth_curr)
        sigma_m += collisions.Collisions.synthetic_tcs(gx_ev, g2._analytic_cross_section_type)
        

    if "g2Regul" in collisions_included:
        g2  = collisions.eAr_G2(cross_section="g2Regul")
        g2.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g2, mw_vth, vth_curr)
        sigma_m += collisions.Collisions.synthetic_tcs(gx_ev, g2._analytic_cross_section_type)
        

    if "g2Const" in collisions_included:
        g2  = collisions.eAr_G2(cross_section="g2Const")
        g2.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g2, mw_vth, vth_curr)
        sigma_m += collisions.Collisions.synthetic_tcs(gx_ev, g2._analytic_cross_section_type)
        

    if "g2step" in collisions_included:
        g2  = collisions.eAr_G2(cross_section="g2step")
        g2.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g2, mw_vth, vth_curr)
        sigma_m += collisions.Collisions.synthetic_tcs(gx_ev, g2._analytic_cross_section_type)
        

    if "g2smoothstep" in collisions_included:
        g2  = collisions.eAr_G2(cross_section="g2smoothstep")
        g2.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g2, mw_vth, vth_curr)
        sigma_m += collisions.Collisions.synthetic_tcs(gx_ev, g2._analytic_cross_section_type)
        

    t2=time()
    print("Assembled the collision op. for Vth : ", vth_curr)
    print("Collision Operator assembly time (s): ",(t2-t1))

    u    = mass_op / (np.sqrt(np.pi)**3)
    # FOpt = np.matmul(Minv,FOp)
    # g_rate_fop = np.matmul(u.reshape(1,-1), FOpt).reshape(num_p*num_sh)

    sigma_m   = sigma_m * np.sqrt(gx_ev) * c_gamma * collisions.AR_NEUTRAL_N
    sigma_bdy = sigma_m[-1]
    sigma_m   = sigma_m[0:-1]
    
    h_prev    = np.array(h_init[0::num_sh])
    u         = u[0::num_sh]
    print("initial mass : %.12E"%(np.dot(u, h_prev)))
    h_prev    = h_prev / np.dot(u, h_prev)

    FOp    = FOp[0::num_sh, 0::num_sh]
    Mmat   = Mmat[0::num_sh, 0::num_sh]

    Minv = BEUtils.choloskey_inv(Mmat)
    mm_inv_error = np.linalg.norm(np.matmul(Mmat,Minv)-np.eye(Mmat.shape[0]))/np.linalg.norm(np.eye(Mmat.shape[0]))
    print("cond(M) = %.4E"%np.linalg.cond(Mmat))
    print("|I-M M^{-1}| error = %.12E"%(mm_inv_error))

    g_rate = np.matmul(u.reshape(1,-1), np.matmul(Minv, FOp))
    Ev     = E_field * collisions.ELECTRON_CHARGE_MASS_RATIO
    Vq_d1  = spec_sp.Vdq_r(gx,0,1,1)
    Wt     = Mmat 
    
    ### if we need to apply some boundary conditions. 
    fx      = k_vec[num_p -4 :  num_p] 
    #fd_coef = [0.5, -2, 1.5]
    fd_coef = [-1/3, 3/2, -3, 11/6]
    bdy_op  = 0
    for ii in range(4):
        bdy_op+=np.array([ fd_coef[ii] * spec_sp.basis_eval_radial(fx[ii], p, 0) for p in range(num_p)])
    dx_bdy = (fx[-1]-fx[-2])
    bdy_op = bdy_op/dx_bdy

    def assemble_f1_op(h_prev):
        g_rate_mu = np.dot(g_rate, h_prev)
        w_ev =  1./(sigma_m + g_rate_mu)
        Bt = np.array([ w_ev * (gx**2) * spec_sp.basis_eval_radial(gx, p, 0) * Vq_d1[k,:] for p in range(num_p) for k in range(num_p)]).reshape((num_p, num_p, -1))
        Bt = np.dot(Bt, gw) 
        return Bt

    if steady_state:
        iteration_error = 1
        iteration_steps = 0 
        
        nn       = FOp.shape[0]
        Ji       = np.zeros((nn+1,nn))
        Rf       = np.zeros(nn+1)
        
        iteration_error = 1
        iteration_steps = 0

        def residual_func(x):
            g_rate_mu = np.dot(g_rate, x) 
            w_ev =  1./(sigma_m + g_rate_mu)
            tmp  = np.sqrt(w_ev * gw) * gx * Vq_d1
            At   = np.matmul(tmp, np.transpose(tmp))

            fx = k_vec[dg_idx[-1] + sp_order]
            assert fx == k_vec[-1], "flux assembly face coords does not match at the boundary"
            At[dg_idx[-1] , dg_idx[-1]] += -fx**2 * spec_sp.basis_derivative_eval_radial(fx - 2 * np.finfo(float).eps, dg_idx[-1],0,1) / (sigma_bdy + g_rate_mu) 
            
            Rop = -(1/3) * ((Ev/MVTH)**2) * (At) + FOp -  Wt * g_rate_mu
            
            y  = np.matmul(Rop, x)
            return np.append(y, np.dot(u, x)-1 )

        def jacobian_func(x):
            g_rate_mu = np.dot(g_rate, x) 
            w_ev =  1./(sigma_m + g_rate_mu)
            tmp  = np.sqrt(w_ev * gw) * gx * Vq_d1
            At   = np.matmul(tmp, np.transpose(tmp))

            fx = k_vec[dg_idx[-1] + sp_order]
            assert fx == k_vec[-1], "flux assembly face coords does not match at the boundary"
            At[dg_idx[-1] , dg_idx[-1]] += -fx**2 * spec_sp.basis_derivative_eval_radial(fx - 2 * np.finfo(float).eps, dg_idx[-1],0,1) / (sigma_bdy + g_rate_mu) 
            
            Ji = (-(1/3) * ((Ev/MVTH)**2) * (At) + FOp -  2 * Wt * g_rate_mu)
            Ji = np.vstack((Ji,u))
            return Ji

        while (iteration_error > 1e-14 and iteration_steps < 1000):
            Rf = residual_func(h_prev) 
            if(iteration_steps%100==0):
                print("Iteration ", iteration_steps, ": Residual =", np.linalg.norm(Rf))
            
            Ji = jacobian_func(h_prev)
            p  = np.matmul(np.linalg.pinv(Ji,rcond=1e-14), -Rf)
            
            alpha=1e-1
            nRf=np.linalg.norm(Rf)
            is_diverged = False
            while (np.linalg.norm(residual_func(h_prev + alpha *p)) >= nRf):
                alpha*=0.5
                if alpha < 1e-100:
                    is_diverged = True
                    break
            
            if(is_diverged):
                print("Iteration ", iteration_steps, ": Residual =", np.linalg.norm(Rf))
                print("line search step size becomes too small")
                break

            iteration_error = np.linalg.norm(residual_func(h_prev + alpha *p))
            h_prev += p*alpha
            iteration_steps = iteration_steps + 1


        hh = spec_sp.create_vec().reshape(num_p * num_sh)
        hh[0::num_sh] = h_prev

        Bt = assemble_f1_op(h_prev)
        hh[1::num_sh] = (Ev/np.sqrt(3)) * np.matmul(Minv,np.matmul(Bt,h_prev)) / MVTH

        solution_vector = np.zeros((2,h_init.shape[0]))
        solution_vector[0,:] = h_init
        solution_vector[1,:] = hh
        return solution_vector

    else:
        raise NotImplementedError
   

parser = argparse.ArgumentParser()

parser.add_argument("-Nr", "--NUM_P_RADIAL"                   , help="Number of polynomials in radial direction", type=int, default=16)
parser.add_argument("-T", "--T_END"                           , help="Simulation time", type=float, default=1e-4)
parser.add_argument("-dt", "--T_DT"                           , help="Simulation time step size ", type=float, default=1e-7)
parser.add_argument("-o",  "--out_fname"                      , help="output file name", type=str, default='coll_op')
parser.add_argument("-ts_tol", "--ts_tol"                     , help="adaptive timestep tolerance", type=float, default=1e-15)
parser.add_argument("-l_max", "--l_max"                       , help="max polar modes in SH expansion", type=int, default=1)
parser.add_argument("-c", "--collisions"                      , help="collisions included (g0, g0Const, g0NoLoss, g2, g2Const)",nargs='+', type=str, default=["g0Const"])
parser.add_argument("-ev", "--electron_volt"                  , help="initial electron volt", type=float, default=0.25)
parser.add_argument("-bscale", "--basis_scale"                , help="basis electron volt", type=float, default=1.0)
parser.add_argument("-q_vr", "--quad_radial"                  , help="quadrature in r"        , type=int, default=200)
parser.add_argument("-q_vt", "--quad_theta"                   , help="quadrature in polar"    , type=int, default=8)
parser.add_argument("-q_vp", "--quad_phi"                     , help="quadrature in azimuthal", type=int, default=8)
parser.add_argument("-q_st", "--quad_s_theta"                 , help="quadrature in scattering polar"    , type=int, default=8)
parser.add_argument("-q_sp", "--quad_s_phi"                   , help="quadrature in scattering azimuthal", type=int, default=8)
parser.add_argument("-radial_poly", "--radial_poly"           , help="radial basis", type=str, default="bspline")
parser.add_argument("-sp_order", "--spline_order"             , help="b-spline order", type=int, default=1)
parser.add_argument("-spline_qpts", "--spline_q_pts_per_knot" , help="q points per knots", type=int, default=7)
parser.add_argument("-E", "--E_field"                         , help="Electric field in V/m", type=float, default=100)
parser.add_argument("-dv", "--dv_target"                      , help="target displacement of distribution in v_th units", type=float, default=0)
parser.add_argument("-nt", "--num_timesteps"                  , help="target number of time steps", type=float, default=100)
parser.add_argument("-steady", "--steady_state"               , help="Steady state or transient", type=int, default=1)
parser.add_argument("-run_bolsig_only", "--run_bolsig_only"   , help="run the bolsig code only", type=bool, default=False)
parser.add_argument("-bolsig", "--bolsig_dir"                 , help="Bolsig directory", type=str, default="../../Bolsig/")
parser.add_argument("-sweep_values", "--sweep_values"         , help="Values for parameter sweep", nargs='+', type=float, default=[24, 48, 96])
parser.add_argument("-sweep_param", "--sweep_param"           , help="Paramter to sweep: Nr, ev, bscale, E, radial_poly", type=str, default="Nr")
parser.add_argument("-dg", "--use_dg"                         , help="enable dg splines", type=int, default=0)
parser.add_argument("-Tg", "--Tg"                             , help="Gass temperature (K)" , type=float, default=1e-12)
parser.add_argument("-ion_deg", "--ion_deg"                   , help="Ionization degreee"   , type=float, default=1)
parser.add_argument("-store_eedf", "--store_eedf"             , help="store eedf"   , type=int, default=0)
parser.add_argument("-store_csv", "--store_csv"               , help="store csv format of QoI comparisons", type=int, default=0)
parser.add_argument("-ee_collisions", "--ee_collisions"       , help="Enable electron-electron collisions", type=float, default=1)

args                = parser.parse_args()
# EbyN_Td             = np.array([1,5,20,100])
# ##e_values          = np.logspace(np.log10(0.148), np.log10(1e5), 100, base=10) #np.array([1e0, 1e1, 1e2])
# e_values            = EbyN_Td * collisions.AR_NEUTRAL_N * 1e-21
# ion_deg_values      = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
e_values            = np.array([args.E_field])
ion_deg_values      = np.array([args.ion_deg])

SAVE_EEDF    = args.store_eedf
SAVE_CSV     = args.store_csv

if not args.ee_collisions:
    ion_deg_values *= 0.0

run_params          = [(e_values[i], ion_deg_values[j]) for i in range(len(e_values)) for j in range(len(ion_deg_values))]
print(run_params)

#np.array([210.2110528, 363.5566248, 628.765318, 1087.439475, 1880.70903, 3252.655928, 5625.415959, 9729.066156]) #np.logspace(np.log10(0.148), np.log10(114471.000) , 4, base=10)
str_datetime = datetime.now().strftime("%m_%d_%Y_%H:%M:%S")

collisions.AR_NEUTRAL_N = 3.22e22
collisions.MAXWELLIAN_N = 1
collisions.AR_IONIZED_N = collisions.AR_NEUTRAL_N 
collisions.AR_TEMP_K    = args.Tg 

COLLISOIN_NAMES=dict()
for  col_idx, col in enumerate(args.collisions):
    COLLISOIN_NAMES[col]=col

COLLISOIN_NAMES["g0"] = "elastic"
COLLISOIN_NAMES["g2"] = "ionization"

# if SAVE_EEDF:
#     with open('eedf_%s.npy'%(str_datetime), 'ab') as f:
#         np.save(f, e_values)
    
for run_id in range(len(run_params)):
    
    args.E_field = run_params[run_id][0] #e_values[run_id]
    args.ion_deg = run_params[run_id][1] #e_values[run_id]
    print(args)

    try:
        bolsig.run_bolsig(args)
        [bolsig_ev, bolsig_f0, bolsig_a, bolsig_E, bolsig_mu, bolsig_M, bolsig_D, bolsig_rates,bolsig_cclog] = bolsig.parse_bolsig(args.bolsig_dir+"argon.out",len(args.collisions))
    except:
        print(args.bolsig_dir+"argon.out file not found due to Bolsig+ run faliure")
        sys.exit(0)

    
    if (args.run_bolsig_only):
        sys.exit(0)    

    if 0:
        ne           = collisions.AR_NEUTRAL_N * args.ion_deg 
        eps_0        = scipy.constants.epsilon_0
        me           = scipy.constants.electron_mass
        qe           = scipy.constants.e
        kT           = (bolsig_mu /1.5) * scipy.constants.electron_volt
        
        # kp_op      = vth**5 * mw(0) * np.sqrt(4*np.pi) * (2/(3*(2 / me))) * np.array([np.dot(gmw_a, gmx_a**4 * B(gmx_a,k,0)) for k in range(num_p)])
        # kT         = np.dot(kp_op, fb[0::num_sh])        
        # print(kT)

        #kT         = ((vth**5 * p40_a[-1] * qe) / (3 * (2/me))) #/ (2 * sph_l0(0))

        #Tev        = kT/ scipy.constants.electron_volt
        #kT         = 2 * np.pi * ((2 * vth**5 * p40_a[-1] * qe) / (3 * (2/me))) / m0 
        #c_lambda   = 2 * np.pi * (2 * vth**5 * p40_a[-1] / (3 * (2/me))) #np.exp(23.5 - np.log(np.sqrt(ne* 1e-6) * Tev **(5/4) - np.sqrt(1e-5 + ((np.log(Tev)-2)**2 )/16)))
        #c_lambda   = np.exp(23 - np.log(np.sqrt(ne * 1e-6) * (kT /scipy.constants.electron_volt)**(-1.5)))

        # b_min       = max(qe**2 / (4 * np.pi * eps_0 * 3 * kT/(0.5 * me)), scipy.constants.Planck / (np.sqrt(me * 3 * kT)))
        # b_max       = np.sqrt((eps_0 * kT) / (ne * qe**2))
        # c_lambda    = b_max/b_min

        # cc_freq_op    = vth**4 * mw(0) * np.sqrt(4*np.pi) * n0 * np.array([np.dot(gmw_a, gmx_a**3 * sigma_m * B(gmx_a,k,0)) for k in range(num_p)])
        # cc_freq       = np.dot(cc_freq_op, fb[0::num_sh])
        # M             = (np.sqrt(6) * cc_freq ) / (2 * np.sqrt((qe**2 * ne) / (eps_0 * me)))
        # c_lambda      = ((12 * np.pi * (eps_0 * kT)**(1.5))/(qe**3 * np.sqrt(ne)))
        # c_lambda      = (c_lambda + M) / (1 + M)
        #c_lambda      = np.exp(6.314)

        # Tev        = kT/ scipy.constants.electron_volt
        # c_lambda     = np.exp(23.5 - np.log(np.sqrt(ne* 1e-6) * Tev **(5/4) - np.sqrt(1e-5 + ((np.log(Tev)-2)**2 )/16)))

        total_cs = 0
        ev       = bolsig_ev 
        
        for col_idx, col in enumerate(args.collisions):
            cs = collisions.Collisions.synthetic_tcs(ev, col)
            total_cs += cs
        M1              = (np.sqrt(6) * collisions.AR_NEUTRAL_N * np.sqrt(2/me) * np.trapz(bolsig_f0 * ev * total_cs ,x=ev) ) / (2 * np.sqrt((qe**2 * ne) / (eps_0 * me)))

        print(M1)

        M              = M1 * 1e-6
        #bolsig_f1     =  bolsig_f0*bolsig_a * spec_sp._sph_harm_real(0, 0, 0, 0)/spec_sp._sph_harm_real(1, 0, 0, 0)
        c_lambda       = ((12 * np.pi * (eps_0 * kT)**(1.5))/(qe**3 * np.sqrt(ne)))
        M_req          = (np.exp(bolsig_cclog) -c_lambda)/(1- np.exp(bolsig_cclog))

        c_lambda       = (c_lambda + M) / (1 + M)
        print("Mreq : ", M_req, "M computed ", M)
        print("computed cl from bolsig : %.8E kT %.8E ratio %.8E" %(np.log(c_lambda), kT, np.log(c_lambda)/bolsig_cclog))
        



    # setting electron volts from bolsig results for now
    print("blolsig temp : %.8E"%((bolsig_mu /1.5)))
    args.electron_volt = (bolsig_mu/1.5) 

    run_data=list()
    run_temp=list()
    v = np.linspace(-2,2,100)
    vx, vz = np.meshgrid(v,v,indexing='ij')
    vy = np.zeros_like(vx)
    v_sph_coord = BEUtils.cartesian_to_spherical(vx, vy, vz)

    ev = bolsig_ev

    params.BEVelocitySpace.SPH_HARM_LM = [[i,0] for i in range(args.l_max+1)]
    num_sph_harm = len(params.BEVelocitySpace.SPH_HARM_LM)

    radial            = np.zeros((len(args.sweep_values), num_sph_harm, len(ev)))
    radial_base       = np.zeros((len(args.sweep_values), num_sph_harm, len(ev)))
    radial_cg         = np.zeros((len(args.sweep_values), num_sph_harm, len(ev)))
    radial_projection = np.zeros((len(args.sweep_values), num_sph_harm, len(ev)))
    #radial_base = np.zeros((len(args.sweep_values), len(ev)))
    #radial_intial = np.zeros((num_sph_harm, len(ev)))

    coeffs_projection = list()

    density_slice         = np.zeros((len(args.sweep_values),len(vx[0]),len(vx[1])))
    density_slice_initial = np.zeros((len(args.sweep_values),len(vx[0]),len(vx[1])))

    SPLINE_ORDER = args.spline_order
    basis.BSPLINE_BASIS_ORDER=SPLINE_ORDER
    basis.XLBSPLINE_NUM_Q_PTS_PER_KNOT=args.spline_q_pts_per_knot

    mu = []
    M = []
    D = []
    rates = []
    solver_tol = []
    rdata_list = []
    for i in enumerate(args.collisions):
        rates.append([])

    for i, value in enumerate(args.sweep_values):

        BASIS_EV                       = args.electron_volt*args.basis_scale
        collisions.MAXWELLIAN_TEMP_K   = BASIS_EV * collisions.TEMP_K_1EV
        collisions.ELECTRON_THEMAL_VEL = collisions.electron_thermal_velocity(collisions.MAXWELLIAN_TEMP_K) 
        VTH                            = collisions.ELECTRON_THEMAL_VEL
        maxwellian                     = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)
        c_gamma                        = np.sqrt(2*collisions.ELECTRON_CHARGE_MASS_RATIO)
        if "g2Smooth" in args.collisions or "g2" in args.collisions or "g2step" in args.collisions or "g2Regul" in args.collisions:
            sig_pts = np.array([np.sqrt(15.76) * c_gamma/VTH])
        else:
            sig_pts = None #np.array([np.sqrt(0.5 * (ev[0] + ev[-1])) * c_gamma/VTH])
            
        ev_range = ((0*VTH/c_gamma)**2, (1.0 + 1e-8) * ev[-1])
        #ev_range = ((0*VTH/c_gamma)**2, (4*VTH/c_gamma)**2)
        k_domain = (np.sqrt(ev_range[0]) * c_gamma / VTH, np.sqrt(ev_range[1]) * c_gamma / VTH)
        print("target ev range : (%.4E, %.4E) ----> knots domain : (%.4E, %.4E)" %(ev_range[0], ev_range[1], k_domain[0],k_domain[1]))
        if(sig_pts is not None):
            print("singularity pts : ", sig_pts, "v/vth and" , (sig_pts * VTH/c_gamma)**2,"eV")

        if args.sweep_param == "Nr":
            args.NUM_P_RADIAL = int(value)
        # elif args.sweep_param == "l_max":
        #     args.l_max = value
        elif args.sweep_param == "ev":
            args.electron_volt = value
            args.electron_volt_basis = value
        elif args.sweep_param == "bscale":
            args.basis_scale = value
        elif args.sweep_param == "E":
            args.E_field = value
        elif args.sweep_param == "radial_poly":
            args.radial_poly = value
        elif args.sweep_param == "q_vr":
            args.quad_radial = value

        params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER = args.NUM_P_RADIAL
        tt_vec = None
        if (args.radial_poly == "maxwell"):
            params.BEVelocitySpace.NUM_Q_VR  = args.quad_radial
            spec_sp  = sp.SpectralExpansionSpherical(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER, basis.Maxwell(), params.BEVelocitySpace.SPH_HARM_LM)
            spec_sp._num_q_radial = params.BEVelocitySpace.NUM_Q_VR
        elif (args.radial_poly == "maxwell_energy"):
            params.BEVelocitySpace.NUM_Q_VR  = args.quad_radial
            spec_sp  = sp.SpectralExpansionSpherical(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER, basis.MaxwellEnergy(), params.BEVelocitySpace.SPH_HARM_LM)
            spec_sp._num_q_radial = params.BEVelocitySpace.NUM_Q_VR
        elif (args.radial_poly == "laguerre"):
            params.BEVelocitySpace.NUM_Q_VR  = args.quad_radial
            spec_sp  = sp.SpectralExpansionSpherical(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER, basis.Laguerre(), params.BEVelocitySpace.SPH_HARM_LM)
            spec_sp._num_q_radial = params.BEVelocitySpace.NUM_Q_VR
        elif (args.radial_poly == "chebyshev"):
            params.BEVelocitySpace.NUM_Q_VR  = args.quad_radial
            spec_sp  = sp.SpectralExpansionSpherical(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER, basis.Chebyshev(domain=(-1,1), window=k_domain) , params.BEVelocitySpace.SPH_HARM_LM)
            spec_sp._num_q_radial = params.BEVelocitySpace.NUM_Q_VR
        elif (args.radial_poly == "bspline"):
            r_mode                          = basis.BasisType.SPLINES
            max_lev                         = int(np.ceil(np.log2(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)))

            def refine_func(x):
                y    = np.zeros_like(x)
                x_ev = (x * VTH / c_gamma)**2
                collisions_included = args.collisions
                for c in collisions_included:
                    y+= VTH * (x**3) * maxwellian(x) * collisions.Collisions.synthetic_tcs(x_ev, c) * collisions.AR_NEUTRAL_N
                
                return y

            # def refine_func_energy(x_ev):
            #     y    = np.zeros_like(x_ev)
            #     x    = np.sqrt(x_ev) * c_gamma / VTH
            #     collisions_included = args.collisions
            #     for c in collisions_included:
            #         y+= VTH * (x**3) * maxwellian(x) * collisions.Collisions.synthetic_tcs(x_ev, c) * collisions.AR_NEUTRAL_N
                
            #     return maxwellian(x)
            # tt_vec_ev = np.linspace(ev_range[0], ev_range[1],1<<max_lev) #basis.BSpline.adaptive_fit(refine_func_energy, (ev_range[0],ev_range[1]), sp_order=SPLINE_ORDER, min_lev=4, max_lev=max_lev, sig_pts=sig_pts, atol=1e-100, rtol=1e-16)
            # tt_vec = np.sqrt(tt_vec_ev) * c_gamma / VTH
            # print(tt_vec)
            # tt_vec = basis.BSpline.adaptive_fit(refine_func, k_domain, sp_order=SPLINE_ORDER, min_lev=4, max_lev=max_lev, sig_pts=sig_pts, atol=1e-100, rtol=1e-10)
            bb     = basis.BSpline(k_domain, SPLINE_ORDER, params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER+1, sig_pts=sig_pts, knots_vec=None, dg_splines=args.use_dg)
            params.BEVelocitySpace.NUM_Q_VR = bb._num_knot_intervals * args.spline_q_pts_per_knot
            params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER = bb._num_p
            # if args.sweep_param == "Nr":
            #     args.sweep_values[i] = params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER
            spec_sp               = sp.SpectralExpansionSpherical(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER-1, bb,params.BEVelocitySpace.SPH_HARM_LM)
            spec_sp._num_q_radial = params.BEVelocitySpace.NUM_Q_VR

        params.BEVelocitySpace.NUM_Q_VT  = args.quad_theta
        params.BEVelocitySpace.NUM_Q_VP  = args.quad_phi
        params.BEVelocitySpace.NUM_Q_CHI = args.quad_s_theta
        params.BEVelocitySpace.NUM_Q_PHI = args.quad_s_phi
        params.BEVelocitySpace.VELOCITY_SPACE_DT = args.T_DT

        print("""===========================Parameters ======================""")
        print("\tMAXWELLIAN_N        : ", collisions.MAXWELLIAN_N)
        print("\tELECTRON_THEMAL_VEL : ", VTH," ms^-1")
        print("\tBASIS_EV            : ", BASIS_EV,"eV")
        print("\tDT : ", params.BEVelocitySpace.VELOCITY_SPACE_DT, " s")
        print("""============================================================""")
        params.print_parameters()

        cf   = colOpSp.CollisionOpSP(spec_sp = spec_sp)
        
        mass_op   = BEUtils.mass_op(spec_sp, None, 64, 2, 1)
        temp_op   = BEUtils.temp_op(spec_sp, None, 64, 2, 1)
        avg_vop   = BEUtils.mean_velocity_op(spec_sp, None, 64, 4, 1)
        eavg_to_K = (2/(3*scipy.constants.Boltzmann))
        ev_fac    = (collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT)

        rdata     = solve_collop_dg(args.steady_state, cf, maxwellian, VTH, args.E_field, args.T_END, args.T_DT, args.ts_tol, collisions_included=args.collisions)
        rdata_list.append(rdata)

        data      = rdata['sol']
        htest     = rdata['htest']
        abs_tol   = rdata['atol']
        rel_tol   = rdata['rtol']    

        #data               = solve_bte(args.steady_state, cf, maxwellian, VTH, args.E_field, args.T_END, args.T_DT, args.ts_tol, collisions_included=args.collisions)
        radial_base[i,:,:]  = BEUtils.compute_radial_components(ev, spec_sp, data[0,:], maxwellian, VTH, 1)
        scale               = 1./( np.trapz(radial_base[i,0,:]*np.sqrt(ev),x=ev) )
        radial_base[i,:,:] *= scale

        radial[i, :, :]     = BEUtils.compute_radial_components(ev, spec_sp, data[-1,:], maxwellian, VTH, 1)
        scale               = 1./( np.trapz(radial[i,0,:]*np.sqrt(ev),x=ev) )
        radial[i, :, :]    *= scale

        # print(data[-1,:]*scale)
        # print(htest)
        #print((data[-1,:]*scale - htest) / htest)
        run_data.append(data)
        solver_tol.append((abs_tol, rel_tol))


        nt = len(data[:,0])
        temp_evolution = np.zeros(nt)
        for k in range(nt):
            current_vth      = VTH
            current_mw       = maxwellian
            current_mass     = np.dot(data[k,:],mass_op) * current_vth**3 * current_mw(0)
            current_temp     = np.dot(data[k,:],temp_op) * current_vth**5 * current_mw(0) * 0.5 * collisions.MASS_ELECTRON * eavg_to_K / current_mass
            temp_evolution[k] = current_temp/collisions.TEMP_K_1EV

        mu.append(1.5*temp_evolution[-1])
        print("PDE code found temp (ev) = %.8E mean energy (ev) = .%8E"%(temp_evolution[-1] , 1.5*temp_evolution[-1]))

        total_cs = 0

        for col_idx, col in enumerate(args.collisions):
            cs = collisions.Collisions.synthetic_tcs(ev, col)
            total_cs += cs
            rates[col_idx].append( np.sqrt(2.*collisions.ELECTRON_CHARGE_MASS_RATIO)*np.trapz(radial[i,0,:]*ev*cs,x=ev) )

            # if col == "g2" or col == "g2Const" or col == "g2Smooth" or col=="g2step":
            #     total_cs += rates[col_idx][-1]/np.sqrt(ev)/np.sqrt(2.*collisions.ELECTRON_CHARGE_MASS_RATIO)
        
        c_gamma = np.sqrt(2.*collisions.ELECTRON_CHARGE_MASS_RATIO)

        #### diffusion coefficient 
        D.append((c_gamma / 3.) * np.trapz(radial[i,0,:] * ev / total_cs,x=ev))

        #### mobility is computed from f1
        # f1 = deriv_fd(ev, radial[i,0,:]) * (args.E_field / (collisions.AR_NEUTRAL_N * total_cs))
        # f1 = radial[i,1,:]
        # M.append( -(c_gamma / (3 * (args.E_field/collisions.AR_NEUTRAL_N))) * np.trapz(f1 * ev, x=ev)) 
        # M.append( -(c_gamma / 3.) * np.trapz(deriv_fd(ev,radial[i,0,:])*ev/total_cs,x=ev) )
        # f1 = deriv_fd(ev,radial[i,0,:])  * (args.E_field/total_cs/collisions.AR_NEUTRAL_N)/np.sqrt(3)
        # g1 = radial[i,1,:]
        
        # plt.semilogy(ev, np.abs(f1), label="f1")
        # plt.semilogy(ev, np.abs(g1), label="g1")
        # plt.grid()
        # plt.legend()
        # plt.show()
        # plt.close()
        f1 = radial[i,1,:] * np.sqrt(3)
        M.append( -(c_gamma / (3 * (args.E_field / collisions.AR_NEUTRAL_N))) * np.trapz(f1 * ev ,x=ev))

        print("D computed from bolsig %.8E vs. reported %.8E from PDE code %.8E"%(np.sqrt(2.*collisions.ELECTRON_CHARGE_MASS_RATIO)/3.*np.trapz(bolsig_f0*ev/total_cs,x=ev), bolsig_D, D[-1]))
        print("M computed from bolsig %.8E vs. reported %.8E from PDE code %.8E"%(-np.sqrt(2.*collisions.ELECTRON_CHARGE_MASS_RATIO)/3.*np.trapz(deriv_fd(ev, bolsig_f0)*ev/total_cs,x=ev), bolsig_M, M[-1]))

        run_temp.append(temp_evolution)

    if SAVE_CSV:
        with open("pde_vs_bolsig_%s.csv"%str_datetime, 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            if run_id == 0:
                # write the header
                header = ["E/N(Td)","E(V/m)","Nr","energy","diffusion","mobility","bolsig_energy","bolsig_defussion","bolsig_mobility","l2_f0","l2_f1", "Tg", "ion_deg", "atol", "rtol"]
                for g in args.collisions:
                    header.append(str(g))
                    header.append("bolsig_"+str(g))

                writer.writerow(header)

            for i, value in enumerate(args.sweep_values):
                # write the data
                bolsig_f1 =  bolsig_f0*bolsig_a * spec_sp._sph_harm_real(0, 0, 0, 0)/spec_sp._sph_harm_real(1, 0, 0, 0)

                l2_f0     = np.linalg.norm(radial[i, 0]-bolsig_f0)/np.linalg.norm(bolsig_f0)
                l2_f1     = np.linalg.norm(radial[i, 1]-bolsig_f1)/np.linalg.norm(bolsig_f1)

                data = [args.E_field/collisions.AR_NEUTRAL_N/1e-21, args.E_field, args.sweep_values[i], mu[i], D[i], M[i], bolsig_mu, bolsig_D, bolsig_M, l2_f0, l2_f1, args.Tg, args.ion_deg, solver_tol[i][0], solver_tol[i][1]]
                for col_idx , _ in enumerate(args.collisions):
                    data.append(rates[col_idx][i])
                    data.append(bolsig_rates[col_idx])

                writer.writerow(data)
        

    if SAVE_EEDF:
        with open('eedf_%s.npy'%(str_datetime), 'ab') as f:
            np.save(f, spec_sp._p + 1)
            np.save(f, spec_sp._sph_harm_lm[-1][0])
            np.save(f, args.E_field)
            np.save(f, args.ion_deg)
            np.save(f, args.Tg)
            np.save(f, ev)

            for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
                np.save(f, radial[-1,lm_idx,:])
                
            np.save(f, bolsig_f0)
            np.save(f, bolsig_f1)

    if (1):
        fig = plt.figure(figsize=(21, 9), dpi=300)

        num_subplots = num_sph_harm + 2

        plt.subplot(2, num_subplots,  1 + 0)
        plt.semilogy(bolsig_ev,  abs(bolsig_f0), '-k', label="bolsig")
        
        # print(np.trapz( bolsig[:,1]*np.sqrt(bolsig[:,0]), x=bolsig[:,0] ))
        # print(np.trapz( scale*radial[i, 0]*np.sqrt(ev), x=ev ))

        plt.subplot(2, num_subplots,  1 + 1)
        plt.semilogy(bolsig_ev,  abs(bolsig_f0*bolsig_a * spec_sp._sph_harm_real(0, 0, 0, 0)/spec_sp._sph_harm_real(1, 0, 0, 0)), '-k', label="bolsig")

        for i, value in enumerate(args.sweep_values):
            data=run_data[i]
            data_projection=coeffs_projection[i]

            lbl = args.sweep_param+"="+str(value)

            if args.steady_state == 0:
                rdata   = rdata_list[i]
                data_tt = rdata['sol']
                tgrid   = list(rdata['tgrid'])

                radial_tt = np.zeros((len(tgrid), num_sph_harm, len(ev)))

                for t_idx, tt in enumerate(tgrid):
                    radial_tt[t_idx, :, : ] = BEUtils.compute_radial_components(ev, spec_sp, data_tt[t_idx,:], maxwellian, VTH, 1)
                    scale                   = 1./( np.trapz(radial_tt[t_idx,0,:]*np.sqrt(ev),x=ev) )
                    radial_tt[t_idx, :, :]  *= scale

            # spherical components plots

            for l_idx in range(num_sph_harm):

                plt.subplot(2, num_subplots, num_subplots + 1+l_idx)

                color = next(plt.gca()._get_lines.prop_cycler)['color']
                plt.plot(np.abs(data[-1,l_idx::num_sph_harm]),label=lbl, color=color)
                
                plt.title(label="l=%d"%l_idx)
                plt.yscale('log')
                plt.xlabel("coeff #")
                plt.ylabel("abs(coeff)")
                plt.grid(visible=True)
                if l_idx == 0:
                    plt.legend()

                plt.subplot(2, num_subplots, 1 + l_idx)
                color = next(plt.gca()._get_lines.prop_cycler)['color']
                plt.semilogy(ev,  abs(radial[i, l_idx]), '-', label=lbl, color=color)

                if args.steady_state == 0:
                    for t_idx, tt in enumerate(tgrid[:-1]):
                        color = next(plt.gca()._get_lines.prop_cycler)['color']
                        # print(tt)
                        # print(np.abs(radial_tt[t_idx, l_idx]))
                        plt.semilogy(ev,  abs(radial_tt[t_idx, l_idx]), '--', label=lbl+" t=%.2E"%(tt), color=color)

                
                plt.xlabel("Energy, eV")
                plt.ylabel("Radial component")
                plt.title("f%d"%(l_idx))
                plt.grid(visible=True)
                if l_idx == 0:
                    #plt.legend(loc=(1.04, 0))
                    plt.legend(prop={'size': 8})
                # plt.legend()


        plt.subplot(2, num_subplots, num_sph_harm + 1)
        plt.semilogy(args.sweep_values, abs(np.array(mu)/bolsig_mu-1), 'o-', label='us')
        plt.xlabel(args.sweep_param)
        plt.ylabel("Rel. error in mean energy")
        plt.grid()

        plt.subplot(2, num_subplots, num_sph_harm + 2)
        for col_idx, col in enumerate(args.collisions):
            if bolsig_rates[col_idx] != 0:
                plt.semilogy(args.sweep_values, abs(rates[col_idx]/bolsig_rates[col_idx]-1), 'o-', label=COLLISOIN_NAMES[col])
                # plt.axhline(y=0, label='bolsig '+col, color='k')
        plt.legend()
        plt.xlabel(args.sweep_param)
        plt.ylabel("Rel. error in reaction rates")
        plt.grid()

        # print(M, bolsig_M)
        # print(D, bolsig_D)

        plt.subplot(2, num_subplots, num_subplots + num_sph_harm + 1)
        plt.semilogy(args.sweep_values, abs(np.array(M)/bolsig_M-1), 'o-', label='us')
        plt.xlabel(args.sweep_param)
        plt.ylabel("Rel. error in mobility")
        plt.grid()


        plt.subplot(2, num_subplots, num_subplots + num_sph_harm + 2)
        plt.semilogy(args.sweep_values, abs(np.array(D)/bolsig_D-1), 'o-', label='us')
        plt.xlabel(args.sweep_param)
        plt.ylabel("Rel. error in diffusion coefficient")
        plt.grid()

        
        fig.subplots_adjust(hspace=0.3)
        fig.subplots_adjust(wspace=0.4)

        if (args.radial_poly == "bspline"):
            fig.suptitle("Collisions: " + str(args.collisions) + ", E = " + str(args.E_field)+"V/m"+ "  %.2E"%(args.E_field/collisions.AR_NEUTRAL_N/1e-21)+ "(Td)\n" + ", polys = " + str(args.radial_poly)+", sp_order= " + str(args.spline_order) + ", Nr = " + str(args.NUM_P_RADIAL) + ", bscale = " + str(args.basis_scale) + " (sweeping " + args.sweep_param + ")" + " q_per_knot="+str(args.spline_q_pts_per_knot) + " ne/N = %.2E"%(args.ion_deg) + " Tg = %.2E(K)"%(args.Tg))
            # plt.show()
            if len(spec_sp._basis_p._dg_idx)==2:
                plt.savefig("us_vs_bolsig_cg_" + "_".join(args.collisions) + "_E" + str(args.E_field) + "_poly_" + str(args.radial_poly)+ "_sp_"+ str(args.spline_order) + "_nr" + str(args.NUM_P_RADIAL)+"_qpn_" + str(args.spline_q_pts_per_knot) + "_bscale" + str(args.basis_scale) + "_sweeping_" + args.sweep_param + "_lmax_" + str(args.l_max) +"_ion_deg_%.2E"%(args.ion_deg) + "_Tg%.2E"%(args.Tg) +".png")
            else:
                plt.savefig("us_vs_bolsig_dg_" + "_".join(args.collisions) + "_E" + str(args.E_field) + "_poly_" + str(args.radial_poly)+ "_sp_"+ str(args.spline_order) + "_nr" + str(args.NUM_P_RADIAL)+"_qpn_" + str(args.spline_q_pts_per_knot) + "_bscale" + str(args.basis_scale) + "_sweeping_" + args.sweep_param + "_lmax_" + str(args.l_max)+"_ion_deg_%.2E"%(args.ion_deg) +".png")
        else:
            fig.suptitle("Collisions: " + str(args.collisions) + ", E = " + str(args.E_field) + ", polys = " + str(args.radial_poly) + ", Nr = " + str(args.NUM_P_RADIAL) + ", bscale = " + str(args.basis_scale) + " (sweeping " + args.sweep_param + ")")
            # plt.show()
            plt.savefig("us_vs_bolsig_" + "_".join(args.collisions) + "_E" + str(args.E_field) + "_poly_" + str(args.radial_poly) + "_nr" + str(args.NUM_P_RADIAL) + "_bscale" + str(args.basis_scale) + "_sweeping_" + args.sweep_param + ".png")


        plt.close()





