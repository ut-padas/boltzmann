"""
@package Boltzmann collision operator solver. 
"""
import os
os.environ["OMP_NUM_THREADS"] = "8"
import enum
import scipy
import scipy.optimize
import scipy.interpolate
from   maxpoly import maxpolyserieseval
import basis
import spec_spherical as sp
import numpy as np
import collision_operator_spherical as colOpSp
import collisions 
import parameters as params
from   time import perf_counter as time, sleep
import utils as BEUtils
import argparse
import scipy.integrate
from   scipy.integrate import ode
from   advection_operator_spherical_polys import *
import scipy.ndimage
import matplotlib.pyplot as plt
from   scipy.interpolate import interp1d
import scipy.sparse.linalg
import sys
import bolsig
import csv
from   datetime import datetime

plt.rcParams.update({
    "text.usetex": False,
    "font.size": 12,
    #"ytick.major.size": 3,
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

def normalized_distribution(spec_sp: sp.SpectralExpansionSpherical, mm_op, f_vec, maxwellian,vth):
    c_gamma      = np.sqrt(2*collisions.ELECTRON_CHARGE_MASS_RATIO)
    num_p        = spec_sp._p +1
    # radial_proj  =  BEUtils.compute_radial_components(ev, spec_sp, f_vec, maxwellian, vth, 1)
    # scale        =  1./(np.trapz(radial_proj[0,:]*np.sqrt(ev),x=ev))
    # NUM_Q_VR     = params.BEVelocitySpace.NUM_Q_VR
    # NUM_Q_VT     = params.BEVelocitySpace.NUM_Q_VT
    # NUM_Q_VP     = params.BEVelocitySpace.NUM_Q_VP
    # sph_harm_lm  = params.BEVelocitySpace.SPH_HARM_LM 
    # num_sh = len(sph_harm_lm)
    # gmx_a, gmw_a = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR)
    # mm1_op  = np.array([np.dot(gmw_a, spec_sp.basis_eval_radial(gmx_a, k, 0) * gmx_a**2 ) * 2 * (vth/c_gamma)**3 for k in range(num_p)])

    mm_fac       = spec_sp._sph_harm_real(0, 0, 0, 0) * 4 * np.pi
    scale        = np.dot(f_vec, mm_op / mm_fac) * (2 * (vth/c_gamma)**3)
    #scale         = np.dot(f_vec,mm_op) * maxwellian(0) * vth**3
    return f_vec/scale

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
    ftestt = lambda v,vt,vp : f0_cf(.5*(v*VTH)**2/collisions.ELECTRON_CHARGE_MASS_RATIO)*(1. - fa_cf(.5*(v*VTH)**2/collisions.ELECTRON_CHARGE_MASS_RATIO)*np.cos(vt))
    
    h_bolsig                     =  BEUtils.function_to_basis(spec_sp,ftestt,maxwellian,None,None,None,Minv=Minv)
    h_bolsig                     =  normalized_distribution(spec_sp, mass_op, h_bolsig, maxwellian, vth)
    radial_projection[-1, :, :]  =  BEUtils.compute_radial_components(ev, spec_sp, h_bolsig, maxwellian, vth, 1)
    coeffs_projection.append(h_bolsig)

    #h_init = h_bolsig

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

    FOp      = 0
    sigma_m  = 0
    gg_list  = list()
    fig      = plt.figure(figsize=(10, 10), dpi=300) 
    t1 = time()
    for col_idx, col in enumerate(collisions_included):
        print("collision %d included %s"%(col_idx, col))
        if "g0NoLoss" == col:
            g  = collisions.eAr_G0_NoEnergyLoss()
            g.reset_scattering_direction_sp_mat()
            FOp       = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g, mw_vth, vth_curr)
            sigma_m  += g.total_cross_section(gx_ev)
        elif "g0ConstNoLoss" == col:
            g  = collisions.eAr_G0_NoEnergyLoss(cross_section="g0Const")
            g.reset_scattering_direction_sp_mat()
            FOp       = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g, mw_vth, vth_curr)
            sigma_m  += g.total_cross_section(gx_ev)
        elif "g0" in col:
            g       = collisions.eAr_G0(cross_section=col)
            g.reset_scattering_direction_sp_mat()
            FOp       = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g, mw_vth, vth_curr)
            sigma_m  += g.total_cross_section(gx_ev)
        elif "g2" in col:
            g  = collisions.eAr_G2(cross_section=col)
            g.reset_scattering_direction_sp_mat()
            FOp       = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g, mw_vth, vth_curr)
            sigma_m  += g.total_cross_section(gx_ev)
        else:
            print("%s unknown collision"%(col))
            sys.exit(0)

        plt.loglog(gx_ev, g._total_cs_interp1d(gx_ev) , ".",label="%s (lxcat)"%(col))
        plt.loglog(gx_ev, g.total_cross_section(gx_ev), "x",label="%s (approx)"%(col))
        gg_list.append(g)
    t2 = time()
    print("Assembled the collision op. for Vth : ", vth_curr)
    print("Collision Operator assembly time (s): ",(t2-t1))
    plt.xlabel(r"energy (eV)")
    plt.ylabel(r"cross section (m^2)")
    plt.grid(visible=True)
    plt.legend()
    fig.savefig("cs_%s.svg"%("_".join(collisions_included)))
    plt.close()
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

    u               = mass_op * mw_vth(0) * vth**3 
    u               = np.matmul(np.transpose(u), qA)
    p_vec           = u.reshape((u.shape[0], 1)) / np.sqrt(np.dot(u, u))
    

    ion_deg         = args.ion_deg
    Cmat_p_Emat     = Cmat + Emat
    Cmat_p_Emat     = np.matmul(Minv, Cmat_p_Emat)
    Wmat            = np.dot(u,Cmat_p_Emat)
    #Wmat            = Cmat_p_Emat

    Imat            = np.eye(FOp.shape[0])
    Imat_r          = np.eye(Imat.shape[0]-1)
    Impp            = (Imat - np.outer(p_vec, p_vec))
    Qm,Rm           = np.linalg.qr(Impp)

    Q               = np.delete(Qm,(num_p-1) * num_sh + 0, axis=1)
    R               = np.delete(Rm,(num_p-1) * num_sh + 0, axis=0)
    QT              = np.transpose(Q)

    print("|I - QT Q| = %.8E"%(np.linalg.norm(np.dot(QT,Q)-np.eye(QT.shape[0]))))
    
    h_prev          = np.copy(h_init)
    h_prev          = h_prev / (np.dot(u, h_prev))

    f1              = u / np.dot(u, u)

    eff_rr_op       = BEUtils.reaction_rates_op(spec_sp, gg_list, maxwellian,vth) * collisions.AR_NEUTRAL_N

    g_rate  = np.zeros(FOp.shape[0])
    for col_idx, col in enumerate(collisions_included):
        if "g2" in col:
            g  = collisions.eAr_G2(cross_section=col)
            g.reset_scattering_direction_sp_mat()
            g_rate[0::num_sh] += collisions.AR_NEUTRAL_N *  BEUtils.reaction_rates_op(spec_sp, [g], maxwellian, vth)

    # Wmat[0::num_sh] = np.dot(u[0::num_sh], np.dot(Minv[0::num_sh, 0::num_sh], Cmat[0::num_sh, 0::num_sh]))
    # Wmat[1::num_sh] = 0.0
    
    if args.ee_collisions:
        t1 = time()
        hl_op, gl_op     =  collOp.compute_rosenbluth_potentials_op(maxwellian, vth, 1, Minv)
        cc_op_a, cc_op_b =  collOp.coulomb_collision_op_assembly(maxwellian, vth)
        cc_op            =  np.dot(cc_op_a, hl_op) + np.dot(cc_op_b, gl_op)
        cc_op            =  np.dot(Minv,cc_op.reshape((num_p*num_sh,-1))).reshape((num_p * num_sh, num_p * num_sh, num_p * num_sh))

        cc_op_l1         =  cc_op
        cc_op_l2         =  np.swapaxes(cc_op, 1, 2)

        t2 = time()
        print("Coulomb collision Op. assembly %.8E"%(t2-t1))

    if steady_state:

        def solver_0(h0, rtol = 1e-5 , atol=1e-5, max_iter=1000):
            
            def res_func_cc(x):
                gamma_a     = collisions.AR_NEUTRAL_N * ion_deg * collOp.gamma_a(x, maxwellian, vth, collisions.AR_NEUTRAL_N, ion_deg,eff_rr_op)
                y           = np.dot(QT, np.dot(Cmat_p_Emat  + gamma_a * np.dot(cc_op,x), x)) - np.dot(Wmat,x) * np.dot(QT,x)
                return y
            def res_func(x):
                y           = np.dot(QT, np.dot(Cmat_p_Emat, x)) - np.dot(Wmat,x) * np.dot(QT, x)
                return y
                
            if args.ee_collisions == 1:
                rf = res_func_cc
            else:
                rf = res_func

            abs_error       = 1.0
            rel_error       = 1.0 
            iteration_steps = 0        
            
            h_prev  = np.copy(h0)
            # Q  = np.eye(len(h_prev))
            # QT = Q
            while ((rel_error> rtol and abs_error > atol) and iteration_steps < max_iter):
                if args.ee_collisions:
                    gamma_a     = collisions.AR_NEUTRAL_N * ion_deg * collOp.gamma_a(h_prev, maxwellian, vth, collisions.AR_NEUTRAL_N, ion_deg, eff_rr_op)
                    Lmat        = np.dot(QT, Cmat_p_Emat + gamma_a * np.dot(cc_op_l1,h_prev) + gamma_a * np.dot(cc_op_l2,h_prev)) - 2 * np.dot(Wmat,h_prev) * QT
                    Lmat        = np.dot(Lmat, Q)
                    rhs_vec     = -rf(h_prev)
                else:
                    Lmat        = np.dot(QT, Cmat_p_Emat) - 2 * np.dot(Wmat, h_prev) * QT 
                    Lmat        = np.dot(Lmat, Q)
                    rhs_vec     = -rf(h_prev)
                
                abs_error = np.linalg.norm(rf(h_prev))

                #p      = np.linalg.solve(Lmat,rhs_vec)
                p       = np.linalg.lstsq(Lmat, rhs_vec, rcond=1e-16 /np.linalg.cond(Lmat))[0]
                print("|Jp - b| / |b| = %.8E"%(np.linalg.norm(np.dot(Lmat,p)-rhs_vec)/np.linalg.norm(rhs_vec)))
                p      = np.dot(Q,p)

                alpha  = 1e0
                is_diverged = False

                while (np.linalg.norm(rf(h_prev + alpha * p))  >  abs_error):
                    alpha*=0.5
                    if alpha < 1e-30:
                        is_diverged = True
                        break
                
                if(is_diverged):
                    print("Iteration ", iteration_steps, ": Residual =", abs_error, "line search step size becomes too small")
                    break
                
                h_curr      = h_prev + alpha * p
                
                if iteration_steps % 1 == 0:
                    rel_error = np.linalg.norm(h_prev-h_curr)/np.linalg.norm(h_curr)
                    print("Iteration ", iteration_steps, ": abs residual = %.8E rel residual=%.8E mass =%.8E"%(abs_error, rel_error, np.dot(u, h_curr)))
                

                h_prev      = h_curr
                iteration_steps+=1

            print("Nonlinear solver (1) atol=%.8E , rtol=%.8E"%(abs_error, rel_error))
            return h_curr, abs_error, rel_error

        
        h_curr  , atol, rtol  = solver_0(h_prev, rtol = 1e-13, atol=1e-2, max_iter=300)
        h_pde                 = normalized_distribution(spec_sp, mass_op, h_curr, maxwellian, vth)
        norm_L2               = lambda vv : np.dot(vv, np.dot(Mmat, vv))
        
        def res_func_cc(x):
            gamma_a     = collisions.AR_NEUTRAL_N * ion_deg * collOp.gamma_a(x, maxwellian, vth, collisions.AR_NEUTRAL_N, ion_deg, eff_rr_op)
            y           = np.dot(Cmat_p_Emat  + gamma_a * np.dot(cc_op,x), x) - np.dot(Wmat,x) * x
            return y
        
        def res_func(x):
            y           = np.dot(Cmat_p_Emat, x) - np.dot(Wmat,x) * x
            return y
            
        if args.ee_collisions == 1:
            rf_func = res_func_cc
        else:
            rf_func = res_func

        
        cf_L2    = norm_L2(h_pde-h_bolsig)
        cf_l2    = np.linalg.norm(h_pde-h_bolsig)

        rf_b_L2  = norm_L2(rf_func(h_bolsig))
        rf_p_L2  = norm_L2(rf_func(h_pde))
        
        mm2      = rf_b_L2 / rf_p_L2 
        print("|f_bolsig - f_pde|_L2            = %.8E"%(cf_L2))
        print("|f_bolsig - f_pde|_l2            = %.8E"%(cf_l2))
        print("|R(f_bolsig)|_L2                 = %.8E"%(rf_b_L2))
        print("|R(f_pde)|_L2                    = %.8E"%(rf_p_L2))
        print("|R(f_bolsig)|_L2 / |R(f_pde)|_L2 = %.8E"%(mm2))

        solution_vector      = np.zeros((2,h_init.shape[0]))
        solution_vector[0,:] = normalized_distribution(spec_sp, mass_op, np.matmul(qA, h_init), maxwellian, vth) 
        solution_vector[1,:] = normalized_distribution(spec_sp, mass_op, np.matmul(qA, h_curr), maxwellian, vth) 
        return {'sol':solution_vector, 'h_bolsig': h_bolsig, 'atol': norm_L2(rf_func(h_curr)), 'rtol':rtol, 'tgrid':None, "cf_L2":cf_L2, "cf_l2": cf_l2, "rf_b_L2": rf_b_L2, "rf_p_L2" : rf_p_L2}

    else:
        
        atol   = 1.0
        rtol   = 1.0
        if args.ee_collisions:
            pass
        else:
            Pmat        = Imat_r - dt * np.dot(QT, np.dot(Cmat_p_Emat,Q)) 
            Pmat_inv    = np.linalg.pinv(Pmat, rcond=1e-14/np.linalg.cond(Pmat))

        
        num_time_samples = 5
        tgrid            = np.append(np.array([0]), np.logspace(np.log10(2*dt), np.log10(t_end), num_time_samples-1, base=10)) #np.linspace(0, t_end, num_time_samples)
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

        h_pde                = normalized_distribution(spec_sp, mass_op, h_prev, maxwellian, vth)
        solution_vector[0,:] = h_pde
        

        f1      = u / np.dot(u, u)
        fb_prev = np.dot(R,h_prev)
        if args.ee_collisions:
            while t_curr < t_end:
                if sample_idx < num_time_samples and t_step == tgrid_idx[sample_idx]:
                    h_pde                         = normalized_distribution(spec_sp, mass_op, h_prev, maxwellian, vth)
                    solution_vector[sample_idx,:] = h_pde
                    sample_idx+=1

                gamma_a     = collOp.gamma_a(h_prev, maxwellian, vth, collisions.AR_NEUTRAL_N, ion_deg, eff_rr_op)
                cc_ee_1     = gamma_a * collisions.AR_NEUTRAL_N * ion_deg * np.dot(cc_op_l1, h_prev)
                #cc_ee_2     = gamma_a * collisions.AR_NEUTRAL_N * ion_deg * np.dot(cc_op_l2, h_prev)
                
                
                Lmat        = Cmat_p_Emat + cc_ee_1
                Pmat        = Imat_r - dt * np.dot(QT, np.dot(Lmat ,Q)) #+ dt * np.dot(u,np.dot(Wmat,h_prev)) * Imat_r
                rhs_vec     = fb_prev + dt * np.dot(np.dot(QT, Lmat), f1) - dt * np.dot(np.dot(Wmat, h_prev) * QT, h_prev)

                fb_curr     = np.linalg.solve(Pmat, rhs_vec) 
                h_curr      = f1 + np.dot(Q,fb_curr)
                
                rtol= (np.linalg.norm(h_prev - h_curr))/np.linalg.norm(h_curr)
                atol= (np.linalg.norm(h_prev - h_curr))

                t_curr+= dt
                t_step+=1
                h_prev  = h_curr
                fb_prev = fb_curr

                if t_step%100 == 0: 
                    print("time = %.3E solution convergence atol = %.8E rtol = %.8E mass %.10E"%(t_curr, atol, rtol, np.dot(u,h_curr)))
        else:
            while t_curr < t_end:
                if sample_idx < num_time_samples and t_step == tgrid_idx[sample_idx]:
                    h_pde                         = normalized_distribution(spec_sp, mass_op, h_prev, maxwellian, vth)
                    solution_vector[sample_idx,:] = h_pde
                    sample_idx+=1

                # fully implicit on mass growth term
                # Pmat        = Imat_r - dt * np.dot(QT, np.dot(Cmat_p_Emat,Q)) + dt * np.dot(u,np.dot(Wmat,h_prev)) * Imat_r
                # Pmat_inv    = np.linalg.inv(Pmat)
                # rhs_vec     = fb_prev + dt * np.dot((np.dot(QT, Cmat_p_Emat) - np.dot(u,np.dot(Wmat,h_prev)) * QT),f1)

                # semi-explit on mass growth term (only need to compute the inverse matrix once)
                rhs_vec     = fb_prev + dt * np.dot(np.dot(QT, Cmat_p_Emat),f1) - dt * np.dot(np.dot(Wmat,h_prev) * QT, h_prev)

                fb_curr     = np.linalg.solve(Pmat, rhs_vec) #np.dot(Pmat_inv,rhs_vec)
                h_curr  = f1 + np.dot(Q,fb_curr)

                rtol= (np.linalg.norm(h_prev - h_curr))/np.linalg.norm(h_curr)
                atol= (np.linalg.norm(h_prev - h_curr))

                t_curr+= dt
                t_step+=1
                h_prev  = h_curr
                fb_prev = fb_curr

                if t_step%100 == 0: 
                    print("time = %.3E solution convergence atol = %.8E rtol = %.8E mass %.10E"%(t_curr, atol, rtol, np.dot(u,h_curr)))

        h_pde    = normalized_distribution(spec_sp, mass_op, h_curr, maxwellian, vth)
        solution_vector[-1,:] = h_pde
        norm_L2  = lambda vv : np.dot(vv, np.dot(Mmat, vv))
        cf_L2    = norm_L2(h_pde-h_bolsig)
        cf_l2    = np.linalg.norm(h_pde-h_bolsig)

        rf_b_L2  = 0
        rf_p_L2  = 0
        return {'sol':solution_vector, 'h_bolsig': h_bolsig, 'atol': atol, 'rtol':rtol, 'tgrid':tgrid, "cf_L2":cf_L2, "cf_l2": cf_l2, "rf_b_L2": rf_b_L2, "rf_p_L2" : rf_p_L2}
        

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
    
    h_bolsig                     = BEUtils.function_to_basis(spec_sp,ftestt,maxwellian,None,None,None,Minv=Minv)
    h_bolsig                     = normalized_distribution(spec_sp, mass_op, h_bolsig, maxwellian, vth)
    radial_projection[i, :, :]   = BEUtils.compute_radial_components(ev, spec_sp, h_bolsig, maxwellian, VTH, 1)
    coeffs_projection.append(h_bolsig)

    FOp      = 0
    g_rate   = np.zeros(num_p * num_sh)
    

    k_vec    = spec_sp._basis_p._t
    dg_idx   = spec_sp._basis_p._dg_idx
    sp_order = spec_sp._basis_p._sp_order

    sigma_m  = np.zeros(len(gx)+1)
    gx_ev    = (gx * MVTH / c_gamma)**2
    
    FOp      = 0
    sigma_m  = 0
    t1 = time()
    for col_idx, col in enumerate(collisions_included):
        print("collision %d included %s"%(col_idx, col))
        if "g0NoLoss" == col:
            g  = collisions.eAr_G0_NoEnergyLoss()
            g.reset_scattering_direction_sp_mat()
            FOp       = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g, mw_vth, vth_curr)
            sigma_m  += g.total_cross_section(gx_ev)
        elif "g0ConstNoLoss" == col:
            g  = collisions.eAr_G0_NoEnergyLoss(cross_section="g0Const")
            g.reset_scattering_direction_sp_mat()
            FOp       = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g, mw_vth, vth_curr)
            sigma_m  += g.total_cross_section(gx_ev)
        elif "g0" in col:
            g       = collisions.eAr_G0(cross_section=col)
            g.reset_scattering_direction_sp_mat()
            FOp       = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g, mw_vth, vth_curr)
            sigma_m  += g.total_cross_section(gx_ev)
        elif "g2" in col:
            g  = collisions.eAr_G2(cross_section=col)
            g.reset_scattering_direction_sp_mat()
            FOp       = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g, mw_vth, vth_curr)
            sigma_m  += g.total_cross_section(gx_ev)
        else:
            print("%s unknown collision"%(col))
            sys.exit(0)
    t2 = time()
    print("Assembled the collision op. for Vth : ", vth_curr)
    print("Collision Operator assembly time (s): ",(t2-t1))
    FOp       =  FOp[0::num_sh, 0::num_sh]
    Minv      = Minv[0::num_sh, 0::num_sh]
    Mmat      = Mmat[0::num_sh, 0::num_sh]

    u    = mass_op / (np.sqrt(np.pi)**3)
    
    sigma_m   = sigma_m * np.sqrt(gx_ev) * c_gamma * collisions.AR_NEUTRAL_N
    
    h_prev    = np.array(h_init[0::num_sh])
    u         = u[0::num_sh]
    h_prev    = h_prev / np.dot(u, h_prev)

    p_vec     = u / np.sqrt(np.dot(u, u))

    Imat      = np.eye(p_vec.shape[0])
    Imat_r    = np.eye(Imat.shape[0]-1)
    Impp      = (Imat - np.outer(p_vec, p_vec))
    Qm,Rm     = np.linalg.qr(Impp)

    Q         = np.delete(Qm,(num_p-1), axis=1)
    R         = np.delete(Rm,(num_p-1), axis=0)
    QT        = np.transpose(Q)
    f1        = u / np.dot(u, u)
    
    print("Initial mass = %.12E"%(np.dot(u, h_prev)))
    print("|I - QT Q| = %.8E"%(np.linalg.norm(np.dot(QT,Q)-np.eye(QT.shape[0]))))
    print("|Impp -QR| = %.8E"%(np.linalg.norm(Impp - np.dot(Q, R))))

    g_rate = np.matmul(u.reshape(1,-1), np.matmul(Minv, FOp))
    # g_rate  = np.zeros(FOp.shape[0])
    # for col_idx, col in enumerate(collisions_included):
    #     if "g2" in col:
    #         g  = collisions.eAr_G2(cross_section=col)
    #         g.reset_scattering_direction_sp_mat()
    #         g_rate += collisions.AR_NEUTRAL_N *  BEUtils.reaction_rates_op(spec_sp, [g], maxwellian, vth)
    
    # print(g_rate1)
    # print(g_rate)

    Ev     = E_field * collisions.ELECTRON_CHARGE_MASS_RATIO
    Vq_d1  = spec_sp.Vdq_r(gx,0,1,1)
    Wt     = Mmat 
    
    ### if we need to apply some boundary conditions. 
    # fx      = k_vec[num_p -4 :  num_p] 
    # #fd_coef = [0.5, -2, 1.5]
    # fd_coef = [-1/3, 3/2, -3, 11/6]
    # bdy_op  = 0
    # for ii in range(4):
    #     bdy_op+=np.array([ fd_coef[ii] * spec_sp.basis_eval_radial(fx[ii], p, 0) for p in range(num_p)])
    # dx_bdy = (fx[-1]-fx[-2])
    # bdy_op = bdy_op/dx_bdy

    def assemble_f1_op(h_prev):
        g_rate_mu = np.dot(g_rate, h_prev)
        w_ev =  1./(sigma_m + g_rate_mu)
        Bt = np.array([ w_ev * (gx**2) * spec_sp.basis_eval_radial(gx, p, 0) * Vq_d1[k,:] for p in range(num_p) for k in range(num_p)]).reshape((num_p, num_p, -1))
        Bt = np.dot(Bt, gw) 
        return Bt

    if steady_state:

        def residual_op(x):
            g_rate_mu = np.dot(g_rate, x) 
            w_ev =  1./(sigma_m + g_rate_mu)
            tmp  = np.sqrt(w_ev * gw) * gx * Vq_d1
            At   = np.matmul(tmp, np.transpose(tmp))

            #print("mass = ",np.dot(u,x))

            # fx = k_vec[dg_idx[-1] + sp_order]
            # assert fx == k_vec[-1], "flux assembly face coords does not match at the boundary"
            # At[dg_idx[-1] , dg_idx[-1]] += -fx**2 * spec_sp.basis_derivative_eval_radial(fx - 2 * np.finfo(float).eps, dg_idx[-1],0,1) / (sigma_bdy + g_rate_mu) 
            
            Rop = -(1/3) * ((Ev/MVTH)**2) * (At) + FOp -  Wt * g_rate_mu
            return Rop

        def residual_func(x):
            Rop = residual_op(x)
            y   = np.dot(Rop,x)
            return y

        def jacobian_func(x):
            g_rate_mu = np.dot(g_rate, x) 
            w_ev =  1./(sigma_m + g_rate_mu)
            tmp  = np.sqrt(w_ev * gw) * gx * Vq_d1
            At   = np.matmul(tmp, np.transpose(tmp))

            # fx = k_vec[dg_idx[-1] + sp_order]
            # assert fx == k_vec[-1], "flux assembly face coords does not match at the boundary"
            # At[dg_idx[-1] , dg_idx[-1]] += -fx**2 * spec_sp.basis_derivative_eval_radial(fx - 2 * np.finfo(float).eps, dg_idx[-1],0,1) / (sigma_bdy + g_rate_mu) 
            
            Ji = (-(1/3) * ((Ev/MVTH)**2) * (At) + FOp -  2 * Wt * g_rate_mu)
            return Ji
        
        def solver_0(h0, rtol = 1e-5 , atol=1e-5, max_iter=1000):
            nn              = FOp.shape[0]
            Ji              = np.zeros((nn+1,nn))
            Rf              = np.zeros(nn+1)
        
            abs_tol = 1
            rel_tol = 1 
            iteration_steps = 0
            h_prev          = np.copy(h0)
            rf_initial      = np.linalg.norm(residual_func(h_prev))
            while (rel_tol > rtol and iteration_steps < max_iter):
                Rf              = residual_func(h_prev)
                #Rf              = np.append(Rf,np.dot(u,h_prev)-1) 
                abs_tol         = np.linalg.norm(Rf)
                rel_tol         = abs_tol/rf_initial
                         
                if(iteration_steps%1==0):
                    print("iteration %d abs. res l2 = %.8E rel. res = %.8E mass = %.8E"%(iteration_steps, abs_tol, rel_tol, np.dot(u,h_prev)))
                    
                Ji = jacobian_func(h_prev)
                #Ji = np.vstack((Ji,u))
                p  = np.matmul(np.linalg.pinv(Ji,rcond=1e-16/np.linalg.cond(Ji)), -Rf)
                
                alpha=1e0
                is_diverged = False
                while (np.linalg.norm(residual_func(h_prev + alpha *p)) >= abs_tol):
                    alpha*=0.5
                    if alpha < 1e-16:
                        is_diverged = True
                        break
                
                if(is_diverged):
                    print("iteration %d abs. res l2 = %.8E rel. res = %.8E"%(iteration_steps, abs_tol, rel_tol))
                    print("line search step size becomes too small")
                    break

                h_prev += p*alpha
                h_prev  /= np.dot(u,h_prev)
                iteration_steps = iteration_steps + 1
            
            hh = spec_sp.create_vec().reshape(num_p * num_sh)
            hh[0::num_sh] = h_prev

            Bt = assemble_f1_op(h_prev)
            hh[1::num_sh] = (Ev/np.sqrt(3)) * np.matmul(Minv,np.matmul(Bt,h_prev)) / MVTH

            return hh, abs_tol, rel_tol

        ### attempt to do mass projection, for some reason this does not work. 
        def solver_1(h0, rtol = 1e-5 , atol=1e-5, max_iter=1000):
            abs_tol = 1
            rel_tol = 1 
            
            iteration_steps = 0
            h_prev       = np.copy(h0)
            rf_initial   = np.linalg.norm(residual_func(h_prev))

            while (rel_tol > rtol and iteration_steps < max_iter):
                Rop             = residual_op(h_prev)
                Rf              = np.dot(Rop, h_prev)
                abs_tol         = np.linalg.norm(Rf)
                rel_tol         = abs_tol/rf_initial
                         
                if(iteration_steps%1==0):
                    print("iteration %d abs. res l2 = %.8E rel. res = %.8E mass = %.8E"%(iteration_steps, abs_tol, rel_tol, np.dot(u,h_prev)))
                    
                Ji = np.dot(QT, np.dot(Rop, Q))
                p  = np.linalg.solve(Ji , -np.dot(QT, Rf))
                p  = np.dot(Q,p)
                #p  = np.dot(Q, np.dot(QT,p))
                
                alpha=1e0
                is_diverged = False
                # while (np.linalg.norm(residual_func(h_prev + alpha *p)) >= abs_tol):
                #     alpha*=0.5
                #     if alpha < 1e-16:
                #         is_diverged = True
                #         break
                
                if(is_diverged):
                    print("iteration %d abs. res l2 = %.8E rel. res = %.8E"%(iteration_steps, abs_tol, rel_tol))
                    print("line search step size becomes too small")
                    break

                h_prev += p*alpha
                iteration_steps = iteration_steps + 1
            
            hh = spec_sp.create_vec().reshape(num_p * num_sh)
            hh[0::num_sh] = h_prev

            Bt = assemble_f1_op(h_prev)
            hh[1::num_sh] = (Ev/np.sqrt(3)) * np.matmul(Minv,np.matmul(Bt,h_prev)) / MVTH

            return hh, abs_tol, rel_tol

        hh, abs_tol, rel_tol  = solver_0(h_prev, rtol=1e-8, atol=1e-4, max_iter=100)
        h_pde                 = normalized_distribution(spec_sp, mass_op, hh, maxwellian, vth)
        norm_L2               = lambda vv : np.dot(vv, np.dot(Mmat, vv))

        rf_func  = residual_func
        cf_L2    = norm_L2(h_pde[0::num_sh]-h_bolsig[0::num_sh])
        cf_l2    = np.linalg.norm(h_pde-h_bolsig)

        rf_b_L2  = norm_L2(rf_func(h_bolsig[0::num_sh]))
        rf_p_L2  = norm_L2(rf_func(h_pde[0::num_sh]))
        
        mm2      = rf_b_L2 / rf_p_L2 
        print("|f_bolsig - f_pde|_L2            = %.8E"%(cf_L2))
        print("|f_bolsig - f_pde|_l2            = %.8E"%(cf_l2))
        print("|R(f_bolsig)|_L2                 = %.8E"%(rf_b_L2))
        print("|R(f_pde)|_L2                    = %.8E"%(rf_p_L2))
        print("|R(f_bolsig)|_L2 / |R(f_pde)|_L2 = %.8E"%(mm2))

        solution_vector = np.zeros((2,h_init.shape[0]))
        solution_vector[0,:] = normalized_distribution(spec_sp, mass_op, h_init, maxwellian, vth)
        solution_vector[1,:] = normalized_distribution(spec_sp, mass_op, hh,     maxwellian, vth)
        return {'sol':solution_vector, 'h_bolsig': h_bolsig, 'atol': abs_tol, 'rtol':rel_tol, 'tgrid':None, "cf_L2":cf_L2, "cf_l2": cf_l2, "rf_b_L2": rf_b_L2, "rf_p_L2" : rf_p_L2}

    else:
        raise NotImplementedError
   

parser = argparse.ArgumentParser()

parser.add_argument("-Nr", "--NUM_P_RADIAL"                       , help="Number of polynomials in radial direction", type=int, default=16)
parser.add_argument("-T", "--T_END"                               , help="Simulation time", type=float, default=1e-4)
parser.add_argument("-dt", "--T_DT"                               , help="Simulation time step size ", type=float, default=1e-7)
parser.add_argument("-o",  "--out_fname"                          , help="output file name", type=str, default='coll_op')
parser.add_argument("-ts_tol", "--ts_tol"                         , help="adaptive timestep tolerance", type=float, default=1e-15)
parser.add_argument("-l_max", "--l_max"                           , help="max polar modes in SH expansion", type=int, default=1)
parser.add_argument("-c", "--collisions"                          , help="collisions included (g0, g0Const, g0NoLoss, g2, g2Const)",nargs='+', type=str, default=["g0Const"])
parser.add_argument("-ev", "--electron_volt"                      , help="initial electron volt", type=float, default=0.25)
parser.add_argument("-bscale", "--basis_scale"                    , help="basis electron volt", type=float, default=1.0)
parser.add_argument("-q_vr", "--quad_radial"                      , help="quadrature in r"        , type=int, default=200)
parser.add_argument("-q_vt", "--quad_theta"                       , help="quadrature in polar"    , type=int, default=8)
parser.add_argument("-q_vp", "--quad_phi"                         , help="quadrature in azimuthal", type=int, default=8)
parser.add_argument("-q_st", "--quad_s_theta"                     , help="quadrature in scattering polar"    , type=int, default=8)
parser.add_argument("-q_sp", "--quad_s_phi"                       , help="quadrature in scattering azimuthal", type=int, default=8)
parser.add_argument("-radial_poly", "--radial_poly"               , help="radial basis", type=str, default="bspline")
parser.add_argument("-sp_order", "--spline_order"                 , help="b-spline order", type=int, default=1)
parser.add_argument("-spline_qpts", "--spline_q_pts_per_knot"     , help="q points per knots", type=int, default=7)
parser.add_argument("-E", "--E_field"                             , help="Electric field in V/m", type=float, default=100)
parser.add_argument("-dv", "--dv_target"                          , help="target displacement of distribution in v_th units", type=float, default=0)
parser.add_argument("-nt", "--num_timesteps"                      , help="target number of time steps", type=float, default=100)
parser.add_argument("-steady", "--steady_state"                   , help="Steady state or transient", type=int, default=1)
parser.add_argument("-run_bolsig_only", "--run_bolsig_only"       , help="run the bolsig code only", type=bool, default=False)
parser.add_argument("-bolsig", "--bolsig_dir"                     , help="Bolsig directory", type=str, default="../../Bolsig/")
parser.add_argument("-sweep_values", "--sweep_values"             , help="Values for parameter sweep", nargs='+', type=float, default=[24, 48, 96])
parser.add_argument("-sweep_param", "--sweep_param"               , help="Paramter to sweep: Nr, ev, bscale, E, radial_poly", type=str, default="Nr")
parser.add_argument("-dg", "--use_dg"                             , help="enable dg splines", type=int, default=0)
parser.add_argument("-Tg", "--Tg"                                 , help="Gass temperature (K)" , type=float, default=1e-12)
parser.add_argument("-ion_deg", "--ion_deg"                       , help="Ionization degreee"   , type=float, default=1)
parser.add_argument("-store_eedf", "--store_eedf"                 , help="store eedf"   , type=int, default=0)
parser.add_argument("-store_csv", "--store_csv"                   , help="store csv format of QoI comparisons", type=int, default=0)
parser.add_argument("-ee_collisions", "--ee_collisions"           , help="Enable electron-electron collisions", type=float, default=1)
parser.add_argument("-bolsig_precision", "--bolsig_precision"     , help="precision value for bolsig code", type=float, default=1e-11)
parser.add_argument("-bolsig_convergence", "--bolsig_convergence" , help="convergence value for bolsig code", type=float, default=1e-8)
parser.add_argument("-bolsig_grid_pts", "--bolsig_grid_pts"       , help="grid points for bolsig code"      , type=int, default=1024)


args                = parser.parse_args()
#EbyN_Td             = np.array([1, 5, 20, 50])
#EbyN_Td             = np.array([20])
#e_values           = np.logspace(np.log10(0.148), np.log10(4e4), 60, base=10) 
#e_values            = EbyN_Td * collisions.AR_NEUTRAL_N * 1e-21
#ion_deg_values      = np.array([1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10])
#ion_deg_values      = np.array([0.0])
#ion_deg_values     = np.array([0.0])
e_values            = np.array([args.E_field])
ion_deg_values      = np.array([args.ion_deg])

SAVE_EEDF    = args.store_eedf
SAVE_CSV     = args.store_csv

if not args.ee_collisions:
    ion_deg_values *= 0.0

run_params          = [(e_values[i], ion_deg_values[j]) for i in range(len(e_values)) for j in range(len(ion_deg_values))]
print(run_params)
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
COLLISOIN_NAMES["g2Regul"] = "ionization"

for run_id in range(len(run_params)):
    args.E_field = run_params[run_id][0] #e_values[run_id]
    args.ion_deg = run_params[run_id][1] #e_values[run_id]
    print(args)

    try:
        bolsig.run_bolsig(args)
        [bolsig_ev_, bolsig_f0, bolsig_a, bolsig_E, bolsig_mu, bolsig_M, bolsig_D, bolsig_rates,bolsig_cclog] = bolsig.parse_bolsig(args.bolsig_dir+"argon.out",len(args.collisions))
        bolsig_ev  = bolsig_ev_ 
        # bolsig_ev  = np.linspace(bolsig_ev_[0], bolsig_ev_[-1], np.int64((bolsig_ev_[-1]- bolsig_ev_[0])/1e-4))
        # bolsig_f0  = scipy.interpolate.interp1d(bolsig_ev_, bolsig_f0)(bolsig_ev)
        # bolsig_a   = scipy.interpolate.interp1d(bolsig_ev_, bolsig_a)(bolsig_ev)

    except:
        print(args.bolsig_dir+"argon.out file not found due to Bolsig+ run faliure")
        sys.exit(0)
    
    if (args.run_bolsig_only):
        sys.exit(0)    

    # setting electron volts from bolsig results for now
    print("bolsig temp      = %.8E"%((bolsig_mu /1.5)))
    print("bolsig mobility  = %.8E"%((bolsig_M)))
    print("bolsig diffusion = %.8E"%((bolsig_D)))
    print("bolsig coulomb logarithm = %.8E"%((bolsig_cclog)))
    print("bolsig collision rates")
    for  col_idx, col in enumerate(args.collisions):
        print("%s = %.8E"%(COLLISOIN_NAMES[col], bolsig_rates[col_idx]))

    #print("setting PDE code ev va")
    args.electron_volt = (bolsig_mu/1.5) 

    run_data=list()
    run_temp=list()
    # v = np.linspace(-2,2,100)
    # vx, vz = np.meshgrid(v,v,indexing='ij')
    # vy = np.zeros_like(vx)
    # v_sph_coord = BEUtils.cartesian_to_spherical(vx, vy, vz)

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

    # density_slice         = np.zeros((len(args.sweep_values),len(vx[0]),len(vx[1])))
    # density_slice_initial = np.zeros((len(args.sweep_values),len(vx[0]),len(vx[1])))

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
            
        ev_range = ((0*VTH/c_gamma)**2, (1.2) * ev[-1])
        #ev_range = ((0*VTH/c_gamma)**2, (6*VTH/c_gamma)**2)
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
            print("----------knots in energy (ev)----------")
            print((bb._t_unique * VTH/c_gamma)**2)
            print("----------knots in energy (ev)----------")
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
        #rdata      = solve_bte(args.steady_state, cf, maxwellian, VTH, args.E_field, args.T_END, args.T_DT, args.ts_tol, collisions_included=args.collisions)
        rdata_list.append(rdata)

        data      = rdata['sol']
        h_bolsig  = rdata['h_bolsig']
        abs_tol   = rdata['atol']
        rel_tol   = rdata['rtol']    

        radial_base[i,:,:]  = BEUtils.compute_radial_components(ev, spec_sp, data[0,:], maxwellian, VTH, 1)
        radial[i, :, :]     = BEUtils.compute_radial_components(ev, spec_sp, data[-1,:], maxwellian, VTH, 1)
        
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
        print("PDE code found temp (ev) = %.8E mean energy (ev) = %.8E"%(temp_evolution[-1] , 1.5*temp_evolution[-1]))

        g_list  = list()
        for col_idx, col in enumerate(args.collisions):
            if "g0NoLoss" == col:
                g  = collisions.eAr_G0_NoEnergyLoss()
                g.reset_scattering_direction_sp_mat()
            elif "g0ConstNoLoss" == col:
                g  = collisions.eAr_G0_NoEnergyLoss(cross_section="g0Const")
                g.reset_scattering_direction_sp_mat()
            elif "g0" in str(col):
                g = collisions.eAr_G0(cross_section=col)
            elif "g2" in str(col):
                g = collisions.eAr_G2(cross_section=col)
            else:
                print("unknown collision %s"%(col))
                sys.exit(0)
            
            g_list.append(g)
            rr_op        = BEUtils.reaction_rates_op(spec_sp, [g], current_mw, current_vth)
            rates[col_idx].append(np.dot(rr_op,data[-1,0::num_sph_harm]))

        mobility_op  = BEUtils.mobility_op(spec_sp, current_mw, current_vth)
        diffusion_op = BEUtils.diffusion_op(spec_sp, g_list, current_mw, current_vth)

        c_gamma = np.sqrt(2.*collisions.ELECTRON_CHARGE_MASS_RATIO)
        D.append((c_gamma / 3.) * np.dot(diffusion_op,data[-1, 0::num_sph_harm]))
        M.append(-(c_gamma / (3 * (args.E_field / collisions.AR_NEUTRAL_N))) * np.dot(mobility_op, np.sqrt(3) * data[-1,1::num_sph_harm]))

        # total_cs = 0
        # for col_idx, col in enumerate(args.collisions):
        #     if "g0NoLoss" == col:
        #         g  = collisions.eAr_G0_NoEnergyLoss()
        #         g.reset_scattering_direction_sp_mat()
        #     elif "g0ConstNoLoss" == col:
        #         g  = collisions.eAr_G0_NoEnergyLoss(cross_section="g0Const")
        #         g.reset_scattering_direction_sp_mat()
        #     elif "g0" in str(col):
        #         g = collisions.eAr_G0(cross_section=col)
        #     elif "g2" in str(col):
        #         g = collisions.eAr_G2(cross_section=col)
        #     else:
        #         print("unknown collision %s"%(col))
        #         sys.exit(0)
        #     cs        = g.total_cross_section(ev)
        #     total_cs += cs
        #     rr_op     = BEUtils.reaction_rates_op(spec_sp, g, current_mw, current_vth)
        #     #print(np.sqrt(2.*collisions.ELECTRON_CHARGE_MASS_RATIO)*np.trapz(radial[i,0,:]*ev*cs,x=ev), np.dot(rr_op,data[-1,0::num_sph_harm]))
        #     rates[col_idx].append(np.dot(rr_op,data[-1,0::num_sph_harm])) 
        #     #rates[col_idx].append(np.sqrt(2.*collisions.ELECTRON_CHARGE_MASS_RATIO)*np.trapz(radial[i,0,:]*ev*cs,x=ev)) 
        # c_gamma = np.sqrt(2.*collisions.ELECTRON_CHARGE_MASS_RATIO)
        # #### diffusion coefficient 
        # D.append((c_gamma / 3.) * np.trapz(radial[i,0,:] * ev / total_cs,x=ev))
        # f1 = radial[i,1,:] * np.sqrt(3)
        # M.append( -(c_gamma / (3 * (args.E_field / collisions.AR_NEUTRAL_N))) * np.trapz(f1 * ev ,x=ev))

        for  col_idx, col in enumerate(args.collisions):
            print("%s \t bolsig = %.8E from PDE = %.8E " %(COLLISOIN_NAMES[col], bolsig_rates[col_idx], rates[col_idx][-1]))

        print("D from bolisg = %.8E from PDE = %.8E"%(bolsig_D, D[-1]))
        print("M from bolisg = %.8E from PDE = %.8E"%(bolsig_M, M[-1]))
        run_temp.append(temp_evolution)

    if SAVE_CSV:
        with open("pde_vs_bolsig_%s.csv"%str_datetime, 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            if run_id == 0:
                # write the header
                header = ["E/N(Td)","E(V/m)","Nr","energy","diffusion","mobility","bolsig_energy","bolsig_defussion","bolsig_mobility","l2_f0","l2_f1", "Tg", "ion_deg", "atol", "rtol","cf_L2", "cf_l2", "Rf_b_L2", "Rf_p_L2","residual_ratio"]
                for g in args.collisions:
                    header.append(str(g))
                    header.append("bolsig_"+str(g))

                writer.writerow(header)

            for i, value in enumerate(args.sweep_values):
                # write the data
                bolsig_f1 =  bolsig_f0*bolsig_a * spec_sp._sph_harm_real(0, 0, 0, 0)/spec_sp._sph_harm_real(1, 0, 0, 0)

                l2_f0     = np.linalg.norm(np.abs(radial[i, 0])-np.abs(bolsig_f0))/np.linalg.norm(np.abs(bolsig_f0))
                l2_f1     = np.linalg.norm(np.abs(radial[i, 1])-np.abs(bolsig_f1))/np.linalg.norm(np.abs(bolsig_f1))

                data = [args.E_field/collisions.AR_NEUTRAL_N/1e-21, args.E_field, args.sweep_values[i], mu[i], D[i], M[i], bolsig_mu, bolsig_D, bolsig_M, l2_f0, l2_f1, args.Tg, args.ion_deg, solver_tol[i][0], solver_tol[i][1],rdata["cf_L2"], rdata["cf_l2"], rdata["rf_b_L2"], rdata["rf_p_L2"], (rdata["rf_b_L2"]/rdata["rf_p_L2"])]
                for col_idx , _ in enumerate(args.collisions):
                    data.append(rates[col_idx][i])
                    data.append(bolsig_rates[col_idx])

                writer.writerow(data)
        

    if SAVE_EEDF:
        with open('eedf_%s.npy'%(str_datetime), 'ab') as f:
            np.save(f, np.array([spec_sp._p + 1]))
            np.save(f, np.array([spec_sp._sph_harm_lm[-1][0]]))
            np.save(f, np.array([args.E_field]))
            np.save(f, np.array([args.ion_deg]))
            np.save(f, np.array([args.Tg]))
            np.save(f, ev)

            for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
                np.save(f, radial[-1,lm_idx,:])
                
            np.save(f, bolsig_f0)
            np.save(f, bolsig_f1)

    if (1):
        num_subplots = num_sph_harm + 2 + 2 + 1
        num_plt_cols = 4
        num_plt_rows = np.int64(np.ceil(num_subplots/num_plt_cols))
        
        fig       = plt.figure(figsize=(num_plt_cols * 5 + 0.5*(num_plt_cols-1), num_plt_rows * 5 + 0.5*(num_plt_rows-1)), dpi=300)
        bolsig_f1 = abs(bolsig_f0*bolsig_a * spec_sp._sph_harm_real(0, 0, 0, 0)/spec_sp._sph_harm_real(1, 0, 0, 0))

        #f0
        plt.subplot(num_plt_rows, num_plt_cols,  1)
        plt.semilogy(bolsig_ev,  abs(bolsig_f0), '-k', label="bolsig")
        # f1
        plt.subplot(num_plt_rows, num_plt_cols,  2)
        plt.semilogy(bolsig_ev,  abs(bolsig_f1), '-k', label="bolsig")

        pde_vs_bolsig_L2 = list()
        for i, value in enumerate(args.sweep_values):
            data=run_data[i]
            data_projection=coeffs_projection[i]
            pde_vs_bolsig_L2.append(rdata_list[i]["cf_l2"])

            lbl = args.sweep_param+"="+str(value)

            if args.steady_state == 0:
                rdata   = rdata_list[i]
                data_tt = rdata['sol']
                tgrid   = list(rdata['tgrid'])

                radial_tt = np.zeros((len(tgrid), num_sph_harm, len(ev)))

                for t_idx, tt in enumerate(tgrid):
                    radial_tt[t_idx, :, : ] = BEUtils.compute_radial_components(ev, spec_sp, data_tt[t_idx,:], maxwellian, VTH, 1)

            # spherical components plots
            plt_idx=1
            for l_idx in range(num_sph_harm):

                # plt.subplot(2, num_subplots, num_subplots + 1+l_idx)
                # color = next(plt.gca()._get_lines.prop_cycler)['color']
                # plt.plot(np.abs(data[-1,l_idx::num_sph_harm]),label=lbl, color=color)
                
                # plt.title(label="l=%d"%l_idx)
                # plt.yscale('log')
                # plt.xlabel("coeff #")
                # plt.ylabel("abs(coeff)")
                # plt.grid(visible=True)
                # if l_idx == 0:
                #     plt.legend()

                plt.subplot(num_plt_rows, num_plt_cols, plt_idx)
                color = next(plt.gca()._get_lines.prop_cycler)['color']
                plt.semilogy(ev,  abs(radial[i, l_idx]), '-', label=lbl, color=color)

                if args.steady_state == 0:
                    for t_idx, tt in enumerate(tgrid[:-1]):
                        color = next(plt.gca()._get_lines.prop_cycler)['color']
                        # print(tt)
                        # print(np.abs(radial_tt[t_idx, l_idx]))
                        plt.semilogy(ev,  abs(radial_tt[t_idx, l_idx]), '--', label=lbl+" t=%.2E"%(tt), color=color)

                
                plt.xlabel(r"energy (eV)")
                plt.ylabel(r"radial component")
                plt.title("f%d"%(l_idx))
                plt.grid(visible=True)
                if l_idx == 0:
                    plt.legend(prop={'size': 8})
                
                plt_idx+=1

            plt.subplot(num_plt_rows, num_plt_cols, plt_idx)
            plt.semilogy(ev,  abs(radial[i, 0]/bolsig_f0-1), '-', label=lbl, color=color)
            plt.ylim((None, 1))
            plt.ylabel(r"relative error")
            plt.xlabel(r"evergy (eV)")
            plt.grid(visible=True)
            plt.title("f0 (PDE vs. Bolsig)")
            
            plt.subplot(num_plt_rows, num_plt_cols, plt_idx + 1)
            plt.semilogy(ev,  abs(abs(radial[i, 1])/bolsig_f1-1), '-', label=lbl, color=color)
            plt.ylim((None, 1))
            plt.ylabel(r"relative error")
            plt.xlabel(r"evergy (eV)")
            plt.grid(visible=True)
            plt.title("f1 (PDE vs. Bolsig)")

            plt.subplot(num_plt_rows, num_plt_cols, plt_idx + 2)
            plt.semilogy(ev,  np.abs(np.abs(radial[i, 0])/np.abs(radial[-1, 0])-1), '-', label=lbl, color=color)
            plt.ylabel(r"relative error")
            plt.xlabel(r"evergy (eV)")
            plt.grid(visible=True)
            plt.title("f0 (PDE vs. PDE)")

            plt.subplot(num_plt_rows, num_plt_cols, plt_idx + 3)
            plt.semilogy(ev,  np.abs(np.abs(radial[i, 1])/np.abs(radial[-1, 1])-1), '-', label=lbl, color=color)
            plt.ylabel(r"relative error")
            plt.xlabel(r"evergy (eV)")
            plt.grid(visible=True)
            plt.title("f1 (PDE vs. PDE)")

                

        plt.subplot(num_plt_rows, num_plt_cols, plt_idx + 4)
        plt.semilogy(args.sweep_values, abs(np.array(mu)/bolsig_mu-1), 'o-', label='mean energy')
        
        for col_idx, col in enumerate(args.collisions):
            if bolsig_rates[col_idx] != 0:
                plt.semilogy(args.sweep_values, abs(rates[col_idx]/bolsig_rates[col_idx]-1), 'o-', label=COLLISOIN_NAMES[col])
        
        plt.semilogy(args.sweep_values, abs(np.array(M)/bolsig_M-1), 'o-', label='mobility')
        #plt.semilogy(args.sweep_values, abs(np.array(pde_vs_bolsig_L2)), 'o-', label='L2 (f_pde-f_bolsig)')
        #plt.semilogy(args.sweep_values, abs(np.array(D)/bolsig_D-1), 'o-', label='diffusion')
        plt.xlabel(args.sweep_param)
        plt.ylabel(r"relative error")
        plt.title("PDE vs. Bolsig")
        plt.legend()
        plt.grid()

        plt.subplot(num_plt_rows, num_plt_cols, plt_idx + 5)
        plt.semilogy(args.sweep_values, abs(np.array(mu)/mu[-1]-1), 'o-', label='mean energy')
        
        for col_idx, col in enumerate(args.collisions):
            if bolsig_rates[col_idx] != 0:
                plt.semilogy(args.sweep_values, abs(rates[col_idx]/rates[col_idx][-1]-1), 'o-', label=COLLISOIN_NAMES[col])
        
        plt.semilogy(args.sweep_values, abs(np.array(M)/M[-1]-1), 'o-', label='mobility')
        #plt.semilogy(args.sweep_values, abs(np.array(D)/bolsig_D-1), 'o-', label='diffusion')
        plt.xlabel(args.sweep_param)
        plt.ylabel(r"relative error")
        plt.title("PDE vs. PDE")
        plt.legend()
        plt.grid()

        


        if (args.radial_poly == "bspline"):
            fig.suptitle("E=%.4EV/m  E/N=%.4ETd ne/N=%.2E gas temp.=%.2EK, N=%.4E $m^{-3}$"%(args.E_field, args.E_field/collisions.AR_NEUTRAL_N/1e-21, args.ion_deg, args.Tg, collisions.AR_NEUTRAL_N))
            # plt.show()
            if len(spec_sp._basis_p._dg_idx)==2:
                if args.steady_state == 1 : 
                    plt.savefig("us_vs_bolsig_cg_" + "_".join(args.collisions) + "_E" + str(args.E_field) + "_poly_" + str(args.radial_poly)+ "_sp_"+ str(args.spline_order) + "_nr" + str(args.NUM_P_RADIAL)+"_qpn_" + str(args.spline_q_pts_per_knot) + "_bscale" + str(args.basis_scale) + "_sweeping_" + args.sweep_param + "_lmax_" + str(args.l_max) +"_ion_deg_%.2E"%(args.ion_deg) + "_Tg%.2E"%(args.Tg) +".svg")
                else:
                    plt.savefig("us_vs_bolsig_cg_" + "_".join(args.collisions) + "_E" + str(args.E_field) + "_poly_" + str(args.radial_poly)+ "_sp_"+ str(args.spline_order) + "_nr" + str(args.NUM_P_RADIAL)+"_qpn_" + str(args.spline_q_pts_per_knot) + "_bscale" + str(args.basis_scale) + "_sweeping_" + args.sweep_param + "_lmax_" + str(args.l_max) +"_ion_deg_%.2E"%(args.ion_deg) + "_Tg%.2E"%(args.Tg)+"_ts%.2E_T%.2E"%(args.T_DT, args.T_END) +".svg")
            else:
                plt.savefig("us_vs_bolsig_dg_" + "_".join(args.collisions) + "_E" + str(args.E_field) + "_poly_" + str(args.radial_poly)+ "_sp_"+ str(args.spline_order) + "_nr" + str(args.NUM_P_RADIAL)+"_qpn_" + str(args.spline_q_pts_per_knot) + "_bscale" + str(args.basis_scale) + "_sweeping_" + args.sweep_param + "_lmax_" + str(args.l_max)+"_ion_deg_%.2E"%(args.ion_deg) +".png")
        else:
            fig.suptitle("Collisions: " + str(args.collisions) + ", E = " + str(args.E_field) + ", polys = " + str(args.radial_poly) + ", Nr = " + str(args.NUM_P_RADIAL) + ", bscale = " + str(args.basis_scale) + " (sweeping " + args.sweep_param + ")")
            # plt.show()
            plt.savefig("us_vs_bolsig_" + "_".join(args.collisions) + "_E" + str(args.E_field) + "_poly_" + str(args.radial_poly) + "_nr" + str(args.NUM_P_RADIAL) + "_bscale" + str(args.basis_scale) + "_sweeping_" + args.sweep_param + ".png")


        plt.close()





