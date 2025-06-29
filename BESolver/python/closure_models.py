"""
Module to handle different closure models for the electron BTE
    1. Maximum entropy closure
    2. 0D-BTE closure
    
"""
import numpy as np
import cross_section
import spec_spherical as sp
import basis
import utils as bte_utils
import plot_scripts.plot_utils as plot_utils
import scipy.constants
import matplotlib.pyplot as plt
import h5py
import os
import sys

def quadrature_grid(spec_sp : sp.SpectralExpansionSpherical, num_vr, num_vt, num_vp, coord="spherical"):
    [glx,glw]    = basis.Legendre().Gauss_Pn(num_vt)
    vt_q, vt_w   = np.arccos(glx), glw
    
    [glx,glw]    = basis.Legendre().Gauss_Pn(num_vp)
    vp_q, vp_w   = np.pi * glx + np.pi , np.pi * glw

    vr_q, vr_w   = spec_sp._basis_p.Gauss_Pn(num_vr)

    vg           = np.meshgrid(vr_q, vt_q, vp_q, indexing="ij")

    if (coord == "spherical"):
        return vg, (vr_w, vt_w, vp_w)
    elif (coord == "cartesian"):
        vg = (vg[0] * np.sin(vg[1]) * np.cos(vg[2]), vg[0] * np.sin(vg[1]) * np.sin(vg[2]), vg[0] * np.cos(vg[1]))
        return vg, (vr_w, vt_w, vp_w) 
    else:
        raise NotImplementedError
    
def max_entropy_reconstruction(spec_sp:sp.SpectralExpansionSpherical, x0, num_vr, num_vt, num_vp, m_vec : np.array, m_funcs : list, xp:np, rtol, atol, iter_max):
    num_m = m_vec.shape[0]
    num_x = m_vec.shape[1]

    ###
    # We seek solutions of the form R = (<exp(\lambda_k \phi_k) \phi> - m)
    # find lambda values that minimizes the R. i.e., maximum entropy approximation that best
    # match the moments
    ###

    x                             = xp.copy(x0)
    vg, (vr_w, vt_w, vp_w)        = quadrature_grid(spec_sp, num_vr, num_vt, num_vp)
    mf                            = np.zeros(tuple([num_m]) + vg[0].shape)

    for i in range(num_m):
       mf[i]  = m_funcs[i](vg[0], vg[1], vg[2])
    
    if spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:
        wx    = vg[0]**2 
        beta  = 0.0 
        nfac  = 2.0

    elif spec_sp.get_radial_basis_type() == basis.BasisType.LAGUERRE:
        wx   = np.ones_like(vg[0])
        beta = 1.0
        nfac = 4 * xp.pi**1.5 
    else:
        raise NotImplementedError

    m_vec_l2 = xp.linalg.norm(m_vec, axis=1)
    #print(m_vec_l2)
    # print("gme : ")
    # print(m_vec)
    #m_vec_l2[m_vec_l2 < 1e-10] = 1.0
    rtheta = 1e-8
    
    def residual(x):
        xmf = xp.einsum("ix,iklm->xklm", x, mf)
        y   = 2 * xp.exp(xmf) * (xp.exp(beta * vg[0]**2) * wx)[xp.newaxis, :, :, :]
        y   = xp.einsum("iklm,xklm->ixklm", mf, y)
        y   = xp.einsum("ixklm,k,l,m->ix", y, vr_w, vt_w, vp_w) 

        # rx  = xp.zeros_like(x)
        # rx[2, x[2]>0] = x[2][x[2]>0]
        return y - m_vec + rtheta * x**3

    def jacobian(x):
        xmf = xp.einsum("ix,iklm->xklm", x, mf)
        y   = 2 * xp.exp(xmf) * (xp.exp(beta * vg[0]**2) * wx)[xp.newaxis, :, :, :]
        y   = xp.einsum("iklm,jklm,xklm->ijxklm", mf, mf, y)
        y   = xp.einsum("ijxklm,k,l,m->ijx", y, vr_w, vt_w, vp_w)
        y   = (y.reshape((num_m * num_m, num_x)).T).reshape((num_x, num_m, num_m))
        # rx  = xp.zeros((num_x, num_m, num_m))
        # mx   = 0.0 * xp.eye(num_m)
        # mx[2, 2] = 1.0
        # rx[x[2]>0]  = xp.einsum("x,ij->xij", xp.ones_like(x[2, x[2]>0]), mx) 
        I   = xp.eye(num_m)
        y   += xp.einsum("ix,ij->xij", 3 * rtheta * x**2, I)
        return y
    
        
    alpha = 1e0
    eta   = 1e-8
    rcond = 1e-14
    while (alpha > 1e-1):
        x         = xp.copy(x0)
        count     = 0
        r0        = residual(x)
        rr        = xp.copy(r0)
        norm_rr   = norm_r0 = xp.linalg.norm(r0, axis=1)
        converged = ((norm_rr/m_vec_l2 < rtol).all() or (norm_rr < atol).all())
        

        
        print("iter = %04d norm_rr = %.4E norm_rr/norm_r0 = %.4E"%(count, xp.max(norm_rr), xp.max(norm_rr/m_vec_l2)))
        while( not converged and (count < iter_max) ):
            jmat     = jacobian(x)
            #jinv     = xp.linalg.pinv(jmat, rcond=rcond)
            jinv     = xp.linalg.inv(jmat)
            pk       = xp.einsum("xil,lx->ix", jinv, rr)
            xk       = x  - alpha * pk
            rk       = residual(xk)
            norm_rk  = xp.linalg.norm(rk, axis=1)
            norm_pk  = xp.linalg.norm(pk, axis=1)

            if ((xp.linalg.norm(pk)< eta).all()):
                print("terminted due to singular jacobian (%.4E)"%(xp.max(norm_pk)))
            ## line search
            # while ( alpha > 1e-8 ):
            #     xk       = x  - alpha * xp.einsum("xil,lx->ix", jinv, rr)
            #     rk       = residual(xk)
            #     norm_rk  = xp.linalg.norm(rk, axis=1)

            #     # print(norm_rk)
            #     # print(norm_rr)
            #     # print(alpha)
        
            #     if ( (norm_rk < norm_rr).any() ):
            #         break
            #     else:
            #         alpha = 0.25 * alpha

            if (xp.isnan(norm_rr).any()==True):
                break
            # if((norm_rr > norm_r0).all()):
            #     break
        
        
            x         = xk
            rr        = rk
            norm_rr   = norm_rk
            count    += 1
            converged = ((norm_rr/m_vec_l2 < rtol).all() or (norm_rr < atol).all()) or (xp.linalg.norm(pk) < eta).all()
            # print(norm_rr.shape, m_vec_l2.shape, (norm_rr/m_vec_l2).shape)
            #print(norm_rr, norm_rr/m_vec_l2, x)
            #print(xp.linalg.norm(pk))
            #print("iter: ", iter, x, rr)
            if(count % 10 ==0):
                print("iter = %04d norm_rr = %.4E norm_rr/norm_r0 = %.4E"%(count, xp.max(norm_rr), xp.max(norm_rr/m_vec_l2)))
            
            
        if (not converged):
            alpha = alpha * 0.1 
        else:
            break
        #print(rr)
    

    print("x: \n", x, x.shape)
    #print("norm_rr: \n", norm_rr, norm_rr.shape)
    print("m_vec: \n", m_vec, m_vec.shape)
    print("rr: \n", rr, rr.shape)
    # print("norm_rr: \n", norm_rr, norm_rr.shape)
    # #print("norm_rr: \n", norm_rr)
    # if (not converged):
    #     assert False

    print("converged at iter = %04d norm_rr = %.4E norm_rr/norm_r0 = %.4E"%(count, xp.max(norm_rr), xp.max(norm_rr/m_vec_l2)))

    num_p        = spec_sp._p +1
    sph_harm_lm  = spec_sp._sph_harm_lm
    num_sh       = len(sph_harm_lm)
    mm           = spec_sp.compute_mass_matrix()

    xmf          = xp.einsum("ix,iklm->xklm", x, mf)
    q            = nfac * xp.exp(xmf) * (xp.exp(beta * vg[0]**2) * wx)[xp.newaxis, :, :, :]

    Vq           = xp.zeros(tuple([num_p * num_sh]) + vg[0].shape)
    
    for p in range(num_p):
        for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
            Vq[p * num_sh + lm_idx] = spec_sp.basis_eval_radial(vg[0], p, l=0) * spec_sp.basis_eval_spherical(vg[1], vg[2], lm[0], lm[1])
    
    fklm = np.einsum("ixklm,k,l,m->ix", np.einsum("iklm,xklm->ixklm", Vq, q), vr_w, vt_w, vp_w)
    fklm = xp.linalg.solve(mm, fklm)
    return x, fklm
    
def relative_entropy(spec_sp_psh: sp.SpectralExpansionSpherical, psh,
                     spec_sp_qsh: sp.SpectralExpansionSpherical, qsh,
                     qg, xp:np, type="KL"):
    """
    psh : true distribution function
    qsh : model distribution function
    """

    [glx,glw]    = basis.Legendre().Gauss_Pn(qg[1])
    vt_q, vt_w   = np.arccos(glx), glw
    
    [glx,glw]    = basis.Legendre().Gauss_Pn(qg[2])
    vp_q, vp_w   = np.pi * glx + np.pi , np.pi * glw

    vr_q, vr_w   = spec_sp_psh._basis_p.Gauss_Pn(qg[0])
    vg           = np.meshgrid(vr_q, vt_q, vp_q, indexing="ij")


    if spec_sp_psh.get_radial_basis_type() == basis.BasisType.SPLINES:
        wx    = vg[0]**2 
        beta  = 0.0 
        nfac  = 1.0
    elif spec_sp_psh.get_radial_basis_type() == basis.BasisType.LAGUERRE:
        wx   = np.ones_like(vg[0])
        beta = 1.0
        nfac = 4 * xp.pi**1.5
    else:
        raise NotImplementedError
    
    def compute_vq(spec_sp, vg):
        num_p        = spec_sp._p +1
        sph_harm_lm  = spec_sp._sph_harm_lm
        num_sh       = len(sph_harm_lm)
        Vq           = xp.zeros(tuple([num_p * num_sh]) + vg[0].shape)
        
        for p in range(num_p):
            for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
                Vq[p * num_sh + lm_idx] = spec_sp.basis_eval_radial(vg[0], p, l=0) * spec_sp.basis_eval_spherical(vg[1], vg[2], lm[0], lm[1])

        return Vq
    

    psh_vg = nfac * xp.einsum("ix,iklm->xklm", psh, compute_vq(spec_sp_psh, vg))
    qsh_vg = nfac * xp.einsum("ix,iklm->xklm", qsh, compute_vq(spec_sp_qsh, vg))

    psh_vg[psh_vg < 0] = 1e-15
    qsh_vg[qsh_vg < 0] = 1e-15

    return xp.einsum("xklm,k,l,m", wx * psh_vg * xp.log(xp.abs(psh_vg/qsh_vg)), vr_w, vt_w, vp_w)

def bte_0d_closure(args, Ef, fprefix):
    """
    computes the time normalized EEDFs with 0D bte
    """
    
    try:
        import cupy as cp
        #CUDA_NUM_DEVICES=cp.cuda.runtime.getDeviceCount()
    except:
        print("cupy is required for 0D-BTE closure model")
        raise ModuleNotFoundError
    
    from   bte_0d3v_batched import bte_0d3v_batched
    import glow1d_utils

    class A:
        par_file = args["par_file"].replace("'", "")

    class bte_0d_params():
        param           = glow1d_utils.parameters(A)
        threads         = 4
        out_fname       = fprefix
        solver_type     = "transient"
        l_max           = int(args["l_max"])
        collisions      = "lxcat_data/eAr_crs.Biagi.3sp2r" #args["collisions"]
        sp_order        = int(args["sp_order"])
        spline_qpts     = int(args["spline_qpts"])
        atol            = float(args["atol"])
        rtol            = float(args["rtol"])
        max_iter        = 1000
        Te              = float(args["Te"])
        n0              = param.n0 * param.np0
        ee_collisions   = 0 
        use_dg          = 0
        ev_max          = float(args["ev_max"])
        Nr              = int(args["Nr"])
        profile         = 0
        store_eedf      = 1
        store_csv       = 1
        plot_data       = 1
        verbose         = 1
        use_gpu         = 1
        cycles          = 100
        dt              = 1e-2
        Efreq           = 13.56e6
        n_grids         = 1
        n_pts           = int(args["Np"])

    args_0d_bte = bte_0d_params()
    print(args_0d_bte.collisions, args_0d_bte.out_fname)
    n_grids     = bte_0d_params.n_grids
    lm_modes    = [[l,0] for l in range(args_0d_bte.l_max+1)]    
    bte_solver  = bte_0d3v_batched(args_0d_bte, np.array([args_0d_bte.ev_max]),
                                         np.array([args_0d_bte.Te]),
                                         np.array([args_0d_bte.Nr], dtype=np.int32), [lm_modes], args_0d_bte.n_grids, [args_0d_bte.collisions])
    
    grid_idx    = 0
    n_pts       = args_0d_bte.n_pts

    bte_solver.assemble_operators(grid_idx)
    f0          = bte_solver.initialize(grid_idx, n_pts,"maxwellian")

    bte_solver.set_boltzmann_parameter(grid_idx, "n0"       , np.ones(n_pts) * args_0d_bte.n0)
    bte_solver.set_boltzmann_parameter(grid_idx, "ne"       , np.ones(n_pts) * 1.0)
    bte_solver.set_boltzmann_parameter(grid_idx, "ns_by_n0" , np.ones(n_pts))
    bte_solver.set_boltzmann_parameter(grid_idx, "Tg"       , np.ones(n_pts) * args_0d_bte.param.Tg)
    bte_solver.set_boltzmann_parameter(grid_idx, "eRe"      , np.ones(n_pts) * 0.0)
    bte_solver.set_boltzmann_parameter(grid_idx, "eIm"      , np.ones(n_pts) * 0.0)
    bte_solver.set_boltzmann_parameter(grid_idx, "f0"       , f0)
    bte_solver.set_boltzmann_parameter(grid_idx,  "E"       , np.ones(n_pts) * 0.0)

    bte_solver.set_boltzmann_parameter(grid_idx, "u0"       , f0)
    bte_solver.set_boltzmann_parameter(grid_idx, "u1"       , 0 * f0)


    if args_0d_bte.use_gpu==1:
        dev_id   = 0
        bte_solver.host_to_device_setup(dev_id, grid_idx)
    
    xp     = bte_solver.xp_module
    Ex, Ek = Ef[0], Ef[1]
    
    Ex     = xp.asarray(Ex)
    Ek     = xp.asarray(Ek)

    def Et(t):
        at = xp.array([[xp.cos(2 * xp.pi * Ek[i] * t ), xp.sin(2 * xp.pi * Ek[i] * t)] for i in range(len(Ek))])
        return xp.sum(Ex * at[:, :, xp.newaxis], axis=(0, 1))

    def solve_step(time, delta_t, Et):
        """
        perform a single timestep in 0d-BTE
        """
        with cp.cuda.Device(dev_id):
            print(Et(time), Et(time).shape)
            bte_solver.set_boltzmann_parameter(grid_idx, "E", Et(time))
            u0    = bte_solver.get_boltzmann_parameter(grid_idx, "u0")
            v     = bte_solver.step(grid_idx, u0, args_0d_bte.atol, args_0d_bte.rtol, args_0d_bte.max_iter, time, delta_t)
            bte_solver.set_boltzmann_parameter(grid_idx, "u1", v)

        return 

    tt               = 0
    dt               = args_0d_bte.dt
    steps_per_cycle  = int(1/dt)
    max_steps        = args_0d_bte.cycles * steps_per_cycle + 1
    print("0d bte : steps_per_cycles = %d max cycles = %d"%(steps_per_cycle, args_0d_bte.cycles))

    
    bte_u            = [0 for i in range(n_grids)]
    bte_v            = [0 for i in range(n_grids)]
    u_avg            = [0 for i in range(n_grids)]
    abs_error        = [0 for i in range(n_grids)]
    rel_error        = [0 for i in range(n_grids)]
    cycle_f1         = (0.5 * dt/ (steps_per_cycle * dt))
    xp               = bte_solver.xp_module
    tt               = 0
    with cp.cuda.Device(dev_id):
        for tidx in range(0, max_steps):   
            if (tidx % steps_per_cycle == 0):
                u0      = bte_solver.get_boltzmann_parameter(grid_idx, "u0")
                abs_error[grid_idx] = xp.max(xp.abs(bte_v[grid_idx]-u0))
                rel_error[grid_idx] = abs_error[grid_idx] / xp.max(xp.abs(u0))
                bte_v[grid_idx] = xp.copy(u0)

                print("cycle = %04d ||u0-u1|| = %.4E  ||u0-u1||/||u0|| = %.4E"%(tidx//steps_per_cycle, abs_error[grid_idx], rel_error[grid_idx]))

                if max(abs_error) < args_0d_bte.atol or max(rel_error)< args_0d_bte.rtol:
                    break

                if tidx < (max_steps-1):
                    u_avg  = [0 for i in range(n_grids)]
            
            if tidx == (max_steps-1):
                break    

            u_avg[grid_idx] += cycle_f1 * bte_solver.get_boltzmann_parameter(grid_idx, "u0")
            solve_step(tt, dt, Et)
            u_avg[grid_idx] += cycle_f1 * bte_solver.get_boltzmann_parameter(grid_idx, "u1")
            bte_solver.set_boltzmann_parameter(grid_idx, "u0", bte_solver.get_boltzmann_parameter(grid_idx, "u1"))
            tt+=dt

    return u_avg[grid_idx]
    # do the time-steping till convergence
















