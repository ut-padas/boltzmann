"""
@package : Spectral based, Petrov-Galerkin discretization of the collission integral
"""
import abc
import basis
import spec_spherical as sp
import collisions
import scipy
import scipy.constants
import numpy as np
import time
#import quadpy
import utils as BEUtils
import scipy.integrate
import scipy.sparse
import sym_cc
import sympy
from multiprocess import Pool

class CollissionOp(abc.ABC):

    def __init__(self,dim,p_order):
        self._dim = dim
        self._p = p_order
        pass

    @abc.abstractmethod
    def assemble_mat(collision,maxwellian):
        pass

class CollisionOpSP():
    """
    3D- velocity space, collision operator for species s-collisions
    with background heavy particles, where heavy particles obey the
    direc delta distribution function. 
    """
    
    def __init__(self, spec_sp:sp.SpectralExpansionSpherical) -> None:
        
        self._spec              = spec_sp
        self._r_basis_type      = spec_sp.get_radial_basis_type()
        
        self._num_p               = spec_sp._p +1
        self._num_sh              = len(spec_sp._sph_harm_lm)
        self._sph_harm_lm         = spec_sp._sph_harm_lm  
        

        # self._gmx,self._gmw     = spec_sp._basis_p.Gauss_Pn(self._NUM_Q_VR)
        # legendre                = basis.Legendre()
        # self._glx,self._glw     = legendre.Gauss_Pn(self._NUM_Q_VT)
        # self._VTheta_q          = np.arccos(self._glx)
        # self._VPhi_q            = np.linspace(0,2*np.pi,self._NUM_Q_VP)

        # self._glx_s,self._glw_s = legendre.Gauss_Pn(self._NUM_Q_CHI)
        # self._Chi_q             = np.arccos(self._glx_s)
        # self._Phi_q             = np.linspace(0,2*np.pi,self._NUM_Q_PHI)

        # assert self._NUM_Q_VP  > 1 
        # assert self._NUM_Q_PHI > 1
        
        # sq_fac_v = (2*np.pi/(self._NUM_Q_VP-1))
        # sq_fac_s = (2*np.pi/(self._NUM_Q_PHI-1))

        # self._WPhi_q   = np.ones(self._NUM_Q_PHI)*sq_fac_s
        # self._WVPhi_q  = np.ones(self._NUM_Q_VP)*sq_fac_v

        # #trap. weights
        # self._WPhi_q[0]  = 0.5 * self._WPhi_q[0]
        # self._WPhi_q[-1] = 0.5 * self._WPhi_q[-1]

        # self._WVPhi_q[0]  = 0.5 * self._WVPhi_q[0]
        # self._WVPhi_q[-1] = 0.5 * self._WVPhi_q[-1]

        self._NUM_Q_VR = self._spec._num_q_radial
        return 

    def _LOp_eulerian_radial_only(self, collision, maxwellian, vth, tgK, mp_pool_sz):
        V_TH          = vth     
        g             = collision
        spec_sp       = self._spec
        num_p         = spec_sp._p+1
        num_sh        = len(spec_sp._sph_harm_lm)
        
        if self._r_basis_type == basis.BasisType.SPLINES:
            k_vec        = spec_sp._basis_p._t
            dg_idx       = spec_sp._basis_p._dg_idx
            sp_order     = spec_sp._basis_p._sp_order
            cc_collision = spec_sp.create_mat()
            c_gamma      = np.sqrt(2*collisions.ELECTRON_CHARGE_MASS_RATIO)

            k_vec_uq     = np.unique(k_vec)
            k_vec_dx     = k_vec_uq[1] - k_vec_uq[0]

            if len(dg_idx) == 2:
                # continuous basis grid
                gx_e , gw_e  = spec_sp._basis_p.Gauss_Pn(self._NUM_Q_VR)
                if(g._type == collisions.CollisionType.EAR_G0):
                    total_cs = g.total_cross_section((gx_e * V_TH / c_gamma)**2) 
                    c_mu     = 2 * g._mByM 
                    v_scale  = np.sqrt(1- c_mu)
                    v_post   = gx_e * v_scale
                    kappa    = (scipy.constants.Boltzmann * tgK * c_mu * 0.5 / scipy.constants.electron_mass) / V_TH
                    
                    def t1(p, k):
                        k_min  = k_vec[k]
                        k_max  = k_vec[k + sp_order + 1]
                        qx_idx = np.logical_and(gx_e >= k_min, gx_e <= k_max)

                        gmx    = gx_e[qx_idx] 
                        gmw    = gw_e[qx_idx]
                        vp     = v_post[qx_idx]
                        t_cs   = total_cs[qx_idx]

                        phi_k    = spec_sp.basis_eval_radial(gmx, k, 0)
                        dx_phi_k = spec_sp.basis_derivative_eval_radial(gmx, k, 0,1)
                        
                        psi_pp       = spec_sp.basis_eval_radial(vp , p, 0)
                        psi_pm       = spec_sp.basis_eval_radial(gmx, p, 0)
                        q0           = np.dot(gmw, V_TH  * gmx**3 * t_cs * phi_k * (psi_pp-psi_pm)  - kappa * gmx **3 * t_cs * spec_sp.basis_derivative_eval_radial(gmx, p, 0, 1) * dx_phi_k)
                        q1           = np.dot(gmw, -V_TH * gmx**3 * t_cs * phi_k * psi_pm)
                        
                        return (p, k, q0, q1)
                        
                    with Pool(mp_pool_sz) as process_pool:
                        result = process_pool.starmap(t1,[(p,k) for p in range(num_p) for k in range(num_p)])
                    
                    tmp_q0   = np.zeros((num_p,num_p))
                    tmp_q1   = np.zeros((num_p,num_p))
                    
                    for r in result:
                        tmp_q0[r[0], r[1]] = r[2]
                        tmp_q1[r[0], r[1]] = r[3]

                    for qs_idx, (q,s) in enumerate(self._sph_harm_lm):
                        if q==0:
                            for p in range(num_p):
                                for k in range(num_p):
                                    cc_collision[p * num_sh + qs_idx , k * num_sh + qs_idx] =tmp_q0[p,k]
                        else:
                            for p in range(num_p):
                                for k in range(num_p):
                                    cc_collision[p * num_sh + qs_idx , k * num_sh + qs_idx] =tmp_q1[p,k]

                elif(g._type == collisions.CollisionType.EAR_G1 or g._type == collisions.CollisionType.EAR_G2 or g._type == collisions.CollisionType.EAR_G3):
                    
                    energy_split = 1.0
                    if g._type == collisions.CollisionType.EAR_G2:
                        energy_split= 2.0

                    if (g._type == collisions.CollisionType.EAR_G1 or g._type == collisions.CollisionType.EAR_G2):
                        check_1          = (gx_e * V_TH/c_gamma)**2 >= g._reaction_threshold
                        gx_e             = gx_e[check_1]
                        gw_e             = gw_e[check_1]
                        v_post           = c_gamma * np.sqrt( (1/energy_split) * ((gx_e * V_TH /c_gamma)**2  - g._reaction_threshold)) / V_TH
                    
                    # elif (g._type == collisions.CollisionType.EAR_G3):
                    #     v_post           = c_gamma * np.sqrt( (1/energy_split) * ((gx_e * V_TH /c_gamma)**2  + g._attachment_energy)) / V_TH
                    else:
                        raise ValueError

                    total_cs         = g.total_cross_section((gx_e * V_TH / c_gamma)**2) 
                    def t1(p, k):
                        k_min   = k_vec[k]
                        k_max   = k_vec[k + sp_order + 1]
                        qx_idx  = np.logical_and(gx_e >= k_min, gx_e <= k_max)
                    
                        gmx     = gx_e[qx_idx] 
                        gmw     = gw_e[qx_idx]
                        vp      = v_post[qx_idx]
                        t_cs    = total_cs[qx_idx] 

                        phi_k   = spec_sp.basis_eval_radial(gmx, k, 0)

                        psi_pp  = spec_sp.basis_eval_radial(vp , p, 0)
                        psi_pm  = spec_sp.basis_eval_radial(gmx, p, 0)
                        q0      = np.dot(gmw,  V_TH * gmx**3 * t_cs * phi_k * (energy_split * psi_pp - psi_pm))
                        q1      = np.dot(gmw, -V_TH * gmx**3 * t_cs * phi_k * psi_pm)
                        
                        return (p, k, q0, q1)
                        
                    with Pool(mp_pool_sz) as process_pool:
                        result = process_pool.starmap(t1,[(p,k) for p in range(num_p) for k in range(num_p)])
                    
                    tmp_q0   = np.zeros((num_p,num_p))
                    tmp_q1   = np.zeros((num_p,num_p))
                    
                    for r in result:
                        tmp_q0[r[0], r[1]] = r[2]
                        tmp_q1[r[0], r[1]] = r[3]
                    
                    # tmp_q0   = np.zeros((num_p,num_p))
                    # tmp_q1   = np.zeros((num_p,num_p))

                    # for k in range(num_p):
                    #     k_min  = k_vec[k]
                    #     k_max  = k_vec[k + sp_order + 1]
                    #     qx_idx = np.logical_and(gx_e >= k_min, gx_e <= k_max)
                        
                    #     gmx    = gx_e[qx_idx] 
                    #     gmw    = gw_e[qx_idx]
                    #     vp     = v_post[qx_idx]
                    #     t_cs   = total_cs[qx_idx] 

                    #     phi_k  = spec_sp.basis_eval_radial(gmx, k, 0)

                    #     for p in range(num_p):
                    #         psi_pp       = spec_sp.basis_eval_radial(vp , p, 0)
                    #         psi_pm       = spec_sp.basis_eval_radial(gmx, p, 0)
                    #         tmp_q0[p,k]  = np.dot(gmw,  V_TH * gmx**3 * t_cs * phi_k * (energy_split * psi_pp - psi_pm))
                    #         tmp_q1[p,k]  = np.dot(gmw, -V_TH * gmx**3 * t_cs * phi_k * psi_pm)

                    for qs_idx, (q,s) in enumerate(self._sph_harm_lm):
                        if q==0:
                            for p in range(num_p):
                                for k in range(num_p):
                                    cc_collision[p * num_sh + qs_idx , k * num_sh + qs_idx] =tmp_q0[p,k]
                        else:
                            for p in range(num_p):
                                for k in range(num_p):
                                    cc_collision[p * num_sh + qs_idx , k * num_sh + qs_idx] =tmp_q1[p,k]

                else:
                    raise NotImplementedError
            else:
                # discontinuous basis grid 
                #gx_e , gw_e  = spec_sp._basis_p.Gauss_Pn(self._NUM_Q_VR)
                #total_cs     = g.total_cross_section((gx_e * V_TH / c_gamma)**2)

                dg_q_per_knot = sp_order + 2
                if(g._type == collisions.CollisionType.EAR_G0):

                    c_mu     = 2 * g._mByM 
                    v_scale  = np.sqrt(1- c_mu)
                    kappa    = (scipy.constants.Boltzmann * tgK * c_mu * 0.5 / scipy.constants.electron_mass) / V_TH

                    tmp_q0   = np.zeros((num_p,num_p))
                    tmp_q1   = np.zeros((num_p,num_p))

                    gx_m     = np.array([])
                    gw_m     = np.array([])

                    gx_p     = np.array([])
                    gw_p     = np.array([])


                    for e_id in range(0,len(dg_idx),2):
                        ib = dg_idx[e_id]
                        ie = dg_idx[e_id+1]

                        xb = k_vec[ib]
                        xe = k_vec[ie+sp_order+1]

                        xb_eps = xb / v_scale
                        xe_eps = xe / v_scale

                        if xb_eps > xb:
                            a0, b0 = basis.gauss_legendre_quad(dg_q_per_knot, xb, xb_eps)
                            gx_m   = np.append(gx_m, a0)
                            gw_m   = np.append(gw_m, b0)
                            #print("e_id=%d (%.4E,%.4E) sp_idx pre (%d,%d) dx = (%.6E, %.6E)"%(e_id,xb,xe,ib,ie, xb, xb_eps))
                            
                        # computing the collision loss term xb to xe domain integration
                        for ii in range(ib, ie + sp_order + 1):
                            if k_vec[ii]< k_vec[ii+1]:
                                a0, b0 = basis.gauss_legendre_quad(dg_q_per_knot, max(xb_eps,k_vec[ii]), k_vec[ii+1])

                                gx_m = np.append(gx_m, a0)
                                gw_m = np.append(gw_m, b0)

                                gx_p = np.append(gx_p, a0)
                                gw_p = np.append(gw_p, b0)
                                #print("e_id=%d (%.4E,%.4E) internal dx = (%.6E, %.6E)"%(e_id,xb, xe, max(xb_eps,k_vec[ii]), k_vec[ii+1]))

                        if xe_eps > xe:
                            a0, b0 = basis.gauss_legendre_quad(dg_q_per_knot, xe, xe_eps)
                            gx_p   = np.append(gx_p, a0)
                            gw_p   = np.append(gw_p, b0)
                            #print("e_id=%d (%.4E,%.4E) sp_idx post (%d,%d) dx = (%.6E, %.6E)"%(e_id,xb,xe,ib,ie, xe, xe_eps))

                    total_cs_m  = g.total_cross_section((gx_m * V_TH / c_gamma)**2)
                    total_cs_p  = g.total_cross_section((gx_p * V_TH / c_gamma)**2)

                    v_post_p    = gx_p * v_scale

                    for e_id in range(0,len(dg_idx),2):
                        ib = dg_idx[e_id]
                        ie = dg_idx[e_id+1]

                        xb = k_vec[ib]
                        xe = k_vec[ie+sp_order+1]

                        # computing the collision loss term xb to xe domain integration
                        for k in range(ib,ie+1):
                            k_min  = k_vec[k]
                            k_max  = k_vec[k + sp_order + 1]
                            qx_idx = np.logical_and(gx_m >= k_min, gx_m <= k_max)

                            gmx    = gx_m[qx_idx] 
                            gmw    = gw_m[qx_idx]
                            t_cs   = total_cs_m[qx_idx]

                            phi_k    = spec_sp.basis_eval_radial(gmx, k, 0)
                            dx_phi_k = spec_sp.basis_derivative_eval_radial(gmx, k, 0,1)

                            for p in range(ib,ie+1):
                                psi_pm       = spec_sp.basis_eval_radial(gmx, p, 0)
                                tmp_q0[p,k]  += np.dot(gmw, -V_TH * gmx**3 * t_cs * phi_k * psi_pm  - kappa * gmx **3 * t_cs * spec_sp.basis_derivative_eval_radial(gmx, p, 0, 1) * dx_phi_k)
                                tmp_q1[p,k]  += np.dot(gmw, -V_TH * gmx**3 * t_cs * phi_k * psi_pm)

                        for k in range(ib, min(ie + sp_order + 2, num_p)):
                            k_min  = k_vec[k]
                            k_max  = k_vec[k + sp_order + 1]
                            qx_idx = np.logical_and(gx_p >= k_min, gx_p <= k_max)

                            gmx    = gx_p[qx_idx] 
                            gmw    = gw_p[qx_idx]

                            vp     = v_post_p[qx_idx]
                            t_cs   = total_cs_p[qx_idx]

                            phi_k    = spec_sp.basis_eval_radial(gmx, k, 0)
                            for p in range(ib,ie+1):
                                psi_pp        = spec_sp.basis_eval_radial(vp, p, 0)
                                tmp_q0[p,k]  += np.dot(gmw, V_TH * gmx**3 * t_cs * phi_k * psi_pp)
                    
                    for qs_idx, (q,s) in enumerate(self._sph_harm_lm):
                        if q==0:
                            for p in range(num_p):
                                for k in range(num_p):
                                    cc_collision[p * num_sh + qs_idx , k * num_sh + qs_idx] =tmp_q0[p,k]
                        else:
                            for p in range(num_p):
                                for k in range(num_p):
                                    cc_collision[p * num_sh + qs_idx , k * num_sh + qs_idx] =tmp_q1[p,k]
                
                elif(g._type == collisions.CollisionType.EAR_G1 or g._type == collisions.CollisionType.EAR_G2):
                    
                    energy_split = 1.0
                    if g._type == collisions.CollisionType.EAR_G2:
                        energy_split= 2.0

                    tmp_q0   = np.zeros((num_p,num_p))
                    tmp_q1   = np.zeros((num_p,num_p))

                    gx_m     = np.array([])
                    gw_m     = np.array([])

                    gx_p     = np.array([])
                    gw_p     = np.array([])

                    v_pre    = lambda x : np.sqrt(energy_split * x**2 + c_gamma**2 * g._reaction_threshold/V_TH**2)


                    for e_id in range(0,len(dg_idx),2):
                        ib = dg_idx[e_id]
                        ie = dg_idx[e_id+1]

                        xb = k_vec[ib]
                        xe = k_vec[ie+sp_order+1]

                      
                        # computing the collision loss term xb to xe domain integration
                        for ii in range(ib, ie + sp_order + 1):
                            if k_vec[ii]< k_vec[ii+1]:
                                a0, b0 = basis.gauss_legendre_quad(dg_q_per_knot, k_vec[ii], k_vec[ii+1])

                                gx_m = np.append(gx_m, a0)
                                gw_m = np.append(gw_m, b0)

                                a0, b0 = basis.gauss_legendre_quad(dg_q_per_knot, v_pre(k_vec[ii]), v_pre(k_vec[ii+1]))

                                #print(k_vec[ii], k_vec[ii+1], "to ", v_pre(k_vec[ii]), v_pre(k_vec[ii+1]))

                                gx_p = np.append(gx_p, a0)
                                gw_p = np.append(gw_p, b0)
                                #print("e_id=%d (%.4E,%.4E) internal dx = (%.6E, %.6E)"%(e_id,xb, xe, max(xb_eps,k_vec[ii]), k_vec[ii+1]))

                    #print(np.sum(gw_m),k_vec[-1])
                    total_cs_m  = g.total_cross_section((gx_m * V_TH / c_gamma)**2)
                    total_cs_p  = g.total_cross_section((gx_p * V_TH / c_gamma)**2)

                    v_post_p    = c_gamma * np.sqrt( (1/energy_split) * ((gx_p * V_TH /c_gamma)**2  - g._reaction_threshold)) / V_TH

                    for e_id in range(0,len(dg_idx),2):
                        ib = dg_idx[e_id]
                        ie = dg_idx[e_id+1]

                        xb = k_vec[ib]
                        xe = k_vec[ie+sp_order+1]

                        # computing the collision loss term xb to xe domain integration
                        for k in range(ib,ie+1):
                            k_min  = k_vec[k]
                            k_max  = k_vec[k + sp_order + 1]
                            qx_idx = np.logical_and(gx_m >= k_min, gx_m <= k_max)

                            gmx    = gx_m[qx_idx] 
                            gmw    = gw_m[qx_idx]
                            t_cs   = total_cs_m[qx_idx]

                            phi_k    = spec_sp.basis_eval_radial(gmx, k, 0)
                            
                            for p in range(ib,ie+1):
                                psi_pm       = spec_sp.basis_eval_radial(gmx, p, 0)
                                tmp_q0[p,k]  += np.dot(gmw, -V_TH * gmx**3 * t_cs * phi_k * psi_pm)
                                tmp_q1[p,k]  += np.dot(gmw, -V_TH * gmx**3 * t_cs * phi_k * psi_pm)

                        for k in range(ib, num_p):
                            k_min  = k_vec[k]
                            k_max  = k_vec[k + sp_order + 1]
                            qx_idx = np.logical_and(gx_p >= k_min, gx_p <= k_max)

                            gmx    = gx_p[qx_idx] 
                            gmw    = gw_p[qx_idx]

                            vp     = v_post_p[qx_idx]
                            t_cs   = total_cs_p[qx_idx]

                            phi_k    = spec_sp.basis_eval_radial(gmx, k, 0)
                            for p in range(ib,ie+1):
                                psi_pp        = energy_split * spec_sp.basis_eval_radial(vp, p, 0)
                                tmp_q0[p,k]  += np.dot(gmw, V_TH * gmx**3 * t_cs * phi_k * psi_pp)
                    
                    for qs_idx, (q,s) in enumerate(self._sph_harm_lm):
                        if q==0:
                            for p in range(num_p):
                                for k in range(num_p):
                                    cc_collision[p * num_sh + qs_idx , k * num_sh + qs_idx] =tmp_q0[p,k]
                        else:
                            for p in range(num_p):
                                for k in range(num_p):
                                    cc_collision[p * num_sh + qs_idx , k * num_sh + qs_idx] =tmp_q1[p,k]

                else:
                    raise NotImplementedError
                                
            return cc_collision
        else:
            raise NotImplementedError

    def electron_gas_temperature(self, collision, maxwellian, vth, mp_pool_sz=4):
        V_TH          = vth     
        g             = collision
        spec_sp       = self._spec
        num_p         = spec_sp._p+1
        num_sh        = len(spec_sp._sph_harm_lm)
        
        assert g._type == collisions.CollisionType.EAR_G0
        
        if self._r_basis_type == basis.BasisType.SPLINES:
            k_vec        = spec_sp._basis_p._t
            dg_idx       = spec_sp._basis_p._dg_idx
            sp_order     = spec_sp._basis_p._sp_order
            cc_collision = spec_sp.create_mat()
            c_gamma      = np.sqrt(2*collisions.ELECTRON_CHARGE_MASS_RATIO)

            k_vec_uq      = np.unique(k_vec)
            k_vec_dx      = k_vec_uq[1] - k_vec_uq[0]
            tgK           = 1.0

            if len(dg_idx) == 2:
                # continuous basis grid
                gx_e , gw_e  = spec_sp._basis_p.Gauss_Pn(self._NUM_Q_VR)
                total_cs     = g.total_cross_section((gx_e * V_TH / c_gamma)**2) 
                c_mu         = 2 * g._mByM 
                #v_scale      = np.sqrt(1- c_mu)
                #v_post       = gx_e * v_scale
                kappa        = (scipy.constants.Boltzmann * tgK * c_mu * 0.5 / scipy.constants.electron_mass) / V_TH
                
                def t1(p,k):
                    k_min    = k_vec[k]
                    k_max    = k_vec[k + sp_order + 1]
                    qx_idx   = np.logical_and(gx_e >= k_min, gx_e <= k_max)

                    gmx      = gx_e[qx_idx] 
                    gmw      = gw_e[qx_idx]
                    t_cs     = total_cs[qx_idx]

                    dx_phi_k = spec_sp.basis_derivative_eval_radial(gmx, k, 0,1)
                    q0       = np.dot(gmw, - kappa * gmx **3 * t_cs * spec_sp.basis_derivative_eval_radial(gmx, p, 0, 1) * dx_phi_k)
                    
                    return (p,k, q0)
                
                tmp_q0     = np.zeros((num_p,num_p))
                with Pool(mp_pool_sz) as process_pool:
                    result = process_pool.starmap(t1,[(p,k) for p in range(num_p) for k in range(num_p)])
                    
                for r in result:
                    tmp_q0[r[0], r[1]] = r[2]
                
                for qs_idx, (q,s) in enumerate(self._sph_harm_lm):
                    if q==0:
                        for p in range(num_p):
                            for k in range(num_p):
                                cc_collision[p * num_sh + qs_idx , k * num_sh + qs_idx] =tmp_q0[p,k]
        
            else:
                dg_q_per_knot = sp_order + 2
                c_mu          = 2 * g._mByM 
                v_scale       = np.sqrt(1- c_mu)
                kappa         = (scipy.constants.Boltzmann * tgK * c_mu * 0.5 / scipy.constants.electron_mass) / V_TH
                tmp_q0        = np.zeros((num_p,num_p))
                
                gx_m          = np.array([])
                gw_m          = np.array([])
                
                gx_p          = np.array([])
                gw_p          = np.array([])


                for e_id in range(0,len(dg_idx),2):
                    ib = dg_idx[e_id]
                    ie = dg_idx[e_id+1]

                    xb = k_vec[ib]
                    xe = k_vec[ie+sp_order+1]

                    xb_eps = xb / v_scale
                    xe_eps = xe / v_scale

                    if xb_eps > xb:
                        a0, b0 = basis.gauss_legendre_quad(dg_q_per_knot, xb, xb_eps)
                        gx_m   = np.append(gx_m, a0)
                        gw_m   = np.append(gw_m, b0)
                        #print("e_id=%d (%.4E,%.4E) sp_idx pre (%d,%d) dx = (%.6E, %.6E)"%(e_id,xb,xe,ib,ie, xb, xb_eps))
                        
                    # computing the collision loss term xb to xe domain integration
                    for ii in range(ib, ie + sp_order + 1):
                        if k_vec[ii]< k_vec[ii+1]:
                            a0, b0 = basis.gauss_legendre_quad(dg_q_per_knot, max(xb_eps,k_vec[ii]), k_vec[ii+1])

                            gx_m = np.append(gx_m, a0)
                            gw_m = np.append(gw_m, b0)

                            gx_p = np.append(gx_p, a0)
                            gw_p = np.append(gw_p, b0)
                            #print("e_id=%d (%.4E,%.4E) internal dx = (%.6E, %.6E)"%(e_id,xb, xe, max(xb_eps,k_vec[ii]), k_vec[ii+1]))

                    if xe_eps > xe:
                        a0, b0 = basis.gauss_legendre_quad(dg_q_per_knot, xe, xe_eps)
                        gx_p   = np.append(gx_p, a0)
                        gw_p   = np.append(gw_p, b0)
                        #print("e_id=%d (%.4E,%.4E) sp_idx post (%d,%d) dx = (%.6E, %.6E)"%(e_id,xb,xe,ib,ie, xe, xe_eps))

                total_cs_m  = g.total_cross_section((gx_m * V_TH / c_gamma)**2)
                for e_id in range(0,len(dg_idx),2):
                    ib = dg_idx[e_id]
                    ie = dg_idx[e_id+1]

                    xb = k_vec[ib]
                    xe = k_vec[ie+sp_order+1]

                    # computing the collision loss term xb to xe domain integration
                    for k in range(ib,ie+1):
                        k_min  = k_vec[k]
                        k_max  = k_vec[k + sp_order + 1]
                        qx_idx = np.logical_and(gx_m >= k_min, gx_m <= k_max)

                        gmx    = gx_m[qx_idx] 
                        gmw    = gw_m[qx_idx]
                        t_cs   = total_cs_m[qx_idx]

                        dx_phi_k = spec_sp.basis_derivative_eval_radial(gmx, k, 0,1)

                        for p in range(ib,ie+1):
                            tmp_q0[p,k]  += np.dot(gmw,- kappa * gmx **3 * t_cs * spec_sp.basis_derivative_eval_radial(gmx, p, 0, 1) * dx_phi_k)
                
                for qs_idx, (q,s) in enumerate(self._sph_harm_lm):
                    if q==0:
                        for p in range(num_p):
                            for k in range(num_p):
                                cc_collision[p * num_sh + qs_idx , k * num_sh + qs_idx] =tmp_q0[p,k]
                    
            return cc_collision    
        else:
            raise NotImplementedError
    
    def assemble_mat(self,collision : collisions.Collisions , maxwellian, vth,v0=np.zeros(3), tgK=0.0, mp_pool_sz=4, use_hsph=False):
        if (use_hsph):
            Lij  = self._Lop_eulerian_full(collision, maxwellian, vth, tgK, use_hemi_sph_harm=True, azimuthal_symmetry=True)
        else:
            Lij = self._LOp_eulerian_radial_only(collision,maxwellian, vth, tgK, mp_pool_sz)
        
        return Lij
    
    def rosenbluth_potentials(self, hl_op, gl_op, fb, mw, vth):
        
        spec_sp  :sp.SpectralExpansionSpherical     = self._spec
        num_p         = spec_sp._p+1
        sph_lm        = spec_sp._sph_harm_lm
        num_sh        = len(spec_sp._sph_harm_lm)
        sp_order      = spec_sp._basis_p._sp_order

        gmx, gmw = spec_sp._basis_p.Gauss_Pn(self._NUM_Q_VR)
        hl = np.zeros(num_p * num_sh)
        gl = np.zeros(num_p * num_sh)

        hl = np.dot(hl_op, fb)
        gl = np.dot(gl_op, fb)

        # for lm_idx, lm in enumerate(sph_lm):
        #     hl[lm_idx::num_sh] = np.dot(hl_op[lm_idx] ,fb[lm_idx::num_sh]) 
        #     gl[lm_idx::num_sh] = np.dot(gl_op[lm_idx] ,fb[lm_idx::num_sh]) 

        # p20_a      = (np.sqrt(4 * np.pi / (2 * 0 + 1))) * 4*np.pi * np.dot(fb[0::num_sh], self._p2)
        # p40_a      = (np.sqrt(4 * np.pi / (2 * 0 + 1))) * 4*np.pi * np.dot(fb[0::num_sh], self._p4) 
        # q10_a      = (np.sqrt(4 * np.pi / (2 * 0 + 1))) * 4*np.pi * np.dot(fb[0::num_sh], self._q1) 
        # q30_a      = (np.sqrt(4 * np.pi / (2 * 0 + 1))) * 4*np.pi * np.dot(fb[0::num_sh], self._q3)

        # p31_a      = (np.sqrt(4 * np.pi / (2 * 0 + 1))) * 4*np.pi * np.dot(fb[1::num_sh], self._p3)
        # p51_a      = (np.sqrt(4 * np.pi / (2 * 0 + 1))) * 4*np.pi * np.dot(fb[1::num_sh], self._p5)
        # q01_a      = (np.sqrt(4 * np.pi / (2 * 0 + 1))) * 4*np.pi * np.dot(fb[1::num_sh], self._q0)

        # Vqr = spec_sp.Vq_r(gmx, 0, 1)

        # import matplotlib.pyplot as plt
        # # hl_0 = np.dot(np.transpose(Vqr),hl[0::num_sh])
        # # plt.plot(gmx, hl_0/(4*np.pi), label="new h0")
        # # plt.plot(gmx, (2) * (p20_a/gmx + q10_a), label="old h0")

        # # hl_1 = np.dot(np.transpose(Vqr),hl[1::num_sh])
        # # plt.plot(gmx, hl_1/(4*np.pi), label="new h1")
        # # plt.plot(gmx, (2) * (p31_a/gmx**2/3 + q01_a * gmx/3), label="old h1")

        # gl_0 = np.dot(np.transpose(Vqr),gl[0::num_sh])
        # plt.plot(gmx, gl_0, label="new g0")
        # plt.plot(gmx, gmx * p20_a + q30_a + (1/gmx/3) * p40_a + gmx**2 * q10_a/3, label="olg g0")

        # plt.legend()
        # plt.show()
        # plt.close()

        return hl, gl
    
    def coulomb_collision_mat(self, alpha, ionization_degree, n0, fb, mw, vth, sigma_m, full_assembly=True):
        """
        compute the weak form of the coulomb collision operator based on fokker-plank equation
        with Rosenbluth's potentials

        Assumptions: 
            - Currently for l=0, l=1 modes only, others assumed to be zero
            - assumes azimuthal symmetry
        """

        spec_sp  :sp.SpectralExpansionSpherical     = self._spec
        num_p         = spec_sp._p+1
        num_sh        = len(spec_sp._sph_harm_lm)

        if self._r_basis_type != basis.BasisType.SPLINES:
            raise NotImplementedError
        
        cc_collision = spec_sp.create_mat()

        k_vec      = spec_sp._basis_p._t
        dg_idx     = spec_sp._basis_p._dg_idx
        sp_order   = spec_sp._basis_p._sp_order

        gmx_a , gmw_a  = spec_sp._basis_p.Gauss_Pn(self._NUM_Q_VR)
        
        sph_l0     = lambda l : np.sqrt((2 * l +1) / (4 * np.pi) )

        B          = spec_sp.basis_eval_radial
        DB         = spec_sp.basis_derivative_eval_radial


        p20_a      = 2 * sph_l0(0) * np.dot(fb[0::num_sh], self._p2)
        p40_a      = 2 * sph_l0(0) * np.dot(fb[0::num_sh], self._p4) 
        q10_a      = 2 * sph_l0(0) * np.dot(fb[0::num_sh], self._q1) 

        p31_a      = 2 * sph_l0(1) * np.dot(fb[1::num_sh], self._p3)
        p51_a      = 2 * sph_l0(1) * np.dot(fb[1::num_sh], self._p5)
        q01_a      = 2 * sph_l0(1) * np.dot(fb[1::num_sh], self._q0)

        for p in range(num_p):
            for k in range(max(0, p - (sp_order+3) ), min(num_p, p + (sp_order+3))):
                
                k_min  = min(k_vec[p], k_vec[k])
                k_max  = max(k_vec[p + sp_order + 1], k_vec[k + sp_order + 1])

                qx_idx = np.logical_and(gmx_a >= k_min, gmx_a <= k_max)
                gmx    = gmx_a[qx_idx]
                gmw    = gmw_a[qx_idx]

                p20    = p20_a[qx_idx] 
                p40    = p40_a[qx_idx] 
                q10    = q10_a[qx_idx] 

                p31    = p31_a[qx_idx] 
                p51    = p51_a[qx_idx] 
                q01    = q01_a[qx_idx] 

                tmp = -(alpha * p20 * B(gmx,k,0) + (1/(3*gmx)) * (p40 + gmx**3 * q10) * DB(gmx, k, 0, 1)) * DB(gmx, p, 0, 1)
                cc_collision[p * num_sh + 0 , k * num_sh + 0] = np.dot(tmp, gmw) 
                
                if full_assembly:
                    tmp  = (1 + alpha) * gmx * q01 * B(gmx, k, 0) * B(gmx, p, 0) - (
                        (1/3) * B(gmx, k, 0) * ((2 * alpha -1) * p31 - (1+alpha) * gmx**3 * q01) 
                        + (DB(gmx, k, 0, 1) / (5 * gmx)) * (p51 + gmx**5 * q01)
                        ) * (gmx * DB(gmx, p, 0, 1) - B(gmx, p, 0)) / gmx**2
                    
                    cc_collision[p * num_sh + 1 , k * num_sh + 0] = (sph_l0(0)/ sph_l0(1)) * np.dot(tmp, gmw) 

                    tmp  =  -((1 + alpha)/gmx) * p20 * B(gmx, k , 0) * B(gmx, p, 0) - \
                    ( alpha * gmx * p20 * B(gmx, k, 0) 
                    + (1./3) * gmx * (p40 + gmx**3 * q10) * ((gmx * DB(gmx, k , 0, 1) - B(gmx, k, 0))/gmx**2) ) * (((gmx * DB(gmx, p , 0, 1) - B(gmx, p, 0))/gmx**2))

                    cc_collision[p * num_sh + 1 , k * num_sh + 1] = np.dot(tmp, gmw)


        ne           = n0 * ionization_degree
        eps_0        = scipy.constants.epsilon_0
        me           = scipy.constants.electron_mass
        qe           = scipy.constants.e
        m0           = mw(0) * np.dot(fb,self._mass_op) * vth**3 
        kT           = mw(0) * (np.dot(fb, self._temp_op) * vth**5 * 0.5 * scipy.constants.electron_mass * (2./ 3) / m0) 

        kT           = np.abs(kT) 
        
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

        # cc_freq_op    = vth**4 * np.sqrt(4*np.pi) * n0 * np.array([np.dot(gmw_a, gmx_a**3 * sigma_m * B(gmx_a,k,0)) for k in range(num_p)])
        # cc_freq       = scipy.constants.electron_volt * np.dot(cc_freq_op, fb[0::num_sh])
        # # print(cc_freq)
        # M             = (np.sqrt(6) * cc_freq ) / (2 * np.sqrt((qe**2 * ne) / (eps_0 * me)))
        # c_lambda      = ((12 * np.pi * (eps_0 * kT)**(1.5))/(qe**3 * np.sqrt(ne)))
        # c_lambda      = (c_lambda + M) / (1 + M)
        #c_lambda      = np.exp(6.314)

        # Tev        = kT/ scipy.constants.electron_volt
        # c_lambda     = np.exp(23.5 - np.log(np.sqrt(ne* 1e-6) * Tev **(5/4) - np.sqrt(1e-5 + ((np.log(Tev)-2)**2 )/16)))
        c_lambda      = ((12 * np.pi * (eps_0 * kT)**(1.5))/(qe**3 * np.sqrt(ne)))
        gamma_a       = (np.log(c_lambda) * (qe**4)) / (4 * np.pi * (eps_0 * me)**2) / (vth)**3

        

        # print( p40[-1] * vth**5 * mw(0) * 0.5 * scipy.constants.electron_mass * (2./ 3) / m0)
        print("mass=%.8E\t Coulomb logarithm %.8E \t gamma_a %.8E \t gamma_a * ne %.8E  \t kT=%.8E temp(ev)=%.8E temp (K)=%.8E " %(m0, np.log(c_lambda) , gamma_a, n0 * ionization_degree * gamma_a, kT, kT/scipy.constants.electron_volt, kT/scipy.constants.Boltzmann))

        

        
        return cc_collision * gamma_a

    def gamma_a(self, fb, mw, vth, n0, ion_deg, eff_rr_op):
        spec_sp  :sp.SpectralExpansionSpherical     = self._spec
        num_p         = spec_sp._p+1
        sph_lm        = spec_sp._sph_harm_lm
        num_sh        = len(spec_sp._sph_harm_lm)

        ne           = n0 * ion_deg 
        eps_0        = scipy.constants.epsilon_0
        me           = scipy.constants.electron_mass
        qe           = scipy.constants.e
        m0           = mw(0) * np.dot(fb,self._mass_op) * vth**3 
        kT           = mw(0) * (np.dot(fb, self._temp_op) * vth**5 * 0.5 * scipy.constants.electron_mass * (2./ 3) / m0) 
        kT           = np.abs(kT)
        
        M            = 0.0
        # wp           = np.sqrt((qe**2 * ne) / (eps_0 * me))
        # M            = (np.sqrt(6)/(2 * wp)) * np.dot(eff_rr_op,fb[0::num_sh])
        
        c_lambda     = ((12 * np.pi * (eps_0 * kT)**(1.5))/(qe**3 * np.sqrt(ne)))
        c_lambda     = (c_lambda + M) / (1+M)

        gamma_a      = (np.log(c_lambda) * (qe**4)) / (4 * np.pi * (eps_0 * me)**2) / (vth)**3
        
        #print("mass=%.8E\t Coulomb logarithm %.8E \t gamma_a %.8E \t gamma_a * ne %.8E  \t kT=%.8E temp(ev)=%.8E temp (K)=%.8E " %(m0, np.log(c_lambda) , gamma_a, n0 * ion_deg * gamma_a, kT, kT/scipy.constants.electron_volt, kT/scipy.constants.Boltzmann))

        return gamma_a
    
    def compute_rosenbluth_potentials_op(self, mw, vth, m_ab, Minv, mp_pool_sz=4):
        
        spec_sp  :sp.SpectralExpansionSpherical     = self._spec
        num_p         = spec_sp._p+1
        sph_lm        = spec_sp._sph_harm_lm
        num_sh        = len(spec_sp._sph_harm_lm)

        if self._r_basis_type != basis.BasisType.SPLINES:
            raise NotImplementedError

        gmx, gmw = spec_sp._basis_p.Gauss_Pn(self._NUM_Q_VR)

        k_vec    = spec_sp._basis_p._t
        dg_idx   = spec_sp._basis_p._dg_idx
        sp_order = spec_sp._basis_p._sp_order


        def integrate_xm_splines(m, sp_idx, xb, xe, qx_e, qw_e):
            q_int  = 0
            q_int1 = 0
            for kn_idx in range(sp_idx + 1, sp_idx + sp_order + 2):
                xx_b = max(xb,k_vec[kn_idx-1]) 
                xx_e = min(xe,k_vec[kn_idx]) 
                if xx_e > xx_b:
                    qx      = 0.5 * ((xx_e - xx_b) * qx_e +  (xx_e + xx_b))
                    qw      = 0.5 *  (xx_e - xx_b) * qw_e
                    #print(xx_b, xx_e, np.dot(qw, (qx**m) * spec_sp.basis_eval_radial(qx, sp_idx , 0)))
                    q_int  += np.dot(qw, (qx**m) * spec_sp.basis_eval_radial(qx, sp_idx , 0))
                    q_int1 += np.dot(qw, np.ones_like(qx))
            
            assert np.abs(q_int1 - (xe-xb))<1e-14, "[coulomb potential computation]--integration interval is invalid dx=%.12E vs q_dx=%.12E diff =%.12E"%(q_int1, (xe-xb), np.abs(xe-xb -q_int1))
            return q_int

        # compute the moment vectors
        def Pm(m):
            q_tol = 1e-12
            assert m>=0 , "m=%d"%(m)
            q_order       = ((sp_order + abs(m) + 1)//2) + 1
            q_order      *= 4
            qq_re         = np.polynomial.legendre.leggauss(q_order) 
            qq_re_points  = qq_re[0]
            qq_re_weights = qq_re[1]

            qq_re1         = np.polynomial.legendre.leggauss(q_order*2) 
            qq_re_points1  = qq_re1[0]
            qq_re_weights1 = qq_re1[1]
            
            pm       = np.zeros((num_p, len(gmx)))
            
            for i in range(num_p):
                a_idx = np.where(k_vec[i] <= gmx)[0]
                b_idx = np.where(gmx <= k_vec[i + sp_order + 1])[0]
                idx   = np.sort(np.intersect1d(a_idx, b_idx))
                for j in idx:
                    xb      = k_vec[i]
                    xe      = gmx[j]

                    tmp0    = integrate_xm_splines(m, i, xb, xe, qq_re_points, qq_re_weights)
                    tmp1    = integrate_xm_splines(m, i, xb, xe, qq_re_points1, qq_re_weights1)
                    rel_error = abs(1-tmp0/tmp1)
                    if rel_error > q_tol:
                        print("Pm ij = (%d,%d) domain = (%.3E, %.3E) dx=%.4E 2*q points = %.16E q = %.16E rel_error=%.8E m=%d"%(i,j, xb, xe, (xe-xb), tmp1, tmp0, rel_error,m))
                    
                    pm[i,j] = tmp0

                
                a_idx = np.where(k_vec[i + sp_order + 1] < gmx)[0]
                xb  = k_vec[i]
                xe  = k_vec[i + sp_order + 1]

                tmp0    = integrate_xm_splines(m, i, xb, xe, qq_re_points, qq_re_weights)
                tmp1    = integrate_xm_splines(m, i, xb, xe, qq_re_points1, qq_re_weights1)
                rel_error = abs(1-tmp0/tmp1)
                if rel_error > q_tol:
                    print("Pm ij = (%d,%d) domain = (%.3E, %.3E) dx=%.4E 2*q points = %.16E q = %.16E rel_error=%.8E m=%d"%(i,j, xb, xe, (xe-xb), tmp1, tmp0, rel_error,m))

                pm[i, a_idx] = tmp0

            return pm
        
        def Qm(m):
            q_tol = 1e-12
            #assert m<=0 , "m=%d"%(m)
            q_order       = ((sp_order + abs(m) + 1)//2) + 1
            q_order       *= 32
            qq_re         = np.polynomial.legendre.leggauss(q_order) #quadpy.c1.gauss_legendre(q_order)
            qq_re_points  = qq_re[0]
            qq_re_weights = qq_re[1]

            qq_re1         = np.polynomial.legendre.leggauss(q_order*2) 
            qq_re_points1  = qq_re1[0]
            qq_re_weights1 = qq_re1[1]
            
            
            qm       = np.zeros((num_p, len(gmx)))
            for i in range(num_p):
                a_idx = np.where(k_vec[i] <= gmx)[0]
                b_idx = np.where(gmx <= k_vec[i + sp_order + 1])[0]
                idx   = np.sort(np.intersect1d(a_idx, b_idx))
                for j in idx:
                    xb  = gmx[j]
                    xe  = k_vec[i + sp_order+1]

                    tmp0    = integrate_xm_splines(-m, i, xb, xe, qq_re_points, qq_re_weights)
                    tmp1    = integrate_xm_splines(-m, i, xb, xe, qq_re_points1, qq_re_weights1)
                    rel_error = abs(1-tmp0/tmp1)
                    if rel_error > q_tol:
                        print("Qm ij = (%d,%d) domain = (%.3E, %.3E) dx=%.4E 2*q points = %.16E q = %.16E rel_error=%.8E m=%d"%(i,j, xb, xe, (xe-xb), tmp1, tmp0, rel_error,m))
                    
                    qm[i,j] =  tmp0

                a_idx = np.where(gmx < k_vec[i])[0]
                xb  = k_vec[i]
                xe  = k_vec[i + sp_order + 1]

                tmp0    = integrate_xm_splines(-m, i, xb, xe, qq_re_points, qq_re_weights)
                tmp1    = integrate_xm_splines(-m, i, xb, xe, qq_re_points1, qq_re_weights1)
                rel_error = abs(1-tmp0/tmp1)
                if rel_error > q_tol:
                    print("bQm ij = (%d,%d) domain = (%.3E, %.3E) dx=%.4E 2*q points = %.16E q = %.16E rel_error=%.8E m=%d"%(i,j, xb, xe, (xe-xb), tmp1, tmp0, rel_error,m))
                    
                qm[i, a_idx] = tmp0

            return qm


        hl_v = np.zeros((num_sh, len(gmx), num_p))
        gl_v = np.zeros((num_sh, len(gmx), num_p))
        
        for lm_idx, lm in enumerate(sph_lm):
            ll = lm[0]
            f1 = (ll-0.5) / (ll + 1.5)

            m1 = (2 / (2*ll+1) ) * (1 + m_ab) * (Pm(ll + 2) / (gmx**(ll+1)) + Qm(ll-1) * gmx** (ll))
            m2 = -( 2 / (4* ll**2 - 1)) * ((Pm(ll + 2) / gmx**(ll-1) - f1 * Pm(ll+4)/gmx**(ll+1)) + (Qm(ll - 3) * gmx**(ll)   - f1 * Qm(ll-1) * gmx**(ll+2)))

            
            # import matplotlib.pyplot as plt
            # for i in range(0,num_p,10):
            #     plt.plot(gmx, m1[i,:], label="m1 l=%d k=%d"%(lm_idx, i))
            
            # plt.legend()
            # plt.show()
            # plt.close()


            hl_v[lm_idx, : , :] = np.transpose(m1) 
            gl_v[lm_idx, : , :] = np.transpose(m2)

        # hl_op = hl_v
        # gl_op = gl_v
        hl_op = np.zeros((num_p * num_sh, num_p * num_sh))
        gl_op = np.zeros((num_p * num_sh, num_p * num_sh))
        
        def t1(lm_idx, p, k):
            bp   = spec_sp.basis_eval_radial(gmx, p, 0)
            hl   = np.dot(gmx**2 * bp * hl_v[lm_idx, : , k], gmw)
            gl   = np.dot(gmx**2 * bp * gl_v[lm_idx, : , k], gmw)
            
            return (lm_idx, p, k, hl, gl)
        
        with Pool(mp_pool_sz) as process_pool:
            result = process_pool.starmap(t1, [(lm_idx, p, k) for lm_idx, lm in enumerate(sph_lm) for p in range(num_p) for k in range(num_p)])
            
        for r in result:
            lm_idx = r[0]
            p      = r[1]
            k      = r[2]
            
            hl     = r[3]
            gl     = r[4]
            
            hl_op[p * num_sh + lm_idx, k * num_sh + lm_idx] = hl
            gl_op[p * num_sh + lm_idx, k * num_sh + lm_idx] = gl
            
            
        # for lm_idx, lm in enumerate(sph_lm):
        #     for p in range(num_p):
        #         bp  = spec_sp.basis_eval_radial(gmx, p, 0)
        #         for k in range(num_p):
        #             hl_op[p * num_sh + lm_idx, k * num_sh + lm_idx] = np.dot(gmx**2 * bp * hl_v[lm_idx, : , k], gmw)
        #             gl_op[p * num_sh + lm_idx, k * num_sh + lm_idx] = np.dot(gmx**2 * bp * gl_v[lm_idx, : , k], gmw)
                
        hl_op = np.dot(Minv, hl_op)
        gl_op = np.dot(Minv, gl_op)
        return hl_op, gl_op

    def coulomb_collision_op_assembly(self, mw, vth, gen_code=False, mp_pool_sz=4):
        spec_sp  :sp.SpectralExpansionSpherical     = self._spec
        num_p         = spec_sp._p+1
        sph_lm        = spec_sp._sph_harm_lm
        num_sh        = len(spec_sp._sph_harm_lm)

        if self._r_basis_type != basis.BasisType.SPLINES:
            raise NotImplementedError

        self._mass_op = BEUtils.mass_op(spec_sp, 1)
        self._temp_op = BEUtils.temp_op(spec_sp, 1)
        
        v      = sympy.Symbol('vr')
        mu     = sympy.Symbol('mu')
        phi    = sympy.Symbol('phi')

        if gen_code:
            # generate integrals for evaluation
            metric = sympy.Matrix([[1,0,0], [0, v**2/(1-mu**2), 0], [0, 0, v**2 * (1-mu**2)]])
            coords = [v, mu, phi]
            Ia, Ib = sym_cc.assemble_symbolic_cc_op(metric, coords, sph_lm[-1][0])

        
        k_vec     = spec_sp._basis_p._t
        dg_idx    = spec_sp._basis_p._dg_idx
        sp_order  = spec_sp._basis_p._sp_order

        B         = lambda vr, a : spec_sp.basis_eval_radial(vr, a, 0)
        DB        = lambda vr, a, d : spec_sp.basis_derivative_eval_radial(vr, a, 0, d) 

        gmx_a , gmw_a  = spec_sp._basis_p.Gauss_Pn(self._NUM_Q_VR)
        cc_mat_a       = np.zeros((num_p * num_sh, num_p * num_sh, num_p *num_sh))
        cc_mat_b       = np.zeros((num_p * num_sh, num_p * num_sh, num_p *num_sh))
        lmax           = sph_lm[-1][0]

        import cc_terms
        def t1(p,k,r):
            k_min  = min(min(k_vec[p], k_vec[k]), k_vec[r])
            k_max  = max(k_vec[r + sp_order + 1] , max(k_vec[p + sp_order + 1], k_vec[k + sp_order + 1]))
    
            qx_idx = np.logical_and(gmx_a >= k_min, gmx_a <= k_max)
            gmx    = gmx_a[qx_idx] 
            gmw    = gmw_a[qx_idx] 

            B_p_vr       =  B(gmx, p)
            DB_p_dvr     = DB(gmx, p, 1)
            DB_p_dvr_dvr = DB(gmx, p, 2)

            B_k_vr       =  B(gmx, k)
            DB_k_dvr     = DB(gmx, k, 1)
            DB_k_dvr_dvr = DB(gmx, k, 2)

            B_r_vr       =  B(gmx, r)
            DB_r_dvr     = DB(gmx, r, 1)
            DB_r_dvr_dvr = DB(gmx, r, 2)
            
            
            ca = list()
            cb = list()
            
            for idx in cc_terms.Ia_nz:
                if idx[0] > lmax or idx[1] > lmax or idx[2] > lmax:
                    continue
                a  = np.dot(gmw, cc_terms.Ia(B, DB, gmx, p, k, r, idx[0], idx[1], idx[2], B_p_vr, B_k_vr, B_r_vr, DB_p_dvr, DB_k_dvr, DB_r_dvr, DB_p_dvr_dvr, DB_k_dvr_dvr, DB_r_dvr_dvr))
                
                ca.append((idx[0], idx[1], idx[2], a))
                
            for idx in cc_terms.Ib_nz:
                if idx[0] > lmax or idx[1] > lmax or idx[2] > lmax:
                    continue    
                b  = np.dot(gmw, cc_terms.Ib(B, DB, gmx, p, k, r, idx[0], idx[1], idx[2], B_p_vr, B_k_vr, B_r_vr, DB_p_dvr, DB_k_dvr, DB_r_dvr, DB_p_dvr_dvr, DB_k_dvr_dvr, DB_r_dvr_dvr))
                cb.append((idx[0], idx[1], idx[2], b))
            
            return (p, k, r, ca, cb)
        
        with Pool(mp_pool_sz) as process_pool:
            result = process_pool.starmap(t1,[(p, k, r) for p in range(num_p) for k in range(max(0, p - (sp_order+3) ), min(num_p, p + (sp_order+3))) for r in range(max(0, p - (sp_order+3) ), min(num_p, p + (sp_order+3)))] )
        
        # for idx in range(len(result)):
        #     p  = result[idx][0]
        #     k  = result[idx][1]
        #     r  = result[idx][2]
            
        #     ca = result[idx][3]
        #     cb = result[idx][4]
        #     print(idx)
        #     for (idx0, idx1, idx2, val) in ca:
        #         cc_mat_a[p * num_sh +  idx0, k * num_sh + idx1, r * num_sh +  idx2] = val
                
        #     for (idx0, idx1, idx2, val) in cb:
        #         cc_mat_b[p * num_sh +  idx0, k * num_sh + idx1, r * num_sh +  idx2] = val
        
        
        from multiprocessing.pool import ThreadPool
        def t2(idx):
            p  = result[idx][0]
            k  = result[idx][1]
            r  = result[idx][2]
            
            ca = result[idx][3]
            cb = result[idx][4]
            
            for (idx0, idx1, idx2, val) in ca:
                cc_mat_a[p * num_sh +  idx0, k * num_sh + idx1, r * num_sh +  idx2] = val
                
            for (idx0, idx1, idx2, val) in cb:
                cc_mat_b[p * num_sh +  idx0, k * num_sh + idx1, r * num_sh +  idx2] = val
        
        pool = ThreadPool(mp_pool_sz)    
        pool.map(t2, [i for i in range(len(result))])
        pool.close()
        pool.join()
                
        # for p in range(num_p):
        #     for k in range(max(0, p - (sp_order+3) ), min(num_p, p + (sp_order+3))):
        #         for r in range(max(0, p - (sp_order+3) ), min(num_p, p + (sp_order+3))):
                    
        #             k_min  = min(min(k_vec[p], k_vec[k]), k_vec[r])
        #             k_max  = max(k_vec[r + sp_order + 1] , max(k_vec[p + sp_order + 1], k_vec[k + sp_order + 1]))
            
        #             qx_idx = np.logical_and(gmx_a >= k_min, gmx_a <= k_max)
        #             gmx    = gmx_a[qx_idx] 
        #             gmw    = gmw_a[qx_idx] 

        #             B_p_vr       =  B(gmx, p)
        #             DB_p_dvr     = DB(gmx, p, 1)
        #             DB_p_dvr_dvr = DB(gmx, p, 2)

        #             B_k_vr       =  B(gmx, k)
        #             DB_k_dvr     = DB(gmx, k, 1)
        #             DB_k_dvr_dvr = DB(gmx, k, 2)

        #             B_r_vr       =  B(gmx, r)
        #             DB_r_dvr     = DB(gmx, r, 1)
        #             DB_r_dvr_dvr = DB(gmx, r, 2)
            
        #             for idx in cc_terms.Ia_nz:
        #                 if idx[0] > lmax or idx[1] > lmax or idx[2] > lmax:
        #                     continue
        #                 cc_mat_a[p * num_sh +  idx[0], k * num_sh + idx[1], r * num_sh +  idx[2]] = np.dot(gmw, cc_terms.Ia(B, DB, gmx, p, k, r, idx[0], idx[1], idx[2], B_p_vr, B_k_vr, B_r_vr, DB_p_dvr, DB_k_dvr, DB_r_dvr, DB_p_dvr_dvr, DB_k_dvr_dvr, DB_r_dvr_dvr))
                    
        #             for idx in cc_terms.Ib_nz:
        #                 if idx[0] > lmax or idx[1] > lmax or idx[2] > lmax:
        #                     continue
        #                 cc_mat_b[p * num_sh +  idx[0], k * num_sh + idx[1], r * num_sh +  idx[2]] = np.dot(gmw, cc_terms.Ib(B, DB, gmx, p, k, r, idx[0], idx[1], idx[2], B_p_vr, B_k_vr, B_r_vr, DB_p_dvr, DB_k_dvr, DB_r_dvr, DB_p_dvr_dvr, DB_k_dvr_dvr, DB_r_dvr_dvr))
                        

        return cc_mat_a, cc_mat_b

    ### hemispherical harmonics related collision operator discretization
    def _Lop_eulerian_full(self, collision, maxwellian, vth, tgK, use_hemi_sph_harm = False, azimuthal_symmetry = True):
        V_TH          = vth     
        g             = collision
        spec_sp       = self._spec
        num_p         = spec_sp._p+1
        num_sh        = len(spec_sp._sph_harm_lm)
        
        assert azimuthal_symmetry, "only azimuthal symmetry is implemented for binary collisions operator"
        
        if self._r_basis_type == basis.BasisType.SPLINES:
            k_vec        = spec_sp._basis_p._t
            dg_idx       = spec_sp._basis_p._dg_idx
            sp_order     = spec_sp._basis_p._sp_order
            cc_collision = spec_sp.create_mat()
            c_gamma      = np.sqrt(2*collisions.ELECTRON_CHARGE_MASS_RATIO)

            k_vec_uq     = np.unique(k_vec)
            k_vec_dx     = k_vec_uq[1] - k_vec_uq[0]

            assert len(dg_idx) == 2, "only CG allowed for hsph lop eulerian"
            gx_e , gw_e        = spec_sp._basis_p.Gauss_Pn(self._NUM_Q_VR)
            
            if(g._type == collisions.CollisionType.EAR_G0):
                energy_split     = 1
                c_mu             = 2 * g._mByM 
                v_scale          = np.sqrt(1- c_mu)
                v_post           = gx_e * v_scale
                
            elif(g._type == collisions.CollisionType.EAR_G1):
                energy_split     = 1
                check_1          = (gx_e * V_TH/c_gamma)**2 >= g._reaction_threshold
                gx_e             = gx_e[check_1]
                gw_e             = gw_e[check_1]
                v_post           = c_gamma * np.sqrt( (1/energy_split) * ((gx_e * V_TH /c_gamma)**2  - g._reaction_threshold)) / V_TH
            elif(g._type == collisions.CollisionType.EAR_G2):
                energy_split     = 2
                check_1          = (gx_e * V_TH/c_gamma)**2 >= g._reaction_threshold
                gx_e             = gx_e[check_1]
                gw_e             = gw_e[check_1]
                v_post           = c_gamma * np.sqrt( (1/energy_split) * ((gx_e * V_TH /c_gamma)**2  - g._reaction_threshold)) / V_TH
            else:
                raise NotImplementedError("only EAR G0, G1, G2 are implemented for hsph lop eulerian")
            
            total_cs           = g.total_cross_section((gx_e * V_TH / c_gamma)**2) 
            if (use_hemi_sph_harm==True):
                
                Nvt      = 32#(spec_sp._sph_harm_lm[-1][0] * 2 + 2) 
                Nvp      = 32
                Nvts     = 128
                Nvps     = 128

                def __Wqs_lm__(Nvt, Nvp, Nvts, Nvps):
                    try:
                        import cupy as cp
                        import cupyx.scipy
                        _  = cp.random.rand(3)
                        xp = cp
                        print("using cupy for angular integrals")
                        
                        def sph_func(l, m, theta, phi, mode):
                            Y            = xp.zeros_like(theta)
                            t            = xp.zeros_like(theta)
                        
                            if mode=="+":
                                idx      = xp.logical_and(theta <= xp.pi/2, theta >= 0)
                                t[idx]   = xp.arccos(2 * xp.cos(theta[idx]) - 1)
                            else:
                                assert mode=="-", "invalid hemisphere mode"
                                idx      = xp.logical_and(theta >= xp.pi/2, theta <= xp.pi)
                                t[idx]   = xp.arccos(2 * xp.cos(theta[idx]) + 1)
                            
                            Ysh  = cupyx.scipy.special.sph_harm(abs(m), l, phi[idx], t[idx])
                        
                            if m < 0:
                                Y[idx] = xp.sqrt(2) * (-1)**m * Ysh.imag
                            elif m > 0:
                                Y[idx] = xp.sqrt(2) * (-1)**m * Ysh.real
                            else:
                                Y[idx] = Ysh.real

                            return xp.sqrt(2) * Y
                        

                    except Exception as e:
                        print(e)
                        xp = np
                        sph_func = spec_sp._hemi_sph_harm_real
                    
                    Wqs_lm_p    = xp.zeros((2 * num_sh, 2 * num_sh))
                    Wqs_lm_m    = xp.zeros((2 * num_sh, 2 * num_sh))

                    for didx_i, di in enumerate(["+", "-"]):
                        for didx_j, dj in enumerate(["+", "-"]):
                            for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
                                vt, vtw   =  spec_sp.gl_vt(Nvt, mode="np") if dj == "+" else  spec_sp.gl_vt(Nvt, mode="sp")
                                vp, vpw   =  spec_sp.gl_vp(Nvp)
                                
                                vts, vtsw = spec_sp.gl_vt(Nvts, mode="npsp")
                                vps, vpsw = spec_sp.gl_vp(Nvps)

                                mg        = xp.meshgrid(xp.asarray(vt), xp.asarray(vp), xp.asarray(vts), xp.asarray(vps), indexing="ij")
                                vt_p      = xp.acos(xp.cos(mg[0]) * xp.cos(mg[2]) + xp.sin(mg[0]) * xp.sin(mg[2]) * xp.cos(mg[1] - mg[3]))
                                vp_p      = 0 * vt_p
                                Ylm       = xp.asarray(sph_func(lm[0], lm[1], mg[0], mg[1], dj))
                                
                                
                                for ps_idx, qs in enumerate(spec_sp._sph_harm_lm):
                                    Yqs = xp.asarray(sph_func(qs[0], qs[1], vt_p, vp_p, di))
                                    Wqs_lm_p[didx_i * num_sh + ps_idx, didx_j * num_sh + lm_idx] = xp.einsum("abcd,a,b,c,d->", Yqs * Ylm, xp.asarray(vtw), xp.asarray(vpw), xp.asarray(vtsw), xp.asarray(vpsw))


                                # for ps_idx, qs in enumerate(spec_sp._sph_harm_lm):
                                #     Yqs = xp.asarray(sph_func(qs[0], qs[1], mg[0], mg[1], di))
                                #     Wqs_lm_m[didx_i * num_sh + ps_idx, didx_j * num_sh + lm_idx] = xp.einsum("abcd,a,b,c,d->", Yqs * Ylm, xp.asarray(vtw), xp.asarray(vpw), xp.asarray(vtsw), xp.asarray(vpsw))
                                        

                    Wqs_lm_m[0:num_sh, 0:num_sh] = 4 * xp.pi * xp.diag(xp.ones(num_sh))
                    Wqs_lm_m[num_sh: , num_sh: ] = 4 * xp.pi * xp.diag(xp.ones(num_sh))

                    # print("p\n", Wqs_lm_p)
                    # print("m\n", Wqs_lm_m)

                    if (xp == cp):
                        Wqs_lm_p = cp.asnumpy(Wqs_lm_p)
                        Wqs_lm_m = cp.asnumpy(Wqs_lm_m)

                    return Wqs_lm_p, Wqs_lm_m
            else:
                sph_func = spec_sp._sph_harm_real
                Nvt      = spec_sp._sph_harm_lm[-1][0] * 2 + 2
                Nvp      = 8
                Nvts     = Nvt
                Nvps     = Nvp

                def __Wqs_lm__(Nvt, Nvp, Nvts, Nvps):
                    Wqs_lm_p    = np.zeros((num_sh, num_sh))
                    Wqs_lm_m    = np.zeros((num_sh, num_sh))

                    vt, vt_qw   = spec_sp.gl_vt(Nvt, hspace_split=True)
                    vp, vp_qw   = spec_sp.gl_vp(Nvp)
                    
                    svt, svt_qw = spec_sp.gl_vt(Nvts, hspace_split=True)
                    svp, svp_qw = spec_sp.gl_vp(Nvps)

                    mg      = np.meshgrid(vt, vp, svt, svp, indexing='ij')
                    vt_p    = (np.cos(mg[0]) * np.cos(mg[2]) + np.sin(mg[0]) * np.sin(mg[2]) * np.cos(mg[1] - mg[3]))
                    
                    if (np.max(vt_p) > 1):
                        assert np.abs(np.max(vt_p) - 1) < 1e-10
                        vt_p[vt_p > 1]  = 1.0
                    
                    if (np.min(vt_p) < -1):
                        assert np.abs(np.min(vt_p) + 1) < 1e-10
                        vt_p[vt_p < -1] = -1.0

                    vt_p    = np.arccos(vt_p)
                    vp_p    = 0 * vt_p  # does not matter for aszimuthally symmetric distributions. 

                    for ps_idx, qs in enumerate(spec_sp._sph_harm_lm):
                        Yqs = sph_func(qs[0], qs[1], vt_p, vp_p)
                        for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
                            Ylm = sph_func(lm[0], lm[1], mg[0], mg[1])
                            Wqs_lm_p[ps_idx, lm_idx] = np.einsum("abcd,a,b,c,d->", Yqs * Ylm, vt_qw, vp_qw, svt_qw, svp_qw)

                    # for ps_idx, qs in enumerate(spec_sp._sph_harm_lm):
                    #     Yqs = sph_func(qs[0], qs[1], mg[0], mg[1])
                    #     for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
                    #         Ylm = sph_func(lm[0], lm[1], mg[0], mg[1])
                    #         Wqs_lm_m[ps_idx, lm_idx] = np.einsum("abcd,a,b,c,d->", Yqs * Ylm, vt_qw, vp_qw, svt_qw, svp_qw)
                    
                    Wqs_lm_m[0:num_sh, 0:num_sh] = 4 * np.pi * np.diag(np.ones(num_sh))

                    return Wqs_lm_p, Wqs_lm_m
            
            

            Wqs_lm_0 = __Wqs_lm__(Nvt, Nvp, Nvts, Nvps)
            Wqs_lm_1 = __Wqs_lm__(2 * Nvt, 2 * Nvp, 2 * Nvts, 2 * Nvps)

            
            for i in range(len(Wqs_lm_0)):
                rel_error_angular = np.linalg.norm(Wqs_lm_0[i] - Wqs_lm_1[i])/np.linalg.norm(Wqs_lm_1[i])
                print("angular quadrature error term %d : %.8E"%(i, rel_error_angular))

            Wqs_lm = Wqs_lm_1
            assert Wqs_lm[0].shape == Wqs_lm[1].shape
            assert Wqs_lm[0].shape[0] == Wqs_lm[0].shape[1]

            nl           = Wqs_lm[0].shape[0]
            cc_collision = np.zeros((num_p , nl, num_p , nl))
            
            for p in range(num_p):
                bp     = spec_sp.basis_eval_radial(gx_e,   p, 0)
                bp_vp  = spec_sp.basis_eval_radial(v_post, p, 0)
                for k in range(num_p):
                    bk = spec_sp.basis_eval_radial(gx_e, k, 0)
                    
                    cc_collision[p, :, k, :] =  np.einsum("p,prs->rs", gw_e, np.einsum("p,prs->prs", (1/4/np.pi) * gx_e**3 * total_cs * bk,
                                                                            (energy_split * np.einsum("p,rs->prs", bp_vp, Wqs_lm[0]) - np.einsum("p,rs->prs", bp, Wqs_lm[1]))))


            cc_collision = vth * cc_collision.reshape((num_p * nl, num_p * nl))                    
            return cc_collision
                      


                    

        else:
            raise NotImplementedError("only splines basis is implemented for hsph lop eulerian")
        
            


        

        
