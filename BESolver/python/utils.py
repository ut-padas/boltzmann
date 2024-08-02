"""
@package : Utility functions needed for the Boltzmann solver. 
"""
import numpy as np
import parameters as params
import spec_spherical
import basis
import scipy.constants
from numba import jit

MAX_GMX_Q_VR_PTS=278

def choloskey_inv(M):
    #return np.linalg.pinv(M,rcond=1e-30)
    rtol=1e-14
    atol=1e-14

    T, Q = scipy.linalg.schur(M)
    Tinv = scipy.linalg.solve_triangular(T, np.identity(M.shape[0]),lower=False)
    #print("cholesky solver inverse : ", np.linalg.norm(np.matmul(T,Tinv)-np.eye(T.shape[0]))/np.linalg.norm(np.eye(T.shape[0])))
    return np.matmul(np.linalg.inv(np.transpose(Q)), np.matmul(Tinv, np.linalg.inv(Q))) 
    #print(np.linalg.cond(Q), np.linalg.cond(T))

    # L    = np.linalg.cholesky(M)
    # Linv = scipy.linalg.solve_triangular(L, np.identity(M.shape[0]),lower=True) 
    # print("cholesky solver inverse : ", np.linalg.norm(np.matmul(L,Linv)-np.eye(L.shape[0]))/np.linalg.norm(np.eye(L.shape[0])))
    # return np.matmul(np.transpose(Linv),Linv)

def block_jacobi_inv(M,num_partitions=8):
    #return choloskey_inv(M)
    #return np.linalg.pinv(M,rcond=1e-30)
    rtol=1e-14
    atol=1e-14
    Pinv = np.zeros_like(M)
    for r in range(num_partitions):
        lb = (r * M.shape[0])//num_partitions
        le = ((r+1) * M.shape[0])//num_partitions
        D  = M[lb:le,lb:le]
        Pinv[lb:le,lb:le] = np.linalg.inv(D)

    Mp   = np.matmul(Pinv,M)
    print("preconditioned mass matrix conditioning: %.8E"%np.linalg.cond(Mp))
    Minv = np.matmul(np.linalg.inv(Mp),Pinv)
    assert np.allclose(np.matmul(Minv,M),np.eye(M.shape[0]), rtol=rtol, atol=atol), "preconditioned inverse failed with %.2E rtol"%(rtol)
    return Minv

def cartesian_to_spherical(vx,vy,vz):
    
    r1              = np.sqrt(vx**2 + vy**2 + vz**2)
    theta_p         = np.arccos(np.divide(vz, r1, where=r1>0))
    phi_p           = np.arctan2(vy, vx)
    phi_p           = phi_p % (2 * np.pi)

    return [r1,theta_p,phi_p]

def spherical_to_cartesian(v_abs, v_theta, v_phi):
    return [v_abs * np.sin(v_theta) * np.cos(v_phi), v_abs * np.sin(v_theta) * np.sin(v_phi), v_abs * np.cos(v_theta)]

def mass_op(spec_sp: spec_spherical.SpectralExpansionSpherical, scale=1.0):
    """
    returns the operator that captures the mass of the distribution function. 
    mass = (maxwellian(0) * VTH**3) * np.dot(cf,M_klm) 
    we need to get this properly scaled for the appropiate maxwellian i.e., v_thermal
    """
    num_p        = spec_sp._p +1
    num_sh       = len(spec_sp._sph_harm_lm)

    if spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:
        k_vec        = spec_sp._basis_p._t
        dg_idx       = spec_sp._basis_p._dg_idx
        sp_order     = spec_sp._basis_p._sp_order

        gx_m , gw_m  = spec_sp._basis_p.Gauss_Pn(2 * (sp_order + 2) * spec_sp._basis_p._num_knot_intervals)
        MP_klm       = np.zeros(num_p * num_sh)
        sph_fac      = spec_sp.basis_eval_spherical(0,0,0,0) * 4 * np.pi

        for e_id in range(0,len(dg_idx),2):
            ib = dg_idx[e_id]
            ie = dg_idx[e_id+1]

            for k in range(ib,ie+1):
                k_min  = k_vec[k]
                k_max  = k_vec[k + sp_order + 1]
                qx_idx = np.logical_and(gx_m >= k_min, gx_m <= k_max)

                gmx    = gx_m[qx_idx] 
                gmw    = gw_m[qx_idx]

                MP_klm[k * num_sh + 0] = sph_fac * np.dot(gmw, gmx**2 * spec_sp.basis_eval_radial(gmx, k, 0))
        return MP_klm
    else:
        legendre     = basis.Legendre()
        [glx,glw]    = legendre.Gauss_Pn(NUM_Q_VT)
        VTheta_q     = np.arccos(glx)
        VPhi_q       = np.linspace(0,2*np.pi,NUM_Q_VP)

        assert NUM_Q_VP>1
        sq_fac_v = (2*np.pi/(NUM_Q_VP-1))
        WVPhi_q  = np.ones(NUM_Q_VP)*sq_fac_v

        #trap. weights
        WVPhi_q[0]  = 0.5 * WVPhi_q[0]
        WVPhi_q[-1] = 0.5 * WVPhi_q[-1]
        l_modes      = list(set([l for l,_ in spec_sp._sph_harm_lm])) 
        
        mm_g  = np.array([])

        for e_id, ele_domain in enumerate(spec_sp._r_grid):
            spec_sp._basis_p = spec_sp._r_basis_p[e_id]
            [gmx,gmw]    = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR)
            quad_grid    = np.meshgrid(gmx,VTheta_q,VPhi_q,indexing='ij')
            Y_lm = spec_sp.Vq_sph(quad_grid[1],quad_grid[2])

            if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY\
                or spec_sp.get_radial_basis_type() == basis.BasisType.LAGUERRE\
                or spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_ENERGY_POLY:
                MP_klm = np.array([ spec_sp.Vq_r(quad_grid[0],l) * Y_lm[lm_idx] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm)])
            elif spec_sp.get_radial_basis_type() == basis.BasisType.CHEBYSHEV_POLY:
                MP_klm = np.array([ spec_sp._basis_p.Wx()(quad_grid[0]) * spec_sp.Vq_r(quad_grid[0],l) * Y_lm[lm_idx] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm)])
            else:
                raise NotImplementedError

            MP_klm = np.dot(MP_klm,WVPhi_q)
            MP_klm = np.dot(MP_klm,glw)
            MP_klm = np.dot(MP_klm,gmw)
            MP_klm = np.transpose(MP_klm)
            MP_klm = MP_klm.reshape(num_p*num_sph_harm)

            for lm_idx, lm in enumerate(sph_harm_lm):
                if lm[0]>0:
                    MP_klm[lm_idx::num_sph_harm] = 0.0

            mm_g = np.append(mm_g, MP_klm)
            MP_klm   = mm_g
    
        return MP_klm

def mean_velocity_op(spec_sp: spec_spherical.SpectralExpansionSpherical, NUM_Q_VR, NUM_Q_VT, NUM_Q_VP, scale=1.0):
    
    num_p        = spec_sp._p +1
    sph_harm_lm  = spec_sp._sph_harm_lm
    num_sh       = len(sph_harm_lm)
    
    Vop          = np.zeros((3, num_p * num_sh))
    
    [glx,glw]    = basis.Legendre().Gauss_Pn(NUM_Q_VT)
    vt_q, vt_w   = np.arccos(glx), glw
    
    [glx,glw]    = basis.Legendre().Gauss_Pn(NUM_Q_VP)
    vp_q, vp_w   = np.pi * glx + np.pi , np.pi * glw
    
    def alpha_phi(m, x):
        if m ==0 :
            return np.ones_like(x)
        elif m > 0:
            return np.cos(m * x)
        else:
            assert m < 0
            return np.sin((-m) * x)
    
    angular_q = list()
    for lm_idx, lm in enumerate(sph_harm_lm):
        l = lm [0]
        m = lm [1]
        
        sin_vt_l = np.dot(vt_w, np.sin(vt_q) * spec_sp.basis_eval_spherical(vt_q, 0 * vt_q, l, 0))
        cos_vt_l = np.dot(vt_w, np.cos(vt_q) * spec_sp.basis_eval_spherical(vt_q, 0 * vt_q, l, 0))
        
        sin_vp_m = np.dot(vp_w, np.sin(vp_q) * alpha_phi(m, vp_q))
        cos_vp_m = np.dot(vp_w, np.cos(vp_q) * alpha_phi(m, vp_q))
        i1_m     = np.dot(vp_w, alpha_phi(m, vp_q))
        
        angular_q.append((cos_vt_l, sin_vt_l, cos_vp_m, sin_vp_m, i1_m))
    
    
    gx_e, gw_e = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR)
    if spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:
        k_vec        = spec_sp._basis_p._t
        dg_idx       = spec_sp._basis_p._dg_idx
        sp_order     = spec_sp._basis_p._sp_order
        
        for e_id in range(0,len(dg_idx),2):
            ib = dg_idx[e_id]
            ie = dg_idx[e_id+1]

            xb = k_vec[ib]
            xe = k_vec[ie+sp_order+1]
            
            for idx in range(ib, ie):
                k_min, k_max  = k_vec[idx], k_vec[idx + sp_order + 1]
                
                qx_idx = np.logical_and(gx_e >= k_min, gx_e <= k_max)
                gmx    = gx_e[qx_idx] 
                gmw    = gw_e[qx_idx]
                
                radial_q = np.dot(gmw, gmx**3 * spec_sp.basis_eval_radial(gmx, idx, 0))
                
                for lm_idx , lm in enumerate(sph_harm_lm):
                    tmp = angular_q[lm_idx]
                    
                    cos_vt_l = tmp[0]
                    sin_vt_l = tmp[1]
                    
                    cos_vp_m = tmp[2]
                    sin_vp_m = tmp[3]
                    
                    i1_m     = tmp[4]
                    
                    Vop[0, idx * num_sh + lm_idx ] = radial_q * sin_vt_l * cos_vp_m
                    Vop[1, idx * num_sh + lm_idx ] = radial_q * sin_vt_l * sin_vp_m
                    Vop[2, idx * num_sh + lm_idx ] = radial_q * cos_vt_l * i1_m
        
        return Vop
                
    else:
        raise NotImplementedError

def kinetic_stress_tensor_op(spec_sp: spec_spherical.SpectralExpansionSpherical, NUM_Q_VR, NUM_Q_VT, NUM_Q_VP, scale=1.0):
    num_p        = spec_sp._p +1
    sph_harm_lm  = spec_sp._sph_harm_lm
    num_sh       = len(sph_harm_lm)
    
    Vop          = np.zeros((6, num_p * num_sh))
    [glx,glw]    = basis.Legendre().Gauss_Pn(NUM_Q_VT)
    vt_q, vt_w   = np.arccos(glx), glw
    
    [glx,glw]    = basis.Legendre().Gauss_Pn(NUM_Q_VP)
    vp_q, vp_w   = np.pi * glx + np.pi , np.pi * glw
    
    def alpha_phi(m, x):
        if m ==0 :
            return np.ones_like(x)
        elif m > 0:
            return np.cos(m * x)
        else:
            assert m < 0
            return np.sin((-m) * x)
    
    angular_q = list()
    for lm_idx, lm in enumerate(sph_harm_lm):
        l  = lm [0]
        m  = lm [1]
        
        qq = dict()
        alpha_phi_m               = alpha_phi(m, vp_q)
        Y_l0                      = spec_sp.basis_eval_spherical(vt_q, 0 * vt_q, l, 0)
        
        qq["cos^2(vphi)"]         = np.dot(vp_w, np.cos(vp_q)**2                * alpha_phi_m)
        qq["cos(vphi)sin(vphi)"]  = np.dot(vp_w, np.cos(vp_q)* np.sin(vp_q)     * alpha_phi_m)
        qq["cos(vphi)"]           = np.dot(vp_w, np.cos(vp_q)                   * alpha_phi_m)
        qq["sin^2(vphi)"]         = np.dot(vp_w, np.sin(vp_q)**2                * alpha_phi_m)
        qq["sin(vphi)"]           = np.dot(vp_w, np.sin(vp_q)                   * alpha_phi_m)
        qq["1_phi"]               = np.dot(vp_w, 1                              * alpha_phi_m)
        
        qq["sin^2(vtheta)"]                = np.dot(vt_w, np.sin(vt_q)**2                            * Y_l0)
        qq["cos(vtheta)sin(vtheta)"]       = np.dot(vt_w, np.sin(vt_q)* np.cos(vt_q)                 * Y_l0)
        qq["cos^2(vtheta)"]                = np.dot(vt_w, np.cos(vt_q)**2                            * Y_l0)
        
        
        
        
        angular_q.append(qq)
        
    
    gx_e, gw_e = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR)
    if spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:
        k_vec        = spec_sp._basis_p._t
        dg_idx       = spec_sp._basis_p._dg_idx
        sp_order     = spec_sp._basis_p._sp_order
        
        for e_id in range(0,len(dg_idx),2):
            ib = dg_idx[e_id]
            ie = dg_idx[e_id+1]

            xb = k_vec[ib]
            xe = k_vec[ie+sp_order+1]
            
            for idx in range(ib, ie):
                k_min, k_max  = k_vec[idx], k_vec[idx + sp_order + 1]
                
                qx_idx = np.logical_and(gx_e >= k_min, gx_e <= k_max)
                gmx    = gx_e[qx_idx] 
                gmw    = gw_e[qx_idx]
                
                radial_q = np.dot(gmw, gmx**4 * spec_sp.basis_eval_radial(gmx, idx, 0))
                
                for lm_idx , lm in enumerate(sph_harm_lm):
                    qq = angular_q[lm_idx]
                    
                    # xx, xy, xz
                    Vop[0, idx * num_sh + lm_idx ] = radial_q * qq["sin^2(vtheta)"]          * qq["cos^2(vphi)"]
                    Vop[1, idx * num_sh + lm_idx ] = radial_q * qq["sin^2(vtheta)"]          * qq["cos(vphi)sin(vphi)"]
                    Vop[2, idx * num_sh + lm_idx ] = radial_q * qq["cos(vtheta)sin(vtheta)"] * qq["cos(vphi)"]
                    
                    # yy, yz
                    Vop[3, idx * num_sh + lm_idx ] = radial_q * qq["sin^2(vtheta)"]          * qq["sin^2(vphi)"]
                    Vop[4, idx * num_sh + lm_idx ] = radial_q * qq["cos(vtheta)sin(vtheta)"] * qq["sin(vphi)"]
                    
                    # zz 
                    Vop[5, idx * num_sh + lm_idx ] = radial_q * qq["cos^2(vtheta)"] * qq["1_phi"]
                    
        return Vop
                
    else:
        raise NotImplementedError

def temp_op(spec_sp: spec_spherical.SpectralExpansionSpherical, scale=1.0):
    
    num_p        = spec_sp._p +1
    num_sh       = len(spec_sp._sph_harm_lm)

    if spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:
        
        k_vec        = spec_sp._basis_p._t
        dg_idx       = spec_sp._basis_p._dg_idx
        sp_order     = spec_sp._basis_p._sp_order

        gx_m , gw_m  = spec_sp._basis_p.Gauss_Pn(2 * (sp_order + 3) * spec_sp._basis_p._num_knot_intervals)
        MP_klm       = np.zeros(num_p * num_sh)
        sph_fac      = spec_sp.basis_eval_spherical(0,0,0,0) * 4 * np.pi

        for e_id in range(0,len(dg_idx),2):
            ib = dg_idx[e_id]
            ie = dg_idx[e_id+1]

            for k in range(ib,ie+1):
                k_min  = k_vec[k]
                k_max  = k_vec[k + sp_order + 1]
                qx_idx = np.logical_and(gx_m >= k_min, gx_m <= k_max)

                gmx    = gx_m[qx_idx] 
                gmw    = gw_m[qx_idx]

                MP_klm[k * num_sh + 0] = sph_fac * np.dot(gmw, gmx**4 * spec_sp.basis_eval_radial(gmx, k, 0))
        return MP_klm

    else:
        legendre     = basis.Legendre()
        [glx,glw]    = legendre.Gauss_Pn(NUM_Q_VT)
        VTheta_q     = np.arccos(glx)
        VPhi_q       = np.linspace(0,2*np.pi,NUM_Q_VP)

        assert NUM_Q_VP>1
        sq_fac_v = (2*np.pi/(NUM_Q_VP-1))
        WVPhi_q  = np.ones(NUM_Q_VP)*sq_fac_v

        #trap. weights
        WVPhi_q[0]  = 0.5 * WVPhi_q[0]
        WVPhi_q[-1] = 0.5 * WVPhi_q[-1]
        l_modes      = list(set([l for l,_ in spec_sp._sph_harm_lm])) 

        mm_g         = np.array([])
        for e_id, ele_domain in enumerate(spec_sp._r_grid):
            spec_sp._basis_p = spec_sp._r_basis_p[e_id]
            [gmx,gmw]    = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR)
            quad_grid    = np.meshgrid(gmx,VTheta_q,VPhi_q,indexing='ij')
            Y_lm = spec_sp.Vq_sph(quad_grid[1],quad_grid[2])

            if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY\
                or spec_sp.get_radial_basis_type() == basis.BasisType.LAGUERRE\
                or spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_ENERGY_POLY:
                MP_klm = np.array([(quad_grid[0]**(2)) * spec_sp.Vq_r(quad_grid[0],l) * Y_lm[lm_idx] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm)])
            elif spec_sp.get_radial_basis_type() == basis.BasisType.CHEBYSHEV_POLY:
                MP_klm = np.array([quad_grid[0]**2 * spec_sp._basis_p.Wx()(quad_grid[0]) * spec_sp.Vq_r(quad_grid[0],l) * Y_lm[lm_idx] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm)])
            else:
                raise NotImplementedError

            MP_klm = np.dot(MP_klm,WVPhi_q)
            MP_klm = np.dot(MP_klm,glw)
            MP_klm = np.dot(MP_klm,gmw)
            MP_klm = np.transpose(MP_klm)
            MP_klm = MP_klm.reshape(num_p*num_sph_harm)

            for lm_idx, lm in enumerate(sph_harm_lm):
                if lm[0]>0:
                    MP_klm[lm_idx::num_sph_harm] = 0.0
                    
            mm_g   = np.append(mm_g, MP_klm)

        MP_klm = mm_g
        return MP_klm

def moment_n_f(spec_sp: spec_spherical.SpectralExpansionSpherical,cf, maxwellian, V_TH, moment, NUM_Q_VR, NUM_Q_VT, NUM_Q_VP, scale=1.0, v0=np.zeros(3)):
    """
    Computes the zero th velocity moment, i.e. number density
    \int_v f(v) dv
    """
    if NUM_Q_VR is None:
        NUM_Q_VR     = params.BEVelocitySpace.NUM_Q_VR
        
    if NUM_Q_VT is None:
        NUM_Q_VT     = params.BEVelocitySpace.NUM_Q_VT
    
    if NUM_Q_VP is None:
        NUM_Q_VP     = params.BEVelocitySpace.NUM_Q_VP
    
    num_p        = spec_sp._p +1
    sph_harm_lm  = spec_sp._sph_harm_lm 
    num_sph_harm = len(sph_harm_lm)
    
    legendre     = basis.Legendre()
    [glx,glw]    = legendre.Gauss_Pn(NUM_Q_VT)
    VTheta_q     = np.arccos(glx)
    VPhi_q       = np.linspace(0,2*np.pi,NUM_Q_VP)

    assert NUM_Q_VP>1
    sq_fac_v = (2*np.pi/(NUM_Q_VP-1))
    WVPhi_q  = np.ones(NUM_Q_VP)*sq_fac_v

    #trap. weights
    WVPhi_q[0]  = 0.5 * WVPhi_q[0]
    WVPhi_q[-1] = 0.5 * WVPhi_q[-1]
    l_modes      = list(set([l for l,_ in spec_sp._sph_harm_lm]))
    maxwellian_fac = maxwellian(0)
    mm_g         = np.array([])
    for e_id, ele_domain in enumerate(spec_sp._r_grid):
        spec_sp._basis_p = spec_sp._r_basis_p[e_id]
        [gmx,gmw]    = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR)
        quad_grid    = np.meshgrid(gmx,VTheta_q,VPhi_q,indexing='ij')
        Y_lm = spec_sp.Vq_sph(quad_grid[1],quad_grid[2])
    
        quad_grid_cart = spherical_to_cartesian(quad_grid[0],quad_grid[1],quad_grid[2])
        quad_grid_cart[0]+=v0[0]
        quad_grid_cart[1]+=v0[1]
        quad_grid_cart[2]+=v0[2]
        norm_v  = np.sqrt(quad_grid_cart[0]**2 + quad_grid_cart[1]**2 + quad_grid_cart[2]**2) 

        if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY\
            or spec_sp.get_radial_basis_type() == basis.BasisType.LAGUERRE\
            or spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_ENERGY_POLY:
            MP_klm = np.array([ (norm_v**moment) * spec_sp.Vq_r(quad_grid[0],l) * Y_lm[lm_idx] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm)])
        elif spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:
            MP_klm = np.array([ (norm_v**moment) * (quad_grid[0]**(2)) * spec_sp.Vq_r(quad_grid[0],l) * Y_lm[lm_idx] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm)])
        elif spec_sp.get_radial_basis_type() == basis.BasisType.CHEBYSHEV_POLY:
            MP_klm = np.array([ (norm_v**moment) * spec_sp._basis_p.Wx()(quad_grid[0]) * spec_sp.Vq_r(quad_grid[0],l) * Y_lm[lm_idx] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm)])
        else:
            raise NotImplementedError
        
        MP_klm = np.dot(MP_klm,WVPhi_q)
        MP_klm = np.dot(MP_klm,glw)
        MP_klm = np.dot(MP_klm,gmw)
        MP_klm = np.transpose(MP_klm)
        MP_klm = MP_klm.reshape(num_p*num_sph_harm)

        mm_g = np.append(mm_g, MP_klm)

    MP_klm = mm_g
    m_k    = (maxwellian_fac * (V_TH**(3+moment))) * scale * np.dot(cf, MP_klm)
    return m_k

def compute_avg_temp(particle_mass,spec_sp: spec_spherical.SpectralExpansionSpherical,cf, maxwellian, V_TH, NUM_Q_VR, NUM_Q_VT, NUM_Q_VP, m0=None, scale=1.0, v0=np.zeros(3)):
    if m0 is None:
        m0         = moment_n_f(spec_sp,cf,maxwellian,V_TH,0,NUM_Q_VR,NUM_Q_VT,NUM_Q_VP,scale)

    m2         = moment_n_f(spec_sp,cf,maxwellian,V_TH,2,NUM_Q_VR,NUM_Q_VT,NUM_Q_VP,scale, v0)
    #avg energy per particle
    avg_energy = (0.5*particle_mass*m2)/m0
    temp       = (2/(3*scipy.constants.Boltzmann)) * avg_energy
    return temp

def function_to_basis(spec_sp: spec_spherical.SpectralExpansionSpherical, hv, maxwellian, NUM_Q_VR, NUM_Q_VT, NUM_Q_VP, Minv=None):
    """
    Note the function is assumed to be f(v) = M(v) h(v) and computes the projection coefficients
    should be compatible with the weight function of the polynomials. 
    """
    num_p        = spec_sp._p +1
    sph_harm_lm  = spec_sp._sph_harm_lm
    num_sph_harm = len(sph_harm_lm)
    
    legendre     = basis.Legendre()
    [glx,glw]    = legendre.Gauss_Pn(NUM_Q_VT)
    VTheta_q     = np.arccos(glx)
    VPhi_q       = np.linspace(0,2*np.pi,NUM_Q_VP)

    assert NUM_Q_VP>1
    sq_fac_v = (2*np.pi/(NUM_Q_VP-1))
    WVPhi_q  = np.ones(NUM_Q_VP)*sq_fac_v

    #trap. weights
    WVPhi_q[0]  = 0.5 * WVPhi_q[0]
    WVPhi_q[-1] = 0.5 * WVPhi_q[-1]

    l_modes      = list(set([l for l,_ in spec_sp._sph_harm_lm])) 
    mm_g         = np.array([])
    for e_id, ele_domain in enumerate(spec_sp._r_grid):
        spec_sp._basis_p = spec_sp._r_basis_p[e_id]
        [gmx,gmw]    = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR)
        quad_grid    = np.meshgrid(gmx,VTheta_q,VPhi_q,indexing='ij')
        Y_lm = spec_sp.Vq_sph(quad_grid[1],quad_grid[2])

        if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_ENERGY_POLY:
            hq   = hv(quad_grid[0],quad_grid[1],quad_grid[2]) / (np.exp(-quad_grid[0]**4) + 1.e-16)
        elif spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY\
            or spec_sp.get_radial_basis_type() == basis.BasisType.LAGUERRE:
            hq   = hv(quad_grid[0],quad_grid[1],quad_grid[2]) / (np.exp(-quad_grid[0]**2) + 1.e-16)
        elif spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:
            hq   = hv(quad_grid[0],quad_grid[1],quad_grid[2]) * quad_grid[0]**2
        elif spec_sp.get_radial_basis_type() == basis.BasisType.CHEBYSHEV_POLY:
            vw     = spec_sp._basis_p._window
            vt     = (quad_grid[0] - 0.5 * (vw[0] + vw[1])) / (0.5 * (vw[1]- vw[0]))
            hq     = hv(quad_grid[0] , quad_grid[1],quad_grid[2]) * (quad_grid[0] **2) * np.sqrt(1-(vt)**2)
        else:
            raise NotImplementedError
        
        M_klm = np.array([ hq * spec_sp.Vq_r(quad_grid[0],l) * Y_lm[lm_idx] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm)])
        M_klm = np.dot(M_klm,WVPhi_q)
        M_klm = np.dot(M_klm,glw)
        M_klm = np.dot(M_klm,gmw)
        M_klm = np.transpose(M_klm)
        M_klm = M_klm.reshape(num_p*num_sph_harm)
        mm_g  = np.append(mm_g, M_klm)

    M_klm = mm_g
    if Minv is not None:
        M_klm = np.matmul(Minv, M_klm)
    
    return M_klm

def thermal_projection(spec_sp: spec_spherical.SpectralExpansionSpherical,mw_1,vth_1,mw_2,vth_2, NUM_Q_VR, NUM_Q_VT, NUM_Q_VP, scale=1.0):
    """
    v_a = v/v_th 
    computes the \int_{R^3} M(v_a) f(v_a) Pi(vth_1) Pj(vth_2) dv
    !! Needs to multiply by mass mat inverse before use. 
    """
    
    num_p        = spec_sp._p +1
    sph_harm_lm  = spec_sp._sph_harm_lm
    num_sh       = len(sph_harm_lm)
    
    # legendre     = basis.Legendre()
    # [glx,glw]    = legendre.Gauss_Pn(NUM_Q_VT)
    # VTheta_q     = np.arccos(glx)
    # VPhi_q       = np.linspace(0,2*np.pi,NUM_Q_VP)

    # assert NUM_Q_VP>1
    # sq_fac_v = (2*np.pi/(NUM_Q_VP-1))
    # WVPhi_q  = np.ones(NUM_Q_VP)*sq_fac_v

    # #trap. weights
    # WVPhi_q[0]  = 0.5 * WVPhi_q[0]
    # WVPhi_q[-1] = 0.5 * WVPhi_q[-1]
    
    if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY:
        
        if NUM_Q_VR is None:
            NUM_Q_VR     = min(MAX_GMX_Q_VR_PTS,params.BEVelocitySpace.NUM_Q_VR)
        
        if NUM_Q_VT is None:
            NUM_Q_VT     = params.BEVelocitySpace.NUM_Q_VT
        
        if NUM_Q_VP is None:
            NUM_Q_VP     = params.BEVelocitySpace.NUM_Q_VP
        
        [gx,gw]    = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR)
        weight_func  = spec_sp._basis_p.Wx()
        
        l_modes    = list(set([l for l,_ in spec_sp._sph_harm_lm]))
        vth1_by_vth2 = vth_1 / vth_2
        
        mm=np.zeros((num_p*num_sh, num_p*num_sh))
        for j,l in enumerate(l_modes):
            v1_qr = spec_sp.Vq_r(vth1_by_vth2 * gx, l)
            v2_qr = spec_sp.Vq_r(gx, l)
            mm_l = np.array([ v1_qr[p,:] * v2_qr[k,:] for p in range(num_p) for k in range(num_p)])
            mm_l = np.dot(mm_l,gw).reshape(num_p,num_p)
            
            for lm_idx, (l1,m) in enumerate(spec_sp._sph_harm_lm):
                if(l==l1):
                    for p in range(num_p):
                        for k in range(num_p):
                            idx_pqs = p * num_sh + lm_idx
                            idx_klm = k * num_sh + lm_idx
                            mm[idx_pqs, idx_klm] = mm_l[p,k]
            
        
        return mm
    
    elif spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:
        
        if NUM_Q_VR is None:
            NUM_Q_VR     = params.BEVelocitySpace.NUM_Q_VR
        
        if NUM_Q_VT is None:
            NUM_Q_VT     = params.BEVelocitySpace.NUM_Q_VT
        
        if NUM_Q_VP is None:
            NUM_Q_VP     = params.BEVelocitySpace.NUM_Q_VP
        
        [gx,gw]    = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR)
        weight_func  = spec_sp._basis_p.Wx()
        
        l_modes    = list(set([l for l,_ in spec_sp._sph_harm_lm]))
        vth1_by_vth2 = vth_1 / vth_2
        
        # note: integration perfomed in the vth2 normalize coord, so the factors cancel out with the mass mat. 
        mm = np.zeros((num_p*num_sh, num_p*num_sh))
        for j,l in enumerate(l_modes):
            mr   = gx**2 
            v1_qr = spec_sp.Vq_r(vth1_by_vth2 * gx, l)
            v2_qr = spec_sp.Vq_r(gx, l)
            mm_l = np.array([ mr * v1_qr[p,:] * v2_qr[k,:] for p in range(num_p) for k in range(num_p)])
            mm_l = np.dot(mm_l,gw).reshape(num_p,num_p)
            for lm_idx, (l1,m) in enumerate(spec_sp._sph_harm_lm):
                if(l==l1):
                    for p in range(num_p):
                        for k in range(num_p):
                            idx_pqs = p * num_sh + lm_idx
                            idx_klm = k * num_sh + lm_idx
                            mm[idx_pqs, idx_klm] = mm_l[p,k]
            
        
        return mm
    else:
        raise NotImplementedError("not implemented for specified basis")

def vcenter_projection(spec_sp: spec_spherical.SpectralExpansionSpherical,mw,vth, NUM_Q_VR, NUM_Q_VT, NUM_Q_VP, v0, scale=1.0):
    """
    compute the v center projection operator
    v0- shift in the choosen v center. 
    P= Minv \int_{R^3} \psi_{pqs}(v) \phi_{klm}(v-v0) dv 
    """

    num_p        = spec_sp._p +1
    sph_harm_lm  = spec_sp._sph_harm_lm
    num_sh       = len(sph_harm_lm)

    if NUM_Q_VR is None:
        if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY\
            or spec_sp.get_radial_basis_type() == basis.BasisType.LAGUERRE\
            or spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_ENERGY_POLY:
            NUM_Q_VR     = min(MAX_GMX_Q_VR_PTS,params.BEVelocitySpace.NUM_Q_VR)
        elif spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:
            NUM_Q_VR     =  params.BEVelocitySpace.NUM_Q_VR

    if NUM_Q_VT is None:
        NUM_Q_VT     = params.BEVelocitySpace.NUM_Q_VT
    
    if NUM_Q_VP is None:
        NUM_Q_VP     = params.BEVelocitySpace.NUM_Q_VP
    
    
    legendre     = basis.Legendre()
    [glx,glw]    = legendre.Gauss_Pn(NUM_Q_VT)
    VTheta_q     = np.arccos(glx)
    VPhi_q       = np.linspace(0,2*np.pi,NUM_Q_VP)

    assert NUM_Q_VP>1
    sq_fac_v = (2*np.pi/(NUM_Q_VP-1))
    WVPhi_q  = np.ones(NUM_Q_VP)*sq_fac_v

    #trap. weights
    WVPhi_q[0]  = 0.5 * WVPhi_q[0]
    WVPhi_q[-1] = 0.5 * WVPhi_q[-1]

    
    if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY\
        or spec_sp.get_radial_basis_type() == basis.BasisType.LAGUERRE\
        or spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_ENERGY_POLY:
        raise NotImplementedError
        
    
    elif spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:
        
        [gmx,gmw]    = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR,True)
        weight_func  = spec_sp._basis_p.Wx()

        quad_grid    = np.meshgrid(gmx,VTheta_q,VPhi_q,indexing='ij')
        l_modes      = list(set([l for l,_ in spec_sp._sph_harm_lm])) 
        
        quad_grid_v0 = spherical_to_cartesian(quad_grid[0], quad_grid[1], quad_grid[2])

        quad_grid_v0[0] -=v0[0] 
        quad_grid_v0[1] -=v0[1] 
        quad_grid_v0[2] -=v0[2] 

        quad_grid_v0 = cartesian_to_spherical(quad_grid_v0[0], quad_grid_v0[1], quad_grid_v0[2])
        
        Vq_sph_lm    = spec_sp.Vq_sph(quad_grid_v0[1], quad_grid_v0[2], 1)
        Vq_sph_qs    = spec_sp.Vq_sph(quad_grid[1], quad_grid[2], 1)

        psi_pqs = np.array([ spec_sp.basis_eval_radial(quad_grid[0],p,q) * Vq_sph_qs[qs_idx] for qs_idx, (q,s) in enumerate(sph_harm_lm) for p in range(num_p)])
        psi_pqs = psi_pqs.reshape(tuple([num_sh,num_p]) + quad_grid[0].shape)
        psi_pqs = np.swapaxes(psi_pqs,0,1)
        
        phi_klm = np.array([ spec_sp.basis_eval_radial(quad_grid_v0[0],k,l) * Vq_sph_lm[lm_idx] for lm_idx, (l,m) in enumerate(sph_harm_lm) for k in range(num_p)])
        phi_klm = phi_klm.reshape(tuple([num_sh,num_p]) + quad_grid[0].shape)
        phi_klm = np.swapaxes(phi_klm,0,1)
        
        mm = np.array([ (quad_grid[0] **2) * psi_pqs[p,qs_idx] * phi_klm[k,lm_idx] for p in range(num_p) for qs_idx in range(num_sh) for k in range(num_p) for lm_idx in range(num_sh)])
        mm = np.dot(mm, WVPhi_q)
        mm = np.dot(mm, glw)
        mm = np.dot(mm, gmw)
        mm = mm.reshape(num_p * num_sh , num_p * num_sh)
        return mm
    else:
        raise NotImplementedError("not implemented for specified basis")

def get_maxwellian_3d(vth,n_scale=1):
    M = lambda x: (n_scale / ((vth * np.sqrt(np.pi))**3) ) * np.exp(-x**2)
    return M

def get_eedf(ev_pts, spec_sp : spec_spherical.SpectralExpansionSpherical, cf, maxwellian, vth,scale=1, v0=np.zeros(3)):
    """
    Assumes spherical harmonic basis in vtheta vphi direction. 
    the integration over the spherical harmonics is done analytically. 
    """
    EV       = scipy.constants.electron_volt
    E_MASS   = scipy.constants.electron_mass

    v0_abs   = np.sqrt(v0[0]**2 + v0[1]**2 + v0[2]**2)
    vr       = np.sqrt(2* EV * ev_pts /E_MASS)/vth
    
    vr[vr<v0_abs]  = 0
    vr[vr>=v0_abs] = vr[vr>=v0_abs] -v0_abs
    
    if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY\
        or spec_sp.get_radial_basis_type() == basis.BasisType.LAGUERRE\
        or spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_ENERGY_POLY:
        NUM_Q_VR     = min(MAX_GMX_Q_VR_PTS,params.BEVelocitySpace.NUM_Q_VR)
    elif spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:
        NUM_Q_VR     =  params.BEVelocitySpace.NUM_Q_VR
    
    NUM_Q_VT     = params.BEVelocitySpace.NUM_Q_VT
    NUM_Q_VP     = params.BEVelocitySpace.NUM_Q_VP
    
    num_p        = spec_sp._p +1
    sph_harm_lm  = spec_sp._sph_harm_lm 
    num_sph_harm = len(sph_harm_lm)
    
    legendre     = basis.Legendre()
    [glx,glw]    = legendre.Gauss_Pn(NUM_Q_VT)
    VTheta_q     = np.arccos(glx)
    VPhi_q       = np.linspace(0,2*np.pi,NUM_Q_VP)

    assert NUM_Q_VP>1
    sq_fac_v = (2*np.pi/(NUM_Q_VP-1))
    WVPhi_q  = np.ones(NUM_Q_VP)*sq_fac_v

    #trap. weights
    WVPhi_q[0]  = 0.5 * WVPhi_q[0]
    WVPhi_q[-1] = 0.5 * WVPhi_q[-1]

    quad_grid = np.meshgrid(vr,VTheta_q,VPhi_q,indexing='ij')

    if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY\
        or spec_sp.get_radial_basis_type() == basis.BasisType.LAGUERRE\
        or spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_ENERGY_POLY:

        l_modes      = list(set([l for l,_ in spec_sp._sph_harm_lm])) 
        
        Y_lm = spec_sp.Vq_sph(quad_grid[1],quad_grid[2])
        
        MP_klm = np.array([ np.exp(-quad_grid[0]**2) * quad_grid[0]**(l) * spec_sp.Vq_r(quad_grid[0],l) * Y_lm[lm_idx] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm)])
        MP_klm = np.dot(MP_klm,WVPhi_q)
        MP_klm = np.dot(MP_klm,glw)
        MP_klm = np.swapaxes(MP_klm,0,1)
        MP_klm = MP_klm.reshape((num_p*num_sph_harm,-1))
        
        return np.dot(np.transpose(MP_klm),cf) 

    elif spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:

        l_modes      = list(set([l for l,_ in spec_sp._sph_harm_lm])) 
        
        Y_lm = spec_sp.Vq_sph(quad_grid[1],quad_grid[2])
        
        MP_klm = np.array([spec_sp.Vq_r(quad_grid[0],l) * Y_lm[lm_idx] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm)])
        MP_klm = np.dot(MP_klm,WVPhi_q)
        MP_klm = np.dot(MP_klm,glw)
        MP_klm = np.swapaxes(MP_klm,0,1)
        MP_klm = MP_klm.reshape((num_p*num_sph_harm,-1))
        return np.dot(np.transpose(MP_klm),cf) 

def sample_distriubtion(vx, vy, vz, spec_sp : spec_spherical.SpectralExpansionSpherical, cf, maxwellian, vth,scale=1):

    # vx, vy, vz must have the same shape

    num_sph = len(spec_sp._sph_harm_lm)

    sph_coord = cartesian_to_spherical(vx, vy, vz)

    result = np.zeros_like(vx)

    for k in range(spec_sp._p+1):
        for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
            result += cf[k*num_sph+lm_idx]*spec_sp.basis_eval_full(sph_coord[0], sph_coord[1], sph_coord[2], k, lm[0], lm[1])*maxwellian(sph_coord[0])*(sph_coord[0]**lm[0])

    return result

def sample_distriubtion_spherical(sph_coord, spec_sp : spec_spherical.SpectralExpansionSpherical, cf, maxwellian, vth,scale=1):

    # vx, vy, vz must have the same shape

    num_sph = len(spec_sp._sph_harm_lm)

    result = np.zeros_like(sph_coord[0])

    for k in range(spec_sp._p+1):
        for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
            result += cf[k*num_sph+lm_idx]*spec_sp.basis_eval_full(sph_coord[0], sph_coord[1], sph_coord[2], k, lm[0], lm[1])*maxwellian(sph_coord[0])*(sph_coord[0]**lm[0])

    return result
    
def compute_radial_components(ev_pts, spec_sp : spec_spherical.SpectralExpansionSpherical, cf, maxwellian,  vth,scale=1, v0=np.zeros(3), mass_op=None):
    """
    Assumes spherical harmonic basis in vtheta vphi direction. 
    the integration over the spherical harmonics is done analytically. 
    """
    EV       = scipy.constants.electron_volt
    E_MASS   = scipy.constants.electron_mass

    #v0_abs  = np.sqrt(v0[0]**2 + v0[1]**2 + v0[2]**2)
    vr       = np.sqrt(2* EV * ev_pts /E_MASS)/vth
    num_p    = spec_sp._p +1 
    num_sh   = len(spec_sp._sph_harm_lm)
    output   = np.zeros((num_sh, len(vr)))
    
    for l_idx, lm in enumerate(spec_sp._sph_harm_lm):

        for e_id, ele_domain in enumerate(spec_sp._r_grid):
            spec_sp._basis_p = spec_sp._r_basis_p[e_id]
            if spec_sp.get_radial_basis_type() == basis.BasisType.CHEBYSHEV_POLY:
                Vqr   = spec_sp.dg_Vq_r(vr,lm[0],e_id,1)
            else:
                Vqr   = spec_sp.Vq_r(vr,lm[0],1)
            output[l_idx, :] += np.dot(cf[e_id * num_p * num_sh : (e_id+1) * num_p *num_sh][l_idx::num_sh], Vqr)
    
    if spec_sp.get_radial_basis_type() == basis.BasisType.LAGUERRE or spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY:
        output *= np.exp(-vr**2)
    
    if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_ENERGY_POLY:
        output *= np.exp(-vr**4)

    if spec_sp.get_radial_basis_type() == basis.BasisType.CHEBYSHEV_POLY:
        output *= np.exp(-vr**2)

    return output

def reaction_rates_op(spec_sp : spec_spherical.SpectralExpansionSpherical, g_list, mw, vth):
    """
    reaction rate op R for collision g

    np.dot(R,f) * vth**3 / mass(f) 

    """
    num_p        = spec_sp._p +1
    sph_harm_lm  = spec_sp._sph_harm_lm 
    num_sh = len(sph_harm_lm)

    if spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:

        k_vec      = spec_sp._basis_p._t
        dg_idx     = spec_sp._basis_p._dg_idx
        sp_order   = spec_sp._basis_p._sp_order

        gmx_a, gmw_a = spec_sp._basis_p.Gauss_Pn(spec_sp._num_q_radial)
        c_gamma  = np.sqrt(2*scipy.constants.e / scipy.constants.m_e)
        cs_total = 0
        for g in g_list:
            cs_total += g.total_cross_section((gmx_a * vth / c_gamma)**2)

        rr_op    = np.zeros(num_p)

        for p in range(num_p):
            qx_idx = np.logical_and(gmx_a >= k_vec[p], gmx_a <= k_vec[p + sp_order + 1])
            gmx    = gmx_a[qx_idx]
            gmw    = gmw_a[qx_idx]

            rr_op[p] = (2 * vth**4 / c_gamma**3) * np.dot(gmw, gmx**3 * cs_total[qx_idx] * spec_sp.basis_eval_radial(gmx, p, 0))

        return rr_op
    else:
        raise NotImplementedError
    
def mobility_op(spec_sp : spec_spherical.SpectralExpansionSpherical, mw, vth):
    
    num_p        = spec_sp._p +1
    sph_harm_lm  = spec_sp._sph_harm_lm 
    num_sh = len(sph_harm_lm)

    c_gamma  = np.sqrt(2*scipy.constants.e / scipy.constants.m_e)
    rr_op    = np.zeros(num_p)
    
    if spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:
        k_vec      = spec_sp._basis_p._t
        dg_idx     = spec_sp._basis_p._dg_idx
        sp_order   = spec_sp._basis_p._sp_order

        gmx_a, gmw_a = spec_sp._basis_p.Gauss_Pn(2 * (sp_order + 3) * spec_sp._basis_p._num_knot_intervals)

        for p in range(num_p):
            qx_idx   = np.logical_and(gmx_a >= k_vec[p], gmx_a <= k_vec[p + sp_order + 1])
            gmx      = gmx_a[qx_idx]
            gmw      = gmw_a[qx_idx]
            rr_op[p] = (2 * vth**4 / c_gamma**4) * np.dot(gmw, gmx**3 * spec_sp.basis_eval_radial(gmx, p, 0))
        return rr_op
    else:
        raise NotImplementedError
    
def diffusion_op(spec_sp : spec_spherical.SpectralExpansionSpherical, g_list, mw, vth):

    c_gamma      = np.sqrt(2*scipy.constants.e / scipy.constants.m_e)
    
    num_p        = spec_sp._p +1
    sph_harm_lm  = spec_sp._sph_harm_lm 
    num_sh       = len(sph_harm_lm)

    
    rr_op    = np.zeros(num_p)
    
    if spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:
        k_vec      = spec_sp._basis_p._t
        dg_idx     = spec_sp._basis_p._dg_idx
        sp_order   = spec_sp._basis_p._sp_order

        gmx_a, gmw_a = spec_sp._basis_p.Gauss_Pn(spec_sp._num_q_radial)
        total_cs     = 0
        
        for g in g_list:
            total_cs += g.total_cross_section((gmx_a * vth / c_gamma)**2)
    
        for p in range(num_p):
            qx_idx = np.logical_and(gmx_a >= k_vec[p], gmx_a <= k_vec[p + sp_order + 1])
            gmx    = gmx_a[qx_idx]
            gmw    = gmw_a[qx_idx]
            rr_op[p] = (2 * vth**4 / c_gamma**4) * np.dot(gmw, gmx**3 * spec_sp.basis_eval_radial(gmx, p, 0) / total_cs[qx_idx])
        return rr_op
    else:
        raise NotImplementedError

def growth_rates_op(spec_sp : spec_spherical.SpectralExpansionSpherical, g_list, mw, vth):
    """
    growth rate op R for collision g
    np.dot(R,f) * vth**3 / mass(f) 
    """
    num_p        = spec_sp._p +1
    sph_harm_lm  = spec_sp._sph_harm_lm 
    num_sh       = len(sph_harm_lm)

    if spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:

        k_vec      = spec_sp._basis_p._t
        dg_idx     = spec_sp._basis_p._dg_idx
        sp_order   = spec_sp._basis_p._sp_order

        gmx_a, gmw_a = spec_sp._basis_p.Gauss_Pn((sp_order + 3) * spec_sp._basis_p._num_knot_intervals)
        c_gamma  = np.sqrt(2*scipy.constants.e / scipy.constants.m_e)
        cs_total = 0
        for g in g_list:
            cs_total += g.total_cross_section((gmx_a * vth / c_gamma)**2)

        rr_op    = np.zeros(num_p)

        for p in range(num_p):
            qx_idx = np.logical_and(gmx_a >= k_vec[p], gmx_a <= k_vec[p + sp_order + 1])
            gmx    = gmx_a[qx_idx]
            gmw    = gmw_a[qx_idx]

            rr_op[p] = vth * np.dot(gmw, gmx**3 * cs_total[qx_idx] * spec_sp.basis_eval_radial(gmx, p, 0))

        return rr_op
    else:
        raise NotImplementedError
    
def normalized_distribution(spec_sp, mm_op, f_vec, maxwellian,vth):
    c_gamma      = np.sqrt(2*scipy.constants.e / scipy.constants.m_e)
    num_p        = spec_sp._p +1
    num_sh       = len(spec_sp._sph_harm_lm)
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

def mcmc_chain(target_pdf, prior_samples, n_samples, burn_in):

    chain_sz        = np.int64(n_samples  + burn_in * n_samples)
    state_offset    = 1
    
    uniform         = np.log(np.random.uniform(0, 1, size=chain_sz))
    xp              = prior_samples(size=chain_sz+1)
    xpp             = prior_samples(size=chain_sz+1)

    p_xp            = target_pdf(xp)
    p_xpp           = target_pdf(xpp)

    p_xp[p_xp<=1e-14]    = 1e-14
    p_xpp[p_xpp<=1e-14]  = 1e-14

    p_xp            = np.log(p_xp)
    p_xpp           = np.log(p_xpp)

    a_ratio         = np.zeros(chain_sz,dtype=np.int64)
    states          = np.zeros_like(xp)

    for i in range(chain_sz):
        states[i + state_offset] = xp[i]
        acceptance = min(p_xpp[i] - p_xp[i], 0) #min(p_xpp[i]/p_xp[i],1) more stable log evaluation
        
        if uniform[i] <= acceptance:
            a_ratio[i] = 1.0
            xp[i+1]    = xpp[i]
            p_xp[i+1]  = p_xpp[i]
        else:
            xp[i+1]    = xp[i]
            p_xp[i+1]  = p_xp[i]

    return states[-n_samples:]

def mcmc_sampling(dist_pdf, prior, n_samples, burn_in=0.3, num_chains=4):
    
    # #print(hops,burn_in)
    # import multiprocessing as mp
    # pool = mp.Pool()

    # result_list = []
    # def log_result(result):
    #     # This is called whenever foo_pool(i) returns a result.
    #     # result_list is modified only by the main process, not the pool workers.
    #     result_list.append(result)
    
    # for i in range(num_chains):
    #     pi = pool.apply_async(mcmc_chain, args=(dist_pdf, prior, n_samples, burn_in), callback=log_result)
        
    # pool.close()
    # pool.join()

    # samples = np.copy(result_list[0])
    # for i in range(1,num_chains):
    #     samples = np.append(samples, result_list[i],axis=0)
    samples = mcmc_chain(dist_pdf, prior, n_samples, burn_in)
    return samples
    
def sample_distribution_with_uniform_prior(spec_sp, ss_sol, x_domain, vth, fname, n_samples):
    """
    @brief sample given distribution function with uniform prior.    
    """
    num_p     = spec_sp._p + 1
    num_sh    = len(spec_sp._sph_harm_lm)

    def ss_dist(xx):
        s=0
        vr = np.linalg.norm(xx,axis=1)
        vt = np.arccos(xx[:,2]/vr)
        vp = np.arctan2(xx[:,1], xx[:,0]) % (2 * np.pi)
        
        for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
            sph_v = spec_sp.basis_eval_spherical(vt, vp, lm[0],lm[1])
            for k in range(num_p):
                bk_vr = spec_sp.basis_eval_radial(vr,k,0)
                s+=(ss_sol[k * num_sh + lm_idx] * bk_vr * sph_v) 
        return np.abs(s)
    
    def prior_dist(size):
        return np.random.uniform([-x_domain[1], -x_domain[1], -x_domain[1]], [x_domain[1], x_domain[1], x_domain[1]], size=(size,3))

    x_cart   = mcmc_sampling(ss_dist, prior_dist, n_samples, burn_in=0.9, num_chains = 1)
    x_cart  *= vth

    np.save("%s"%(fname), x_cart)
    return 0
    