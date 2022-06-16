"""
@package : Utility functions needed for the Boltzmann solver. 
"""
import numpy as np
import parameters as params
import spec_spherical
import basis
import scipy.constants

MAX_GMX_Q_VR_PTS=278

def choloskey_inv(M):
    #return np.linalg.pinv(M,rcond=1e-30)
    rtol=1e-14
    atol=1e-14
    L    = np.linalg.cholesky(M)
    Linv = scipy.linalg.solve_triangular(L, np.identity(M.shape[0]),lower=True) 
    return np.matmul(np.transpose(Linv),Linv)

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

def maxwellian_normalized(v_abs):
    """
    Normalized Maxwellian without 
    any properties of the application parameters. 
    v_abs - norm(v), scalar

    The steady state maxwellian depends on, 
    mass of the particle, k_B, steady state temperature, and the number density
    M(vp) = A exp(-m * vp**2/ 2 * k_B * T)

    For the computations we can use the normalized maxwellian and use 
    v =  \sqrt(m/(k_B * T)) vp go to application specific maxwellian. 

    """
    #v_l2 = np.linalg.norm(v,2)
    return np.exp(-0.5*(v_abs**2))

def gaussian(v,mu=None,sigma=1.0):
    """
    Gaussian distribution function for d dim
    v  = np.array (dim,)
    mu = None, then mu=0 will be used. 
    sigma =1.0 standard deviation
    """
    if mu is None:
        mu = np.zeros(v.shape)
    return ( 1.0/(sigma * np.sqrt(2 * np.pi)) ) * np.exp(-0.5 * np.linalg.norm((v-mu),2)**2/(sigma**2))

def cartesian_to_spherical(vx,vy,vz):
    
    r1              = np.sqrt(vx**2 + vy**2 + vz**2)
    theta_p         = np.arccos(np.divide(vz, r1, where=r1>0))
    phi_p           = np.arctan2(vy, vx)
    phi_p           = phi_p % (2 * np.pi)

    return [r1,theta_p,phi_p]

def spherical_to_cartesian(v_abs, v_theta, v_phi):
    return [v_abs * np.sin(v_theta) * np.cos(v_phi), v_abs * np.sin(v_theta) * np.sin(v_phi), v_abs * np.cos(v_theta)]

def mass_op(spec_sp: spec_spherical.SpectralExpansionSpherical, NUM_Q_VR, NUM_Q_VT, NUM_Q_VP, scale=1.0):
    """
    returns the operator that captures the mass of the distribution function. 
    mass = (maxwellian(0) * VTH**3) * np.dot(cf,M_klm) 
    we need to get this properly scaled for the appropiate maxwellian i.e., v_thermal
    """

    if NUM_Q_VR is None:
        if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY or spec_sp.get_radial_basis_type() == basis.BasisType.LAGUERRE:
            NUM_Q_VR     = min(MAX_GMX_Q_VR_PTS,params.BEVelocitySpace.NUM_Q_VR)
        elif spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:
            NUM_Q_VR     =  basis.BSpline.get_num_q_pts(spec_sp._p,spec_sp._basis_p._sp_order,spec_sp._basis_p._q_per_knot)
        
    if NUM_Q_VT is None:
        NUM_Q_VT     = params.BEVelocitySpace.NUM_Q_VT
    
    if NUM_Q_VP is None:
        NUM_Q_VP     = params.BEVelocitySpace.NUM_Q_VP
    
    num_p        = spec_sp._p +1
    sph_harm_lm  = params.BEVelocitySpace.SPH_HARM_LM 
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

    if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY or spec_sp.get_radial_basis_type() == basis.BasisType.LAGUERRE:
        [gmx,gmw]    = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR)
        #weight_func  = spec_sp._basis_p.Wx()

        quad_grid    = np.meshgrid(gmx,VTheta_q,VPhi_q,indexing='ij')
        l_modes      = list(set([l for l,_ in spec_sp._sph_harm_lm])) 
        
        Y_lm = spec_sp.Vq_sph(quad_grid[1],quad_grid[2])

        MP_klm = np.array([ spec_sp.Vq_r(quad_grid[0],l) * Y_lm[lm_idx] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm)])
        MP_klm = np.dot(MP_klm,WVPhi_q)
        MP_klm = np.dot(MP_klm,glw)
        MP_klm = np.dot(MP_klm,gmw)
        MP_klm = np.transpose(MP_klm)
        MP_klm = MP_klm.reshape(num_p*num_sph_harm)
        
    elif spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:

        [gmx,gmw]    = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR,True)
        weight_func  = spec_sp._basis_p.Wx()

        quad_grid = np.meshgrid(gmx,VTheta_q,VPhi_q,indexing='ij')
        l_modes      = list(set([l for l,_ in spec_sp._sph_harm_lm])) 
        
        Y_lm = spec_sp.Vq_sph(quad_grid[1],quad_grid[2])

        MP_klm = np.array([(quad_grid[0]**(2)) * spec_sp.Vq_r(quad_grid[0],l) * Y_lm[lm_idx] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm)])
        MP_klm = np.dot(MP_klm,WVPhi_q)
        MP_klm = np.dot(MP_klm,glw)
        MP_klm = np.dot(MP_klm,gmw)
        MP_klm = np.transpose(MP_klm)
        MP_klm = MP_klm.reshape(num_p*num_sph_harm)
    else:
        raise NotImplementedError
    
    return MP_klm

def mean_velocity_op(spec_sp: spec_spherical.SpectralExpansionSpherical, NUM_Q_VR, NUM_Q_VT, NUM_Q_VP, scale=1.0):
    
    if NUM_Q_VR is None:
        if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY or spec_sp.get_radial_basis_type() == basis.BasisType.LAGUERRE:
            NUM_Q_VR     = min(MAX_GMX_Q_VR_PTS,params.BEVelocitySpace.NUM_Q_VR)
        elif spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:
            NUM_Q_VR     =  basis.BSpline.get_num_q_pts(spec_sp._p,spec_sp._basis_p._sp_order,spec_sp._basis_p._q_per_knot)

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

    if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY or spec_sp.get_radial_basis_type() == basis.BasisType.LAGUERRE:
        [gmx,gmw]    = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR)
        #weight_func  = spec_sp._basis_p.Wx()

        quad_grid    = np.meshgrid(gmx,VTheta_q,VPhi_q,indexing='ij')
        l_modes      = list(set([l for l,_ in spec_sp._sph_harm_lm])) 
        
        Y_lm = spec_sp.Vq_sph(quad_grid[1],quad_grid[2])

        vx     = quad_grid[0] * np.sin(quad_grid[1]) * np.cos(quad_grid[2])
        vy     = quad_grid[0] * np.sin(quad_grid[1]) * np.sin(quad_grid[2])
        vz     = quad_grid[0] * np.cos(quad_grid[1]) 

        Vr_klm = np.array([ vx * spec_sp.basis_eval_radial(quad_grid[0],p,l) * Y_lm[lm_idx,:] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm) for p in range(num_p)])
        Vt_klm = np.array([ vy * spec_sp.basis_eval_radial(quad_grid[0],p,l) * Y_lm[lm_idx,:] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm) for p in range(num_p)])
        Vp_klm = np.array([ vz * spec_sp.basis_eval_radial(quad_grid[0],p,l) * Y_lm[lm_idx,:] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm) for p in range(num_p)])

        Vr_klm = np.dot(Vr_klm,WVPhi_q)
        Vr_klm = np.dot(Vr_klm,glw)
        Vr_klm = np.dot(Vr_klm,gmw)

        Vt_klm = np.dot(Vt_klm,WVPhi_q)
        Vt_klm = np.dot(Vt_klm,glw)
        Vt_klm = np.dot(Vt_klm,gmw)

        Vp_klm = np.dot(Vp_klm,WVPhi_q)
        Vp_klm = np.dot(Vp_klm,glw)
        Vp_klm = np.dot(Vp_klm,gmw)

        Vr_klm = Vr_klm.reshape(num_sph_harm, num_p)
        Vr_klm = np.transpose(Vr_klm)
        Vr_klm = Vr_klm.reshape(num_p*num_sph_harm)
        
        Vt_klm = Vt_klm.reshape(num_sph_harm, num_p)
        Vt_klm = np.transpose(Vt_klm)
        Vt_klm = Vt_klm.reshape(num_p*num_sph_harm)

        Vp_klm = Vp_klm.reshape(num_sph_harm, num_p)
        Vp_klm = np.transpose(Vp_klm)
        Vp_klm = Vp_klm.reshape(num_p*num_sph_harm)

        return [Vr_klm,Vt_klm, Vp_klm]

    elif spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:

        [gmx,gmw]    = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR,True)
        weight_func  = spec_sp._basis_p.Wx()

        quad_grid = np.meshgrid(gmx,VTheta_q,VPhi_q,indexing='ij')
        l_modes      = list(set([l for l,_ in spec_sp._sph_harm_lm])) 

        Y_lm = spec_sp.Vq_sph(quad_grid[1],quad_grid[2])

        vx     = quad_grid[0] * np.sin(quad_grid[1]) * np.cos(quad_grid[2])
        vy     = quad_grid[0] * np.sin(quad_grid[1]) * np.sin(quad_grid[2])
        vz     = quad_grid[0] * np.cos(quad_grid[1]) 

        mr    = (quad_grid[0]**(2))

        Vr_klm = np.array([ vx * mr * spec_sp.basis_eval_radial(quad_grid[0],p,l) * Y_lm[lm_idx,:] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm) for p in range(num_p)])
        Vt_klm = np.array([ vy * mr * spec_sp.basis_eval_radial(quad_grid[0],p,l) * Y_lm[lm_idx,:] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm) for p in range(num_p)])
        Vp_klm = np.array([ vz * mr * spec_sp.basis_eval_radial(quad_grid[0],p,l) * Y_lm[lm_idx,:] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm) for p in range(num_p)])

        Vr_klm = np.dot(Vr_klm,WVPhi_q)
        Vr_klm = np.dot(Vr_klm,glw)
        Vr_klm = np.dot(Vr_klm,gmw)

        Vt_klm = np.dot(Vt_klm,WVPhi_q)
        Vt_klm = np.dot(Vt_klm,glw)
        Vt_klm = np.dot(Vt_klm,gmw)

        Vp_klm = np.dot(Vp_klm,WVPhi_q)
        Vp_klm = np.dot(Vp_klm,glw)
        Vp_klm = np.dot(Vp_klm,gmw)

        Vr_klm = Vr_klm.reshape(num_sph_harm, num_p)
        Vr_klm = np.transpose(Vr_klm)
        Vr_klm = Vr_klm.reshape(num_p*num_sph_harm)
        
        Vt_klm = Vt_klm.reshape(num_sph_harm, num_p)
        Vt_klm = np.transpose(Vt_klm)
        Vt_klm = Vt_klm.reshape(num_p*num_sph_harm)

        Vp_klm = Vp_klm.reshape(num_sph_harm, num_p)
        Vp_klm = np.transpose(Vp_klm)
        Vp_klm = Vp_klm.reshape(num_p*num_sph_harm)
        
        return [Vr_klm,Vt_klm, Vp_klm]
    else:
        raise NotImplementedError

def temp_op(spec_sp: spec_spherical.SpectralExpansionSpherical, NUM_Q_VR, NUM_Q_VT, NUM_Q_VP, scale=1.0):
    if NUM_Q_VR is None:
        if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY or spec_sp.get_radial_basis_type() == basis.BasisType.LAGUERRE:
            NUM_Q_VR     = min(MAX_GMX_Q_VR_PTS,params.BEVelocitySpace.NUM_Q_VR)
        elif spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:
            NUM_Q_VR     =  basis.BSpline.get_num_q_pts(spec_sp._p,spec_sp._basis_p._sp_order,spec_sp._basis_p._q_per_knot)
        
    if NUM_Q_VT is None:
        NUM_Q_VT     = params.BEVelocitySpace.NUM_Q_VT
    
    if NUM_Q_VP is None:
        NUM_Q_VP     = params.BEVelocitySpace.NUM_Q_VP
    
    num_p        = spec_sp._p +1
    sph_harm_lm  = params.BEVelocitySpace.SPH_HARM_LM 
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

    if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY or spec_sp.get_radial_basis_type() == basis.BasisType.LAGUERRE:
        [gmx,gmw]    = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR)
        
        quad_grid    = np.meshgrid(gmx,VTheta_q,VPhi_q,indexing='ij')
        l_modes      = list(set([l for l,_ in spec_sp._sph_harm_lm])) 
        
        Y_lm = spec_sp.Vq_sph(quad_grid[1],quad_grid[2])
        

        MP_klm = np.array([(quad_grid[0]**(2)) * spec_sp.Vq_r(quad_grid[0],l) * Y_lm[lm_idx] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm)])
        MP_klm = np.dot(MP_klm,WVPhi_q)
        MP_klm = np.dot(MP_klm,glw)
        MP_klm = np.dot(MP_klm,gmw)
        MP_klm = np.transpose(MP_klm)
        MP_klm = MP_klm.reshape(num_p*num_sph_harm)
        return MP_klm

    elif spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:

        [gmx,gmw]    = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR,True)
        weight_func  = spec_sp._basis_p.Wx()

        quad_grid = np.meshgrid(gmx,VTheta_q,VPhi_q,indexing='ij')
        l_modes      = list(set([l for l,_ in spec_sp._sph_harm_lm])) 
       
        Y_lm = spec_sp.Vq_sph(quad_grid[1],quad_grid[2])

        MP_klm = np.array([ (quad_grid[0]**(2 + 2)) * spec_sp.Vq_r(quad_grid[0],l) * Y_lm[lm_idx] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm)])
        MP_klm = np.dot(MP_klm,WVPhi_q)
        MP_klm = np.dot(MP_klm,glw)
        MP_klm = np.dot(MP_klm,gmw)
        MP_klm = np.transpose(MP_klm)
        MP_klm = MP_klm.reshape(num_p*num_sph_harm)
        return MP_klm
    else:
        raise NotImplementedError

def moment_n_f(spec_sp: spec_spherical.SpectralExpansionSpherical,cf, maxwellian, V_TH, moment, NUM_Q_VR, NUM_Q_VT, NUM_Q_VP, scale=1.0, v0=np.zeros(3)):
    """
    Computes the zero th velocity moment, i.e. number density
    \int_v f(v) dv
    """
    if NUM_Q_VR is None:
        if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY or spec_sp.get_radial_basis_type() == basis.BasisType.LAGUERRE:
            NUM_Q_VR     = min(MAX_GMX_Q_VR_PTS,params.BEVelocitySpace.NUM_Q_VR)
        elif spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:
            NUM_Q_VR     =  basis.BSpline.get_num_q_pts(spec_sp._p,spec_sp._basis_p._sp_order,spec_sp._basis_p._q_per_knot)
        
    if NUM_Q_VT is None:
        NUM_Q_VT     = params.BEVelocitySpace.NUM_Q_VT
    
    if NUM_Q_VP is None:
        NUM_Q_VP     = params.BEVelocitySpace.NUM_Q_VP
    
    num_p        = spec_sp._p +1
    sph_harm_lm  = params.BEVelocitySpace.SPH_HARM_LM 
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

    if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY or spec_sp.get_radial_basis_type() == basis.BasisType.LAGUERRE:
        [gmx,gmw]    = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR)
        #weight_func  = spec_sp._basis_p.Wx()

        quad_grid    = np.meshgrid(gmx,VTheta_q,VPhi_q,indexing='ij')
        l_modes      = list(set([l for l,_ in spec_sp._sph_harm_lm])) 
        
        quad_grid_cart = spherical_to_cartesian(quad_grid[0],quad_grid[1],quad_grid[2])
        quad_grid_cart[0]+=v0[0]
        quad_grid_cart[1]+=v0[1]
        quad_grid_cart[2]+=v0[2]
        norm_v  = np.sqrt(quad_grid_cart[0]**2 + quad_grid_cart[1]**2 + quad_grid_cart[2]**2) 
        
        Y_lm = spec_sp.Vq_sph(quad_grid[1],quad_grid[2])

        maxwellian_fac = maxwellian(0)
        MP_klm = np.array([ (norm_v**(moment)) * spec_sp.Vq_r(quad_grid[0],l) * Y_lm[lm_idx] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm)])
        MP_klm = np.dot(MP_klm,WVPhi_q)
        MP_klm = np.dot(MP_klm,glw)
        MP_klm = np.dot(MP_klm,gmw)
        MP_klm = np.transpose(MP_klm)
        MP_klm = MP_klm.reshape(num_p*num_sph_harm)
        m_k    = (maxwellian_fac * (V_TH**(3+moment))) * scale * np.dot(cf, MP_klm)
        return m_k

    elif spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:

        [gmx,gmw]    = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR,True)
        weight_func  = spec_sp._basis_p.Wx()

        quad_grid = np.meshgrid(gmx,VTheta_q,VPhi_q,indexing='ij')
        l_modes      = list(set([l for l,_ in spec_sp._sph_harm_lm])) 
        
        Y_lm = spec_sp.Vq_sph(quad_grid[1],quad_grid[2])

        quad_grid_cart = spherical_to_cartesian(quad_grid[0],quad_grid[1],quad_grid[2])
        quad_grid_cart[0]+=v0[0]
        quad_grid_cart[1]+=v0[1]
        quad_grid_cart[2]+=v0[2]
        norm_v  = np.sqrt(quad_grid_cart[0]**2 + quad_grid_cart[1]**2 + quad_grid_cart[2]**2) 

        maxwellian_fac = maxwellian(0)
        MP_klm = np.array([ (norm_v**moment) * (quad_grid[0]**(2)) * spec_sp.Vq_r(quad_grid[0],l) * Y_lm[lm_idx] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm)])
        MP_klm = np.dot(MP_klm,WVPhi_q)
        MP_klm = np.dot(MP_klm,glw)
        MP_klm = np.dot(MP_klm,gmw)
        MP_klm = np.transpose(MP_klm)
        MP_klm = MP_klm.reshape(num_p*num_sph_harm)
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

def function_to_basis(spec_sp: spec_spherical.SpectralExpansionSpherical, hv, maxwellian, NUM_Q_VR, NUM_Q_VT, NUM_Q_VP):
    """
    Note the function is assumed to be f(v) = M(v) h(v) and computes the projection coefficients
    should be compatible with the weight function of the polynomials. 
    """
    if NUM_Q_VR is None:
        if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY or spec_sp.get_radial_basis_type() == basis.BasisType.LAGUERRE:
            NUM_Q_VR     = min(MAX_GMX_Q_VR_PTS,params.BEVelocitySpace.NUM_Q_VR)
        elif spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:
            NUM_Q_VR     =  basis.BSpline.get_num_q_pts(spec_sp._p,spec_sp._basis_p._sp_order,spec_sp._basis_p._q_per_knot)
    
    if NUM_Q_VT is None:
        NUM_Q_VT     = params.BEVelocitySpace.NUM_Q_VT
    
    if NUM_Q_VP is None:
        NUM_Q_VP     = params.BEVelocitySpace.NUM_Q_VP
            
    num_p        = spec_sp._p +1
    sph_harm_lm  = params.BEVelocitySpace.SPH_HARM_LM 
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

    if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY or spec_sp.get_radial_basis_type() == basis.BasisType.LAGUERRE:
        """
        assumes orthonormal mass matrix. 
        """
        [gmx,gmw]    = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR)
        weight_func  = spec_sp._basis_p.Wx()
        quad_grid    = np.meshgrid(gmx,VTheta_q,VPhi_q,indexing='ij')

        l_modes      = list(set([l for l,_ in spec_sp._sph_harm_lm])) 
        
        Y_lm = spec_sp.Vq_sph(quad_grid[1],quad_grid[2])
        hq   = hv(quad_grid[0],quad_grid[1],quad_grid[2]) 

        M_klm = np.array([ hq * spec_sp.Vq_r(quad_grid[0],l) * Y_lm[lm_idx] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm)])
        M_klm = np.dot(M_klm,WVPhi_q)
        M_klm = np.dot(M_klm,glw)
        M_klm = np.dot(M_klm,gmw)
        M_klm = np.transpose(M_klm)
        M_klm = M_klm.reshape(num_p*num_sph_harm)
        
        return M_klm

    elif spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:

        [gmx,gmw]    = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR,False)
        weight_func  = spec_sp._basis_p.Wx()

        quad_grid = np.meshgrid(gmx,VTheta_q,VPhi_q,indexing='ij')

        l_modes      = list(set([l for l,_ in spec_sp._sph_harm_lm])) 
        
        Y_lm = spec_sp.Vq_sph(quad_grid[1],quad_grid[2])
        hq   = hv(quad_grid[0],quad_grid[1],quad_grid[2]) * (quad_grid[0]**2)
        MM   = spec_sp.compute_mass_matrix()
        MMinv= choloskey_inv(MM)
        
        M_klm  = np.array([ hq * spec_sp.Vq_r(quad_grid[0],l) * Y_lm[lm_idx] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm)])
        M_klm  = np.dot(M_klm,WVPhi_q)
        M_klm  = np.dot(M_klm,glw)
        M_klm  = np.dot(M_klm,gmw)
        M_klm  = np.transpose(M_klm)
        M_klm  = M_klm.reshape(num_p*num_sph_harm)
        #M_klm = np.matmul(np.linalg.pinv(MM),M_klm)
        M_klm = np.matmul(MMinv,M_klm)
        
        # import matplotlib.pyplot as plt
        # cmat        = MP_klm.reshape(num_p,num_sph_harm)
        # fv    = np.array([cmat[i,j] * P_kr[i]*Y_lm[j] for i in range(num_p) for j in range(num_sph_harm)])
        # fv    = fv[:,:,0,0]
        # fv    = np.sum(fv,axis=0)
        
        # Fv    = np.exp(-gmx**2)

        # plt.plot(gmx,fv-Fv)
        # plt.yscale("log")

        # #plt.plot(gmx,fv)
        # #plt.yscale("log")

        # plt.grid()
        # plt.show()
        
        # print(MP_klm)
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
        if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY or spec_sp.get_radial_basis_type() == basis.BasisType.LAGUERRE:
            NUM_Q_VR     = min(MAX_GMX_Q_VR_PTS,params.BEVelocitySpace.NUM_Q_VR)
        elif spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:
            NUM_Q_VR     =  basis.BSpline.get_num_q_pts(spec_sp._p,spec_sp._basis_p._sp_order,spec_sp._basis_p._q_per_knot)

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

    
    if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY:
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
    
    if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY or spec_sp.get_radial_basis_type() == basis.BasisType.LAGUERRE:
        NUM_Q_VR     = min(MAX_GMX_Q_VR_PTS,params.BEVelocitySpace.NUM_Q_VR)
    elif spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:
        NUM_Q_VR     =  basis.BSpline.get_num_q_pts(spec_sp._p,spec_sp._basis_p._sp_order,spec_sp._basis_p._q_per_knot)
    
    NUM_Q_VT     = params.BEVelocitySpace.NUM_Q_VT
    NUM_Q_VP     = params.BEVelocitySpace.NUM_Q_VP
    
    num_p        = spec_sp._p +1
    sph_harm_lm  = params.BEVelocitySpace.SPH_HARM_LM 
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

    if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY or spec_sp.get_radial_basis_type() == basis.BasisType.LAGUERRE:

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

def reaction_rate(spec_sp : spec_spherical.SpectralExpansionSpherical, g, cf, maxwellian, vth, scale=1):
    """
    Compute the reaction rates for specified collision data, 
    """    

    if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY or spec_sp.get_radial_basis_type() == basis.BasisType.LAGUERRE:
        NUM_Q_VR     = min(MAX_GMX_Q_VR_PTS,params.BEVelocitySpace.NUM_Q_VR)
    elif spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:
        NUM_Q_VR     =  basis.BSpline.get_num_q_pts(spec_sp._p,spec_sp._basis_p._sp_order,spec_sp._basis_p._q_per_knot)

    NUM_Q_VT     = params.BEVelocitySpace.NUM_Q_VT
    NUM_Q_VP     = params.BEVelocitySpace.NUM_Q_VP
    
    num_p        = spec_sp._p +1
    sph_harm_lm  = params.BEVelocitySpace.SPH_HARM_LM 
    num_sph_harm = len(sph_harm_lm)

    gmx,gmw      = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR)
    
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

    quad_grid = np.meshgrid(gmx,VTheta_q,VPhi_q,indexing='ij')
    P_kr = spec_sp.Vq_r(quad_grid[0]) 
    Y_lm = spec_sp.Vq_sph(quad_grid[1],quad_grid[2])

    ev_qx    =  (0.5 * scipy.constants.electron_mass * (quad_grid[0]*vth)**2 )/scipy.constants.electron_volt
    cs_total =  g.total_cross_section(ev_qx)

    gama_electron    = 1 #np.sqrt(2*collisions.ELECTRON_CHARGE/collisions.MASS_ELECTRON)
    mass_m0          = moment_n_f(spec_sp,cf,maxwellian,vth,0,NUM_Q_VR=NUM_Q_VR,NUM_Q_VT=None,NUM_Q_VP=None,scale=1)
    
    if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY:
        MP_klm = np.array([ cs_total * P_kr[i] * Y_lm[j] for i in range(num_p) for j in range(num_sph_harm)])
        MP_klm = np.dot(MP_klm,WVPhi_q)
        MP_klm = np.dot(MP_klm,glw)
        MP_klm = np.dot(MP_klm,gmw)

        reaction_rate = (np.dot(np.transpose(MP_klm),cf)/mass_m0) * gama_electron * scale * (1/(4*np.pi))
        return reaction_rate

    elif spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:
        MP_klm = np.array([((quad_grid[0]**2) * (vth**3)) * cs_total * P_kr[i] * Y_lm[j] for i in range(num_p) for j in range(num_sph_harm)])
        MP_klm = np.dot(MP_klm,WVPhi_q)
        MP_klm = np.dot(MP_klm,glw)
        MP_klm = np.dot(MP_klm,gmw)
        
        reaction_rate = (np.dot(np.transpose(MP_klm),cf)/mass_m0) * gama_electron * scale
        return reaction_rate

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
    
def compute_radial_components(ev_pts, spec_sp : spec_spherical.SpectralExpansionSpherical, cf, maxwellian, vth,scale=1, v0=np.zeros(3)):
    """
    Assumes spherical harmonic basis in vtheta vphi direction. 
    the integration over the spherical harmonics is done analytically. 
    """
    EV       = scipy.constants.electron_volt
    E_MASS   = scipy.constants.electron_mass

    # v0_abs   = np.sqrt(v0[0]**2 + v0[1]**2 + v0[2]**2)
    vr       = np.sqrt(2* EV * ev_pts /E_MASS)/vth
    
    # TODO: check this
    # vr[vr<v0_abs]  = 0
    # vr[vr>=v0_abs] = vr[vr>=v0_abs] -v0_abs

    sph_harm_lm  = params.BEVelocitySpace.SPH_HARM_LM 
    num_sph_harm = len(sph_harm_lm)
    num_p        = spec_sp._p + 1  

    output = np.zeros((num_sph_harm, len(vr)))

    for l_idx, lm in enumerate(params.BEVelocitySpace.SPH_HARM_LM):
        if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY:
            output[l_idx, :] = basis.maxpoly.maxpolyserieseval(2*lm[0]+2, vr, cf[l_idx::num_sph_harm])*np.exp(-vr**2)*(vr**lm[0])
        else:
            # radial_component = 0
            # for i, coeff in enumerate(cf[l_idx::num_sph_harm]):
            #     radial_component = radial_component + coeff*spec_sp.basis_eval_radial(vr, i, lm[0]) * sph_factor
            # output[l_idx, : ] = radial_component
                        
            sph_factor = spec_sp.basis_eval_spherical(0,0,lm[0],lm[1])
            #print("l,m=(%d,%d)=%.8E" %(lm[0], lm[1], sph_factor))
            Vqr=spec_sp.Vq_r(vr,lm[0],1)
            output[l_idx, :] = sph_factor * np.dot( cf[l_idx::num_sph_harm], Vqr)
    return output




