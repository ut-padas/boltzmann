"""
@package : Utility functions needed for the Boltzmann solver. 
"""
import numpy as np
import parameters as params
import spec_spherical
import basis
import scipy.constants

MAX_GMX_Q_VR_PTS=278

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
    v_abs = np.sqrt(vx**2 + vy**2 + vz**2)
    check_1  = np.isclose(v_abs,0.0)
    check_2  = np.isclose(vx,0.0)

    not_check_1 = np.logical_not(check_1)
    not_check_2 = np.logical_not(check_2)
    
    not_check_1_and_check_2 = np.logical_and(not_check_1,not_check_2)
    
    vr = np.zeros_like(v_abs)
    vt = np.zeros_like(v_abs)
    vp = np.zeros_like(v_abs)

    vr = v_abs
    vt[not_check_1_and_check_2] =  np.arccos(vz[not_check_1_and_check_2]/v_abs[not_check_1_and_check_2])
    vp[not_check_1_and_check_2] =  np.arctan(vy[not_check_1_and_check_2]/vx[not_check_1_and_check_2])

    # to fix the range to 0-2pi
    vp[vp < 0]+= (np.pi * 2)
    vp[check_2]                 =  np.pi/2
    
    return [vr,vt,vp]

def spherical_to_cartesian(v_abs, v_theta, v_phi):
    return [v_abs * np.sin(v_theta) * np.cos(v_phi), v_abs * np.sin(v_theta) * np.sin(v_phi), v_abs * np.cos(v_theta)]

def moment_n_f(spec_sp: spec_spherical.SpectralExpansionSpherical,cf, maxwellian, V_TH, moment, NUM_Q_VR, NUM_Q_VT, NUM_Q_VP, scale=1.0):
    """
    Computes the zero th velocity moment, i.e. number density
    \int_v f(v) dv
    """
    if NUM_Q_VR is None:
        if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY:
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

    if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY:
        [gmx,gmw]    = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR)
        #weight_func  = spec_sp._basis_p.Wx()

        quad_grid    = np.meshgrid(gmx,VTheta_q,VPhi_q,indexing='ij')
        l_modes      = list(set([l for l,_ in spec_sp._sph_harm_lm])) 
        Vq_radial_l  = list()
        
        for l in l_modes:
            Vq_radial_l.append(spec_sp.Vq_r(quad_grid[0],l))

        Y_lm = spec_sp.Vq_sph(quad_grid[1],quad_grid[2])

        maxwellian_fac = maxwellian(0)
        MP_klm = np.array([ (quad_grid[0]**(moment + l)) * Vq_radial_l[l_modes.index(l)] * Y_lm[lm_idx] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm)])
        MP_klm = np.dot(MP_klm,WVPhi_q)
        MP_klm = np.dot(MP_klm,glw)
        MP_klm = np.dot(MP_klm,gmw)
        MP_klm = np.transpose(MP_klm)
        MP_klm = MP_klm.reshape(num_p*num_sph_harm)
        m_k    = (maxwellian_fac * (V_TH**(3+moment))) * scale * np.dot(MP_klm,cf)
        return m_k

    elif spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:

        [gmx,gmw]    = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR,True)
        weight_func  = spec_sp._basis_p.Wx()

        quad_grid = np.meshgrid(gmx,VTheta_q,VPhi_q,indexing='ij')
        l_modes      = list(set([l for l,_ in spec_sp._sph_harm_lm])) 
        Vq_radial_l  = list()
        
        for l in l_modes:
            Vq_radial_l.append(spec_sp.Vq_r(quad_grid[0],l))

        Y_lm = spec_sp.Vq_sph(quad_grid[1],quad_grid[2])

        maxwellian_fac = maxwellian(0)
        MP_klm = np.array([ (quad_grid[0]**(moment + 2)) * Vq_radial_l[l_modes.index(l)] * Y_lm[lm_idx] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm)])
        MP_klm = np.dot(MP_klm,WVPhi_q)
        MP_klm = np.dot(MP_klm,glw)
        MP_klm = np.dot(MP_klm,gmw)
        MP_klm = np.transpose(MP_klm)
        MP_klm = MP_klm.reshape(num_p*num_sph_harm)
        m_k    = (maxwellian_fac * (V_TH**(3+moment))) * scale * np.dot(MP_klm,cf)
        return m_k

def compute_avg_temp(particle_mass,spec_sp: spec_spherical.SpectralExpansionSpherical,cf, maxwellian, V_TH, NUM_Q_VR, NUM_Q_VT, NUM_Q_VP, m0=None, scale=1.0):
    if m0 is None:
        m0         = moment_n_f(spec_sp,cf,maxwellian,V_TH,0,NUM_Q_VR,NUM_Q_VT,NUM_Q_VP,scale)

    m2         = moment_n_f(spec_sp,cf,maxwellian,V_TH,2,NUM_Q_VR,NUM_Q_VT,NUM_Q_VP,scale)
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
        if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY:
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

    if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY:
        """
        assumes orthonormal mass matrix. 
        """
        [gmx,gmw]    = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR)
        weight_func  = spec_sp._basis_p.Wx()
        quad_grid    = np.meshgrid(gmx,VTheta_q,VPhi_q,indexing='ij')

        l_modes      = list(set([l for l,_ in spec_sp._sph_harm_lm])) 
        Vq_radial_l  = list()
        
        for l in l_modes:
            Vq_radial_l.append(spec_sp.Vq_r(quad_grid[0],l)) 
        
        Y_lm = spec_sp.Vq_sph(quad_grid[1],quad_grid[2])
        hq   = hv(quad_grid[0],quad_grid[1],quad_grid[2]) 

        M_klm = np.array([ hq * quad_grid[0]**(l) * Vq_radial_l[l_modes.index(l)] * Y_lm[lm_idx] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm)])
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
        Vq_radial_l  = list()
        
        for l in l_modes:
            Vq_radial_l.append(spec_sp.Vq_r(quad_grid[0],l)) 
        
        Y_lm = spec_sp.Vq_sph(quad_grid[1],quad_grid[2])
        hq   = hv(quad_grid[0],quad_grid[1],quad_grid[2]) * np.exp(-quad_grid[0]**2) * (quad_grid[0]**2)
        MM   = spec_sp.compute_mass_matrix()
        
        M_klm  = np.array([ hq * Vq_radial_l[l_modes.index(l)] * Y_lm[lm_idx] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm)])
        M_klm  = np.dot(M_klm,WVPhi_q)
        M_klm  = np.dot(M_klm,glw)
        M_klm  = np.dot(M_klm,gmw)
        M_klm  = np.transpose(M_klm)
        M_klm  = M_klm.reshape(num_p*num_sph_harm)
        M_klm = np.matmul(np.linalg.pinv(MM),M_klm)
        
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

def compute_Mvth1_Pi_vth2_Pj_vth1(spec_sp: spec_spherical.SpectralExpansionSpherical,mw_1,vth_1,mw_2,vth_2, NUM_Q_VR, NUM_Q_VT, NUM_Q_VP, scale=1.0):
    """
    v_a = v/v_th 
    computes the \int_{R^3} M(v_a) f(v_a) Pi(vth_1) Pj(vth_2) dv
    """
    if NUM_Q_VR is None:
        NUM_Q_VR     = min(MAX_GMX_Q_VR_PTS,params.BEVelocitySpace.NUM_Q_VR)
    
    if NUM_Q_VT is None:
        NUM_Q_VT     = params.BEVelocitySpace.NUM_Q_VT
    
    if NUM_Q_VP is None:
        NUM_Q_VP     = params.BEVelocitySpace.NUM_Q_VP
            
    num_p        = spec_sp._p +1
    sph_harm_lm  = params.BEVelocitySpace.SPH_HARM_LM 
    num_sph_harm = len(sph_harm_lm)
    [gmx,gmw]    = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR)
    weight_func  = spec_sp._basis_p.Wx()
    
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
    Q_kr = spec_sp.Vq_r(quad_grid[0]*(vth_1/vth_2)) 
    Y_lm = spec_sp.Vq_sph(quad_grid[1],quad_grid[2])
    
    M_klm_pqs = np.array([1 * Q_kr[i]*Y_lm[j] * P_kr[p]*Y_lm[q] for i in range(num_p) for j in range(num_sph_harm) for p in range(num_p) for q in range(num_sph_harm)])

    M_klm_pqs  = np.dot(M_klm_pqs,WVPhi_q)
    M_klm_pqs  = np.dot(M_klm_pqs,glw)
    M_klm_pqs  = np.dot(M_klm_pqs,gmw)
    M_klm_pqs  = M_klm_pqs.reshape(num_p*num_sph_harm, num_p*num_sph_harm)
    
    return M_klm_pqs

def get_maxwellian_3d(vth,n_scale=1):
    M = lambda x: (n_scale / ((vth * np.sqrt(np.pi))**3) ) * np.exp(-x**2)
    return M

def get_eedf(ev_pts, spec_sp : spec_spherical.SpectralExpansionSpherical, cf, maxwellian, vth,scale=1):
    """
    Assumes spherical harmonic basis in vtheta vphi direction. 
    the integration over the spherical harmonics is done analytically. 
    """
    EV       = scipy.constants.electron_volt
    E_MASS   = scipy.constants.electron_mass
    vr       = np.sqrt(2* EV * ev_pts /E_MASS)/vth
    

    if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY:
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

    if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY:

        l_modes      = list(set([l for l,_ in spec_sp._sph_harm_lm])) 
        Vq_radial_l  = list()
        
        for l in l_modes:
            Vq_radial_l.append(spec_sp.Vq_r(quad_grid[0],l)) 
        
        Y_lm = spec_sp.Vq_sph(quad_grid[1],quad_grid[2])
        
        MP_klm = np.array([ np.exp(-quad_grid[0]**2) * quad_grid[0]**(l) * Vq_radial_l[l_modes.index(l)] * Y_lm[lm_idx] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm)])
        MP_klm = np.dot(MP_klm,WVPhi_q)
        MP_klm = np.dot(MP_klm,glw)
        MP_klm = np.swapaxes(MP_klm,0,1)
        MP_klm = MP_klm.reshape((num_p*num_sph_harm,-1))
        
        return np.dot(np.transpose(MP_klm),cf) 

    elif spec_sp.get_radial_basis_type() == basis.BasisType.SPLINES:

        l_modes      = list(set([l for l,_ in spec_sp._sph_harm_lm])) 
        Vq_radial_l  = list()
        
        for l in l_modes:
            Vq_radial_l.append(spec_sp.Vq_r(quad_grid[0],l)) 
        
        Y_lm = spec_sp.Vq_sph(quad_grid[1],quad_grid[2])
        
        MP_klm = np.array([Vq_radial_l[l_modes.index(l)] * Y_lm[lm_idx] for lm_idx, (l,m) in enumerate(spec_sp._sph_harm_lm)])
        MP_klm = np.dot(MP_klm,WVPhi_q)
        MP_klm = np.dot(MP_klm,glw)
        MP_klm = np.swapaxes(MP_klm,0,1)
        MP_klm = MP_klm.reshape((num_p*num_sph_harm,-1))
        return np.dot(np.transpose(MP_klm),cf) 

def reaction_rate(spec_sp : spec_spherical.SpectralExpansionSpherical, g, cf, maxwellian, vth, scale=1):
    """
    Compute the reaction rates for specified collision data, 
    """    

    if spec_sp.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY:
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




    






