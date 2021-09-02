"""
@package : Utility functions needed for the Boltzmann solver. 
"""
import numpy as np
import collisions
import parameters as params
import spec_spherical
import basis

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
    if np.allclose(v_abs,0.0):
        return [0.0,0.0,0.0]
    
    if np.allclose(vx,0.0):
        return [v_abs, np.arccos(vz/v_abs), np.pi/2]
        
    return [v_abs, np.arccos(vz/v_abs), np.arctan(vy/vx)]

def spherical_to_cartesian(v_abs, v_theta, v_phi):
    return [v_abs * np.sin(v_theta) * np.cos(v_phi), v_abs * np.sin(v_theta) * np.sin(v_phi), v_abs * np.cos(v_theta)]

def is_collission_mat_converged(spec,cf,collision,maxwellian,tol):
    
    """
    check if the assembled collision operator
    converged with increasing quadrature points. 
    """
    print("-- converge on the spherical integration")
    num_qv_s = 1 #params.BEVelocitySpace.NUM_Q_PTS_ON_SPHERE
    num_qv   = 1 #params.BEVelocitySpace.NUM_Q_PTS_ON_V

    V_TH  = collisions.ELECTRON_THEMAL_VEL
    M     = spec.compute_maxwellian_mm(maxwellian, V_TH)
    invM  = np.linalg.inv(M)

    L     = params.BEVelocitySpace.VELOCITY_SPACE_DT * np.matmul( invM, cf.assemble_mat(collision, maxwellian, num_qv , num_qv_s) ) 
    Lp    = params.BEVelocitySpace.VELOCITY_SPACE_DT * np.matmul( invM, cf.assemble_mat(collision, maxwellian, num_qv+1 , num_qv_s+1) ) #cf.assemble_mat(collision, maxwellian, num_qv , num_qv_s+1) 
    error_func = lambda L,L1 : np.max( np.abs( (L1-L) ) )

    if ( error_func(Lp,L) < tol ):
        print("collision mat converged with %d number of quadrature points on the sphere" %(num_qv_s))
    else:
        num_qv += 1
        num_qv_s+= 1
        
    while(error_func(Lp,L)  > tol):
        L        = Lp
        Lp       = params.BEVelocitySpace.VELOCITY_SPACE_DT * np.matmul(invM, cf.assemble_mat(collision, maxwellian, num_qv+1 , num_qv_s+1) ) 
        print(" |Lp-L| : %.16f  with num q points on sphere %d " %(error_func(Lp,L), num_qv_s) )
        print("number of # of quadrature point in r %d, # quadrature points for angular %d \n" %(num_qv,num_qv_s))
        print(L)
        print("\n")
        print(Lp)
        num_qv += 1
        num_qv_s+= 1

    return 

def moment_zero_f(spec_sp: spec_spherical.SpectralExpansionSpherical,cf, maxwellian, V_TH, NUM_Q_VR, NUM_Q_VT, NUM_Q_VP, scale=1.0):
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
    sph_harm_lm  = params.BEVelocitySpace.SPH_HARM_LM 
    num_sph_harm = len(sph_harm_lm)
    [gmx,gmw]    = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR)
    weight_func  = spec_sp._basis_p.Wx()
    
    legendre     = basis.Legendre()
    [glx,glw]    = legendre.Gauss_Pn(NUM_Q_VT)
    VTheta_q     = np.arccos(glx)
    VPhi_q       = np.linspace(0,2*np.pi,NUM_Q_VP)

    # [glx_s,glw_s] = legendre.Gauss_Pn(NUM_Q_CHI)
    # Chi_q         = np.arccos(glx_s)
    # Phi_q         = np.linspace(0,2*np.pi,NUM_Q_PHI)
    
    assert NUM_Q_VP>1
    sq_fac_v = (2*np.pi/(NUM_Q_VP-1))
    WVPhi_q  = np.ones(NUM_Q_VP)*sq_fac_v

    #trap. weights
    WVPhi_q[0]  = 0.5 * WVPhi_q[0]
    WVPhi_q[-1] = 0.5 * WVPhi_q[-1]

    quad_grid = np.meshgrid(gmx,VTheta_q,VPhi_q,indexing='ij')
    P_kr = spec_sp.Vq_r(quad_grid[0]) 
    Y_lm = spec_sp.Vq_sph(quad_grid[1],quad_grid[2])

    #np.sum(glw)
    #print(P_kr.shape)
    maxwellian_fac = maxwellian(0)
    MP_klm = np.array([P_kr[i]*Y_lm[j] for i in range(num_p) for j in range(num_sph_harm)])
    #print(MP_klm.shape)
    #print(np.allclose(np.sum(WVPhi_q),2*np.pi))
    MP_klm = np.dot(MP_klm,WVPhi_q)
    MP_klm = np.dot(MP_klm,glw)
    #print(MP_klm)
    MP_klm = np.dot(MP_klm,gmw)
    #print(MP_klm)
    #print(cf)
    m0     = (np.sqrt(np.pi)/4) * (maxwellian_fac * (V_TH**3)) * scale * np.dot(MP_klm,cf)
    return m0

def moment_second_f(spec_sp: spec_spherical.SpectralExpansionSpherical,cf, maxwellian, V_TH, NUM_Q_VR, NUM_Q_VT, NUM_Q_VP, scale=1.0):
    """
    Computes the second velocity moment, i.e. related to average kinetic energy, hence the temperature
    \int_v v^2 f(v) dv
    """
    if NUM_Q_VR is None:
        NUM_Q_VR     = params.BEVelocitySpace.NUM_Q_VR
    
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

    # [glx_s,glw_s] = legendre.Gauss_Pn(NUM_Q_CHI)
    # Chi_q         = np.arccos(glx_s)
    # Phi_q         = np.linspace(0,2*np.pi,NUM_Q_PHI)
    
    assert NUM_Q_VP>1
    sq_fac_v = (2*np.pi/(NUM_Q_VP-1))
    WVPhi_q  = np.ones(NUM_Q_VP)*sq_fac_v

    #trap. weights
    WVPhi_q[0]  = 0.5 * WVPhi_q[0]
    WVPhi_q[-1] = 0.5 * WVPhi_q[-1]

    quad_grid = np.meshgrid(gmx,VTheta_q,VPhi_q,indexing='ij')
    P_kr = spec_sp.Vq_r(quad_grid[0]) 
    Y_lm = spec_sp.Vq_sph(quad_grid[1],quad_grid[2])

    maxwellian_fac = maxwellian(0)
    MP_klm = np.array([(quad_grid[0]**2) * P_kr[i] * Y_lm[j] for i in range(num_p) for j in range(num_sph_harm)])
    #print(MP_klm.shape)
    #print(np.allclose(np.sum(WVPhi_q),2*np.pi))
    MP_klm = np.dot(MP_klm,WVPhi_q)
    MP_klm = np.dot(MP_klm,glw)
    #print(MP_klm)
    MP_klm = np.dot(MP_klm,gmw)
    #print(MP_klm)
    #print(cf)
    m2     = (np.sqrt(np.pi)/4) * (maxwellian_fac * (V_TH**5)) * scale * np.dot(MP_klm,cf)
    return m2


def compute_avg_temp(particle_mass,spec_sp: spec_spherical.SpectralExpansionSpherical,cf, maxwellian, V_TH, NUM_Q_VR, NUM_Q_VT, NUM_Q_VP, m0=None, scale=1.0):
    if m0 is None:
        m0         = moment_zero_f(spec_sp,cf,maxwellian,V_TH,NUM_Q_VR,NUM_Q_VT,NUM_Q_VP,scale)

    m2         = moment_second_f(spec_sp,cf,maxwellian,V_TH,NUM_Q_VR,NUM_Q_VT,NUM_Q_VP,scale)
    #avg energy per particle
    avg_energy = (0.5*particle_mass*m2)/m0
    temp       = (2/(3*collisions.BOLTZMANN_CONST)) * avg_energy
    return temp

def compute_coefficients(spec_sp: spec_spherical.SpectralExpansionSpherical, hv, maxwellian, NUM_Q_VR, NUM_Q_VT, NUM_Q_VP):
    """
    Note the function is assumed to be f(v) = M(v) h(v), M(v)
    should be compatible with the weight function of the polynomials. 
    """
    V_TH         = collisions.ELECTRON_THEMAL_VEL

    if NUM_Q_VR is None:
        NUM_Q_VR     = params.BEVelocitySpace.NUM_Q_VR
    
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

    # [glx_s,glw_s] = legendre.Gauss_Pn(NUM_Q_CHI)
    # Chi_q         = np.arccos(glx_s)
    # Phi_q         = np.linspace(0,2*np.pi,NUM_Q_PHI)
    
    assert NUM_Q_VP>1
    sq_fac_v = (2*np.pi/(NUM_Q_VP-1))
    WVPhi_q  = np.ones(NUM_Q_VP)*sq_fac_v

    #trap. weights
    WVPhi_q[0]  = 0.5 * WVPhi_q[0]
    WVPhi_q[-1] = 0.5 * WVPhi_q[-1]

    quad_grid = np.meshgrid(gmx,VTheta_q,VPhi_q,indexing='ij')
    P_kr = spec_sp.Vq_r(quad_grid[0]) 
    Y_lm = spec_sp.Vq_sph(quad_grid[1],quad_grid[2])

    hq   = hv(quad_grid[0],quad_grid[1],quad_grid[2]) 
    integral_fac   = 1

    MP_klm = np.array([hq * P_kr[i]*Y_lm[j] for i in range(num_p) for j in range(num_sph_harm)])
    MP_klm = np.dot(MP_klm,WVPhi_q)
    MP_klm = np.dot(MP_klm,glw)
    MP_klm = np.dot(MP_klm,gmw)

    return MP_klm

    