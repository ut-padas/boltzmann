"""
@package : Utility functions needed for the Boltzmann solver. 
"""
import numpy as np
import collisions
import parameters as params

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

    