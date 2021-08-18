"""
@package : Spectral based, Petrov-Galerkin discretization of the collission integral
"""

import abc
import basis
import spec_spherical as sp
import collisions
import scipy.constants
import numpy as np
import maxpoly
import parameters as params
import utils


class CollissionOp(abc.ABC):

    def __init__(self,dim,p_order):
        self._dim = dim
        self._p = p_order
        pass

    @abc.abstractmethod
    def assemble_mat(collision,maxwellian):
        pass

#SPEC_HERMITE_E = sp.SpectralExpansion(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VEL_SPACE_POLY_ORDER, basis.BasisType.HERMITE_E_POLY)
SPEC_SPHERICAL  = sp.SpectralExpansionSpherical(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,basis.Maxwell(),params.BEVelocitySpace.SPH_HARM_LM) 

class CollisionOpSP():
    """
    3D- velocity space, collision operator for species s-collisions
    with background heavy particles, where heavy particles obey the
    direc delta distribution function. 
    """
    def __init__(self,dim,p_order) -> None:
        self._dim  = dim
        self._p    = p_order
        self._spec = SPEC_SPHERICAL

    @staticmethod
    def assemble_mat(collision : collisions.Collisions , maxwellian):
        """
        Compute the spectral PG discretization for specified Collision. 
        """
        spec  = SPEC_SPHERICAL
        num_p = spec._p +1
        num_q_pts_on_v      = params.BEVelocitySpace.NUM_Q_PTS_ON_V
        num_q_pts_on_sphere = params.BEVelocitySpace.NUM_Q_PTS_ON_SPHERE
        sph_harm_lm         = params.BEVelocitySpace.SPH_HARM_LM 

        num_sph_harm = len(sph_harm_lm)
        [gmx,gmw]    = maxpoly.maxpolygauss(num_q_pts_on_v)
        weight_func  = maxpoly.maxpolyweight
        
        legendre = basis.Legendre()
        [glx,glw] = legendre.Gauss_Pn(num_q_pts_on_sphere)
        theta_q = np.arccos(glx)
        phi_q = np.linspace(0,2*np.pi,2*(num_q_pts_on_sphere))

        L_ij =spec.create_mat()
        spherical_quadrature_fac = (np.pi/num_q_pts_on_sphere)
        qr = np.zeros([num_p, num_sph_harm, num_sph_harm, len(gmx)])

        # pi = p
        # pj = k
        # lm1 = lm
        # lm2 = qs

        V_TH         = collisions.ELECTRON_THEMAL_VEL
        ELE_VOLT     = collisions.ELECTRON_VOLT
        MAXWELLIAN_N = collisions.MAXWELLIAN_N
        AR_NEUTRAL_N = collisions.AR_NEUTRAL_N

        for qi,v_abs in enumerate(gmx): # loop over quadrature points radial
            # \int_S^2 from R^3 integral
            energy_in_ev = (0.5*collisions.MASS_ELECTRON * (v_abs * V_TH)**2) / ELE_VOLT   
            if (energy_in_ev < collision.min_energy_threshold()):
                print("skipping energy : ", energy_in_ev," for v: ",v_abs," Eth : ", collision.min_energy_threshold()," col type: ",collision._type)
                continue
            total_cs     = collision.total_cross_section(energy_in_ev)
            #print(energy_in_ev, " -> ", total_cs)
            for v_theta_i, v_theta in enumerate(theta_q):
                for v_phi in phi_q:
                    v_in         = np.array( utils.spherical_to_cartesian(v_abs * V_TH, v_theta, v_phi) )
                    #print("incident velocity : ",v_in, " speed: ",np.linalg.norm(v_in,2), "sp: ",[v_abs,v_theta,v_phi])
                    #\int_S^2 from scattering integral
                    for theta_i, theta in enumerate(theta_q):
                        diff_cs      = ( np.linalg.norm(v_in,2) ) *  collision.differential_cross_section(total_cs,energy_in_ev,theta)
                        #print(diff_cs)
                        for phi in phi_q:
                                v_sc_ct        = collision.compute_scattering_velocity(v_in,theta,phi)
                                if(collision._type == collisions.CollisionType.EAR_G2):
                                    # v_sp_sp1- scattered electron v_sc_sp2: ejected electron
                                    v_sc_sp1   = utils.cartesian_to_spherical(v_sc_ct[0][0], v_sc_ct[0][1], v_sc_ct[0][2])
                                    v_sc_sp2   = utils.cartesian_to_spherical(v_sc_ct[1][0], v_sc_ct[1][1], v_sc_ct[1][2])
                                    
                                    v_sc_sp1[0]   = v_sc_sp1[0]/V_TH
                                    v_sc_sp2[0]   = v_sc_sp2[0]/V_TH

                                    mr1  = maxwellian(v_sc_sp1[0]) / maxwellian(v_abs)
                                    mr2  = maxwellian(v_sc_sp2[0]) / maxwellian(v_abs)

                                    for pj in range(num_p): # k
                                        for lm2_idx,lm2 in enumerate(sph_harm_lm): # qs
                                            for lm1_idx,lm1 in enumerate(sph_harm_lm): #lm
                                                qr[pj, lm2_idx, lm1_idx, qi] += (AR_NEUTRAL_N*(spherical_quadrature_fac**2) * glw[theta_i] * glw[v_theta_i] \
                                                                            * spec.basis_eval_spherical(v_theta, v_phi, lm2[0], lm2[1]) \
                                                                            * diff_cs\
                                                                            * (mr2*spec.basis_eval_full(v_sc_sp2[0], v_sc_sp2[1], v_sc_sp2[2], pj, lm1[0], lm1[1]) + mr1*spec.basis_eval_full(v_sc_sp1[0], v_sc_sp1[1], v_sc_sp1[2], pj, lm1[0], lm1[1])  -   spec.basis_eval_full(v_abs,v_theta, v_phi, pj, lm1[0], lm1[1])))
                                else:
                                    v_sc_sp         = utils.cartesian_to_spherical(v_sc_ct[0], v_sc_ct[1], v_sc_ct[2])
                                    # scale back to the thermal velocity
                                    v_sc_sp[0]      = v_sc_sp[0]/V_TH
                                    mr  = maxwellian(v_sc_sp[0]) / maxwellian(v_abs)
                                    for pj in range(num_p): # k
                                        for lm2_idx,lm2 in enumerate(sph_harm_lm): # qs
                                            for lm1_idx,lm1 in enumerate(sph_harm_lm): #lm
                                                qr[pj, lm2_idx, lm1_idx, qi] += (AR_NEUTRAL_N*(spherical_quadrature_fac**2) * glw[theta_i] * glw[v_theta_i] \
                                                                            * spec.basis_eval_spherical(v_theta, v_phi, lm2[0], lm2[1]) \
                                                                            * diff_cs\
                                                                            * (mr*spec.basis_eval_full(v_sc_sp[0], v_sc_sp[1], v_sc_sp[2], pj, lm1[0], lm1[1])  -   spec.basis_eval_full(v_abs,v_theta, v_phi, pj, lm1[0], lm1[1])))

        # for pj in range(num_p): # k
        #     for lm2_idx,lm2 in enumerate(sph_harm_lm): # qs
        #         for lm1_idx,lm1 in enumerate(sph_harm_lm): # lm                                                                        
        #             print("k: ",pj," qs : ",lm2," lm: ",lm1," value : ",qr[pj,lm2_idx,lm1_idx,:])
        for pi in range(num_p):
            for lm2_idx,lm2 in enumerate(sph_harm_lm):
                for pj in range(num_p):
                    for lm1_idx,lm1 in enumerate(sph_harm_lm):
                        for qi,v_abs in enumerate(gmx):
                            i_id = pi*num_sph_harm + lm2_idx
                            j_id = pj*num_sph_harm + lm1_idx
                            w_factor = ((v_abs**2 ) * maxwellian(v_abs) * (V_TH**3)) / weight_func(v_abs)
                            L_ij[i_id,j_id] += w_factor * gmw[qi] * qr[pj, lm2_idx, lm1_idx, qi]  \
                                * spec.basis_eval_radial(v_abs, pi)

        return L_ij

    