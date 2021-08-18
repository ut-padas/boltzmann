"""
@package : Spectral based, Petrov-Galerkin discretization of the collission integral
"""
import abc
from math import sin
from typing import MappingView

from numpy.lib.function_base import _quantile_is_valid
import basis
import spec as sp
import collisions
import scipy.constants
import pint
import parameters as params
import numpy as np

class CollissionOp(abc.ABC):

    def __init__(self,dim,p_order):
        self._dim = dim
        self._p = p_order
        pass

    @abc.abstractmethod
    def assemble_mat(collision,maxwellian):
        pass


SPEC_HERMITE_E = sp.SpectralExpansion(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER, basis.HermiteE())

class CollisionOp3D():
    """
    3D- velocity space, collision operator for species s-collisions
    with background heavy particles, where heavy particles obey the
    direc delta distribution function. 
    """
    def __init__(self,dim,p_order) -> None:
        self._dim  = dim
        self._p    = p_order
        self._spec = SPEC_HERMITE_E

    @staticmethod
    def assemble_mat(collision : collisions.Collisions , maxwellian):
        spec          = SPEC_HERMITE_E
        dim           = spec._dim
        num_p         = spec._p + 1
        num_q_v       = params.BEVelocitySpace().NUM_Q_PTS_ON_V
        num_q_s       = params.BEVelocitySpace().NUM_Q_PTS_ON_SPHERE

        [ghx,ghw]     = spec._basis_p.Gauss_Pn(num_q_v)
        weight_func   = spec._basis_p.Wx()

        legendre      = basis.Legendre()
        [glx,glw]     = legendre.Gauss_Pn(num_q_s)
        theta_q       = np.arccos(glx)
        phi_q         = np.linspace(0,2*np.pi,2*(num_q_s))

        L_ij =spec.create_mat()

        # this part of the integral is independent of the collision type, 
        # basically computes based on the pre-collision velocity.         
        #Lij = - \int_{v} M(v) Pi(v) Pj(v) sigma(||v||) dv
        #Pi
        V_TH         = collisions.ELECTRON_THEMAL_VEL
        ELE_VOLT     = collisions.ELECTRON_VOLT
        MAXWELLIAN_N = collisions.MAXWELLIAN_N
        AR_NEUTRAL_N = collisions.AR_NEUTRAL_N

        # v'= v'(v_g,\chi,\phi)
        # qr[Pj,V_g] = \int_{\chi} \int_{\phi} M(v') P_j(v') \sigma(|v_g|,\chi) sin\chi d\chi d\phi
        #qr                       = np.zeros((num_p**dim,num_q_v**dim))
        spherical_quadrature_fac = (np.pi/num_q_s)
        v_in = np.zeros(3)

        for qk,vz in enumerate(ghx):
            for qj,vy in enumerate(ghx):
                for qi,vx in enumerate(ghx):
                    v_in[0]   = vx * V_TH
                    v_in[1]   = vy * V_TH
                    v_in[2]   = vz * V_TH
                    energy_in_ev = (0.5*collisions.MASS_ELECTRON * (np.linalg.norm(v_in,2))**2) / ELE_VOLT
                    if (energy_in_ev <= collision.min_energy_threshold()):
                        print("skipping energy : ", energy_in_ev," for v: ",v_in," Eth : ", collision.min_energy_threshold()," col type: ",collision._type)
                        continue
                    total_cs     = collision.total_cross_section(energy_in_ev)
                    # spherical quadrature
                    for theta_i, theta in enumerate(theta_q):
                        diff_cs = np.linalg.norm(v_in,2)*collision.differential_cross_section(total_cs,energy_in_ev,theta)
                        for phi in phi_q:
                            # solid angle for spherical quadrature points. 
                            #print(v_in)
                            vs      = collision.compute_scattering_velocity(v_in,theta,phi)
                            if(collision._type == collisions.CollisionType.EAR_G2):
                                vs1_th   = vs[0]/V_TH
                                vs2_th   = vs[1]/V_TH
                                vs1_th_abs = np.linalg.norm(vs1_th,2)
                                vs2_th_abs = np.linalg.norm(vs2_th,2)

                                for pk in range(num_p):
                                    for pj in range(num_p):
                                        for pi in range(num_p):
                                            Pi_v = spec.basis_eval3d(vx,vy,vz,(pi,pj,pk))
                                            Mv   = maxwellian(np.sqrt(vx**2 + vy**2 + vz**2))
                                            # Pj
                                            for tk in range(num_p):
                                                for tj in range(num_p):
                                                    for ti in range(num_p):
                                                        i_id       = pk * num_p * num_p + pj * num_p + pi
                                                        j_id       = tk * num_p * num_p + tj * num_p + ti
                                                        Pj_vs1     = spec.basis_eval3d(vs1_th[0],vs1_th[1],vs1_th[2],(ti,tj,tk))
                                                        Pj_vs2     = spec.basis_eval3d(vs2_th[0],vs2_th[1],vs2_th[2],(ti,tj,tk))
                                                        Pj_v       = spec.basis_eval3d(vx,vy,vz,(ti,tj,tk))
                                                        Mvs1        = maxwellian(vs1_th_abs)
                                                        Mvs2        = maxwellian(vs2_th_abs)
                                                        wf_inv     = (1.0/weight_func(np.sqrt(vx**2 + vy**2 + vz**2)))
                                                        L_ij[i_id,j_id] += ( (V_TH**3) * ghw[qi] * ghw[qj] * ghw[qk] * spherical_quadrature_fac * glw[theta_i] * AR_NEUTRAL_N * diff_cs * wf_inv * ( Mvs2 * Pi_v * Pj_vs2 + Mvs1 * Pi_v * Pj_vs1 - Mv * Pi_v * Pj_v) )
                            else:
                                vs_th   = vs/V_TH
                                vs_th_abs = np.linalg.norm(vs_th,2)
                                for pk in range(num_p):
                                    for pj in range(num_p):
                                        for pi in range(num_p):
                                            Pi_v = spec.basis_eval3d(vx,vy,vz,(pi,pj,pk))
                                            Mv   = maxwellian(np.sqrt(vx**2 + vy**2 + vz**2))
                                            # Pj
                                            for tk in range(num_p):
                                                for tj in range(num_p):
                                                    for ti in range(num_p):
                                                        i_id       = pk * num_p * num_p + pj * num_p + pi
                                                        j_id       = tk * num_p * num_p + tj * num_p + ti
                                                        Pj_vs      = spec.basis_eval3d(vs_th[0],vs_th[1],vs_th[2],(ti,tj,tk))
                                                        Pj_v       = spec.basis_eval3d(vx,vy,vz,(ti,tj,tk))
                                                        Mvs        = maxwellian(vs_th_abs)
                                                        wf_inv     = (1.0/weight_func(np.sqrt(vx**2 + vy**2 + vz**2)))
                                                        L_ij[i_id,j_id] += ( (V_TH**3) * ghw[qi] * ghw[qj] * ghw[qk] * spherical_quadrature_fac * glw[theta_i] * AR_NEUTRAL_N * diff_cs * wf_inv * (Mvs*Pi_v * Pj_vs - Mv*Pi_v * Pj_v) )
        
        return L_ij
