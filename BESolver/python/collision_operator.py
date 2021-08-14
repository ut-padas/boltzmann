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
    def assemble_mat(self,collision,maxwellian):
        pass


SPEC_HERMITE_E = sp.SpectralExpansion(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VEL_SPACE_POLY_ORDER, basis.BasisType.HERMITE_E_POLY)

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

    def assemble_mat(collision : collisions.Collisions , maxwellian):
        spec          = SPEC_HERMITE_E
        dim           = spec._dim
        num_p         = spec._p + 1
        num_q_v       = num_p
        num_q_s       = num_p

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
        for pk in range(num_p):
            for pj in range(num_p):
                for pi in range(num_p):
                    # Pj
                    for tk in range(num_p):
                        for tj in range(num_p):
                            for ti in range(num_p):
                                # GH quad loop
                                for qk,vz in enumerate(ghx):
                                    for qj,vy in enumerate(ghx):
                                        for qi,vx in enumerate(ghx):
                                            energy     = 0.5 * collisions.MASS_ELECTRON * (vx**2 + vy**2 + vz**2)
                                            total_cs   = collision.total_cross_section(energy)
                                            v_mag      = np.sqrt(vx**2 + vy**2 + vz**2)
                                            Mv         = maxwellian(v_mag)
                                            i_id       = pk * num_p * num_p + pj * num_p + pi
                                            j_id       = tk * num_p * num_p + tj * num_p + ti
                                            wf_inv     = (Mv/weight_func(v_mag))
                                            L_ij[i_id,j_id] -= ((ghw[qi] * ghw[qj] * ghw[qk]) * wf_inv * spec.basis_eval3d(vx,vy,vz,(pi,pj,pk))*spec.basis_eval3d(vx,vy,vz,(ti,tj,tk))*total_cs) 

        
        # v'= v'(v_g,\chi,\phi)
        # qr[Pj,V_g] = \int_{\chi} \int_{\phi} M(v') P_j(v') \sigma(|v_g|,\chi) sin\chi d\chi d\phi
        qr                       = np.zeros((num_p**dim,num_q_v**dim))
        spherical_quadrature_fac = (np.pi/num_q_s)

        v_in = np.zeros(3)

        # Pj
        for tk in range(num_p):
            for tj in range(num_p):
                for ti in range(num_p):
                    # GH quad loop
                    for qk,vz in enumerate(ghx):
                        for qj,vy in enumerate(ghx):
                            for qi,vx in enumerate(ghx):
                                v_in[0]   = vx
                                v_in[1]   = vy
                                v_in[2]   = vz
                                v_mag     = np.sqrt(vx**2 + vy**2 + vz**2)
                                energy    = 0.5 * collisions.MASS_ELECTRON * (vx**2 + vy**2 + vz**2)
                                total_cs  = collision.total_cross_section(energy)
                                Pj_id     = tk * num_p * num_p + tj * num_p + ti
                                Vq_id     = qk * num_q_v * num_q_v + qj * num_q_v + qi
                                sq_value  = 0
                                # spherical quadrature
                                for theta_i, theta in enumerate(theta_q):
                                    diff_cs = collision.differential_cross_section(energy,theta)
                                    s_theta = np.sin(theta)
                                    for phi in phi_q:
                                        # solid angle for spherical quadrature points. 
                                        vs      = collision.compute_scattering_velocity(v_in,theta,phi)
                                        Pj_vs   = spec.basis_eval3d(vs[0],vs[1],vs[2],(ti,tj,tk))
                                        # maxwellian value for v, and vp
                                        M_vs      = maxwellian(np.linalg.norm(vs,2))
                                        wf_inv    = (M_vs/weight_func(v_mag))
                                        sq_value += (glw[theta_i] * wf_inv * Pj_vs *diff_cs * s_theta)

                                sq_value *= spherical_quadrature_fac
                                qr[Pj_id,Vq_id] = sq_value


        # now add the \int_{v} ||v|| Pi(v) qr to Lij
        for pk in range(num_p):
            for pj in range(num_p):
                for pi in range(num_p):
                    # GH quad loop
                    sq_value=0
                    for qk,vz in enumerate(ghx):
                        for qj,vy in enumerate(ghx):
                            for qi,vx in enumerate(ghx):
                                v_mag     = np.sqrt(vx**2 + vy**2 + vz**2)
                                Pi_v      = spec.basis_eval3d(vx,vy,vz,(pi,pj,pk))
                                Vq_id     = qk * num_q_v * num_q_v + qj * num_q_v + qi

                                
                                sq_value += (ghw[qi] * ghw[qj] * ghw[qk] * v_mag * Pi_v * qr[pk*num_p*num_p + pj*num_p + pi])