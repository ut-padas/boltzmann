"""
@package : Spectral based, Petrov-Galerkin discretization of the collission integral
"""
import abc
from re import U

import basis
import spec_spherical as sp
import binary_collisions
import scipy.constants
import pint
import boltzmann_parameters as bp
import numpy as np
import maxpoly

class SpecWeakFormCollissionOpSpherical(abc.ABC):

    def __init__(self,p_order,sph_harm_lm,num_q_pts_on_v,num_q_pts_on_sphere):
        self._p = p_order # number of polynomials in radial direction
        self._sph_harm_lm = sph_harm_lm # (l,m) index pairs of shperical harmonics to use
        self._num_q_pts_on_v = num_q_pts_on_v
        self._num_q_pts_on_sphere = num_q_pts_on_sphere
        pass

    @abc.abstractmethod
    def assemble_collision_mat(self,collision):
        pass


class CollisionOpElectronNeutral3DSpherical(SpecWeakFormCollissionOpSpherical):
    """
    Collission operator for electron-neutral. Density function 
    for the neutral assumed to be delta function for all times. 
    """

    def __init__(self,p_order,sph_harm_lm,num_q_pts_on_v,num_q_pts_on_sphere,basis_p):
        """
        p_order - polynomial order of the approximation
        basis_p - which basis to use for the expansion
        """
        super().__init__(p_order,sph_harm_lm,num_q_pts_on_v,num_q_pts_on_sphere)
        self._basis_p    = basis_p
        self._spec       = sp.SpectralExpansionSpherical(p_order, basis_p, sph_harm_lm)

    def get_spectral_structure(self):
        """
        Returns the underlying spectral data structure object. 
        """
        return self._spec

    def assemble_collision_mat(self,collision,maxwellian):
        """
        Compute the spectral PG discretization for specified Collision. 
        """
        assert (self._p == self._spec._p)
        num_p = self._spec._p +1
        num_q_pts_on_v      = self._num_q_pts_on_v
        num_q_pts_on_sphere = self._num_q_pts_on_sphere

        num_sph_harm = len(self._sph_harm_lm)
        
        [gmx,gmw] = maxpoly.maxpolygauss(num_q_pts_on_v)
        
        legendre = basis.Legendre()
        [glx,glw] = legendre.Gauss_Pn(num_q_pts_on_sphere)
        theta_q = np.arccos(glx)
        phi_q = np.linspace(0,2*np.pi,2*(num_q_pts_on_sphere))

        L_ij =self._spec.create_mat()
        spherical_quadrature_fac = (np.pi/num_q_pts_on_sphere)
        qr = np.zeros([num_p, num_sph_harm, num_sph_harm, len(gmx)])

        for qi,v_abs in enumerate(gmx): # loop over quadrature points
            for pj in range(num_p): # loop over polynomial term
                for lm1_idx,lm1 in enumerate(self._sph_harm_lm): # loop over first spherical harmonics term 
                    for lm2_idx,lm2 in enumerate(self._sph_harm_lm): # loop over second spherical harmonics term 
                        # first integration over sphere
                        for phi in phi_q:
                            for theta_i, theta in enumerate(theta_q):
                                #second integration over sphere
                                for v_phi in phi_q:
                                    for v_theta_i, v_theta in enumerate(theta_q):
                                        # velocity at quadrature point. 
                                        v = [v_abs, v_theta, v_phi]
                                        # solid angle for spherical quadrature points. 
                                        omega = (theta,phi)

                                        # prior velocity, this determined by the collision type 
                                        vp = collision.post_vel_to_pre_vel_sph(v,0,omega)[0]

                                        # ratio of maxwellian values for v, and vp (this is kinda ugly)
                                        # ~exp(-vp^2)/exp(-v^2) = exp(-(vp^2-v^2)) = exp(-(vp-v)*(vp+v)) = exp(-(sqrt((vp-v)*(vp+v)))^2)
                                        mr = np.real(maxwellian(np.sqrt(complex((vp[0]-v[0])*(vp[0]+v[0])))))

                                        qr[pj, lm1_idx, lm2_idx, qi] += spherical_quadrature_fac**2 * glw[theta_i] * glw[v_theta_i] \
                                            * self._spec.basis_eval_spherical(v[1], v[2], lm2[0], lm2[1]) \
                                            * collision.cross_section_sph(v, omega) \
                                            * (mr*self._spec.basis_eval_full(vp[0], vp[1], vp[2], pj, lm1[0], lm1[1])
                                                - self._spec.basis_eval_full(v[0], v[1], v[2], pj, lm1[0], lm1[1]))
        
        for pj in range(num_p):
            for pi in range(num_p):
                for lm1_idx,lm1 in enumerate(self._sph_harm_lm):
                    for lm2_idx,lm2 in enumerate(self._sph_harm_lm):
                        for qi,v_abs in enumerate(gmx):
                            i_id = pi*num_sph_harm + lm1_idx
                            j_id = pj*num_sph_harm + lm2_idx
                            L_ij[i_id,j_id] += gmw[qi] * qr[pj, lm1_idx, lm2_idx, qi]  \
                                * self._spec.basis_eval_radial(v_abs, pi)

        return L_ij
