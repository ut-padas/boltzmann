"""
@package : Spectral based, Petrov-Galerkin discretization of the collission integral
"""
import abc
from re import U

import basis
import spec as sp
import binary_collisions
import scipy.constants
import pint
import boltzmann_parameters as bp
import numpy as np

class SpecWeakFormCollissionOp(abc.ABC):

    def __init__(self,dim,p_order):
        self._dim = dim
        self._p = p_order
        pass

    @abc.abstractmethod
    def assemble_collision_mat(self,collision):
        pass


class CollisionOpElectronNeutral3D(SpecWeakFormCollissionOp):
    """
    Collission operator for electron-neutral. Density function 
    for the neutral assumed to be delta function for all times. 
    """

    def __init__(self,p_order,basis_p):
        """
        p_order - polynomial order of the approximation
        basis_p - which basis to use for the expansion
        """
        super().__init__(3, p_order)
        self._basis_p    = basis_p
        self._spec       = sp.SpectralExpansion(self._dim,self._p,self._basis_p)

    def get_spectral_structure(self):
        """
        Returns the underlying spectral data structure object. 
        """
        return self._spec

    def assemble_collision_mat(self,collision):
        """
        Compute the spectral PG discretization for specified Collision. 
        """
        legendre = basis.Legendre()
        # quadrature over the sphere, 
        assert (self._p == self._spec._p)
        num_p = self._spec._p +1
        [glx,glw] = legendre.Gauss_Pn(num_p)
        theta_q = np.arccos(glx)
        #print(glx)
        #print(theta_q)
        phi_q = np.linspace(0,2*np.pi,2*(num_p))
        
        # maxwellian projected on to the basis and the integration. 
        # M_i = \int_{R^3} \int_{S^2} (M(v\prime) - M(v) P_i(v) ) B(|v|,\omega) d\omega dv
        Mi = self._spec.create_vec()
        # Lij collision integral as written on the nodes. 
        # L_{ij} = \int_{R^3} \int_{S^2} (M(v\prime)P_i(v)P_j(v^\prime) - M(v) P_i(v)P_j(v)) B(|v|,\omega) d\omega dv
        L_ij =self._spec.create_mat()

        
        # integration over the sphere for the values of v defined on Gauss-Hermite quadrature points.
        # !!!! Note currently assumes for binary elastic collisions, v= -v',
        # for other collisions, I think we need to do a change of variables to correct the weight function. 
        
        [ghx,ghw] = self._spec._basis_p.Gauss_Pn(num_p)
        qr = np.zeros(num_p**self._dim)
        for qk,vz in enumerate(ghx):
            for qj,vy in enumerate(ghx):
                for qi,vx in enumerate(ghx):
                    # loop for quadrature over sphere
                    for phi in phi_q:
                        for theta_i, theta in enumerate(theta_q):
                            r_id      = qk * num_p * num_p + qj * num_p + qi
                            v         = np.array([vx,vy,vz])
                            omega     = (theta,phi)
                            qr[r_id] += glw[theta_i]*collision.cross_section(v,omega)

        qr = (np.pi/num_p) * qr 

        # loop for 3d polynomial
        for pk in range(num_p):
            for pj in range(num_p):
                for pi in range(num_p):
                    # GH quadrature loop
                    for qk,vz in enumerate(ghx):
                        for qj,vy in enumerate(ghx):
                            for qi,vx in enumerate(ghx):
                                r_id      = pk * num_p * num_p + pj * num_p + pi
                                quad_id   = qk * num_p * num_p + qj * num_p + qi
                                Mi[r_id] += (ghw[qi] * ghw[qj] * ghw[qk]) * self._spec.basis_eval3d(vx,vy,vz,(pi,pj,pk)) * qr[quad_id]
        
        ## computation of Lij
        # Pi(x)
        for pk in range(num_p):
            for pj in range(num_p):
                for pi in range(num_p):
                    #Pj(x)
                    for tk in range(num_p):
                        for tj in range(num_p):
                            for ti in range(num_p):
                                # GH quadrature loop
                                for qk,vz in enumerate(ghx):
                                    for qj,vy in enumerate(ghx):
                                        for qi,vx in enumerate(ghx):
                                            r_id   = pk * num_p * num_p + pj * num_p + pi
                                            c_id   = tk * num_p * num_p + tj * num_p + ti
                                            quad_id  = qk * num_p * num_p + qj * num_p + qi
                                            v      = np.array([vx,vy,vz])
                                            u      = np.array([0,0,0])
                                            [vp,up]=collision.post_vel_to_pre_vel(v,u,(0,0))
                                            L_ij[r_id,c_id] += (ghw[qi] * ghw[qj] * ghw[qk]) *(self._spec.basis_eval3d(vx,vy,vz,(pi,pj,pk)) * self._spec.basis_eval3d(vp[0],vp[1],vp[2],(ti,tj,tk)) - self._spec.basis_eval3d(vx,vy,vz,(pi,pj,pk)) * self._spec.basis_eval3d(vx,vy,vz,(ti,tj,tk))) * qr[quad_id]
        
        #print("Mi  = \n %s" %Mi)
        #print("Lij = \n %s" %L_ij)
        return [Mi,L_ij]


                                            



                    










                    






        

        

        