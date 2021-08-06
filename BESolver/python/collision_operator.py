"""
@package : Spectral based, Petrov-Galerkin discretization of the collission integral
"""
import abc
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

    def assemble_collision_mat(self,collision,maxwellian):
        """
        Compute the spectral PG discretization for specified Collision. 
        """
        assert (self._p == self._spec._p)
        num_p = self._spec._p +1
        num_q_pts_on_v      = num_p
        num_q_pts_on_sphere = num_p
        
        [ghx,ghw]   = self._spec._basis_p.Gauss_Pn(num_q_pts_on_v)
        weight_func = self._spec._basis_p.Wx()

        
        legendre = basis.Legendre()
        [glx,glw] = legendre.Gauss_Pn(num_q_pts_on_sphere)
        theta_q = np.arccos(glx)
        phi_q = np.linspace(0,2*np.pi,2*(num_q_pts_on_sphere))

        L_ij =self._spec.create_mat()
        qr = np.zeros( (num_p**self._dim, num_q_pts_on_v**self._dim))
        spherical_quadrature_fac = (np.pi/num_q_pts_on_sphere)

        for tk in range(num_p):
            for tj in range(num_p):
                for ti in range(num_p):
                    for qk,vz in enumerate(ghx):
                        for qj,vy in enumerate(ghx):
                            for qi,vx in enumerate(ghx):
                                # loop for quadrature over sphere
                                for phi in phi_q:
                                    for theta_i, theta in enumerate(theta_q):
                                        j_id      = tk * num_p * num_p + tj * num_p + ti
                                        quad_id   = qk * num_q_pts_on_v * num_q_pts_on_v + qj * num_q_pts_on_v + qi
                                        # velocity at quadrature point. 
                                        v         = np.array([vx,vy,vz])
                                        v_abs     = np.linalg.norm(v,2)
                                        # solid angle for spherical quadrature points. 
                                        omega     = (theta,phi)

                                        # prior velocity, this determined by the collision type 
                                        #v_abs * np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
                                        vp          = collision.post_vel_to_pre_vel(v,0,omega)[0] 
                                        
                                        # maxwellian value for v, and vp
                                        w_vp        = maxwellian(np.linalg.norm(vp,2))
                                        w_v         = maxwellian(np.linalg.norm(v,2))
                                        #print("w_v %f and w_vp %f" %(w_v,w_vp))

                                        #print(f'v = %s and vp =%s '%(v,vp))
                                        qr[j_id,quad_id] += spherical_quadrature_fac * glw[theta_i] * ( w_vp * self._spec.basis_eval3d(vp[0],vp[1],vp[2],(ti,tj,tk)) - w_v * self._spec.basis_eval3d(v[0],v[1],v[2],(ti,tj,tk))) * collision.cross_section(v,omega)


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
                                            v          = np.array([vx,vy,vz])
                                            v_abs      = np.linalg.norm(v,2)
                                            i_id       = pk * num_p * num_p + pj * num_p + pi
                                            j_id       = tk * num_p * num_p + tj * num_p + ti
                                            quad_id    = qk * num_q_pts_on_v * num_q_pts_on_v + qj * num_q_pts_on_v + qi
                                            wf_inv = (1.0/weight_func(v_abs))
                                            L_ij[i_id,j_id] += (ghw[qi] * ghw[qj] * ghw[qk]) * wf_inv * self._spec.basis_eval3d(vx,vy,vz,(pi,pj,pk)) * qr[j_id,quad_id]


        return L_ij


                                            



                    










                    






        

        

        