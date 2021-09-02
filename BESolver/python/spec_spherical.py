"""
@package: Generic class to store spectral discretization in spherical coordinates, 

"""
import numpy as np
import basis 
from scipy.special import sph_harm
import maxpoly # to perform integration over [0, +inf)

class SpectralExpansionSpherical:
    """
    Handles spectral decomposition with specified orders of expansion, w.r.t. basis_p
    domain and window by default set to None. None assumes the domain is unbounded. 
    """
    def __init__(self, p_order, basis_p, sph_harm_lm, domain=None, window=None):
        """
        @param p_order : number of basis functions used in radial direction
        @param basis_p : np.polynomial object
        @param sph_harm_lm : (l,m) indices of spherical harmonics to use
        """
        self._sph_harm_lm = sph_harm_lm
        self._p = p_order
        self._domain = domain
        self._window = window

        self._basis_p  = basis_p
        self._basis_1d = list()
        
        for deg in range(self._p+1):
            self._basis_1d.append(self._basis_p.Pn(deg,self._domain,self._window))
    

    def _sph_harm_real(self, l, m, theta, phi):
        # in python's sph_harm phi and theta are swapped
        Y = sph_harm(abs(m), l, phi, theta)
        if m < 0:
            Y = np.sqrt(2) * (-1)**m * Y.imag
        elif m > 0:
            Y = np.sqrt(2) * (-1)**m * Y.real
        else:
            Y = Y.real

        return Y 
    
    def basis_eval_full(self,r,theta,phi,k,l,m):
        """
        Evaluates 
        """
        return self._basis_1d[k](r) * self._sph_harm_real(l, m, theta, phi)
    
    def basis_eval_radial(self,r,k):
        """
        Evaluates 
        """
        return self._basis_1d[k](r)
    
    def basis_eval_spherical(self, theta, phi,l,m):
        """
        Evaluates 
        """
        return self._sph_harm_real(l, m, theta, phi)

    def create_vec(self,dtype=float):
        num_c = (self._p +1)*len(self._sph_harm_lm)
        return np.zeros((num_c,1),dtype=dtype)

    def create_mat(self,dtype=float):
        """
        Create a matrix w.r.t the number of spectral coefficients. 
        """
        num_c = (self._p +1)*len(self._sph_harm_lm)
        return np.zeros((num_c,num_c),dtype=dtype)
    
    def get_num_coefficients(self):
        """
        returns the number of coefficients in  the spectral
        representation. 
        """
        return (self._p +1)*len(self._sph_harm_lm)

    def compute_mass_matrix(self,is_diagonal=False):
        """
        Compute the mass matrix w.r.t the basis polynomials
        if the chosen basis is orthogonal set is_diagonal to True. 
        Given we always use spherical harmonics we will integrate them exactly
        """
        num_p = self._p+1
        num_sph_harm = len(self._sph_harm_lm)
        [gx, gw] = self._basis_p.Gauss_Pn(num_p)
        
        # note : r_id, c_id defined inside the loops on purpose (loops become pure nested), assuming it will allow
        # more agressive optimizations from the python
        if(is_diagonal):
            mm_diag = self.create_vec()
            for pi in range(num_p):
                for yi in range(num_sph_harm): # todo: optimize this out
                    # quadrature loop. 
                    for qi,v_abs in enumerate(gx): # loop over quadrature points
                        r_id = pi*num_sph_harm + yi
                        mm_diag[r_id] += gw[qi] * (self.basis_eval_radial(v_abs, pi)**2)
            return mm_diag
            
        else:
            mm = self.create_mat()
            # loop over polynomials i
            for yi in range(num_sph_harm): # todo: optimize this out
                for pi in range(num_p):
                    # loop over polynomials j
                    for yj in range(num_sph_harm): # todo: optimize this out
                        for pj in range(num_p):
                            # quadrature loop. 
                            for qi,v_abs in enumerate(gx): # loop over quadrature points
                                r_id = pi*num_sph_harm + yi
                                c_id = pj*num_sph_harm + yj
                                # this is only true for spherical harmonic basis. 
                                if (yi is not yj):
                                    continue
                                mm[r_id,c_id]+= gw[qi] * self.basis_eval_radial(v_abs, pi) * self.basis_eval_radial(v_abs, pj)
            return mm

    def compute_maxwellian_mm(self,maxwellian,v_th):
        """
        computs the the mass matrix w.r.t specified maxwellian
        for generic maxwellian mm might not be diagonal. 
        """
        num_p = self._p+1
        num_sph_harm = len(self._sph_harm_lm)
        [gx, gw] = self._basis_p.Gauss_Pn(num_p)
        w_func   =self._basis_p.Wx()
        mm = self.create_mat()
        # loop over polynomials i
        for yi in range(num_sph_harm): # todo: optimize this out
            for pi in range(num_p):
                # loop over polynomials j
                for yj in range(num_sph_harm): # todo: optimize this out
                    for pj in range(num_p):
                        # quadrature loop. 
                        for qi,v_abs in enumerate(gx): # loop over quadrature points
                            r_id = pi*num_sph_harm + yi
                            c_id = pj*num_sph_harm + yj
                            # this is only true for spherical harmonic basis. 
                            if (yi is not yj):
                                continue
                            mr = 1#((v_abs**2) * maxwellian(v_abs) * (v_th**3))/w_func(v_abs)
                            mm[r_id,c_id]+= gw[qi] * mr * self.basis_eval_radial(v_abs, pi) * self.basis_eval_radial(v_abs, pj)
        
        return mm

    def Vq_r(self, v_r, scale=1):
        """
        compute the basis Vandermonde for the all the basis function
        for the specified quadrature points.  
        """
        num_p        = self._p+1
        num_q_v_r    = len(v_r)

        _shape = tuple([num_p]) + v_r.shape
        Vq = np.zeros(_shape)

        for i in range(num_p):
            Vq[i] = scale * self.basis_eval_radial(v_r,i)
        
        return Vq

    def Vq_sph_mg(self, v_theta, v_phi, scale=1):
        """
        compute the basis Vandermonde for the all the basis function
        for the specified quadrature points.  
        """

        num_sph_harm = len(self._sph_harm_lm)
        num_q_v_theta   = len(v_theta)
        num_q_v_phi     = len(v_phi)

        [Vt,Vp]      = np.meshgrid(v_theta, v_phi,indexing='ij')

        Vq = np.zeros((num_sph_harm,num_q_v_theta,num_q_v_phi))

        for lm_i, lm in enumerate(self._sph_harm_lm):
            Vq[lm_i] = scale * self.basis_eval_spherical(Vt,Vp,lm[0],lm[1])
        
        return Vq

    def Vq_sph(self, v_theta, v_phi, scale=1):
        """
        compute the basis Vandermonde for the all the basis function
        for the specified quadrature points.  
        """

        num_sph_harm = len(self._sph_harm_lm)
        assert v_theta.shape == v_phi.shape, "invalid shapes, use mesh grid to get matching shapes"
        _shape = tuple([num_sph_harm]) + v_theta.shape
        Vq = np.zeros(_shape)

        for lm_i, lm in enumerate(self._sph_harm_lm):
            Vq[lm_i] = scale * self.basis_eval_spherical(v_theta,v_phi,lm[0],lm[1])
        
        return Vq

    def Vq_full(self, v_r, v_theta, v_phi, scale=1):
        """
        compute the basis Vandermonde for the all the basis function
        for the specified quadrature points.  
        """

        V_r_q   = self.Vq_r(v_r,1.0)
        V_sph_q = self.Vq_sph(v_theta,v_phi,1.0)
        Vq= np.kron(V_r_q,V_sph_q)
        
        return Vq



