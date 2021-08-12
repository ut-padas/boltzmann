"""
@package: Generic class to store spectral discretization in spherical coordinates, 

"""
import numpy as np
from numpy.lib.twodim_base import diag
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
        return Y
    
    def basis_eval_full(self,r,phi,theta,k,l,m):
        """
        Evaluates 
        """
        return self._basis_1d[k](r) * self._sph_harm_real(l, m, phi, theta)
    
    def basis_eval_radial(self,r,k):
        """
        Evaluates 
        """
        return self._basis_1d[k](r)
    
    def basis_eval_spherical(self,phi,theta,l,m):
        """
        Evaluates 
        """
        return self._sph_harm_real(l, m, phi, theta)

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
                        mm_diag[r_id] += gw[qi] \
                            * self.basis_eval_radial(v_abs, pi)**2
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
                                mm[r_id,c_id]+= gw[qi] \
                                    * self.basis_eval_radial(v_abs, pi) \
                                    * self.basis_eval_radial(v_abs, pj)
            return mm

    # todo
    # def compute_coefficients(self,func,mm_diag=None):
    #     """
    #     computes basis coefficients for a given function,
    #     for basis orthogonal w.r.t. weight function w(x)
    #     f(x) = w(x) \sum_i c_i P_i(x)
    #     """


