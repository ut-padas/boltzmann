"""
@package: Generic class to store spectral discretization in spherical coordinates, 

"""
import numpy as np
import basis 
import enum
from scipy.special import sph_harm
import enum

class QuadMode(enum.Enum):
    GMX     = 0 # default
    SIMPSON = 1


MM_SIMPSON_NUM_Q_PTS=2049
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
        self._q_mode   = QuadMode.GMX
        
        for deg in range(self._p+1):
            self._basis_1d.append(self._basis_p.Pn(deg,self._domain,self._window))

    def get_radial_basis_type(self):
        return self._basis_p._basis_type   

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
        return self.basis_eval_radial(r,k) * self._sph_harm_real(l, m, theta, phi)
    
    def basis_eval_radial(self,r,k):
        """
        Evaluates 
        """
        return np.nan_to_num(self._basis_1d[k](r))
    
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
        num_p  = self._p+1
        num_sh = len(self._sph_harm_lm)
        
        if self.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY:
            [gx, gw] = self._basis_p.Gauss_Pn(num_p)
            Vr = self.Vq_r(gx)
            mr = np.ones_like(gx) 
        elif self.get_radial_basis_type() == basis.BasisType.SPLINES:
            [gx, gw] = self._basis_p.Gauss_Pn(MM_SIMPSON_NUM_Q_PTS,False)
            Vr = self.Vq_r(gx)
            mr = gx**2 
        
        mm = np.array([ mr * Vr[i,:] * Vr[j,:] for i in range(num_p) for j in range(num_p)])
        mm = np.dot(mm,gw).reshape(num_p,num_p)
        Lm = np.eye(num_sh)
        mm = np.kron(mm,Lm).reshape(num_p*num_sh,num_p*num_sh)
        return mm

    def compute_maxwellian_mm(self,maxwellian,v_th):
        """
        computs the the mass matrix w.r.t specified maxwellian
        for generic maxwellian mm might not be diagonal. 
        """
        num_p = self._p+1
        num_sh = len(self._sph_harm_lm)
        
        if self.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY:
            [gx, gw] = self._basis_p.Gauss_Pn(num_p)
            Vr = self.Vq_r(gx)
            mr = np.ones_like(gx) 
        elif self.get_radial_basis_type() == basis.BasisType.SPLINES:
            [gx, gw] = self._basis_p.Gauss_Pn(MM_SIMPSON_NUM_Q_PTS,False)
            Vr = self.Vq_r(gx)
            mr = np.exp(-gx**2) * (gx**2)

        mm = np.array([ mr * Vr[i,:] * Vr[j,:] for i in range(num_p) for j in range(num_p)])
        mm = np.dot(mm,gw).reshape(num_p,num_p)
        Lm = np.eye(num_sh)
        mm = np.kron(mm,Lm).reshape(num_p*num_sh,num_p*num_sh)
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



