"""
@package: basis functions utilities
"""

import numpy as np
import enum
import abc

class BasisType(enum.Enum):
    """
    Currently supported basis types. 
    """
    HERMITE_E_POLY=0
    HERMITE_POLY=1
    MAXWELLIAN_POLY=2
    SPHERICAL_HARMONIC=3
    LEGENDRE=4


class Basis(abc.ABC):
    abc.abstractmethod
    def __init__(self):
        pass
    
    abc.abstractmethod
    def Pn(self,deg,domain=None,window=None):
        pass

    abc.abstractmethod
    def Gauss_Pn(self,deg):
        pass
    
    abc.abstractmethod
    def Wx(self):
        """
        Weight function w.r.t. the polynomials are orthogonal
        """
        pass

class HermiteE(Basis):
    def __init__(self):
        self._basis_type = BasisType.HERMITE_E_POLY

    def Pn(self,deg,domain=None,window=None):
        """
        returns 1d He polynomial the specified degree (normalized probabilist's)
        """
        return np.polynomial.hermite_e.HermiteE.basis(deg,domain,window)
    
    def Gauss_Pn(self,deg):
        """
        Quadrature points and the corresponding weights for 1d Gauss-HermineE quadrature. 
        The specified quadrature is exact to poly degree <= 2*degree-1, over [-inf,inf] domain
        """
        return np.polynomial.hermite_e.hermegauss(deg)

    def Wx(self):
        """
        Weight function w.r.t. the polynomials are orthogonal
        """
        return np.polynomial.hermite_e.hermeweight


class Hermite(Basis):
    def __init__(self):
        self._basis_type = BasisType.HERMITE_POLY

    def Pn(self,deg,domain=None,window=None):
        """
        returns 1d Hn polynomial the specified degree 
        """
        return np.polynomial.hermite.Hermite.basis(deg,domain,window)
    
    def Gauss_Pn(self,deg):
        """
        Quadrature points and the corresponding weights for 1d Gauss-Hermite quadrature. 
        The specified quadrature is exact to poly degree <= 2*degree-1, over [-inf,inf] domain
        """
        return np.polynomial.hermite.hermgauss(deg)

    def Wx(self):
        """
        Weight function w.r.t. the polynomials are orthogonal
        """
        return np.polynomial.hermite.hermweight

class Legendre(Basis):
    def __init__(self):
        self._basis_type = BasisType.LEGENDRE

    def Pn(self,deg,domain=None,window=None):
        """
        returns 1d He polynomial the specified degree 
        """
        return np.polynomial.legendre.Legendre.basis(deg,domain,window)
        
    def Gauss_Pn(self,deg):
        """
        Gauss-legendre quadrature 1d points for specified degree.
        """
        return np.polynomial.legendre.leggauss(deg)

    def Wx(self):
        """
        Weight function w.r.t. the polynomials are orthogonal
        """
        I = lambda x: 1
        return I






