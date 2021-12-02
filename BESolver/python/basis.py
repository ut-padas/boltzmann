"""
@package: basis functions utilities
"""

import numpy as np
import enum
import abc
import maxpoly
import lagpoly
import scipy.interpolate

class BasisType(enum.Enum):
    """
    Currently supported basis types. 
    """
    HERMITE_E_POLY=0
    HERMITE_POLY=1
    MAXWELLIAN_POLY=2
    SPHERICAL_HARMONIC=3
    LEGENDRE = 4
    LAGUERRE = 5
    SPLINES  = 6


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
        
class Maxwell(Basis):
    def __init__(self):
        self._basis_type = BasisType.MAXWELLIAN_POLY

    def Pn(self,deg,domain=None,window=None):
        """
        returns 1d Maxwell polynomial the specified degree 
        """
        return maxpoly.basis(deg)
    
    def Gauss_Pn(self,deg):
        """
        Quadrature points and the corresponding weights for 1d Gauss quadrature. 
        The specified quadrature is exact to poly degree <= 2*degree-1, over [0,inf] domain
        """
        return maxpoly.maxpolygauss(max(1,deg-1))

    def Wx(self):
        """
        Weight function w.r.t. the polynomials are orthogonal
        """
        return maxpoly.maxpolyweight
        
class Laguerre(Basis):
    def __init__(self):
        self._basis_type = BasisType.LAGUERRE

    def Pn(self,deg,domain=None,window=None):
        """
        returns 1d associated (k=1/2) Laguerre polynomial the specified degree 
        """
        return lagpoly.basis(deg)
    
    def Gauss_Pn(self,deg):
        """
        Quadrature points and the corresponding weights for 1d Gauss quadrature. 
        The specified quadrature is exact to poly degree <= 2*degree-1, over [0,inf] domain
        """
        return lagpoly.lagpolygauss(max(1,deg-1))

    def Wx(self):
        """
        Weight function w.r.t. the polynomials are orthogonal
        """
        return lagpoly.lagpolyweight


class BSpline(Basis):

    def __init__(self,knots,spline_order, num_c_pts):
        self._basis_type = BasisType.SPLINES
        assert len(knots) == num_c_pts + (spline_order+1) + 1, "knots vector length does not match the spline order"
        self._num_c_pts = num_c_pts
        self._sp_order = spline_order
        self._t        = knots
        self._splines = [scipy.interpolate.BSpline.basis_element(knots[i:i+spline_order+2],False) for i in range(num_c_pts)]

    def Pn(self,deg,domain=None,window=None):
        return self._splines[deg]

    def Gauss_Pn(self,deg,from_zero=True):
        """
        Quadrature points and the corresponding weights for 1d Gauss quadrature. 
        The specified quadrature is exact to poly degree <= 2*degree-1, over [0,inf] domain
        """
        if from_zero:
            return uniform_simpson((0,self._t[-1]),deg)
        else:
            return uniform_simpson((self._t[0],self._t[-1]),deg)
    
    def Wx(self):
        """
        Weight function w.r.t. the polynomials are orthogonal
        """
        I = lambda x: 1
        return I

def uniform_simpson(domain, pts):
    assert pts%2==1, "simpson requires even number of intervals"
    assert pts>1   , "composite simpson require more than 1"
    gx = np.linspace(domain[0],domain[1],pts)
    dx = (domain[1]-domain[0])/(pts-1)
    gw = np.ones_like(gx) * (dx/3.0)
    gw[range(2,pts-1,2)]  *= 2.0
    gw[range(1,pts,2)]    *= 4.0
    return gx,gw



