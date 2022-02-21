"""
@package: basis functions utilities
"""

import numpy as np
import enum
import abc
import maxpoly
import lagpoly
import scipy.interpolate

# some parameters related to splines. 
XLBSPLINE_NUM_Q_PTS_PER_KNOT   = 31
BSPLINE_NUM_Q_PTS_PER_KNOT     = 7
BSPLINE_BASIS_ORDER            = 1

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

    @staticmethod
    def get_num_q_pts(p_order, s_order, pts_per_knot):
        return pts_per_knot*(p_order+1 + s_order + 2  - 2*(s_order+1))

    def __init__(self,k_domain,spline_order, num_p):
        self._basis_type = BasisType.SPLINES
        """
        x x x     | * * * * * * * * * * | x  x  x
         sp + 1          num_p -2          sp+1   
        """
        # first and last splines have repeated knots, 
        num_k            = 2*spline_order + (num_p -2) + 2
        self._t          = (k_domain[0])*np.ones(spline_order+1)
        knot_base        = 1.5
        self._t          = np.append(self._t,np.logspace(-2, np.log(k_domain[1]-2)/np.log(knot_base) , num_k-2*spline_order -2 ,base=knot_base))
        self._t          = np.append(self._t,k_domain[1]*np.ones(spline_order+1))
        self._num_c_pts  = num_p
        self._sp_order   = spline_order
        self._q_per_knot = BSPLINE_NUM_Q_PTS_PER_KNOT
        self._splines    = [scipy.interpolate.BSpline.basis_element(self._t[i:i+spline_order+2],False) for i in range(num_p)]

    def Pn(self,deg,domain=None,window=None):
        return self._splines[deg]

    def Gauss_Pn(self,deg,from_zero=True):
        """
        Quadrature points and the corresponding weights for 1d Gauss quadrature. 
        The specified quadrature is exact to poly degree <= 2*degree-1, over [0,inf] domain
        """
        assert np.allclose(self._t[self._sp_order],0), "specified knot vector element %d is not aligned with zero"%(self._sp_order)
        num_intervals = self._num_c_pts -1
        assert deg % num_intervals == 0, "specified # quadrature points %d not evenly divided by the number of control points %d" %(deg,num_intervals)
        qx = np.zeros(deg)
        qw = np.zeros(deg)
        for i in range(self._sp_order, self._sp_order + num_intervals):
            qx[(i-self._sp_order)*self._q_per_knot: (i-self._sp_order)*self._q_per_knot + self._q_per_knot], qw[(i-self._sp_order)*self._q_per_knot: (i-self._sp_order)*self._q_per_knot + self._q_per_knot] = uniform_simpson((self._t[i], self._t[i+1]),self._q_per_knot)
        
        # qx,qw=uniform_simpson((self._t[ti], self._t[-ti]),4097)
        # print(qx.shape)
        # print(qx)
        # print(np.sum(qw))
        # print(qx)
        assert np.allclose(np.sum(qw),(self._t[-1]-self._t[0])), "simpson weights computed for splines does not match the knots domain"
        return qx,qw
    
    def Wx(self):
        """
        Weight function w.r.t. the polynomials are orthogonal
        """
        I = lambda x: np.ones_like(x)
        return I

    def derivative(self, deg, dorder):
        assert dorder==1, "not implemented for HO derivatives"
        i=deg
        p=self._sp_order
        c1 = self._t[i+p] - self._t[i]
        c2 = self._t[i+p+1] - self._t[i+1]

        if(c2!=0):
            f_c2 = lambda x : - (p / c2) * np.nan_to_num(scipy.interpolate.BSpline.basis_element(self._t[i+1:i+1 + p+1],False)(x)) 
        else:
            f_c2 = lambda x : np.zeros_like(x)

        if(c1!=0):
            f_c1 = lambda x : (p / c1) * np.nan_to_num(scipy.interpolate.BSpline.basis_element(self._t[i:i + p + 1],False)(x))
        else:
            f_c1 = lambda x : np.zeros_like(x)

        return lambda x : np.nan_to_num(f_c1(x)) + np.nan_to_num(f_c2(x))

    def diff(self,deg, dorder):
        return self._splines[deg].derivative(dorder)

class XlBSpline(Basis):
    """
    Scaled b-splines for advection. 
    b_{kl}= b_k(x) x^l
    b_0 and b_last are used as boundary splines with repeated knots. 
    """
    @staticmethod
    def get_num_q_pts(p_order, s_order, pts_per_knot):
        return pts_per_knot*(p_order)

    def __init__(self,k_domain,spline_order, num_p):
        self._basis_type = BasisType.SPLINES
        """
        x x x     | * * * * * * * * * * | x  x  x
         sp + 1          num_p -2          sp+1   
        """
        # first and last splines have repeated knots, 
        num_k            = 2*spline_order + (num_p -2) + 2
        self._t          = (k_domain[0])*np.ones(spline_order+1)
        knot_base        = 1.5
        self._t          = np.append(self._t,np.logspace(-2, np.log(k_domain[1]-2)/np.log(knot_base) , num_k-2*spline_order -2 ,base=knot_base))
        self._t          = np.append(self._t,k_domain[1]*np.ones(spline_order+1))
        #print("len_t ",len(self._t) , " num_k ",num_k)
        self._num_c_pts  = num_p
        self._sp_order   = spline_order
        self._q_per_knot = XLBSPLINE_NUM_Q_PTS_PER_KNOT
        self._splines    = [scipy.interpolate.BSpline.basis_element(self._t[i:i+spline_order+2],False) for i in range(num_p)]
        
        #print(self._t)
        # import matplotlib.pyplot as plt
        # x=np.linspace(k_domain[0],2,1000)
        # for i in range(1,5):
        #     for l in range(0,3):
        #         #print(self._t[i:i+spline_order+2])
        #         plt.plot(x,self.Pn(i)(l,x),label="kl=(%d,%d)"%(i,l))
        #         #plt.plot(x,self.derivative(i,1)(x),label="b'_i=%d"%i)
        #         #plt.plot(x,self._splines[i].derivative(1)(x),label="python b'_i=%d"%i)
        # #plt.xscale("log")
        # plt.legend()
        # plt.grid()
        # plt.savefig("splines.png")
        # plt.show()
        


    def Pn(self,deg,domain=None,window=None):
        return lambda l,x : np.nan_to_num(self._splines[deg](x)) * (x**(max(l-deg,0)))
        
        
        
    def Gauss_Pn(self,deg,from_zero=True):
        """
        Quadrature points and the corresponding weights for 1d Gauss quadrature. 
        The specified quadrature is exact to poly degree <= 2*degree-1, over [0,inf] domain
        """
        assert np.allclose(self._t[self._sp_order],0), "specified knot vector element %d is not aligned with zero"%(self._sp_order)
        num_intervals = self._num_c_pts -1
        assert deg % num_intervals == 0, "specified # quadrature points %d not evenly divided by the number of control points %d" %(deg,num_intervals)
        qx = np.zeros(deg)
        qw = np.zeros(deg)
        for i in range(self._sp_order, self._sp_order + num_intervals):
            qx[(i-self._sp_order)*self._q_per_knot: (i-self._sp_order)*self._q_per_knot + self._q_per_knot], qw[(i-self._sp_order)*self._q_per_knot: (i-self._sp_order)*self._q_per_knot + self._q_per_knot] = uniform_simpson((self._t[i], self._t[i+1]),self._q_per_knot)
        
        # qx,qw=uniform_simpson((self._t[ti], self._t[-ti]),4097)
        # print(qx.shape)
        # print(qx)
        # print(np.sum(qw))
        # print(qx)
        assert np.allclose(np.sum(qw),(self._t[-1]-self._t[0])), "simpson weights computed for splines does not match the knots domain"
        return qx,qw
    
    def Wx(self):
        """
        Weight function w.r.t. the polynomials are orthogonal
        """
        I = lambda x: np.ones_like(x)
        return I

    def derivative(self, deg, dorder):
        assert dorder==1, "not implemented for HO derivatives"
        i=deg
        p=self._sp_order
        c1 = self._t[i+p] - self._t[i]
        c2 = self._t[i+p+1] - self._t[i+1]

        if(c2!=0):
            f_c2 = lambda x : - (p / c2) * np.nan_to_num(scipy.interpolate.BSpline.basis_element(self._t[i+1:i+1 + p+1],False)(x)) 
        else:
            f_c2 = lambda x : np.zeros_like(x)

        if(c1!=0):
            f_c1 = lambda x : (p / c1) * np.nan_to_num(scipy.interpolate.BSpline.basis_element(self._t[i:i + p + 1],False)(x))
        else:
            f_c1 = lambda x : np.zeros_like(x)

        return lambda x : np.nan_to_num(f_c1(x)) + np.nan_to_num(f_c2(x))

    def diff(self,deg, dorder):
        assert dorder==1, "not implemented for HO derivatives"
        
        b_deriv = self.derivative(deg,1)
        return lambda l,x : b_deriv(x) if deg>=(l) else x**(l-deg) *b_deriv(x) + self.Pn(deg)(0,x) * (l-deg) * x**(l-deg-1)
        #return lambda l,x : b_deriv(x) if l==0 else x**(l) *b_deriv(x) + self.Pn(deg)(0,x) * (l) * x**(l-1)



def uniform_simpson(domain, pts):
    assert pts%2==1, "simpson requires even number of intervals"
    assert pts>1   , "composite simpson require more than 1"
    gx = np.linspace(domain[0],domain[1],pts)
    dx = (domain[1]-domain[0])/(pts-1)
    gw = np.ones_like(gx) * (dx/3.0)
    gw[range(2,pts-1,2)]  *= 2.0
    gw[range(1,pts,2)]    *= 4.0
    return gx,gw



