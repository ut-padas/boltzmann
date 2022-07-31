"""
@package: basis functions utilities
"""

from random import uniform
import numpy as np
import enum
import abc
import maxpoly
import maxpoly_frac
import lagpoly
import scipy.interpolate
import scipy
import scipy.special
import quadpy

# some parameters related to splines. 
XLBSPLINE_NUM_Q_PTS_PER_KNOT   = 3
BSPLINE_NUM_Q_PTS_PER_KNOT     = 3
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
    MAXWELLIAN_ENERGY_POLY=7
    CHEBYSHEV_POLY=8


class Basis(abc.ABC):
    abc.abstractmethod
    def __init__(self, domain, window):
        pass
    
    abc.abstractmethod
    def Pn(self,deg):
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
    def __init__(self, domain=None, window=None):
        self._basis_type = BasisType.HERMITE_E_POLY
        self._domain     = domain
        self._window     = window

    def Pn(self,deg):
        """
        returns 1d He polynomial the specified degree (normalized probabilist's)
        """
        return np.polynomial.hermite_e.HermiteE.basis(deg)
    
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
    def __init__(self,domain=None, window=None):
        self._basis_type = BasisType.HERMITE_POLY
        self._domain     = domain
        self._window     = window

    def Pn(self,deg):
        """
        returns 1d Hn polynomial the specified degree 
        """
        return np.polynomial.hermite.Hermite.basis(deg,self._domain, self._window)
    
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
    def __init__(self, domain=None, window=None):
        self._basis_type = BasisType.LEGENDRE
        self._domain     = domain
        self._window     = window

    def Pn(self,deg,domain=None,window=None):
        """
        returns 1d He polynomial the specified degree 
        """
        return np.polynomial.legendre.Legendre.basis(deg,self._domain, self._window)
        
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

class Chebyshev(Basis):
    def __init__(self,domain=(-1,1), window=(-1,1)):
        self._basis_type = BasisType.CHEBYSHEV_POLY
        self._domain     = domain
        self._window     = window
        assert (self._domain[0],self._domain[1]) == (-1,1), "Chebushev polynomial domain should match [-1,1]"

    def Pn(self,deg):
        """
        returns 1d He polynomial the specified degree 
        """
        (a,b) = (self._window[0], self._window[1])
        return lambda x, l : scipy.special.eval_chebyt(deg, ((x - 0.5 * (a+b))/(0.5 * (b-a)))) 
        
    def Gauss_Pn(self,deg):
        """
        Gauss-legendre quadrature 1d points for specified degree.
        """
        gx,gw  = scipy.special.roots_chebyt(deg, mu=False)
        (a,b)  = (self._window[0], self._window[1])
        gx     = (gx * (b-a) * 0.5) +  0.5 * (a+b)  
        gw     = gw  * (b-a) * 0.5
        
        return gx,gw

    def Wx(self):
        """
        Weight function w.r.t. the polynomials are orthogonal
        """
        (a,b)  = (self._window[0], self._window[1])
        return lambda x : (1.0/np.polynomial.chebyshev.chebweight((x - 0.5 * (a+b))/(0.5 * (b-a)))) * (x**2) * np.exp(-x**2)
        

    def diff(self,deg, dorder,domain=None,window=None):
        (a,b) = (self._window[0], self._window[1])
        return lambda x, l : deg * scipy.special.eval_chebyu(deg-1, ((x - 0.5 * (a+b))/(0.5 * (b-a)))) / (0.5 * (b-a))
        

class Maxwell(Basis):
    def __init__(self, domain=None, window=None):
        self._basis_type = BasisType.MAXWELLIAN_POLY
        self._domain     = domain
        self._window     = window

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
        
class MaxwellEnergy(Basis):
    def __init__(self, domain=None, window=None):
        self._basis_type = BasisType.MAXWELLIAN_ENERGY_POLY
        self._domain     = domain
        self._window     = window

    def Pn(self,deg,domain=None,window=None):
        """
        returns 1d Maxwell polynomial the specified degree 
        """
        return maxpoly_frac.basis(deg)
    
    def Gauss_Pn(self,deg):
        """
        Quadrature points and the corresponding weights for 1d Gauss quadrature. 
        The specified quadrature is exact to poly degree <= 2*degree-1, over [0,inf] domain
        """
        [gmx,gmw] = maxpoly_frac.maxpolygauss(max(1,deg-1))
        return [np.sqrt(gmx), .5*gmw]

    def Wx(self):
        """
        Weight function w.r.t. the polynomials are orthogonal
        """
        return maxpoly.maxpolyweight
        
class Laguerre(Basis):
    def __init__(self, domain=None, window=None):
        self._basis_type = BasisType.LAGUERRE
        self._domain     = domain
        self._window     = window

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
        return pts_per_knot*(p_order)

    def __init__(self,k_domain,spline_order, num_p):
        self._basis_type = BasisType.SPLINES
        self._domain     = k_domain
        self._window     = k_domain
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

    def Pn(self,deg):
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

    @staticmethod
    def uniform_knots(k_domain, num_p, sp_order):
        """
        x x x     | * * * * * * * * * * | x  x  x
         sp + 1          num_p -2          sp+1   
        """
        num_k      = 2*sp_order + (num_p -(sp_order+1)) + 2
        t          = (k_domain[0])*np.ones(sp_order)
        t          = np.append(t,np.linspace(k_domain[0] , k_domain[1] , num_k-2*sp_order -1 , endpoint=False))
        t          = np.append(t, k_domain[1]*np.ones(sp_order+1))
        assert len(t)==num_k , "knot length of %d does not match with spline order %d"%(len(t),sp_order)
        return t

    @staticmethod
    def uniform_dg_knots(k_domain, num_p, sp_order, dg_pts):
        if dg_pts is None:
            return XlBSpline.uniform_knots(k_domain, num_p, sp_order)

        _dg_pts = dg_pts[np.logical_and(dg_pts>k_domain[0], dg_pts<k_domain[1])]
        if len(_dg_pts)==0:
            return XlBSpline.uniform_knots(k_domain, num_p, sp_order)
        
        dg_domains = [(k_domain[0], _dg_pts[0])]
        dg_domains.extend([(dg_pts[i-1], dg_pts[i]) for i in range(1, len(_dg_pts))])
        dg_domains.append((_dg_pts[-1], k_domain[1]))
        
        num_d      = len(dg_domains)
        _num_p     = np.ones(num_d,dtype=int) * ((num_p) //num_d) #np.array(,dtype=int)
        #_num_p     = np.array([max(2*sp_order + 2, int(np.floor(num_p*(dg_domains[i][1]-dg_domains[i][0])/(k_domain[1]-k_domain[0])))) for i in range(0,num_d)])
        _num_p[-1] = num_p-np.sum(_num_p[0:-1])

        # if(_num_p[-1]<2*sp_order):
        #     _num_p[-2]-=2*sp_order
        #     _num_p[-1]+=2*sp_order

        print("domains : ", dg_domains)
        print("nr: ", _num_p)
        assert (_num_p>0).all(), "invalide radial polynomial partition"
        
        t = XlBSpline.uniform_knots(dg_domains[0], _num_p[0], sp_order)
        for i in range(1, num_d):
            t=np.append(t, XlBSpline.uniform_knots(dg_domains[i], _num_p[i], sp_order)[sp_order+1:])
        
        #print(t)
        num_k      = np.sum(np.array([(2*sp_order + (_num_p[i] -(sp_order+1)) + 2) for i in range(num_d)])) - (num_d-1)*(sp_order+1)
        assert len(t)==num_k , "knot length of %d does not match with spline order %d"%(len(t),sp_order)
        return t
    

    def __init__(self,k_domain,spline_order, num_p, sig_pts=None, knots_vec=None):
        self._basis_type = BasisType.SPLINES
        self._domain     = k_domain
        self._window     = k_domain
        self._kdomain    = (k_domain[0], k_domain[1])
        
        # first and last splines have repeated knots, 
        num_k            = 2*spline_order + (num_p -(spline_order+1)) + 2
        knot_base        = 2

        if knots_vec is None:
            self._t      = XlBSpline.uniform_dg_knots(k_domain, num_p, spline_order, sig_pts)
        else:
            self._t = np.copy(knots_vec)
            self._kdomain    = (self._t[0], self._t[-1])
        
        self._num_c_pts  = num_p
        self._sp_order   = spline_order
        self._q_per_knot = XLBSPLINE_NUM_Q_PTS_PER_KNOT
        self._splines    = [scipy.interpolate.BSpline.basis_element(self._t[i:i+spline_order+2],False) for i in range(num_p)]

        self._dg_idx     = list()
        for i in range(num_p):
            #print("i", i , "uq", np.unique(self._t[i:i+spline_order+2]), "k", self._t[i:i+spline_order+2])
            if len(np.unique(self._t[i:i+spline_order+2])) == 2:
                self._dg_idx.append(i)

        
        x_bdy = np.unique(np.array([self._t[self._dg_idx[i] + spline_order] for i in range(len(self._dg_idx))])) 
        self._dg_domains = [(x_bdy[i], x_bdy[i+1]) for i in range(len(x_bdy)-1)]

        self._num_knot_intervals=0
        for i in range(len(self._t)-1):
            if self._t[i] != self._t[i+1]:
                self._num_knot_intervals+=1

        #print("bsplines knots:")
        #print(self._t)
        #print(self._num_knot_intervals)
        print("dg basis idx ", self._dg_idx)
        print("dg domains   ", self._dg_domains)
        self._scheme     = quadpy.c1.gauss_legendre(self._q_per_knot)
        
    def Pn(self,deg):
        return lambda x,l : np.nan_to_num(self._splines[deg](x))
        #return lambda x,l : np.nan_to_num(self._splines[deg](x)) if deg>=l else np.nan_to_num(self._splines[deg](x)) * x**(l-deg)
        
    def Gauss_Pn(self,deg,from_zero=True):
        """
        Quadrature points and the corresponding weights for 1d Gauss quadrature. 
        The specified quadrature is exact to poly degree <= 2*degree-1, over [0,inf] domain
        """
        # num_intervals = self._num_c_pts -1
        # assert deg % num_intervals == 0, "specified # quadrature points %d not evenly divided by the number of control points %d" %(deg,num_intervals)
        # qx = np.zeros(deg)
        # qw = np.zeros(deg)
        # for i in range(self._sp_order, self._sp_order + num_intervals):
        #     qx[(i-self._sp_order)*self._q_per_knot: (i-self._sp_order)*self._q_per_knot + self._q_per_knot], qw[(i-self._sp_order)*self._q_per_knot: (i-self._sp_order)*self._q_per_knot + self._q_per_knot] = 0.5 * (self._t[i+1] - self._t[i]) * self._scheme.points + 0.5 * (self._t[i+1] + self._t[i]), 0.5 * (self._t[i+1]-self._t[i]) * self._scheme.weights 
        
        # assert np.allclose(np.sum(qw),(self._t[-1]-self._t[0]), rtol=1e-14, atol=1e-14), "weights computed for splines does not match the knots domain"
        # return qx,qw
        qx=np.array([])
        qw=np.array([])
        for i in range(len(self._t)-1):
            if self._t[i] != self._t[i+1]:
                qx = np.append(qx, 0.5 * (self._t[i+1] - self._t[i]) * self._scheme.points + 0.5 * (self._t[i+1] + self._t[i]))
                qw = np.append(qw, 0.5 * (self._t[i+1]-self._t[i]) * self._scheme.weights)

        assert deg == len(qx), "qx len %d is not the requested %d"%(len(qx), deg)
        assert np.allclose(np.sum(qw),(self._t[-1]-self._t[0]), rtol=1e-14, atol=1e-14), "weights computed for splines does not match the knots domain"
        return qx,qw
    
    def Wx(self):
        """
        Weight function w.r.t. the polynomials are orthogonal
        """
        I = lambda x: np.ones_like(x)
        return I

    def derivative(self, deg, dorder):
        if(dorder > 1):
            raise NotImplementedError
        
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
        if(dorder > 1):
            raise NotImplementedError
        
        b_deriv = self.derivative(deg,1)
        return lambda x,l : b_deriv(x)
        #return lambda x, l: b_deriv(x) if l==0 else x**(l) *b_deriv(x) + self.Pn(deg)(x,0) * (l) * x**(l-1)
        #return lambda x,l : b_deriv(x) if deg>=l else x**(l-deg) *b_deriv(x) + self.Pn(deg)(x,0) * (l-deg) * x**(l-deg-1)


