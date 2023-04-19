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
#import quadpy
import math
import matplotlib.pyplot as plt

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


def gauss_legendre_quad(deg, a, b):
    q_pts   = np.polynomial.legendre.leggauss(deg) #quadpy.c1.gauss_legendre(q_per_knot)
    qx      = q_pts[0]
    qw      = q_pts[1]

    qx      = 0.5 * (b - a) * qx + 0.5 * (a + b)
    qw      = 0.5 * (b - a) * qw

    return qx, qw

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
    
    def __init__(self, k_domain, spline_order, num_p, q_per_knot=None , sig_pts=None, knots_vec=None, dg_splines=False):
        self._basis_type = BasisType.SPLINES
        self._domain     = k_domain
        self._window     = k_domain
        self._kdomain    = (k_domain[0], k_domain[1])
        self._sp_order   = spline_order
        
        if knots_vec is None:
            if dg_splines:
                self._t , self._ele, self._ele_p  = BSpline.uniform_dg_knots(k_domain, num_p, spline_order)
            else:
                self._t     = BSpline.uniform_knots(k_domain, num_p, spline_order)
                #self._t      = BSpline.uniform_knots_with_extended_bdy(k_domain, num_p, spline_order, ext_kdomain = 2 * k_domain[1])
                self._ele    = None
                self._ele_p  = None 

                if sig_pts is not None:
                    for i in range(len(sig_pts)):
                        if sig_pts[i] > k_domain[0] and sig_pts[i] < k_domain[1]:
                            s   = sig_pts[i]
                            idx = np.argmin( np.abs(self._t - s))
                            if (idx > spline_order+1) and (idx < len(self._t)-spline_order-1):
                                self._t[idx] = s
            
            self._num_p   = num_p
            

        else:
            if sig_pts is not None and False:
                ii              = list(knots_vec).index(sig_pts[0])
                self._t         = np.append(np.ones(self._sp_order) * knots_vec[0], knots_vec[0:ii+1])
                self._t         = np.append(self._t, np.ones(self._sp_order) * knots_vec[ii])
                self._t         = np.append(self._t, knots_vec[ii+1:])
                self._t         = np.append(self._t, np.ones(self._sp_order) * knots_vec[-1])
            else:
                self._t         = np.append(np.ones(self._sp_order) * knots_vec[0], knots_vec)
                self._t         = np.append(self._t, np.ones(self._sp_order) * knots_vec[-1])
            self._kdomain   = (self._t[0], self._t[-1])
            self._num_p     = len(self._t) - (self._sp_order+1)
            
        self._splines            = [scipy.interpolate.BSpline.basis_element(self._t[i:i+spline_order+2],False) for i in range(self._num_p)]
        self._t_unique           = np.unique(self._t)
        self._num_knot_intervals = len(self._t_unique)

        if self._ele is not None:
            self._dg_idx    = list()
            ele_offsets     = np.zeros_like(self._ele_p)
            ele_offsets[1:] = np.cumsum(self._ele_p)[0:-1]
            for i in range(len(self._ele_p)):
                self._dg_idx.append(0 + ele_offsets[i])
                self._dg_idx.append(self._ele_p[i]-1 + ele_offsets[i])
        else:
            self._dg_idx    = [0, num_p-1]

        print("num_p \n ", self._ele_p)
        print("dg node indices\n", self._dg_idx)
        print("knots vector\n", self._t)
        print("knots vector (unique)\n", self._t_unique)
        # print(self._t)
        # for i in range(self._num_p):
        #     print("spline ", i , self._t[i:i+spline_order+2])
        #self.plot()
        
    def Pn(self,deg):
        return lambda x,l : np.nan_to_num(self._splines[deg](x))
        
    def Gauss_Pn(self,deg):
        q_per_knot    = deg // self._num_knot_intervals
        self._q_pts   = np.polynomial.legendre.leggauss(q_per_knot) #quadpy.c1.gauss_legendre(q_per_knot)
        self._q_pts_points = self._q_pts[0]
        self._q_pts_weights = self._q_pts[1]

        qx=np.array([])
        qw=np.array([])
        for i in range(len(self._t)-1):
            if self._t[i] != self._t[i+1]:
                qx = np.append(qx, 0.5 * (self._t[i+1] - self._t[i]) * self._q_pts_points + 0.5 * (self._t[i+1] + self._t[i]))
                qw = np.append(qw, 0.5 * (self._t[i+1]-self._t[i])   * self._q_pts_weights)

        #assert deg == len(qx), "qx len %d is not the requested %d"%(len(qx), deg)
        assert np.allclose(np.sum(qw),(self._t[-1]-self._t[0]), rtol=1e-14, atol=1e-14), "weights computed for splines does not match the knots domain"
        return qx,qw
    
    def Gauss_Pn_gl(self,deg):
        """
        Quadrature points with gauss lobatto points. 
        """
        q_per_knot    = deg // self._num_knot_intervals
        self._q_pts   = np.polynomial.legendre.leggauss(q_per_knot) #quadpy.c1.gauss_legendre(q_per_knot)
        #self._q_pts   = quadpy.c1.gauss_lobatto(q_per_knot)
        assert False, "not supported"
        self._q_pts_points = self._q_pts[0]
        self._q_pts_weights = self._q_pts[1]

        qx=np.array([])
        qw=np.array([])
        for i in range(len(self._t)-1):
            if self._t[i] != self._t[i+1]:
                qx = np.append(qx, 0.5 * (self._t[i+1] - self._t[i]) * self._q_pts_points + 0.5 * (self._t[i+1] + self._t[i]))
                qw = np.append(qw, 0.5 * (self._t[i+1]-self._t[i]) * self._q_pts_weights)

        #assert deg == len(qx), "qx len %d is not the requested %d"%(len(qx), deg)
        assert np.allclose(np.sum(qw),(self._t[-1]-self._t[0]), rtol=1e-14, atol=1e-14), "weights computed for splines does not match the knots domain"
        return qx,qw
    
    def Wx(self):
        """
        Weight function w.r.t. the polynomials are orthogonal
        """
        I = lambda x: np.ones_like(x)
        return I

    def derivative(self, deg, dorder):
        i=deg
        p=self._sp_order
        f1 = math.factorial(p)/math.factorial(p-dorder)

        def f_alpha(k,j):
            if k==0 and j==0:
                return 1
            elif j==0:
                d1 = self._t[i+p-k+1] - self._t[i]
                if d1!=0:
                    return f_alpha(k-1,0) / d1
                else:
                    return 0
            elif j>0 and j<k:
                d1 = self._t[i+p+j-k+1] - self._t[i+j]
                if d1!=0:
                    return (f_alpha(k-1,j)- f_alpha(k-1,j-1))/d1
                else:
                    return 0
            elif k==j:
                d1 = self._t[i+p+1] - self._t[i+k]
                if d1!=0:
                    return -f_alpha(k-1,k-1)/d1
                else:
                    return 0
            else:
                raise NotImplementedError

        def deriv_spline(x):
            y=0
            for j in range(0,dorder+1):
                #print(i+j, "to",  (i+j) + (p-dorder) + 2, self._t[i+j: (i+j) + (p-dorder) + 2], f_alpha(dorder,j))
                y+=f_alpha(dorder,j) * np.nan_to_num(scipy.interpolate.BSpline.basis_element(self._t[i+j: (i+j) + (p-dorder) + 2],False)(x))
            
            return y*f1
        
        return deriv_spline

    def diff(self,deg, dorder):
        b_deriv = self.derivative(deg,dorder)
        return lambda x,l : b_deriv(x)

    def plot(self):
        dd = self._kdomain
        x  = np.linspace(dd[0], dd[1], self._num_p * 100)
        for i in range(self._num_p):
            plt.plot(x, self.Pn(i)(x,0), label="b_%d"%(i), linewidth=2)

        plt.legend()
        plt.grid()
        plt.show()
        plt.close()

    def fit(self, f):
        num_p    = self._num_p
        sp_order = self._sp_order 
        m_mat    = np.zeros((num_p, num_p))
        gx, gw   = self.Gauss_Pn( (self._sp_order+1) * self._num_knot_intervals)
        
        for i in range(num_p):
            for j in range(num_p):#(max(0, i-sp_order-1), min(num_p, i + sp_order+1)):
                m_mat[i,j] = np.dot(gw, self.Pn(i)(gx,0) * self.Pn(j)(gx,0))

        t_mat, q_mat = scipy.linalg.schur(m_mat)
        t_mat_inv    = scipy.linalg.solve_triangular(t_mat, np.identity(m_mat.shape[0]),lower=False)
        m_mat_inv    = np.matmul(np.linalg.inv(np.transpose(q_mat)), np.matmul(t_mat_inv, np.linalg.inv(q_mat))) 
        
        # l_mat      = np.linalg.cholesky(m_mat)
        # l_inv_mat  = scipy.linalg.solve_triangular(l_mat, np.identity(m_mat.shape[0]),lower=True)
        # m_mat_inv  = np.matmul(np.transpose(l_inv_mat),l_inv_mat)
        f_vec      = np.array([np.dot(gw, f(gx) * self.Pn(i)(gx,0)) for i in range(num_p)])
        assert np.allclose(np.matmul(m_mat_inv,m_mat),np.eye(m_mat.shape[0]), rtol=1e-12, atol=1e-12), "mass mat inverse failed with %.2E rtol"%(1e-12)
        return np.dot(m_mat_inv,f_vec)

    @staticmethod
    def adaptive_fit(f, k_domain, rtol=1e-12, atol=1e-12, sp_order=4, min_lev=3, max_lev=10, sig_pts=np.array([])):
        t0 = k_domain[0]
        t1 = k_domain[1]

        ele_old    = []
        ele_new    = []

        if sig_pts is None:
            sp    = BSpline((t0, t1), sp_order, sp_order+1)
            ele_old.append(sp)
        else:
            ss    = sig_pts[sig_pts>t0 and sig_pts<t1]
            if len(ss)>0:
                sp    = BSpline((t0, ss[0]), sp_order, sp_order+1)
                ele_old.append(sp)
                for i in range(1, len(ss)):
                    sp    = BSpline((ss[i-1], ss[i]), sp_order, sp_order+1)
                    ele_old.append(sp)
                
                sp    = BSpline((ss[-1], t1), sp_order, sp_order+1)
                ele_old.append(sp)
            else:
                sp    = BSpline((t0, t1), sp_order, sp_order+1)
                ele_old.append(sp)
                

        is_refine  = True
        refine_lev = int(np.log2(len(ele_old)))

        while is_refine and refine_lev < max_lev:
            is_refine=False
            for sp in ele_old:
                t0   = sp._kdomain[0]
                t1   = sp._kdomain[1]

                fc       = sp.fit(f)
                tt , _   = sp.Gauss_Pn((sp_order+1) * 8)
                f1       = np.sum(np.array([fc[i] * sp.Pn(i)(tt,0) for i in range(sp._num_p)]),axis=0)
                f2       = f(tt)
                aerror    = np.max(np.abs(f2 -f1))
                rerror   = aerror/np.max(np.abs(f2))

                
                if aerror  > atol and rerror >rtol :#(atol + rtol * np.max(np.abs(f2))) or refine_lev < min_lev:
                    #print("ele = (%f, %f)  split (%f,%f) and (%f, %f) "  %(t0,t1, t0, 0.5 * (t0 + t1), 0.5 * (t0 + t1), t1))
                    ele_new.append(BSpline((t0, 0.5 * (t0+t1)), sp_order, sp_order+1))
                    ele_new.append(BSpline((0.5 * (t0+t1), t1), sp_order, sp_order+1))
                    is_refine=True
                else:
                    ele_new.append(BSpline((t0, t1), sp_order, sp_order+1))

            
            ele_old = ele_new
            ele_new = list()
            refine_lev += 1
            

        tt = np.array([])
        for sp in ele_old:
            tt=np.append(tt, np.unique(sp._t))
        
        tt = np.sort(np.unique(tt))
        assert tt[0] == k_domain[0]
        return tt
        
    @staticmethod
    def uniform_knots(k_domain, num_p, sp_order):
        # t          = (k_domain[0])*np.ones(sp_order+1)
        # glx, _        = Legendre().Gauss_Pn(num_p-sp_order-1)
        # # Np         = num_p-sp_order-1 + 2
        # # glx        = -np.cos(np.pi*np.linspace(0,Np-1,Np)/(Np-1))
        # # glx        = glx[1:-1]
        # glx        = glx * (k_domain[1]-k_domain[0]) * 0.5 + (k_domain[1] + k_domain[0]) * 0.5
        # t          = np.append(t,glx)
        # t          = np.append(t, k_domain[1]*np.ones(sp_order+1))

        t          = (k_domain[0])*np.ones(sp_order)
        t          = np.append(t,np.linspace(k_domain[0] , k_domain[1] , num_p-sp_order , endpoint=False))
        t          = np.append(t, k_domain[1]*np.ones(sp_order+1))
        return t

    @staticmethod
    def uniform_knots_with_extended_bdy(k_domain, num_p, sp_order, ext_kdomain=10.0):
        # t          = (k_domain[0])*np.ones(sp_order+1)
        # glx, _        = Legendre().Gauss_Pn(num_p-sp_order-1)
        # # Np         = num_p-sp_order-1 + 2
        # # glx        = -np.cos(np.pi*np.linspace(0,Np-1,Np)/(Np-1))
        # # glx        = glx[1:-1]
        # glx        = glx * (k_domain[1]-k_domain[0]) * 0.5 + (k_domain[1] + k_domain[0]) * 0.5
        # t          = np.append(t,glx)
        # t          = np.append(t, k_domain[1]*np.ones(sp_order+1))

        num_p1     = 3 * (num_p //4)
        num_p2     = num_p -num_p1

        t          = (k_domain[0])*np.ones(sp_order)
        t          = np.append(t , np.linspace(k_domain[0] , k_domain[1] , num_p1 , endpoint=False))
        t          = np.append(t , np.logspace(np.log10(k_domain[1]), np.log10(ext_kdomain), num_p2-sp_order,endpoint=False))
        t          = np.append(t , ext_kdomain * np.ones(sp_order+1))

        print(len(t), num_p+sp_order+1)
        return t
    
    @staticmethod
    def uniform_dg_knots(k_domain, num_p, sp_order):
        
        dg_pts = np.linspace(k_domain[0], k_domain[1], num_p//(sp_order+1) + 1)[1:-1]

        dg_domains = [(k_domain[0], dg_pts[0])]
        dg_domains.extend([(dg_pts[i-1], dg_pts[i]) for i in range(1, len(dg_pts))])
        dg_domains.append((dg_pts[-1], k_domain[1]))
        
        num_d      = len(dg_domains)
        _num_p     = np.ones(num_d,dtype=int) * ((num_p) //num_d) 
        _num_p[-1] = num_p-np.sum(_num_p[0:-1])
        
        t = BSpline.uniform_knots(dg_domains[0], _num_p[0], sp_order)
        for i in range(1, num_d):
            t=np.append(t, BSpline.uniform_knots(dg_domains[i], _num_p[i], sp_order)[sp_order+1:])
        
        num_k          = np.sum(np.array([(2*sp_order + (_num_p[i] -(sp_order+1)) + 2) for i in range(num_d)])) - (num_d-1)*(sp_order+1)
        assert len(t) == num_k , "knot length of %d does not match with spline order %d"%(len(t),sp_order)
        return t, dg_domains, _num_p

    @staticmethod
    def uniform_dg_knots_1(k_domain, num_p, sp_order, dg_pts):
        if dg_pts is None:
            return BSpline.uniform_knots(k_domain, num_p, sp_order)

        _dg_pts = dg_pts[np.logical_and(dg_pts>k_domain[0], dg_pts<k_domain[1])]
        if len(_dg_pts)==0:
            return BSpline.uniform_knots(k_domain, num_p, sp_order)
        
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
        
        t = BSpline.uniform_knots(dg_domains[0], _num_p[0], sp_order)
        for i in range(1, num_d):
            t=np.append(t, BSpline.uniform_knots(dg_domains[i], _num_p[i], sp_order)[sp_order+1:])
        
        #print(t)
        num_k      = np.sum(np.array([(2*sp_order + (_num_p[i] -(sp_order+1)) + 2) for i in range(num_d)])) - (num_d-1)*(sp_order+1)
        assert len(t)==num_k , "knot length of %d does not match with spline order %d"%(len(t),sp_order)
        return t