import numpy as np
from math import factorial
import scipy as scipy

maxpoly_nmax    = 554 # max degree allowed
maxpoly_data    = np.genfromtxt('polynomials/maxpoly_upto555.dat',delimiter=',')
maxpoly_nodes   = maxpoly_data[:,0]
maxpoly_weights = maxpoly_data[:,1]

norm = lambda k,alpha: scipy.special.gamma(k+alpha+1)/factorial(max(k,0))
lagp = lambda k,alpha,x2: scipy.special.eval_genlaguerre(k, alpha, x2)*np.sqrt(2./norm(k,alpha))
xlagp_der = lambda k,alpha,x2: -2.*x2*scipy.special.eval_genlaguerre(k-1, alpha+1, x2)*np.sqrt(2./norm(k,alpha))

def idx_s(p):
    return int(p*(p+1)/2)

def idx_e(p):
    return int(p*(p+1)/2+p)

# TODO: using maxwell based quadratures for now
def lagpolygauss(p):
    return [maxpoly_nodes[idx_s(p):idx_e(p)+1], maxpoly_weights[idx_s(p):idx_e(p)+1]]

# lagpolyweight = lambda x: 4/np.sqrt(np.pi)*x**2*np.exp(-x**2)
lagpolyweight = lambda x: x**2*np.exp(-x**2)

# quick and dirty 
# probably should reimplement this using ABCPolyBase
class basis:
    """
    Class for a 'Laguerre' polynomial of a given degree
    """
    def __init__(self, deg):
        """
        @param deg : degree of polynomial
        """
        self._deg = deg

    def __call__(self, x, l):
        return (x**l) * lagp(self._deg, l+0.5, x**2)
    
