import numpy as np

# Coefficients of Maxwell polynomials up to 20th power
# organized as [a00, a10, a11, a20, a21, a22, ... ], that is 
# coefficients of n-degree polynomial are given by max_poly_coeffs[n*(n+1)/2]

maxpoly_nmax    = 20 # max degree allowed
maxpoly_data    = np.genfromtxt('polynomials/maxpoly_upto20.dat',delimiter=',')
maxpoly_coeffs  = maxpoly_data[:,0]
maxpoly_nodes   = maxpoly_data[:,1]
maxpoly_weights = maxpoly_data[:,2]

def idx_s(p):
    return int(p*(p+1)/2)

def idx_e(p):
    return int(p*(p+1)/2+p)

def maxpolygauss(p):
    return [maxpoly_nodes[idx_s(p):idx_e(p)+1], maxpoly_weights[idx_s(p):idx_e(p)+1]]

def maxpolyeval(x, p):
    result = maxpoly_coeffs[idx_e(p)]
    for i in range(p):
        result = result*x + maxpoly_coeffs[idx_e(p)-i-1]
    return result

maxpolyweight = lambda x: 4/np.sqrt(np.pi)*x**2*np.exp(-x**2)

# quick and dirty 
# probably should reimplement this using ABCPolyBase
class basis:
    """
    Class for a 'Maxwell' polynomial of a given degree
    """
    def __init__(self, deg):
        """
        @param deg : degree of polynomial
        """
        self._deg = deg

    def __call__(self, x):
        return maxpolyeval(x, self._deg)