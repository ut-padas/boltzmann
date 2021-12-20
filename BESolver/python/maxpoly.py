import numpy as np

# Coefficients of Maxwell polynomials up to 20th power
# organized as [a00, a10, a11, a20, a21, a22, ... ], that is 
# coefficients of n-degree polynomial are given by max_poly_coeffs[n*(n+1)/2]

maxpoly_nmax    = 554 # max degree allowed
maxpoly_data    = np.genfromtxt('polynomials/maxpoly_upto555.dat',delimiter=',')
# maxpoly_coeffs  = maxpoly_data[:,0]
maxpoly_nodes   = maxpoly_data[:,0]
maxpoly_weights = maxpoly_data[:,1]

maxpoly_rec_data = np.genfromtxt('polynomials/maxpoly_upto555_recursive.dat',delimiter=',')
maxpoly_rec_a    = maxpoly_rec_data[:,0]
maxpoly_rec_b    = maxpoly_rec_data[:,1]
# maxpoly_rec_n    = maxpoly_rec_data[:,2]

def idx_s(p):
    return int(p*(p+1)/2)

def idx_e(p):
    return int(p*(p+1)/2+p)

def maxpolygauss(p):
    return [maxpoly_nodes[idx_s(p):idx_e(p)+1], maxpoly_weights[idx_s(p):idx_e(p)+1]]

def maxpolyeval_naive(x, p):

    if p == 0:
        return np.ones(np.shape(x))

    result = maxpoly_coeffs[idx_e(p)]
    for i in range(p):
        result = result*x + maxpoly_coeffs[idx_e(p)-i-1]
    return result

def maxpolyeval(x, p):

    if p == 0:
        return np.ones(np.shape(x))/np.sqrt(np.sqrt(np.pi)/4.)

    Bn   = 0
    Bnp1 = 0
    Bnp2 = 0
    for i in range(p,0,-1):
        Bnp2 = Bnp1
        Bnp1 = Bn
        if i == p:
            Bn = np.ones(np.shape(x)) + (x-maxpoly_rec_a[i])*Bnp1/np.sqrt(maxpoly_rec_b[i+1]) - np.sqrt(maxpoly_rec_b[i+1]/maxpoly_rec_b[i+2])*Bnp2
        else:
            Bn = (x-maxpoly_rec_a[i])*Bnp1/np.sqrt(maxpoly_rec_b[i+1]) - np.sqrt(maxpoly_rec_b[i+1]/maxpoly_rec_b[i+2])*Bnp2

    return Bn*(x-2./np.sqrt(np.pi))/np.sqrt(np.sqrt(np.pi)*(1.5-4./np.pi)/4.) - np.sqrt(maxpoly_rec_b[1]/maxpoly_rec_b[2])*Bnp1/np.sqrt(np.sqrt(np.pi)/4.)

def maxpolyeval_non_norm(x, p):

    if p == 0:
        return np.ones(np.shape(x))

    Bn   = 0
    Bnp1 = 0
    Bnp2 = 0
    for i in range(p,0,-1):
        Bnp2 = Bnp1
        Bnp1 = Bn
        if i == p:
            Bn = np.ones(np.shape(x)) + (x-maxpoly_rec_a[i])*Bnp1 - maxpoly_rec_b[i+1]*Bnp2
        else:
            Bn = (x-maxpoly_rec_a[i])*Bnp1 - maxpoly_rec_b[i+1]*Bnp2

    return (Bn*(x-2./np.sqrt(np.pi)) - maxpoly_rec_b[1]*Bnp1)/np.sqrt(maxpoly_rec_n[p])

maxpolyweight = lambda x: 4/np.sqrt(np.pi)*x**2*np.exp(-x**2)

def diff_matrix(n):

    D = np.zeros([n+1, n+1])

    for j in range(n+1):

        if j > 0:
            D[j-1,j] = j/np.sqrt(maxpoly_rec_b[j])

        if j > 1:
            D[j-2,j] = (sum(maxpoly_rec_a[0:j]) - j*maxpoly_rec_a[j-1])/np.sqrt(maxpoly_rec_b[j]*maxpoly_rec_b[j-1])

        if j > 2:
            D[j-3,j] = (2.*np.sqrt(maxpoly_rec_b[j]*maxpoly_rec_b[j-1]) - np.sqrt(maxpoly_rec_b[j-1])*D[j-1,j] - maxpoly_rec_a[j-2]*D[j-2,j])/np.sqrt(maxpoly_rec_b[j-2])

        if j > 3:
            for i in range(4,j+1):
                D[j-i,j] = - (np.sqrt(maxpoly_rec_b[j+2-i])*D[j+2-i,j] + maxpoly_rec_a[j+1-i]*D[j+1-i,j])/np.sqrt(maxpoly_rec_b[j+1-i])


    return D

def lift_matrix(n):

    D = np.zeros([n+1, n+1])

    D[0,0] = maxpoly_rec_a[0]
    D[0,1] = np.sqrt(maxpoly_rec_b[1])

    for j in range(1,n):

        D[j,j-1] = np.sqrt(maxpoly_rec_b[j])
        D[j,j]   = maxpoly_rec_a[j]
        D[j,j+1] = np.sqrt(maxpoly_rec_b[j+1])

    D[n,n-1] = np.sqrt(maxpoly_rec_b[n])
    D[n,n] = maxpoly_rec_a[n]

    return D

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