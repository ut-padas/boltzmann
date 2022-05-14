import numpy as np
from scipy.special import gamma

# Coefficients of Maxwell polynomials up to 20th power
# organized as [a00, a10, a11, a20, a21, a22, ... ], that is 
# coefficients of n-degree polynomial are given by max_poly_coeffs[n*(n+1)/2]

maxpoly_nmax    = 299 # max degree allowed
# maxpoly_data    = np.genfromtxt('polynomials/maxpoly_upto555.dat',delimiter=',')
# maxpoly_coeffs  = maxpoly_data[:,0]
# maxpoly_nodes   = maxpoly_data[:,0]
# maxpoly_weights = maxpoly_data[:,1]

# maxpoly_rec_data = np.genfromtxt('polynomials/maxpoly_upto555_recursive.dat',delimiter=',')
# maxpoly_rec_a    = maxpoly_rec_data[:,0]
# maxpoly_rec_b    = maxpoly_rec_data[:,1]
# maxpoly_rec_n    = maxpoly_rec_data[:,2]

# maxpoly_alpha = np.genfromtxt('polynomials/maxpoly_alpha_upto300_new.dat',delimiter=',')
# maxpoly_beta = np.genfromtxt('polynomials/maxpoly_beta_upto300_new.dat',delimiter=',')

maxpoly_alpha = np.genfromtxt('polynomials/maxpoly_alpha_300_65.dat',delimiter=',')
maxpoly_beta = np.genfromtxt('polynomials/maxpoly_beta_300_65.dat',delimiter=',')
maxpoly_nodes   = np.genfromtxt('polynomials/maxpoly_nodes_300_65.dat',delimiter=',')
maxpoly_weights = np.genfromtxt('polynomials/maxpoly_weights_300_65.dat',delimiter=',')
maxpoly_deriv = np.genfromtxt('polynomials/maxpoly_deriv_300_65.dat',delimiter=',')

def idx_s(p):
    return int(p*(p+1)/2)

def idx_e(p):
    return int(p*(p+1)/2+p)

def maxpolygauss(p, G=2):
    return [maxpoly_nodes[G, idx_s(p):idx_e(p)+1], maxpoly_weights[G, idx_s(p):idx_e(p)+1]]

# def maxpolyeval_naive(x, p):

#     if p == 0:
#         return np.ones(np.shape(x))

#     result = maxpoly_coeffs[idx_e(p)]
#     for i in range(p):
#         result = result*x + maxpoly_coeffs[idx_e(p)-i-1]
#     return result

def maxpolyeval(G, x, p):

    idx = G

    g1 = gamma((1.+G)/2.)
    g2 = gamma((2.+G)/2.)
    g3 = gamma((3.+G)/2.)

    if p == 0:
        return np.ones(np.shape(x))*np.sqrt(2./g1)

    Bn   = 0
    Bnp1 = 0
    Bnp2 = 0
    for i in range(p,0,-1):
        Bnp2 = Bnp1
        Bnp1 = Bn
        if i == p:
            Bn = np.ones(np.shape(x)) + (x-maxpoly_alpha[idx,i])*Bnp1/np.sqrt(maxpoly_beta[idx,i+1]) - np.sqrt(maxpoly_beta[idx,i+1]/maxpoly_beta[idx,i+2])*Bnp2
        else:
            Bn = (x-maxpoly_alpha[idx,i])*Bnp1/np.sqrt(maxpoly_beta[idx,i+1]) - np.sqrt(maxpoly_beta[idx,i+1]/maxpoly_beta[idx,i+2])*Bnp2

    return Bn*(x-g2/g1)*np.sqrt(2)/np.sqrt(g3-g2**2/g1) - np.sqrt(maxpoly_beta[idx,1]/maxpoly_beta[idx,2])*Bnp1*np.sqrt(2./g1)

def maxpolyserieseval(G, x, coeffs):

    idx = G

    g1 = gamma((1.+G)/2.)
    g2 = gamma((2.+G)/2.)
    g3 = gamma((3.+G)/2.)

    p = len(coeffs)

    if p == 0:
        return 0

    if p == 1:
        return coeffs[0]*np.sqrt(2./g1)*np.ones(np.shape(x))

    Bn   = 0
    Bnp1 = 0
    Bnp2 = 0
    for i in range(p-1,0,-1):
        Bnp2 = Bnp1
        Bnp1 = Bn
        Bn = coeffs[i] + (x-maxpoly_alpha[idx,i])*Bnp1/np.sqrt(maxpoly_beta[idx,i+1]) - np.sqrt(maxpoly_beta[idx,i+1]/maxpoly_beta[idx,i+2])*Bnp2

    return coeffs[0]*np.sqrt(2./g1) + Bn*(x-g2/g1)*np.sqrt(2)/np.sqrt(g3-g2**2/g1) - np.sqrt(maxpoly_beta[idx,1]/maxpoly_beta[idx,2])*Bnp1*np.sqrt(2./g1)

# def maxpolyeval_non_norm(x, p):

#     if p == 0:
#         return np.ones(np.shape(x))

#     Bn   = 0
#     Bnp1 = 0
#     Bnp2 = 0
#     for i in range(p,0,-1):
#         Bnp2 = Bnp1
#         Bnp1 = Bn
#         if i == p:
#             Bn = np.ones(np.shape(x)) + (x-maxpoly_rec_a[i])*Bnp1 - maxpoly_rec_b[i+1]*Bnp2
#         else:
#             Bn = (x-maxpoly_rec_a[i])*Bnp1 - maxpoly_rec_b[i+1]*Bnp2

#     return (Bn*(x-2./np.sqrt(np.pi)) - maxpoly_rec_b[1]*Bnp1)/np.sqrt(maxpoly_rec_n[p])

maxpolyweight = lambda x: x**2*np.exp(-x**2)

# def diff_matrix(G, n):

#     idx = int(G/2)

#     D = np.zeros([n+1, n+1])

#     for j in range(n+1):

#         if j > 0:
#             D[j-1,j] = j/np.sqrt(maxpoly_beta[idx,j])

#         if j > 1:
#             D[j-2,j] = (sum(maxpoly_alpha[idx,0:j]) - j*maxpoly_alpha[idx,j-1])/np.sqrt(maxpoly_beta[idx,j]*maxpoly_beta[idx,j-1])

#         if j > 2:
#             D[j-3,j] = (2.*np.sqrt(maxpoly_beta[idx,j]*maxpoly_beta[idx,j-1]) - np.sqrt(maxpoly_beta[idx,j-1])*D[j-1,j] - maxpoly_alpha[idx,j-2]*D[j-2,j])/np.sqrt(maxpoly_beta[idx,j-2])

#         if j > 3:
#             for i in range(4,j+1):
#                 D[j-i,j] = - (np.sqrt(maxpoly_beta[idx,j+2-i])*D[j+2-i,j] + maxpoly_alpha[idx,j+1-i]*D[j+1-i,j])/np.sqrt(maxpoly_beta[idx,j+1-i])


#     return D


def diff_matrix(G, n):

    idx = G

    D = np.zeros([n+1, n+1])

    for j in range(n+1):
        for i in range(j):
            D[i,j] = maxpoly_deriv[idx, int((j-1)*j/2) + i]

    return D

def lift_matrix(G, n):

    idx = int(G/2)

    D = np.zeros([n+1, n+1])

    D[0,0] = maxpoly_alpha[idx,0]
    D[0,1] = np.sqrt(maxpoly_beta[idx,1])

    for j in range(1,n):

        D[j,j-1] = np.sqrt(maxpoly_beta[idx,j])
        D[j,j]   = maxpoly_alpha[idx,j]
        D[j,j+1] = np.sqrt(maxpoly_beta[idx,j+1])

    D[n,n-1] = np.sqrt(maxpoly_beta[idx,n])
    D[n,n] = maxpoly_alpha[idx,n]

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

    def __call__(self, x, l):
        return (x**l) * maxpolyeval(2*l+2, x, self._deg)