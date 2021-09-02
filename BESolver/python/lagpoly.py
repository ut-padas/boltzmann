import numpy as np
import scipy.special

# Coefficients of Laguerre polynomials up to 20th power
# organized as [a00, a10, a11, a20, a21, a22, ... ], that is 
# coefficients of n-degree polynomial are given by max_poly_coeffs[n*(n+1)/2]

lagpoly_nmax    = 50 # max degree allowed
lagpoly_data    = np.genfromtxt('polynomials/lagpoly_upto50.dat',delimiter=',')
lagpoly_coeffs  = lagpoly_data[:,0]
lagpoly_nodes   = lagpoly_data[:,1]
lagpoly_weights = lagpoly_data[:,2]

lagpoly_rec_data = np.genfromtxt('polynomials/lagpoly_upto50_recursive.dat',delimiter=',')
lagpoly_rec_a    = lagpoly_rec_data[:,0]
lagpoly_rec_b    = lagpoly_rec_data[:,1]
lagpoly_rec_n    = lagpoly_rec_data[:,2]

def idx_s(p):
    return int(p*(p+1)/2)

def idx_e(p):
    return int(p*(p+1)/2+p)

def lagpolygauss(p):
    return [lagpoly_nodes[idx_s(p):idx_e(p)+1], lagpoly_weights[idx_s(p):idx_e(p)+1]]

def lagpolyeval_naive(x, p):

    if p == 0:
        return np.ones(np.shape(x))

    result = lagpoly_coeffs[idx_e(p)]
    for i in range(p):
        result = result*x + lagpoly_coeffs[idx_e(p)-i-1]
    return result

def lagpolyeval(x, p):

    if p == 0:
        return np.ones(np.shape(x))

    Bn   = 0
    Bnp1 = 0
    Bnp2 = 0
    for i in range(p,0,-1):
        Bnp2 = Bnp1
        Bnp1 = Bn
        if i == p:
            Bn = np.ones(np.shape(x)) + (x-lagpoly_rec_a[i])*Bnp1 - lagpoly_rec_b[i+1]*Bnp2
        else:
            Bn = (x-lagpoly_rec_a[i])*Bnp1 - lagpoly_rec_b[i+1]*Bnp2

    return (Bn*(x-3./2.) - lagpoly_rec_b[1]*Bnp1)/np.sqrt(lagpoly_rec_n[p])
    
