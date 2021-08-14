import numpy as np
import matplotlib.pyplot as plt
import scipy.special

# Coefficients of Laguerre polynomials up to 20th power
# organized as [a00, a10, a11, a20, a21, a22, ... ], that is 
# coefficients of n-degree polynomial are given by max_poly_coeffs[n*(n+1)/2]

lagpoly_nmax    = 20 # max degree allowed
lagpoly_data    = np.genfromtxt('polynomials/lagpoly_upto20.dat',delimiter=',')
lagpoly_coeffs  = lagpoly_data[:,0]
lagpoly_nodes   = lagpoly_data[:,1]
lagpoly_weights = lagpoly_data[:,2]

def idx_s(p):
    return int(p*(p+1)/2)

def idx_e(p):
    return int(p*(p+1)/2+p)

def lagpolygauss(p):
    return [lagpoly_nodes[idx_s(p):idx_e(p)+1], lagpoly_weights[idx_s(p):idx_e(p)+1]]

def lagpolyeval(x, p):
    result = lagpoly_coeffs[idx_e(p)]
    for i in range(p):
        result = result*x + lagpoly_coeffs[idx_e(p)-i-1]
    return result
