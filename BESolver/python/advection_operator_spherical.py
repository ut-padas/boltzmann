"""
@package : Spectral based, Petrov-Galerkin discretization of the advection term
"""

# import abc
# from BESolver.python.maxpoly import diff_matrix, lift_matrix
# import basis
import spec_spherical as sp
# import collisions
# import scipy.constants
import numpy as np
# import parameters as params
# import time
import maxpoly

phimat = np.genfromtxt('sph_harm_del/phimat.dat',delimiter=',')
psimat = np.genfromtxt('sph_harm_del/phimat.dat',delimiter=',')

def assemble_advection_matix(Nr, sph_harm_lm):

    num_sh = len(sph_harm_lm)
    num_total = (Nr+1)*num_sh
    adv_mat = np.zeros([num_total, num_total])

    diff_mat = maxpoly.diff_matrix(Nr)
    lift_mat = maxpoly.lift_matrix(Nr)

    g_mat = diff_mat - 2*lift_mat
    
    for p in range(Nr+1):
        for k in range(Nr+1):
            for lm_idx,lm in enumerate(sph_harm_lm):
                for qs_idx,qs in enumerate(sph_harm_lm):
                    lm_mat = lm[0]**2+1+lm[0]+lm[1]
                    qs_mat = qs[0]**2+1+qs[0]+qs[1]
                    klm = k*num_sh + lm_idx
                    pqs = p*num_sh + qs_idx
                    adv_mat[pqs,klm] = g_mat[k,p]*psimat[lm_mat,qs_mat] + \
                        .5*(sum(g_mat[k,:]*g_mat[:,p])+sum(g_mat[k,:]*diff_mat[p,:])*phimat[lm_mat,qs_mat])

    return adv_mat
    
print(assemble_advection_matix(5, [[0,0], [1,-1], [1,0], [1,1]]))