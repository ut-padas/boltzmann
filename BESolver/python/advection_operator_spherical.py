# import abc
# from BESolver.python.maxpoly import diff_matrix, lift_matrix
import basis
import spec_spherical as sp
# import collisions
import scipy.integrate
import numpy as np
# import parameters as params
# import time
import maxpoly
import matplotlib.pyplot as plt

phimat = np.genfromtxt('sph_harm_del/phimat.dat',delimiter=',')
psimat = np.genfromtxt('sph_harm_del/psimat.dat',delimiter=',')

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
    
Nr = 20
lmax = 5
Ntotal = (Nr+1)*(lmax+1)**2

lm_all = []

for l in range(lmax+1):
    for m in range(-l,l+1):
        lm_idx = l**2+1+l+m
        lm_all.append([l,m])

num_sph = len(lm_all)

coeffs = np.zeros(Ntotal)
coeffs[0] = 1

advmat = assemble_advection_matix(Nr, lm_all)

print(np.shape(advmat))

func = lambda t,a: -np.matmul(advmat,a)

t_end = 1e-2
sol = scipy.integrate.solve_ivp(func, (0,t_end), coeffs)

print(coeffs)
print(sol.y[:,-1])

coeffs_new = sol.y[:,-1]

maxpolybasis = basis.Maxwell()
sph_basis = sp.SpectralExpansionSpherical(Nr, maxpolybasis, lm_all)

x = np.linspace(0,3,100)
f = np.zeros(np.shape(x))
f_ex = np.zeros(np.shape(x))

for k in range(Nr+1):
    for lm_idx, lm in enumerate(lm_all):
        f += coeffs_new[k*num_sph+lm_idx]*sph_basis.basis_eval_spherical(0,0, lm[0], lm[1])*maxpoly.maxpolyeval(x,k)*np.exp(-x**2)
        f_ex += coeffs[k*num_sph+lm_idx]*sph_basis.basis_eval_spherical(0,0, lm[0], lm[1])*maxpoly.maxpolyeval(x-t_end,k)*np.exp(-(x-t_end)**2)

plt.plot(x,f)
plt.plot(x,f_ex)
plt.show()