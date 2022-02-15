from math import factorial
import numpy as np
import scipy as scipy
from maxpoly import *

lmax = 33
Nr_max = 33
CpsiDphiP = np.zeros([lmax,Nr_max+1,Nr_max+1])
CphiDpsiP = np.zeros([lmax,Nr_max+1,Nr_max+1])
CpsiDphiM = np.zeros([lmax,Nr_max+1,Nr_max+1])
CphiDpsiM = np.zeros([lmax,Nr_max+1,Nr_max+1])

norm = lambda k,alpha: scipy.special.gamma(k+alpha+1)/factorial(max(k,0))
lagp = lambda k,alpha,x2: scipy.special.eval_genlaguerre(k, alpha, x2)*np.sqrt(2./norm(k,alpha))
xlagp_der = lambda k,alpha,x2: -2.*x2*scipy.special.eval_genlaguerre(k-1, alpha+1, x2)*np.sqrt(2./norm(k,alpha))

gauss_q_order_proj = 256
[xg, wg] = maxpolygauss(gauss_q_order_proj)

xg2 = xg**2

for q in range(lmax):
    
    for p in range(Nr_max+1):
        subintegrand = q*lagp(p, q+.5, xg2) + xlagp_der(p, q+.5, xg2)
        for k in range(Nr_max+1):
            integrand = xg**(2*q)*lagp(k, q+1.5, xg2)*subintegrand
            CphiDpsiP[q,p,k] = sum(integrand*wg)

            if q > 0:
                integrand = xg**(2*q-2)*lagp(k, q-0.5, xg2)*subintegrand
                CphiDpsiM[q,p,k] = sum(integrand*wg)

    for k in range(Nr_max+1):
        subintegrand = ((q+1)-2*xg2)*lagp(k,q+1.5,xg2) + xlagp_der(k,q+1.5,xg2)
        for p in range(Nr_max+1):
            integrand = xg**(2*q)*lagp(p,q+.5,xg2)*subintegrand
            CpsiDphiP[q,p,k] = sum(integrand*wg)

        if q > 0:
            subintegrand = ((q-1)-2*xg2)*lagp(k,q-.5,xg2) + xlagp_der(k,q-.5,xg2)
            for p in range(Nr_max+1):
                integrand = xg**(2*q-2)*lagp(p,q+.5,xg2)*subintegrand
                CpsiDphiM[q,p,k] = sum(integrand*wg)

    print(q)

np.save('polynomials/lagpoly_CpsiDphiP.npy', CpsiDphiP)
np.save('polynomials/lagpoly_CpsiDphiM.npy', CpsiDphiM)
np.save('polynomials/lagpoly_CphiDpsiP.npy', CphiDpsiP)
np.save('polynomials/lagpoly_CphiDpsiM.npy', CphiDpsiM)