import numpy as np
from maxpoly_frac import *
import matplotlib.pyplot as plt

lmax = 3
Nr_max = 140
CpsiDphiP = np.zeros([lmax,Nr_max+1,Nr_max+1])
CphiDpsiP = np.zeros([lmax,Nr_max+1,Nr_max+1])
CpsiDphiM = np.zeros([lmax,Nr_max+1,Nr_max+1])
CphiDpsiM = np.zeros([lmax,Nr_max+1,Nr_max+1])

for q in range(lmax):
    if q > 0:
        diff_mat0 = diff_matrix(q-0.5, Nr_max+1)
    diff_mat1 = diff_matrix(q+0.5, Nr_max+1)
    diff_mat2 = diff_matrix(q+1.5, Nr_max+1)

    for p in range(Nr_max+1):
        # for k in range(max(0,p-2),Nr_max+1):
        for k in range(Nr_max+1):
            [xg, wg] = maxpolygauss(int((p+k)/2)+1,q+0.5)
            subintegrand = q*maxpolyeval(q+0.5, xg, p) + 2*xg*maxpolyserieseval(q+0.5, xg, diff_mat1[:,p])
            integrand = maxpolyeval(q+1.5, xg, k)*subintegrand
            CphiDpsiP[q,p,k] = np.dot(integrand,wg)

        if q > 0:
            # for k in range(0,p+1):
            for k in range(Nr_max+1):
                [xg, wg] = maxpolygauss(int((p+k)/2)+1,q-0.5)
                subintegrand = q*maxpolyeval(q+0.5, xg, p) + 2*xg*maxpolyserieseval(q+0.5, xg, diff_mat1[:,p])
                integrand = maxpolyeval(q-0.5, xg, k)*subintegrand
                CphiDpsiM[q,p,k] = np.dot(integrand,wg)

    for k in range(Nr_max+1):
        # for p in range(min(Nr_max,k+2)+1):
        for p in range(Nr_max+1):
            [xg, wg] = maxpolygauss(int(2+(p+k)/2)+1, q+0.5)
            subintegrand = ((q+1)-4*xg**2)*maxpolyeval(q+1.5, xg, k) + 2*xg*maxpolyserieseval(q+1.5, xg, diff_mat2[:,k])
            maxpoly_2q2_p = maxpolyeval(q+0.5, xg, p)
            integrand = maxpoly_2q2_p*subintegrand
            CpsiDphiP[q,p,k] = np.dot(integrand,wg)

        if q > 0:
            # for p in range(k, Nr_max+1):
            for p in range(Nr_max+1):
                [xg, wg] = maxpolygauss(int((p+k)/2)+1, q-0.5)
                subintegrand = ((q-1)-4*xg**2)*maxpolyeval(q-0.5, xg, k) + 2*xg*maxpolyserieseval(q-0.5, xg, diff_mat0[:,k])
                maxpoly_2q2_p = maxpolyeval(q+0.5, xg, p)
                integrand = maxpoly_2q2_p*subintegrand
                CpsiDphiM[q,p,k] = np.dot(integrand,wg)

    print(q)

np.save('polynomials/maxpoly_frac_CpsiDphiP.npy', CpsiDphiP)
np.save('polynomials/maxpoly_frac_CpsiDphiM.npy', CpsiDphiM)
np.save('polynomials/maxpoly_frac_CphiDpsiP.npy', CphiDpsiP)
np.save('polynomials/maxpoly_frac_CphiDpsiM.npy', CphiDpsiM)