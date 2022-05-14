import numpy as np
from maxpoly import *
import matplotlib.pyplot as plt

lmax = 31
Nr_max = 260
CpsiDphiP = np.zeros([lmax,Nr_max+1,Nr_max+1])
CphiDpsiP = np.zeros([lmax,Nr_max+1,Nr_max+1])
CpsiDphiM = np.zeros([lmax,Nr_max+1,Nr_max+1])
CphiDpsiM = np.zeros([lmax,Nr_max+1,Nr_max+1])

# gauss_q_order_proj = 150
# [xg, wg] = maxpolygauss(gauss_q_order_proj)

# for q in range(lmax):
#     diff_mat0 = diff_matrix(2*q, Nr_max)
#     diff_mat1 = diff_matrix(2*(q+1), Nr_max)
#     diff_mat2 = diff_matrix(2*(q+2), Nr_max)

#     plt.semilogy(abs(diff_mat2[0,:]))
#     plt.show()

#     for p in range(Nr_max+1):
#         subintegrand = q*maxpolyeval(2*q+2, xg, p) + xg*maxpolyserieseval(2*q+2, xg, diff_mat1[:,p])
#         for k in range(max(0,p-2),Nr_max+1):
#             integrand = xg**(2*q)*maxpolyeval(2*q+4, xg, k)*subintegrand
#             CphiDpsiP[q,p,k] = np.dot(integrand,wg)

#         for k in range(0,p+1):
#             if q > 0:
#                 integrand = xg**(2*q-2)*maxpolyeval(2*q, xg, k)*subintegrand
#                 CphiDpsiM[q,p,k] = np.dot(integrand,wg)

#     for k in range(Nr_max+1):
#         subintegrand = ((q+1)-2*xg**2)*maxpolyeval(2*q+4, xg, k)*0 + xg*maxpolyserieseval(2*q+4, xg, diff_mat2[0:k,k])
#         for p in range(min(Nr_max,k+2)+1):
#             maxpoly_2q2_p = maxpolyeval(2*q+2, xg, p)
#             # integrand = (xg**(2*q))*maxpoly_2q2_p*subintegrand
#             integrand = (xg**(2*q+1))*subintegrand
#             CpsiDphiP[q,p,k] = diff_mat2[0,k]

#         if q > 0:
#             subintegrand = ((q-1)-2*xg**2)*maxpolyeval(2*q, xg, k) + xg*maxpolyserieseval(2*q, xg, diff_mat0[:,k])
#             for p in range(k, Nr_max+1):
#                 maxpoly_2q2_p = maxpolyeval(2*q+2, xg, p)
#                 integrand = xg**(2*q-2)*maxpoly_2q2_p*subintegrand
#                 CpsiDphiM[q,p,k] = np.dot(integrand,wg)

#     print(q)


# gauss_q_order_proj = 70
# [xg, wg] = maxpolygauss(gauss_q_order_proj)

for q in range(lmax):
    diff_mat0 = diff_matrix(2*q, Nr_max)
    diff_mat1 = diff_matrix(2*(q+1), Nr_max)
    diff_mat2 = diff_matrix(2*(q+2), Nr_max)

    for p in range(Nr_max+1):
        for k in range(max(0,p-2),Nr_max+1):
            [xg, wg] = maxpolygauss(int((p+k)/2)+1,2*q+2)
            subintegrand = q*maxpolyeval(2*q+2, xg, p) + xg*maxpolyserieseval(2*q+2, xg, diff_mat1[:,p])
            integrand = maxpolyeval(2*q+4, xg, k)*subintegrand
            CphiDpsiP[q,p,k] = np.dot(integrand,wg)

        if q > 0:
            for k in range(0,p+1):
                [xg, wg] = maxpolygauss(int((p+k)/2)+1,2*q)
                subintegrand = q*maxpolyeval(2*q+2, xg, p) + xg*maxpolyserieseval(2*q+2, xg, diff_mat1[:,p])
                integrand = maxpolyeval(2*q, xg, k)*subintegrand
                CphiDpsiM[q,p,k] = np.dot(integrand,wg)

    for k in range(Nr_max+1):
        for p in range(min(Nr_max,k+2)+1):
            [xg, wg] = maxpolygauss(int(2+(p+k)/2)+1, 2*q+2)
            subintegrand = ((q+1)-2*xg**2)*maxpolyeval(2*q+4, xg, k) + xg*maxpolyserieseval(2*q+4, xg, diff_mat2[:,k])
            maxpoly_2q2_p = maxpolyeval(2*q+2, xg, p)
            integrand = maxpoly_2q2_p*subintegrand
            CpsiDphiP[q,p,k] = np.dot(integrand,wg)

        if q > 0:
            for p in range(k, Nr_max+1):
                [xg, wg] = maxpolygauss(int((p+k)/2)+1,2*q)
                subintegrand = ((q-1)-2*xg**2)*maxpolyeval(2*q, xg, k) + xg*maxpolyserieseval(2*q, xg, diff_mat0[:,k])
                maxpoly_2q2_p = maxpolyeval(2*q+2, xg, p)
                integrand = maxpoly_2q2_p*subintegrand
                CpsiDphiM[q,p,k] = np.dot(integrand,wg)

    print(q)

np.save('polynomials/maxpoly_CpsiDphiP.npy', CpsiDphiP)
np.save('polynomials/maxpoly_CpsiDphiM.npy', CpsiDphiM)
np.save('polynomials/maxpoly_CphiDpsiP.npy', CphiDpsiP)
np.save('polynomials/maxpoly_CphiDpsiM.npy', CphiDpsiM)