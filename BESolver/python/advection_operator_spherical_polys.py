import numpy as np
import math

def sph_harm_norm(l,m):
    if m == 0:
        return np.sqrt((2.*l+1)/(4.*np.pi))
    else:
        return (-1)**m * np.sqrt((2.*l+1)/(2.*np.pi)*math.factorial(l-abs(m))/math.factorial(l+abs(m)))

AM = lambda l,m: (l+abs(m))/(2.*l+1.)*sph_harm_norm(l,m)/sph_harm_norm(l-1,m)
BM = lambda l,m: (l-abs(m)+1.)/(2.*l+1.)*sph_harm_norm(l,m)/sph_harm_norm(l+1,m)

AD = lambda l,m: (l+abs(m))*(l+1.)/(2.*l+1.)*sph_harm_norm(l,m)/sph_harm_norm(l-1,m)
BD = lambda l,m: -l*(l-abs(m)+1.)/(2.*l+1.)*sph_harm_norm(l,m)/sph_harm_norm(l+1,m)

CpsiDphiP_lag = np.load('polynomials/lagpoly_CpsiDphiP.npy')
CpsiDphiM_lag = np.load('polynomials/lagpoly_CpsiDphiM.npy')
CphiDpsiP_lag = np.load('polynomials/lagpoly_CphiDpsiP.npy')
CphiDpsiM_lag = np.load('polynomials/lagpoly_CphiDpsiM.npy')

CpsiDphiP_max = np.load('polynomials/maxpoly_CpsiDphiP.npy')
CpsiDphiM_max = np.load('polynomials/maxpoly_CpsiDphiM.npy')
CphiDpsiP_max = np.load('polynomials/maxpoly_CphiDpsiP.npy')
CphiDpsiM_max = np.load('polynomials/maxpoly_CphiDpsiM.npy')

def assemble_advection_matix_lp(Nr, sph_harm_lm, CpsiDphiP, CphiDpsiP, CpsiDphiM, CphiDpsiM):

    num_sh = len(sph_harm_lm)
    num_total = (Nr+1)*num_sh
    adv_mat = np.zeros([num_total, num_total])

    for qs_idx,qs in enumerate(sph_harm_lm):

        lm = [qs[0]+1,qs[1]]
        if lm in sph_harm_lm:
            lm_idx = sph_harm_lm.index(lm)

            for p in range(Nr+1):
                for k in range(Nr+1):
                    klm = k*num_sh + lm_idx
                    pqs = p*num_sh + qs_idx

                    val = (AM(lm[0],lm[1]) - .5*AD(lm[0],lm[1]))*CpsiDphiP[qs[0],p,k] - .5*AD(lm[0],lm[1])*CphiDpsiP[qs[0],p,k]

                    adv_mat[pqs,klm] += (0 if abs(val) < 1e-30 else val)
        
        lm = [qs[0]-1,qs[1]]
        if lm in sph_harm_lm:
            lm_idx = sph_harm_lm.index(lm)

            for p in range(Nr+1):
                for k in range(Nr+1):
                    klm = k*num_sh + lm_idx
                    pqs = p*num_sh + qs_idx

                    val = (BM(lm[0],lm[1]) - .5*BD(lm[0],lm[1]))*CpsiDphiM[qs[0],p,k] - .5*BD(lm[0],lm[1])*CphiDpsiM[qs[0],p,k]

                    adv_mat[pqs,klm] += (0 if abs(val) < 1e-30 else val)

    return adv_mat

def assemble_advection_matix_lp_max(Nr, sph_harm_lm):

    return assemble_advection_matix_lp(Nr, sph_harm_lm, CpsiDphiP_max, CphiDpsiP_max, CpsiDphiM_max, CphiDpsiM_max)

def assemble_advection_matix_lp_lag(Nr, sph_harm_lm):

    return assemble_advection_matix_lp(Nr, sph_harm_lm, CpsiDphiP_lag, CphiDpsiP_lag, CpsiDphiM_lag, CphiDpsiM_lag)