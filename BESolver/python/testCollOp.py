"""
simple cases to test the collision operator, 
compute the collision operator, for known expression 
for cross-section and numerically validate the result
"""
from re import S

from numpy.lib.function_base import diff
import basis
import numpy as np
import math
import collision_operator_spherical as colOpSp
import collision_operator as colOp
import collisions 
import parameters as params
import visualize_utils
import ets 
import os
import utils as BEUtils
import maxpoly
import time

NUM_Q_VR     = params.BEVelocitySpace.NUM_Q_VR
NUM_Q_VT     = params.BEVelocitySpace.NUM_Q_VT
NUM_Q_VP     = params.BEVelocitySpace.NUM_Q_VP

NUM_Q_CHI    = params.BEVelocitySpace.NUM_Q_CHI
NUM_Q_PHI    = params.BEVelocitySpace.NUM_Q_PHI


maxwellian = lambda x: (collisions.MAXWELLIAN_N / ((collisions.ELECTRON_THEMAL_VEL * np.sqrt(np.pi))**3) ) * np.exp(-x**2)
cf_sp    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)
spec_sp  = colOpSp.SPEC_SPHERICAL

g0  = collisions.eAr_G0()
g1  = collisions.eAr_G1()
g2  = collisions.eAr_G1()

def Lm_l(collision,maxwellian):
    return colOpSp.CollisionOpSP._Lm_l(collision,maxwellian)

def Lp_l(collision,maxwellian):
    return colOpSp.CollisionOpSP._Lp_l(collision,maxwellian)

def Lm(collision,maxwellian):
    return colOpSp.CollisionOpSP._Lm(collision,maxwellian)

def Lp(collision: collisions.Collisions, maxwellian):
    return colOpSp.CollisionOpSP._Lp(collision,maxwellian)

def Lp_v1(collision: collisions.Collisions, maxwellian):
    V_TH         = collisions.ELECTRON_THEMAL_VEL
    ELE_VOLT     = collisions.ELECTRON_VOLT
    MAXWELLIAN_N = collisions.MAXWELLIAN_N
    AR_NEUTRAL_N = collisions.AR_NEUTRAL_N

    num_p        = spec_sp._p +1
    sph_harm_lm  = params.BEVelocitySpace.SPH_HARM_LM 
    num_sph_harm = len(sph_harm_lm)
    [gmx,gmw]    = maxpoly.maxpolygauss(NUM_Q_VR-1)
    weight_func  = maxpoly.maxpolyweight
    
    legendre     = basis.Legendre()
    [glx,glw]    = legendre.Gauss_Pn(NUM_Q_VT)
    VTheta_q      = np.arccos(glx)
    VPhi_q        = np.linspace(0,2*np.pi,NUM_Q_VP)

    [glx_s,glw_s] = legendre.Gauss_Pn(NUM_Q_CHI)
    Chi_q         = np.arccos(glw_s)
    Phi_q         = np.linspace(0,2*np.pi,NUM_Q_PHI)
    
    assert NUM_Q_VP>1
    assert NUM_Q_PHI>1
    sq_fac_v = (2*np.pi/(NUM_Q_VP-1))
    sq_fac_s = (2*np.pi/(NUM_Q_PHI-1))

    WPhi_q   = np.ones(NUM_Q_PHI)*sq_fac_s
    WVPhi_q  = np.ones(NUM_Q_VP)*sq_fac_v

    #trap. weights
    WPhi_q[0]  = 0.5 * WPhi_q[0]
    WPhi_q[-1] = 0.5 * WPhi_q[-1]

    WVPhi_q[0]  = 0.5 * WVPhi_q[0]
    WVPhi_q[-1] = 0.5 * WVPhi_q[-1]

    energy_ev = (0.5 * collisions.MASS_ELECTRON * ((gmx*V_TH)**2))/ELE_VOLT
    energy_ev = energy_ev.reshape(len(gmx),1)

    g=collision
    
    scattering_mg = np.meshgrid(gmx,VTheta_q,VPhi_q,Chi_q,Phi_q,indexing='ij')
    diff_cs       = g.assemble_diff_cs_mat(scattering_mg[0]*V_TH , scattering_mg[3]) * AR_NEUTRAL_N
    
    Sd    = g.compute_scattering_velocity_sp(scattering_mg[0]*V_TH,scattering_mg[1],scattering_mg[2],scattering_mg[3],scattering_mg[4])
    Pp_kr = spec_sp.Vq_r(Sd[0]/V_TH) 
    Yp_lm = spec_sp.Vq_sph(Sd[1],Sd[2])
    Mp_r  = maxwellian(Sd[0]/V_TH) * ( (scattering_mg[0]*V_TH)**3) * ((V_TH)/weight_func(scattering_mg[0]))
    
    num_p  = spec_sp._p+1
    num_sh = len(spec_sp._sph_harm_lm)

    # Ap_klm1 = np.zeros(tuple([num_p,num_sh]) + incident_mg[0].shape)
    # for i in range(num_p):
    #     for j in range(num_sh):
    #         Ap_klm1[i,j] = diff_cs * Mp_r * Pp_kr[i] * Yp_lm[j]

    Ap_klm = np.array([diff_cs * Mp_r* Pp_kr[i] * Yp_lm[j] for i in range(num_p) for j in range(num_sh)])
    Ap_klm = Ap_klm.reshape(tuple([num_p,num_sh]) + scattering_mg[0].shape)

    P_pr  = spec_sp.Vq_r(scattering_mg[0])
    Y_qs  = spec_sp.Vq_sph(scattering_mg[1],scattering_mg[2])

    B_qrs = np.array([P_pr[i] * Y_qs[j] for i in range(num_p) for j in range(num_sh)])
    B_qrs = B_qrs.reshape(tuple([num_p,num_sh]) + scattering_mg[0].shape)


    D_pqs_klm = np.array([Ap_klm[pi,li] * B_qrs[pj,lj] for pi in range(num_p) for li in range(num_sh) for pj in range(num_p) for lj in range(num_sh)])
    D_pqs_klm = D_pqs_klm.reshape(tuple([num_p,num_sh,num_p,num_sh]) + scattering_mg[0].shape)
    
    D_pqs_klm = np.dot(D_pqs_klm,WPhi_q)
    D_pqs_klm = np.dot(D_pqs_klm,glw_s)
    D_pqs_klm = np.dot(D_pqs_klm,WVPhi_q)
    D_pqs_klm = np.dot(D_pqs_klm,glw)
    D_pqs_klm = np.dot(D_pqs_klm,gmw)

    D_pqs_klm = D_pqs_klm.reshape((num_p*num_sh,num_p*num_sh))
    return D_pqs_klm


# t1 = time.time()
# Lm1 = Lm_l(g0,maxwellian)
# t2 = time.time()
# print("loop based : %.10f" %(t2-t1))
# t1 = time.time()
# Lm2 = Lm(g0,maxwellian)
# t2 = time.time()
# print("np based : %.10f" %(t2-t1))
# print(Lm1)
# print(Lm2)
# print(np.allclose((Lm2-Lm1)/np.max(Lm1),0))

# t1 = time.time()
# Lp1 = Lp_l(g0,maxwellian)
# t2 = time.time()
# print(Lp1)
# print("loop based : %.10f" %(t2-t1))

# t1 = time.time()
# Lp2 = Lp(g0,maxwellian)
# t2 = time.time()
# print("np based : %.10f" %(t2-t1))
# print((Lp2-Lp1)/np.max(Lp1))

params.BEVelocitySpace.NUM_Q_VR                   = 2
params.BEVelocitySpace.NUM_Q_VT                   = 10
params.BEVelocitySpace.NUM_Q_VP                   = 10
params.BEVelocitySpace.NUM_Q_CHI                  = 2
params.BEVelocitySpace.NUM_Q_PHI                  = 10

L0=Lp(g0,maxwellian)
for i in range(20):
    print("VR %d VT %d VP %d CHI %d PHI %d" %(params.BEVelocitySpace.NUM_Q_VR,params.BEVelocitySpace.NUM_Q_VT,params.BEVelocitySpace.NUM_Q_VP,params.BEVelocitySpace.NUM_Q_CHI,params.BEVelocitySpace.NUM_Q_PHI))
    params.BEVelocitySpace.NUM_Q_VR                   = min(20,2 * params.BEVelocitySpace.NUM_Q_VR)
    #params.BEVelocitySpace.NUM_Q_VT                   = 2 * params.BEVelocitySpace.NUM_Q_VT
    #params.BEVelocitySpace.NUM_Q_VP                   = 2 * params.BEVelocitySpace.NUM_Q_VP
    params.BEVelocitySpace.NUM_Q_CHI                  = 2 * params.BEVelocitySpace.NUM_Q_CHI
    #params.BEVelocitySpace.NUM_Q_PHI                  = 2 * params.BEVelocitySpace.NUM_Q_PHI

    L1=Lp(g0,maxwellian)
    print("L0")
    print(L0)

    print("L1")
    print(L1)
    print("(L1-L0)/L0")
    print((L1-L0)/L0)

    L0=L1

    


