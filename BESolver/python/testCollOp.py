"""
simple cases to test the collision operator, 
compute the collision operator, for known expression 
for cross-section and numerically validate the result
"""
from re import S

import basis
import numpy as np
import math
import spec_spherical as sp
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
import matplotlib.pyplot as plt

NUM_Q_VR     = params.BEVelocitySpace.NUM_Q_VR
NUM_Q_VT     = params.BEVelocitySpace.NUM_Q_VT
NUM_Q_VP     = params.BEVelocitySpace.NUM_Q_VP
NUM_Q_CHI    = params.BEVelocitySpace.NUM_Q_CHI
NUM_Q_PHI    = params.BEVelocitySpace.NUM_Q_PHI

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
    #Mp_r  = maxwellian(Sd[0]/V_TH) * ( (scattering_mg[0]*V_TH)**3) * ((V_TH)/weight_func(scattering_mg[0]))
    Mp_r  = np.exp((Sd[0]/V_TH)**2 - (scattering_mg[0])**2) *(scattering_mg[0]*V_TH)
    
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


maxwellian = lambda x: (collisions.MAXWELLIAN_N / ((collisions.ELECTRON_THEMAL_VEL * np.sqrt(np.pi))**3) ) * np.exp(-x**2)
g0  = collisions.eAr_G0()
g1  = collisions.eAr_G1()
g2  = collisions.eAr_G1()



def collision_op_test():
    params.BEVelocitySpace.NUM_Q_VR    = 10
    params.BEVelocitySpace.NUM_Q_VT    = 5
    params.BEVelocitySpace.NUM_Q_VP    = 2
    params.BEVelocitySpace.NUM_Q_CHI   = 10
    params.BEVelocitySpace.NUM_Q_PHI   = 2
    
    cf_sp    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)
    colOpSp.SPEC_SPHERICAL  = sp.SpectralExpansionSpherical(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,basis.Maxwell(),params.BEVelocitySpace.SPH_HARM_LM) 
    spec_sp  = colOpSp.SPEC_SPHERICAL

    t1 = time.time()
    L_loop = Lp_l(g0,maxwellian) - Lm_l(g0,maxwellian)
    t2 = time.time()
    print("loop based (s) : %.10f" %(t2-t1))

    t1 = time.time()
    L_np   = Lp(g0,maxwellian) - Lm(g0,maxwellian)
    t2 = time.time()
    print("np(tensor) based (s) : %.10f" %(t2-t1))
    print("np(tensor) based |L_np| : %.10f" %(np.linalg.norm(L_np)))
    print("loop based |L_loop|     : %.10f" %(np.linalg.norm(L_loop)))
    print("loop based |L_np - L_loop|       : %.10f" %(np.linalg.norm(L_loop-L_np)))
    print("Relative diff (L_np - L_loop)/|L_loop|: ")
    print((L_np-L_loop)/np.linalg.norm(L_loop))


def collision_op_conv():
    params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER=4
    params.BEVelocitySpace.SPH_HARM_LM = [ [0,0], [1,0]]
    params.BEVelocitySpace.NUM_Q_VR    = 20
    params.BEVelocitySpace.NUM_Q_VT    = 10
    params.BEVelocitySpace.NUM_Q_VP    = 10
    params.BEVelocitySpace.NUM_Q_CHI   = 2
    params.BEVelocitySpace.NUM_Q_PHI   = 10

    L0=Lp(g0,maxwellian)
    for i in range(5):
        print("VR %d VT %d VP %d CHI %d PHI %d" %(params.BEVelocitySpace.NUM_Q_VR,params.BEVelocitySpace.NUM_Q_VT,params.BEVelocitySpace.NUM_Q_VP,params.BEVelocitySpace.NUM_Q_CHI,params.BEVelocitySpace.NUM_Q_PHI))
        params.BEVelocitySpace.NUM_Q_VR                   = min(20,2 * params.BEVelocitySpace.NUM_Q_VR)
        #params.BEVelocitySpace.NUM_Q_VT                  = 2 * params.BEVelocitySpace.NUM_Q_VT
        #params.BEVelocitySpace.NUM_Q_VP                  = 2 * params.BEVelocitySpace.NUM_Q_VP
        params.BEVelocitySpace.NUM_Q_CHI                  = 2 * params.BEVelocitySpace.NUM_Q_CHI
        #params.BEVelocitySpace.NUM_Q_PHI                 = 2 * params.BEVelocitySpace.NUM_Q_PHI

        L1=Lp(g0,maxwellian)
        print("L0")
        print(L0)
        print("L1")
        print(L1)
        print("np.norm(L1-L0) : %.10f"%np.linalg.norm(L1-L0))
        L0=L1

    print("Refine on scattering azimuthal angle")
    params.BEVelocitySpace.NUM_Q_PHI                 = 2 * params.BEVelocitySpace.NUM_Q_PHI
    print("VR %d VT %d VP %d CHI %d PHI %d" %(params.BEVelocitySpace.NUM_Q_VR,params.BEVelocitySpace.NUM_Q_VT,params.BEVelocitySpace.NUM_Q_VP,params.BEVelocitySpace.NUM_Q_CHI,params.BEVelocitySpace.NUM_Q_PHI))
    L1=Lp(g0,maxwellian)
    print("np.norm(L1-L0) : %.10f"%np.linalg.norm(L1-L0))
    
    L0=L1
    print("Refine on V polar angle")
    params.BEVelocitySpace.NUM_Q_PHI                 = params.BEVelocitySpace.NUM_Q_PHI//2
    params.BEVelocitySpace.NUM_Q_VT                  = 2 * params.BEVelocitySpace.NUM_Q_VT
    print("VR %d VT %d VP %d CHI %d PHI %d" %(params.BEVelocitySpace.NUM_Q_VR,params.BEVelocitySpace.NUM_Q_VT,params.BEVelocitySpace.NUM_Q_VP,params.BEVelocitySpace.NUM_Q_CHI,params.BEVelocitySpace.NUM_Q_PHI))
    L1=Lp(g0,maxwellian)
    print("np.norm(L1-L0) : %.10f"%np.linalg.norm(L1-L0))


    L0=L1
    print("Refine on V azimuthal angle")
    params.BEVelocitySpace.NUM_Q_VT                 = params.BEVelocitySpace.NUM_Q_VT//2
    params.BEVelocitySpace.NUM_Q_VP                  = 2 * params.BEVelocitySpace.NUM_Q_VP
    print("VR %d VT %d VP %d CHI %d PHI %d" %(params.BEVelocitySpace.NUM_Q_VR,params.BEVelocitySpace.NUM_Q_VT,params.BEVelocitySpace.NUM_Q_VP,params.BEVelocitySpace.NUM_Q_CHI,params.BEVelocitySpace.NUM_Q_PHI))
    L1=Lp(g0,maxwellian)
    print("np.norm(L1-L0) : %.10f"%np.linalg.norm(L1-L0))




    


def analytical_time_integration(collision,maxwellian,h_init):

    params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER=4
    params.BEVelocitySpace.SPH_HARM_LM = [ [0,0], [1,0],[1,1]]
    params.BEVelocitySpace.NUM_Q_VR    = 20
    params.BEVelocitySpace.NUM_Q_VT    = 10
    params.BEVelocitySpace.NUM_Q_VP    = 10
    params.BEVelocitySpace.NUM_Q_CHI   = 64
    params.BEVelocitySpace.NUM_Q_PHI   = 10
    


    colOpSp.SPEC_SPHERICAL  = sp.SpectralExpansionSpherical(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,basis.Maxwell(),params.BEVelocitySpace.SPH_HARM_LM) 
    cf_sp    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)
    spec_sp  = colOpSp.SPEC_SPHERICAL

    sph_modes = params.BEVelocitySpace.SPH_HARM_LM
    num_p     = spec_sp._p +1
    num_sh    = len(params.BEVelocitySpace.SPH_HARM_LM)

    L    = cf_sp.assemble_mat(collision,maxwellian)
    M    = spec_sp.compute_maxwellian_mm(collision,maxwellian)
    FOp  = np.matmul(np.linalg.inv(M),L)
    W,Q  = np.linalg.eig(FOp)
    print(W)
    Qinv = np.linalg.inv(Q)
    #W    = np.diag(W)
    #FOp1 = np.matmul(np.matmul(Q,np.diag(W)), Qinv)
    #print(np.linalg.norm(FOp-FOp1))
    #y_init = np.dot(Qinv,h_init)

    # plt.plot(np.diag(W))
    # plt.xlabel("klm mode")
    # plt.ylabel("eigen value")
    # plt.show()

    h_init    = np.zeros(spec_sp.get_num_coefficients())
    h_init[0] = 1.0
    y_init    = np.dot(Qinv,h_init)
    
    #print([y_init[i] * Q[:,i] for i in range(num_p*num_sh)])
    # print(Q)
    # print(Q[:,0])
    num_pts = 80
    X = np.linspace(-5,5,num_pts)
    Y = np.linspace(-5,5,num_pts)
    Z = np.array([0])

    X,Y,Z=np.meshgrid(X,Y,Z,indexing='ij')

    R     = np.sqrt(X**2 + Y**2 + Z**2)
    THETA = np.arccos(Z/R)
    PHI   = np.arctan(Y/X)  

    P_klm = np.array([np.exp(-R**2)*spec_sp.basis_eval_full(R,THETA,PHI,k,sph_modes[lm_i][0],sph_modes[lm_i][1]) for k in range(num_p) for lm_i in range(num_sh)])
    #print(P_klm.shape)
    P_klm = P_klm.reshape(num_p*num_sh,-1)
    P_klm = np.transpose(P_klm)
    #print(P_klm.shape)
    #print(P_klm[:,3])

    fun_sol = np.array([y_init[i] * Q[:,i] for i in range(num_p*num_sh)])
    #print(fun_sol)
    fun_sol = np.transpose(fun_sol)

    print("check if original initial condition can be recovered: ")
    print(fun_sol.sum(axis=1))

    fun_sol_on_g = np.matmul(P_klm,fun_sol)
    fun_sol_on_g = np.transpose(fun_sol_on_g)


    #print(fun_sol)
    # fun_sol_on_g = [np.dot(P_klm,fun_sol[i]) for ]
    # fun_sol_on_g = np.transpose(fun_sol_on_g)
    # fun_sol_on_g = fun_sol_on_g.reshape(num_p*num_sh,num_pts,num_pts)
    #print(fun_sol_on_g)
    #print(fun_sol_on_g.shape)

    fig, axs = plt.subplots(num_p, num_sh)
    fig.set_size_inches(6,16)
    #fig.set_size_inches(3.5*num_sh, 5*num_p)
    for pk in range(num_p):
        for lm_i,lm in enumerate(sph_modes):
            im=axs[pk, lm_i].imshow(fun_sol_on_g[pk*num_sh + lm_i].reshape(num_pts,num_pts))
            plt.colorbar(im, ax=axs[pk, lm_i])
            axs[pk, lm_i].set_title("P_%d Ylm[%d, %d], eig : %.5E" %(pk,lm[0],lm[1],W[pk*num_sh + lm_i]),fontsize = 6.0)
            axs[pk, lm_i].tick_params(axis='both', which='major', labelsize=8)
    plt.tight_layout()
    plt.show()
    fig.savefig("fun_sol")
    
    # print(y_init)
    # print(Qinv[:,0])
    # T      = 1e-2
    # YT     = y_init*np.exp(np.diag(W)*T)
    # HT     = np.dot(Q,YT)
    # print(HT)










    
    #print(FOp -np.transpose(FOp))
    
    # print(FOp)
    # print(np.transpose(FOp))
    

    # print(np.linalg.cond(FOp))
    # D,U = np.linalg.eig(FOp)

    # h_init    = np.zeros(spec_sp.get_num_coefficients())
    # h_init[0] = 1.0

    # print(np.allclose(np.matmul(np.matmul(U,np.diag(D)),np.linalg.inv(U)),FOp))
    # print(np.linalg.norm(np.matmul(np.matmul(U,np.diag(D)),np.transpose(U))-FOp))
    # print(FOp)

    # print(np.linalg.norm(FOp-np.transpose(FOp)))
    # # print(W)

    # # #print(W)
    # # print(np.abs(U))
    # plt.plot(np.abs(D))
    # plt.show()

    # X= U * np.exp(W*T)
    # print(X)

    




#collision_op_test()
collision_op_conv()
analytical_time_integration(g0,maxwellian,0)





    

