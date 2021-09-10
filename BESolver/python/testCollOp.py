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


NUM_Q_VR     = params.BEVelocitySpace.NUM_Q_VR
NUM_Q_VT     = params.BEVelocitySpace.NUM_Q_VT
NUM_Q_VP     = params.BEVelocitySpace.NUM_Q_VP
NUM_Q_CHI    = params.BEVelocitySpace.NUM_Q_CHI
NUM_Q_PHI    = params.BEVelocitySpace.NUM_Q_PHI

def Lm_l(collision,maxwellian, vth):
    return colOpSp.CollisionOpSP._Lm_l(collision,maxwellian,vth)

def Lp_l(collision,maxwellian, vth):
    return colOpSp.CollisionOpSP._Lp_l(collision,maxwellian, vth)

def Lm(collision,maxwellian, vth):
    return colOpSp.CollisionOpSP._Lm(collision,maxwellian, vth)

def Lp(collision: collisions.Collisions, maxwellian, vth):
    return colOpSp.CollisionOpSP._Lp(collision,maxwellian, vth)

VTH = collisions.ELECTRON_THEMAL_VEL
maxwellian = lambda x: (collisions.MAXWELLIAN_N / ((collisions.ELECTRON_THEMAL_VEL * np.sqrt(np.pi))**3) ) * np.exp(-x**2)
g0  = collisions.eAr_G0()
g1  = collisions.eAr_G1()
g2  = collisions.eAr_G1()


def plot_crosssection(collision: collisions.Collisions,num_q):
    #np.sqrt(2*BOLTZMANN_CONST*MAXWELLIAN_TEMP_K/MASS_ELECTRON)
    import matplotlib.pyplot as plt

    ELE_VOLT        = collisions.ELECTRON_VOLT
    BOLTZMANN_CONST = collisions.BOLTZMANN_CONST

    TEMP_K_1EV          = ELE_VOLT/BOLTZMANN_CONST
    g            = collision 
    ev   = collision._energy
    tcs  = collision._total_cs

    [gmx,gmw]    = maxpoly.maxpolygauss(num_q-1)
    print(gmx)
    weight_func  = maxpoly.maxpolyweight

    MAXWELLIAN_TEMP_K   = 5*TEMP_K_1EV
    V_TH                = np.sqrt(2*BOLTZMANN_CONST*MAXWELLIAN_TEMP_K/collisions.MASS_ELECTRON)
    energy_ev = (0.5 * collisions.MASS_ELECTRON * ((gmx*V_TH)**2))/ELE_VOLT
    tcs_q     = g.total_cross_section(energy_ev)

    # p1 = plt.plot(ev,tcs,'o-b')
    # p2 = plt.plot(energy_ev,tcs_q,'x-g')
    plt.plot(ev,tcs,'o-b',energy_ev,tcs_q,'x-g')
    plt.legend(["LXCAT","V_thermal=1EV"])

    # MAXWELLIAN_TEMP_K   = 5*TEMP_K_1EV
    # V_TH                = np.sqrt(2*BOLTZMANN_CONST*MAXWELLIAN_TEMP_K/collisions.MASS_ELECTRON)
    # energy_ev = (0.5 * collisions.MASS_ELECTRON * ((gmx*V_TH)**2))/ELE_VOLT
    # tcs_q     = g.total_cross_section(energy_ev)
    # p3 = plt.plot(energy_ev,tcs_q,'x-r',label='GMX-5eV')

    # MAXWELLIAN_TEMP_K   = 10*TEMP_K_1EV
    # V_TH                = np.sqrt(2*BOLTZMANN_CONST*MAXWELLIAN_TEMP_K/collisions.MASS_ELECTRON)
    # energy_ev = (0.5 * collisions.MASS_ELECTRON * ((gmx*V_TH)**2))/ELE_VOLT
    # tcs_q     = g.total_cross_section(energy_ev)
    # p4 =plt.plot(energy_ev,tcs_q,'x-y',label='GMX-10eV')

    # MAXWELLIAN_TEMP_K   = 20*TEMP_K_1EV
    # V_TH                = np.sqrt(2*BOLTZMANN_CONST*MAXWELLIAN_TEMP_K/collisions.MASS_ELECTRON)
    # energy_ev = (0.5 * collisions.MASS_ELECTRON * ((gmx*V_TH)**2))/ELE_VOLT
    # tcs_q     = g.total_cross_section(energy_ev)
    # p5 = plt.plot(energy_ev,tcs_q,'x-m',label='GMX-20eV')

    # plt.legend([p1,p2,p3,p4,p5],["lxcat",'GMX-1eV','GMX-5eV','GMX-10eV','GMX-20eV'])
    plt.xlabel("eV")
    plt.ylabel("total cs (m2)")
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


def collision_op_test():
    params.BEVelocitySpace.NUM_Q_VR    = 20
    params.BEVelocitySpace.NUM_Q_VT    = 5
    params.BEVelocitySpace.NUM_Q_VP    = 2
    params.BEVelocitySpace.NUM_Q_CHI   = 10
    params.BEVelocitySpace.NUM_Q_PHI   = 2
    
    cf_sp    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)
    colOpSp.SPEC_SPHERICAL  = sp.SpectralExpansionSpherical(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,basis.Maxwell(),params.BEVelocitySpace.SPH_HARM_LM) 
    spec_sp  = colOpSp.SPEC_SPHERICAL

    t1 = time.time()
    L_loop = Lp_l(g0,maxwellian,VTH) - Lm_l(g0,maxwellian,VTH)
    t2 = time.time()
    print("loop based (s) : %.10f" %(t2-t1))

    t1 = time.time()
    L_np   = Lp(g0,maxwellian,VTH) - Lm(g0,maxwellian,VTH)
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
    params.BEVelocitySpace.NUM_Q_VR    = 10
    params.BEVelocitySpace.NUM_Q_VT    = 10
    params.BEVelocitySpace.NUM_Q_VP    = 10
    params.BEVelocitySpace.NUM_Q_CHI   = 32
    params.BEVelocitySpace.NUM_Q_PHI   = 10

    colOpSp.SPEC_SPHERICAL  = sp.SpectralExpansionSpherical(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,basis.Maxwell(),params.BEVelocitySpace.SPH_HARM_LM) 
    cf_sp    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)
    spec_sp  = colOpSp.SPEC_SPHERICAL

    print("VR %d VT %d VP %d CHI %d PHI %d" %(params.BEVelocitySpace.NUM_Q_VR,params.BEVelocitySpace.NUM_Q_VT,params.BEVelocitySpace.NUM_Q_VP,params.BEVelocitySpace.NUM_Q_CHI,params.BEVelocitySpace.NUM_Q_PHI))
    L0=Lp(g0,maxwellian,VTH) - Lm(g0,maxwellian,VTH)
    for i in range(2):
        params.BEVelocitySpace.NUM_Q_VR                   = min(20,2 * params.BEVelocitySpace.NUM_Q_VR)
        #params.BEVelocitySpace.NUM_Q_VT                  = 2 * params.BEVelocitySpace.NUM_Q_VT
        #params.BEVelocitySpace.NUM_Q_VP                  = 2 * params.BEVelocitySpace.NUM_Q_VP
        params.BEVelocitySpace.NUM_Q_CHI                  = 2 * params.BEVelocitySpace.NUM_Q_CHI
        #params.BEVelocitySpace.NUM_Q_PHI                 = 2 * params.BEVelocitySpace.NUM_Q_PHI
        print("VR %d VT %d VP %d CHI %d PHI %d" %(params.BEVelocitySpace.NUM_Q_VR,params.BEVelocitySpace.NUM_Q_VT,params.BEVelocitySpace.NUM_Q_VP,params.BEVelocitySpace.NUM_Q_CHI,params.BEVelocitySpace.NUM_Q_PHI))
        L1=Lp(g0,maxwellian,VTH) - Lm(g0,maxwellian,VTH)
        print("L0")
        print(L0)
        print("L1")
        print(L1)
        print("np.norm(L1-L0) : %.10f"%np.linalg.norm(L1-L0))
        L0=L1

    L0=L1
    print("Refine on VR ")
    params.BEVelocitySpace.NUM_Q_VR                  = min(20,2 * params.BEVelocitySpace.NUM_Q_VR)
    print("VR %d VT %d VP %d CHI %d PHI %d" %(params.BEVelocitySpace.NUM_Q_VR,params.BEVelocitySpace.NUM_Q_VT,params.BEVelocitySpace.NUM_Q_VP,params.BEVelocitySpace.NUM_Q_CHI,params.BEVelocitySpace.NUM_Q_PHI))
    L1=Lp(g0,maxwellian,VTH) - Lm(g0,maxwellian,VTH)
    print("np.norm(L1-L0) : %.10f"%np.linalg.norm(L1-L0))


    L0=L1
    print("Refine on scattering azimuthal angle")
    params.BEVelocitySpace.NUM_Q_PHI                 = 2 * params.BEVelocitySpace.NUM_Q_PHI
    print("VR %d VT %d VP %d CHI %d PHI %d" %(params.BEVelocitySpace.NUM_Q_VR,params.BEVelocitySpace.NUM_Q_VT,params.BEVelocitySpace.NUM_Q_VP,params.BEVelocitySpace.NUM_Q_CHI,params.BEVelocitySpace.NUM_Q_PHI))
    L1=Lp(g0,maxwellian,VTH) - Lm(g0,maxwellian,VTH)
    print("np.norm(L1-L0) : %.10f"%np.linalg.norm(L1-L0))
    
    L0=L1
    print("Refine on V polar angle")
    params.BEVelocitySpace.NUM_Q_PHI                 = params.BEVelocitySpace.NUM_Q_PHI//2
    params.BEVelocitySpace.NUM_Q_VT                  = 2 * params.BEVelocitySpace.NUM_Q_VT
    print("VR %d VT %d VP %d CHI %d PHI %d" %(params.BEVelocitySpace.NUM_Q_VR,params.BEVelocitySpace.NUM_Q_VT,params.BEVelocitySpace.NUM_Q_VP,params.BEVelocitySpace.NUM_Q_CHI,params.BEVelocitySpace.NUM_Q_PHI))
    L1=Lp(g0,maxwellian,VTH) - Lm(g0,maxwellian,VTH)
    print("np.norm(L1-L0) : %.10f"%np.linalg.norm(L1-L0))


    L0=L1
    print("Refine on V azimuthal angle")
    params.BEVelocitySpace.NUM_Q_VT                 = params.BEVelocitySpace.NUM_Q_VT//2
    params.BEVelocitySpace.NUM_Q_VP                  = 2 * params.BEVelocitySpace.NUM_Q_VP
    print("VR %d VT %d VP %d CHI %d PHI %d" %(params.BEVelocitySpace.NUM_Q_VR,params.BEVelocitySpace.NUM_Q_VT,params.BEVelocitySpace.NUM_Q_VP,params.BEVelocitySpace.NUM_Q_CHI,params.BEVelocitySpace.NUM_Q_PHI))
    L1=Lp(g0,maxwellian,VTH) - Lm(g0,maxwellian,VTH)
    print("np.norm(L1-L0) : %.10f"%np.linalg.norm(L1-L0))


def eigenvec_collision_op(collision,maxwellian):

    import matplotlib.pyplot as plt

    params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER=4
    params.BEVelocitySpace.SPH_HARM_LM = [ [0,0], [1,0],[1,1],[2,0]]
    params.BEVelocitySpace.NUM_Q_VR    = 51
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

    L    = cf_sp.assemble_mat(collision,maxwellian,VTH)
    M    = spec_sp.compute_maxwellian_mm(maxwellian,VTH)

    FOp  = np.matmul(np.linalg.inv(M),L)
    W,Q  = np.linalg.eig(FOp)
    print("Eigenvalues: ")
    print(W)
    Qinv = np.linalg.inv(Q)
    
    y_init = np.ones(num_p*num_sh)
    
    q_norm=np.array([np.linalg.norm(Q[:,i]) for i in range(num_p*num_sh)])
    print("Eigenvectors normalized : ", np.allclose(q_norm,np.ones_like(q_norm)))

    fun_sol = np.array([y_init[i] * Q[:,i] for i in range(num_p*num_sh)])
    #print(fun_sol)
    fun_sol = np.transpose(fun_sol)

    # if(h_init is not None):
    #     print("Scaled for the initial condition : " ,np.allclose(fun_sol.sum(axis=1),h_init))
    
    #print([y_init[i] * Q[:,i] for i in range(num_p*num_sh)])
    # print(Q)
    # print(Q[:,0])
    
    num_pts = 100
    X = np.linspace(-1,1,num_pts)
    Y = np.linspace(-1,1,num_pts)
    #Z = np.linspace(-1,1,num_pts)
    Z = np.array([0.5])

    X,Y,Z=np.meshgrid(X,Y,Z,indexing='ij')

    R     = np.sqrt(X**2 + Y**2 + Z**2)
    THETA = np.arccos(Z/R)
    PHI   = np.arctan(Y/X)  

    #P_klm = np.array([np.exp(-R**2)*spec_sp.basis_eval_full(R,THETA,PHI,k,sph_modes[lm_i][0],sph_modes[lm_i][1]) for k in range(num_p) for lm_i in range(num_sh)])
    #P_klm = np.array([ maxwellian(R) *spec_sp.basis_eval_full(R,THETA,PHI,k,sph_modes[lm_i][0],sph_modes[lm_i][1]) for k in range(num_p) for lm_i in range(num_sh)])
    P_klm = np.array([spec_sp.basis_eval_full(R,THETA,PHI,k,sph_modes[lm_i][0],sph_modes[lm_i][1]) for k in range(num_p) for lm_i in range(num_sh)])
    #print(P_klm.shape)
    P_klm = P_klm.reshape(num_p*num_sh,-1)
    P_klm = np.transpose(P_klm)
    #print(P_klm.shape)
    #print(P_klm[:,3])
    
    fun_sol_on_g = np.matmul(P_klm,fun_sol)
    fun_sol_on_g = np.transpose(fun_sol_on_g)

    # point_data=dict()
    # for pk in range(num_p):
    #     for lm_i,lm in enumerate(sph_modes):
    #         point_data["klm_%d_%d_%d"%(pk,lm[0],lm[1])] = np.array(np.real(fun_sol_on_g[pk*num_sh + lm_i]).reshape(num_pts,num_pts,num_pts))
    
    # visualize_utils.vtk_structured_grid("eigen_vec3",X,Y,Z,point_data,None)

    
    fig, axs = plt.subplots(num_p, num_sh)
    fig.set_size_inches(8,30)
    #fig.set_size_inches(3.5*num_sh, 5*num_p)
    for pk in range(num_p):
        for lm_i,lm in enumerate(sph_modes):
            im=axs[pk, lm_i].imshow(np.real(fun_sol_on_g[pk*num_sh + lm_i].reshape(num_pts,num_pts)))
            plt.colorbar(im, ax=axs[pk, lm_i])
            axs[pk, lm_i].set_title("klm=(%d,%d,%d) eig:%.2E+%.2Ej" %(pk,lm[0],lm[1],np.real(W[pk*num_sh + lm_i]), np.imag(W[pk*num_sh + lm_i]) ),fontsize = 7.0)
            axs[pk, lm_i].tick_params(axis='both', which='major', labelsize=7)
    plt.tight_layout()
    plt.show()
    fig.savefig("fun_sol")
    plt.close()

    return [W,fun_sol]



def maxwellian_test():
    """
    try to compute the expansion coefficients function 
    written in a maxwellian 1 w.r.t. basis maxwellian 2. 
    """

    params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER=15
    params.BEVelocitySpace.SPH_HARM_LM = [ [0,0],[1,1]]
    params.BEVelocitySpace.NUM_Q_VR    = 51
    params.BEVelocitySpace.NUM_Q_VT    = 16
    params.BEVelocitySpace.NUM_Q_VP    = 16
    params.BEVelocitySpace.NUM_Q_CHI   = 64
    params.BEVelocitySpace.NUM_Q_PHI   = 16

    colOpSp.SPEC_SPHERICAL  = sp.SpectralExpansionSpherical(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,basis.Maxwell(),params.BEVelocitySpace.SPH_HARM_LM) 
    cf_sp    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)
    spec_sp  = colOpSp.SPEC_SPHERICAL

    MT1  = 1.0*collisions.TEMP_K_1EV
    VTH1 = collisions.electron_thermal_velocity(MT1)
    maxwellian_1 = BEUtils.get_maxwellian_3d(VTH1,1)

    # set f to be the maxwellian 1
    #hv1     = lambda v,vt,vp : 3.54490770e+00 * spec_sp.basis_eval_full(v,vt,vp,0,0,0) -1.22241517e-1 * spec_sp.basis_eval_full(v,vt,vp,2,1,0) #+ 5.55490164e-01 * spec_sp.basis_eval_full(v,vt,vp,2,0,0) -1.56114596e-01 * spec_sp.basis_eval_full(v,vt,vp,3,0,0) + 2.17803558e-02 *spec_sp.basis_eval_full(v,vt,vp,4,0,0)
    hv1_vec = np.zeros(spec_sp.get_num_coefficients())
    hv1_vec[0]=1
    print(hv1_vec)

    import matplotlib.pyplot as plt
    plt.plot(hv1_vec,'b-o',label='orig(T1)')
    # try to expand it using maxwellian 2. 
    for t_fac in np.linspace(1,3,3):
        MT2  = (1/t_fac)*collisions.TEMP_K_1EV
        VTH2 = collisions.electron_thermal_velocity(MT2)
        maxwellian_2 = BEUtils.get_maxwellian_3d(VTH2,1)
        #print(VTH2,MT2)
        maxwellian_ratio = lambda vr,vt,vp: ( (VTH2**2) / (VTH1**2) ) * np.exp( -(vr*VTH2/VTH1)**2 + vr**2)
        mm_h2     = spec_sp.compute_maxwellian_mm(maxwellian_2,VTH2) #BEUtils.compute_func_PiPj(spec_sp,maxwellian_ratio,maxwellian_2,VTH2,None,None,None,1)
        mm_h1     = BEUtils.compute_Mvth1_Pi_vth2_Pj_vth1(spec_sp,maxwellian_1,VTH1,maxwellian_2,VTH2,None,None,None,1)
        hv2_vec   = np.dot(np.matmul(np.linalg.inv(mm_h2),mm_h1),hv1_vec)
        #hv2_vec = np.dot(np.linalg.inv(mm), BEUtils.compute_func_proj_to_basis(spec_sp,mr_h1,maxwellian_2,VTH2,None,None,None,1))
        #print(mm)
        print(hv2_vec)
        print("diff |hv2-hv1| = ",np.linalg.norm(hv2_vec-hv1_vec))
        
        m0_1 = BEUtils.moment_zero_f(spec_sp,hv1_vec,maxwellian_1,VTH1,None,None,None,1)
        print("h1 m0: ", m0_1)

        m0_2 = BEUtils.moment_zero_f(spec_sp,hv2_vec,maxwellian_2,VTH2,None,None,None,1)
        print("h2 m0: ", m0_2)
        print("mass diff : ", (m0_2-m0_1)/m0_1)

        plt.plot(hv2_vec,'-x',label='T2/T1 = %.2E'%(1/t_fac))
        plt.legend()

    plt.show()
    plt.close()

    plt.plot(hv1_vec,'b-o',label='orig(T1)')
    # try to expand it using maxwellian 2. 
    for t_fac in np.linspace(1,3,3):
        MT2  = (t_fac)*collisions.TEMP_K_1EV
        VTH2 = collisions.electron_thermal_velocity(MT2)
        maxwellian_2 = BEUtils.get_maxwellian_3d(VTH2,1)
        #print(VTH2,MT2)
        maxwellian_ratio = lambda vr,vt,vp: ( (VTH2**2) / (VTH1**2) ) * np.exp( -(vr*VTH2/VTH1)**2 + vr**2)
        mm_h2     = spec_sp.compute_maxwellian_mm(maxwellian_2,VTH2) #BEUtils.compute_func_PiPj(spec_sp,maxwellian_ratio,maxwellian_2,VTH2,None,None,None,1)
        mm_h1     = BEUtils.compute_Mvth1_Pi_vth2_Pj_vth1(spec_sp,maxwellian_1,VTH1,maxwellian_2,VTH2,None,None,None,1)
        hv2_vec   = np.dot(np.matmul(np.linalg.inv(mm_h2),mm_h1),hv1_vec)
        #hv2_vec = np.dot(np.linalg.inv(mm), BEUtils.compute_func_proj_to_basis(spec_sp,mr_h1,maxwellian_2,VTH2,None,None,None,1))
        #print(mm)
        print(hv2_vec)
        print("diff |hv2-hv1| = ",np.linalg.norm(hv2_vec-hv1_vec))
        
        m0_1 = BEUtils.moment_zero_f(spec_sp,hv1_vec,maxwellian_1,VTH1,None,None,None,1)
        print("h1 m0: ", m0_1)

        m0_2 = BEUtils.moment_zero_f(spec_sp,hv2_vec,maxwellian_2,VTH2,None,None,None,1)
        print("h2 m0: ", m0_2)
        print("mass diff : ", (m0_2-m0_1)/m0_1)

        plt.plot(hv2_vec,'-x',label='T2/T1 = %.2E'%(t_fac))
        plt.legend()

    plt.show()
    plt.close()

    
def collision_op_thermal_test():

    params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER = 1
    params.BEVelocitySpace.SPH_HARM_LM = [[0,0],[1,0]]
    params.BEVelocitySpace.NUM_Q_VR  = 21
    params.BEVelocitySpace.NUM_Q_VT  = 16
    params.BEVelocitySpace.NUM_Q_VP  = 16
    params.BEVelocitySpace.NUM_Q_CHI = 64
    params.BEVelocitySpace.NUM_Q_PHI = 16

    cf_sp    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)
    colOpSp.SPEC_SPHERICAL  = sp.SpectralExpansionSpherical(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,basis.Maxwell(),params.BEVelocitySpace.SPH_HARM_LM) 
    spec_sp  = colOpSp.SPEC_SPHERICAL

    VTH1 = 1.0 * VTH
    MW1 = BEUtils.get_maxwellian_3d(VTH1,1)

    L_np1   = Lp(g0,MW1,VTH1) - Lm(g0,MW1,VTH1)
    print("Coll_Op at VTH: ",VTH1)
    print(L_np1)
    
    VTH2 = (1 - 1e-3)* VTH1 # +   #1e-2 * VTH1
    MW2 = BEUtils.get_maxwellian_3d(VTH2,1)
    L_np2   = Lp(g0,MW2,VTH2) - Lm(g0,MW2,VTH2)
    
    print("Coll_Op at VTH: ",VTH2)
    print(L_np2)
    
    print("diff relative : ", np.linalg.norm(L_np1-L_np2)/np.linalg.norm(L_np2))


collision_op_thermal_test()
#collision_op_test()
#collision_op_conv()
#maxwellian_test()
#eigenvec_collision_op(g0,maxwellian)
#plot_crosssection(g0,50)




    


