"""
simple cases to test the collision operator, 
compute the collision operator, for known expression 
for cross-section and numerically validate the result
"""
from re import S

from matplotlib import collections
from matplotlib.pyplot import axis, sci
import scipy
from scipy.interpolate.polyint import approximate_taylor_polynomial

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


def plot_crosssection(collision: collisions.Collisions,num_q,ev_q):
    """
    plot: how quadrature points hits the experimental cross section curve. 
    """
    #np.sqrt(2*BOLTZMANN_CONST*MAXWELLIAN_TEMP_K/MASS_ELECTRON)
    import matplotlib.pyplot as plt

    ELE_VOLT        = collisions.ELECTRON_VOLT
    BOLTZMANN_CONST = collisions.BOLTZMANN_CONST

    TEMP_K_1EV          = ELE_VOLT/BOLTZMANN_CONST
    g            = collision 
    ev   = collision._energy
    tcs  = collision._total_cs

    [gmx,gmw]    = maxpoly.maxpolygauss(num_q-1)
    weight_func  = maxpoly.maxpolyweight

    MAXWELLIAN_TEMP_K   = ev_q * TEMP_K_1EV
    V_TH                = np.sqrt(2*BOLTZMANN_CONST*MAXWELLIAN_TEMP_K/collisions.MASS_ELECTRON)
    energy_ev           = (0.5 * collisions.MASS_ELECTRON * ((gmx*V_TH)**2))/ELE_VOLT
    
    tcs_q     = g.total_cross_section(energy_ev)

    # p1 = plt.plot(ev,tcs,'o-b')
    # p2 = plt.plot(energy_ev,tcs_q,'x-g')
    plt.plot(ev,tcs,'o-b',energy_ev,tcs_q,'x-g')
    plt.legend(["LXCAT","quadrature points"])

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
    plt.xlabel("energy (eV)")
    plt.ylabel("cross section (m2)")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid()
    plt.show()

def collision_op_test(ev=1.0):
    """
    Compute the collision op. with numpy based and the loop based, 
    Check to make sure the tensorized coll. op. is equivalent to the 
    the loop based variation. 
    """

    collisions.AR_NEUTRAL_N=3.22e22
    collisions.MAXWELLIAN_N=1e18
    collisions.AR_IONIZED_N=1e18

    collisions.MAXWELLIAN_TEMP_K   = ev * collisions.TEMP_K_1EV
    collisions.ELECTRON_THEMAL_VEL = collisions.electron_thermal_velocity(collisions.MAXWELLIAN_TEMP_K) 
    VTH = collisions.ELECTRON_THEMAL_VEL

    maxwellian = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)


    params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER=2
    params.BEVelocitySpace.SPH_HARM_LM = [[0,0]]
    params.BEVelocitySpace.NUM_Q_VR    = 2
    params.BEVelocitySpace.NUM_Q_VT    = 2
    params.BEVelocitySpace.NUM_Q_VP    = 2
    params.BEVelocitySpace.NUM_Q_CHI   = 2
    params.BEVelocitySpace.NUM_Q_PHI   = 2
    
    cf_sp    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)
    g0.reset_scattering_direction_sp_mat()
    t1 = time.time()
    L_loop = cf_sp._LOp_l(g0,maxwellian,VTH) #- cf_sp._Lm_l(g0,maxwellian,VTH)
    t2 = time.time()
    print("loop based (s) : %.10f" %(t2-t1))
    
    g0.reset_scattering_direction_sp_mat()
    t1 = time.time()
    L_np   = cf_sp._LOp(g0,maxwellian,VTH) #cf_sp._Lp(g0,maxwellian,VTH) - cf_sp._Lm(g0,maxwellian,VTH)
    t2 = time.time()
    print("np(tensor) based (s) : %.10f" %(t2-t1))
    print("np(tensor) based |L_np| : %.10f" %(np.linalg.norm(L_np)))
    print("loop based |L_loop|     : %.10f" %(np.linalg.norm(L_loop)))
    print("loop based |L_np - L_loop|       : %.10f" %(np.linalg.norm(L_loop-L_np)))
    print("Relative diff (L_np - L_loop)/|L_loop|: ")
    print((L_np-L_loop)/np.linalg.norm(L_loop))
    print("np based")
    print(L_np)
    print("loop based")
    print(L_loop)

def collision_op_conv(collision,ev=1.0):
    """
    Simplified test to check if the collision op. 
    converges with the current quadrature grid. 
    """

    collisions.AR_NEUTRAL_N=3.22e22
    collisions.MAXWELLIAN_N=1e18
    collisions.AR_IONIZED_N=1e18

    collisions.MAXWELLIAN_TEMP_K   = ev * collisions.TEMP_K_1EV
    collisions.ELECTRON_THEMAL_VEL = collisions.electron_thermal_velocity(collisions.MAXWELLIAN_TEMP_K) 
    V_TH = collisions.ELECTRON_THEMAL_VEL

    maxwellian = BEUtils.get_maxwellian_3d(V_TH,collisions.MAXWELLIAN_N)
    print("collision op: convergence test for ",ev," eV")

    NR=[4,8,16,32]
    NSH=1
    for nr in NR:
        params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER=nr
        params.BEVelocitySpace.SPH_HARM_LM = [[0,0]]
        params.BEVelocitySpace.NUM_Q_VR    = 118
        params.BEVelocitySpace.NUM_Q_VT    = 2
        params.BEVelocitySpace.NUM_Q_VP    = 2
        params.BEVelocitySpace.NUM_Q_CHI   = 2
        params.BEVelocitySpace.NUM_Q_PHI   = 2

        q_mode=sp.QuadMode.GMX
        cf_sp    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,q_mode=q_mode)
        
        #print("NR=%d NSH=%d Quadrature grid VR=%d VT=%d VP=%d CHI=%d PHI=%d" %(nr,NSH,params.BEVelocitySpace.NUM_Q_VR,params.BEVelocitySpace.NUM_Q_VT,params.BEVelocitySpace.NUM_Q_VP,params.BEVelocitySpace.NUM_Q_CHI,params.BEVelocitySpace.NUM_Q_PHI))
        
        L0=cf_sp._spec.create_mat()
        for g in collision:
            g.reset_scattering_direction_sp_mat()
            L0+=cf_sp.assemble_mat(g,maxwellian,V_TH)
        
        for i in range(10):

            if(cf_sp._spec._q_mode== sp.QuadMode.GMX):
                params.BEVelocitySpace.NUM_Q_VR                  = min(118,2 * params.BEVelocitySpace.NUM_Q_VR)
            else:
                params.BEVelocitySpace.NUM_Q_VR              = 2 * params.BEVelocitySpace.NUM_Q_VR + 1

            params.BEVelocitySpace.NUM_Q_VT                  = 2 * params.BEVelocitySpace.NUM_Q_VT
            params.BEVelocitySpace.NUM_Q_VP                  = 2 * params.BEVelocitySpace.NUM_Q_VP
            params.BEVelocitySpace.NUM_Q_CHI                 = 2 * params.BEVelocitySpace.NUM_Q_CHI
            params.BEVelocitySpace.NUM_Q_PHI                 = 2 * params.BEVelocitySpace.NUM_Q_PHI
            g.reset_scattering_direction_sp_mat()
            
            cf_sp1    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,q_mode=q_mode)
            L1=cf_sp1._spec.create_mat()
            for g in collision:
                g.reset_scattering_direction_sp_mat()
                L1+=cf_sp1.assemble_mat(g,maxwellian,V_TH)
            
            # print("L0")
            # print(L0)
            # print("L1")
            # print(L1)
            error = np.linalg.norm(L1-L0)/np.linalg.norm(L0)
            print("np.norm(L1-L0) : %.10E"%(error))
            L0=L1
            if(error < 1e-6):
                print("NR=%d NSH=%d Quadrature grid VR=%d VT=%d VP=%d CHI=%d PHI=%d" %(nr,NSH,params.BEVelocitySpace.NUM_Q_VR,params.BEVelocitySpace.NUM_Q_VT,params.BEVelocitySpace.NUM_Q_VP,params.BEVelocitySpace.NUM_Q_CHI,params.BEVelocitySpace.NUM_Q_PHI))
                break

    return

def spec_convergence_collision_op(collision,ev=1.0):
    """
    Uses the linear eigen based solver, 
    no reassemble or re-projection just 
    eigen base analytical solution. 
    """
    
    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (6,6)

    collisions.AR_NEUTRAL_N=3.22e22
    collisions.MAXWELLIAN_N=1e18
    collisions.AR_IONIZED_N=1e18

    collisions.MAXWELLIAN_TEMP_K   = ev * collisions.TEMP_K_1EV
    collisions.ELECTRON_THEMAL_VEL = collisions.electron_thermal_velocity(collisions.MAXWELLIAN_TEMP_K) 
    VTH = collisions.ELECTRON_THEMAL_VEL

    maxwellian = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)
    
    params.BEVelocitySpace.SPH_HARM_LM = [[0,0]]
    params.BEVelocitySpace.NUM_Q_VR    = 118
    params.BEVelocitySpace.NUM_Q_VT    = 2
    params.BEVelocitySpace.NUM_Q_VP    = 2
    params.BEVelocitySpace.NUM_Q_CHI   = 2
    params.BEVelocitySpace.NUM_Q_PHI   = 2

    TEV = collisions.MAXWELLIAN_TEMP_K * collisions.BOLTZMANN_CONST / collisions.ELECTRON_VOLT
    print("temp (ev): ", TEV, " VTH : ",VTH)

    EIGEN_DECOMP_TOL=1e-6
    MNE = collisions.MAXWELLIAN_N
    NR        = [4,8,16,32]
    num_steps = 300
    T_TOTAL   = 1e-7
    SPEC_QUAD_MODE = sp.QuadMode.GMX
         
    data = list()
    dt_steps = np.linspace(0,T_TOTAL,num_steps)
    tail_norm = lambda x, i: np.linalg.norm(x[:,i:],axis=1)/np.linalg.norm(x[:,:],axis=1)

    def assemble_and_eigen_factorize(nr, q_mode, t_total):
        params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER=nr
        for g in collision:
            g.reset_scattering_direction_sp_mat()
        
        cf_sp    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,q_mode=SPEC_QUAD_MODE)
        spec_sp  = cf_sp._spec

        L = spec_sp.create_mat()
        for g in collision:
            L +=cf_sp.assemble_mat(g,maxwellian,VTH)

        M=cf_sp._spec.compute_maxwellian_mm(maxwellian,VTH)
        L=np.matmul(np.linalg.inv(M),L)
        
        W,Q  = scipy.linalg.eig(L)#np.linalg.eig(L)
        Qinv = np.linalg.inv(Q)
        LL= np.matmul(Q , np.matmul(np.diag(W),Qinv)) 

        #if(np.linalg.norm(L-LL)/np.linalg.norm(L) > EIGEN_DECOMP_TOL):
        #print("NORM(L-QW Q^{-1})/NORM(L) : ", np.linalg.norm(L-LL)/np.linalg.norm(L))
        LQ = np.matmul(L,Q)
        QW = np.matmul(Q,np.diag(W))
        print("condition number : %.8E"%(np.linalg.cond(Q)))
        print("error eigen decomp: %.8E"%(np.linalg.norm(LQ-QW)/np.linalg.norm(LQ)))

        ET = np.real(np.matmul(Q,np.matmul(np.diag(np.exp(W*t_total)),Qinv)))

        return ET,L,W,Q

    for nr in NR:
        
        Et_h, L_h, W_h, Q_h     = assemble_and_eigen_factorize(nr,SPEC_QUAD_MODE,T_TOTAL)
        Et_2h, L_2h, W_2h, Q_2h = assemble_and_eigen_factorize(2*nr,SPEC_QUAD_MODE,T_TOTAL)

        m_truncate=nr//2
        Pm=np.eye(m_truncate + 1, nr+1)
        E_h = np.matmul(Pm,np.matmul(Et_h,np.transpose(Pm)))

        Pm   =  np.eye(m_truncate + 1, 2*nr+1)
        E_2h = np.matmul(Pm,np.matmul(Et_2h,np.transpose(Pm)))

        # # print(ET_h_m)
        # # print(ET_2h_m)

        # convg_error = np.linalg.norm(E_2h-E_h)/np.linalg.norm(E_2h)
        # print("Nr=%d to Nr=%d error : %.10E"%(nr,2*nr,convg_error))

        # Et_h, L_h, W_h, Q_h     = assemble_and_eigen_factorize(nr,SPEC_QUAD_MODE,T_TOTAL)
        # Et_2h, L_2h, W_2h, Q_2h = assemble_and_eigen_factorize(2*nr,SPEC_QUAD_MODE,T_TOTAL)
        
        # m_truncate=nr//2
        # Pm=np.eye(m_truncate + 1, nr+1)
        # Y=scipy.linalg.solve(Q_h,np.transpose(Pm))
        # E_h = np.matmul(np.matmul(np.matmul(Pm,Q_h),np.diag(np.exp(W_h*T_TOTAL))),Y)

        # Pm=np.eye(m_truncate + 1, 2*nr+1)
        # Y=scipy.linalg.solve(Q_2h,np.transpose(Pm))
        # E_2h = np.matmul(np.matmul(np.matmul(Pm,Q_2h),np.diag(np.exp(W_2h*T_TOTAL))),Y)

        convg_error = np.linalg.norm(E_2h-E_h)/np.linalg.norm(E_2h)
        print("Nr=%d to Nr=%d error : %.10E\n\n"%(nr,2*nr,convg_error))

    return

def eigenvec_collision_op(collision,maxwellian):

    import matplotlib.pyplot as plt

    params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER=31
    params.BEVelocitySpace.SPH_HARM_LM = [[0,0]]
    params.BEVelocitySpace.NUM_Q_VR    = 51
    params.BEVelocitySpace.NUM_Q_VT    = 8
    params.BEVelocitySpace.NUM_Q_VP    = 4
    params.BEVelocitySpace.NUM_Q_CHI   = 64
    params.BEVelocitySpace.NUM_Q_PHI   = 2

    TEV = collisions.MAXWELLIAN_TEMP_K * collisions.BOLTZMANN_CONST / collisions.ELECTRON_VOLT
    print("temp (ev): ", TEV, " VTH : ",VTH)
    
    cf_sp    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)
    spec_sp  = cf_sp._spec
    
    sph_modes = params.BEVelocitySpace.SPH_HARM_LM
    num_p     = spec_sp._p +1
    num_sh    = len(params.BEVelocitySpace.SPH_HARM_LM)
    L = spec_sp.create_mat()
    for g in collision:
        g.reset_scattering_direction_sp_mat()
        L +=cf_sp.assemble_mat(g,maxwellian,VTH)

    
    W,Q  = np.linalg.eig(L)
    Qinv = np.linalg.inv(Q)
    LL   = np.matmul(Q,np.matmul(np.diag(W),Qinv)) 
    print("Eigen decomposition error : ", np.linalg.norm(L-LL)/np.linalg.norm(L))
    

    fname="g"
    for g in collision:
        fname = fname + str(g._type)
    
    num_cols = 8
    fig, axs = plt.subplots(int(np.ceil(num_p*num_sh/num_cols)), num_cols)
    fig.set_size_inches(20,8)
    
    for dt in np.linspace(0,3e-8,3):
        E    = np.real(np.matmul(Q,np.matmul(np.diag(np.exp(W*dt)),Qinv)))
        for r in range(num_p):
            for c in range(num_sh):
                rid   = r*num_sh + c
                plt_r = rid//num_cols
                plt_c = rid%num_cols
                axs[plt_r,plt_c].plot(range(len(E[:,rid])),E[:,rid],label="dt=%.2E"%(dt))
                axs[plt_r,plt_c].set_title("(k,l,m)=(%d,%d,%d)" %(r,sph_modes[c][0],sph_modes[c][1]),fontsize=7)
                axs[plt_r,plt_c].tick_params(axis='both', which='major', labelsize=7)
                axs[plt_r,plt_c].grid()
                axs[plt_r,plt_c].set_yscale("log")
                axs[0,0].legend()

    
    
    fname1=fname + f"_Nr_%d_eig_fun_modes_%.2E_NI_%.2E_N0_%.2E_NE_%.2E.png" %(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,TEV,collisions.AR_IONIZED_N,collisions.AR_NEUTRAL_N,collisions.MAXWELLIAN_N)
    print("writing file to : ", fname1)
    plt.savefig(fname1)
    plt.show()
    plt.close()

    # plt.plot(np.abs(W))
    # plt.grid()
    # plt.xlabel("Eigen mode")
    # plt.ylabel("magnitude abs(\lambda)")
    # #plt.show()
    # fname1=fname + f"_Nr_%d_eig_abs_%.2E_NI_%.2E_N0_%.2E_NE_%.2E.png" %(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,TEV,collisions.AR_IONIZED_N,collisions.AR_NEUTRAL_N,collisions.MAXWELLIAN_N)
    # plt.savefig(fname1)
    # plt.show()
    # plt.close()


    # plt.scatter(np.real(W),np.imag(W))
    # plt.grid()
    # plt.xlabel("Real")
    # plt.ylabel("Imag")
    # fname1=fname + f"_Nr_%d_eig_cplx_%.2E_NI_%.2E_N0_%.2E_NE_%.2E.png" %(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,TEV,collisions.AR_IONIZED_N,collisions.AR_NEUTRAL_N,collisions.MAXWELLIAN_N)
    # plt.savefig(fname1)
    # plt.show()
    # plt.close()

    fig, axs = plt.subplots(int(np.ceil(num_p*num_sh/num_cols)), num_cols)
    fig.set_size_inches(20,8)

    for r in range(num_p):
            for c in range(num_sh):
                rid   = r*num_sh + c
                plt_r = rid//num_cols
                plt_c = rid%num_cols

                axs[plt_r,plt_c].plot(range(len(Q[:,rid])),np.real(Q[:,rid]),label="Re")
                axs[plt_r,plt_c].plot(range(len(Q[:,rid])),np.imag(Q[:,rid]),label ="Im")
                axs[plt_r,plt_c].set_title("(%d,%d,%d) = %.2E,%.2E" %(r,sph_modes[c][0],sph_modes[c][1],np.real(W[r*num_sh + c]),np.imag(W[r*num_sh + c])),fontsize=7)
                axs[plt_r,plt_c].tick_params(axis='both', which='major', labelsize=7)
                axs[plt_r,plt_c].grid()
                axs[0,0].legend()
    
    fname1=fname + f"_Nr_%d_eig_vec_%.2E_NI_%.2E_N0_%.2E_NE_%.2E.png" %(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,TEV,collisions.AR_IONIZED_N,collisions.AR_NEUTRAL_N,collisions.MAXWELLIAN_N)
    plt.tight_layout()
    plt.savefig(fname1)
    plt.show()
    plt.close()

    
    ## maxwellian initial condition over time. 
    fig, axs = plt.subplots(int(np.ceil(num_p*num_sh/num_cols)), num_cols)
    fig.set_size_inches(20,8)

    num_steps=30
    TT  = 1e-3
    tt    =  np.linspace(0,TT,num_steps)
    c_vec =  np.zeros((num_p*num_sh,num_steps))

    for ii,dt in enumerate(tt):
        E = np.real(np.matmul(Q,np.matmul(np.diag(np.exp(W*dt)),Qinv)))
        for r in range(num_p):
            for c in range(num_sh):
                rid   = r*num_sh + c    
                c_vec[rid,ii] = E[rid,0]

    for r in range(num_p):
        for c in range(num_sh):
            rid   = r*num_sh + c
            plt_r = rid//num_cols
            plt_c = rid%num_cols

            axs[plt_r,plt_c].plot(tt,c_vec[rid,:])
            axs[plt_r,plt_c].set_title("(%d,%d,%d) = %.2E,%.2E" %(r,sph_modes[c][0],sph_modes[c][1],np.real(W[r*num_sh + c]),np.imag(W[r*num_sh + c])),fontsize=7)
            axs[plt_r,plt_c].tick_params(axis='both', which='major', labelsize=7)
            axs[plt_r,plt_c].grid()
            axs[plt_r,plt_c].set_xlabel("time(s)")
            #axs[0,0].legend()

    fname1=fname + f"_Nr_%d_coeff_vs_time_%.2E_NI_%.2E_N0_%.2E_NE_%.2E.png" %(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,TEV,collisions.AR_IONIZED_N,collisions.AR_NEUTRAL_N,collisions.MAXWELLIAN_N)
    plt.tight_layout()
    plt.savefig(fname1)
    plt.show()
    plt.close()
    
def maxwellian_basis_change_test():
    """
    try to compute the expansion coefficients function 
    written in a maxwellian 1 w.r.t. basis maxwellian 2. 
    """

    params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER=31
    params.BEVelocitySpace.SPH_HARM_LM = [[0,0]]
    params.BEVelocitySpace.NUM_Q_VR    = 51
    params.BEVelocitySpace.NUM_Q_VT    = 8
    params.BEVelocitySpace.NUM_Q_VP    = 4
    params.BEVelocitySpace.NUM_Q_CHI   = 64
    params.BEVelocitySpace.NUM_Q_PHI   = 2

    import matplotlib.pyplot as plt

    MNE = 1e18
    MT1  = 1.0*collisions.TEMP_K_1EV
    VTH1 = collisions.electron_thermal_velocity(MT1)
    maxwellian_1 = BEUtils.get_maxwellian_3d(VTH1,MNE)
    grid_pts         = int(1e5)
    grid_bds         = (0,10)
    vr1              = np.linspace(grid_bds[0]*VTH1,grid_bds[1]*VTH1,grid_pts)
    v1               = vr1/VTH1
    Mr1              = maxwellian_1(v1)

    
    for NR in [3,7,15,31]:
        params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER=NR

        cf_sp    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)
        spec_sp  = cf_sp._spec

        TAIL_NORM_INDEX = (spec_sp._p+1) * len(spec_sp._sph_harm_lm) // 2
        tail_norm = lambda x, i: np.linalg.norm(x[i:],ord=2)/np.linalg.norm(x,ord=2)

        h_vec = spec_sp.create_vec()
        h_vec = np.zeros_like(h_vec)
        h_vec[0]=1.0

        Pr1              = spec_sp.Vq_r(v1).transpose()
        fv1              = Mr1 * (np.dot(Pr1,h_vec).reshape(-1))
        # plt.figure(NR)
        # plt.plot(vr1,fv1,label="NR=%d T1=%.2f"%(NR,MT1*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT))
        
        eps_ls        = np.linspace(-2e-1,2e-1,41)
        error_eval    = np.zeros_like(eps_ls) # error in the function eval on the grid.
        error_tail     = np.zeros_like(eps_ls) # error after projection back. 
        for ii, eps in enumerate(eps_ls):
            tmp_red_fac = 1 + eps
            MT2  = tmp_red_fac * MT1

            VTH2 = collisions.electron_thermal_velocity(MT2)
            maxwellian_2 = BEUtils.get_maxwellian_3d(VTH2,MNE)

            mm_h12    = BEUtils.compute_Mvth1_Pi_vth2_Pj_vth1(spec_sp,maxwellian_1,VTH1,maxwellian_2,VTH2,None,None,None,1)
            h_im      = np.dot(mm_h12,h_vec)
            
            vr2              = vr1#np.linspace(grid_bds[0]*VTH2,grid_bds[1]*VTH2,grid_pts)
            v2               = vr2/VTH2
            Mr2              = maxwellian_2(v2)
            Pr2              = spec_sp.Vq_r(v2).transpose()
            fv2              = Mr2 * (np.dot(Pr2,h_im).reshape(-1))
            # print(np.linalg.norm(fv1))
            # print(np.linalg.norm(fv2))
            error_eval[ii]   = np.linalg.norm(fv2-fv1)/np.linalg.norm(fv1)
            mm_h21    = BEUtils.compute_Mvth1_Pi_vth2_Pj_vth1(spec_sp,maxwellian_2,VTH2,maxwellian_1,VTH1,None,None,None,1)
            h_new     = np.dot(mm_h21,h_im)
            error_tail[ii]    = tail_norm(h_im,TAIL_NORM_INDEX)
            print(f"Nr = {NR:d} T1 = {MT1:.4E} T2 = {MT2:.4E} epsilon = {eps:.4E} error= {error_eval[ii]:.16E}")

            if(ii%20==0):
                plt.figure(NR)
                plt.plot(vr2,fv2,label="NR=%d T2=%.2f"%(NR,MT2*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT))

        plt.figure(0)
        plt.plot(eps_ls,error_eval,label="Nr=%d"%NR)
        plt.figure(1)
        plt.plot(eps_ls,error_tail,label="Nr=%d"%NR)
    
    #plt.plot(eps_ls,error_pjb ,label="|h1 - M_12 * M_21 * h1|")
    plt.figure(0)
    plt.xlabel("epsilon")
    plt.ylabel("relative error")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.savefig("basis_1ev_grid.png")
    
    plt.figure(1)
    plt.xlabel("epsilon")
    plt.ylabel("tail norm")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.savefig("basis_1ev_tail.png")

    for NR in [3,7,15,31]:
        plt.figure(NR)
        plt.xlabel("vr")
        plt.ylabel("f(vr)")
        #plt.yscale("log")
        plt.grid()
        plt.legend()
        plt.savefig(f"basis_1ev_f_NR_%d.png"%NR)

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
        
        m0_1 = BEUtils.moment_n_f(spec_sp,hv1_vec,maxwellian_1,VTH1,0,None,None,None,1)
        print("h1 m0: ", m0_1)

        m0_2 = BEUtils.moment_n_f(spec_sp,hv2_vec,maxwellian_2,VTH2,0,None,None,None,1)
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
        
        m0_1 = BEUtils.moment_n_f(spec_sp,hv1_vec,maxwellian_1,VTH1,0,None,None,None,1)
        print("h1 m0: ", m0_1)

        m0_2 = BEUtils.moment_n_f(spec_sp,hv2_vec,maxwellian_2,VTH2,0,None,None,None,1)
        print("h2 m0: ", m0_2)
        print("mass diff : ", (m0_2-m0_1)/m0_1)

        plt.plot(hv2_vec,'-x',label='T2/T1 = %.2E'%(t_fac))
        plt.legend()

    plt.show()
    plt.close()
    
def collission_op_thermal_test():

    params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER = 4
    params.BEVelocitySpace.SPH_HARM_LM = [[0,0],[1,0]]
    params.BEVelocitySpace.NUM_Q_VR  = 21
    params.BEVelocitySpace.NUM_Q_VT  = 16
    params.BEVelocitySpace.NUM_Q_VP  = 16
    params.BEVelocitySpace.NUM_Q_CHI = 64
    params.BEVelocitySpace.NUM_Q_PHI = 16

    cf_sp    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)
    spec_sp  = cf_sp._spec

    VTH1 = 1.0 * VTH
    MW1 = BEUtils.get_maxwellian_3d(VTH1,1)

    L_np1   = spec_sp.Lp(g0,MW1,VTH1) - spec_sp.Lm(g0,MW1,VTH1)
    print("Coll_Op at VTH: ",VTH1)
    print(L_np1)
    
    NUM_COLL_STEPS=1000
    VTH2 = (np.sqrt(1 - 2*collisions.MASS_R_EARGON)**NUM_COLL_STEPS) * VTH1 # +   #1e-2 * VTH1
    MW2 = BEUtils.get_maxwellian_3d(VTH2,1)
    L_np2   = spec_sp.Lp(g0,MW2,VTH2) - spec_sp.Lm(g0,MW2,VTH2)
    
    print("Coll_Op at VTH: ",VTH2)
    print(L_np2)
    
    print("diff relative : ", np.linalg.norm(L_np1-L_np2)/np.linalg.norm(L_np2))

    W1,U1 = np.linalg.eig(L_np1)
    W2,U2 = np.linalg.eig(L_np2)

    for i in range(U1.shape[1]):
        print("EigenVec i : ",i ," angle change", np.arccos(np.dot(U1[:,i],U2[:,i])))

def collission_op_themal_sensitivity(collision):
    """
    To numerically see how sensitive the collission operator 
    for the v thermal. 
    """

    import matplotlib.pyplot as plt

    for NR in [7,15,31]:
        params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER=NR
        params.BEVelocitySpace.SPH_HARM_LM = [[0,0]]
        params.BEVelocitySpace.NUM_Q_VR    = 51
        params.BEVelocitySpace.NUM_Q_VT    = 8
        params.BEVelocitySpace.NUM_Q_VP    = 4
        params.BEVelocitySpace.NUM_Q_CHI   = 64
        params.BEVelocitySpace.NUM_Q_PHI   = 2

        collisions.AR_NEUTRAL_N=3.22e22
        collisions.MAXWELLIAN_N=1e18
        collisions.AR_IONIZED_N=1e18

        cf_sp    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)
        spec_sp  = cf_sp._spec
        sph_modes = params.BEVelocitySpace.SPH_HARM_LM
        num_p     = spec_sp._p +1
        num_sh    = len(params.BEVelocitySpace.SPH_HARM_LM)
        
        for g in collision:
            g.reset_scattering_direction_sp_mat()
        
        #L_list = list()
        x_ev = np.linspace(0.1,2,100)
        L_fd = np.zeros_like(x_ev)
        for i,ev in enumerate(x_ev):
            
            collisions.MAXWELLIAN_TEMP_K   = ev * collisions.TEMP_K_1EV
            collisions.ELECTRON_THEMAL_VEL = collisions.electron_thermal_velocity(collisions.MAXWELLIAN_TEMP_K) 
            VTH = collisions.ELECTRON_THEMAL_VEL

            TEV = collisions.MAXWELLIAN_TEMP_K * collisions.BOLTZMANN_CONST / collisions.ELECTRON_VOLT
            print("temp (ev): ", TEV, " VTH : ",VTH)

            maxwellian = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)
            L = spec_sp.create_mat()
            for g in collision:
                L +=cf_sp.assemble_mat(g,maxwellian,VTH)

            if i > 0:
                L_fd[i] = np.linalg.norm(L- L_prev)/(x_ev[i]-x_ev[i-1])
            
            L_prev = np.copy(L)

            
                
        
        plt.plot(x_ev[1:],L_fd[1:],label="Nr=%d"%NR)
        print(x_ev)
        print(L_fd)
    
    plt.xlabel("Vth (eV)")
    plt.ylabel("dL/d(Vth)")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()

def collission_op_themal_sensitivity_eig(collision):
    
    import matplotlib.pyplot as plt

    for NR in [31]:
        params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER=NR
        params.BEVelocitySpace.SPH_HARM_LM = [[0,0]]
        params.BEVelocitySpace.NUM_Q_VR    = 51
        params.BEVelocitySpace.NUM_Q_VT    = 8
        params.BEVelocitySpace.NUM_Q_VP    = 4
        params.BEVelocitySpace.NUM_Q_CHI   = 64
        params.BEVelocitySpace.NUM_Q_PHI   = 2

        collisions.AR_NEUTRAL_N=3.22e22
        collisions.MAXWELLIAN_N=1e18
        collisions.AR_IONIZED_N=1e18

        cf_sp    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)
        spec_sp  = cf_sp._spec
        sph_modes = params.BEVelocitySpace.SPH_HARM_LM
        num_p     = spec_sp._p +1
        num_sh    = len(params.BEVelocitySpace.SPH_HARM_LM)
        
        for g in collision:
            g.reset_scattering_direction_sp_mat()
        
        #L_list = list()

        fname="g"
        for g in collision:
            fname = fname + str(g._type)
    
        num_cols = 4
        fig, axs = plt.subplots(int(np.ceil(num_p*num_sh/num_cols)), num_cols)
        fig.set_size_inches(20,8)
    
        x_ev = np.linspace(0.975,1.0,5)
        L_fd = np.zeros_like(x_ev)
        for i,ev in enumerate(x_ev):
            
            collisions.MAXWELLIAN_TEMP_K   = ev * collisions.TEMP_K_1EV
            collisions.ELECTRON_THEMAL_VEL = collisions.electron_thermal_velocity(collisions.MAXWELLIAN_TEMP_K) 
            VTH = collisions.ELECTRON_THEMAL_VEL

            TEV = collisions.MAXWELLIAN_TEMP_K * collisions.BOLTZMANN_CONST / collisions.ELECTRON_VOLT
            print("temp (ev): ", TEV, " VTH : ",VTH)

            maxwellian = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)
            L = spec_sp.create_mat()
            for g in collision:
                L +=cf_sp.assemble_mat(g,maxwellian,VTH)

            W,Q  = np.linalg.eig(L)
            # plt.figure(1)
            # plt.plot(np.abs(W),label="eV=%.4f"%ev)

            dt=1e-10
            E    = np.real(np.matmul(np.matmul(np.linalg.inv(Q),np.diag(np.exp(W*dt))),Q))
            for r in range(num_p):
                for c in range(num_sh):
                    rid   = r*num_sh + c
                    plt_r = rid//num_cols
                    plt_c = rid%num_cols

                    # axs[plt_r,plt_c].plot(range(len(Q[:,rid])),np.real(Q[:,rid]),label="Re,ev=%.3f"%TEV)
                    # axs[plt_r,plt_c].plot(range(len(Q[:,rid])),np.imag(Q[:,rid]),label ="Im,ev=%.3f"%TEV)
                    axs[plt_r,plt_c].plot(range(len(E[:,rid])),E[:,rid],label="dt=%.2E"%(dt))
                    axs[plt_r,plt_c].set_title("(%d,%d,%d) = %.2E" %(r,sph_modes[c][0],sph_modes[c][1],np.abs(W[r*num_sh + c])),fontsize=7)
                    axs[plt_r,plt_c].tick_params(axis='both', which='major', labelsize=7)
                    axs[plt_r,plt_c].grid()
                    axs[plt_r,plt_c].set_yscale("log")
                    axs[0,0].legend()
        
            #fname1=fname + f"_Nr_%d_eig_vec_%.2E_NI_%.2E_N0_%.2E_NE_%.2E.png" %(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,TEV,collisions.AR_IONIZED_N,collisions.AR_NEUTRAL_N,collisions.MAXWELLIAN_N)
            #plt.tight_layout()
            #plt.savefig(fname1)
            #plt.show()



            
    # plt.figure(1)    
    # plt.xlabel("eigenmode")
    # plt.ylabel("magnitude")
    # plt.legend()
    # plt.grid()

    plt.show()

def q_mode_gmx_simpson_test(collision,ev=1.0):

    collisions.AR_NEUTRAL_N=3.22e22
    collisions.MAXWELLIAN_N=1e18
    collisions.AR_IONIZED_N=1e18

    collisions.MAXWELLIAN_TEMP_K   = ev * collisions.TEMP_K_1EV
    collisions.ELECTRON_THEMAL_VEL = collisions.electron_thermal_velocity(collisions.MAXWELLIAN_TEMP_K) 
    VTH = collisions.ELECTRON_THEMAL_VEL

    maxwellian = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)

    params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER=32
    params.BEVelocitySpace.SPH_HARM_LM = [[0,0]]
    params.BEVelocitySpace.NUM_Q_VT    = 4
    params.BEVelocitySpace.NUM_Q_VP    = 4
    params.BEVelocitySpace.NUM_Q_CHI   = 4
    params.BEVelocitySpace.NUM_Q_PHI   = 4

    TEV = collisions.MAXWELLIAN_TEMP_K * collisions.BOLTZMANN_CONST / collisions.ELECTRON_VOLT
    print("temp (ev): ", TEV, " VTH : ",VTH)

    EIGEN_DECOMP_TOL=1e-6
    MNE = collisions.MAXWELLIAN_N
    NR        = [4]
    num_steps = 300
    T_TOTAL   = 1e-6

    for g in collision:
        g.reset_scattering_direction_sp_mat()

    params.BEVelocitySpace.NUM_Q_VR    = 118    
    cf_sp_gmx    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,q_mode=sp.QuadMode.GMX)
    spec_sp_gmx  = cf_sp_gmx._spec

    L_gmx = spec_sp_gmx.create_mat()
    for g in collision:
        g.reset_scattering_direction_sp_mat()
        L_gmx +=cf_sp_gmx.assemble_mat(g,maxwellian,VTH)

    params.BEVelocitySpace.NUM_Q_VR    = 401
    cf_sp_smp    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,q_mode=sp.QuadMode.SIMPSON)
    spec_sp_smp  = cf_sp_smp._spec
    

    L_smp = spec_sp_smp.create_mat()
    for g in collision:
        g.reset_scattering_direction_sp_mat()
        L_smp +=cf_sp_smp.assemble_mat(g,maxwellian,VTH)
    
    print("||L_smp-L_gmx||/||L_gmx|| = %.16E "%(np.linalg.norm(L_smp-L_gmx)/np.linalg.norm(L_gmx)))

def plot_synthetic_cs():
    import matplotlib.pyplot as plt
    collisions.AR_NEUTRAL_N=3.22e22
    collisions.MAXWELLIAN_N=1e18
    collisions.AR_IONIZED_N=1e18

    collisions.MAXWELLIAN_TEMP_K   = 1.0 * collisions.TEMP_K_1EV
    collisions.ELECTRON_THEMAL_VEL = collisions.electron_thermal_velocity(collisions.MAXWELLIAN_TEMP_K) 
    VTH = collisions.ELECTRON_THEMAL_VEL

    maxwellian = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)
    gx,gw      = basis.uniform_simpson((0,10),1601)
    v          = gx * VTH
    energy_ev  = (0.5 * collisions.MASS_ELECTRON * (v**2))/collisions.ELECTRON_VOLT
    # with np.printoptions(threshold=np.inf):
    #     print(energy_ev)
    t_cs       = g0._total_cs_interp1d(energy_ev)
    plt.plot(energy_ev,t_cs,label="ar")
    for m in range(0,10):
        t_cs       = collisions.Collisions.synthetic_tcs(energy_ev,m)
        plt.plot(energy_ev,t_cs,label="f_%d"%m)
    
    
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("energy (eV)")
    plt.ylabel("cross section (m^2)")
    plt.legend()
    plt.grid()
    plt.show()

def plot_maxwell_poly():
    basis_p = basis.Maxwell()
    x=np.linspace(0,6,10000)
    import matplotlib.pyplot as plt

    for d in range(3,7,1):
        m  = 1<<d
        pm = basis_p.Pn(m)
        plt.plot(x,pm(x),label="d=%d"%m)
    
    plt.xlabel("x")
    plt.ylabel("P(x)")
    #plt.yscale("log")
    plt.grid()
    plt.legend()
    #plt.savefig("maxpoly.png")
    plt.show()

def test_scattering():
    collisions.AR_NEUTRAL_N=3.22e22
    collisions.MAXWELLIAN_N=1e18
    collisions.AR_IONIZED_N=1e18

    ev = 1.0
    collisions.MAXWELLIAN_TEMP_K   = ev * collisions.TEMP_K_1EV
    collisions.ELECTRON_THEMAL_VEL = collisions.electron_thermal_velocity(collisions.MAXWELLIAN_TEMP_K) 
    VTH = 1#collisions.ELECTRON_THEMAL_VEL

    maxwellian = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)

    params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER=32
    params.BEVelocitySpace.SPH_HARM_LM = [[0,0]]
    params.BEVelocitySpace.NUM_Q_VT    = 2
    params.BEVelocitySpace.NUM_Q_VP    = 2
    params.BEVelocitySpace.NUM_Q_CHI   = 2
    params.BEVelocitySpace.NUM_Q_PHI   = 2

    params.BEVelocitySpace.NUM_Q_VR    = 4
    cf_sp_gmx    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,q_mode=sp.QuadMode.GMX)
    spec_sp_gmx  = cf_sp_gmx._spec

    incident_mg   = cf_sp_gmx._incident_mg_plus
    scattering_mg = cf_sp_gmx._scattering_mg_plus
    g  = collisions.eAr_G0()
    g.reset_scattering_direction_sp_mat()
    sd = g.compute_scattering_direction_sp(scattering_mg[0],scattering_mg[1],scattering_mg[2],scattering_mg[3],scattering_mg[4])
    #print(incident_mg)
    #print(sd)
    # v0=np.zeros(tuple([3])+incident_mg[0].shape)
    # v0[0] = incident_mg[0] * np.sin(incident_mg[1]) * np.cos(incident_mg[2])
    # v0[1] = incident_mg[0] * np.sin(incident_mg[1]) * np.sin(incident_mg[2])
    # v0[2] = incident_mg[0] * np.cos(incident_mg[1]) 

    # g.compute_scattering_direction()
    v0 = np.zeros(3)
    v1 = np.zeros(3)

    error = np.zeros_like(scattering_mg[0])
    
    for vr in range(params.BEVelocitySpace.NUM_Q_VR):
        for vt in range(params.BEVelocitySpace.NUM_Q_VT):
            for vp in range(params.BEVelocitySpace.NUM_Q_VP):
                for s1 in range(params.BEVelocitySpace.NUM_Q_CHI):
                    for s2 in range(params.BEVelocitySpace.NUM_Q_PHI):
                        v0[0] = scattering_mg[0][vr,vt,vp,s1,s2] * np.sin(scattering_mg[1][vr,vt,vp,s1,s2]) * np.cos(scattering_mg[2][vr,vt,vp,s1,s2])
                        v0[1] = scattering_mg[0][vr,vt,vp,s1,s2] * np.sin(scattering_mg[1][vr,vt,vp,s1,s2]) * np.sin(scattering_mg[2][vr,vt,vp,s1,s2])
                        v0[2] = scattering_mg[0][vr,vt,vp,s1,s2] * np.cos(scattering_mg[1][vr,vt,vp,s1,s2])

                        v1[0] = sd[0][vr,vt,vp,s1,s2] * np.sin(sd[1][vr,vt,vp,s1,s2]) * np.cos(sd[2][vr,vt,vp,s1,s2])
                        v1[1] = sd[0][vr,vt,vp,s1,s2] * np.sin(sd[1][vr,vt,vp,s1,s2]) * np.sin(sd[2][vr,vt,vp,s1,s2])
                        v1[2] = sd[0][vr,vt,vp,s1,s2] * np.cos(sd[1][vr,vt,vp,s1,s2])

                        vs    = g.compute_scattering_direction(v0,scattering_mg[3][vr,vt,vp,s1,s2],scattering_mg[4][vr,vt,vp,s1,s2])

                        #print("vs: ",vs," v1: ",v1)
                        # print("vIn = %s s1 = %.4E s2 = %.4E \t vOut1 = %s \t vOut2 = %s " %(np.array_str(v0,precision=8),scattering_mg[3][vr,vt,vp,s1,s2],scattering_mg[4][vr,vt,vp,s1,s2],np.array_str(vs,precision=8),np.array_str(v1,precision=8)))
                        error[vr,vt,vp,s1,s2] = np.linalg.norm(vs-v1)
    
    error=error.reshape(-1)
    print("np.allclose(error,np.zeros_like(error): ",np.allclose(error,np.zeros_like(error)))

g0  = collisions.eAr_G0()
g1  = collisions.eAr_G1()
g2  = collisions.eAr_G2()
g0_p = collisions.eAr_G0_NoEnergyLoss()

#q_mode_gmx_simpson_test([g0],1.0)
#collision_op_conv([g0],1.0)    
#spec_convergence_collision_op([g0],1.0)

#plot_synthetic_cs()
#plot_maxwell_poly()
#test_scattering()

#collision_op_test(1.0)
plot_crosssection(g2,118,1)




