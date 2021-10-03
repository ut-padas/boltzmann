"""
@package Boltzmann collision operator solver. 
"""

import enum

import scipy
import basis
import spec_spherical as sp
import numpy as np
import collision_operator_spherical as colOpSp
import collisions 
import parameters as params
import os
import ets
import utils as BEUtils
import profiler
import argparse
import scipy.integrate
import pyevtk
t_M = profiler.profile_t("mass_assembly")
t_L = profiler.profile_t("collOp_assembly")
t_ts = profiler.profile_t("ts")


collisions.AR_NEUTRAL_N=3.22e22
collisions.MAXWELLIAN_N=1e6
collisions.AR_IONIZED_N=3.22e22#collisions.MAXWELLIAN_N



if not os.path.exists('plots'):
    print("creating folder `plots`, output will be written into it")
    os.makedirs('plots')

def eigen_solve_coll_op(FOp,h_init):

    W,Q  = np.linalg.eig(FOp)
    # print("Eigenvalues: ")
    # print(W)
    Qinv = np.linalg.inv(Q)
    # AA = np.matmul(np.matmul(Q,np.diag(W)),Qinv)
    # print((FOp-AA)/np.linalg.norm(FOp))

    y_init = np.dot(Qinv,h_init)

    q_norm=np.array([np.linalg.norm(Q[:,i]) for i in range(Q.shape[1])])
    print("Eigenvectors normalized : ", np.allclose(q_norm,np.ones_like(q_norm)))

    fun_sol = np.array([y_init[i] * Q[:,i] for i in range(Q.shape[1])])
    fun_sol = np.transpose(fun_sol)

    print("Scaled for the initial condition (t=0) : " ,np.allclose(fun_sol.sum(axis=1),h_init))
    # print(W)
    # print(np.transpose(fun_sol))
    return W,fun_sol

def ode_numerical_solve(collOp:colOpSp.CollisionOpSP, col_list, h_init, maxwellian, vth, t_end, dt , ml_tol):
    MVTH  = vth
    MNE   = maxwellian(0) * (np.sqrt(np.pi)**3) * (vth**3)
    MTEMP = collisions.electron_temperature(MVTH)
    IO_COUT_FEQ=5

    ts_integrator = ets.ExplicitODEIntegrator(ets.TSType.FORWARD_EULER)
    ts_integrator.set_ts_size(dt)
    
    
    vth_prev = MVTH
    vth_curr = MVTH
    spec_sp = collOp._spec
    h_t = np.array(h_init)
    ev_pts     = np.linspace(0,50,1000) 
    m_ht=h_init[0]
    mw_prev   = BEUtils.get_maxwellian_3d(vth_prev,MNE)
    m0_t0     = BEUtils.moment_n_f(spec_sp,h_t,mw_prev,vth_prev,0,None,None,None,1)
    temp_t0   = BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec_sp,h_t,mw_prev,vth_prev,None,None,None,m0_t0,1)

    while ts_integrator.current_ts()[0] < t_end:
        ts_info = ts_integrator.current_ts()
        t_step  = ts_info[1]
        if(t_step == 0):
            fout = open(OUTPUT_FILE_NAME, "w")
            print(f"T\tNe_T\tNe_0\tRE_Ne\tTmp_T\tTmp_0",file=fout)
            fout.close()
        
        fout   = open(OUTPUT_FILE_NAME, "a")
        t_curr = ts_info[0] 
        mw_prev   = BEUtils.get_maxwellian_3d(vth_prev,MNE)
        m0        = BEUtils.moment_n_f(spec_sp,h_t,mw_prev,vth_prev,0,None,None,None,1)
        temp_curr = BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec_sp,h_t,mw_prev,vth_prev,None,None,None,m0,1)
        vth_curr  = collisions.electron_thermal_velocity(temp_curr)
        mw_curr   = BEUtils.get_maxwellian_3d(vth_curr,MNE)

        vth_g0     = np.sqrt(1-2*collisions.MASS_R_EARGON) * vth_curr 
        
        print(f"dt={dt:.2E} N_e(t={t_curr:.2E})={m0:.12E} maxwellian N_e={m0_t0:.12E} relative error: {(m0_t0-m0)/m0_t0:.12E} temperature(K)={temp_curr:.12E} ")
        if(t_step % IO_COUT_FEQ ==0):
            fout = open(OUTPUT_FILE_NAME, "a")
            dat_ht = np.append(np.array([t_curr,m0,(m0_t0-m0)/m0_t0,temp_curr]),h_t)
            np.savetxt(fout,dat_ht,newline=" ")
            print("",file=fout)
            fout.close()
        
        print("vth_prev = %.12E  vth_curr = %.12E" %(vth_prev,vth_curr))
        mm_h1  = BEUtils.compute_Mvth1_Pi_vth2_Pj_vth1(spec_sp, mw_prev, vth_prev, mw_curr, vth_curr, None, None, None, 1)
        h_t    = np.dot(mm_h1,h_t)

        t_L.start()
        FOp    = spec_sp.create_mat()
        for g in col_list:
            FOp += collOp.assemble_mat(g,mw_curr,vth_curr)
        t_L.stop()
        print("Assembling the collision op. for Vth : ", vth_curr)
        print("Collision Operator assembly time (s): ",t_L.snap)

        def f_rhs(f,t):
            return np.matmul(FOp,f)

        ts_integrator.set_rhs_func(f_rhs)

        h_t = ts_integrator.evolve(h_t)
        
        mw_prev   = mw_curr
        vth_prev  = vth_curr
        temp_prev = temp_curr



def ode_first_order_linear(collOp:colOpSp.CollisionOpSP, col_list, h_init, maxwellian, vth, t_end, dt , ml_tol):
    MVTH  = vth
    MNE   = maxwellian(0) * (np.sqrt(np.pi)**3) * (vth**3)
    MTEMP = collisions.electron_temperature(MVTH)
    COL_OP_TEMP_TOL = 1e-2
    COL_OP_VTH_TOL  = 1e-5
    COL_OP_FREQ     = 1
    MASS_LOSS_TOL   = ml_tol
    IO_COUT_FEQ     = 5

    dt_tau = 1/collisions.PLASMA_FREQUENCY
    print("==========================================================================")

    t_ts.start()
    h_t = np.array(h_init)
    spec_sp = collOp._spec
    print("Data output : ", OUTPUT_FILE_NAME)
    
    
    t_curr = 0.0
    t_step = 0

    vth_prev = MVTH
    vth_curr = MVTH

    # num_pts = 100
    # X = np.linspace(-3,3,num_pts)
    # Y = np.linspace(-3,3,num_pts)
    # Z = np.linspace(-3,3,num_pts)
    # #Z = np.array([0.5])
    # X,Y,Z=np.meshgrid(X,Y,Z,indexing='ij')
    # VTK_IO_FREQ=5

    # R     = np.sqrt(X**2 + Y**2 + Z**2)
    # THETA = np.arccos(Z/R)
    # PHI   = np.arctan(Y/X)
    # sph_modes = spec_sp._sph_harm_lm
    # num_p     = spec_sp._p + 1
    # num_sh    = len(sph_modes)
    # P_klm = np.array([ maxwellian(R) *spec_sp.basis_eval_full(R,THETA,PHI,k,sph_modes[lm_i][0],sph_modes[lm_i][1]) for k in range(num_p) for lm_i in range(num_sh)])
    # P_klm = P_klm.reshape(num_p*num_sh,-1)
    # P_klm = np.transpose(P_klm)
    # point_data=dict()
    
    ev_pts     = np.linspace(0,50,1000) 
    m_ht=h_init[0]
    mw_prev   = BEUtils.get_maxwellian_3d(vth_prev,MNE)
    m0_t0     = BEUtils.moment_n_f(spec_sp,h_t,mw_prev,vth_prev,0,None,None,None,1)
    temp_t0   = BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec_sp,h_t,mw_prev,vth_prev,None,None,None,m0_t0,1)

    while(t_curr < t_end):
        
        if(t_step == 0):
            fout = open(OUTPUT_FILE_NAME, "w")
            print(f"T\tNe_T\tRE_Ne\tTmp_T\tC_klm[%d,%s]"%(spec_sp._p,spec_sp._sph_harm_lm),file=fout)
            fout.close()
        
        

        mw_prev   = BEUtils.get_maxwellian_3d(vth_prev,MNE)
        m0        = BEUtils.moment_n_f(spec_sp,h_t,mw_prev,vth_prev,0,None,None,None,1)
        temp_curr = BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec_sp,h_t,mw_prev,vth_prev,None,None,None,m0,1)
        vth_curr  = collisions.electron_thermal_velocity(temp_curr)
        mw_curr   = BEUtils.get_maxwellian_3d(vth_curr,MNE)

        print(f"dt={dt:.2E} N_e(t={t_curr:.2E})={m0:.12E} maxwellian N_e={m0_t0:.12E} relative error: {(m0_t0-m0)/m0_t0:.12E} temperature(K)={temp_curr:.12E} ")
        
        if(t_step % IO_COUT_FEQ ==0):
            fout = open(OUTPUT_FILE_NAME, "a")
            dat_ht = np.append(np.array([t_curr,m0,(m0_t0-m0)/m0_t0,temp_curr]),h_t)
            np.savetxt(fout,dat_ht,newline=" ")
            print("",file=fout)
            fout.close()

        print("vth_prev = %.12E  vth_curr = %.12E" %(vth_prev,vth_curr))
        mm_h1  = BEUtils.compute_Mvth1_Pi_vth2_Pj_vth1(spec_sp, mw_prev, vth_prev, mw_curr, vth_curr, None, None, None, 1)
        h_t    = np.dot(mm_h1,h_t)
        #h_t[0] = m_ht

        # if(t_step % VTK_IO_FREQ  is 0):
        #     point_data["f(v)"] = np.matmul(P_klm,h_t)
        #     pyevtk.hl.gridToVTK(OUTPUT_FILE_NAME+"_vtk_%04d"%(t_step//VTK_IO_FREQ), X, Y, Z, cellData = None, pointData = point_data,fieldData=None)

        t_L.start()
        FOp    = spec_sp.create_mat()
        for g in col_list:
            FOp += collOp.assemble_mat(g,mw_curr,vth_curr)
        t_L.stop()
        print("Assembling the collision op. for Vth : ", vth_curr)
        print("Collision Operator assembly time (s): ",t_L.snap)

        W,fun_sol = eigen_solve_coll_op(FOp, h_t)

        #check the time step size, 
        # while(True):
        #     c_t = np.exp(W*dt)
        #     h_test     =  np.dot(fun_sol,c_t)
        #     m0_test    = BEUtils.moment_zero_f(spec_sp,h_test,mw_curr,vth_curr,None,None,None,1)
        #     temp_test  = BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec_sp,h_test,mw_curr,vth_curr,None,None,None,m0,1)
        #     vth_test   = collisions.electron_thermal_velocity(temp_test)
        #     #if (np.abs(m0_test-m0)> MASS_LOSS_TOL or vth_g0>vth_test):
        #     if (np.abs(m0_test-m0)> MASS_LOSS_TOL):
        #         dt=dt/2
        #     else:
        #         break
        c_t = np.exp(W*dt)
        h_t = np.dot(fun_sol,c_t) #h_test
        assert np.allclose(np.imag(h_t),np.zeros(h_t.shape)) , "imaginary part is not close to zero"
        h_t   = np.real(h_t)
        #h_t[0]   = m_ht

        mw_prev   = mw_curr
        vth_prev  = vth_curr
        temp_prev = temp_curr
        
        t_curr+=dt
        t_step+=1
        #dt=dt*2
    
    
    t_ts.stop()

    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, 2)
    # axs[0].plot(ts,mass_ts,'b-o')
    # axs[0].set_title("n(0)= %.2E " %(MNE))
    # axs[0].set_xlabel('time (s)')
    # axs[0].set_ylabel('mass loss (n(t)-n(0))/n(0)')

    # axs[1].plot(ts,temp_ts,'b-o')
    # axs[1].set_title("T(0)= %.2E K " %(MTEMP))
    # axs[1].set_xlabel('time (s)')
    # axs[1].set_ylabel('Temperature (K) ')
    # plt.tight_layout()
    # plt.show()
    # plt.close()
    print("Collision Operator assembly total : (s): ",t_L.seconds)
    print("Total ts time : (s): ",t_ts.seconds)

parser = argparse.ArgumentParser()
parser.add_argument("-Nr", "--NUM_P_RADIAL", help="Number of polynomials in radial direction", type=int, default=4)
parser.add_argument("-Te", "--T_END", help="Simulation time", type=float, default=1e-14)
parser.add_argument("-dt", "--T_DT", help="Simulation time step size ", type=float, default=1e-15)
parser.add_argument("-o",  "--out_fname", help="output file name", type=str, default='out.dat')
parser.add_argument("-m_tol",  "--mass_tol", help="mass loss tolerance", type=float, default=1e-10)
args = parser.parse_args()


params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER = args.NUM_P_RADIAL
params.BEVelocitySpace.SPH_HARM_LM = [[0,0],[1,0]]
params.BEVelocitySpace.NUM_Q_VR  = 21
params.BEVelocitySpace.NUM_Q_VT  = 16
params.BEVelocitySpace.NUM_Q_VP  = 16
params.BEVelocitySpace.NUM_Q_CHI = 64
params.BEVelocitySpace.NUM_Q_PHI = 16
params.BEVelocitySpace().VELOCITY_SPACE_DT = args.T_DT
NUM_T_STEPS      = int(args.T_END/args.T_DT)
OUTPUT_FILE_NAME = args.out_fname
colOpSp.SPEC_SPHERICAL  = sp.SpectralExpansionSpherical(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,basis.Maxwell(),params.BEVelocitySpace.SPH_HARM_LM) 

# instance of the collision operator
cf    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)
spec  = colOpSp.SPEC_SPHERICAL
VTH   = collisions.ELECTRON_THEMAL_VEL

print("""===========================Parameters ======================""")
print("\tMAXWELLIAN_N : ", collisions.MAXWELLIAN_N)
print("\tELECTRON_THEMAL_VEL : ", VTH," ms^-1")
print("\tDT : ", params.BEVelocitySpace().VELOCITY_SPACE_DT, " s")
print("""============================================================""")
params.print_parameters()


maxwellian = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)
hv         = lambda v,vt,vp : np.ones_like(v) #* collisions.MAXWELLIAN_N
h_vec      = BEUtils.compute_func_projection_coefficients(spec,hv,maxwellian,None,None,None)

col_g0_no_E_loss = collisions.eAr_G0_NoEnergyLoss()
col_g0 = collisions.eAr_G0()
col_g1 = collisions.eAr_G1()
col_g2 = collisions.eAr_G2()

t_M.start()
M  = spec.compute_maxwellian_mm(maxwellian,VTH)
t_M.stop()
#print(M)
print("Mass assembly time (s): ", t_M.seconds)
#ode_first_order_linear(cf,[col_g0_no_E_loss],h_vec,maxwellian,VTH,args.T_END,args.T_DT,args.mass_tol)
ode_first_order_linear(cf,[col_g0,col_g2],h_vec,maxwellian,VTH,args.T_END,args.T_DT,args.mass_tol)
#ode_numerical_solve(cf,[[col_g0,1],[col_g1,1]],h_vec,maxwellian,VTH,args.T_END,args.T_DT,args.mass_tol)
#ode_numerical_solve(cf,[[col_g0,1]],h_vec,maxwellian,VTH,args.T_END,args.T_DT,args.mass_tol)
