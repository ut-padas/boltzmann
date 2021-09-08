"""
@package Boltzmann collision operator solver. 
"""

import enum
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

t_M = profiler.profile_t("mass_assembly")
t_L = profiler.profile_t("collOp_assembly")
t_ts = profiler.profile_t("ts")

if not os.path.exists('plots'):
    print("creating folder `plots`, output will be written into it")
    os.makedirs('plots')

def eigen_solve_coll_op(FOp,h_init):

    W,Q  = np.linalg.eig(FOp)
    # print("Eigenvalues: ")
    # print(W)

    Qinv = np.linalg.inv(Q)
    y_init = np.dot(Qinv,h_init)

    q_norm=np.array([np.linalg.norm(Q[:,i]) for i in range(Q.shape[1])])
    print("Eigenvectors normalized : ", np.allclose(q_norm,np.ones_like(q_norm)))

    fun_sol = np.array([y_init[i] * Q[:,i] for i in range(Q.shape[1])])
    fun_sol = np.transpose(fun_sol)

    print("Scaled for the initial condition (t=0) : " ,np.allclose(fun_sol.sum(axis=1),h_init))
    
    return W,fun_sol

def ode_first_order_linear(collOp:colOpSp.CollisionOpSP, col_list, h_init, maxwellian, vth, t_end, dt):
    MVTH  = vth
    MNE   = maxwellian(0) * (np.sqrt(np.pi)**3) * (vth**3)
    MTEMP = collisions.electron_temperature(MVTH)
    COL_OP_TEMP_TOL = 1e-2
    COL_OP_VTH_TOL = 1e-3

    dt_tau = 1/params.PLASMA_FREQUENCY
    print("==========================================================================")
    vth_t = float(MVTH)
    vth_coll_op  = vth_t
    temp_coll_op = MVTH
    # mass_ts=np.zeros_like(ts)
    # temp_ts=np.zeros_like(ts)

    t_ts.start()

    h_t = np.array(h_init)
    
    t_L.start()
    mw_cop = BEUtils.get_maxwellian_3d(vth_coll_op,1)
    spec_p = collOp._spec
    FOp    = spec_p.create_mat()
    for g in col_list:
        FOp+=collOp.assemble_mat(g,mw_cop,vth_coll_op)
    t_L.stop()
    print("Collision Operator assembly time (s): ",t_L.snap)
    W,fun_sol = eigen_solve_coll_op(FOp, h_t)

    #print("coll_op_vth ", vth_coll_op, "vth _curr: ", vth_t)
    print("Data output : ", OUTPUT_FILE_NAME)
    fout = open(OUTPUT_FILE_NAME, "w")
    print(f"T\tNe_T\tNe_0\tRE_Ne\tTmp_T\tTmp_0",file=fout)
    
    t_curr =0
    while(t_curr < t_end):
        mw_t   = BEUtils.get_maxwellian_3d(vth_t,1)
        m0     = BEUtils.moment_zero_f(spec_p,h_t,mw_t,vth_t,None,None,None,1)
        temp_t = BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec_p,h_t,mw_t,vth_t,None,None,None,m0,1)
        vth_t  = collisions.electron_thermal_velocity(temp_t)

        if((np.abs(vth_t-vth_coll_op)/vth_coll_op)> COL_OP_VTH_TOL):
            spec_sp = collOp._spec
            print("coll_op_vth ", vth_coll_op, "vth _curr: ", vth_t)
            
            mm_h1  = BEUtils.compute_Mvth1_Pi_vth2_Pj_vth1(spec_sp, mw_cop, vth_coll_op, mw_t, vth_t, None, None, None, 1)
            #print("before ", h_t)
            h_t    = np.dot(mm_h1,h_t)
            #print("after ", h_t)
            
            print("Assembling the collision op. for temperature : ", temp_t)
            t_L.start()
            vth_coll_op = vth_t
            temp_coll_op = temp_t
            mw_cop = BEUtils.get_maxwellian_3d(vth_coll_op,1)
            FOp    = spec_sp.create_mat()
            for g in col_list:
                FOp+=collOp.assemble_mat(g,mw_cop,vth_coll_op)
            t_L.stop()
            print("Collision Operator assembly time (s): ",t_L.snap)
            W,fun_sol = eigen_solve_coll_op(FOp, h_t)
            
        c_t = np.exp(W*t_curr)
        h_t = np.dot(fun_sol,c_t)
        if not np.allclose(np.imag(h_t),np.zeros_like(h_t)):
            print("non-zero imag coefficients ")
            print(np.imag(h_t))
        h_t   = np.real(h_t)
        # print("W>0 : ", c_t[W>0])
        # print("W<0 : ", c_t[W<0])
        
        #print(" imag part close to zero : ", np.allclose(np.imag(h_t),np.zeros(h_t.shape)))
        assert np.allclose(np.imag(h_t),np.zeros(h_t.shape)) , "imaginary part is not close to zero"
        # mass_ts[t_i] = (MNE-m0)/MNE
        # temp_ts[t_i] = temp_t
        print(f"N_e(t={t_curr:.2E})={m0:.10E} maxwellian N_e={MNE:.2E} relative error: {(MNE-m0)/MNE:.8E} t/dt={t_curr/dt_tau:.2E} temperature(K)={temp_t:.5E} maxwellian temp(K)={MTEMP:.5E}")
        print(f"{t_curr:.6E}\t{m0:.10E}\t{MNE:.2E}\t{(MNE-m0)/MNE:.10E}\t{temp_t:.8E}\t{MTEMP:.8E}",file=fout)
        t_curr+=dt
    
    fout.close()
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

print("""===========================Parameters ======================""")
print("\tMAXWELLIAN_N : ", collisions.MAXWELLIAN_N)
print("\tELECTRON_THEMAL_VEL : ", collisions.ELECTRON_THEMAL_VEL," ms^-1")
print("\tDT : ", params.BEVelocitySpace().VELOCITY_SPACE_DT, " s")
print("""============================================================""")
params.print_parameters()

maxwellian = BEUtils.get_maxwellian_3d(collisions.ELECTRON_THEMAL_VEL,1)
hv         = lambda v,vt,vp : np.ones_like(v)
h_vec      = BEUtils.compute_func_projection_coefficients(spec,hv,maxwellian,None,None,None)

col_g0 = collisions.eAr_G0()
col_g0_no_E_loss = collisions.eAr_G0_NoEnergyLoss()
#col_g1 = collisions.eAr_G1()
#col_g2 = collisions.eAr_G2()

t_M.start()
M  = spec.compute_maxwellian_mm(maxwellian,collisions.ELECTRON_THEMAL_VEL)
t_M.stop()
print("Mass assembly time (s): ", t_M.seconds)
ode_first_order_linear(cf,[col_g0],h_vec,maxwellian,collisions.ELECTRON_THEMAL_VEL,args.T_END,args.T_DT)
