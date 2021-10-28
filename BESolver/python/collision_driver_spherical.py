"""
@package Boltzmann collision operator solver. 
"""

import enum
from math import fabs

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
collisions.MAXWELLIAN_N=1e18
collisions.AR_IONIZED_N=3.22e22 #collisions.MAXWELLIAN_N

if not os.path.exists('plots'):
    print("creating folder `plots`, output will be written into it")
    os.makedirs('plots')

def eigen_solve_coll_op(FOp,h_init):

    W,Q  = np.linalg.eig(FOp)
    # print(FOp)
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
    #print(fun_sol)
    

    print("Scaled for the initial condition (t=0) : " ,np.allclose(fun_sol.sum(axis=1),h_init))
    # print(W)
    # print(np.transpose(fun_sol))
    return W,fun_sol

def ode_first_order_linear_adaptive_v1(collOp:colOpSp.CollisionOpSP, col_list, h_init, maxwellian, vth, t_end, dt, ts_tol):
    spec_sp = collOp._spec
    MVTH  = vth
    MNE   = maxwellian(0) * (np.sqrt(np.pi)**3) * (vth**3)
    MTEMP = collisions.electron_temperature(MVTH)
    COL_OP_TEMP_TOL     = ts_tol
    TAIL_NORM_TOLERANCE = ts_tol
    TAIL_NORM_INDEX = (spec_sp._p+1) * len(spec_sp._sph_harm_lm) // 2
    IO_COUT_FEQ     = 1

    dt_tau = 1/collisions.PLASMA_FREQUENCY
    print("==========================================================================")

    t_ts.start()
    h_t = np.array(h_init)
    print("Data output : ", OUTPUT_FILE_NAME)
    t_curr = 0.0
    t_step = 0
    
    tail_norm = lambda x, i: np.linalg.norm(x[i:],ord=2)/np.linalg.norm(x,ord=2)
    
    vth_curr  = vth 
    mw_curr   = BEUtils.get_maxwellian_3d(vth_curr,MNE)
    m0_t0     = BEUtils.moment_n_f(spec_sp,h_t,mw_curr,vth_curr,0,None,None,None,1)

    while(t_curr < t_end):
        mw_curr   = BEUtils.get_maxwellian_3d(vth_curr,MNE)
        m0        = BEUtils.moment_n_f(spec_sp,h_t,mw_curr,vth_curr,0,None,None,None,1)
        temp_curr = BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec_sp,h_t,mw_curr,vth_curr,None,None,None,m0,1)
        
        print(f"N_e(t={t_curr:.2E})={m0:.12E} maxwellian N_e={m0_t0:.12E} relative error: {(m0_t0-m0)/m0_t0:.12E} temperature(eV)={temp_curr*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT:.12E} tail={tail_norm(h_t,TAIL_NORM_INDEX):.12E}")
        
        if(t_step == 0):
            fout = open(OUTPUT_FILE_NAME, "w")
            print(f"T\tNe_T\tRE_Ne\tTmp_T\tC_klm[%d,%s]"%(spec_sp._p,spec_sp._sph_harm_lm),file=fout)
            fout.close()
        
        if(t_step % IO_COUT_FEQ ==0):
            fout = open(OUTPUT_FILE_NAME, "a")
            dat_ht = np.append(np.array([t_curr,m0,(m0_t0-m0)/m0_t0,temp_curr]),h_t)
            np.savetxt(fout,dat_ht,newline=" ")
            print("",file=fout)
            fout.close()
        
        # Assemble the collision operator
        t_L.start()
        FOp    = spec_sp.create_mat()
        for g in col_list:
            FOp += collOp.assemble_mat(g,mw_curr,vth_curr)
        t_L.stop()
        print("Assembled the collision op. for Vth : ", vth_curr)
        print("Collision Operator assembly time (s): ",t_L.snap)
        W,fun_sol = eigen_solve_coll_op(FOp, h_t)
        ht_zeros = np.zeros(h_t.shape)

        # now try to perform a timestep. 
        fac_dt   = 1
        while True:
            dt_curr  = fac_dt * dt
            c_t      = np.exp(W*dt_curr)
            h_c      = np.dot(fun_sol,c_t)
            # if not we need to use this as a fundemental solution. 
            assert np.allclose(np.imag(h_c),ht_zeros) , "imaginary part is not close to zero"
            h_c     = np.real(h_c)

            temp_c  = BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec_sp,h_c,mw_curr,vth_curr,None,None,None,m0,1)
            vth_c   = collisions.electron_thermal_velocity(temp_c)
            mw_c    = BEUtils.get_maxwellian_3d(vth_c,MNE)
            mm_h1   = BEUtils.compute_Mvth1_Pi_vth2_Pj_vth1(spec_sp, mw_curr, vth_curr, mw_c, vth_c, None, None, None, 1)
            h_cc    = np.dot(mm_h1,h_c)

            if(dt_curr > 1e-5 or  tail_norm(h_cc,TAIL_NORM_INDEX) > TAIL_NORM_TOLERANCE):
                if(fac_dt>1):
                    dt_curr  =  (fac_dt>>1) * dt
                else:
                    dt_curr  =   dt
                c_t      = np.exp(W*dt_curr)
                h_t      = np.dot(fun_sol,c_t)
                h_t      = np.real(h_t)

                temp_c   = BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec_sp,h_c,mw_curr,vth_curr,None,None,None,m0,1)
                vth_c    = collisions.electron_thermal_velocity(temp_c)
                mw_c     = BEUtils.get_maxwellian_3d(vth_c,MNE)
                mm_h1    = BEUtils.compute_Mvth1_Pi_vth2_Pj_vth1(spec_sp, mw_curr, vth_curr, mw_c, vth_c, None, None, None, 1)
                h_t      = np.dot(mm_h1,h_t)
                print("dt_curr = %.2E   (T_new - T) / T = %.6E tail = %.6E"  %(dt_curr,fabs(temp_c-temp_curr)/temp_curr, tail_norm(h_t,TAIL_NORM_INDEX)) )
                break
            else:
                print("dt_curr = %.2E   (T_new - T) / T = %.6E tail = %.6E"  %(dt_curr,fabs(temp_c-temp_curr)/temp_curr, tail_norm(h_t,TAIL_NORM_INDEX)) )
            
            fac_dt=fac_dt<<1
        
        
        vth_curr = vth_c
        t_curr+=dt_curr
        t_step+=1
        
    
    t_ts.stop()
    print("Collision Operator assembly total : (s): ",t_L.seconds)
    print("Total ts time : (s): ",t_ts.seconds)

def ode_first_order_linear(collOp:colOpSp.CollisionOpSP, col_list, h_init, maxwellian, vth, t_end, dt, ts_tol):
    
    spec_sp = collOp._spec
    MVTH  = vth
    MNE   = maxwellian(0) * (np.sqrt(np.pi)**3) * (vth**3)
    MTEMP = collisions.electron_temperature(MVTH)
    COL_OP_TEMP_TOL = ts_tol
    TAIL_NORM_INDEX = (spec_sp._p+1) * len(spec_sp._sph_harm_lm) // 2
    IO_COUT_FEQ     = 1

    dt_tau = 1/collisions.PLASMA_FREQUENCY
    print("==========================================================================")

    t_ts.start()
    h_t = np.array(h_init)
    print("Data output : ", OUTPUT_FILE_NAME)
    t_curr = 0.0
    t_step = 0

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
    
    mw_vth    = BEUtils.get_maxwellian_3d(vth,MNE)
    m0_t0     = BEUtils.moment_n_f(spec_sp,h_t,mw_vth,vth,0,None,None,None,1)
    temp_t0   = BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec_sp,h_t,mw_vth,vth,None,None,None,m0_t0,1)
    vth_curr  = collisions.electron_thermal_velocity(temp_t0) 
    print("Initial Ev : " , temp_t0 * collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT)
    tail_norm = lambda x, i: np.linalg.norm(x[i:],ord=2)/np.linalg.norm(x,ord=2)

    

    while(t_curr < t_end):
        mw_curr   = BEUtils.get_maxwellian_3d(vth_curr,MNE)
        m0        = BEUtils.moment_n_f(spec_sp,h_t,mw_curr,vth_curr,0,None,None,None,1)
        temp_curr = BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec_sp,h_t,mw_curr,vth_curr,None,None,None,m0,1)
        # # mw_curr   = BEUtils.get_maxwellian_3d(vth_curr,MNE)
        # m0        = BEUtils.moment_n_f(spec_sp,h_t,mw_vth,vth,0,None,None,None,1)
        # temp_curr = BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec_sp,h_t,mw_vth,vth,None,None,None,m0,1)
        
        
        print(f"dt={dt:.2E} N_e(t={t_curr:.2E})={m0:.12E} maxwellian N_e={m0_t0:.12E} relative error: {(m0_t0-m0)/m0_t0:.12E} temperature(eV)={temp_curr*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT:.12E} tail={tail_norm(h_t,TAIL_NORM_INDEX):.12E}")
        
        if(t_step == 0):
            fout = open(OUTPUT_FILE_NAME, "w")
            print(f"T\tNe_T\tRE_Ne\tTmp_T\tC_klm[%d,%s]"%(spec_sp._p,spec_sp._sph_harm_lm),file=fout)
            fout.close()
        
        if(t_step % IO_COUT_FEQ ==0):
            fout = open(OUTPUT_FILE_NAME, "a")
            dat_ht = np.append(np.array([t_curr,m0,(m0_t0-m0)/m0_t0,temp_curr]),h_t)
            np.savetxt(fout,dat_ht,newline=" ")
            print("",file=fout)
            fout.close()
        t_L.start()
        FOp    = spec_sp.create_mat()
        mw_curr   = BEUtils.get_maxwellian_3d(vth_curr,MNE)
        for g in col_list:
            FOp += collOp.assemble_mat(g,mw_curr,vth_curr)
        t_L.stop()
        print("Assembled the collision op. for Vth : ", vth_curr)
        print("Collision Operator assembly time (s): ",t_L.snap)
        W,fun_sol = eigen_solve_coll_op(FOp, h_t)
        ht_zeros = np.zeros(h_t.shape)

        # now try to perform a timestep. 
        c_t      = np.exp(W*dt)
        h_c      = np.dot(fun_sol,c_t)
        assert np.allclose(np.imag(h_c),ht_zeros) , "imaginary part is not close to zero"
        h_c      = np.real(h_c)
        temp_c   = BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec_sp,h_c,mw_curr,vth_curr,None,None,None,m0,1)
        vth_c    = collisions.electron_thermal_velocity(temp_c)

        h_t    = h_c
        mw_c   = BEUtils.get_maxwellian_3d(vth_c,MNE)
        print("vth = %.12E changed to vth  = %.12E" %(vth_curr,vth_c))
        mm_h1  = BEUtils.compute_Mvth1_Pi_vth2_Pj_vth1(spec_sp, mw_curr, vth_curr, mw_c, vth_c, None, None, None, 1)
        tn_bf  = tail_norm(h_t,TAIL_NORM_INDEX)
        h_t    = np.dot(mm_h1,h_t)
        tn_af  = tail_norm(h_t,TAIL_NORM_INDEX)
        print("gain in the tail norm : %12E" %((tn_af-tn_bf)/tn_bf))

        vth_curr = vth_c
        t_curr  += dt
        t_step  += 1
        
    
    t_ts.stop()
    print("Collision Operator assembly total : (s): ",t_L.seconds)
    print("Total ts time : (s): ",t_ts.seconds)

def ode_numerical_solve(collOp:colOpSp.CollisionOpSP, col_list, h_init, maxwellian, vth, t_end, dt , ts_tol,restore=0,quasi_neutral=True):
    
    spec_sp = collOp._spec
    MVTH  = vth
    MNE   = maxwellian(0) * (np.sqrt(np.pi)**3) * (vth**3)
    MTEMP = collisions.electron_temperature(MVTH)
    COL_OP_TEMP_TOL = ts_tol
    TAIL_NORM_INDEX = (spec_sp._p+1) * len(spec_sp._sph_harm_lm) // 2
    IO_COUT_FEQ     = 1

    dt_tau = 1/collisions.PLASMA_FREQUENCY
    print("==========================================================================")

    t_ts.start()
    h_t = np.array(h_init)
    print("Data output : ", OUTPUT_FILE_NAME)
    t_curr = 0.0
    t_step = 0

    ts_integrator = ets.ExplicitODEIntegrator(ets.TSType.RK4)
    ts_integrator.set_ts_size(dt)
    
    mw_vth    = BEUtils.get_maxwellian_3d(vth,MNE)
    m0_t0     = BEUtils.moment_n_f(spec_sp,h_t,mw_vth,vth,0,None,None,None,1)
    temp_t0   = BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec_sp,h_t,mw_vth,vth,None,None,None,m0_t0,1)
    vth_curr  = collisions.electron_thermal_velocity(temp_t0) 
    m0_curr   = m0_t0
    print("Initial Ev : " , temp_t0 * collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT)
    tail_norm = lambda x, i: np.linalg.norm(x[i:],ord=2)/np.linalg.norm(x,ord=2)
    

    step_restored = -1
    if(restore):
        [time, mass, temp, h_t] = restore_solver(OUTPUT_FILE_NAME)
        print("Solver restore at time : %8E mass : %.8E temp: %8E" %(time,mass,temp))
        print("spectral coefficients: ")
        print(h_t)

        ts_integrator._t_coord = time
        ts_integrator._t_step  = int(time/dt)
        step_restored = int(time/dt)

        vth_curr = collisions.electron_thermal_velocity(temp)
        m0_curr  = mass

    if(quasi_neutral):
        collisions.AR_IONIZED_N = m0_curr

    while ts_integrator.current_ts()[0] < t_end:
        ts_info = ts_integrator.current_ts()
        t_curr  = ts_info[0]
        t_step  = ts_info[1]

        mw_curr   = BEUtils.get_maxwellian_3d(vth_curr,m0_curr)
        m0        = BEUtils.moment_n_f(spec_sp,h_t,mw_curr,vth_curr,0,None,None,None,1)
        temp_curr = BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec_sp,h_t,mw_curr,vth_curr,None,None,None,m0,1)

        if(quasi_neutral):
            collisions.AR_IONIZED_N = m0
        
        print(f"dt={dt:.2E} N_e(t={t_curr:.2E})={m0:.12E} maxwellian N_e={m0_t0:.12E} relative error: {(m0_t0-m0)/m0_t0:.12E} temperature(eV)={temp_curr*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT:.12E} tail ={tail_norm(h_t,TAIL_NORM_INDEX):.12E}")
        
        if(t_step == 0):
            fout = open(OUTPUT_FILE_NAME, "w")
            print(f"T\tNe_T\tRE_Ne\tTmp_T\tC_klm[%d,%s]"%(spec_sp._p,spec_sp._sph_harm_lm),file=fout)
            fout.close()
        
        if((t_step % IO_COUT_FEQ ==0)  and (t_step!=step_restored)):
            fout = open(OUTPUT_FILE_NAME, "a")
            dat_ht = np.append(np.array([t_curr,m0,(m0_t0-m0)/m0_t0,temp_curr]),h_t)
            np.savetxt(fout,dat_ht,newline=" ")
            print("",file=fout)
            fout.close()

        t_L.start()
        FOp    = spec_sp.create_mat()
        for g in col_list:
            FOp += collOp.assemble_mat(g,mw_curr,vth_curr)
        t_L.stop()
        print("Assembled the collision op. for Vth : ", vth_curr)
        print("Collision Operator assembly time (s): ",t_L.snap)

        def f_rhs(f,t):
            return np.matmul(FOp,f)
        
        ts_integrator.set_rhs_func(f_rhs)

        h_c      = ts_integrator.evolve(h_t)
        
        m0       = BEUtils.moment_n_f(spec_sp,h_c,mw_curr,vth_curr,0,None,None,None,1)
        temp_c   = BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec_sp,h_c,mw_curr,vth_curr,None,None,None,m0,1)
        vth_c    = collisions.electron_thermal_velocity(temp_c)

        h_t    = h_c
        mw_c   = BEUtils.get_maxwellian_3d(vth_c,m0)
        mm_h1  = BEUtils.compute_Mvth1_Pi_vth2_Pj_vth1(spec_sp, mw_curr, vth_curr, mw_c, vth_c, None, None, None, 1)
        h_t    = np.dot(mm_h1,h_t)
        
        print("temp(eV) = %.12E changed to temp (eV)  = %.12E tail after basis change = %.12E " %(temp_curr*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,temp_c*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,tail_norm(h_t,TAIL_NORM_INDEX)))
        
        m0_curr  = m0
        vth_curr = vth_c
        t_curr  += dt
        t_step  += 1

    t_ts.stop()
    print("Collision Operator assembly total : (s): ",t_L.seconds)
    print("Total ts time : (s): ",t_ts.seconds)



def restore_solver(fname):
    TIME_INDEX=0
    MASS_INDEX=1
    TEMP_INDEX=3
    C000_INDEX=4
    h_vec = spec.create_vec()
    data  = np.loadtxt(fname,skiprows=1)
    assert len(data[-1,C000_INDEX:]) == len(h_vec), "restore spec dim does not match with the data file in the disk"
    h_vec = data[-1,C000_INDEX:]
    time  = data[-1,TIME_INDEX]
    mass  = data[-1,MASS_INDEX]
    temp  = data[-1,TEMP_INDEX]
    return[time, mass, temp, h_vec]



parser = argparse.ArgumentParser()
parser.add_argument("-Nr", "--NUM_P_RADIAL", help="Number of polynomials in radial direction", type=int, default=4)
parser.add_argument("-Te", "--T_END", help="Simulation time", type=float, default=1e-14)
parser.add_argument("-dt", "--T_DT", help="Simulation time step size ", type=float, default=1e-15)
parser.add_argument("-o",  "--out_fname", help="output file name", type=str, default='out.dat')
parser.add_argument("-ts_tol", "--ts_tol", help="adaptive timestep tolerance", type=float, default=1e-6)
parser.add_argument("-c", "--collision_mode", help="collision mode", type=str, default="g0")
parser.add_argument("-r", "--restore", help="if 1 try to restore solution from a checkpoint", type=int, default=0)
args = parser.parse_args()


params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER = args.NUM_P_RADIAL
params.BEVelocitySpace.SPH_HARM_LM = [[0,0],[1,0]]
params.BEVelocitySpace.NUM_Q_VR  = 21
params.BEVelocitySpace.NUM_Q_VT  = 16
params.BEVelocitySpace.NUM_Q_VP  = 16
params.BEVelocitySpace.NUM_Q_CHI = 64
params.BEVelocitySpace.NUM_Q_PHI = 16
params.BEVelocitySpace.VELOCITY_SPACE_DT = args.T_DT
NUM_T_STEPS      = int(args.T_END/args.T_DT)
OUTPUT_FILE_NAME = args.out_fname

### mpi to run multiple convergence study runs. 
## ============================================
CONV_NR    = np.array([3,7,15])
CONV_STEPS = np.array([1e3,2e4,4e4])
CONV_DT    = args.T_END/CONV_STEPS
RESTORE_SOLVER = args.restore

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
npes = comm.Get_size()

if(not rank):
    print("Total MPI tasks : ", npes)

args.T_DT         = CONV_DT[rank%len(CONV_DT)]
args.NUM_P_RADIAL = CONV_NR[rank//len(CONV_DT)]
params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER=args.NUM_P_RADIAL
params.BEVelocitySpace.VELOCITY_SPACE_DT = args.T_DT
print(params.BEVelocitySpace.VELOCITY_SPACE_DT)
## ============================================

colOpSp.SPEC_SPHERICAL  = sp.SpectralExpansionSpherical(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,basis.Maxwell(),params.BEVelocitySpace.SPH_HARM_LM) 
collisions.MAXWELLIAN_TEMP_K = 20 * collisions.TEMP_K_1EV
collisions.ELECTRON_THEMAL_VEL = collisions.electron_thermal_velocity(collisions.MAXWELLIAN_TEMP_K) 
    
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

col_g0_no_E_loss = collisions.eAr_G0_NoEnergyLoss()
col_g0 = collisions.eAr_G0()
col_g1 = collisions.eAr_G1()
col_g2 = collisions.eAr_G2()

# maxwellian = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)
# hv         = lambda v,vt,vp : np.ones_like(v) #+ v + v**2   #* collisions.MAXWELLIAN_N
# h_vec      = BEUtils.compute_func_projection_coefficients(spec,hv,maxwellian,None,None,None)

def perturbed_mw_test():
    vth_fac    = 1 - 5e-1
    maxwellian = BEUtils.get_maxwellian_3d(vth_fac * VTH,collisions.MAXWELLIAN_N)
    hv         = lambda v,vt,vp : np.ones_like(v) #+  (v**2) * np.exp(-v**2)#+ v + v**2   #* collisions.MAXWELLIAN_N
    h_vec      = BEUtils.compute_func_projection_coefficients(spec,hv,maxwellian,None,None,None)

    mw_high    = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)
    mm_change  = BEUtils.compute_Mvth1_Pi_vth2_Pj_vth1(spec,maxwellian,vth_fac* VTH, mw_high, VTH, None,None,None,1)
    h_vec      = np.dot(mm_change,h_vec)
    h_vec = np.ones_like(h_vec)

    t_M.start()
    M  = spec.compute_maxwellian_mm(maxwellian,VTH)
    t_M.stop()
    print("Mass assembly time (s): ", t_M.seconds)
    global OUTPUT_FILE_NAME
    OUTPUT_FILE_NAME = f"%s/g0_pw_dt_%.8E_Nr_%d.dat" %(OUTPUT_FILE_NAME,params.BEVelocitySpace.VELOCITY_SPACE_DT,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)
    print(OUTPUT_FILE_NAME)

    ode_first_order_linear_adaptive_v1(cf,[col_g0_no_E_loss],h_vec,mw_high,VTH,args.T_END,args.T_DT, ts_tol=args.ts_tol)


def g0_test():
    maxwellian = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)
    hv         = lambda v,vt,vp : np.ones_like(v)
    h_vec      = BEUtils.compute_func_projection_coefficients(spec,hv,maxwellian,None,None,None)

    t_M.start()
    M  = spec.compute_maxwellian_mm(maxwellian,VTH)
    t_M.stop()
    print("Mass assembly time (s): ", t_M.seconds)
    
    global OUTPUT_FILE_NAME
    OUTPUT_FILE_NAME = f"%s/g0_dt_%.8E_Nr_%d.dat" %(OUTPUT_FILE_NAME,params.BEVelocitySpace.VELOCITY_SPACE_DT,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)
    print(OUTPUT_FILE_NAME)

    ode_numerical_solve(cf,[col_g0],h_vec,maxwellian,VTH,args.T_END,args.T_DT, ts_tol=args.ts_tol,restore=RESTORE_SOLVER)

def g1_test():
    maxwellian = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)
    hv         = lambda v,vt,vp : np.ones_like(v)
    h_vec      = BEUtils.compute_func_projection_coefficients(spec,hv,maxwellian,None,None,None)

    t_M.start()
    M  = spec.compute_maxwellian_mm(maxwellian,VTH)
    t_M.stop()
    print("Mass assembly time (s): ", t_M.seconds)

    global OUTPUT_FILE_NAME
    OUTPUT_FILE_NAME = f"%s/g1_dt_%.8E_Nr_%d.dat" %(OUTPUT_FILE_NAME,params.BEVelocitySpace.VELOCITY_SPACE_DT,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)
    print(OUTPUT_FILE_NAME)


    ode_numerical_solve(cf,[col_g1],h_vec,maxwellian,VTH,args.T_END,args.T_DT, ts_tol=args.ts_tol,restore=RESTORE_SOLVER)
    


def g2_test():
    maxwellian = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)
    hv         = lambda v,vt,vp : np.ones_like(v)
    h_vec      = BEUtils.compute_func_projection_coefficients(spec,hv,maxwellian,None,None,None)

    t_M.start()
    M  = spec.compute_maxwellian_mm(maxwellian,VTH)
    t_M.stop()
    print("Mass assembly time (s): ", t_M.seconds)
    
    global OUTPUT_FILE_NAME
    OUTPUT_FILE_NAME = f"%s/g2_dt_%.8E_Nr_%d.dat" %(OUTPUT_FILE_NAME,params.BEVelocitySpace.VELOCITY_SPACE_DT,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)
    print(OUTPUT_FILE_NAME)


    ode_numerical_solve(cf,[col_g2],h_vec,maxwellian,VTH,args.T_END,args.T_DT, ts_tol=args.ts_tol,restore=RESTORE_SOLVER,quasi_neutral=False)


def g01_test():
    maxwellian = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)
    hv         = lambda v,vt,vp : np.ones_like(v)
    h_vec      = BEUtils.compute_func_projection_coefficients(spec,hv,maxwellian,None,None,None)

    t_M.start()
    M  = spec.compute_maxwellian_mm(maxwellian,VTH)
    t_M.stop()
    print("Mass assembly time (s): ", t_M.seconds)
    
    global OUTPUT_FILE_NAME
    OUTPUT_FILE_NAME = f"%s/g01_dt_%.8E_Nr_%d.dat" %(OUTPUT_FILE_NAME,params.BEVelocitySpace.VELOCITY_SPACE_DT,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)
    print(OUTPUT_FILE_NAME)

    ode_numerical_solve(cf,[col_g0,col_g1],h_vec,maxwellian,VTH,args.T_END,args.T_DT, ts_tol=args.ts_tol,restore=RESTORE_SOLVER)


def g02_test():
    maxwellian = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)
    hv         = lambda v,vt,vp : np.ones_like(v)
    h_vec      = BEUtils.compute_func_projection_coefficients(spec,hv,maxwellian,None,None,None)

    t_M.start()
    M  = spec.compute_maxwellian_mm(maxwellian,VTH)
    t_M.stop()
    print("Mass assembly time (s): ", t_M.seconds)
    
    global OUTPUT_FILE_NAME
    OUTPUT_FILE_NAME = f"%s/g02_dt_%.8E_Nr_%d.dat" %(OUTPUT_FILE_NAME,params.BEVelocitySpace.VELOCITY_SPACE_DT,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)
    print(OUTPUT_FILE_NAME)


    ode_numerical_solve(cf,[col_g0,col_g2],h_vec,maxwellian,VTH,args.T_END,args.T_DT, ts_tol=args.ts_tol,restore=RESTORE_SOLVER,quasi_neutral=True)
    

if args.collision_mode  == "g0":
    g0_test()
elif args.collision_mode  == "g1":
    g1_test()
elif args.collision_mode  == "g2":
    g2_test()
elif args.collision_mode  == "g01":
    g01_test()
elif args.collision_mode  == "g02":
    g02_test()
else:
    print("unknown collision mode")

