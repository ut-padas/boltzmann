"""
@package Boltzmann collision operator solver. 
"""

import enum
import scipy
from sympy import arg
import basis
import spec_spherical as sp
import numpy as np
import collision_operator_spherical as colOpSp
import collisions 
import parameters as params
import os
from time import perf_counter as time
import utils as BEUtils
import argparse
import scipy.integrate
from scipy.integrate import ode

class CollissionMode(enum.Enum):
    ELASTIC_ONLY=0
    ELASTIC_W_EXCITATION=1
    ELASTIC_W_IONIZATION=2
    ELASTIC_W_EXCITATION_W_IONIZATION=3

collisions.AR_NEUTRAL_N=3.22e22
collisions.MAXWELLIAN_N=1e18
collisions.AR_IONIZED_N=collisions.MAXWELLIAN_N
parser = argparse.ArgumentParser()

def spec_tail(cf, num_p, num_sh):
    return np.linalg.norm(cf[num_p//2 * num_sh :])

def spec_tail_timeseries(cf, num_p, num_sh):
    return np.array([np.linalg.norm(cf[i, num_p//2 * num_sh :]) for i in range(data.shape[0])])


def solve_collop(collOp:colOpSp.CollisionOpSP, h_init, maxwellian, vth, t_end, dt,t_tol, mode:CollissionMode):
    spec_sp = collOp._spec
    MVTH  = vth
    MNE   = maxwellian(0) * (np.sqrt(np.pi)**3) * (vth**3)
    MTEMP = collisions.electron_temperature(MVTH)
    print("==========================================================================")

    h_t = np.array(h_init)
    t_curr = 0.0
    t_step = 0

    ne_t      = MNE
    mw_vth    = BEUtils.get_maxwellian_3d(vth,ne_t)
    m0_t0     = BEUtils.moment_n_f(spec_sp,h_t,mw_vth,vth,0,None,None,None,1)
    temp_t0   = BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec_sp,h_t,mw_vth,vth,None,None,None,m0_t0,1)
    vth_curr  = collisions.electron_thermal_velocity(temp_t0) 
    print("Initial Ev : " , temp_t0 * collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT)
    
    t1=time()
    M  = spec_sp.compute_mass_matrix()
    t2=time()
    print("Mass assembly time (s): ", (t2-t1))
    print("Condition number of M= %.8E"%np.linalg.cond(M))
    #print(M)

    Minv = np.linalg.inv(M)

    if(mode == CollissionMode.ELASTIC_ONLY):
        g0  = collisions.eAr_G0()
        g0.reset_scattering_direction_sp_mat()
        t1=time()
        FOp = collOp.assemble_mat(g0,mw_vth,vth_curr)
        t2=time()
        print("Assembled the collision op. for Vth : ", vth_curr)
        print("Collision Operator assembly time (s): ",(t2-t1))
        #print("FOp: ",FOp)
        #print("Condition number of C= %.8E"%np.linalg.cond(FOp))
        FOp= np.matmul(Minv,FOp)
        #print("Condition number of n0 x (M^-1 x C)= ",np.linalg.cond(FOp))
        # u,s,v = np.linalg.svd(FOp)
        # print("Singular values of FOp:", s)
        def f_rhs(t,y,n0):
            return n0*np.matmul(FOp,y)
        
        ode_solver = ode(f_rhs,jac=None).set_integrator("dopri5",verbosity=1, rtol=t_tol, atol=t_tol, nsteps=10000)
        #ode_solver = ode(f_rhs,jac=None).set_integrator("dop853",verbosity=1,rtol=t_tol)
        ode_solver.set_initial_value(h_init,t=0.0)
        ode_solver.set_f_params(collisions.AR_NEUTRAL_N)

        total_steps     = int(t_end/dt)
        solution_vector = np.zeros((total_steps,h_init.shape[0]))
        while ode_solver.successful() and t_step < total_steps: 
            t_curr = ode_solver.t
            ht     = ode_solver.y
            solution_vector[t_step,:] = ht
            ode_solver.integrate(ode_solver.t + dt)
            t_step+=1

    elif(mode == CollissionMode.ELASTIC_W_IONIZATION):
        g0  = collisions.eAr_G0()
        g0.reset_scattering_direction_sp_mat()

        g2  = collisions.eAr_G2()
        g2.reset_scattering_direction_sp_mat()
        t1=time()
        FOp_g0 = collOp.assemble_mat(g0,mw_vth,vth_curr) 
        FOp_g2 = collOp.assemble_mat(g2,mw_vth,vth_curr)
        t2=time()
        print("Assembled the collision op. for Vth : ", vth_curr)
        print("Collision Operator assembly time (s): ",(t2-t1))
        
        FOp_g0 = np.matmul(Minv,FOp_g0)
        FOp_g2 = np.matmul(Minv,FOp_g2)

        def f_rhs(t,y,n0,ni):
            return n0*np.matmul(FOp_g0,y) #+ ni*np.matmul(FOp_g2,y)

        ode_solver = ode(f_rhs,jac=None).set_integrator("dopri5",verbosity=1, rtol=t_tol, atol=t_tol, nsteps=10000)
        ode_solver.set_initial_value(h_init,t=0.0)
        
        t_step = 0
        total_steps = int(t_end/dt)
        solution_vector = np.zeros((total_steps,h_init.shape[0]))
        while ode_solver.successful() and t_step < total_steps: 
            t_curr   = ode_solver.t
            m0_t     = BEUtils.moment_n_f(spec_sp,ode_solver.y,mw_vth,vth,0,300,16,16,1)
            ode_solver.set_f_params(collisions.AR_NEUTRAL_N,m0_t)
            ht     = ode_solver.y
            solution_vector[t_step,:] = ht
            ode_solver.integrate(ode_solver.t + dt)
            t_step+=1
    else:
        raise NotImplementedError("Unknown collision type")

    return solution_vector

parser.add_argument("-Nr", "--NUM_P_RADIAL", help="Number of polynomials in radial direction", nargs='+', type=int, default=[4,8,16,32,64])
parser.add_argument("-Te", "--T_END", help="Simulation time", type=float, default=1e-6)
parser.add_argument("-dt", "--T_DT", help="Simulation time step size ", type=float, default=1e-10)
parser.add_argument("-o",  "--out_fname", help="output file name", type=str, default='.')
parser.add_argument("-ts_tol", "--ts_tol", help="adaptive timestep tolerance", type=float, default=1e-15)
parser.add_argument("-l_max", "--l_max", help="max polar modes in SH expansion", type=int, default=1)
parser.add_argument("-c", "--collision_mode", help="collision mode", type=str, default="g0")
parser.add_argument("-ev", "--electron_volt", help="initial electron volt", type=float, default=1.0)
parser.add_argument("-r", "--restore", help="if 1 try to restore solution from a checkpoint", type=int, default=0)
args = parser.parse_args()

run_data=list()
ev     = np.linspace(0.3,40,1000)
eedf   = np.zeros((len(args.NUM_P_RADIAL),len(ev)))
for i, nr in enumerate(args.NUM_P_RADIAL):
    params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER = nr
    params.BEVelocitySpace.SPH_HARM_LM = [[i,j] for i in range(args.l_max) for j in range(i+1)]

    # q_mode = sp.QuadMode.SIMPSON
    # r_mode = basis.BasisType.SPLINES
    # params.BEVelocitySpace.NUM_Q_VR  = basis.BSpline.get_num_q_pts(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,basis.BSPLINE_BASIS_ORDER,basis.BSPLINE_NUM_Q_PTS_PER_KNOT)

    q_mode = sp.QuadMode.GMX
    r_mode = basis.BasisType.MAXWELLIAN_POLY
    params.BEVelocitySpace.NUM_Q_VR  = 300
    
    params.BEVelocitySpace.NUM_Q_VT  = 8
    params.BEVelocitySpace.NUM_Q_VP  = 8
    params.BEVelocitySpace.NUM_Q_CHI = 2
    params.BEVelocitySpace.NUM_Q_PHI = 2
    params.BEVelocitySpace.VELOCITY_SPACE_DT = args.T_DT

    INIT_EV    = args.electron_volt
    collisions.MAXWELLIAN_TEMP_K   = INIT_EV * collisions.TEMP_K_1EV
    collisions.ELECTRON_THEMAL_VEL = collisions.electron_thermal_velocity(collisions.MAXWELLIAN_TEMP_K) 
    cf    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,q_mode,r_mode)
    spec  = cf._spec
    VTH   = collisions.ELECTRON_THEMAL_VEL

    print("""===========================Parameters ======================""")
    print("\tMAXWELLIAN_N : ", collisions.MAXWELLIAN_N)
    print("\tELECTRON_THEMAL_VEL : ", VTH," ms^-1")
    print("\tDT : ", params.BEVelocitySpace().VELOCITY_SPACE_DT, " s")
    print("""============================================================""")
    params.print_parameters()

    maxwellian = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)
    hv         = lambda v,vt,vp : np.ones_like(v)
    h_vec      = BEUtils.function_to_basis(spec,hv,maxwellian,None,None,None)
    data       = solve_collop(cf, h_vec, maxwellian, VTH, args.T_END, args.T_DT,args.ts_tol,mode=CollissionMode.ELASTIC_ONLY)
    #data       = solve_collop(cf, h_vec, maxwellian, VTH, args.T_END, args.T_DT,args.ts_tol,mode=CollissionMode.ELASTIC_W_IONIZATION)
    eedf[i]      = BEUtils.get_eedf(ev, spec, data[-1,:], maxwellian, VTH, 1)
    eedf_initial = BEUtils.get_eedf(ev, spec, data[0,:], maxwellian, VTH, 1)
    run_data.append(data)


import matplotlib.pyplot as plt
fig = plt.figure()#(figsize=(6, 6), dpi=300)

ts= np.linspace(0,args.T_END, int(args.T_END/args.T_DT))
plt.subplot(1, 2, 1)
for i, nr in enumerate(args.NUM_P_RADIAL):
    plt.plot(ts, spec_tail_timeseries(run_data[i],nr, len(params.BEVelocitySpace.SPH_HARM_LM)),label="Nr=%d"%args.NUM_P_RADIAL[i])

plt.legend()
plt.yscale('log')
plt.xlabel("time (s)")
plt.ylabel("spectral tail l2(h[nr/2: ])")


plt.subplot(1, 2, 2)
plt.plot(ev, eedf_initial, label="initial")
for i, nr in enumerate(args.NUM_P_RADIAL):
    data=run_data[i]
    plt.plot(ev, eedf[i],label="final")

plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel("energy (ev)")
plt.ylabel("eedf")
plt.tight_layout()

plt.show()


# for i in range(eedf.shape[0]-1):
#     plt.plot(ev,np.abs(eedf[i]-eedf[i+1]),label="Nr=%d"%(args.NUM_P_RADIAL[i]))

# for i in range(eedf.shape[0]):
#     plt.plot(ev,eedf[i],linewidth=1,label="Nr=%d"%(args.NUM_P_RADIAL[i]))


# # plt.xscale("log")
# # plt.yscale("log")
# # plt.xlabel("energy(ev)")
# # plt.ylabel("error")
# # plt.legend()
# #plt.savefig('g0_maxwell.png')
# #plt.savefig('g0_bspline_linear.png', dpi=300)
# #plt.savefig('g0_bspline_quad.png', dpi=300)

# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("energy(ev)")
# plt.ylabel("EEDF")
# plt.legend()
# #plt.show()
# plt.savefig('g0_maxwell_eedf.png')
#plt.savefig('g0_bspline_linear_eedf.png', dpi=300)
#plt.savefig('g0_bspline_quad_eedf.png', dpi=300)






    
# instance of the collision operator


# col_g0_no_E_loss = collisions.eAr_G0_NoEnergyLoss()
# col_g0 = collisions.eAr_G0()
# col_g1 = collisions.eAr_G1()
# col_g2 = collisions.eAr_G2()

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

    # t_M.start()
    # M  = spec.compute_maxwellian_mm(maxwellian,VTH)
    # t_M.stop()
    # print("Mass assembly time (s): ", t_M.seconds)
    # global OUTPUT_FILE_NAME
    # OUTPUT_FILE_NAME = f"%s/g0_pw_dt_%.8E_Nr_%d.dat" %(OUTPUT_FILE_NAME,params.BEVelocitySpace.VELOCITY_SPACE_DT,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)
    # print(OUTPUT_FILE_NAME)

    #ode_first_order_linear_adaptive_v1(cf,[col_g0_no_E_loss],h_vec,mw_high,VTH,args.T_END,args.T_DT, ts_tol=args.ts_tol)



#ode_numerical_solve_no_reassembly_and_projection(cf, h_vec, maxwellian, VTH, args.T_END, args.T_DT,args.ts_tol,mode=CollissionMode.ELASTIC_ONLY)
