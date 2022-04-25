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

def spec_tail(cf, num_p, num_sh):
    return np.linalg.norm(cf[(num_p//2) * num_sh :])

def spec_tail_timeseries(cf, num_p, num_sh):
    return np.array([np.linalg.norm(cf[i, (num_p//2) * num_sh :])/len(cf[i, (num_p//2) * num_sh :]) for i in range(data.shape[0])])


def svd_truncate(FOp,r_tol=1e-6):
    u, s, v = np.linalg.svd(FOp)
    FOp_svd = np.matmul(u, np.matmul(np.diag(s), v))  
    print("SVD rel. error : %.12E"%(np.linalg.norm(FOp_svd-FOp)/np.linalg.norm(FOp)))
    st = np.where(s/s[0]>r_tol) 
    s_k = st[0][-1]
    print("truncated at =%d out of %d "%(s_k,len(s)))
    FOp_svd = np.matmul(u[:, 0:s_k], np.matmul(np.diag(s[0:s_k]), v[0:s_k,:]))  
    print("SVD after truncation rel. error : %.12E"%(np.linalg.norm(FOp_svd-FOp)/np.linalg.norm(FOp)))
    return FOp_svd


def constant_r_eval(spec : sp.SpectralExpansionSpherical, cf, r):
    theta = np.linspace(0,np.pi,50)
    phi   = np.linspace(0,2*np.pi,100)
    grid  = np.meshgrid(theta, phi, indexing='ij')
    
    num_p  = spec._p + 1
    num_sh = len(spec._sph_harm_lm)
    
    b_eval = np.array([ np.exp(-r**2) * spec.basis_eval_radial(r,k,l) * spec.basis_eval_spherical(grid[0],grid[1],l,m) for lm_idx, (l,m) in enumerate(spec._sph_harm_lm) for k in range(num_p)])
    b_eval = b_eval.reshape(num_sh,num_p, -1)
    b_eval = np.swapaxes(b_eval,0,1)
    b_eval = b_eval.reshape(num_p * num_sh,-1)
    
    return np.dot(cf, b_eval).reshape(50,100), theta, phi
    

def solve_collop(collOp:colOpSp.CollisionOpSP, h_init, maxwellian, vth, t_end, dt,t_tol, mode:CollissionMode):
    spec_sp = collOp._spec

    t1=time()
    M  = spec_sp.compute_mass_matrix()
    t2=time()
    _, s, _ = np.linalg.svd(M)
    m_cond=np.linalg.cond(M)
    #print("s=",s)
    print("Mass assembly time (s): ", (t2-t1))
    print("Condition number of M= %.8E"%m_cond)
    Minv = np.linalg.pinv(M)

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
    print("Initial Ev : "   , temp_t0 * collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT)
    print("Initial mass : " , m0_t0 )
    
    if(mode == CollissionMode.ELASTIC_ONLY):
        #g0  = collisions.eAr_G0_NoEnergyLoss()
        g0  = collisions.eAr_G0()
        g0.reset_scattering_direction_sp_mat()
        t1=time()
        FOp = collOp.assemble_mat(g0,mw_vth,vth_curr)
        t2=time()
        print("Assembled the collision op. for Vth : ", vth_curr)
        print("Collision Operator assembly time (s): ",(t2-t1))
        FOp = np.matmul(Minv,FOp)
        print("Condition number of C= %.8E"%np.linalg.cond(FOp))
        
        def f_rhs(t,y,n0):
            return n0*np.matmul(FOp,y)
        
        ode_solver = ode(f_rhs,jac=None).set_integrator("dopri5",verbosity=1, rtol=t_tol, atol=t_tol, nsteps=10000)
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
        print("Condition number of C= %.8E"%np.linalg.cond(FOp_g0 + FOp_g2 ))

        def f_rhs(t,y,n0,ni):
            return n0*np.matmul(FOp_g0,y) + ni*np.matmul(FOp_g2,y)

        ode_solver = ode(f_rhs,jac=None).set_integrator("dopri5",verbosity=1, rtol=t_tol, atol=t_tol, nsteps=10000)
        ode_solver.set_initial_value(h_init,t=0.0)
        ode_solver.set_f_params(collisions.AR_NEUTRAL_N,m0_t0)
        
        t_step = 0
        total_steps = int(t_end/dt)
        solution_vector = np.zeros((total_steps,h_init.shape[0]))
        while ode_solver.successful() and t_step < total_steps: 
            t_curr   = ode_solver.t
            # m0_t     = BEUtils.moment_n_f(spec_sp,ode_solver.y,mw_vth,vth,0,300,16,16,1)
            # ode_solver.set_f_params(collisions.AR_NEUTRAL_N,m0_t)
            ht     = ode_solver.y
            solution_vector[t_step,:] = ht
            ode_solver.integrate(ode_solver.t + dt)
            t_step+=1
    else:
        raise NotImplementedError("Unknown collision type")

    return solution_vector

parser = argparse.ArgumentParser()
parser.add_argument("-Nr", "--NUM_P_RADIAL"                   , help="Number of polynomials in radial direction", nargs='+', type=int, default=[4,8,16,32,64,128,256])
parser.add_argument("-T", "--T_END"                           , help="Simulation time", type=float, default=1e-6)
parser.add_argument("-dt", "--T_DT"                           , help="Simulation time step size ", type=float, default=1e-10)
parser.add_argument("-o",  "--out_fname"                      , help="output file name", type=str, default='coll_op')
parser.add_argument("-ts_tol", "--ts_tol"                     , help="adaptive timestep tolerance", type=float, default=1e-15)
parser.add_argument("-l_max", "--l_max"                       , help="max polar modes in SH expansion", type=int, default=0)
parser.add_argument("-c", "--collision_mode"                  , help="collision mode g- elastic with no E loss, g0-elastic", type=str, default="g0")
parser.add_argument("-ev", "--electron_volt"                  , help="initial electron volt", type=float, default=1.0)
parser.add_argument("-q_vr", "--quad_radial"                  , help="quadrature in r"        , type=int, default=270)
parser.add_argument("-q_vt", "--quad_theta"                   , help="quadrature in polar"    , type=int, default=2)
parser.add_argument("-q_vp", "--quad_phi"                     , help="quadrature in azimuthal", type=int, default=2)
parser.add_argument("-q_st", "--quad_s_theta"                 , help="quadrature in scattering polar"    , type=int, default=2)
parser.add_argument("-q_sp", "--quad_s_phi"                   , help="quadrature in scattering azimuthal", type=int, default=2)
parser.add_argument("-radial_poly", "--radial_poly"           , help="radial basis", type=str, default="maxwell")
parser.add_argument("-sp_order", "--spline_order"             , help="b-spline order", type=int, default=2)
parser.add_argument("-spline_qpts", "--spline_q_pts_per_knot" , help="q points per knots", type=int, default=11)
parser.add_argument("-vth_fac", "--vth_fac" , help="expand the function vth_fac *V_TH", type=float, default=1.0)
#parser.add_argument("-r", "--restore", help="if 1 try to restore solution from a checkpoint", type=int, default=0)
args = parser.parse_args()

run_data=list()
ev           = np.linspace(args.electron_volt/50.,100.*args.electron_volt,1000)
eedf         = np.zeros((len(args.NUM_P_RADIAL),len(ev)))
temperature  = list()
SPLINE_ORDER = args.spline_order
basis.BSPLINE_BASIS_ORDER=SPLINE_ORDER
basis.XLBSPLINE_NUM_Q_PTS_PER_KNOT=args.spline_q_pts_per_knot
vth_factor_temp=[1,0.95, 0.9, 0.85, 0.8]
for i, nr in enumerate(args.NUM_P_RADIAL):
    params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER = nr
    params.BEVelocitySpace.SPH_HARM_LM = [[i,j] for i in range(args.l_max+1) for j in range(-i,i+1)]
    if (args.radial_poly == "maxwell"):
        r_mode = basis.BasisType.MAXWELLIAN_POLY
        params.BEVelocitySpace.NUM_Q_VR  = args.quad_radial
        
    elif (args.radial_poly == "laguerre"):
        r_mode = basis.BasisType.LAGUERRE
        params.BEVelocitySpace.NUM_Q_VR  = args.quad_radial

    elif (args.radial_poly == "bspline"):
        r_mode = basis.BasisType.SPLINES
        params.BEVelocitySpace.NUM_Q_VR  = basis.BSpline.get_num_q_pts(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER, SPLINE_ORDER, basis.XLBSPLINE_NUM_Q_PTS_PER_KNOT)

    params.BEVelocitySpace.NUM_Q_VT  = args.quad_theta
    params.BEVelocitySpace.NUM_Q_VP  = args.quad_phi
    params.BEVelocitySpace.NUM_Q_CHI = args.quad_s_theta
    params.BEVelocitySpace.NUM_Q_PHI = args.quad_s_phi
    params.BEVelocitySpace.VELOCITY_SPACE_DT = args.T_DT

    INIT_EV    = args.electron_volt
    collisions.MAXWELLIAN_TEMP_K   = INIT_EV * collisions.TEMP_K_1EV
    collisions.ELECTRON_THEMAL_VEL = collisions.electron_thermal_velocity(collisions.MAXWELLIAN_TEMP_K) 
    cf    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,poly_type=r_mode)
    spec  = cf._spec
    VTH   = collisions.ELECTRON_THEMAL_VEL

    print("""===========================Parameters ======================""")
    print("\tMAXWELLIAN_N : ", collisions.MAXWELLIAN_N)
    print("\tELECTRON_THEMAL_VEL : ", VTH," ms^-1")
    print("\tDT : ", params.BEVelocitySpace().VELOCITY_SPACE_DT, " s")
    print("""============================================================""")
    params.print_parameters()

    vth_factor   = args.vth_fac
    print("using vth factor : ", vth_factor, "for Nr: ", nr)
    VTH_C        = vth_factor * VTH
    maxwellian   = BEUtils.get_maxwellian_3d(VTH_C, collisions.MAXWELLIAN_N)
    #hv          = lambda v,vt,vp : np.ones_like(v) +  np.tan(vt)  
    if (args.radial_poly == "maxwell" or args.radial_poly == "laguerre"):
        hv           = lambda v,vt,vp : (vth_factor**3) * np.exp(v**2  -(v*vth_factor)**2)
    if (args.radial_poly == "bspline"):
        hv           = lambda v,vt,vp : (vth_factor**3) * np.exp(-(v*vth_factor)**2)
        
    h_vec        = BEUtils.function_to_basis(spec,hv,maxwellian,None,None,None)
    data         = solve_collop(cf, h_vec, maxwellian, VTH_C, args.T_END, args.T_DT,args.ts_tol,mode=CollissionMode.ELASTIC_ONLY)
    #data        = solve_collop(cf, h_vec, maxwellian, VTH_C, args.T_END, args.T_DT,args.ts_tol,mode=CollissionMode.ELASTIC_W_IONIZATION)
    eedf[i]      = BEUtils.get_eedf(ev, spec, data[-1,:], maxwellian, VTH_C, 1)
    eedf_initial = BEUtils.get_eedf(ev, spec, data[0,:], maxwellian, VTH_C, 1)
    
    # temp       = np.array([(BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec, data[w], maxwellian, VTH_C, None, None, None, None)) * (collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT) for w in range(data.shape[0])])
    temp         = BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec, data, maxwellian, VTH_C, None, None, None, None) * (collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT)
    temperature.append(temp)
    run_data.append(data)

import matplotlib.pyplot as plt

if (1):
    fig = plt.figure(figsize=(16, 10), dpi=300)
    plt.subplot(2, 3, 1)
    for i, nr in enumerate(args.NUM_P_RADIAL):
        data=run_data[i]
        plt.plot(abs(data[1,:]),label="Nr=%d"%args.NUM_P_RADIAL[i])

    plt.yscale('log')
    plt.xlabel("Coeff. #")
    plt.ylabel("Coeff. magnitude")
    plt.grid()
    plt.legend()

    plt.subplot(2, 3, 2)
    for i, nr in enumerate(args.NUM_P_RADIAL):
        data=run_data[i]
        plt.plot(abs(data[-1,:]),label="Nr=%d"%args.NUM_P_RADIAL[i])

    plt.yscale('log')
    plt.xlabel("Coeff. #")
    plt.ylabel("Coeff. magnitude")
    plt.grid()
    plt.legend()

    plt.subplot(2, 3, 3)
    for i, nr in enumerate(args.NUM_P_RADIAL):
        data_last=run_data[-1]
        data=run_data[i]
        plt.plot(abs(data[-1,:] - data_last[-1,0:len(data[-1,:])]),label="Nr=%d"%args.NUM_P_RADIAL[i])

    plt.yscale('log')
    plt.xlabel("Coeff. #")
    plt.ylabel("Error in coeff. magnitude")
    plt.grid()
    plt.legend()

    ts= np.linspace(0,args.T_END, int(args.T_END/args.T_DT))
    plt.subplot(2, 3, 4)
    
    if (args.radial_poly == "maxwell"):
        for i, nr in enumerate(args.NUM_P_RADIAL):
            plt.plot(ts, spec_tail_timeseries(run_data[i],nr, len(params.BEVelocitySpace.SPH_HARM_LM)),label="Nr=%d"%args.NUM_P_RADIAL[i])

        plt.legend()
        plt.yscale('log')
        plt.xlabel("time (s)")
        plt.ylabel("spectral tail l2(h[nr/2: ])")
        plt.grid()
        
    elif (args.radial_poly == "bspline"):
        for i, nr in enumerate(args.NUM_P_RADIAL):
            plt.plot(ev, np.abs(eedf[i]-eedf[-1]),label="Nr=%d"%args.NUM_P_RADIAL[i])

        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.xlabel("energy (ev)")
        plt.ylabel("error in eedf")
        plt.grid()


    plt.subplot(2, 3, 5)
    plt.plot(ev, eedf_initial, label="initial")
    for i, nr in enumerate(args.NUM_P_RADIAL):
        data=run_data[i]
        plt.plot(ev, abs(eedf[i]),label="Nr=%d"%args.NUM_P_RADIAL[i])

    # plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlabel("energy (ev)")
    plt.ylabel("eedf")
    # plt.tight_layout()
    plt.grid()

    plt.subplot(2, 3, 6)
    for i, nr in enumerate(args.NUM_P_RADIAL):
        data=temperature[i]
        plt.plot(ts, temperature[i], label="Nr=%d vth_fac=%.2f"%(args.NUM_P_RADIAL[i],vth_factor_temp[i]))

    # plt.xscale('log')
    # plt.yscale('log')
    plt.legend()
    plt.xlabel("time (s)")
    plt.ylabel("temperature (eV)")
    # plt.tight_layout()
    plt.grid()

    plt.tight_layout()
    #plt.show()
    plt.savefig("%s_coeff.png"%(args.out_fname))
    
    

if (0):
    fig = plt.figure(figsize=(10, 4), dpi=300)

    if (args.radial_poly == "maxwell"):
        ts= np.linspace(0,args.T_END, int(args.T_END/args.T_DT))
        plt.subplot(1, 2, 1)
        for i, nr in enumerate(args.NUM_P_RADIAL):
            plt.plot(ts, spec_tail_timeseries(run_data[i],nr+1, len(params.BEVelocitySpace.SPH_HARM_LM)),label="Nr=%d"%args.NUM_P_RADIAL[i])

        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel("time (s)")
        plt.ylabel("tail l2(h[nr/2: ])")

    elif (args.radial_poly == "bspline"):
        plt.subplot(1, 2, 1)
        for i, nr in enumerate(args.NUM_P_RADIAL):
            plt.plot(ev, np.abs(eedf[i]-eedf[-1]),label="Nr=%d"%args.NUM_P_RADIAL[i])

        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.xlabel("energy (ev)")
        plt.ylabel("error in eedf")

    plt.subplot(1, 2, 2)
    plt.plot(ev, eedf_initial, label="initial")
    for i, nr in enumerate(args.NUM_P_RADIAL):
        data=run_data[i]
        plt.plot(ev, eedf[i],label="Nr=%d"%(nr))

    #plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlabel("energy (ev)")
    plt.ylabel("eedf")
    plt.tight_layout()

    #plt.show()
    plt.savefig("%s.png"%(args.out_fname))
    plt.close()

    fig = plt.figure(figsize=(10, 4), dpi=300)
    plt.subplot(1,2,1)
    polar_plot, _ , __ = constant_r_eval(spec,run_data[-1][0,:],1)
    plt.imshow(polar_plot,extent=[0,2*np.pi,0,np.pi])
    plt.colorbar()
    plt.xlabel("azimuthal angle")
    plt.ylabel("polar angle")
    plt.title("initial, v_r = 1")

    plt.subplot(1,2,2)
    polar_plot, _ , __ = constant_r_eval(spec,run_data[-1][-1,:],1)
    plt.imshow(polar_plot,extent=[0,2*np.pi,0,np.pi])
    plt.colorbar()
    plt.clim(np.min(polar_plot),np.max(polar_plot)*1.01)
    plt.xlabel("azimuthal angle")
    plt.ylabel("polar angle")
    plt.title("final, v_r = 1")
    plt.savefig("%s_const_r.png"%(args.out_fname))

