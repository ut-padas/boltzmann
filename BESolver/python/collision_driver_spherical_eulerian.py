"""
@package Boltzmann collision operator solver. 
"""

from cProfile import run
import enum
import string
import scipy
from sympy import arg, eye
from maxpoly import maxpolyserieseval
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
from advection_operator_spherical_polys import *

import matplotlib.pyplot as plt

class CollissionMode(enum.Enum):
    ELASTIC_ONLY=0
    ELASTIC_W_EXCITATION=1
    ELASTIC_W_IONIZATION=2
    ELASTIC_W_EXCITATION_W_IONIZATION=3

collisions.AR_NEUTRAL_N=3.22e22
collisions.MAXWELLIAN_N=1e18
collisions.AR_IONIZED_N=collisions.MAXWELLIAN_N
parser = argparse.ArgumentParser()

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



def solve_collop(steady_state, collOp:colOpSp.CollisionOpSP, h_init, maxwellian, vth, E_field, t_end, dt,t_tol, collisions_included):
    spec_sp = collOp._spec

    t1=time()
    M  = spec_sp.compute_mass_matrix()
    t2=time()
    print("Mass assembly time (s): ", (t2-t1))
    print("Condition number of M= %.8E"%np.linalg.cond(M))
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
    # vth_curr  = collisions.electron_thermal_velocity(temp_t0) 
    vth_curr  = vth 
    print("Initial Ev : "   , temp_t0 * collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT)
    print("Initial mass : " , m0_t0 )

    spec_sp.get_num_coefficients

    t1=time()
    advmat = spec_sp.compute_advection_matix()
    t2=time()
    print("Advection Operator assembly time (s): ",(t2-t1))
    # advmat = assemble_advection_matix_lp_lag(spec_sp._p, spec_sp._sph_harm_lm)

    FOp = 0

    t1=time()
    if "g0" in collisions_included:
        g0  = collisions.eAr_G0()
        g0.reset_scattering_direction_sp_mat()
        FOp = FOp + collOp.assemble_mat(g0, mw_vth, vth_curr)

    if "g0NoLoss" in collisions_included:
        g0noloss  = collisions.eAr_G0_NoEnergyLoss()
        g0noloss.reset_scattering_direction_sp_mat()
        FOp = FOp + collOp.assemble_mat(g0noloss, mw_vth, vth_curr)

    if "g0Const" in collisions_included:
        g0const  = collisions.eAr_G0(cross_section="g0Const")
        g0const.reset_scattering_direction_sp_mat()
        FOp = FOp + collOp.assemble_mat(g0const, mw_vth, vth_curr)

    if "g2" in collisions_included:
        g2  = collisions.eAr_G2()
        g2.reset_scattering_direction_sp_mat()
        FOp = FOp + collOp.assemble_mat(g2, mw_vth, vth_curr)

    if "g2Const" in collisions_included:
        g2const  = collisions.eAr_G2(cross_section="g2Const", threshold=0)
        g2const.reset_scattering_direction_sp_mat()
        FOp = FOp + collOp.assemble_mat(g2const, mw_vth, vth_curr)

    t2=time()
    print("Assembled the collision op. for Vth : ", vth_curr)
    print("Collision Operator assembly time (s): ",(t2-t1))
    
    FOp = np.matmul(Minv, FOp)

    # plt.show()
    if steady_state == True:

        h_red = np.zeros(len(h_init)-1)

        Cmat = collisions.AR_NEUTRAL_N*FOp
        Emat = E_field/VTH*collisions.ELECTRON_CHARGE_MASS_RATIO*advmat

        iteration_error = 1
        iteration_steps = 0
        while (iteration_error > 1e-14 and iteration_steps < 30) or iteration_steps < 5:
            h_prev = h_red
            h_red = - np.linalg.solve(Cmat[1:,1:] - Emat[1:,1:] - (Cmat[0,0] + np.dot(h_red, Cmat[0,1:]))*np.eye(len(h_red)), Cmat[1:,0] - Emat[1:,0])
            iteration_error = np.linalg.norm(h_prev-h_red)
            print("Iteration ", iteration_steps, ": Residual =", iteration_error)
            iteration_steps = iteration_steps + 1


        solution_vector = np.zeros((1,h_init.shape[0]))
        solution_vector[0,0] = 1
        solution_vector[0,1:] = h_red

    else:
        def f_rhs(t,y,n0,ni):
            return np.matmul(n0*FOp - E_field/VTH*collisions.ELECTRON_CHARGE_MASS_RATIO*advmat,y)

        ode_solver = ode(f_rhs,jac=None).set_integrator("dopri5",verbosity=1, rtol=t_tol, atol=t_tol, nsteps=10000)
        ode_solver.set_initial_value(h_init,t=0.0)
        t_step = 0
        total_steps = int(t_end/dt)
        solution_vector = np.zeros((total_steps,h_init.shape[0]))
        ht = h_init
        while ode_solver.successful() and t_step < total_steps: 
            t_curr   = ode_solver.t
            m0_t     = BEUtils.moment_n_f(spec_sp,ode_solver.y,mw_vth,vth,0,300,16,16,1)
            ode_solver.set_f_params(collisions.AR_NEUTRAL_N,m0_t)
            ht     = ode_solver.y
            solution_vector[t_step,:] = ht
            ode_solver.integrate(ode_solver.t + dt)
            t_step+=1

    return solution_vector

parser.add_argument("-Nr", "--NUM_P_RADIAL"                   , help="Number of polynomials in radial direction", nargs='+', type=int, default=32)
parser.add_argument("-T", "--T_END"                           , help="Simulation time", type=float, default=1e-4)
parser.add_argument("-dt", "--T_DT"                           , help="Simulation time step size ", type=float, default=1e-7)
parser.add_argument("-o",  "--out_fname"                      , help="output file name", type=str, default='coll_op')
parser.add_argument("-ts_tol", "--ts_tol"                     , help="adaptive timestep tolerance", type=float, default=1e-10)
parser.add_argument("-l_max", "--l_max"                       , help="max polar modes in SH expansion", type=int, default=1)
# parser.add_argument("-c", "--collisions"                      , help="collisions included (g0, g0Const, g0NoLoss, g2, g2Const)", type=str, default=["g0", "g2"])
parser.add_argument("-c", "--collisions"                      , help="collisions included (g0, g0Const, g0NoLoss, g2, g2Const)", type=str, default=["g0Const", "g2Const"])
parser.add_argument("-ev", "--electron_volt"                  , help="initial electron volt", type=float, default=.36)
parser.add_argument("-ev_basis", "--electron_volt_basis"      , help="basis electron volt", type=float, default=.36)
parser.add_argument("-q_vr", "--quad_radial"                  , help="quadrature in r"        , type=int, default=64)
parser.add_argument("-q_vt", "--quad_theta"                   , help="quadrature in polar"    , type=int, default=8)
parser.add_argument("-q_vp", "--quad_phi"                     , help="quadrature in azimuthal", type=int, default=8)
parser.add_argument("-q_st", "--quad_s_theta"                 , help="quadrature in scattering polar"    , type=int, default=8)
parser.add_argument("-q_sp", "--quad_s_phi"                   , help="quadrature in scattering azimuthal", type=int, default=8)
parser.add_argument("-radial_poly", "--radial_poly"           , help="radial basis", type=str, default="maxwell")
# parser.add_argument("-radial_poly", "--radial_poly"           , help="radial basis", type=str, default="laguerre")
parser.add_argument("-sp_order", "--spline_order"             , help="b-spline order", type=int, default=2)
parser.add_argument("-spline_qpts", "--spline_q_pts_per_knot" , help="q points per knots", type=int, default=11)
parser.add_argument("-E", "--E_field"                         , help="Electric field in V/m", type=float, default=10)
parser.add_argument("-dv", "--dv_target"                      , help="target displacement of distribution in v_th units", type=float, default=0)
parser.add_argument("-nt", "--num_timesteps"                  , help="target number of time steps", type=float, default=100)
parser.add_argument("-steady", "--steady_state"               , help="Steady state or transient", type=bool, default=True)
parser.add_argument("-bolsig", "--bolsig_data"                , help="Path to bolsig generated EEDF", type=str, default="./")

# parser.add_argument("-sweep_values", "--sweep_values"         , help="Values for parameter sweep", type=str, default=[0.125, 0.25, 0.5, 0.87, 1.0, 1.5])
# parser.add_argument("-sweep_param", "--sweep_param"           , help="Paramter to sweep: Nr, ev, ev_basis, E, radial_poly", type=str, default="ev_basis")

parser.add_argument("-sweep_values", "--sweep_values"         , help="Values for parameter sweep", type=str, default=[32])
parser.add_argument("-sweep_param", "--sweep_param"           , help="Paramter to sweep: Nr, ev, ev_basis, E, radial_poly", type=str, default="Nr")

# parser.add_argument("-sweep_values", "--sweep_values"         , help="Values for parameter sweep", type=str, default=["maxwell", "laguerre"])
# parser.add_argument("-sweep_param", "--sweep_param"           , help="Paramter to sweep: Nr, ev, ev_basis, E, radial_poly", type=str, default="radial_poly")

#parser.add_argument("-r", "--restore", help="if 1 try to restore solution from a checkpoint", type=int, default=0)
args = parser.parse_args()

run_data=list()
run_temp=list()

v = np.linspace(-2,2,100)
vx, vz = np.meshgrid(v,v,indexing='ij')
vy = np.zeros_like(vx)
v_sph_coord = BEUtils.cartesian_to_spherical(vx, vy, vz)

# vr = np.linspace(1e-3,5,100)
ev = np.linspace(args.electron_volt/20.,20.*args.electron_volt,100)

# assuming l_max is not changing
# params.BEVelocitySpace.SPH_HARM_LM = [[i,j] for i in range(args.l_max+1) for j in range(-i,i+1)]
params.BEVelocitySpace.SPH_HARM_LM = [[i,0] for i in range(args.l_max+1)]
num_sph_harm = len(params.BEVelocitySpace.SPH_HARM_LM)

radial = np.zeros((len(args.sweep_values), num_sph_harm, len(ev)))
radial_intial = np.zeros((num_sph_harm, len(ev)))

density_slice         = np.zeros((len(args.sweep_values),len(vx[0]),len(vx[1])))
density_slice_initial = np.zeros((len(args.sweep_values),len(vx[0]),len(vx[1])))

SPLINE_ORDER = args.spline_order
basis.BSPLINE_BASIS_ORDER=SPLINE_ORDER
basis.XLBSPLINE_NUM_Q_PTS_PER_KNOT=args.spline_q_pts_per_knot

for i, value in enumerate(args.sweep_values):

    if args.sweep_param == "Nr":
        args.NUM_P_RADIAL = value
    # elif args.sweep_param == "l_max":
    #     args.l_max = value
    elif args.sweep_param == "ev":
        args.electron_volt = value
        args.electron_volt_basis = value
    elif args.sweep_param == "ev_basis":
        args.electron_volt_basis = value
    elif args.sweep_param == "E":
        args.E_field = value
    elif args.sweep_param == "radial_poly":
        args.radial_poly = value

    if args.dv_target != 0 and args.E_field != 0:
        args.T_END = args.dv_target/args.E_field*collisions.electron_thermal_velocity(args.electron_volt*collisions.TEMP_K_1EV)/collisions.ELECTRON_CHARGE_MASS_RATIO

    if args.num_timesteps != 0:
        args.T_DT = args.T_END/args.num_timesteps

    params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER = args.NUM_P_RADIAL
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

    INIT_EV    = args.electron_volt_basis
    vratio = np.sqrt(args.electron_volt/args.electron_volt_basis)

    # INIT_EV    = evcurr
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

    maxwellian = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)
    hv         = lambda v,vt,vp : np.exp((v**2)*(1.-1./(vratio**2)))/vratio**3
    h_vec      = BEUtils.function_to_basis(spec,hv,maxwellian,None,None,None)

    data       = solve_collop(args.steady_state, cf, h_vec, maxwellian, VTH, args.E_field, args.T_END, args.T_DT, args.ts_tol, collisions_included=args.collisions)
    
    radial[i, :, :] = BEUtils.compute_radial_components(ev, spec, data[-1,:], maxwellian, VTH, 1)#/VTH**3

    if args.steady_state == False and i == 0:
        radial_initial = BEUtils.compute_radial_components(ev, spec, data[0,:], maxwellian, VTH, 1)#/VTH**3

    density_slice[i]  = BEUtils.sample_distriubtion_spherical(v_sph_coord, spec, data[-1,:], maxwellian, VTH, 1)
    density_slice_initial[i]  = BEUtils.sample_distriubtion_spherical(v_sph_coord, spec, data[0,:], maxwellian, VTH, 1)
    run_data.append(data)


    nt = len(data[:,0])
    temp_evolution = np.zeros(nt)

    for k in range(nt):
        current_mass     = BEUtils.moment_n_f(spec,data[k,:], maxwellian, VTH, 0,None,None,None,1)
        current_temp     = BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec,data[k,:], maxwellian, VTH, None,None,None,current_mass,1)
        temp_evolution[k] = current_temp/collisions.TEMP_K_1EV

    print(temp_evolution[-1])
    
    run_temp.append(temp_evolution)


# np.set_printoptions(precision=16)
# print(data[1,:])

if (1):
    fig = plt.figure(figsize=(12, 4), dpi=150)

    num_subplots = num_sph_harm + 1

    for i, value in enumerate(args.sweep_values):
        data=run_data[i]

        lbl = args.sweep_param+"="+str(value)

        # spherical components plots
        for l_idx in range(num_sph_harm):

            plt.subplot(2, num_subplots, 1+l_idx)

            plt.plot(abs(data[-1,l_idx::num_sph_harm]),label=lbl)

            plt.title(label="l=%d"%l_idx)
            plt.yscale('log')
            plt.xlabel("Coeff. #")
            plt.ylabel("Coeff. magnitude")
            plt.grid(visible=True)
            if l_idx == 0:
                plt.legend()

            plt.subplot(2, num_subplots, num_subplots + 1 + l_idx)

            if args.steady_state == False:
                plt.semilogy(ev,  abs(radial_initial[l_idx]), '-', label="Initial")

            color = next(plt.gca()._get_lines.prop_cycler)['color']
            plt.semilogy(ev,  abs(radial[i, l_idx]), '-', label=lbl, color=color)
            plt.semilogy(ev, -radial[i, l_idx], 'o', label=lbl, color=color, markersize=3, markerfacecolor='white')

            plt.yscale('log')
            plt.xlabel("Scaled speed")
            plt.ylabel("Radial component")
            plt.grid(visible=True)
            # plt.legend()

        plt.subplot(2, num_subplots, num_sph_harm + 1)
        temp = run_temp[i]
        plt.plot(temp, label=lbl)

    plt.subplot(2, num_subplots, num_subplots + num_sph_harm + 1)

    lvls = np.linspace(0, np.amax(density_slice_initial[0]), 10)

    plt.contour(vx, vz, density_slice_initial[-1], levels=lvls, linestyles='solid', colors='grey', linewidths=1) 
    plt.contour(vx, vz, density_slice[-1], levels=lvls, linestyles='dotted', colors='red', linewidths=1)  
    plt.gca().set_aspect('equal')
    
    fig.subplots_adjust(hspace=0.15)
    fig.subplots_adjust(wspace=0.2)
    plt.show()
    # plt.savefig("%s_coeff.png"%(args.out_fname))