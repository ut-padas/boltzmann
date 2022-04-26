from cProfile import label
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

collisions.AR_NEUTRAL_N=3.22e22
collisions.MAXWELLIAN_N=1e18
collisions.AR_IONIZED_N=collisions.MAXWELLIAN_N
parser = argparse.ArgumentParser()

parser.add_argument("-Nr", "--NUM_P_RADIAL"                   , help="Number of polynomials in radial direction", nargs='+', type=int, default=[4,8,16,32,64])
parser.add_argument("-T", "--T_END"                           , help="Simulation time", type=float, default=1e-6)
parser.add_argument("-dt", "--T_DT"                           , help="Simulation time step size ", type=float, default=1e-10)
parser.add_argument("-o",  "--out_fname"                      , help="output file name", type=str, default='coll_op')
parser.add_argument("-ts_tol", "--ts_tol"                     , help="adaptive timestep tolerance", type=float, default=1e-15)
parser.add_argument("-l_max", "--l_max"                       , help="max polar modes in SH expansion", type=int, default=0)
parser.add_argument("-c", "--collision_mode"                  , help="collision mode", type=str, default="g0")
parser.add_argument("-ev", "--electron_volt"                  , help="initial electron volt", type=float, default=1.0)
parser.add_argument("-q_vr", "--quad_radial"                  , help="quadrature in r"        , type=int, default=270)
parser.add_argument("-q_vt", "--quad_theta"                   , help="quadrature in polar"    , type=int, default=2)
parser.add_argument("-q_vp", "--quad_phi"                     , help="quadrature in azimuthal", type=int, default=2)
parser.add_argument("-q_st", "--quad_s_theta"                 , help="quadrature in scattering polar"    , type=int, default=2)
parser.add_argument("-q_sp", "--quad_s_phi"                   , help="quadrature in scattering azimuthal", type=int, default=2)
parser.add_argument("-radial_poly", "--radial_poly"           , help="radial basis", type=str, default="maxwell")
parser.add_argument("-sp_order", "--spline_order"             , help="b-spline order", type=int, default=2)
parser.add_argument("-spline_qpts", "--spline_q_pts_per_knot" , help="q points per knots", type=int, default=11)
#parser.add_argument("-r", "--restore", help="if 1 try to restore solution from a checkpoint", type=int, default=0)
args = parser.parse_args()

L_MAX=args.l_max
r_mode = basis.BasisType.MAXWELLIAN_POLY
params.BEVelocitySpace.NUM_Q_VR  = args.quad_radial
params.BEVelocitySpace.NUM_Q_VT  = args.quad_theta
params.BEVelocitySpace.NUM_Q_VP  = args.quad_phi
params.BEVelocitySpace.NUM_Q_CHI = args.quad_s_theta
params.BEVelocitySpace.NUM_Q_PHI = args.quad_s_phi
params.BEVelocitySpace.VELOCITY_SPACE_DT = args.T_DT
params.BEVelocitySpace.SPH_HARM_LM = [[i,j] for i in range(L_MAX+1) for j in range(-i,i+1)]

INIT_EV    = args.electron_volt
collisions.MAXWELLIAN_TEMP_K   = INIT_EV * collisions.TEMP_K_1EV
collisions.ELECTRON_THEMAL_VEL = collisions.electron_thermal_velocity(collisions.MAXWELLIAN_TEMP_K) 
VTH   = collisions.ELECTRON_THEMAL_VEL
maxwellian = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)

Nr=args.NUM_P_RADIAL
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(4, 4),dpi=300) #(figsize=(6, 6), dpi=300)

SPLINE_ORDER = args.spline_order
basis.BSPLINE_BASIS_ORDER=SPLINE_ORDER
basis.XLBSPLINE_NUM_Q_PTS_PER_KNOT=args.spline_q_pts_per_knot
for i, nr in enumerate(Nr):
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
    params.print_parameters()
    
    cf    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM, nr,poly_type=r_mode)
    spec  = cf._spec
    mm    = spec.compute_mass_matrix()
    mm    = np.linalg.inv(mm)

    if args.collision_mode == "g0":
        g  = collisions.eAr_G0()
    if args.collision_mode == "g2":
        g  = collisions.eAr_G2()
        
    g.reset_scattering_direction_sp_mat()
    t1=time()
    FOp = cf.assemble_mat(g,maxwellian,VTH)
    t2=time()
    #FOp = np.matmul(mm, FOp)
    FOp  = FOp / ((spec._p +1) * len(spec._sph_harm_lm))
    print("Assembled the collision op. for Vth : ", VTH)
    print("Collision Operator assembly time (s): ",(t2-t1))
    #coll_mats.append(FOp)
    u, s, v = np.linalg.svd(FOp)
    # pt_c=1
    # for k in range(Nr[0]):
    #     plt.subplot(Nr[0], 2, pt_c)
    #     plt.plot(u[:,k],label="L Nr=%d"%(nr))
    #     #plt.legend()
    #     plt.subplot(Nr[0], 2, pt_c+1)
    #     plt.plot(v[k,:],label="R Nr=%d"%(nr))
    #     #plt.legend()
    #     pt_c+=2
    plt.plot(s,label="Nr=%d"%(nr),linewidth=0.5)
    plt.yscale("log")
    plt.xscale("log")


plt.legend()
plt.tight_layout()
plt.savefig("%s.png"%(args.out_fname))
#plt.close()





