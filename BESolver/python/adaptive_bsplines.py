"""
linear adaptive trees
"""
import numpy as np
import collisions
import basis
import quadpy
import utils as BEUtils
import bolsig
import sys
import scipy.interpolate
import parameters as params
import argparse
import spec_spherical as sp
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("-Nr", "--NUM_P_RADIAL"   , help="Number of polynomials in radial direction", nargs='+', type=int, default=16)
parser.add_argument("-T", "--T_END"           , help="Simulation time", type=float, default=1e-4)
parser.add_argument("-dt", "--T_DT"           , help="Simulation time step size ", type=float, default=1e-7)
parser.add_argument("-o",  "--out_fname"      , help="output file name", type=str, default='coll_op')
parser.add_argument("-ts_tol", "--ts_tol"     , help="adaptive timestep tolerance", type=float, default=1e-15)
parser.add_argument("-l_max", "--l_max"       , help="max polar modes in SH expansion", type=int, default=1)
parser.add_argument("-c", "--collisions"      , help="collisions included (g0, g0Const, g0NoLoss, g2, g2Const)",nargs='+', type=str, default=["g0Const"])
parser.add_argument("-ev", "--electron_volt"  , help="initial electron volt", type=float, default=0.25)
parser.add_argument("-bscale", "--basis_scale", help="basis electron volt", type=float, default=1.0)
parser.add_argument("-q_vr", "--quad_radial"  , help="quadrature in r"        , type=int, default=200)
parser.add_argument("-q_vt", "--quad_theta"   , help="quadrature in polar"    , type=int, default=8)
parser.add_argument("-q_vp", "--quad_phi"     , help="quadrature in azimuthal", type=int, default=8)
parser.add_argument("-q_st", "--quad_s_theta" , help="quadrature in scattering polar"    , type=int, default=8)
parser.add_argument("-q_sp", "--quad_s_phi"   , help="quadrature in scattering azimuthal", type=int, default=8)
parser.add_argument("-radial_poly", "--radial_poly"           , help="radial basis", type=str, default="bspline")
parser.add_argument("-sp_order", "--spline_order"             , help="b-spline order", type=int, default=1)
parser.add_argument("-spline_qpts", "--spline_q_pts_per_knot" , help="q points per knots", type=int, default=7)
parser.add_argument("-E", "--E_field"                         , help="Electric field in V/m", type=float, default=100)
parser.add_argument("-dv", "--dv_target"                      , help="target displacement of distribution in v_th units", type=float, default=0)
parser.add_argument("-nt", "--num_timesteps"                  , help="target number of time steps", type=float, default=100)
parser.add_argument("-steady", "--steady_state"               , help="Steady state or transient", type=bool, default=True)
parser.add_argument("-run_bolsig_only", "--run_bolsig_only"   , help="run the bolsig code only", type=bool, default=False)
parser.add_argument("-bolsig", "--bolsig_dir"                 , help="Bolsig directory", type=str, default="../../Bolsig/")
parser.add_argument("-sweep_values", "--sweep_values"         , help="Values for parameter sweep", nargs='+', type=int, default=[24, 48, 96])
parser.add_argument("-sweep_param", "--sweep_param"           , help="Paramter to sweep: Nr, ev, bscale, E, radial_poly", type=str, default="Nr")

args = parser.parse_args()
print(args)
# bolsig.run_bolsig(args)

# if (args.run_bolsig_only):
#     sys.exit(0)

# [bolsig_ev, bolsig_f0, bolsig_a, bolsig_E, bolsig_mu, bolsig_M, bolsig_D, bolsig_rates] = bolsig.parse_bolsig(args.bolsig_dir+"argon.out",len(args.collisions))

# print("blolsig temp : %.8E"%((bolsig_mu /1.5)))
# args.electron_volt = (bolsig_mu/1.5)

# f0_cf = scipy.interpolate.interp1d(bolsig_ev, bolsig_f0, kind='cubic', bounds_error=False, fill_value=(bolsig_f0[0],bolsig_f0[-1]))
# fa_cf = scipy.interpolate.interp1d(bolsig_ev, bolsig_a,  kind='cubic', bounds_error=False, fill_value=(bolsig_a[0],bolsig_a[-1]))


BASIS_EV                       = args.electron_volt*args.basis_scale
collisions.MAXWELLIAN_TEMP_K   = BASIS_EV * collisions.TEMP_K_1EV
collisions.ELECTRON_THEMAL_VEL = collisions.electron_thermal_velocity(collisions.MAXWELLIAN_TEMP_K) 
VTH                            = collisions.ELECTRON_THEMAL_VEL
maxwellian                     = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)
c_gamma                        = np.sqrt(2*collisions.ELECTRON_CHARGE_MASS_RATIO)

if "g2Smooth" in args.collisions or "g2" in args.collisions or "g2step" in args.collisions or "g2Regul" in args.collisions:
    sig_pts = np.array([np.sqrt(15.76) * c_gamma/VTH])
else:
    sig_pts = None

ev_range = ((0*VTH/c_gamma)**2, (4*VTH/c_gamma)**2)
k_domain = (np.sqrt(ev_range[0]) * c_gamma / VTH, np.sqrt(ev_range[1]) * c_gamma / VTH)
print("target ev range : (%.4E, %.4E) ----> knots domain : (%.4E, %.4E)" %(ev_range[0], ev_range[1], k_domain[0],k_domain[1]))
if(sig_pts is not None):
    print("singularity pts : ", sig_pts,"v/vth", (sig_pts * VTH/c_gamma)**2,"eV")

params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER = args.NUM_P_RADIAL
tt_vec = None

def refine_func(x):
    y    = np.zeros_like(x)
    x_ev = (x * VTH / c_gamma)**2
    collisions_included = args.collisions
    for c in collisions_included:
        y+= VTH * (x**3) * maxwellian(x) * collisions.Collisions.synthetic_tcs(x_ev, c) * collisions.AR_NEUTRAL_N
    
    return y#maxwellian(x)

max_lev  = int(np.ceil(np.log2(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER)))
tt_vec   = basis.BSpline.adaptive_fit(refine_func, k_domain, sp_order=args.spline_order, min_lev=4, max_lev=max_lev, sig_pts=sig_pts, atol=1e-40, rtol=1e-10)
bb       = basis.BSpline(k_domain, args.spline_order, 0, sig_pts=None, knots_vec=tt_vec)
bb_u     = basis.BSpline(k_domain, args.spline_order, bb._num_p, sig_pts=sig_pts, knots_vec=None)

params.BEVelocitySpace.NUM_Q_VR = bb._num_knot_intervals * args.spline_q_pts_per_knot
params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER = bb._num_p-1

spec_sp               = sp.SpectralExpansionSpherical(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER, bb,params.BEVelocitySpace.SPH_HARM_LM)
spec_sp._num_q_radial = params.BEVelocitySpace.NUM_Q_VR
params.BEVelocitySpace.NUM_Q_VT  = args.quad_theta
params.BEVelocitySpace.NUM_Q_VP  = args.quad_phi
params.BEVelocitySpace.NUM_Q_CHI = args.quad_s_theta
params.BEVelocitySpace.NUM_Q_PHI = args.quad_s_phi
params.BEVelocitySpace.VELOCITY_SPACE_DT = args.T_DT

print("""===========================Parameters ======================""")
print("\tMAXWELLIAN_N        : ", collisions.MAXWELLIAN_N)
print("\tELECTRON_THEMAL_VEL : ", VTH," ms^-1")
print("\tBASIS_EV            : ", BASIS_EV,"eV")
print("\tDT : ", params.BEVelocitySpace.VELOCITY_SPACE_DT, " s")
print("""============================================================""")
params.print_parameters()

fc   = bb.fit(refine_func)
fc_u = bb_u.fit(refine_func)


vx   = np.linspace(k_domain[0],k_domain[1],1000)
fx   = np.sum(np.array([fc[i] * bb.Pn(i)(vx,0) for i in range(bb._num_p)]),axis=0)
fx_u = np.sum(np.array([fc_u[i] * bb_u.Pn(i)(vx,0) for i in range(bb_u._num_p)]),axis=0)

ff   = refine_func(vx)

l2_u = np.linalg.norm(fx_u   - ff)   / np.linalg.norm(ff)
l2   = np.linalg.norm(fx - ff)   / np.linalg.norm(ff)


plt.semilogy(vx, ff   , 'b--', label="f(x)")
#plt.semilogy(vx, fx   , label=" adaptive")
#plt.semilogy(vx, fx_u , label=" uniform")
plt.semilogy(bb._t  , refine_func(bb._t)  , "*" , label="adaptive knots")
plt.semilogy(bb_u._t, refine_func(bb_u._t), "x" , label="uniform knots")
plt.legend()
plt.grid()
plt.show()

print("l2 error adaptive : %8E"%(l2))
print("l2 error uniform  : %8E"%(l2_u))




