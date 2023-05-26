"""
Markov-Chain MC sampling for distribution functions. 
"""
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["OMP_NUM_THREADS"] = "8"
import scipy
import scipy.optimize
import scipy.interpolate
import basis
import spec_spherical as sp
import numpy as np
import collision_operator_spherical as colOpSp
import collisions 
import parameters as params
from   time import perf_counter as time, sleep
import utils as BEUtils
import argparse
import scipy.integrate
from   scipy.integrate import ode
from   advection_operator_spherical_polys import *
import scipy.ndimage
import matplotlib.pyplot as plt
from   scipy.interpolate import interp1d
import scipy.sparse.linalg
import sys
import bolsig
import csv
import bte_0d3v_solver as bte_0d3v
import utils as bte_utils

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 10,
    #"ytick.major.size": 3,
    #"font.family": "Helvetica",
    "lines.linewidth":2.0
})

parser = argparse.ArgumentParser()
parser.add_argument("-Nr", "--NUM_P_RADIAL"                       , help="Number of polynomials in radial direction", type=int, default=16)
parser.add_argument("-T", "--T_END"                               , help="Simulation time", type=float, default=1e-4)
parser.add_argument("-dt", "--T_DT"                               , help="Simulation time step size ", type=float, default=1e-7)
parser.add_argument("-o",  "--out_fname"                          , help="output file name", type=str, default='pde_vs_bolsig')
parser.add_argument("-l_max", "--l_max"                           , help="max polar modes in SH expansion", type=int, default=1)
parser.add_argument("-l_pt_mode", "--l_pt_mode"                   , help="perturb mode", type=int, default=2)
parser.add_argument("-c", "--collisions"                          , help="collisions included (g0, g0Const, g0NoLoss, g2, g2Const)",nargs='+', type=str, default=["g0Const"])
parser.add_argument("-ev", "--electron_volt"                      , help="initial electron volt", type=float, default=0.25)
parser.add_argument("-q_vr", "--quad_radial"                      , help="quadrature in r"        , type=int, default=200)
parser.add_argument("-q_vt", "--quad_theta"                       , help="quadrature in polar"    , type=int, default=8)
parser.add_argument("-q_vp", "--quad_phi"                         , help="quadrature in azimuthal", type=int, default=8)
parser.add_argument("-q_st", "--quad_s_theta"                     , help="quadrature in scattering polar"    , type=int, default=8)
parser.add_argument("-q_sp", "--quad_s_phi"                       , help="quadrature in scattering azimuthal", type=int, default=8)
parser.add_argument("-sp_order", "--sp_order"                     , help="b-spline order", type=int, default=3)
parser.add_argument("-spline_qpts", "--spline_qpts"               , help="q points per knots", type=int, default=5)
parser.add_argument("-EbyN", "--EbyN"                             , help="Effective electric field in Td", type=float, nargs='+', default=1)
parser.add_argument("-efield_period", "--efield_period"           , help="Oscillation period in seconds", type=float, default=0.0)
parser.add_argument("-E", "--E_field"                             , help="Electric field in V/m", type=float, default=100)
parser.add_argument("-steady", "--steady_state"                   , help="Steady state or transient", type=int, default=1)
parser.add_argument("-sweep_values", "--sweep_values"             , help="Values for parameter sweep", nargs='+', type=float, default=[32, 64, 128])
parser.add_argument("-sweep_param", "--sweep_param"               , help="Parameter to sweep: Nr, ev, bscale, E, radial_poly", type=str, default="Nr")
parser.add_argument("-dg", "--use_dg"                             , help="enable dg splines", type=int, default=0)
parser.add_argument("-Tg", "--Tg"                                 , help="Gas temperature (K)" , type=float, default=1e-12)
parser.add_argument("-ion_deg", "--ion_deg"                       , help="Ionization degree"   , type=float, nargs='+', default=[1e-1])
parser.add_argument("-store_eedf", "--store_eedf"                 , help="store EEDF"          , type=int, default=0)
parser.add_argument("-store_csv", "--store_csv"                   , help="store csv format of QoI comparisons", type=int, default=0)
parser.add_argument("-ee_collisions", "--ee_collisions"           , help="Enable electron-electron collisions", type=float, default=1)
parser.add_argument("-bolsig", "--bolsig_dir"                     , help="Bolsig directory", type=str, default="../../Bolsig/")
parser.add_argument("-bolsig_precision", "--bolsig_precision"     , help="precision value for bolsig code", type=float, default=1e-11)
parser.add_argument("-bolsig_convergence", "--bolsig_convergence" , help="convergence value for bolsig code", type=float, default=1e-8)
parser.add_argument("-bolsig_grid_pts", "--bolsig_grid_pts"       , help="grid points for bolsig code"      , type=int, default=1024)

args                = parser.parse_args()
EbyN_Td             = np.logspace(np.log10(args.EbyN[0]), np.log10(args.EbyN[1]), int(args.EbyN[2]), base=10)
e_values            = EbyN_Td * collisions.AR_NEUTRAL_N * 1e-21
ion_deg_values      = np.array(args.ion_deg)

if not args.ee_collisions:
    ion_deg_values = np.array([0])

run_params          = [(e_values[i], ion_deg_values[j]) for i in range(len(e_values)) for j in range(len(ion_deg_values))]
for run_id in range(len(run_params)):
    args.E_field = run_params[run_id][0]
    args.ion_deg = run_params[run_id][1]
    #print(args)

    bte_solver = bte_0d3v.bte_0d3v(args)
    bte_solver.run_bolsig_solver()
    ev = bte_solver._bolsig_data["ev"]
    num_sh = args.l_max + 1

    radial_base       = np.zeros((num_sh, len(ev)))
    radial            = np.zeros((num_sh, len(ev)))

    assert len(args.sweep_values)==1
    assert args.sweep_param == "Nr"

    if args.sweep_param == "Nr":
        args.NUM_P_RADIAL = int(args.sweep_values[0])
    else:
        raise NotImplementedError
        
    bte_solver._args = args
    bte_solver.setup()

    if args.steady_state == 1:
        
        if args.ee_collisions ==0 and args.l_max ==1:
            r_data   = bte_solver.steady_state_solver_two_term()
        else:
            r_data    = bte_solver.steady_state_solver()
    
    else:
        r_data    = bte_solver.transient_solver(args.T_END, args.T_DT)
    
    data      = r_data['sol']
    h_bolsig  = r_data['h_bolsig']
    abs_tol   = r_data['atol']
    rel_tol   = r_data['rtol']

    spec_sp   = bte_solver._spec_sp 
    mw        = bte_solver._mw
    vth       = bte_solver._vth

    num_p     = spec_sp._p + 1
    num_sh    = len(spec_sp._sph_harm_lm)    

    # radial_base[:,:]  = BEUtils.compute_radial_components(ev, spec_sp, data[0,:] , mw, vth, 1)
    radial[:, :]      = BEUtils.compute_radial_components(ev, spec_sp, data[-1,:], mw, vth, 1)


    ss_sol = np.zeros(num_p * num_sh) 
    ss_sol[0::num_sh] = data[-1,0::num_sh]
    ss_sol[1::num_sh] = data[-1,1::num_sh]
    mm_op  = bte_solver._mass_op * mw(0) * vth**3

    ss_sol /= np.dot(mm_op,ss_sol)
    if num_sh> args.l_pt_mode:
        # to perturb mode 2 and sets all the other modes to zero. 
        ss_sol[args.l_pt_mode::num_sh] = data[0,0::num_sh]
    
    def ss_dist(vv):
        s=0
        for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
            sph_v = spec_sp.basis_eval_spherical(vv[1],vv[2],lm[0],lm[1])
            for k in range(num_p):
                bk_vr = spec_sp.basis_eval_radial(vv[0],k,0)
                s+=ss_sol[k * num_sh + lm_idx] * bk_vr * sph_v * vv[0]**2 * np.sin(vv[1])
        return s

    mu  = np.zeros(3)
    cov = np.array([[np.sqrt(0.5), 0, 0], [0, np.sqrt(0.5), 0], [0, 0, np.sqrt(0.5)]])
    def prior_dist():
        x   = np.random.multivariate_normal(mu,cov)
        xp  = bte_utils.cartesian_to_spherical(x[0], x[1], x[2])
        return np.array(xp)

    samples = bte_utils.mcmc_sampling(ss_dist, prior_dist, 2000, burn_in_fac=0.5)
    
    vr    = np.linspace(0,3,100)
    vq    = spec_sp.Vq_r(vr,0,1)
    ss_r0 = np.dot(ss_sol[0::num_sh], vq) * vr**2 

    num_plt_cols = 3
    num_plt_rows = 2

    fig  = plt.figure(figsize=(num_plt_cols * 6 + 0.5*(num_plt_cols-1), num_plt_rows * 6 + 0.5*(num_plt_rows-1)), dpi=300, constrained_layout=True)

    plt.subplot(num_plt_rows, num_plt_cols, 1)
    plt.semilogy(bte_solver._bolsig_data["ev"], np.abs(bte_solver._bolsig_data["f0"]), 'k-', label="bolsig")
    plt.semilogy(ev, np.abs(radial[0,:]),'r-', label="pde")
    plt.xlabel(r"ev")
    plt.title(r"$f_0$")
    plt.legend()
    plt.grid(visible=True)
    
    plt.subplot(num_plt_rows, num_plt_cols, 2)
    plt.semilogy(bte_solver._bolsig_data["ev"], np.abs(bte_solver._bolsig_data["f1"]), 'k-', label="bolsig")
    plt.semilogy(ev, np.abs(radial[1,:]),'r-', label="pde")
    plt.xlabel(r"ev")
    plt.title(r"$f_1$")
    plt.grid(visible=True)
   
    plt.subplot(num_plt_rows, num_plt_cols, 4)
    plt.plot(vr, ss_r0, 'b-')
    plt.hist(samples[:,0],bins=30,density=1)
    plt.xlabel(r"$v_r$")
    plt.grid(visible=True)

    plt.subplot(num_plt_rows, num_plt_cols, 5)
    plt.hist(samples[:,1],bins=30,density=1)
    plt.xlabel(r"$v_\theta$")
    plt.grid(visible=True)

    plt.subplot(num_plt_rows, num_plt_cols, 6)
    plt.hist(samples[:,2],bins=30,density=1)
    plt.xlabel(r"$v_\phi$")
    plt.grid(visible=True)

    #plt.show()
    plt.savefig("sampling_example.png")
    plt.close()




    

    
