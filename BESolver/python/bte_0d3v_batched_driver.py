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
from   datetime import datetime
from   bte_0d3v_batched import bte_0d3v_batched
import cupy as cp
import matplotlib.pyplot as plt
import csv
import sys


plt.rcParams.update({
    "text.usetex": False,
    "font.size": 24,
    #"ytick.major.size": 3,
    #"font.family": "Helvetica",
    "lines.linewidth":1.0
})


parser = argparse.ArgumentParser()
parser.add_argument("-threads", "--threads"                       , help="number of cpu threads", type=int, default=4)
parser.add_argument("-out_fname", "--out_fname"                     , help="output file name for the qois", type=str, default="bte")
parser.add_argument("-l_max", "--l_max"                           , help="max polar modes in SH expansion", type=int, default=1)
parser.add_argument("-c", "--collisions"                          , help="collisions model",nargs='+', type=str, default=["g0","g2"])
parser.add_argument("-sp_order", "--sp_order"                     , help="b-spline order", type=int, default=3)
parser.add_argument("-spline_qpts", "--spline_qpts"               , help="q points per knots", type=int, default=5)
parser.add_argument("-steady", "--steady_state"                   , help="steady state or transient", type=int, default=1)
parser.add_argument("-max_iter", "--max_iter"                     , help="max number of iterations for newton solve", type=int, default=300)
parser.add_argument("-Tg", "--Tg"                                 , help="gas temperature (K)" , type=float, default=0.0)
parser.add_argument("-n0", "--n0"                                 , help="heavy density (1/m^3)" , type=float, default=3.22e22)
parser.add_argument("-Nr", "--Nr"                                 , help="radial refinement", type=int, default=128)
parser.add_argument("-profile", "--profile"                       , help="profile", type=int, default=0)
parser.add_argument("-warm_up", "--warm_up"                       , help="warm up", type=int, default=5)
parser.add_argument("-runs", "--runs"                             , help="runs "  , type=int, default=10)
parser.add_argument("-n_pts", "--n_pts"                           , help="number of points for batched solver", type=int, default=10)
parser.add_argument("-store_eedf", "--store_eedf"                 , help="store EEDF"          , type=int, default=0)
parser.add_argument("-store_csv", "--store_csv"                   , help="store csv format of QoI comparisons", type=int, default=1)
parser.add_argument("-plot_data", "--plot_data"                   , help="plot data", type=int, default=1)
parser.add_argument("-ee_collisions", "--ee_collisions"           , help="enable electron-electron collisions", type=float, default=0)
parser.add_argument("-verbose", "--verbose"                       , help="verbose with debug information", type=int, default=0)

args        = parser.parse_args()
n_grids     = 1
n_pts       = args.n_pts
Te          = np.ones(n_grids) * 1.6 * collisions.TEMP_K_1EV

ef          = np.linspace(0.6  , 1, n_pts) * 100
n0          = np.linspace(0.5  , 1, n_pts) * 3.22e22
ne          = np.linspace(1e-3 , 1, n_pts) * 3.22e21
ni          = np.linspace(1e-3 , 1, n_pts) * 3.22e21
Tg          = np.linspace(0.5  , 1, n_pts) * 13000#0.5 * collisions.TEMP_K_1EV

# ef          = np.ones(n_pts) * 96.6
# n0          = np.ones(n_pts) * 3.22e22
# ne          = np.ones(n_pts) * 3.22e21
# ni          = np.ones(n_pts) * 3.22e21
# Tg          = np.ones(n_pts) * 13000#0.5 * collisions.TEMP_K_1EV

lm_modes    = [[l,0] for l in range(args.l_max+1)]
nr          = np.ones(n_pts, dtype=np.int32) * args.Nr

bte_solver = bte_0d3v_batched(args,Te, nr, lm_modes, n_grids, args.collisions)
f0         = bte_solver.initialize(0, n_pts,"maxwellian")

bte_solver.set_boltzmann_parameters(0, n0, ne, ni, ef, Tg)
bte_solver.host_to_device_setup(0)
f0       = cp.asarray(f0)

if args.profile==1:
    res_func, jac_func = bte_solver.get_rhs_and_jacobian(0, f0.shape[1], 16)
    for i in range(args.warm_up):
        a=res_func(f0)
        b=jac_func(f0)
    
    bte_solver.profile_reset()
    for i in range(args.runs):
        a=res_func(f0)
        b=jac_func(f0)
    
    bte_solver.profile_stats()
    sys.exit(0)

ff , qoi = bte_solver.steady_state_solve(0, f0, 1e-15, 1e-10, args.max_iter)
ev       = np.linspace(1e-3, bte_solver._par_ev_range[0][1], 500)
ff_r     = bte_solver.compute_radial_components(0, ev, ff)
bte_solver.device_to_host_setup(0)

ff_r     = cp.asnumpy(ff_r)
for k, v in qoi.items():
    qoi[k] = cp.asnumpy(v)

csv_write = args.store_csv
if csv_write:
    fname = args.out_fname
    with open("%s_qoi.csv"%fname, 'w', encoding='UTF8') as f:
        writer = csv.writer(f,delimiter=',')
        # write the header
        header = ["n0", "ne", "ni", "Tg", "E",  "energy", "mobility", "diffusion"]
        for col_idx, g in enumerate(args.collisions):
            header.append(str(g))
            
        writer.writerow(header)
        
        data = np.concatenate((n0.reshape(-1,1), ne.reshape(-1,1), ni.reshape(-1,1), Tg.reshape(-1,1), ef.reshape(-1,1), qoi["energy"].reshape(-1,1), qoi["mobility"].reshape(-1,1), qoi["diffusion"].reshape(-1,1)), axis=1)
        for col_idx, g in enumerate(args.collisions):
            data = np.concatenate((data, qoi["rates"][col_idx].reshape(-1,1)), axis=1)
        
        writer.writerows(data)


plot_data    = args.plot_data
if plot_data:
    num_sh       = len(lm_modes)
    num_subplots = num_sh 
    num_plt_cols = min(num_sh, 4)
    num_plt_rows = np.int64(np.ceil(num_subplots/num_plt_cols))
    fig        = plt.figure(figsize=(num_plt_cols * 8 + 0.5*(num_plt_cols-1), num_plt_rows * 8 + 0.5*(num_plt_rows-1)), dpi=300, constrained_layout=True)
    plt_idx    =  1
    n_pts_step =  4

    for lm_idx, lm in enumerate(lm_modes):
        plt.subplot(num_plt_rows, num_plt_cols, plt_idx)
        for ii in range(0, n_pts, n_pts_step):
            fr = np.abs(ff_r[ii, lm_idx, :])
            valid_idx = fr>1e-11
            plt.semilogy(ev[valid_idx], fr[valid_idx], label="Tg=%.2E[K], E=%.2E[V/m], n0=%.2E[1/m^3], ne=%.2E[1/m^3]"%(Tg[ii], ef[ii], n0[ii], ne[ii]))
        
        plt.xlabel(r"energy (eV)")
        plt.ylabel(r"$f_%d$"%(lm[0]))
        plt.grid(visible=True)
        if lm_idx==0:
            plt.legend(prop={'size': 6})
            
        plt_idx +=1
    
    #plt_idx = num_sh
    plt.savefig("%s_plot.png"%(args.out_fname))

bte_solver.profile_stats()






