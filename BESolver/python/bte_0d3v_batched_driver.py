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
import scipy.cluster
from itertools import cycle
import cross_section

plt.rcParams.update({
    "text.usetex": False,
    "font.size": 24,
    #"ytick.major.size": 3,
    #"font.family": "Helvetica",
    "lines.linewidth":1.0
})


parser = argparse.ArgumentParser()
parser.add_argument("-threads", "--threads"                       , help="number of cpu threads", type=int, default=4)
parser.add_argument("-out_fname", "--out_fname"                   , help="output file name for the qois", type=str, default="bte")
parser.add_argument("-solver_type", "--solver_type"               , help="solver type", type=str, default="steady-state")
parser.add_argument("-l_max", "--l_max"                           , help="max polar modes in SH expansion", type=int, default=1)
parser.add_argument("-c", "--collisions"                          , help="collisions model", type=str, default="lxcat_data/eAr_crs.nominal.Biagi.3sp")
parser.add_argument("-sp_order", "--sp_order"                     , help="b-spline order", type=int, default=3)
parser.add_argument("-spline_qpts", "--spline_qpts"               , help="q points per knots", type=int, default=5)
parser.add_argument("-atol", "--atol"                             , help="absolute tolerance", type=float, default=1e-10)
parser.add_argument("-rtol", "--rtol"                             , help="relative tolerance", type=float, default=1e-10)
parser.add_argument("-max_iter", "--max_iter"                     , help="max number of iterations for newton solve", type=int, default=30)
parser.add_argument("-Te", "--Te"                                 , help="approximate electron temperature (eV)" , type=float, default=1.0)
parser.add_argument("-n0"    , "--n0"                             , help="heavy density (1/m^3)" , type=float, default=3.22e22)
parser.add_argument("-ev_max", "--ev_max"                         , help="max energy in the v-space grid" , type=float, default=30)
parser.add_argument("-Nr", "--Nr"                                 , help="radial refinement", type=int, default=128)
parser.add_argument("-profile", "--profile"                       , help="profile", type=int, default=0)
parser.add_argument("-warm_up", "--warm_up"                       , help="warm up", type=int, default=5)
parser.add_argument("-runs", "--runs"                             , help="runs "  , type=int, default=10)
parser.add_argument("-n_pts", "--n_pts"                           , help="number of points for batched solver", type=int, default=10)
parser.add_argument("-store_eedf", "--store_eedf"                 , help="store EEDF"          , type=int, default=0)
parser.add_argument("-store_csv", "--store_csv"                   , help="store csv format of QoI comparisons", type=int, default=1)
parser.add_argument("-plot_data", "--plot_data"                   , help="plot data", type=int, default=1)
parser.add_argument("-ee_collisions", "--ee_collisions"           , help="enable electron-electron collisions", type=int, default=0)
parser.add_argument("-verbose", "--verbose"                       , help="verbose with debug information", type=int, default=0)
parser.add_argument("-use_gpu", "--use_gpu"                       , help="use gpus for batched solver", type=int, default=1)
parser.add_argument("-cycles", "--cycles"                         , help="number of max cycles to evolve to compute cycle average rates", type=float, default=100)
parser.add_argument("-dt"    , "--dt"                             , help="1/dt number of denotes the number of steps for cycle", type=float, default=1e-3)
parser.add_argument("-Efreq" , "--Efreq"                          , help="electric field frequency Hz", type=float, default=13.56e6)
parser.add_argument("-input", "--input"                           , help="tps data file", type=str,  default="")

args                  = parser.parse_args()
read_input_from_file  = 1

if (args.input==""):
    read_input_from_file = 0

Td_fac                = 1e-21
c_gamma               = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))
all_species           = cross_section.read_available_species(args.collisions)

if (read_input_from_file):
    
    n_grids       = 1
    grid_idx      = 0
    file_idx      = 3
    ev_to_K       = collisions.TEMP_K_1EV
    fprefix       = args.input
    
    Te            = np.load("%s_Tg_%02d.npy"            %(fprefix, file_idx))
    Tg            = np.load("%s_Tg_%02d.npy"            %(fprefix, file_idx))
    ns_by_n0      = np.load("%s_ns_by_n0_%02d.npy"      %(fprefix, file_idx))
    n0            = np.load("%s_n0_%02d.npy"            %(fprefix, file_idx))
    ne            = np.load("%s_ne_%02d.npy"            %(fprefix, file_idx))
    ni            = np.load("%s_ni_%02d.npy"            %(fprefix, file_idx))
    E             = np.load("%s_E_%02d.npy"             %(fprefix, file_idx))
    EbyN          = E/n0/Td_fac
    Te_mean       = np.mean(Te/ev_to_K)
    vth           = np.sqrt(Te_mean) * c_gamma
    
    args.Te       = Te_mean
    args.n_pts    = len(Te)
    args.ev_max   = (6 * vth / c_gamma)**2
    
    ef            = EbyN * n0 * Td_fac

else:
    n_grids     = 1
    n_pts       = args.n_pts
    grid_idx    = 0

    n0          = np.linspace(1     , 1, n_pts) * 3.22e22
    ef          = np.linspace(1.8   , 2, n_pts)
    ef          = ef * n0 * Td_fac
    
    if(len(all_species)==1):
        ns_by_n0    = np.ones(n_pts).reshape((1, n_pts))
    else:
        ns_by_n0    = np.linspace(0.999   , 0.999, n_pts)
        ns_by_n0    = np.concatenate([ns_by_n0] + [(1-ns_by_n0)/(len(all_species)-1) for i in range(1, len(all_species))] ,  axis = 0).reshape(len(all_species), n_pts)
        print(ns_by_n0[:, -1])

    ne          = np.linspace(1     , 1, n_pts) * 3.22e20
    ni          = np.linspace(1     , 1, n_pts) * 3.22e20
    Tg          = np.linspace(1     , 1, n_pts) * 6000 #0.5 * collisions.TEMP_K_1EV
    
Te          = np.ones(n_grids) * args.Te 
ev_max      = np.ones(n_grids) * args.ev_max     
lm_modes    = [[[l,0] for l in range(args.l_max+1)] for i in range(n_grids)]
n_pts       = args.n_pts
nr          = np.ones(n_grids, dtype=np.int32) * args.Nr

bte_solver  = bte_0d3v_batched(args,ev_max, Te, nr, lm_modes, n_grids, args.collisions)

bte_solver.assemble_operators(grid_idx)
f0         = bte_solver.initialize(grid_idx, n_pts,"maxwellian")

bte_solver.set_boltzmann_parameter(grid_idx, "n0"       , n0)
bte_solver.set_boltzmann_parameter(grid_idx, "ne"       , ne)
bte_solver.set_boltzmann_parameter(grid_idx, "ns_by_n0" , ns_by_n0)
bte_solver.set_boltzmann_parameter(grid_idx, "Tg"       , Tg)
bte_solver.set_boltzmann_parameter(grid_idx, "eRe"      , 0*ef)
bte_solver.set_boltzmann_parameter(grid_idx, "eIm"      , ef)
bte_solver.set_boltzmann_parameter(grid_idx, "f0"       , f0)
bte_solver.set_boltzmann_parameter(grid_idx,  "E"       , ef)

if args.use_gpu==1:
    dev_id   = 0
    bte_solver.host_to_device_setup(dev_id, grid_idx)
    
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

f0       = bte_solver.get_boltzmann_parameter(grid_idx,"f0")
ff , qoi = bte_solver.solve(grid_idx, f0, args.atol, args.rtol, args.max_iter, args.solver_type)
ev       = np.linspace(1e-3, bte_solver._par_ev_range[grid_idx][1], 500)
ff_r     = bte_solver.compute_radial_components(grid_idx, ev, ff)

if args.use_gpu==1:
    bte_solver.device_to_host_setup(dev_id, grid_idx)

ff_r     = cp.asnumpy(ff_r)
for k, v in qoi.items():
    qoi[k] = cp.asnumpy(v)

collision_names = bte_solver.get_collision_names()
csv_write = args.store_csv
if csv_write:
    fname = args.out_fname
    with open("%s_qoi.csv"%fname, 'w', encoding='UTF8') as f:
        writer = csv.writer(f,delimiter=',')
        # write the header
        header = ["n0", "ne", "ni", "Tg", "E",  "energy", "mobility", "diffusion"]
        for col_idx, g in enumerate(collision_names):
            header.append(str(g))
            
        writer.writerow(header)
        
        data = np.concatenate((n0.reshape(-1,1), ne.reshape(-1,1), ni.reshape(-1,1), Tg.reshape(-1,1), ef.reshape(-1,1), qoi["energy"].reshape(-1,1), qoi["mobility"].reshape(-1,1), qoi["diffusion"].reshape(-1,1)), axis=1)
        for col_idx, g in enumerate(collision_names):
            data = np.concatenate((data, qoi["rates"][col_idx].reshape(-1,1)), axis=1)
        
        writer.writerows(data)


plot_data    = args.plot_data
if plot_data:
    num_sh       = len(lm_modes[0])
    num_subplots = num_sh 
    num_plt_cols = min(num_sh, 4)
    num_plt_rows = np.int64(np.ceil(num_subplots/num_plt_cols))
    fig        = plt.figure(figsize=(num_plt_cols * 8 + 0.5*(num_plt_cols-1), num_plt_rows * 8 + 0.5*(num_plt_rows-1)), dpi=300, constrained_layout=True)
    plt_idx    =  1
    n_pts_step =  max(1, n_pts // 20)

    for lm_idx, lm in enumerate(lm_modes[0]):
        plt.subplot(num_plt_rows, num_plt_cols, plt_idx)
        for ii in range(0, n_pts, n_pts_step):
            fr = np.abs(ff_r[ii, lm_idx, :])
            plt.semilogy(ev, fr, label=r"$T_g$=%.2E [K], $E/n_0$=%.2E [Td], $n_e/n_0$ = %.2E "%(Tg[ii], ef[ii]/n0[ii]/1e-21, ne[ii]/n0[ii]))
        
        plt.xlabel(r"energy (eV)")
        plt.ylabel(r"$f_%d$"%(lm[0]))
        plt.grid(visible=True)
        if lm_idx==0:
            plt.legend(prop={'size': 6})
            
        plt_idx +=1
    
    #plt_idx = num_sh
    plt.savefig("%s_plot.png"%(args.out_fname))

bte_solver.profile_stats(fname="%s_profile.csv"%(args.out_fname))


