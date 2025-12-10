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
import os

print("imported all modules")

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
parser.add_argument("-c", "--collisions"                          , help="collisions model", type=str, default="lxcat_data/fully_lumped_argon_mechanism_cs.lxcat")
parser.add_argument("-sp_order", "--sp_order"                     , help="b-spline order", type=int, default=3)
parser.add_argument("-spline_qpts", "--spline_qpts"               , help="q points per knots", type=int, default=5)
parser.add_argument("-atol", "--atol"                             , help="absolute tolerance", type=float, default=1e-10)
parser.add_argument("-rtol", "--rtol"                             , help="relative tolerance", type=float, default=1e-8)
parser.add_argument("-max_iter", "--max_iter"                     , help="max number of iterations for newton solve", type=int, default=1000)
parser.add_argument("-Te", "--Te"                                 , help="approximate electron temperature (eV)" , type=float, default=0.5)
parser.add_argument("-n0"    , "--n0"                             , help="heavy density (1/m^3)" , type=float, default=3.22e22)
parser.add_argument("-ev_max", "--ev_max"                         , help="max energy in the v-space grid" , type=float, default=30)
parser.add_argument("-Nr", "--Nr"                                 , help="radial refinement", type=int, default=127)
parser.add_argument("-profile", "--profile"                       , help="profile", type=int, default=0)
parser.add_argument("-warm_up", "--warm_up"                       , help="warm up", type=int, default=5)
parser.add_argument("-runs", "--runs"                             , help="runs "  , type=int, default=10)
parser.add_argument("-n_pts", "--n_pts"                           , help="number of points for batched solver", type=int, default=10)
parser.add_argument("-store_eedf", "--store_eedf"                 , help="store EEDF"          , type=int, default=1)
parser.add_argument("-store_csv", "--store_csv"                   , help="store csv format of QoI comparisons", type=int, default=1)
parser.add_argument("-plot_data", "--plot_data"                   , help="plot data", type=int, default=0)
parser.add_argument("-ee_collisions", "--ee_collisions"           , help="enable electron-electron collisions", type=int, default=0)
parser.add_argument("-verbose", "--verbose"                       , help="verbose with debug information", type=int, default=0)
parser.add_argument("-use_gpu", "--use_gpu"                       , help="use gpus for batched solver", type=int, default=1)
parser.add_argument("-cycles", "--cycles"                         , help="number of max cycles to evolve to compute cycle average rates", type=float, default=5)
parser.add_argument("-dt"    , "--dt"                             , help="1/dt number of denotes the number of steps for cycle", type=float, default=5e-3)
parser.add_argument("-Efreq" , "--Efreq"                          , help="electric field frequency Hz", type=float, default=6e6)
parser.add_argument("-input", "--input"                           , help="tps data file", type=str,  default="")
#python3 bte_0d3v_batched_driver.py --threads 1 -out_fname bte_ss -solver_type steady-state -c lxcat_data/eAr_crs.synthetic.3sp2r -sp_order 3 -spline_qpts 5 -atol 1e-10 -rtol 1e-10 -max_iter 300 -Te 3 -n0 3.22e22 -ev_max 30 -Nr 127 -n_pts 1 -ee_collisions 1 -cycles 2 -dt 1e-3
args                  = parser.parse_args()
read_input_from_file  = 1

# python3 bte_solver_pointwise.py --threads 1 -out_fname bte_ss -solver_type steady-state -c lxcat_data/fully_lumped_argon_mechanism_cs.lxcat -sp_order 3 -spline_qpts 5 -atol 1e-10 -rtol 1e-8 -max_iter 1000 -Te 0.5 -ev_max 30 -Nr 127 -ee_collisions 0 -cycles 2 -dt 5e-3
if (args.input==""):
    read_input_from_file = 0

ev_to_K               = collisions.TEMP_K_1EV
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
    # Tarr        = np.array([10254.3, 4919.96, 3046.67])
    Tarr        = np.array([3046.67]) # in Kelvin
    n_pts       = len(Tarr)

    # nAr         = np.array([6.84e23, 1.5e24, 2.41e24])
    # nAr_s       = np.array([3.68e19, 5.15e17, 3.64e17])
    # ne          = np.array([1.58e22, 1.26e20, 7.92e18])
    # ni          = np.array([1.58e22, 1.26e20, 7.92e18])

    nAr         = np.array([2.41e24]) #in m^-3
    nAr_s       = np.array([3.64e17]) #in m^-3
    ne          = np.array([7.92e18]) #in m^-3
    ni          = np.array([7.92e18]) #in m^-3

    n0          = nAr + nAr_s + ni - ne

    # Er          = np.array([134.888, 198.817, 1e-20])
    # Ei          = np.array([-12.61, 369.285, 1e-20])

    Er          = np.array([1e-20]) # in V/m
    Ei          = np.array([1e-20])  # in V/m

    Te          = Tarr 
    Tg          = Tarr

ion_deg = (ne / n0)

ns_by_n0 = np.zeros((len(all_species), n_pts))

print("len(all_species) = ", len(all_species), ", all_species = ", all_species)

for i in range(len(all_species)):
    print("i = ", i, ", all_species[i] = ", all_species[i])
    if all_species[i] == 'Ar':
        ns_by_n0[i,0:n_pts] = nAr / n0
    elif all_species[i] == "Ar*":
        ns_by_n0[i,0:n_pts] = nAr_s / n0
    elif all_species[i] == "Ar+":
        ns_by_n0[i,0:n_pts] = ni / n0

print("ns_by_n0 = ", ns_by_n0, ", ion_deg = ", ion_deg)

args.n_pts = n_pts

Emag = np.sqrt(Er**2 + Ei**2)
EbyN = Emag/n0/Td_fac

Te_mean       = np.mean(Te/ev_to_K)
args.Te       = Te_mean
vth           = np.sqrt(Te_mean) * c_gamma
args.ev_max   = (6 * vth / c_gamma)**2

ef            = EbyN * n0 * Td_fac

# Setup grids for BTE solve
n_grids       = 1
grid_idx      = 0

Te          = np.ones(n_grids) * args.Te 
ev_max      = np.ones(n_grids) * args.ev_max     
lm_modes    = [[[l,0] for l in range(args.l_max+1)] for i in range(n_grids)]
nr          = np.ones(n_grids, dtype=np.int32) * args.Nr

bte_solver  = bte_0d3v_batched(args,ev_max, Te, nr, lm_modes, n_grids, [args.collisions])

bte_solver.assemble_operators(grid_idx)

f0         = bte_solver.initialize(grid_idx, n_pts,"maxwellian")
bte_solver.set_boltzmann_parameter(grid_idx, "n0"       , n0)
bte_solver.set_boltzmann_parameter(grid_idx, "ne"       , ne)
bte_solver.set_boltzmann_parameter(grid_idx, "ns_by_n0" , ns_by_n0)
bte_solver.set_boltzmann_parameter(grid_idx, "Tg"       , Tg)   
bte_solver.set_boltzmann_parameter(grid_idx, "eRe"      , Er)
bte_solver.set_boltzmann_parameter(grid_idx, "eIm"      , Ei)
bte_solver.set_boltzmann_parameter(grid_idx, "f0"       , f0)
bte_solver.set_boltzmann_parameter(grid_idx,  "E"       , ef)

if args.use_gpu==1:
    dev_id   = 0
    bte_solver.host_to_device_setup(dev_id, grid_idx)

if args.profile==1:
    res_func, jac_func = bte_solver.get_rhs_and_jacobian(0, f0.shape[1])
    f0  = bte_solver.get_boltzmann_parameter(grid_idx, "f0")
    for i in range(args.warm_up):
        xp = bte_solver.xp_module
        a  = res_func(f0, 0, 0)
        b  = jac_func(f0, 0, 0)
        b  = bte_solver.batched_inv(grid_idx, b)
        b  = xp.einsum("ijk,ki->ji", b, a)
    
    bte_solver.profile_reset()
    for i in range(args.runs):
        xp = bte_solver.xp_module
        a  = res_func(f0, 0, 0)
        b  = jac_func(f0, 0, 0)
        b  = bte_solver.batched_inv(grid_idx, b)
        b  = xp.einsum("ijk,ki->ji", b, a)
    
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

# FIX THE RECOMBINATION RATES
# SINCE THE BTE ASSUMES A PSEUDO 2-BODY COLLISION MODEL,
# THE RECOMBINATION RATES COMING FROM BTE MUST BE MULTIPLIED BY ne
# TO ACCOUNT FOR THE 2ND ELECTRON IN THE COLLISION
# cs_data_all   = cross_section.read_cross_section_data(args.collisions)
# collision_count = 0
# for col_str, col_data in cs_data_all.items():
#     if col_data["type"] == "ATTACHMENT":
#         qoi["rates"][collision_count] = ne[0]*qoi["rates"][collision_count]
#     collision_count+=1

dirname = args.out_fname
os.makedirs(dirname, exist_ok = True)

collision_names = bte_solver.get_collision_names()
csv_write = args.store_csv
if csv_write:
    fname = "%s/%s_Tg%.2E_qoi.csv" %(dirname, args.out_fname, Tg[0])
    with open(fname, 'w', encoding='UTF8') as f:
        writer = csv.writer(f,delimiter=',')
        # write the header
        header = ["n0", "ne", "ni", "Tg", "E", "energy", "mobility", "diffusion"]
        for col_idx, g in enumerate(collision_names):
            header.append(str(g))
            
        writer.writerow(header)
        
        data = np.concatenate((n0.reshape(-1,1), ne.reshape(-1,1), ni.reshape(-1,1), Tg.reshape(-1,1), ef.reshape(-1,1), qoi["energy"].reshape(-1,1), qoi["mobility"].reshape(-1,1), qoi["diffusion"].reshape(-1,1)), axis=1)
        for col_idx in range(len(qoi["rates"])):
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

store_eedf = args.store_eedf
if store_eedf:
    num_sh       = len(lm_modes[0])
    num_subplots = num_sh 
    n_pts_step =  max(1, n_pts // 20)

    f0 = np.abs(ff_r[0, 0, :])
    f1 = np.abs(ff_r[0, 1, :])

    # GET THE MAXWELLIAN DISTRIBUTION
    # fmw = ( n0[0] / ( (np.pi**1.5) * (vth**3) ) ) * np.exp( -ev * c_gamma**2 / vth**2 )
    # fmw = fmw / np.trapezoid(fmw, ev)
    fmw = (2 / np.sqrt(np.pi)) * (Te_mean**(-1.5)) * np.exp(-ev / Te_mean)
    fmw_int = np.trapezoid(fmw*np.sqrt(ev), ev)
    outarr = np.column_stack((ev, f0, f1, fmw))

    filename = "%s/%s_EEDF_Tg%.2E.txt" %(dirname, args.out_fname, Tg[0])
    print("shape(f0) = ", f0.shape, ", shape(f1) = ", f1.shape, ", shape(ev) = ", ev.shape, ", fmw.shape = ", fmw.shape, ", outarr.shape = ", outarr.shape)
    print("filename = ", filename)

    # WRITE TO FILE
    with open(filename, 'w') as f:
        header = "# Energy (eV) \t f0 (BTE) \t f1 (BTE) \t f0 (Maxwellian) Nr = %d" %(args.Nr)
        f.write(header + '\n')
        # Then, use numpy.savetxt to write the array below the header
        np.savetxt(f, outarr, fmt='%.4E', delimiter='\t')  # '%d' specifies integer format

bte_solver.profile_stats(fname="%s_profile.csv"%(args.out_fname))
bte_solver.profile_reset()


