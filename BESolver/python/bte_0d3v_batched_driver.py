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
from mpi4py import MPI

path = os.path.abspath(os.path.dirname(sys.argv[0]))
print("path = ", path)
sys.path.append(path)
sys.path.append("/scratch/10565/ashwathsv/tps-venv/vista/tps/torch-chemistry/")
import argon.scripts.synthetic_cs as synthetic_cs

plt.rcParams.update({
    "text.usetex": False,
    "font.size": 24,
    #"ytick.major.size": 3,
    #"font.family": "Helvetica",
    "lines.linewidth":1.0
})

cs_datbase = "/scratch/10565/ashwathsv/tps-venv/vista/tps/tps-inputs/axisymmetric/argon/highP/cs_data"
nist_file = "/scratch/10565/ashwathsv/tps-venv/vista/tps/torch-chemistry/argon/input-data/ArI-levels-nist.csv"
g0 = 1 # Ar ground state degeneracy
g1 = 4 # Ar ion degeneracy

delE_ex = 11.512684347610431 # Excitation threshold energy in eV
delE_ig = 15.650008568830833 # Ionization threshold energy from ground state
delE_il = delE_ig - delE_ex
enforce_detailed_balance = 0
print("enforce_detailed_balance = ", enforce_detailed_balance)

avail_recomb_cs = 1

parser = argparse.ArgumentParser()
parser.add_argument("-threads", "--threads"                       , help="number of cpu threads", type=int, default=4)
parser.add_argument("-out_fname", "--out_fname"                   , help="output file name for the qois", type=str, default="bte")
parser.add_argument("-solver_type", "--solver_type"               , help="solver type", type=str, default="steady-state")
parser.add_argument("-l_max", "--l_max"                           , help="max polar modes in SH expansion", type=int, default=1)
parser.add_argument("-c", "--collisions"                          , help="collisions model", type=str, default="lxcat_data/fully_lumped_argon_mechanism_cs_recomb.lxcat")
parser.add_argument("-sp_order", "--sp_order"                     , help="b-spline order", type=int, default=3)
parser.add_argument("-spline_qpts", "--spline_qpts"               , help="q points per knots", type=int, default=11)
parser.add_argument("-atol", "--atol"                             , help="absolute tolerance", type=float, default=1e-4)
parser.add_argument("-rtol", "--rtol"                             , help="relative tolerance", type=float, default=1e-8)
parser.add_argument("-max_iter", "--max_iter"                     , help="max number of iterations for newton solve", type=int, default=50)
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
parser.add_argument("-input", "--input"                           , help="tps csv file", type=str,  default="")
parser.add_argument("-cs_datbase", "--cs_database"                , help="path to directory of cross-section database", type=str,  default="/scratch/10565/ashwathsv/tps-venv/vista/tps/tps-inputs/axisymmetric/argon/highP/cs_data")
#python3 bte_0d3v_batched_driver.py --threads 1 -out_fname bte_ss -solver_type steady-state -c lxcat_data/eAr_crs.synthetic.3sp2r -sp_order 3 -spline_qpts 5 -atol 1e-10 -rtol 1e-10 -max_iter 300 -Te 3 -n0 3.22e22 -ev_max 30 -Nr 127 -n_pts 1 -ee_collisions 1 -cycles 2 -dt 1e-3
args                  = parser.parse_args()
read_input_from_file  = 1

N_Avo                 = scipy.constants.Avogadro
m_e                   = scipy.constants.electron_mass
k_B                   = scipy.constants.Boltzmann
h_pl                  = scipy.constants.Planck
ev_to_K               = collisions.TEMP_K_1EV

varyT_cs = 1

if (args.input==""):
    read_input_from_file = 0

Td_fac                = 1e-21
c_gamma               = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))
all_species           = cross_section.read_available_species(args.collisions)

comm = MPI.COMM_WORLD
rank_ = comm.Get_rank()
size_ = comm.Get_size()

if (read_input_from_file):
    n_grids       = 1
    grid_idx      = 0
    ev_to_K       = collisions.TEMP_K_1EV

    import pandas as pd
    try:
        dat = pd.read_csv(args.input)
        Tg            = np.array(dat["Tg"])
        n0            = np.array(dat["n0"])
        n_pts         = len(Tg)
        ne            = np.array(dat["ne/n0"]) * n0
        eRe           = np.array(dat["eRe"])
        eIm           = np.array(dat["eIm"])
        E             = np.array(dat["E"])
        ni            = ne
        ns_by_n0      = np.zeros((len(all_species), len(Tg)))
        
        for i in range(len(all_species)):
            ns_by_n0[i] = np.array(dat["(%s)/n0"%(all_species[i])])
            # if all_species[i] == "Ar*":
            #     ns_by_n0[i] = 0.0
            #     ns_by_n0[i] = ni / n0

        # enforce mixture mass is normalized
        ns_by_n0 = ns_by_n0/np.sum(ns_by_n0, axis=0)

        for i in range(len(all_species)):
            #### We should have more robust recombination detection process. 
            if all_species[i] == 'Ar+':
                print("Two body recombination detected")
                ns_by_n0[i,:] = (ne/n0)**2 * n0
        
    except Exception as e:
        print("Error reading file %s"%(args.input))
        print(e)

    EbyN          = E/n0/Td_fac
    Te_mean       = max(1e-1, np.mean(Tg/ev_to_K)) #[ev] #np.mean(Te/ev_to_K)
    vth           = np.sqrt(Te_mean) * c_gamma

    # print("ns_by_n0=\n", ns_by_n0, "\nEbyN=\n", EbyN, "\nN=\n", n0, "\nne=\n", ne, "\nTg=\n", Tg)
    print("all_species = ", all_species, len(all_species))


    args.Te       = Te_mean
    args.n_pts    = len(Tg)
    args.ev_max   = 40 * np.mean(Tg/ev_to_K) #(6 * vth / c_gamma)**2

    Tg_max = np.amax(Tg)
    nen0max = np.amax(ne/n0)
    
    evmax_thermal = 50 * Tg_max / ev_to_K
    if Tg_max < 3000:
        evmax_thermal = 50 * Tg_max / ev_to_K

    if Tg_max > 6000:
        evmax_thermal = 50 * Tg_max / ev_to_K 
            
    if args.ee_collisions==1:
        evmax_thermal = 80 * Tg_max / ev_to_K
    
    efmax = np.amax(EbyN)
    
    maxwellian_bypass = 0
    if efmax <= 0.1:
        print(f"Low E/N WARNING: efmax = {efmax}, is low. Bad coefficients possible.", flush=True)
        maxwellian_bypass = 0
        print(f"Due to low E/N, we assume a Maxwellian EEDF for this grid to avoid errors in computed EEDF.")
    evmax_efield = 6.0
    if efmax > 0.1 and efmax <= 0.5:
        evmax_efield = 16.0
        if Tg_max < 3000:
            evmax_efield = 12.0
    elif efmax > 0.5 and efmax <= 1.0:
        evmax_efield = 18.0
    elif efmax > 1.0 and efmax <= 2.0:
        evmax_efield = 20.0
    elif efmax > 2.0 and efmax <= 5.0:
        evmax_efield = 30.0
    elif efmax > 5.0 and efmax <= 10.0:
        evmax_efield = 50.0
    elif efmax > 10.0:
        evmax_efield = 100.0
    
    # if maxwellian_bypass == 1:
    #     args.ee_collisions = 0

    args.ev_max = max(evmax_thermal, evmax_efield)
    
    ef            = EbyN * n0 * Td_fac

else:
    n_grids     = 1
    n_pts       = args.n_pts
    grid_idx    = 0

    n0          = np.linspace(1     , 1, n_pts) * 3.22e22
    ef          = np.linspace(1.0e-2 , 0.1, n_pts)
    ef          = ef * n0 * Td_fac
    
    if(len(all_species)==1):
        ns_by_n0    = np.ones(n_pts).reshape((1, n_pts))
    else:
        ns_by_n0    = np.linspace(0.999998   , 1, n_pts)
        ns_by_n0    = np.concatenate([ns_by_n0] + [(1-ns_by_n0)/(len(all_species)-1) for i in range(1, len(all_species))] ,  axis = 0).reshape(len(all_species), n_pts)
        print(ns_by_n0[:, -1])
        print(ns_by_n0)

    ne          = np.linspace(0.1     , 1, n_pts) * 3.22e15
    ni          = np.linspace(0.1     , 1, n_pts) * 3.22e15
    Tg          = np.linspace(0.8     , 1.2, n_pts) * 400 #0.5 * collisions.TEMP_K_1EV
    
Te          = np.ones(n_grids) * args.Te 
ev_max      = np.ones(n_grids) * args.ev_max     
lm_modes    = [[[l,0] for l in range(args.l_max+1)] for i in range(n_grids)]
n_pts       = args.n_pts
nr          = np.ones(n_grids, dtype=np.int32) * args.Nr

# args.collisions contains reactions which will be modeled by BTE
# allow_cols() is a list of strings that tells which collision data 
# goes into the CS file
cs_data_all   = cross_section.read_cross_section_data(args.collisions)
coll_type_list = list()
allow_cols     = list()
for col_str, col_data in cs_data_all.items():
    coll_type_list.append(col_data["type"])
    allow_cols.append(col_str)

# generate crs Tg depended crs data 
crs_folder = "lxcat_data/argon/crs_lxcat"
if not os.path.exists(crs_folder):
    os.makedirs(crs_folder)
col_cs = list()
if varyT_cs == 1:
    Tlfile = cs_datbase + "/Tlump.txt"
    Tlarr = np.loadtxt(Tlfile, delimiter=",")

    # glfile = cs_datbase + "/glump.txt"
    # glarr = np.loadtxt(glfile, delimiter="\t")

for idx in range(n_grids):
    cs_fname = args.collisions
    if varyT_cs == 1:
        Temp = Tlarr[np.argmin(np.abs((Te[idx]*ev_to_K) - Tlarr))]

        # Lumped state degeneracy is determined at Temp
        # glump = glarr[np.argmin(np.abs(glarr[:,0] - Temp)), 1]

        cs_input = "/crs_Tlump%06d.lxcat" %(Temp)
        cs_input = cs_datbase + cs_input
        cs_fname = "%s/crs_rank_%06d_npes_%06d_%06d.txt"%(crs_folder, rank_, size_, idx)
        synthetic_cs.read_and_write_lumpedcs(cs_fname, cs_input, ev_max[idx], allow_cols, rank_, idx)

        print("idx = ", idx, ", Temp = ", Temp, ", glump = ", glump)
    col_cs.append(cs_fname)

# Find out for which collisions we need to get the reverse rate coefficient from detailed balance
need_reverse_kreac = np.zeros(len(allow_cols))
for i in range(len(allow_cols)):
    check_reverse_kreac = 0
    if allow_cols[i] == "E + Ar -> E + Ar*":
        reverse_key = "E + Ar* -> E + Ar"
        check_reverse_kreac = 1
    elif allow_cols[i] == "E + Ar -> E + E + Ar+":
        reverse_key = "E + E + Ar+ -> E + Ar"
        check_reverse_kreac = 1
    elif allow_cols[i] == "E + Ar* -> E + E + Ar+":
        reverse_key = "E + E + Ar+ -> E + Ar*"
        check_reverse_kreac = 1
    
    if check_reverse_kreac == 1:
        if reverse_key not in allow_cols:
            need_reverse_kreac[i] = 1

total_cols = len(allow_cols) + np.sum(need_reverse_kreac)

# print("allow_cols = ", allow_cols)
print("degeneracy of lumped state = ", glump)

is_recomb = np.zeros(len(coll_type_list))
for idx in range(len(is_recomb)):
    if coll_type_list[idx] == "ATTACHMENT":
        is_recomb[idx] = 1
        
# print("is_recomb = ", is_recomb)

bte_solver  = bte_0d3v_batched(args,ev_max, Te, nr, lm_modes, n_grids, col_cs)

bte_solver.assemble_operators(grid_idx)
f0         = bte_solver.initialize(grid_idx, n_pts,"maxwellian")

bte_solver.set_boltzmann_parameter(grid_idx, "n0"       , n0)
bte_solver.set_boltzmann_parameter(grid_idx, "ne"       , ne)
bte_solver.set_boltzmann_parameter(grid_idx, "ns_by_n0" , ns_by_n0)
bte_solver.set_boltzmann_parameter(grid_idx, "Tg"       , Tg)
bte_solver.set_boltzmann_parameter(grid_idx, "eRe"      , eRe)
bte_solver.set_boltzmann_parameter(grid_idx, "eIm"      , eIm)
bte_solver.set_boltzmann_parameter(grid_idx, "f0"       , f0)
bte_solver.set_boltzmann_parameter(grid_idx,  "E"       , ef)

if args.use_gpu==1 and maxwellian_bypass == 0:
    dev_id   = 0
    bte_solver.host_to_device_setup(dev_id, grid_idx)
    
if args.profile==1 and maxwellian_bypass == 0:
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
ls_c     = 1e-4 #min(0.9, np.exp(-(np.min(np.abs(ef))/2)**2))
f0       = bte_solver.get_boltzmann_parameter(grid_idx,"f0")

if args.ee_collisions == 1 and maxwellian_bypass == 0:
    # disabling ee-collisions temporary by force to get a good initial guess. 
    args.ee_collisions = 0  
    ff , qoi = bte_solver.solve(grid_idx, f0, args.atol, args.rtol, args.max_iter, args.solver_type,
                                alpha_min = 1e-16, alpha_rho=0.5, line_search_c = ls_c)
    bte_solver.set_boltzmann_parameter(grid_idx, "f0"       , ff)

    # enable ee-collisions
    args.ee_collisions = 1

if maxwellian_bypass == 0:  
    f0       = bte_solver.get_boltzmann_parameter(grid_idx,"f0")
    ff , qoi = bte_solver.solve(grid_idx, f0, args.atol, args.rtol, args.max_iter, args.solver_type,
                            alpha_min = 1e-16, alpha_rho=0.5, line_search_c = ls_c)
    ev       = np.linspace(1e-3, bte_solver._par_ev_range[grid_idx][1], 500)
    ff_r     = bte_solver.compute_radial_components(grid_idx, ev, ff)
else:
    ff       = bte_solver.get_boltzmann_parameter(grid_idx,"f0")
    ev       = np.linspace(1e-3, bte_solver._par_ev_range[grid_idx][1], 500)
    ff_r     = bte_solver.compute_radial_components(grid_idx, ev, ff)
    
    bte_solver.set_boltzmann_parameter(grid_idx, "u_avg", ff)
    qA        = bte_solver._op_diag_dg[grid_idx]
    u0        = bte_solver.get_boltzmann_parameter(grid_idx, "u_avg")
    h_curr    = bte_solver.normalized_distribution(grid_idx, u0)
    qoi       = bte_solver.compute_QoIs(grid_idx, h_curr, effective_mobility=False)

if args.use_gpu==1 and maxwellian_bypass == 0:
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
        for col_idx, g in enumerate(collision_names[0]):
            print("col_idx = ", col_idx, ", len(collision_names) = ", len(collision_names[0]), ", g = ", g)
            header.append(str(g))
            if col_idx == len(collision_names[0])-1:
                # reached end of collision_names
                # Headers for reverse reactions computed from detailed balance
                ii = len(collision_names[0])
                for i in range(len(allow_cols)):
                    if need_reverse_kreac[i] == 1:
                        headstr = "C" + str(ii)
                        header.append(headstr)
                        ii += 1
            
        writer.writerow(header)
        
        data = np.concatenate((n0.reshape(-1,1), ne.reshape(-1,1), ni.reshape(-1,1), Tg.reshape(-1,1), ef.reshape(-1,1), qoi["energy"].reshape(-1,1), qoi["mobility"].reshape(-1,1), qoi["diffusion"].reshape(-1,1)), axis=1)
        for col_idx, g in enumerate(collision_names[0]):
            ratearr = qoi["rates"][col_idx].reshape(-1,1) # numpy array

            if enforce_detailed_balance == 1:
                # Get the mean electron temperature in the v-space grid
                Te_arr = (2.0/3.0) * qoi["energy"].reshape(-1,1)
                prefac = ((2 * np.pi * m_e * k_B * Te_arr * ev_to_K) / (h_pl**2))**1.5
                if allow_cols[col_idx] == "E + Ar -> E + E + Ar+":
                    # Ground ionization (compare electron temperature to ionization threshold)
                    ig_thresh = 0.05 * delE_ig
                    lo_cutoff = 0.9 * ig_thresh
                    hi_cutoff = 1.1 * ig_thresh
                    Keq = (2 * g1 / g0) * prefac * np.exp(-delE_ig / Te_arr)

                    # First, confirm that the cross-section file has ground recombination
                    irec = -1
                    ii = 0
                    for ii in range(len(allow_cols)):
                        if allow_cols[ii] == "E + E + Ar+ -> E + Ar":
                            irec = ii
                    
                    if irec == -1:
                        raise RuntimeError("FATAL Runtime error: Collisions file does not have ground recombination")

                    if np.amax(Te_arr) <= hi_cutoff:
                        print(f"Ground ionization, max(Te) = {np.amax(Te_arr)} <= hi_cutoff ({hi_cutoff} eV)")
                        krec = qoi["rates"][irec].reshape(-1,1) # recombination rate as computed by BTE

                        if np.amax(Te_arr) <= lo_cutoff:
                            # Ground ionization coefficient determined by detailed balance using recombination
                            # Since Te << deltaE_ig, electron energy is very low. 
                            # BTE computed ground ionization coefficient may be corrupted by numerical noise
                            print(f"Ground ionization, max(Te) = {np.amax(Te_arr)} < lo_cutoff ({lo_cutoff}), ground ionization obtained from recombination")
                            ratearr = krec * Keq.reshape(-1,1)
                        else:
                            print(f"Ground ionization, {lo_cutoff} < max(Te) = {np.amax(Te_arr)} < {hi_cutoff}, coefficient obtained by blending BTE and detailed balance")
                            # Blending the BTE obtained rate coefficient with detailed balance rate coefficient
                            # Transition Zone: Smooth blending with Quintic Smoothstep
                            kBTE = ratearr # Rate coefficient for forward ionization obtained from BTE
                            x = (Te_arr - lo_cutoff) / (hi_cutoff - lo_cutoff)
                            w = 6.0 * (x**5) - 15.0 * (x**4) + 10.0 * (x**3)

                            krec_high = kBTE / Keq # Recombination coefficient obtained based on detailed balance from forward reaction

                            # Blend the recombination rates and derive ionization rate via inverted detailed balance (use Keq and krec to determine kion)
                            k_rec_fluid = (1.0 - w) * krec + w * krec_high
                            k_ion_fluid = k_rec_fluid * Keq

                            ratearr = k_ion_fluid
                            # UPDATE THE RECOMBINATION RATE COEFFICIENT 
                            print(f"Updating recombination to ground rate coefficient after blending (irec = {irec}).")
                            qoi["rates"][irec] = k_rec_fluid.reshape(-1)
                    else:
                        print(f"Ground ionization, max(Te) = {np.amax(Te_arr)} > hi_cutoff ({hi_cutoff}), recombination determined from ground ionization.")
                        # The ionization rate coefficient stays as is
                        # Determine recombination rate coefficient from detailed balance based on forward reaction
                        krec = ratearr / Keq
                        # Update the recomination rate
                        print(f"Updating recombination to ground rate coefficient based on detailed balance (irec = {irec}).")
                        qoi["rates"][irec] = krec.reshape(-1)
                # ------------------ Ground ionization ends here ---------------------------------
                elif allow_cols[col_idx] == "E + Ar* -> E + E + Ar+":
                    # Lumped ionization (compare electron temperature to ionization threshold)
                    il_thresh = 0.05 * delE_il
                    lo_cutoff = 0.9 * il_thresh
                    hi_cutoff = 1.1 * il_thresh
                    Keq = (2 * glump / g0) * prefac * np.exp(-delE_il / Te_arr)

                    # First, confirm that the cross-section file has lumped recombination
                    irec = -1
                    ii = 0
                    for ii in range(len(allow_cols)):
                        if allow_cols[ii] == "E + E + Ar+ -> E + Ar*":
                            irec = ii
                    
                    if irec == -1:
                        raise RuntimeError("FATAL Runtime error: Collisions file does not have lumped recombination")

                    if np.amax(Te_arr) <= hi_cutoff:
                        print(f"Lumped ionization, max(Te) = {np.amax(Te_arr)} <= hi_cutoff ({hi_cutoff}) eV")
                        # Lumped Ionization rate coefficient obtained from recombination via detailed balance
                        krec = qoi["rates"][irec].reshape(-1,1) # recombination rate as computed by BTE
                    
                        if np.amax(Te_arr) <= lo_cutoff:
                            # Lumped ionization coefficient determined by detailed balance using recombination
                            print(f"Lumped ionization, max(Te) = {np.amax(Te_arr)} < lo_cutoff ({lo_cutoff}), lumped ionization obtained from recombination")
                            ratearr = krec * Keq.reshape(-1,1)
                        else:
                            print(f"Lumped ionization, {lo_cutoff} < max(Te) = {np.amax(Te_arr)} < {hi_cutoff}, coefficient obtained by blending BTE and detailed balance")
                            # Blending the BTE obtained rate coefficient with detailed balance rate coefficient
                            # Transition Zone: Smooth blending with Quintic Smoothstep
                            kBTE = ratearr # Rate coefficient for forward ionization obtained from BTE
                            x = (Te_arr - lo_cutoff) / (hi_cutoff - lo_cutoff)
                            w = 6.0 * (x**5) - 15.0 * (x**4) + 10.0 * (x**3)

                            krec_high = kBTE / Keq # Recombination coefficient obtained based on detailed balance from forward reaction

                            # Blend the recombination rates and derive ionization rate via inverted detailed balance
                            k_rec_fluid = (1.0 - w) * krec + w * krec_high
                            k_ion_fluid = k_rec_fluid * Keq

                            ratearr = k_ion_fluid
                            # UPDATE THE RECOMBINATION RATE COEFFICIENT 
                            print(f"Updating recombination to lumped rate coefficient after blending (irec = {irec}).")
                            print("Shape(qoi) = ", qoi["rates"][irec].shape, ", shape(k_rec_fluid) = ", k_rec_fluid.shape)
                            qoi["rates"][irec] = k_rec_fluid.reshape(-1)
                    else:
                        print(f"Lumped ionization, max(Te) = {np.amax(Te_arr)} > hi_cutoff ({hi_cutoff}), recombination determined from lumped ionization.")
                        # The ionization rate coefficient stays as is
                        # Determine recombination rate coefficient from detailed balance based on forward reaction
                        krec = ratearr / Keq
                        # Update the recomination rate
                        print(f"Updating recombination to ground rate coefficient based on detailed balance (irec = {irec}).")
                        qoi["rates"][irec] = krec.reshape(-1)

            data = np.concatenate((data, ratearr), axis=1)

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
bte_solver.profile_reset()


