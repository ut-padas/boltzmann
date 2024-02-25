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
parser.add_argument("-Tg", "--Tg"                                 , help="gas temperature (K)" , type=float, default=0.0)
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

#
# examnple use : 
# python3 bte_0d3v_batched_driver.py -threads 64 -solver_type steady-state -l_max 1 -sp_order 3 -spline_qpts 10 -n_pts 18 -Efreq 0.0 -dt 1e-4 -plot_data 1 -Nr 63 --ee_collisions 0 -ev_max 15.8 -Te 0.5 -use_gpu 1 -cycles 10 -out_fname batched_bte/bte_0d_batched_ss_r2 -verbose 1 -max_iter 10000 -atol 1e-15 -rtol 1e-12 -c g0 g2

args        = parser.parse_args()
n_grids     = 1
n_pts       = args.n_pts
grid_idx    = 0

Te          = np.ones(n_grids) * args.Te 
ev_max      = np.ones(n_grids) * args.ev_max
ef          = np.linspace(1e-10 , 1, n_pts) * 3.22e1
n0          = np.linspace(1     , 1, n_pts) * 3.22e22

all_species = cross_section.read_available_species(args.collisions)

if(len(all_species)==1):
    ns_by_n0    = np.ones(n_pts).reshape((1, n_pts))
else:
    ns_by_n0    = np.linspace(0.6   , 1, n_pts)
    ns_by_n0    = np.concatenate([ns_by_n0] + [(1-ns_by_n0)/(len(all_species)-1) for i in range(1, len(all_species))] ,  axis = 0).reshape(len(all_species), n_pts)


ne          = np.linspace(1     , 1, n_pts) * 3.22e20
ni          = np.linspace(1     , 1, n_pts) * 3.22e20
Tg          = np.linspace(1     , 1, n_pts) * 6000 #0.5 * collisions.TEMP_K_1EV

clustering_experiment = 0
if (clustering_experiment == 1): 
    ev_to_K       = (scipy.constants.electron_volt/scipy.constants.Boltzmann) 
    Td_fac        = 1e-21 #[Vm^2]
    c_gamma       = np.sqrt(2 * scipy.constants.elementary_charge / scipy.constants.electron_mass) #[(C/kg)^{1/2}]

    Te            = np.load("../../../tps-inputs/axisymmetric/argon/lowP/single-rxn/Te.npy")/ev_to_K
    Tg            = np.load("../../../tps-inputs/axisymmetric/argon/lowP/single-rxn/Tg.npy")
    n0            = np.load("../../../tps-inputs/axisymmetric/argon/lowP/single-rxn/n0.npy")
    ne            = np.load("../../../tps-inputs/axisymmetric/argon/lowP/single-rxn/ne.npy")
    ni            = np.load("../../../tps-inputs/axisymmetric/argon/lowP/single-rxn/ni.npy")
    E             = np.load("../../../tps-inputs/axisymmetric/argon/lowP/single-rxn/E.npy")
    EbyN          = E/n0/Td_fac

    n_grids       = 4
    Tew           = scipy.cluster.vq.whiten(Te)
    # Tecw          = scipy.cluster.vq.kmeans(Tew, np.linspace(np.min(Tew), np.max(Tew), n_grids), iter=1000, thresh=1e-8)[0] #scipy.cluster.vq.kmeans2(Tew, n_grids, iter=1000, thresh=1e-8, minit='points')[0]
    # Tec           = Tecw * np.std(Te, axis=0)
    Tecw          = np.load("batched_bte/Tew_clusters.npy")
    Tec           = np.load("batched_bte/Te_clusters.npy")
    assert len(Tec)==n_grids

    dist_mat      = np.array([np.linalg.norm((Tew - Tecw[i]).reshape((-1, 1)), axis=1) for i in range(n_grids)]).T
    membership_Te = np.argmin(dist_mat, axis=1)
    gidx_to_pidx  = [np.argwhere(membership_Te==i)[:,0] for i in range(n_grids)]

    cluster_idx    = np.argmax(Tec)
    n_sub_clusters = 200
    t1_cluster    = time()
    m             = np.concatenate((EbyN.reshape((-1, 1)), Tg.reshape((-1, 1)), ne.reshape((-1, 1))), axis=1)
    #m            = np.concatenate((E.reshape((-1, 1)), Tg.reshape((-1, 1)), n0.reshape((-1, 1))), axis=1)
    m             = m[gidx_to_pidx[cluster_idx]]
    mw            = scipy.cluster.vq.whiten(m)
    mcw           = scipy.cluster.vq.kmeans2(mw, n_sub_clusters, iter=1000, thresh=1e-8, minit='points')[0] 
    mc            = mcw * np.std(m, axis=0)
    dist_mat      = np.array([np.linalg.norm(mw - mcw[i], axis=1) for i in range(n_sub_clusters)]).T
    membership_m  = np.argmin(dist_mat, axis=1)
    t2_cluster    = time()
    print("clustering time: %.8E s"%(t2_cluster-t1_cluster))
    # print("membership ", (membership_m == np.array(range(n_sub_clusters))).all())
    # print("membership, \n", np.linalg.norm(np.array(membership_m) - np.eye(n_sub_clusters))/np.linalg.norm(np.eye(n_sub_clusters)))

    plt.figure(figsize=(16, 16), dpi=300)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors    = cycle(prop_cycle.by_key()['color'])
    for i in range(n_sub_clusters):
        idx = np.argwhere(membership_m==i)[:,0]
        c   = next(colors)
        plt.scatter(m[idx, 1], m[idx, 0], s=10, marker='.', color=c)
        plt.scatter(mc[i,1], mc[i,0], s=100, marker='*', color=c)
        

    plt.tight_layout()      
    plt.xlabel(r"$T_g$ [K]")
    plt.ylabel(r"$E/n_0$ [Td]")
    plt.grid()
    plt.savefig("batched_bte/bte_clustering_sc%d.png"%(n_sub_clusters))
    plt.close()


    # n_grids     = 1
    # n_pts       = len(gidx_to_pidx[cluster_idx])
    # Te          = np.ones(n_grids) * Tec[cluster_idx] 
    # ev_max      = np.ones(n_grids) * (c_gamma * np.sqrt(Tec[cluster_idx]) * 6 / c_gamma )**2
    # n0          = n0[gidx_to_pidx[cluster_idx]]
    # ne          = ne[gidx_to_pidx[cluster_idx]]
    # ni          = ni[gidx_to_pidx[cluster_idx]] 
    # Tg          = Tg[gidx_to_pidx[cluster_idx]]
    # ef          = E [gidx_to_pidx[cluster_idx]]
    # grid_idx    = 0
    # np.save("batched_bte/Te_clusters.npy", Tec)
    # np.save("batched_bte/Tew_clusters.npy", Tecw)
    # np.save("batched_bte/mc_clusters.npy", mc)

    n_grids     = 1
    n_pts       = n_sub_clusters 
    Te          = np.ones(n_grids) * Tec[cluster_idx] 
    ev_max      = np.ones(n_grids) * (c_gamma * np.sqrt(Tec[cluster_idx]) * 6 / c_gamma )**2
    n0          = 3.22e22 * np.ones(n_pts)
    ne          = mc[:, 2]
    ni          = mc[:, 2] 
    Tg          = mc[:, 1]
    ef          = mc[:, 0] * n0 * Td_fac
    grid_idx    = 0

lm_modes    = [[[l,0] for l in range(args.l_max+1)] for i in range(n_grids)]
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
    n_pts_step =  n_pts // 20

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

bte_solver.profile_stats()


if (clustering_experiment==1):
    torch_full_qois = np.genfromtxt("batched_bte/torch_core_ss_qoi.csv", delimiter=",", skip_header=True)
    Te            = np.load("../../../tps-inputs/axisymmetric/argon/lowP/single-rxn/Te.npy")/ev_to_K
    Tg            = np.load("../../../tps-inputs/axisymmetric/argon/lowP/single-rxn/Tg.npy")
    n0            = np.load("../../../tps-inputs/axisymmetric/argon/lowP/single-rxn/n0.npy")
    ne            = np.load("../../../tps-inputs/axisymmetric/argon/lowP/single-rxn/ne.npy")
    ni            = np.load("../../../tps-inputs/axisymmetric/argon/lowP/single-rxn/ni.npy")
    E             = np.load("../../../tps-inputs/axisymmetric/argon/lowP/single-rxn/E.npy")

    idx           = np.argwhere(membership_Te==cluster_idx)[:,0]
    print(torch_full_qois[:,0].shape)
    print(n0[idx].shape)
    print(torch_full_qois[:,0] - n0[idx])
    assert (torch_full_qois[:,0]==n0[idx]).all() == True
    assert (torch_full_qois[:,1]==ne[idx]).all() == True
    assert (torch_full_qois[:,2]==ni[idx]).all() == True
    assert (torch_full_qois[:,3]==Tg[idx]).all() == True
    assert (torch_full_qois[:,4]==E[idx]).all()  == True

    ki_n0_ne     = torch_full_qois[:, 9] * torch_full_qois[:, 0] * torch_full_qois[:, 1]
    ki_n0_ne_max = np.max(ki_n0_ne)
    n0_ne        = torch_full_qois[:, 0] * torch_full_qois[:, 1]

    plt.figure(figsize=(20, 24), dpi=300)
    for c_idx in range(n_sub_clusters):
        pts_idx = np.argwhere(membership_m==c_idx)[:, 0]
        if len(pts_idx)==0:
            print("cluster = %d is empty "%(c_idx))
            continue
        
        mu_e     = torch_full_qois[pts_idx, 6]
        De       = torch_full_qois[pts_idx, 7]
        g0       = torch_full_qois[pts_idx, 8]
        g2       = torch_full_qois[pts_idx, 9]
        
        
        
        mu_e_inp = qoi["mobility"][c_idx]
        De_inp   = qoi["diffusion"][c_idx]
        g0_inp   = qoi["rates"][0][c_idx]
        g2_inp   = qoi["rates"][1][c_idx]
        
        #print(c_idx, g2, g2_inp, np.abs(1-(g2/g2_inp)), pts_idx)
        
        c        = next(colors)
        x_idx    = pts_idx #np.array(range(len(pts_idx)))
        plt.subplot(4, 1, 1)
        #plt.semilogy(x_idx, np.abs(1-g2_inp/g2), ".", color=c, label="cluster=%d"%(c_idx))
        
        plt.semilogy(x_idx, np.abs((ki_n0_ne[pts_idx] - g2_inp * n0_ne[pts_idx])/ki_n0_ne_max), ".", color=c, label="cluster=%d"%(c_idx))
        
        plt.subplot(4, 1, 2)
        plt.semilogy(x_idx, np.abs(1-g0_inp/g0), ".", color=c, label="cluster=%d"%(c_idx))
        
        plt.subplot(4, 1, 3)
        plt.semilogy(x_idx, np.abs(1-De_inp/De), ".", color=c, label="cluster=%d"%(c_idx))
        
        plt.subplot(4, 1, 4)
        plt.semilogy(x_idx, np.abs(1-mu_e_inp/mu_e), ".", color=c, label="cluster=%d"%(c_idx))
        
    plt.subplot(4, 1, 1)
    plt.xlabel(r"spatial point index")
    plt.ylabel(r"relative error")
    plt.grid(visible=True)
    plt.title(r"ionization")

    plt.subplot(4, 1, 2)
    plt.xlabel(r"spatial point index")
    plt.ylabel(r"relative error")
    plt.title(r"elastic")
    plt.grid(visible=True)

    plt.subplot(4, 1, 3)
    plt.xlabel(r"spatial point index")
    plt.ylabel(r"relative error")
    plt.title(r"diffusion")
    plt.grid(visible=True)

    plt.subplot(4, 1, 4)
    plt.xlabel(r"spatial point index")
    plt.ylabel(r"relative error")
    plt.title(r"mobility")
    plt.grid(visible=True)

    plt.tight_layout()
    plt.savefig("batched_bte/rel_errors_sc%d.png"%(n_sub_clusters))
    plt.close()