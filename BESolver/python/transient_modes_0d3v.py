"""
@brief Analyze the transient modes, for the 0d3v boltzmann equation. 
"""

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
from   datetime import datetime
import bte_0d3v_solver as bte_0d3v
import utils as bte_utils

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 24,
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
parser.add_argument("-n0", "--n0"                                 , help="heavy density (1/m^3)" , type=float, default=3.22e22)
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

    if args.ion_deg == 0:
        args.ee_collisions = 0
        args.use_dg        = 1
    else:
        args.ee_collisions = 1
        args.use_dg        = 0
    
    print(args)
    bte_solver = bte_0d3v.bte_0d3v(args)
    bte_solver.run_bolsig_solver()
    ev = bte_solver._bolsig_data["ev"]
    num_sh = args.l_max + 1

    bolsig_ev = bte_solver._bolsig_data["ev"]
    bolsig_f0 = bte_solver._bolsig_data["f0"]
    bolsig_f1 = bte_solver._bolsig_data["f1"]

    ev = bolsig_ev

    ts_qoi_all = dict()
    ts_ss_all  = dict()
    
    ss_list    = list()
    spec_list  = list()
    
    COLLISOIN_NAMES = bte_solver._collision_names

    pb_mode_begin   = 2 

    for i, value in enumerate(args.sweep_values):
        if args.sweep_param == "Nr":
            args.NUM_P_RADIAL = int(value)
        else:
            raise NotImplementedError
        
        bte_solver._args = args
        bte_solver.setup()

        ss_sol  = bte_solver.steady_state_solver()["sol"]
        spec_sp = bte_solver._spec_sp
        spec_list.append(spec_sp)
        ss_list.append(ss_sol)

        mw      = bte_solver._mw
        vth     = bte_solver._vth

        mass_op = bte_solver._mass_op * mw(0) * vth**3
        temp_op = bte_solver._temp_op_ev * mw(0) * vth**5

        num_p   = spec_sp._p + 1

        ss_init    = np.zeros_like(ss_sol)
        ss_init[0] = ss_sol[0] / np.dot(mass_op, ss_sol[0])
        ss_init[1] = ss_sol[1] / np.dot(mass_op, ss_sol[1])

        for l in range(pb_mode_begin, num_sh):
            h_init            = spec_sp.create_vec().reshape(-1)
            h_init[0::num_sh] = ss_init[-1][0::num_sh]
            h_init[l::num_sh] = (1- 1e-10) * ss_init[-1][0::num_sh]
            # for k in range(l, num_sh):
            #     h_init[k::num_sh] = (1- 1e-10) * ss_init[-1][0::num_sh]
            
            # h_init            = np.copy(ss_init[-1])
            # h_init[l::num_sh] = ss_init[0][0::num_sh] 

            # if args.store_csv == 1 and i==(len(args.sweep_values)-1):
            #     n_samples = int(1e6)
            #     fname    = "%s_nr%d_lmax%d_E%.2E_id_%.2E_Tg%.2E_perturb_l_%d.npy"%(args.out_fname, spec_sp._p, spec_sp._sph_harm_lm[-1][0], args.E_field, args.ion_deg, args.Tg, l)
            #     print("sampling for l-perturb %d size %.2E"%(l,n_samples))
            #     c_gamma      = bte_solver._c_gamma
            #     x_domain     = (np.sqrt(0) * c_gamma / vth , np.sqrt(bte_solver._args.ev_max) * c_gamma / vth)
            #     bte_utils.sample_distribution_with_uniform_prior(spec_sp, h_init, x_domain, vth, fname, n_samples)

            m0 = np.dot(mass_op, h_init)
            t0 = np.dot(temp_op, h_init)/m0

            dt = args.T_DT/(1<<i) 

            print("perturbing mode %d"%l)
            print("  mass = %.8E"%(m0))
            print("  temp = %.8E"%(t0))
            print("  dt   = %.8E"%(dt))

            if args.efield_period == 0:
                ts_sol = bte_solver.transient_solver(args.T_END, dt, num_time_samples=200, h_init=h_init)
            else:
                ts_sol = bte_solver.transient_solver_time_harmonic_efield(args.T_END, dt, h_init)
            ts_qoi_all[(i,l)] = bte_solver.compute_QoIs(ts_sol["sol"], ts_sol["tgrid"])

            ss_diff = np.linalg.norm(ss_sol[1]-ts_sol["sol"][-1])/np.linalg.norm(ss_sol[1])
            print("relative error with ss solution = %.4E"%(ss_diff))

            if args.store_csv == 1 :
                fname    = "%s_nr%d_lmax%d_E%.2E_id_%.2E_Tg%.2E_perturb_l_%d_dt%.2E.csv"%(args.out_fname, spec_sp._p, spec_sp._sph_harm_lm[-1][0], args.E_field, args.ion_deg, args.Tg, l, dt)
                sol_data = np.zeros((len(ts_sol["tgrid"]),  len(args.collisions) + 3 ))
                
                sol_data[:,0] = ts_sol["tgrid"]
                sol_data[:,1] = ts_qoi_all[(i,l)]["energy"]
                sol_data[:,2] = ts_qoi_all[(i,l)]["mobility"]
                rr            = ts_qoi_all[(i,l)]["rates"]

                for col_idx, col in enumerate(args.collisions):
                    sol_data[:,3 + col_idx] = rr[col_idx]
                
                header_str = "time\tenergy\tmobility\t"
                for col_idx, col in enumerate(args.collisions):
                    header_str+=str(col)+"\t"

                np.savetxt(fname, sol_data, delimiter='\t',header=header_str,comments='')


            # ts_ss     = np.zeros((2,len(h_init)))
            # ts_ss[0]  = ts_sol["sol"][0]
            # ts_ss[-1] = ts_sol["sol"][-1]
            ts_ss_all [(i,l)] = ts_sol
    

    if (1):
        maxwellian   = bte_solver._mw
        vth          = bte_solver._vth
        num_subplots = int(np.ceil(num_sh/4)) * 4 + len(args.sweep_values) * 4
        num_plt_cols = 4
        num_plt_rows = np.int64(np.ceil(num_subplots/num_plt_cols))
        
        fig       = plt.figure(figsize=(num_plt_cols * 8 + 0.5*(num_plt_cols-1), num_plt_rows * 8 + 0.5*(num_plt_rows-1)), dpi=300, constrained_layout=True)

        #f0
        plt.subplot(num_plt_rows, num_plt_cols,  1)
        plt.semilogy(bolsig_ev,  abs(bolsig_f0), '-k', label="bolsig")
        # f1
        plt.subplot(num_plt_rows, num_plt_cols,  2)
        plt.semilogy(bolsig_ev,  abs(bolsig_f1), '-k', label="bolsig")

    
        for i, value in enumerate(args.sweep_values):
            spec_sp   = spec_list[i]
            data      = ss_list[i]

            lbl       = args.sweep_param+"="+str(value)
            radial    = bte_utils.compute_radial_components(ev, spec_sp, data[-1],maxwellian,vth)
            
            # spherical components plots
            plt_idx=1
            for l_idx in range(num_sh):
                plt.subplot(num_plt_rows, num_plt_cols, plt_idx)
                color = next(plt.gca()._get_lines.prop_cycler)['color']
                plt.semilogy(ev,  abs(radial[l_idx]), '-', label=lbl, color=color)

                plt.xlabel(r"energy (eV)")
                plt.ylabel(r"radial component")
                plt.title("f%d"%(l_idx))
                plt.grid(visible=True)
                if l_idx == 0:
                    plt.legend(prop={'size': 24})

                # for pl_idx in range(pb_mode_begin,num_sh):
                #     ts_sol   = ts_ss_all[(i,pl_idx)]
                #     ts_data  = ts_sol["sol"]
                #     ts_radial   = bte_utils.compute_radial_components(ev, spec_sp, ts_data[-1],maxwellian,vth)
                #     plt.semilogy(ev,  abs(ts_radial[l_idx]), '--', label=lbl+" ts ", color=color)


                
                plt_idx+=1
        
        for i, value in enumerate(args.sweep_values):
            
                spec_sp  = spec_list[i]

                radial   = bte_utils.compute_radial_components(ev, spec_sp, data[-1],maxwellian,vth)



        plt_idx = int(np.ceil(num_sh/4)) * 4 +1
        color_list = list()
        for l_idx in range(0,num_sh):
            color = next(plt.gca()._get_lines.prop_cycler)['color']
            color_list.append(color)

        for i, value in enumerate(args.sweep_values):
            for l_idx in range(pb_mode_begin,num_sh):
                color  = color_list[l_idx] 
                qois   = ts_qoi_all[(i,l_idx)]
                ts_sol = ts_ss_all[(i,l_idx)]

                tgrid  = ts_sol["tgrid"]
                dt     = args.T_DT / (1<<(1 * i ))
                if(l_idx==0):
                    lbl =r"$f_%d$"%(l_idx)
                else:
                    lbl =r"$f_0 + f_%d$"%(l_idx)
                plt.subplot(num_plt_rows, num_plt_cols, plt_idx)
                plt.semilogy(tgrid,  qois["energy"], '-', label=lbl, color=color)
                plt.ylabel(r"energy (ev)")
                plt.xlabel(r"time (s)")
                plt.title("Nr = %d dt =%.4E"%(value, dt))
                plt.grid(visible=True)
                plt.legend()

                plt.subplot(num_plt_rows, num_plt_cols, plt_idx + 1)
                plt.semilogy(tgrid,  np.abs(qois["mobility"]), '-', label=lbl, color=color)
                plt.ylabel(r"mobility ($N (1/m/V/s $))")
                plt.xlabel(r"time (s)")
                plt.title("Nr = %d dt =%.4E"%(value, dt))
                plt.legend()
                plt.grid(visible=True)

                for col_idx, col in enumerate(args.collisions):
                    plt.subplot(num_plt_rows, num_plt_cols, plt_idx + 2 + col_idx)
                    plt.semilogy(tgrid, qois["rates"][col_idx], '-', label=lbl, color=color)

                    plt.title(COLLISOIN_NAMES[col])
                    plt.ylabel(r"reaction rate ($m^3s^{-1}$)")
                    plt.xlabel(r"time (s)")
                    #plt.title("Nr = %d dt =%.4E"%(value, dt))
                    plt.grid(visible=True)
            
            plt_idx+= 2 + len(args.collisions)
        
        fig.suptitle("E=%.4EV/m  E/N=%.4ETd ne/N=%.2E gas temp.=%.2EK, N=%.4E $m^{-3}$"%(args.E_field, args.E_field/collisions.AR_NEUTRAL_N/1e-21, args.ion_deg, args.Tg, collisions.AR_NEUTRAL_N))

        plt.savefig(args.out_fname+"_perturb_modes_use_dg"+str(args.use_dg)+"_" + "_".join(args.collisions) + "_E%.2E"%args.E_field + "_sp_"+ str(args.sp_order) + "_nr" + str(args.NUM_P_RADIAL)+"_qpn_" + str(args.spline_qpts) + "_sweeping_" + args.sweep_param + "_lmax_" + str(args.l_max) +"_ion_deg_%.2E"%(args.ion_deg) + "_Tg%.2E"%(args.Tg) +".svg")

        plt.close()



        



