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
import pandas as pd

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
parser.add_argument("-E", "--E_field"                             , help="Electric field in V/m", type=float, default=100)
parser.add_argument("-efield_period", "--efield_period"           , help="Oscillation period in seconds", type=float, default=0.0)
parser.add_argument("-num_tsamples", "--num_tsamples"             , help="number of samples to the time to collect QoIs", type=int, default=500)
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

interpolation_kind  = 'linear'

run_params          = [(e_values[i], ion_deg_values[j]) for i in range(len(e_values)) for j in range(len(ion_deg_values))]
for run_id in range(len(run_params)):
    args.E_field = run_params[run_id][0]
    args.ion_deg = run_params[run_id][1]

    if args.ion_deg == 0:
        args.ee_collisions = 0
        args.use_dg        = 1
        #df                 = pd.read_csv(r'pde_vs_bolsig/g0g2Nr128/pde_vs_bolsig_04_21_2023_18:06:18.csv')
        #r_data             = df[df['Nr']==128.0][df['ion_deg']==args.ion_deg]
        
        df                 = pd.read_csv(r'pde_vs_bolsig/g0g2_two_term/pde_vs_bolsig_09_14_2022_collop_approx.csv')
        r_data             = df[df['Nr']==128.0]

    else:
        args.ee_collisions = 1
        args.use_dg        = 0
        df                 = pd.read_csv(r'pde_vs_bolsig/g0g2eeNr64/pde_vs_bolsig_03_31_2023_11:44:46.csv')
        r_data             = df[df['Nr']==64.0][df['ion_deg']==args.ion_deg]
    
    
    
    x         = np.array(r_data["bolsig_energy"])
    y         = np.array(r_data["bolsig_mobility"])
    b_mu      = scipy.interpolate.interp1d(x,y,kind = interpolation_kind, bounds_error=False, fill_value=0.0)

    y         = np.array(r_data["bolsig_g0"])
    b_g0      = scipy.interpolate.interp1d(x,y,kind = interpolation_kind, bounds_error=False, fill_value=0.0)

    y         = np.array(r_data["bolsig_g2"])
    b_g2      = scipy.interpolate.interp1d(x,y,kind = interpolation_kind, bounds_error=False, fill_value=0.0)


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

    ts_qoi_all_ss = dict()
    ts_ss_all_ss  = dict()
    
    ss_list    = list()
    spec_list  = list()
    
    COLLISOIN_NAMES = bte_solver._collision_names

    pb_mode_begin   = 2 

    E_field = args.E_field
    ion_deg = args.ion_deg

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


        h_init     = np.copy(ss_init[1])

        m0 = np.dot(mass_op, h_init)
        t0 = np.dot(temp_op, h_init)/m0

        assert args.efield_period > 0
        dt          = args.T_DT/(1<<i)
        ts_sol1     = bte_solver.transient_solver_time_harmonic_efield(args.T_END, dt , num_time_samples=args.num_tsamples, h_init = h_init)
        #ts_sol2      = bte_solver.time_harmonic_efield_with_series_ss_solves(args.efield_period, args.efield_period/128, num_time_samples = 128)

        
        ts_qoi_all[i]    = bte_solver.compute_QoIs(ts_sol1["sol"], ts_sol1["tgrid"], effective_mobility=False)
        
        ts_ss_all[i]         = ts_sol1
        ss_sol               = dict()
        ss_sol["energy"]     = ts_qoi_all[i]["energy"]
        ss_sol["diffusion"]  = np.zeros_like(ts_qoi_all[i]["energy"])

        c_gamma              = np.sqrt(2*collisions.ELECTRON_CHARGE_MASS_RATIO)
        ss_sol["mobility"]   = b_mu(ss_sol["energy"]) * E_field * np.cos(np.pi * 2 * ts_sol1["tgrid"] / args.efield_period)
        ss_sol["rates"]      = [b_g0(ss_sol["energy"]), b_g2(ss_sol["energy"])]
        ts_qoi_all_ss[i]     = ss_sol
        #ts_ss_all_ss [i] = ts_sol2

        
        args.E_field = E_field
        args.ion_deg = ion_deg

        if args.store_csv == 1 :
            # fname    = "%s_nr%d_lmax%d_E%.2E_id_%.2E_Tg%.2E_dt%.2E.csv"%(args.out_fname, spec_sp._p, spec_sp._sph_harm_lm[-1][0], args.E_field, args.ion_deg, args.Tg, dt)
            # sol_data = np.zeros((len(ts_sol1["tgrid"]),  2 * (len(args.collisions) + 3) - 1))
            
            # sol_data[:,0] = ts_sol1["tgrid"]
            # sol_data[:,1] = ts_qoi_all[i]["energy"]
            # sol_data[:,2] = ts_qoi_all[i]["mobility"]
            # rr            = ts_qoi_all[i]["rates"]

            # for col_idx, col in enumerate(args.collisions):
            #     sol_data[:,3 + col_idx] = rr[col_idx]
            
            # sol_data[:, 5] = ts_qoi_all_ss[i]["energy"]
            # sol_data[:, 6] = ts_qoi_all_ss[i]["mobility"]
            # rr             = ts_qoi_all_ss[i]["rates"]

            # for col_idx, col in enumerate(args.collisions):
            #     sol_data[:,7 + col_idx] = rr[col_idx]
            
            # header_str = "time\tenergy\tmobility\t"
            # for col_idx, col in enumerate(args.collisions):
            #     header_str+=str(col)+"\t"

            # header_str += "energy_inp\tmobility_inp\t"
            # for col_idx, col in enumerate(args.collisions):
            #     header_str+=str(col)+"_inp\t"
            
            # np.savetxt(fname, sol_data, delimiter='\t',header=header_str,comments='')

            normL2 = lambda f1, f2, x : np.sqrt(np.trapz((f1 - f2)**2, x) / np.trapz(f2**2 , x))
            fname = "%s_lmax%d_Tg%.2E_Ep%.2E.csv"%(args.out_fname, spec_sp._sph_harm_lm[-1][0], args.Tg, args.efield_period)
            with open("%s_qois.csv"%fname, 'a', encoding='UTF8') as f:
                writer = csv.writer(f,delimiter='\t')
                if run_id == 0 and i==0:
                    header = ["E/N(Td)", "E(V/m)", "ion_deg", "dt", "T", "Nr", "energy_L2", "mobility_L2"]
                    for col_idx, col in enumerate(args.collisions):
                        header.append(str(col)+"_L2")
                
                    writer.writerow(header)

                mu_L2 = normL2(np.abs(ts_qoi_all_ss[i]["energy"]   ) , np.abs(ts_qoi_all[i]["energy"]  ) , ts_sol1["tgrid"])
                M_L2  = normL2(np.abs(ts_qoi_all_ss[i]["mobility"] ) , np.abs(ts_qoi_all[i]["mobility"]) , ts_sol1["tgrid"])
                data = [args.E_field/collisions.AR_NEUTRAL_N/1e-21, args.E_field, args.ion_deg, dt, args.T_END, args.sweep_values[i], mu_L2, M_L2]

                rr_ss         = ts_qoi_all_ss[i]["rates"]
                rr            = ts_qoi_all[i]["rates"]

                for col_idx, col in enumerate(args.collisions):
                    rr_L2 = normL2(np.abs(rr_ss[col_idx]), np.abs(rr[col_idx]), ts_sol1["tgrid"])
                    data.append(rr_L2)

                writer.writerow(data)



    
    if (1):
        maxwellian   = bte_solver._mw
        vth          = bte_solver._vth
        num_subplots = 4 * len(args.sweep_values) #int(np.ceil(num_sh/4)) * 4 + len(args.sweep_values) * 4
        num_plt_cols = 4
        num_plt_rows = np.int64(np.ceil(num_subplots/num_plt_cols))

        fig       = plt.figure(figsize=(num_plt_cols * 10 + 0.5*(num_plt_cols-1), num_plt_rows * 6 + 0.5*(num_plt_rows-1)), dpi=300, constrained_layout=True)

        # #f0
        # plt.subplot(num_plt_rows, num_plt_cols,  1)
        # plt.semilogy(bolsig_ev,  abs(bolsig_f0), '-k', label="bolsig")
        # # f1
        # plt.subplot(num_plt_rows, num_plt_cols,  2)
        # plt.semilogy(bolsig_ev,  abs(bolsig_f1), '-k', label="bolsig")

    
        # for i, value in enumerate(args.sweep_values):
        #     spec_sp   = spec_list[i]
        #     data      = ss_list[i]

        #     lbl       = args.sweep_param+"="+str(value)
        #     radial    = bte_utils.compute_radial_components(ev, spec_sp, data[-1],maxwellian,vth)
            
        #     # spherical components plots
        #     plt_idx=1
        #     for l_idx in range(num_sh):
        #         plt.subplot(num_plt_rows, num_plt_cols, plt_idx)
        #         color = next(plt.gca()._get_lines.prop_cycler)['color']
        #         plt.semilogy(ev,  abs(radial[l_idx]), '-', label=lbl, color=color)

        #         plt.xlabel(r"energy (eV)")
        #         plt.ylabel(r"radial component")
        #         plt.title("f%d"%(l_idx))
        #         plt.grid(visible=True)
        #         if l_idx == 0:
        #             plt.legend(prop={'size': 8})
                
        #         plt_idx+=1
        # plt_idx = max(num_sh, 4) + 1
        plt_idx = 1
        
        for i, value in enumerate(args.sweep_values):
            color = "red"#next(plt.gca()._get_lines.prop_cycler)['color']
            
            qois1    = ts_qoi_all[i]
            ts_sol1  = ts_ss_all [i]

            qois2    = ts_qoi_all_ss[i]
            #ts_sol2  = ts_ss_all_ss [i]

            tgrid1    = ts_sol1["tgrid"]
            tgrid2    = ts_sol1["tgrid"]#ts_sol2["tgrid"]

            dt       = args.T_DT / (1<<(1 * i ))

            plt.subplot(num_plt_rows, num_plt_cols, plt_idx)
            plt.semilogy(tgrid1,  qois1["energy"], '-', label ="transient"   , color=color)
            plt.semilogy(tgrid2,  qois2["energy"], '--*b', label="steady-state", markersize=1)
            plt.ylabel(r"energy (ev)")
            plt.xlabel(r"time (s)")
            plt.title("Nr = %d dt =%.4E"%(value, dt))
            plt.grid(visible=True)
            plt.legend()

            plt.subplot(num_plt_rows, num_plt_cols, plt_idx + 1)
            plt.semilogy(tgrid1, np.abs(qois1["mobility"]), '-', label ="transient"   , color=color)
            plt.semilogy(tgrid2, np.abs(qois2["mobility"]), '--*b', label="steady-state",markersize=1)
            plt.ylabel(r"mobility ($N (1/m/V/s $))")
            plt.xlabel(r"time (s)")
            plt.title("Nr = %d dt =%.4E"%(value, dt))
            plt.grid(visible=True)
            plt.legend()

            for col_idx, col in enumerate(args.collisions):
                plt.subplot(num_plt_rows, num_plt_cols, plt_idx + 2 + col_idx)
                plt.semilogy(tgrid1,  qois1["rates"][col_idx], '-', label ="transient"   , color=color)
                plt.semilogy(tgrid2,  qois2["rates"][col_idx], '--*b', label="steady-state", markersize=1)
                #plt.semilogy(tgrid1,  np.abs(qois1["rates"][col_idx]/qois2["rates"][col_idx]-1), '-', color=color)

                plt.title(COLLISOIN_NAMES[col])
                plt.ylabel(r"reaction rate ($m^3s^{-1}$)")
                #plt.ylabel(r"relative error")
                plt.xlabel(r"time (s)")
                plt.title("Nr = %d dt =%.4E"%(value, dt))
                plt.grid(visible=True)
                plt.legend()
        
            plt_idx+= 2 + len(args.collisions)
        
        fig.suptitle("E=%.4EV/m  E/N=%.4ETd ne/N=%.2E gas temp.=%.2EK, N=%.4E $m^{-3}$"%(args.E_field, args.E_field/collisions.AR_NEUTRAL_N/1e-21, args.ion_deg, args.Tg, collisions.AR_NEUTRAL_N))

        plt.savefig("rs_" + "_".join(args.collisions) + "_E%.2E"%(args.E_field/collisions.AR_NEUTRAL_N/1e-21) + "_sp_"+ str(args.sp_order) + "_nr" + str(args.NUM_P_RADIAL)+"_qpn_" + str(args.spline_qpts) + "_sweeping_" + args.sweep_param + "_lmax_" + str(args.l_max) +"_ion_deg_%.2E"%(args.ion_deg) + "_Tg%.2E"%(args.Tg)+ "Ep%.2E"%(args.efield_period) +".svg")

        plt.close()



        



