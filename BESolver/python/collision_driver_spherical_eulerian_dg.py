"""
@package Boltzmann collision operator solver. 
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
parser.add_argument("-c", "--collisions"                          , help="collisions included (g0, g0Const, g0NoLoss, g2, g2Const)", type=str, default="lxcat_data/eAr_crs.nominal.Biagi.3sp")
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
parser.add_argument("-ns_by_n0", "--ns_by_n0"                     , help="ns/n0 ratio for each collision specified in the file in that order", type=float, nargs='+', default=[1, 1, 1, 1, 1, 1, 1, 1])
parser.add_argument("-store_eedf", "--store_eedf"                 , help="store EEDF"          , type=int, default=0)
parser.add_argument("-store_csv", "--store_csv"                   , help="store csv format of QoI comparisons", type=int, default=0)
parser.add_argument("-ee_collisions", "--ee_collisions"           , help="Enable electron-electron collisions", type=float, default=1)
parser.add_argument("-bolsig", "--bolsig_dir"                     , help="Bolsig directory", type=str, default="../../Bolsig/")
parser.add_argument("-bolsig_precision", "--bolsig_precision"     , help="precision value for bolsig code", type=float, default=1e-11)
parser.add_argument("-bolsig_convergence", "--bolsig_convergence" , help="convergence value for bolsig code", type=float, default=1e-8)
parser.add_argument("-bolsig_grid_pts", "--bolsig_grid_pts"       , help="grid points for bolsig code"      , type=int, default=1024)
#python3 collision_driver_spherical_eulerian_dg.py -l_max 1 -c g0 g2 -sp_order 3 -spline_qpts 5 -steady 1 --sweep_values 31 63 127  -EbyN 1e-1 1 5 -Tg 5000 -ion_deg 0
args                = parser.parse_args()
EbyN_Td             = np.logspace(np.log10(args.EbyN[0]), np.log10(args.EbyN[1]), int(args.EbyN[2]), base=10)
e_values            = EbyN_Td * args.n0 * 1e-21
ion_deg_values      = np.array(args.ion_deg)

SAVE_EEDF    = args.store_eedf
SAVE_CSV     = args.store_csv

if not args.ee_collisions:
    ion_deg_values = np.array([0])

run_params          = [(e_values[i], ion_deg_values[j]) for i in range(len(e_values)) for j in range(len(ion_deg_values))]
for run_id in range(len(run_params)):
    args.E_field = run_params[run_id][0]
    args.ion_deg = run_params[run_id][1]
    #print(args)

    if args.ion_deg == 0:
        args.ee_collisions = 0
        if args.Tg > 0: 
            args.use_dg        = 0
        else:
            args.use_dg        = 1
    else:
        args.ee_collisions = 1
        args.use_dg        = 0

    bte_solver = bte_0d3v.bte_0d3v(args)
    bte_solver.run_bolsig_solver()
    ev = bte_solver._bolsig_data["ev"]
    num_sh = args.l_max + 1

    radial_base       = np.zeros((len(args.sweep_values), num_sh, len(ev)))
    radial            = np.zeros((len(args.sweep_values), num_sh, len(ev)))
    
    mu          = list()
    M           = list()
    D           = list()
    rates       = list()
    spec_list   = list()
    qoi_list    = list()
    solver_data = list()
    
    collision_list  = bte_solver.get_collision_list()
    collision_names = bte_solver.get_collision_names()

    for i , _ in enumerate(collision_list):
        rates.append([])

    bolsig_ev = bte_solver._bolsig_data["ev"]
    bolsig_f0 = bte_solver._bolsig_data["f0"]
    bolsig_f1 = bte_solver._bolsig_data["f1"]

    bolsig_mu       = bte_solver._bolsig_data["energy"]
    bolsig_M        = bte_solver._bolsig_data["mobility"]
    bolsig_D        = bte_solver._bolsig_data["diffusion"]
    bolsig_rates    = bte_solver._bolsig_data["rates"]

    for i, value in enumerate(args.sweep_values):
        if args.sweep_param == "Nr":
            args.NUM_P_RADIAL = int(value)
        else:
            raise NotImplementedError
        
        bte_solver._args = args
        bte_solver.setup()

        if args.steady_state == 1:
            r_data    = bte_solver.steady_state_solver()
            #r_data   = bte_solver.steady_state_solver_two_term()
        else:
            r_data    = bte_solver.transient_solver(args.T_END, args.T_DT/(1<<(i)), num_time_samples=args.num_tsamples)
        
        solver_data.append(r_data)
        spec_list.append(bte_solver._spec_sp)

        data      = r_data['sol']
        h_bolsig  = r_data['h_bolsig']
        abs_tol   = r_data['atol']
        rel_tol   = r_data['rtol']

        spec_sp   = bte_solver._spec_sp 
        mw        = bte_solver._mw
        vth       = bte_solver._vth    

        radial_base[i,:,:]  = BEUtils.compute_radial_components(ev, spec_sp, data[0,:] , mw, vth, 1)
        radial[i, :, :]     = BEUtils.compute_radial_components(ev, spec_sp, data[-1,:], mw, vth, 1)

        qois = bte_solver.compute_QoIs(data, r_data["tgrid"])
        qoi_list.append(qois)
        
        mu.append(qois["energy"][-1])
        M.append(qois["mobility"][-1])
        D.append(qois["diffusion"][-1])
        rr = qois["rates"]

        for col_idx, g in enumerate(collision_list):
            rates[col_idx].append(rr[col_idx][-1])
        
        # tmp       = BEUtils.compute_radial_components(ev, spec_sp, h_bolsig,mw, vth, 1)
        # bolsig_f0 = tmp[0]
        # bolsig_f1 = tmp[1]

        if SAVE_EEDF:
            # with open("%s.npy"%args.out_fname, 'ab') as f:
            #     np.save(f, np.array([spec_sp._p + 1]))
            #     np.save(f, np.array([spec_sp._sph_harm_lm[-1][0]]))
            #     np.save(f, np.array([args.E_field]))
            #     np.save(f, np.array([args.ion_deg]))
            #     np.save(f, np.array([args.Tg]))
            #     np.save(f, ev)

            #     for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
            #         np.save(f, radial[-1,lm_idx,:])
                    
            #     np.save(f, bolsig_f0)
            #     np.save(f, bolsig_f1)

            fname    = "%s_nr%d_lmax%d_E%.2E_id_%.2E_Tg%.2E.csv"%(args.out_fname, spec_sp._p, spec_sp._sph_harm_lm[-1][0], args.E_field, args.ion_deg, args.Tg)
            sol_data = np.zeros((len(ev),  1 + 2 + args.l_max+1))
            sol_data[:,0] = ev 
            sol_data[:,1] = bolsig_f0
            sol_data[:,2] = bolsig_f1

            header_str = "ev\tbolsig_f0\tbolsig_f1\t"
            for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
                sol_data[:,3 + lm_idx] = radial[i,lm_idx,:]
                header_str+="f%d\t"%lm[0]

            np.savetxt(fname, sol_data, delimiter='\t',header=header_str,comments='')

        
    if SAVE_CSV:
        fname = "%s_lmax%d_Tg%.2E.csv"%(args.out_fname, spec_sp._sph_harm_lm[-1][0], args.Tg)
        with open("%s_qois.csv"%fname, 'a', encoding='UTF8') as f:
            writer = csv.writer(f,delimiter='\t')
            if run_id == 0:
                # write the header
                header = ["E/N(Td)", "E(V/m)", "Nr", "energy", "diffusion", "mobility", "bolsig_energy", "bolsig_defussion", "bolsig_mobility", "l2_f0", "l2_f1", "Tg", "ion_deg", "atol", "rtol"]
                for col_idx, g in collision_names:
                    header.append(str(g))
                    header.append("bolsig_"+str(g))

                header.append("rel_energy")
                header.append("rel_mobility")
                header.append("rel_diffusion")
                for col_idx, g in collision_names:
                    header.append("rel_"+str(g))
                
                header.append("rel_energy_vs_bolsig")
                header.append("rel_mobility_vs_bolsig")
                header.append("rel_diffusion_vs_bolsig")
                for col_idx, g in collision_names:
                    header.append("rel_"+str(g)+"_vs_bolsig")
                
                for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
                    header.append("l2_f%d%d"%(lm[0],lm[1]))

                writer.writerow(header)
            
            normL2 = lambda f1, f2, x : np.trapz(x * (f1 - f2)**2, x) / np.trapz(x * f2**2 , x)

            for i, value in enumerate(args.sweep_values):
                # write the data
                l2_f0     = normL2(np.abs(radial[i, 0]), np.abs(bolsig_f0), ev) #np.linalg.norm(np.abs(radial[i, 0])-np.abs(bolsig_f0))/np.linalg.norm(np.abs(bolsig_f0))
                l2_f1     = normL2(np.abs(radial[i, 1]), np.abs(bolsig_f1), ev) #np.linalg.norm(np.abs(radial[i, 1])-np.abs(bolsig_f1))/np.linalg.norm(np.abs(bolsig_f1))

                data = [args.E_field/args.n0/1e-21, args.E_field, args.sweep_values[i], mu[i], D[i], M[i], bolsig_mu, bolsig_D, bolsig_M, l2_f0, l2_f1, args.Tg, args.ion_deg, solver_data[i]["atol"], solver_data[i]["rtol"]]
                for col_idx , _ in enumerate(collision_list):
                    data.append(rates[col_idx][i])
                    data.append(bolsig_rates[col_idx])

                data.append(np.abs(mu[i]/ mu[-1]-1))
                data.append(np.abs(M[i] / M[-1] -1))
                data.append(np.abs(D[i] / D[-1] -1))
                for col_idx , _ in enumerate(collision_list):
                    data.append(np.abs(rates[col_idx][i]/rates[col_idx][-1]-1))

                data.append(np.abs(mu[i]/ bolsig_mu-1))
                data.append(np.abs(M[i] / bolsig_M -1))
                data.append(np.abs(D[i] / bolsig_D -1))
                for col_idx , _ in enumerate(collision_list):
                    data.append(np.abs(rates[col_idx][i]/bolsig_rates[col_idx]-1))

                for lm_idx, lm in enumerate(spec_sp._sph_harm_lm):
                    l2_error = normL2(np.abs(radial[i, lm[0]]), np.abs(radial[-1, lm[0]]), ev) #np.linalg.norm(np.abs(radial[i, lm_idx])-np.abs(radial[-1, lm_idx]))/np.linalg.norm(np.abs(radial[-1, lm_idx]))
                    data.append(l2_error)

                writer.writerow(data)
        

    if (1):
        maxwellian   = bte_solver._mw
        if args.steady_state == 0:
            num_subplots = num_sh + 2 + 2 + 2 + 4
        else:
            num_subplots = num_sh + 2 + 2 + 2
        num_plt_cols = 4
        num_plt_rows = np.int64(np.ceil(num_subplots/num_plt_cols))
        
        fig       = plt.figure(figsize=(num_plt_cols * 8 + 0.5*(num_plt_cols-1), num_plt_rows * 8 + 0.5*(num_plt_rows-1)), dpi=300, constrained_layout=True)
        
        #f0
        plt.subplot(num_plt_rows, num_plt_cols,  3)
        plt.semilogy(bolsig_ev,  abs(bolsig_f0), '-k', label="bolsig")
        # f1
        plt.subplot(num_plt_rows, num_plt_cols,  4)
        plt.semilogy(bolsig_ev,  abs(bolsig_f1), '-k', label="bolsig")

        plt.subplot(num_plt_rows, num_plt_cols, 1)
        plt.semilogy(args.sweep_values, abs(np.array(mu)/bolsig_mu-1), 'o-', label='mean energy')
        
        for col_idx, col in enumerate(collision_list):
            if bolsig_rates[col_idx] != 0:
                plt.semilogy(args.sweep_values, abs(rates[col_idx]/bolsig_rates[col_idx]-1), 'o-', label=collision_names[col_idx])
        
        plt.semilogy(args.sweep_values, abs(np.array(M)/bolsig_M-1), 'o-', label='mobility')
        plt.xlabel(args.sweep_param)
        plt.ylabel(r"relative error")
        plt.title("PDE vs. Bolsig")
        plt.legend()
        plt.grid(visible=True)

        plt.subplot(num_plt_rows, num_plt_cols, 2)
        plt.semilogy(args.sweep_values, abs(np.array(mu)/mu[-1]-1), 'o-', label='mean energy')
        
        for col_idx, col in enumerate(collision_list):
            if bolsig_rates[col_idx] != 0:
                plt.semilogy(args.sweep_values, abs(rates[col_idx]/rates[col_idx][-1]-1), 'o-', label=collision_names[col_idx])
        
        plt.semilogy(args.sweep_values, abs(np.array(M)/M[-1]-1), 'o-', label='mobility')
        plt.xlabel(args.sweep_param)
        plt.ylabel(r"relative error")
        #plt.title("PDE vs. PDE")
        plt.legend()
        plt.grid(visible=True)

        pde_vs_bolsig_L2 = list()
        for i, value in enumerate(args.sweep_values):
            qois      = qoi_list[i]
            spec_sp   = spec_list[i]
            r_data    = solver_data[i]
            data      = r_data["sol"]
            lbl = args.sweep_param+"="+str(value)
            if args.steady_state == 0:
                lbl += " dt =%.2E"%(args.T_DT/(1<<(2 * i)))

            if args.steady_state == 0:
                data_tt    = r_data['sol']
                tgrid      = r_data['tgrid']
                tgrid_step = int(len(tgrid)/5)

                # time_idx   = [0, 1, 2, 20, 30]
                # radial_tt = np.zeros((len(time_idx), num_sh, len(ev)))
                # for t_idx, tt in enumerate(time_idx):
                #     radial_tt[t_idx, :, : ] = BEUtils.compute_radial_components(ev, spec_sp, data_tt[tt,:], maxwellian, vth, 1)

            # spherical components plots
            plt_idx=3
            for l_idx in range(num_sh):
                plt.subplot(num_plt_rows, num_plt_cols, plt_idx)
                color = next(plt.gca()._get_lines.prop_cycler)['color']
                plt.semilogy(ev,  abs(radial[i, l_idx]), '-', label=lbl, color=color)

                # if args.steady_state == 0:
                #     for t_idx, tt in enumerate(time_idx):
                #         #color = next(plt.gca()._get_lines.prop_cycler)['color']
                #         tt_time = tgrid[tt]    
                #         plt.semilogy(ev,  abs(radial_tt[t_idx, l_idx]), '--', label=lbl+" t=%.2E"%(tt_time), color=color)

                
                plt.xlabel(r"energy (eV)")
                plt.ylabel(r"radial component")
                plt.title(r"$f_%d$"%(l_idx))
                plt.grid(visible=True)
                if l_idx == 0 or l_idx==2:
                    #plt.legend(prop={'size': 16})
                    plt.legend()
                
                plt_idx+=1

            plt.subplot(num_plt_rows, num_plt_cols, plt_idx)
            plt.semilogy(ev,  np.abs(np.abs(radial[i, 0]) - np.abs(bolsig_f0))/np.max(np.abs(bolsig_f0)), '-', label=lbl, color=color)
            plt.ylim((None, 1))
            plt.ylabel(r"relative error")
            plt.xlabel(r"evergy (eV)")
            plt.grid(visible=True)
            plt.title("f0 (PDE vs. Bolsig)")
            
            plt.subplot(num_plt_rows, num_plt_cols, plt_idx + 1)
            plt.semilogy(ev,  np.abs(np.abs(radial[i, 1]) - np.abs(bolsig_f1))/np.max(np.abs(bolsig_f1)), '-', label=lbl, color=color)
            #plt.semilogy(ev,  abs(abs(radial[i, 1])/abs(bolsig_f1)-1), '-', label=lbl, color=color)
            plt.ylim((None, 1))
            plt.ylabel(r"relative error")
            plt.xlabel(r"evergy (eV)")
            plt.grid(visible=True)
            plt.title("f1 (PDE vs. Bolsig)")

            plt.subplot(num_plt_rows, num_plt_cols, plt_idx + 2)
            plt.semilogy(ev,  np.abs(np.abs(radial[i, 0]) - np.abs(radial[-1, 0]))/np.max(np.abs(radial[-1, 0])), '-', label=lbl, color=color)
            plt.ylabel(r"relative error")
            plt.xlabel(r"evergy (eV)")
            plt.grid(visible=True)
            plt.title("f0 (PDE vs. PDE)")

            plt.subplot(num_plt_rows, num_plt_cols, plt_idx + 3)
            plt.semilogy(ev,  np.abs(np.abs(radial[i, 1]) - np.abs(radial[-1, 1]))/np.max(np.abs(radial[-1, 1])), '-', label=lbl, color=color)
            plt.ylabel(r"relative error")
            plt.xlabel(r"evergy (eV)")
            plt.grid(visible=True)
            plt.title("f1 (PDE vs. PDE)")

            if args.steady_state == 0:
                data_tt   = r_data['sol']
                tgrid     = list(r_data['tgrid'])

                plt.subplot(num_plt_rows, num_plt_cols, plt_idx + 4)
                plt.semilogy(tgrid,  qois["energy"], '-', label=lbl, color=color)
                plt.ylabel(r"energy (ev)")
                plt.xlabel(r"time (s)")
                plt.grid(visible=True)

                plt.subplot(num_plt_rows, num_plt_cols, plt_idx + 5)
                plt.semilogy(tgrid,  np.abs(qois["mobility"]), '-', label=lbl, color=color)
                plt.ylabel(r"mobility ($N (1/m/V/s $))")
                plt.xlabel(r"time (s)")
                plt.grid(visible=True)

                for col_idx, col in enumerate(collision_list):
                    plt.subplot(num_plt_rows, num_plt_cols, plt_idx + 6 + col_idx)
                    plt.semilogy(tgrid, qois["rates"][col_idx], '-', label=lbl, color=color)

                    plt.title(collision_names[col_idx])
                    plt.ylabel(r"reaction rate ($m^3s^{-1}$)")
                    plt.xlabel(r"time (s)")
                    plt.grid(visible=True)
                

        fig.suptitle("E=%.4EV/m  E/N=%.4ETd ne/N=%.2E gas temp.=%.2EK, N=%.4E $m^{-3}$"%(args.E_field, args.E_field/args.n0/1e-21, args.ion_deg, args.Tg, args.n0))
        # plt.show()
        effective_efield = args.E_field/args.n0/1e-21
        colfname = args.collisions.split("/")[-1].strip()
        if len(spec_sp._basis_p._dg_idx)==2:
            if args.steady_state == 1 : 
                plt.savefig("pde_cg_" + colfname + "_E%.2ETd"%effective_efield + "_sp_"+ str(args.sp_order) + "_nr" + str(args.NUM_P_RADIAL)+"_qpn_" + str(args.spline_qpts) + "_sweeping_" + args.sweep_param + "_lmax_" + str(args.l_max) +"_ion_deg_%.2E"%(args.ion_deg) + "_Tg%.2E"%(args.Tg) +".png")
            else:
                plt.savefig("pde_cg_" + colfname + "_E%.2ETd"%effective_efield + "_sp_"+ str(args.sp_order) + "_nr" + str(args.NUM_P_RADIAL)+"_qpn_" + str(args.spline_qpts) + "_sweeping_" + args.sweep_param + "_lmax_" + str(args.l_max) +"_ion_deg_%.2E"%(args.ion_deg) + "_Tg%.2E"%(args.Tg)+"_ts%.2E_T%.2E"%(args.T_DT, args.T_END) +".png")
        else:
            if args.steady_state == 1 : 
                plt.savefig("pde_dg_" + colfname + "_E%.2ETd"%effective_efield + "_sp_"+ str(args.sp_order) + "_nr" + str(args.NUM_P_RADIAL)+"_qpn_" + str(args.spline_qpts) + "_sweeping_" + args.sweep_param + "_lmax_" + str(args.l_max) +"_ion_deg_%.2E"%(args.ion_deg) + "_Tg%.2E"%(args.Tg) +".png")
            else:
                plt.savefig("pde_dg_" + colfname + "_EbyN%.2ETd"%effective_efield + "_sp_"+ str(args.sp_order) + "_nr" + str(args.NUM_P_RADIAL)+"_qpn_" + str(args.spline_qpts) + "_sweeping_" + args.sweep_param + "_lmax_" + str(args.l_max) +"_ion_deg_%.2E"%(args.ion_deg) + "_Tg%.2E"%(args.Tg)+"_ts%.2E_T%.2E"%(args.T_DT, args.T_END) +".png")

        plt.close()





