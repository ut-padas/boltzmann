import bolsig
import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.interpolate
import collisions

parser = argparse.ArgumentParser()
parser.add_argument("-Nr", "--NUM_P_RADIAL"                       , help="Number of polynomials in radial direction", type=int, default=16)
parser.add_argument("-T", "--T_END"                               , help="Simulation time", type=float, default=1e-4)
parser.add_argument("-dt", "--T_DT"                               , help="Simulation time step size ", type=float, default=1e-7)
parser.add_argument("-o",  "--out_fname"                          , help="output file name", type=str, default='coll_op')
parser.add_argument("-ts_tol", "--ts_tol"                         , help="adaptive timestep tolerance", type=float, default=1e-15)
parser.add_argument("-l_max", "--l_max"                           , help="max polar modes in SH expansion", type=int, default=1)
parser.add_argument("-c", "--collisions"                          , help="collisions included (g0, g0Const, g0NoLoss, g2, g2Const)",nargs='+', type=str, default=["g0Const"])
parser.add_argument("-ev", "--electron_volt"                      , help="initial electron volt", type=float, default=0.25)
parser.add_argument("-bscale", "--basis_scale"                    , help="basis electron volt", type=float, default=1.0)
parser.add_argument("-q_vr", "--quad_radial"                      , help="quadrature in r"        , type=int, default=200)
parser.add_argument("-q_vt", "--quad_theta"                       , help="quadrature in polar"    , type=int, default=8)
parser.add_argument("-q_vp", "--quad_phi"                         , help="quadrature in azimuthal", type=int, default=8)
parser.add_argument("-q_st", "--quad_s_theta"                     , help="quadrature in scattering polar"    , type=int, default=8)
parser.add_argument("-q_sp", "--quad_s_phi"                       , help="quadrature in scattering azimuthal", type=int, default=8)
parser.add_argument("-radial_poly", "--radial_poly"               , help="radial basis", type=str, default="bspline")
parser.add_argument("-sp_order", "--spline_order"                 , help="b-spline order", type=int, default=1)
parser.add_argument("-spline_qpts", "--spline_q_pts_per_knot"     , help="q points per knots", type=int, default=7)
parser.add_argument("-E", "--E_field"                             , help="Electric field in V/m", type=float, default=100)
parser.add_argument("-dv", "--dv_target"                          , help="target displacement of distribution in v_th units", type=float, default=0)
parser.add_argument("-nt", "--num_timesteps"                      , help="target number of time steps", type=float, default=100)
parser.add_argument("-steady", "--steady_state"                   , help="Steady state or transient", type=int, default=1)
parser.add_argument("-run_bolsig_only", "--run_bolsig_only"       , help="run the bolsig code only", type=bool, default=False)
parser.add_argument("-bolsig", "--bolsig_dir"                     , help="Bolsig directory", type=str, default="../../Bolsig/")
parser.add_argument("-sweep_values", "--sweep_values"             , help="Values for parameter sweep", nargs='+', type=float, default=[24, 48, 96])
parser.add_argument("-sweep_param", "--sweep_param"               , help="Paramter to sweep: Nr, ev, bscale, E, radial_poly", type=str, default="Nr")
parser.add_argument("-dg", "--use_dg"                             , help="enable dg splines", type=int, default=0)
parser.add_argument("-Tg", "--Tg"                                 , help="Gass temperature (K)" , type=float, default=0.0)
parser.add_argument("-ion_deg", "--ion_deg"                       , help="Ionization degreee"   , type=float, default=1)
parser.add_argument("-store_eedf", "--store_eedf"                 , help="store eedf"   , type=int, default=0)
parser.add_argument("-store_csv", "--store_csv"                   , help="store csv format of QoI comparisons", type=int, default=0)
parser.add_argument("-ee_collisions", "--ee_collisions"           , help="Enable electron-electron collisions", type=float, default=1)
parser.add_argument("-bolsig_precision", "--bolsig_precision"     , help="precision value for bolsig code", type=float, default=1e-15)
parser.add_argument("-bolsig_convergence", "--bolsig_convergence" , help="convergence value for bolsig code", type=float, default=1e-8)
parser.add_argument("-bolsig_grid_pts", "--bolsig_grid_pts"       , help="grid points for bolsig code"      , type=int, default=256)

args                = parser.parse_args()
run_data = list()
for i, value in enumerate(args.sweep_values):
    if args.sweep_param == "Nr":
        args.bolsig_grid_pts = int(value)
    elif args.sweep_param == "Tolerance":
        args.bolsig_convergence = value
    else:
        print("unknown sweep parameter value")
        exit(0)
    print(args)

    try:
        bolsig.run_bolsig(args)
        [bolsig_ev, bolsig_f0, bolsig_a, bolsig_E, bolsig_mu, bolsig_M, bolsig_D, bolsig_rates,bolsig_cclog] = bolsig.parse_bolsig(args.bolsig_dir+"argon.out",len(args.collisions))
        run_data.append([bolsig_ev, bolsig_f0, bolsig_a, bolsig_E, bolsig_mu, bolsig_M, bolsig_D, bolsig_rates,bolsig_cclog])
    except:
        print(args.bolsig_dir+"argon.out file not found due to Bolsig+ run faliure")
        sys.exit(0)
        


num_subplots = 2 + 2 + 1
num_plt_cols = 5
num_plt_rows = np.int64(np.ceil(num_subplots/num_plt_cols))
fig       = plt.figure(figsize=(num_plt_cols * 5 + 0.5*(num_plt_cols-1), num_plt_rows * 5 + 0.5*(num_plt_rows-1)), dpi=300)
fig.suptitle("E=%.4EV/m  E/N=%.4ETd ne/N=%.2E gas temp.=%.2EK, N=%.4E $m^{-3}$"%(args.E_field, args.E_field/collisions.AR_NEUTRAL_N/1e-21, args.ion_deg, args.Tg, collisions.AR_NEUTRAL_N))

bolsig_ev_hs = run_data[-1][0]
bolsig_f0_hs = run_data[-1][1]
bolsig_a_hs  = run_data[-1][2]
bolsig_f1_hs = abs(bolsig_f0_hs*bolsig_a_hs* np.sqrt(1/3))

mu_list    = list()
M_list     = list()
D_list     = list()
rates_list = list() 

for i, value in enumerate(args.sweep_values):
    run_i = run_data[i]

    bolsig_ev = run_i[0]
    bolsig_f0 = run_i[1]
    bolsig_a = run_i[2]
    bolsig_E = run_i[3]
    bolsig_mu = run_i[4]
    bolsig_M = run_i[5]
    bolsig_D = run_i[6]
    bolsig_rates = run_i[7]
    bolsig_cclog = run_i[8]
    
    mu_list.append(bolsig_mu)
    M_list.append(bolsig_M)
    D_list.append(bolsig_D)
    rates_list.append(bolsig_rates)

    bolsig_f1 = abs(bolsig_f0*bolsig_a * np.sqrt(1/3))

    bolsig_f0_inp  =  scipy.interpolate.interp1d(bolsig_ev, bolsig_f0, fill_value="extrapolate")(bolsig_ev_hs)
    bolsig_f1_inp  =  scipy.interpolate.interp1d(bolsig_ev, bolsig_f1, fill_value="extrapolate")(bolsig_ev_hs)
    
    plt.subplot(num_plt_rows, num_plt_cols, 1)
    color = next(plt.gca()._get_lines.prop_cycler)['color']
    lbl = "%s=%.2E"%(args.sweep_param, value)
    plt.semilogy(bolsig_ev, bolsig_f0,label=lbl,color=color)
    plt.title("f0")
    plt.xlabel("energy (eV)")
    plt.grid(visible=True)
    

    plt.subplot(num_plt_rows, num_plt_cols, 2)
    lbl = "%s=%.2E"%(args.sweep_param, value)
    plt.semilogy(bolsig_ev, bolsig_f1,label=lbl,color=color)
    plt.xlabel("energy (eV)")
    plt.title("f1")
    plt.grid(visible=True)
    

    plt.subplot(num_plt_rows, num_plt_cols, 3)
    lbl = "%s=%.2E"%(args.sweep_param, value)
    plt.semilogy(bolsig_ev_hs, np.abs(1-np.abs(bolsig_f0_inp)/np.abs(bolsig_f0_hs)),label=lbl,color=color)
    plt.title("f0")
    plt.xlabel("energy (eV)")
    plt.ylabel("relative error")
    plt.grid(visible=True)
    

    plt.subplot(num_plt_rows, num_plt_cols, 4)
    lbl = "%s=%.2E"%(args.sweep_param, value)
    plt.semilogy(bolsig_ev_hs, np.abs(1-np.abs(bolsig_f1_inp)/np.abs(bolsig_f1_hs)),label=lbl,color=color)
    plt.title("f1")
    plt.xlabel("energy (eV)")
    plt.ylabel("relative error")
    plt.grid(visible=True)

plt.subplot(num_plt_rows, num_plt_cols, 5)
mu_list = np.array(mu_list)
M_list = np.array(M_list)
D_list = np.array(D_list)
rates_list = np.array(rates_list)


plt.semilogy(args.sweep_values, np.abs(1-mu_list/mu_list[-1]),'.-', label="mean energy")
plt.semilogy(args.sweep_values, np.abs(1-M_list/M_list[-1]),  '^-', label="mobility")
plt.semilogy(args.sweep_values, np.abs(1-D_list/D_list[-1]),  'v-', label="diffusion")

plt.semilogy(args.sweep_values, np.abs(1-rates_list[:,0]/rates_list[-1,0]),  '1-', label="elastic")
if rates_list[-1,1] > 0:
    plt.semilogy(args.sweep_values, np.abs(1-rates_list[:,1]/rates_list[-1,1]),  '2-', label="elastic")
#plt.semilogy(args.sweep_values, np.abs(1-mu_list/mu_list[-1]),'.', label="mean energy")
plt.xlabel(args.sweep_param)
plt.ylabel("relative error")
plt.grid()
plt.legend()

    

plt.subplot(num_plt_rows, num_plt_cols, 1)
plt.legend()
plt.savefig("bolsig_convergence_" + "_".join(args.collisions) + "_E" + str(args.E_field) + "_sweeping_" + args.sweep_param + "_lmax_" + str(args.l_max) +"_ion_deg_%.2E"%(args.ion_deg) + "_Tg%.2E"%(args.Tg) +".svg")