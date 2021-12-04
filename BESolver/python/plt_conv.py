"""
@package : simple convergence plots
"""
import matplotlib as mpl
#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
import matplotlib.pyplot as plt
import spec_spherical as sp
import argparse
import numpy as np
import scipy.constants 
import utils as BEUtils
import collisions
import collision_operator_spherical as colOpSp
import parameters as params
from tabulate import tabulate
import basis
import glob

TIME_INDEX=0
C000_INDEX=1


def plot_convgence(folder, fnames_list, nr_list,ev,q_mode,r_mode):
    DATA_FOLDER_NAME = folder
    #DATA_FOLDER_NAME="dat_1ev_cs_m"+str(mm)
    EV = 1.0
    params.BEVelocitySpace.SPH_HARM_LM = [[i,j] for i in range(1) for j in range(i+1)]
    params.BEVelocitySpace.NUM_Q_VR    = 4097
    params.BEVelocitySpace.NUM_Q_VT    = 2
    params.BEVelocitySpace.NUM_Q_VP    = 2
    params.BEVelocitySpace.NUM_Q_CHI   = 2
    params.BEVelocitySpace.NUM_Q_PHI   = 2
    
    collisions.AR_NEUTRAL_N=3.22e22
    collisions.MAXWELLIAN_N=1e18
    collisions.AR_IONIZED_N=1e18
    collisions.MAXWELLIAN_TEMP_K   = EV * collisions.TEMP_K_1EV
    collisions.ELECTRON_THEMAL_VEL = collisions.electron_thermal_velocity(collisions.MAXWELLIAN_TEMP_K) 
    VTH = collisions.ELECTRON_THEMAL_VEL
    maxwellian = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)

    NR   = nr_list
    f_names= fnames_list
    data = list()
    for f in f_names:
        data.append(np.loadtxt(DATA_FOLDER_NAME+"/"+f))

    import matplotlib.pylab as plt
    for i in range(len(NR)):
        nr = NR[i]
        num_sh = len(params.BEVelocitySpace.SPH_HARM_LM)
        tail_end   = (nr+1)*num_sh
        tail_begin = tail_end//2
        tail_norm = np.linalg.norm(data[i][:,C000_INDEX + tail_begin: C000_INDEX + tail_end],axis=1)/np.linalg.norm(data[i][:,C000_INDEX: C000_INDEX + tail_end],axis=1)
        plt.plot(data[i][:,TIME_INDEX], tail_norm,label="Nr=%d"%(nr))
    
    plt.xlabel("time (s)")
    plt.ylabel("tail norm")
    plt.title("spectral tails")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    fname=DATA_FOLDER_NAME+"/tails.png"
    plt.savefig(fname)
    plt.close()

    mass        = np.zeros((len(NR),data[0].shape[0]))
    temperature = np.zeros((len(NR),data[0].shape[0]))

    for i in range(len(NR)):
        nr = NR[i]
        params.BEVelocitySpace().VELOCITY_SPACE_POLY_ORDER=nr
        cf_sp    = colOpSp.CollisionOpSP(3,nr,q_mode=q_mode,poly_type=r_mode)
        spec_sp  = cf_sp._spec

        mass[i,:]         = BEUtils.moment_n_f(spec_sp,np.transpose(data[i][:,C000_INDEX:]),maxwellian,VTH,0,None,None,None,1)
        temperature[i,:]  = BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec_sp,np.transpose(data[i][:,C000_INDEX:]),maxwellian,VTH,None,None,None, mass[i,:],1)


    for i in range(0,len(NR)):
        nr = NR[i]
        temp_ev  = temperature[i] * collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT
        plt.plot(data[i][:,TIME_INDEX],temp_ev,label="Nr=%d"%(NR[i]))
        
    plt.xlabel("time (s)")
    plt.ylabel("temperature (eV)")
    plt.legend()
    plt.grid()
    plt.title("temperature")
    fname=DATA_FOLDER_NAME+"/temp.png"
    plt.savefig(fname)
    plt.close()

    for i in range(1,len(NR)):
        temp_ev_0  = temperature[i-1]
        temp_ev_1  = temperature[i]
        assert np.allclose(np.abs(data[i][:,TIME_INDEX]-data[i-1][:,TIME_INDEX]),np.zeros_like(data[i][:,TIME_INDEX])), "mismatching time points in the temperature plot"
        plt.plot(data[i-1][:,TIME_INDEX],np.abs(temp_ev_0-temp_ev_1)/temp_ev_1,label="Nr=%d vs. Nr=%d"%(NR[i-1],NR[i]))
        
        
    plt.xlabel("time (s)")
    plt.ylabel("error")
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.grid()
    plt.title("temperature relative error")
    fname=DATA_FOLDER_NAME+"/temp_error.png"
    plt.savefig(fname)
    plt.close()



    for i in range(0,len(NR)):
        nr = NR[i]
        plt.plot(data[i][:,TIME_INDEX],mass[i]/mass[i,0],label="Nr=%d"%(NR[i]))
        
    plt.xlabel("time (s)")
    plt.ylabel("mass/m(t=0)")
    plt.legend()
    plt.grid()
    plt.title("mass / mass_initial")
    fname=DATA_FOLDER_NAME+"/mass.png"
    plt.savefig(fname)
    plt.close()


    ev=np.linspace(0,30,300)
    for i in range(len(NR)):
        nr = NR[i]
        params.BEVelocitySpace().VELOCITY_SPACE_POLY_ORDER=nr
        cf_sp    = colOpSp.CollisionOpSP(3,nr,q_mode=q_mode,poly_type=r_mode)
        spec_sp  = cf_sp._spec

        eedf_1 = BEUtils.get_eedf(ev,spec_sp,data[i][-1,C000_INDEX:],maxwellian,VTH,1)
        plt.plot(ev,eedf_1,label="Nr=%d t=%.2E"%(NR[i],data[i][-1,TIME_INDEX]))
    
    plt.xlabel("energy (eV)")
    plt.ylabel("EEDF")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.title("EEDF final")
    fname=DATA_FOLDER_NAME+"/eedf_final.png"
    plt.savefig(fname)
    plt.close()

    
    for i in range(1,len(NR)):
        cf_sp0    = colOpSp.CollisionOpSP(3,NR[i-1],q_mode=q_mode,poly_type=r_mode)
        spec_sp0  = cf_sp0._spec

        cf_sp1    = colOpSp.CollisionOpSP(3,NR[i],q_mode=q_mode,poly_type=r_mode)
        spec_sp1  = cf_sp1._spec

        eedf_0 = BEUtils.get_eedf(ev,spec_sp0,data[i-1][-1,C000_INDEX:],maxwellian,VTH,1)
        eedf_1 = BEUtils.get_eedf(ev,spec_sp1,data[i][-1,C000_INDEX:],maxwellian,VTH,1)

        plt.plot(ev,np.abs(eedf_0-eedf_1),label="Nr=%d vs Nr=%d t=%.2E"%(NR[i-1],NR[i], data[i][-1,TIME_INDEX]))
    
    plt.xlabel("energy (eV)")
    plt.ylabel("error")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.title("EEDF final error")
    fname=DATA_FOLDER_NAME+"/eedf_final_error.png"
    plt.savefig(fname)
    plt.close()


    for i in range(len(NR)):
        nr = NR[i]
        params.BEVelocitySpace().VELOCITY_SPACE_POLY_ORDER=nr
        cf_sp    = colOpSp.CollisionOpSP(3,nr,q_mode=q_mode,poly_type=r_mode)
        spec_sp  = cf_sp._spec

        eedf_0 = BEUtils.get_eedf(ev,spec_sp,data[i][0,C000_INDEX:],maxwellian,VTH,1)
        plt.plot(ev,eedf_0,label="Nr=%d t=%.2E"%(NR[i],data[i][0,TIME_INDEX]))
        
    
    plt.xlabel("energy (eV)")
    plt.ylabel("EEDF")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.title("EEDF initial")
    fname=DATA_FOLDER_NAME+"/eedf_initial.png"
    plt.savefig(fname)
    plt.close()


    for i in range(1,len(NR)):
        cf_sp0    = colOpSp.CollisionOpSP(3,NR[i-1],q_mode=q_mode,poly_type=r_mode)
        spec_sp0  = cf_sp0._spec

        cf_sp1    = colOpSp.CollisionOpSP(3,NR[i],q_mode=q_mode,poly_type=r_mode)
        spec_sp1  = cf_sp1._spec

        eedf_0 = BEUtils.get_eedf(ev,spec_sp0,data[i-1][0,C000_INDEX:],maxwellian,VTH,1)
        eedf_1 = BEUtils.get_eedf(ev,spec_sp1,data[i][0,C000_INDEX:],maxwellian,VTH,1)

        plt.plot(ev,np.abs(eedf_0-eedf_1),label="Nr=%d vs Nr=%d t=%.2E"%(NR[i-1],NR[i], data[i][-1,TIME_INDEX]))
        
    
    plt.xlabel("energy (eV)")
    plt.ylabel("error")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.title("EEDF initial error")
    fname=DATA_FOLDER_NAME+"/eedf_initial_error.png"
    plt.savefig(fname)
    plt.close()



folder="dat_1ev_g0_maxpoly"
fname_list=["g0_dt_tol_1.00000000E-12_Nr_4.dat",
            "g0_dt_tol_1.00000000E-12_Nr_8.dat",
            "g0_dt_tol_1.00000000E-12_Nr_16.dat",
            "g0_dt_tol_1.00000000E-12_Nr_32.dat",
            "g0_dt_tol_1.00000000E-12_Nr_64.dat"]
nr_list=[4,8,16,32,64]
plot_convgence(folder,fname_list, nr_list, 1.0,sp.QuadMode.GMX,basis.BasisType.MAXWELLIAN_POLY)


folder="dat_1ev_g0_bspline"
fname_list=[#"g0_dt_tol_1.00000000E-12_Nr_4.dat",
            #"g0_dt_tol_1.00000000E-12_Nr_8.dat",
            "g0_dt_tol_1.00000000E-12_Nr_16.dat",
            "g0_dt_tol_1.00000000E-12_Nr_32.dat",
            "g0_dt_tol_1.00000000E-12_Nr_64.dat",
            "g0_dt_tol_1.00000000E-12_Nr_128.dat",
            "g0_dt_tol_1.00000000E-12_Nr_256.dat"
            #"g0_dt_tol_1.00000000E-12_Nr_512.dat"
            ]

nr_list=[16,32,64,128,256]
plot_convgence(folder,fname_list, nr_list, 1.0, sp.QuadMode.SIMPSON, basis.BasisType.SPLINES)