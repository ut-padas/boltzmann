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


def plot_convgence(folder, fnames_list, nr_list,ev,q_mode,r_mode,fname_prefix=""):
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

    import matplotlib.pyplot as plt
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
    fname=DATA_FOLDER_NAME+"/"+fname_prefix+"_tails.png"
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
    fname=DATA_FOLDER_NAME+"/"+fname_prefix+"_temp.png"
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
    fname=DATA_FOLDER_NAME+"/"+fname_prefix+"_temp_error.png"
    plt.savefig(fname)
    plt.close()



    for i in range(0,len(NR)):
        nr = NR[i]
        plt.plot(data[i][:,TIME_INDEX],mass[i]/mass[i,0],label="Nr=%d"%(NR[i]))

    plt.xlabel("time (s)")
    plt.ylabel("# of electrons / # of electrons (t=0) ")
    plt.legend()
    #plt.yscale("log")
    #plt.xscale("log")
    plt.grid()
    plt.title("number of electrons vs. time")
    fname=DATA_FOLDER_NAME+"/"+fname_prefix+"_mass.png"
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
    fname=DATA_FOLDER_NAME+"/"+fname_prefix+"_eedf_final.png"
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
    fname=DATA_FOLDER_NAME+"/"+fname_prefix+"_eedf_final_error.png"
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
    fname=DATA_FOLDER_NAME+"/"+fname_prefix+"_eedf_initial.png"
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
    fname=DATA_FOLDER_NAME+"/"+fname_prefix+"_eedf_initial_error.png"
    plt.savefig(fname)
    plt.close()


def plot_maxwell_vs_bspline(mw_folder,mw_fname,bs_folder,bs_fname,mw_nr,bs_nr,file_prefix):


    assert len(bs_nr) == len(mw_nr) , "NR size does not match"
    EV = 1.0
    params.BEVelocitySpace.SPH_HARM_LM = [[i,j] for i in range(1) for j in range(i+1)]
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

    
    data_mw = list()
    data_bs = list()

    for f in mw_fname:
        data_mw.append(np.loadtxt(mw_folder + "/" +f))

    for f in bs_fname:
        data_bs.append(np.loadtxt(bs_folder + "/" +f))
    
    mass_bs = list()
    temperature_bs=list()
    eedf_b_bs=list()
    eedf_e_bs=list()
    ev_pts = np.linspace(0,30,300)
    q_mode = sp.QuadMode.SIMPSON
    r_mode = basis.BasisType.SPLINES
    params.BEVelocitySpace.NUM_Q_VR  = 4049

    for i,nr in enumerate(bs_nr):
        params.BEVelocitySpace().VELOCITY_SPACE_POLY_ORDER=nr
        cf_sp    = colOpSp.CollisionOpSP(3,nr,q_mode=q_mode,poly_type=r_mode)
        spec_sp  = cf_sp._spec

        mass_bs.append(BEUtils.moment_n_f(spec_sp,np.transpose(data_bs[i][:,C000_INDEX:]),maxwellian,VTH,0,None,None,None,1))
        temperature_bs.append(BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec_sp,np.transpose(data_bs[i][:,C000_INDEX:]),maxwellian,VTH,None,None,None, mass_bs[i],1))

        eedf_b_bs.append(BEUtils.get_eedf(ev_pts,spec_sp,data_bs[i][0,C000_INDEX:],maxwellian,VTH,1))
        eedf_e_bs.append(BEUtils.get_eedf(ev_pts,spec_sp,data_bs[i][-1,C000_INDEX:],maxwellian,VTH,1))

    mass_mw = list()
    temperature_mw=list()
    eedf_b_mw=list()
    eedf_e_mw=list()
    q_mode = sp.QuadMode.GMX
    r_mode = basis.BasisType.MAXWELLIAN_POLY
    params.BEVelocitySpace.NUM_Q_VR  = 118
    for i,nr in enumerate(mw_nr):

        params.BEVelocitySpace().VELOCITY_SPACE_POLY_ORDER=nr
        cf_sp    = colOpSp.CollisionOpSP(3,nr,q_mode=q_mode,poly_type=r_mode)
        spec_sp  = cf_sp._spec

        mass_mw.append(BEUtils.moment_n_f(spec_sp,np.transpose(data_mw[i][:,C000_INDEX:]),maxwellian,VTH,0,None,None,None,1))
        temperature_mw.append(BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec_sp,np.transpose(data_mw[i][:,C000_INDEX:]),maxwellian,VTH,None,None,None, mass_mw[i],1))

        eedf_b_mw.append(BEUtils.get_eedf(ev_pts,spec_sp,data_mw[i][0,C000_INDEX:],maxwellian,VTH,1))
        eedf_e_mw.append(BEUtils.get_eedf(ev_pts,spec_sp,data_mw[i][-1,C000_INDEX:],maxwellian,VTH,1))

    for i in range(len(bs_nr)):
        assert np.allclose(np.abs(data_bs[i][:,TIME_INDEX] - data_mw[i][:,TIME_INDEX]),np.zeros_like(data_bs[i][:,TIME_INDEX]))," mismatching time points"  
    
    import matplotlib.pyplot as plt
    #plt.plot(data[0][:,TIME_INDEX],np.abs(temperature_mw-temperature_bs)/temperature_mw,label="nr=%d maxwell vs. nr=%d bspline"%(mw_nr,bs_nr))

    for i in range(len(bs_nr)):
        plt.plot(data_bs[i][:,TIME_INDEX],np.abs(temperature_mw[i]-temperature_bs[i])/temperature_mw[i],label="nr=%d maxwell vs. nr=%d bspline"%(mw_nr[i],bs_nr[i]))
        
    
    plt.xlabel("time (s)")
    plt.ylabel("error")
    plt.yscale("log")
    plt.title("Temperature relative error")
    plt.grid()
    plt.legend()
    plt.savefig(file_prefix+"_temp_error"+".png")
    plt.close()

    for i in range(len(bs_nr)):
        plt.plot(ev_pts,np.abs(eedf_b_bs[i]-eedf_b_mw[i]),label="nr=%d maxwell vs. nr=%d bspline"%(mw_nr[i],bs_nr[i]))

    plt.xlabel("time (s)")
    plt.ylabel("error")
    plt.yscale("log")
    plt.xscale("log")
    plt.title("EEDF initial error")
    plt.grid()
    plt.legend()
    plt.savefig(file_prefix+"_eedf_initial_error"+".png")
    plt.close()

    for i in range(len(bs_nr)):
        plt.plot(ev_pts,np.abs(eedf_e_bs[i]-eedf_e_mw[i]),label="nr=%d maxwell vs. nr=%d bspline"%(mw_nr[i],bs_nr[i]))

    plt.xlabel("time (s)")
    plt.ylabel("error")
    plt.yscale("log")
    plt.xscale("log")
    plt.title("EEDF final error")
    plt.grid()
    plt.legend()
    plt.savefig(file_prefix+"_eedf_final_error"+".png")
    plt.close()




folder_g0_mw_m0     = "dat_1ev_cs_m0"
fname_list_g0_mw_m0 = [ "g0_dt_tol_1.00000000E-12_Nr_4.dat",
                        "g0_dt_tol_1.00000000E-12_Nr_8.dat",
                        "g0_dt_tol_1.00000000E-12_Nr_16.dat",
                        "g0_dt_tol_1.00000000E-12_Nr_32.dat",
                        "g0_dt_tol_1.00000000E-12_Nr_64.dat"]

plot_convgence(folder_g0_mw_m0, fname_list_g0_mw_m0, [4,8,16,32,64], 1.0,sp.QuadMode.GMX,basis.BasisType.MAXWELLIAN_POLY,"g0_mw_m0")

folder_g0_mw_m1     = "dat_1ev_cs_m1"
fname_list_g0_mw_m1 = [ "g0_dt_tol_1.00000000E-12_Nr_4.dat",
                        "g0_dt_tol_1.00000000E-12_Nr_8.dat",
                        "g0_dt_tol_1.00000000E-12_Nr_16.dat",
                        "g0_dt_tol_1.00000000E-12_Nr_32.dat",
                        "g0_dt_tol_1.00000000E-12_Nr_64.dat"]
plot_convgence(folder_g0_mw_m1, fname_list_g0_mw_m1, [4,8,16,32,64], 1.0,sp.QuadMode.GMX,basis.BasisType.MAXWELLIAN_POLY,"g0_mw_m1")




folder_g0_mw     = "dat_1ev_g0_maxpoly"
fname_list_g0_mw = ["g0_dt_tol_1.00000000E-12_Nr_4.dat",
            "g0_dt_tol_1.00000000E-12_Nr_8.dat",
            "g0_dt_tol_1.00000000E-12_Nr_16.dat",
            "g0_dt_tol_1.00000000E-12_Nr_32.dat",
            "g0_dt_tol_1.00000000E-12_Nr_64.dat"]

#plot_convgence(folder_g0_mw,fname_list_g0_mw, [4,8,16,32,64], 1.0,sp.QuadMode.GMX,basis.BasisType.MAXWELLIAN_POLY,"g0_mw")


folder_g02_mw     = "dat_1ev_g02_maxpoly"
fname_list_g02_mw = ["g02_dt_tol_1.00000000E-12_Nr_4.dat",
            "g02_dt_tol_1.00000000E-12_Nr_8.dat",
            "g02_dt_tol_1.00000000E-12_Nr_16.dat",
            "g02_dt_tol_1.00000000E-12_Nr_32.dat",
            "g02_dt_tol_1.00000000E-12_Nr_64.dat"]
#plot_convgence(folder_g02_mw,fname_list_g02_mw, [4,8,16,32,64], 1.0,sp.QuadMode.GMX,basis.BasisType.MAXWELLIAN_POLY,"g02_mw")



folder_g0_bs    ="dat_1ev_g0_bspline"
fname_list_g0_bs=["g0_dt_tol_1.00000000E-12_Nr_16.dat",
            "g0_dt_tol_1.00000000E-12_Nr_32.dat",
            "g0_dt_tol_1.00000000E-12_Nr_64.dat",
            "g0_dt_tol_1.00000000E-12_Nr_128.dat",
            "g0_dt_tol_1.00000000E-12_Nr_256.dat"]
#plot_convgence(folder_g0_bs,fname_list_g0_bs, [16,32,64,128,256], 1.0, sp.QuadMode.SIMPSON, basis.BasisType.SPLINES,"g0_bs")


folder_g02_bs     ="dat_1ev_g02_bspline"
fname_list_g02_bs =["g02_dt_tol_1.00000000E-12_Nr_16.dat",
                    "g02_dt_tol_1.00000000E-12_Nr_32.dat",
                    "g02_dt_tol_1.00000000E-12_Nr_64.dat",
                    "g02_dt_tol_1.00000000E-12_Nr_128.dat",
                    "g02_dt_tol_1.00000000E-12_Nr_256.dat"]
#plot_convgence(folder_g02_bs,fname_list_g02_bs, [16,32,64,128,256], 1.0, sp.QuadMode.SIMPSON, basis.BasisType.SPLINES,"g02_bs")


#plot_maxwell_vs_bspline(folder_g0_mw,fname_list_g0_mw,folder_g0_bs,fname_list_g0_bs,[4,8,16,32,64],[16,32,64,128,256],"g0_mw_bs")


# did you change the correct bounds
# folder="dat_1ev_g02_bspline1"
# fname_list=["g02_dt_tol_1.00000000E-12_Nr_32.dat",
#             "g02_dt_tol_1.00000000E-12_Nr_64.dat",
#             "g02_dt_tol_1.00000000E-12_Nr_128.dat",
#             "g02_dt_tol_1.00000000E-12_Nr_256.dat"
#             ]

# nr_list=[32,64,128,256]
# plot_convgence(folder,fname_list, nr_list, 1.0, sp.QuadMode.SIMPSON, basis.BasisType.SPLINES)