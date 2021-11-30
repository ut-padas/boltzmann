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


TIME_INDEX=0
MASS_INDEX=1
TEMP_INDEX=3
C000_INDEX=4
DATA_FOLDER_NAME="dat_1ev_no_proj"

#f = plt.figure()
#f.set_figwidth(100)
#f.set_figheight(100)
#f.set_size_inches(8, 8, forward=True)
plt.rcParams["figure.figsize"] = (6,6)


def spec_tail(cf):
    tail_index = len(cf[0,C000_INDEX:])//2
    #print(tail_index)
    return np.linalg.norm(cf[:,(C000_INDEX + tail_index):],axis=1) / np.linalg.norm(cf[:,C000_INDEX:],axis=1)

def g0_convergence():
    file_names=[["dat/g0_dt_1.00000000E-10_Nr_3.dat",  "dat/g0_dt_5.00000000E-11_Nr_3.dat",  "dat/g0_dt_2.50000000E-11_Nr_3.dat",  "dat/g0_dt_6.25000000E-12_Nr_3.dat"],
                ["dat/g0_dt_1.00000000E-10_Nr_7.dat",  "dat/g0_dt_5.00000000E-11_Nr_7.dat",  "dat/g0_dt_2.50000000E-11_Nr_7.dat",  "dat/g0_dt_6.25000000E-12_Nr_7.dat"],
                ["dat/g0_dt_1.00000000E-10_Nr_15.dat", "dat/g0_dt_5.00000000E-11_Nr_15.dat", "dat/g0_dt_2.50000000E-11_Nr_15.dat", "dat/g0_dt_6.25000000E-12_Nr_15.dat"]]


    data_nr_1x_dt1 = np.loadtxt(file_names[0][0],skiprows=1)
    data_nr_2x_dt1 = np.loadtxt(file_names[1][0],skiprows=1)
    data_nr_4x_dt1 = np.loadtxt(file_names[2][0],skiprows=1)


    data_nr_1x_dt2 = np.loadtxt(file_names[0][1],skiprows=1)
    data_nr_2x_dt2 = np.loadtxt(file_names[1][1],skiprows=1)
    data_nr_4x_dt2 = np.loadtxt(file_names[2][1],skiprows=1)

    data_nr_1x_dt4 = np.loadtxt(file_names[0][2],skiprows=1)
    data_nr_2x_dt4 = np.loadtxt(file_names[1][2],skiprows=1)
    data_nr_4x_dt4 = np.loadtxt(file_names[2][2],skiprows=1)

    plt.close()
    plt.plot(data_nr_1x_dt1[:,TIME_INDEX],data_nr_1x_dt1[:,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0, dt=1E-10, Nr=4")
    plt.plot(data_nr_2x_dt1[:,TIME_INDEX],data_nr_2x_dt1[:,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0, dt=1E-10, Nr=8")
    plt.plot(data_nr_4x_dt1[:,TIME_INDEX],data_nr_4x_dt1[:,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0, dt=1E-10, Nr=16")

    plt.plot(data_nr_1x_dt2[:,TIME_INDEX],data_nr_1x_dt2[:,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0, dt=5E-11, Nr=4")
    plt.plot(data_nr_2x_dt2[:,TIME_INDEX],data_nr_2x_dt2[:,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0, dt=5E-11, Nr=8")
    plt.plot(data_nr_4x_dt2[:,TIME_INDEX],data_nr_4x_dt2[:,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0, dt=5E-11, Nr=16")
    plt.title("G0 with cooling")
    plt.grid()
    plt.xlabel("time(s)")
    plt.ylabel("temperature (eV)")
    plt.legend()
    plt.show()
    
    


    plt.close()

    plt.plot(data_nr_1x_dt1[:,TIME_INDEX],abs(data_nr_4x_dt1[:,TEMP_INDEX]-data_nr_1x_dt1[:,TEMP_INDEX])/data_nr_4x_dt1[:,TEMP_INDEX],label="G0, dt=1E-10, Nr=4 vs. Nr=16")
    plt.plot(data_nr_1x_dt1[:,TIME_INDEX],abs(data_nr_4x_dt1[:,TEMP_INDEX]-data_nr_2x_dt1[:,TEMP_INDEX])/data_nr_4x_dt1[:,TEMP_INDEX],label="G0, dt=1E-10, Nr=8 vs. Nr=16")

    plt.plot(data_nr_1x_dt2[:,TIME_INDEX],abs(data_nr_4x_dt2[:,TEMP_INDEX]-data_nr_1x_dt2[:,TEMP_INDEX])/data_nr_4x_dt2[:,TEMP_INDEX],label="G0, dt=5E-11, Nr=4 vs. Nr=16")
    plt.plot(data_nr_1x_dt2[:,TIME_INDEX],abs(data_nr_4x_dt2[:,TEMP_INDEX]-data_nr_2x_dt2[:,TEMP_INDEX])/data_nr_4x_dt2[:,TEMP_INDEX],label="G0, dt=5E-11, Nr=8 vs. Nr=16")

    plt.plot(data_nr_1x_dt4[:,TIME_INDEX],abs(data_nr_4x_dt4[:,TEMP_INDEX]-data_nr_1x_dt4[:,TEMP_INDEX])/data_nr_4x_dt4[:,TEMP_INDEX],label="G0, dt=2.5E-11, Nr=4 vs. Nr=16")
    plt.plot(data_nr_1x_dt4[:,TIME_INDEX],abs(data_nr_4x_dt4[:,TEMP_INDEX]-data_nr_2x_dt4[:,TEMP_INDEX])/data_nr_4x_dt4[:,TEMP_INDEX],label="G0, dt=2.5E-11, Nr=8 vs. Nr=16")
    
    plt.title("G0 with cooling (refinement Nr)")
    plt.xlabel("time (s)")
    plt.ylabel("relative error (temperature)")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.show()

    norm_1dt_1x_4x = np.max(abs(data_nr_4x_dt1[:,TEMP_INDEX]-data_nr_1x_dt1[:,TEMP_INDEX])/data_nr_4x_dt1[:,TEMP_INDEX])
    norm_1dt_2x_4x = np.max(abs(data_nr_4x_dt1[:,TEMP_INDEX]-data_nr_2x_dt1[:,TEMP_INDEX])/data_nr_4x_dt1[:,TEMP_INDEX])

    norm_2dt_1x_4x = np.max(abs(data_nr_4x_dt2[:,TEMP_INDEX]-data_nr_1x_dt2[:,TEMP_INDEX])/data_nr_4x_dt2[:,TEMP_INDEX])
    norm_2dt_2x_4x = np.max(abs(data_nr_4x_dt2[:,TEMP_INDEX]-data_nr_2x_dt2[:,TEMP_INDEX])/data_nr_4x_dt2[:,TEMP_INDEX])

    norm_4dt_1x_4x = np.max(abs(data_nr_4x_dt4[:,TEMP_INDEX]-data_nr_1x_dt4[:,TEMP_INDEX])/data_nr_4x_dt4[:,TEMP_INDEX])
    norm_4dt_2x_4x = np.max(abs(data_nr_4x_dt4[:,TEMP_INDEX]-data_nr_2x_dt4[:,TEMP_INDEX])/data_nr_4x_dt4[:,TEMP_INDEX])

    table_data = [
        [1e-10, norm_1dt_1x_4x, norm_1dt_2x_4x],
        [5e-11, norm_2dt_1x_4x, norm_2dt_2x_4x],
        [2.5e-11, norm_4dt_1x_4x, norm_4dt_2x_4x]]

    print("========================================")
    print("          G0 refinement in Nr           ")
    print("========================================")
    print(tabulate(table_data,headers=["dt","Nr=4 vs Nr=16","Nr=8 vs Nr=16"]))
    
    plt.close()
    sk_tail = 1
    data1 = data_nr_1x_dt4[range(0,len(data_nr_1x_dt4[:-sk_tail,TIME_INDEX]),4),TEMP_INDEX]
    data2 = data_nr_1x_dt4[range(0,len(data_nr_1x_dt4[:-sk_tail,TIME_INDEX]),2),TEMP_INDEX]

    data3 = data_nr_2x_dt4[range(0,len(data_nr_2x_dt4[:-sk_tail,TIME_INDEX]),4),TEMP_INDEX]
    data4 = data_nr_2x_dt4[range(0,len(data_nr_2x_dt4[:-sk_tail,TIME_INDEX]),2),TEMP_INDEX]

    data5 = data_nr_4x_dt4[range(0,len(data_nr_4x_dt4[:-sk_tail,TIME_INDEX]),4),TEMP_INDEX]
    data6 = data_nr_4x_dt4[range(0,len(data_nr_4x_dt4[:-sk_tail,TIME_INDEX]),2),TEMP_INDEX]

    assert np.allclose((data_nr_1x_dt1[:len(data1),TIME_INDEX] - data_nr_1x_dt4[range(0,len(data_nr_1x_dt4[:-sk_tail,TIME_INDEX]),4),TIME_INDEX]), np.zeros_like(data1)), "times does not match"
    assert np.allclose((data_nr_1x_dt2[:len(data2),TIME_INDEX] - data_nr_1x_dt4[range(0,len(data_nr_1x_dt4[:-sk_tail,TIME_INDEX]),2),TIME_INDEX]), np.zeros_like(data2)), "times does not match"

    assert np.allclose((data_nr_2x_dt1[:len(data3),TIME_INDEX] - data_nr_2x_dt4[range(0,len(data_nr_2x_dt4[:-sk_tail,TIME_INDEX]),4),TIME_INDEX]), np.zeros_like(data3)), "times does not match"
    assert np.allclose((data_nr_2x_dt2[:len(data4),TIME_INDEX] - data_nr_2x_dt4[range(0,len(data_nr_2x_dt4[:-sk_tail,TIME_INDEX]),2),TIME_INDEX]), np.zeros_like(data4)), "times does not match"

    assert np.allclose((data_nr_4x_dt1[:len(data3),TIME_INDEX] - data_nr_4x_dt4[range(0,len(data_nr_4x_dt4[:-sk_tail,TIME_INDEX]),4),TIME_INDEX]), np.zeros_like(data5)), "times does not match"
    assert np.allclose((data_nr_4x_dt2[:len(data4),TIME_INDEX] - data_nr_4x_dt4[range(0,len(data_nr_4x_dt4[:-sk_tail,TIME_INDEX]),2),TIME_INDEX]), np.zeros_like(data6)), "times does not match"

    plt.plot(data_nr_1x_dt1[:len(data1),TIME_INDEX], abs(data1-data_nr_1x_dt1[:len(data1),TEMP_INDEX])/data1,label="G0, dt=1E-10 error w.r.t dt= 2.5E-11, Nr=4")
    plt.plot(data_nr_1x_dt2[:len(data2),TIME_INDEX], abs(data2-data_nr_1x_dt2[:len(data2),TEMP_INDEX])/data2,label="G0, dt=5E-11 error w.r.t dt= 2.5E-11, Nr=4")

    plt.plot(data_nr_2x_dt1[:len(data3),TIME_INDEX], abs(data3-data_nr_2x_dt1[:len(data3),TEMP_INDEX])/data3,label="G0, dt=1E-10 error w.r.t dt= 2.5E-11, Nr=8")
    plt.plot(data_nr_2x_dt2[:len(data4),TIME_INDEX], abs(data4-data_nr_2x_dt2[:len(data4),TEMP_INDEX])/data4,label="G0, dt=5E-11 error w.r.t dt= 2.5E-11, Nr=8")

    plt.plot(data_nr_4x_dt1[:len(data5),TIME_INDEX], abs(data5-data_nr_4x_dt1[:len(data5),TEMP_INDEX])/data5,label="G0, dt=1E-10 error w.r.t dt= 2.5E-11, Nr=16")
    plt.plot(data_nr_4x_dt2[:len(data6),TIME_INDEX], abs(data6-data_nr_4x_dt2[:len(data6),TEMP_INDEX])/data6,label="G0, dt=5E-11 error w.r.t dt= 2.5E-11, Nr=16")

    norm_nr4_dt1_dt4 = np.max(abs(data1-data_nr_1x_dt1[:len(data1),TEMP_INDEX])/data1)
    norm_nr4_dt2_dt4 = np.max(abs(data2-data_nr_1x_dt2[:len(data2),TEMP_INDEX])/data2)

    norm_nr8_dt1_dt4 = np.max(abs(data3-data_nr_2x_dt1[:len(data3),TEMP_INDEX])/data3)
    norm_nr8_dt2_dt4 = np.max(abs(data4-data_nr_2x_dt2[:len(data4),TEMP_INDEX])/data4)

    norm_nr16_dt1_dt4 = np.max(abs(data5-data_nr_4x_dt1[:len(data5),TEMP_INDEX])/data5)
    norm_nr16_dt2_dt4 = np.max(abs(data6-data_nr_4x_dt2[:len(data6),TEMP_INDEX])/data6)

    table_data = [
         [4,norm_nr4_dt1_dt4,norm_nr4_dt2_dt4],
         [8,norm_nr8_dt1_dt4,norm_nr8_dt2_dt4],
        [16,norm_nr16_dt1_dt4,norm_nr16_dt2_dt4]]


    print("========================================")
    print("          G0 refinement in dt           ")
    print("========================================")
    print(tabulate(table_data,headers=["Nr","1e-10 vs. 2.5E-11","5e-11 vs. 2.5E-11"]))
    
    plt.title("G0 with cooling (refinement in time)")
    plt.xlabel("time (s)")
    plt.ylabel("relative error (temperature)")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.show()

    sk_tail = 1
    data1   = data_nr_4x_dt4[range(0,len(data_nr_4x_dt4[:-sk_tail,TIME_INDEX]),4),:]
    data2   = data_nr_4x_dt4[range(0,len(data_nr_4x_dt4[:-sk_tail,TIME_INDEX]),2),:]

    data3   = data_nr_1x_dt1[0:len(data1[:,TEMP_INDEX]),:]
    data4   = data_nr_2x_dt2[0:len(data2[:,TEMP_INDEX]),:]

    assert np.allclose((data1[:,TIME_INDEX]-data3[:,TIME_INDEX]), np.zeros_like(data1[:,TIME_INDEX])), "convergence comparisons,  time data points do not match"
    assert np.allclose((data2[:,TIME_INDEX]-data4[:,TIME_INDEX]), np.zeros_like(data2[:,TIME_INDEX])), "convergence comparisons,  time data points do not match"

    norm_r13 = np.max(abs(data1[:,TEMP_INDEX]-data3[:,TEMP_INDEX])/data1[:,TEMP_INDEX])
    norm_r24 = np.max(abs(data2[:,TEMP_INDEX]-data4[:,TEMP_INDEX])/data2[:,TEMP_INDEX])
    
    table_data = [
         ["Nr=4 dt=1e-10",norm_r13],
         ["Nr=8 dt=5e-11",norm_r24]]


    print("========================================")
    print("G0 refinement in both Nr and dt (20ev)")
    print("========================================")
    print(tabulate(table_data,headers=["run","rel. error (temperature)"]))


def g02_quasi_neutral():

    file_names=[["dat/g02_dt_1.00000000E-10_Nr_3.dat",  "dat/g02_dt_5.00000000E-11_Nr_3.dat",  "dat/g02_dt_2.50000000E-11_Nr_3.dat"],
                ["dat/g02_dt_1.00000000E-10_Nr_7.dat",  "dat/g02_dt_5.00000000E-11_Nr_7.dat",  "dat/g02_dt_2.50000000E-11_Nr_7.dat"],
                ["dat/g02_dt_1.00000000E-10_Nr_15.dat", "dat/g02_dt_5.00000000E-11_Nr_15.dat", "dat/g02_dt_2.50000000E-11_Nr_15.dat"]]


    data_nr_1x_dt1 = np.loadtxt(file_names[0][0],skiprows=1)
    data_nr_2x_dt1 = np.loadtxt(file_names[1][0],skiprows=1)
    data_nr_4x_dt1 = np.loadtxt(file_names[2][0],skiprows=1)


    data_nr_1x_dt2 = np.loadtxt(file_names[0][1],skiprows=1)
    data_nr_2x_dt2 = np.loadtxt(file_names[1][1],skiprows=1)
    data_nr_4x_dt2 = np.loadtxt(file_names[2][1],skiprows=1)

    data_nr_1x_dt4 = np.loadtxt(file_names[0][2],skiprows=1)
    data_nr_2x_dt4 = np.loadtxt(file_names[1][2],skiprows=1)
    data_nr_4x_dt4 = np.loadtxt(file_names[2][2],skiprows=1)

    plt.close()
    sk_tail = 10
    plt.plot(data_nr_1x_dt1[:-sk_tail,TIME_INDEX],data_nr_1x_dt1[:-sk_tail,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0 + G2, dt=1E-10, Nr=4")
    plt.plot(data_nr_2x_dt1[:-sk_tail,TIME_INDEX],data_nr_2x_dt1[:-sk_tail,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0 + G2, dt=1E-10, Nr=8")
    plt.plot(data_nr_4x_dt1[:-sk_tail,TIME_INDEX],data_nr_4x_dt1[:-sk_tail,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0 + G2, dt=1E-10, Nr=16")
    
    sk_tail = 10
    plt.plot(data_nr_1x_dt2[:-sk_tail,TIME_INDEX],data_nr_1x_dt2[:-sk_tail,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0 + G2, dt=5E-11, Nr=4")
    plt.plot(data_nr_2x_dt2[:-sk_tail,TIME_INDEX],data_nr_2x_dt2[:-sk_tail,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0 + G2, dt=5E-11, Nr=8")
    plt.plot(data_nr_4x_dt2[:-sk_tail,TIME_INDEX],data_nr_4x_dt2[:-sk_tail,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0 + G2, dt=5E-11, Nr=16")

    sk_tail = 10
    plt.plot(data_nr_1x_dt4[:-sk_tail,TIME_INDEX],data_nr_1x_dt4[:-sk_tail,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0 + G2, dt=2.5E-11, Nr=4")
    plt.plot(data_nr_2x_dt4[:-sk_tail,TIME_INDEX],data_nr_2x_dt4[:-sk_tail,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0 + G2, dt=2.5E-11, Nr=8")
    plt.plot(data_nr_4x_dt4[:-sk_tail,TIME_INDEX],data_nr_4x_dt4[:-sk_tail,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0 + G2, dt=2.5E-11, Nr=16")

    plt.title("G0 + G2 with quasi neutrality (Temperature)")

    #plt.plot(data_nr_1x_dt2[:,TIME_INDEX],data_nr_1x_dt2[:,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0, dt=5E-11, Nr=4")
    #plt.plot(data_nr_2x_dt2[:,TIME_INDEX],data_nr_2x_dt2[:,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0, dt=5E-11, Nr=8")
    #plt.plot(data_nr_4x_dt2[:,TIME_INDEX],data_nr_4x_dt2[:,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0, dt=5E-11, Nr=16")
    plt.grid()
    plt.xlabel("time(s)")
    plt.ylabel("temperature (eV)")
    plt.xscale("log")
    plt.legend()
    plt.show()



    plt.close()
    sk_tail = 10
    plt.plot(data_nr_1x_dt1[:-sk_tail,TIME_INDEX],data_nr_1x_dt1[:-sk_tail,MASS_INDEX],label="G0 + G2, dt=1E-10, Nr=4")
    plt.plot(data_nr_2x_dt1[:-sk_tail,TIME_INDEX],data_nr_2x_dt1[:-sk_tail,MASS_INDEX],label="G0 + G2, dt=1E-10, Nr=8")
    plt.plot(data_nr_4x_dt1[:-sk_tail,TIME_INDEX],data_nr_4x_dt1[:-sk_tail,MASS_INDEX],label="G0 + G2, dt=1E-10, Nr=16")
    
    sk_tail = 10
    plt.plot(data_nr_1x_dt2[:-sk_tail,TIME_INDEX],data_nr_1x_dt2[:-sk_tail,MASS_INDEX],label="G0 + G2, dt=5E-11, Nr=4")
    plt.plot(data_nr_2x_dt2[:-sk_tail,TIME_INDEX],data_nr_2x_dt2[:-sk_tail,MASS_INDEX],label="G0 + G2, dt=5E-11, Nr=8")
    plt.plot(data_nr_4x_dt2[:-sk_tail,TIME_INDEX],data_nr_4x_dt2[:-sk_tail,MASS_INDEX],label="G0 + G2, dt=5E-11, Nr=16")

    sk_tail = 10
    plt.plot(data_nr_1x_dt4[:-sk_tail,TIME_INDEX],data_nr_1x_dt4[:-sk_tail,MASS_INDEX],label="G0 + G2, dt=2.5E-11, Nr=4")
    plt.plot(data_nr_2x_dt4[:-sk_tail,TIME_INDEX],data_nr_2x_dt4[:-sk_tail,MASS_INDEX],label="G0 + G2, dt=2.5E-11, Nr=8")
    plt.plot(data_nr_4x_dt4[:-sk_tail,TIME_INDEX],data_nr_4x_dt4[:-sk_tail,MASS_INDEX],label="G0 + G2, dt=2.5E-11, Nr=16")

    plt.title("G0 + G2 with quasi neutrality (mass) ")

    #plt.plot(data_nr_1x_dt2[:,TIME_INDEX],data_nr_1x_dt2[:,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0, dt=5E-11, Nr=4")
    #plt.plot(data_nr_2x_dt2[:,TIME_INDEX],data_nr_2x_dt2[:,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0, dt=5E-11, Nr=8")
    #plt.plot(data_nr_4x_dt2[:,TIME_INDEX],data_nr_4x_dt2[:,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0, dt=5E-11, Nr=16")
    plt.grid()
    plt.xlabel("time(s)")
    plt.ylabel("# of electrons (m0)")
    plt.yscale("log")
    plt.legend()
    plt.show()


    plt.close()
    sk_tail = 100
    plt.plot(data_nr_1x_dt1[:-sk_tail,TIME_INDEX],abs(data_nr_4x_dt1[:-sk_tail,TEMP_INDEX]-data_nr_1x_dt1[:-sk_tail,TEMP_INDEX])/data_nr_4x_dt1[:-sk_tail,TEMP_INDEX],label="G0 + G2, dt=1E-10, Nr=4 vs. Nr=16")
    plt.plot(data_nr_1x_dt1[:-sk_tail,TIME_INDEX],abs(data_nr_4x_dt1[:-sk_tail,TEMP_INDEX]-data_nr_2x_dt1[:-sk_tail,TEMP_INDEX])/data_nr_4x_dt1[:-sk_tail,TEMP_INDEX],label="G0 + G2, dt=1E-10, Nr=8 vs. Nr=16")
    
    sk_tail = 100
    plt.plot(data_nr_1x_dt2[:-sk_tail,TIME_INDEX],abs(data_nr_4x_dt2[:-sk_tail,TEMP_INDEX]-data_nr_1x_dt2[:-sk_tail,TEMP_INDEX])/data_nr_4x_dt2[:-sk_tail,TEMP_INDEX],label="G0 + G2, dt=5E-11, Nr=4 vs. Nr=16")
    plt.plot(data_nr_1x_dt2[:-sk_tail,TIME_INDEX],abs(data_nr_4x_dt2[:-sk_tail,TEMP_INDEX]-data_nr_2x_dt2[:-sk_tail,TEMP_INDEX])/data_nr_4x_dt2[:-sk_tail,TEMP_INDEX],label="G0 + G2, dt=5E-11, Nr=8 vs. Nr=16")

    sk_tail = 10
    plt.plot(data_nr_1x_dt4[:-sk_tail,TIME_INDEX],abs(data_nr_4x_dt4[:-sk_tail,TEMP_INDEX]-data_nr_1x_dt4[:-sk_tail,TEMP_INDEX])/data_nr_4x_dt4[:-sk_tail,TEMP_INDEX],label="G0 + G2, dt=2.5E-11, Nr=4 vs. Nr=16")
    plt.plot(data_nr_1x_dt4[:-sk_tail,TIME_INDEX],abs(data_nr_4x_dt4[:-sk_tail,TEMP_INDEX]-data_nr_2x_dt4[:-sk_tail,TEMP_INDEX])/data_nr_4x_dt4[:-sk_tail,TEMP_INDEX],label="G0 + G2, dt=2.5E-11, Nr=8 vs. Nr=16")

    norm_1dt_1x_4x = np.max(abs(data_nr_4x_dt1[:,TEMP_INDEX]-data_nr_1x_dt1[:,TEMP_INDEX])/data_nr_4x_dt1[:,TEMP_INDEX])
    norm_1dt_2x_4x = np.max(abs(data_nr_4x_dt1[:,TEMP_INDEX]-data_nr_2x_dt1[:,TEMP_INDEX])/data_nr_4x_dt1[:,TEMP_INDEX])

    norm_2dt_1x_4x = np.max(abs(data_nr_4x_dt2[:,TEMP_INDEX]-data_nr_1x_dt2[:,TEMP_INDEX])/data_nr_4x_dt2[:,TEMP_INDEX])
    norm_2dt_2x_4x = np.max(abs(data_nr_4x_dt2[:,TEMP_INDEX]-data_nr_2x_dt2[:,TEMP_INDEX])/data_nr_4x_dt2[:,TEMP_INDEX])

    norm_4dt_1x_4x = np.max(abs(data_nr_4x_dt4[:,TEMP_INDEX]-data_nr_1x_dt4[:,TEMP_INDEX])/data_nr_4x_dt4[:,TEMP_INDEX])
    norm_4dt_2x_4x = np.max(abs(data_nr_4x_dt4[:,TEMP_INDEX]-data_nr_2x_dt4[:,TEMP_INDEX])/data_nr_4x_dt4[:,TEMP_INDEX])

    table_data = [
        [1e-10, norm_1dt_1x_4x, norm_1dt_2x_4x],
        [5e-11, norm_2dt_1x_4x, norm_2dt_2x_4x],
        [2.5e-11, norm_4dt_1x_4x, norm_4dt_2x_4x]]

    print("========================================")
    print("      G0 + G2 refinement in Nr          ")
    print("========================================")
    print(tabulate(table_data,headers=["dt","Nr=4 vs Nr=16","Nr=8 vs Nr=16"]))


    

    plt.title("G0 + G2 with quasi neutrality (Nr refinement)")
    plt.xlabel("time (s)")
    plt.ylabel("relative error (temperature)")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.show()


    plt.close()
    sk_tail = 50
    data1 = data_nr_1x_dt4[range(0,len(data_nr_1x_dt4[:-sk_tail,TIME_INDEX]),4),TEMP_INDEX]
    data2 = data_nr_1x_dt4[range(0,len(data_nr_1x_dt4[:-sk_tail,TIME_INDEX]),2),TEMP_INDEX]

    data3 = data_nr_2x_dt4[range(0,len(data_nr_2x_dt4[:-sk_tail,TIME_INDEX]),4),TEMP_INDEX]
    data4 = data_nr_2x_dt4[range(0,len(data_nr_2x_dt4[:-sk_tail,TIME_INDEX]),2),TEMP_INDEX]

    data5 = data_nr_4x_dt4[range(0,len(data_nr_4x_dt4[:-sk_tail,TIME_INDEX]),4),TEMP_INDEX]
    data6 = data_nr_4x_dt4[range(0,len(data_nr_4x_dt4[:-sk_tail,TIME_INDEX]),2),TEMP_INDEX]

    assert np.allclose((data_nr_1x_dt1[:len(data1),TIME_INDEX] - data_nr_1x_dt4[range(0,len(data_nr_1x_dt4[:-sk_tail,TIME_INDEX]),4),TIME_INDEX]), np.zeros_like(data1)), "times does not match"
    assert np.allclose((data_nr_1x_dt2[:len(data2),TIME_INDEX] - data_nr_1x_dt4[range(0,len(data_nr_1x_dt4[:-sk_tail,TIME_INDEX]),2),TIME_INDEX]), np.zeros_like(data2)), "times does not match"

    assert np.allclose((data_nr_2x_dt1[:len(data3),TIME_INDEX] - data_nr_2x_dt4[range(0,len(data_nr_2x_dt4[:-sk_tail,TIME_INDEX]),4),TIME_INDEX]), np.zeros_like(data3)), "times does not match"
    assert np.allclose((data_nr_2x_dt2[:len(data4),TIME_INDEX] - data_nr_2x_dt4[range(0,len(data_nr_2x_dt4[:-sk_tail,TIME_INDEX]),2),TIME_INDEX]), np.zeros_like(data4)), "times does not match"

    assert np.allclose((data_nr_4x_dt1[:len(data5),TIME_INDEX] - data_nr_4x_dt4[range(0,len(data_nr_4x_dt4[:-sk_tail,TIME_INDEX]),4),TIME_INDEX]), np.zeros_like(data5)), "times does not match"
    assert np.allclose((data_nr_4x_dt2[:len(data6),TIME_INDEX] - data_nr_4x_dt4[range(0,len(data_nr_4x_dt4[:-sk_tail,TIME_INDEX]),2),TIME_INDEX]), np.zeros_like(data6)), "times does not match"

    plt.plot(data_nr_1x_dt1[:len(data1),TIME_INDEX], abs(data1-data_nr_1x_dt1[:len(data1),TEMP_INDEX])/data1,label="G0 + G2, dt=1E-10 error w.r.t dt= 2.5E-11, Nr=4")
    plt.plot(data_nr_1x_dt2[:len(data2),TIME_INDEX], abs(data2-data_nr_1x_dt2[:len(data2),TEMP_INDEX])/data2,label="G0 + G2, dt=5E-11 error w.r.t dt= 2.5E-11, Nr=4")

    plt.plot(data_nr_2x_dt1[:len(data3),TIME_INDEX], abs(data3-data_nr_2x_dt1[:len(data3),TEMP_INDEX])/data3,label="G0 + G2, dt=1E-10 error w.r.t dt= 2.5E-11, Nr=8")
    plt.plot(data_nr_2x_dt2[:len(data4),TIME_INDEX], abs(data4-data_nr_2x_dt2[:len(data4),TEMP_INDEX])/data4,label="G0 + G2, dt=5E-11 error w.r.t dt= 2.5E-11, Nr=8")

    plt.plot(data_nr_4x_dt1[:len(data5),TIME_INDEX], abs(data5-data_nr_4x_dt1[:len(data5),TEMP_INDEX])/data5,label="G0 + G2, dt=1E-10 error w.r.t dt= 2.5E-11, Nr=16")
    plt.plot(data_nr_4x_dt2[:len(data6),TIME_INDEX], abs(data6-data_nr_4x_dt2[:len(data6),TEMP_INDEX])/data6,label="G0 + G2, dt=5E-11 error w.r.t dt= 2.5E-11, Nr=16")

    norm_nr4_dt1_dt4 = np.max(abs(data1-data_nr_1x_dt1[:len(data1),TEMP_INDEX])/data1)
    norm_nr4_dt2_dt4 = np.max(abs(data2-data_nr_1x_dt2[:len(data2),TEMP_INDEX])/data2)

    norm_nr8_dt1_dt4 = np.max(abs(data3-data_nr_2x_dt1[:len(data3),TEMP_INDEX])/data3)
    norm_nr8_dt2_dt4 = np.max(abs(data4-data_nr_2x_dt2[:len(data4),TEMP_INDEX])/data4)

    norm_nr16_dt1_dt4 = np.max(abs(data5-data_nr_4x_dt1[:len(data5),TEMP_INDEX])/data5)
    norm_nr16_dt2_dt4 = np.max(abs(data6-data_nr_4x_dt2[:len(data6),TEMP_INDEX])/data6)

    table_data = [
         [4,norm_nr4_dt1_dt4,norm_nr4_dt2_dt4],
         [8,norm_nr8_dt1_dt4,norm_nr8_dt2_dt4],
        [16,norm_nr16_dt1_dt4,norm_nr16_dt2_dt4]]
    
    print("========================================")
    print("      G0 + G2 refinement in dt          ")
    print("========================================")
    print(tabulate(table_data,headers=["Nr","1e-10 vs. 2.5E-11","5e-11 vs. 2.5E-11"]))

    plt.title("G0 + G2 with quasi neutrality (time refinement)")
    plt.xlabel("time (s)")
    plt.ylabel("relative error (temperature)")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.show()

    plt.close()


    sk_tail = 1
    data1   = data_nr_4x_dt4[range(0,len(data_nr_4x_dt4[:-sk_tail,TIME_INDEX]),4),:]
    data2   = data_nr_4x_dt4[range(0,len(data_nr_4x_dt4[:-sk_tail,TIME_INDEX]),2),:]

    data3   = data_nr_1x_dt1[0:len(data1[:,TEMP_INDEX]),:]
    data4   = data_nr_2x_dt2[0:len(data2[:,TEMP_INDEX]),:]

    assert np.allclose((data1[:,TIME_INDEX]-data3[:,TIME_INDEX]), np.zeros_like(data1[:,TIME_INDEX])), "convergence comparisons,  time data points do not match"
    assert np.allclose((data2[:,TIME_INDEX]-data4[:,TIME_INDEX]), np.zeros_like(data2[:,TIME_INDEX])), "convergence comparisons,  time data points do not match"

    norm_r13 = np.max(abs(data1[:,TEMP_INDEX]-data3[:,TEMP_INDEX])/data1[:,TEMP_INDEX])
    norm_r24 = np.max(abs(data2[:,TEMP_INDEX]-data4[:,TEMP_INDEX])/data2[:,TEMP_INDEX])
    
    table_data = [
         ["Nr=4 dt=1e-10",norm_r13],
         ["Nr=8 dt=5e-11",norm_r24]]


    print("========================================")
    print("G0 + G2 refinement in both Nr and dt (20ev)")
    print("========================================")
    print(tabulate(table_data,headers=["run","rel. error (temperature)"]))


def g02_quasi_neutral_1ev():

    file_names=[["dat_1ev/g02_dt_1.00000000E-10_Nr_3.dat",  "dat_1ev/g02_dt_5.00000000E-11_Nr_3.dat",  "dat_1ev/g02_dt_2.50000000E-11_Nr_3.dat"],
                ["dat_1ev/g02_dt_1.00000000E-10_Nr_7.dat",  "dat_1ev/g02_dt_5.00000000E-11_Nr_7.dat",  "dat_1ev/g02_dt_2.50000000E-11_Nr_7.dat"],
                ["dat_1ev/g02_dt_1.00000000E-10_Nr_15.dat", "dat_1ev/g02_dt_5.00000000E-11_Nr_15.dat", "dat_1ev/g02_dt_2.50000000E-11_Nr_15.dat"]]


    data_nr_1x_dt1 = np.loadtxt(file_names[0][0],skiprows=1)
    data_nr_2x_dt1 = np.loadtxt(file_names[1][0],skiprows=1)
    data_nr_4x_dt1 = np.loadtxt(file_names[2][0],skiprows=1)


    data_nr_1x_dt2 = np.loadtxt(file_names[0][1],skiprows=1)
    data_nr_2x_dt2 = np.loadtxt(file_names[1][1],skiprows=1)
    data_nr_4x_dt2 = np.loadtxt(file_names[2][1],skiprows=1)

    data_nr_1x_dt4 = np.loadtxt(file_names[0][2],skiprows=1)
    data_nr_2x_dt4 = np.loadtxt(file_names[1][2],skiprows=1)
    data_nr_4x_dt4 = np.loadtxt(file_names[2][2],skiprows=1)

    plt.close()
    sk_tail = 10
    plt.plot(data_nr_1x_dt1[:-sk_tail,TIME_INDEX],data_nr_1x_dt1[:-sk_tail,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0 + G2, dt=1E-10, Nr=4")
    plt.plot(data_nr_2x_dt1[:-sk_tail,TIME_INDEX],data_nr_2x_dt1[:-sk_tail,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0 + G2, dt=1E-10, Nr=8")
    plt.plot(data_nr_4x_dt1[:-sk_tail,TIME_INDEX],data_nr_4x_dt1[:-sk_tail,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0 + G2, dt=1E-10, Nr=16")
    
    sk_tail = 10
    plt.plot(data_nr_1x_dt2[:-sk_tail,TIME_INDEX],data_nr_1x_dt2[:-sk_tail,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0 + G2, dt=5E-11, Nr=4")
    plt.plot(data_nr_2x_dt2[:-sk_tail,TIME_INDEX],data_nr_2x_dt2[:-sk_tail,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0 + G2, dt=5E-11, Nr=8")
    plt.plot(data_nr_4x_dt2[:-sk_tail,TIME_INDEX],data_nr_4x_dt2[:-sk_tail,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0 + G2, dt=5E-11, Nr=16")

    sk_tail = 10
    plt.plot(data_nr_1x_dt4[:-sk_tail,TIME_INDEX],data_nr_1x_dt4[:-sk_tail,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0 + G2, dt=2.5E-11, Nr=4")
    plt.plot(data_nr_2x_dt4[:-sk_tail,TIME_INDEX],data_nr_2x_dt4[:-sk_tail,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0 + G2, dt=2.5E-11, Nr=8")
    plt.plot(data_nr_4x_dt4[:-sk_tail,TIME_INDEX],data_nr_4x_dt4[:-sk_tail,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0 + G2, dt=2.5E-11, Nr=16")

    plt.title("G0 + G2 with quasi neutrality (Temperature)")

    #plt.plot(data_nr_1x_dt2[:,TIME_INDEX],data_nr_1x_dt2[:,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0, dt=5E-11, Nr=4")
    #plt.plot(data_nr_2x_dt2[:,TIME_INDEX],data_nr_2x_dt2[:,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0, dt=5E-11, Nr=8")
    #plt.plot(data_nr_4x_dt2[:,TIME_INDEX],data_nr_4x_dt2[:,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0, dt=5E-11, Nr=16")
    plt.grid()
    plt.xlabel("time(s)")
    plt.ylabel("temperature (eV)")
    plt.xscale("log")
    plt.legend()
    plt.show()



    plt.close()
    sk_tail = 10
    plt.plot(data_nr_1x_dt1[:-sk_tail,TIME_INDEX],data_nr_1x_dt1[:-sk_tail,MASS_INDEX],label="G0 + G2, dt=1E-10, Nr=4")
    plt.plot(data_nr_2x_dt1[:-sk_tail,TIME_INDEX],data_nr_2x_dt1[:-sk_tail,MASS_INDEX],label="G0 + G2, dt=1E-10, Nr=8")
    plt.plot(data_nr_4x_dt1[:-sk_tail,TIME_INDEX],data_nr_4x_dt1[:-sk_tail,MASS_INDEX],label="G0 + G2, dt=1E-10, Nr=16")
    
    sk_tail = 10
    plt.plot(data_nr_1x_dt2[:-sk_tail,TIME_INDEX],data_nr_1x_dt2[:-sk_tail,MASS_INDEX],label="G0 + G2, dt=5E-11, Nr=4")
    plt.plot(data_nr_2x_dt2[:-sk_tail,TIME_INDEX],data_nr_2x_dt2[:-sk_tail,MASS_INDEX],label="G0 + G2, dt=5E-11, Nr=8")
    plt.plot(data_nr_4x_dt2[:-sk_tail,TIME_INDEX],data_nr_4x_dt2[:-sk_tail,MASS_INDEX],label="G0 + G2, dt=5E-11, Nr=16")

    sk_tail = 10
    plt.plot(data_nr_1x_dt4[:-sk_tail,TIME_INDEX],data_nr_1x_dt4[:-sk_tail,MASS_INDEX],label="G0 + G2, dt=2.5E-11, Nr=4")
    plt.plot(data_nr_2x_dt4[:-sk_tail,TIME_INDEX],data_nr_2x_dt4[:-sk_tail,MASS_INDEX],label="G0 + G2, dt=2.5E-11, Nr=8")
    plt.plot(data_nr_4x_dt4[:-sk_tail,TIME_INDEX],data_nr_4x_dt4[:-sk_tail,MASS_INDEX],label="G0 + G2, dt=2.5E-11, Nr=16")

    plt.title("G0 + G2 with quasi neutrality (mass) ")

    #plt.plot(data_nr_1x_dt2[:,TIME_INDEX],data_nr_1x_dt2[:,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0, dt=5E-11, Nr=4")
    #plt.plot(data_nr_2x_dt2[:,TIME_INDEX],data_nr_2x_dt2[:,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0, dt=5E-11, Nr=8")
    #plt.plot(data_nr_4x_dt2[:,TIME_INDEX],data_nr_4x_dt2[:,TEMP_INDEX]*collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT,label="G0, dt=5E-11, Nr=16")
    plt.grid()
    plt.xlabel("time(s)")
    plt.ylabel("# of electrons (m0)")
    plt.yscale("log")
    plt.legend()
    plt.show()


    plt.close()
    sk_tail = 1
    plt.plot(data_nr_1x_dt1[:-sk_tail,TIME_INDEX],abs(data_nr_4x_dt1[:-sk_tail,TEMP_INDEX]-data_nr_1x_dt1[:-sk_tail,TEMP_INDEX])/data_nr_4x_dt1[:-sk_tail,TEMP_INDEX],label="G0 + G2, dt=1E-10, Nr=4 vs. Nr=16")
    plt.plot(data_nr_1x_dt1[:-sk_tail,TIME_INDEX],abs(data_nr_4x_dt1[:-sk_tail,TEMP_INDEX]-data_nr_2x_dt1[:-sk_tail,TEMP_INDEX])/data_nr_4x_dt1[:-sk_tail,TEMP_INDEX],label="G0 + G2, dt=1E-10, Nr=8 vs. Nr=16")
    
    sk_tail = 1
    plt.plot(data_nr_1x_dt2[:-sk_tail,TIME_INDEX],abs(data_nr_4x_dt2[:-sk_tail,TEMP_INDEX]-data_nr_1x_dt2[:-sk_tail,TEMP_INDEX])/data_nr_4x_dt2[:-sk_tail,TEMP_INDEX],label="G0 + G2, dt=5E-11, Nr=4 vs. Nr=16")
    plt.plot(data_nr_1x_dt2[:-sk_tail,TIME_INDEX],abs(data_nr_4x_dt2[:-sk_tail,TEMP_INDEX]-data_nr_2x_dt2[:-sk_tail,TEMP_INDEX])/data_nr_4x_dt2[:-sk_tail,TEMP_INDEX],label="G0 + G2, dt=5E-11, Nr=8 vs. Nr=16")

    sk_tail = 1
    plt.plot(data_nr_1x_dt4[:-sk_tail,TIME_INDEX],abs(data_nr_4x_dt4[:-sk_tail,TEMP_INDEX]-data_nr_1x_dt4[:-sk_tail,TEMP_INDEX])/data_nr_4x_dt4[:-sk_tail,TEMP_INDEX],label="G0 + G2, dt=2.5E-11, Nr=4 vs. Nr=16")
    plt.plot(data_nr_1x_dt4[:-sk_tail,TIME_INDEX],abs(data_nr_4x_dt4[:-sk_tail,TEMP_INDEX]-data_nr_2x_dt4[:-sk_tail,TEMP_INDEX])/data_nr_4x_dt4[:-sk_tail,TEMP_INDEX],label="G0 + G2, dt=2.5E-11, Nr=8 vs. Nr=16")

    norm_1dt_1x_4x = np.max(abs(data_nr_4x_dt1[:,TEMP_INDEX]-data_nr_1x_dt1[:,TEMP_INDEX])/data_nr_4x_dt1[:,TEMP_INDEX])
    norm_1dt_2x_4x = np.max(abs(data_nr_4x_dt1[:,TEMP_INDEX]-data_nr_2x_dt1[:,TEMP_INDEX])/data_nr_4x_dt1[:,TEMP_INDEX])

    norm_2dt_1x_4x = np.max(abs(data_nr_4x_dt2[:,TEMP_INDEX]-data_nr_1x_dt2[:,TEMP_INDEX])/data_nr_4x_dt2[:,TEMP_INDEX])
    norm_2dt_2x_4x = np.max(abs(data_nr_4x_dt2[:,TEMP_INDEX]-data_nr_2x_dt2[:,TEMP_INDEX])/data_nr_4x_dt2[:,TEMP_INDEX])

    norm_4dt_1x_4x = np.max(abs(data_nr_4x_dt4[:,TEMP_INDEX]-data_nr_1x_dt4[:,TEMP_INDEX])/data_nr_4x_dt4[:,TEMP_INDEX])
    norm_4dt_2x_4x = np.max(abs(data_nr_4x_dt4[:,TEMP_INDEX]-data_nr_2x_dt4[:,TEMP_INDEX])/data_nr_4x_dt4[:,TEMP_INDEX])

    table_data = [
        [1e-10, norm_1dt_1x_4x, norm_1dt_2x_4x],
        [5e-11, norm_2dt_1x_4x, norm_2dt_2x_4x],
        [2.5e-11, norm_4dt_1x_4x, norm_4dt_2x_4x]]

    print("========================================")
    print("      G0 + G2 refinement in Nr          ")
    print("========================================")
    print(tabulate(table_data,headers=["dt","Nr=4 vs Nr=16","Nr=8 vs Nr=16"]))


    

    plt.title("G0 + G2 with quasi neutrality (Nr refinement)")
    plt.xlabel("time (s)")
    plt.ylabel("relative error (temperature)")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.show()


    plt.close()
    sk_tail = 10
    data1 = data_nr_1x_dt4[range(0,len(data_nr_1x_dt4[:-sk_tail,TIME_INDEX]),4),TEMP_INDEX]
    data2 = data_nr_1x_dt4[range(0,len(data_nr_1x_dt4[:-sk_tail,TIME_INDEX]),2),TEMP_INDEX]

    data3 = data_nr_2x_dt4[range(0,len(data_nr_2x_dt4[:-sk_tail,TIME_INDEX]),4),TEMP_INDEX]
    data4 = data_nr_2x_dt4[range(0,len(data_nr_2x_dt4[:-sk_tail,TIME_INDEX]),2),TEMP_INDEX]

    data5 = data_nr_4x_dt4[range(0,len(data_nr_4x_dt4[:-sk_tail,TIME_INDEX]),4),TEMP_INDEX]
    data6 = data_nr_4x_dt4[range(0,len(data_nr_4x_dt4[:-sk_tail,TIME_INDEX]),2),TEMP_INDEX]

    assert np.allclose((data_nr_1x_dt1[:len(data1),TIME_INDEX] - data_nr_1x_dt4[range(0,len(data_nr_1x_dt4[:-sk_tail,TIME_INDEX]),4),TIME_INDEX]), np.zeros_like(data1)), "times does not match"
    assert np.allclose((data_nr_1x_dt2[:len(data2),TIME_INDEX] - data_nr_1x_dt4[range(0,len(data_nr_1x_dt4[:-sk_tail,TIME_INDEX]),2),TIME_INDEX]), np.zeros_like(data2)), "times does not match"

    assert np.allclose((data_nr_2x_dt1[:len(data3),TIME_INDEX] - data_nr_2x_dt4[range(0,len(data_nr_2x_dt4[:-sk_tail,TIME_INDEX]),4),TIME_INDEX]), np.zeros_like(data3)), "times does not match"
    assert np.allclose((data_nr_2x_dt2[:len(data4),TIME_INDEX] - data_nr_2x_dt4[range(0,len(data_nr_2x_dt4[:-sk_tail,TIME_INDEX]),2),TIME_INDEX]), np.zeros_like(data4)), "times does not match"

    assert np.allclose((data_nr_4x_dt1[:len(data5),TIME_INDEX] - data_nr_4x_dt4[range(0,len(data_nr_4x_dt4[:-sk_tail,TIME_INDEX]),4),TIME_INDEX]), np.zeros_like(data5)), "times does not match"
    assert np.allclose((data_nr_4x_dt2[:len(data6),TIME_INDEX] - data_nr_4x_dt4[range(0,len(data_nr_4x_dt4[:-sk_tail,TIME_INDEX]),2),TIME_INDEX]), np.zeros_like(data6)), "times does not match"

    plt.plot(data_nr_1x_dt1[:len(data1),TIME_INDEX], abs(data1-data_nr_1x_dt1[:len(data1),TEMP_INDEX])/data1,label="G0 + G2, dt=1E-10 error w.r.t dt= 2.5E-11, Nr=4")
    plt.plot(data_nr_1x_dt2[:len(data2),TIME_INDEX], abs(data2-data_nr_1x_dt2[:len(data2),TEMP_INDEX])/data2,label="G0 + G2, dt=5E-11 error w.r.t dt= 2.5E-11, Nr=4")

    plt.plot(data_nr_2x_dt1[:len(data3),TIME_INDEX], abs(data3-data_nr_2x_dt1[:len(data3),TEMP_INDEX])/data3,label="G0 + G2, dt=1E-10 error w.r.t dt= 2.5E-11, Nr=8")
    plt.plot(data_nr_2x_dt2[:len(data4),TIME_INDEX], abs(data4-data_nr_2x_dt2[:len(data4),TEMP_INDEX])/data4,label="G0 + G2, dt=5E-11 error w.r.t dt= 2.5E-11, Nr=8")

    plt.plot(data_nr_4x_dt1[:len(data5),TIME_INDEX], abs(data5-data_nr_4x_dt1[:len(data5),TEMP_INDEX])/data5,label="G0 + G2, dt=1E-10 error w.r.t dt= 2.5E-11, Nr=16")
    plt.plot(data_nr_4x_dt2[:len(data6),TIME_INDEX], abs(data6-data_nr_4x_dt2[:len(data6),TEMP_INDEX])/data6,label="G0 + G2, dt=5E-11 error w.r.t dt= 2.5E-11, Nr=16")

    norm_nr4_dt1_dt4 = np.max(abs(data1-data_nr_1x_dt1[:len(data1),TEMP_INDEX])/data1)
    norm_nr4_dt2_dt4 = np.max(abs(data2-data_nr_1x_dt2[:len(data2),TEMP_INDEX])/data2)

    norm_nr8_dt1_dt4 = np.max(abs(data3-data_nr_2x_dt1[:len(data3),TEMP_INDEX])/data3)
    norm_nr8_dt2_dt4 = np.max(abs(data4-data_nr_2x_dt2[:len(data4),TEMP_INDEX])/data4)

    norm_nr16_dt1_dt4 = np.max(abs(data5-data_nr_4x_dt1[:len(data5),TEMP_INDEX])/data5)
    norm_nr16_dt2_dt4 = np.max(abs(data6-data_nr_4x_dt2[:len(data6),TEMP_INDEX])/data6)

    table_data = [
         [4,norm_nr4_dt1_dt4,norm_nr4_dt2_dt4],
         [8,norm_nr8_dt1_dt4,norm_nr8_dt2_dt4],
        [16,norm_nr16_dt1_dt4,norm_nr16_dt2_dt4]]
    
    print("========================================")
    print("      G0 + G2 refinement in dt          ")
    print("========================================")
    print(tabulate(table_data,headers=["Nr","1e-10 vs. 2.5E-11","5e-11 vs. 2.5E-11"]))

    plt.title("G0 + G2 with quasi neutrality (time refinement)")
    plt.xlabel("time (s)")
    plt.ylabel("relative error (temperature)")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.show()


    plt.close()
    sk_tail = 1
    data1   = data_nr_4x_dt4[range(0,len(data_nr_4x_dt4[:-sk_tail,TIME_INDEX]),4),:]
    data2   = data_nr_4x_dt4[range(0,len(data_nr_4x_dt4[:-sk_tail,TIME_INDEX]),2),:]

    data3   = data_nr_1x_dt1[0:len(data1[:,TEMP_INDEX]),:]
    data4   = data_nr_2x_dt2[0:len(data2[:,TEMP_INDEX]),:]

    assert np.allclose((data1[:,TIME_INDEX]-data3[:,TIME_INDEX]), np.zeros_like(data1[:,TIME_INDEX])), "convergence comparisons,  time data points do not match"
    assert np.allclose((data2[:,TIME_INDEX]-data4[:,TIME_INDEX]), np.zeros_like(data2[:,TIME_INDEX])), "convergence comparisons,  time data points do not match"

    norm_r13 = np.max(abs(data1[:,TEMP_INDEX]-data3[:,TEMP_INDEX])/data1[:,TEMP_INDEX])
    norm_r24 = np.max(abs(data2[:,TEMP_INDEX]-data4[:,TEMP_INDEX])/data2[:,TEMP_INDEX])
    
    table_data = [
         ["Nr=4 dt=1e-10",norm_r13],
         ["Nr=8 dt=5e-11",norm_r24]]


    print("========================================")
    print("  G0 + G2 refinement in both Nr and dt (1ev)  ")
    print("========================================")
    print(tabulate(table_data,headers=["run","rel. error (temperature)"]))

    #data2   = data_nr_1x_dt4[range(0,len(data_nr_1x_dt4[:-sk_tail,TIME_INDEX]),2),TEMP_INDEX]

    # data3   = data_nr_2x_dt4[range(0,len(data_nr_2x_dt4[:-sk_tail,TIME_INDEX]),4),TEMP_INDEX]
    # data4   = data_nr_2x_dt4[range(0,len(data_nr_2x_dt4[:-sk_tail,TIME_INDEX]),2),TEMP_INDEX]

    # data5   = data_nr_4x_dt4[range(0,len(data_nr_4x_dt4[:-sk_tail,TIME_INDEX]),4),TEMP_INDEX]
    # data6   = data_nr_4x_dt4[range(0,len(data_nr_4x_dt4[:-sk_tail,TIME_INDEX]),2),TEMP_INDEX]


def g0_rk4_1ev_convergence():

    file_names=[[DATA_FOLDER_NAME +"/g0_dt_1.00000000E-10_Nr_3.dat",  DATA_FOLDER_NAME+"/g0_dt_5.00000000E-11_Nr_3.dat",  DATA_FOLDER_NAME+"/g0_dt_2.50000000E-11_Nr_3.dat",  DATA_FOLDER_NAME+"/g0_dt_1.25000000E-11_Nr_3.dat"],
                [DATA_FOLDER_NAME +"/g0_dt_1.00000000E-10_Nr_7.dat",  DATA_FOLDER_NAME+"/g0_dt_5.00000000E-11_Nr_7.dat",  DATA_FOLDER_NAME+"/g0_dt_2.50000000E-11_Nr_7.dat",  DATA_FOLDER_NAME+"/g0_dt_1.25000000E-11_Nr_7.dat"],
                [DATA_FOLDER_NAME +"/g0_dt_1.00000000E-10_Nr_15.dat", DATA_FOLDER_NAME+"/g0_dt_5.00000000E-11_Nr_15.dat", DATA_FOLDER_NAME+"/g0_dt_2.50000000E-11_Nr_15.dat", DATA_FOLDER_NAME+"/g0_dt_1.25000000E-11_Nr_15.dat"],
                [DATA_FOLDER_NAME +"/g0_dt_1.00000000E-10_Nr_31.dat", DATA_FOLDER_NAME+"/g0_dt_5.00000000E-11_Nr_31.dat", DATA_FOLDER_NAME+"/g0_dt_2.50000000E-11_Nr_31.dat", DATA_FOLDER_NAME+"/g0_dt_1.25000000E-11_Nr_31.dat"]]

    data = list()
    for t_res in file_names:
        tmp=list()
        for fname in t_res:
            tmp.append(np.loadtxt(fname,skiprows=1))
        data.append(tmp)

    highest_res = data[3][3]
    hr_sampled  = [ highest_res[range(0,len(highest_res[:,TIME_INDEX]),8),:],
                    highest_res[range(0,len(highest_res[:,TIME_INDEX]),4),:],
                    highest_res[range(0,len(highest_res[:,TIME_INDEX]),2),:],
                    highest_res[range(0,len(highest_res[:,TIME_INDEX]),1),:]]
    
    error_mat = np.zeros((4,4))
    for row in range(0,4):
        for col in range(0,4):
            # check if time points match first. 
            data_base = data[row][col][0:len(hr_sampled[col][:,TIME_INDEX]),:]
            assert np.allclose( (data_base[:,TIME_INDEX] - hr_sampled[col][:,TIME_INDEX]) , np.zeros_like(hr_sampled[col][:,TIME_INDEX])) , "convergence test time sample points does not match"
            error_mat[row,col] = np.max(abs(data_base[:,TEMP_INDEX] - hr_sampled[col][:,TEMP_INDEX])/hr_sampled[col][:,TEMP_INDEX])
    
    #print(error_mat)
    table_data = [
        ["3" , error_mat[0,0], error_mat[0,1], error_mat[0,2], error_mat[0,3]],
        ["7" , error_mat[1,0], error_mat[1,1], error_mat[1,2], error_mat[1,3]],
        ["15", error_mat[2,0], error_mat[2,1], error_mat[2,2], error_mat[2,3]],
        ["31", error_mat[3,0], error_mat[3,1], error_mat[3,2], error_mat[3,3]],
    ]
    print(tabulate(table_data,headers=["Nr","dt=1E-10","dt=5E-11","dt=2.5E-11","dt=1.25E-11"]))
    
    pl_lable = ["Nr=3, dt=1E-10", "Nr=7, dt=5E-11", "Nr=15, dt=2.5E-11"]
    for row,col in ((0,0),(1,1),(2,2)):
        data_base = data[row][col][0:len(hr_sampled[col][:,TIME_INDEX]),:]
        plt.plot(hr_sampled[col][:,TIME_INDEX], abs(data_base[:,TEMP_INDEX] - hr_sampled[col][:,TEMP_INDEX])/hr_sampled[col][:,TEMP_INDEX],label=pl_lable[row])

    plt.xlabel(r"$time(s) \rightarrow$")
    plt.ylabel(r"$relative \ error (temp)\rightarrow $")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.show()

    plt.close()

    pl_lable = ["Nr=3, dt=1E-10", "Nr=7, dt=5E-11", "Nr=15, dt=2.5E-11", "Nr=31, dt=1.25E-11"]
    for row,col in ((0,0),(1,1),(2,2),(3,3)):
        data_base = data[row][col][0:len(hr_sampled[col][:,TIME_INDEX]),:]
        plt.plot(hr_sampled[col][:,TIME_INDEX], spec_tail(data_base),label=pl_lable[row])
    
    plt.yscale("log")
    plt.grid()
    plt.xlabel(r"$time(s) \rightarrow$")
    plt.ylabel(r"$|h_{tail}|/ |h| \rightarrow $")
    plt.legend()
    plt.show()


    # plt.close()
    # MNE  = data[0][0][0,MASS_INDEX]
    # MT1  = data[0][0][0,TEMP_INDEX]
    # VTH1             = collisions.electron_thermal_velocity(MT1)
    # NR=[3,7,15,31]
    # pl_lable = ["Nr=3, dt=1E-10", "Nr=7, dt=5E-11", "Nr=15, dt=2.5E-11", "Nr=31, dt=1.25E-11"]
    # cf_sp_base   = colOpSp.CollisionOpSP(3,NR[-1])
    # spec_base    = cf_sp_base._spec 
    # for row in range(4):
    #     cf_sp    = colOpSp.CollisionOpSP(3,NR[row])
    #     spec_sp  = cf_sp._spec
    #     for col in range(4):
    #         data_base = data[row][col][0:len(hr_sampled[col][:,TIME_INDEX]),:]
    #         error_mat[row,col]=0.0
    #         num_sample_pts = 10
    #         step_sz        = data_base.shape[0] // num_sample_pts
    #         tt = np.zeros(num_sample_pts)
    #         er = np.zeros(num_sample_pts)
            
    #         for ii,t in enumerate(range(0,data_base.shape[0],step_sz)):
    #             grid_pts         = int(1e5)
    #             grid_bds         = (0.0,8)
    #             MNE1             = data_base[t,MASS_INDEX]
    #             VTH1             = collisions.electron_thermal_velocity(data_base[t,TEMP_INDEX]) 
    #             MW1              = BEUtils.get_maxwellian_3d(VTH1 ,MNE1)
    #             vr1              = np.linspace(grid_bds[0]*VTH1,grid_bds[1]*VTH1,grid_pts)
                

    #             v1               = vr1/VTH1
    #             Mr1              = MW1(v1)
    #             Pr1              = spec_sp.Vq_r(v1).transpose()
    #             fv1              = Mr1 * (np.dot(Pr1,data_base[t,C000_INDEX:]).reshape(-1))


    #             MNE2             = hr_sampled[col][t,MASS_INDEX]
    #             VTH2             = collisions.electron_thermal_velocity(hr_sampled[col][t,TEMP_INDEX]) 
    #             MW2              = BEUtils.get_maxwellian_3d(VTH2 ,MNE2)
    #             v2               = vr1/VTH2
    #             Mr2              = MW2(v2)
    #             Pr2              = spec_base.Vq_r(v2).transpose()
    #             fv2              = Mr2 * (np.dot(Pr2,hr_sampled[col][t,C000_INDEX:]).reshape(-1))

    #             # plt.plot(v1,fv1,label="fv1")
    #             # plt.plot(v2,fv2,label="fv2")
    #             # plt.legend()
    #             # #plt.yscale("log")
    #             # plt.grid()
    #             # plt.show()
    #             # plt.close()
    #             # print("DT: %.16E VTH1 : %.16E MNE1: %.16E VTH2 : %.16E MNE2: %.16E |fv1-fv2| %.8E " %(abs(data_base[t,TIME_INDEX]-hr_sampled[col][t,TIME_INDEX]),VTH1, MNE1,VTH2, MNE2,np.linalg.norm(fv2-fv1)/np.linalg.norm(fv2)))
    #             # print(np.linalg.norm(fv2-fv1)/np.linalg.norm(fv2))
    #             assert np.allclose( (data_base[t,TIME_INDEX] - hr_sampled[col][t,TIME_INDEX]) , np.zeros_like(hr_sampled[col][t,TIME_INDEX])) , "convergence test time sample points does not match"
    #             tt[ii] = data_base[t,TIME_INDEX]
    #             er[ii] = np.linalg.norm((fv2-fv1))/np.linalg.norm(fv2)

    #         error_mat[row,col] = np.max(er)
    #         if row==col:
    #             plt.plot(tt,er,label=pl_lable[row])
    

    # plt.yscale("log")
    # plt.grid()
    # plt.legend()
    # plt.xlabel("time(s)")
    # plt.ylabel("|f2-f1|/f2")
    # plt.show()

    # table_data = [
    #     ["3" , error_mat[0,0], error_mat[0,1], error_mat[0,2], error_mat[0,3]],
    #     ["7" , error_mat[1,0], error_mat[1,1], error_mat[1,2], error_mat[1,3]],
    #     ["15", error_mat[2,0], error_mat[2,1], error_mat[2,2], error_mat[2,3]],
    #     ["31", error_mat[3,0], error_mat[3,1], error_mat[3,2], error_mat[3,3]],
    # ]
    # print(tabulate(table_data,headers=["Nr","dt=1E-10","dt=5E-11","dt=2.5E-11","dt=1.25E-11"]))


def g02_rk4_1ev_convergence():
    
    file_names=[[DATA_FOLDER_NAME+"/g02_dt_1.00000000E-10_Nr_3.dat",  DATA_FOLDER_NAME+"/g02_dt_5.00000000E-11_Nr_3.dat",  DATA_FOLDER_NAME+"/g02_dt_2.50000000E-11_Nr_3.dat",  DATA_FOLDER_NAME+"/g02_dt_1.25000000E-11_Nr_3.dat"],
                [DATA_FOLDER_NAME+"/g02_dt_1.00000000E-10_Nr_7.dat",  DATA_FOLDER_NAME+"/g02_dt_5.00000000E-11_Nr_7.dat",  DATA_FOLDER_NAME+"/g02_dt_2.50000000E-11_Nr_7.dat",  DATA_FOLDER_NAME+"/g02_dt_1.25000000E-11_Nr_7.dat"],
                [DATA_FOLDER_NAME+"/g02_dt_1.00000000E-10_Nr_15.dat", DATA_FOLDER_NAME+"/g02_dt_5.00000000E-11_Nr_15.dat", DATA_FOLDER_NAME+"/g02_dt_2.50000000E-11_Nr_15.dat", DATA_FOLDER_NAME+"/g02_dt_1.25000000E-11_Nr_15.dat"],
                [DATA_FOLDER_NAME+"/g02_dt_1.00000000E-10_Nr_31.dat", DATA_FOLDER_NAME+"/g02_dt_5.00000000E-11_Nr_31.dat", DATA_FOLDER_NAME+"/g02_dt_2.50000000E-11_Nr_31.dat", DATA_FOLDER_NAME+"/g02_dt_1.25000000E-11_Nr_31.dat"]]

    data = list()
    for t_res in file_names:
        tmp=list()
        for fname in t_res:
            tmp.append(np.loadtxt(fname,skiprows=1))
        data.append(tmp)

    highest_res = data[3][3]
    hr_sampled  = [ highest_res[range(0,len(highest_res[:,TIME_INDEX]),8),:],
                    highest_res[range(0,len(highest_res[:,TIME_INDEX]),4),:],
                    highest_res[range(0,len(highest_res[:,TIME_INDEX]),2),:],
                    highest_res[range(0,len(highest_res[:,TIME_INDEX]),1),:]]
    
    error_mat = np.zeros((4,4))
    for row in range(0,4):
        for col in range(0,4):
            # check if time points match first. 
            data_base = data[row][col][0:len(hr_sampled[col][:,TIME_INDEX]),:]
            assert np.allclose( (data_base[:,TIME_INDEX] - hr_sampled[col][:,TIME_INDEX]) , np.zeros_like(hr_sampled[col][:,TIME_INDEX])) , "convergence test time sample points does not match"
            error_mat[row,col] = np.max(abs(data_base[:,TEMP_INDEX] - hr_sampled[col][:,TEMP_INDEX])/hr_sampled[col][:,TEMP_INDEX])
    
    #print(error_mat)
    print("error in temperature")
    table_data = [
        ["3" , error_mat[0,0], error_mat[0,1], error_mat[0,2], error_mat[0,3]],
        ["7" , error_mat[1,0], error_mat[1,1], error_mat[1,2], error_mat[1,3]],
        ["15", error_mat[2,0], error_mat[2,1], error_mat[2,2], error_mat[2,3]],
        ["31", error_mat[3,0], error_mat[3,1], error_mat[3,2], error_mat[3,3]],
    ]
    print(tabulate(table_data,headers=["Nr","dt=1E-10","dt=5E-11","dt=2.5E-11","dt=1.25E-11"]))
    
    pl_lable = ["Nr=3, dt=1E-10", "Nr=7, dt=5E-11", "Nr=15, dt=2.5E-11"]
    for row,col in ((0,0),(1,1),(2,2)):
        data_base = data[row][col][0:len(hr_sampled[col][:,TIME_INDEX]),:]
        plt.plot(hr_sampled[col][:,TIME_INDEX], abs(data_base[:,TEMP_INDEX] - hr_sampled[col][:,TEMP_INDEX])/hr_sampled[col][:,TEMP_INDEX],label=pl_lable[row])

    plt.xlabel(r"$time(s) \rightarrow$")
    plt.ylabel(r"$relative \ error (temp)\rightarrow $")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.show()


    plt.close()

    for row in range(0,4):
        for col in range(0,4):
            # check if time points match first. 
            data_base = data[row][col][0:len(hr_sampled[col][:,TIME_INDEX]),:]
            assert np.allclose( (data_base[:,TIME_INDEX] - hr_sampled[col][:,TIME_INDEX]) , np.zeros_like(hr_sampled[col][:,TIME_INDEX])) , "convergence test time sample points does not match"
            error_mat[row,col] = np.max(abs(data_base[:,MASS_INDEX] - hr_sampled[col][:,MASS_INDEX])/hr_sampled[col][:,MASS_INDEX])
    
    #print(error_mat)
    print("error in mass growth")
    table_data = [
        ["3" , error_mat[0,0], error_mat[0,1], error_mat[0,2], error_mat[0,3]],
        ["7" , error_mat[1,0], error_mat[1,1], error_mat[1,2], error_mat[1,3]],
        ["15", error_mat[2,0], error_mat[2,1], error_mat[2,2], error_mat[2,3]],
        ["31", error_mat[3,0], error_mat[3,1], error_mat[3,2], error_mat[3,3]],
    ]
    print(tabulate(table_data,headers=["Nr","dt=1E-10","dt=5E-11","dt=2.5E-11","dt=1.25E-11"]))
    pl_lable = ["Nr=3, dt=1E-10", "Nr=7, dt=5E-11", "Nr=15, dt=2.5E-11"]
    for row,col in ((0,0),(1,1),(2,2)):
        data_base = data[row][col][0:len(hr_sampled[col][:,TIME_INDEX]),:]
        plt.plot(hr_sampled[col][:,TIME_INDEX], abs(data_base[:,MASS_INDEX] - hr_sampled[col][:,MASS_INDEX])/hr_sampled[col][:,MASS_INDEX],label=pl_lable[row])

    plt.xlabel(r"$time(s) \rightarrow$")
    plt.ylabel(r"$relative \ error (mass)\rightarrow $")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.show()

    plt.close()
    pl_lable = ["Nr=3, dt=1E-10", "Nr=7, dt=5E-11", "Nr=15, dt=2.5E-11", "Nr=31, dt=1.25E-11"]
    for row,col in ((0,0),(1,1),(2,2),(3,3)):
        data_base = data[row][col][0:len(hr_sampled[col][:,TIME_INDEX]),:]
        plt.plot(hr_sampled[col][:,TIME_INDEX], spec_tail(data_base),label=pl_lable[row])
    
    plt.yscale("log")
    plt.grid()
    plt.xlabel(r"$time(s) \rightarrow$")
    plt.ylabel(r"$|h_{tail}|/ |h| \rightarrow $")
    plt.legend()
    plt.show()


    # plt.close()
    # MNE  = data[0][0][0,MASS_INDEX]
    # MT1  = data[0][0][0,TEMP_INDEX]
    # VTH1             = collisions.electron_thermal_velocity(MT1)
    # NR=[3,7,15,31]
    # pl_lable = ["Nr=3, dt=1E-10", "Nr=7, dt=5E-11", "Nr=15, dt=2.5E-11", "Nr=31, dt=1.25E-11"]
    # cf_sp_base   = colOpSp.CollisionOpSP(3,NR[-1])
    # spec_base    = cf_sp_base._spec 
    # for row in range(4):
    #     cf_sp    = colOpSp.CollisionOpSP(3,NR[row])
    #     spec_sp  = cf_sp._spec
    #     for col in range(4):
    #         data_base = data[row][col][0:len(hr_sampled[col][:,TIME_INDEX]),:]
    #         error_mat[row,col]=0.0
    #         num_sample_pts = 10
    #         step_sz        = data_base.shape[0] // num_sample_pts
    #         tt = np.zeros(num_sample_pts)
    #         er = np.zeros(num_sample_pts)
            
    #         for ii,t in enumerate(range(0,data_base.shape[0],step_sz)):
    #             grid_pts         = int(1e5)
    #             grid_bds         = (0.0,8)
    #             MNE1             = data_base[t,MASS_INDEX]
    #             VTH1             = collisions.electron_thermal_velocity(data_base[t,TEMP_INDEX]) 
    #             MW1              = BEUtils.get_maxwellian_3d(VTH1 ,MNE1)
    #             vr1              = np.linspace(grid_bds[0]*VTH1,grid_bds[1]*VTH1,grid_pts)
                

    #             v1               = vr1/VTH1
    #             Mr1              = MW1(v1)
    #             Pr1              = spec_sp.Vq_r(v1).transpose()
    #             fv1              = Mr1 * (np.dot(Pr1,data_base[t,C000_INDEX:]).reshape(-1))


    #             MNE2             = hr_sampled[col][t,MASS_INDEX]
    #             VTH2             = collisions.electron_thermal_velocity(hr_sampled[col][t,TEMP_INDEX]) 
    #             MW2              = BEUtils.get_maxwellian_3d(VTH2 ,MNE2)
    #             v2               = vr1/VTH2
    #             Mr2              = MW2(v2)
    #             Pr2              = spec_base.Vq_r(v2).transpose()
    #             fv2              = Mr2 * (np.dot(Pr2,hr_sampled[col][t,C000_INDEX:]).reshape(-1))

    #             # plt.plot(v1,fv1,label="fv1")
    #             # plt.plot(v2,fv2,label="fv2")
    #             # plt.legend()
    #             # #plt.yscale("log")
    #             # plt.grid()
    #             # plt.show()
    #             # plt.close()
    #             #print("DT: %.16E VTH1 : %.16E MNE1: %.16E VTH2 : %.16E MNE2: %.16E |fv1-fv2| %.8E " %(abs(data_base[t,TIME_INDEX]-hr_sampled[col][t,TIME_INDEX]),VTH1, MNE1,VTH2, MNE2,np.linalg.norm(fv2-fv1)/np.linalg.norm(fv2)))
    #             #print(np.linalg.norm(fv2-fv1)/np.linalg.norm(fv2))
    #             assert np.allclose( (data_base[t,TIME_INDEX] - hr_sampled[col][t,TIME_INDEX]) , np.zeros_like(hr_sampled[col][t,TIME_INDEX])) , "convergence test time sample points does not match"
    #             tt[ii] = data_base[t,TIME_INDEX]
    #             er[ii] = np.linalg.norm((fv2-fv1))/np.linalg.norm(fv2)

    #         error_mat[row,col] = np.max(er)
    #         if row==col:
    #             plt.plot(tt,er,label=pl_lable[row])
    

    # plt.yscale("log")
    # plt.grid()
    # plt.legend()
    # plt.xlabel("time(s)")
    # plt.ylabel("|f2-f1|/f2")
    # plt.show()

    # table_data = [
    #     ["3" , error_mat[0,0], error_mat[0,1], error_mat[0,2], error_mat[0,3]],
    #     ["7" , error_mat[1,0], error_mat[1,1], error_mat[1,2], error_mat[1,3]],
    #     ["15", error_mat[2,0], error_mat[2,1], error_mat[2,2], error_mat[2,3]],
    #     ["31", error_mat[3,0], error_mat[3,1], error_mat[3,2], error_mat[3,3]],
    # ]
    # print(tabulate(table_data,headers=["Nr","dt=1E-10","dt=5E-11","dt=2.5E-11","dt=1.25E-11"]))


    
def g0_proj_vs_no_proj():

    DATA_FOLDER_NAME="dat_1ev_no_proj"
    file_names_np=[[DATA_FOLDER_NAME +"/g0_dt_1.00000000E-10_Nr_3.dat",  DATA_FOLDER_NAME+"/g0_dt_5.00000000E-11_Nr_3.dat",  DATA_FOLDER_NAME+"/g0_dt_2.50000000E-11_Nr_3.dat",  DATA_FOLDER_NAME+"/g0_dt_1.25000000E-11_Nr_3.dat"],
                [DATA_FOLDER_NAME +"/g0_dt_1.00000000E-10_Nr_7.dat",  DATA_FOLDER_NAME+"/g0_dt_5.00000000E-11_Nr_7.dat",  DATA_FOLDER_NAME+"/g0_dt_2.50000000E-11_Nr_7.dat",  DATA_FOLDER_NAME+"/g0_dt_1.25000000E-11_Nr_7.dat"],
                [DATA_FOLDER_NAME +"/g0_dt_1.00000000E-10_Nr_15.dat", DATA_FOLDER_NAME+"/g0_dt_5.00000000E-11_Nr_15.dat", DATA_FOLDER_NAME+"/g0_dt_2.50000000E-11_Nr_15.dat", DATA_FOLDER_NAME+"/g0_dt_1.25000000E-11_Nr_15.dat"],
                [DATA_FOLDER_NAME +"/g0_dt_1.00000000E-10_Nr_31.dat", DATA_FOLDER_NAME+"/g0_dt_5.00000000E-11_Nr_31.dat", DATA_FOLDER_NAME+"/g0_dt_2.50000000E-11_Nr_31.dat", DATA_FOLDER_NAME+"/g0_dt_1.25000000E-11_Nr_31.dat"]]

    DATA_FOLDER_NAME="dat_1ev"
    file_names_p =[[DATA_FOLDER_NAME +"/g0_dt_1.00000000E-10_Nr_3.dat",  DATA_FOLDER_NAME+"/g0_dt_5.00000000E-11_Nr_3.dat",  DATA_FOLDER_NAME+"/g0_dt_2.50000000E-11_Nr_3.dat",  DATA_FOLDER_NAME+"/g0_dt_1.25000000E-11_Nr_3.dat"],
                [DATA_FOLDER_NAME +"/g0_dt_1.00000000E-10_Nr_7.dat",  DATA_FOLDER_NAME+"/g0_dt_5.00000000E-11_Nr_7.dat",  DATA_FOLDER_NAME+"/g0_dt_2.50000000E-11_Nr_7.dat",  DATA_FOLDER_NAME+"/g0_dt_1.25000000E-11_Nr_7.dat"],
                [DATA_FOLDER_NAME +"/g0_dt_1.00000000E-10_Nr_15.dat", DATA_FOLDER_NAME+"/g0_dt_5.00000000E-11_Nr_15.dat", DATA_FOLDER_NAME+"/g0_dt_2.50000000E-11_Nr_15.dat", DATA_FOLDER_NAME+"/g0_dt_1.25000000E-11_Nr_15.dat"],
                [DATA_FOLDER_NAME +"/g0_dt_1.00000000E-10_Nr_31.dat", DATA_FOLDER_NAME+"/g0_dt_5.00000000E-11_Nr_31.dat", DATA_FOLDER_NAME+"/g0_dt_2.50000000E-11_Nr_31.dat", DATA_FOLDER_NAME+"/g0_dt_1.25000000E-11_Nr_31.dat"]]


    data_proj = list()
    for t_res in file_names_p:
        tmp=list()
        for fname in t_res:
            tmp.append(np.loadtxt(fname,skiprows=1))
        data_proj.append(tmp)

    data_no_proj = list()
    for t_res in file_names_np:
        tmp=list()
        for fname in t_res:
            tmp.append(np.loadtxt(fname,skiprows=1))
        data_no_proj.append(tmp)
    Nr=[3,7,15,31]

    for (r,c) in ((0,0),(1,1),(2,2),(3,3)):
        plt.plot(data_proj[r][c][:,TIME_INDEX],abs(data_proj[r][c][:,TEMP_INDEX]-data_no_proj[r][c][:,TEMP_INDEX])/data_proj[r][c][:,TEMP_INDEX],label="Nr=%d"%Nr[r])

    plt.xlabel("time(s)")
    plt.ylabel("relative error")
    plt.grid()
    plt.legend()
    plt.yscale("log")
    plt.show()
            
def g0_conv_plot():

    for mm in range(0,10):
        DATA_FOLDER_NAME="dat_1ev_cs_m"+str(mm)
        EV = 1.0

        params.BEVelocitySpace.SPH_HARM_LM = [[0,0]]
        params.BEVelocitySpace.NUM_Q_VR    = 118
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

        f_names=[
                    DATA_FOLDER_NAME+"/g0_dt_1.00000000E-14_Nr_4.dat",
                    DATA_FOLDER_NAME+"/g0_dt_5.00000000E-15_Nr_8.dat",
                    DATA_FOLDER_NAME+"/g0_dt_2.50000000E-15_Nr_16.dat",
                    DATA_FOLDER_NAME+"/g0_dt_1.25000000E-15_Nr_32.dat",
                    DATA_FOLDER_NAME+"/g0_dt_6.25000000E-16_Nr_64.dat" 
                ]
        
        TIME_INDEX=0
        C000_INDEX=1
        
        NR   = [4, 8, 16, 32, 64]
        DT   = [1e-10, 5e-11, 2.5e-11, 1.25e-11, 6.25e-12]
        data = list()
        
        for f in f_names:
            data.append(np.loadtxt(f))
            #print(data[-1].shape)

        import matplotlib.pylab as plt
        for i in range(len(NR)):
            nr = NR[i]
            tail_end   = (nr+1)
            tail_begin = tail_end//2
            tail_norm = np.linalg.norm(data[i][:,C000_INDEX + tail_begin: C000_INDEX + tail_end],axis=1)/np.linalg.norm(data[i][:,C000_INDEX: C000_INDEX + tail_end],axis=1)
            plt.plot(data[i][:,TIME_INDEX], tail_norm,label="nr= %d dt=%.2E"%(nr,DT[i]))
        
        plt.xlabel("time (s)")
        plt.ylabel("tail norm")
        plt.yscale("log")
        plt.legend()
        plt.grid()
        fname=DATA_FOLDER_NAME+"_tail.png"
        plt.savefig(fname)
        plt.show()
        plt.close()

        temperature = np.zeros((len(NR),data[0].shape[0]))

        for i in range(len(NR)):
            nr = NR[i]
            params.BEVelocitySpace().VELOCITY_SPACE_POLY_ORDER=nr
            cf_sp    = colOpSp.CollisionOpSP(3,nr,q_mode=sp.QuadMode.GMX)
            spec_sp  = cf_sp._spec

            m0_t0             = BEUtils.moment_n_f(spec_sp,np.transpose(data[i][:,C000_INDEX:]),maxwellian,VTH,0,None,None,None,1)
            temperature[i,:]  = BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec_sp,np.transpose(data[i][:,C000_INDEX:]),maxwellian,VTH,None,None,None,m0_t0,1)
        
        for i in range(1,len(NR)):
            nr = NR[i]
            # cf_sp    = colOpSp.CollisionOpSP(3,nr,q_mode=sp.QuadMode.GMX)
            # spec_sp  = cf_sp._spec
            #rel_error = temperature[i] * collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT
            rel_error  = abs(temperature[i-1,:]-temperature[i,:])/temperature[i,:]
            plt.plot(data[i][:,TIME_INDEX],rel_error,label="nr= %d dt=%.2E"%(NR[i-1],DT[i-1]))
        
        plt.xlabel("time (s)")
        plt.ylabel("relative error")
        plt.yscale("log")
        plt.legend()
        plt.grid()
        fname=DATA_FOLDER_NAME+"_temp_error.png"
        plt.savefig(fname)
        plt.close()

        for i in range(1,len(NR)):
            nr = NR[i]
            # cf_sp    = colOpSp.CollisionOpSP(3,nr,q_mode=sp.QuadMode.GMX)
            # spec_sp  = cf_sp._spec
            rel_error = temperature[i] * collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT
            plt.plot(data[i][:,TIME_INDEX],rel_error,label="nr= %d dt=%.2E"%(NR[i-1],DT[i-1]))
        
        plt.xlabel("time (s)")
        plt.ylabel("temperature (eV)")
        #plt.yscale("log")
        plt.legend()
        plt.grid()
        fname=DATA_FOLDER_NAME+"_temp.png"
        plt.savefig(fname)
        plt.close()


g0_conv_plot()
#g0_convergence()
#g02_quasi_neutral()
#g02_quasi_neutral_1ev()

#print("g0 convergence results")
#g0_rk4_1ev_convergence()

#print("g02 convergence results")
#g02_rk4_1ev_convergence()

#g0_proj_vs_no_proj()
