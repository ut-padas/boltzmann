"""
@package : simple convergence plots
"""
import matplotlib as mpl
#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
import matplotlib.pyplot as plt
import spec_spherical
import argparse
import numpy as np
import scipy.constants 
import utils as BEUtils
import collisions
from tabulate import tabulate

TIME_INDEX=0
MASS_INDEX=1
TEMP_INDEX=3
C000_INDEX=4

#f = plt.figure()
#f.set_figwidth(100)
#f.set_figheight(100)
#f.set_size_inches(8, 8, forward=True)
plt.rcParams["figure.figsize"] = (6,6)



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

    file_names=[["dat_1ev/g0_dt_1.00000000E-10_Nr_3.dat",  "dat_1ev/g0_dt_5.00000000E-11_Nr_3.dat",  "dat_1ev/g0_dt_2.50000000E-11_Nr_3.dat",  "dat_1ev/g0_dt_1.25000000E-11_Nr_3.dat"],
                ["dat_1ev/g0_dt_1.00000000E-10_Nr_7.dat",  "dat_1ev/g0_dt_5.00000000E-11_Nr_7.dat",  "dat_1ev/g0_dt_2.50000000E-11_Nr_7.dat",  "dat_1ev/g0_dt_1.25000000E-11_Nr_7.dat"],
                ["dat_1ev/g0_dt_1.00000000E-10_Nr_15.dat", "dat_1ev/g0_dt_5.00000000E-11_Nr_15.dat", "dat_1ev/g0_dt_2.50000000E-11_Nr_15.dat", "dat_1ev/g0_dt_1.25000000E-11_Nr_15.dat"]]

    data = list()
    for t_res in file_names:
        tmp=list()
        for fname in t_res:
            tmp.append(np.loadtxt(fname,skiprows=1))
        data.append(tmp)

    #print(data.shape)

    # data_nr_1x_dt1 = np.loadtxt(file_names[0][0],skiprows=1)
    # data_nr_2x_dt1 = np.loadtxt(file_names[1][0],skiprows=1)
    # data_nr_4x_dt1 = np.loadtxt(file_names[2][0],skiprows=1)


    # data_nr_1x_dt2 = np.loadtxt(file_names[0][1],skiprows=1)
    # data_nr_2x_dt2 = np.loadtxt(file_names[1][1],skiprows=1)
    # data_nr_4x_dt2 = np.loadtxt(file_names[2][1],skiprows=1)

    # data_nr_1x_dt4 = np.loadtxt(file_names[0][2],skiprows=1)
    # data_nr_2x_dt4 = np.loadtxt(file_names[1][2],skiprows=1)
    # data_nr_4x_dt4 = np.loadtxt(file_names[2][2],skiprows=1)

    # data_nr_1x_dt8 = np.loadtxt(file_names[0][3],skiprows=1)
    # data_nr_2x_dt8 = np.loadtxt(file_names[1][3],skiprows=1)
    # data_nr_4x_dt8 = np.loadtxt(file_names[2][3],skiprows=1)


    highest_res = data[2][3]
    hr_sampled  = [ highest_res[range(0,len(highest_res[:,TIME_INDEX]),8),:],
                    highest_res[range(0,len(highest_res[:,TIME_INDEX]),4),:],
                    highest_res[range(0,len(highest_res[:,TIME_INDEX]),2),:],
                    highest_res[range(0,len(highest_res[:,TIME_INDEX]),1),:]]
    
    error_mat = np.zeros((3,4))
    for row in range(0,3):
        for col in range(0,4):
            # check if time points match first. 
            data_base = data[row][col][0:len(hr_sampled[col][:,TIME_INDEX]),:]
            assert np.allclose( (data_base[:,TIME_INDEX] - hr_sampled[col][:,TIME_INDEX]) , np.zeros_like(hr_sampled[col][:,TIME_INDEX])) , "convergence test time sample points does not match"
            error_mat[row,col] = np.max(abs(data_base[:,TEMP_INDEX] - hr_sampled[col][:,TEMP_INDEX]))
    
    #print(error_mat)
    table_data = [
        ["3" , error_mat[0,0], error_mat[0,1], error_mat[0,2], error_mat[0,3]],
        ["7" , error_mat[1,0], error_mat[1,1], error_mat[1,2], error_mat[1,3]],
        ["15", error_mat[2,0], error_mat[2,1], error_mat[2,2], error_mat[2,3]],
    ]
    print(tabulate(table_data,headers=["Nr","dt=1E-10","dt=5E-11","dt=2.5E-11","dt=1.25E-11"]))
    
    pl_lable = ["Nr=3, dt=1E-10", "Nr=7, dt=5E-11", "Nr=15, dt=2.5E-11"]
    for row,col in ((0,0),(1,1),(2,2)):
        data_base = data[row][col][0:len(hr_sampled[col][:,TIME_INDEX]),:]
        plt.plot(hr_sampled[col][:,TIME_INDEX], (collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT) * abs(data_base[:,TEMP_INDEX] - hr_sampled[col][:,TEMP_INDEX]),label=pl_lable[row])

    plt.xlabel(r"$time(s) \rightarrow$")
    plt.ylabel(r"$Diff(T) (eV) \rightarrow $")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.show()




    # data2 = data_nr_1x_dt4[range(0,len(data_nr_1x_dt4[:-sk_tail,TIME_INDEX]),2),TEMP_INDEX]

    # data3 = data_nr_2x_dt4[range(0,len(data_nr_2x_dt4[:-sk_tail,TIME_INDEX]),4),TEMP_INDEX]
    # data4 = data_nr_2x_dt4[range(0,len(data_nr_2x_dt4[:-sk_tail,TIME_INDEX]),2),TEMP_INDEX]

    # data5 = data_nr_4x_dt4[range(0,len(data_nr_4x_dt4[:-sk_tail,TIME_INDEX]),4),TEMP_INDEX]
    # data6 = data_nr_4x_dt4[range(0,len(data_nr_4x_dt4[:-sk_tail,TIME_INDEX]),2),TEMP_INDEX]



#g0_convergence()
#g02_quasi_neutral()
#g02_quasi_neutral_1ev()

g0_rk4_1ev_convergence()


