"""
Simple script to run Bolsig through Python. 
"""

import numpy as np
from cProfile import run
from dataclasses import replace
import collisions
import os

def parse_bolsig(file, num_collisions):
    eveclist = []
    dveclist = []
    aveclist = []
    Elist = []
    mulist = []
    mobillist = []
    difflist = []
    ratelist = []

    rate = np.zeros(2)

    with open(file, 'r') as F:
        line = F.readline()
        
        while (line!='Energy (eV) EEDF (eV-3/2) Anisotropy\n'):

            if (len(line)>=23):
                if (line[0:23]=='Electric field / N (Td)'):
                    EbN = float(line.split(")",2)[1]) # 1e-21 converts from Td to V*m^2
                    print("Found EbN = ", EbN)
                    Elist.append(EbN)

            if (len(line)>=16):
                if (line[0:16]=='Mean energy (eV)'):
                    mu = float(line.split(")",2)[1]) # while it says eV, the unit here is actually V
                    print("Found mu = ", mu)
                    mulist.append(mu)

            if (len(line)>=21):
                if (line[0:21]=='Mobility *N (1/m/V/s)'):
                    mobil = float(line.split(")",2)[1]) 
                    print("Found mobility = ", mobil)
                    mobillist.append(mobil)

            if (len(line)>=32):
                if (line[0:32]=='Diffusion coefficient *N (1/m/s)'):
                    diff = float(line.split(")",2)[1]) 
                    print("Found diffusion = ", diff)
                    difflist.append(diff)

            for i in range(num_collisions):
                if (len(line)>=8):
                    if (line[0:8]=='C'+str(i+1)+'    Ar'):
                        rate[i] = float(line.split()[-1]) 
                        print("Found collision rate no. "+str(i+1)+" = ", rate[i])
                        ratelist.append(rate)

            line = F.readline()

        # while (line!=''):
        line = F.readline()
        elist = []
        dlist = []
        alist = []
        while (line!=' \n'):
            col = line.split()
            energy = float(col[0])
            distrib = float(col[1])
            anisotropy = float(col[2])

            elist.append(energy)
            dlist.append(distrib)
            alist.append(anisotropy)

            line = F.readline()

        print("Adding vectors to lists for Etil = {0:.3e}...".format(EbN))
        eveclist.append(np.array(elist))
        dveclist.append(np.array(dlist))
        aveclist.append(np.array(alist))

    return [eveclist[0], dveclist[0], aveclist[0], Elist[0], mulist[0], mobillist[0], difflist[0], ratelist[0]]

def replace_line(file_name, line_num, text):
    lines = open(file_name, 'r').readlines()
    lines[line_num] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()

def run_bolsig(args, run_convergence=False):
    """
    run the bolsig code. 
    """
    bolsig_cs_file = args.collisions[0]
    for col in args.collisions[1:]:
        bolsig_cs_file = bolsig_cs_file + "_" + col
    bolsig_cs_file = bolsig_cs_file + ".txt"

    g0_str="""
    EFFECTIVE
    Ar
    1.373235e-5
    SPECIES: e / Ar
    PROCESS: E + Ar -> E + Ar, Effective
    PARAM.:  m/M = 1.373235e-5, complete set
    COMMENT: EFFECTIVE Momentum transfer CROSS SECTION.
    UPDATED: 2011-06-06 11:19:56
    COLUMNS: Energy (eV) | Cross section (m2)
    """
    g2_str="""
    IONIZATION
    Ar -> Ar^+
    1.576000e+1
    SPECIES: e / Ar
    PROCESS: E + Ar -> E + E + Ar+, Ionization
    PARAM.:  E = 15.76 eV, complete set
    COMMENT: Ionization - RAPP-SCHRAM.
    UPDATED: 2010-10-01 07:49:50
    COLUMNS: Energy (eV) | Cross section (m2)
    """

    for i, cc in enumerate(args.collisions):
        if "g0" in str(cc):
            prefix_line=g0_str

            ev1=np.logspace(-4,4,500,base=10)
            tcs = collisions.Collisions.synthetic_tcs(ev1,cc)

            cs_data=np.concatenate((ev1,tcs),axis=0)
            cs_data=cs_data.reshape((2,-1))
            cs_data=np.transpose(cs_data)

        elif "g2" in str(cc):
            prefix_line=g2_str

            ev1=np.logspace(np.log10(15.76),4,500,base=10)
            tcs = collisions.Collisions.synthetic_tcs(ev1,cc)

            cs_data=np.concatenate((ev1,tcs),axis=0)
            cs_data=cs_data.reshape((2,-1))
            cs_data=np.transpose(cs_data)
            
        f_mode="a"
        if i==0:
            f_mode="w"
        with open("../../Bolsig/%s"%(bolsig_cs_file), f_mode) as file:
            cs_str=np.array_str(cs_data)
            cs_str=cs_str.replace("[","")
            cs_str=cs_str.replace("]","")
            cs_str=" "+cs_str
            file.writelines(prefix_line)
            file.write("\n-----------------------------\n")
            cs_str=["%.14E %14E"%(cs_data[i][0],cs_data[i][1]) for i in range(cs_data.shape[0])]
            cs_str = '\n'.join(cs_str)
            file.writelines(cs_str)
            file.write("\n-----------------------------\n")
        

    replace_line(args.bolsig_dir+"run.sh", 2, "cd " + args.bolsig_dir + "\n")
    replace_line(args.bolsig_dir+"minimal-argon.dat", 8, "\""+bolsig_cs_file+"\"   / File\n")
    replace_line(args.bolsig_dir+"minimal-argon.dat", 13, "%.8E"%(args.E_field/collisions.AR_NEUTRAL_N/1e-21)+"\t\t/ Electric field / N (Td)\n")
    replace_line(args.bolsig_dir+"minimal-argon.dat", 16, "%.8E"%(args.Tg) +"\t\t/ Gas temperature (K)\n")

    if (run_convergence):
        bolsig_rundata=list()
        for i, value in enumerate(args.sweep_values):
            if args.sweep_param == "Nr":
                args.NUM_P_RADIAL = value
            else:
                print("Warning : unsupported sweep argument for the Bolsig code, setting num pts to 500")
                args.NUM_P_RADIAL = 500

            replace_line(args.bolsig_dir+"minimal-argon.dat", 27, str(args.NUM_P_RADIAL)+" / # of grid points\n")
            os.system("sh "+args.bolsig_dir+"run.sh")
            sleep(2)
            #[0,            1,             2,       3,          4,        5,        6,          7  ]
            #[bolsig_ev, bolsig_f0, bolsig_a, bolsig_E, bolsig_mu, bolsig_M, bolsig_D, bolsig_rates] 
            b_data=parse_bolsig(args.bolsig_dir+"argon.out",len(args.collisions))
            bolsig_rundata.append(b_data)

        if (1):
            mpl.style.use('default')
            fig = plt.figure(figsize=(21, 9), dpi=300)
            num_sph_harm = 2
            num_subplots = num_sph_harm + 2

            def sph_harm_real(l, m, theta, phi):
                # in python's sph_harm phi and theta are swapped
                Y = scipy.special.sph_harm(abs(m), l, phi, theta)
                if m < 0:
                    Y = np.sqrt(2) * (-1)**m * Y.imag
                elif m > 0:
                    Y = np.sqrt(2) * (-1)**m * Y.real
                else:
                    Y = Y.real

                return Y 

            mu = list()
            M  = list()
            D  = list()

            rates = list()
            
            bolsig_mu    = bolsig_rundata[-1][4] 
            bolsig_M     = bolsig_rundata[-1][5] 
            bolsig_D     = bolsig_rundata[-1][6] 
            bolsig_rates = bolsig_rundata[-1][7] 
            sph_fac = sph_harm_real(0, 0, 0, 0)/sph_harm_real(1, 0, 0, 0)

            for i in range(len(args.sweep_values)):
                value=args.sweep_values[i]

                data=bolsig_rundata[i]
                bolsig_ev = data[0] 
                bolsig_f0 = data[1] 
                bolsig_a  = data[2]

                # print("i", i)
                # print(bolsig_ev)

                mu.append(data[4])  
                M .append(data[5]) 
                D .append(data[6])
                rates.append(data[7])
                    

                lbl = "b_" + args.sweep_param+"="+str(value)
                color = 'C%d'%(i+1)#next(plt.gca()._get_lines.prop_cycler)['color']

                plt.subplot(2, num_subplots,  1)
                plt.semilogy(bolsig_ev,  abs(bolsig_f0), label=lbl, color=color)
                plt.grid(True)
                plt.legend()

                plt.subplot(2, num_subplots,  2)
                plt.semilogy(bolsig_ev,  abs(bolsig_f0*bolsig_a * sph_fac),color=color)
                plt.grid(True)

                #print(bolsig_f0.shape)
                #print(bolsig_rundata[-1][0].shape)
                plt.subplot(2, num_subplots,  5)
                # rel_error= (bolsig_f0/bolsig_rundata[-1][1]-1)
                # plt.semilogy(bolsig_ev,  abs(rel_error) ,color=color)
                # plt.xlabel("Energy (eV)")
                # plt.ylabel("Rel. error f0")
                # plt.grid()

                plt.subplot(2, num_subplots,  6)
                # rel_error= bolsig_f0*bolsig_a/bolsig_rundata[-1][1] * bolsig_rundata[-1][2]-1
                # plt.semilogy(bolsig_ev,  abs(rel_error) ,color=color)
                # plt.xlabel("Energy (eV)")
                # plt.ylabel("Rel. error f1")
                # plt.grid()

            rates=np.array(rates)
            plt.subplot(2, num_subplots,  3)
            plt.plot(args.sweep_values, np.abs(np.array(mu)/bolsig_mu-1), 'o-')
            plt.xlabel(args.sweep_param)
            plt.ylabel("Rel. error in mean energy")
            plt.grid()

            plt.subplot(2, num_subplots,  4)
            for col_idx, col in enumerate(args.collisions):
                if bolsig_rates[col_idx] != 0:
                    plt.semilogy(args.sweep_values, abs(rates[:,col_idx]/bolsig_rates[col_idx]-1), 'o-', label=col)
                    # plt.axhline(y=0, label='bolsig '+col, color='k')
            plt.legend()
            plt.xlabel(args.sweep_param)
            plt.ylabel("Rel. error in reaction rates")
            plt.grid()

            plt.subplot(2, num_subplots,  7)
            plt.semilogy(args.sweep_values, np.abs(np.array(M)/bolsig_M-1), 'o-')
            # plt.legend()
            plt.xlabel(args.sweep_param)
            plt.ylabel("Rel. error in mobility")
            plt.grid()

            plt.subplot(2, num_subplots, 8)
            # plt.plot(args.sweep_values, D, 'o-', label='us')
            # plt.axhline(y=bolsig_D, label='bolsig', color='k')
            plt.semilogy(args.sweep_values, np.abs(np.array(D)/bolsig_D-1), 'o-')
            # plt.legend()
            plt.xlabel(args.sweep_param)
            plt.ylabel("Rel. error in diffusion coefficient")
            plt.grid()

        fig.suptitle("Collisions: " + str(args.collisions) + ", E = " + str(args.E_field) + ", Nr = " + str(args.NUM_P_RADIAL) +  " (sweeping " + args.sweep_param + ")")
        plt.savefig("bolsig_vs_bolsig_" + "_".join(args.collisions) + "_E" + str(args.E_field) + "_nr" + str(args.NUM_P_RADIAL) +  "_sweeping_" + args.sweep_param + ".png")

    replace_line(args.bolsig_dir+"minimal-argon.dat", 27, str(1000)+" / # of grid points\n")
    os.system("sh "+args.bolsig_dir+"run.sh")
    return
    