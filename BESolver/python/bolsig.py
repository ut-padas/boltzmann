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
    clogarithmlist = []

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
            
            if "Coulomb logarithm" in line:
                cl = float(line.strip().split(" ")[-1]) 
                print("Found coulomb logarithm = ", cl)
                clogarithmlist.append(cl)
            else:
                clogarithmlist.append(0)

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

    return [eveclist[0], dveclist[0], aveclist[0], Elist[0], mulist[0], mobillist[0], difflist[0], ratelist[0], clogarithmlist[0]]

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
    ev_max = 1e4

    for i, cc in enumerate(args.collisions):
        if "g0" in str(cc):
            prefix_line = g0_str
            g           = collisions.eAr_G0(cross_section=cc)
            #ev1         = g._energy
            ev1         = np.logspace(np.log10(g._energy[0] + 1e-12), np.log10(ev_max), 800, base=10)
            tcs         = g.total_cross_section(ev1)

            # ev1         = np.append(ev1, ev_max)
            # tcs         = np.append(tcs, tcs[-1])
            
            cs_data=np.concatenate((ev1,tcs),axis=0)
            cs_data=cs_data.reshape((2,-1))
            cs_data=np.transpose(cs_data)

        elif "g2" in str(cc):
            prefix_line = g2_str
            g           = collisions.eAr_G2(cross_section=cc)
            #ev1         = g._energy 
            ev1         = np.logspace(np.log10(g._energy[0]), np.log10(ev_max), 800, base=10)
            tcs         = g.total_cross_section(ev1)

            # ev1         = np.append(ev1, ev_max)
            # tcs         = np.append(tcs, tcs[-1])

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
    if args.ee_collisions:
        replace_line(args.bolsig_dir+"minimal-argon.dat", 19, "%.8E"%(args.ion_deg) +"\t\t/ Ionization degree\n")
        replace_line(args.bolsig_dir+"minimal-argon.dat", 20, "%.8E"%(args.ion_deg * collisions.AR_NEUTRAL_N) +"\t\t/ Plasma Density (1/m^3)\n")
        replace_line(args.bolsig_dir+"minimal-argon.dat", 23, "%d"%(1) +"\t\t// e-e momentum effects: 0=No; 1=Yes*\n")
    else:
        replace_line(args.bolsig_dir+"minimal-argon.dat", 19, "%.8E"%(0) +"\t\t/ Ionization degree\n")
        replace_line(args.bolsig_dir+"minimal-argon.dat", 20, "%.8E"%(collisions.AR_NEUTRAL_N) +"\t\t/ Plasma Density (1/m^3)\n")
        replace_line(args.bolsig_dir+"minimal-argon.dat", 23, "%d"%(0) +"\t\t// e-e momentum effects: 0=No; 1=Yes*\n")

    replace_line(args.bolsig_dir+"minimal-argon.dat", 27, str(args.bolsig_grid_pts)+    "   / # of grid points\n")
    replace_line(args.bolsig_dir+"minimal-argon.dat", 30, str(args.bolsig_precision)+   "   / Precision\n")
    replace_line(args.bolsig_dir+"minimal-argon.dat", 31, str(args.bolsig_convergence)+ "   / Convergence\n")
    os.system("sh "+args.bolsig_dir+"run.sh")
    return
    