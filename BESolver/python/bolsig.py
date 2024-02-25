"""
Simple script to run Bolsig through Python. 
"""

import numpy as np
from cProfile import run
from dataclasses import replace
import collisions
import os
import lxcat_data_parser as ldp
import sys
import cross_section

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

    rate = np.zeros(num_collisions)

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

            if "Rate coefficients (m3/s)" in line:
                for i in range(num_collisions):
                    rr_line = F.readline()
                    rate[i] = float(rr_line.split()[-1]) 
                    print("Found collision rate no. "+str(i+1)+" = ", rate[i])
                    ratelist.append(rate)

                # if (len(line)>=8):
                #     print(line)
                #     if (line[0:8]=='C'+str(i+1)+'    Ar'):
                #         print(line[0:8])
                #         rate[i] = float(line.split()[-1]) 
                #         print("Found collision rate no. "+str(i+1)+" = ", rate[i])
                #         ratelist.append(rate)

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
    cs_species  = cross_section.read_available_species(args.collisions)
    cs_data_all = cross_section.read_cross_section_data(args.collisions)
    
    bolsig_cs_file = "crs_file.txt"
    for i, (col_str, col_data) in enumerate(cs_data_all.items()):
        col_str   = col_str
        col_type  = col_data["type"]
        g         = collisions.electron_heavy_binary_collision(cross_section=col_str, collision_type=col_type)
        ev        = g._energy
        tcs       = g.total_cross_section(ev)
        
        cs_data=np.concatenate((ev,tcs),axis=0)
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
            
            collision_prefix  = col_data["type"] + "\n"
            collision_prefix += col_data["species"] + "\n"
            
            if col_data["mass_ratio"] is not None:
                collision_prefix += str(col_data["mass_ratio"]) + "\n"
                
            if col_data["threshold"] is not None:
                collision_prefix += str(col_data["threshold"]) + "\n"
            
            for k, v in col_data["info"].items():
                collision_prefix += str(k) + ": "+ str(v) + "\n"
                
            file.writelines(collision_prefix)
            file.write("\n-----------------------------\n")
            cs_str=["%.14E %14E"%(cs_data[i][0],cs_data[i][1]) for i in range(cs_data.shape[0])]
            cs_str = '\n'.join(cs_str)
            file.writelines(cs_str)
            file.write("\n-----------------------------\n")
        

    replace_line(args.bolsig_dir+"run.sh", 2, "cd " + args.bolsig_dir + "\n")
    replace_line(args.bolsig_dir+"minimal-argon.dat", 8, "\""+bolsig_cs_file+"\"   / File\n")
    replace_line(args.bolsig_dir+"minimal-argon.dat", 9, " ".join([s for s in cs_species])+   "/ species\n")
    replace_line(args.bolsig_dir+"minimal-argon.dat", 13, "%.8E"%(args.E_field/args.n0/1e-21)+"   / Electric field / N (Td)\n")
    replace_line(args.bolsig_dir+"minimal-argon.dat", 16, "%.8E"%(args.Tg) +"   / Gas temperature (K)\n")
    if args.ee_collisions:
        replace_line(args.bolsig_dir+"minimal-argon.dat", 19, "%.8E"%(args.ion_deg) +"   / Ionization degree\n")
        replace_line(args.bolsig_dir+"minimal-argon.dat", 20, "%.8E"%(args.ion_deg * args.n0) +"   / Plasma Density (1/m^3)\n")
        replace_line(args.bolsig_dir+"minimal-argon.dat", 23, "%d"%(1) +"   / e-e momentum effects: 0=No; 1=Yes*\n")
    else:
        replace_line(args.bolsig_dir+"minimal-argon.dat", 19, "%.8E"%(0) +"   / Ionization degree\n")
        replace_line(args.bolsig_dir+"minimal-argon.dat", 20, "%.8E"%(args.n0) +"   / Plasma Density (1/m^3)\n")
        replace_line(args.bolsig_dir+"minimal-argon.dat", 23, "%d"%(0) +"   / e-e momentum effects: 0=No; 1=Yes*\n")

    replace_line(args.bolsig_dir+"minimal-argon.dat", 27, str(args.bolsig_grid_pts)+    "   / # of grid points\n")
    replace_line(args.bolsig_dir+"minimal-argon.dat", 30, str(args.bolsig_precision)+   "   / Precision\n")
    replace_line(args.bolsig_dir+"minimal-argon.dat", 31, str(args.bolsig_convergence)+ "   / Convergence\n")
    replace_line(args.bolsig_dir+"minimal-argon.dat", 33, " ".join([str(args.ns_by_n0[i]) for i in range(len(cs_species))])         + "   / Gas composition fractions\n")
    
    os.system("sh "+args.bolsig_dir+"run.sh")
    return
    