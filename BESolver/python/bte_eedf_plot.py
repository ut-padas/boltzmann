import scipy
import basis
import spec_spherical as sp
import numpy as np 
import parameters as params
import matplotlib.pyplot as plt
import os
import collisions 

Tg = np.array([4919.96])
Tgstr = "%.2E" %Tg[0]
outnames  = ["bte_ss_ee_recomb", "bte_ts_ee_recomb", "bte_ss_ee_recomb", "bte_ts_ee_recomb", "bte_ts_noee_noIL", "bte_ts_noee", "bte_ts_ee_noIL", "bte_ts_ee"]
legstr    = ["steady-state, e-e, recomb", "transient, e-e, recomb", "steady-state, e-e, recomb", "transient, e-e, recomb",
            "transient, no e-e, no IL", "transient, no e-e", "transient, e-e, no IL", "transient, e-e"]
linestr   = ["-", "-", "-", ".", "-", "-.", "--", "."]
markerstr = ["s", "v", "o", "D", "s", "v", "o", "D"]
colorstr  = ["blue", "red", "orange", "gold", "blue", "red", "gold", "orange"]

legloc = 'lower left'
legsize = 18
do_save = 1


datfiles = []
for i in range(len(outnames)):
    tempstr = outnames[i] + "/" + outnames[i] + "_EEDF_Tg" + Tgstr + ".txt"
    datfiles.append(tempstr)

EV = 0; F0 = 1; F1 = 2; FM = 3

print("datfiles = ", datfiles)
ilo = 0; ihi = 2
for i in range(ilo, ihi):
    dat = np.loadtxt(datfiles[i], skiprows=1)

# eV versus f0
    if i == ilo:
        fig1, ax1 = plt.subplots(nrows=1, figsize=(10,10))
        fig1.patch.set_facecolor('white')
        fig1.patch.set_alpha(1.0)

        ax1.set_xlabel(r'Energy (eV)', fontsize=28)
        ax1.set_ylabel(r'f0 (eV$^{-1.5}$)', fontsize=28)

        plt.xticks(fontsize=20); plt.yticks(fontsize=20); 
        ax1.tick_params(axis='y', colors='k')
        plt.locator_params(axis='x', nbins=10)

    ax1.semilogy(dat[:,EV], dat[:,F0], label=legstr[i], color=colorstr[i], linewidth=2, linestyle=linestr[i])

    if i == ihi-1:
        ax1.semilogy(dat[:,EV], dat[:,FM], label="Maxwellian", color="gold", linewidth=2, linestyle="-")

        # Straight lines at threshold of excitation, step ionization, ground ionization
        # ax1.axvline(x = 11.512684347610431, linestyle=":", linewidth=2, color="black")
        # ax1.axvline(x = 15.650008568830833, linestyle=":", linewidth=2, color="black")

        ax1.legend(fontsize=legsize, labelcolor='k')

        if do_save == 1:
            imgstr    = "energy_vs_f0_bte_ss_ts_ee_recombination_Tg%.2E.png" %Tg[0]
            print("imgname = ", imgstr)

            titlestr = "Temperature = %.2E K" %(Tg[0])
            ax1.set_title(titlestr, fontsize=24)

            fig1.savefig(imgstr,bbox_inches='tight')
            print("Saved first image")

# # eV versus f1
    if i == ilo:
        fig2, ax2 = plt.subplots(nrows=1, figsize=(10,10))
        fig2.patch.set_facecolor('white')
        fig2.patch.set_alpha(1.0)

        ax2.set_xlabel(r'Energy (eV)', fontsize=28)
        ax2.set_ylabel(r'f1 (eV$^{-1.5}$)', fontsize=28)

        # Straight lines at threshold of excitation, step ionization, ground ionization
        # ax2.axvline(x = 11.512684347610431, linestyle=":", linewidth=2, color="black")
        # ax2.axvline(x = 15.650008568830833, linestyle=":", linewidth=2, color="black")
        
        plt.xticks(fontsize=20); plt.yticks(fontsize=20); 
        ax2.tick_params(axis='y', colors='k')
        plt.locator_params(axis='x', nbins=10)

    ax2.semilogy(dat[:,EV], dat[:,F1], label=legstr[i], color=colorstr[i], linewidth=2, linestyle=linestr[i])

    if i == ihi-1:
        ax2.legend(fontsize=legsize, labelcolor='k')

        if do_save == 1:
            imgstr    = "energy_vs_f1_bte_ss_ts_ee_recombination_Tg%.2E.png" %Tg[0]
            print("imgname = ", imgstr)
            titlestr = "Temperature = %.2E K" %(Tg[0])
            ax2.set_title(titlestr, fontsize=24)

            fig2.savefig(imgstr,bbox_inches='tight')


# # PLOT THE EEDFS AT TWO RESOLUTIONS TO CHECK CONVERGENCE
# Tg = np.array([10254.3])
# Tgstr = "%.2E" %Tg[0]
# file1 = "bte_ss_noee/bte_ss_noee_EEDF_Tg1.03E+04_Nr127.txt"
# file2 = "bte_ss_noee/bte_ss_noee_EEDF_Tg1.03E+04_Nr256.txt"

# dat1 = np.loadtxt(file1, skiprows=1)
# dat2 = np.loadtxt(file2, skiprows=1)
# EV = 0; F0 = 1; F1 = 2; FM = 3

# f1 = dat1[:,F0]
# f2 = dat2[:,F0]

# abserr = np.abs(f1 - f2)
# relerr = np.abs(f1 - f2) / (1E-20 + f2)

# print("max(abserr) = ", np.amax(abserr), ", mean(abserr) = ", np.mean(abserr))
# print("max(relerr) = ", np.amax(relerr), ", mean(relerr) = ", np.mean(relerr))

# # CONVERGENCE FOR F0
# fig1, ax1 = plt.subplots(nrows=1, figsize=(10,10))
# fig1.patch.set_facecolor('white')
# fig1.patch.set_alpha(1.0)

# ax1.set_xlabel(r'Energy (eV)', fontsize=28)
# ax1.set_ylabel(r'f0', fontsize=28)

# plt.xticks(fontsize=20); plt.yticks(fontsize=20); 
# ax1.tick_params(axis='y', colors='k')
# plt.locator_params(axis='x', nbins=10)

# ax1.semilogy(dat1[:,EV], dat1[:,F0], label="Nr = 127", color="blue", linewidth=2)
# ax1.semilogy(dat2[:,EV], dat2[:,F0], label="Nr = 256", color="red", linewidth=2)

# ax1.legend(fontsize=14, labelcolor='k')

# imgstr    = "f0convergence_Tg%.2E.png" %Tg[0]
# titlestr = "Temperature = %.2E K" %(Tg[0])
# ax1.set_title(titlestr, fontsize=24)

# fig1.savefig(imgstr,bbox_inches='tight')

# # CONVERGENCE FOR F1
# fig2, ax2 = plt.subplots(nrows=1, figsize=(10,10))
# fig2.patch.set_facecolor('white')
# fig2.patch.set_alpha(1.0)

# ax2.set_xlabel(r'Energy (eV)', fontsize=28)
# ax2.set_ylabel(r'f1', fontsize=28)

# plt.xticks(fontsize=20); plt.yticks(fontsize=20); 
# ax2.tick_params(axis='y', colors='k')
# plt.locator_params(axis='x', nbins=10)

# ax2.semilogy(dat1[:,EV], dat1[:,F1], label="Nr = 127", color="blue", linewidth=2)
# ax2.semilogy(dat2[:,EV], dat2[:,F1], label="Nr = 256", color="red", linewidth=2)

# ax2.legend(fontsize=14, labelcolor='k')

# imgstr    = "f1convergence_Tg%.2E.png" %Tg[0]
# titlestr = "Temperature = %.2E K" %(Tg[0])
# ax2.set_title(titlestr, fontsize=24)

# fig2.savefig(imgstr,bbox_inches='tight')