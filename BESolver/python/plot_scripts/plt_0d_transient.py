import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

# file to process
fname = sys.argv[1]

with h5py.File(fname) as F:

    print("processing - ", fname)
    print(F.keys())
    tt = np.array(F["time[s]"][()])
    f0 = np.array(F["f0[eV^-1.5]"][()])
    f1 = np.array(F["f1[eV^-1.5]"][()])
    ev = np.array(F["evgrid[eV]"][()])

    
    De   = np.array(F["D[m^2s^{-1}]"][()])
    mueE = np.array(F["mu[ms^{-1}]"][()])
    Te   = np.array(F["Te[eV]"][()])
    C1   = np.array(F["C1[m^3s^-1]"][()])
    C2   = np.array(F["C2[m^3s^-1]"][()])
    
    plt.figure(figsize=(10, 8), dpi=200)
    plt.subplot(3, 2, 1)
    plt.semilogy(ev, np.abs(f0[0::10, :].T))
    plt.xlabel(r"energy (eV)")
    plt.ylabel(r"$f_0$ ($eV^{-3/2}$)")
    plt.grid(visible=True)

    plt.subplot(3, 2, 2)
    plt.semilogy(ev, np.abs(f1[0::10, :].T))
    plt.xlabel(r"energy (eV)")
    plt.ylabel(r"$f_1$ ($eV^{-3/2}$)")
    plt.grid(visible=True)

    plt.subplot(3, 2, 3)
    plt.plot(tt, C1)
    plt.xlabel(r"time [s]")
    plt.ylabel(r"rate coefficient [$m^3s^{-1}$]")
    plt.title(r"Momentum transfer [$e + Ar \rightarrow e + Ar$]")
    plt.grid(visible=True)

    plt.subplot(3, 2, 4)
    plt.semilogy(tt, C2)
    plt.xlabel(r"time [s]")
    plt.ylabel(r"rate coefficient [$m^3s^{-1}$]")
    plt.title(r"Ionization [$e + Ar \rightarrow 2e + Ar^+$]")
    plt.grid(visible=True)

    plt.subplot(3, 2, 5)
    plt.plot(tt, Te)
    plt.xlabel(r"time [s]")
    plt.ylabel(r"temperature [eV]")
    plt.grid(visible=True)

    plt.subplot(3, 2, 6)
    plt.plot(tt, De)
    plt.xlabel(r"time [s]")
    plt.ylabel(r"$D_e$ [$m^2s^{-1}$]")
    plt.grid(visible=True)

    # plt.semilogy(tt, np.abs(mueE))
    # plt.xlabel(r"time [s]")
    # plt.ylabel(r"abs($\mu_e E$) [ms^{-1}]")
    # plt.grid(visible=True)

    plt.tight_layout()
    plt.show()
    F.close()