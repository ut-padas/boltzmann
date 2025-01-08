import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
from scipy.special import sph_harm
import scipy.constants
from matplotlib.colors import LogNorm

def load_run_args(folder):
    args   = dict()
    
    f  = open(folder+"/1d_glow_args.txt")
    st = f.read().strip()
    st = st.split(":")[1].strip()
    st = st.split("(")[1]
    st = st.split(")")[0].strip()
    st = st.split(",")

    for s in st:
        kv = s.split("=")
        if len(kv)!=2 or kv[0].strip()=="collisions":
            continue
        args[kv[0].strip()]=kv[1].strip().replace("'", "")
    return args

def sph_harm_real(l, m, theta, phi):
    # in python's sph_harm phi and theta are swapped
    Y = sph_harm(abs(m), l, phi, theta)
    if m < 0:
        Y = np.sqrt(2) * (-1)**m * Y.imag
    elif m > 0:
        Y = np.sqrt(2) * (-1)**m * Y.real
    else:
        Y = Y.real

    return Y 

def sph_vander(theta, num_l):
    Yl = lambda l, theta : sph_harm(abs(0), l, 0.0, theta).real
    V  = np.zeros((num_l, len(theta)))
    for l in range(num_l):
        V[l,:] = Yl(l, theta)

    return V
def extract_flux_and_E(folder, xl, xr):
    args   = load_run_args(folder)
    num_l  = int(args["l_max"]) + 1
    vtheta = np.linspace(0, np.pi, 64)
    ff     = h5py.File("%s/macro.h5"%(folder), "r")
    #print(ff.keys())
    
    xx        = np.array(ff["x[-1,1]"][()])
    tt        = np.array(ff["time[T]"][()])
    ne        = np.array(ff["ne[m^-3]"][()])
    fl        = np.array(ff["fl[eV^-1.5]"][()])
    Ef        = np.array(ff["E[Vm^-1]"][()])
    evgrid    = np.array(ff["evgrid[eV]"][()])
    idx       = evgrid<20
    evgrid    = evgrid[idx]
    fl        = fl[:, :, :, idx]
    Np        = len(xx)
    deg       = Np-1
    xp        = -np.cos(np.pi*np.linspace(0,deg,Np)/deg)
    assert np.linalg.norm(xp-xx)/np.linalg.norm(xx) < 1e-14, "Chebyshev point mismatch found"
    

    V0p = np.polynomial.chebyshev.chebvander(xx, deg)
    # V0pinv: xp values to coefficients
    ident  = np.eye(Np)
    V0pinv = np.linalg.solve(V0p, ident)

    xlidx  = np.argmin(np.abs(xx-xl))
    xridx  = np.argmin(np.abs(xx-xr))

    print(xl, xx[xlidx], xr, xx[xridx])

    #xnew   = np.array([xl, xr])
    xnew   = np.array([xx[xlidx], xx[xridx]])
    P1     = np.dot(np.polynomial.chebyshev.chebvander(xnew, deg), V0pinv)

    flx    = np.einsum("il,albc->aibc", P1, fl)
    nex    = np.einsum("il,al->ai"    , P1, ne)

    c_gamma       = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))
    # from : https://github.com/ut-padas/boltzmann/blob/main/BESolver/python/qoi_process.py - compute_radial_components
    vth           = (float) (args["Te"])**0.5 * c_gamma
    mm_fac        = np.sqrt(4 * np.pi) 
    #scale        = np.array([np.dot(mm_op / mm_fac, v_lm[idx]) * (2 * (vth/c_gamma)**3) for idx in range(n_pts)])
    norm_const    = nex * 2 * (vth/c_gamma)**3  / mm_fac  
    flx           = np.einsum("ik,iklm->iklm", norm_const, flx)

    flx_l         = flx[:, 0, :, :]
    flx_r         = flx[:, 1, :, :]

    Vsh           = sph_vander(vtheta, num_l)

    flx_l         = np.einsum("iam,al->ilm", flx_l, Vsh)
    flx_r         = np.einsum("iam,al->ilm", flx_r, Vsh)
    vr            = np.sqrt(evgrid) * c_gamma 
    cos_vt        = np.cos(vtheta)

    dvt           = 0.5 * (vtheta[1] - vtheta[0])
    dev           = 0.5 * (evgrid[1] - evgrid[0])
    dvr           = 0.5 * (vr[1] - vr[0])
    extent        = [evgrid[0]-dev , evgrid[-1] + dev, vtheta[0]-dvt, vtheta[-1] + dvt]
    #extent        = [vr[0]-dvr , vr[-1] + dvr, vtheta[0]-dvt, vtheta[-1] + dvt]
    Sv_l          = np.einsum("a,b,tba->tba", vr, cos_vt, flx_l)
    Sv_r          = np.einsum("a,b,tba->tba", vr, cos_vt, flx_r)


    plt.figure(figsize=(12, 4), dpi=200)
    plt.subplot(1, 3, 1)
    #plt.imshow(np.abs(Sv_l[0]), aspect='auto', extent=extent, norm=LogNorm(vmin=1e16, vmax=1e21))
    #plt.title(r"abs($v\cos(v_{\theta})f(x_L, v, v_{\theta})$)")
    plt.imshow(Sv_l[0], aspect='auto', extent=extent)
    plt.title(r"$v\cos(v_{\theta})f(x_L, v, v_{\theta})$")
    plt.colorbar()
    plt.xlabel(r"energy (eV)")
    plt.ylabel(r"$v_{\theta}$")
    

    plt.subplot(1, 3, 2)
    #plt.imshow(np.abs(Sv_r[0]), aspect='auto', extent=extent, norm=LogNorm(vmin=1e16, vmax=1e21))
    #plt.title(r"abs($v\cos(v_{\theta})f(x_R, v, v_{\theta})$)")
    plt.imshow(Sv_r[0], aspect='auto', extent=extent)
    plt.title(r"$v\cos(v_{\theta})f(x_R, v, v_{\theta})$")
    plt.colorbar()
    plt.xlabel(r"energy (eV)")
    plt.ylabel(r"$v_{\theta}$")

    plt.subplot(1, 3, 3)
    plt.imshow((Sv_r[0] - Sv_l[0]), aspect='auto', extent=extent)
    plt.title(r"net flux")
    plt.colorbar()
    plt.xlabel(r"energy (eV)")
    plt.ylabel(r"$v_{\theta}$")
    

    plt.tight_layout()
    plt.show()

    return {"ev": evgrid, "vr": vr/vth, "vtheta": vtheta, "time": tt, "flux_left_bdy": Sv_l, "flux_right_bdy": Sv_r, "Ef": Ef, "vth": vth}
    
    # flx_l         = flx[:, 0, :, :]
    # flx_r         = flx[:, 1, :, :]
    # plt.subplot(2, 1, 1)
    # plt.semilogy(evgrid, np.abs(flx_r[0::10, 0, :].T))
    # plt.grid(visible=True)
    # plt.xlabel(r"energy [eV]")
    # plt.ylabel(r"$f_0$  [$eV^{-3/2}$]")
    # plt.title(r"x=%.4E"%xr)

    # plt.subplot(2, 1, 2)
    # plt.semilogy(evgrid, np.abs(flx_l[0::10, 0, :].T))
    # plt.grid(visible=True)
    # plt.xlabel(r"energy [eV]")
    # plt.ylabel(r"$f_0$  [$eV^{-3/2}$]")
    # plt.title(r"x=%.4E"%xl)
    # plt.show()
    

data   = extract_flux_and_E(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))
ev     = data["ev"]
vr     = data["vr"]
vtheta = data["vtheta"]
tt     = data["time"]
Fvt_L  = data["flux_left_bdy"] 
Fvt_R  = data["flux_right_bdy"] 
Ef     = data["Ef"]




