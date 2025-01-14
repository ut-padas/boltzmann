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

def time_average(qoi, tt):
    """
    computes the time average on grids
    """
    # check if tt is uniform
    nT   = len(tt)
    T    = (tt[-1]-tt[0])
    dt   = T/(nT-1)
    
    assert abs(tt[1] -tt[0] - dt) < 1e-10
    
    tw    = np.ones_like(tt) * dt
    tw[0] = 0.5 * tw[0]; tw[-1] = 0.5 * tw[-1];
    
    assert np.abs(T-np.sum(tw)) < 1e-12
    return np.dot(tw, qoi)

def trapz_w(qx):
    # check if tt is uniform
    nx   = len(qx)
    dx   = qx[1:]-qx[0:-1]
    
    qw   = np.array([(dx[i-1] + dx[i]) * 0.5 for i in range(1, nx-1)])
    qw   = np.append(np.array([0.5 * dx[0]]), qw)
    qw   = np.append(qw, np.array([0.5 * dx[-1]]))
    assert np.abs((qx[-1] - qx[0])-np.sum(qw)) < 1e-12
    return qw

def extract_flux_and_E(folder, xl, xr, ev_cutoff=80):
    args   = load_run_args(folder)
    num_l  = int(args["l_max"]) + 1
    vtheta = np.linspace(0, np.pi, 64)
    ff     = h5py.File("%s/macro.h5"%(folder), "r")
    print(ff.keys())
    
    xx        = np.array(ff["x[-1,1]"][()])
    tt        = np.array(ff["time[T]"][()])
    ne        = np.array(ff["ne[m^-3]"][()])
    ni        = np.array(ff["ni[m^-3]"][()])
    fl        = np.array(ff["fl[eV^-1.5]"][()])
    Ef        = np.array(ff["E[Vm^-1]"][()])
    evgrid    = np.array(ff["evgrid[eV]"][()])
    ki        = np.array(ff["ki[m^3s^{-1}]"][()])
    ip_avg    = np.array(ff["avg_ion_prod[m^-3s^{-1}]"][()])
    uz        = np.array(ff["uz[ms^{-1}]"][()])
    ux        = np.array(ff["ux[ms^{-1}]"][()])
    L         = (2.54e-2 / 2)
    tau       = 1/13.56e6
    c_gamma       = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))
    # from : https://github.com/ut-padas/boltzmann/blob/main/BESolver/python/qoi_process.py - compute_radial_components
    vth           = (float) (args["Te"])**0.5 * c_gamma
    mm_fac        = np.sqrt(4 * np.pi) 

    neuz      = ne * uz 
    neuz_avg  = time_average(neuz, tt)
    ne_avg    = time_average(ne, tt)
    ni_avg    = time_average(ni, tt)
    E_avg     = time_average(Ef, tt)

    charge_density = ((ni_avg - ne_avg) / ni_avg)
    sheath_tol     = 1e-3
    sidx           = np.argmin(np.abs(charge_density -sheath_tol))
    print("sheath begin for (ne-ni)/ni rtol = %.4E sheath = %.4E, E_avg = %.4E [V/m]"%(sheath_tol, xx[sidx], E_avg[sidx]))
    
    idx       = evgrid<ev_cutoff
    evgrid    = evgrid[idx]
    fl        = fl[:, :, :, idx]
    Np        = len(xx)
    deg       = Np-1
    xp        = -np.cos(np.pi*np.linspace(0,deg,Np)/deg)
    vr        = np.sqrt(evgrid) * c_gamma 
    gmx       = vr/vth
    gmw       = trapz_w(gmx)
    evw       = trapz_w(evgrid)
    vthetaw   = trapz_w(vtheta)
    eedf_const= 2 * (vth/c_gamma)**3  / mm_fac  
    cos_vt    = np.cos(vtheta)
    assert np.linalg.norm(xp-xx)/np.linalg.norm(xx) < 1e-14, "Chebyshev point mismatch found"
    

    V0p = np.polynomial.chebyshev.chebvander(xx, deg)
    # V0pinv: xp values to coefficients
    ident  = np.eye(Np)
    V0pinv = np.linalg.solve(V0p, ident)

    xlidx  = np.argmin(np.abs(xx-xl))
    xridx  = np.argmin(np.abs(xx-xr))

    print(xl, xx[xlidx], xr, xx[xridx])

    dx      = (xx[xridx] - xx[xlidx]) * L
    dne     = tau * np.abs(0.5 * dx * (ip_avg[xlidx] + ip_avg[xridx]) - (neuz_avg[xridx] - neuz_avg[xlidx])) / (0.5 * dx * (ne_avg[xlidx] + ne_avg[xridx]))
    print("(prod - flux)/ne_cell = %.8E"%(dne))
    
    #print(ne[-1,:] - ne[0,:])
    #plt.semilogy(xx, np.abs(ne[0,:]-ne[-1,:])/np.max(ne[0,:]))
    # plt.semilogy(xx, np.abs(ne[0,:]-ne[-1,:])/np.abs(ne[0,:]))
    # #plt.semilogy(xx, ne[-1,:])
    # plt.show()

    #xnew   = np.array([xl, xr])
    xnew   = np.array([xx[xlidx], xx[xridx]])
    P1     = np.dot(np.polynomial.chebyshev.chebvander(xnew, deg), V0pinv)

    flx    = np.einsum("il,albc->aibc", P1, fl)
    nex    = np.einsum("il,al->ai"    , P1, ne)
    neuzx  = np.einsum("il,al->ai"    , P1, neuz)
    
    #scale        = np.array([np.dot(mm_op / mm_fac, v_lm[idx]) * (2 * (vth/c_gamma)**3) for idx in range(n_pts)])
    flx           = eedf_const * flx

    flx_l         = flx[:, 0, :, :]
    flx_r         = flx[:, 1, :, :]

    Vsh           = sph_vander(vtheta, num_l)

    flx_l         = np.einsum("iam,al->ilm", flx_l, Vsh)
    flx_r         = np.einsum("iam,al->ilm", flx_r, Vsh)

    mx_l          = 2 * np.pi * np.dot(np.dot(flx_l * gmx**2 , gmw) * np.sin(vtheta), vthetaw) 
    mx_r          = 2 * np.pi * np.dot(np.dot(flx_r * gmx**2 , gmw) * np.sin(vtheta), vthetaw)

    flx_l         = np.einsum("t,tar->tar",(1 / mx_l), flx_l)
    flx_r         = np.einsum("t,tar->tar",(1 / mx_r), flx_r)
    #print(flx_r.shape)
    

    dvt           = 0.5 * (vtheta[1] - vtheta[0])
    dev           = 0.5 * (evgrid[1] - evgrid[0])
    dvr           = 0.5 * (vr[1] - vr[0])
    extent        = [evgrid[0]-dev , evgrid[-1] + dev, vtheta[0]-dvt, vtheta[-1] + dvt]
    #extent        = [vr[0]-dvr , vr[-1] + dvr, vtheta[0]-dvt, vtheta[-1] + dvt]
    Sv_l          = np.einsum("a,b,tba->tba", gmx * vth, cos_vt, flx_l)
    Sv_r          = np.einsum("a,b,tba->tba", gmx * vth, cos_vt, flx_r)


    a1_l          = 2 * np.pi * np.dot(np.dot(Sv_l * gmx**2 , gmw) * np.sin(vtheta), vthetaw) * nex[:, 0]
    a1_r          = 2 * np.pi * np.dot(np.dot(Sv_r * gmx**2 , gmw) * np.sin(vtheta), vthetaw) * nex[:, 1]

    a2_l          = neuzx[:, 0]
    a2_r          = neuzx[:, 1]

    Sv_l          = np.einsum("t,tar->tar", (a2_l/a1_l) * nex[:, 0], Sv_l)
    Sv_r          = np.einsum("t,tar->tar", (a2_r/a1_r) * nex[:, 1], Sv_r)

    a1_l          = 2 * np.pi * np.dot(np.dot(Sv_l * gmx**2 , gmw) * np.sin(vtheta), vthetaw) 
    a1_r          = 2 * np.pi * np.dot(np.dot(Sv_r * gmx**2 , gmw) * np.sin(vtheta), vthetaw) 

    # for tidx in range(0, len(tt), 10):
    #     print("a1_l = %.8E a2_l = %.8E a1_r=%.8E a2_r=%.8E"%(a1_l[tidx], a2_l[tidx], a1_r[tidx], a2_r[tidx]))

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

    return {"tt": tt, "ev": evgrid, "vr": vr/vth, "vtheta": vtheta, "time": tt, "flux_left_bdy": Sv_l, "flux_right_bdy": Sv_r, "Ef": Ef, "vth": vth, "xx":xx, "xlidx":xlidx, "xridx":xridx, "uz": uz, "ne": ne}
    
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
tt     = data["tt"]
ev     = data["ev"]
vr     = data["vr"]
vtheta = data["vtheta"]
tt     = data["time"]
Fvt_L  = data["flux_left_bdy"] 
Fvt_R  = data["flux_right_bdy"] 
Ef     = data["Ef"]
xlidx  = data["xlidx"]
xridx  = data["xridx"]
xx     = data["xx"]
uz     = data["uz"]
ne     = data["ne"]
dx     = xx[xridx] - xx[xlidx]
L      = 2.54e-2 / 2 

vr_w   = trapz_w(vr)
vt_w   = trapz_w(vtheta)

print("\
r1      = ne[tidx, xlidx] * uz[tidx, xlidx]\n\
a1      = 2*np.pi * np.dot(np.dot(Fvt_L[tidx] * vr**2 , vr_w) * np.sin(vtheta), vt_w)\n\
r2      = ne[tidx, xridx] * uz[tidx, xridx] \n\
a2      = 2*np.pi * np.dot(np.dot(Fvt_R[tidx] * vr**2 , vr_w) * np.sin(vtheta), vt_w) \n\
S       = (a2-a1)\n\
R       = (r2-r1)")


for tidx in range(0, len(tt), 10):
    # left boundary
    r1      = ne[tidx, xlidx] * uz[tidx, xlidx]
    a1      = 2*np.pi * np.dot(np.dot(Fvt_L[tidx] * vr**2 , vr_w) * np.sin(vtheta), vt_w)
    
    # right boundary
    r2      = ne[tidx, xridx] * uz[tidx, xridx] 
    a2      = 2*np.pi * np.dot(np.dot(Fvt_R[tidx] * vr**2 , vr_w) * np.sin(vtheta), vt_w) 

    S       = (a2-a1)
    R       = (r2-r1)

    print("a1 = %.8E r1 = %.8E a2 = %.8E r2 = %.8E S=%.8E R=%.8E np.abs(1-S/R) = %.8E"%(a1, r1, a2, r2, S, R, np.abs(1-R/S)))
    #print("t = %.4E [s] at left ||uzne - int(fvtL)||/||uzne|| = %.8E  and right ||uzne - int(fvtL)||/||uzne|| = %.8E net flux relative error=%.8E "%(tt[tidx], np.abs(1-a1/r1), np.abs(1-a2/r2), np.abs(1-(S/R))))

    
    








