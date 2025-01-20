import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
from scipy.special import sph_harm
import scipy.constants
from matplotlib.colors import LogNorm

def gauss_legendre_lobatto(n):
    assert n>2, "quadrature order is less than 2"
    xi, wi = np.zeros(n), np.zeros(n)
    
    c1      = np.zeros(n)
    c1[n-1] = 1.0
    
    c1_x      = np.polynomial.legendre.legder(c1, 1)
    xi[1:-1]  = np.polynomial.legendre.legroots(c1_x)
    
    xi[0]     = -1
    xi[-1]    = 1
    
    wi[0]     = 2/n/(n-1)
    wi[-1]    = 2/n/(n-1)
    
    wi[1:-1]  = 2/n/(n-1)/(np.polynomial.legendre.legval(xi[1:-1], c1)**2)
    
    return xi, wi

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

def quad_on_grid(vr, vt, qoi):
    '''
    Note assumes, qoi axis (0-time, 1-vr, 2-vt)
    '''
    assert qoi.shape[1] == len(vr)
    assert qoi.shape[2] == len(vt)

    # evaluate v-space quad on v-space grid
    # i - denotes vr cell index. 
    # j - denotes vt cell index. 

    dvr  = vr[1:] - vr[0:-1]
    dvt  = vt[1:] - vt[0:-1]

    iqoi = np.zeros((qoi.shape[0], len(dvr), len(dvt)))
    iqoi = 0.5 * (qoi[:, 1:, :] * (vr[1:]**2)[np.newaxis, :, np.newaxis] + qoi[:, 0:-1, :] * (vr[0:-1]**2)[np.newaxis, :, np.newaxis]) * dvr[np.newaxis, :, np.newaxis]
    iqoi = 0.5 * (iqoi[:, :, 1:] * np.sin(vt[1:]) + iqoi[:, :, 0:-1] * np.sin(vt[0:-1])) * dvt * np.pi * 2

    return iqoi

def extract_data(folder, xl, xr, num_vt=16, ev_cutoff=80):
    args   = load_run_args(folder)
    num_l  = int(args["l_max"]) + 1
    
    # vtheta = np.linspace(0, np.pi, 64)
    # vthetaw= trapz_w(vtheta)

    cvtq,vtqw = gauss_legendre_lobatto(num_vt) #np.polynomial.legendre.leggauss(num_vt)
    cvtq,vtqw = np.flip(cvtq), np.flip(vtqw)
    vtq       = np.arccos(cvtq)
    #print(vtq)
    
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
    c_gamma   = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))
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
    eedf_const= 2 * (vth/c_gamma)**3  / mm_fac  
    cos_vt    = cvtq
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
    
    flx           = eedf_const * flx

    flx_l         = flx[:, 0, :, :]
    flx_r         = flx[:, 1, :, :]

    Vsh           = sph_vander(vtq, num_l)

    flx_l         = np.swapaxes(np.einsum("iam,al->ilm", flx_l, Vsh), 1, 2)
    flx_r         = np.swapaxes(np.einsum("iam,al->ilm", flx_r, Vsh), 1, 2)

    mx_l          = 2 * np.pi * np.dot(np.dot(flx_l, vtqw) * gmx**2, gmw) 
    mx_r          = 2 * np.pi * np.dot(np.dot(flx_r, vtqw) * gmx**2, gmw)

    flx_l         = np.einsum("t,tar->tar",(1 / mx_l), flx_l)
    flx_r         = np.einsum("t,tar->tar",(1 / mx_r), flx_r)
    
    Fv_l          = flx_l * (nex[:, 0])[:, np.newaxis, np.newaxis]
    Fv_r          = flx_r * (nex[:, 1])[:, np.newaxis, np.newaxis]

    # norm_cl       = 2 * np.pi * np.dot(np.dot(Fv_l, vtqw) * gmx**2, gmw) /nex[:, 0]
    # norm_cr       = 2 * np.pi * np.dot(np.dot(Fv_r, vtqw) * gmx**2, gmw) /nex[:, 1]

    # for tidx in range(0, len(tt),10):
    #     print("time = %.2E norm_const_left = %.8E norm_const_right = %.8E  with ev cutoff = %.2E"%(tt[tidx],norm_cl[tidx], norm_cr[tidx], ev_cutoff))

    
    Sv_l          = np.einsum("a,b,tab->tab", gmx * vth, cos_vt, flx_l)
    Sv_r          = np.einsum("a,b,tab->tab", gmx * vth, cos_vt, flx_r)


    a1_l          = 2 * np.pi * np.dot(np.dot(Sv_l, vtqw) * gmx**2, gmw) * nex[:, 0]
    a1_r          = 2 * np.pi * np.dot(np.dot(Sv_r, vtqw) * gmx**2, gmw) * nex[:, 1]

    a2_l          = neuzx[:, 0]
    a2_r          = neuzx[:, 1]

    norm_cl       = (a2_l/a1_l)
    norm_cr       = (a2_r/a1_r)

    # for tidx in range(0, len(tt),10):
    #     print("time = %.2E norm_const_left = %.8E norm_const_right = %.8E  with ev cutoff = %.2E"%(tt[tidx],norm_cl[tidx], norm_cr[tidx], ev_cutoff))

    Sv_l          = np.einsum("t,tar->tar", (a2_l/a1_l) * nex[:, 0], Sv_l)
    Sv_r          = np.einsum("t,tar->tar", (a2_r/a1_r) * nex[:, 1], Sv_r)

    a1_l          = 2 * np.pi * np.dot(np.dot(Sv_l, vtqw) * gmx**2, gmw) 
    a1_r          = 2 * np.pi * np.dot(np.dot(Sv_r, vtqw) * gmx**2, gmw) 

    # for tidx in range(0, len(tt), 10):
    #     print("a1_l = %.8E a2_l = %.8E a1_r=%.8E a2_r=%.8E"%(a1_l[tidx], a2_l[tidx], a1_r[tidx], a2_r[tidx]))

    return {"tt": tt, 
            "ev": evgrid, 
            "vr": vr/vth, 
            "vr_weights":gmw,
            "vtheta": vtq,
            "vtheta_weights": vtqw,
            "time": tt,
            "fv_left_bdy": Fv_l,
            "fv_right_bdy": Fv_r,
            "flux_left_bdy": Sv_l,
            "flux_right_bdy": Sv_r,
            "Ef": Ef, 
            "vth": vth,
            "xx":xx,
            "xlidx":xlidx,
            "xridx":xridx,
            "uz": uz,
            "ne": ne}
    
def plot_data(data, tidx):
    c_gamma       = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))
    vth           = data["vth"] 
    vr            = data["vr"]
    vtheta        = data["vtheta"]
    Sv_l          = data["flux_left_bdy"] 
    Sv_r          = data["flux_right_bdy"] 

    evgrid        = (vr * vth)**2 /c_gamma**2
    iFvt_L        = quad_on_grid(vr, vtheta, Sv_l)
    iFvt_R        = quad_on_grid(vr, vtheta, Sv_r)
    iS            = (iFvt_R - iFvt_L)

    
    dvt           = 0.5 * (vtheta[1] - vtheta[0])
    dev           = 0.5 * (evgrid[1] - evgrid[0])
    extent        = [evgrid[0]-dev , evgrid[-1] + dev, vtheta[0]-dvt, vtheta[-1] + dvt]

    plt.figure(figsize=(12, 6), dpi=200)
    plt.subplot(2, 3, 1)
    plt.imshow(np.abs(Sv_l[tidx].T), aspect='auto', extent=extent, norm=LogNorm(vmin=1e11, vmax=1e21))
    plt.title(r"abs($v\cos(v_{\theta})f(x_L, v, v_{\theta})$)")
    #plt.imshow(Sv_l[0], aspect='auto', extent=extent)
    #plt.title(r"$v\cos(v_{\theta})f(x_L, v, v_{\theta})$")
    plt.colorbar()
    plt.xlabel(r"energy (eV)")
    plt.ylabel(r"$v_{\theta}$")
    

    plt.subplot(2, 3, 2)
    plt.imshow(np.abs(Sv_r[tidx].T), aspect='auto', extent=extent, norm=LogNorm(vmin=1e11, vmax=1e21))
    plt.title(r"abs($v\cos(v_{\theta})f(x_R, v, v_{\theta})$)")
    #plt.imshow(Sv_r[0], aspect='auto', extent=extent)
    #plt.title(r"$v\cos(v_{\theta})f(x_R, v, v_{\theta})$")
    plt.colorbar()
    plt.xlabel(r"energy (eV)")
    plt.ylabel(r"$v_{\theta}$")

    plt.subplot(2, 3, 3)
    plt.imshow(np.abs(Sv_r[tidx].T-Sv_l[tidx].T), aspect='auto', extent=extent, norm=LogNorm(vmin=1e10, vmax=1e19))
    plt.title(r"abs(net flux)")
    #plt.imshow((Sv_r[0] - Sv_l[0]), aspect='auto', extent=extent)
    #plt.title(r"net flux")
    plt.colorbar()
    plt.xlabel(r"energy (eV)")
    plt.ylabel(r"$v_{\theta}$")

    plt.subplot(2, 3, 4)
    plt.imshow((iFvt_L[tidx].T), aspect='auto', extent=extent)
    plt.colorbar()
    plt.title(r"$\int_{\Omega_v}v\cos(v_{\theta})f(x_L, v, v_{\theta}) d \Omega_v$")
    plt.xlabel(r"energy (eV)")
    plt.ylabel(r"$v_{\theta}$")

    plt.subplot(2, 3, 5)
    plt.imshow((iFvt_R[tidx].T), aspect='auto', extent=extent)
    plt.colorbar()
    plt.title(r"$\int_{\Omega_v}v\cos(v_{\theta})f(x_L, v, v_{\theta}) d \Omega_v$")
    plt.xlabel(r"energy (eV)")
    plt.ylabel(r"$v_{\theta}$")

    plt.subplot(2, 3, 6)
    plt.imshow((iS[tidx].T), aspect='auto', extent=extent)
    plt.title(r"$\int_{\Omega_v} \text{net flux} d \Omega_v$")
    plt.colorbar()
    plt.xlabel(r"energy (eV)")
    plt.ylabel(r"$v_{\theta}$")
    
    plt.suptitle("time = %.4E T"%(data["tt"][tidx]))
    plt.tight_layout()
    
    plt.show()
    plt.close()    

data   = extract_data(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), num_vt=32, ev_cutoff=30)
tt     = data["tt"]
ev     = data["ev"]
vr     = data["vr"]
vr_w   = data["vr_weights"]
vtheta = data["vtheta"]
vt_w   = data["vtheta_weights"]
tt     = data["time"]
Fv_L   = data["fv_left_bdy"]
Fv_R   = data["fv_right_bdy"]
Flux_L = data["flux_left_bdy"] 
Flux_R = data["flux_right_bdy"] 
Ef     = data["Ef"]
xlidx  = data["xlidx"]
xridx  = data["xridx"]
xx     = data["xx"]
uz     = data["uz"]
ne     = data["ne"]
dx     = xx[xridx] - xx[xlidx]
L      = 2.54e-2 / 2 

print("\
r1      = ne[tidx, xlidx] * uz[tidx, xlidx]\n\
a1      = 2*np.pi * np.dot(np.dot(Fvt_L[tidx] * vr**2 , vr_w) * np.sin(vtheta), vt_w)\n\
r2      = ne[tidx, xridx] * uz[tidx, xridx] \n\
a2      = 2*np.pi * np.dot(np.dot(Fvt_R[tidx] * vr**2 , vr_w) * np.sin(vtheta), vt_w) \n\
S       = (a2-a1)\n\
R       = (r2-r1)")

#### Example on how to do the cell-wise quad. 
# this will do the cell wise quad. for all time points, 
# iFvt_R - (axis-0 is time, axis-1 vr_i, axis-2, vtheta_j) where vr_i in [0, len(vr)-1] vtheta_j in [0, len(vtheta)-1]
iFlux_L  = quad_on_grid(vr, vtheta, Flux_L)
iFlux_R  = quad_on_grid(vr, vtheta, Flux_R)

iFv_L    = quad_on_grid(vr, vtheta, Fv_L)
iFv_R    = quad_on_grid(vr, vtheta, Fv_R)
iFv_ic   = 0.5 * (iFv_L + iFv_R)
#print(np.sum(iFv_L[0]), ne[0,xlidx])




plot_data(data, tidx=0)
for tidx in range(0, len(tt), 10):
    # left boundary
    r1      = ne[tidx, xlidx] * uz[tidx, xlidx]
    a1      = 2*np.pi * np.dot(np.dot(Flux_L[tidx], vt_w) * vr**2, vr_w)
    b1      = np.sum(iFlux_L[tidx])
    
    # right boundary
    r2      = ne[tidx, xridx] * uz[tidx, xridx] 
    a2      = 2*np.pi * np.dot(np.dot(Flux_R[tidx], vt_w) * vr**2, vr_w) 
    b2      = np.sum(iFlux_R[tidx])

    S       = (a2-a1)
    R       = (r2-r1)


    print("np.abs(1-b1/a1) = %.4E , np.abs(1-b2/a2) = %.4E, a1 = %.8E r1 = %.8E a2 = %.8E r2 = %.8E S=%.8E R=%.8E np.abs(1-S/R) = %.8E"%(np.abs(1-b1/a1), np.abs(1-b1/a2), a1, r1, a2, r2, S, R, np.abs(1-R/S)))
    #print("t = %.4E [s] at left ||uzne - int(fvtL)||/||uzne|| = %.8E  and right ||uzne - int(fvtL)||/||uzne|| = %.8E net flux relative error=%.8E "%(tt[tidx], np.abs(1-a1/r1), np.abs(1-a2/r2), np.abs(1-(S/R))))

    
    
    








