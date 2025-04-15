import matplotlib
#matplotlib.use('tkagg')  # Or any other X11 back-end
import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
from scipy.special import sph_harm
import scipy.constants
from matplotlib.colors import LogNorm
from matplotlib.colors import Colormap
import os

def make_dir(dir_name):
    # Check whether the specified path exists or not
    isExist = os.path.exists(dir_name)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(dir_name)
       print("directory %s is created!"%(dir_name))

def gauss_legendre_lobatto(n, domain=(-1, 1)):
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

    a, b      = domain[0], domain[1]
    
    xi        = 0.5 * (b-a) * xi + 0.5 * (b+a)
    wi        = 0.5 * (b-a) * wi
    
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

def vr_coarsen(vr, vr_p):
    num_vr       = len(vr)
    num_vr_cells = (num_vr -1) // (vr_p-1) +1 
    num_vr0      = num_vr_cells * (vr_p-1) + 1
    
    
    vrw          = np.zeros((num_vr_cells , vr_p))
    vr_e2n       = np.array([range(i, num_vr0 - (vr_p-1-i), vr_p-1) for i in range(vr_p)]).T
    vr_e2n[vr_e2n>=num_vr]= num_vr-1

    vr_idx_width = vr_e2n[:, -1] - vr_e2n[:, 0]
    #vr_idx_width[vr_idx_width>0]
    #print(vr_idx_width.shape, (vr_idx_width>0).shape, vr_e2n.shape)
    vr_e2n       = vr_e2n[vr_idx_width>0, :]
    #print(vr_e2n.shape)

    vr1          = np.array([vr[vr_e2n[i]]   for i in range(vr_e2n.shape[0])])
    vrw1         = np.array([trapz_w(vr1[i]) for i in range(vr_e2n.shape[0])])

    a1 = np.sum(vrw1,axis=1)
    a2 = (vr1[:, -1] - vr1[:, 0])
    assert np.linalg.norm(a1-a2)/np.linalg.norm(a1) < 1e-14

    return vr1.reshape((-1)), vrw1.reshape((-1)), vr_e2n


def grid_to_cell(vr, vrw, vr_e2n, vt, vtw, qoi, vr_p, vt_p):
    
    vr       = vr.reshape((-1, vr_p))
    vrw      = vrw.reshape((-1, vr_p))
    vt       = vt.reshape((-1, vt_p))
    vtw      = vtw.reshape((-1, vt_p))

    nbins_vr = vr.shape[0]
    nbins_vt = vt.shape[0]
    num_vt   = nbins_vt + 1

    iqoi     = np.array([qoi[:, vr_e2n[i], :] for i in range(nbins_vr)])
    iqoi     = np.swapaxes(iqoi, 0, 1).reshape((qoi.shape[0], nbins_vr * vr_p, qoi.shape[2]))
    #iqoi     = 2 * np.pi * np.dot(np.dot(iqoi, vtw), vrw)

    iqoi     = np.pi * 2 * np.array([ np.dot(np.dot((vr[vr_idx])[np.newaxis, :, np.newaxis]**2 *
                                      iqoi[:, vr_idx * vr_p : (vr_idx+1) * vr_p, vt_idx * vt_p : (vt_idx+1) * vt_p ], vtw[vt_idx]), vrw[vr_idx])
                                      for vr_idx in range(nbins_vr) for vt_idx in range(nbins_vt)]).T

    iqoi     = iqoi.reshape((qoi.shape[0], nbins_vr , nbins_vt))    
    



    #print(iqoi.shape, qoi.shape)
    #sys.exit(0)

    # iqoi  = np.array([ 0.5 * (vr[vr_idx+1]-vr[vr_idx]) * (vr[vr_idx]**2   * np.dot(qoi[:, vr_idx  , vt_idx * vt_p : (vt_idx+1) * vt_p], vtw[vt_idx]) + 
    #                                                         vr[vr_idx+1]**2 * np.dot(qoi[:, vr_idx+1, vt_idx * vt_p : (vt_idx+1) * vt_p], vtw[vt_idx])) 
    #                                                     for vr_idx in range(nbins_vr) for vt_idx in range(nbins_vt)]).T

    # iqoi  = 2 * np.pi * iqoi.reshape((qoi.shape[0], nbins_vr, nbins_vt))    
    return iqoi

def vt_coarsen(vr, vt, iqoi):
    """
    coarsening the vt grid - only vt coarsening
    """
    assert len(vt)-1 == iqoi.shape[2]
    idx1  = np.argmin(np.abs(vt-np.pi/2)) + 1
    idx2  = idx1

    assert np.abs(vt[idx1-1] - 0.5 * np.pi)/(np.pi/2) < 1e-14
    assert np.abs(vt[idx2]   - 0.5 * np.pi)/(np.pi/2) < 1e-14

    # coarsen vt grid
    vt_c = np.append(np.array(vt[0:idx1])[0::2], np.array(vt[idx2:])[0::2])
    
    # for i in range(0, idx1-1, 2):
    #     print(i, i+1)
    
    # for i in range(idx2, len(vt)-1, 2):
    #     print(i, i+1)

    a1   = np.array([iqoi[:, :, i] + iqoi[:, :, i+1]  for i in range(0, idx1-1, 2)])
    a2   = np.array([iqoi[:, :, i] + iqoi[:, :, i+1]  for i in range(idx2, len(vt)-1, 2)])

    iqoi_c = np.append(a1, np.zeros((1, iqoi.shape[0], iqoi.shape[1])), axis=0)
    iqoi_c = np.append(iqoi_c, a2, axis=0)
    iqoi_c = (iqoi_c.reshape((iqoi_c.shape[0], -1)).T).reshape((iqoi.shape[0], iqoi.shape[1], -1))
    return vt_c, iqoi_c

def extract_data(folder, xl, xr, vt_k= 9, ev_cutoff=80, fname="macro.h5"):
    args   = load_run_args(folder)
    num_l  = int(args["l_max"]) + 1

    assert vt_k %2 == 1
    num_vt = 2 * vt_k 
    
    # vtheta = np.linspace(0, np.pi, 64)
    # vthetaw= trapz_w(vtheta)

    # cvtq,vtqw = gauss_legendre_lobatto(num_vt, domain=(-1, 1)) 
    # cvtq,vtqw = np.flip(cvtq), np.flip(vtqw)
    # vtq       = np.arccos(cvtq)
    # print(vtq)

    cvtq,vtqw     = gauss_legendre_lobatto(num_vt//2, domain=(-1, 0)) 
    cvtq_0,vtqw_0 = np.flip(cvtq), np.flip(vtqw)
    

    cvtq,vtqw     = gauss_legendre_lobatto(num_vt//2, domain=(0, 1)) 
    cvtq_1,vtqw_1 = np.flip(cvtq), np.flip(vtqw)
    
    
    cvtq          = np.append(cvtq_1 , cvtq_0)
    vtq           = np.arccos(cvtq)
    vtqw          = np.append(vtqw_1 , vtqw_0)

    #print(vtq)


    
    
    ff     = h5py.File("%s/%s"%(folder,fname), "r")
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

def extract_data_on_cell(folder, xl, xr, vr_p, num_vt, vt_p, ev_cutoff=80, fname="macro.h5"):
    args   = load_run_args(folder)
    num_l  = int(args["l_max"]) + 1

    assert vr_p > 1
    assert num_vt % 2 == 0
    
    nvt_by_2              = num_vt//2
    cvtq_0, _             = gauss_legendre_lobatto(num_vt//2, domain=(-1, 0)) 
    cvtq_1, _             = gauss_legendre_lobatto(num_vt//2, domain=(0, 1)) 
    gll_vt , gll_vt_w     = gauss_legendre_lobatto(vt_p, domain=(-1, 1))


    cvt_cell              = np.append(cvtq_0, cvtq_1)


    cvtq                  = np.array([gll_vt   * 0.5 * (cvt_cell[i+1] - cvt_cell[i]) + 0.5 * (cvt_cell[i+1] + cvt_cell[i]) for i in range(len(cvt_cell)-1)]).reshape((-1))  
    vtqw                  = np.array([gll_vt_w * 0.5 * (cvt_cell[i+1] - cvt_cell[i])                                       for i in range(len(cvt_cell)-1)]).reshape((-1))

    vtq                   = np.flip(np.acos(cvtq))
    cvtq                  = np.cos(vtq)
    svtq                  = np.sin(vtq)
    vtqw                  = np.flip(vtqw)

    # print(vtq)
    # print(vtq.reshape((-1,vt_p)))
    # print(cvtq.reshape((-1,vt_p)))
    # print(vtqw)
    # print(vtqw.reshape((-1,vt_p)))
    # print(np.sum(vtqw.reshape((-1,vt_p)), axis=1))
    # sys.exit(0)

    vt_cell               = np.flip(np.acos(cvt_cell))
    cvt_cell              = np.cos(vt_cell)

    
    assert np.abs(np.dot(vtqw     , 2* np.pi* sph_harm_real(0, 0, vtq, 0)**2 )-1) < 1e-14 
    assert np.abs(np.dot(gll_vt_w , 2* np.pi* sph_harm_real(0, 0, np.arccos(gll_vt), 0)**2 )-1) < 1e-14 

    ff                    = h5py.File("%s/%s"%(folder,fname), "r")
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
    vth       = (float) (args["Te"])**0.5 * c_gamma
    mm_fac    = np.sqrt(4 * np.pi) 

    neuz      = ne * uz 
    neuz_avg  = time_average(neuz, tt)
    ne_avg    = time_average(ne, tt)
    ni_avg    = time_average(ni, tt)
    E_avg     = time_average(Ef, tt)

    # charge_density = ((ni_avg - ne_avg) / ni_avg)
    # sheath_tol     = 1e-3
    # sidx           = np.argmin(np.abs(charge_density -sheath_tol))
    # print("sheath begin for (ne-ni)/ni rtol = %.4E sheath = %.4E, E_avg = %.4E [V/m]"%(sheath_tol, xx[sidx], E_avg[sidx]))


    idx       = evgrid<ev_cutoff
    evgrid    = evgrid[idx]
    fl        = fl[:, :, :, idx]
    Np        = len(xx)
    deg       = Np-1
    xp        = -np.cos(np.pi*np.linspace(0,deg,Np)/deg)
    vr        = np.sqrt(evgrid) * c_gamma 
    gmx       = vr/vth
    gmw       = trapz_w(gmx)

    eedf_const= 2 * (vth/c_gamma)**3  / mm_fac  
    cos_vt    = cvtq
    sin_vt    = svtq
    assert np.linalg.norm(xp-xx)/np.linalg.norm(xx) < 1e-14, "Chebyshev point mismatch found"
    
    V0p = np.polynomial.chebyshev.chebvander(xx, deg)
    # V0pinv: xp values to coefficients
    ident  = np.eye(Np)
    V0pinv = np.linalg.solve(V0p, ident)

    xlidx  = np.argmin(np.abs(xx-xl))
    xridx  = np.argmin(np.abs(xx-xr))

    print(xl, xx[xlidx], xr, xx[xridx], xlidx, xridx)

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

    
    Svz_l         = np.einsum("a,b,tab->tab", gmx * vth, cos_vt, flx_l)
    Svz_r         = np.einsum("a,b,tab->tab", gmx * vth, cos_vt, flx_r)

    Svzp_l        = np.einsum("a,b,tab->tab", gmx * vth, sin_vt, flx_l)
    Svzp_r        = np.einsum("a,b,tab->tab", gmx * vth, sin_vt, flx_r)


    a1_l          = 2 * np.pi * np.dot(np.dot(Svz_l, vtqw) * gmx**2, gmw) * nex[:, 0]
    a1_r          = 2 * np.pi * np.dot(np.dot(Svz_r, vtqw) * gmx**2, gmw) * nex[:, 1]

    a2_l          = neuzx[:, 0]
    a2_r          = neuzx[:, 1]

    norm_cl       = (a2_l/a1_l)
    norm_cr       = (a2_r/a1_r)

    # for tidx in range(0, len(tt),10):
    #     print("time = %.2E norm_const_left = %.8E norm_const_right = %.8E  with ev cutoff = %.2E"%(tt[tidx],norm_cl[tidx], norm_cr[tidx], ev_cutoff))

    Svz_l         = np.einsum("t,tar->tar", (a2_l/a1_l) * nex[:, 0], Svz_l)
    Svz_r         = np.einsum("t,tar->tar", (a2_r/a1_r) * nex[:, 1], Svz_r)

    Svzp_l        = Svzp_l * (nex[:, 0])[:, np.newaxis, np.newaxis]
    Svzp_r        = Svzp_r * (nex[:, 1])[:, np.newaxis, np.newaxis]

    # Svzp_l        = np.einsum("t,tar->tar", (a2_l/a1_l) * nex[:, 0], Svzp_l)
    # Svzp_r        = np.einsum("t,tar->tar", (a2_r/a1_r) * nex[:, 1], Svzp_r)

    a1_l          = 2 * np.pi * np.dot(np.dot(Svz_l, vtqw) * gmx**2, gmw) 
    a1_r          = 2 * np.pi * np.dot(np.dot(Svz_r, vtqw) * gmx**2, gmw)

    # for tidx in range(0, len(tt), 10):
    #     print("a1_l = %.8E a2_l = %.8E a1_r=%.8E a2_r=%.8E"%(a1_l[tidx], a2_l[tidx], a1_r[tidx], a2_r[tidx]))

    gmx1, gmw1, vr_e2n = vr_coarsen(gmx, vr_p)
    vr_cell            = np.append(gmx1[0::vr_p], np.array([gmx1[-1]]))
    assert(len(vr_cell) == vr_e2n.shape[0] + 1)

    vr_cell_width      = np.array([(-vr_cell[i] + vr_cell[i+1]) for i in range(len(vr_cell)-1)])
    assert (vr_cell_width > 0).all() == True
    m_cell           = grid_to_cell(gmx1, gmw1, vr_e2n,  vtq, vtqw, np.ones_like(Fv_l[0])[np.newaxis, :, :],  vr_p=vr_p, vt_p=vt_p) 
    vz_cell          = grid_to_cell(gmx1, gmw1, vr_e2n,  vtq, vtqw, np.einsum("a,b->ab", gmx * vth, cos_vt)[np.newaxis, :, :],  vr_p=vr_p, vt_p=vt_p)
    vzp_cell         = grid_to_cell(gmx1, gmw1, vr_e2n,  vtq, vtqw, np.einsum("a,b->ab", gmx * vth, sin_vt)[np.newaxis, :, :],  vr_p=vr_p, vt_p=vt_p)

    Fv_cell_l        = grid_to_cell(gmx1, gmw1, vr_e2n,  vtq, vtqw, Fv_l ,  vr_p=vr_p, vt_p=vt_p)
    Fv_cell_r        = grid_to_cell(gmx1, gmw1, vr_e2n,  vtq, vtqw, Fv_r ,  vr_p=vr_p, vt_p=vt_p)
    Svz_cell_l       = grid_to_cell(gmx1, gmw1, vr_e2n,  vtq, vtqw, Svz_l,  vr_p=vr_p, vt_p=vt_p)
    Svz_cell_r       = grid_to_cell(gmx1, gmw1, vr_e2n,  vtq, vtqw, Svz_r,  vr_p=vr_p, vt_p=vt_p)

    Svzp_cell_l      = grid_to_cell(gmx1, gmw1, vr_e2n,  vtq, vtqw, Svzp_l, vr_p=vr_p, vt_p=vt_p)
    Svzp_cell_r      = grid_to_cell(gmx1, gmw1, vr_e2n,  vtq, vtqw, Svzp_r, vr_p=vr_p, vt_p=vt_p)

    ne_cell_sum_l    = np.sum(Fv_cell_l.reshape((len(tt), -1)), axis=1)
    ne_cell_sum_r    = np.sum(Fv_cell_r.reshape((len(tt), -1)), axis=1)

    neuz_cell_sum_l  = np.sum(Svz_cell_l.reshape((len(tt), -1)), axis=1)
    neuz_cell_sum_r  = np.sum(Svz_cell_r.reshape((len(tt), -1)), axis=1)

    #print(np.linalg.norm(nex[:, 0]-ne_cell_sum_l)/np.linalg.norm(nex[:, 0]))
    assert np.linalg.norm(nex[:, 0]-ne_cell_sum_l)/np.linalg.norm(nex[:, 0]) < 1e-10
    assert np.linalg.norm(nex[:, 1]-ne_cell_sum_r)/np.linalg.norm(nex[:, 1]) < 1e-10

    assert np.linalg.norm(neuzx[:, 0]-neuz_cell_sum_l)/np.linalg.norm(neuzx[:, 0]) < 1e-10
    assert np.linalg.norm(neuzx[:, 1]-neuz_cell_sum_r)/np.linalg.norm(neuzx[:, 1]) < 1e-10

    vt_mid_idx       = np.argmin(np.abs(vt_cell-np.pi/2))

    uz_cell_l        = np.zeros_like(Svz_cell_l)
    uz_cell_r        = np.zeros_like(Svz_cell_r)

    uzp_cell_l       = np.zeros_like(Svzp_cell_l)
    uzp_cell_r       = np.zeros_like(Svzp_cell_r)

    uz_cell_l[:, : ,    0 : vt_mid_idx]   = Svz_cell_l[:, : , 0 : vt_mid_idx]        / Fv_cell_l[:, : , 0 : vt_mid_idx]
    uz_cell_r[:, : ,    0 : vt_mid_idx]   = Svz_cell_r[:, : , 0 : vt_mid_idx]        / Fv_cell_r[:, : , 0 : vt_mid_idx]
    uz_cell_l[:, : , (vt_mid_idx + 1):]   = Svz_cell_l[:, : , (vt_mid_idx + 1):]     / Fv_cell_l[:, : , (vt_mid_idx + 1):]
    uz_cell_r[:, : , (vt_mid_idx + 1):]   = Svz_cell_r[:, : , (vt_mid_idx + 1):]     / Fv_cell_r[:, : , (vt_mid_idx + 1):]

    uzp_cell_l[:, :,    0 : vt_mid_idx]   = Svzp_cell_l[:, :, 0 : vt_mid_idx]        / Fv_cell_l[:, : , 0 : vt_mid_idx]
    uzp_cell_r[:, :,    0 : vt_mid_idx]   = Svzp_cell_r[:, :, 0 : vt_mid_idx]        / Fv_cell_r[:, : , 0 : vt_mid_idx]
    uzp_cell_l[:, :, (vt_mid_idx + 1):]   = Svzp_cell_l[:, :, (vt_mid_idx + 1):]     / Fv_cell_l[:, : , (vt_mid_idx + 1):]
    uzp_cell_r[:, :, (vt_mid_idx + 1):]   = Svzp_cell_r[:, :, (vt_mid_idx + 1):]     / Fv_cell_r[:, : , (vt_mid_idx + 1):]

    vz_cell[:, :,    0 : vt_mid_idx]      = vz_cell[:, :, 0 : vt_mid_idx]        / m_cell[:, : , 0 : vt_mid_idx]
    vz_cell[:, : , (vt_mid_idx + 1):]     = vz_cell[:, : , (vt_mid_idx + 1):]    / m_cell[:, : , (vt_mid_idx + 1):]

    vzp_cell[:, :,    0 : vt_mid_idx]     = vzp_cell[:, :, 0 : vt_mid_idx]       / m_cell[:, : , 0 : vt_mid_idx]
    vzp_cell[:, : , (vt_mid_idx + 1):]    = vzp_cell[:, : , (vt_mid_idx + 1):]   / m_cell[:, : , (vt_mid_idx + 1):]
    
    ## added to make sure we don't get currupted vz due to tail osicllations. 04-15-2025
    uz_cell_l [Fv_cell_l < 0] = 0.0
    uz_cell_r [Fv_cell_r < 0] = 0.0
    uzp_cell_l[Fv_cell_l < 0] = 0.0
    uzp_cell_r[Fv_cell_r < 0] = 0.0

    # print("%.8E " %np.sum(uzp_cell_l))
    # print("%.8E " %np.sum(uzp_cell_r))
    # a = np.sqrt(uz_cell_l**2 + uzp_cell_l**2) /vth
    # print(np.min(a), np.max(a))

    # a = np.sqrt(uz_cell_r**2 + uzp_cell_r**2) /vth
    # print(np.min(a), np.max(a))

    vz_cell          = vz_cell[0]
    vzp_cell         = vzp_cell[0]
    vr_c             = np.sqrt(vz_cell**2 + vzp_cell**2)
    vt_c             = np.ones_like(vr_c) * np.pi/2 
    vt_c[vr_c>0]     = np.arccos(vz_cell[vr_c>0]/vr_c[vr_c>0])

    return {"tt": tt, 
            "ev": evgrid, 
            "vr": vr/vth, 
            "vr_weights":gmw,
            "vtheta": vtq,
            "vtheta_weights": vtqw,
            "vt_p": vt_p,
            "time": tt,
            "fv_left_bdy": Fv_l,
            "fv_right_bdy": Fv_r,
            "flux_left_bdy": Svz_l,
            "flux_right_bdy": Svz_r,
            "Ef": Ef, 
            "vth": vth,
            "xx":xx,
            "xlidx":xlidx,
            "xridx":xridx,
            "uz": uz,
            "ne": ne,

            #"vr_cell_width" : np.array([(-vr_cell[i] + vr_cell[i+1]) for i in range(len(vr_cell)-1)]),
            "vr_cell_coord"  : np.array([0.5 * (vr_cell[i] + vr_cell[i+1]) for i in range(len(vr_cell)-1)]),
            "vt_cell_coord"  : np.array([0.5 * (vt_cell[i] + vt_cell[i+1]) for i in range(len(vt_cell)-1)]),
            "vr_cell"        : vr_c,
            "vt_cell"        : vt_c,
            "vz_cell"        :  vz_cell,
            "vzp_cell"       : vzp_cell,
            "Svz_cell_left" : Svz_cell_l,
            "Svz_cell_right": Svz_cell_r,
            # "Svz_cell_left" : Svz_cell_l,
            # "Svz_cell_right": Svz_cell_r,
            "Fv_cell_left"  : Fv_cell_l,
            "Fv_cell_right" : Fv_cell_r,
            "uz_cell_left"  : uz_cell_l,
            "uz_cell_right" : uz_cell_r,
            "uzp_cell_left" : uzp_cell_l,
            "uzp_cell_right": uzp_cell_r

            }

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


def use_extract_data():
    data   = extract_data(folder = folder_name, fname=macro_fname, xl=xl, xr=xr, vt_k=65, ev_cutoff=80)
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

    #print(xx[81], xx[82])
    print(" xlidx = ", xlidx, " xridx = ",  xridx, "xl = ", xx[xlidx], "xr = ", xx[xridx])

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
    
    vtheta_bins = 0.5 * (vtheta[0:-1] + vtheta[1:])
    
    pi_by_2_idx = np.argmin(np.abs(vtheta-np.pi/2))+1
    assert np.abs(vtheta[pi_by_2_idx-1]   - np.pi/2) < 1e-15
    assert np.abs(vtheta[pi_by_2_idx]     - np.pi/2) < 1e-15
    vtheta_0    = vtheta[0:pi_by_2_idx]
    vtheta_1    = vtheta[pi_by_2_idx:]

    assert np.abs(vtheta_0[0]-0.0)< 1e-15         and np.abs(vtheta_0[-1] - np.pi/2) < 1e-15
    assert np.abs(vtheta_1[0]-np.pi/2)< 1e-15     and np.abs(vtheta_1[-1] - np.pi  ) < 1e-15

    iFlux_Lp    = quad_on_grid(vr, vtheta_0, Flux_L[:, :, 0:pi_by_2_idx]) 
    iFlux_Ln    = quad_on_grid(vr, vtheta_1, Flux_L[:, :, pi_by_2_idx: ]) 

    iFlux_Rp    = quad_on_grid(vr, vtheta_0, Flux_R[:, :, 0:pi_by_2_idx]) 
    iFlux_Rn    = quad_on_grid(vr, vtheta_1, Flux_R[:, :, pi_by_2_idx: ]) 

    plt.figure(figsize=(12,8), dpi=200)

    plt.subplot(3, 2, 1)
    plt.title(r"$x_L=%.4E$"%(xx[xlidx]))
    plt.semilogy(tt, np.abs(np.sum(iFlux_Lp.reshape((len(tt), -1)), axis=1)), label=r'S_L > 0 (incoming)')
    plt.semilogy(tt, np.abs(np.sum(iFlux_Ln.reshape((len(tt), -1)), axis=1)), label=r'S_L < 0 (outgoing)')
    #plt.plot(tt, np.sum(iFlux_Rp.reshape((len(tt), -1)), axis=1), label=r'S_R < 0')
    plt.ylabel(r"")
    plt.xlabel(r"time [T]")
    plt.grid(visible=True)
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.title(r"$x_L=%.4E$"%(xx[xridx]))
    plt.semilogy(tt, np.abs(np.sum(iFlux_Rp.reshape((len(tt), -1)), axis=1)), label=r'S_R > 0 (outgoing)')
    plt.semilogy(tt, np.abs(np.sum(iFlux_Rn.reshape((len(tt), -1)), axis=1)), label=r'S_R < 0 (incoming)')
    plt.ylabel(r"")
    plt.xlabel(r"time [T]")
    plt.grid(visible=True)
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.semilogy(tt, np.abs(np.sum(iFlux_Lp.reshape((len(tt), -1)), axis=1) + np.sum(iFlux_Ln.reshape((len(tt), -1)), axis=1)), label=r'|S_L|')
    plt.ylabel(r"")
    plt.xlabel(r"time [T]")
    plt.grid(visible=True)
    plt.legend()
    
    plt.subplot(3, 2, 4)
    plt.semilogy(tt, np.abs(np.sum(iFlux_Rp.reshape((len(tt), -1)), axis=1) + np.sum(iFlux_Rn.reshape((len(tt), -1)), axis=1)), label=r'|S_R|')
    plt.ylabel(r"")
    plt.xlabel(r"time [T]")
    plt.grid(visible=True)
    plt.legend()


    plt.subplot(3, 2, 5)
    plt.plot(tt, (np.sum(iFlux_Lp.reshape((len(tt), -1)), axis=1) + np.sum(iFlux_Ln.reshape((len(tt), -1)), axis=1)), label=r'S_L')
    plt.ylabel(r"total incoming flux")
    plt.xlabel(r"time [T]")
    plt.grid(visible=True)
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(tt, (-np.sum(iFlux_Rp.reshape((len(tt), -1)), axis=1) - np.sum(iFlux_Rn.reshape((len(tt), -1)), axis=1)), label=r'S_R')
    plt.ylabel(r"total incoming flux")
    plt.xlabel(r"time [T]")
    plt.grid(visible=True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("flux_eulerian_xl_%.4E_xr_%.4E.png"%(float(sys.argv[3]), float(sys.argv[4])))
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(tt, Ef[:, 0] , label=r"E(x=-1)")
    plt.plot(tt, Ef[:, -1], label=r"E(x= 1)")
    plt.legend()
    plt.xlabel(r"time [T]")
    plt.ylabel(r"E [V/m]")
    plt.grid(visible=True)
    plt.savefig("electric_field.png")
    plt.close()

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(tt, uz[:, xlidx] , label=r"uz(x=%.4E)"%(xx[xlidx]))
    plt.plot(tt, uz[:, xridx] , label=r"uz(x=%.4E)"%(xx[xridx]))
    plt.legend()
    plt.xlabel(r"time [T]")
    plt.ylabel(r"uz [m/s]")
    plt.grid(visible=True)
    plt.savefig("uz.png")
    plt.close()

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(tt, ne[:, xlidx] , label=r"uz(x=%.4E)"%(xx[xlidx]))
    plt.plot(tt, ne[:, xridx] , label=r"uz(x=%.4E)"%(xx[xridx]))
    plt.legend()
    plt.xlabel(r"time [T]")
    plt.ylabel(r"ne [1/m^3]")
    plt.grid(visible=True)
    plt.savefig("ne.png")
    plt.close()

    

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


def use_extract_data_on_cell():
    io_out_folder = "t2"
    make_dir(io_out_folder)    
    data          = extract_data_on_cell(folder = folder_name, fname=macro_fname, xl=xl, xr=xr, vr_p=2, num_vt=130, vt_p=3 , ev_cutoff=80)
    # data_low      = extract_data_on_cell(folder = folder_name, fname=macro_fname, xl=xl, xr=xr, vr_p=2, num_vt=130, vt_p=3 , ev_cutoff=160)

    # for i in ["Fv_cell_left", "Svz_cell_left", "uz_cell_left", "uzp_cell_left", "Fv_cell_right", "Svz_cell_right", "uz_cell_right", "uzp_cell_right"]:
    #     print("rel error [%s]= %.8E"%(i, np.linalg.norm(data[i]-data_low[i])/np.linalg.norm(data_low[i])))

    tt            = data["tt"]
    vr_cell_coord = data["vr_cell_coord"]  # vr bucket centers (unit : []) 
    vt_cell_coord = data["vt_cell_coord"]  # vt bucket centers (unit : [])
    Sv_cell_left  = data["Svz_cell_left" ] # Sv (left)  integrated on the cell centers (unit : [m^-2s^-1])
    Sv_cell_right = data["Svz_cell_right"] # Sv (right) integrated on the cell centers (unit : [m^-2s^-1])
    Fv_cell_left  = data["Fv_cell_left" ]  # Sv (left)  integrated on the cell centers (unit : [m^-3])
    Fv_cell_right = data["Fv_cell_right"]  # Sv (right) integrated on the cell centers (unit : [m^-3])
    uz_cell_left  = data["uz_cell_left" ]  # electron vz_cell_left   (unit : [ms^-1])
    uz_cell_right = data["uz_cell_right"]  # electron vz_cell_right  (unit : [ms^-1])
    uzp_cell_left = data["uzp_cell_left" ] # electron (perpendicular) vzp_cell_left   (unit : [ms^-1])
    uzp_cell_right= data["uzp_cell_right"] # electron (perpendicular) vzp_cell_right  (unit : [ms^-1])

    vz            = data["vz_cell"]        # vz mean velocity of the cell (without EDF weight) (unit: [ms^-1])
    vzp           = data["vzp_cell"]       # (vz perpendicular) vzp mean velocity of the cell (without EDF weight) (unit: [ms^-1])
    vr_cell       = data["vr_cell"]        # mean radial velocity of the cell (ms^{-1})
    vt_cell       = data["vt_cell"]        # polar angle corresponds to the mean velocity of the cell. 
    vth           = data["vth"]

    c_gamma       = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))

    ev_cell       = (vr_cell_coord * vth)**2 /c_gamma**2
    
    dvt           = 0.5 * (vt_cell_coord[1] - vt_cell_coord[0])
    dev           = 0.5 * (ev_cell[1]  - ev_cell[0])
    extent        = [ev_cell[0]-dev , ev_cell[-1] + dev, vt_cell_coord[-1] + dvt, vt_cell_coord[0]-dvt]
    vt_mid_idx    = np.argmin(np.abs(vt_cell_coord-np.pi/2))


    # uz_energy_left= uz_cell_left ** 2 / c_gamma**2
    # #print(np.max(uz_energy_left))
    # print(uz_energy_left.shape)
    # print(uz_energy_left)
    # for tidx in range(0, len(tt), 1):
    #     print(tt[tidx], np.max(uz_energy_left[tidx]), np.min(Fv_cell_left[tidx]), np.max(Fv_cell_left[tidx]), np.min(uz_cell_left[tidx]), np.max(uz_cell_left[tidx]))

    # sys.exit(0)

    # for tidx in range(0, len(tt), 1):
    #     print("time = %.2E [L] min: %.8E max: %.8E rel error per cell (min/max): %.8E"%(tt[tidx], np.min(Sv_cell_left[tidx,:,0: vt_mid_idx]),
    #                                                                     np.max(Sv_cell_left[tidx, :, 0:vt_mid_idx]),
    #                                                                     np.min(Sv_cell_left[tidx, :, 0:vt_mid_idx])/np.max(Sv_cell_left[tidx, :, 0:vt_mid_idx])))

    #     print("time = %.2E [R] min: %.8E max: %.8E rel error per cell (min/max): %.8E"%(tt[tidx], np.min(Sv_cell_right[tidx,:,0: vt_mid_idx]),
    #                                                                     np.max(Sv_cell_right[tidx, :, 0:vt_mid_idx]),
    #                                                                     np.min(Sv_cell_right[tidx, :, 0:vt_mid_idx])/np.max(Sv_cell_right[tidx, :, 0:vt_mid_idx])))
    # for tidx in range(0, len(tt), 1):
    #     print("time = %.2E [L+] sum (Sv_cell_left[tidx, :, 0:vt_mid_idx]) = %.8E [L-] sum (Sv_cell_left[tidx, :, vt_mid_idx+1:])= %.8E "%(tt[tidx], np.sum(Sv_cell_left[tidx, :, 0:vt_mid_idx]), np.sum(Sv_cell_left[tidx, :, vt_mid_idx+1:])))

        

    tstep         = 100
    cmap_str="plasma"
    for tidx in range(0, len(tt), tstep):
        fig = plt.figure(figsize=(12, 6), dpi=200)
        plt.subplot(2, 3, 1)
        plt.imshow(np.abs(Fv_cell_left[tidx].T), aspect='auto', extent=extent, norm=LogNorm(vmin=1e3, vmax=np.max(Fv_cell_left)), cmap=cmap_str)
        plt.colorbar()
        plt.xlabel(r"energy [eV]")
        plt.ylabel(r"$v_\theta$")
        plt.title(r"$n_e(x_L)$")

        plt.subplot(2, 3, 4)
        plt.imshow(np.abs(Fv_cell_right[tidx].T), aspect='auto', extent=extent, norm=LogNorm(vmin=1e3, vmax=np.max(Fv_cell_right)), cmap=cmap_str)
        plt.colorbar()
        plt.xlabel(r"energy [eV]")
        plt.ylabel(r"$v_\theta$")
        plt.title(r"$n_e(x_R)$")

        plt.subplot(2, 3, 2)
        #m = np.copy(Sv_cell_left[tidx])
        #m[:, vt_mid_idx:] = 0.0
        #plt.imshow(np.abs(m.T), aspect='auto', extent=extent, vmin=np.min(Sv_cell_left), vmax=np.max(Sv_cell_left[:, :, 0:vt_mid_idx]))
        #plt.imshow(Sv_cell_left[tidx].T, aspect='auto', extent=extent, vmin=np.min(Sv_cell_left), vmax=np.max(Sv_cell_left))
        plt.imshow(np.abs(Sv_cell_left[tidx].T), aspect='auto', extent=extent, norm=LogNorm(vmin=1e12, vmax=np.max(np.abs(Sv_cell_left))), cmap=cmap_str)
        plt.colorbar()
        plt.xlabel(r"energy [eV]")
        plt.ylabel(r"$v_\theta$")
        plt.title(r"$flux(x_L)$")
        #plt.xlim(0, 30)

        plt.subplot(2, 3, 5)
        #plt.imshow(Sv_cell_right[tidx].T, aspect='auto', extent=extent, vmin=np.min(Sv_cell_right), vmax=np.max(Sv_cell_right))
        plt.imshow(np.abs(Sv_cell_right[tidx].T), aspect='auto', extent=extent, norm=LogNorm(vmin=1e12, vmax=np.max(np.abs(Sv_cell_right))), cmap=cmap_str)
        plt.colorbar()
        plt.xlabel(r"energy [eV]")
        plt.ylabel(r"$v_\theta$")
        plt.title(r"$flux(x_R)$")
        #plt.xlim(0, 30)


        plt.subplot(2, 3, 3)
        vr_t                                    = np.sqrt(uz_cell_left[tidx]**2 + uzp_cell_left[tidx]**2)
        vt_t                                    = np.zeros_like(vr_t) * (np.pi/2)
        vt_t[: , 0:vt_mid_idx     ]             = np.arccos(uz_cell_left[tidx, :, 0: vt_mid_idx  ]/vr_t[:, 0:vt_mid_idx   ])
        vt_t[: , (vt_mid_idx+1):  ]             = np.arccos(uz_cell_left[tidx, :, (vt_mid_idx+1):]/vr_t[:, (vt_mid_idx+1):])
        
        assert (vr_t[vr_cell==0] == 0).all() == True
        vrt_error                               = 0.5 * (np.abs(1-vr_t/vr_cell) + np.abs(1-vt_t/vt_cell))

        
        plt.imshow(vrt_error.T, aspect='auto', extent=extent, cmap=cmap_str, norm=LogNorm(vmin=1e-10, vmax=1))
        #plt.imshow((vtotal.T), aspect='auto', extent=extent, cmap=cmap_str)
        #plt.imshow(uz_cell_left[tidx].T, aspect='auto', extent=extent)
        #plt.imshow(np.abs(uz_cell_left[tidx].T), aspect='auto', extent=extent, norm=LogNorm(vmin=1e1, vmax=1e6))
        plt.colorbar()
        plt.xlabel(r"energy [eV]")
        plt.ylabel(r"$v_\theta$")
        
        plt.subplot(2, 3, 6)
        vr_t                                    = np.sqrt(uz_cell_right[tidx]**2 + uzp_cell_right[tidx]**2)
        vt_t                                    = np.zeros_like(vr_t) * (np.pi/2)
        vt_t[: , 0:vt_mid_idx     ]             = np.arccos(uz_cell_right[tidx, :, 0: vt_mid_idx  ]/vr_t[:, 0:vt_mid_idx   ])
        vt_t[: , (vt_mid_idx+1):  ]             = np.arccos(uz_cell_right[tidx, :, (vt_mid_idx+1):]/vr_t[:, (vt_mid_idx+1):])
        
        assert (vr_t[vr_cell==0] == 0).all() == True
        vrt_error                               = 0.5 * (np.abs(1-vr_t/vr_cell) + np.abs(1-vt_t/vt_cell))
        #print(np.abs(1-vr_t/vr_cell)[0])
        #print(np.abs(1-vt_t/vt_cell)[:, 0])
        plt.imshow(vrt_error.T, aspect='auto', extent=extent, cmap=cmap_str, norm=LogNorm(vmin=1e-10, vmax=1))
        #plt.imshow((vtotal.T), aspect='auto', extent=extent, cmap=cmap_str)
        #plt.imshow(uz_cell_right[tidx].T, aspect='auto', extent=extent)
        #plt.imshow(np.abs(uz_cell_right[tidx].T), aspect='auto', extent=extent, norm=LogNorm(vmin=1e1, vmax=1e6))
        plt.colorbar()
        plt.xlabel(r"energy [eV]")
        plt.ylabel(r"$v_\theta$")
        plt.suptitle("time = %.4E"%(tt[tidx]))

        plt.tight_layout()

        plt.savefig("%s/p_%04d.png"%(io_out_folder, tidx))
        #plt.show()
        plt.close()
        #plt.show(block=False)
        #plt.pause(2)
        #os.system('read -p "a" ') 
        #plt.close('all')

    

def make_movie(xloc, movie_data_folder, tidx_step):
    assert len(xloc)%2 == 0

    io_out_folder = movie_data_folder
    make_dir(io_out_folder)    
    
    data  = [extract_data_on_cell(folder = folder_name, fname=macro_fname, xl=xloc[i], xr=xloc[i+1], num_vt=64, vt_p=8, ev_cutoff=50) for i in range(0, len(xloc), 2)] 
    nrows = 2
    ncols = len(xloc)

    tt    = data[0]["tt"]
    xx    = data[0]["xx"] 
    c_gamma = np.sqrt(2 * (scipy.constants.elementary_charge/ scipy.constants.electron_mass))
    cmap_str="plasma"
    vr_cell       = data[0]  ["vr_cell"      ]  # vr bucket centers (unit : []) 
    vt_cell       = data[0]  ["vt_cell"      ]  # vt bucket centers (unit : [])
    vth           = data[0]  ["vth"]
    ev_cell       = (vr_cell * vth)**2 /c_gamma**2
    dvt           = 0.5 * (vt_cell[1] - vt_cell[0])
    dev           = 0.5 * (ev_cell[1]  - ev_cell[0])
    extent        = [ev_cell[0]-dev , ev_cell[-1] + dev, vt_cell[-1] + dvt, vt_cell[0]-dvt]
    vt_mid_idx    = np.argmin(np.abs(vt_cell-np.pi/2))

    for tidx in range(0, len(tt), tidx_step):
        plt.figure(figsize=(ncols * 4 + 2, nrows*4 + 2), dpi=200)
        for xidx in range(len(xloc)//2):
            xlidx         = data[xidx]["xlidx"]
            xridx         = data[xidx]["xridx"]
            Sv_cell_left  = data[xidx]["Svz_cell_left" ] # Sv (left)  integrated on the cell centers (unit : [m^-2s^-1])
            Sv_cell_right = data[xidx]["Svz_cell_right"] # Sv (right) integrated on the cell centers (unit : [m^-2s^-1])
            Fv_cell_left  = data[xidx]["Fv_cell_left" ]  # Sv (left)  integrated on the cell centers (unit : [m^-3])
            Fv_cell_right = data[xidx]["Fv_cell_right"]  # Sv (right) integrated on the cell centers (unit : [m^-3])
            uz_cell_left  = data[xidx]["uz_cell_left" ]  # vz_cell_left   (unit : [ms^-1])
            uz_cell_right = data[xidx]["uz_cell_right"]  # vz_cell_right  (unit : [ms^-1])
            uzp_cell_left = data[xidx]["uzp_cell_left" ] # (perpendicular) vzp_cell_left   (unit : [ms^-1])
            uzp_cell_right= data[xidx]["uzp_cell_right"] # (perpendicular) vzp_cell_right  (unit : [ms^-1])


            plt.subplot(nrows, ncols, 0 * ncols + (2 * xidx + 1))
            plt.title(r"$n_e(\hat{x} = %.4f, \varepsilon, v_\theta)[m^{-3}]$"%(xx[xlidx]))
            plt.imshow(np.abs(Fv_cell_left[tidx].T), aspect='auto', extent=extent, norm=LogNorm(vmin=1e5, vmax=np.max(Fv_cell_left)), cmap=cmap_str)
            plt.colorbar()
            plt.xlabel(r"energy [eV]")
            plt.ylabel(r"$v_\theta$")
            
            plt.subplot(nrows, ncols, 0 * ncols + (2 * xidx + 2))
            plt.title(r"$n_e(\hat{x} = %.4f, \varepsilon, v_\theta) [m^{-3}]$"%(xx[xridx]))
            plt.imshow(np.abs(Fv_cell_right[tidx].T), aspect='auto', extent=extent, norm=LogNorm(vmin=1e5, vmax=np.max(Fv_cell_right)), cmap=cmap_str)
            plt.colorbar()
            plt.xlabel(r"energy [eV]")
            plt.ylabel(r"$v_\theta$")


            plt.subplot(nrows, ncols, 1 * ncols + (2 * xidx + 1))
            plt.title(r"electron flux$(\hat{x} = %.4f, \varepsilon, v_\theta) [m^{-2}s^{-1}]$"%(xx[xlidx]))
            plt.imshow(np.abs(Sv_cell_left[tidx].T), aspect='auto', extent=extent, norm=LogNorm(vmin=1e12, vmax=np.max(Sv_cell_left)), cmap=cmap_str)
            plt.colorbar()
            plt.xlabel(r"energy [eV]")
            plt.ylabel(r"$v_\theta$")
            
            plt.subplot(nrows, ncols, 1 * ncols + (2 * xidx + 2))
            plt.title(r"electron flux$(\hat{x} = %.4f, \varepsilon, v_\theta) [m^{-2}s^{-1}]$"%(xx[xridx]))
            plt.imshow(np.abs(Sv_cell_right[tidx].T), aspect='auto', extent=extent, norm=LogNorm(vmin=1e12, vmax=np.max(Sv_cell_right)), cmap=cmap_str)
            plt.colorbar()
            plt.xlabel(r"energy [eV]")
            plt.ylabel(r"$v_\theta$")
    
        plt.tight_layout()
        plt.savefig("%s/p_%04d.png"%(io_out_folder, tidx))
        #plt.show()
        plt.close()
            

            

            
    
    



if __name__ == "__main__":
    folder_name = sys.argv[1]
    macro_fname = sys.argv[2]
    xl          = float(sys.argv[3])
    xr          = float(sys.argv[4])

    #use_extract_data()
    use_extract_data_on_cell()
    
    # q, _ = gauss_legendre_lobatto(12, domain=(-1, 1))
    # # xloc = q[2:-2]
    # # print(xloc)
    # # make_movie(xloc, movie_data_folder="movie", tidx_step=1)


    # xloc = np.array([q[2], q[3], -q[3], -q[2]])
    # print(xloc)
    # make_movie(xloc, movie_data_folder="movie1", tidx_step=1)


    

    
    
    
    








