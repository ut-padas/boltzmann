"""
To test vtheta mollification
"""

import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import scipy.special
from scipy.special import sph_harm
import basis
import scipy.ndimage
import mesh
import scipy.optimize
import sys


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

def Vq_sph(lm_modes, v_theta, v_phi, scale=1):
    """
    compute the basis Vandermonde for the all the basis function
    for the specified quadrature points.  
    """

    num_sph_harm = len(lm_modes)
    assert v_theta.shape == v_phi.shape, "invalid shapes, use mesh grid to get matching shapes"
    _shape = tuple([num_sph_harm]) + v_theta.shape
    Vq = np.zeros(_shape)

    for lm_i, lm in enumerate(lm_modes):
        Vq[lm_i] = scale * sph_harm_real(lm[0], lm[1], v_theta, v_phi) 
    
    return Vq

def compute_proj_ops(Nvt,  lmax):
    gx, gw             = basis.Legendre().Gauss_Pn(Nvt//2)
    gx_m1_0 , gw_m1_0  = 0.5 * gx - 0.5, 0.5 * gw
    gx_0_p1 , gw_0_p1  = 0.5 * gx + 0.5, 0.5 * gw
    xp_vt              = np.append(np.arccos(gx_m1_0), np.arccos(gx_0_p1)) 
    xp_vt_qw           = np.append(gw_m1_0, gw_0_p1)

    Vq                 = Vq_sph([(l,0) for l in range(lmax+1)], xp_vt, np.zeros_like(xp_vt))
    
    op_po2sh           = (Vq @ np.diag(xp_vt_qw)) * 2 * np.pi
    op_psh2o           = Vq.T 

    return xp_vt, xp_vt_qw, op_po2sh, op_psh2o

def benstien_poly(n:int, k:int, x):
    return scipy.special.comb(n, k, exact=True) * (x**k) * (1-x)**(n-k)

def vtheta_phi(n:int, k:int, vtheta, supp, xp=np):
    y = xp.zeros_like(vtheta)
    if (supp=="L"):
        idx    = xp.logical_and(vtheta >= 0 , vtheta < 0.5 * xp.pi)
        x      = 1-xp.cos(vtheta[idx])
        y[idx] = benstien_poly(n, k, x)
    else:
        assert supp == "R", "invalid support specified"
        idx    = xp.logical_and(vtheta > 0.5 * xp.pi , vtheta <=xp.pi) 
        x      = 1 + xp.cos(vtheta[idx])
        y[idx] = benstien_poly(n, k, x)
    
    return y

def mollify(f, m, mode="nearest", xp=np):

    # ksz    = ((int(10 * sigma) // 2) + 1) * 2 
    # x      = xp.arange(-ksz//2, ksz//2 + 1)
    # g      = xp.exp(-(x**2) / (2 * sigma**2))
    # g      = g/xp.sum(g)
    assert np.abs(1-np.sum(m)) < 1e-12, "molifiler is not normalized"
    assert len(m)%2 == 1, "molifier is not symmetric"
    
    ksz        = len(m) -1 
    if mode == "nearest":
        fp     = xp.pad(f, ksz//2, 'constant', constant_values = (f[0], f[-1]))
    elif mode == "zeros":
        fp     = xp.pad(f, ksz//2, 'constant', constant_values = (0, 0))
    else:
        raise NotImplementedError
    
    return xp.convolve(fp, m, mode="valid")

def optimize_mollifier(Nb, num_qpts, domain, xp_vt, xp_vt_w, P):
    
    qx, qw = np.polynomial.legendre.leggauss(num_qpts)
    qx, qw = 0.5 * (qx * (domain[1] - domain[0])  + (domain[1] + domain[0])) , 0.5 * (domain[1] - domain[0]) * qw

    #print(np.sum(qw)- 2 * np.pi)

    xp     = np
    M      = xp.zeros((Nb+1, len(xp_vt), num_qpts))
    x0     = xp.zeros((Nb+1, len(xp_vt)))
    supp   = "L"

    Lp     = mesh.central_dxx(qx)

    for kb in range(Nb+1):
        u      = xp_vt[:, xp.newaxis] - qx * xp.ones((len(xp_vt), len(qx))) 
        M[kb]  = vtheta_phi(Nb, kb, u, supp=supp, xp=xp) * qw
        x0[kb] = vtheta_phi(Nb, kb, xp_vt, supp=supp, xp=xp) 

        # u     = vtheta_phi(Nb, kb, xp_vt, supp=supp, xp=xp)
        # g     = xp.exp(-qx**2/ (2 * 1e-8))
        # g     = g/xp.sum(g)

        # plt.semilogy(xp_vt, u, label="k=%d"%(kb))
        # plt.semilogy(xp_vt, M[kb]@ g, label="k=%d (*)"%(kb))
        # plt.grid(visible=True)
        # plt.show()
        # plt.close()

    
    pgamma = 1e0
    ck     = 1e0
    rk     = 1e0
    def L(x):
        g = xp.copy(x[0:num_qpts])
        u = xp.copy(x[num_qpts:])

        s = 0
        for kb in range(Nb+1):
            y  = M[kb] @ g
            Mp = P @ M[kb]
            #0* + \
            s+= np.sum((y - x0[kb])**2)  + \
                ck * u[kb] * xp.dot(xp_vt_w , (y-x0[kb])**2) + \
                pgamma * xp.sum(xp.log(1 + xp.exp((-Mp @ g)))) + \
                rk * xp.sum((Lp @ g)**2)

        return s
    
    def DL(x):
        g = xp.copy(x[0:num_qpts])
        u = xp.copy(x[num_qpts:])

        s = xp.zeros_like(x)
        for kb in range(Nb+1):
            y  = M[kb] @ g
            Mp = P @ M[kb]
             ##+ ck * u[kb] * xp.einsum("l,lp->p", xp_vt_w, M[kb]) \
             #- 0 * 
            s[0:num_qpts]    += xp.einsum("l,lp->p", 2 * (y -x0[kb]) , M[kb]) +\
                                     ck * u[kb] * xp.einsum("l,l,lp->p", xp_vt_w, 2 * (y-x0[kb]), M[kb]) + \
                                    -pgamma * xp.einsum("l,lp->p", (1 / (1 + xp.exp(Mp @ g)) ), Mp) + \
                                    rk * xp.einsum("i,ip->p", (2 * (Lp @ g)), Lp)
            s[num_qpts + kb]  = ck * xp.dot(xp_vt_w , (y-x0[kb])**2)

        return s
    
    # cons = [{'type': 'eq', 'fun': lambda x:  xp.dot(xp_vt_w, (M[kb]@ x- x0[kb])),
    #                        } for kb in range(Nb+1)]

    cons = [{'type': 'eq', 'fun': lambda x: xp.dot(xp_vt_w, M[kb]@x - x0[kb]),
                           'jac': lambda x: xp.einsum("l,lp->p", xp_vt_w, M[kb]),
                            } for kb in range(Nb+1)]
    
    #
    #cons.append({'type': 'ineq', 'fun': lambda x:  M[kb]@ x})

    g0  = xp.zeros(num_qpts + (Nb +1))
    #g0  = xp.ones(num_qpts)
    gs = 1e-1
    g0[0:num_qpts] = xp.exp(-qx**2 /(2 * gs **2) ) 
    g0[0:num_qpts] = g0[0:num_qpts]/xp.sum(g0[0:num_qpts])
    #print(g0)

    # plt.semilogy(qx, g0[0:num_qpts])
    # plt.semilogy(qx, xp.abs(Lp @ g0[0:num_qpts]))
    # plt.show()
    # plt.close()
    # sys.exit()

    # idx= 150
    # p0 = xp.copy(g0)
    # p0[idx] += 1e-10
    # w  = (L(p0) - L(g0))/1e-10

    # print(w, DL(g0)[idx])
    # sys.exit(0)
    r0 = L(g0)
    d0 = xp.linalg.norm(DL(g0))

    
    
    res = scipy.optimize.minimize(L, g0, callback=lambda x: print("res / r0 : %.8E norm(grad) = %.8E"%(np.abs(L(x)/r0), xp.linalg.norm(DL(x)))),
                                   #options={'maxiter': 1000, 'disp': True},
                                   #jac = DL, 
                                   #method="CG", options={'maxiter': 50000, 'disp': True, "gtol": 1e-10, "c1": 1e-4, "c2": 1e-1},
                                   jac = DL, method="BFGS",
                                   #jac = DL, method="trust-exact",
                                   #constraints=tuple(cons),
                                   #jac =DL,
                                   #method="SLSQP",
                                   options={'maxiter': 10000, 'disp': True},

                                   )

    


    #print(res.x)
    ###res.x[0:num_qpts] = g0[0:num_qpts]

    plt.subplot(1, 2, 1)
    
    plt.semilogy(qx, xp.abs(g0[0:num_qpts]), '--', label=r"g(initial)")
    plt.semilogy(qx, xp.abs(res.x[0:num_qpts]), label=r"g (terminated)")
    plt.xlabel(r"$v_{\theta}$")
    plt.ylabel(r"$g(v_\theta)$")
    plt.legend()
    plt.grid(visible=True)

    plt.subplot(1, 2, 2)
    for kb in range(Nb+1):
        plt.semilogy(xp_vt, x0[kb], label=r"$b_{%d,%d}$"%(kb, Nb))
        plt.semilogy(xp_vt, (M[kb]@res.x[0:num_qpts]), label=r"$b^{*}_{%d, %d}$ (mollified)"%(kb, Nb))
        plt.semilogy(xp_vt, np.abs(P @ M[kb]@res.x[0:num_qpts]), label=r"$P \cdot b^{*}_{%d, %d}$ (SPH)"%(kb, Nb))


    for kb in range(Nb+1):
        print("kb = %d mom_constraint : %.8E"%(kb, xp.dot(xp_vt_w, M[kb]@res.x[0:num_qpts]  - x0[kb])))

    #print((P @ M[0]@res.x[0:num_qpts]) >=0)

    plt.xlabel(r"$v_{\theta}$")
    #plt.ylabel(r"$b(v_\theta)$")
    plt.grid(visible=True)
    plt.legend()
    plt.show()
    plt.close()





    return


    



if __name__ == "__main__":
    Nvt   =  128
    l_max =  32
    Nb    =  1
    xp    =  np
    
    xp_vt, xp_vt_w, Ps, Po = compute_proj_ops(Nvt, l_max)
    P  = Po @ Ps

    optimize_mollifier(Nb, 2 * Nvt, (-1.0 * np.pi, 1.0 * np.pi), xp_vt, xp_vt_w, P)
    #optimize_mollifier(Nb, Nvt, (0, 0.5 * np.pi), xp_vt, xp_vt_w, P)

    # #Dp                     = mesh.upwinded_dx(xp_vt, "LtoR")
    # Lp                     = mesh.central_dxx(xp_vt)
    # Imat                   = np.eye(Lp.shape[0])
    # Lp[0]                  = 0.0
    # Lp[-1]                 = 0.0
    # #Lp                     = Imat - 1e-3 * Lp


    # # for n in range(Nb+1):
    # #     for k in range(n+1):
    # #         plt.semilogy(xp_vt, vtheta_phi(n, k, xp_vt, "L"), label=r"L (n=%d, k=%d)"%(n,k))
    # #         plt.semilogy(xp_vt, vtheta_phi(n, k, xp_vt, "R"), label=r"R (n=%d, k=%d)"%(n,k))

    # # plt.grid(visible=True)
    # # plt.legend()
    # # plt.show()
    # # plt.close()
    # # m   = xp_vt_w @ gaussian_molifier(xp_vt, 1e-3)
    # # print("mass of the molifier : %.8E relative error %.8E"%( m, np.abs(1-m)))

    # #plt.semilogy(xp_vt, gaussian_molifier(xp_vt - xp_vt[0], eps=1e-3, xp=xp))
    # #plt.show()

    # P  = Po @ Ps
    # nb = 1
    # kb = 0
    # u  = vtheta_phi(nb, kb, xp_vt, "L")
    
    
    # plt.subplot(1, 2, 1)
    # plt.semilogy(xp_vt, u   , label=r"$\phi_L$ (n=%d, k=%d)"%(nb,kb))
    # for s in range(1, 5, 1):
    #     ss = Nvt>>s
    #     m0 = np.dot(xp_vt_w, u)
    #     us = scipy.ndimage.gaussian_filter(u, sigma=ss, mode="constant")#gaussian_molifier(u, , xp=xp)
    #     m1 = np.dot(xp_vt_w, us)
    #     us = us * (m0/m1)
        
    #     #print((us>=0).all())
    #     plt.semilogy(xp_vt, xp.abs(us)        , label=r"$\phi^{*, s=%d}_L$ (n=%d, k=%d)"%(ss, nb, kb))
    #     plt.semilogy(xp_vt, xp.abs(P@us), '--', label=r"$T^{%d}\phi^{*, s=%d}_L$ (n=%d, k=%d)"%(l_max, ss, nb,kb))

    # # for s in xp.array(xp.logspace(-8, 0, 4)):
    # #     us = xp.linalg.solve(Imat - s * Lp, u)
    # #     plt.semilogy(xp_vt, us          , label=r"$\phi^{*, s=%d}_L$ (n=%d, k=%d)"%(s, nb, kb))
    # #     plt.semilogy(xp_vt, P @ us, '--', label=r"$T^{%d}\phi^{*, s=%d}_L$ (n=%d, k=%d)"%(l_max, s, nb,kb))

    # plt.grid(visible=True)
    # plt.legend()
    
    # plt.subplot(1, 2, 2)
    # plt.plot(xp_vt, u   , label=r"$\phi_L$ (n=%d, k=%d)"%(nb,kb))
    # for s in range(1, 5, 1):
    #     ss = Nvt>>s
    #     m0 = np.dot(xp_vt_w, u)
    #     us = scipy.ndimage.gaussian_filter(u, sigma=ss, mode="constant")#gaussian_molifier(u, , xp=xp)
    #     m1 = np.dot(xp_vt_w, us)
    #     us = us * (m0/m1)
    #     #print((us>=0).all())
    #     plt.plot(xp_vt,  us        , label=r"$\phi^{*, s=%d}_L$ (n=%d, k=%d)"%(ss, nb, kb))
    #     plt.plot(xp_vt,  P@us, '--', label=r"$T^{%d}\phi^{*, s=%d}_L$ (n=%d, k=%d)"%(l_max, ss, nb,kb))

    # # for s in xp.array(xp.logspace(-8, 0, 4)):
    # #     us = xp.linalg.solve(Imat - s * Lp, u)
    # #     plt.semilogy(xp_vt, us          , label=r"$\phi^{*, s=%d}_L$ (n=%d, k=%d)"%(s, nb, kb))
    # #     plt.semilogy(xp_vt, P @ us, '--', label=r"$T^{%d}\phi^{*, s=%d}_L$ (n=%d, k=%d)"%(l_max, s, nb,kb))

    # plt.grid(visible=True)
    # plt.legend()
    # plt.show()
    # plt.close()


