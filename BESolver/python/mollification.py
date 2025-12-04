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
import h5py


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

def qpts(n, domain):
    # qx, qw   = np.polynomial.legendre.leggauss(num_qpts)
    # qx, qw   = 0.5 * (qx * (domain[1] - domain[0])  + (domain[1] + domain[0])) , 0.5 * (domain[1] - domain[0]) * qw

    # return qx, qw

    def trapz_w(qx):
        # check if tt is uniform
        nx   = len(qx)
        dx   = qx[1:]-qx[0:-1]
        
        qw   = np.array([(dx[i-1] + dx[i]) * 0.5 for i in range(1, nx-1)])
        qw   = np.append(np.array([0.5 * dx[0]]), qw)
        qw   = np.append(qw, np.array([0.5 * dx[-1]]))
        assert np.abs((qx[-1] - qx[0])-np.sum(qw)) < 1e-12
        return qw
    
    qx = np.linspace(domain[0], domain[1], n)
    qw = trapz_w(qx)
    return qx, qw

    # assert n>2, "quadrature order is less than 2"
    # xi, wi = np.zeros(n), np.zeros(n)
    
    # c1      = np.zeros(n)
    # c1[n-1] = 1.0
    
    # c1_x      = np.polynomial.legendre.legder(c1, 1)
    # xi[1:-1]  = np.polynomial.legendre.legroots(c1_x)
    
    # xi[0]     = -1
    # xi[-1]    = 1
    
    # wi[0]     = 2/n/(n-1)
    # wi[-1]    = 2/n/(n-1)
    
    # wi[1:-1]  = 2/n/(n-1)/(np.polynomial.legendre.legval(xi[1:-1], c1)**2)

    # a, b      = domain[0], domain[1]
    
    # xi        = 0.5 * (b-a) * xi + 0.5 * (b+a)
    # wi        = 0.5 * (b-a) * wi
    
    # return xi, wi

def optimize_mollifier(g0, Nb, num_qpts, domain, xp_vt, xp_vt_w, P, mfuncs):
    
    xp     = np
    supp   = "L"
    qx, qw = qpts(num_qpts, domain)
    assert len(qx) == num_qpts
    
    M      = xp.zeros((Nb+1, len(xp_vt), len(qx)))
    B      = xp.zeros((Nb+1, len(xp_vt)))
    Mp     = xp.zeros((Nb+1, len(xp_vt), len(qx)))

    Nm     = len(mfuncs)
    

    # constraint matrix - currently only considers mass
    C      = xp.zeros(((Nb+1) * Nm, len(qx)))
    mc     = xp.zeros( (Nb+1) * Nm)
    Lp     = mesh.central_dxx(qx)#mesh.upwinded_dx(qx, "LtoR")


    for kb in range(Nb+1):
        u      = xp_vt[:, xp.newaxis] - qx * xp.ones((len(xp_vt), len(qx))) 
        M[kb]  = vtheta_phi(Nb, kb, u, supp=supp, xp=xp) * qw
        Mp[kb] = P @ M[kb]
        B[kb]  = vtheta_phi(Nb, kb, xp_vt, supp=supp, xp=xp) 
        for mi in range(Nm):
            mvt              = mfuncs[mi](xp_vt)
            C[kb * Nm + mi]  = xp.dot(xp_vt_w,  M[kb] * mvt[:, xp.newaxis])
            mc[kb * Nm + mi] = xp.dot(xp_vt_w, B[kb] * mvt)
        

    Uc, Sc, Vct = xp.linalg.svd(C, full_matrices=True, compute_uv=True, hermitian=False)
    Vc          = Vct.T
    iSc         = xp.diag(1/Sc)
    Nc          = C.shape[0]
    
    Vc, Vc_b    = Vc[:, 0:Nc], Vc[:, Nc:]
    Vct         = Vc.T
    
    
    pgamma      = 1e-1#100e-1
    rk          = 1e-1#100e-1
    gc          = iSc @ Uc.T @ mc

    pterm       = 1
    pterm_a     = 1e0
    

    def L(gb):
        g = Vc @ gc + Vc_b @ gb
        s = 0
        for kb in range(Nb+1):
            #print(kb, Mp[kb] @ g)
            y  = M[kb] @ g
            yp = Mp[kb] @ g

            if (yp < 0).any():
                # print(yp[yp<0])
                # print("negativity detected in the penalty term, returning inf")
                # assert np.max(np.abs(yp[yp<0])) < 1e-12, "negativity detected in the penalty term"
                # yp[yp<0] = 1e-16
                return np.inf
                #yp = xp.maximum(yp, 1e-10)
            
            if (pterm == 0):
                s += np.sum((y - B[kb])**2)  + pgamma * xp.sum(xp.log(1 + xp.exp((-pterm_a * (yp))))) 
            else:
                s  += np.sum((y - B[kb])**2)  - pgamma * xp.sum(xp.log(yp)) 
        
        s+=rk * xp.sum((Lp @ g)**2)
        return s
    
    # jacobian
    def DL(gb):
        s = xp.zeros_like(gb)
        g = Vc @ gc + Vc_b @ gb
        for kb in range(Nb+1):
            y  = M[kb] @ g
            if (pterm == 0):
                s += xp.einsum("j,jp->p", 2 * (y -B[kb]) , M[kb] @ Vc_b) + \
                    -pterm_a * pgamma * xp.einsum("j,jp->p", (1 / (1 + xp.exp(Mp[kb] @ g)) ), Mp[kb] @ Vc_b) 

            else:
                s += xp.einsum("j,jp->p", 2 * (y -B[kb]) , M[kb] @ Vc_b) + \
                    -pgamma * xp.einsum("j,jp->p", (1 / (Mp[kb] @ g) ), Mp[kb] @ Vc_b) 
                                
            
        s+= rk * xp.einsum("j,jp->p", (2 * (Lp @ g)), Lp @ Vc_b)
        return s
    
    # # heassian
    def DDL(gb):
        H = xp.zeros((len(gb), len(gb)))
        g = Vc @ gc + Vc_b @ gb

        for kb in range(Nb+1):
            y   = M[kb]  @ g            
            gp  = Mp[kb] @ g

            Mb  = M[kb]  @ Vc_b
            Mpb = Mp[kb] @ Vc_b

            if(pterm == 0):
                H   += 2 * xp.einsum("jp,jq->pq", Mb, Mb) + \
                    pterm_a**2 * pgamma * xp.einsum("j,jp,jq->pq", (xp.exp(gp)/ (1+xp.exp(gp))**2), Mpb, Mpb) 
            else:
                H   +=  2 * xp.einsum("jp,jq->pq", Mb, Mb) + \
                    pgamma * xp.einsum("j,jp,jq->pq", (1/(gp**2)), Mpb, Mpb) 
                  
            
              
        Lpb = Lp @ Vc_b            
        H  += rk * xp.einsum("jp,jq->pq", Lpb, Lpb) * 2 
        
        return H
    
    #gb = (Vc_b.T @ (g0 - Vc @ gc))
    # # gb = xp.linalg.lstsq(Vc_b, (g0 - Vc @ gc), rcond=1e-2)[0]
    # # print("n ", xp.linalg.norm(gb - Vc_b.T @ (g0 - Vc @ gc)))
    # # print(gb.shape)
    # # for kb in range(Nb+1):
    # #     #xp.linalg.svd()
    # #     print(Mp[kb] @ Vc_b @ Vc_b.T)

    # g1 = Vc @ gc + Vc_b @ gb
    # plt.semilogy(qx, xp.abs(g0), '--', label=r"g(initial)")
    # plt.semilogy(qx, xp.abs(g1), label=r"g (projected)")
    # plt.xlabel(r"$v_{\theta}$")
    # plt.ylabel(r"$g(v_\theta)$")
    # plt.legend()
    # plt.grid(visible=True)
    # plt.show()
    # plt.close()

    # Q, R = xp.linalg.qr(Vc_b, mode="complete")
    # print(Vc_b.shape)
    # print(Q, R)
    
    # print(Q.T @ Q)
    # sys.exit(0)

    ## use LP to solve for a initial guess

    q0  = -xp.array([Mp[kb] @ Vc @ gc for kb in range(Nb+1)]).reshape((Nb+1, len(xp_vt))).reshape((-1)) 
    Am  = xp.array([Mp[kb] @ Vc_b for kb in range(Nb+1)]).reshape((Nb+1, len(xp_vt), num_qpts-Nc)).reshape((-1, num_qpts-Nc))
    w0  = xp.zeros(Am.shape[1])
    res = scipy.optimize.linprog(w0, A_ub=-Am, b_ub=-(q0+1e-8), method="highs", bounds=(None, None), 
                                options={'disp': True}, x0=(Vc_b.T @ (g0 - Vc @ gc)))

    # print(Am.shape, q0.shape)
    # c   = xp.zeros((2, Am.shape[1]))
    # c[1]= xp.ones(Am.shape[1])
    # c   = c.reshape((-1)) 
    # #Am  = xp.vstack((Am, xp.eye(Am.shape[1])))
    # Ams = xp.zeros((Am.shape[0] + Am.shape[1], 2 * Am.shape[1]))
    # Ams[0:Am.shape[0], 0:Am.shape[1]] = Am
    # Ams[Am.shape[0]: , Am.shape[1]:]  = xp.eye(Am.shape[1])
    # q0  = xp.append(q0, xp.zeros(Am.shape[1]))
    # Am  = Ams
    
    # print(q0.shape, Am.shape, c.shape)
    # res = scipy.optimize.linprog(c, A_ub=-Am, b_ub=-(q0), method="highs", bounds=(None, None), 
    #                             options={'disp': True})



    # res = scipy.optimize.minimize(lambda x: xp.sum(x**2), x0=xp.zeros(Am.shape[1]), 
    #                               jac=lambda x: 2 * xp.einsum("j,jp->p", x, xp.eye(len(x))),
    #                               constraints={'type': 'ineq', 'fun': lambda x: -Am @ x + q0, 'jac' : lambda x: -Am},
    #                               method="SLSQP", options={'maxiter': 1000, 'disp': True, "ftol":1e-12})

    # d   = (g0- Vc @ gc)
    # res = scipy.optimize.minimize(lambda x: xp.sum((Vc_b @ x -d)**2), x0=Vc_b.T @ d, #xp.zeros(Am.shape[1]), 
    #                               jac=lambda x: 2 * xp.einsum("j,jp->p", (Vc_b @ x -d), Vc_b),
    #                               constraints={'type': 'ineq', 'fun': lambda x: -Am @ x + q0, 'jac' : lambda x: -Am},
    #                               method="SLSQP", options={'maxiter': 1000, 'disp': True, "ftol":1e-12})
    
    # res = scipy.optimize.minimize(lambda x: xp.sum((Vc @ gc * 0 + Vc_b @ x)**2), x0=xp.zeros(Am.shape[1]),
    #                               jac=lambda x: 2 * xp.einsum("j,jp->p", (Vc @ gc  + Vc_b @ x), Vc_b),
    #                               constraints={'type': 'ineq', 'fun': lambda x: -Am @ x + q0, 'jac' : lambda x: -Am},
    #                               method="SLSQP", options={'maxiter': 1000, 'disp': True, "ftol":1e-12})



    if(res.success):
        gb = res.x
        #print(gb)
        # print(Am @ gb < q0)
        # print(Am @ gb -q0)
        # print(res.status)
        # print((-q0) -(-Am) @ gb)
        # print(L(gb))
        
        assert (Am @ gb >= q0).all()==True, "LP solution does not satisfy the constraint"
        print("LP succeeded, using LP initial guess")
    else:
        print("LP failed, using zero initial guess")
        raise ValueError
    
    r0 = L(gb)
    d0 = xp.linalg.norm(DL(gb))

    
    
    res = scipy.optimize.minimize(L, gb, callback=lambda x: print("||res|| = %.8E ||grad|| = %.8E"%(np.abs(L(x)), xp.linalg.norm(DL(x)))),
                                   #options={'maxiter': 1000, 'disp': True},
                                   #jac = DL, 
                                   #tions={'maxiter': 50000, 'disp': True, "gtol": 1e-10, "c1": 1e-4, "c2": 1e-1},
                                   jac = DL, hess=DDL, method="Newton-CG",options={'maxiter': 100, 'disp': True, 'xtol':1e-12},
                                   #jac = DL, method="CG",options={'maxiter': 10000, 'disp': True},
                                   #jac = DL, method="trust-exact",
                                   #constraints=tuple(cons),
                                   #jac =DL,
                                   #method="SLSQP",
                                   #options={'maxiter': 10000, 'disp': True},

                                   )
    gopt = Vc @ gc + Vc_b @ res.x
    
    # gopt = Vc @ gc + Vc_b @ gb
    # print("norm : ", xp.linalg.norm(g0 - gopt) / xp.linalg.norm(g0))

    plt.subplot(1, 2, 1)
    
    plt.semilogy(qx, xp.abs(g0), '--', label=r"g(initial)")
    plt.semilogy(qx, xp.abs(gopt), label=r"g (terminated)")
    plt.xlabel(r"$v_{\theta}$")
    plt.ylabel(r"$g(v_\theta)$")
    plt.legend()
    plt.grid(visible=True)

    plt.subplot(1, 2, 2)
    for kb in range(Nb+1):
        #print(Mp[kb]@ gopt, kb)
        plt.semilogy(xp_vt, B[kb], label=r"$b_{%d,%d}$"%(kb, Nb))
        #plt.semilogy(xp_vt, (M[kb] @ gopt), label=r"$b^{*}_{%d, %d}$ (mollified)"%(kb, Nb))
        plt.semilogy(xp_vt, (Mp[kb]@ gopt), label=r"$P \cdot b^{*}_{%d, %d}$ (SPH)"%(kb, Nb))
        #plt.semilogy(xp_vt, (P @ M[kb]@res.x[0:num_qpts]),'--')


    for kb in range(Nb+1):
        print("kb = %d mom_constraint : %.8E"%(kb, xp.abs(xp.dot(xp_vt_w, (M[kb] @ gopt  - B[kb])))))

    
    plt.xlabel(r"$v_{\theta}$")
    plt.grid(visible=True)
    plt.legend()
    plt.show()
    plt.close()

    return scipy.interpolate.interp1d(qx, gopt)


def check_bernstien_projection():
    xp                     = np
    ff                     = h5py.File("bte1d3v/r4_upfd/bte1d3v_cp_0001.h5", 'r')
    v                      = np.array(ff["edf"][()])
    mass_op                = xp.load("bte1d3v/r4_upfd/bte1d3v_bte_mass_op.npy")
    num_p                  = 256
    num_x                  = v.shape[1]
    num_sh                 = mass_op.shape[0]// num_p
    num_vt                 = v.shape[0] // num_p
    xp_vt, xp_vt_w, Ps, Po = compute_proj_ops(num_vt, 12)
    v                      = v.reshape((num_p, num_vt, -1))

    xp_vt_d0               = xp_vt[xp_vt < 0.5 * xp.pi]
    xp_vt_d1               = xp_vt[xp_vt > 0.5 * xp.pi]

    xp_vt_d0_n             = 2 * xp_vt_d0 / xp.pi
    xp_vt_d1_n             = (2 * xp_vt_d0 - xp.pi) / xp.pi

    v_d0_inp               = scipy.interpolate.interp1d(xp_vt_d0_n, v[:, xp_vt < 0.5 * xp.pi, -10], kind="linear", axis=1, bounds_error=False, fill_value="extrapolate")
    v_d1_inp               = scipy.interpolate.interp1d(xp_vt_d1_n, v[:, xp_vt > 0.5 * xp.pi, 0] , kind="linear", axis=1, bounds_error=False, fill_value="extrapolate")

    plt.figure(figsize=(16, 8))
    for i in range(8):
        plt.subplot(2, 4, i+1)
        plt.semilogy(xp_vt, v[0::10, :, -(i+1)].T)
        plt.title(r"x[%d]"%(-(i+1)))
        plt.grid(visible=True)
        plt.xlabel(r"$v_{\theta}$")
        plt.ylabel(r"$f_k(v_{\theta})$")

    #plt.show()
    plt.tight_layout()
    plt.savefig("bte1d3v/r4_upfd/vtheta_dist.png", dpi=300)
    # plt.plot(xp_vt_d0_n, v_d0_inp(xp_vt_d0_n)[0::10,:].T, label=r"$v_{\theta} < \pi/2$")
    # plt.show()
    # plt.close()

    # plt.plot(xp_vt_d1_n, v_d1_inp(xp_vt_d0_n)[0::10,:].T, label=r"$v_{\theta} > \pi/2$")
    # plt.show()
    # plt.close()





    # Nb = 4
    # for n in range(Nb+1):
    #     for k in range(n+1):
    #         u  = vtheta_phi(n, k, xp_vt, "L", xp=xp)
    #         up = P @ u
    #         plt.semilogy(xp_vt, u , label=r"$b_{%d,%d}$"%(n,k))
    #         plt.semilogy(xp_vt, up, '--', label=r"$P \cdot b_{%d,%d}$"%(n,k))
    
    # plt.grid(visible=True)
    # plt.legend()
    # plt.show()
    # plt.close()

    



if __name__ == "__main__":
    xp       = np
    Nb       =  0
    Nvtr     = [16, 32, 64]
    domain   = (-1.0 * np.pi, 1.0 * np.pi)
    k_nvt    = 2

    num_qpts = k_nvt * Nvtr[0]
    g0       = xp.zeros(num_qpts)
    qx,qw    = qpts(num_qpts, domain) 
    g0       = xp.ones(num_qpts)#xp.exp(-qx**2 /(2 * (1e-1**2)) ) 
    g0       = g0/xp.sum(g0)
    g0       = scipy.interpolate.interp1d(qx, g0)
    mfuncs   = [lambda x: np.ones_like(x)]#, lambda x: np.cos(x)]#, lambda x: np.sin(x)]

    # check_bernstien_projection()
    # sys.exit(0)
    
    for Nvt in Nvtr:
        l_max =  8#Nvt//2
        xp_vt, xp_vt_w, Ps, Po = compute_proj_ops(Nvt, l_max)

        qx, qw = qpts(k_nvt * Nvt, domain)
        g0 = g0(qx)
        
        P  = Po @ Ps
        g0 = optimize_mollifier(g0, Nb, k_nvt * Nvt, domain, xp_vt, xp_vt_w, P, mfuncs)
        
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


