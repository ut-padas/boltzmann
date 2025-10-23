"""
Hybrid formulation of generalized moments with BTE
"""
import numpy as np
import spec_spherical as sp
import utils as bte_utils
import scipy.constants
import enum
import mesh
import collisions
import matplotlib.pyplot as plt
import sys
import h5py
import collocation_op
import bte_1d3v_solver
import argparse
try:
    import cupy as cp
    import cupyx.scipy.sparse.linalg
except:
    print("Cupy module not found !")
    #raise ModuleNotFoundError

def fft_filtering(tt, q, rk, xp):
    assert len(q.shape) == 2
    ## fft filtering
    fk = xp.fft.rfftfreq(len(tt), d=1/len(tt))
    qh = xp.fft.rfft(q, axis=0)
    assert rk < len(fk)

    sf          = xp.ones_like(fk) * (2/(len(tt)))
    sf[0]       = (1/(len(tt)))
    qh          = sf[:, xp.newaxis] * qh
    qh          = xp.concatenate(((xp.real(qh))[:, :, xp.newaxis],(xp.imag(qh))[:, :, xp.newaxis]), axis=2)
    qh          = xp.swapaxes(qh, 1, 2)

    def m(t):
        at = xp.array([[xp.cos(2 * xp.pi * fk[i] * t ), -xp.sin(2 * xp.pi * fk[i] * t)] for i in range(rk)])
        y  = xp.sum(qh[0:rk, :, :] * at[:, :, xp.newaxis], axis=(0, 1))
        return y
    
    return m


"""
generalized moments BTE hybrid solver
"""
class gm_bte_hybrid_solver():

    def __init__(self, args):
        self.bte_solver     = bte_1d3v_solver.bte_1d3v(args)
        self.bte_params     = self.bte_solver.params
        
        self.bte_solver.init(self.bte_params.dt)
        self.xp_module      = np
        self.vt, self.vt_w  = self.bte_solver.xp_vt, self.bte_solver.xp_vt_qw
        
        xp                  = self.xp_module
        self.spec_sp        = self.bte_solver.op_spec_sp
        
        self.num_vr         = (self.spec_sp._basis_p._num_knot_intervals) * self.bte_params.spline_qpts
        self.num_vt         = self.bte_params.Nvt
        self.num_vp         = 2
        vth                 = self.bte_solver.bs_vth
        c_gamma             = self.bte_solver.c_gamma
        ev0                 = self.bte_params.ev_max
        ev1                 = self.bte_params.ev_max
        ev2                 = self.bte_params.ev_max
        
        self.mfuncs         = [ lambda vr, vt, vp : np.ones_like(vr) * np.heaviside((ev0 - (vr * vth / c_gamma)**2), 1),
                                lambda vr, vt, vp : vth**2 * vr **2 * (2/3/c_gamma**2) * np.heaviside((ev0 - (vr * vth / c_gamma)**2), 1)#np.exp(-(vr)**2),
                                #lambda vr, vt, vp : vth    * vr * np.cos(vt),
                                ]
        
        self.mfuncs_Jx      = [ lambda vr, vt, vp:  vth * vr * np.cos(vt)  * np.ones_like(vr) * np.heaviside((ev1 - (vr * vth / c_gamma)**2), 1), 
                                lambda vr, vt, vp:  vth * vr * np.cos(vt)  * vth**2 * vr **2 * (2/3/c_gamma**2) * np.heaviside((ev1 - (vr * vth / c_gamma)**2), 1),
                                #lambda vr, vt, vp:  vth * vr * np.cos(vt)  * vth    * vr * np.cos(vt),
                                ]
        
        self.mfuncs_Sx      = [ lambda vr, vt, vp : np.ones_like(vr) * np.heaviside((ev2 - (vr * vth / c_gamma)**2), 1),
                                lambda vr, vt, vp : vth**2 * vr **2 * (2/3/c_gamma**2) * np.heaviside((ev2 - (vr * vth / c_gamma)**2), 1)#np.exp(-(vr)**2),
                                #lambda vr, vt, vp : vth    * vr * np.cos(vt),
                                ]
        
        self.mfuncs_names   = [r"$n_e$", r"$T_e$"] 

        
        self.mfuncs_ops_ords    = bte_utils.assemble_moment_ops_ords(self.spec_sp, self.vt, self.vt_w, self.num_vr, self.num_vp, self.mfuncs, scale=1.0)
        self.mfuncs_Jx_ops_ords = bte_utils.assemble_moment_ops_ords(self.spec_sp, self.vt, self.vt_w, self.num_vr, self.num_vp, self.mfuncs_Jx, scale=1.0)
        self.mfuncs_Sx_ops_ords = bte_utils.assemble_moment_ops_ords(self.spec_sp, self.vt, self.vt_w, self.num_vr, self.num_vp, self.mfuncs_Sx, scale=1.0)

        self.I_Nx           = xp.eye(self.bte_params.Np)
        self.Dp             = self.bte_solver.Dp
        self.Dp2            = self.bte_solver.mesh.D2[0]
        self.Dp4            = self.bte_solver.mesh.D4[0]

    def host_to_device(self, dev_id):
        with cp.cuda.Device(dev_id):
            self.I_Nx = cp.asarray(self.I_Nx)
            self.Dp   = cp.asarray(self.Dp)
            
            self.mfuncs_ops_ords    = cp.asarray(self.mfuncs_ops_ords)
            self.mfuncs_Jx_ops_ords = cp.asarray(self.mfuncs_Jx_ops_ords)

    def device_to_host(self, dev_id):
        with cp.cuda.Device(dev_id):
            self.I_Nx               = cp.asnumpy(self.I_Nx)
            self.Dp                 = cp.asnumpy(self.Dp)
            self.mfuncs_ops_ords    = cp.asnumpy(self.mfuncs_ops_ords)
            self.mfuncs_Jx_ops_ords = cp.asnumpy(self.mfuncs_Jx_ops_ords)
    
    def asnumpy(self, x):
        if (type(x) == cp.ndarray):
            return cp.asnumpy(x)
        else:
            type(x) == np.ndarray
            return x

    def bte_vspace_mvec(self, v, E):
        params = self.bte_params
        bte    = self.bte_solver

        assert v.shape == ((params.Nr+1) * params.Nvt, params.Np)

        xp     = bte.xp_module
        x      = bte.op_po2sh @ v 
        y      = params.tau * (params.n0 * params.np0 * (xp.dot(bte.op_col_en, x) + params.Tg * xp.dot(bte.op_col_gT, x))  + E * xp.dot(bte.op_adv_v, x))
        y      = bte.op_psh2o @ y 

        return y
    
    def eval_bte(self, ne, Te, Ext, sfreq:float, warmup_cycles:int, it):
        bte     = self.bte_solver
        params  = self.bte_params

        xp      = self.xp_module 
        xp_bte  = bte.xp_module

        v       = bte.maxwellian_eedf(ne, Te)
        dt      = params.dt

        print("Warmup cycles begin")

        cycle_freq = (int) (1/dt)
        io_freq    = (int) (0.1/dt)
        vnp        = bte.asnumpy(v)
        m0         = self.mfuncs_ops_ords @ vnp
        for tidx in range(warmup_cycles * (int) (1/dt) + 1):
            tt  = tidx * dt

            if (tidx % cycle_freq == 0):
                vnp = bte.asnumpy(v)
                m   = self.mfuncs_ops_ords @ vnp
                a1  = np.linalg.norm(m[0] - m0[0]) / np.linalg.norm(m0[0])
                b1  = np.linalg.norm(m[1]/m[0] - m0[1]/m0[0]) / np.linalg.norm(m0[1]/m0[0])
                print("warmup cycle = %04d ne rtol = %.4E Te rtol = %.4E"%(tidx//cycle_freq, a1, b1))
                m0  = np.copy(m)

            if (tidx % io_freq == 0):
                bte.plot(Ext(tt), v, "%s_1d3v_%04d.png"%(params.fname, tidx //io_freq), tt)

            if(tidx == warmup_cycles * (int) (1/dt)):
                break


            v   = bte.step(Ext(tt), v, None, tt, dt, verbose=int(tidx%100==0))

        print("Warmup cycles end")

        assert sfreq >= dt
        sample_freq  = int(sfreq/dt)
        assert abs(dt * sample_freq - sfreq) < 1e-14
        
        num_m = len(self.mfuncs)
        num_t = int(1/sfreq)
        m     = xp.zeros((3, num_t, num_m, params.Np))

        vnp   = bte.asnumpy(v)
        vT    = xp.copy(vnp)
        m0    = self.mfuncs_ops_ords @ vnp
        for tidx in range((int) (1/dt)):
            tt = tidx * dt

            if (tidx % sample_freq == 0):
                idx       = tidx // sample_freq 
                vnp       = bte.asnumpy(v)

                m[0, idx] = self.mfuncs_ops_ords @ vnp
                m[1, idx] = self.mfuncs_Jx_ops_ords @ vnp

                # if (self.xp_module == np):
                #     fv        = bte.asnumpy(self.bte_vspace_mvec(v, Ext(tt)))
                # else:
                #     fv        = self.bte_vspace_mvec(v, Ext(tt))
                
                # # # fvp             = self.bte_solver.step_bte_v(Ext(tt), v, None, tt, dt, verbose=0)
                # # # fv              = (fvp - v)/dt
                # # # if (self.xp_module == np):
                # # #     fv          = self.bte_solver.asnumpy(fv)
                
                # m[2, idx] = self.mfuncs_ops_ords @ fv

                if (self.xp_module == np):
                    fv        = bte.asnumpy(self.bte_vspace_mvec(v, 0 * Ext(tt)))
                else:
                    fv        = self.bte_vspace_mvec(v, 0 * Ext(tt))

                m[2, idx, 0] = self.mfuncs_ops_ords[0] @ fv
                m[2, idx, 1] = (self.mfuncs_ops_ords[1] @ fv) - params.tau * ((2/3) * bte.asnumpy(Ext(tt)) * m[1, idx, 0]) 

                # a1      = - params.tau * ((2/3) * bte.asnumpy(Ext(tt)) * m[1, idx, 0])
                # xp_bte  = bte.xp_module
                
                # a2      = params.tau * self.mfuncs_Sx_ops_ords[1] @ (bte.asnumpy( bte.op_psh2o @ (Ext(tt) * xp_bte.dot(bte.op_adv_v,  bte.op_po2sh @ v))) )
                
                # # #print(a1, a2, np.linalg.norm(a1-a2)/np.linalg.norm(a2))
                # # plt.plot(bte.xp, a1, label=r"a1")
                # # plt.plot(bte.xp, a2, label=r"a2")
                # # plt.legend()
                # # plt.grid(visible=True)
                # # plt.show()
                # # plt.close()

                # a3      = (self.mfuncs_Sx_ops_ords[1] @ fv)
                # # xp_bte  = bte.xp_module
                
                # a4      = -(1/3) * (15.76) * (self.mfuncs_Sx_ops_ords[0] @ fv)
                
                # # #print(a1, a2, np.linalg.norm(a1-a2)/np.linalg.norm(a2))
                # plt.plot(bte.xp, a1, label=r"a1")
                # plt.plot(bte.xp, a2, label=r"a2")
                # plt.plot(bte.xp, a3, label=r"a3")
                # plt.plot(bte.xp, a4, label=r"a4")
                # plt.legend()
                # plt.grid(visible=True)
                # plt.show()
                # plt.close()

                vT        = xp.copy(vnp)
            
            if (tidx % io_freq == 0 ):
                idx = warmup_cycles * (cycle_freq//io_freq) +  tidx //io_freq
                bte.plot(Ext(tt), v, "%s_1d3v_%04d.png"%(params.fname, idx), tt)
            
            v   = bte.step(Ext(tt), v, None, tt, dt, verbose=int(tidx%100==0))

            
        v       = bte.step(Ext(tt + dt), v, None, tt + dt, dt, verbose=int(tidx%100==0))
        vnp     = bte.asnumpy(v)
        vT      = xp.copy(vnp)

        m1      = self.mfuncs_ops_ords @ vnp
        a1      = xp.linalg.norm(m0[0]-m1[0])/xp.linalg.norm(m0[0])
        a2      = xp.linalg.norm(m0[1]/m0[0]-m1[1]/m1[0])/xp.linalg.norm(m0[1]/m0[0])

        print("delta_ne = %.4E delta_Te = %.4E"%(a1, a2))

        jx      = xp.copy(m[1])
        sx      = xp.copy(m[2])

        # idx     = m[0] > 1e-8
        # jx[idx] = m[1][idx] / m[0][idx]
        # sx[idx] = m[2][idx] / m[0][idx]

        #jxm_max   = xp.max(xp.abs(m[1]), axis=(1, 2)) 
        #m[1, :, 0]
        
        # for i in range(num_m):
        #     jxm = m[1,:, i, :]
        #     a1  = xp.max(xp.abs(jxm), axis=1).reshape((-1, 1))
        #     jxm[xp.abs(jxm) < a1 * 1e-3 ] = 1e-15

        #     sxm = m[2,:, i, :]
        #     a1  = xp.max(xp.abs(sxm), axis=1).reshape((-1, 1))
        #     sxm[xp.abs(sxm) < a1 * 1e-3 ] = 0.0#1e-15
        


        jx = m[1] / m[0]
        sx = m[2] / m[0]

        

        tt                = xp.linspace(0, 1, num_t + 1)[:-1]
        xx                = self.bte_solver.xp
        num_m             = len(self.mfuncs)

        dt                = 0.5 * (tt[1] - tt[0])
        dx                = 0.5 * (xx[1] - xx[0])
        extent            = [xx[0]-dx , xx[-1] + dx, tt[-1] + dt, tt[0]-dt]

        #jx[:, 0, -1]      = 0.0
        #jx[:, 1, -1]      = 0.0

        use_flux_smooth       = False
        if (use_flux_smooth):
            sigma             = 1e-1
            Lp                = gm_bte.I_Nx - sigma * (params.tau/params.L**2) * gm_bte.Dp2
            Lp[0 , :]         = gm_bte.I_Nx[0]
            Lp[-1, :]         = gm_bte.I_Nx[-1]
            LpD               = gm_bte.xp_module.linalg.inv(Lp)

            jx_s              = xp.einsum("tik,mk->tim", jx, LpD)
            sx_s              = xp.einsum("tik,mk->tim", sx, LpD)

            jx                = jx_s
            sx                = sx_s


            # for i in range(num_m):
            #     jxt               = fft_filtering(tt, jx[:, i, :], num_t//4, xp)
            #     sxt               = fft_filtering(tt, sx[:, i, :], num_t//4, xp)

            #     jx[:, i, :]       = xp.array([jxt(t) for t in tt]).reshape((num_t, params.Np))
            #     sx[:, i, :]       = xp.array([sxt(t) for t in tt]).reshape((num_t, params.Np))


        for tidx in range(num_t):
            if (tidx % (100) == 0):
                plt.figure(figsize=(12, 4 * num_m), dpi=200)
                for i in range(num_m):
                    plt.subplot(3, num_m, 0 * num_m + i + 1)
                    plt.semilogy(bte.xp, m[0, tidx, i], label="%s"%(self.mfuncs_names[i]))
                    
                    plt.xlabel(r"$\hat{x}$")
                    plt.ylabel(r"$m$")
                    plt.grid(visible=True)
                    plt.legend()

                for i in range(num_m):
                    plt.subplot(3, num_m, 1 * num_m + i + 1)
                    plt.semilogy(bte.xp, np.abs(m[1, tidx, i]), label="%s"%(self.mfuncs_names[i]))
                
                    plt.xlabel(r"$\hat{x}$")
                    plt.ylabel(r"$J(m)$")
                    plt.grid(visible=True)
                    plt.legend()

                for i in range(num_m):
                    plt.subplot(3, num_m, 2 * num_m + i + 1)
                    plt.semilogy(bte.xp, m[2, tidx, i], label="%s"%(self.mfuncs_names[i]))
                
                    plt.xlabel(r"$\hat{x}$")
                    plt.ylabel(r"$S(m)$")
                    plt.grid(visible=True)
                    plt.legend()
                
                plt.tight_layout()
                plt.savefig("%s_flux_snapshot_%04d.png"%(params.fname, tidx))
                plt.close()

                plt.figure(figsize=(12, 4 * num_m), dpi=200)
                for i in range(num_m):
                    plt.subplot(3, num_m, 0 * num_m + i + 1)
                    plt.plot(bte.xp, m[0, tidx, i], label="%s"%(self.mfuncs_names[i]))
                
                    plt.xlabel(r"$\hat{x}$")
                    plt.ylabel(r"$m$")
                    plt.grid(visible=True)
                    plt.legend()

                for i in range(num_m):
                    plt.subplot(3, num_m, 1 * num_m + i + 1)
                    plt.plot(bte.xp, m[1, tidx, i]/m[0, tidx, i], label="%s"%(self.mfuncs_names[i]))
                    plt.plot(bte.xp, jx[tidx, i],              '--', label="%s-smoothed"%(self.mfuncs_names[i]))
                
                    plt.xlabel(r"$\hat{x}$")
                    plt.ylabel(r"$J(m)$")
                    plt.grid(visible=True)
                    plt.legend()

                for i in range(num_m):
                    plt.subplot(3, num_m, 2 * num_m + i + 1)
                    plt.plot(bte.xp, m[2, tidx, i]/m[0, tidx, i], label="%s"%(self.mfuncs_names[i]))
                    plt.plot(bte.xp, sx[tidx, i],              '--', label="%s-smoothed"%(self.mfuncs_names[i]))
                
                    plt.xlabel(r"$\hat{x}$")
                    plt.ylabel(r"$S(m)$")
                    plt.grid(visible=True)
                    plt.legend()
                
                plt.tight_layout()
                plt.savefig("%s_flux_snapshot_normalized_%04d.png"%(params.fname, tidx))
                plt.close()



        plt.figure(figsize=(12, 4 * num_m), dpi=200)
        plt_idx = 1 

        for i in range(num_m):
            plt.subplot(3, num_m, plt_idx + i)

            if (i == 0):
                plt.imshow(m[0, :, i, :], extent=extent, aspect="auto")
            else:
                plt.imshow(m[0, :, i, :]/m[0, :, 0, :], extent=extent, aspect="auto")

            plt.xlabel(r"$\hat{x}$")
            plt.ylabel(r"$t$")
            plt.title(r"(%s)"%(self.mfuncs_names[i]))
            plt.colorbar()

        for i in range(num_m):
            plt.subplot(3, num_m, plt_idx + num_m + i)
            plt.imshow(jx[:, i, :], extent=extent, aspect="auto")
            plt.xlabel(r"$\hat{x}$")
            plt.ylabel(r"$t$")
            plt.title(r"$\hat{J}$(%s)"%(self.mfuncs_names[i]))
            plt.colorbar()

        for i in range(num_m):
            plt.subplot(3, num_m, plt_idx + 2 * num_m + i)
            plt.imshow(sx[:, i, :], extent=extent, aspect="auto")
            plt.xlabel(r"$\hat{x}$")
            plt.ylabel(r"$t$")
            plt.title(r"$\hat{S}$(%s)"%(self.mfuncs_names[i]))
            plt.colorbar()
        
        plt.tight_layout()
        plt.savefig("%s_usm_%d_moments_it_%04d.png"%(self.bte_params.fname, int(use_flux_smooth) ,it))
        plt.close()

        jxt = list()
        sxt = list()

        # for i in range(num_m):
        #     jxt.append(fft_filtering(tt, jx[:, i, :], num_t//2, xp))
        #     sxt.append(fft_filtering(tt, sx[:, i, :], num_t//2, xp))

        for i in range(num_m):
            jxt.append(jx[:, i, :])
            sxt.append(sx[:, i, :])
            
        return m, jxt, sxt, vT, a1, a2
    
    def init(self, type=0):
        v = self.bte_solver.initial_condition(type)
        
        if self.xp_module == np:
            return self.bte_solver.asnumpy(v) 
        else:
            return v 

    def step(self, m, jxt, sxt, time, dt, tidx):
        bte  = self.bte_solver
        xp   = self.xp_module

        params= bte.params
        num_m = len(self.mfuncs)
        Imat  = self.I_Nx
        Dp    = self.Dp
        Dp2d  = xp.copy(self.Dp2)

        Dp2d[0  , :] = 0 
        Dp2d[-1 , :] = 0 
        

        mp    = xp.zeros_like(m)
        r     = 0.0

        assert time == tidx * dt, "time = %.4E tidx * dt = %.4E"%(time, tidx * dt)
        cycle_freq = int(1/dt)

        for i in range(num_m):
            jxp  = jxt[i][(tidx + 1) % cycle_freq]
            sxp  = sxt[i][(tidx + 1) % cycle_freq]
            
            jxm  = jxt[i][(tidx) % cycle_freq]
            sxm  = sxt[i][(tidx) % cycle_freq]
            

            jxp  = 0.5 * (jxp + jxm)
            sxp  = 0.5 * (sxp + sxm)
            
            # a2   = (Dp @ jxm) * m[i] + (Dp @ m[i]) * jxm
            # a1   = ((Dp @ jxm) * Imat +(xp.diag(jxm) @ Dp)) @ m[i] 
            #print("libniz: %.2E"%(xp.linalg.norm(a1-a2)/xp.linalg.norm(a1)))

            # Lmat = Imat + (1-r) * dt * (-sxp * Imat + (params.tau/params.L) * (Dp @ xp.diag(jxp)))
            # bvec = m[i] + r * dt * (sxm * m[i] - (params.tau/params.L) * (Dp @ (jxm * m[i])))
            
            Lmat = Imat + (1-r) * dt * (-sxp * Imat + (params.tau/params.L) * ((Dp @ jxp) * Imat + xp.diag(jxp) @ Dp)) - (1-r) * dt * (params.tau / params.L**2) *  5e1 * Dp2d
            bvec = m[i] + r * dt * (sxm * m[i] - (params.tau/params.L) * ((Dp @ jxm) * m[i] + (Dp @ m[i]) * jxm))
            #bvec = m[i] + r * dt * (sxm * m[i] - (params.tau/params.L) * (Dp @ (jxm * m[i])))

            mp[i]= xp.linalg.solve(Lmat, bvec)

            # if (np.any(mp[i] < 0)==True):
            #     print(mp[i][mp[i] < 0])
            #     print(i, time, dt, tidx)
                #sys.exit(0)


        return mp

    def evolve(self, m, jxt, sxt, T, dt):
        bte        = self.bte_solver
        xp         = self.xp_module
        
        cycle_freq = (int) (1/dt)
        u0         = xp.copy(m)
        
        for tidx in range((int)(T/dt) + 1):
            tt = tidx * dt
            if(tidx > 0 and (tidx % cycle_freq==0)):
                a1 = xp.linalg.norm(m[0] - u0[0])
                r1 = xp.linalg.norm(1 - m[0]/u0[0])
                print(" (ne(T) - ne(0)) = %.4E  -- (ne(T) - ne(0))/ne(T) = %.4E "%(a1, r1))
                print("Te : %.4E, %.4E"%(xp.min(m[1]/m[0]), xp.max(m[1]/m[0])))    

                u0 = xp.copy(m)


            m  = self.step(m, jxt, sxt, tt, dt)
        
        return m

    def assemble_cycle_step_op(self, jxt, sxt, dt):
        xp         = self.xp_module
        cycle_freq = (int) (1/dt)
        num_m      = len(self.mfuncs)
        Imat       = self.I_Nx
        Dp         = self.Dp

        M                 = xp.zeros((num_m, Imat.shape[0], Imat.shape[1]))
        
        # sigma             = 1e-1
        # Lp                = gm_bte.I_Nx - gm_dt * sigma * (params.tau/params.L**2) * gm_bte.Dp2
        # Lp[0 , :]         = gm_bte.I_Nx[0]
        # Lp[-1, :]         = gm_bte.I_Nx[-1]
        # LpD               = gm_bte.xp_module.linalg.inv(Lp)
        for i in range(num_m):
            M[i]   = xp.copy(Imat)

        for tidx in range(cycle_freq):
            #print(tidx, "tidx")
            for i in range(num_m):

                # jxp  = jxt[i](tidx * dt + dt)
                # sxp  = sxt[i](tidx * dt + dt)
                jxp    = 0.5 * (jxt[i][(tidx) % cycle_freq] + jxt[i][(tidx + 1) % cycle_freq])
                sxp    = 0.5 * (sxt[i][(tidx) % cycle_freq] + sxt[i][(tidx + 1) % cycle_freq])
                Lmat   = Imat + dt * (-sxp * Imat + (params.tau/params.L) * ((Dp @ jxp) * Imat + xp.diag(jxp) @ Dp))
                M[i]   = xp.linalg.inv(Lmat) @ M[i]
            
        return M

    def store_checkpoint(self, m, time, dt, fprefix):
        try:
            with h5py.File("%s.h5"%(fprefix), 'w') as F:
                F.create_dataset("time[T]"      , data = np.array([time]))
                F.create_dataset("dt[T]"        , data = np.array([dt]))
                F.create_dataset("m"            , data = self.asnumpy(m))
                F.close()
        except:
           print("checkpoint file write failed at time = %.4E"%(time), " : ", "%s.h5"%(fprefix) )
        
        return
    
    def restore_checkpoint(self, fprefix):
        xp = self.xp_module
        try:
            with h5py.File("%s.h5"%(fprefix), 'r') as F:
                time = xp.array(F["time[T]"][()])[0]
                dt   = xp.array(F["dt[T]"][()])[0]
                m    = xp.array(F["m"][()])
                F.close()
        except:
           print("Error while reading the checkpoint file", " : ", "%s.h5"%(fprefix) )

        return time, dt, m

    def time_periodic_solve(self, m0, jxt, sxt, dt, ncycles):
        xp = self.xp_module
        M  = self.assemble_cycle_step_op(jxt, sxt, dt)
        m  = xp.copy(m0)
        mp = xp.copy(m)
        
        for cidx in range(ncycles):
            
            a1 = xp.linalg.norm(m[0] - mp[0]) / xp.linalg.norm(mp[0])
            a2 = xp.linalg.norm(m[1]/m[0] - mp[1]/mp[0]) / xp.linalg.norm(mp[1]/mp[0])

            ne = m[0]
            Te = m[1]/m[0]

            mp = xp.copy(m)

            print ("cycle = %04d ne : (%.4E, %4E) Te : (%.4E, %.4E) delta_ne_r = %.4E delta_Te_r = %.4E"
                   %(cidx, xp.min(ne), xp.max(ne), xp.min(Te), xp.max(Te), a1, a2))
            
            xx = bte.xp
            plt.figure(figsize=(12, 4), dpi=200)
            plt.subplot(1, 2, 1)
            plt.plot(xx, params.np0 * m[0]           , label="GM");
            plt.legend() 
            plt.xlabel(r"$\hat{x}$")
            plt.ylabel(r"$n_e[m^{-3}]$")
            plt.grid(visible=True)

            plt.subplot(1, 2, 2)
            plt.plot(xx, m[1]/m[0]        , label="GM");
            plt.legend()
            plt.xlabel(r"$\hat{x}$")
            plt.ylabel(r"$T_e [eV]$")
            plt.grid(visible=True)

            plt.suptitle("t = %.2E"%(cidx))
            plt.tight_layout()
            #plt.show()
            plt.savefig("%s_gm_cycle_%04d.png"%(params.fname, cidx))
            plt.close()

            m  = xp.einsum("ijk,ik->ij", M, m)

    def sample_fluxes_bte1d3v(self, Ext, v, ncycles, dt):
        bte              = self.bte_solver 
        xp_bte           = bte.xp_module
        bte.init(dt)

        vnp              = bte.asnumpy(v)
        m0               = gm_bte.mfuncs_ops_ords @ vnp
        freq             = int(1/dt)

        xp               = self.xp_module
        num_m            = len(self.mfuncs)
        num_t            = int(1/dt)

        m                = xp.zeros((3, num_t, num_m, params.Np))

        tt                = xp.linspace(0, 1, num_t + 1)[:-1]
        xx                = self.bte_solver.xp
        num_m             = len(self.mfuncs)

        dtt               = 0.5 * (tt[1] - tt[0])
        dxx               = 0.5 * (xx[1] - xx[0])
        extent            = [xx[0]-dxx , xx[-1] + dxx, tt[-1] + dtt, tt[0]-dtt]

        for tidx in range((int)(ncycles/dt)):
            tt                 = tidx * dt

            if (tidx > 0 and tidx % freq ==0):

                cidx = tidx // freq
                bte.store_checkpoint(v, 0.0, dt, "%s_%04d_cp"%(params.fname, cidx % 2))


                vnp      = bte.asnumpy(v)
                m_bte_np = gm_bte.mfuncs_ops_ords @ vnp
                
                a1 = np.linalg.norm(m_bte_np[0] - m0[0])/np.linalg.norm(m0[0])
                b1 = np.linalg.norm(m_bte_np[1] - m0[1])/np.linalg.norm(m0[1])
                c1 = np.linalg.norm(m_bte_np[1]/m_bte_np[0] - m0[1]/m0[0])/np.linalg.norm(m0[1]/m0[0])
                
                ne, Te, neTe = m_bte_np[0], m_bte_np[1]/m_bte_np[0], m_bte_np[1]
                print("cycle = %04d ne : (%.4E, %4E) Te : (%.4E, %.4E)  neTe : (%.4E, %.4E) -- delta_ne_r = %.4E delta_Te_r = %.4E delta_neTe_r = %.4E"%(tidx//freq,
                    xp.min(ne), xp.max(ne),
                    xp.min(Te), xp.max(Te),
                    xp.min(neTe), xp.max(neTe),
                    a1, c1, b1))
                
                m0       = np.copy(m_bte_np)
                
                
                plt.figure(figsize=(12, 4 * num_m), dpi=200)
                plt_idx = 1 

                jx = m[1]/m[0]
                sx = m[2]/m[0]

                for i in range(num_m):
                    plt.subplot(3, num_m, plt_idx + i)
                    
                    if (i == 0):
                        plt.imshow(m[0, :, i, :], extent=extent, aspect="auto")
                    else:
                        plt.imshow(m[0, :, i, :]/m[0, :, 0, :], extent=extent, aspect="auto")
                    
                    plt.xlabel(r"$\hat{x}$")
                    plt.ylabel(r"$t$")
                    plt.title(r"(%s)"%(self.mfuncs_names[i]))
                    plt.colorbar()
                
                for i in range(num_m):
                    plt.subplot(3, num_m, plt_idx +  num_m + i)
                    plt.imshow(jx[:, i, :], extent=extent, aspect="auto")
                    plt.xlabel(r"$\hat{x}$")
                    plt.ylabel(r"$t$")
                    plt.title(r"$\hat{J}$(%s)"%(self.mfuncs_names[i]))
                    plt.colorbar()

                for i in range(num_m):
                    plt.subplot(3, num_m, plt_idx + 2 * num_m + i)
                    plt.imshow(sx[:, i, :], extent=extent, aspect="auto")
                    plt.xlabel(r"$\hat{x}$")
                    plt.ylabel(r"$t$")
                    plt.title(r"$\hat{S}$(%s)"%(self.mfuncs_names[i]))
                    plt.colorbar()
                
                plt.tight_layout()
                plt.savefig("%s_moments_%04d.png"%(self.bte_params.fname, tidx // freq))
                plt.close()

                with h5py.File("%s_moments_%04d.h5"%(self.bte_params.fname, tidx // freq), 'w') as f:
                    f.create_dataset("m"  ,data =m[0])
                    f.create_dataset("jx" ,data =jx)
                    f.create_dataset("sx" ,data =sx)
                    f.create_dataset("tt" ,data =tt)
                    f.create_dataset("xx" ,data =xx)
                    f.close()



            u                  = v #bte.op_psh2o @ (bte.op_po2sh @ v)
            vnp                = bte.asnumpy(u)
            m[0, tidx % num_t] = self.mfuncs_ops_ords    @ vnp
            m[1, tidx % num_t] = self.mfuncs_Jx_ops_ords @ vnp
            m[2, tidx % num_t] = self.mfuncs_ops_ords    @ (bte.asnumpy(self.bte_vspace_mvec(v, Ext(tt))))
            
            v   = bte.step(Ext(tt), v, None, tt, dt, verbose=int(tidx %100 == 0))
            
        
        return

    def anderson_accleration(self, Ext, v, dimM, tol, verbose):
        dt  = self.params.dt
        bte = self.bte_solver 
        self.init(dt)

        def res(x):
            v0 = bte.xp_module.copy(x)
            v  = bte.xp_module.copy(x)

            m0 = self.mfuncs_ops_ords @ bte.asnumpy(v0)
                
            for tidx in range((int) (1/dt)):
                tt = tidx * dt
                v  = bte.step(Ext(tt), v, None, tt, dt, verbose=int(tidx%100==0))

            m  = self.mfuncs_ops_ords @ bte.asnumpy(v)
            return self.xp_module.linalg.norm(m-m0)/self.xp_module.linalg.norm(m0)
            #return #bte.asnumpy((v -v0).reshape((-1))  / self.xp_module.linalg.norm(v0.reshape((-1))))
        
        vc = scipy.optimize.anderson(res, v, M=dimM, f_tol=tol, verbose=verbose)
        return vc
    
    def bte1d3v(self, Ext, v, ncycles, dt, fprefix=""):
        params = self.bte_params
        bte    = self.bte_solver
        xp     = self.xp_module
        xp_bte = bte.xp_module
        v0     = xp_bte.copy(v)

        for it, p  in enumerate([(params.dt, ncycles), (params.dt/2, ncycles//10), (params.dt/4, ncycles//50)]):
            print(" it ", it, " dt " , p[0], " ncycles ", p[1])
            # if (it == 0):
            #     _, _, v = bte.restore_checkpoint("%s_%04d_cp"%(params.fname, 1))
            #     print("restored solver")
            #     continue
            gm_dt, ncycles   = p[0], p[1]
            gm_freq          = int(1/gm_dt)
            
            bte.init(gm_dt)
            vnp              = bte.asnumpy(v)
            m0               = gm_bte.mfuncs_ops_ords @ vnp

            for tidx in range((int)(ncycles/gm_dt) + 1):
                
                if (tidx % gm_freq ==0):
                    
                    vnp      = bte.asnumpy(v)
                    m_bte_np = gm_bte.mfuncs_ops_ords @ vnp
                    
                    a1       = xp.linalg.norm(m_bte_np[0] - m0[0])/xp.linalg.norm(m0[0])
                    b1       = xp.linalg.norm(m_bte_np[1] - m0[1])/xp.linalg.norm(m0[1])
                    c1       = xp.linalg.norm(m_bte_np[1]/m_bte_np[0] - m0[1]/m0[0])/xp.linalg.norm(m0[1]/m0[0])
                    
                    ne, Te, neTe = m_bte_np[0], m_bte_np[1]/m_bte_np[0], m_bte_np[1]
                    print("cycle = %04d ne : (%.4E, %4E) Te : (%.4E, %.4E)  neTe : (%.4E, %.4E) -- delta_ne_r = %.4E delta_Te_r = %.4E delta_neTe_r = %.4E"%(tidx//gm_freq,
                        xp.min(ne), xp.max(ne),
                        xp.min(Te), xp.max(Te),
                        xp.min(neTe), xp.max(neTe),
                        a1, c1, b1))
                    
                    m0       = np.copy(m_bte_np)
                    xx       = bte.xp

                    plt.figure(figsize=(12, 4), dpi=200)
                    plt.subplot(1, 2, 1)
                    plt.plot(xx, params.np0 * m_bte_np[0]       , label="BTE");
                    
                    plt.legend() 
                    plt.xlabel(r"$\hat{x}$")
                    plt.ylabel(r"$n_e[m^{-3}]$")
                    plt.grid(visible=True)

                    plt.subplot(1, 2, 2)
                    plt.plot(xx, m_bte_np[1]/m_bte_np[0], label="BTE");
                    plt.legend()
                    plt.xlabel(r"$\hat{x}$")
                    plt.ylabel(r"$T_e [eV]$")
                    plt.grid(visible=True)

                    plt.suptitle("t = %.2E"%(tidx * gm_dt))
                    plt.tight_layout()
                    plt.savefig("%s%s_it_%04d_idx_%04d.png"%(params.fname, fprefix, it, tidx//gm_freq))
                    plt.close()
                
                if (tidx  == (int)(ncycles/gm_dt)):
                    break
                
                v   = bte.step(Ext(tidx * gm_dt), v, None, tidx * gm_dt, gm_dt, verbose=int(tidx %100 == 0))
                
            
            bte.store_checkpoint(v, 0.0, gm_dt, "%s%s_%04d_cp"%(params.fname, fprefix, it))
        
        return v 
        
    




    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-par_file", "--par_file"                       , help="toml par file to specify run parameters" , type=str)
    parser.add_argument("-E",        "--E"                              , help="E magnitude"  , type=float)
    parser.add_argument("-wc",       "--wc"                             , help="warmup cycles", type=int)
    parser.add_argument("-run_1d3v", "--run_1d3v"                       , help="run 1d3V", type=int, default=0)
    parser.add_argument("-maxiter",  "--maxiter"                        , help="hybrid solver maxiter", type=int, default=1000)
    parser.add_argument("-hybrid_solver_freq",  "--hybrid_solver_freq"  , help="hybrid solver freq"   , type=int, default=5)
    parser.add_argument("-plot_mode","--plot_mode"                      , help="plot_mode"            , type=int, default=0)
    parser.add_argument("-rbte_with_gm","--rbte_with_gm"                , help="run bte solver with GM", type=int, default=0)
    parser.add_argument("-staticE","--staticE"                          , help="static E field", type=int, default=0)
    parser.add_argument("-gm_restore", "--gm_restore"                   , help="restore from gm_bte solver", type=str, default="")
    
    
    args    = parser.parse_args()
    gm_bte  = gm_bte_hybrid_solver(args)
    bte     = gm_bte.bte_solver
    params  = bte.params

    if(args.plot_mode==1):
        cp_idx = 0
        ff     = h5py.File("%s__gm_%04d_cp.h5"%(params.fname, cp_idx))
        # ne_bte = np.array(ff["ne[m^-3]"][()])
        # Te_bte = np.array(ff["Te[eV]"][()])
        v        = np.array(ff["edf"][()])
        m_bte    = gm_bte.mfuncs_ops_ords @ v 
        ne_bte   = m_bte[0]
        Te_bte   = m_bte[1]/m_bte[0]
        ff.close()

        cp_idx = 29
        ff     = h5py.File("%s_gm_cp_%04d.h5"%(params.fname, cp_idx))
        m_gm   = np.array(ff["m"][()])
        ne_gm  = m_gm[0] #np.array(ff["ne[m^-3]"][()])
        Te_gm  = m_gm[1] / m_gm[0] #np.array(ff["Te[eV]"][()])
        ff.close()

        xx = bte.xp

        plt.figure(figsize=(8, 4), dpi=300)
        plt.subplot(1, 2, 1)
        plt.plot(xx, ne_bte * params.np0 , label=r"$n_e$ (BTE)")
        plt.plot(xx, ne_gm  * params.np0 , label=r"$n_e$ (GM)")
        
        plt.xlabel(r"$\hat{x}$")
        plt.ylabel(r"$n_e$ [$m^{-3}$]")
        plt.legend()
        plt.grid(visible=True)

        plt.subplot(1, 2, 2)
        plt.plot(xx, Te_bte             , label=r"$T_e$ (BTE)")
        plt.plot(xx, Te_gm              , label=r"$T_e$ (GM)")
        
        plt.xlabel(r"$\hat{x}$")
        plt.ylabel(r"$T_e$ [eV]")
        plt.legend()
        plt.grid(visible=True)

        #plt.show()
        plt.tight_layout()
        plt.savefig("%s_converged_plot.png"%(params.fname))
        plt.close()

        sys.exit(0)

    
    if (params.use_gpu == 1):
        import cupy as cp
        bte.copy_operators_H2D(params.dev_id)
        bte.xp_module = cp
        cp.cuda.Device(params.dev_id).use()

        # reinitialization for the GPU
        bte.init(params.dt)


        # gm_bte.host_to_device(params.dev_id)
        # gm_bte.xp_module = cp

    v       = gm_bte.init()
    xp      = bte.xp_module
    
    ne      = gm_bte.mfuncs_ops_ords[0] @ v
    Te      = (gm_bte.mfuncs_ops_ords[1] @ v)/ne
    if (args.staticE == 1):
        Ext     = lambda t : bte.xp_module.ones(params.Np) * args.E #* bte.xp_module.sin(2 * bte.xp_module.pi * t)
    else:
        Ext     = lambda t : bte.xp_module.array(bte.xp)**3 * args.E * bte.xp_module.sin(2 * bte.xp_module.pi * t) 
        #Ext     = lambda t :bte.xp_module.ones(params.Np) * args.E * bte.xp_module.sin(2 * bte.xp_module.pi * t)


    # gm_bte.sample_fluxes_bte1d3v(Ext, xp.asarray(v), params.T, params.dt)
    # sys.exit(0)
    #vc = bte.anderson_solve(Ext, xp.asarray(v))

    if args.run_1d3v == 1:
        #v        = bte.maxwellian_eedf(ne * 1e-3, Te)
        #v         = bte.maxwellian_eedf(ne, Te)
        #_, _, v  = bte.restore_checkpoint("bte1d3v/E1e3/bte1d3v_gmbte_hybrid_cp_0001")
        #_, _, v  = bte.restore_checkpoint("bte1d3v/E1e2_static/bte1d3v_gmbte_hybrid_cp_0002") 
        #_, _, v           = bte.restore_checkpoint("bte1d3v/E1e3_static_40eV/bte1d3v_0000_cp") 
        gm_bte.bte1d3v(Ext, bte.xp_module.array(v), params.T, params.dt, "")
        
        
    
    if (args.gm_restore == ""):
        run_bte = (args.rbte_with_gm == 1)
        
        # _, _, v           = bte.restore_checkpoint("bte1d3v/E1e3_static_40eV/bte1d3v_0000_cp") 
        # ne                =  gm_bte.mfuncs_ops_ords[0] @ bte.asnumpy(v)
        # Te                = (gm_bte.mfuncs_ops_ords[1] @ bte.asnumpy(v))/ne

        rtol    = 1e-3
        ne      = ne * 1e-4
        ne0     = gm_bte.xp_module.copy(ne)

        for it in range(0, args.maxiter):
            ne_tol = gm_bte.xp_module.linalg.norm(ne0 - ne) / gm_bte.xp_module.linalg.norm(ne0)
            
            if (it > 0):
                print("it = %04d ne_tol = %.4E"%(it, ne_tol))
                ne0    = gm_bte.xp_module.copy(ne)

                if (it > 0 and ne_tol < rtol):
                    print("ne relative change : %.4E"%(ne_tol))
                    break
            
            mxt, jxt, sxt, vT , delta_ne, delta_Te = gm_bte.eval_bte(gm_bte.asnumpy(ne), gm_bte.asnumpy(Te), Ext, params.dt, args.wc, it)
            #bte.store_checkpoint(xp.asarray(vT), 0.0, params.dt, "%s_gmbte_hybrid_cp_%04d"%(params.fname, it))
            

            #Mop               = gm_bte.assemble_cycle_step_op(jxt, sxt, params.dt)
            gm_dt             = params.dt
            gm_freq           = int(1/ gm_dt)
            ncycles           = args.hybrid_solver_freq
            v                 = xp.asarray(vT) 
            vnp               = gm_bte.bte_solver.asnumpy(v)
            m                 = gm_bte.mfuncs_ops_ords @ vnp

            Te0               = m[1]/m[0]

            m0_bte_np         = np.copy(m)
            m0_np             = np.copy(m)
            for tidx in range((int)(ncycles/gm_dt) + 1):
                
                if (tidx % gm_freq ==0):
                    vnp      = gm_bte.bte_solver.asnumpy(v)
                    m_bte    = gm_bte.mfuncs_ops_ords @ vnp
                    m_bte_np = gm_bte.asnumpy(m_bte)
                    m_np     = gm_bte.asnumpy(m)

                    a1 = np.linalg.norm(m_bte_np[0] - m0_bte_np[0]) / np.linalg.norm(m0_bte_np[0])
                    b1 = np.linalg.norm(m_np[0] - m0_np[0]) / np.linalg.norm(m0_np[0])


                    a2 = np.linalg.norm(m_bte_np[1]/m_bte_np[0] - m0_bte_np[1]/m0_bte_np[0]) / np.linalg.norm(m0_bte_np[1]/m0_bte_np[0])
                    b2 = np.linalg.norm(m_np[1]/m_np[0]         - m0_np[1]/m0_np[0]) / np.linalg.norm(m0_np[1]/m0_np[0])


                    print("it ", it, " cycle = %04d bte ne : (%.4E, %.4E) gm ne : (%.4E, %.4E) | bte Te : (%.4E, %.4E) gm Te : (%.4E, %.4E) | delta_ne (gm): %.4E delta_ne (bte): %.4E delta_Te (gm): %.4E delta_Te (bte): %.4E"%(tidx//gm_freq, np.min(m_bte_np[0]), np.max(m_bte_np[0]),
                                                                        np.min(m_np[0]), np.max(m_np[0]),
                                                                        np.min(m_bte_np[1]/m_bte_np[0]), np.max(m_bte_np[1]/m_bte_np[0]),
                                                                        np.min(m_np[1]/m_np[0]), np.max(m_np[1]/m_np[0]),
                                                                        b1, a1, b2, a2))
                    

                    m0_bte_np = np.copy(m_bte_np)
                    m0_np     = np.copy(m_np)

                    xx = bte.xp
                    plt.figure(figsize=(12, 4), dpi=200)
                    plt.subplot(1, 2, 1)
                    
                    if run_bte==True:
                        plt.plot(xx, params.np0 * m_bte_np[0]       , label="BTE");
                    
                    plt.plot(xx, params.np0 * m_np[0]           , label="GM");
                    plt.legend() 
                    plt.xlabel(r"$\hat{x}$")
                    plt.ylabel(r"$n_e[m^{-3}]$")
                    plt.grid(visible=True)

                    plt.subplot(1, 2, 2)
                    
                    if run_bte==True:
                        plt.plot(xx, m_bte_np[1]/m_bte_np[0], label="BTE");
                    
                    plt.plot(xx, m_np[1]/m_np[0]        , label="GM");
                    plt.legend()
                    plt.xlabel(r"$\hat{x}$")
                    plt.ylabel(r"$T_e [eV]$")
                    plt.grid(visible=True)

                    plt.suptitle("t = %.2E"%(tidx * gm_dt))
                    plt.tight_layout()
                    plt.savefig("%s_gm_bte_iter_%04d_%04d.png"%(params.fname, it, tidx//gm_freq))
                    plt.close()

                    #m = gm_bte.xp_module.einsum("ijk,ik->ij", Mop, m)
                
                if (tidx == (int)(ncycles/gm_dt)):
                    break

                m   = gm_bte.step(m, jxt, sxt, tidx * gm_dt, gm_dt, tidx)
                # #m   = (LpD @ m.T).T
                if run_bte==True:
                    v   = gm_bte.bte_solver.step(Ext(tidx * gm_dt), v, None, tidx * gm_dt, gm_dt, verbose=0)
                    

            ne, Te = m[0], m[1]/m[0]
            gm_bte.store_checkpoint(m, 0.0, params.dt, "%s_gm_cp_%04d"%(params.fname, it))
            #Te     = Te0 #gm_bte.xp_module.ones_like(Te) * params.Te

    else:
        _, _, m = gm_bte.restore_checkpoint(args.gm_restore)
        ne      = m[0]
        Te      = m[1]/ne

    v       = bte.maxwellian_eedf(ne, Te)

    gm_bte.bte1d3v(Ext, v, 100, params.dt, fprefix="__gm")

    
        
        
