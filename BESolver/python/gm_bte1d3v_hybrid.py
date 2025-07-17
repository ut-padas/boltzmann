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

def fft_filtering(tt, q, rk, xp):
    assert len(q.shape) == 2
    ## fft filtering
    fk = xp.fft.rfftfreq(len(tt), d=1/len(tt))
    qh = xp.fft.rfft(q, axis=0)
    assert rk < len(fk)

    sf          = xp.ones_like(fk) * (2/(len(tt)))
    sf[0]       = (1/(len(tt)))
    qh          = sf[:, np.newaxis] * qh
    qh          = np.concatenate(((np.real(qh))[:, :, np.newaxis],(np.imag(qh))[:, :, np.newaxis]), axis=2)
    qh          = np.swapaxes(qh, 1, 2)

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

        self.mfuncs         = [ lambda vr, vt, vp : np.ones_like(vr),
                                lambda vr, vt, vp : vth**2 * vr **2 * (2/3/c_gamma**2),
                                #lambda vr, vt, vp : vth    * vr * np.cos(vt),
                                ]
        
        self.mfuncs_Jx      = [ lambda vr, vt, vp:  vth * vr * np.cos(vt)  * np.ones_like(vr), 
                                lambda vr, vt, vp:  vth * vr * np.cos(vt)  * vth**2 * vr **2 * (2/3/c_gamma**2),
                                #lambda vr, vt, vp:  vth * vr * np.cos(vt)  * vth    * vr * np.cos(vt),
                                ]

        
        self.mfuncs_ops_ords    = bte_utils.assemble_moment_ops_ords(self.spec_sp, self.vt, self.vt_w, self.num_vr, self.num_vp, self.mfuncs, scale=1.0)
        self.mfuncs_Jx_ops_ords = bte_utils.assemble_moment_ops_ords(self.spec_sp, self.vt, self.vt_w, self.num_vr, self.num_vp, self.mfuncs_Jx, scale=1.0)

    def bte_vspace_mvec(self, v, E):
        params = self.bte_params
        bte    = self.bte_solver

        assert v.shape == ((params.Nr+1) * params.Nvt, params.Np)

        xp     = bte.xp_module
        x      = bte.op_po2sh @ v 
        y      = params.tau * (params.n0 * params.np0 * (xp.dot(bte.op_col_en, x) + params.Tg * xp.dot(bte.op_col_gT, x))  + E * xp.dot(bte.op_adv_v, x))
        y      = bte.op_psh2o @ y 

        return y
    
    def eval_bte(self, ne, Te, Ext, sfreq:float, warmup_cycles:int):
        bte     = self.bte_solver
        params  = self.bte_params

        xp      = bte.xp_module

        v       = bte.maxwellian_eedf(ne, Te)
        dt      = params.dt
        
        print("Warmup cycles begin")

        
        # for tidx in range(warmup_cycles * (int) (1/dt) + 1):
        #     tt  = tidx * dt
        #     v   = bte.step(Ext(tt), v, None, tt, dt, verbose=int(tidx%100==0))

        print("Warmup cycles end")

        assert sfreq >= dt
        sample_freq  = int(sfreq/dt)
        assert abs(dt * sample_freq - sfreq) < 1e-14
        
        num_m = len(self.mfuncs)
        num_t = int(1/sfreq)
        m     = xp.zeros((4, num_t, num_m, params.Np))

        vT    = xp.copy(v)
        m0    = self.mfuncs_ops_ords @ v
        for tidx in range((int) (1/dt)):
            tt = tidx * dt

            if (tidx % sample_freq == 0):
                idx       = tidx // sample_freq 
                
                m[0, idx] = self.mfuncs_ops_ords @ v
                m[1, idx] = self.mfuncs_Jx_ops_ords @ v
                fv        = self.bte_vspace_mvec(v, Ext(tt))
                m[2, idx] = self.mfuncs_ops_ords @ fv
                vT        = xp.copy(v)

            v   = bte.step(Ext(tt), v, None, tt, dt, verbose=int(tidx%100==0))
        
        m1      = self.mfuncs_ops_ords @ v 
        print("ne relative change = %.4E"%(xp.linalg.norm(m0[0]-m1[0])/xp.max(m0[0])))

        ## fft filtering
        jx = m[1] / m[0]
        sx = m[2] / m[0]
        
        jxt = list()
        sxt = list()
        
        tt   = xp.linspace(0, 1, num_t + 1)[:-1]

        for i in range(num_m):
            jxt.append(fft_filtering(tt, jx[:, i, :], num_t//2, xp))
            sxt.append(fft_filtering(tt, sx[:, i, :], num_t//2, xp))
            
        # print(jx[:, 0, 0])
        # print(jx[:, 0, -1])
        #print(jx[:, 1, :])

        return m, jxt, sxt, vT
    
    def init(self, type=0):
        v = self.bte_solver.initial_condition(type)
        return v 

    def step(self, m, jxt, sxt, time, dt):
        bte  = self.bte_solver
        xp   = bte.xp_module

        params= bte.params
        num_m = len(self.mfuncs)
        Imat  = bte.I_Nx
        Dp    = bte.Dp

        mp    = xp.zeros_like(m)
        r     = 0.5
        for i in range(num_m):
            jxp  = jxt[i](time + dt)
            sxp  = sxt[i](time + dt)
            
            jxm  = jxt[i](time)
            sxm  = sxt[i](time)
            
            
            # a2   = (Dp @ jxm) * m[i] + (Dp @ m[i]) * jxm
            # a1   = ((Dp @ jxm) * Imat +(xp.diag(jxm) @ Dp)) @ m[i] 
            #print("libniz: %.2E"%(xp.linalg.norm(a1-a2)/xp.linalg.norm(a1)))

            # Lmat = Imat + (1-r) * dt * (-sxp * Imat + (params.tau/params.L) * (Dp @ xp.diag(jxp)))
            # bvec = m[i] + r * dt * (sxm * m[i] - (params.tau/params.L) * (Dp @ (jxm * m[i])))
            
            Lmat = Imat + (1-r) * dt * (-sxp * Imat + (params.tau/params.L) * ((Dp @ jxp) * Imat + xp.diag(jxp) @ Dp))
            bvec = m[i] + r * dt * (sxm * m[i] - (params.tau/params.L) * ((Dp @ jxm) * m[i] + (Dp @ m[i]) * jxm))
            #bvec = m[i] + r * dt * (sxm * m[i] - (params.tau/params.L) * (Dp @ (jxm * m[i])))

            mp[i]= xp.linalg.solve(Lmat, bvec)

        return mp

    def evolve(self, m, jxt, sxt, T, dt):
        bte        = self.bte_solver
        xp         = bte.xp_module
        
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-par_file", "--par_file" , help="toml par file to specify run parameters" , type=str)
    args    = parser.parse_args()

    gm_bte  = gm_bte_hybrid_solver(args)
    v       = gm_bte.init()

    ne      = gm_bte.mfuncs_ops_ords[0] @ v
    Te      = (gm_bte.mfuncs_ops_ords[1] @ v)/ne


    bte     = gm_bte.bte_solver
    xp      = bte.xp_module
    params  = bte.params

    Ext     = lambda t : xp.ones(params.Np) * 1e3 * xp.sin(2 * xp.pi * t)

    #print(ne)
    #print(Te)
    #mxt1, jxt1, sxt1 = gm_bte.eval_bte(ne, Te, Ext, 2 * params.dt, 2)
    mxt, jxt, sxt, vT = gm_bte.eval_bte(ne, Te, Ext, params.dt, 1)
    

    gm_dt             = params.dt
    gm_freq           = int(1e-2 / gm_dt)
    ncycles           = 5
    #m                 = xp.copy(mxt[0,-1])
    v                 = gm_bte.bte_solver.maxwellian_eedf(ne, Te)#xp.copy(vT)
    m                 = gm_bte.mfuncs_ops_ords @ v

    for tidx in range((int)(ncycles/gm_dt) + 1):
        m_bte = gm_bte.mfuncs_ops_ords @ v
        if (tidx % gm_freq ==0):
            xx = bte.xp
            plt.figure(figsize=(12, 4), dpi=200)
            plt.subplot(1, 2, 1)
            plt.plot(xx, params.np0 * m_bte[0]       , label="BTE");
            plt.plot(xx, params.np0 * m[0]           , label="GM");
            plt.legend() 
            plt.xlabel(r"$\hat{x}$")
            plt.ylabel(r"$n_e[m^{-3}]$")
            plt.grid(visible=True)

            plt.subplot(1, 2, 2)
            plt.plot(xx, m_bte[1]/m_bte[0], label="BTE");
            plt.plot(xx, m[1]/m[0]        , label="GM");
            plt.legend()
            plt.xlabel(r"$\hat{x}$")
            plt.ylabel(r"$T_e [eV]$")
            plt.grid(visible=True)

            plt.suptitle("t = %.2E"%(tidx * gm_dt))
            plt.tight_layout()
            #plt.show()
            plt.savefig("%s_%04d.png"%(params.fname,tidx//gm_freq))
            plt.close()
        
        m = gm_bte.step(m, jxt, sxt, tidx * gm_dt, gm_dt)
        v = gm_bte.bte_solver.step(Ext(tidx * gm_dt), v, None, tidx * gm_dt, gm_dt, verbose=0)











    # while(True):
    #     mxt, jxt, sxt     = gm_bte.eval_bte(ne, Te, Ext, params.dt, 1)
    #     m0                = mxt[0, -1]
    #     m                 = xp.copy(m0)

    #     gm_dt             = 2e-4
    #     gm_freq           = int(params.dt / gm_dt)

    #     m                 = gm_bte.evolve(m0, jxt, sxt, 5, gm_dt)

    #     ne                = m[0]
    #     Te                = m[1]/m[0]

        # for tidx in range((int)(1/gm_dt)):
            
        #     if (tidx % gm_freq ==0):
        #         xx = bte.xp
        #         plt.figure(figsize=(16, 8), dpi=200)
        #         plt.subplot(1, 2, 1)
        #         plt.plot(xx, mxt[0, tidx//gm_freq, 0], label="BTE");
        #         plt.plot(xx, m[0]           , label="GM");
        #         plt.legend() 
        #         plt.grid(visible=True)

        #         plt.subplot(1, 2, 2)
        #         plt.plot(xx, mxt[0, tidx//gm_freq, 1]/mxt[0, tidx//gm_freq, 0], label="BTE");
        #         plt.plot(xx, m[1]/m[0]                                        , label="GM");
        #         plt.legend()
        #         plt.grid(visible=True)

        #         plt.suptitle("t = %.2E"%(tidx * gm_dt))
        #         plt.tight_layout()
        #         #plt.show()
        #         plt.savefig("%s_%04d.png"%(params.fname,tidx//gm_freq))
        #         plt.close()


        #     m = gm_bte.step(m, jxt, sxt, tidx * gm_dt, gm_dt)


    # m             = gm_bte.evolve(m0, jxt, sxt, 1, params.dt)

    # xx = bte.xp
    # plt.subplot(1, 2, 1)
    # plt.plot(xx, m0[0], label="IC");
    # plt.plot(xx, mxt[0, 1, 0], label="BTE");
    # plt.plot(xx, m[0]         , label="GM");
    # plt.legend(); 

    # plt.subplot(1, 2, 2)
    # plt.plot(xx, m0[1]/m0[0], label="IC");
    # plt.plot(xx, mxt[0, 1, 1]/mxt[0, -1, 0], label="BTE");
    # plt.plot(xx, m[1]/m[0]         , label="GM");
    # plt.legend(); 

    # plt.tight_layout()
    # plt.show()
        
        
