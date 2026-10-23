"""
Planar probe implementation for plasma diagnostics, using 1D2V BTE model with prescribed potential. 
"""

import numpy as np
import argparse
import bte_1d3v_solver
import enum
import sys
import matplotlib.pyplot as plt
import utils as bte_utils
import scipy.constants
from time import perf_counter, sleep
import matplotlib as mpl
from datetime import datetime
bte_utils.use_pgfplots_style(mpl)

try:
    import cupy as cp
except:
    print("cupy module not found")


class qoi_idx():
    NE_IDX = 0 
    TE_IDX = 1
    LAST   = 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-par_file", "--par_file" , help="toml par file to specify run parameters" , type=str)
    parser.add_argument("-eta", "--eta"           , help="drift velocity factor (used for initialization)" , type=float, default=0.0)
    
    args    = parser.parse_args()
    bte     = bte_1d3v_solver.bte_1d3v(args)
    params  = bte.params
    v       = bte.initial_condition(type=0, eta = args.eta)

    if params.use_gpu == 1:
        print("using GPUs")
        cp.cuda.Device(params.dev_id).use()
        bte.copy_operators_H2D(params.dev_id)
        v             = cp.array(v)
        bte.xp_module = cp

    output_cycle_averaged_qois = True
    xp    = bte.xp_module

    dt    = params.dt
    steps = int(params.T / params.dt)+1

    io_freq    = int(np.round(params.io_freq/dt))
    cp_freq    = int(np.round(params.cp_freq/dt))
    cycle_freq = int(1/params.dt)

    print("io freq = %d cycle_freq = %d cp_freq = %d" %(io_freq, cycle_freq, cp_freq))
    ts_idx_b   = 0 

    if params.restore==1:
        ts_idx_b  = int(params.rs_idx * cp_freq)
        tt        = ts_idx_b * dt
        print("restoring solver from ts_idx = ", int(params.rs_idx * cp_freq), "at time = ",tt)
        tt, _ , v = bte.restore_checkpoint("%s_cp_%04d"%(params.fname, ts_idx_b//cp_freq))
        
        
    num_p               = bte.op_spec_sp._p + 1
    num_sh              = len(bte.op_spec_sp._sph_harm_lm)

    def qoi(v, xp):

        if (bte.params.vspace_type == bte_1d3v_solver.vspace_discretization.SPECTRAL_BSPH):
            vsh                 = bte.ords_to_sph(v)
            qoi                 = xp.zeros((qoi_idx.LAST, params.Np))
            qoi[qoi_idx.NE_IDX] = bte.op_mass @ vsh
            qoi[qoi_idx.TE_IDX] = (bte.op_temp @ vsh) / qoi[qoi_idx.NE_IDX]
        elif(bte.params.vspace_type == bte_1d3v_solver.vspace_discretization.FVM):
            qoi                 = xp.zeros((qoi_idx.LAST, params.Np))
            qoi[qoi_idx.NE_IDX] = bte.op_mass  @ v
            qoi[qoi_idx.TE_IDX] = (bte.op_temp @ v) / qoi[qoi_idx.NE_IDX]

        return qoi

    u0   = qoi(v, xp)
    u1   = xp.copy(u0)


    tp_rtol = 1e-10

    Emax        = bte.params.tau * 4e4 * bte.params.qe / bte.params.me / bte.bs_vth
    dx          = np.min(bte.xp[1:] - bte.xp[0:-1])
    dvr         = np.min(bte.xp_vr[1:] - bte.xp_vr[0:-1])
    dvt         = np.min((-bte.xp_vt[1:] + bte.xp_vt[0:-1]))

    if bte.params.solver_type == bte_1d3v_solver.solver_type.FULLY_IMPLICIT:
        r_ax        = 1/(1 + (bte.bs_vth * bte.params.tau/bte.params.L) *(bte.op_spec_sp._basis_p._kdomain[1] * dt) / dx )
        r_vr        = r_ax #1/(1 + Emax * dt/dvr)
        r_vt        = r_ax #1/(1 + Emax * dt/dvt/bte.xp_vr[0])
    else:
        r_ax        = 1.0
        r_vr        = 1.0
        r_vt        = 1.0

    bte.init(dt, 1, r_ax)
    # vn  = bte.normalize_edf(v)
    # mn  = xp.dot(bte.op_mass, v)
    # a1  = xp.dot(bte.op_mass, xp.dot(bte.op_col_en, v/mn))
    # a2  = xp.dot(bte.op_rate[1], vn)
    # print(a1,"\n",  a2)
    #print()
    neb_r = u0[qoi_idx.NE_IDX, -1]
    Teb_r = u0[qoi_idx.TE_IDX, -1]
    qe    = bte.params.qe
    me    = bte.params.me

    print("neb_r = %.4E, Teb_r = %.4E, qe = %.4E, me = %.4E"%(neb_r* bte.params.np0, Teb_r, qe, me))
    Ia    = lambda vp: 0.25 * neb_r * qe * np.sqrt(8 * qe * Teb_r / (np.pi * me)) * bte.params.np0 * np.exp(min(0, vp/Teb_r))
    y1    = xp.copy(v).reshape((bte.dof_vr,bte.dof_vt, bte.dof_x))[:, :,-1]
    y1    = xp.einsum("i,j,ij->ij", bte.xp_vr**2, xp.ones_like(bte.xp_vt), y1)
    y1    = 2 * np.pi * xp.einsum("i,j,ij->", bte.xp_vr_qw, bte.xp_vt_qw, y1) * bte.params.np0
    print("y1 = %.4E, Isat = %.4E, I/Isat = %.4E"%(y1, Ia(0), y1/Ia(0)))

    vb_R  = xp.copy(v).reshape((bte.dof_vr,bte.dof_vt, bte.dof_x)) [:, bte.idx_sp_vt, -1]
    xx    = xp.asarray(bte.xp)
    a     = 0.5

    def solve_probe(vp, v0, rtol=1e-6, max_cycles=params.T):

        Ext   = lambda t : vp * np.exp(-((xx+1)/a)**2) * 2 * (xx+1) / a**2
        v     = xp.copy(v0)
        
        for dt_i, dt in enumerate([1e2 * params.dt, 2.5e1 * params.dt]):#:, 1e0 * params.dt]):
            
            steps      = int(max_cycles/dt)+1
            io_freq    = int(np.round(params.io_freq/dt))
            cp_freq    = int(np.round(params.cp_freq/dt))
            cycle_freq = int(1/dt)

            bte.init(dt, 1, r_ax)
            u0          = qoi(v, xp)
            u1          = xp.copy(u0)


            for ts_idx in range(0, steps):
                tt   = ts_idx * dt
                # if (ts_idx % cycle_freq == 0):
                #     bte.plot(Ext(tt), v, "%s_vp%.4f_cycle_%04d.png"%(params.fname, vp, ts_idx//cycle_freq), tt)

                if (ts_idx % cycle_freq == 0 and ts_idx > 0):
                    u1 = qoi(v,xp)
                    a1 = xp.max(xp.abs(u1[:,0] -u0[:, 0]))
                    a2 = xp.max(xp.abs((u1[:,0]-u0[:,0]) / xp.max(xp.abs(u0[:,0]))))
                    u0 = u1
                    
                    print("ne : ||u(t+T) - u(t)|| = %.8E and ||u(t+T) - u(t)||/||u(t)|| = %.8E"% (a1, a2))

                    if (a2 < rtol):
                        print("dt = %.4E -- converged at ts_idx = %d, time = %.2E"%(dt, ts_idx, tt))
                        break

                v  = bte.step(Ext(tt), v, None, tt, dt, 0, vb_R, (ts_idx % (cycle_freq//4) == 0),r_vr, r_vt, pc_left = False)

            bte.plot(Ext(0), v, "%s_vp%.4f_%04d.png"%(params.fname, vp, dt_i), 0)
            bte.store_checkpoint(v, 0, 0, "%s_vp%.4f_cp_%04d"%(params.fname, vp, dt_i))

            vb_L = v.reshape((bte.dof_vr,bte.dof_vt, bte.dof_x)) [:, :, 0]
            vb_L = xp.einsum("i,j,ij->ij", bte.xp_vr**3, bte.xp_cos_vt, vb_L)
            I    = xp.pi * 2 * bte.qe * bte.params.np0 * bte.bs_vth* xp.abs(xp.einsum("i,j,ij->", bte.xp_vr_qw, bte.xp_vt_qw, vb_L))

            print("dt = %.4E, vp = %.4f, I = %.4E, Isat = %.4E, I/Isat = %.4E"%(dt, vp, I, Ia(vp), I/Ia(vp)))


        vb_L = v.reshape((bte.dof_vr,bte.dof_vt, bte.dof_x)) [:, :, 0]
        print("vb_L shape = ", vb_L.shape)
        print("min, max", xp.min(xp.abs(vb_L[:, bte.idx_np_vt])), xp.max(xp.abs(vb_L[:, bte.idx_np_vt])))
        
        vb_L = xp.einsum("i,j,ij->ij", bte.xp_vr**3, bte.xp_cos_vt, vb_L)
        I    = xp.pi * 2 * bte.qe * bte.params.np0 * bte.bs_vth* xp.abs(xp.einsum("i,j,ij->", bte.xp_vr_qw, bte.xp_vt_qw, vb_L))
        
        return v, I ,a2

    vp    = np.append(np.linspace(-300, 0, 20), np.linspace(0.5, 30, 4))#np.linspace(-30, 30, 30)
    v0    = xp.copy(v)
    Ip    = np.zeros_like(vp)

    vp    = np.flip(vp)
    for i, vpi in enumerate(vp):
        v, I, rtol = solve_probe(vpi, v, rtol=1e-6, max_cycles=params.T)
        Ip[i]      = I

    vp = np.flip(vp)
    Ip = np.flip(Ip)
    #bte.plot(Ext(0), v, "%s_%04d.png"%(params.fname, 0), 0)
    np.savetxt("%s_iv_curve.txt"%(params.fname), np.vstack((vp * bte.params.L, Ip)).T, header="vp [V], Ip [A]")
    plt.figure(figsize=(6,4),dpi=300)
    plt.plot(vp, Ip, "-o", label="1D2V BTE")
    plt.xlabel("Probe potential [V]")
    plt.ylabel("Probe current [A]")
    plt.title("Planar probe I-V curve")
    plt.grid() 
    plt.grid(which='minor', alpha=0.2)
    plt.savefig("%s_iv_curve.png"%(params.fname), bbox_inches='tight', dpi=300)


