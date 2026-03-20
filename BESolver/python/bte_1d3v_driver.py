import numpy as np
import argparse
import bte_1d3v_solver
import enum
import sys
import matplotlib.pyplot as plt

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
    parser.add_argument("-ef_mode" , "--ef_mode",   help="E-field mode" , type=int)
    parser.add_argument("-benchmark" , "--benchmark",   help="fully-implicit benchmark solve" , type=int, default=0)
    
    args    = parser.parse_args()
    bte     = bte_1d3v_solver.bte_1d3v(args)
    params  = bte.params

    v       = bte.initial_condition(type=0)
    if params.use_gpu == 1:
        print("using GPUs")
        cp.cuda.Device(params.dev_id).use()
        bte.copy_operators_H2D(params.dev_id)
        v             = cp.array(v)
        bte.xp_module = cp
    
    output_cycle_averaged_qois = True
    xp    = bte.xp_module
    if (args.ef_mode == 0):
        Ext   = lambda t : xp.ones(params.Np) * 1e4 * xp.sin(2 * xp.pi * t)
    elif (args.ef_mode == 1):
        xx    = xp.asarray(bte.xp)
        Ext   = lambda t : xx**3 * 1e4 * xp.sin(2 * xp.pi * t)
    elif (args.ef_mode == 2):
        import h5py
        import scipy.interpolate
        ff        = h5py.File("1dglow_fluid_Nx400/1Torr300K_100V_Ar_3sp2r_tab_cycle/macro.h5", 'r')
        tt_sp     = np.array(ff["time[T]"][()])
        E_sp      = np.array(ff['E[Vm^-1]'][()])
        xx_sp     = np.array(ff['x[-1,1]'][()])
        if(len(xx_sp)!=bte.params.Np):
            print("resampling E field to match BTE grid")
            E_sp      = scipy.interpolate.interp1d(xx_sp, E_sp, axis=1, bounds_error=True)(np.asarray(bte.xp))

        Et_inp    = scipy.interpolate.interp1d(tt_sp, E_sp, axis=0, bounds_error=True)

        dt        = bte.params.dt
        
        def Ext(t):
            idx  = int(np.round(t / dt))
            cfrq = int(1 / dt)
            if (t > 0):
                assert (np.abs(idx * dt - t)/np.abs(t) < 1e-10) , "time = %.14E idx * dt = %.14E"%(t, idx * dt)
            tta  = (idx % cfrq) * dt
            return xp.array(Et_inp(tta))
            #return xp.array(Et_inp(0.25))
    elif (args.ef_mode == 3):
        Ext   = lambda t : xp.ones(params.Np) * 1e4
    elif (args.ef_mode == 4):
        Ext   = lambda t : xp.ones(params.Np) * 0.0
    elif (args.ef_mode == 5):
        Ext   = lambda t : xp.ones(params.Np) * 1e3

    else:
        raise NotImplementedError
    
    if args.benchmark == 1:
        ## benchmark the fully-implicit solver

        _rtol   = params.rtol
        _dt     = params.dt


        Emax = bte.params.tau * bte.params.qe * 4e4 / bte.params.me/ bte.bs_vth
        dtx  = np.min(bte.xp[1:] - bte.xp[0:-1]) / (bte.op_spec_sp._basis_p._kdomain[1] * bte.bs_vth  * bte.params.tau/bte.params.L)
        dtvr = np.min(bte.xp_vr[1:] - bte.xp_vr[0:-1]) / Emax
        dtvt = np.min(bte.xp_vt[0:-1]- bte.xp_vt[1:]) * np.min(bte.xp_vr) / Emax
        dtex = min(dtx, min(dtvr, dtvt))

        print("explicit dt = %.4E min(dtx = %.4E, dtvr = %.4E dtvt = %.4E)"%(dtex, dtx, dtvr, dtvt))
        print("dt base: %.2E"%(_dt/dtex))
        
        gmres_info = [list(), list(), list(), list()]
        #dt_info    = np.array([dtex, 10 * dtex, 100 * dtex, 500 * dtex, 1000 * dtex])
        dt_info    = np.array([1e-5, 1e-4, 5e-4, 1e-3, 2e-3])
        rtol_info  = np.array([1e-12]           )      

        ndt        = dt_info.shape[0]
        nrtol      = rtol_info.shape[0]
        nE         = 2

        
        iter_counts = np.zeros((nE, ndt, nrtol))
        res_rel     = np.zeros((nE, ndt, nrtol))

        dx          = np.min(bte.xp[1:] - bte.xp[0:-1])
        dvr         = np.min(bte.xp_vr[1:] - bte.xp_vr[0:-1])
        dvt         = np.min((-bte.xp_vt[1:] + bte.xp_vt[0:-1]))


        print("===========================================")
        print("=====FULLY IMPLICIT SCHEME BENCMARK========")
        print("===========================================")

        for i in range(ndt):
            rt = rtol_info[0]
            dt = dt_info[i]

            print("\n\nrunning : dt = %.4E rtol = %.4E"%(dt, rt))

            params.dt   = dt
            params.rtol = rt

            
            Emax        = bte.params.tau * 4e4 * bte.params.qe / bte.params.me / bte.bs_vth
            
            r_ax        = 1/(1 + (bte.bs_vth * bte.params.tau/bte.params.L) *(bte.op_spec_sp._basis_p._kdomain[1] * dt) / dx )
            r_vr        = r_ax #1/(1 + Emax * dt/dvr)
            r_vt        = r_ax #1/(1 + Emax * dt/dvt/bte.xp_vr[0])

            print("GMRES relaxation parameters rx = %.4E, rvr = %.4E rvt = %.4E"%(r_ax, r_vr, r_vt))
            bte.init(dt, gmres_pc_ax=r_ax)
            params.pc_type = bte_1d3v_solver.pc_type.CXVTVR
            try:
                y0  = bte.step(Ext(0)   , v, None, 0.0, dt, params.verbose, gmres_pc_vr=r_vr, gmres_pc_vt=r_vt)
            except RuntimeError:
                print("fully-implicit solve did not converge")

            gmres_info[0].append(bte.gmres_info["gmres_callback"])


            params.pc_type = bte_1d3v_solver.pc_type.XCVTVR
            try:
                y0  = bte.step(Ext(0)   , v, None, 0.0, dt, params.verbose, gmres_pc_vr=r_vr, gmres_pc_vt=r_vt)
            except RuntimeError:
                print("fully-implicit solve did not converge")
            gmres_info[1].append(bte.gmres_info["gmres_callback"])


            params.pc_type = bte_1d3v_solver.pc_type.XVTVRC
            try:
                y0  = bte.step(Ext(0)   , v, None, 0.0, dt, params.verbose, gmres_pc_vr=r_vr, gmres_pc_vt=r_vt)
            except RuntimeError:
                print("fully-implicit solve did not converge")

            gmres_info[2].append(bte.gmres_info["gmres_callback"])

            params.pc_type = bte_1d3v_solver.pc_type.NONE
            try:
                y0  = bte.step(Ext(0)   , v, None, 0.0, dt, params.verbose, gmres_pc_vr=r_vr, gmres_pc_vt=r_vt)
            except RuntimeError:
                print("fully-implicit solve did not converge")
            
            gmres_info[3].append(bte.gmres_info["gmres_callback"])
        

        import matplotlib.ticker

        fig, axes = plt.subplots(1, ndt, figsize=(4 * ndt, 5), dpi=200)
        for ax in axes.flatten():
            ax.minorticks_on() # Explicitly turn on minor ticks
            ax.grid(visible=True, which='both', axis='both', linestyle='--', linewidth=0.25)
            y_major = matplotlib.ticker.LogLocator(base = 10.0, numticks = 10)
            ax.yaxis.set_major_locator(y_major)

            x_major = matplotlib.ticker.MaxNLocator(integer=True)
            ax.xaxis.set_major_locator(x_major)

            y_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
            ax.yaxis.set_minor_locator(y_minor)
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

        for i in range(ndt):
            plt.subplot(1, ndt, i+1)
            
            x0 = np.arange(1, len(bte.asnumpy(bte.xp_module.array(gmres_info[0][i].residuals)))+1)
            y0 = bte.asnumpy(bte.xp_module.array(gmres_info[0][i].residuals))

            x1 = np.arange(1, len(bte.asnumpy(bte.xp_module.array(gmres_info[1][i].residuals)))+1)
            y1 = bte.asnumpy(bte.xp_module.array(gmres_info[1][i].residuals))

            x2 = np.arange(1, len(bte.asnumpy(bte.xp_module.array(gmres_info[2][i].residuals)))+1)
            y2 = bte.asnumpy(bte.xp_module.array(gmres_info[2][i].residuals))

            x3 = np.arange(1, len(bte.asnumpy(bte.xp_module.array(gmres_info[3][i].residuals)))+1)
            y3 = bte.asnumpy(bte.xp_module.array(gmres_info[3][i].residuals))

            plt.semilogy(x0, y0, ':',  label=r"$CXV_{\theta} V_{r}$")
            plt.semilogy(x1, y1, '-.',  label=r"$XCV_{\theta} V_{r}$")
            plt.semilogy(x2, y2, '--', label=r"$XV_{\theta} V_{r}C$")
            plt.semilogy(x3, y3, '-', label=r"none")

            
            plt.xlabel(r"iteration")
            plt.ylabel(r"residual")
            plt.title(r"$\frac{\Delta t}{\tau}$ = %.2E"%(dt_info[i]))
            plt.legend()
            
        plt.tight_layout()
        plt.savefig("%s_gmres_benchmark.png"%(params.fname))
        sys.exit(0)
    elif args.benchmark==2:
        _dt  = bte.params.dt
        Emax = bte.params.tau * bte.params.qe * 4e4 / bte.params.me/ bte.bs_vth
        dtx  = np.min(bte.xp[1:] - bte.xp[0:-1]) / (bte.op_spec_sp._basis_p._kdomain[1] * bte.bs_vth  * bte.params.tau/bte.params.L)
        dtvr = np.min(bte.xp_vr[1:] - bte.xp_vr[0:-1]) / Emax
        dtvt = np.min(bte.xp_vt[0:-1]- bte.xp_vt[1:]) * np.min(bte.xp_vr) / Emax
        dtex = min(dtx, min(dtvr, dtvt))

        print("explicit dt = %.4E min(dtx = %.4E, dtvr = %.4E dtvt = %.4E)"%(dtex, dtx, dtvr, dtvt))
        print("dt base: %.2E"%(_dt/dtex))
        
        gmres_info   = list()
        #dt_info     = np.array([dtex, 10 * dtex, 100 * dtex, 500 * dtex, 1000 * dtex])
        # dt_info      = np.array([1e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3])
        # rtol_info    = np.array([1e-12])
        # # x, vr, vt, c
        # relax_params = [(1.0, 1.0, 1.0, 1.0), (0.5, 0.5, 0.5, 1), (0.25, 0.25, 0.25, 1), (0.125, 0.125, 0.125, 1), (0.0625, 0.0625, 0.06625, 1)]

        dt_info      = np.array([2e-3, 5e-3])
        rtol_info    = np.array([1e-12])
        relax_params = [(1/2**i, 1/2**i, 1/2**i, 1) for i in range(4)]
        

        ndt        = dt_info.shape[0]
        nrtol      = rtol_info.shape[0]
        nE         = 2
        pc_left    = False

        
        print("===========================================")
        print("=====FULLY IMPLICIT SCHEME BENCMARK========")
        print("===========================================")

        for i in range(ndt):
            rt = rtol_info[0]
            dt = dt_info[i]
            bte.init(dt)
            params.pc_type = bte_1d3v_solver.pc_type.NONE
            try:
                y0  = bte.step(Ext(0)   , v, None, 0.0, dt, params.verbose)
            except RuntimeError:
                print("fully-implicit solve did not converge")

            gmres_info.append((dt, rt, None, bte.gmres_info["gmres_callback"]))

            
            for idx, rp in enumerate(relax_params):
                
                print("\n\nrunning : dt = %.4E rtol = %.4E"%(dt, rt), " with ", rp)
                bte.init(dt,rp[-1], rp[0])

                params.pc_type = bte_1d3v_solver.pc_type.XVTVRC
                try:
                    y0  = bte.step(Ext(0)   , v, None, 0.0, dt, params.verbose, rp[1], rp[2], pc_left)
                except RuntimeError:
                    print("fully-implicit solve did not converge")

                gmres_info.append((dt, rt, rp, bte.gmres_info["gmres_callback"]))
                
        

        import matplotlib.ticker
        fig, axes = plt.subplots(1, ndt, figsize=(4 * ndt, 5), dpi=200)

        for ax in axes.flatten():
            ax.minorticks_on() # Explicitly turn on minor ticks
            ax.grid(visible=True, which='both', axis='both', linestyle='--', linewidth=0.25)
            y_major = matplotlib.ticker.LogLocator(base = 10.0, numticks = 10)
            ax.yaxis.set_major_locator(y_major)

            x_major = matplotlib.ticker.MaxNLocator(integer=True)
            ax.xaxis.set_major_locator(x_major)

            y_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
            ax.yaxis.set_minor_locator(y_minor)
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


        for i in range(ndt):
            plt.subplot(1, ndt, i+1)
            y0 = bte.asnumpy(bte.xp_module.array(gmres_info[i * (1+ len(relax_params)) + 0][-1].residuals))
            x0 = np.arange(1, len(y0)+1)
            plt.semilogy(x0, y0, '-', label=r"none")
            for idx, rp in enumerate(relax_params):
                yi = bte.asnumpy(bte.xp_module.array(gmres_info[i * (1+ len(relax_params)) + idx+1][-1].residuals))
                xi = np.arange(1, len(yi)+1)

                assert gmres_info[i * (1+ len(relax_params)) + idx+1][-2] == rp
                assert gmres_info[i * (1+ len(relax_params)) + idx+1][-3] == rtol_info[0]
                assert gmres_info[i * (1+ len(relax_params)) + idx+1][-4] == dt_info[i]

                #plt.semilogy(xi, yi, '--', label=r"($R_x=%.4f$, $R_{vr}=%.4f$, $R_{vt}=%.4f$, $R_c=%.4f$)"%(rp[0], rp[1], rp[2], rp[3]))
                plt.semilogy(xi, yi, '--', label=r"1/$2^{%d}, 1.0$"%(idx))

            plt.xlabel(r"iteration")
            plt.ylabel(r"residual")
            plt.title(r"$\frac{\Delta t}{\tau}$ = %.2E"%(dt_info[i]))
            plt.grid(visible=True, which='both', axis='both', linestyle='--', linewidth=0.25)
            plt.legend(fontsize=10)
            
        plt.tight_layout()
        plt.savefig("%s_gmres_benchmark_rp_pcl%d.png"%(params.fname, pc_left))
        sys.exit(0)
    elif args.benchmark==3:
        _dt  = bte.params.dt
        Emax = bte.params.tau * bte.params.qe * 4e4 / bte.params.me/ bte.bs_vth
        dtx  = np.min(bte.xp[1:] - bte.xp[0:-1]) / (bte.op_spec_sp._basis_p._kdomain[1] * bte.bs_vth  * bte.params.tau/bte.params.L)
        dtvr = np.min(bte.xp_vr[1:] - bte.xp_vr[0:-1]) / Emax
        dtvt = np.min(bte.xp_vt[0:-1]- bte.xp_vt[1:]) * np.min(bte.xp_vr) / Emax
        dtex = min(dtx, min(dtvr, dtvt))
        
        
        dt_info      = np.array([5e-4, 5e-5, 5e-6, 1e-6])
        rtol_info    = np.array([1e-12])
        T            = 1e-2
        E            = Ext(0.25) # max E-field
        
        ndt          = dt_info.shape[0]
        nrtol        = rtol_info.shape[0]
        nE           = 2

        dx           = np.min(bte.xp[1:] - bte.xp[0:-1])
        dvr          = np.min(bte.xp_vr[1:] - bte.xp_vr[0:-1])
        dvt          = np.min((-bte.xp_vt[1:] + bte.xp_vt[0:-1]))

        qoi_all      = xp.zeros((ndt, 2 + bte.num_collisions, bte.dof_x))

        for i in range(len(dt_info)):
            dt              = dt_info[i]    
            bte.params.dt   = dt
            bte.params.rtol = rtol_info[0]

            if bte.params.solver_type == bte_1d3v_solver.solver_type.FULLY_IMPLICIT:
                r_ax        = 1/(1 + (bte.bs_vth * bte.params.tau/bte.params.L) *(bte.op_spec_sp._basis_p._kdomain[1] * dt) / dx )
                r_vr        = r_ax #1/(1 + Emax * dt/dvr)
                r_vt        = r_ax #1/(1 + Emax * dt/dvt/bte.xp_vr[0])
            else:
                r_ax        = 1.0
                r_vr        = 1.0
                r_vt        = 1.0


            bte.init(dt,1, r_ax)
            bte.params.pc_type = bte_1d3v_solver.pc_type.XVTVRC
            
            steps       = int(np.ceil(T/dt))
            print(dt, rtol_info[0], steps)
            assert np.abs(1 - T/steps/dt) < 1e-12

            u           = xp.copy(v)
            
                        
            for ts_idx in range(steps+1):
                q              = bte.compute_qoi(u, ts_idx * dt, dt)
                qoi_all[i, 0] += (dt/T) * 0.5 * q["ne"]
                qoi_all[i, 1] += (dt/T) * 0.5 * q["Te"] * q["ne"]
                
                for m in range(bte.num_collisions):
                    qoi_all[i, 2 + m] += (dt/T) * 0.5 * q["rates"][:,m] * q["ne"]

                u       = bte.step(Ext(0.25 + ts_idx * dt), u, None, ts_idx * dt, dt, (ts_idx % 100 == 0))

                q              = bte.compute_qoi(u, (ts_idx +1) * dt, dt)
                
                qoi_all[i, 0] += (dt/T) * 0.5 * q["ne"]
                qoi_all[i, 1] += (dt/T) * 0.5 * q["Te"] * q["ne"]

                for m in range(bte.num_collisions):
                    qoi_all[i, 2 + m] += (dt/T) * 0.5 * q["rates"][:,m] * q["ne"]
        
        import matplotlib.ticker
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=300)
        for ax in axes.flatten():
            ax.minorticks_on() # Explicitly turn on minor ticks
            ax.grid(visible=True, which='both', axis='both', linestyle='--', linewidth=0.25)
            y_major = matplotlib.ticker.LogLocator(base = 10.0, numticks = 10)
            ax.yaxis.set_major_locator(y_major)

            x_major = matplotlib.ticker.MaxNLocator(integer=True)
            ax.xaxis.set_major_locator(x_major)

            y_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = np.arange(1.0, 10.0) * 0.1, numticks = 10)
            ax.yaxis.set_minor_locator(y_minor)
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

        qoi_rerrors = xp.zeros((qoi_all.shape[0], qoi_all.shape[1]))
        qx_q        = bte.grid_info["x_qw"]

        for i in range(ndt-1):
            for m in range(qoi_all.shape[1]):
                qoi_rerrors[i, m] = xp.sqrt(xp.dot(qx_q, (qoi_all[i, m] - qoi_all[-1, m])**2) / xp.dot(qx_q, qoi_all[-1, m]**2))
        
        qoi_all     = bte.asnumpy(qoi_all)
        qoi_rerrors = bte.asnumpy(qoi_rerrors)
        xx          = bte.asnumpy(bte.xp)
        
        print(qoi_rerrors.shape, qoi_rerrors)
        plt.subplot(1, 3, 1)
        for i in range(ndt):
            plt.semilogy(xx, qoi_all[i][0], label=r"dt=%.2E"%(dt_info[i]))
        
        plt.xlabel(r"$\hat{x}$")
        plt.ylabel(r"$n_e[m^-3]$")
        plt.legend()

        plt.subplot(1, 3, 2)
        for i in range(ndt):
            plt.semilogy(xx, qoi_all[i][1], label=r"dt=%.2E"%(dt_info[i]))
        
        plt.xlabel(r"$\hat{x}$")
        plt.ylabel(r"$n_e T_e$ [eV]")
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.loglog(dt_info, qoi_rerrors[:, 0], 'x--', label=r"$n_e$")
        plt.loglog(dt_info, qoi_rerrors[:, 1], 'o--', label=r"$n_eT_e$")
        plt.loglog(dt_info, qoi_rerrors[:, 2], '+--', label=r"$n_ek_{0}$")
        plt.loglog(dt_info, qoi_rerrors[:, 3], 's--', label=r"$n_ek_{1}$")

        plt.xlabel(r"$\frac{\Delta t}{\tau}$")
        plt.ylabel(r"$L_2(\Omega_{txv})$")
        plt.legend()

        plt.tight_layout()
        plt.savefig("%s_dt_conv.png"%(bte.params.fname))
        plt.close()
        sys.exit(0)

        # for i in range(ndt-1):
        #     
        
        # plt.xlabel(r"$\hat{x}$")
        # plt.ylabel(r"")

    
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
    for ts_idx in range(ts_idx_b, steps):
        tt   = ts_idx * params.dt
        
        if (ts_idx % cycle_freq == 0 and ts_idx > ts_idx_b):
          u1 = qoi(v,xp)
          a1 = xp.max(xp.abs(u1[:,0] -u0[:, 0]))
          a2 = xp.max(xp.abs((u1[:,0]-u0[:,0]) / xp.max(xp.abs(u0[:,0]))))
          u0 = u1
          
        #   print("ne : ||u(t+T) - u(t)|| = %.8E and ||u(t+T) - u(t)||/||u(t)|| = %.8E"% (a1, a2))
        #   if (a2 < tp_rtol):
        #       break

        if (ts_idx % io_freq == 0):
          print("time = %.2E step=%d/%d"%(tt, ts_idx, steps))
          bte.plot(Ext(tt), v, "%s_%04d.png"%(params.fname, ts_idx//io_freq), tt)
          
        if (ts_idx % cp_freq == 0):
           bte.store_checkpoint(v, tt, params.dt, "%s_cp_%04d"%(params.fname, ts_idx//io_freq))
           
        
        v  = bte.step(Ext(tt), v, None, tt, dt, (ts_idx % 100 == 0),r_vr, r_vt, pc_left = False)
        #v   = bte.step_adv_vx_sl(Ext(tt), v, tt, dt, verbose=0)
        #v = bte.step_bte_x(v, tt           , 0.5 * dt, verbose=1)
        #v = bte.step_bte_x(v, tt + 0.5 * dt, 0.5 * dt, verbose=1)
        #v = bte.step_bte_v(Ext(tt), v, None, tt, dt, verbose=0)
        tt += 0.5 * dt

        #tt+= dt

    

        
        
    
        


    

    

    
