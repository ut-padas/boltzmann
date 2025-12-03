import numpy as np
import argparse
import bte_1d3v_solver
import enum

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
    
    bte.init(params.dt)

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
    elif (args.ef_mode == 3):
        Ext   = lambda t : xp.ones(params.Np) * 1e4
    elif (args.ef_mode == 4):
        Ext   = lambda t : xp.ones(params.Np) * 0.0
    elif (args.ef_mode == 5):
        Ext   = lambda t : xp.ones(params.Np) * 1e3

    else:
        raise NotImplementedError
    
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
            print("q", bte.op_mass.shape, v.shape)
            qoi[qoi_idx.NE_IDX] = bte.op_mass  @ v
            qoi[qoi_idx.TE_IDX] = (bte.op_temp @ v) / qoi[qoi_idx.NE_IDX]

        return qoi
    
    u0   = qoi(v, xp)
    u1   = xp.copy(u0)

    tp_rtol = 1e-4
      
    for ts_idx in range(ts_idx_b, steps):
        tt   = ts_idx * params.dt
        
        if (ts_idx % cycle_freq == 0 and ts_idx > ts_idx_b):
          u1 = qoi(v,xp)
          a1 = xp.max(xp.abs(u1[:,0] -u0[:, 0]))
          a2 = xp.max(xp.abs((u1[:,0]-u0[:,0]) / xp.max(xp.abs(u0[:,0]))))
          u0 = u1
          
          print("ne : ||u(t+T) - u(t)|| = %.8E and ||u(t+T) - u(t)||/||u(t)|| = %.8E"% (a1, a2))
          if (a2 < tp_rtol):
              break

        if (ts_idx % io_freq == 0):
          print("time = %.2E step=%d/%d"%(tt, ts_idx, steps))
          bte.plot(Ext(tt), v, "%s_%04d.png"%(params.fname, ts_idx//io_freq), tt)
          
        if (ts_idx % cp_freq == 0):
           bte.store_checkpoint(v, tt, params.dt, "%s_cp_%04d"%(params.fname, ts_idx//io_freq))
           
          
        v  = bte.step(Ext(tt), v, None, tt, dt, params.verbose)
        #v = bte.step_bte_x(v, tt           , 0.5 * dt, verbose=1)
        #v = bte.step_bte_x(v, tt + 0.5 * dt, 0.5 * dt, verbose=1)
        #v = bte.step_bte_v(Ext(tt), v, None, tt, dt, verbose=0)
        tt += 0.5 * dt

        #tt+= dt

    

        
        
    
        


    

    

    
