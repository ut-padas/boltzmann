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
    Ext   = lambda t : xp.ones(params.Np) * 1e4 #* xp.sin(2 * xp.pi * t)
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
        vsh                 = bte.ords_to_sph(v)
        qoi                 = xp.zeros((qoi_idx.LAST, params.Np))
        qoi[qoi_idx.NE_IDX] = bte.op_mass @ vsh
        qoi[qoi_idx.TE_IDX] = (bte.op_temp @ vsh) / qoi[qoi_idx.NE_IDX]

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
           
          
        #v  = bte.step(Ext(tt), v, None, tt, dt, params.verbose)
        v = bte.step_bte_x(v, tt           , 0.5 * dt, verbose=1)
        v = bte.step_bte_x(v, tt + 0.5 * dt, 0.5 * dt, verbose=1)
        #v = bte.step_bte_v(Ext(tt), v, None, tt, dt, verbose=0)

        tt+= dt

    

        
        
    
        


    

    

    
