import numpy as np
import matplotlib.pyplot as plt
import plot_utils
from itertools import cycle
plt.style.use('tableau-colorblind10')

plt.rcParams.update({
    #"text.usetex": True,
    "font.size": 16,
    #"ytick.major.size": 3,
    #"font.family": "Helvetica",
    "lines.linewidth":2.0
})

def plot_data1(data, fprefix, cycles, xloc, time_fac, plot_eedf=False, num_cols = 4, num_rows=1):
    ki_all = list() 
    np0    = 8e16
    for d_idx, d in enumerate(data):
        args  = d[0]
        Np    = (int)(args["Np"])
        Nr    = (int)(args["Nr"])
        Nvt   = (int)(args["Nvt"])
        l_max = (int)(args["l_max"])
        dt    = (float)(args["cfl"])
        ev_max= (float)(args["ev_max"])
        op    = plot_utils.op(Np)
        ki_d  = plot_utils.compute_rate_coefficients(args, d[3], d[2], collisions=["g2"])
        ki_all.append(ki_d)
        
        
    for idx, cycle_id in enumerate(cycles):
        print("plotting data cycle idx = %d"%(cycle_id))
        plt.figure(figsize=(24,14), dpi=100)
        for d_idx, d in enumerate(data):
            args  = d[0]
            Np    = (int)(args["Np"])
            Nr    = (int)(args["Nr"])
            Nvt   = (int)(args["Nvt"])
            l_max = (int)(args["l_max"])
            ev_max= (float)(args["ev_max"])
            dt    = (float)(args["cfl"])
            op    = plot_utils.op(Np)
            xp    = op.xp
            
            u      = d[1]
            v      = d[2]
            #bte_op = d[3]
            f_lm   = d[4]
            
            
            lbl = r"$dt=%.2E,N_p=%d,N_r=%d,N_\theta=%d,l=%d$"%(dt, Np, Nr + 1, Nvt, l_max)
            ne  = u[idx][:, 0]
            ni  = u[idx][:, 1]
            Te  = u[idx][:, 2]
            E   = np.ones_like(xp) * 1e4 * np.sin(2*np.pi * cycle_id * 1e-1) #-(op256.V0/op256.L) * np.dot(op256.Dp, op256.solve_poisson(ne1, ni1,0))
            ki  = ki_all[d_idx][0][idx]
            
            ne_h  = data[-1][1][idx][:, 0]
            ni_h  = data[-1][1][idx][:, 1]
            Te_h  = data[-1][1][idx][:, 2]
            E_h   = np.ones_like(xp) * 1e4 * np.sin(2*np.pi * cycle_id * 1e-1) #-(op256.V0/op256.L) * np.dot(op256.Dp, op256.solve_poisson(ne1, ni1,0))
            ki_h  = ki_all[-1][0][idx]
            
            plt.subplot(2,4, 1)
            plt.plot(xp, ne * np0, '-', label=r"" + lbl)
            
            plt.subplot(2, 4, 2)
            plt.plot(xp, E, label=lbl)
            
            plt.subplot(2, 4, 3)
            plt.plot(xp, Te, label=lbl)
            
            plt.subplot(2, 4, 4)
            plt.semilogy(xp, ki, label=lbl)
            
            plt.subplot(2, 4, 5)
            #plt.semilogy(xp, np.abs(ne-ne_h)/np.max(np.abs(ne_h)), '-', label=r"ne  " + lbl)
            plt.semilogy(xp, np.abs(1-ne/ne_h), '-', label=r"" + lbl)
            
            plt.subplot(2, 4, 6)
            #plt.semilogy(xp, np.abs(1-E/E_h), label=lbl)
            
            plt.subplot(2, 4, 7)
            #plt.semilogy(xp, np.abs(Te_h-Te)/np.max(np.abs(Te_h)), label=lbl)
            plt.semilogy(xp, np.abs(1-Te/Te_h), label=lbl)
            
            plt.subplot(2, 4, 8)
            plt.semilogy(xp, np.abs(1-ki/ki_h), label=lbl)
            
            
        plt.subplot(2, 4, 1)    
        plt.ylabel(r"density$(m^{-3})$")
        plt.grid()
        plt.legend(fontsize="10", loc ="lower left")
        
        plt.subplot(2, 4, 2)
        plt.ylabel(r"E (V/m)")
        plt.legend(fontsize="10", loc ="upper left")
        plt.grid()
        
        plt.subplot(2, 4, 3)
        plt.ylabel(r"temperature (eV)")
        plt.legend(fontsize="10", loc ="upper left")
        plt.grid()
        
        plt.subplot(2, 4, 4)
        plt.ylabel(r"$k_i$ ($m^3s^{-1}$)")
        plt.grid()
        plt.legend(fontsize="10", loc ="upper left")
        
        plt.subplot(2, 4, 5)    
        plt.ylabel(r"relative error ($n_e$)")
        plt.grid()
        plt.legend(fontsize="10", loc ="upper left")
        
        plt.subplot(2, 4, 6)
        plt.ylabel(r"relative error ($E$)")
        #plt.legend()
        #plt.legend(fontsize="10", loc ="upper left")
        plt.grid()
        
        plt.subplot(2, 4, 7)
        plt.ylabel(r"relative error ($T_e$)")
        plt.legend(fontsize="10", loc ="upper left")
        plt.grid()
        
        plt.subplot(2, 4, 8)
        plt.ylabel(r"relative error ($k_i$)")
        plt.grid()
        plt.legend(fontsize="10", loc ="upper left")
        
        
        plt.suptitle("cycle = %.4E"%(cycle_id * time_fac))
        plt.savefig("%s_%04d.png"%(fprefix, cycle_id))
        plt.close()
        
        if plot_eedf:
            for x_idx, x in enumerate(xloc):
                plt.figure(figsize=(num_cols * 8 + 0.5 * (num_cols-1), num_rows * 8 + 0.5 * (num_rows-1)), dpi=100)
                for d_idx, d in enumerate(data):
                    args     = d[0]
                    Np       = (int)(args["Np"])
                    Nr       = (int)(args["Nr"])
                    Nvt      = (int)(args["Nvt"])
                    l_max    = (int)(args["l_max"])
                    dt       = (float)(args["cfl"])
                    ev_max   = (float)(args["ev_max"])
                    ev_grid  = np.linspace(0, ev_max * 4, 1024)
                    plt_idx = 1
                    for lm_idx in range(l_max + 1):
                        plt.subplot(num_rows, num_cols, plt_idx)
                        plt.semilogy(ev_grid, np.abs(d[4][idx][x_idx][lm_idx]), label=r"x=%.1f, $N_r$=%d $l_{max}=%d$, $ev_{max}$=%.2E,dt=%.2E"%(x, Nr+1, l_max, ev_max, dt))

                        plt.title(r"$l = %d$"%(lm_idx))
                        plt.xlabel(r"energy (eV)")
                        plt.ylabel(r"$abs(f_{%d})$"%(lm_idx))
                        plt.grid(visible=True)
                        plt.legend(fontsize="10", loc ="upper right")
                        plt.xlim((1e-6, None))
                        plt_idx+=1
                    
                plt.suptitle("cycle = %.4E"%(cycle_id * time_fac))
                plt.savefig("%s_eedf_at_x%04d_%04d.png"%(fprefix, x_idx, cycle_id))
                plt.close()
                

def plot_data2(data, fprefix, cycles, xloc, time_fac, plot_eedf=False, num_cols = 4, num_rows=1):
    ki_all = list() 
    np0    = 8e16
    for d_idx, d in enumerate(data):
        args  = d[0]
        Np    = (int)(args["Np"])
        Nr    = (int)(args["Nr"])
        Nvt   = (int)(args["Nvt"])
        l_max = (int)(args["l_max"])
        dt    = (float)(args["cfl"])
        ev_max= (float)(args["ev_max"])
        op    = plot_utils.op(Np)
        ki_d  = plot_utils.compute_rate_coefficients(args, d[3], d[2], collisions=["g2"])
        ki_all.append(ki_d)
        
    for idx, cycle_id in enumerate(cycles):
        print("plotting data cycle idx = %d"%(cycle_id))
        plt.figure(figsize=(34,14), dpi=200)
        colors    = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        for d_idx, d in enumerate(data):
            args  = d[0]
            Np    = (int)(args["Np"])
            Nr    = (int)(args["Nr"])
            Nvt   = (int)(args["Nvt"])
            l_max = (int)(args["l_max"])
            dt    = (float)(args["cfl"])
            ev_max= (float)(args["ev_max"])
            op    = plot_utils.op(Np)
            xp    = op.xp
            
            u      = d[1]
            v      = d[2]
            #bte_op = d[3]
            f_lm   = d[4]
            
            
            clr    = next(colors)
            
            
            #lbl = r"$dt=%.2E,N_p=%d,N_r=%d,N_\theta=%d,l=%d$,ev_max=%.2E"%(dt, Np, Nr+1, Nvt, l_max, ev_max)
            lbl = r"$dt=%.2E,N_p=%d,N_r=%d,N_\theta=%d,l=%d$"%(dt, Np, Nr+1, Nvt, l_max)
            ne  = u[idx][:, 0]
            ni  = u[idx][:, 1]
            Te  = u[idx][:, 2]
            E   = -(op.V0/op.L) * np.dot(op.Dp, op.solve_poisson(ne, ni, cycle_id * time_fac))
            ki  = ki_all[d_idx][0][idx]
            
            ne_h  = data[-1][1][idx][:, 0]
            ni_h  = data[-1][1][idx][:, 1]
            Te_h  = data[-1][1][idx][:, 2]
            op_h  = plot_utils.op((int)(data[-1][0]["Np"]))
            E_h   = -(op_h.V0/op_h.L) * np.dot(op.Dp, op_h.solve_poisson(ne_h, ni_h, cycle_id * time_fac))
            ki_h  = ki_all[-1][0][idx]
            
            plt.subplot(2,4, 1)
            plt.plot(xp, ne * np0, '-', label=r"" + lbl, color=clr)
            #plt.plot(xp, ni * np0, '--', label=r"" + lbl)
            lbl=""
            
            plt.subplot(2, 4, 2)
            plt.plot(xp, E, label=lbl, color=clr)
            
            plt.subplot(2, 4, 3)
            plt.plot(xp, Te, label=lbl, color=clr)
            
            plt.subplot(2, 4, 4)
            plt.semilogy(xp, ki, label=lbl, color=clr)
            
            plt.subplot(2, 4, 5)
            plt.semilogy(xp, np.abs(ne_h-ne)/np.max(np.abs(ne_h)), '-', label=r"" + lbl, color=clr)
            #plt.semilogy(xp, np.abs(1-ni/ni_h), '--', label=r"" + lbl)
            
            plt.subplot(2, 4, 6)
            #plt.semilogy(xp, np.abs(1-E/E_h), label=lbl)
            plt.semilogy(xp, np.abs(E_h - E)/np.max(np.abs(E_h)), '-', label=r"" + lbl, color=clr)
            
            plt.subplot(2, 4, 7)
            #plt.semilogy(xp, np.abs(1-Te/Te_h), label=lbl)
            plt.semilogy(xp, np.abs(Te_h - Te)/np.max(np.abs(Te_h)), '-', label=r"" + lbl, color=clr)
            
            
            plt.subplot(2, 4, 8)
            #plt.semilogy(xp, np.abs(1-ki/ki_h), label=lbl)
            plt.semilogy(xp, np.abs(ki_h - ki)/np.max(np.abs(ki_h)), '-', label=r"" + lbl, color=clr)
            
            
        plt.subplot(2, 4, 1)    
        plt.ylabel(r"density$(m^{-3})$")
        plt.grid(visible=True)
        plt.legend(fontsize="14", loc ="lower left")
        
        plt.subplot(2, 4, 2)
        plt.ylabel(r"E (V/m)")
        plt.legend(fontsize="8", loc ="upper left")
        plt.grid(visible=True)
        
        plt.subplot(2, 4, 3)
        plt.ylabel(r"temperature (eV)")
        plt.legend(fontsize="8", loc ="upper left")
        plt.grid(visible=True)
        
        plt.subplot(2, 4, 4)
        plt.ylabel(r"$k_i$ ($m^3s^{-1}$)")
        plt.grid(visible=True)
        plt.legend(fontsize="8", loc ="upper left")
        
        plt.subplot(2, 4, 5)    
        plt.ylabel(r"relative error ($n_e$)")
        plt.grid(visible=True)
        plt.legend(fontsize="8", loc ="upper left")
        
        plt.subplot(2, 4, 6)
        plt.ylabel(r"relative error ($E$)")
        #plt.legend()
        #plt.legend(fontsize="8", loc ="upper left")
        plt.grid(visible=True)
        
        plt.subplot(2, 4, 7)
        plt.ylabel(r"relative error ($T_e$)")
        plt.legend(fontsize="8", loc ="upper left")
        plt.grid(visible=True)
        
        plt.subplot(2, 4, 8)
        plt.ylabel(r"relative error ($k_i$)")
        plt.grid(visible=True)
        plt.legend(fontsize="8", loc ="upper left")
        
        
        plt.suptitle("cycle = %.4E"%(cycle_id * time_fac))
        plt.savefig("%s_%04d.png"%(fprefix, idx))
        plt.close()
        
        if plot_eedf:
            plt.figure(figsize=(num_cols * 8 + 0.5 * (num_cols-1), num_rows * 8 + 0.5 * (num_rows-1)), dpi=200)
            colors    = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
            for x_idx, x in enumerate(xloc):
                clr      = next(colors)
                for d_idx, d in enumerate(data):
                    args     = d[0]
                    Np       = (int)(args["Np"])
                    Nr       = (int)(args["Nr"])
                    Nvt      = (int)(args["Nvt"])
                    l_max    = (int)(args["l_max"])
                    dt       = (float)(args["cfl"])
                    ev_max   = (float)(args["ev_max"])
                    ev_grid  = np.linspace(0, 4 * ev_max, 1024)
                    
                    plt_idx = 1
                    for lm_idx in range(l_max + 1):
                        plt.subplot(num_rows, num_cols, plt_idx)
                        #plt.semilogy(ev_grid, np.abs(d[4][idx][x_idx][lm_idx]), label=r"dt=%.2E, x=%.2f, $N_r$=%d, $l_{max}$ = %d"%(dt, x, Nr, l_max))
                        if (d_idx==0):
                            plt.semilogy(ev_grid, np.abs(d[4][idx][x_idx][lm_idx]), '--', color=clr)
                        else:
                            plt.semilogy(ev_grid, np.abs(d[4][idx][x_idx][lm_idx]), label=r"l_{max}=%d, x=%.2f"%(l_max, x), color=clr)

                        plt.title(r"$l = %d$"%(lm_idx))
                        plt.xlabel(r"energy (eV)")
                        plt.ylabel(r"$abs(f_{%d})$"%(lm_idx))
                        plt.grid(visible=True)
                        plt.xlim((0, 30))
                        plt.legend()
                        plt_idx+=1

            plt.suptitle("cycle = %.4E"%(cycle_id * time_fac))
            plt.savefig("%s_eedf_at_%04d.png"%(fprefix, idx))
            plt.close()

def glow_plot(data, fprefix, cycles, xloc, num_cols = 4, num_rows=2):
    plt.figure(figsize=(num_cols * 5 + 0.5 * (num_cols-1), num_rows * 5 + 0.5 * (num_rows-1)), dpi=300)
    args  = data[0]
    Np    = (int)(args["Np"])
    Nr    = (int)(args["Nr"])
    Nvt   = (int)(args["Nvt"])
    l_max = (int)(args["l_max"])
    dt    = (float)(args["cfl"])
    ev_max= (float)(args["ev_max"])
    ev_grid  = np.linspace(0 + 1e-6, ev_max * 25 -1e-6, 1024)
    op    = plot_utils.op(Np)
    xp    = op.xp
    
    ki    = plot_utils.compute_rate_coefficients(args, data[3], data[2], collisions=["g0", "g2"])
    
    u      = data[1]
    v      = data[2]
    #bte_op = data[3]
    f_lm   = data[4]
    
    for idx, cycle_id in enumerate(cycles):
        
        ne  = u[idx][:, 0]
        ni  = u[idx][:, 1]
        Te  = u[idx][:, 2]
        E   = -(op.V0/op.L) * np.dot(op.Dp, op.solve_poisson(ne, ni, cycle_id))
        
        ki0 = ki[0][idx]
        ki1 = ki[1][idx]
        
        lbl = r"T=%d cycles"%(cycle_id)
        
        plt.subplot(num_rows, num_cols, 1)
        plt.plot(xp, ne * op.np0, label=lbl)
        plt.plot(xp, ni * op.np0, '--', label=lbl)
        plt.grid(visible=True)
        plt.xlabel(r"x")
        plt.ylabel(r"density $(m^{-3})$")
        plt.legend()
        
        plt.subplot(num_rows, num_cols, 2)
        plt.plot(xp, Te, label=lbl)
        plt.grid(visible=True)
        plt.xlabel(r"x")
        plt.ylabel(r"electron temperature $(eV)$")
        plt.legend()
        
        plt.subplot(num_rows, num_cols, 3)
        plt.plot(xp, E, label=lbl)
        plt.grid(visible=True)
        plt.xlabel(r"x")
        plt.ylabel(r"E $(V/m)$")
        plt.legend()
        
        plt.subplot(num_rows, num_cols, 4)
        plt.semilogy(xp, ki0, label=lbl + " elastic ")
        plt.semilogy(xp, ki1, label=lbl + " ionization ")
        plt.grid(visible=True)
        plt.xlabel(r"x")
        plt.ylabel(r"reaction rate $m^3s^{-1}$")
        plt.legend()
        
        for x_idx, x in enumerate(xloc):
            plt_idx = 5
            for lm_idx in range(l_max+1):
                plt.subplot(num_rows, num_cols, plt_idx)
                plt.semilogy(ev_grid, np.abs(f_lm[idx][x_idx][lm_idx]), label=lbl + r" x=%.2f"%(x))
                plt.title(r"$l = %d$"%(lm_idx))
                plt.xlabel(r"energy (eV)")
                plt.ylabel(r"$abs(f_{%d})$"%(lm_idx))
                plt.xlim(0, 300)
                plt.grid(visible=True)
                plt.legend()
                plt_idx+=1
                
                if plt_idx > num_cols * num_rows:
                    break
        
    #plt.show()
    #plt.close()
    plt.savefig("%s_glow_plot.png"%(fprefix))
        
# op        = plot_utils.op(200)
# idx       = (np.abs(op.xp - 0)).argmin()
# eedf_idx  = [0,idx-1, -1]
# xloc      = op.xp[eedf_idx]
# plot_utils.make_dir("1D_bte_10K_Vm_Nr")
# batch_sz  = 2 
# for i in range(0, 11, batch_sz):
#     idx_range = range(i, min(i+batch_sz, 11), 1)
#     d1        = plot_utils.load_data_bte("./../1d2v_bte_convergence/e1" , idx_range, eedf_idx)
#     d2        = plot_utils.load_data_bte("./../1d2v_bte_convergence/e2" , idx_range, eedf_idx)
#     d3        = plot_utils.load_data_bte("./../1d2v_bte_convergence/e3" , idx_range, eedf_idx)
#     d4        = plot_utils.load_data_bte("./../1d2v_bte_convergence/e4" , idx_range, eedf_idx)
#     d5        = plot_utils.load_data_bte("./../1d2v_bte_convergence/e5" , idx_range, eedf_idx)
#     data      = [d1, d2, d3, d4, d5]
#     plot_data1(data, "./1D_bte_10K_Vm_Nr/1d_glow", list(idx_range), xloc, time_fac=0.1, plot_eedf=True, num_cols=3, num_rows=1)
    
# op        = plot_utils.op(200)
# idx       = (np.abs(op.xp - 0)).argmin()
# eedf_idx  = [0,idx-1, -1]
# xloc      = op.xp[eedf_idx]
# plot_utils.make_dir("1D_bte_10K_Vm_lmax")
# batch_sz  = 4 
# for i in range(0, 11, batch_sz):
#     idx_range = range(i, min(i+batch_sz, 11), 1)
#     d3        = plot_utils.load_data_bte("./../1d2v_bte_convergence/e3" , idx_range, eedf_idx)
#     d6        = plot_utils.load_data_bte("./../1d2v_bte_convergence/e6" , idx_range, eedf_idx)
#     d7        = plot_utils.load_data_bte("./../1d2v_bte_convergence/e7" , idx_range, eedf_idx)
#     d8        = plot_utils.load_data_bte("./../1d2v_bte_convergence/e8" , idx_range, eedf_idx)
#     data      = [d3, d6, d7, d8]
#     plot_data1(data, "./1D_bte_10K_Vm_lmax/1d_glow", list(idx_range), xloc, time_fac=0.1, plot_eedf=True, num_cols=3, num_rows=3)
    
    
# op        = plot_utils.op(200)
# idx       = (np.abs(op.xp - 0)).argmin()
# eedf_idx  = [0,idx-1, -1]
# xloc      = op.xp[eedf_idx]
# plot_utils.make_dir("1D_bte_10K_Vm_time")
# batch_sz  = 4 
# for i in range(0, 11, batch_sz):
#     idx_range = range(i, min(i+batch_sz, 11), 1)
#     d1        = plot_utils.load_data_bte("./../1d2v_bte_convergence/e7" , idx_range, eedf_idx)
#     d2        = plot_utils.load_data_bte("./../1d2v_bte_convergence/f1" , idx_range, eedf_idx)
#     d3        = plot_utils.load_data_bte("./../1d2v_bte_convergence/f2" , idx_range, eedf_idx)
#     data      = [d1, d2, d3]
#     plot_data1(data, "./1D_bte_10K_Vm_time/1d_glow", list(idx_range), xloc, time_fac=0.1, plot_eedf=True, num_cols=3, num_rows=3)
    

# op        = plot_utils.op(200)
# idx       = (np.abs(op.xp - 0)).argmin()
# eedf_idx  = [0,idx-1, -1]
# xloc      = op.xp[eedf_idx]
# idx_range = range(0, 11, 1)
# d1        = plot_utils.load_data_bte("./../1dglow/a1" , idx_range, eedf_idx)
# d2        = plot_utils.load_data_bte("./../1dglow/a2" , idx_range, eedf_idx)
# d3        = plot_utils.load_data_bte("./../1dglow/a3" , idx_range, eedf_idx)
# #d4        = plot_utils.load_data_bte("./../junk/a4" , idx_range, eedf_idx)
# data      = [d1, d2, d3]
# plot_utils.make_dir("1D_glow_bte_Nr")
# plot_data2(data, "./1D_glow_bte_Nr/1d_glow", list(idx_range), xloc, time_fac=1, plot_eedf=True, num_cols=3, num_rows=1)
# # batch_sz  = 4 
# # for i in range(0, 11, batch_sz):

# op        = plot_utils.op(200)
# idx       = (np.abs(op.xp - 0)).argmin()
# eedf_idx  = [0,idx-1, -1]
# xloc      = op.xp[eedf_idx]
# idx_range = range(0, 11, 1)
# d1        = plot_utils.load_data_bte("./../1dglow/a2" , idx_range, eedf_idx)
# d2        = plot_utils.load_data_bte("./../1dglow/a5" , idx_range, eedf_idx)
# d3        = plot_utils.load_data_bte("./../1dglow/a6" , idx_range, eedf_idx)
# #d4        = plot_utils.load_data_bte("./../junk/a4" , idx_range, eedf_idx)
# data      = [d1, d2, d3]
# plot_utils.make_dir("1D_glow_bte_lmax")
# plot_data2(data, "./1D_glow_bte_lmax/1d_glow", list(idx_range), xloc, time_fac=1, plot_eedf=True, num_cols=3, num_rows=3)


# op        = plot_utils.op(200)
# idx       = (np.abs(op.xp - 0)).argmin()
# eedf_idx  = [0,idx-1, -1]
# xloc      = op.xp[eedf_idx]
# idx_range = range(0, 11, 1)
# d1        = plot_utils.load_data_bte("./../1dglow/a6" , idx_range, eedf_idx)
# d2        = plot_utils.load_data_bte("./../1dglow/a7" , idx_range, eedf_idx)
# d3        = plot_utils.load_data_bte("./../1dglow/a8" , idx_range, eedf_idx)
# data      = [d1, d2, d3]
# plot_utils.make_dir("1D_glow_bte_dt")
# plot_data2(data, "./1D_glow_bte_dt/1d_glow", list(idx_range), xloc, time_fac=1, plot_eedf=True, num_cols=3, num_rows=3)

# op        = plot_utils.op(200)
# idx       = (np.abs(op.xp - 0)).argmin()
# eedf_idx  = [0,idx-1, -1]
# xloc      = op.xp[eedf_idx]
# idx_range = range(0, 11, 1)
# d1        = plot_utils.load_data_bte("./../1d2v_bte_convergence/m1" , idx_range, eedf_idx)
# d2        = plot_utils.load_data_bte("./../1d2v_bte_convergence/m2" , idx_range, eedf_idx)
# data      = [d1, d2]
# plot_utils.make_dir("m")
# plot_data1(data, "./m/1d_glow", list(idx_range), xloc, time_fac=1, plot_eedf=True, num_cols=3, num_rows=3)


# op        = plot_utils.op(200)
# idx       = (np.abs(op.xp - 0)).argmin()
# eedf_idx  = [0,idx-1, -1]
# xloc      = op.xp[eedf_idx]
# idx_range = range(0, 91, 10)
# d1        = plot_utils.load_data_bte("./../1dglow/r3" , idx_range, eedf_idx)
# d2        = plot_utils.load_data_bte("./../1dglow/r4" , idx_range, eedf_idx)
# data      = [d1, d2]
# plot_utils.make_dir("1dglow_lmax")
# plot_data1(data, "./1dglow_lmax/1d_glow", list(idx_range), xloc, time_fac=0.1, plot_eedf=True, num_cols=3, num_rows=3)


# op          = plot_utils.op(200)
# idx         = (np.abs(op.xp - 0)).argmin()
# #eedf_idx   = [0,idx-1, -1]
# x_probe_loc = np.linspace(0.1,  0.9, 10)
# eedf_idx    = [(np.abs(op.xp - x_probe_loc[i])).argmin() for i in range(len(x_probe_loc))]

# xloc        = op.xp[eedf_idx]
# d1          = plot_utils.load_data_bte("./../1dglow/r3" , range(401, 415, 1), eedf_idx, read_cycle_avg=True)
# d2          = plot_utils.load_data_bte("./../1dglow/r4" , range(1  , 15 , 1), eedf_idx, read_cycle_avg=True)
# data        = [d1, d2]

# plot_utils.make_dir("dt_conv")
# plot_data2(data, "./dt_conv/1d_glow", range(401, 415, 1) , xloc, time_fac = 1.0, plot_eedf=True, num_cols=3, num_rows=1)

op          = plot_utils.op(200)
idx         = (np.abs(op.xp - 0)).argmin()
#eedf_idx   = [0,idx-1, -1]
x_probe_loc = np.linspace(0.1,  0.9, 5)
eedf_idx    = [(np.abs(op.xp - x_probe_loc[i])).argmin() for i in range(len(x_probe_loc))]

xloc        = op.xp[eedf_idx]
d1          = plot_utils.load_data_bte("./../1dglow/r3" , range(1  , 21, 1), eedf_idx, read_cycle_avg=True)
d2          = plot_utils.load_data_bte("./../1dglow/r5" , range(1  , 21 , 1), eedf_idx, read_cycle_avg=True)
data        = [d1, d2]

plot_utils.make_dir("lmax_conv")
plot_data2(data, "./lmax_conv/1d_glow", range(1, 21, 1) , xloc, time_fac = 1.0, plot_eedf=True, num_cols=3, num_rows=2)