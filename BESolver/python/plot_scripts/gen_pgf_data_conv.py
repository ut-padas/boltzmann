import numpy as np
import matplotlib.pyplot as plt
import plot_utils
from itertools import cycle
import scipy.interpolate
import sys


# idx_range   = range(0, 11, 2)
# d1          = plot_utils.load_data_bte("./../1dglow_hybrid_Nx400/Ewt_10K_Nx100_Nr127_Nvt16_l2_dt_5e-4"  , idx_range , None, read_cycle_avg=True)
# d2          = plot_utils.load_data_bte("./../1dglow_hybrid_Nx400/Ewt_10K_Nx200_Nr255_Nvt32_l4_dt_2e-4"  , idx_range , None, read_cycle_avg=True)
# d3          = plot_utils.load_data_bte("./../1dglow_hybrid_Nx400/Ewt_10K_Nx200_Nr511_Nvt64_l8_dt_1e-4"  , idx_range , None, read_cycle_avg=True)

# data        = [d1, d2, d3]
# lbl         = [r"$N_r$=128,$N_\theta$=16,$l_{max}=2,dt=5e-4$", r"$N_r$=256,$N_\theta$=32,$l_{max}=4,dt=2e-4$", r"$N_r$=512,$N_\theta$=64,$l_{max}=8,dt=1e-4$"]

# fnames      = ["../1dglow_hybrid_Nx400/Ewt_10K_Nx100_Nr127_l2_dt_5e-4", 
#                "../1dglow_hybrid_Nx400/Ewt_10K_Nx200_Nr255_l4_dt_2e-4", 
#                "../1dglow_hybrid_Nx400/Ewt_10K_Nx200_Nr511_l8_dt_1e-4"]
# #xx_grid    = np.linspace(-1, 1, 100)
# xx_grid     = plot_utils.op(200).xp


# idx_range   = range(0, 11, 10)
# d1          = plot_utils.load_data_bte("./../1dglow_hybrid_Nx400/Ewt_10K_Nx100_Nr127_Nvt8_l1_dt_2e-4"  , idx_range , None, read_cycle_avg=False)
# d2          = plot_utils.load_data_bte("./../1dglow_hybrid_Nx400/Ewt_10K_Nx200_Nr256_Nvt16_l2_dt_1e-4" , idx_range , None, read_cycle_avg=False)
# d3          = plot_utils.load_data_bte("./../1dglow_hybrid_Nx400/Ewt_10K_Nx400_Nr512_Nvt32_l4_dt_5e-5" , idx_range , None, read_cycle_avg=False)

# data        = [d1, d2, d3]
# lbl         = [r"$N_x$=%d $N_r$=%d,$N_\theta$=%d,$l_{max}=%d,dt=%.2E$"%(int(data[i][0]["Np"]), int(data[i][0]["Nr"]), int(data[i][0]["Nvt"]), int(data[i][0]["l_max"]), float(data[i][0]["cfl"])) for i in range(len(data))]

# fnames      = ["../1dglow_hybrid_Nx400/Ewt_10K_Nx100_Nr128_Nvt8_l1_dt_2e-4", 
#                "../1dglow_hybrid_Nx400/Ewt_10K_Nx200_Nr256_Nvt16_l2_dt_1e-4", 
#                "../1dglow_hybrid_Nx400/Ewt_10K_Nx400_Nr512_Nvt32_l4_dt_5e-5"]

# xx_grid    = np.linspace(-1, 1, 1000)
# xx_grid     = plot_utils.op(400).xp

# idx_range   = range(0, 11, 2)
# d1          = plot_utils.load_data_bte("./../1dglow_hybrid_Nx400/Ewt_10K_Nx100_Nr64_Nvt16_l2_dt_5e-4"     , idx_range , None, read_cycle_avg=False)
# d2          = plot_utils.load_data_bte("./../1dglow_hybrid_Nx400/Ewt_10K_Nx200_Nr128_Nvt32_l4_dt_2.5e-4"  , idx_range , None, read_cycle_avg=False)
# d3          = plot_utils.load_data_bte("./../1dglow_hybrid_Nx400/Ewt_10K_Nx400_Nr256_Nvt64_l8_dt_1.25e-4" , idx_range , None, read_cycle_avg=False)

# data        = [d1, d2, d3]
# lbl         = [r"$N_x$=%d $N_r$=%d,$N_\theta$=%d,$l_{max}=%d,dt=%.2E$"%(int(data[i][0]["Np"]), int(data[i][0]["Nr"]), int(data[i][0]["Nvt"]), int(data[i][0]["l_max"]), float(data[i][0]["cfl"])) for i in range(len(data))]

# fnames      = ["../1dglow_hybrid_Nx400/Ewt_10K_Nx100_Nr64_Nvt16_l2_dt_5e-4", 
#                "../1dglow_hybrid_Nx400/Ewt_10K_Nx200_Nr128_Nvt32_l4_dt_2.5e-4", 
#                "../1dglow_hybrid_Nx400/Ewt_10K_Nx400_Nr256_Nvt64_l8_dt_1.25e-4"]

# #xx_grid    = np.linspace(-1, 1, 1000)
# xx_grid     = plot_utils.op(400).xp

# idx_range   = range(0, 8, 7)
# #d1          = plot_utils.load_data_bte("./../1dglow_hybrid_Nx400/1Torr300K_100V_Ar_3sp2r_l2_100x64x16_dt5e-5"     , idx_range , None, read_cycle_avg=True)
# d2          = plot_utils.load_data_bte("./../1dglow_hybrid_Nx400/1Torr300K_100V_Ar_3sp2r_l4_200x128x32_dt2.5e-5"  , idx_range , None, read_cycle_avg=True)
# d3          = plot_utils.load_data_bte("./../1dglow_hybrid_Nx400/1Torr300K_100V_Ar_3sp2r_l8_400x256x64_dt1.25e-5" , idx_range , None, read_cycle_avg=True)

# data        = [d2, d3]
# lbl         = [r"$N_x$=%d $N_r$=%d,$N_\theta$=%d,$l_{max}=%d,dt=%.2E$"%(int(data[i][0]["Np"]), int(data[i][0]["Nr"]), int(data[i][0]["Nvt"]), int(data[i][0]["l_max"]), float(data[i][0]["cfl"])) for i in range(len(data))]

# fnames      = [#"../1dglow_hybrid_Nx400/1Torr300K_100V_Ar_3sp2r_l2_100x64x16_dt5e-5", 
#                "../1dglow_hybrid_Nx400/1Torr300K_100V_Ar_3sp2r_l4_200x128x32_dt2.5e-5", 
#                "../1dglow_hybrid_Nx400/1Torr300K_100V_Ar_3sp2r_l8_400x256x64_dt1.25e-5"]

# xx_grid    = np.linspace(-1, 1, 100)
# xx_grid     = plot_utils.op(400).xp

idx_range   = range(0, 11, 10)
d1          = plot_utils.load_data_bte("./../1dglow_hybrid_Nx400/1Torr300K_100V_Ar_3sp2r_l2_400x256x16"     , idx_range , None, read_cycle_avg=True)
d2          = plot_utils.load_data_bte("./../1dglow_hybrid_Nx400/1Torr300K_100V_Ar_3sp2r_l4_400x256x16"     , idx_range , None, read_cycle_avg=True)
d3          = plot_utils.load_data_bte("./../1dglow_hybrid_Nx400/1Torr300K_100V_Ar_3sp2r_l8_400x256x16"     , idx_range , None, read_cycle_avg=True)
d4          = plot_utils.load_data_bte("./../1dglow_hybrid_Nx400/1Torr300K_100V_Ar_3sp2r_l8_400x256x16_dt2.5e-5"     , idx_range , None, read_cycle_avg=True)
#d5          = plot_utils.load_data_bte("./../1dglow_hybrid_Nx400/1Torr300K_100V_Ar_3sp2r_l8_400x256x16_dt1.25e-5"    , idx_range , None, read_cycle_avg=True)
d6          = plot_utils.load_data_bte("./../1dglow_hybrid_Nx400/1Torr300K_100V_Ar_3sp2r_l8_400x256x32_dt2.5e-5"     , idx_range , None, read_cycle_avg=True)

data        = [d1, d2, d3, d4, d6]
lbl         = [r"$N_x$=%d $N_r$=%d,$N_\theta$=%d,$l_{max}=%d,dt=%.2E$"%(int(data[i][0]["Np"]), int(data[i][0]["Nr"]), int(data[i][0]["Nvt"]), int(data[i][0]["l_max"]), float(data[i][0]["cfl"])) for i in range(len(data))]

fnames      = ["/home/mfernando/research/papers/boltzmann1d-paper/dat/1Torr_300K_100V_conv/1Torr300K_100V_Ar_3sp2r_l2_400x256x16", 
               "/home/mfernando/research/papers/boltzmann1d-paper/dat/1Torr_300K_100V_conv/1Torr300K_100V_Ar_3sp2r_l4_400x256x16", 
               "/home/mfernando/research/papers/boltzmann1d-paper/dat/1Torr_300K_100V_conv/1Torr300K_100V_Ar_3sp2r_l8_400x256x16",
               "/home/mfernando/research/papers/boltzmann1d-paper/dat/1Torr_300K_100V_conv/1Torr300K_100V_Ar_3sp2r_l8_400x256x16_dt2.5e-5",
               #"/home/mfernando/research/papers/boltzmann1d-paper/dat/1Torr_300K_100V_conv/1Torr300K_100V_Ar_3sp2r_l8_400x256x16_dt1.25e-5",
               "/home/mfernando/research/papers/boltzmann1d-paper/dat/1Torr_300K_100V_conv/1Torr300K_100V_Ar_3sp2r_l8_400x256x32_dt2.5e-5"]

#xx_grid    = np.linspace(-1, 1, 100)
xx_grid     = plot_utils.op(400).xp



macro_xx    = list()
for d in data:
    u   = d[1]
    Np  = u.shape[1]
    
    op  = plot_utils.op(Np)
    v0  = np.polynomial.chebyshev.chebvander(xx_grid, Np-1)
    p0  = np.dot(v0, op.V0pinv)
    
    # op  = plot_utils.op(Np)
    # p0  = op.interp_op_galerkin(xx_grid)
    
    #macro_xx.append(np.einsum("ik,akc->aic", p0, u))
    #print(u.shape)
    a = np.array([scipy.interpolate.interp1d(op.xp, u[i, : , j], kind='linear')(xx_grid) for i in range(u.shape[0]) for j in range(u.shape[2])]).reshape((u.shape[0], u.shape[2], -1))
    a = np.swapaxes(a, 1, 2)
    macro_xx.append(a)

for idx, cycle_id in enumerate(idx_range):
    plt.figure(figsize=(10, 4), dpi=200)
    for i, u in enumerate(macro_xx):
        ne = u[idx, : , 0]
        ni = u[idx, : , 1]
        Te = u[idx, : , 2]
        
        ne_h = macro_xx[-1][idx, : , 0]
        ni_h = macro_xx[-1][idx, : , 1]
        Te_h = macro_xx[-1][idx, : , 2]
    
        plt.subplot(1, 2, 1)
        plt.plot(xx_grid, ne, label=lbl[i])
        plt.grid(visible=True)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(xx_grid, Te, label=lbl[i])
        plt.grid(visible=True)
        plt.legend()
        
        np.savetxt("%s_%02d.csv"%(fnames[i], cycle_id), np.array([xx_grid, ne, Te]).T, header="x\tne\tTe", delimiter="\t", comments="")
        np.savetxt("%s_rel_error_%02d.csv"%(fnames[i], cycle_id), np.array([xx_grid, np.abs(ne-ne_h)/np.max(np.abs(ne_h)), np.abs(Te-Te_h)/np.max(np.abs(Te_h))]).T, header="x\tne\tTe", delimiter="\t", comments="")
    
    
    plt.show()
    plt.close()

#plt.figure(figsize=(10, 4), dpi=200)
#rel_error_ne = list()
#rel_error_Te = list()

macro_avg_xt = list()
for d in data:
    u   = d[1]
    Np  = u.shape[1]
    
    Nph = data[-1][1].shape[1]
    
    op     = plot_utils.op(Nph)
    qx, qw = np.polynomial.legendre.leggauss(Nph)
    v0     = np.polynomial.chebyshev.chebvander(qx, Nph-1).T
    Q      = (v0 @ qw).T @ op.V0pinv

    
    p0     = plot_utils.op(Np).interp_op_galerkin(xx_grid)
    u      = np.einsum("ik,akc->aic", p0, u)
    
    #macro_avg_xt.append(0.5 * np.einsum("k,akc->ac", Q, u))
    macro_avg_xt.append(np.sqrt(np.einsum("k,akc->ac", Q, (u -data[-1][1])**2) / np.einsum("k,akc->ac", Q, (data[-1][1])**2)))

run_idx  = np.arange(len(macro_avg_xt))[:, np.newaxis]
#csv_data = np.concatenate((run_idx, ne_xt[:, np.newaxis] * op.np0, Te_xt[:, np.newaxis], rel_error_ne[:,np.newaxis], rel_error_Te[:,np.newaxis], rel_error_ne_linear[:,np.newaxis], rel_error_Te_linear[:,np.newaxis]), axis=1)

rel_error_ne = np.array([macro_avg_xt[i][-1, 0] for i in range(len(macro_avg_xt))])
rel_error_Te = np.array([macro_avg_xt[i][-1, 2] for i in range(len(macro_avg_xt))])
csv_data = np.concatenate((run_idx, rel_error_ne[:, np.newaxis], rel_error_Te[:, np.newaxis]), axis=1)
np.savetxt("%s_conv.csv"%(fnames[-1]), csv_data, delimiter="\t", header="run_id\trel_error_ne\trel_error_Te",comments="")

plt.figure(figsize=(16,16), dpi=200)
plt.subplot(1, 2, 1)
#for i in range(len(macro_avg_xt)):
plt.semilogy(run_idx, rel_error_ne, 'o', label=lbl[i])

plt.xlabel(r"run id")
plt.ylabel(r"$n_e$")
plt.grid(visible=True)

plt.subplot(1, 2, 2)
plt.semilogy(run_idx, rel_error_Te, 'o', label=lbl[i])

plt.xlabel(r"run id")
plt.ylabel(r"$T_e$")
plt.grid(visible=True)

# plt.subplot(2, 2, 3)
# rel_error_ne         = np.array([np.abs(1-macro_avg_xt[i][-1, 0]/macro_avg_xt[-1][-1, 0]) for i in range(len(macro_avg_xt))])
# rel_error_ne_linear  = np.array([np.abs(1-macro_avg_xt[0][-1, 0]/macro_avg_xt[-1][-1, 0])/float(2**i) for i in range(len(macro_avg_xt))])
# #print(rel_error_ne)
# plt.plot(rel_error_ne, 'o-')
# plt.plot(rel_error_ne_linear, "o--")
# plt.xticks([i for i in range(len(macro_avg_xt))], ['r%d'%i for i in range(len(macro_avg_xt))])
# #plt.semilogy(np.array(idx_range), , label=lbl[i])
# #plt.semilogy(np.array(idx_range), np.abs(1-macro_avg_xt[0][:, 0]/macro_avg_xt[-1][:, 0])/2**i, 'k--')
# plt.xlabel(r"run id")
# plt.ylabel(r"relative error $n_e$")
# plt.grid(visible=True)

# plt.subplot(2, 2, 4)
# rel_error_Te         = np.array([np.abs(1-macro_avg_xt[i][-1, 2]/macro_avg_xt[-1][-1, 2]) for i in range(len(macro_avg_xt))])
# rel_error_Te_linear  = np.array([np.abs(1-macro_avg_xt[0][-1, 2]/macro_avg_xt[-1][-1, 2])/float(2**i) for i in range(len(macro_avg_xt))])
# plt.plot(rel_error_Te, 'o-')
# plt.plot(rel_error_Te_linear, "o--")
# plt.xticks([i for i in range(len(macro_avg_xt))], ['r%d'%i for i in range(len(macro_avg_xt))])
# #plt.semilogy(np.array(idx_range), , label=lbl[i])
# #plt.semilogy(np.array(idx_range), np.abs(1-macro_avg_xt[0][:, 0]/macro_avg_xt[-1][:, 0])/2**i, 'k--')
# plt.xlabel(r"run id")
# plt.ylabel(r"relative error $T_e$")
# plt.grid(visible=True)

# for i in range(len(macro_avg_xt)-1):
#     plt.semilogy(np.array(idx_range), np.abs(1-macro_avg_xt[i][:, 2]/macro_avg_xt[-1][:, 2]), label=lbl[i])
#     #plt.semilogy(np.array(idx_range), np.abs(1-macro_avg_xt[0][:, 2]/macro_avg_xt[-1][:, 2])/2**i, 'k--')

# plt.xlabel(r"cycle")
# plt.ylabel(r"$T_e$")
# plt.grid(visible=True)
plt.tight_layout()
plt.show()
plt.close()





#sys.exit(0)

    





# for idx, cycle_id in enumerate(idx_range):
#     for i, u in enumerate(macro_xx):
#         ne = u[idx, : , 0]
#         ni = u[idx, : , 1]
#         Te = u[idx, : , 2]
        
#         ne_h = macro_xx[-1][idx, : , 0]
#         ni_h = macro_xx[-1][idx, : , 1]
#         Te_h = macro_xx[-1][idx, : , 2]

#         rel_error_ne.append(np.linalg.norm(ne - ne_h)/np.linalg.norm(ne_h))
#         rel_error_Te.append(np.linalg.norm(ne * Te - ne_h * Te_h)/np.linalg.norm(ne_h * Te_h))

# rel_error_ne = np.array(rel_error_ne).reshape((len(idx_range), -1))
# rel_error_Te = np.array(rel_error_Te).reshape((len(idx_range), -1))
# tt = np.linspace(0, 1, len(idx_range))

# plt.subplot(1, 2, 1)
# plt.semilogy(tt, rel_error_ne[:, 0],"-",label=r"$n_e (r0 vs r2)$")
# plt.semilogy(tt, rel_error_ne[:, 1],"-",label=r"$n_e (r1 vs r2)$")
# #plt.semilogy(tt, rel_error_Te,"x-",label=r"$T_e$")
# plt.xlabel(r"time [T]")
# plt.ylabel(r"relative error")

# plt.subplot(1, 2, 2)
# plt.semilogy(tt, rel_error_Te[:, 0],"-",label=r"$T_e (r0 vs r2)$")
# plt.semilogy(tt, rel_error_Te[:, 1],"-",label=r"$T_e (r1 vs r2)$")
# #plt.semilogy(tt, rel_error_Te,"x-",label=r"$T_e$")
# plt.xlabel(r"time [T]")
# plt.ylabel(r"relative error")
# plt.tight_layout()
# plt.show()
# plt.close()



        



#xx_grid = plot_utils.op(400).xp

for xidx in [50, 100, 200]:
    ev_grid = np.linspace(1e-4, 40, 4000)
    plt.figure(figsize=(16, 4), dpi=200)
    for idx, d in enumerate(data):
        u   = d[1]
        Np  = u.shape[1]
        
        op  = plot_utils.op(Np)
        v0  = np.polynomial.chebyshev.chebvander(xx_grid, Np-1)
        p0  = np.dot(v0, op.V0pinv)
        
        v   = np.einsum("ik,abk->abi", p0, d[2])
        u   = np.einsum("ik,akc->aic", p0, d[1])
        
        spec_sp, col_list = plot_utils.gen_spec_sp(d[0])
        rr                = plot_utils.compute_radial_components(d[0], d[3], spec_sp, ev_grid, v)
        
        for tidx, cycle_id in enumerate(idx_range):
            np.savetxt("%s_fl_x%.3f_%02d.csv"%(fnames[idx], xx_grid[xidx], cycle_id), np.array([ev_grid, rr[tidx, xidx, 0], rr[tidx, xidx, 1]]).T, header="x\tf0\tf1", delimiter="\t", comments="")
        
        
        for l in range(2):
            plt.subplot(1, 2, l + 1)    
            plt.semilogy(ev_grid, np.abs(rr[-1, xidx, l]), label=lbl[idx])
            
            # plt.subplot(3, 3, 3 * l + 2)
            # plt.semilogy(ev_grid, np.abs(rr[-2, xidx, l]), label=lbl[idx])
            
            # plt.subplot(3, 3, 3 * l + 3)
            # plt.semilogy(ev_grid, np.abs(rr[-1, xidx, l]), label=lbl[idx])
            

    plt.subplot(1, 2, 1)
    plt.legend()
    plt.grid(visible=True)
    plt.xlabel(r"eV")
    plt.ylabel(r"$f_%d [eV^{-3/2}]$"%(1))

    plt.subplot(1, 2, 2)
    plt.legend()
    plt.grid(visible=True)
    plt.xlabel(r"eV")
    plt.ylabel(r"$f_%d [eV^{-3/2}]$"%(2))

    # plt.subplot(1, 3, 3)
    # plt.legend()
    # plt.grid(visible=True)
    # plt.xlabel(r"eV")
    # plt.ylabel(r"$f_%d [eV^{-3/2}]$"%(3))

    # for l in range(2):    
    #     plt.subplot(1, 3, 3 * l +1)
    #     plt.legend()
    #     plt.grid(visible=True)
    #     plt.xlabel(r"eV")
    #     plt.ylabel(r"$f_%d [eV^{-3/2}]$"%(l))
        
    #     plt.subplot(1, 3, 3 * l +2)
    #     plt.legend()
    #     plt.grid(visible=True)
    #     plt.xlabel(r"eV")
    #     plt.ylabel(r"$f_%d [eV^{-3/2}]$"%(l))
        
    #     plt.subplot(1, 3, 3 * l +3)
    #     plt.legend()
    #     plt.grid(visible=True)
    #     plt.xlabel(r"eV")
    #     plt.ylabel(r"$f_%d [eV^{-3/2}]$"%(l))


    plt.show()
    plt.close()






