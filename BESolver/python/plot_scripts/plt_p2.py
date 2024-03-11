import numpy as np
import matplotlib.pyplot as plt
import plot_utils

# ne_pic = 1e6 * np.array([np.fromfile(open("./../1d2v_pic/E10K_g0g2_PIC/case4_Emanual_fullcolls_ne_CA_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in range(0, 11)])
# Te_pic = np.array([np.fromfile(open("./../1d2v_pic/E10K_g0g2_PIC/case4_Emanual_fullcolls_Te_CA_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in range(0, 11)])
# g0_pic = np.array([np.fromfile(open("./../1d2v_pic/E10K_g0g2_PIC/case4_Emanual_fullcolls_g0_ar_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in range(0, 11)])
# g2_pic = np.array([np.fromfile(open("./../1d2v_pic/E10K_g0g2_PIC/case4_Emanual_fullcolls_g2_ar_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in range(0, 11)])
# xx_pic = np.linspace(-1,1,len(ne_pic[0]))


#cycle_list=[1, 10, 21, 41, 101, 151, 191]
#cycle_list = list(range(1, 151, 4))
cycle_list = [150]

ne_pic0 = 1e6 * np.array([np.fromfile(open("./../1dglow/1dglow_pic/MLOAD_V2_ne_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in cycle_list])
Te_pic0 = np.array([np.fromfile(open("./../1dglow/1dglow_pic/MLOAD_V2_Te_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in cycle_list])
g0_pic0 = np.array([np.fromfile(open("./../1dglow/1dglow_pic/MLOAD_V2_g0_ar_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in cycle_list])
g2_pic0 = np.array([np.fromfile(open("./../1dglow/1dglow_pic/MLOAD_V2_g2_ar_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in cycle_list])

ne_pic1 = 1e6 * np.array([np.fromfile(open("./../1dglow/1dglow_pic_high_res/run_N4M_ne_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in cycle_list])
Te_pic1 = np.array([np.fromfile(open("./../1dglow/1dglow_pic_high_res/run_N4M_Te_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in cycle_list])
g0_pic1 = np.array([np.fromfile(open("./../1dglow/1dglow_pic_high_res/run_N4M_g0_ar_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in cycle_list])
g2_pic1 = np.array([np.fromfile(open("./../1dglow/1dglow_pic_high_res/run_N4M_g2_ar_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in cycle_list])

ne_pic2 = 1e6 * np.array([np.fromfile(open("./../1dglow/N40M_cycle150/run_N40M_M1000ne_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in cycle_list])
ni_pic2 = 1e6 * np.array([np.fromfile(open("./../1dglow/N40M_cycle150/run_N40M_M1000ni_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in cycle_list])
E_pic2  = np.array([np.fromfile(open("./../1dglow/N40M_cycle150/run_N40M_M1000E_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in cycle_list])
Te_pic2 = np.array([np.fromfile(open("./../1dglow/N40M_cycle150/run_N40M_M1000Te_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in cycle_list])
g0_pic2 = np.array([np.fromfile(open("./../1dglow/N40M_cycle150/run_N40M_M1000g0_ar_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in cycle_list])
g2_pic2 = np.array([np.fromfile(open("./../1dglow/N40M_cycle150/run_N40M_M1000g2_ar_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in cycle_list])

# ne_pic1 = 1e6 * np.array([np.fromfile(open("./../1dglow/1dglow_pic_40T/MLOAD_V2_ne_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in cycle_list])
# Te_pic1 = np.array([np.fromfile(open("./../1dglow/1dglow_pic_40T/MLOAD_V2_Te_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in cycle_list])
# g0_pic1 = np.array([np.fromfile(open("./../1dglow/1dglow_pic_40T/MLOAD_V2_g0_ar_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in cycle_list])
# g2_pic1 = np.array([np.fromfile(open("./../1dglow/1dglow_pic_40T/MLOAD_V2_g2_ar_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in cycle_list])

# print(np.max(np.abs(ne_pic1 - ne_pic)))
# print(np.max(np.abs(Te_pic1 - ne_pic)))
# print(np.max(np.abs(g0_pic1 - ne_pic)))
# print(np.max(np.abs(g2_pic1 - ne_pic)))


#g0_pic = np.array([np.fromfile(open("./../1dglow/1dglow_pic_20T/MLOAD_V2_g0avg_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in cycle_list])
#g2_pic = np.array([np.fromfile(open("./../1dglow/1dglow_pic_20T/MLOAD_V2_g2avg_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in cycle_list])

#xx_pic = np.linspace(-1,1,len(ne_pic[0]))
xx_pic0 = np.linspace(-1,1,len(ne_pic0[0]))
xx_pic1 = np.linspace(-1,1,len(ne_pic1[0]))
xx_pic2 = np.linspace(-1,1,len(ne_pic2[0]))

d       = plot_utils.load_data_bte("./../1dglow/r3/", cycle_list, None, read_cycle_avg=False)
op      = plot_utils.op(200)
ne_pde  = d[1][:, :, 0] * op.np0 
ni_pde  = d[1][:, :, 1] * op.np0 
Te_pde  = d[1][:, :, 2]
xx_pde  = op.xp
g0_pde  = np.load("./../1dglow/r3/rates_avg_elastic.npy")
g2_pde  = np.load("./../1dglow/r3/rates_avg_ionization.npy")

g0_pde  = g0_pde[np.array(cycle_list),:]
g2_pde  = g2_pde[np.array(cycle_list),:]


print("dx: min PDE", np.min(np.array([xx_pde[i]-xx_pde[i-1] for i in range(1, len(xx_pde))])), "dx max PDE", np.max(np.array([xx_pde[i]-xx_pde[i-1] for i in range(1, len(xx_pde))])) )
print(len(xx_pic0), len(xx_pic1), len(xx_pic2))
print("pic 0", xx_pic0[1]-xx_pic0[0], "pic 1", xx_pic1[1]-xx_pic1[0], "pic 2", xx_pic2[1]-xx_pic2[0])

u_fluid      = np.load("../1dglow_fluid/r1_teb_flux/1d_glow_1000_avg.npy")
ne_fluid     = u_fluid[:, 0] * op.np0 
ni_fluid     = u_fluid[:, 1] * op.np0
Te_fluid_avg = u_fluid[:, 2]/ u_fluid[:,0] 

u_fluid  = np.load("../1dglow_fluid/r1_teb_flux/1d_glow_1000.npy")
ne_fluid = u_fluid[:, 0] * op.np0 
ni_fluid = u_fluid[:, 1] * op.np0
Te_fluid = u_fluid[:, 2]/ u_fluid[:,0] 

xx_fluid = plot_utils.op(len(ne_fluid)).xp

folder_name = "1dglow_pic_pde_fluid_tmp"
plot_utils.make_dir(folder_name)
for idx, cycle in enumerate(cycle_list):
    
    phi     = op.solve_poisson((ne_pde/op.np0).reshape((-1)), (ni_pde/op.np0).reshape((-1)), 0)
    E_pde   = -np.dot(op.Dp, phi) * (op.V0/op.L)
    
    plt.figure(figsize=(15, 10), dpi=300)
    plt.subplot(2, 3, 1)
    plt.plot(xx_fluid, ne_fluid, "g--" , label="Fluid")
    plt.plot(xx_pde  , ne_pde[idx], "r"   , label="PDE")
    #plt.plot(xx_pic  , ne_pic[idx], "o"   , markersize=2, color="blue", label="PIC-DSMC")
    plt.plot(xx_pic0  , ne_pic0[idx], "."   , markersize=1.5, label="PIC-DSMC 0")
    plt.plot(xx_pic1  , ne_pic1[idx], "."   , markersize=1.5, label="PIC-DSMC 1")
    plt.plot(xx_pic2  , ne_pic2[idx], "."   , markersize=1.5, label="PIC-DSMC 2")
    
    plt.grid()
    plt.legend(loc ="lower left")
    plt.xlabel(r"$2x/L  -1$")
    plt.ylabel(r"$n_e\ (m^3s^{-1})$")

    plt.subplot(2, 3, 2)
    plt.plot(xx_fluid, Te_fluid, "g--" , label="Fluid")
    plt.plot(xx_pde  , Te_pde[idx], "r"   , label="PDE")
    #plt.plot(xx_pic  , Te_pic[idx], "."   , markersize=2, color="blue", label="PIC-DSMC 0")
    
    plt.plot(xx_pic0  , Te_pic0[idx], "."   , markersize=1.5, label="PIC-DSMC 0")
    plt.plot(xx_pic1  , Te_pic1[idx], "."   , markersize=1.5, label="PIC-DSMC 1")
    plt.plot(xx_pic2  , Te_pic2[idx], "."   , markersize=1.5, label="PIC-DSMC 2")
    
    plt.ylim((0,10))
    plt.grid()
    plt.legend(loc ="upper right")
    plt.xlabel(r"$2x/L  -1$")
    plt.ylabel(r"$T_e$ (eV)")

    plt.subplot(2, 3, 3)
    plt.plot(xx_pde   , g0_pde[idx], "r" , label="PDE")
    #plt.plot(xx_pic   , g0_pic[idx], "o" , markersize=2, color="blue", label="PIC-DSMC")
    
    plt.plot(xx_pic0  , g0_pic0[idx], "."   , markersize=1.5, label="PIC-DSMC 0")
    plt.plot(xx_pic1  , g0_pic1[idx], "."   , markersize=1.5, label="PIC-DSMC 1")
    plt.plot(xx_pic2  , g0_pic2[idx], "."   , markersize=1.5, label="PIC-DSMC 2")
    
    plt.grid()
    plt.legend(loc ="upper right")
    plt.xlabel(r"$2x/L  -1$")
    plt.ylabel(r"$k_{elastic}$ $(m^3s^{-1})$")

    plt.subplot(2, 3, 4)
    plt.semilogy(xx_fluid, op.ki(Te_fluid_avg, 1) * op.r_fac , "g--" , label="Fluid")
    plt.semilogy(xx_pde  , g2_pde[idx]        , "r"   , label="PDE")
    #plt.plot(xx_pic      , g2_pic[idx]        , "o"   , markersize=2, color="blue", label="PIC-DSMC")
    
    plt.plot(xx_pic0  , g2_pic0[idx], "."   , markersize=1, label="PIC-DSMC 0")
    plt.plot(xx_pic1  , g2_pic1[idx], "."   , markersize=1, label="PIC-DSMC 1")
    plt.plot(xx_pic2  , g2_pic2[idx], "."   , markersize=1, label="PIC-DSMC 2")
    
    plt.grid()
    plt.legend(loc ="upper right")
    plt.xlabel(r"$2x/L  -1$")
    plt.ylabel(r"$k_{ionization}$ $(m^3s^{-1})$")
    
    plt.subplot(2, 3, 5)
    plt.plot(xx_pde  , ni_pde[idx], "r"     , label="PDE")
    plt.plot(xx_pic2  , ne_pic2[idx], "."   , markersize=1.5, label="PIC-DSMC 2")
    
    plt.grid()
    plt.legend(loc ="upper right")
    plt.xlabel(r"$2x/L  -1$")
    plt.ylabel(r"$n_i$ $(m^3)$")
    
    plt.subplot(2, 3, 6)
    plt.plot(xx_pde  , E_pde, "r"     , label="PDE")
    plt.plot(xx_pic2[:-1] , 1e2 * E_pic2[idx], "."   , markersize=1.5, label="PIC-DSMC 2")
    plt.grid()
    plt.legend(loc ="upper right")
    plt.xlabel(r"$2x/L  -1$")
    plt.ylabel(r"$E$(V/m)$")
    
    plt.suptitle("t=%04dT"%(cycle))
    plt.savefig("%s/cycle_%04d.png"%(folder_name, idx))
    plt.close()