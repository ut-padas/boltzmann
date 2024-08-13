import numpy as np
import matplotlib.pyplot as plt
import plot_utils

op = plot_utils.op(200)
cycle_list = range(0, 11, 1)
 



ne_pic = 1e6 * np.array([np.fromfile(open("../1dglow_hybrid/E10K_g0g2_PIC/case4_Emanual_fullcolls_ne_CA_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in cycle_list])
Te_pic = np.array([np.fromfile(open("../1dglow_hybrid/E10K_g0g2_PIC/case4_Emanual_fullcolls_Te_CA_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in cycle_list])
g0_pic = np.array([np.fromfile(open("../1dglow_hybrid/E10K_g0g2_PIC/case4_Emanual_fullcolls_g0_ar_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in cycle_list])
g2_pic = np.array([np.fromfile(open("../1dglow_hybrid/E10K_g0g2_PIC/case4_Emanual_fullcolls_g2_ar_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in cycle_list])
xx_pic = np.linspace(-1,1,len(ne_pic[0]))

xx_pde   = op.xp 
P1       = np.dot(np.polynomial.chebyshev.chebvander(xx_pde, op.Np-1), op.V0pinv)

ne       = np.array([np.load("../1dglow_hybrid/Ewt_10K_g0g2_PDE/1d_glow_%04d_u_avg.npy"%i)[: , 0] for i in cycle_list]) * op.np0  
Te       = np.array([np.load("../1dglow_hybrid/Ewt_10K_g0g2_PDE/1d_glow_%04d_u_avg.npy"%i)[: , 2] for i in cycle_list])

rates    = [ np.load("../1dglow_hybrid/Ewt_10K_g0g2_PDE/rates_avg_elastic.npy"), 
             np.load("../1dglow_hybrid/Ewt_10K_g0g2_PDE/rates_avg_ionization.npy")]

ne   = np.dot(P1, ne.T).T
Te   = np.dot(P1, Te.T).T

rates = [np.dot(P1, rates[0].T).T, np.dot(P1, rates[1].T).T]

for idx in cycle_list:
      if idx == 0:
            continue
      
      plt.figure(figsize=(16, 16), dpi=200)
      plt.subplot(2, 2, 1)
      plt.plot(xx_pde , ne [idx]   , label=r"Eulerian")      
      plt.plot(xx_pic, ne_pic[idx], label=r"PIC-DSMC")
      
      plt.xlabel(r"x")
      plt.ylabel(r"number density [$m^{-3}$]")
      plt.legend()
      plt.grid(visible=True)
      
      plt.subplot(2, 2, 2)
      plt.plot(xx_pde, Te[idx]    , label=r"Eulerian")      
      plt.plot(xx_pic, Te_pic[idx], label=r"PIC-DSMC")
      
      plt.xlabel(r"x")
      plt.ylabel(r"temperature [$eV$]")
      plt.legend()
      plt.grid(visible=True)
      
      plt.subplot(2, 2, 3)
      plt.semilogy(xx_pde, rates[0][idx]    , label=r"Eulerian")      
      plt.semilogy(xx_pic, g0_pic[idx]      , label=r"PIC-DSMC")
      
      plt.xlabel(r"x")
      plt.ylabel(r"elastic rate coefficient [$m^{3}s^{-1}$]")
      plt.legend()
      plt.grid(visible=True)
      
      plt.subplot(2, 2, 4)
      plt.semilogy(xx_pde, rates[1][idx]    , label=r"Eulerian")      
      plt.semilogy(xx_pic, g2_pic[idx]      , label=r"PIC-DSMC")
      
      plt.xlabel(r"x")
      plt.ylabel(r"ionization rate coefficient [$m^{3}s^{-1}$]")
      plt.legend()
      plt.grid(visible=True)

      plt.tight_layout()
      plt.savefig("pic_pde_ewt_idx%04d.png"%(idx))
      plt.close()



for i in cycle_list:
      np.savetxt("../1dglow_hybrid/Ewt_10K_pde_idx_%04d.csv"%(i), 
      np.array([xx_pde, ne[i], Te[i], rates[0][i], rates[1][i]] , dtype=np.float64).T, 
      header="x\tne_pde\tTe_pde\telastic_pde\tionization_pde",comments="", delimiter="\t")
      
      np.savetxt("../1dglow_hybrid/Ewt_10K_pic_idx_%04d.csv"%(i)   , 
      np.array([xx_pic, ne_pic[i], Te_pic[i], g0_pic[i], g2_pic[i]], dtype=np.float64).T, 
      header="x\tne_pic\tTe_pic\telastic_pic\tionization_pic",comments="", delimiter="\t")
      


ne_pic = 1e6 * np.array([np.fromfile(open("../1dglow_hybrid/Ewtx_10K_g0g2_PIC/georgetest_ne_CA_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in cycle_list])
Te_pic = np.array([np.fromfile(open("../1dglow_hybrid/Ewtx_10K_g0g2_PIC/georgetest_Te_CA_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in cycle_list])
g0_pic = np.array([np.fromfile(open("../1dglow_hybrid/Ewtx_10K_g0g2_PIC/georgetest_g0_ar_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in cycle_list])
g2_pic = np.array([np.fromfile(open("../1dglow_hybrid/Ewtx_10K_g0g2_PIC/georgetest_g2_ar_cycle%d.dat"%(max(i,1))),dtype=np.double) for i in cycle_list])
xx_pic = np.linspace(-1,1,len(ne_pic[0]))

xx_pde   = op.xp 
P1       = np.dot(np.polynomial.chebyshev.chebvander(xx_pde, op.Np-1), op.V0pinv)

ne       = np.array([np.load("../1dglow_hybrid/Ewtx_10K_g0g2_PDE/1d_glow_%04d_u_avg.npy"%i)[: , 0] for i in cycle_list]) * op.np0  
Te       = np.array([np.load("../1dglow_hybrid/Ewtx_10K_g0g2_PDE/1d_glow_%04d_u_avg.npy"%i)[: , 2] for i in cycle_list])

rates    = [ np.load("../1dglow_hybrid/Ewtx_10K_g0g2_PDE/rates_avg_elastic.npy"), 
             np.load("../1dglow_hybrid/Ewtx_10K_g0g2_PDE/rates_avg_ionization.npy")]

ne   = np.dot(P1, ne.T).T
Te   = np.dot(P1, Te.T).T

rates = [np.dot(P1, rates[0].T).T, np.dot(P1, rates[1].T).T]

      
for idx in cycle_list:
      if idx == 0:
            continue
      
      plt.figure(figsize=(16, 16), dpi=200)
      plt.subplot(2, 2, 1)
      plt.plot(xx_pde , ne [idx]   , label=r"Eulerian")      
      plt.plot(xx_pic, ne_pic[idx], label=r"PIC-DSMC")
      
      plt.xlabel(r"x")
      plt.ylabel(r"number density [$m^{-3}$]")
      plt.legend()
      plt.grid(visible=True)
      
      plt.subplot(2, 2, 2)
      plt.plot(xx_pde, Te[idx]    , label=r"Eulerian")      
      plt.plot(xx_pic, Te_pic[idx], label=r"PIC-DSMC")
      
      plt.xlabel(r"x")
      plt.ylabel(r"temperature [$eV$]")
      plt.legend()
      plt.grid(visible=True)
      
      plt.subplot(2, 2, 3)
      plt.semilogy(xx_pde, rates[0][idx]    , label=r"Eulerian")      
      plt.semilogy(xx_pic, g0_pic[idx]      , label=r"PIC-DSMC")
      
      plt.xlabel(r"x")
      plt.ylabel(r"elastic rate coefficient [$m^{3}s^{-1}$]")
      plt.legend()
      plt.grid(visible=True)
      
      plt.subplot(2, 2, 4)
      plt.semilogy(xx_pde, np.abs(ne [idx] * rates[1][idx])    , label=r"Eulerian")      
      plt.semilogy(xx_pic, np.abs(ne_pic [idx] * g2_pic[idx]  )    , label=r"PIC-DSMC")
      
      plt.xlabel(r"x")
      plt.ylabel(r"ionization rate coefficient [$m^{3}s^{-1}$]")
      plt.legend()
      plt.grid(visible=True)

      plt.tight_layout()
      plt.savefig("pic_pde_ewtx_idx%04d.png"%(idx))
      plt.close()



for i in cycle_list:
      np.savetxt("../1dglow_hybrid/Ewtx_10K_pde_idx_%04d.csv"%(i), 
      np.array([xx_pde, ne[i], Te[i], rates[0][i], rates[1][i]] , dtype=np.float64).T, 
      header="x\tne_pde\tTe_pde\telastic_pde\tionization_pde",comments="", delimiter="\t")
      
      np.savetxt("../1dglow_hybrid/Ewtx_10K_pic_idx_%04d.csv"%(i)   , 
      np.array([xx_pic, ne_pic[i], Te_pic[i], g0_pic[i], g2_pic[i]], dtype=np.float64).T, 
      header="x\tne_pic\tTe_pic\telastic_pic\tionization_pic",comments="", delimiter="\t")