import sys
sys.path.append("../.")
import collisions
import spec_spherical as sp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plot_utils
import scipy.constants
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use('ggplot')

font = {#'family' : 'normal',
        'weight' : 'bold',
        'size'   : 6}
matplotlib.rc('font', **font)



ne = np.load("../1dglow_hybrid/1Torr300K_Ar_3sp2r_l2_cycle/species_densities.npy")[:, : ,0]
ni = np.load("../1dglow_hybrid/1Torr300K_Ar_3sp2r_l2_cycle/species_densities.npy")[:, : ,1]
op = plot_utils.op(ne.shape[1])
tt = np.linspace(0, 1, ne.shape[0])
xx = op.xp


hybrid_data       = dict()
hybrid_data["x"]  = xx
hybrid_data["tt"] = tt
hybrid_data["ne"] = ne * op.np0
hybrid_data["ni"] = ni * op.np0
hybrid_data["Te"] = np.load("../1dglow_hybrid/1Torr300K_Ar_3sp2r_l2_cycle/Te.npy")
hybrid_data["uz"] = np.load("../1dglow_hybrid/1Torr300K_Ar_3sp2r_l2_cycle/u_z.npy")
hybrid_data["mueE"] = np.load("../1dglow_hybrid/1Torr300K_Ar_3sp2r_l2_cycle/mueE.npy")
hybrid_data["De"]   = np.load("../1dglow_hybrid/1Torr300K_Ar_3sp2r_l2_cycle/De.npy")

hybrid_data["elastic"]    = np.load("../1dglow_hybrid/1Torr300K_Ar_3sp2r_l2_cycle/rates_elastic.npy")
hybrid_data["ionization"] = np.load("../1dglow_hybrid/1Torr300K_Ar_3sp2r_l2_cycle/rates_ionization.npy")

hybrid_data["charge_density"]    = hybrid_data["ni"] - hybrid_data["ne"]
hybrid_data["plasma_potential"]  = np.array([op.solve_poisson(ne[i], ni[i], tt[i]) for i in range(len(tt))]) * op.V0
hybrid_data["E"]                 = (np.dot(op.Dp, hybrid_data["plasma_potential"].T).T) * 1 / op.L
hybrid_data["Je"]                = hybrid_data["uz"] * hybrid_data["ne"]

hybrid_data["Je1"]               = np.array([hybrid_data["mueE"][i] * hybrid_data["ne"][i] - (1.0/op.L) * hybrid_data["De"][i] * np.dot(op.Dp, hybrid_data["ne"][i])  for i in range(len(tt))])

extent  = [xx[0] - 0.5 * (xx[1]-xx[0]), xx[-1] + 0.5 * (xx[-1]-xx[-2]), tt[0] - 0.5 * (tt[1] - tt[0]), tt[-1] + 0.5 * (tt[-1] - tt[-2])]

def plot_image(qoi, extent, fname, cb_range=None, use_log=True):
    plt.figure(figsize=(6,6), dpi=200)
    im_ratio = qoi.shape[0]/qoi.shape[0]
    
    if cb_range is None:
        vmin = np.min(qoi)
        vmax = np.max(qoi)
    else:
        vmin = cb_range[0]
        vmax = cb_range[1]
    
    if use_log == True:
        plt.imshow(qoi, interpolation="gaussian", cmap='plasma', extent=extent, norm=LogNorm(vmin=vmin, vmax=vmax), aspect=2)
    else:
        plt.imshow(qoi, interpolation="gaussian", cmap='plasma', extent=extent, vmin=vmin, vmax=vmax, aspect=2)
    plt.grid(False)
    fmt = lambda x, pos: "%.2E"%(x)#'{:.2E%}'.format(x)
    plt.colorbar(fraction=0.045*im_ratio, format=FuncFormatter(fmt))
    plt.xlabel(r"$\hat{x}$")
    plt.ylabel(r"$t/T$")
    #plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    return

fprefix = "../../../../papers/boltzmann1d-paper/fig/1Torr300K"
plot_image(hybrid_data["ne"], extent=extent,  fname="%s_hybrid_ne.png"%(fprefix), cb_range=(1e10, 4e16))
plot_image(hybrid_data["ni"], extent=extent,  fname="%s_hybrid_ni.png"%(fprefix), cb_range=(1e16, 4e16))
plot_image(hybrid_data["charge_density"] * scipy.constants.elementary_charge, extent=extent,  fname="%s_hybrid_charge.png"%(fprefix), cb_range=(0, 8e-4), use_log=False)
plot_image(hybrid_data["plasma_potential"] , extent=extent,  fname="%s_hybrid_potential.png"%(fprefix), cb_range=(-100, 100), use_log=False)

kine = hybrid_data["ionization"] * hybrid_data["ne"] * op.np0 * op.n0
plot_image(kine, extent=extent,  fname="%s_hybrid_production.png"%(fprefix), cb_range=(1e18, 1e22), use_log=False)

plot_image(hybrid_data["ne"] * hybrid_data["Te"] * 1.5 * scipy.constants.electron_mass  , extent=extent, fname="%s_hybrid_energy_density.png"%(fprefix), cb_range=(1e-19, 1e-13), use_log=True)
plot_image(hybrid_data["E"]           , extent=extent,  fname="%s_hybrid_E.png"%(fprefix) , cb_range=(-8e4, 8e4), use_log=False)
plot_image(hybrid_data["Je"]          , extent=extent,  fname="%s_hybrid_Je.png"%(fprefix), cb_range=(-1.3e20, 1.3e20), use_log=False)




ne   = np.load("../1dglow_fluid/1Torr300K_Ar_3sp2r_tab_cycle/1d_glow_cycle.npy")[:, : ,0]
ni   = np.load("../1dglow_fluid/1Torr300K_Ar_3sp2r_tab_cycle/1d_glow_cycle.npy")[:, : ,1]
neTe = np.load("../1dglow_fluid/1Torr300K_Ar_3sp2r_tab_cycle/1d_glow_cycle.npy")[:, : ,2] 
Te   = neTe/ne

op = plot_utils.op(ne.shape[1])
tt = np.linspace(0, 1, ne.shape[0])
xx = op.xp

extent  = [xx[0] - 0.5 * (xx[1]-xx[0]), xx[-1] + 0.5 * (xx[-1]-xx[-2]), tt[0] - 0.5 * (tt[1] - tt[0]), tt[-1] + 0.5 * (tt[-1] - tt[-2])]


fluid_data       = dict()
fluid_data["x"]  = xx
fluid_data["tt"] = tt

fluid_data["ne"] = ne * op.np0
fluid_data["ni"] = ni * op.np0
fluid_data["Te"] = Te

fluid_data["ionization"]         = np.array([op.ki(Te[i],1) * op.r_fac for i in range(Te.shape[0])]) #np.load("../1dglow_hybrid/1Torr300K_Ar_3sp2r_l2_cycle/rates_ionization.npy")
fluid_data["charge_density"]     = fluid_data["ni"] - fluid_data["ne"]
fluid_data["plasma_potential"]   = np.array([op.solve_poisson(ne[i], ni[i], tt[i]) for i in range(len(tt))]) * op.V0
fluid_data["E"]                  = (np.dot(op.Dp, fluid_data["plasma_potential"].T).T) * 1 / op.L
fluid_data["mueE"]               = np.array([op.mu_e(Te[i],1) * op.mu_fac * fluid_data["E"][i] for i in range(Te.shape[0])])
fluid_data["De"]                 = np.array([op.De(Te[i],1)   * op.D_fac for i in range(Te.shape[0])])
fluid_data["Je"]                 = np.array([fluid_data["mueE"][i] * fluid_data["ne"][i] - (1/op.L) * fluid_data["De"][i] * np.dot(op.Dp, fluid_data["ne"][i])  for i in range(Te.shape[0])])

plot_image(fluid_data["ne"]                                                            , extent=extent,  fname="%s_fluid_ne.png"%(fprefix), cb_range=(1e10, 4e16))
plot_image(fluid_data["ni"]                                                            , extent=extent,  fname="%s_fluid_ni.png"%(fprefix), cb_range=(1e16, 4e16))
plot_image(fluid_data["charge_density"] * scipy.constants.elementary_charge            , extent=extent,  fname="%s_fluid_charge.png"%(fprefix), cb_range=(0, 8e-4), use_log=False)
plot_image(fluid_data["plasma_potential"]                                              , extent=extent,  fname="%s_fluid_potential.png"%(fprefix), cb_range=(-100, 100), use_log=False)
plot_image(fluid_data["ne"] * hybrid_data["Te"] * 1.5 * scipy.constants.electron_mass  , extent=extent,  fname="%s_fluid_energy_density.png"%(fprefix), cb_range=(1e-19, 1e-13), use_log=True)
plot_image(fluid_data["E"]                                                             , extent=extent,  fname="%s_fluid_E.png"%(fprefix), cb_range=(-8e4, 8e4), use_log=False)

kine = fluid_data["ionization"] * fluid_data["ne"] * op.np0 * op.n0
plot_image(kine, extent=extent,  fname="%s_fluid_production.png"%(fprefix), cb_range=(1e18, 1e22), use_log=False)

plot_image(fluid_data["Je"]                                                            , extent=extent,  fname="%s_fluid_Je.png"%(fprefix), cb_range=(-1.3e20, 1.3e20), use_log=False)

ue_h      = hybrid_data["uz"]
ue_f      = fluid_data["Je"]/fluid_data["ne"]  
qoi_diff  = np.abs(ue_h - ue_f)/np.max(np.abs(ue_h))
plot_image(qoi_diff , extent=extent,  fname="%s_diff_ue.png"%(fprefix), use_log=True)
qoi_diff  = np.abs(hybrid_data["Je"] - fluid_data["Je"])/np.max(np.abs(hybrid_data["Je"]))
plot_image(qoi_diff , extent=extent,  fname="%s_diff_Je.png"%(fprefix), use_log=True)

plt.plot(hybrid_data["x"] , (plot_utils.time_average(hybrid_data["Je"]  , hybrid_data["tt"])), label="hybrid")
#plt.plot(hybrid_data["x"] , (plot_utils.time_average(hybrid_data["Je1"] , hybrid_data["tt"])) , label="hybrid - (drift-diffusion)")
plt.plot(fluid_data["x"]  , (plot_utils.time_average(fluid_data["Je"]   , fluid_data["tt"]) ), label="fluid")
plt.grid(visible=True)
plt.legend()
#plt.show()
plt.savefig("%s_electron_flux_ca.png"%(fprefix))
plt.close()

# plot_image(fluid_data["ne"], extent=extent,  fname="%s_fluid_ne.png"%(fprefix))
# plot_image(fluid_data["ni"], extent=extent,  fname="%s_fluid_ni.png"%(fprefix))
# plot_image(fluid_data["charge_density"] * scipy.constants.elementary_charge, extent=extent,  fname="%s_fluid_charge.png"%(fprefix), use_log=False)
# plot_image(fluid_data["plasma_potential"] , extent=extent,  fname="%s_fluid_potential.png"%(fprefix), use_log=False)
# plot_image(fluid_data["ne"] * fluid_data["Te"] * 1.5 * scipy.constants.electron_mass  , extent=extent,  fname="%s_fluid_energy_density.png"%(fprefix), use_log=True)
# plot_image(fluid_data["E"], extent=extent,  fname="%s_fluid_E.png"%(fprefix), use_log=False)


data                = hybrid_data
ca_ne_hybrid        = plot_utils.time_average(data["ne"]                                                     ,  data["tt"])
ca_ion_prod_hybrid  = plot_utils.time_average(data["ionization"] * data["ne"] * op.np0 * op.n0                       ,  data["tt"])
ca_energy_hybrid    = plot_utils.time_average(data["Te"] * data["ne"] * 1.5 * scipy.constants.electron_mass  ,  data["tt"])


data               = fluid_data
ca_ne_fluid        = plot_utils.time_average(data["ne"]                                                     ,  data["tt"])
ca_ion_prod_fluid  = plot_utils.time_average(data["ionization"] * data["ne"] * op.np0 * op.n0                       ,  data["tt"])
ca_energy_fluid    = plot_utils.time_average(data["Te"] * data["ne"] * 1.5 * scipy.constants.electron_mass  ,  data["tt"])

