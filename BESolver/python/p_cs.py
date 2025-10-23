import matplotlib.pyplot as plt
import numpy as np
import collisions
import cross_section
import scipy.interpolate
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica",
#     "font.size": 20,
#     "lines.linewidth":0.5
# })

avail_species                     = cross_section.read_available_species("./lxcat_data/eAr_crs.Biagi.3sp2r")
cross_section.CROSS_SECTION_DATA  = cross_section.read_cross_section_data("./lxcat_data/eAr_crs.Biagi.3sp2r")
cross_section_data                = cross_section.CROSS_SECTION_DATA

g0 = collisions.eAr_G0()
g2 = collisions.eAr_G2()



c_list = [g0, g2]
ev     = np.logspace(-3, 3, 1000,base=10)

fig=plt.figure(figsize=(9, 7), dpi=300)
plt.loglog(ev, g0.total_cross_section(ev),'rx--', markersize=2,label="elastic")
cs_data                  = cross_section.CROSS_SECTION_DATA["E + Ar -> E + Ar"]
energy                   = cs_data["energy"]
total_cs                 = cs_data["cross section"]
g0_linear                = scipy.interpolate.interp1d(energy, total_cs, kind='linear', bounds_error=False, fill_value=0.0)
plt.loglog(ev, g0_linear(ev),'k-', markersize=2,label="g0-linear")

cs_data                  = cross_section.CROSS_SECTION_DATA["E + Ar -> E + E + Ar+"]
energy                   = cs_data["energy"]
total_cs                 = cs_data["cross section"]
g2_linear                = scipy.interpolate.interp1d(energy, total_cs, kind='linear', bounds_error=False, fill_value=0.0)
plt.loglog(ev, g2.total_cross_section(ev),'bx--', markersize=2,label="ionization")
plt.loglog(ev, g2_linear(ev),'k-', markersize=2,label="g2-linear")

#plt.tight_layout()
plt.grid()
plt.legend()
plt.xlabel(r"energy (eV)")
plt.ylabel(r"cross section ($m^2$)")
#fig.savefig("g0_g2_cs.png")
plt.show()
plt.close()

a1  = g0.total_cross_section(ev)
plt.semilogy(ev, np.abs(1-a1/g0_linear(ev)),'rx--', markersize=2,label="elastic")
idx = ev>15.76
a2  = g2.total_cross_section(ev)
plt.loglog(ev[idx], np.abs(1-a2[idx]/g2_linear(ev)[idx]),'bx--', markersize=2,label="ionization")
plt.grid(visible=True)
plt.legend()
plt.xlabel(r"energy (eV)")
plt.ylabel(r"relative error")
plt.show()

np.save("eAr_g0_g2.npy", np.concatenate([ev.reshape((-1,1)), g0.total_cross_section(ev).reshape((-1,1)), g2.total_cross_section(ev).reshape((-1,1))], axis=1))

    

