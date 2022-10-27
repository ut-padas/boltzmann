import matplotlib.pyplot as plt
import numpy as np
import collisions

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    "font.size": 20,
    "lines.linewidth":0.5
})

g0 = collisions.eAr_G0()
g2 = collisions.eAr_G2()

c_list = [g0, g2]
ev     = np.logspace(-3, 3, 100,base=10)

fig=plt.figure(figsize=(9, 7), dpi=300)
plt.loglog(ev, g0._total_cs_interp1d(ev),'rx--', markersize=2,label="elastic")
plt.loglog(ev, g2._total_cs_interp1d(ev),'bx--', markersize=2,label="ionization")
#plt.tight_layout()
plt.grid()
plt.legend()
plt.xlabel(r"energy (eV)")
plt.ylabel(r"cross section ($m^2$)")
fig.savefig("g0_g2_cs.png")
plt.close()

    

