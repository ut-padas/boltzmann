import numpy as np
import scipy.constants

def get_maxwellian_3d(vth,n_scale=1):
    M = lambda x: (n_scale / ((vth * np.sqrt(np.pi))**3) ) * np.exp(-x**2)
    return M

def synthetic_tcs(ev, mode):
    """
    synthetic cross-sections for testing. 
    """
    if mode==0:
        return 2e-20 * np.ones_like(ev)
    
    elif mode == "g0":
        """
        G0 cross section data fit with analytical function
        """
        ev =     ev+1e-13
        a0 =    0.008787
        b0 =     0.07243
        c  =    0.007048
        d  =      0.9737
        a1 =        3.27
        b1 =       3.679
        x0 =      0.2347
        x1 =       11.71
        y=9.900000e-20*(a1+b1*(np.log(ev/x1))**2)/(1+b1*(np.log(ev/x1))**2)*(a0+b0*(np.log(ev/x0))**2)/(1+b0*(np.log(ev/x0))**2)/(1+c*ev**d)
        assert len(y[y<0]) == 0 , "g0 cross section is negative" 
        return  y
    
    elif mode == "g2":
        """
        G2 cross section data fit with analytical function (ionization)
        """
        y               = np.zeros_like(ev)
        threshold_value = 15.76
        y[ev>threshold_value] = (2.860000e-20/np.log(90-threshold_value)) * np.log((ev[ev>threshold_value]-threshold_value + 1)) * np.exp(-1e-2*((ev[ev>threshold_value]-90)/90)**2)
        y[ev>=10000]=0
        return  y
    else:
        raise NotImplementedError

kB      = scipy.constants.Boltzmann
me      = scipy.constants.electron_mass
qe      = scipy.constants.electron_volt
ev_1    = (qe/kB)
c_gamma = np.sqrt(2 * qe / me)
VTH     = lambda Te : np.sqrt(Te) * c_gamma  #np.sqrt(2 * kB * Te / scipy.constants.electron_mass)

Te         = 0.5 # ev
vth        = VTH(Te)
mw         = get_maxwellian_3d(vth, 1)

vr         = np.linspace(0, 5, 10000)
ev_grid    = (vr * vth/c_gamma)**2
f0         = 4* np.pi * mw(vr)
mass       = vth**3      * np.trapz(vr**2 * f0 , vr, dx=(vr[1]-vr[0]))
f0_n       = f0/mass
avg_energy = (0.5 * me ) * vth**5 * np.trapz(vr**4 * f0 , vr, dx=(vr[1]-vr[0])) / mass
temp       = (2.0/3.0/kB) * avg_energy / (qe/kB)

rate_g0    = vth**4 * np.trapz(vr**3 * synthetic_tcs(ev_grid, "g0") * (f0/ mass), vr, dx=(vr[1]-vr[0]))
rate_g2    = vth**4 * np.trapz(vr**3 * synthetic_tcs(ev_grid, "g2") * (f0/ mass), vr, dx=(vr[1]-vr[0]))

print("input maxwellian at %.8E eV"%(Te))
print("mass [m^{-3}] = %.8E"%(mass))
print("temp [eV]     = %.8E"%(temp))
print("rates (g0)    = %.8E"%(rate_g0))
print("rates (g2)    = %.8E"%(rate_g2))
