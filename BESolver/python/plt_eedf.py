"""
@package : Simple script to plot EEDF from the simulation data. 
"""
import matplotlib.pyplot as plt
import spec_spherical
import argparse
import numpy as np
import scipy.constants 
import utils as BEUtils
import collisions


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="filename with spec coefficients", type=str)
parser.add_argument("-freq", "--frequency", help="freq. to extract EEDF", type=int,default=100)
parser.add_argument("-o", "--output", help="output filename", type=str)
args = parser.parse_args()

data = np.loadtxt(args.filename,skiprows=1)
ev_pts = np.linspace(0,100,10000)
EV       = scipy.constants.electron_volt
E_MASS   = scipy.constants.electron_mass
vr       = np.sqrt(2* EV * ev_pts /E_MASS)

TIME_INDEX=0
TEMP_INDEX=3
C000_INDEX=4

INIT_MASS = data[0,1]

for i in range(0,data.shape[0],args.frequency):
    time = data[i,TIME_INDEX]
    temp = data[i,TEMP_INDEX]

    vth  = collisions.electron_thermal_velocity(temp)
    Mv   = BEUtils.get_maxwellian_3d(vth,INIT_MASS)

    eedf_pts = data[i,C000_INDEX] * Mv(vr/vth) 

    plt.plot(ev_pts,eedf_pts,label="t=%.4E"%time)
    plt.xlabel("Energy (eV)")
    plt.ylabel("EEDF")
    plt.grid()
    plt.legend()
    

plt.show()
plt.savefig(args.output)
plt.close()






# def plot_EEDF(ev_pts, spec : spec_spherical.SpectralExpansionSpherical, cf_list, maxwellian_list, vth_list, scale=1):
#     """
#     Assumes spherical harmonic basis in vtheta vphi direction. 
#     the integration over the spherical harmonics is done analytically. 
#     """
#     EV       = scipy.constants.electron_volt
#     E_MASS   = scipy.constants.electron_mass
#     vr       = np.sqrt(2* EV * ev_pts /E_MASS)
#     #print(vr)
#     #print(vr/vth)
#     #print(maxwellian(vr/vth))
    
#     for i,cf in enumerate(cf_list):
#         eedf_pts = cf[0] * maxwellian_list[i](vr/vth_list[i]) * spec.basis_eval_radial(vr/vth_list[i],0)
#         plt.plot(ev_pts,eedf_pts)
#         plt.xlabel("Energy (eV)")
#         plt.ylabel("Number")
#         plt.grid()
#     plt.show()