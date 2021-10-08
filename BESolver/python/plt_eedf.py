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
MASS_INDEX=1
TEMP_INDEX=3
C000_INDEX=4


INIT_MASS = data[0,1]

def spec_tail(cf):
    return np.linalg.norm(cf[cf.shape[0]//2:]) / np.linalg.norm(cf)

plt.rcParams['axes.grid'] = True
fig,ax_plt = plt.subplots(2,2)
fig.set_figheight(8)
fig.set_figwidth(8)
sp_tail=list()

for i in range(0,data.shape[0]):
    sp_tail.append(spec_tail(data[i,C000_INDEX:]))

for i in range(0,data.shape[0],args.frequency):
    time = data[i,TIME_INDEX]
    temp = data[i,TEMP_INDEX]

    vth  = collisions.electron_thermal_velocity(temp)
    Mv   = BEUtils.get_maxwellian_3d(vth,INIT_MASS)

    eedf_pts = data[i,C000_INDEX] * Mv(vr/vth) 

    #print(np.append(time,sp_tail))


    ax_plt[0,0].plot(ev_pts,eedf_pts,label="t=%.4E"%time)
    ax_plt[0,0].set_xlabel("Energy (eV)")
    ax_plt[0,0].set_ylabel("EEDF")
    ax_plt[0,0].set_yscale("log")
    ax_plt[0,0].set_xscale("log")
    #ax_plt[0,0].grid()
    ax_plt[0,0].legend()

    
ax_plt[0,1].plot(data[:,TIME_INDEX],data[:,MASS_INDEX])
ax_plt[0,1].set_xlabel("time(s)")
ax_plt[0,1].set_ylabel("mass")    
#ax_plt[0,1].grid()

ax_plt[1,0].plot(data[:,TIME_INDEX],data[:,TEMP_INDEX])
ax_plt[1,0].set_xlabel("time(s)")
ax_plt[1,0].set_ylabel("temp(K)")    
#ax_plt[1,0].grid()

ax_plt[1,1].plot(data[:,TIME_INDEX],np.array(sp_tail))
ax_plt[1,1].set_xlabel("time(s)")
ax_plt[1,1].set_ylabel("norm_tail/norm_overall")    
#ax_plt[1,1].grid()

plt.tight_layout()

#plt.show()
plt.savefig(args.filename+".png")
#plt.savefig(args.output)
#plt.close()
#f.show()
#f.close()






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