import numpy as np
import plot_utils
import scipy.constants
import sys
sys.path.append("../.")
import basis
import cross_section
import utils as bte_utils
import spec_spherical as sp
import collisions
import collision_operator_spherical as collOpSp
import matplotlib.pyplot as plt
from itertools import cycle
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = cycle(prop_cycle.by_key()['color'])

data               = plot_utils.load_data_bte("../../../../bte-docs/1dglow/Ar_3_species_100V/", [1900], eedf_idx=None, read_cycle_avg=True, use_ionization=1)
spec_sp, coll_list = plot_utils.gen_spec_sp(data[0])
bte_op             = data[3]
mass_op = bte_op["mass"]
temp_op = bte_op["temp"] 
num_p   = spec_sp._p + 1
num_sh  = len(spec_sp._sph_harm_lm)
Nvt     = (int)(data[0]["Nvt"])

cross_section_data = cross_section.CROSS_SECTION_DATA
collision_op       = collOpSp.CollisionOpSP(spec_sp)
vth                = collisions.electron_thermal_velocity((float)(data[0]["Te"]) * (scipy.constants.elementary_charge / scipy.constants.Boltzmann))
tscale             = 1/(vth **2 * 0.5 * scipy.constants.electron_mass * (2/3/scipy.constants.Boltzmann) / collisions.TEMP_K_1EV)
maxwellian         = bte_utils.get_maxwellian_3d(vth, 1) 
# mm_mat             = spec_sp.compute_mass_matrix(vth)
# mm_mat_inv         = spec_sp.inverse_mass_mat(vth, Mmat=mm_mat)

# Cop_list           = list()
# for col_idx, (col_str, col_data) in enumerate(cross_section_data.items()):
#     g = coll_list[col_idx]
#     g.reset_scattering_direction_sp_mat()
#     col = g._col_name
#     print("collision %d  %s %s"%(col_idx, col, col_data["type"]))
        
#     # if col_data["type"] == "ELASTIC":
#     #     FOp_g     = collision_op.electron_gas_temperature(g, maxwellian, vth, mp_pool_sz=4)
#     Cop  = np.dot(mm_mat_inv, collision_op.assemble_mat(g, maxwellian, vth, tgK=0.0, mp_pool_sz=4))    
#     Cop_list.append(Cop)

v       = data[2][0]
v_lm    = np.dot(bte_op["po2sh"], v)
v1      = np.dot(bte_op["psh2o"], v_lm)

import glow1d_utils
param = glow1d_utils.parameters()
Emax  = 8e4
Cen   = bte_op["Cmat"]
Av    = bte_op["Emat"]
dt    = 5e-5

Jmat  = np.eye(Av.shape[0]) - dt * param.tau * (param.n0 * param.np0 * Cen + 8e4 * Av)
Jmat_inv = np.linalg.inv(Jmat)
w_lm     = np.dot(Jmat_inv, v_lm)
v2       = np.dot(bte_op["psh2o"], w_lm)

Jmat     = np.eye(Av.shape[0]) - dt * param.tau * (param.n0 * param.np0 * Cen + 1e3 * Av)
Jmat_inv = np.linalg.inv(Jmat)
w_lm     = np.dot(Jmat_inv, v_lm)
v3       = np.dot(bte_op["psh2o"], w_lm)

Jmat     = np.eye(Av.shape[0]) - dt * param.tau * (param.n0 * param.np0 * Cen + Av)
Jmat_inv = np.linalg.inv(Jmat)
w_lm     = np.dot(Jmat_inv, v_lm)
v4       = np.dot(bte_op["psh2o"], w_lm)

assert Nvt%2 == 0 
gx, gw             = basis.Legendre().Gauss_Pn(Nvt//2)
gx_m1_0 , gw_m1_0  = 0.5 * gx - 0.5, 0.5 * gw
gx_0_p1 , gw_0_p1  = 0.5 * gx + 0.5, 0.5 * gw
xp_vt              = np.append(np.arccos(gx_m1_0), np.arccos(gx_0_p1)) 
xp_vt_qw           = np.append(gw_m1_0, gw_0_p1)
xp                 = plot_utils.op((int)(data[0]["Np"])).xp

xp_idx_0           = list(range(0, 10))
xp_idx_1           = list(range(np.argmin(np.abs(xp + 0.875)) -5 ,  np.argmin(np.abs(xp + 0.875)) + 5))
xp_idx_2           = list(range(np.argmin(np.abs(xp-0)) -5 ,  np.argmin(np.abs(xp-0))))

plt.figure(figsize=(16,8), dpi=300)
plt.subplot(1, 2, 1)
for i in xp_idx_0:
    vt      =  v[:,i].reshape((num_p, Nvt))
    vt1     = v1[:,i].reshape((num_p, Nvt))
    vt2     = v2[:,i].reshape((num_p, Nvt))
    
    ne      = np.dot(mass_op[0::num_sh], vt)
    ne1     = np.dot(mass_op[0::num_sh], vt1)
    ne2     = np.dot(mass_op[0::num_sh], vt2)
    c       = next(colors)
    
    plt.plot(xp_vt, ne , '*--', label=r"x=%.4E"%(xp[i]), color=c)
    plt.plot(xp_vt, ne1, 'o-' ,                          color=c)
    #plt.plot(xp_vt, ne2, '+-' ,                          color=c)

plt.suptitle(r"*-ordinates, o- ordinates-> SPH -> ordinates, +- ordinates -> SPH -> (v-space solve $C_{en} + 8\times 10^4 A_v$) -> ordinates")    
plt.xlabel(r"$v_\theta$")
plt.ylabel(r"$\int_{v} v^2 f(v, v_\theta) dv$")
plt.legend()
plt.grid(visible=True)


plt.subplot(1, 2, 2)
for i in xp_idx_0:
    vt      =  v[:,i].reshape((num_p, Nvt))
    vt1     = v1[:,i].reshape((num_p, Nvt))
    vt2     = v2[:,i].reshape((num_p, Nvt))
    
    ne      = np.dot(mass_op[0::num_sh], vt)
    ne1     = np.dot(mass_op[0::num_sh], vt1)
    ne2     = np.dot(mass_op[0::num_sh], vt2)
    
    Te      = tscale * np.dot(temp_op[0::num_sh], vt)
    Te1     = tscale * np.dot(temp_op[0::num_sh], vt1)
    Te2     = tscale * np.dot(temp_op[0::num_sh], vt2)
    
    c       = next(colors)
    
    plt.plot(xp_vt, Te , '*--', label=r"x=%.4E"%(xp[i]), color=c)
    plt.plot(xp_vt, Te1, 'o-',                           color=c)
    #plt.plot(xp_vt, Te2, '+-',                           color=c)
    
plt.xlabel(r"$v_\theta$")
plt.ylabel(r"$\int_{v} v^4 f(v, v_\theta) dv$")
plt.legend()
plt.grid(visible=True)
plt.tight_layout()
plt.savefig("plt_wall.png")


plt.figure(figsize=(16,8), dpi=300)
plt.subplot(1, 2, 1)
for i in xp_idx_1:
    vt      =  v[:,i].reshape((num_p, Nvt))
    vt1     = v1[:,i].reshape((num_p, Nvt))
    vt3     = v3[:,i].reshape((num_p, Nvt))
    
    ne      = np.dot(mass_op[0::num_sh], vt)
    ne1     = np.dot(mass_op[0::num_sh], vt1)
    ne3     = np.dot(mass_op[0::num_sh], vt3)
    c       = next(colors)
    
    plt.plot(xp_vt, ne , '*--', label=r"x=%.4E"%(xp[i]), color=c)
    plt.plot(xp_vt, ne1, 'o-' ,                          color=c)
    #plt.plot(xp_vt, ne3, '+-' ,                          color=c)

plt.suptitle(r"*-ordinates, o- ordinates-> SPH -> ordinates, +- ordinates -> SPH -> (v-space solve $C_{en} + 8\times 10^4 A_v$) -> ordinates")    
plt.xlabel(r"$v_\theta$")
plt.ylabel(r"$\int_{v} v^2 f(v, v_\theta) dv$")
plt.legend()
plt.grid(visible=True)
plt.subplot(1, 2, 2)
for i in xp_idx_1:
    vt      =  v[:,i].reshape((num_p, Nvt))
    vt1     = v1[:,i].reshape((num_p, Nvt))
    vt3     = v3[:,i].reshape((num_p, Nvt))
    
    ne      = np.dot(mass_op[0::num_sh], vt)
    ne1     = np.dot(mass_op[0::num_sh], vt1)
    ne3     = np.dot(mass_op[0::num_sh], vt3)
    
    Te      = tscale * np.dot(temp_op[0::num_sh], vt)
    Te1     = tscale * np.dot(temp_op[0::num_sh], vt1)
    Te3     = tscale * np.dot(temp_op[0::num_sh], vt3)
    
    c       = next(colors)
    
    plt.plot(xp_vt, Te , '*--', label=r"x=%.4E"%(xp[i]), color=c)
    plt.plot(xp_vt, Te1, 'o-',                           color=c)
    #plt.plot(xp_vt, Te3, '+-',                           color=c)
    
plt.xlabel(r"$v_\theta$")
plt.ylabel(r"$\int_{v} v^4 f(v, v_\theta) dv$")
plt.legend()
plt.grid(visible=True)
plt.tight_layout()
plt.savefig("plt_sheath.png")

plt.figure(figsize=(16,8), dpi=300)
plt.subplot(1, 2, 1)
for i in xp_idx_2:
    vt      =  v[:,i].reshape((num_p, Nvt))
    vt1     = v1[:,i].reshape((num_p, Nvt))
    vt4     = v4[:,i].reshape((num_p, Nvt))
    
    ne      = np.dot(mass_op[0::num_sh], vt)
    ne1     = np.dot(mass_op[0::num_sh], vt1)
    ne4     = np.dot(mass_op[0::num_sh], vt4)
    
    c       = next(colors)
    
    plt.plot(xp_vt, ne , '*--', label=r"x=%.4E"%(xp[i]), color=c)
    plt.plot(xp_vt, ne1, 'o-',                          color=c)
    #plt.plot(xp_vt, ne4, '+-',                          color=c)
    
plt.xlabel(r"$v_\theta$")
plt.ylabel(r"$\int_{v} v^2 f(v, v_\theta) dv$")
plt.legend()
plt.grid(visible=True)

plt.subplot(1, 2, 2)
for i in xp_idx_2:
    vt      =  v[:,i].reshape((num_p, Nvt))
    vt1     = v1[:,i].reshape((num_p, Nvt))
    vt4     = v4[:,i].reshape((num_p, Nvt))
    
    ne      = np.dot(mass_op[0::num_sh], vt)
    ne1     = np.dot(mass_op[0::num_sh], vt1)
    ne4     = np.dot(mass_op[0::num_sh], vt4)
    
    Te      = tscale * np.dot(temp_op[0::num_sh], vt)
    Te1     = tscale * np.dot(temp_op[0::num_sh], vt1)
    Te4     = tscale * np.dot(temp_op[0::num_sh], vt4)
    
    c       = next(colors)
    
    plt.plot(xp_vt, Te , '*--', label=r"x=%.4E"%(xp[i]), color=c)
    plt.plot(xp_vt, Te1, 'o-',                           color=c)
    #plt.plot(xp_vt, Te4, '+-',                           color=c)
    
plt.xlabel(r"$v_\theta$")
plt.ylabel(r"$\int_{v} v^4 f(v, v_\theta) dv$")
plt.legend()
plt.grid(visible=True)
plt.suptitle(r"*-ordinates, o- ordinates-> SPH -> ordinates, +- ordinates -> SPH -> (v-space solve $C_{en} + 1 A_v$) -> ordinates")    
plt.tight_layout()
plt.savefig("plt_center.png")
plt.close()
    


    
    
    
