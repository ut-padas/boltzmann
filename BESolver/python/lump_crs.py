"""
lump crs for 6sp
"""
import cross_section
import collisions
import basis
import matplotlib.pyplot as plt
import utils
import numpy as np
import scipy.interpolate
import scipy.constants
from datetime import datetime
cs_species                       = cross_section.read_available_species("lxcat_data/eAr_crs.15sp106r")
crs_data                         = cross_section.read_cross_section_data("lxcat_data/eAr_crs.15sp106r")
cross_section.CROSS_SECTION_DATA = crs_data

Tg                               = 0.8 #eV
out_cs_fname                     = "lxcat_data/eAr_crs.6sp_Tg_0.8eV"

mass_ar    = scipy.constants.electron_mass / crs_data["E + Ar -> E + Ar"]["mass_ratio"]
c_gamma_Ar = np.sqrt(2 * scipy.constants.elementary_charge / mass_ar)
vth        = np.sqrt(Tg) * c_gamma_Ar

def boltzmann_distribution(Tg):
    return lambda x : np.exp(-x/Tg)

mw         = boltzmann_distribution(Tg)#utils.get_maxwellian_3d(vth=vth, n_scale=1)

lump_mechanism = {
            
            "E + Ar -> E + Ar"     : ["E + Ar -> E + Ar"],
            
            "E + Ar -> E + Ar*(1)" : ["E + Ar -> E + Ar(1S5)", "E + Ar -> E + Ar(1S3)"],
            
            "E + Ar -> E + Ar*(2)" : ["E + Ar -> E + Ar(1S4)", "E + Ar -> E + Ar(1S2)"],
            
            "E + Ar -> E + Ar*(3)" : ["E + Ar -> E + Ar(2P10)", 
                                      "E + Ar -> E + Ar(2P9)",
                                      "E + Ar -> E + Ar(2P8)",
                                      "E + Ar -> E + Ar(2P7)",
                                      "E + Ar -> E + Ar(2P6)",
                                      "E + Ar -> E + Ar(2P5)",
                                      "E + Ar -> E + Ar(2P4)",
                                      "E + Ar -> E + Ar(2P3)",
                                      "E + Ar -> E + Ar(2P2)",
                                      "E + Ar -> E + Ar(2P1)"], 
            
            "E + Ar -> E + E + Ar(+)"     : ["E + Ar -> E + E + Ar(+)"],
            
            "E + Ar*(1) -> E + E + Ar(+)" : ["E + Ar5 -> E + E + Ar(+)", "E + Ar3 -> E + E + Ar(+)"],
            
            "E + Ar*(2) -> E + E + Ar(+)" : ["E + Ar4 -> E + E + Ar(+)", "E + Ar2 -> E + E + Ar(+)"], 
            
            "E + Ar*(3) -> E + E + Ar(+)" : ["E + Ar(2p10) -> E + E + Ar(+)", 
                                            "E + Ar(2p9) -> E + E + Ar(+)", 
                                            "E + Ar(2p8) -> E + E + Ar(+)", 
                                            "E + Ar(2p7) -> E + E + Ar(+)", 
                                            "E + Ar(2p6) -> E + E + Ar(+)", 
                                            "E + Ar(2p5) -> E + E + Ar(+)", 
                                            "E + Ar(2p4) -> E + E + Ar(+)", 
                                            "E + Ar(2p3) -> E + E + Ar(+)", 
                                            "E + Ar(2p2) -> E + E + Ar(+)", 
                                            "E + Ar(2p1) -> E + E + Ar(+)"]
            }

species_energy = {"Ar": 0,  "Ar*(1)":11.54835 , "Ar5":11.54835 , "Ar4":11.62359, "Ar*(2)":11.72316, "Ar3":11.72316, "Ar2":11.82807, "Ar*(3)":12.907, "Ar(2p10)":12.907, "Ar(2p9)":13.076, "Ar(2p8)":13.095, "Ar(2p7)":13.153, "Ar(2p6)":13.172, "Ar(2p5)":13.273, "Ar(2p4)":13.283, "Ar(2p3)":13.302, "Ar(2p2)":13.328, "Ar(2p1)":13.48}
num_cs_pts  = 256
for idx, (k, lump_r) in enumerate(lump_mechanism.items()):
    cs_type    = list()
    ev_min     = list()
    ev_max     = list()
    
    cs_interp  = list()
    
    lump_sp    = k.split("+")[1].strip()
    
    cs_weight = np.array([mw(species_energy[crs_data[r]["species"]]) for r in lump_r])
    #cs_weight = np.array([mw(np.sqrt(species_energy[crs_data[r]["species"]]) * c_gamma_Ar/vth) for r in lump_r])
    #cs_weight = np.array([(np.sqrt(species_energy[crs_data[r]["species"]]) * c_gamma_Ar/vth) for r in lump_r])
    cs_weight = cs_weight/np.sum(cs_weight)
    
    assert np.allclose(np.sum(cs_weight),1) == True, "lump weights does not add to one"
    
    for r in lump_r:
        cs = crs_data[r]
        cs_type.append(cs["type"])
        ev_min.append(cs["energy"][0])
        ev_max.append(cs["energy"][-1])
        cs_interp.append(scipy.interpolate.interp1d(cs["energy"], cs["cross section"], kind="linear", bounds_error=False, fill_value=(cs["cross section"][0], cs["cross section"][-1])))
    
    assert len(list(set(cs_type))) == 1, "Error attempt to lump unmatched reaction mechanism %s"%str(cs_type)
        
    ev_min = np.array(ev_min)
    ev_max = np.array(ev_max)
    
    ev_domain = (max(np.min(ev_min), 1e-4), np.min(ev_max))
    print("lumped reaction %s lumping input %s \n"%(k, str(lump_r)), cs_weight)
    #print(ev_min)
    
    if ev_domain[0]== 1e-4:
        ev_grid   = np.append(np.array([0]), np.logspace(np.log10(ev_domain[0]), np.log10(ev_domain[1]), num_cs_pts-1, base=10))
    else:
        ev_grid   = np.logspace(np.log10(ev_domain[0]), np.log10(ev_domain[1]), num_cs_pts, base=10)
    
    cs_eff    = np.array([cs_interp[r_idx](ev_grid) for r_idx, r in enumerate(lump_r)]).reshape((len(lump_r), len(ev_grid)))
    cs_eff    = np.dot(cs_eff.T, cs_weight).reshape((-1))
    
    open_mode = "w" if idx==0 else "a"
    with open(out_cs_fname, open_mode) as f:
        g_type      = cs_type[0].upper()
        g_species   = k.split("->")[0].split("+")[1].strip()
        g_threshold = ev_domain[0]
        f.write(g_type + "\n")
        f.write(g_species + "\n")
        if (g_type == "ELASTIC"):
            assert(len(lump_r)==1), "elastic collisions cannot be lumped"
            g_mbyM      =  crs_data[lump_r[0]]["mass_ratio"]
            f.write("%.6E"%(g_mbyM) + "\n")
        else:
            f.write("%.6E"%(g_threshold) + "\n")
        
        f.write("SPECIES: %s/%s\n"%("e", g_species))
        f.write("PROCESS: %s, %s\n"%(k,g_type))
        if (g_type == "ELASTIC"):
            f.write("PARAM.: m/M = %.6E, complete set\n"%(g_mbyM))
        
        f.write("COMMENT: lumped crs at Maxwellian for Ar with %.2E eV\n"%(Tg))
        f.write("UPDATED: %s\n"%(datetime.now().strftime("%m/%d/%Y, %H:%M:%S")))
        f.write("COLUMNS: Energy (eV) | Cross section (m2)\n")
        f.write("------------------------------\n")
        data_str = "\n".join(["%.6E\t%.6E"%(ev_grid[idx], cs_eff[idx]) for idx in range(len(ev_grid))])
        f.write(data_str+"\n")
        f.write("------------------------------\n")
        
        
            
            
        
    plt.figure(figsize=(8, 8), dpi=200)
    plt.loglog(ev_grid, cs_eff, '--', label=r"effective")
    for r in lump_r:
        cs = crs_data[r]
        plt.loglog(cs["energy"], cs["cross section"], label=r"%s"%(r), alpha=0.3)
    
    plt.grid(visible=True)
    plt.legend()
    plt.xlabel(r"energy (eV)")
    plt.ylabel(r"cross section (m^2)")
    plt.savefig("lxcat_data/%s.png"%(k))
    plt.close()
    
    
        
    
    
    
    


     

# e1  = 11.54835
# e2  = 11.62359

# for te in Te:
#     vth = (te)**0.5 * c_gamma
#     mw  = utils.get_maxwellian_3d(vth, 1)
    
#     a1  = mw(np.sqrt(e1) * c_gamma/vth) 
#     a2  = mw(np.sqrt(e2) * c_gamma/vth) 

#     print("w1 = %.6E w2=%.6E"%(a1/(a1+a2), a2/(a1+a1)))

# coll_list  = list()
# for col_str, col_data in crs_data.items():
#     print(col_str, col_data["type"])
#     g = collisions.electron_heavy_binary_collision(col_str, collision_type=col_data["type"])
#     coll_list.append(g)

# def expand_basis_cs(g: collisions.electron_heavy_binary_collision):
#     sp_order        = 3
#     num_p           = 256
#     kd_threshold    = 1e-8
#     k_domain        = (0, 1e4) 
#     k_vec           = basis.BSpline.logspace_knots(k_domain, num_p, sp_order, 1e-4 , base=2)
#     bb              = basis.BSpline(k_domain, sp_order, num_p, sig_pts=None, knots_vec=k_vec, dg_splines=0, verbose=False)
    

#     num_intervals     = bb._num_knot_intervals
#     q_pts             = (2 * sp_order + 1) * 2
#     gx, gw            = basis.Legendre().Gauss_Pn(q_pts)
    
#     total_cs_interp1d = scipy.interpolate.interp1d(g._energy, g._total_cs, kind='linear', bounds_error=False,fill_value=(g._total_cs[0],g._total_cs[-1]))

#     mm_mat    = np.zeros((num_p, num_p))  
#     b_rhs     = np.zeros(num_p)
#     for p in range(num_p):
#         k_min   = bb._t[p]
#         k_max   = bb._t[p + sp_order + 1]

#         gmx     = 0.5 * (k_max-k_min) * gx + 0.5 * (k_min + k_max)
#         gmw     = 0.5 * (k_max-k_min) * gw
#         b_p     = bb.Pn(p)(gmx, 0)
#         b_rhs[p] = np.dot(gmw, b_p * total_cs_interp1d(gmx))
#         for k in range(max(0, p - (sp_order+3) ), min(num_p, p + (sp_order+3))):
#             b_k          = bb.Pn(k)(gmx, 0)
#             mm_mat[p,k]  = np.dot(gmw, b_p * b_k)

#     def schur_inv(M):
#         rtol=1e-14
#         atol=1e-14

#         T, Q = scipy.linalg.schur(M)
#         Tinv = scipy.linalg.solve_triangular(T, np.identity(M.shape[0]),lower=False)
#         #print("spline cross-section fit mass mat inverse = %.6E "%(np.linalg.norm(np.matmul(T,Tinv)-np.eye(T.shape[0]))/np.linalg.norm(np.eye(T.shape[0]))))
#         return np.matmul(np.linalg.inv(np.transpose(Q)), np.matmul(Tinv, np.linalg.inv(Q)))

#     mm_inv = schur_inv(mm_mat)
#     sigma_k = np.dot(mm_inv, b_rhs)
#     bb      = bb
    
#     return bb, sigma_k

# Te  = list(np.linspace(0.1, 7, 10))

# e1  = 11.54835
# e2  = 11.62359

# for te in Te:
#     vth = (te)**0.5 * c_gamma
#     mw  = utils.get_maxwellian_3d(vth, 1)
    
#     a1  = mw(np.sqrt(e1) * c_gamma/vth) 
#     a2  = mw(np.sqrt(e2) * c_gamma/vth) 

#     print("w1 = %.6E w2=%.6E"%(a1/(a1+a2), a2/(a1+a1)))
    






# print("all collisions ", len(crs_data.items()))
# print("all background species ", cs_species, len(cs_species))
# for s in cs_species:
#     print("species ", s)
#     for k, v in crs_data.items():
#         if v["species"]==s:
#             print("\t", k, v["threshold"])
        
