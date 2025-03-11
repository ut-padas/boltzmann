"""
@package To build efficient interpolation methods to deal with the experimental cross section data. 
"""

import numpy as np
import typing as tp
# this is the read lxCat data file.
import lxcat_data_parser as ldp
from scipy import interpolate
import scipy.constants
import h5py
import sys
import os
import datetime

import scipy.interpolate
import scipy.optimize
def lxcat_cross_section_to_numpy(file : str, column_fields : tp.List[str] )->list:
    try:
        data    = ldp.CrossSectionSet(file)
    except Exception as e:
        print("Error while cross section file read: %s "%str(e))
        sys.exit(0)
    
    np_data = list()
    for f in column_fields:
        np_data.append(data.cross_sections[0].data[f].to_numpy())
    
    return np_data

def read_available_species(file:str):
    try:
        with open(file,'r') as f:
            species = [line.split(":")[1].split("/")[1].strip() for line in f if "SPECIES:" in line]
    except Exception as e:
        print("Error while cross section file read: %s "%str(e))
        sys.exit(0)
    
    species = list(sorted(set(species), key=species.index))
    return species

def read_cross_section_data(file: str):
    
    species = read_available_species(file)
    cs_dict = dict()
    
    for s in species:
        try:
            data = ldp.CrossSectionSet(file, imposed_species=s)
        except Exception as e:
            print("Error while cross section file read: %s "%str(e))
            sys.exit(0)
            
        #print("reading species: ", data.species)
        #print("number of cross sections read: ", len(data.cross_sections))
        #print(data.cross_sections)
    
        for i in range(len(data.cross_sections)):
            process          = data.cross_sections[i].info["PROCESS"].split(",")[0].strip()
            process_str      = data.cross_sections[i].info["PROCESS"].split(",")[1].strip().upper()
            energy           = np.array(data.cross_sections[i].data["energy"])
            cross_section    = np.array(data.cross_sections[i].data["cross section"])
            threshold        = data.cross_sections[i].threshold
            mass_ratio       = data.cross_sections[i].mass_ratio
            sp               = data.cross_sections[i].species
            cs_dict[process] = {"info": data.cross_sections[i].info, "type": process_str, "species": sp, "energy": energy, "cross section": cross_section, "threshold": threshold, "mass_ratio": mass_ratio, "raw": data.cross_sections[i]}
    
    return cs_dict

def compute_mw_reaction_rates(Te:np.array, cs_data:list, normalized_v_domain=(0, 30), num_vr_pts = 1000):
    """
    computes Maxwellian rates
    """
    def composite_trapz(qx):
        N   = len(qx)
        T   = (qx[-1]-qx[0])
        dx  = T/(N-1)

        assert abs(qx[1] -qx[0] - dx) < 1e-10
        qw    = np.ones_like(qx) * dx
        qw[0] = 0.5 * qw[0]; qw[-1] = 0.5 * qw[-1];
        assert (T-np.sum(qw)) < 1e-12
        return qx, qw
    
    kB         = scipy.constants.Boltzmann
    me         = scipy.constants.electron_mass
    qe         = scipy.constants.electron_volt
    ev_to_K    = (qe/kB)
    c_gamma    = np.sqrt(2 * qe / me)
    VTH        = lambda Te : np.sqrt(Te) * c_gamma
    
    vr, vr_w   = composite_trapz(np.linspace(normalized_v_domain[0], normalized_v_domain[1], num_vr_pts))
    vth        = VTH(Te)
    f0         = 4* np.pi * np.array([(1 / ((VTH(Te[i]) * np.sqrt(np.pi))**3)) * np.exp(-vr ** 2) for i in range(Te.shape[0])]).reshape((Te.shape[0], vr.shape[0]))
    
    mass       = (vth**3  * np.dot(vr**2 * f0 , vr_w)).reshape((-1, 1))
    f0_n       = f0/mass
    avg_energy = (0.5 * me ) * vth**5 * np.dot(vr**4 * f0_n , vr_w) 
    temp       = (2.0/3.0/kB) * avg_energy / (qe/kB)
    
    # additional check to see if temperature match with the current quad points. 
    assert np.linalg.norm(Te-temp)/np.linalg.norm(Te) < 1e-10
    
    rf         = np.zeros((Te.shape[0], len(cs_data)))
    for i in range(len(cs_data)):
        for j in range(Te.shape[0]):
            rf[j, i] = vth[j]**4 * np.dot(vr**3 * cs_data[i]( (vth[j] * vr/c_gamma)**2) * f0_n[j, :], vr_w)
            
    return rf

def append_recomb_cs(cs_in_file, cs_out_file, eqc_fname):
    """
    computes the pseudo cross-section for 3-body recombination collision
    - Assumes maxwellian EEDF for all species. 
    """
    import matplotlib.pyplot as plt
    
    cs_dict  = read_cross_section_data(cs_in_file)
    
    eqc_name = {
                "E + Ar -> E + E + Ar(+)":     "Ground",
                "E + Ar*(1) -> E + E + Ar(+)": "Metastable",
                "E + Ar*(2) -> E + E + Ar(+)": "Resonant",
                "E + Ar*(3) -> E + E + Ar(+)": "4p",
                }
    
    kB       = scipy.constants.Boltzmann
    me       = scipy.constants.electron_mass
    qe       = scipy.constants.electron_volt
    ev_to_K  = (qe/kB)
    c_gamma  = np.sqrt(2 * qe / me)
    VTH      = lambda Te : np.sqrt(Te) * c_gamma
    
    ff       = h5py.File(eqc_fname, "r")
    
    cs_data  = list()
    eqc      = list()
    process  = list()    
    for key, cs_metadata in cs_dict.items():
        if cs_metadata["type"] == "IONIZATION":
            process.append(key)
            eqc_data       = ff[eqc_name[key]][()]
            eqc_data[:, 0] = eqc_data[:, 0]/ev_to_K
            cs_data.append(scipy.interpolate.interp1d(cs_metadata["energy"], cs_metadata["cross section"], kind="linear", bounds_error=False, fill_value=0.0))
            eqc.append(scipy.interpolate.interp1d(eqc_data[:, 0], eqc_data[:, 1], kind="linear", bounds_error=True))

    #print(ff["Ground"][()][:, 0]/ev_to_K, (ff["Ground"][()][:, 0]/ev_to_K).shape)
    Te                    = ff["Ground"][()][:, 0]/ev_to_K
    Te                    = np.logspace(np.log10(0.2), np.log10(Te[-1]), 64, base=10)
    num_ev_intervals      = 256
    sigma_ev              = np.logspace(-8, 1, num_ev_intervals, base=10) 
    mvec                  = np.zeros((Te.shape[0], num_ev_intervals))
    
    vth = VTH(Te)
    
    rf = compute_mw_reaction_rates(Te, cs_data)
    a1 = np.zeros((len(Te), len(cs_data)))
    for i in range(len(cs_data)):
        a1[:, i] = eqc[i](Te)
    
    rb      = rf / a1 / scipy.constants.Avogadro
    
    # with h5py.File("lxcat_data/rates.h5", 'w') as F:
    #     F.create_dataset("Te[eV]", data=Te)
        
    #     gf = F.create_group("forward[m^3s^-1]")
    #     for i in range(rf.shape[1]):
    #         gf.create_dataset(eqc_name[process[i]], data=rf[:, i])
        
    #     gf = F.create_group("backward[m^6s^-1]")
    #     for i in range(rf.shape[1]):
    #         gf.create_dataset(eqc_name[process[i]], data=rb[:, i])
            
    #     F.close()
            
    
    # for i in range(len(process)):
    #     #plt.loglog(Te, rf[:, i], label=r"%s (forward)"%process[i])
    #     plt.loglog(Te, rb[:, i], '--', label=r"%s (reverse)"%process[i])
    #     #plt.loglog(Te, eqc[i](Te), label="%d"%i)
    
    # plt.xlabel(r"electron temperature (eV)")
    # plt.grid(visible=True)
    # plt.legend()
    # plt.show()
    # plt.close()
    
    crs_idx = 0 
    crs_rev = dict()
    
    crs_idx=0
    
    Imat    = np.eye(sigma_ev.shape[0]) 
    maxiter = 3000
    atol    = 1e-200
    rtol    = 1e-4
    
    
    plt.figure(figsize=(12, 4), dpi=200)
    
    for key, cs_metadata in cs_dict.items():
        if cs_metadata["type"] == "IONIZATION":
            ev_l   = sigma_ev
            vr     = np.array([np.sqrt(ev_l) * c_gamma / vth[i] for i in range(vth.shape[0])]).reshape((vth.shape[0], -1))
            vr_w   = np.zeros_like(vr)
        
            vr_w[:, 1:-1] = np.array([ 0.5 * (vr[:, i+1] - vr[:, i-1]) for i in range(1, vr.shape[1]-1)]).T
        
            vr_w[:,  0]   = (vr[:,  1]  - vr[:,  0]) * 0.5
            vr_w[:, -1]   = (vr[:, -1]  - vr[:, -2]) * 0.5 
        
            assert (np.abs(np.sum(vr_w, axis=1) - (vr[:, -1]-vr[:, 0])) < 1e-12).all()
        
            f0        = 4* np.pi * np.array([(1 / ((np.sqrt(np.pi))**3)) * np.exp(-vr[i] ** 2) for i in range(Te.shape[0])]).reshape((Te.shape[0], vr.shape[1]))
            mass      = np.array([np.dot(vr_w[i], vr[i]**2 * f0[i]) for i in range(Te.shape[0])])
            Qmat      = f0 * vr_w * vr**3 * vth.reshape((-1, 1))
        
            
            # solve a non-linear least squares problem
            u             = np.zeros_like(sigma_ev)
            norm_b        = np.linalg.norm(rb[:, crs_idx])
            
            def res(x):
                return (np.dot(Qmat, np.exp(x)) - rb[:, crs_idx])#/norm_b
            
            iter          = 0
            r0            = np.linalg.norm(res(u))
            abs_e , rel_e = r0, r0/norm_b
            converged     = rel_e < rtol 
            u1            = np.copy(u)
            
            while((iter < maxiter) and not converged):
                
                if (iter > 0):
                    u1 = np.copy(u)
                
                alpha      = 1e0
                rr         = res(u)
                norm_rr    = np.linalg.norm(rr)
                
                abs_e      = norm_rr
                rel_e      = norm_rr/norm_b
                
                if (iter%100 == 0):
                    print("[NLSL] iter = %06d ||F(u)-b|| = %.10E ||F(u)-b||/||b||= %.10E "%(iter, abs_e, rel_e))
                
                Jmat       = np.dot(Qmat, np.exp(u) * Imat)            
                JTJ        = np.dot(Jmat.T, Jmat)
                JTJ_inv    = np.linalg.pinv(JTJ, rcond=1e-6)
            
                du         = -np.dot(JTJ_inv, np.dot(Jmat.T, rr))
                
                while (alpha> 1e-15):
                    if (np.linalg.norm(res(u + alpha * du)) > norm_rr):
                        alpha = alpha * 0.1
                    else:
                        break
                
                if (alpha < 1e-8):
                    print("[NLLS] : Line search failed")
                    break
                
                u          = u + du * alpha
                
                rr         = res(u)
                norm_rr    = np.linalg.norm(rr)
                abs_e      = norm_rr
                rel_e      = norm_rr/norm_b
                converged  = rel_e < rtol 
                rr_u       = np.linalg.norm(u1-u)/np.linalg.norm(u)
                if  rr_u < 1e-15:
                    converged=True
                
                if (converged):
                    print("[NLLS] : Converged at iter = %06d atol = %.10E rtol= %.10E ||u1-u0||/||u0|| = %.10E "%(iter, abs_e, rel_e, rr_u))
                    break
                
                
                iter+=1
                
            sigma_rev  = np.exp(u)
            a1         = np.dot(Qmat, sigma_rev)
            nlls_rel_e = np.linalg.norm(a1-rb[:, crs_idx])/np.linalg.norm(rb[:, crs_idx])
            print("process : %s NLLS fit relative error = %.4E"%(key, nlls_rel_e))
            
            rev_process = key.split("->")[1] + " -> " + key.split("->")[0]
            
            plt.subplot(121)
            #plt.loglog(Te, rf[:, crs_idx], '.', markersize=2 , label=r"$k_f$ (%s)"%(key))
            plt.loglog(Te, rb[:, crs_idx], 'x', markersize=2 , label=r"$k_b = k_f C_{eq}$")
            plt.loglog(Te, a1            , 'o', markersize=2 , label=r"%s(NLLS fit)"%(rev_process))
            plt.xlabel(r"electron temperature (eV)")
            plt.ylabel(r"rate coefficient [$m^6s^{-1}$]")
            plt.grid(visible=True)
            plt.legend(fontsize=4)
            
            plt.subplot(122)
            #plt.loglog(sigma_ev, cs_data[crs_idx] (sigma_ev), '.',  markersize=2, label=r"%s "%(key))
            plt.loglog(sigma_ev, sigma_rev                  , 'x',  markersize=2, label=r"%s "%(rev_process))
            plt.grid(visible=True)
            plt.xlabel(r"energy (eV)")
            plt.ylabel(r"reference cross-section [$m^2$]")
            plt.legend(fontsize=6)
            
            crs_rev[key]=(sigma_ev, sigma_rev)
            crs_idx+=1
    
    plt.tight_layout()
    #plt.show()
    plt.savefig("cs_reverse.png")
    plt.close()
            
    
    with open(cs_out_file, "w") as f:
        for key, cs_metadata in cs_dict.items():
            f.write(cs_metadata["type"] + "\n")
            f.write(cs_metadata["species"] + "\n")
            if cs_metadata["mass_ratio"]==None:
                f.write("%.6E"%cs_metadata["threshold"] + "\n")
            else:
                f.write("%.6E"%cs_metadata["mass_ratio"] + "\n")
            
            info = cs_metadata["info"]
            for k,v in info.items():
                f.write("%s: %s"%(str(k),str(v)) + "\n")
            f.write("------------------------------\n")    
            
            ev_grid  = cs_metadata["energy"]
            cs_data  = cs_metadata["cross section"]
            
            data_str = "\n".join(["%.6E\t%.6E"%(ev_grid[idx], cs_data[idx]) for idx in range(len(ev_grid))])
            f.write(data_str+"\n")
            f.write("------------------------------\n")

        for key, rev_cs in crs_rev.items():
            ctype   = "ATTACHMENT"
            sp      = key.split("->")[1].split(" ")[-1]
            process = (key.split("->")[1] + " -> " + key.split("->")[0]).strip()
            f.write("%s"%(ctype) + "\n")
            f.write(sp + "\n")
            #f.write("%.6E"%(cs_dict[key]["threshold"]) + "\n")
            f.write("SPECIES: e/%s"%(sp)+"\n")
            f.write("PROCESS: %s, %s"%(process, ctype)+"\n")
            f.write("FORWARD: %.6E"%(cs_dict[key]["threshold"]) + "\n")
            f.write("COMMENT: reverse cross-sections based on the detailed balance\n")
            f.write("UPDATED: %s\n"%(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))
            f.write("COLUMNS: Energy (eV) | Cross section (m2)" + "\n")
            f.write("------------------------------\n")
            ev_grid = rev_cs[0]
            cs_data = rev_cs[1]
            
            ev_grid = np.append(np.array([0]), ev_grid)
            cs_data = np.append(np.array([0]), cs_data)
            
            data_str = "\n".join(["%.6E\t%.6E"%(ev_grid[idx], cs_data[idx]) for idx in range(len(ev_grid))])
            f.write(data_str+"\n")
            f.write("------------------------------\n")
            
    
            
            
        
    
            
    
    #return crs_rev
            
CROSS_SECTION_DATA = "" #read_cross_section_data(os.path.dirname(os.path.abspath(__file__)) + "/lxcat_data/eAr_crs.nominal.Biagi_minimal.txt")