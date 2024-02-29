"""
@package To build efficient interpolation methods to deal with the experimental cross section data. 
"""

import numpy as np
import typing as tp
# this is the read lxCat data file.
import lxcat_data_parser as ldp
from scipy import interpolate
import sys
import os
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

CROSS_SECTION_DATA = "" #read_cross_section_data(os.path.dirname(os.path.abspath(__file__)) + "/lxcat_data/eAr_crs.nominal.Biagi_minimal.txt")