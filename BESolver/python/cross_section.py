"""
@package To build efficient interpolation methods to deal with the experimental cross section data. 
"""

import numpy as np
import typing as tp
# this is the read lxCat data file.
import lxcat_data_parser as ldp
from scipy import interpolate
import sys

def lxcat_cross_section_to_numpy(file : str, column_fields : tp.List[str] )->list:
    try:
        data    = ldp.CrossSectionSet(file)
    except:
        print("Error while cross section file read")
        sys.exit(0)
    
    np_data = list()
    for f in column_fields:
        np_data.append(data.cross_sections[0].data[f].to_numpy())
    
    return np_data

def read_cross_section_data(file: str):
    try:
        data = ldp.CrossSectionSet(file)
    except:
        print("Error while cross section file read")
        sys.exit(0)

    species = data.species
    print("read species: ", species)
    print("number of cross sections read: ", len(data.cross_sections))
    print(data.cross_sections)
    
    cs_dict = dict()
    for i in range(len(data.cross_sections)):
        process          = data.cross_sections[i].info["PROCESS"].split(",")[0]
        energy           = np.array(data.cross_sections[i].data["energy"])
        cross_section    = np.array(data.cross_sections[i].data["cross section"])
        threshold        = 0 
        if data.cross_sections[i].threshold !=None:
            threshold    = data.cross_sections[i].threshold
        
        mass_ratio       = data.cross_sections[i].mass_ratio
        # if data.cross_sections[i].mass_ratio !=None:
        #     mass_ratio   = data.cross_sections[i].mass_ratio 
        
        cs_dict[process] = {"energy": energy, "cross section": cross_section, "threshold": threshold, "mass ratio": mass_ratio}
    
    #print(cs_dict)
    return cs_dict

CROSS_SECTION_DATA = read_cross_section_data("lxcat_data/eAr_chung.txt")