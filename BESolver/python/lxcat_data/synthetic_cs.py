import numpy as np
import sys
sys.path.append("../.")
import collisions
import cross_section
import lxcat_data as ldp
import datetime

cs_input  = "eAr_crs.Biagi.3sp2r"
cs_output = "eAr_crs.synthetic.3sp2r"


avail_species       = cross_section.read_available_species(cs_input)
cross_section_data  = cross_section.read_cross_section_data(cs_input)

with open(cs_output, "w") as f:
    for key, cs_data in cross_section_data.items():
        cs_data["cross section"] = collisions.Collisions.synthetic_tcs(cs_data["energy"], key)
        
        f.write("%s\n"%(cs_data["type"]))
        f.write("%s\n"%(cs_data["species"]))
        if cs_data["mass_ratio"] == None:
            f.write("%s\n"%(cs_data["threshold"]))
        else:
            f.write("%s\n"%(cs_data["mass_ratio"]))
            
        f.write("SPECIES: e / %s\n"%(cs_data["species"]))
        f.write("PROCESS: %s, %s\n"%(key,cs_data["type"].capitalize()))
        if cs_data["mass_ratio"] == None:
            f.write("PARAM.:  E = %.8E, complete set\n"%(cs_data["threshold"]))
        else:
            f.write("PARAM.:  m/M = %.8E, complete set\n"%(cs_data["mass_ratio"]))
            
        f.write("COMMENT: synthetic cross-sections\n")
        f.write("UPDATED: %s\n"%(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))
        f.write("COLUMNS: Energy (eV) | Cross section (m2)\n")
        f.write("------------------------------\n")
        
        ss_data = "\n".join(["%.6E\t%6E"%(cs_data["energy"][i], cs_data["cross section"][i]) for i in range(len(cs_data["energy"]))])
        f.write("%s\n"%(ss_data))
        f.write("------------------------------\n")
        f.write("\n")
        
        
        
        
    
    





