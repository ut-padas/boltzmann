import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy.constants
import os
import scipy.interpolate

import sys
sys.path.append("../.")
sys.path.append("../plot_scripts")
import basis
import cross_section
import utils as bte_utils
import spec_spherical as sp
import collisions
import plot_utils

def load_run_args(fname):
    args   = dict()
    
    f  = open(fname)
    st = f.read().strip()
    st = st.split(":")[1].strip()
    st = st.split("(")[1]
    st = st.split(")")[0].strip()
    st = st.split(",")

    for s in st:
        kv = s.split("=")
        if len(kv)!=2 or kv[0].strip()=="collisions":
            continue
        args[kv[0].strip()]=kv[1].strip().replace("'", "")
    return args


