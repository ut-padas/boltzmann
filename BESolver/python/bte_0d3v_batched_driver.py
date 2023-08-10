import scipy
import scipy.optimize
import scipy.interpolate
import basis
import spec_spherical as sp
import numpy as np
import collision_operator_spherical as colOpSp
import collisions 
import parameters as params
from   time import perf_counter as time, sleep
import utils as BEUtils
import argparse
import scipy.integrate
from   scipy.integrate import ode
from   advection_operator_spherical_polys import *
import scipy.ndimage
import matplotlib.pyplot as plt
from   scipy.interpolate import interp1d
import scipy.sparse.linalg
from   datetime import datetime
from   bte_0d3v_batched import bte_0d3v_batched

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 24,
    #"ytick.major.size": 3,
    #"font.family": "Helvetica",
    "lines.linewidth":2.0
})


parser = argparse.ArgumentParser()
parser.add_argument("-threads", "--threads"                       , help="number of cpu threads", type=int, default=4)
parser.add_argument("-l_max", "--l_max"                           , help="max polar modes in SH expansion", type=int, default=1)
parser.add_argument("-c", "--collisions"                          , help="collisions model",nargs='+', type=str, default=["g0","g2"])
parser.add_argument("-sp_order", "--sp_order"                     , help="b-spline order", type=int, default=3)
parser.add_argument("-spline_qpts", "--spline_qpts"               , help="q points per knots", type=int, default=5)
parser.add_argument("-steady", "--steady_state"                   , help="steady state or transient", type=int, default=1)
parser.add_argument("-Tg", "--Tg"                                 , help="gas temperature (K)" , type=float, default=1e-12)
parser.add_argument("-n0", "--n0"                                 , help="heavy density (1/m^3)" , type=float, default=3.22e22)
parser.add_argument("-ion_deg", "--ion_deg"                       , help="Ionization degree"   , type=float, nargs='+', default=[1e-1])
parser.add_argument("-store_eedf", "--store_eedf"                 , help="store EEDF"          , type=int, default=0)
parser.add_argument("-store_csv", "--store_csv"                   , help="store csv format of QoI comparisons", type=int, default=0)
parser.add_argument("-ee_collisions", "--ee_collisions"           , help="enable electron-electron collisions", type=float, default=0)
parser.add_argument("-verbose", "--verbose"                       , help="verbose with debug information", type=int, default=0)

args        = parser.parse_args()
batch_size  = 10

ef          = np.ones(batch_size) * 100
n0          = np.ones(batch_size) * 3.22e22
ni          = np.ones(batch_size) * 3.22e18
Te          = np.ones(batch_size) * 3 * collisions.TEMP_K_1EV
lm          = [(l,0) for l in range(args.l_max)]
nr          = np.ones(batch_size, dtype=np.int32) * 128

bte_batached = bte_0d3v_batched(args, ef, n0, ni, Te, nr, lm, batch_size, args.collisions)






