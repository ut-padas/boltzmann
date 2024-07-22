import argparse
import bolsig
import cross_section

parser = argparse.ArgumentParser()
parser.add_argument("-o",  "--out_fname"           , help="output file name for the qois", type=str, default="Ar_o_")
parser.add_argument("-i",  "--input_fname"         , help="output file name for the qois", type=str, default="Ar.dat")
parser.add_argument("-c",  "--collisions"          , help="collisions model", type=str, default="lxcat_data/eAr_crs.Biagi.3sp2r")
parser.add_argument("-T0", "--T0"                  , help="ground state temperature (K)", type=float, default=300)
parser.add_argument("-P0", "--P0"                  , help="ground state pressure (Torr)", type=float, default=1)
parser.add_argument("-ion_deg",        "--ion_deg" , help="ionization degree", type=float, default=0)
parser.add_argument("-run_series", "--run_series" , help="run series, E/N (min) E/N (max), num points grid type (1-linear, 2-quadratric, 3-exponential)", nargs='+', type=float, default=[1e-2, 1e2, 100, 1])
args = parser.parse_args()


cross_section.CROSS_SECTION_DATA = cross_section.read_cross_section_data(args.collisions)
bolsig.run_seris(args)