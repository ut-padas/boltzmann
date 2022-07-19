"""
@package Boltzmann collision operator solver. 
"""

from cProfile import run
from cmath import sqrt
from dataclasses import replace
import enum
from math import ceil
import string
import scipy
import scipy.optimize
import scipy.interpolate
from sympy import rad
from maxpoly import maxpolyserieseval
import basis
import spec_spherical as sp
import numpy as np
import collision_operator_spherical as colOpSp
import collisions 
import parameters as params
import os
from time import perf_counter as time
import utils as BEUtils
import argparse
import scipy.integrate
from scipy.integrate import ode
from advection_operator_spherical_polys import *
import scipy.ndimage
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# from adaptive import Runner, Learner1D
# class ALearner1D:
#     def setup(self,x, y, bounds):
#         self.func1d  = scipy.interpolate.interp1d(x,y)
#         self.learner = Learner1D(self.func1d, bounds=bounds)
#         self.learner.tell_many(x, map(self.func1d, x))

#     def get_pts(self, num_points):
#         points, _ = self.learner.ask(num_points)
#         return np.sort(np.array(points))

def deriv_fd(x, y):
    mid = (y[1:len(x)]-y[0:len(x)-1])/(x[1:len(x)]-x[0:len(x)-1])
    d = np.zeros_like(x)
    d[0] = mid[0]
    d[-1] = mid[-1]
    d[1:len(x)-1] = .5*(mid[1:len(x)-1]+mid[0:len(x)-2])
    return d

def parse_bolsig(file, num_collisions):
    eveclist = []
    dveclist = []
    aveclist = []
    Elist = []
    mulist = []
    mobillist = []
    difflist = []
    ratelist = []

    rate = np.zeros(2)

    with open(file, 'r') as F:
        line = F.readline()
        
        while (line!='Energy (eV) EEDF (eV-3/2) Anisotropy\n'):

            if (len(line)>=23):
                if (line[0:23]=='Electric field / N (Td)'):
                    EbN = float(line.split(")",2)[1]) # 1e-21 converts from Td to V*m^2
                    print("Found EbN = ", EbN)
                    Elist.append(EbN)

            if (len(line)>=16):
                if (line[0:16]=='Mean energy (eV)'):
                    mu = float(line.split(")",2)[1]) # while it says eV, the unit here is actually V
                    print("Found mu = ", mu)
                    mulist.append(mu)

            if (len(line)>=21):
                if (line[0:21]=='Mobility *N (1/m/V/s)'):
                    mobil = float(line.split(")",2)[1]) 
                    print("Found mobility = ", mobil)
                    mobillist.append(mobil)

            if (len(line)>=32):
                if (line[0:32]=='Diffusion coefficient *N (1/m/s)'):
                    diff = float(line.split(")",2)[1]) 
                    print("Found diffusion = ", diff)
                    difflist.append(diff)

            for i in range(num_collisions):
                if (len(line)>=8):
                    if (line[0:8]=='C'+str(i+1)+'    Ar'):
                        rate[i] = float(line.split()[-1]) 
                        print("Found collision rate no. "+str(i+1)+" = ", rate[i])
                        ratelist.append(rate)

            line = F.readline()

        # while (line!=''):
        line = F.readline()
        elist = []
        dlist = []
        alist = []
        while (line!=' \n'):
            col = line.split()
            energy = float(col[0])
            distrib = float(col[1])
            anisotropy = float(col[2])

            elist.append(energy)
            dlist.append(distrib)
            alist.append(anisotropy)

            line = F.readline()

        print("Adding vectors to lists for Etil = {0:.3e}...".format(EbN))
        eveclist.append(np.array(elist))
        dveclist.append(np.array(dlist))
        aveclist.append(np.array(alist))

    return [eveclist[0], dveclist[0], aveclist[0], Elist[0], mulist[0], mobillist[0], difflist[0], ratelist[0]]

def replace_line(file_name, line_num, text):
    lines = open(file_name, 'r').readlines()
    lines[line_num] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()

def run_bolsig(args):
    """
    run the bolsig code. 
    """
    bolsig_cs_file = args.collisions[0]
    for col in args.collisions[1:]:
        bolsig_cs_file = bolsig_cs_file + "_" + col
    bolsig_cs_file = bolsig_cs_file + ".txt"

    g0_str="""
    EFFECTIVE
    Ar
    1.373235e-5
    SPECIES: e / Ar
    PROCESS: E + Ar -> E + Ar, Effective
    PARAM.:  m/M = 1.373235e-5, complete set
    COMMENT: EFFECTIVE Momentum transfer CROSS SECTION.
    UPDATED: 2011-06-06 11:19:56
    COLUMNS: Energy (eV) | Cross section (m2)
    """
    g2_str="""
    IONIZATION
    Ar -> Ar^+
    1.576000e+1
    SPECIES: e / Ar
    PROCESS: E + Ar -> E + E + Ar+, Ionization
    PARAM.:  E = 15.76 eV, complete set
    COMMENT: Ionization - RAPP-SCHRAM.
    UPDATED: 2010-10-01 07:49:50
    COLUMNS: Energy (eV) | Cross section (m2)
    """

    for i, cc in enumerate(args.collisions):
        if "g0" in str(cc):
            prefix_line=g0_str

            ev1=np.logspace(-4,4,500,base=10)
            tcs = collisions.Collisions.synthetic_tcs(ev1,cc)

            cs_data=np.concatenate((ev1,tcs),axis=0)
            cs_data=cs_data.reshape((2,-1))
            cs_data=np.transpose(cs_data)

        elif "g2" in str(cc):
            prefix_line=g2_str

            ev1=np.logspace(np.log10(15.76),4,500,base=10)
            tcs = collisions.Collisions.synthetic_tcs(ev1,cc)

            cs_data=np.concatenate((ev1,tcs),axis=0)
            cs_data=cs_data.reshape((2,-1))
            cs_data=np.transpose(cs_data)
            
        f_mode="a"
        if i==0:
            f_mode="w"
        with open("../../Bolsig/%s"%(bolsig_cs_file), f_mode) as file:
            cs_str=np.array_str(cs_data)
            cs_str=cs_str.replace("[","")
            cs_str=cs_str.replace("]","")
            cs_str=" "+cs_str
            file.writelines(prefix_line)
            file.write("\n-----------------------------\n")
            cs_str=["%.14E %14E"%(cs_data[i][0],cs_data[i][1]) for i in range(cs_data.shape[0])]
            cs_str = '\n'.join(cs_str)
            file.writelines(cs_str)
            file.write("\n-----------------------------\n")
        

    replace_line(args.bolsig_dir+"run.sh", 2, "cd " + args.bolsig_dir + "\n")
    replace_line(args.bolsig_dir+"minimal-argon.dat", 8, "\""+bolsig_cs_file+"\"   / File\n")
    replace_line(args.bolsig_dir+"minimal-argon.dat", 13, str(args.E_field/collisions.AR_NEUTRAL_N/1e-21)+" / Electric field / N (Td)\n")
    os.system("sh "+args.bolsig_dir+"run.sh")

    return
    
class CollissionMode(enum.Enum):
    ELASTIC_ONLY=0
    ELASTIC_W_EXCITATION=1
    ELASTIC_W_IONIZATION=2
    ELASTIC_W_EXCITATION_W_IONIZATION=3

collisions.AR_NEUTRAL_N=3.22e22
collisions.MAXWELLIAN_N=1
collisions.AR_IONIZED_N=collisions.AR_NEUTRAL_N #collisions.MAXWELLIAN_N
parser = argparse.ArgumentParser()

def uniform_knots(k_domain, nr, sp_order):
    num_p   = nr + 1
    num_k   = 2*sp_order + (num_p -2) + 2
    
    v_knots = np.array([])
    v_knots = np.append(v_knots, k_domain[0]*np.ones(sp_order))
    v_knots = np.append(v_knots, np.linspace(k_domain[0],k_domain[1],num_p-1, endpoint=False))
    v_knots = np.append(v_knots, k_domain[1]*np.ones(sp_order+1))

    return v_knots

def create_dg_grid(g_domain, singularity_pts, nr, sp_order):
    """num domains will be equal to the number of singularity points"""
    dg_domains=list()
    dg_domains.append((g_domain[0],singularity_pts[0]))

    for i in range(1, len(singularity_pts)-1):
        dg_domains.append((singularity_pts[i], singularity_pts[i+1]))
    
    dg_domains.append((singularity_pts[-1], g_domain[1]))
    print("discontinous domains:  ", dg_domains)
    
    g_len = (g_domain[1]-g_domain[0]) 
    #rr    = np.array([(w[1]-w[0])/g_len for w in dg_domains]) #0.5 * np.ones(len(dg_domains)) #
    rr    = (1/len(dg_domains)) * np.ones(len(dg_domains)) 

    nr_dg     = np.floor(rr * nr)
    nr_dg[-1] = nr-np.sum(nr_dg[0:-1])
    nr_dg     = np.array([int(w) for w in nr_dg])
    print(nr_dg)

    knot_vec=np.array([])
    knot_vec=np.append(knot_vec, uniform_knots(dg_domains[0], nr_dg[0], sp_order))
    
    for i in range(1, len(dg_domains)-1):
        knot_vec=np.append(knot_vec, np.linspace(dg_domains[i][0], dg_domains[i][1], nr_dg[i] + sp_order + 2)[1:])
    
    knot_vec = np.append(knot_vec, np.linspace(dg_domains[-1][0], dg_domains[-1][1], max(0,nr_dg[-1]-  sp_order), endpoint=False)[1:])
    knot_vec = np.append(knot_vec, np.ones(sp_order+1) * dg_domains[-1][1])

    return knot_vec


def solve_collop_dg(steady_state, cf_list : list[colOpSp.CollisionOpSP], maxwellian, vth, E_field, t_end, dt,t_tol, collisions_included):

    collOp  = cf_list[0]
    spec_sp = collOp._spec

    Mmat = spec_sp.compute_mass_matrix()
    print("Condition number of M= %.8E"%np.linalg.cond(Mmat))
    Minv = BEUtils.choloskey_inv(Mmat)
    mm_inv_error=np.linalg.norm(np.matmul(Mmat,Minv)-np.eye(Mmat.shape[0]))/np.linalg.norm(np.eye(Mmat.shape[0]))
    if mm_inv_error >1e-12:
        print("Warning ::: mass matrix inversion error = %.12E"%(mm_inv_error))
    #assert np.allclose(np.matmul(Mmat,Minv),np.eye(Mmat.shape[0]), rtol=1e-12, atol=1e-12), "mass inverse not with in machine precision"
    
    # if not (spec_sp.get_radial_basis_type()==basis.BasisType.CHEBYSHEV_POLY or spec_sp.get_radial_basis_type()==basis.BasisType.SPLINES):
    #     raise NotImplementedError

    MVTH  = vth
    MNE   = maxwellian(0) * (np.sqrt(np.pi)**3) * (vth**3)
    MTEMP = collisions.electron_temperature(MVTH)
    print("==========================================================================")
    vratio = np.sqrt(1.0/args.basis_scale)

    if (args.radial_poly == "maxwell" or args.radial_poly == "laguerre"):
        hv    = lambda v,vt,vp : np.exp((v**2)*(1.-1./(vratio**2)))/vratio**3
    elif (args.radial_poly == "maxwell_energy"):
        hv    = lambda v,vt,vp : np.exp((v**2)*(1.-1./(vratio**2)))/vratio**3
    elif(args.radial_poly == "chebyshev"):
        hv    = lambda v,vt,vp : np.exp(-((v/vratio)**2)) / vratio**3
    elif (args.radial_poly == "bspline"):
        hv    = lambda v,vt,vp : np.exp(-((v/vratio)**2)) / vratio**3
    else:
        raise NotImplementedError

    h_init    = BEUtils.function_to_basis(spec,hv,maxwellian,None,None,None,Minv=Minv)
    h_t       = np.array(h_init)
    
    ne_t      = MNE
    mw_vth    = BEUtils.get_maxwellian_3d(vth,ne_t)
    m0_t0     = BEUtils.moment_n_f(spec_sp,h_t,mw_vth,vth,0,None,None,None,1)
    temp_t0   = BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec_sp,h_t,mw_vth,vth,None,None,None,m0_t0,1)
    
    vth_curr  = vth 
    print("Initial Ev : "   , temp_t0 * collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT)
    print("Initial mass : " , m0_t0 )


    f0_cf = interp1d(ev, bolsig_f0, kind='cubic', bounds_error=False, fill_value=(bolsig_f0[0],bolsig_f0[-1]))
    fa_cf = interp1d(ev, bolsig_a, kind='cubic', bounds_error=False, fill_value=(bolsig_a[0],bolsig_a[-1]))
    ftestt = lambda v,vt,vp : f0_cf(.5*(v*VTH)**2/collisions.ELECTRON_CHARGE_MASS_RATIO)*(1. + fa_cf(.5*(v*VTH)**2/collisions.ELECTRON_CHARGE_MASS_RATIO)*np.cos(vt))
    
    htest      = BEUtils.function_to_basis(spec,ftestt,maxwellian,None,None,None,Minv=Minv)
    radial_projection[i, :, :] = BEUtils.compute_radial_components(ev, spec, htest, maxwellian, VTH, 1)
    scale = 1./( np.trapz(radial_projection[i, 0, :]*np.sqrt(ev),x=ev) )
    radial_projection[i, :, :] *= scale
    coeffs_projection.append(htest)
    # solution_vector = np.zeros((2,h_init.shape[0]))
    # return solution_vector

    
    FOp = 0
    t1=time()
    if "g0" in collisions_included:
        g0  = collisions.eAr_G0()
        g0.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g0, mw_vth, vth_curr)

    if "g0NoLoss" in collisions_included:
        g0noloss  = collisions.eAr_G0_NoEnergyLoss()
        g0noloss.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g0noloss, mw_vth, vth_curr)

    if "g0Const" in collisions_included:
        g0const  = collisions.eAr_G0(cross_section="g0Const")
        g0const.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g0const, mw_vth, vth_curr)

    if "g2" in collisions_included:
        g2  = collisions.eAr_G2()
        g2.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g2, mw_vth, vth_curr)

    if "g2Smooth" in collisions_included:
        g2Smooth  = collisions.eAr_G2(cross_section="g2Smooth")
        g2Smooth.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g2Smooth, mw_vth, vth_curr)

    if "g2Regul" in collisions_included:
        g2Regul  = collisions.eAr_G2(cross_section="g2Regul")
        g2Regul.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g2Regul, mw_vth, vth_curr)

    if "g2Const" in collisions_included:
        g2  = collisions.eAr_G2(cross_section="g2Const")
        g2.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g2, mw_vth, vth_curr)

    if "g2step" in collisions_included:
        g2  = collisions.eAr_G2(cross_section="g2step")
        g2.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g2, mw_vth, vth_curr)

    if "g2smoothstep" in collisions_included:
        g2  = collisions.eAr_G2(cross_section="g2smoothstep")
        g2.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g2, mw_vth, vth_curr)

    t2=time()
    print("Assembled the collision op. for Vth : ", vth_curr)
    print("Collision Operator assembly time (s): ",(t2-t1))

    num_p   = spec_sp._p + 1
    num_sh  = len(spec_sp._sph_harm_lm)

    MVTH  = vth
    MNE   = maxwellian(0) * (np.sqrt(np.pi)**3) * (vth**3)
    MTEMP = collisions.electron_temperature(MVTH)
    
    advmat, eA, qA = spec_sp.compute_advection_matix_ibp()

    qA = np.kron(np.eye(num_p), qA)

    FOp   = np.matmul(np.transpose(qA), np.matmul(FOp, qA))
    h_init= np.dot(np.transpose(qA),h_init)
    
    Cmat = np.matmul(Minv, FOp)
    Emat = (E_field/MVTH) * collisions.ELECTRON_CHARGE_MASS_RATIO * np.matmul(Minv, advmat)

    if steady_state:
        iteration_error = 1
        iteration_steps = 0 

        
        u        = mass_op / (np.sqrt(np.pi)**3) 
        u        = np.matmul(np.transpose(u), qA)
        h_prev   = np.copy(h_init)
        print("initial mass : %.12E"%(np.dot(u, h_prev)))
        h_prev   = h_prev / np.dot(u, h_prev)
        
        
        nn       = Cmat.shape[0]
        Ji       = np.zeros((nn+1,nn))
        Rf       = np.zeros(nn+1)
        
        iteration_error = 1
        iteration_steps = 0

        def residual_func(x):
            y = -np.dot(u, np.matmul(Cmat, x)) * x  + np.matmul((Cmat - Emat), x)
            return np.append(y, np.dot(u, x)-1 )

        while (iteration_error > 1e-14 and iteration_steps < 1000) or iteration_steps < 5:
            Rf = residual_func(h_prev) 
            if(iteration_steps%100==0):
                print("Iteration ", iteration_steps, ": Residual =", np.linalg.norm(Rf))
            
            Ji[0:nn, 0:nn] = -2 * np.eye(nn) * np.dot(u, np.matmul(Cmat,h_prev)) + (Cmat-Emat)
            Ji[nn  , 0:nn] = u

            p = np.matmul(np.linalg.pinv(Ji,rcond=1e-14), -Rf)
            alpha=1e-1
            nRf=np.linalg.norm(Rf)
            is_diverged = False
            while (np.linalg.norm(residual_func(h_prev + alpha *p)) >= nRf):
                alpha*=0.5
                if alpha < 1e-40:
                    is_diverged = True
                    break
            
            if(is_diverged):
                print("Iteration ", iteration_steps, ": Residual =", iteration_error)
                print("line search step size becomes too small")
                break

            iteration_error = np.linalg.norm(residual_func(h_prev + alpha *p))
            h_prev += p*alpha
            iteration_steps = iteration_steps + 1

        solution_vector = np.zeros((2,h_init.shape[0]))
        solution_vector[0,:] = np.matmul(qA, h_init)
        solution_vector[1,:] = np.matmul(qA, h_prev)
        return solution_vector

    else:
        def f_rhs(t,y):
            return np.matmul(Cmat-Emat, y)
            
        ode_solver = ode(f_rhs,jac=None).set_integrator("dopri5",verbosity=1, rtol=t_tol, atol=t_tol, nsteps=1e16)
        #ode_solver = ode(f_rhs,jac=None).set_integrator("lsode", method='bdf', order=1, rtol=t_tol, atol=1e-30, nsteps=1e8)
        ode_solver.set_initial_value(h_init,t=0.0)
        ode_solver.integrate(100*dt)
        solution_vector = np.zeros((2,h_init.shape[0]))
        solution_vector[0,:] = h_init
        solution_vector[1,:] = ode_solver.y
        return solution_vector

   
    
parser.add_argument("-Nr", "--NUM_P_RADIAL"                   , help="Number of polynomials in radial direction", nargs='+', type=int, default=16)
parser.add_argument("-T", "--T_END"                           , help="Simulation time", type=float, default=1e-4)
parser.add_argument("-dt", "--T_DT"                           , help="Simulation time step size ", type=float, default=1e-7)
parser.add_argument("-o",  "--out_fname"                      , help="output file name", type=str, default='coll_op')
parser.add_argument("-ts_tol", "--ts_tol"                     , help="adaptive timestep tolerance", type=float, default=1e-15)
parser.add_argument("-l_max", "--l_max"                       , help="max polar modes in SH expansion", type=int, default=1)
parser.add_argument("-c", "--collisions"                      , help="collisions included (g0, g0Const, g0NoLoss, g2, g2Const)",nargs='+', type=str, default=["g0Const"])
parser.add_argument("-ev", "--electron_volt"                  , help="initial electron volt", type=float, default=0.25)
parser.add_argument("-bscale", "--basis_scale"                , help="basis electron volt", type=float, default=1.0)
parser.add_argument("-q_vr", "--quad_radial"                  , help="quadrature in r"        , type=int, default=200)
parser.add_argument("-q_vt", "--quad_theta"                   , help="quadrature in polar"    , type=int, default=8)
parser.add_argument("-q_vp", "--quad_phi"                     , help="quadrature in azimuthal", type=int, default=8)
parser.add_argument("-q_st", "--quad_s_theta"                 , help="quadrature in scattering polar"    , type=int, default=8)
parser.add_argument("-q_sp", "--quad_s_phi"                   , help="quadrature in scattering azimuthal", type=int, default=8)
parser.add_argument("-radial_poly", "--radial_poly"           , help="radial basis", type=str, default="bspline")
parser.add_argument("-sp_order", "--spline_order"             , help="b-spline order", type=int, default=1)
parser.add_argument("-spline_qpts", "--spline_q_pts_per_knot" , help="q points per knots", type=int, default=7)
parser.add_argument("-E", "--E_field"                         , help="Electric field in V/m", type=float, default=100)
parser.add_argument("-dv", "--dv_target"                      , help="target displacement of distribution in v_th units", type=float, default=0)
parser.add_argument("-nt", "--num_timesteps"                  , help="target number of time steps", type=float, default=100)
parser.add_argument("-steady", "--steady_state"               , help="Steady state or transient", type=bool, default=True)
parser.add_argument("-bolsig", "--bolsig_dir"                 , help="Bolsig directory", type=str, default="../../Bolsig/")
parser.add_argument("-sweep_values", "--sweep_values"         , help="Values for parameter sweep", nargs='+', type=int, default=[24, 48, 96])
parser.add_argument("-sweep_param", "--sweep_param"           , help="Paramter to sweep: Nr, ev, bscale, E, radial_poly", type=str, default="Nr")

args = parser.parse_args()
print(args)

run_bolsig(args)


[bolsig_ev, bolsig_f0, bolsig_a, bolsig_E, bolsig_mu, bolsig_M, bolsig_D, bolsig_rates] = parse_bolsig(args.bolsig_dir+"argon.out",len(args.collisions))

# setting electron volts from bolsig results for now
print("blolsig temp : %.8E"%((bolsig_mu /1.5)))
args.electron_volt = (bolsig_mu/1.5) 

run_data=list()
run_temp=list()
v = np.linspace(-2,2,100)
vx, vz = np.meshgrid(v,v,indexing='ij')
vy = np.zeros_like(vx)
v_sph_coord = BEUtils.cartesian_to_spherical(vx, vy, vz)

ev = bolsig_ev

params.BEVelocitySpace.SPH_HARM_LM = [[i,0] for i in range(args.l_max+1)]
num_sph_harm = len(params.BEVelocitySpace.SPH_HARM_LM)

radial     = np.zeros((len(args.sweep_values), num_sph_harm, len(ev)))
radial_cg  = np.zeros((len(args.sweep_values), num_sph_harm, len(ev)))
radial_projection = np.zeros((len(args.sweep_values), num_sph_harm, len(ev)))
#radial_base = np.zeros((len(args.sweep_values), len(ev)))
#radial_intial = np.zeros((num_sph_harm, len(ev)))

coeffs_projection = list()

density_slice         = np.zeros((len(args.sweep_values),len(vx[0]),len(vx[1])))
density_slice_initial = np.zeros((len(args.sweep_values),len(vx[0]),len(vx[1])))

SPLINE_ORDER = args.spline_order
basis.BSPLINE_BASIS_ORDER=SPLINE_ORDER
basis.XLBSPLINE_NUM_Q_PTS_PER_KNOT=args.spline_q_pts_per_knot

mu = []
M = []
D = []
rates = []
for i in enumerate(args.collisions):
    rates.append([])

for i, value in enumerate(args.sweep_values):

    if args.sweep_param == "Nr":
        args.NUM_P_RADIAL = value
    # elif args.sweep_param == "l_max":
    #     args.l_max = value
    elif args.sweep_param == "ev":
        args.electron_volt = value
        args.electron_volt_basis = value
    elif args.sweep_param == "bscale":
        args.basis_scale = value
    elif args.sweep_param == "E":
        args.E_field = value
    elif args.sweep_param == "radial_poly":
        args.radial_poly = value
    elif args.sweep_param == "q_vr":
        args.quad_radial = value

    params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER = args.NUM_P_RADIAL
    
    if (args.radial_poly == "maxwell"):
        r_mode = basis.BasisType.MAXWELLIAN_POLY
        params.BEVelocitySpace.NUM_Q_VR  = args.quad_radial
    elif (args.radial_poly == "maxwell_energy"):
        r_mode = basis.BasisType.MAXWELLIAN_ENERGY_POLY
        params.BEVelocitySpace.NUM_Q_VR  = args.quad_radial
    elif (args.radial_poly == "laguerre"):
        r_mode = basis.BasisType.LAGUERRE
        params.BEVelocitySpace.NUM_Q_VR  = args.quad_radial
    elif (args.radial_poly == "chebyshev"):
        r_mode = basis.BasisType.CHEBYSHEV_POLY
        params.BEVelocitySpace.NUM_Q_VR  = args.quad_radial
    elif (args.radial_poly == "bspline"):
        r_mode = basis.BasisType.SPLINES
        params.BEVelocitySpace.NUM_Q_VR  = basis.XlBSpline.get_num_q_pts(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER, SPLINE_ORDER, basis.XLBSPLINE_NUM_Q_PTS_PER_KNOT)

    params.BEVelocitySpace.NUM_Q_VT  = args.quad_theta
    params.BEVelocitySpace.NUM_Q_VP  = args.quad_phi
    params.BEVelocitySpace.NUM_Q_CHI = args.quad_s_theta
    params.BEVelocitySpace.NUM_Q_PHI = args.quad_s_phi
    params.BEVelocitySpace.VELOCITY_SPACE_DT = args.T_DT

    BASIS_EV                       = args.electron_volt*args.basis_scale
    collisions.MAXWELLIAN_TEMP_K   = BASIS_EV * collisions.TEMP_K_1EV
    collisions.ELECTRON_THEMAL_VEL = collisions.electron_thermal_velocity(collisions.MAXWELLIAN_TEMP_K) 
    VTH                            = collisions.ELECTRON_THEMAL_VEL
    c_gamma = np.sqrt(2*collisions.ELECTRON_CHARGE_MASS_RATIO)
    sig_pts = np.array([np.sqrt(15.76) * c_gamma/VTH])
    print("singularity pts : ", sig_pts)

    ev_range = ((0*VTH/c_gamma)**2, ev[-1] + 4 * np.finfo(float).eps)
    k_domain = (np.sqrt(ev_range[0]) * c_gamma / VTH, np.sqrt(ev_range[1]) * c_gamma / VTH)
    print("target ev range : (%.4E, %.4E) ----> knots domain : (%.4E, %.4E)" %(ev_range[0], ev_range[1], k_domain[0],k_domain[1]))

    print("""===========================Parameters ======================""")
    print("\tMAXWELLIAN_N        : ", collisions.MAXWELLIAN_N)
    print("\tELECTRON_THEMAL_VEL : ", VTH," ms^-1")
    print("\tBASIS_EV            : ", BASIS_EV,"eV")
    print("\tDT : ", params.BEVelocitySpace().VELOCITY_SPACE_DT, " s")
    print("""============================================================""")
    params.print_parameters()

    maxwellian = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)
    
    # v_knots = uniform_knots(k_domain, params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER, SPLINE_ORDER) 
    # cf      = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,poly_type=r_mode,k_domain=k_domain, sig_pts=sig_pts, knot_vec=v_knots)
    # spec    = cf._spec
    
    # mass_op   = BEUtils.mass_op(spec, None, 64, 2, 1)
    # temp_op   = BEUtils.temp_op(spec, None, 64, 2, 1)
    # avg_vop   = BEUtils.mean_velocity_op(spec, None, 64, 4, 1)
    # eavg_to_K = (2/(3*scipy.constants.Boltzmann))
    # ev_fac    = (collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT)

    # data               = solve_collop_dg(args.steady_state, [cf], maxwellian, VTH, args.E_field, args.T_END, args.T_DT, args.ts_tol, collisions_included=args.collisions)
    # radial_cg[i, :, :] = BEUtils.compute_radial_components(ev, spec, data[-1,:], maxwellian, VTH, 1)
    # scale              = 1./( np.trapz(radial_cg[i,0,:]*np.sqrt(ev),x=ev) )
    # radial_cg[i, :, :] *= scale
    

    v_knots = uniform_knots(k_domain, params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER, SPLINE_ORDER) #create_dg_grid(k_domain, sig_pts, params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER, SPLINE_ORDER)
    cf      = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,poly_type=r_mode,k_domain=k_domain, sig_pts=None, knot_vec=v_knots)
    spec    = cf._spec
    spec._num_q_radial = params.BEVelocitySpace.NUM_Q_VR
    mass_op   = BEUtils.mass_op(spec, None, 64, 2, 1)
    temp_op   = BEUtils.temp_op(spec, None, 64, 2, 1)
    avg_vop   = BEUtils.mean_velocity_op(spec, None, 64, 4, 1)
    eavg_to_K = (2/(3*scipy.constants.Boltzmann))
    ev_fac    = (collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT)

    data            = solve_collop_dg(args.steady_state, [cf], maxwellian, VTH, args.E_field, args.T_END, args.T_DT, args.ts_tol, collisions_included=args.collisions)
    radial[i, :, :] = BEUtils.compute_radial_components(ev, spec, data[-1,:], maxwellian, VTH, 1)
    scale            = 1./( np.trapz(radial[i,0,:]*np.sqrt(ev),x=ev) )
    radial[i, :, :] *= scale
    
    run_data.append(data)


    nt = len(data[:,0])
    temp_evolution = np.zeros(nt)
    for k in range(nt):
        current_vth      = VTH
        current_mw       = maxwellian
        current_mass     = np.dot(data[k,:],mass_op) * current_vth**3 * current_mw(0)
        current_temp     = np.dot(data[k,:],temp_op) * current_vth**5 * current_mw(0) * 0.5 * collisions.MASS_ELECTRON * eavg_to_K / current_mass
        temp_evolution[k] = current_temp/collisions.TEMP_K_1EV

    mu.append(1.5*temp_evolution[-1])
    print(1.5*temp_evolution[-1])

    total_cs = 0

    for col_idx, col in enumerate(args.collisions):
        cs = collisions.Collisions.synthetic_tcs(ev, col)
        total_cs += cs
        rates[col_idx].append( np.sqrt(2.*collisions.ELECTRON_CHARGE_MASS_RATIO)*np.trapz(radial[i,0,:]*ev*cs,x=ev) )

        if col == "g2" or col == "g2Const" or col == "g2Smooth":
            total_cs += rates[col_idx][-1]/np.sqrt(ev)/np.sqrt(2.*collisions.ELECTRON_CHARGE_MASS_RATIO)

    D.append( np.sqrt(2.*collisions.ELECTRON_CHARGE_MASS_RATIO)/3.*np.trapz(radial[i,0,:]*ev/total_cs,x=ev) )
    M.append( -np.sqrt(2.*collisions.ELECTRON_CHARGE_MASS_RATIO)/3.*np.trapz(deriv_fd(ev,radial[i,0,:])*ev/total_cs,x=ev) )

    run_temp.append(temp_evolution)


# np.set_printoptions(precision=16)
# print(data[1,:])

if (1):
    fig = plt.figure(figsize=(19, 9), dpi=300)

    num_subplots = num_sph_harm + 2

    plt.subplot(2, num_subplots,  1 + 0)
    plt.semilogy(bolsig_ev,  abs(bolsig_f0), '-k', label="bolsig")
    
    # print(np.trapz( bolsig[:,1]*np.sqrt(bolsig[:,0]), x=bolsig[:,0] ))
    # print(np.trapz( scale*radial[i, 0]*np.sqrt(ev), x=ev ))

    plt.subplot(2, num_subplots,  1 + 1)
    plt.semilogy(bolsig_ev,  abs(bolsig_f0*bolsig_a * spec._sph_harm_real(0, 0, 0, 0)/spec._sph_harm_real(1, 0, 0, 0)), '-k', label="bolsig")

    for i, value in enumerate(args.sweep_values):
        data=run_data[i]
        data_projection=coeffs_projection[i]

        lbl = args.sweep_param+"="+str(value)

        # spherical components plots

        for l_idx in range(num_sph_harm):

            plt.subplot(2, num_subplots, num_subplots + 1+l_idx)

            color = next(plt.gca()._get_lines.prop_cycler)['color']
            plt.plot(np.abs(data[-1,l_idx::num_sph_harm]),label=lbl, color=color)
            #plt.plot(np.abs(data_projection[l_idx::num_sph_harm]), '--',label=lbl+" (proj)", color=color)
            #plt.plot(ev, abs(radial[i, l_idx]-radial_cg[i,l_idx]), '-', label=lbl, color=color)
            # if l_idx==0:
            #     plt.plot(ev, abs(radial[i, l_idx]-radial_cg[i,l_idx]), '-', label=lbl, color=color)
            #     #plt.plot(ev, abs(radial_cg[i, l_idx]-bolsig_f0), '-', label=lbl+ "(cg)", color=color)
            # elif l_idx==1:
            #     #plt.plot(ev, abs(radial[i, l_idx]-bolsig_f0*bolsig_a * spec._sph_harm_real(0, 0, 0, 0)/spec._sph_harm_real(1, 0, 0, 0)), '-', label=lbl + "(dg)", color=color)
            #     #plt.plot(ev, abs(radial_cg[i, l_idx]-bolsig_f0*bolsig_a * spec._sph_harm_real(0, 0, 0, 0)/spec._sph_harm_real(1, 0, 0, 0)), '-', label=lbl+ "(cg)", color=color)



            plt.title(label="l=%d"%l_idx)
            plt.yscale('log')
            plt.xlabel("coeff #")
            plt.ylabel("abs(coeff)")
            plt.grid(visible=True)
            if l_idx == 0:
                plt.legend()

            plt.subplot(2, num_subplots, 1 + l_idx)

            # if args.steady_state == False:
            #     plt.semilogy(ev,  abs(radial_initial[l_idx]), '-', label="Initial")

            color = next(plt.gca()._get_lines.prop_cycler)['color']
            plt.semilogy(ev,  abs(radial[i, l_idx]), '-', label=lbl, color=color)
            #plt.semilogy(ev,  abs(radial_projection[i, l_idx]), '--', label=lbl+" (proj)", color=color)
            # if l_idx == 0:
            #     plt.semilogy(ev,  abs(radial_base[i]), ':', label=lbl+" (base)", color=color)
                
            # plt.semilogy(ev, -radial[i, l_idx], 'o', label=lbl, color=color, markersize=3, markerfacecolor='white')


            #plt.yscale('log')
            plt.xlabel("Energy, eV")
            plt.ylabel("Radial component")
            plt.grid(visible=True)
            if l_idx == 0:
                plt.legend()
            # plt.legend()

        # plt.subplot(2, num_subplots, num_sph_harm + 1)
        # temp = run_temp[i]
        # plt.plot(temp, label=lbl)

    plt.subplot(2, num_subplots, num_sph_harm + 1)
    #plt.plot(args.sweep_values, mu, 'o-', label='us')
    #plt.axhline(y=bolsig_mu, label='bolsig', color='k')
    plt.semilogy(args.sweep_values, abs(np.array(mu)/bolsig_mu-1), 'o-', label='us')
    # plt.legend()
    plt.xlabel(args.sweep_param)
    plt.ylabel("Rel. error in mean energy")

    # if args.sweep_param != "radial_poly":
        # plt.gca().ticklabel_format(useOffset=False)

    plt.subplot(2, num_subplots, num_sph_harm + 2)
    for col_idx, col in enumerate(args.collisions):
        if bolsig_rates[col_idx] != 0:
            plt.semilogy(args.sweep_values, abs(rates[col_idx]/bolsig_rates[col_idx]-1), 'o-', label=col)
            # plt.axhline(y=0, label='bolsig '+col, color='k')
    plt.legend()
    plt.xlabel(args.sweep_param)
    plt.ylabel("Rel. error in reaction rates")

    # if args.sweep_param != "radial_poly":
        # plt.gca().ticklabel_format(useOffset=False)


    plt.subplot(2, num_subplots, num_subplots + num_sph_harm + 1)
    # plt.plot(args.sweep_values, M, 'o-', label='us')
    # plt.axhline(y=bolsig_M, label='bolsig', color='k')
    plt.semilogy(args.sweep_values, abs(np.array(M)/bolsig_M-1), 'o-', label='us')
    # plt.legend()
    plt.xlabel(args.sweep_param)
    plt.ylabel("Rel. error in mobility")

    # if args.sweep_param != "radial_poly":
        # plt.gca().ticklabel_format(useOffset=False)

    plt.subplot(2, num_subplots, num_subplots + num_sph_harm + 2)
    # plt.plot(args.sweep_values, D, 'o-', label='us')
    # plt.axhline(y=bolsig_D, label='bolsig', color='k')
    plt.semilogy(args.sweep_values, abs(np.array(D)/bolsig_D-1), 'o-', label='us')
    # plt.legend()
    plt.xlabel(args.sweep_param)
    plt.ylabel("Rel. error in diffusion coefficient")

    # if args.sweep_param != "radial_poly":
        # plt.gca().ticklabel_format(useOffset=False)

    # plt.subplot(2, num_subplots, num_subplots + num_sph_harm + 1)

    # lvls = np.linspace(0, np.amax(density_slice_initial[0]), 10)

    # plt.contour(vx, vz, density_slice_initial[-1], levels=lvls, linestyles='solid', colors='grey', linewidths=1) 
    # plt.contour(vx, vz, density_slice[-1], levels=lvls, linestyles='dotted', colors='red', linewidths=1)  
    # plt.gca().set_aspect('equal')
    
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(wspace=0.4)

    if (args.radial_poly == "bspline"):
        fig.suptitle("Collisions: " + str(args.collisions) + ", E = " + str(args.E_field) + ", polys = " + str(args.radial_poly)+", sp_order= " + str(args.spline_order) + ", Nr = " + str(args.NUM_P_RADIAL) + ", bscale = " + str(args.basis_scale) + " (sweeping " + args.sweep_param + ")")
        # plt.show()
        plt.savefig("bspline_dg_vs_bolsig_" + "_".join(args.collisions) + "_E" + str(args.E_field) + "_poly_" + str(args.radial_poly)+ "_sp_"+ str(args.spline_order) + "_nr" + str(args.NUM_P_RADIAL) + "_bscale" + str(args.basis_scale) + "_sweeping_" + args.sweep_param + ".png")
    else:
        fig.suptitle("Collisions: " + str(args.collisions) + ", E = " + str(args.E_field) + ", polys = " + str(args.radial_poly) + ", Nr = " + str(args.NUM_P_RADIAL) + ", bscale = " + str(args.basis_scale) + " (sweeping " + args.sweep_param + ")")
        # plt.show()
        plt.savefig("us_vs_bolsig_" + "_".join(args.collisions) + "_E" + str(args.E_field) + "_poly_" + str(args.radial_poly) + "_nr" + str(args.NUM_P_RADIAL) + "_bscale" + str(args.basis_scale) + "_sweeping_" + args.sweep_param + ".png")


    





