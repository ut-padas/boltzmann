"""
@package Boltzmann collision operator solver. 
"""

from cProfile import run
from cmath import sqrt
from dataclasses import replace
import enum
import string
import scipy
import scipy.optimize
import scipy.interpolate
from sympy import arg, eye
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

import matplotlib.pyplot as plt

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

class CollissionMode(enum.Enum):
    ELASTIC_ONLY=0
    ELASTIC_W_EXCITATION=1
    ELASTIC_W_IONIZATION=2
    ELASTIC_W_EXCITATION_W_IONIZATION=3

collisions.AR_NEUTRAL_N=3.22e22
collisions.MAXWELLIAN_N=1e18
collisions.AR_IONIZED_N=collisions.AR_NEUTRAL_N #collisions.MAXWELLIAN_N
parser = argparse.ArgumentParser()

def constant_r_eval(spec : sp.SpectralExpansionSpherical, cf, r):
    theta = np.linspace(0,np.pi,50)
    phi   = np.linspace(0,2*np.pi,100)
    grid  = np.meshgrid(theta, phi, indexing='ij')
    
    num_p  = spec._p + 1
    num_sh = len(spec._sph_harm_lm)
    
    b_eval = np.array([ np.exp(-r**2) * spec.basis_eval_radial(r,k,l) * spec.basis_eval_spherical(grid[0],grid[1],l,m) for lm_idx, (l,m) in enumerate(spec._sph_harm_lm) for k in range(num_p)])
    b_eval = b_eval.reshape(num_sh,num_p, -1)
    b_eval = np.swapaxes(b_eval,0,1)
    b_eval = b_eval.reshape(num_p * num_sh,-1)
    
    return np.dot(cf, b_eval).reshape(50,100), theta, phi



def solve_collop(steady_state, collOp:colOpSp.CollisionOpSP, h_init, maxwellian, vth, E_field, t_end, dt,t_tol, collisions_included):
    spec_sp = collOp._spec

    t1=time()
    M  = spec_sp.compute_mass_matrix()
    t2=time()
    print("Mass assembly time (s): ", (t2-t1))
    print("Condition number of M= %.8E"%np.linalg.cond(M))
    Minv = BEUtils.choloskey_inv(M)
    #Minv = BEUtils.block_jacobi_inv(M)
    assert np.allclose(np.matmul(M,Minv),np.eye(M.shape[0]), rtol=1e-12, atol=1e-12), "mass inverse not with in machine precision"

    MVTH  = vth
    MNE   = maxwellian(0) * (np.sqrt(np.pi)**3) * (vth**3)
    MTEMP = collisions.electron_temperature(MVTH)
    print("==========================================================================")

    h_t = np.array(h_init)
    
    ne_t      = MNE
    mw_vth    = BEUtils.get_maxwellian_3d(vth,ne_t)
    m0_t0     = BEUtils.moment_n_f(spec_sp,h_t,mw_vth,vth,0,None,None,None,1)
    temp_t0   = BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec_sp,h_t,mw_vth,vth,None,None,None,m0_t0,1)
    # vth_curr  = collisions.electron_thermal_velocity(temp_t0) 
    vth_curr  = vth 
    print("Initial Ev : "   , temp_t0 * collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT)
    print("Initial mass : " , m0_t0 )

    #spec_sp.get_num_coefficients

    t1=time()
    advmat = spec_sp.compute_advection_matix()
    advmat = np.matmul(Minv,advmat)
    t2=time()
    print("Advection Operator assembly time (s): ",(t2-t1))
    # advmat = assemble_advection_matix_lp_lag(spec_sp._p, spec_sp._sph_harm_lm)
    # print(advmat[0:spec_sp._p+1 , 0:spec_sp._p+1])

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

    if "g2Const" in collisions_included:
        g2const  = collisions.eAr_G2(cross_section="g2Const", threshold=0)
        g2const.reset_scattering_direction_sp_mat()
        FOp = FOp + collisions.AR_NEUTRAL_N * collOp.assemble_mat(g2const, mw_vth, vth_curr)

    t2=time()
    print("Assembled the collision op. for Vth : ", vth_curr)
    print("Collision Operator assembly time (s): ",(t2-t1))
    FOp = np.matmul(Minv, FOp)
    print("Cond(C)= %.8E"%np.linalg.cond(FOp))
    print("Cond(E)= %.8E"%np.linalg.cond(E_field/VTH*collisions.ELECTRON_CHARGE_MASS_RATIO*advmat))
    print("Cond(C-E)= %.8E"%np.linalg.cond(FOp-E_field/VTH*collisions.ELECTRON_CHARGE_MASS_RATIO*advmat))

    # plt.show()
    if steady_state == True:
        print("solving steady")
        if (args.radial_poly == "maxwell" or args.radial_poly == "laguerre"):
            # h_red = np.zeros(len(h_init)-1)
            h_red = h_init[1:]

            Cmat = FOp
            Emat = E_field/VTH*collisions.ELECTRON_CHARGE_MASS_RATIO*advmat
            print("Cmat cond: %.8E"%(np.linalg.cond(Cmat)))
            print("Emat cond: %.8E"%(np.linalg.cond(Emat)))

            iteration_error = 1
            iteration_steps = 0
            while (iteration_error > 1e-14 and iteration_steps < 30) or iteration_steps < 5:
                h_prev = h_red
                h_red = - np.linalg.solve(Cmat[1:,1:] - Emat[1:,1:] - (Cmat[0,0] + np.dot(h_red, Cmat[0,1:]))*np.eye(len(h_red)), Cmat[1:,0] - Emat[1:,0])
                # inv_op = BEUtils.block_jacobi_inv(Cmat[1:,1:] - Emat[1:,1:] - (Cmat[0,0] + np.dot(h_red, Cmat[0,1:]))*np.eye(len(h_red)), 4)
                # h_red = - np.matmul(inv_op, Cmat[1:,0] - Emat[1:,0])
                iteration_error = np.linalg.norm(h_prev-h_red)
                print("Iteration ", iteration_steps, ": Residual =", iteration_error)
                iteration_steps = iteration_steps + 1

            solution_vector = np.zeros((1,h_init.shape[0]))
            solution_vector[0,0] = 1
            solution_vector[0,1:] = h_red
            return solution_vector

        elif (args.radial_poly == "bspline"):
            Cmat = FOp
            Emat = E_field/VTH*collisions.ELECTRON_CHARGE_MASS_RATIO*advmat
            iteration_error = 1
            iteration_steps = 0 


            m_scale  = 1/np.sqrt(np.pi)**3
            h_prev   = np.copy(h_init)
            nn       = Cmat.shape[0]
            Ai       = np.zeros((nn+1, nn))
            rr       = np.zeros((nn+1,1))
            rr[-1:0] = 1
            
            raise NotImplementedError

    else:
        Cmat = FOp
        Emat = E_field/VTH*collisions.ELECTRON_CHARGE_MASS_RATIO*advmat
        
        # fname="Cmat_" + "_".join(args.collisions) + "_ev_"+str(args.electron_volt) + "_poly_" + str(args.radial_poly) + "_nr" + str(spec_sp._p+1)+".npy"
        # np.save(fname, FOp)
        # fname="Emat_" + "_".join(args.collisions) + "_ev_"+str(args.electron_volt) + "_poly_" + str(args.radial_poly) + "_nr" + str(spec_sp._p+1)+".npy"
        # np.save(fname, ( (1.0 / VTH) * collisions.ELECTRON_CHARGE_MASS_RATIO ) * advmat)
        
        current_mw  = maxwellian
        current_vth = vth

        eavg_to_K = (2/(3*scipy.constants.Boltzmann))
        ev_fac    = (collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT)

        current_mass     = np.dot(h_init,mass_op) * vth**3 * current_mw(0)
        current_temp     = np.dot(h_init,temp_op) * vth**5 * current_mw(0) * 0.5 * collisions.MASS_ELECTRON * eavg_to_K / current_mass
        mass_initial     = current_mass 
        print("initial Ev = %.14E initial mass = %.14E"%(current_temp * ev_fac, mass_initial))
        
        u      = mass_op / (np.sqrt(np.pi)**3) 
        h_init = h_init/ np.dot(u,h_init)
        pp     = u / np.linalg.norm(u,ord=2)
        w      = np.matmul(u.reshape(1,Cmat.shape[0]),Cmat)
        Imppt  = np.eye(pp.shape[0]) - np.matmul(pp.reshape(-1,1),np.transpose(pp.reshape(-1,1)))
        f1     = h_init-np.matmul(Imppt,h_init)
        
        def f_rhs(t,y):
            f2     = y
            c1     = np.dot(w,f1+f2)
            return np.matmul(Imppt, c1 * (f1+f2) + np.matmul(Cmat-Emat, f1+f2))
            
            
        ode_solver = ode(f_rhs,jac=None).set_integrator("dopri5",verbosity=1, rtol=t_tol, atol=t_tol,max_step=1e-8, nsteps=1e16)
        ode_solver.set_initial_value(np.matmul(Imppt,h_init),t=0.0)
        
        # f1=0
        # def f_rhs(t,y):
        #     y     = y / (np.dot(mass_op,y) / (np.sqrt(np.pi)**3))
        #     Ly    = np.matmul(FOp,y) 
        #     mLy   = np.dot(mass_op,Ly) / (np.sqrt(np.pi)**3)
        #     return -mLy * y +  np.matmul(Cmat-Emat, y)
        
        # ode_solver = ode(f_rhs,jac=None).set_integrator("dopri5",verbosity=1, rtol=t_tol, atol=t_tol, nsteps=1e16)
        # # #ode_solver = ode(f_rhs,jac=None).set_integrator("dop853",verbosity=1, rtol=t_tol, atol=t_tol, nsteps=1e8)
        # # #ode_solver = ode(f_rhs,jac=None).set_integrator("lsode", method='bdf', order=2, rtol=t_tol, atol=1e-30, nsteps=1e8)
        # ode_solver.set_initial_value(h_init,t=0.0)
        
        
        max_steps   = 100
        total_steps = min(int(t_end/dt), max_steps)
        
        ss_norm         = 1
        ss_tol          = 1e-10
        tt              = np.linspace(0,t_end,total_steps)
        num_steps       = len(tt)-1
        solution_vector = np.zeros((num_steps,h_init.shape[0]))

        for t_idx in range(num_steps):
            t_curr                     = tt[t_idx]
            h_t                        = ode_solver.y + f1
            
            solution_vector [t_idx,:]  = h_t
            dt_adap = dt
            ss_reached=False
            while t_curr < tt[t_idx+1]:
                h_prev           = ode_solver.y + f1
                ode_solver.integrate(t_curr + dt)
                h_t              = ode_solver.y + f1
                t_curr          += dt_adap
            
                ss_norm          = np.linalg.norm(h_prev-h_t)/np.linalg.norm(h_t)
            
                current_mass               = np.dot(h_t,mass_op) * current_vth**3 * current_mw(0)
                current_temp               = np.dot(h_t,temp_op) * current_vth**5 * current_mw(0) * 0.5 * collisions.MASS_ELECTRON * eavg_to_K / current_mass

                vc_x                       = np.dot(avg_vop[0],h_t) * current_vth**4 * (current_mw(0)/current_mass) /current_vth
                vc_y                       = np.dot(avg_vop[1],h_t) * current_vth**4 * (current_mw(0)/current_mass) /current_vth
                vc_z                       = np.dot(avg_vop[2],h_t) * current_vth**4 * (current_mw(0)/current_mass) /current_vth
                vc_a                       = 0 + (E_field * t_curr * collisions.ELECTRON_CHARGE_MASS_RATIO /current_vth)
                
                print("time:%.2E mass: %.10E temp: %.10E vc=(%.2E,%.2E,%.2E) adv vc_z=%.2E ss_norm=%.5E dt=%.4E" %(t_curr, current_mass, current_temp * ev_fac, vc_x, vc_y,vc_z,vc_a,ss_norm,dt_adap))

                if(ss_norm < ss_tol):
                    ss_reached=True
                    for w in range(t_idx+1, solution_vector.shape[0]):
                        solution_vector[w,:] = h_t
                    print("sol. convergence norm: %.8E"%ss_norm)
                    break
            
            if(ss_reached):
                break
      
            
        return solution_vector

parser.add_argument("-Nr", "--NUM_P_RADIAL"                   , help="Number of polynomials in radial direction", nargs='+', type=int, default=64)
parser.add_argument("-T", "--T_END"                           , help="Simulation time", type=float, default=1e-4)
parser.add_argument("-dt", "--T_DT"                           , help="Simulation time step size ", type=float, default=1e-7)
parser.add_argument("-o",  "--out_fname"                      , help="output file name", type=str, default='coll_op')
parser.add_argument("-ts_tol", "--ts_tol"                     , help="adaptive timestep tolerance", type=float, default=1e-15)
parser.add_argument("-l_max", "--l_max"                       , help="max polar modes in SH expansion", type=int, default=1)
parser.add_argument("-c", "--collisions"                      , help="collisions included (g0, g0Const, g0NoLoss, g2, g2Const)",nargs='+', type=str, default=["g0", "g2"])
parser.add_argument("-ev", "--electron_volt"                  , help="initial electron volt", type=float, default=0.25)
parser.add_argument("-bscale", "--basis_scale"                , help="basis electron volt", type=float, default=1.0)
parser.add_argument("-q_vr", "--quad_radial"                  , help="quadrature in r"        , type=int, default=200)
parser.add_argument("-q_vt", "--quad_theta"                   , help="quadrature in polar"    , type=int, default=8)
parser.add_argument("-q_vp", "--quad_phi"                     , help="quadrature in azimuthal", type=int, default=8)
parser.add_argument("-q_st", "--quad_s_theta"                 , help="quadrature in scattering polar"    , type=int, default=8)
parser.add_argument("-q_sp", "--quad_s_phi"                   , help="quadrature in scattering azimuthal", type=int, default=8)
parser.add_argument("-radial_poly", "--radial_poly"           , help="radial basis", type=str, default="maxwell")
parser.add_argument("-sp_order", "--spline_order"             , help="b-spline order", type=int, default=2)
parser.add_argument("-spline_qpts", "--spline_q_pts_per_knot" , help="q points per knots", type=int, default=11)
parser.add_argument("-E", "--E_field"                         , help="Electric field in V/m", type=float, default=7500)
parser.add_argument("-dv", "--dv_target"                      , help="target displacement of distribution in v_th units", type=float, default=0)
parser.add_argument("-nt", "--num_timesteps"                  , help="target number of time steps", type=float, default=100)
parser.add_argument("-steady", "--steady_state"               , help="Steady state or transient", type=bool, default=False)
parser.add_argument("-bolsig", "--bolsig_dir"                 , help="Bolsig directory", type=str, default="../../Bolsig/")

# parser.add_argument("-sweep_values", "--sweep_values"         , help="Values for parameter sweep", type=float, default=[1.25, 1., 0.75, 0.5, 0.25, 0.125])
# parser.add_argument("-sweep_param", "--sweep_param"           , help="Paramter to sweep: Nr, ev, bscale, E, radial_poly", type=str, default="bscale")
parser.add_argument("-sweep_values", "--sweep_values"         , help="Values for parameter sweep", nargs='+', type=int, default=[64, 80, 96, 112, 128])
#parser.add_argument("-sweep_values", "--sweep_values"         , help="Values for parameter sweep", type=str, default=[16, 32])
# parser.add_argument("-sweep_values", "--sweep_values"         , help="Values for parameter sweep", type=str, default=[16])
parser.add_argument("-sweep_param", "--sweep_param"           , help="Paramter to sweep: Nr, ev, bscale, E, radial_poly", type=str, default="Nr")
# parser.add_argument("-sweep_values", "--sweep_values"         , help="Values for parameter sweep", type=str, default=["maxwell", "laguerre"])
# parser.add_argument("-sweep_param", "--sweep_param"           , help="Paramter to sweep: Nr, ev, bscale, E, radial_poly", type=str, default="radial_poly")

#parser.add_argument("-r", "--restore", help="if 1 try to restore solution from a checkpoint", type=int, default=0)
args = parser.parse_args()
print(args)

# run bolsig for chosen parameters and parse its output
bolsig_cs_file = args.collisions[0]
for col in args.collisions[1:]:
    bolsig_cs_file = bolsig_cs_file + "_" + col
bolsig_cs_file = bolsig_cs_file + ".txt"

replace_line(args.bolsig_dir+"run.sh", 2, "cd " + args.bolsig_dir + "\n")
replace_line(args.bolsig_dir+"minimal-argon.dat", 8, "\""+bolsig_cs_file+"\"   / File\n")
replace_line(args.bolsig_dir+"minimal-argon.dat", 13, str(args.E_field/collisions.AR_NEUTRAL_N/1e-21)+" / Electric field / N (Td)\n")
os.system("sh "+args.bolsig_dir+"run.sh")
[bolsig_ev, bolsig_f0, bolsig_a, bolsig_E, bolsig_mu, bolsig_M, bolsig_D, bolsig_rates] = parse_bolsig(args.bolsig_dir+"argon.out",len(args.collisions))

# setting electron volts from bolsig results for now
print("blolsig temp : %.8E"%((bolsig_mu /1.5)))
args.electron_volt = bolsig_mu#0.98*(bolsig_mu/1.5) #0.8 * (bolsig_mu/1.5) 
print(args.electron_volt)
# bolsig = np.genfromtxt(args.bolsig_data,delimiter=',')
# plt.plot(bolsig[:,0], bolsig[:,1])
# plt.show()

run_data=list()
run_temp=list()

v = np.linspace(-2,2,100)
vx, vz = np.meshgrid(v,v,indexing='ij')
vy = np.zeros_like(vx)
v_sph_coord = BEUtils.cartesian_to_spherical(vx, vy, vz)

# vr = np.linspace(1e-3,5,100)
# ev = np.linspace(args.electron_volt/20.,20.*args.electron_volt,1000)
# ev = np.linspace(1e-5,10,100)
ev = bolsig_ev
#print(ev)

dev = ev[-1]-ev[-2]
numadd = int(.5*ev[-1]/dev)
ev = np.concatenate((ev, np.linspace(ev[-1] + dev, ev[-1] + numadd*dev, numadd)))

# assuming l_max is not changing
# params.BEVelocitySpace.SPH_HARM_LM = [[i,j] for i in range(args.l_max+1) for j in range(-i,i+1)]
params.BEVelocitySpace.SPH_HARM_LM = [[i,0] for i in range(args.l_max+1)]
num_sph_harm = len(params.BEVelocitySpace.SPH_HARM_LM)

radial = np.zeros((len(args.sweep_values), num_sph_harm, len(ev)))
radial_base = np.zeros((len(args.sweep_values), len(ev)))
radial_intial = np.zeros((num_sph_harm, len(ev)))

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

    # if args.dv_target != 0 and args.E_field != 0:
    #     args.T_END = args.dv_target/args.E_field*collisions.electron_thermal_velocity(args.electron_volt*collisions.TEMP_K_1EV)/collisions.ELECTRON_CHARGE_MASS_RATIO

    # if args.num_timesteps != 0:
    #     args.T_DT = args.T_END/args.num_timesteps

    params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER = args.NUM_P_RADIAL
    if (args.radial_poly == "maxwell"):
        r_mode = basis.BasisType.MAXWELLIAN_POLY
        params.BEVelocitySpace.NUM_Q_VR  = args.quad_radial
        
    elif (args.radial_poly == "laguerre"):
        r_mode = basis.BasisType.LAGUERRE
        params.BEVelocitySpace.NUM_Q_VR  = args.quad_radial

    elif (args.radial_poly == "bspline"):
        r_mode = basis.BasisType.SPLINES
        params.BEVelocitySpace.NUM_Q_VR  = basis.BSpline.get_num_q_pts(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER, SPLINE_ORDER, basis.XLBSPLINE_NUM_Q_PTS_PER_KNOT)

    params.BEVelocitySpace.NUM_Q_VT  = args.quad_theta
    params.BEVelocitySpace.NUM_Q_VP  = args.quad_phi
    params.BEVelocitySpace.NUM_Q_CHI = args.quad_s_theta
    params.BEVelocitySpace.NUM_Q_PHI = args.quad_s_phi
    params.BEVelocitySpace.VELOCITY_SPACE_DT = args.T_DT

    BASIS_EV = args.electron_volt*args.basis_scale
    vratio = np.sqrt(1.0/args.basis_scale)

    collisions.MAXWELLIAN_TEMP_K   = BASIS_EV * collisions.TEMP_K_1EV
    collisions.ELECTRON_THEMAL_VEL = collisions.electron_thermal_velocity(collisions.MAXWELLIAN_TEMP_K) 
    VTH                            = collisions.ELECTRON_THEMAL_VEL
    c_gamma = np.sqrt(2*collisions.ELECTRON_CHARGE_MASS_RATIO)
    sig_pts = np.array([np.sqrt(15.76) * c_gamma/VTH])

    # adaptive_learner = ALearner1D()
    # adaptive_learner.setup(bolsig_ev,bolsig_f0,(bolsig_ev[0],bolsig_ev[-1]))
    # adap_knots = adaptive_learner.get_pts(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER) * c_gamma/VTH
    # sig_pts = np.sort(np.append(sig_pts,adap_knots))
    print("singularity pts : ", sig_pts)

    # uncomment to plot the cross section data used. 
    #collisions.collission_cs_test()

    ev_range = (0, ev[-1])
    #k_domain = np.sqrt(ev_range) * c_gamma / VTH
    k_domain = (np.sqrt(ev_range[0]) * c_gamma / VTH, np.sqrt(ev_range[1]) * c_gamma / VTH)
    print("target ev range : (%.4E, %.4E) ----> knots domain : (%.4E, %.4E)" %(ev_range[0], ev_range[1], k_domain[0],k_domain[1]))
    cf    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,poly_type=r_mode,k_domain=k_domain, sig_pts=sig_pts)
    spec  = cf._spec

    print("""===========================Parameters ======================""")
    print("\tMAXWELLIAN_N        : ", collisions.MAXWELLIAN_N)
    print("\tELECTRON_THEMAL_VEL : ", VTH," ms^-1")
    print("\tBASIS_EV            : ", BASIS_EV,"eV")
    print("\tDT : ", params.BEVelocitySpace().VELOCITY_SPACE_DT, " s")
    print("""============================================================""")
    params.print_parameters()

    maxwellian = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)
    
    if (args.radial_poly == "maxwell" or args.radial_poly == "laguerre"):
        scale = (np.sqrt(2.*collisions.ELECTRON_CHARGE_MASS_RATIO)/VTH)**3*(2./np.sqrt(np.pi))/basis.maxpoly.maxpolyeval(2,0,0)
        hv    = lambda v,vt,vp : np.exp((v**2)*(1.-1./(vratio**2)))/vratio**3
    elif (args.radial_poly == "bspline"):
        scale =  (2 *(np.sqrt(2.*collisions.ELECTRON_CHARGE_MASS_RATIO)/VTH)**3) / np.sqrt(np.pi)
        hv    = lambda v,vt,vp : np.exp((v**2)*(-1./(vratio**2)))/vratio**3

    h_vec      = BEUtils.function_to_basis(spec,hv,maxwellian,None,None,None)
    spec_sp   = spec 
    mass_op   = BEUtils.mass_op(spec_sp, None, 64, 2, 1)
    temp_op   = BEUtils.temp_op(spec_sp, None, 64, 2, 1)
    avg_vop   = BEUtils.mean_velocity_op(spec_sp, None, 64, 4, 1)
    eavg_to_K = (2/(3*scipy.constants.Boltzmann))
    ev_fac    = (collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT)
    # if args.steady_state:
    #     # continuation for quicker solution
    #     if i == 0:
    #         h_vec *= 0
    #         h_vec[0] = 1
    #     else:
    #         data = run_data[i-1]
    #         h_vec *= 0
    #         h_vec[0:len(data[-1])] = data[-1]


    data            = solve_collop(args.steady_state, cf, h_vec, maxwellian, VTH, args.E_field, args.T_END, args.T_DT, args.ts_tol, collisions_included=args.collisions)
    radial[i, :, :] = BEUtils.compute_radial_components(ev, spec, data[-1,:], maxwellian, VTH, 1)*scale
    
    if args.steady_state == False and i == 0:
        radial_initial = BEUtils.compute_radial_components(ev, spec, data[0,:], maxwellian, VTH, 1)*scale

    # empty_cf = data[0,:]*0
    # empty_cf[0] = 1
    empty_cf=h_vec
    radial_base[i,:] = BEUtils.compute_radial_components(ev, spec, empty_cf, maxwellian, VTH, 1)[0,:]*scale

    # density_slice[i]  = BEUtils.sample_distriubtion_spherical(v_sph_coord, spec, data[-1,:], maxwellian, VTH, 1)
    # density_slice_initial[i]  = BEUtils.sample_distriubtion_spherical(v_sph_coord, spec, data[0,:], maxwellian, VTH, 1)
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

        if col == "g2" or col == "g2Const":
            total_cs += rates[col_idx][-1]/np.sqrt(ev)/np.sqrt(2.*collisions.ELECTRON_CHARGE_MASS_RATIO)

    D.append( np.sqrt(2.*collisions.ELECTRON_CHARGE_MASS_RATIO)/3.*np.trapz(radial[i,0,:]*ev/total_cs,x=ev) )
    M.append( -np.sqrt(2.*collisions.ELECTRON_CHARGE_MASS_RATIO)/3.*np.trapz(deriv_fd(ev,radial[i,0,:])*ev/total_cs,x=ev) )

    run_temp.append(temp_evolution)


# np.set_printoptions(precision=16)
# print(data[1,:])

if (1):
    fig = plt.figure(figsize=(19, 9), dpi=200)

    num_subplots = num_sph_harm + 2

    plt.subplot(2, num_subplots, num_subplots + 1 + 0)
    plt.semilogy(bolsig_ev,  bolsig_f0, '-k', label="bolsig")
    # print(np.trapz( bolsig[:,1]*np.sqrt(bolsig[:,0]), x=bolsig[:,0] ))
    # print(np.trapz( scale*radial[i, 0]*np.sqrt(ev), x=ev ))

    plt.subplot(2, num_subplots, num_subplots + 1 + 1)
    plt.semilogy(bolsig_ev,  bolsig_f0*bolsig_a*spec._sph_harm_real(0, 0, 0, 0)/spec._sph_harm_real(1, 0, 0, 0), '-k', label="bolsig")

    for i, value in enumerate(args.sweep_values):
        data=run_data[i]

        lbl = args.sweep_param+"="+str(value)

        # spherical components plots

        for l_idx in range(num_sph_harm):

            plt.subplot(2, num_subplots, 1+l_idx)

            plt.plot(abs(data[-1,l_idx::num_sph_harm]),label=lbl)

            plt.title(label="l=%d"%l_idx)
            plt.yscale('log')
            plt.xlabel("Coeff. #")
            plt.ylabel("Coeff. magnitude")
            plt.grid(visible=True)
            if l_idx == 0:
                plt.legend()

            plt.subplot(2, num_subplots, num_subplots + 1 + l_idx)

            # if args.steady_state == False:
            #     plt.semilogy(ev,  abs(radial_initial[l_idx]), '-', label="Initial")

            color = next(plt.gca()._get_lines.prop_cycler)['color']
            plt.semilogy(ev,  abs(radial[i, l_idx]), '-', label=lbl, color=color)
            if l_idx == 0:
                plt.semilogy(ev,  abs(radial_base[i]), ':', label=lbl+" (base)", color=color)
            # plt.semilogy(ev, -radial[i, l_idx], 'o', label=lbl, color=color, markersize=3, markerfacecolor='white')


            plt.yscale('log')
            plt.xlabel("Energy, eV")
            plt.ylabel("Radial component")
            plt.grid(visible=True)
            # if l_idx == 0:
                # plt.legend()
            # plt.legend()

        # plt.subplot(2, num_subplots, num_sph_harm + 1)
        # temp = run_temp[i]
        # plt.plot(temp, label=lbl)

    plt.subplot(2, num_subplots, num_sph_harm + 1)
    #plt.plot(args.sweep_values, mu, 'o-', label='us')
    #plt.axhline(y=bolsig_mu, label='bolsig', color='k')
    plt.plot(args.sweep_values, abs(np.array(mu)/bolsig_mu-1), 'o-', label='us')
    plt.legend()
    plt.xlabel(args.sweep_param)
    plt.ylabel("Mean energy (eV)")

    if args.sweep_param != "radial_poly":
        plt.gca().ticklabel_format(useOffset=False)

    plt.subplot(2, num_subplots, num_sph_harm + 2)
    for col_idx, col in enumerate(args.collisions):
        if bolsig_rates[col_idx] != 0:
            plt.semilogy(args.sweep_values, abs(rates[col_idx]/bolsig_rates[col_idx]-1), 'o-', label='us '+col)
            # plt.axhline(y=0, label='bolsig '+col, color='k')
    plt.legend()
    plt.xlabel(args.sweep_param)
    plt.ylabel("Reaction rates")

    # if args.sweep_param != "radial_poly":
        # plt.gca().ticklabel_format(useOffset=False)


    plt.subplot(2, num_subplots, num_subplots + num_sph_harm + 1)
    # plt.plot(args.sweep_values, M, 'o-', label='us')
    # plt.axhline(y=bolsig_M, label='bolsig', color='k')
    plt.plot(args.sweep_values, abs(np.array(M)/bolsig_M-1), 'o-', label='us')
    plt.legend()
    plt.xlabel(args.sweep_param)
    plt.ylabel("Mobility *N (1/m/V/s) ")

    if args.sweep_param != "radial_poly":
        plt.gca().ticklabel_format(useOffset=False)

    plt.subplot(2, num_subplots, num_subplots + num_sph_harm + 2)
    # plt.plot(args.sweep_values, D, 'o-', label='us')
    # plt.axhline(y=bolsig_D, label='bolsig', color='k')
    plt.plot(args.sweep_values, abs(np.array(D)/bolsig_D-1), 'o-', label='us')
    plt.legend()
    plt.xlabel(args.sweep_param)
    plt.ylabel("Diffusion coefficient *N (1/m/s)")

    if args.sweep_param != "radial_poly":
        plt.gca().ticklabel_format(useOffset=False)

    # plt.subplot(2, num_subplots, num_subplots + num_sph_harm + 1)

    # lvls = np.linspace(0, np.amax(density_slice_initial[0]), 10)

    # plt.contour(vx, vz, density_slice_initial[-1], levels=lvls, linestyles='solid', colors='grey', linewidths=1) 
    # plt.contour(vx, vz, density_slice[-1], levels=lvls, linestyles='dotted', colors='red', linewidths=1)  
    # plt.gca().set_aspect('equal')
    
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(wspace=0.4)
    fig.suptitle("Collisions: " + str(args.collisions) + ", E = " + str(args.E_field) + ", polys = " + str(args.radial_poly) + ", Nr = " + str(args.NUM_P_RADIAL) + ", bscale = " + str(args.basis_scale) + " (sweeping " + args.sweep_param + ")")
    # plt.show()
    plt.savefig("maxwell_vs_bolsig_" + "_".join(args.collisions) + "_E" + str(args.E_field) + "_poly_" + str(args.radial_poly) + "_nr" + str(args.NUM_P_RADIAL) + "_bscale" + str(args.basis_scale) + "_sweeping_" + args.sweep_param + ".png")