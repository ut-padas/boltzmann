"""
Evolving the 0-d space Boltzmann eq. with collision and advection. 
"""
from ast import arg
import enum
import scipy
import basis
import spec_spherical as sp
import numpy as np
import collision_operator_spherical as colOpSp
import collisions 
import parameters as params
import utils as BEUtils
import argparse
import os
from scipy.integrate import ode
from time import perf_counter as time

class CollissionMode(enum.Enum):
    ELASTIC_ONLY=0
    ELASTIC_W_EXCITATION=1
    ELASTIC_W_IONIZATION=2
    ELASTIC_W_EXCITATION_W_IONIZATION=3

collisions.AR_NEUTRAL_N=3.22e22
collisions.MAXWELLIAN_N=1e18
collisions.AR_IONIZED_N=collisions.MAXWELLIAN_N

def advection_mat(spec: sp.SpectralExpansionSpherical, mw, vth):

    if args.radial_poly== "bspline":
        num_p  = spec._p+1
        num_sh = len(spec._sph_harm_lm)
        
        lmodes = list(set([l for l,_ in spec._sph_harm_lm]))
        num_l  = len(lmodes)
        l_max  = lmodes[-1]
        
        phimat = np.genfromtxt('sph_harm_del/phimat.dat',delimiter=',')
        psimat = np.genfromtxt('sph_harm_del/psimat.dat',delimiter=',')

        psimat= np.transpose(psimat[0:(l_max+1)**2, 0:(l_max+1)**2])
        phimat= np.transpose(phimat[0:(l_max+1)**2, 0:(l_max+1)**2])

        [gx, gw] = spec._basis_p.Gauss_Pn(basis.XlBSpline.get_num_q_pts(spec._p,spec._basis_p._sp_order,spec._basis_p._q_per_knot),True)
        
        Vr  = np.zeros(tuple([num_l,num_p])+gx.shape)
        Vdr = np.zeros(tuple([num_l,num_p])+gx.shape)
        for i,l in enumerate(lmodes):
            Vr[i]  = spec.Vq_r(gx,l)
            Vdr[i] = spec.Vdq_r(gx,l,d_order=1)
            
        mr = gx**2 
        mm1 = np.array([mr * Vr[pl,p,:] * Vdr[kl,k,:] for p in range(num_p) for k in range(num_p) for pl in range(num_l) for kl in range(num_l) ]).reshape(num_p,num_p,num_l,num_l,-1)
        mm1 = np.dot(mm1,gw)
        
        mr = gx
        mm2 = np.array([mr * Vr[pl,p,:] * Vr[kl,k,:] for p in range(num_p) for k in range(num_p) for pl in range(num_l) for kl in range(num_l) ]).reshape(num_p,num_p,num_l,num_l,-1)
        mm2 = np.dot(mm2,gw)

        advec_mat = np.zeros((num_p,num_sh,num_p,num_sh))
        for qs_idx,qs in enumerate(spec._sph_harm_lm):
            for lm in [(max(qs[0]-1,0), qs[1]), (min(qs[0]+1,l_max), qs[1])]:
                qs_mat = qs[0]**2+qs[0]+qs[1]
                lm_mat = lm[0]**2+lm[0]+lm[1]
                advec_mat[:,qs_idx,:,spec._sph_harm_lm.index([lm[0],lm[1]])] = mm1[:,:,qs[0],lm[0]] * psimat[qs_mat,lm_mat] - mm2[:,:,qs[0],lm[0]] * phimat[qs_mat,lm_mat]
                
            
        advec_mat = advec_mat.reshape(num_p*num_sh, num_p*num_sh)
        print("norm adv mat = %.8E"%np.linalg.norm(advec_mat))
        return advec_mat
    elif args.radial_poly=="maxwell":
        raise NotImplementedError
    

def spec_tail(cf, num_p, num_sh):
    return np.linalg.norm(cf[(num_p//2) * num_sh :])

def spec_tail_timeseries(cf, num_p, num_sh):
    return np.array([np.linalg.norm(cf[i, (num_p//2) * num_sh :])/len(cf[i, (num_p//2) * num_sh :]) for i in range(cf.shape[0])])

def svd_truncate(FOp,r_tol=1e-6):
    u, s, v = np.linalg.svd(FOp)
    FOp_svd = np.matmul(u, np.matmul(np.diag(s), v))  
    print("SVD rel. error : %.12E"%(np.linalg.norm(FOp_svd-FOp)/np.linalg.norm(FOp)))
    st = np.where(s/s[0]>r_tol) 
    s_k = st[0][-1]
    print("truncated at =%d out of %d "%(s_k,len(s)))
    FOp_svd = np.matmul(u[:, 0:s_k], np.matmul(np.diag(s[0:s_k]), v[0:s_k,:]))  
    print("SVD after truncation rel. error : %.12E"%(np.linalg.norm(FOp_svd-FOp)/np.linalg.norm(FOp)))
    return FOp_svd

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
    
def first_order_split(E : np.array, y0: np.array, cf : colOpSp.CollisionOpSP, t_final : float,  dt :float, v0: np.array):
    """
    E  : electric field for v-space advection 
    cf : collision operator
    dt : operator splitting time scale. 
    v0 : initial mean velocity
    """
    t_initial = 0.0
    t_curr    = 0.0
    spec      = cf._spec
    
    M=spec.compute_mass_matrix()
    Minv = np.linalg.inv(M)
    maxwellian = BEUtils.get_maxwellian_3d(VTH,collisions.MAXWELLIAN_N)

    def collission_rhs(t,y, C, n0):
        return n0 * np.matmul(C,y)

    f_t = y0
    g0  = collisions.eAr_G0()

    ode_solver = ode(collission_rhs,jac=None).set_integrator("dopri5",verbosity=1, rtol = 1e-14, atol=1e-14, nsteps=1e8)
    
    if(np.allclose(E,np.zeros_like(E))):
        "no splitting"

        g0.reset_scattering_direction_sp_mat()
        collission_mat=cf.assemble_mat(g0, maxwellian, VTH, v0)
        collission_mat= np.matmul(Minv,collission_mat)
        ode_solver.set_f_params(collission_mat,collisions.AR_NEUTRAL_N)
        ode_solver.set_initial_value(f_t,t_curr)
        ode_solver.integrate(ode_solver.t + t_final)
        f_t = ode_solver.y
        return [f_t,v0]
    else:
        "do first order operator split" 
        while (t_curr < t_final):
            "A - advection step"
            v0 = v0 - E * dt

            "C - collision step"
            g0.reset_scattering_direction_sp_mat()
            collission_mat=cf.assemble_mat(g0, maxwellian, VTH, v0)
            collission_mat= np.matmul(Minv,collission_mat)
            ode_solver.set_f_params(collission_mat,collisions.AR_NEUTRAL_N)
            ode_solver.set_initial_value(f_t,t_curr + dt)
            ode_solver.integrate(ode_solver.t + dt)

            if(not ode_solver.successful()):
                print("collission operator solve failed\n")
                assert(False)
            
            f_t = ode_solver.y
            
            t_curr+=2*dt

        
        return [f_t,v0]

def second_order_split(e_field, y0: np.array, cf : colOpSp.CollisionOpSP, t_final : float,  dt :float, v0: np.array, mw, vth, mode):
    """
    E  : electric field for v-space advection 
    cf : collision operator
    dt : operator splitting time scale. 
    v0 : initial mean velocity
    """
    t_initial = 0.0
    t_curr    = 0.0
    
    spec      = cf._spec
    q_by_m    = collisions.ELECTRON_CHARGE/ collisions.MASS_ELECTRON 
    ev_fac    = (collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT)

    mass_op   = BEUtils.mass_op(spec,None, 128, None, 1)
    avg_vop   = BEUtils.mean_velocity_op(spec,None, 128, None, 1)
    temp_op   = BEUtils.temp_op(spec,None, 128, None, 1)
    eavg_to_K = (2/(3*scipy.constants.Boltzmann))

    
    M    = spec.compute_mass_matrix()
    if args.radial_poly == "bspline":
        Minv = BEUtils.choloskey_inv(M) #BEUtils.block_jacobi_inv(M, 8)
    elif args.radial_poly == "maxwell" or args.radial_poly == "laguerre":
        Minv = np.linalg.inv(M)
        assert np.allclose(M,np.eye(M.shape[0]), rtol=1e-10, atol=1e-10), "mass matrix orthogonality test failed"
        
    if (mode == CollissionMode.ELASTIC_ONLY):
        def collission_rhs(t,y, C, n0):
            return n0 * np.matmul(C,y)
    elif (mode == CollissionMode.ELASTIC_W_IONIZATION):
        def collission_rhs(t,y, Ce, n0, Ci,ni):
            return n0 * np.matmul(Ce,y) + ni * np.matmul(Ci,y)
    else:
        raise NotImplementedError

    f_t    = y0
    current_vth      = vth 
    current_mw       = mw
    current_mass     = np.dot(f_t,mass_op)  * current_vth**3 * current_mw(0)
    current_temp     = np.dot(f_t,temp_op)  * current_vth**5 * current_mw(0) * 0.5 * collisions.MASS_ELECTRON * eavg_to_K / current_mass 
    
    print("Initial Ev : %.14E"%(current_temp * ev_fac))
    print("Initial mass : %.14E"%current_mass)

    g0  = collisions.eAr_G0()
    g2  = collisions.eAr_G2()

    ode_solver = ode(collission_rhs,jac=None).set_integrator("dopri5",verbosity=1, rtol = 1e-14, atol=1e-14, nsteps=1e8)
    
    q_pts_efield = 11
    [xt_1, wt_1] = basis.uniform_simpson( (0,  dt/2), q_pts_efield)
    [xt_2, wt_2] = basis.uniform_simpson( (dt/2, dt), q_pts_efield)
    
    sol_vec=list()
    temp_t =list()
    mass_t =list()
    mw_t   =list()
    v0_t   =list()
    while (t_curr < t_final):
        current_mass     = np.dot(f_t,mass_op)  * current_vth**3 * current_mw(0)
        current_temp     = BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec,f_t,current_mw,current_vth,None,None,None,current_mass,1, v0)
        
        current_vc_r     = np.dot(avg_vop[0],f_t) * current_vth**4 * current_mw(0)/current_mass
        current_vc_t     = np.dot(avg_vop[1],f_t) * current_vth**3 * current_mw(0)/current_mass
        current_vc_p     = np.dot(avg_vop[2],f_t) * current_vth**3 * current_mw(0)/current_mass

        current_vc_cart  = np.array(BEUtils.spherical_to_cartesian(current_vc_r, current_vc_t, current_vc_p)) + v0
        
        collisions.AR_IONIZED_N = current_mass
        print("time=%2E\t mass\t=%.12E temp\t=%.12E"%(t_curr, current_mass, current_temp * ev_fac))
        
        sol_vec.append(f_t)
        temp_t.append(current_temp)
        mass_t.append(current_mass)
        mw_t.append(current_mw)
        v0_t.append(v0)
        
        print("reassemble collision Op: (%.2E, %.2E,%.2E)"%(v0[0], v0[1], v0[2]))
        E = np.dot(e_field(t_curr + xt_1), wt_1)
        "A - advection step"
        v1 = v0 - (E * q_by_m/current_vth)

        adv_vc_cart  = np.array(BEUtils.spherical_to_cartesian(current_vc_r, current_vc_t, current_vc_p))/current_vth + v1

        "C - collision step"
        if(mode == CollissionMode.ELASTIC_ONLY):
            g0.reset_scattering_direction_sp_mat()
            t1 = time()
            Ce=cf.assemble_mat(g0, current_mw, current_vth, v1)
            Ce= np.matmul(Minv,Ce)
            t2 = time()
            print("Collission op. assembly time(s)= ",(t2-t1))
            ode_solver.set_f_params(Ce,collisions.AR_NEUTRAL_N)
            
        elif(mode == CollissionMode.ELASTIC_W_IONIZATION):
            g0.reset_scattering_direction_sp_mat()
            t1=time()
            Ce=cf.assemble_mat(g0, current_mw, current_vth, v1)
            Ce= np.matmul(Minv,Ce)
            
            g2.reset_scattering_direction_sp_mat()
            Ci=cf.assemble_mat(g2, current_mw, current_vth, v1)
            Ci= np.matmul(Minv,Ci)
            t2 = time()
            print("Collission op. assembly time(s)= ",(t2-t1))
            
            ode_solver.set_f_params(Ce,collisions.AR_NEUTRAL_N, Ci, collisions.AR_IONIZED_N)
            
        ode_solver.set_initial_value(f_t,0)
        ode_solver.integrate(dt)
        
        if(not ode_solver.successful()):
            print("collission operator solve failed\n")
            assert(False)
        
        f_t    = ode_solver.y

        new_vc_r    = np.dot(avg_vop[0],f_t) * current_vth**4 * current_mw(0)/current_mass
        new_vc_t    = np.dot(avg_vop[1],f_t) * current_vth**3 * current_mw(0)/current_mass
        new_vc_p    = np.dot(avg_vop[2],f_t) * current_vth**3 * current_mw(0)/current_mass
        col_vc_cart = np.array(BEUtils.spherical_to_cartesian(new_vc_r,new_vc_t,new_vc_p))/current_vth + v1

        # print("adv_vc", adv_vc_cart)
        # print("col_vc", col_vc_cart)
        # print("v1 b, ", v1)
        # correction for the vc difference between advection and collision. 
        b_shift      = col_vc_cart - adv_vc_cart
        v1          += b_shift
        # print("v1 a, ", v1)
        # print("bshift , ", b_shift)
        P_vc = BEUtils.vcenter_projection(spec,current_mw,current_vth, None, None, None, np.zeros_like(b_shift), Minv, 1)
        f_t  = np.matmul(P_vc, f_t)
        
        new_mass     = np.dot(f_t,mass_op)  * current_vth**3 * current_mw(0)
        new_temp     = BEUtils.compute_avg_temp(collisions.MASS_ELECTRON,spec,f_t,current_mw,current_vth,None,None,None,current_mass,1, v1)
        print("After collision step mass =%.12E temp=%.12E"%(new_mass,new_temp*ev_fac))
        if new_temp > current_temp:
            print("current temp: %.8E new temp %.8E"%(current_temp*ev_fac, new_temp*ev_fac))
            new_vth     = collisions.electron_thermal_velocity(new_temp)
            mw_new      = BEUtils.get_maxwellian_3d(new_vth,new_mass)
            P_mat       = BEUtils.thermal_projection(spec, current_mw, current_vth, mw_new, new_vth, None, None, None, 1)
            f_t         = np.matmul(P_mat,f_t)
            
            # num_p   = spec_sp._p + 1
            # sph_lm_modes = spec_sp._sph_harm_lm
            # num_sh  = len(sph_lm_modes)
            
            # current_hv = lambda vr,vt,vp : np.sum( np.array([f_t[k * num_sh + lm_idx] * spec.basis_eval_full(vr * (current_vth/new_vth) ,vt,vp,k,l,m) for k in range(num_p) for lm_idx, (l,m) in enumerate(sph_lm_modes)]),axis=0)
            # f_t        = BEUtils.function_to_basis(spec_sp,current_hv, mw_new, None, None, None )
            
            current_mw   = mw_new
            current_vth  = new_vth
            current_mass = np.dot(f_t,mass_op)  * current_vth**3 * current_mw(0)
            print("mass before %.14E mass after %.14E"%(new_mass,current_mass))
            
        
        # scale_fac = collisions.MAXWELLIAN_N/m0_t
        # f_t       = f_t * scale_fac
        
        "A- advection step"
        E = np.dot(e_field(t_curr + xt_2), wt_2)
        v0 = v1 - (E * q_by_m/current_vth)
        
        t_curr+=dt
    ts_steps = len(sol_vec)
    return np.array(sol_vec).reshape((ts_steps,-1)), np.array(temp_t).reshape(ts_steps,-1), np.array(mass_t).reshape(ts_steps,-1), mw_t, v0_t

def eularian_approach(Ez, y0: np.array, cf : colOpSp.CollisionOpSP, t_final : float,  dt :float, v0: np.array, mw, vth, mode):
    t_initial = 0.0
    t_curr    = 0.0
    
    spec      = cf._spec
    
    q_by_m    = -collisions.ELECTRON_CHARGE/ collisions.MASS_ELECTRON 
    ev_fac    = (collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT)
    
    mass_op   = BEUtils.mass_op(spec, None, 64, 2, 1)
    temp_op   = BEUtils.temp_op(spec, None, 64, 2, 1)
    avg_vop   = BEUtils.mean_velocity_op(spec, None, 64, 2, 1)
    eavg_to_K = (2/(3*scipy.constants.Boltzmann))

    current_mass     = np.dot(y0,mass_op) * vth**3 * mw(0)
    current_temp     = np.dot(y0,temp_op) * vth**5 * mw(0) * 0.5 * collisions.MASS_ELECTRON * eavg_to_K / current_mass
    print("Initial Ev   : %.14E"%(current_temp * ev_fac))
    print("Initial mass : %.14E"%current_mass)

    mass_initial    =  current_mass
    
    M    = spec.compute_mass_matrix()
    if args.radial_poly == "bspline":
        Minv = BEUtils.choloskey_inv(M) #BEUtils.block_jacobi_inv(M, 8)
    elif args.radial_poly == "maxwell" or args.radial_poly == "laguerre":
        raise NotImplementedError
        Minv = np.linalg.inv(M)
        assert np.allclose(M,np.eye(M.shape[0]), rtol=1e-10, atol=1e-10), "mass matrix orthogonality test failed"

    current_mw  = mw
    current_vth = vth
    
    if (mode == CollissionMode.ELASTIC_ONLY):
        def rhs(t, y, adv_mat, C, n0):
            return -Ez * np.matmul(adv_mat,y) + n0 * np.matmul(C,y)
    elif (mode == CollissionMode.ELASTIC_W_IONIZATION):
        def rhs(t, y, adv_mat,Ce, n0, Ci, ni):
            return -Ez * np.matmul(adv_mat,y) + n0 * np.matmul(Ce,y) + ni * np.matmul(Ci,y)
    
    ode_solver = ode(rhs,jac=None).set_integrator("dopri5",verbosity=1, rtol = 1e-14, atol=1e-14, nsteps=1e6)
    t1 = time()
    adv_mat = advection_mat(spec,current_mw,current_vth)
    adv_mat = np.matmul(Minv,adv_mat)
    t2 = time()
    print("Advection op. assembly time(s)= ",(t2-t1))
    #print("E scaling factor : ",(-Ez *q_by_m/current_vth))
    
    g0  = collisions.eAr_G0()
    g2  = collisions.eAr_G2()
    
    "C - collision operator"
    if(mode == CollissionMode.ELASTIC_ONLY):
        g0.reset_scattering_direction_sp_mat()
        t1=time()
        Ce = cf.assemble_mat(g0, current_mw, current_vth, v0)
        Ce = np.matmul(Minv,Ce)
        t2 = time()
        print("Collission op. assembly time(s)= ",(t2-t1))
        adv_vth = adv_mat *(q_by_m/current_vth)
        ode_solver.set_f_params(adv_vth, Ce,collisions.AR_NEUTRAL_N)
        
    elif(mode == CollissionMode.ELASTIC_W_IONIZATION):
        g0.reset_scattering_direction_sp_mat()
        t1=time()
        Ce=cf.assemble_mat(g0, current_mw, current_vth, v0)
        Ce= np.matmul(Minv,Ce)
        
        g2.reset_scattering_direction_sp_mat()
        Ci=cf.assemble_mat(g2, current_mw, current_vth, v0)
        Ci= np.matmul(Minv,Ci)
        t2 = time()
        print("Collission op. assembly time(s)= ",(t2-t1))
        adv_vth = adv_mat *(q_by_m/current_vth)
        ode_solver.set_f_params(adv_vth, Ce, collisions.AR_NEUTRAL_N, Ci, collisions.AR_IONIZED_N)
    
    ode_solver.set_initial_value(y0,0.0)
    t_step = 0
    total_steps = int(t_final/dt)
    sol_t = np.zeros((total_steps,y0.shape[0]))
    temp_t = np.zeros(total_steps)
    mass_t = np.zeros(total_steps)
    mw_t   = list()
    v0_t   = list()

    while t_step < total_steps:
        t_curr           = ode_solver.t
        h_t              = ode_solver.y
        
        current_mass     = np.dot(h_t,mass_op) * current_vth**3 * current_mw(0)
        current_temp     = np.dot(h_t,temp_op) * current_vth**5 * current_mw(0) * 0.5 * collisions.MASS_ELECTRON * eavg_to_K / current_mass

        if(mode == CollissionMode.ELASTIC_W_IONIZATION):
            mr = current_mass/mass_initial
            h_t=h_t/mr
            current_mass     = np.dot(h_t,mass_op) * current_vth**3 * current_mw(0)
            current_temp     = np.dot(h_t,temp_op) * current_vth**5 * current_mw(0) * 0.5 * collisions.MASS_ELECTRON * eavg_to_K / current_mass
            ode_solver.set_initial_value(h_t,t_curr)
            #collisions.AR_IONIZED_N = current_mass
            #ode_solver.set_f_params(adv_vth, Ce, collisions.AR_NEUTRAL_N, Ci, collisions.AR_IONIZED_N)

        vc_x             = np.dot(avg_vop[0],h_t) * current_vth**4 * (current_mw(0)/current_mass) /current_vth
        vc_y             = np.dot(avg_vop[1],h_t) * current_vth**4 * (current_mw(0)/current_mass) /current_vth
        vc_z             = np.dot(avg_vop[2],h_t) * current_vth**4 * (current_mw(0)/current_mass) /current_vth
        print("time:%.2E mass: %.10E temp: %.10E vc=(%.2E,%.2E,%.2E) adv vc_z=%.2E " %(t_curr, current_mass, current_temp * ev_fac, vc_x, vc_y,vc_z,0 + (Ez * t_curr * q_by_m/current_vth)))
        
        
        sol_t [t_step,:] = h_t
        temp_t[t_step]   = current_temp
        mass_t[t_step]   = current_mass
        mw_t.append(current_mw)
        v0_t.append(np.array([vc_x,vc_y,vc_z]))
        ode_solver.integrate(ode_solver.t + dt) 
        if not ode_solver.successful():
            print("ode integrator failed...")
            break
        
        t_curr+=dt
        t_step+=1
        h_t       = ode_solver.y
        new_mass  = np.dot(h_t,mass_op) * current_vth**3 * current_mw(0)
        new_temp  = np.dot(h_t,temp_op) * current_vth**5 * current_mw(0) * 0.5 * collisions.MASS_ELECTRON * eavg_to_K / new_mass
        
        # if new_temp > current_temp:
        #     print("current temp: %.8E new temp %.8E"%(current_temp*ev_fac, new_temp*ev_fac))
        #     new_vth      = collisions.electron_thermal_velocity(new_temp)
        #     mw_new       = BEUtils.get_maxwellian_3d(new_vth,new_mass)
        #     P_mat        = BEUtils.thermal_projection(spec, current_mw, current_vth, mw_new, new_vth, None, None, None, 1)
        #     h_t          = np.matmul(P_mat,h_t)
            
        #     current_mw   = mw_new
        #     current_vth  = new_vth
            
        #     ode_solver.set_initial_value(h_t,t_curr)

        #     "C - collision operator"
        #     if(mode == CollissionMode.ELASTIC_ONLY):
        #         g0.reset_scattering_direction_sp_mat()
        #         t1=time()
        #         Ce = cf.assemble_mat(g0, current_mw, current_vth, v0)
        #         Ce = np.matmul(Minv,Ce)
        #         #Ce=np.zeros_like(Minv)
        #         t2 = time()
        #         print("Collission op. assembly time(s)= ",(t2-t1))
        #         adv_vth = adv_mat * (q_by_m/current_vth)
        #         ode_solver.set_f_params(adv_vth, Ce, collisions.AR_NEUTRAL_N)
                
        #     elif(mode == CollissionMode.ELASTIC_W_IONIZATION):
        #         g0.reset_scattering_direction_sp_mat()
        #         t1=time()
        #         Ce=cf.assemble_mat(g0, current_mw, current_vth, v0)
        #         Ce= np.matmul(Minv,Ce)
                
        #         g2.reset_scattering_direction_sp_mat()
        #         Ci=cf.assemble_mat(g2, current_mw, current_vth, v0)
        #         Ci= np.matmul(Minv,Ci)
        #         t2 = time()
        #         print("Collission op. assembly time(s)= ",(t2-t1))
        #         adv_vth = adv_mat * (q_by_m/current_vth)
        #         ode_solver.set_f_params(adv_vth, Ce, collisions.AR_NEUTRAL_N, Ci, collisions.AR_IONIZED_N)
        
    return sol_t, temp_t, mass_t, mw_t, v0_t

parser = argparse.ArgumentParser()
parser.add_argument("-Nr", "--NUM_P_RADIAL"                   , help="Number of polynomials in radial direction", nargs='+', type=int, default=[4,8,16,32,64,128,256])
parser.add_argument("-T", "--T_END"                           , help="Simulation time", type=float, default=1e-6)
parser.add_argument("-dt", "--T_DT"                           , help="Simulation time step size ", type=float, default=1e-10)
parser.add_argument("-o",  "--out_fname"                      , help="output file name", type=str, default='coll_op')
parser.add_argument("-ts_tol", "--ts_tol"                     , help="adaptive timestep tolerance", type=float, default=1e-15)
parser.add_argument("-l_max", "--l_max"                       , help="max polar modes in SH expansion", type=int, default=0)
parser.add_argument("-c", "--collision_mode"                  , help="collision mode g- elastic with no E loss, g0-elastic", type=str, default="g0")
parser.add_argument("-ev", "--electron_volt"                  , help="initial electron volt", type=float, default=1.0)
parser.add_argument("-q_vr", "--quad_radial"                  , help="quadrature in r"        , type=int, default=270)
parser.add_argument("-q_vt", "--quad_theta"                   , help="quadrature in polar"    , type=int, default=2)
parser.add_argument("-q_vp", "--quad_phi"                     , help="quadrature in azimuthal", type=int, default=2)
parser.add_argument("-q_st", "--quad_s_theta"                 , help="quadrature in scattering polar"    , type=int, default=2)
parser.add_argument("-q_sp", "--quad_s_phi"                   , help="quadrature in scattering azimuthal", type=int, default=2)
parser.add_argument("-radial_poly", "--radial_poly"           , help="radial basis", type=str, default="maxwell")
parser.add_argument("-sp_order", "--spline_order"             , help="b-spline order", type=int, default=2)
parser.add_argument("-spline_qpts", "--spline_q_pts_per_knot" , help="q points per knots", type=int, default=11)
parser.add_argument("-vth_fac", "--vth_fac"                   , help="expand the function vth_fac *V_TH", type=float, default=1.0)
parser.add_argument("-vmode", "--vmode"                       , help="Semi-Lagrangian (SL) or Eulerian (eulerian)", type=str, default="eulerian")
parser.add_argument("-E", "--E"                               , help="E field", nargs='+', type=float, default=[0, 0, -1e2])
parser.add_argument("-e_freq", "--e_freq"                     , help="E field frequency", type=float, default=13e6)
parser.add_argument("-v_cfl", "--v_cfl"                       , help="v-space cfl", type=float, default=0.2)
args = parser.parse_args()
print(args)
coll_mode=CollissionMode.ELASTIC_ONLY
if args.collision_mode == "g0":
    coll_mode=CollissionMode.ELASTIC_ONLY
elif args.collision_mode == "g02":
    coll_mode=CollissionMode.ELASTIC_W_IONIZATION
else:
    raise NotImplementedError
run_data=list()
temperature=list()
ev           = np.linspace(args.electron_volt/50.,63.*args.electron_volt,1000)
eedf         = np.zeros((len(args.NUM_P_RADIAL),len(ev)))
eedf_initial = np.zeros((len(args.NUM_P_RADIAL),len(ev)))
temperature  = list()
mass_of_f    = list()
v_center     = list()
SPLINE_ORDER = args.spline_order
basis.BSPLINE_BASIS_ORDER=SPLINE_ORDER
basis.XLBSPLINE_NUM_Q_PTS_PER_KNOT=args.spline_q_pts_per_knot

for i, nr in enumerate(args.NUM_P_RADIAL):
    params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER = nr
    #params.BEVelocitySpace.SPH_HARM_LM = [[i,j] for i in range(args.l_max+1) for j in range(-i,i+1)]
    params.BEVelocitySpace.SPH_HARM_LM = [[i,j] for i in range(args.l_max+1) for j in range(0,1)]
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

    INIT_EV                        = args.electron_volt
    collisions.MAXWELLIAN_TEMP_K   = INIT_EV * collisions.TEMP_K_1EV
    collisions.ELECTRON_THEMAL_VEL = collisions.electron_thermal_velocity(collisions.MAXWELLIAN_TEMP_K) 
    cf    = colOpSp.CollisionOpSP(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,poly_type=r_mode)
    spec  = cf._spec
    VTH   = collisions.ELECTRON_THEMAL_VEL
    
    dt_e  = args.v_cfl * VTH * collisions.MASS_ELECTRON / collisions.ELECTRON_CHARGE / np.linalg.norm(args.E)
    args.T_DT = min(args.T_DT, dt_e)

    print("""===========================Parameters ======================""")
    print("\tMAXWELLIAN_N : ", collisions.MAXWELLIAN_N)
    print("\tELECTRON_THEMAL_VEL : ", VTH," ms^-1")
    #print("\tDT : ", params.BEVelocitySpace().VELOCITY_SPACE_DT, " s")
    print("\ttimestep size : ", args.T_DT, " s")
    print("""============================================================""")
    
    params.print_parameters()
    
    vth_factor = args.vth_fac
    VTH_C      = vth_factor * VTH
    maxwellian = BEUtils.get_maxwellian_3d(VTH_C, collisions.MAXWELLIAN_N)
    #hv         = lambda v,vt,vp : np.ones_like(v) +  np.tan(vt)  
    if (args.radial_poly == "maxwell" or args.radial_poly == "laguerre"):
        hv           = lambda v,vt,vp : (vth_factor**3) * np.exp(v**2  -(v*vth_factor)**2)
    if (args.radial_poly == "bspline"):
        hv           = lambda v,vt,vp : (vth_factor**3) * np.exp(-(v*vth_factor)**2)
    h_vec      = BEUtils.function_to_basis(spec,hv,maxwellian,None,None,None)
    #e_field    = lambda t : np.array([args.E[i] * np.sin(2* np.pi * args.e_freq * t) for i in range(3)])
    e_field    = lambda t : np.array([args.E[i] * np.ones_like(t) for i in range(3)])
    
    v0                               = np.array([0,0,0])
    if args.vmode   == "eulerian":
        data, temp_t, mass_t, mw_t, v0_t = eularian_approach(args.E[2], h_vec, cf, args.T_END, args.T_DT, v0,maxwellian, VTH_C, coll_mode)
        eedf[i]                          = BEUtils.get_eedf(ev, spec, data[-1,:], mw_t[-1], collisions.electron_thermal_velocity(temp_t[-1]), 1)
        eedf_initial[i]                  = BEUtils.get_eedf(ev, spec, data[0,:], maxwellian, VTH_C, 1)
    elif args.vmode == "SL":
        data, temp_t, mass_t, mw_t, v0_t = second_order_split(e_field, h_vec, cf, args.T_END, args.T_DT, v0,maxwellian, VTH_C, coll_mode)
        eedf[i]                          = BEUtils.get_eedf(ev, spec, data[-1,:], mw_t[-1], collisions.electron_thermal_velocity(temp_t[-1]), 1,v0_t[-1])
        eedf_initial[i]                  = BEUtils.get_eedf(ev, spec, data[0,:], maxwellian, VTH_C, 1)
    else:
        raise NotImplementedError

    mass_of_f.append(mass_t)
    temperature.append(temp_t * (collisions.BOLTZMANN_CONST/collisions.ELECTRON_VOLT))
    run_data.append(data)
    v_center.append(v0_t)
    
if (1):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 10), dpi=300)
    plt.subplot(2, 3, 1)
    plt.plot(abs(run_data[-1][0,:]), label = "t=0")
    for i, nr in enumerate(args.NUM_P_RADIAL):
        data=run_data[i]
        plt.plot(abs(data[1,:]),label="Nr=%d"%args.NUM_P_RADIAL[i])

    plt.yscale('log')
    plt.xlabel("Coeff. #")
    plt.ylabel("Coeff. magnitude")
    plt.grid(True, which="both", ls="-")
    plt.legend()

    plt.subplot(2, 3, 2)
    for i, nr in enumerate(args.NUM_P_RADIAL):
        data=run_data[i]
        plt.plot(abs(data[-1,:]),label="Nr=%d"%args.NUM_P_RADIAL[i])

    plt.yscale('log')
    plt.xlabel("Coeff. #")
    plt.ylabel("Coeff. magnitude")
    plt.grid(True, which="both", ls="-")
    plt.legend()

    #ts= np.linspace(0,args.T_END, int(args.T_END/args.T_DT))
    ts= np.linspace(0,args.T_END, run_data[-1].shape[0])
    plt.subplot(2, 3, 3)
    for i, nr in enumerate(args.NUM_P_RADIAL):
        data_last=run_data[-1]
        data=run_data[i]
        plt.plot(ts, np.array([u[2] for u in v_center[i]]), label="Nr=%d"%args.NUM_P_RADIAL[i])

    plt.xlabel("time (s)")
    plt.ylabel("vc_z coordinate")
    plt.grid(True, which="both", ls="-")
    plt.legend()

    plt.subplot(2, 3, 4)
    
    if (args.radial_poly == "maxwell"):
        for i, nr in enumerate(args.NUM_P_RADIAL):
            plt.plot(ts, spec_tail_timeseries(run_data[i],nr, len(params.BEVelocitySpace.SPH_HARM_LM)),label="Nr=%d"%args.NUM_P_RADIAL[i])

        plt.legend()
        plt.yscale('log')
        plt.xlabel("time (s)")
        plt.ylabel("spectral tail l2(h[nr/2: ])")
        plt.grid(True, which="both", ls="-")
        
    elif (args.radial_poly == "bspline"):
        for i, nr in enumerate(args.NUM_P_RADIAL):
            plt.plot(ev, np.abs(eedf[i]-eedf[-1]),label="Nr=%d"%args.NUM_P_RADIAL[i])

        plt.yscale('log')
        #plt.xscale('log')
        plt.legend()
        plt.xlabel("energy (ev)")
        plt.ylabel("error in eedf")
        plt.grid(True, which="both", ls="-")


    plt.subplot(2, 3, 5)
    plt.plot(ev, eedf_initial[-1], label="initial")
    for i, nr in enumerate(args.NUM_P_RADIAL):
        data=run_data[i]
        plt.plot(ev, abs(eedf[i]),label="Nr=%d"%args.NUM_P_RADIAL[i])
        #plt.plot(ev, abs(eedf_initial[i]),label="initial Nr=%d"%args.NUM_P_RADIAL[i])

    # plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlabel("energy (ev)")
    plt.ylabel("eedf")
    plt.grid(True, which="both", ls="-")

    plt.subplot(2, 3, 6)
    for i, nr in enumerate(args.NUM_P_RADIAL):
        data=temperature[i]
        plt.plot(ts, temperature[i], label="Nr=%d"%(args.NUM_P_RADIAL[i]))

    # plt.xscale('log')
    # plt.yscale('log')
    plt.legend()
    plt.xlabel("time (s)")
    plt.ylabel("temperature (eV)")
    plt.grid(True, which="both", ls="-")

    plt.tight_layout()
    #plt.show()
    plt.savefig("%s_vspace_coeff.png"%(args.out_fname))
    

if(0):
    fig = plt.figure(figsize=(10, 4), dpi=300)
    if (args.radial_poly == "maxwell"):
        ts= np.linspace(0,args.T_END, run_data[i].shape[0])
        plt.subplot(1, 2, 1)
        for i, nr in enumerate(args.NUM_P_RADIAL):
            plt.plot(ts, spec_tail_timeseries(run_data[i],nr+1, len(params.BEVelocitySpace.SPH_HARM_LM)),label="Nr=%d"%args.NUM_P_RADIAL[i])

        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel("time (s)")
        plt.ylabel("tail l2(h[nr/2: ])")

    elif (args.radial_poly == "bspline"):
        plt.subplot(1, 2, 1)
        for i, nr in enumerate(args.NUM_P_RADIAL):
            plt.plot(ev, np.abs(eedf[i]-eedf[-1]),label="Nr=%d"%args.NUM_P_RADIAL[i])

        plt.yscale('log')
        plt.xscale('log')
        plt.legend()
        plt.xlabel("energy (ev)")
        plt.ylabel("error in eedf")

    plt.subplot(1, 2, 2)
    plt.plot(ev, eedf_initial, label="initial")
    for i, nr in enumerate(args.NUM_P_RADIAL):
        data=run_data[i]
        plt.plot(ev, eedf[i],label="Nr=%d"%(nr))

    #plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlabel("energy (ev)")
    plt.ylabel("eedf")
    plt.tight_layout()

    #plt.show()
    plt.savefig("%s.png"%(args.out_fname))
    plt.close()

    fig = plt.figure(figsize=(10, 4), dpi=300)
    plt.subplot(1,2,1)
    polar_plot, _ , __ = constant_r_eval(spec,run_data[-1][0,:],1)
    plt.imshow(polar_plot,extent=[0,2*np.pi,0,np.pi])
    plt.colorbar()
    plt.xlabel("azimuthal angle")
    plt.ylabel("polar angle")
    plt.title("initial, v_r = 1")

    plt.subplot(1,2,2)
    polar_plot, _ , __ = constant_r_eval(spec,run_data[-1][-1,:],1)
    plt.imshow(polar_plot,extent=[0,2*np.pi,0,np.pi])
    plt.colorbar()
    plt.clim(np.min(polar_plot),np.max(polar_plot)*1.01)
    plt.xlabel("azimuthal angle")
    plt.ylabel("polar angle")
    plt.title("final, v_r = 1")
    plt.savefig("%s_const_r.png"%(args.out_fname))