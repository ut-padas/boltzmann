"""
@brief: 0d-space, 3d-space boltzmann eq solver
"""
from   time import perf_counter as time, sleep
import utils as bte_utils
import spec_spherical as sp
import numpy as np
import collision_operator_spherical as collOpSp
import collisions 
import parameters as params
import basis
import argparse
import bolsig
import sys
import scipy.interpolate
import scipy.constants
import cross_section


def newton_solve(spec_sp, h0, res_func, jac_func, f1, Qmat, Rmat, mass_op, rtol = 1e-5, atol=1e-5, max_iter=1000):

    k_vec     = spec_sp._basis_p._t
    dg_idx    = spec_sp._basis_p._dg_idx
    sp_order  = spec_sp._basis_p._sp_order

    num_p     = spec_sp._p + 1
    num_sh    = len(spec_sp._sph_harm_lm)

    abs_error       = 1.0
    rel_error       = 1.0 
    iteration_steps = 0        

    fb_prev  = np.dot(Rmat, h0)
    f1p      = f1
    h_prev   = f1p + np.dot(Qmat,fb_prev)

    while ((rel_error> rtol and abs_error > atol) and iteration_steps < max_iter):
        Lmat      =  jac_func(h_prev)
        rhs_vec   = -res_func(h_prev)
        abs_error = np.linalg.norm(rhs_vec)

        p      = np.linalg.solve(Lmat, rhs_vec) #np.linalg.lstsq(Lmat, rhs_vec, rcond=1e-16 /np.linalg.cond(Lmat))[0]
        p      = np.dot(Qmat,p)

        alpha  = 1e0
        is_diverged = False

        while (np.linalg.norm(res_func(h_prev + alpha * p))  >  abs_error):
            alpha*=0.5
            if alpha < 1e-30:
                is_diverged = True
                break
        
        if(is_diverged):
            print("Iteration ", iteration_steps, ": Residual =", abs_error, "line search step size becomes too small")
            break
        
        h_curr      = h_prev + alpha * p
        
        if iteration_steps % 10 == 0:
            rel_error = np.linalg.norm(h_prev-h_curr)/np.linalg.norm(h_curr)
            print("Iteration ", iteration_steps, ": abs residual = %.8E rel residual=%.8E mass =%.8E"%(abs_error, rel_error, np.dot(mass_op, h_prev)))
        
        #fb_prev      = np.dot(Rmat,h_curr)
        h_prev       = h_curr #f1p + np.dot(Qmat,fb_prev)
        iteration_steps+=1

    print("Nonlinear solver (1) atol=%.8E , rtol=%.8E"%(abs_error, rel_error))
    return h_curr, abs_error, rel_error
    
class bte_0d3v():

    def __init__(self, args) -> None:
        """
        Initialize the 0d-space 3v Boltzmann solver
        """

        """parameters for the problem"""
        self._args = args
        self._n0   = args.n0
        self._tgK  = args.Tg
        
        self._collision_names              = list()
        self._coll_list                    = list()
        self._avail_species                = cross_section.read_available_species(self._args.collisions)
        cross_section.CROSS_SECTION_DATA   = cross_section.read_cross_section_data(self._args.collisions)
        self._cross_section_data           = cross_section.CROSS_SECTION_DATA
        print("==========read collissions===========")
        collision_count = 0
        for col_str, col_data in self._cross_section_data.items():
            print(col_str, col_data["type"])
            g = collisions.electron_heavy_binary_collision(col_str, collision_type=col_data["type"])
            self._coll_list.append(g)
            self._collision_names.append("C%d"%(collision_count)) 
            collision_count+=1
        print("=====================================")
        print("number of total collisions = %d " %(len(self._coll_list)))
        self.num_collisions = len(self._coll_list)
        
        species       = cross_section.read_available_species(self._args.collisions)
        mole_fraction = np.array(self._args.ns_by_n0)[0:len(species)]
        assert np.allclose(np.sum(mole_fraction),1), "mole fractions does not add up to 1.0"
        
    
    def get_collision_list(self):
        return self._coll_list    
    
    def get_collision_names(self):
        return self._collision_names
    
    def get_cross_section_data(self):
        return self._cross_section_data
    
    def run_bolsig_solver(self):
        """simple function to run the bolsig code"""
        self._bolsig_data = dict()
        try:
            bolsig.run_bolsig(self._args)
            [bolsig_ev, bolsig_f0, bolsig_a, bolsig_E, bolsig_mu, bolsig_M, bolsig_D, bolsig_rates,bolsig_cclog] = bolsig.parse_bolsig(self._args.bolsig_dir+"argon.out", self.num_collisions)

            self._bolsig_data["ev"] = bolsig_ev
            self._bolsig_data["f0"] = bolsig_f0
            self._bolsig_data["f1"] = np.abs(bolsig_f0 * bolsig_a / np.sqrt(3))

            self._bolsig_data["energy"]    = bolsig_mu
            self._bolsig_data["mobility"]  = bolsig_M
            self._bolsig_data["diffusion"] = bolsig_D
            self._bolsig_data["rates" ]    = bolsig_rates
            self._bolsig_data["cc_log"]    = bolsig_cclog

        except:
           print(self._args.bolsig_dir+"argon.out file not found due to Bolsig+ run failure")
           import traceback
           traceback.print_exc()
           sys.exit(0)


        print("bolsig temp      = %.8E"%((bolsig_mu /1.5)))
        print("bolsig mobility  = %.8E"%((bolsig_M)))
        print("bolsig diffusion = %.8E"%((bolsig_D)))
        #print("bolsig coulomb logarithm = %.8E"%((bolsig_cclog)))
        print("bolsig collision rates")
        for  col_idx, col in enumerate(self._coll_list):
            print("%s = %.8E"%(self._collision_names[col_idx], bolsig_rates[col_idx]))

        #print("setting PDE code ev va")
        self._args.electron_volt = (bolsig_mu/1.5)
        self._args.ev_max        = (1.2) * bolsig_ev[-1]
        return 

    def setup(self):
        """
        setup for the boltzmann solver
            1). Compute the mass matrix, and its inverse
            2). Compute the advection matrix
            3). Compute the collision matrix
            4). Compute the Coulomb collision tensors
        """

        args                = self._args
        self._mw_ne         = 1.0
        self._mw_tmp        = args.electron_volt * collisions.TEMP_K_1EV
        self._vth           = collisions.electron_thermal_velocity(self._mw_tmp)
        self._mw            = bte_utils.get_maxwellian_3d(self._vth, self._mw_ne)
        self._c_gamma       = np.sqrt(2*collisions.ELECTRON_CHARGE_MASS_RATIO)

        sig_pts   =  list()
        for col_idx, g in enumerate(self._coll_list):
            g  = self._coll_list[col_idx]
            if g._reaction_threshold != None and g._reaction_threshold >0:
                sig_pts.append(g._reaction_threshold)
        
        self._sig_pts = np.sort(np.array(list(set(sig_pts))))
        self._sig_pts = np.sqrt(self._sig_pts) * self._c_gamma / self._vth

        ev_range = ((0*self._vth/self._c_gamma)**2, self._args.ev_max)
        k_domain = (np.sqrt(ev_range[0]) * self._c_gamma / self._vth, np.sqrt(ev_range[1]) * self._c_gamma / self._vth)

        print("target ev range : (%.4E, %.4E) ----> knots domain : (%.4E, %.4E)" %(ev_range[0], ev_range[1], k_domain[0],k_domain[1]))
        if(self._sig_pts is not None):
            print("singularity pts : ", self._sig_pts, "v/vth and" , (self._sig_pts * self._vth/self._c_gamma)**2, "eV")

        """currently using the azimuthal symmetry case"""
        self._sph_lm   = [[l,0] for l in range(args.l_max+1)]
        
        bb                    = basis.BSpline(k_domain, self._args.sp_order, self._args.NUM_P_RADIAL + 1, sig_pts=self._sig_pts, knots_vec=None, dg_splines=args.use_dg)
        spec_sp               = sp.SpectralExpansionSpherical(self._args.NUM_P_RADIAL, bb, self._sph_lm)
        spec_sp._num_q_radial = bb._num_knot_intervals * self._args.spline_qpts

        self._collision_op    = collOpSp.CollisionOpSP(spec_sp)
        
        Mmat = spec_sp.compute_mass_matrix()
        Minv = spec_sp.inverse_mass_mat(Mmat=Mmat)
        mm_inv_error=np.linalg.norm(np.matmul(Mmat,Minv)-np.eye(Mmat.shape[0]))/np.linalg.norm(np.eye(Mmat.shape[0]))

        print("mass matrix M cond(M)      = %.4E"  %np.linalg.cond(Mmat))
        print("|I-M M^{-1}| error         = %.12E" %(mm_inv_error))

        if args.ee_collisions==1 and args.ion_deg > 0 and args.use_dg==1:
            print("Coulomb collisions are not supported in the DG mode, switching to CG mode")
            args.use_dg = 0
        
        num_p              = spec_sp._p + 1
        num_sh             = len(spec_sp._sph_harm_lm)

        self._spec_sp      = spec_sp
        self._mass_mat     = Mmat
        self._inv_mass_mat = Minv

        vth                = self._vth
        maxwellian         = self._mw
        collOp             = self._collision_op 

        gx, gw             = spec_sp._basis_p.Gauss_Pn(spec_sp._num_q_radial)
        sigma_m            = np.zeros(len(gx))
        c_gamma            = np.sqrt(2*collisions.ELECTRON_CHARGE_MASS_RATIO)
        gx_ev              = (gx * vth / c_gamma)**2

        FOp      = 0
        sigma_m  = 0
        
        t1 = time()
        for col_idx, (col_str, col_data) in enumerate(self._cross_section_data.items()):
            g         = self._coll_list[col_idx]
            g.reset_scattering_direction_sp_mat()
            mole_idx  = self._avail_species.index(col_data["species"]) 
            FOp       = FOp + args.ns_by_n0[mole_idx] * self._n0 * collOp.assemble_mat(g, maxwellian, vth, tgK=args.Tg)
            sigma_m  += g.total_cross_section(gx_ev)

        t2 = time()
        print("Assembled the collision op. for Vth : ", vth)
        print("Collision Operator assembly time (s): ",(t2-t1))

        gg_list = self._coll_list

        self._mass_op               = bte_utils.mass_op(spec_sp, 1)
        self._temp_op               = bte_utils.temp_op(spec_sp, 1)
        self._temp_op_ev            = self._temp_op * 0.5 * scipy.constants.electron_mass * (2/3/scipy.constants.Boltzmann) / collisions.TEMP_K_1EV

        mw  = self._mw
        self._mobility_op           = bte_utils.mobility_op(spec_sp, mw, vth)
        self._diffusion_op          = bte_utils.diffusion_op(spec_sp, self._coll_list, mw, vth)
        self._rr_op                 = np.zeros((len(gg_list),num_p))

        for col_idx, g in enumerate(gg_list):
            self._rr_op[col_idx,:] = bte_utils.reaction_rates_op(spec_sp, [g], mw, vth)

        if args.use_dg == 1 : 
            advmat, eA, qA = spec_sp.compute_advection_matix_dg(advection_dir=-1.0)
            qA     = np.kron(np.eye(spec_sp.get_num_radial_domains()), np.kron(np.eye(num_p), qA))
        else:
            print("using cg version")
            advmat = spec_sp.compute_advection_matix()
            qA     = np.eye(advmat.shape[0])

        FOp   = np.matmul(np.transpose(qA), np.matmul(FOp, qA))
        
        Cmat  = FOp 
        Emat  = advmat

        self._Cmat   = Cmat
        self._Emat   = Emat
        self._Qa     = qA

        if args.ee_collisions==1:
            print("electron-electron collision setup")
            t1               = time()
            hl_op, gl_op     =  collOp.compute_rosenbluth_potentials_op(maxwellian, vth, 1, Minv)
            cc_op_a, cc_op_b =  collOp.coulomb_collision_op_assembly(maxwellian, vth)
            cc_op            =  np.dot(cc_op_a, hl_op) + np.dot(cc_op_b, gl_op)

            cc_op            = np.dot(cc_op,qA)
            cc_op            = np.dot(np.swapaxes(cc_op,1,2),qA)
            cc_op            = np.swapaxes(cc_op,1,2)

            cc_op            =  np.dot(np.transpose(qA), cc_op.reshape((num_p*num_sh,-1))).reshape((num_p * num_sh, num_p * num_sh, num_p * num_sh))
            cc_op            =  np.dot(Minv,cc_op.reshape((num_p*num_sh,-1))).reshape((num_p * num_sh, num_p * num_sh, num_p * num_sh))

            cc_op_l1         =  cc_op
            cc_op_l2         =  np.swapaxes(cc_op, 1, 2)

            self._cc_op      =  cc_op
            t2 = time()
            print("Coulomb collision Op. assembly %.8E"%(t2-t1))

        ev        = self._bolsig_data["ev"]
        bolsig_f0 = self._bolsig_data["f0"]
        bolsig_a  = (self._bolsig_data["f1"] / bolsig_f0) * np.sqrt(3)

        f0_cf = scipy.interpolate.interp1d(ev, bolsig_f0, kind='cubic', bounds_error=False, fill_value=(bolsig_f0[0],bolsig_f0[-1]))
        fa_cf = scipy.interpolate.interp1d(ev, bolsig_a,  kind='cubic', bounds_error=False, fill_value=(bolsig_a[0],bolsig_a[-1]))
        ff    = lambda v,vt,vp : f0_cf(.5*(v* vth)**2/collisions.ELECTRON_CHARGE_MASS_RATIO) * (1. - fa_cf(.5*(v* vth)**2/collisions.ELECTRON_CHARGE_MASS_RATIO)*np.cos(vt))

        hh    =  bte_utils.function_to_basis(spec_sp, ff, maxwellian, spec_sp._num_q_radial, 2, 2, Minv=Minv)
        hh    =  bte_utils.normalized_distribution(spec_sp, self._mass_op, hh, maxwellian, vth)

        self._bolsig_data["bolsig_hh"] = hh

        return
        
    def initialize(self, init_type = "maxwellian"):
        """
        Initialize the grid to Maxwell-Boltzmann distribution
        """
        args      = self._args
        Minv      = self._inv_mass_mat
        spec_sp   = self._spec_sp
        mw        = self._mw
        vth       = self._vth

        mass_op   = self._mass_op 
        temp_op   = self._temp_op_ev

        if init_type == "maxwellian":
            v_ratio = 1.0 #np.sqrt(1.0/args.basis_scale)
            hv      = lambda v,vt,vp : (1/np.sqrt(np.pi)**3) * np.exp(-((v/v_ratio)**2)) / v_ratio**3
            h_init  = bte_utils.function_to_basis(spec_sp,hv,mw, spec_sp._num_q_radial, 2, 2, Minv=Minv)
        elif init_type == "anisotropic":
            v_ratio = 1.0 #np.sqrt(1.0/args.basis_scale)
            hv      = lambda v,vt,vp : (1/np.sqrt(np.pi)**3 ) * (np.exp(-((v/v_ratio)**2)) / v_ratio**3) * (1 + np.cos(vt))
            h_init  = bte_utils.function_to_basis(spec_sp,hv,mw, spec_sp._num_q_radial, 4, 2, Minv=Minv)
        else:
            raise NotImplementedError
        
        m0 = np.dot(mass_op,h_init) 
        print("initial data")
        print("  mass = %.8E"%(m0))
        print("  temp = %.8E"%(np.dot(temp_op,h_init) * vth**2 /m0))
        return h_init

    def steady_state_solver_two_term(self,h_init=None):

        args = self._args
        if args.ee_collisions == 1 or args.use_dg==1:
            raise NotImplementedError
        
        mass_op = self._mass_op
        mw      = self._mw
        vth     = self._vth

        args    = self._args
        Cmat    = self._Cmat
        Emat    = self._Emat * (args.E_field / vth) * collisions.ELECTRON_CHARGE_MASS_RATIO
        Mmat    = self._mass_mat
        Minv    = self._inv_mass_mat


        Cmat_p_Emat = Cmat + Emat
        Cmat_p_Emat = np.matmul(Minv, Cmat_p_Emat)

        spec_sp  = self._spec_sp
        gx, gw   = spec_sp._basis_p.Gauss_Pn(spec_sp._num_q_radial)
        c_gamma  = np.sqrt(2*collisions.ELECTRON_CHARGE_MASS_RATIO)

        sigma_m  = 0
        gx_ev    = (gx * vth / c_gamma)**2
        for col_idx, col in enumerate(args.collisions):
            g = self._coll_list[col_idx]
            g.reset_scattering_direction_sp_mat()
            assert col == g._col_name
            print("collision %d included %s"%(col_idx, col))
            
            if "g0NoLoss" == col:
                sigma_m  += g.total_cross_section(gx_ev)
            elif "g0ConstNoLoss" == col:
                sigma_m  += g.total_cross_section(gx_ev)
            elif "g0" in col:
                sigma_m  += g.total_cross_section(gx_ev)
            elif "g1" in col:
                sigma_m  += g.total_cross_section(gx_ev)
            elif "g2" in col:
                sigma_m  += g.total_cross_section(gx_ev)
            else:
                print("%s unknown collision"%(col))
                sys.exit(0)

        num_p  = spec_sp._p + 1
        num_sh = len(spec_sp._sph_harm_lm)
        
        FOp       =  Cmat
        FOp       =  FOp[0::num_sh, 0::num_sh]
        Minv      = Minv[0::num_sh, 0::num_sh]
        Mmat      = Mmat[0::num_sh, 0::num_sh]

        u         = mass_op 
        sigma_m   = sigma_m * np.sqrt(gx_ev) * c_gamma * self._n0
        u         = u[0::num_sh]
        
        # g         = collisions.eAr_G2(cross_section=col)
        # Wmat      = self._n0 * self._collision_op.assemble_mat(g, mw, vth)[0::num_sh, 0::num_sh]

        g_rate  = np.matmul(u.reshape(1,-1), np.matmul(Minv, FOp))
        # g_rate2 = np.matmul(u.reshape(1,-1), np.matmul(Minv, FOp))
        # g_rate1 = np.zeros(FOp.shape[0])
        # for col_idx, col in enumerate(args.collisions):
        #     if "g2" in col:
        #         g  = collisions.eAr_G2(cross_section=col)
        #         g.reset_scattering_direction_sp_mat()
        #         g_rate1 += self._n0 *  bte_utils.growth_rates_op(spec_sp, [g], mw, vth) * np.sqrt(4 * np.pi)

        # print(g_rate1)
        # print(g_rate)
        # print(g_rate1/g_rate)
        # print(np.abs(g_rate - g_rate1)/g_rate)
        # g_rate = g_rate1

        Ev     = args.E_field * collisions.ELECTRON_CHARGE_MASS_RATIO
        Vq_d1  = spec_sp.Vdq_r(gx,0,1,1)
        Wt     = Mmat
    
        ### if we need to apply some boundary conditions. 
        # fx      = k_vec[num_p -4 :  num_p] 
        # #fd_coef = [0.5, -2, 1.5]
        # fd_coef = [-1/3, 3/2, -3, 11/6]
        # bdy_op  = 0
        # for ii in range(4):
        #     bdy_op+=np.array([ fd_coef[ii] * spec_sp.basis_eval_radial(fx[ii], p, 0) for p in range(num_p)])
        # dx_bdy = (fx[-1]-fx[-2])
        # bdy_op = bdy_op/dx_bdy

        def assemble_f1_op(h_prev):
            g_rate_mu = np.dot(g_rate, h_prev)
            w_ev =  1./(sigma_m + g_rate_mu)
            Bt = np.array([ w_ev * (gx**2) * spec_sp.basis_eval_radial(gx, p, 0) * Vq_d1[k,:] for p in range(num_p) for k in range(num_p)]).reshape((num_p, num_p, -1))
            Bt = np.dot(Bt, gw) 
            return Bt

        def residual_op(x):
            g_rate_mu = np.dot(g_rate, x) 
            w_ev =  1./(sigma_m + g_rate_mu)
            tmp  = np.sqrt(w_ev * gw) * gx * Vq_d1
            At   = np.matmul(tmp, np.transpose(tmp))

            #print("mass = %.8E vi =%.8E %.8E, %.8E" %(np.dot(u,x), g_rate_mu, np.dot(FOp, x), np.dot(Wt * g_rate_mu, x)))

            # fx = k_vec[dg_idx[-1] + sp_order]
            # assert fx == k_vec[-1], "flux assembly face coords does not match at the boundary"
            # At[dg_idx[-1] , dg_idx[-1]] += -fx**2 * spec_sp.basis_derivative_eval_radial(fx - 2 * np.finfo(float).eps, dg_idx[-1],0,1) / (sigma_bdy + g_rate_mu) 
            
            Rop = -(1/3) * ((Ev/vth)**2) * (At) + FOp -  Wt * g_rate_mu
            return Rop

        def residual_func(x):
            Rop = residual_op(x)
            y   = np.dot(Rop,x)
            return y

        def jacobian_func(x):
            g_rate_mu = np.dot(g_rate, x) 
            w_ev =  1./(sigma_m + g_rate_mu)
            tmp  = np.sqrt(w_ev * gw) * gx * Vq_d1
            At   = np.matmul(tmp, np.transpose(tmp))

            # fx = k_vec[dg_idx[-1] + sp_order]
            # assert fx == k_vec[-1], "flux assembly face coords does not match at the boundary"
            # At[dg_idx[-1] , dg_idx[-1]] += -fx**2 * spec_sp.basis_derivative_eval_radial(fx - 2 * np.finfo(float).eps, dg_idx[-1],0,1) / (sigma_bdy + g_rate_mu) 
            
            Ji = (-(1/3) * ((Ev/vth)**2) * (At) + FOp -  2 * Wt * g_rate_mu)
            return Ji
        
        def solver_0(h0, rtol = 1e-5 , atol=1e-5, max_iter=1000):
            nn              = FOp.shape[0]
            Ji              = np.zeros((nn+1,nn))
            Rf              = np.zeros(nn+1)
        
            abs_tol = atol * 2
            rel_tol = rtol * 2 
            iteration_steps = 0
            h_prev          = h0[0::num_sh]
            rf_initial      = np.linalg.norm(residual_func(h_prev))
            while (abs_tol > atol and rel_tol > rtol and iteration_steps < max_iter):
                Rf              = residual_func(h_prev)
                #Rf              = np.append(Rf,np.dot(u,h_prev)-1) 
                abs_tol         = np.linalg.norm(Rf)
                rel_tol         = abs_tol/rf_initial
                         
                if(iteration_steps%1==0):
                    print("iteration %d abs. res l2 = %.8E rel. res = %.8E mass = %.8E"%(iteration_steps, abs_tol, rel_tol, np.dot(u,h_prev)))
                    
                Ji = jacobian_func(h_prev)
                #Ji = np.vstack((Ji,u))
                #p  = np.matmul(np.linalg.pinv(Ji,rcond=1e-16/np.linalg.cond(Ji)), -Rf)
                p = np.linalg.solve(Ji, -Rf)
                
                alpha=1e0
                is_diverged = False
                while (np.linalg.norm(residual_func(h_prev + alpha *p)) >= abs_tol):
                    alpha*=0.5
                    if alpha < 1e-16:
                        is_diverged = True
                        break
                
                if(is_diverged):
                    print("iteration %d abs. res l2 = %.8E rel. res = %.8E"%(iteration_steps, abs_tol, rel_tol))
                    print("line search step size becomes too small")
                    break

                h_prev += p*alpha
                h_prev  /= np.dot(u,h_prev)
                iteration_steps = iteration_steps + 1
            
            hh = spec_sp.create_vec().reshape(num_p * num_sh)
            hh[0::num_sh] = h_prev

            Bt = assemble_f1_op(h_prev)
            hh[1::num_sh] = (Ev/np.sqrt(3)) * np.matmul(Minv,np.matmul(Bt,h_prev)) / vth

            return hh, abs_tol, rel_tol

        h_bolsig             = self._bolsig_data["bolsig_hh"]
        if h_init==None:
            h_init                = self.initialize(init_type="maxwellian")
            h_init                = h_init/np.dot(u,h_init[0::num_sh])
            

        hh, abs_tol, rel_tol  = solver_0(h_init, rtol=1e-16, atol=1e-6, max_iter=1000)
        h_pde                 = bte_utils.normalized_distribution(spec_sp, mass_op, hh, mw, vth)
        norm_L2               = lambda vv : np.dot(vv, np.dot(Mmat, vv))

        solution_vector = np.zeros((2, h_init.shape[0]))
        solution_vector[0,:] = bte_utils.normalized_distribution(spec_sp, mass_op, h_init, mw, vth)
        solution_vector[1,:] = bte_utils.normalized_distribution(spec_sp, mass_op, hh,     mw, vth)
        
        return {'sol':solution_vector, 'h_bolsig': h_bolsig, 'atol': abs_tol, 'rtol':rel_tol, 'tgrid':None}

    def steady_state_solver(self, h_init=None):
        """steady state solver"""
        mass_op = self._mass_op
        mw      = self._mw
        vth     = self._vth

        args    = self._args
        Cmat    = self._Cmat
        Emat    = self._Emat * (args.E_field / vth) * collisions.ELECTRON_CHARGE_MASS_RATIO
        Mmat    = self._mass_mat
        Minv    = self._inv_mass_mat
        qA      = self._Qa

        collOp  = self._collision_op 
        
        if args.ee_collisions == 1:
            cc_op   = self._cc_op
            def res_func(x):
                gamma_a     = self._n0 * ion_deg * collOp.gamma_a(np.dot(qA,x), mw, vth, self._n0, ion_deg,eff_rr_op)
                y           = np.dot(QT, np.dot(Cmat_p_Emat  + gamma_a * np.dot(cc_op,x), x)) - np.dot(Wmat,x) * np.dot(QT,x)
                return y
            
            cc_op_l1 = cc_op
            cc_op_l2 = np.swapaxes(cc_op,1,2)
            def jac_func(x):
                gamma_a     = self._n0 * ion_deg * collOp.gamma_a(np.dot(qA,x), mw, vth, self._n0, ion_deg, eff_rr_op)
                Lmat        = np.dot(QT, Cmat_p_Emat + gamma_a * np.dot(cc_op_l1,x) + gamma_a * np.dot(cc_op_l2,x)) - np.dot(Wmat,x) * QT
                Lmat        = np.dot(Lmat, Q)
                return Lmat 
        else:
            def res_func(x):
                y           = np.dot(QT, np.dot(Cmat_p_Emat, x)) - np.dot(Wmat,x) * np.dot(QT, x)
                return y
            
            def jac_func(x):
                Lmat        = np.dot(QT, Cmat_p_Emat) - np.dot(Wmat, x) * QT 
                Lmat        = np.dot(Lmat, Q)
                return Lmat
            
        spec_sp = self._spec_sp
        num_p   = spec_sp._p +1 
        num_sh  = len(spec_sp._sph_harm_lm)

        mm_op   = mass_op * mw(0) * vth**3
        u       = mm_op
        u       = np.dot(np.transpose(mm_op),qA)
        p_vec   = u.reshape((u.shape[0], 1)) / np.sqrt(np.dot(u, u))

        ion_deg     = args.ion_deg
        Cmat_p_Emat = Cmat + Emat
        Cmat_p_Emat = np.matmul(Minv, Cmat_p_Emat)
        Wmat        = np.dot(u,Cmat_p_Emat)
        
        Imat        = np.eye(Cmat.shape[0])
        Imat_r      = np.eye(Imat.shape[0]-1)
        Impp        = (Imat - np.outer(p_vec, p_vec))
        Qm,Rm       = np.linalg.qr(Impp)
        
        if args.use_dg == 1 : 
            Q           = np.delete(Qm,(num_p-1) * num_sh + num_sh-1, axis=1)
            R           = np.delete(Rm,(num_p-1) * num_sh + num_sh-1, axis=0)
            QT          = np.transpose(Q)
        else:
            Q           = np.delete(Qm,(num_p-1) * num_sh + 0, axis=1)
            R           = np.delete(Rm,(num_p-1) * num_sh + 0, axis=0)
            QT          = np.transpose(Q)
        
        qr_error1       = np.linalg.norm(Impp - np.dot(Q,R)) / np.linalg.norm(Impp)
        qr_error2       = np.linalg.norm(np.dot(QT,Q)-np.eye(QT.shape[0]))

        print("|Impp - QR|/|Impp| = %.8E"%(qr_error1))
        print("|I - QT Q|         = %.8E"%(qr_error2))

        assert qr_error1 < 1e-10
        assert qr_error2 < 1e-10

        gg_list         = self._coll_list
        f1              = u / np.dot(u, u)
        eff_rr_op       = bte_utils.reaction_rates_op(spec_sp, gg_list, mw, vth) * self._n0

        # g_rate  = np.zeros(FOp.shape[0])
        # for col_idx, col in enumerate(collisions_included):
        # if "g2" in col:
        #     g  = collisions.eAr_G2(cross_section=col)
        #     g.reset_scattering_direction_sp_mat()
        #     g_rate[0::num_sh] += self._n0 *  BEUtils.reaction_rates_op(spec_sp, [g], maxwellian, vth)
        # Wmat[0::num_sh] = np.dot(u[0::num_sh], np.dot(Minv[0::num_sh, 0::num_sh], Cmat[0::num_sh, 0::num_sh]))
        # Wmat[1::num_sh] = 0.0

        if h_init is None:
            h_init                = self.initialize(init_type="maxwellian")
            # hh                    = bte_utils.normalized_distribution(spec_sp, self._mass_op, h_init, self._mw, vth)
            # qois                  = self.compute_QoIs(hh, None)
            # print("Maxwellian at Te= %.4E (ev) ev_max=%.4E g0=%.8E g2=%.8E"%(self._args.electron_volt, self._args.ev_max, qois["rates"][0], qois["rates"][1]))

        h_init                = h_init/np.dot(mm_op,h_init)
        h_init                = np.dot(np.transpose(qA),h_init)

        h_curr , atol, rtol   = newton_solve(spec_sp, h_init, res_func, jac_func, f1, Q, R, u, rtol = 1e-13, atol=1e-8, max_iter=300)
        h_pde                 = bte_utils.normalized_distribution(spec_sp, mass_op, np.dot(qA,h_curr), mw, vth)

        solution_vector = np.zeros((2, h_init.shape[0]))
        solution_vector[0,:] = bte_utils.normalized_distribution(spec_sp, mass_op, np.dot(qA,h_init), mw, vth)
        solution_vector[1,:] = bte_utils.normalized_distribution(spec_sp, mass_op, np.dot(qA,h_curr), mw, vth)
        h_bolsig             = self._bolsig_data["bolsig_hh"]

        return {'sol':solution_vector, 'h_bolsig': h_bolsig, 'atol': atol, 'rtol':rtol, 'tgrid':None}

    def transient_solver(self, T, dt, num_time_samples=500, h_init=None):
        """
        computes the transient solution
        """
        args    = self._args
        spec_sp = self._spec_sp

        mass_op = self._mass_op
        mw      = self._mw
        vth     = self._vth

        args    = self._args
        Cmat    = self._Cmat
        Emat    = self._Emat * (args.E_field / vth) * collisions.ELECTRON_CHARGE_MASS_RATIO
        Mmat    = self._mass_mat
        Minv    = self._inv_mass_mat
        qA      = self._Qa

        collOp  = self._collision_op 

        num_p   = spec_sp._p +1 
        num_sh  = len(spec_sp._sph_harm_lm)
        
        mm_op   = mass_op * mw(0) * vth**3
        u       = mm_op
        u       = np.dot(np.transpose(mm_op),qA)
        p_vec   = u.reshape((u.shape[0], 1)) / np.sqrt(np.dot(u, u))

        ion_deg     = args.ion_deg
        Cmat_p_Emat = Cmat + Emat
        Cmat_p_Emat = np.matmul(Minv, Cmat_p_Emat)
        Wmat        = np.dot(u,Cmat_p_Emat)
        
        Imat        = np.eye(Cmat.shape[0])
        Imat_r      = np.eye(Imat.shape[0]-1)
        Impp        = (Imat - np.outer(p_vec, p_vec))
        Qm,Rm       = np.linalg.qr(Impp)

        if args.use_dg == 1 : 
            Q           = np.delete(Qm,(num_p-1) * num_sh + num_sh-1, axis=1)
            R           = np.delete(Rm,(num_p-1) * num_sh + num_sh-1, axis=0)
            QT          = np.transpose(Q)
        else:
            Q           = np.delete(Qm,(num_p-1) * num_sh + 0, axis=1)
            R           = np.delete(Rm,(num_p-1) * num_sh + 0, axis=0)
            QT          = np.transpose(Q)
        
        qr_error1       = np.linalg.norm(Impp - np.dot(Q,R)) / np.linalg.norm(Impp)
        qr_error2       = np.linalg.norm(np.dot(QT,Q)-np.eye(QT.shape[0]))

        print("|Impp - QR|/|Impp| = %.8E"%(qr_error1))
        print("|I - QT Q|         = %.8E"%(qr_error2))

        assert qr_error1 < 1e-10
        assert qr_error2 < 1e-10


        if args.ee_collisions == 1 :
            pass
        else:
            Pmat        = Imat_r - dt * np.dot(QT, np.dot(Cmat_p_Emat,Q)) 
            Pmat_inv    = np.linalg.pinv(Pmat, rcond=1e-14/np.linalg.cond(Pmat))

        if num_time_samples > int(T/dt) + 1:
            print("provided dt=%.2E is too large to get %d samples. Resetting number of samples to %d"%(dt, num_time_samples, int(T/args.T_DT)))
            num_time_samples = int(T/args.T_DT)
        
        tgrid            = np.linspace(0,T, num_time_samples)
        tgrid_idx        = np.int64(np.floor(tgrid / dt))
        
        if h_init is None:
            h_init           = self.initialize(init_type="maxwellian")
            h_init           = h_init/np.dot(mm_op,h_init)
        
        h_prev       = np.dot(np.transpose(qA), h_init)
        rtol_desired = 1e-8
        atol_desired = 1e-4
        t_curr       = 0.0

        solution_vector = np.zeros((num_time_samples,h_init.shape[0]))

        sample_idx = 1
        t_step     = 0

        gg_list         = self._coll_list
        eff_rr_op       = bte_utils.reaction_rates_op(spec_sp, gg_list, mw, vth) * self._n0

        f1      = u / np.dot(u, u)
        fb_prev = np.dot(R,h_prev)
        
        h_pde                = bte_utils.normalized_distribution(spec_sp, mass_op, np.dot(qA,h_prev), mw, vth)
        solution_vector[0,:] = h_pde

        if args.ee_collisions:
            
            cc_op  = self._cc_op
            cc_op1 = cc_op
            cc_op2 = np.swapaxes(cc_op,1,2)

            QT_Cmat_p_Emat_Q  = np.dot(QT, np.dot(Cmat_p_Emat,Q))
            QT_Cmat_p_Emat    = np.dot(QT, Cmat_p_Emat)
            cc_op1_p_cc_op2   = cc_op1 + cc_op2

            while t_curr < T:
                if sample_idx < num_time_samples and t_step == tgrid_idx[sample_idx]:
                    h_pde                         = bte_utils.normalized_distribution(spec_sp, mass_op, h_prev, mw, vth)
                    solution_vector[sample_idx,:] = h_pde
                    sample_idx+=1

                gamma_a     = collOp.gamma_a(h_prev, mw, vth, self._n0, ion_deg, eff_rr_op)
                cc_f        = gamma_a * self._n0 * ion_deg
                #Lmat        = Cmat_p_Emat +  cc_f * np.dot(cc_op1,h_prev) + cc_f * np.dot(cc_op2,h_prev)
                #Pmat        = Imat_r - dt * np.dot(QT, np.dot(Lmat, Q)) 
                #rhs_vec     = dt * np.dot(np.dot(QT, Cmat_p_Emat + cc_f * np.dot(cc_op,h_prev)), h_prev) - dt * np.dot(np.dot(Wmat, h_prev) * QT, h_prev)

                Pmat        = Imat_r - dt * QT_Cmat_p_Emat_Q - dt * np.dot(QT, np.dot(cc_f * np.dot(cc_op1_p_cc_op2,h_prev), Q))
                rhs_vec     = dt * np.dot(np.dot(QT, Cmat_p_Emat + cc_f * np.dot(cc_op,h_prev)), h_prev) - dt * np.dot(np.dot(Wmat, h_prev) * QT, h_prev)
                
                fb_curr     = fb_prev + np.linalg.solve(Pmat, rhs_vec) 
                h_curr      = f1 + np.dot(Q,fb_curr)
                
                rtol= (np.linalg.norm(h_prev - h_curr))/np.linalg.norm(h_curr)
                atol= (np.linalg.norm(h_prev - h_curr))

                t_curr+= dt
                t_step+=1
                h_prev  = h_curr
                fb_prev = fb_curr

                if t_step%100 == 0: 
                    print("time = %.3E solution convergence dt=%.2E atol = %.8E rtol = %.8E mass %.10E"%(t_curr, dt, atol, rtol, np.dot(u,h_curr)))
        else:
            QT_Cmat_p_Emat_Q  = np.dot(QT, np.dot(Cmat_p_Emat,Q))
            QT_Cmat_p_Emat_f1 = np.dot(np.dot(QT, Cmat_p_Emat),f1)
            QT_f1             = np.dot(QT,f1)

            while t_curr < T:
                if sample_idx < num_time_samples and t_step == tgrid_idx[sample_idx]:
                    h_pde                         = bte_utils.normalized_distribution(spec_sp, mass_op, np.dot(qA,h_prev), mw, vth)
                    solution_vector[sample_idx,:] = h_pde
                    sample_idx+=1

                # fully implicit on mass growth term
                Pmat        = Imat_r  - dt * QT_Cmat_p_Emat_Q  + dt * np.dot(Wmat,h_prev) * Imat_r
                #Pmat_inv    = np.linalg.inv(Pmat)
                rhs_vec     = fb_prev + dt * QT_Cmat_p_Emat_f1 - dt * np.dot(Wmat,h_prev) * QT_f1
                #fb_curr     = np.dot(Pmat_inv, rhs_vec)
                fb_curr     = np.linalg.solve(Pmat, rhs_vec) 

                # semi-explit on mass growth term (only need to compute the inverse matrix once)
                #rhs_vec     = fb_prev + dt * QT_Cmat_p_Emat_f1 - dt * np.dot(np.dot(Wmat,h_prev) * QT, h_prev)
                #fb_curr     = np.dot(Pmat_inv,rhs_vec)
                
                h_curr      = f1 + np.dot(Q,fb_curr)

                rtol= (np.linalg.norm(h_prev - h_curr))/np.linalg.norm(h_curr)
                atol= (np.linalg.norm(h_prev - h_curr))

                t_curr+= dt
                t_step+=1
                h_prev  = h_curr
                fb_prev = fb_curr

                if t_step%100 == 0: 
                    print("time = %.3E solution convergence atol = %.8E rtol = %.8E mass %.10E"%(t_curr, atol, rtol, np.dot(u,h_curr)))
        # atol = 0
        # rtol = 0
        # h_curr = f1 + np.dot(Q,fb_prev)
        # print(h_curr)
        # print(np.dot(qA,h_curr))

        h_pde                 = bte_utils.normalized_distribution(spec_sp, mass_op, np.dot(qA,h_curr), mw, vth)
        solution_vector[-1,:] = h_pde
        h_bolsig              = self._bolsig_data["bolsig_hh"]
        
        return {'sol':solution_vector, 'h_bolsig': h_bolsig, 'atol': atol, 'rtol':rtol, 'tgrid':tgrid}

    def transient_solver_time_harmonic_efield(self, T, dt, num_time_samples=500, h_init=None):
        """
        computes the transient solution
        """
        args    = self._args
        spec_sp = self._spec_sp

        mass_op = self._mass_op
        mw      = self._mw
        vth     = self._vth

        args    = self._args
        Cmat    = self._Cmat
        Emat    = self._Emat 
        Mmat    = self._mass_mat
        Minv    = self._inv_mass_mat
        qA      = self._Qa

        collOp  = self._collision_op 

        num_p   = spec_sp._p +1 
        num_sh  = len(spec_sp._sph_harm_lm)
        
        mm_op   = mass_op * mw(0) * vth**3
        u       = mm_op
        u       = np.dot(np.transpose(mm_op),qA)
        p_vec   = u.reshape((u.shape[0], 1)) / np.sqrt(np.dot(u, u))

        ion_deg     = args.ion_deg
        
        Imat        = np.eye(Cmat.shape[0])
        Imat_r      = np.eye(Imat.shape[0]-1)
        Impp        = (Imat - np.outer(p_vec, p_vec))
        Qm,Rm       = np.linalg.qr(Impp)

        if args.use_dg == 1 : 
            Q           = np.delete(Qm,(num_p-1) * num_sh + num_sh-1, axis=1)
            R           = np.delete(Rm,(num_p-1) * num_sh + num_sh-1, axis=0)
            QT          = np.transpose(Q)
        else:
            Q           = np.delete(Qm,(num_p-1) * num_sh + 0, axis=1)
            R           = np.delete(Rm,(num_p-1) * num_sh + 0, axis=0)
            QT          = np.transpose(Q)
        
        qr_error1       = np.linalg.norm(Impp - np.dot(Q,R)) / np.linalg.norm(Impp)
        qr_error2       = np.linalg.norm(np.dot(QT,Q)-np.eye(QT.shape[0]))

        print("|Impp - QR|/|Impp| = %.8E"%(qr_error1))
        print("|I - QT Q|         = %.8E"%(qr_error2))

        assert qr_error1 < 1e-10
        assert qr_error2 < 1e-10

        if num_time_samples > int(T/dt) + 1:
            print("provided dt=%.2E is too large to get %d samples. Resetting number of samples to %d"%(dt, num_time_samples, int(T/args.T_DT)))
            num_time_samples = int(T/args.T_DT)

        tgrid            = np.linspace(0,T, num_time_samples)
        tgrid_idx        = np.int64(np.floor(tgrid / dt))
        
        if h_init is None:
            h_init           = self.initialize(init_type="maxwellian")
            h_init           = h_init/np.dot(mm_op,h_init)
        
        h_prev       = np.dot(np.transpose(qA), h_init)
        rtol_desired = 1e-8
        atol_desired = 1e-4
        t_curr       = 0.0

        solution_vector = np.zeros((num_time_samples,h_init.shape[0]))

        sample_idx = 1
        t_step     = 0

        gg_list         = self._coll_list
        eff_rr_op       = bte_utils.reaction_rates_op(spec_sp, gg_list, mw, vth) * self._n0

        f1      = u / np.dot(u, u)
        fb_prev = np.dot(R,h_prev)
        
        h_pde                = bte_utils.normalized_distribution(spec_sp, mass_op, np.dot(qA,h_prev), mw, vth)
        solution_vector[0,:] = h_pde

        E_max = args.E_field
        Ef    = lambda t : E_max * np.cos(np.pi * 2 * t / args.efield_period)

        Cmat        = np.dot(Minv, Cmat)
        Emat        = np.dot(Minv, Emat)
        Wmat        = np.dot(u, Cmat)

        if args.ee_collisions:

            cc_op  = self._cc_op
            cc_op1 = cc_op
            cc_op2 = np.swapaxes(cc_op,1,2)

            cc_op1_p_cc_op2   = cc_op1 + cc_op2

            while t_curr < T:
                E_field           = Ef(t_curr)
                Cmat_p_Emat       = Cmat + Emat * (E_field / vth) * collisions.ELECTRON_CHARGE_MASS_RATIO
                QT_Cmat_p_Emat_Q  = np.dot(QT, np.dot(Cmat_p_Emat,Q))
                QT_Cmat_p_Emat    = np.dot(QT, Cmat_p_Emat)

                if sample_idx < num_time_samples and t_step == tgrid_idx[sample_idx]:
                    h_pde                         = bte_utils.normalized_distribution(spec_sp, mass_op, h_prev, mw, vth)
                    solution_vector[sample_idx,:] = h_pde
                    sample_idx+=1

                gamma_a     = collOp.gamma_a(h_prev, mw, vth, self._n0, ion_deg, eff_rr_op)
                cc_f        = gamma_a * self._n0 * ion_deg
                #Lmat        = Cmat_p_Emat +  cc_f * np.dot(cc_op1,h_prev) + cc_f * np.dot(cc_op2,h_prev)
                #Pmat        = Imat_r - dt * np.dot(QT, np.dot(Lmat, Q)) 
                #rhs_vec     = dt * np.dot(np.dot(QT, Cmat_p_Emat + cc_f * np.dot(cc_op,h_prev)), h_prev) - dt * np.dot(np.dot(Wmat, h_prev) * QT, h_prev)

                Pmat        = Imat_r - dt * QT_Cmat_p_Emat_Q - dt * np.dot(QT, np.dot(cc_f * np.dot(cc_op1_p_cc_op2,h_prev), Q))
                rhs_vec     = dt * np.dot(np.dot(QT, Cmat_p_Emat + cc_f * np.dot(cc_op,h_prev)), h_prev) - dt * np.dot(np.dot(Wmat, h_prev) * QT, h_prev)
                
                fb_curr     = fb_prev + np.linalg.solve(Pmat, rhs_vec) 
                h_curr      = f1 + np.dot(Q,fb_curr)
                
                rtol= (np.linalg.norm(h_prev - h_curr))/np.linalg.norm(h_curr)
                atol= (np.linalg.norm(h_prev - h_curr))

                t_curr+= dt
                t_step+=1
                h_prev  = h_curr
                fb_prev = fb_curr

                if t_step%100 == 0: 
                    print("time = %.3E solution convergence dt=%.2E atol = %.8E rtol = %.8E mass %.10E"%(t_curr, dt, atol, rtol, np.dot(u,h_curr)))
        else:
            QT_f1             = np.dot(QT,f1)
            emat_p , _, _     = spec_sp.compute_advection_matix_dg(advection_dir=-1.0)
            emat_m , _, _     = spec_sp.compute_advection_matix_dg(advection_dir=1.0)

            emat_p            = np.matmul(Minv, emat_p)
            emat_m            = np.matmul(Minv, emat_m)

            while t_curr < T:
                E_field           = Ef(t_curr)

                if E_field >= 0:
                    Emat = emat_p
                else:
                    Emat = emat_m

                Cmat_p_Emat       = Cmat + Emat * (E_field / vth) * collisions.ELECTRON_CHARGE_MASS_RATIO
                QT_Cmat_p_Emat_Q  = np.dot(QT, np.dot(Cmat_p_Emat,Q))
                QT_Cmat_p_Emat_f1 = np.dot(np.dot(QT, Cmat_p_Emat),f1)

                if sample_idx < num_time_samples and t_step == tgrid_idx[sample_idx]:
                    h_pde                         = bte_utils.normalized_distribution(spec_sp, mass_op, np.dot(qA,h_prev), mw, vth)
                    solution_vector[sample_idx,:] = h_pde
                    sample_idx+=1

                # fully implicit on mass growth term
                Pmat        = Imat_r  - dt * QT_Cmat_p_Emat_Q  + dt * np.dot(Wmat,h_prev) * Imat_r
                #Pmat_inv    = np.linalg.inv(Pmat)
                rhs_vec     = fb_prev + dt * QT_Cmat_p_Emat_f1 - dt * np.dot(Wmat,h_prev) * QT_f1
                fb_curr     = np.linalg.solve(Pmat, rhs_vec) 

                # semi-explit on mass growth term (only need to compute the inverse matrix once)
                #rhs_vec     = fb_prev + dt * QT_Cmat_p_Emat_f1 - dt * np.dot(np.dot(Wmat,h_prev) * QT, h_prev)
                #fb_curr     = np.dot(Pmat_inv,rhs_vec)
                
                h_curr      = f1 + np.dot(Q,fb_curr)

                rtol= (np.linalg.norm(h_prev - h_curr))/np.linalg.norm(h_curr)
                atol= (np.linalg.norm(h_prev - h_curr))

                t_curr+= dt
                t_step+=1
                h_prev  = h_curr
                fb_prev = fb_curr

                if t_step%100 == 0: 
                    print("time = %.3E solution convergence atol = %.8E rtol = %.8E mass %.10E"%(t_curr, atol, rtol, np.dot(u,h_curr)))
        # atol = 0
        # rtol = 0
        # h_curr = f1 + np.dot(Q,fb_prev)
        # print(h_curr)
        # print(np.dot(qA,h_curr))
        args.E_field          = E_max
        h_pde                 = bte_utils.normalized_distribution(spec_sp, mass_op, np.dot(qA,h_curr), mw, vth)
        solution_vector[-1,:] = h_pde
        h_bolsig              = self._bolsig_data["bolsig_hh"]
        
        return {'sol':solution_vector, 'h_bolsig': h_bolsig, 'atol': atol, 'rtol':rtol, 'tgrid':tgrid}

    def compute_QoIs(self, hh, tgrid, effective_mobility=True):

        args = self._args
        mw   = self._mw
        vth  = self._vth

        spec_sp = self._spec_sp
        num_p   = spec_sp._p + 1
        num_sh  = len(spec_sp._sph_harm_lm)

        E_max   = args.E_field
        if args.efield_period > 0:
            Ef              = lambda t : E_max * np.cos(np.pi * 2 * t / args.efield_period)
            e_field_t       = Ef(tgrid)
            e_field_t[np.abs(e_field_t) < 1e-2] = 0 
        elif tgrid is None:
            e_field_t       = E_max
        else:
            e_field_t       = E_max

        hh =hh.reshape((-1, num_p * num_sh))
        c_gamma   = self._c_gamma
        eavg_to_K = (2/(3*scipy.constants.Boltzmann))

        #print(tgrid/1e-5)
        #print(Ef(tgrid))

        mm  = np.dot(hh, self._mass_op) * mw(0) * vth**3
        mu  = np.dot(hh, self._temp_op) * mw(0) * vth**5 * (0.5 * collisions.MASS_ELECTRON * eavg_to_K / mm ) * 1.5 / collisions.TEMP_K_1EV

        if effective_mobility:
            M   = np.dot(np.sqrt(3) * hh[:, 1::num_sh], self._mobility_op)  * (-(c_gamma / (3 * ( e_field_t / self._n0))))
        else:
            M   = np.dot(np.sqrt(3) * hh[:, 1::num_sh], self._mobility_op)  * (-(c_gamma / (3 * ( 1 / self._n0))))
        
        D   = np.dot(hh[:, 0::num_sh], self._diffusion_op) * (c_gamma / 3.)
        
        rr  = list()
        for col_idx, g in enumerate(self._coll_list):
            reaction_rate = np.dot(hh[:,0::num_sh], self._rr_op[col_idx])
            rr.append(reaction_rate)
        
        return {"energy":mu, "mobility":M, "diffusion": D, "rates": rr}
    
    def time_harmonic_efield_with_series_ss_solves(self, T, dt, num_time_samples=500, h_init=None):
        """
        computes the transient solution
        """
        args    = self._args
        spec_sp = self._spec_sp

        mass_op = self._mass_op
        mw      = self._mw
        vth     = self._vth

        args    = self._args
        Cmat    = self._Cmat
        Emat    = self._Emat 
        Mmat    = self._mass_mat
        Minv    = self._inv_mass_mat
        qA      = self._Qa

        collOp  = self._collision_op 

        num_p   = spec_sp._p +1 
        num_sh  = len(spec_sp._sph_harm_lm)
        
        mm_op   = mass_op * mw(0) * vth**3
        u       = mm_op
        
        tgrid            = np.linspace(0,T, num_time_samples)
        tgrid_idx        = np.int64(np.floor(tgrid / dt))
        
        if h_init is None:
            h_init           = self.initialize(init_type="maxwellian")
            h_init           = h_init/np.dot(mm_op,h_init)
        
        
        solution_vector = np.zeros((num_time_samples,h_init.shape[0]))

        sample_idx = 0
        t_step     = 0
        t_curr     = 0.0

        
        E_max   = args.E_field
        Ef      = lambda t : E_max * np.cos(np.pi * 2 * t / args.efield_period)
        
        while t_curr < T:
            args.E_field       = Ef(t_curr)
            print("time = %.3E E= %4E" %(t_curr, args.E_field))
            ss_sol             = self.steady_state_solver(h_init)
            atol               = ss_sol['atol']
            h_prev             = ss_sol['sol'][-1, :]
            if atol > 1:
                print("ss solver failed")
                h_prev = h_prev * 0

            if sample_idx < num_time_samples and t_step == tgrid_idx[sample_idx]:
                print(sample_idx)
                h_pde                         = bte_utils.normalized_distribution(spec_sp, mass_op, h_prev, mw, vth)
                solution_vector[sample_idx,:] = h_pde
                sample_idx+=1

            t_curr+= dt
            t_step+=1
            

        args.E_field          = E_max
        
        h_pde                 = bte_utils.normalized_distribution(spec_sp, mass_op, np.dot(qA,h_prev), mw, vth)
        solution_vector[-1,:] = h_pde
        h_bolsig              = self._bolsig_data["bolsig_hh"]

        return {'sol':solution_vector, 'h_bolsig': h_bolsig, 'atol': 0.0, 'rtol':0.0, 'tgrid':tgrid}