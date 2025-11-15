import numpy as np
import argparse
import basis
import spec_spherical as sp
import scipy.constants
import cross_section
import collisions
import collision_operator_spherical as collOp
from time import perf_counter as time, sleep
import utils as bte_utils
import matplotlib.pyplot as plt
import os
import itertools
import scipy.interpolate
import sys

class bte_ops():

    def __init__(self, args, nr, nvt):
        self._run_bolsig(args)
        c_gamma                  = np.sqrt(2 * scipy.constants.elementary_charge / scipy.constants.electron_mass) 
        vth                      = c_gamma * args.ev**0.5

        collision_names          = list()
        coll_list                = list()
        avail_species            = cross_section.read_available_species(args.collisions)
        cross_section_data       = cross_section.read_cross_section_data(args.collisions)
               
        print("==========read collissions===========")
        collision_count = 0
        for col_str, col_data in cross_section_data.items():
            print(col_str, col_data["type"])
            g = collisions.electron_heavy_binary_collision(col_str, collision_type=col_data["type"])
            coll_list.append(g)
            collision_names.append("C%d"%(collision_count+1)) 
            collision_count+=1
        print("=====================================")
        print("number of total collisions = %d " %(len(coll_list)))
        num_collisions = len(coll_list)
        species        = cross_section.read_available_species(args.collisions)
        sig_pts   =  list()

        for col_idx, g in enumerate(coll_list):
            g  = coll_list[col_idx]
            if g._reaction_threshold != None and g._reaction_threshold >0:
                sig_pts.append(g._reaction_threshold)
        
        sig_pts = np.sort(np.array(list(set(sig_pts))))
        sig_pts = np.sqrt(sig_pts) * c_gamma / vth

        ev_range            = ((0*vth/c_gamma)**2, args.ev_max)
        k_domain            = (np.sqrt(ev_range[0]) * c_gamma / vth, np.sqrt(ev_range[1]) * c_gamma / vth)


        l_max                   = args.l_max
        sph_harm_lm             = [[l,0] for l in range(l_max+1)]
        mw                      = bte_utils.get_maxwellian_3d(vth)
        sp_order                = args.spline_order
        bsp                     = basis.BSpline(k_domain,sp_order, nr+1, sig_pts=None, knots_vec=None, dg_splines=0)
        spec_sp                 = sp.SpectralExpansionSpherical(nr, bsp, sph_harm_lm)
        spec_sp._num_q_radial   = args.sp_qpts * spec_sp._basis_p._num_knot_intervals
        
        num_vt                  = nvt
        xp_vt, xp_vt_qw         = spec_sp.gl_vt(num_vt, hspace_split=True, mode="npsp")
        Po, Ps                  = spec_sp.sph_ords_projections_ops(xp_vt, xp_vt_qw, mode="sph")

        #xp_vr, xp_vr_qw        = spec_sp.gl_vr(spec_sp._num_q_radial, use_bspline_qgrid=False)
        xp_vr, xp_vr_qw         = spec_sp.gl_vr(args.sp_qpts, use_bspline_qgrid=True)
        num_vr                  = len(xp_vr)
        
        Pr, Pb                  = spec_sp.radial_to_vr_projection_ops(xp_vr, xp_vr_qw)

        ev_fac                  = 0.5 * scipy.constants.electron_mass * (2/3/scipy.constants.Boltzmann) / collisions.TEMP_K_1EV
        mass_vrvt               = mw(0) * vth**3 * np.kron(xp_vr**2 * xp_vr_qw, np.ones_like(xp_vt) * xp_vt_qw) * np.pi * 2
        temp_vrvt               = mw(0) * vth**5 * np.kron(xp_vr**4 * xp_vr_qw, np.ones_like(xp_vt) * xp_vt_qw) * np.pi * 2 * ev_fac
        mass_bsp                = mw(0) * vth**3 * bte_utils.mass_op(spec_sp)
        temp_bsp                = mw(0) * vth**5 * bte_utils.temp_op(spec_sp) * ev_fac

        num_p                   = spec_sp._p + 1
        num_sh                  = len(spec_sp._sph_harm_lm)
        mmat                    = spec_sp.compute_mass_matrix()
        minv                    = spec_sp.inverse_mass_mat(Mmat=mmat)

        advmatEp, advmatEn      = spec_sp.compute_advection_matrix_vrvt_fv(xp_vr, xp_vt, sw_vr=3, sw_vt=2, use_upwinding=True)
        Av                      = advmatEp if args.E > 0 else advmatEn
        Av                      = Av.reshape((num_vr * num_vt, num_vr * num_vt))
        
        Av_bsp                  = spec_sp.compute_advection_matix() 
        
        
        print("num vt = %d num vr = %d num_p = %d num_sh=%d"%(num_vt, num_vr, num_p, num_sh))
        cop     = collOp.CollisionOpSP(spec_sp)
        Fop     = 0.0
        Fop_bsp = 0.0
        t1  = time()
        for col_idx, (col_str, col_data) in enumerate(cross_section_data.items()):
            g         = coll_list[col_idx]
            mole_idx  = avail_species.index(col_data["species"]) 
            Fop       = Fop     + args.n0 * cop._Lop_eulerian_strong_form((xp_vr, xp_vr_qw), (xp_vt, xp_vt_qw), g, mw, vth, 0.0, Nvts=64, Nvps=64, azimuthal_symmetry=True)
            Fop_bsp   = Fop_bsp + args.n0 * cop.assemble_mat(g, mw, vth, use_hsph=False, tgK=0.0)
            #Fop       = Fop + args.n0 * cop.assemble_mat(g, mw, vth, use_hsph=True)

            #Fop       = Fop     + args.n0 * xp.kron(Pr, Po) @ (minv @ cop.assemble_mat(g, mw, vth, use_hsph=False, tgK=0.0)) @ xp.kron(Pb,Ps)

            print(col_str, "u^T C (bsp) = %.9E u^T C (fvm) = %.9E "%(np.linalg.norm(np.dot(mass_bsp, minv @ Fop_bsp)), np.linalg.norm(np.dot(mass_vrvt, Fop))))

        t2 = time()

        #Fop= xp.kron(Pr,Po) @ Fop @ xp.kron(Pb, Ps)


        print("Assembled the collision op. for Vth : ", vth)
        print("Collision Operator assembly time (s): ",(t2-t1))
        print("wmat A = %.8E , %.8E" % (np.linalg.norm(np.dot(mass_vrvt, Av)), np.linalg.norm(np.dot(mass_bsp, minv @ Av_bsp))))
        print("wmat C = %.8E , %.8E" % (np.linalg.norm(np.dot(mass_vrvt, Fop)), np.linalg.norm(np.dot(mass_bsp, minv @ Fop_bsp))))


        self.mmat = mmat
        self.minv = minv

        self.spec_sp  = spec_sp
        self.Pr       = Pr
        self.Pb       = Pb
        self.Ps       = Ps
        self.Po       = Po
        self.cmat_bsp = minv @ Fop_bsp
        self.amat_bsp = (1/vth)* (collisions.ELECTRON_CHARGE_MASS_RATIO) * (minv @ Av_bsp)
        self.cmat_fvm = Fop
        self.amat_fvm = (1/vth)* (collisions.ELECTRON_CHARGE_MASS_RATIO) * (Av)

        self.vr_g     = (xp_vr, xp_vr_qw)
        self.vt_g     = (xp_vt, xp_vt_qw)
        
        self.m0_bsp   = mass_bsp
        self.m0_fvm   = mass_vrvt

        self.m2_bsp   = temp_bsp
        self.m2_fvm   = temp_vrvt
        self.args     = args
        self.vth      = vth
        self.c_gamma  = c_gamma
        self.num_collisions = num_collisions
        self.mw       = bte_utils.get_maxwellian_3d(vth)

        # print(args)
        # self._run_bolsig(args)
        # print(args)

    def _run_bolsig(self, args):
        class bolsig_params():
            collisions = args.collisions
            bolsig_dir = "../../Bolsig/"
            ion_deg    = 0
            n0         = args.n0
            bolsig_grid_pts = 1024
            bolsig_precision = 1e-11
            bolsig_convergence = 1e-8
            ns_by_n0           = [1 for i in range(5)]
            E_field            = args.E
            Tg                 = args.Tg
            ee_collisions      = 0

        avail_species                       = cross_section.read_available_species(args.collisions)
        cross_section_data                  = cross_section.read_cross_section_data(args.collisions)
        print("==========read collissions===========")
        collision_count = 0
        coll_list = list()
        collision_names=list()
        for col_str, col_data in cross_section_data.items():
            print(col_str, col_data["type"])
            g = collisions.electron_heavy_binary_collision(col_str, collision_type=col_data["type"])
            coll_list.append(g)
            collision_names.append("C%d"%(collision_count+1)) 
            collision_count+=1
        print("=====================================")
        print("number of total collisions = %d " %(len(coll_list)))

        try:
            import bolsig
            params = bolsig_params()
            bolsig.run_bolsig(params)
            [bolsig_ev, bolsig_f0, bolsig_a, bolsig_E, bolsig_mu, bolsig_M, bolsig_D, bolsig_rates,bolsig_cclog] = bolsig.parse_bolsig(params.bolsig_dir+"argon.out", len(coll_list))

            self._bolsig_data = dict()
            self._bolsig_data["ev"] = bolsig_ev
            self._bolsig_data["f0"] = bolsig_f0
            self._bolsig_data["f1"] = np.abs(bolsig_f0 * bolsig_a / np.sqrt(3))

            self._bolsig_data["energy"]    = bolsig_mu
            self._bolsig_data["mobility"]  = bolsig_M
            self._bolsig_data["diffusion"] = bolsig_D
            self._bolsig_data["rates" ]    = bolsig_rates
            self._bolsig_data["cc_log"]    = bolsig_cclog

            print("bolsig temp      = %.8E"%((bolsig_mu /1.5)))
            print("bolsig mobility  = %.8E"%((bolsig_M)))
            print("bolsig diffusion = %.8E"%((bolsig_D)))
            #print("bolsig coulomb logarithm = %.8E"%((bolsig_cclog)))

            print("bolsig collision rates")
            for  col_idx, col in enumerate(coll_list):
                print("%s = %.8E"%(collision_names[col_idx], bolsig_rates[col_idx]))

            args.ev            = (bolsig_mu/1.5)
            args.ev_max        = bolsig_ev[-1]
        except Exception as e:
           print("running Bolsig+ solver failed with error: %s"%(e))


        



    def _uQRQT(self, u, rmidx):
        mw              = bte_utils.get_maxwellian_3d(self.vth)
        p_vec           = u.reshape((u.shape[0], 1)) / np.sqrt(np.dot(u, u))
        Imat            = np.eye(u.shape[0])
        Impp            = (Imat - np.outer(p_vec, p_vec))
        Qm,Rm           = np.linalg.qr(Impp)

        Q               = np.delete(Qm,rmidx, axis=1)
        R               = np.delete(Rm,rmidx, axis=0)
        QT              = np.transpose(Q)

        qr_error1       = np.linalg.norm(Impp - np.dot(Q,R)) / np.linalg.norm(Impp)
        qr_error2       = np.linalg.norm(np.dot(QT,Q)-np.eye(QT.shape[0]))

        print("|Impp - QR|/|Impp| = %.8E"%(qr_error1))
        print("|I - QT Q|         = %.8E"%(qr_error2))

        assert qr_error1 < 1e-10
        assert qr_error2 < 1e-10


        return u, Q, R, QT
    
    def _step_normalized(self, x0, t, dt, u, Q, R, QT, Cmat, Emat, E, xp, verbose=0):
        vth               = self.vth
        f1                = u / xp.dot(u, u)
        Cmat_p_Emat       = Cmat + Emat * E
        QT_Cmat_p_Emat_Q  = xp.dot(QT, np.dot(Cmat_p_Emat,Q))
        QT_Cmat_p_Emat_f1 = xp.dot(xp.dot(QT, Cmat_p_Emat),f1)
        Wmat              = xp.dot(u, Cmat_p_Emat)
        #Wmat              = 0*xp.dot(u, Cmat)
        Imat              = xp.eye(Cmat.shape[0])
        Imat_r            = xp.eye(Imat.shape[0]-1)
        QT_f1             = xp.dot(QT,f1)

        h_prev            = xp.copy(x0)
        fb_prev           = xp.dot(R,h_prev)
        
        # fully implicit on mass growth term
        Pmat        = Imat_r  - dt * QT_Cmat_p_Emat_Q  + dt * xp.dot(Wmat,h_prev) * Imat_r
        #Pmat_inv    = np.linalg.inv(Pmat)
        rhs_vec     = fb_prev + dt * QT_Cmat_p_Emat_f1 - dt * xp.dot(Wmat,h_prev) * QT_f1
        fb_curr     = xp.linalg.solve(Pmat, rhs_vec) 

        # semi-explit on mass growth term (only need to compute the inverse matrix once)
        #rhs_vec     = fb_prev + dt * QT_Cmat_p_Emat_f1 - dt * np.dot(np.dot(Wmat,h_prev) * QT, h_prev)
        #fb_curr     = np.dot(Pmat_inv,rhs_vec)
                
        h_curr      = f1 + np.dot(Q,fb_curr)

        rtol= (xp.linalg.norm(h_prev - h_curr))/xp.linalg.norm(h_curr)
        atol= (xp.linalg.norm(h_prev - h_curr))

        if verbose: 
            print("time = %.3E solution convergence atol = %.8E rtol = %.8E mass %.10E"%(t, atol, rtol, xp.dot(u,h_curr)))
        
        return h_curr
    
    def evolve(self, dt, T, plot_sol=False):
        args = self.args
        try:
            import cupy as cp
            import cupyx.scipy.interpolate
            _  = cp.random.rand(3)
            xp = cp
            lin_ip = cupyx.scipy.interpolate.interp1d
        except Exception as e:
            xp           = np

        def asnumpy(x):
            a = cp.asnumpy(x) if xp == cp else x
            return x


        mw           = bte_utils.get_maxwellian_3d(self.vth)
        io_freq      = args.io
        steps        = int(np.ceil(T/dt))
        

        xp_vr, xp_vr_qw = self.vr_g[0], self.vr_g[1]
        xp_vt, xp_vt_qw = self.vt_g[0], self.vt_g[1]
        vth             = self.vth
        num_vr, num_vt  = len(xp_vr), len(xp_vt)
        spec_sp         = self.spec_sp
        minv            = self.minv


        # y0           = xp.repeat((mw(xp_vr) * vth**3).reshape((-1,1)), num_vt, axis=1).reshape((-1))
        # hv           = lambda v,vt,vp : (1/np.sqrt(np.pi)**3 ) * (np.exp(-((v)**2))) 

        y0           = xp.kron(mw(xp_vr) * vth**3, 1+np.cos(xp_vt)).reshape((-1))
        hv           = lambda v,vt,vp : (1/np.sqrt(np.pi)**3 ) * (np.exp(-((v)**2))) * (1 + np.cos(vt))
        y0_bsp       = xp.asarray(bte_utils.function_to_basis(spec_sp, hv, mw, spec_sp._num_q_radial, 8, 2, Minv=minv))

        num_p        = spec_sp._p+1
        num_sh       = len(spec_sp._sph_harm_lm) 

        uQRQt_bsp    = self._uQRQT(self.m0_bsp, (num_p-1) * num_sh)
        uQRQt_fvm    = self._uQRQT(self.m0_fvm, num_vr * num_vt -1)

        mass_vrvt    = xp.asarray(self.m0_fvm)
        temp_vrvt    = xp.asarray(self.m2_fvm)

        mass_bsp     = xp.asarray(self.m0_bsp)
        temp_bsp     = xp.asarray(self.m2_bsp)
        c_gamma      = self.c_gamma

        Pr, Pb       = xp.asarray(self.Pr), xp.asarray(self.Pb)
        Po, Ps       = xp.asarray(self.Po), xp.asarray(self.Ps)

        cmat_fvm     = xp.asarray(self.cmat_fvm)
        amat_fvm     = xp.asarray(self.amat_fvm)

        cmat_bsp     = xp.asarray(self.cmat_bsp)
        amat_bsp     = xp.asarray(self.amat_bsp)


        L            = cmat_fvm + args.E * amat_fvm
        L_bsp        = cmat_bsp + args.E * amat_bsp

        Linv         = xp.linalg.inv(xp.eye(self.cmat_fvm.shape[0]) - dt * L)
        Linv_bsp     = xp.linalg.inv(xp.eye(self.cmat_bsp.shape[0]) - dt * L_bsp)

        qoi          = [[] for i in range(5)]

        ev_bg     = self._bolsig_data["ev"]
        bolsig_f0 = self._bolsig_data["f0"]
        bolsig_a  = (self._bolsig_data["f1"] / bolsig_f0) * np.sqrt(3)

        f0_cf = scipy.interpolate.interp1d(ev_bg, bolsig_f0, kind='cubic', bounds_error=False, fill_value=(bolsig_f0[0],bolsig_f0[-1]))
        fa_cf = scipy.interpolate.interp1d(ev_bg, bolsig_a,  kind='cubic', bounds_error=False, fill_value=(bolsig_a[0],bolsig_a[-1]))
        ff    = lambda v,vt,vp : f0_cf(.5*(v* vth)**2/collisions.ELECTRON_CHARGE_MASS_RATIO) * (1. - fa_cf(.5*(v* vth)**2/collisions.ELECTRON_CHARGE_MASS_RATIO)*np.cos(vt))

        hh    =  bte_utils.function_to_basis(spec_sp, ff, self.mw, spec_sp._num_q_radial, 2, 2, Minv=self.minv)
        hh    =  bte_utils.normalized_distribution(spec_sp, self.m0_bsp, hh, self.mw, vth)

        self._bolsig_data["bolsig_hh"] = hh
        self._bolsig_data["bolsig_fvrvt"] = ff

        vg    = np.meshgrid(xp_vr, xp_vt, indexing="ij")
        y0_bg = ff(vg[0], vg[1], 0.0).reshape((-1))
        #print(y0_bg.shape)

        for tidx in range(steps+1):
            t  = tidx * dt

            if (tidx % io_freq == 0):
                
                ne_fvm  = xp.dot(mass_vrvt, y0)
                te_fvm  = xp.dot(temp_vrvt, y0)/ne_fvm
                
                ne_bsp  = xp.dot(mass_bsp,y0_bsp)
                te_bsp  = xp.dot(temp_bsp,y0_bsp)/ne_bsp

                qoi[0].append(t)
                qoi[1].append(ne_fvm)
                qoi[2].append(te_fvm)

                qoi[3].append(ne_bsp)
                qoi[4].append(te_bsp)

                print("time = %.4E mass = %.12E temp=%12E mass (bsp) = %.12E temp = %.12E"%(t, ne_fvm, te_fvm, ne_bsp, te_bsp))

                if plot_sol:
                    plt.figure(figsize=(8,4), dpi=200)
                    ev = (xp_vr * vth/ c_gamma)**2
                    
                    plt.subplot(1, 2, 1)
                    sxl   = (1/np.dot(mass_vrvt, y0)) 
                    syl   = (1/np.dot(mass_bsp , y0_bsp))
                    #szl   = (1/np.dot(mass_bsp , y0_bg))
                    szl   = (1/np.dot(mass_vrvt , y0_bg))
                    
                    print(sxl, syl, szl)
                    xl    = asnumpy(xp.einsum("li,vi->vl", Ps,y0.reshape((num_vr, num_vt)))    ) * sxl 
                    yl    = asnumpy(xp.einsum("vi,il->vl", Pr,y0_bsp.reshape((num_p, num_sh))) ) * syl 
                    #zl    = asnumpy(xp.einsum("vi,il->vl", Pr,y0_bg.reshape((num_p, num_sh))) ) * szl 
                    zl    = asnumpy(xp.einsum("li,vi->vl", Ps,y0_bg.reshape((num_vr, num_vt)))    ) * szl 

                    xl    = scipy.interpolate.interp1d(ev, xl, axis=0, bounds_error=True)
                    yl    = scipy.interpolate.interp1d(ev, yl, axis=0, bounds_error=True)
                    zl    = scipy.interpolate.interp1d(ev, zl, axis=0, bounds_error=True)

                    evg   = np.linspace(ev[0], ev[-1], 1000)

                    xl    = xl(evg)
                    yl    = yl(evg)
                    zl    = zl(evg)

                    idx   = evg>0#xl[:, 0]>1e-16
                    
                    plt.semilogy(evg[idx], (xl[idx, 0])  , "-"  , markersize=0.2, color='C0', label=r"l=0 w FVM")
                    plt.semilogy(evg[idx], (yl[idx, 0])  , "--" , markersize=0.2, color='C1', label=r"l=0 w Bsp + SPH")
                    plt.semilogy(evg[idx], (zl[idx, 0])  , "--" , markersize=0.2, color='C2', label=r"l=0 Bolsig+")
                    plt.grid(visible=True)
                    plt.legend()
                    plt.xlabel(r"energy (eV)")
                    plt.title(r"t = %.4E s"%(t))
                    #plt.ylim(1e-16, None)

                    plt.subplot(1, 2, 2)
                    plt.semilogy(evg[idx], np.abs(xl[idx, 1])  , "-" , markersize=0.2, color='C0', label=r"l=1 w FVM")
                    plt.semilogy(evg[idx], np.abs(yl[idx, 1])  , "--", markersize=0.2, color='C1', label=r"l=1 w Bsp + SPH")
                    plt.semilogy(evg[idx], np.abs(zl[idx, 1])  , "--", markersize=0.2, color='C2', label=r"l=1 Bolsig+")
                    plt.grid(visible=True)
                    plt.legend()
                    plt.xlabel(r"energy (eV)")
                    plt.title(r"t = %.4E s"%(t))
                    #plt.ylim(1e-16, None)
                
                    plt.tight_layout()
                    plt.savefig("%s/%s_nr%d_nvt%d_idx_%04d.png"%(args.odir, args.o, nr, nvt, tidx//io_freq))
                    plt.close()

            
            y0_bsp = self._step_normalized(y0_bsp, t, dt,
                                xp.asarray(uQRQt_bsp[0]),
                                xp.asarray(uQRQt_bsp[1]),
                                xp.asarray(uQRQt_bsp[2]),
                                xp.asarray(uQRQt_bsp[3]),
                                self.cmat_bsp, self.amat_bsp, args.E, xp, verbose=(tidx%io_freq))
            
            y0     = self._step_normalized(y0,     t, dt,
                                xp.asarray(uQRQt_fvm[0]),
                                xp.asarray(uQRQt_fvm[1]),
                                xp.asarray(uQRQt_fvm[2]),
                                xp.asarray(uQRQt_fvm[3]),
                                self.cmat_fvm, self.amat_fvm, args.E, xp, verbose=(tidx%io_freq))

            # y0_bsp = Linv_bsp @ (y0_bsp ) #+ 0.5 * dt * L_bsp @y0_bsp)
            # y0     = Linv     @ (y0     ) #+ 0.5 * dt * L @ y0)

        if plot_sol:
            plt.figure(figsize=(10, 5), dpi=200)
            plt.subplot(2, 1, 1)
            plt.plot(qoi[0], qoi[1], label=r"$n_e$ - FVM")
            plt.plot(qoi[0], qoi[3], label=r"$n_e$ - Galerkin(BSP+SPH)")
            plt.ylabel(r"mass [m^{-3}]")
            plt.grid(visible=True)
            plt.legend()

            plt.subplot(2, 1, 2)
            plt.plot(qoi[0], qoi[2], label=r"$n_e$ - FVM")
            plt.plot(qoi[0], qoi[4], label=r"$n_e$ - Galerkin(BSP+SPH)")
            plt.ylabel(r"temperature [eV]")
            plt.grid(visible=True)
            plt.legend()
            #plt.show()
            plt.savefig("%s/%s_qoi_NrNvt_%d_%d.png"%(args.odir, args.o, nr, nvt))
            plt.close()

        return {"qoi": asnumpy(qoi), "bsp": asnumpy(y0_bsp), "fvm": asnumpy(y0), "Nr": nr, "Nvt": nvt, "dt": dt, "T":T, "bte": self}



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n0", "--n0"                             , help="n0", type=float, default= 3.22e22)
    parser.add_argument("-nr", "--nr"                             , help="nr", nargs='+', type=int, default = [16, 32, 64])
    parser.add_argument("-l_max", "--l_max"                       , help="max polar modes in SH expansion", type=int, default=2)
    parser.add_argument("-sp_order", "--spline_order"             , help="b-spline order", type=int, default=2)
    parser.add_argument("-sp_qpts" , "--sp_qpts"                  , help="q points per knots", type=int, default=11)
    parser.add_argument("-nvt","--nvt"                            , help="number of ordinates", nargs='+', type=int, default= [16, 32, 64])
    parser.add_argument("-c", "--collisions"                      , help="collision crs file" , type=str, default="lxcat_data/eAr_crs.Biagi.3sp2r")
    parser.add_argument("-ev", "--ev"                             , help="initial electron volt", type=float, default=4.0)
    parser.add_argument("-ev_max", "--ev_max"                     , help="ev max", type=float, default=4.0)
    parser.add_argument("-vmode", "--vmode"                       , help="Semi-Lagrangian (SL) or Eulerian (eulerian)", type=str, default="fvm")
    parser.add_argument("-E", "--E"                               , help="E field", type=float, default=0.0)
    
    
    parser.add_argument("-T" , "--T"                              , help="t final", type=float, default=1e-7)
    parser.add_argument("-dt", "--dt"                             , help="time step size ", type=float, default=1e-10)
    
    parser.add_argument("-odir",  "--odir"                        , help="output dir", type=str, default='vsolve')
    parser.add_argument("-o",  "--o"                              , help="output", type=str, default='vsp')
    parser.add_argument("-io", "--io"                             , help="io every K steps", type=int, default=10)
    parser.add_argument("-Tg", "--Tg"                             , help="background gas temperature", type=float, default=0.0)
    
    args = parser.parse_args()
    print(args)

    isExist = os.path.exists(args.odir)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(args.odir)
       print("directory %s is created!"%(args.odir))

    runs = list()
    for i in range(len(args.nr)):
        nr  = args.nr[i]
        nvt = args.nvt[i]
        bte = bte_ops(args, nr, nvt)
        runs.append(bte.evolve(args.dt/(2**i), args.T, plot_sol=(i==len(args.nr)-1)))

    plt.figure(figsize=(8, 8), dpi=200)
    yl_fvm_hr     = np.einsum("li,vi->vl", runs[-1]["bte"].Ps, runs[-1]["fvm"].reshape((-1, runs[-1]["Nvt"])))
    ev_hr         = ((bte.vth/bte.c_gamma) * runs[-1]["bte"].vr_g[0])**2
    qx            = runs[-1]["bte"].vr_g
    rel_error_l2  = list()
    num_sh        = args.l_max + 1
    
    for i in range(len(args.nr)):
        bte : bte_ops = runs[i]["bte"]
        ev            = ((bte.vth/bte.c_gamma) * bte.vr_g[0])**2

        y_bsp         = runs[i]["bsp"]
        y_fvm         = runs[i]["fvm"]
        dt            = runs[i]["dt"]
        Ps            = bte.Ps
        Pr            = bte.Pr
        num_vr        = len(bte.vr_g[0])
        num_vt        = len(bte.vt_g[0])

        vr_qw         = bte.vr_g[1]
        Lv            = np.sum(vr_qw)
        vr            = np.zeros(num_vr+2) 
        vr[1:-1]      = bte.vr_g[0]
        vr[-1]        = Lv
        print("run", i, num_vr, num_vt, dt)

        yl_fvm         = np.zeros((num_vr+2, num_vt))
        yl_fvm[1:-1,:] = y_fvm.reshape((num_vr, num_vt))

        yl_fvm        = scipy.interpolate.interp1d(vr, np.einsum("li,vi->vl", Ps, yl_fvm), axis=0, bounds_error=True) 
        lbl           = r"$N_r$=%d,$N_\theta$=%d, dt=%.2E"%(num_vr, num_vt, dt)

        

        yl_fvm_i      = np.einsum("li,vi->vl", Ps, y_fvm.reshape((num_vr, num_vt)))
        
        
        
        assert num_sh  == Ps.shape[0]
        for l in range(num_sh):
            plt.subplot(2, 2, l+1)

            #idx = np.abs(yl_fvm(qx[0])[:, l]) > 1e-21
            #plt.semilogy(ev_hr[idx], np.abs(yl_fvm(qx[0])[:, l])[idx], label=lbl)
            idx = np.abs(yl_fvm_i[:, l]) > 1e-21
            plt.semilogy(ev[idx], np.abs(yl_fvm_i[:, l])[idx], label=lbl)
            plt.xlabel(r"energy [eV]")
            plt.ylabel(r"$f_l$")
            plt.grid(visible=True)
            plt.legend()
            plt.ylim((1e-20, None))

        a1 = np.dot(qx[1], np.einsum("v, vl->vl", qx[0]**2, (yl_fvm(qx[0]) - yl_fvm_hr)**2))
        a2 = np.dot(qx[1], np.einsum("v, vl->vl", qx[0]**2, (yl_fvm_hr)**2))
        print(i, a1, np.min(np.abs((yl_fvm(qx[0]) - yl_fvm_hr))), np.max(np.abs((yl_fvm(qx[0]) - yl_fvm_hr))))
        rel_error_l2.append( a1/a2 )

    rel_error_l2 = np.array(rel_error_l2)
    
    plt.subplot(2, 2, 3)
    plt.semilogy(np.arange(len(runs))[:-1], rel_error_l2[:-1, 0], "x--")
    plt.xlabel(r"run id")
    plt.ylabel(r"$||y_l - y_h||_{L^2}/||y_h||_{L^2}$")
    plt.grid(visible=True)
    plt.xticks(range(1,len(args.nr)))

    plt.tight_layout()
    plt.savefig("%s/%s.png"%(args.odir, args.o))
    print("saving to ", "%s/%s.png"%(args.odir, args.o))
    #plt.show()
    plt.close()

    

        




    

    
