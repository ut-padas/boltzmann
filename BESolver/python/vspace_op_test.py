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

class bte_ops():
    def __init__(self, args, nr, nvt):

        c_gamma     = np.sqrt(2 * scipy.constants.elementary_charge / scipy.constants.electron_mass) 
        vth         = c_gamma * args.ev**0.5

        collision_names          = list()
        coll_list                = list()
        avail_species            = cross_section.read_available_species(args.collisions)
        cross_section.CROSS_SECTION_DATA   = cross_section.read_cross_section_data(args.collisions)
        cross_section_data       = cross_section.CROSS_SECTION_DATA
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

        ev_range            = ((0*vth/c_gamma)**2, (12*vth/c_gamma)**2)
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

        advmatEp, advmatEn      = spec_sp.compute_advection_matrix_vrvt_fv(xp_vr, xp_vt, sw_vr=2, sw_vt=2, use_upwinding=True)
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
    
    def _step_normalized(self, x0, t, dt, u, Q, R, QT, Cmat, Emat, E, verbose=0):
        xp                = np
        vth               = self.vth
        f1                = u / xp.dot(u, u)
        Cmat_p_Emat       = Cmat + Emat * E
        QT_Cmat_p_Emat_Q  = xp.dot(QT, np.dot(Cmat_p_Emat,Q))
        QT_Cmat_p_Emat_f1 = xp.dot(xp.dot(QT, Cmat_p_Emat),f1)
        Wmat              = xp.dot(u, Cmat_p_Emat)
        Imat              = xp.eye(Cmat.shape[0])
        Imat_r            = xp.eye(Imat.shape[0]-1)
        QT_f1             = xp.dot(QT,f1)

        h_prev            = xp.copy(x0)
        fb_prev           = np.dot(R,h_prev)
        
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

        if verbose: 
            print("time = %.3E solution convergence atol = %.8E rtol = %.8E mass %.10E"%(t, atol, rtol, xp.dot(u,h_curr)))
        
        return h_curr
    
    def evolve(self):
        args = self.args
        xp           = np
        mw           = bte_utils.get_maxwellian_3d(self.vth)
        io_freq      = args.io
        steps        = int(np.ceil(args.T/args.dt))
        dt           = args.dt

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
        y0_bsp       = bte_utils.function_to_basis(spec_sp, hv, mw, spec_sp._num_q_radial, 8, 2, Minv=minv)

        num_p        = spec_sp._p+1
        num_sh       = len(spec_sp._sph_harm_lm) 

        uQRQt_bsp    = self._uQRQT(self.m0_bsp, (num_p-1) * num_sh)
        uQRQt_fvm    = self._uQRQT(self.m0_fvm, num_vr * num_vt -1)

        mass_vrvt    = self.m0_fvm
        temp_vrvt    = self.m2_fvm

        mass_bsp    = self.m0_bsp
        temp_bsp    = self.m2_bsp
        c_gamma     = self.c_gamma

        Pr, Pb      = self.Pr, self.Pb
        Po, Ps      = self.Po, self.Ps


        Linv     = xp.linalg.inv(xp.eye(self.cmat_fvm.shape[0]) - dt * (self.cmat_fvm + args.E * self.amat_fvm))
        Linv_bsp = xp.linalg.inv(xp.eye(self.cmat_bsp.shape[0]) - dt * (self.cmat_bsp + args.E * self.amat_bsp))

        qoi      = [[] for i in range(5)]

        for tidx in range(steps+1):
            t  = tidx * dt

            if (tidx % io_freq == 0):
                
                ne_fvm  = np.dot(mass_vrvt, y0)
                te_fvm  = np.dot(temp_vrvt, y0)/ne_fvm
                
                ne_bsp  = np.dot(mass_bsp,y0_bsp)
                te_bsp  = np.dot(temp_bsp,y0_bsp)/ne_bsp

                qoi[0].append(t)
                qoi[1].append(ne_fvm)
                qoi[2].append(te_fvm)

                qoi[3].append(ne_bsp)
                qoi[4].append(te_bsp)

                print("time = %.4E mass = %.12E temp=%12E mass (bsp) = %.12E temp = %.12E"%(t, ne_fvm, te_fvm, ne_bsp, te_bsp))
                plt.figure(figsize=(20,10), dpi=200)
                ev = (xp_vr * vth/ c_gamma)**2
                
                plt.subplot(1, 2, 1)
                xl    = np.einsum("li,vi->vl", Ps, y0.reshape((num_vr, num_vt)))    #* (1/np.dot(mass_vrvt, y0))     
                yl    = np.einsum("vi,il->vl", Pr, y0_bsp.reshape((num_p, num_sh))) #* (1/np.dot(mass_bsp , y0_bsp)) 

                xl    = scipy.interpolate.interp1d(ev, xl, axis=0, bounds_error=True)
                yl    = scipy.interpolate.interp1d(ev, yl, axis=0, bounds_error=True)

                evg   = np.linspace(ev[0], ev[-1], 1000)

                plt.semilogy(evg, xl(evg)[:, 0]  , "-" , markersize=0.2, color='C0', label=r"l=0 w FVM")
                plt.semilogy(evg, yl(evg)[:, 0]  , "--", markersize=0.2, color='C1', label=r"l=0 w Bsp + SPH")
                plt.grid(visible=True)
                plt.legend()
                plt.xlabel(r"energy (eV)")
                plt.title(r"t = %.4E s"%(t))
                plt.ylim(1e-16, 1e1)

                plt.subplot(1, 2, 2)
                plt.semilogy(evg, np.abs(xl(evg)[:, 1])  , "-", markersize=0.2, color='C0', label=r"l=1 w FVM$")
                plt.semilogy(evg, np.abs(yl(evg)[:, 1])  , "--", markersize=0.2, color='C1', label=r"l=1 w Bsp + SPH")
                plt.grid(visible=True)
                plt.legend()
                plt.xlabel(r"energy (eV)")
                plt.title(r"t = %.4E s"%(t))
                plt.ylim(1e-16, 1e1)
            
                plt.tight_layout()
                plt.savefig("%s/%s_nr%d_nvt%d_idx_%04d.png"%(args.odir, args.o, nr, nvt, tidx//io_freq))
                plt.close()

            
            # y0_bsp = self._step(y0_bsp, t, dt, uQRQt_bsp[0], uQRQt_bsp[1], uQRQt_bsp[2], uQRQt_bsp[3], self.cmat_bsp, self.amat_bsp, args.E, verbose=(tidx%io_freq))
            # y0     = self._step(y0,     t, dt, uQRQt_fvm[0], uQRQt_fvm[1], uQRQt_fvm[2], uQRQt_fvm[3], self.cmat_fvm, self.amat_fvm, args.E, verbose=(tidx%io_freq))

            y0_bsp = Linv_bsp @ y0_bsp
            y0     = Linv @ y0

        xp.array(qoi)
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
        plt.savefig("%s/%s_ne_Te_comparison.png"%(args.odir, args.o))
        plt.close()



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
    parser.add_argument("-vmode", "--vmode"                       , help="Semi-Lagrangian (SL) or Eulerian (eulerian)", type=str, default="fvm")
    parser.add_argument("-E", "--E"                               , help="E field", type=float, default=0.0)
    
    
    parser.add_argument("-T" , "--T"                              , help="t final", type=float, default=1e-7)
    parser.add_argument("-dt", "--dt"                             , help="time step size ", type=float, default=1e-10)
    
    parser.add_argument("-odir",  "--odir"                        , help="output dir", type=str, default='vsolve')
    parser.add_argument("-o",  "--o"                              , help="output", type=str, default='vsp')
    parser.add_argument("-io", "--io"                             , help="io every K steps", type=int, default=10)
    
    args = parser.parse_args()
    print(args)

    isExist = os.path.exists(args.odir)
    if not isExist:
       # Create a new directory because it does not exist
       os.makedirs(args.odir)
       print("directory %s is created!"%(args.odir))

    for i in range(1):#(len(args.nr)):
        nr  = args.nr[i]
        nvt = args.nvt[i]
        bte = bte_ops(args, nr, nvt)
        bte.evolve()

    
