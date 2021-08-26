"""
@package : Spectral based, Petrov-Galerkin discretization of the collission integral
"""

import abc
import basis
import spec_spherical as sp
import collisions
import scipy.constants
import numpy as np
import maxpoly
import parameters as params
import utils


class CollissionOp(abc.ABC):

    def __init__(self,dim,p_order):
        self._dim = dim
        self._p = p_order
        pass

    @abc.abstractmethod
    def assemble_mat(collision,maxwellian):
        pass

#SPEC_HERMITE_E = sp.SpectralExpansion(params.BEVelocitySpace.VELOCITY_SPACE_DIM,params.BEVelocitySpace.VEL_SPACE_POLY_ORDER, basis.BasisType.HERMITE_E_POLY)
SPEC_SPHERICAL  = sp.SpectralExpansionSpherical(params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER,basis.Maxwell(),params.BEVelocitySpace.SPH_HARM_LM) 

class CollisionOpSP():
    """
    3D- velocity space, collision operator for species s-collisions
    with background heavy particles, where heavy particles obey the
    direc delta distribution function. 
    """
    def __init__(self,dim,p_order) -> None:
        self._dim  = dim
        self._p    = p_order
        self._spec = SPEC_SPHERICAL

    @staticmethod
    def assemble_mat_v1(collision : collisions.Collisions , maxwellian, qv_pts=None, qs_pts=None):
        """
        Compute the spectral PG discretization for specified Collision. 
        """
        spec  = SPEC_SPHERICAL
        num_p = spec._p +1
        
        if qv_pts is None:
            num_q_pts_on_v      = params.BEVelocitySpace.NUM_Q_PTS_ON_V
        else:
            num_q_pts_on_v      = qv_pts
        
        if qs_pts is None:
            num_q_pts_on_sphere = params.BEVelocitySpace.NUM_Q_PTS_ON_SPHERE
        else:
            num_q_pts_on_sphere = qs_pts
        
        print("collision operator for %d num_qv= %d num_qs=%d " %(collision._type,num_q_pts_on_v,num_q_pts_on_sphere))

        sph_harm_lm         = params.BEVelocitySpace.SPH_HARM_LM 

        num_sph_harm = len(sph_harm_lm)
        [gmx,gmw]    = maxpoly.maxpolygauss(num_q_pts_on_v)
        weight_func  = maxpoly.maxpolyweight
        
        legendre = basis.Legendre()
        [glx,glw] = legendre.Gauss_Pn(num_q_pts_on_sphere)
        theta_q = np.arccos(glx)
        phi_q = np.linspace(0,2*np.pi,2*(num_q_pts_on_sphere))

        L_ij =spec.create_mat()
        spherical_quadrature_fac = (np.pi/num_q_pts_on_sphere)
        qr = np.zeros([num_p, num_sph_harm, num_sph_harm, len(gmx)])

        # pi = p
        # pj = k
        # lm1 = lm
        # lm2 = qs

        V_TH         = collisions.ELECTRON_THEMAL_VEL
        ELE_VOLT     = collisions.ELECTRON_VOLT
        MAXWELLIAN_N = collisions.MAXWELLIAN_N
        AR_NEUTRAL_N = collisions.AR_NEUTRAL_N

        for qi,v_abs in enumerate(gmx): # loop over quadrature points radial
            # \int_S^2 from R^3 integral
            energy_in_ev = (0.5*collisions.MASS_ELECTRON * (v_abs * V_TH)**2) / ELE_VOLT   
            if (energy_in_ev < collision.min_energy_threshold()):
                print("skipping energy : ", energy_in_ev," for v: ",v_abs," Eth : ", collision.min_energy_threshold()," col type: ",collision._type)
                continue
            total_cs     = collision.total_cross_section(energy_in_ev)
            #print(energy_in_ev, " -> ", total_cs)
            for v_theta_i, v_theta in enumerate(theta_q):
                for v_phi in phi_q:
                    v_in         = np.array( utils.spherical_to_cartesian(v_abs * V_TH, v_theta, v_phi) )
                    m_in         = maxwellian(v_abs)
                    #print("incident velocity : ",v_in, " speed: ",np.linalg.norm(v_in,2), "sp: ",[v_abs,v_theta,v_phi])
                    #\int_S^2 from scattering integral
                    for theta_i, theta in enumerate(theta_q):
                        diff_cs      = ( np.linalg.norm(v_in,2) ) *  collision.differential_cross_section(total_cs,energy_in_ev,theta)
                        #print(diff_cs)
                        for phi in phi_q:
                                v_sc_ct        = collision.compute_scattering_velocity(v_in,theta,phi)
                                if(collision._type == collisions.CollisionType.EAR_G2):
                                    # v_sp_sp1- scattered electron v_sc_sp2: ejected electron
                                    v_sc_sp1   = utils.cartesian_to_spherical(v_sc_ct[0][0], v_sc_ct[0][1], v_sc_ct[0][2])
                                    v_sc_sp2   = utils.cartesian_to_spherical(v_sc_ct[1][0], v_sc_ct[1][1], v_sc_ct[1][2])
                                    
                                    v_sc_sp1[0]   = v_sc_sp1[0]/V_TH
                                    v_sc_sp2[0]   = v_sc_sp2[0]/V_TH

                                    mr_sc_sp1  = maxwellian(v_sc_sp1[0]) 
                                    mr_sc_sp2  = maxwellian(v_sc_sp2[0]) 

                                    for pj in range(num_p): # k
                                        for lm2_idx,lm2 in enumerate(sph_harm_lm): # qs
                                            for lm1_idx,lm1 in enumerate(sph_harm_lm): #lm
                                                qr[pj, lm2_idx, lm1_idx, qi] += (AR_NEUTRAL_N*(spherical_quadrature_fac**2) * glw[theta_i] * glw[v_theta_i] \
                                                                            * spec.basis_eval_spherical(v_theta, v_phi, lm2[0], lm2[1]) \
                                                                            * diff_cs\
                                                                            * (mr_sc_sp2*spec.basis_eval_full(v_sc_sp2[0], v_sc_sp2[1], v_sc_sp2[2], pj, lm1[0], lm1[1]) + mr_sc_sp1*spec.basis_eval_full(v_sc_sp1[0], v_sc_sp1[1], v_sc_sp1[2], pj, lm1[0], lm1[1])  -   m_in*spec.basis_eval_full(v_abs,v_theta, v_phi, pj, lm1[0], lm1[1])))
                                else:
                                    v_sc_sp         = utils.cartesian_to_spherical(v_sc_ct[0], v_sc_ct[1], v_sc_ct[2])
                                    # scale back to the thermal velocity
                                    v_sc_sp[0]      = v_sc_sp[0]/V_TH
                                    m_sc_sp  = maxwellian(v_sc_sp[0]) 
                                    for pj in range(num_p): # k
                                        for lm2_idx,lm2 in enumerate(sph_harm_lm): # qs
                                            for lm1_idx,lm1 in enumerate(sph_harm_lm): #lm
                                                qr[pj, lm2_idx, lm1_idx, qi] += (AR_NEUTRAL_N*(spherical_quadrature_fac**2) * glw[theta_i] * glw[v_theta_i] \
                                                                            * spec.basis_eval_spherical(v_theta, v_phi, lm2[0], lm2[1]) \
                                                                            * diff_cs\
                                                                            * (m_sc_sp*spec.basis_eval_full(v_sc_sp[0], v_sc_sp[1], v_sc_sp[2], pj, lm1[0], lm1[1])  -   m_in* spec.basis_eval_full(v_abs,v_theta, v_phi, pj, lm1[0], lm1[1])))

        # for pj in range(num_p): # k
        #     for lm2_idx,lm2 in enumerate(sph_harm_lm): # qs
        #         for lm1_idx,lm1 in enumerate(sph_harm_lm): # lm                                                                        
        #             print("k: ",pj," qs : ",lm2," lm: ",lm1," value : ",qr[pj,lm2_idx,lm1_idx,:])
        for pi in range(num_p):
            for lm2_idx,lm2 in enumerate(sph_harm_lm):
                for pj in range(num_p):
                    for lm1_idx,lm1 in enumerate(sph_harm_lm):
                        for qi,v_abs in enumerate(gmx):
                            i_id = pi*num_sph_harm + lm2_idx
                            j_id = pj*num_sph_harm + lm1_idx
                            w_factor = ((v_abs**2 ) * 1 * (V_TH**3)) / weight_func(v_abs)
                            L_ij[i_id,j_id] += w_factor * gmw[qi] * qr[pj, lm2_idx, lm1_idx, qi]  \
                                * spec.basis_eval_radial(v_abs, pi)

        return L_ij
    
    @staticmethod
    def _Lm(collision : collisions.Collisions , maxwellian):
        """
        computes the - part of the collision operator
        \int_r^3 \int_s2 f(v0) dw
        """
        V_TH         = collisions.ELECTRON_THEMAL_VEL
        ELE_VOLT     = collisions.ELECTRON_VOLT
        MAXWELLIAN_N = collisions.MAXWELLIAN_N
        AR_NEUTRAL_N = collisions.AR_NEUTRAL_N

        NUM_Q_VR     = params.BEVelocitySpace.NUM_Q_VR
        NUM_Q_VT     = params.BEVelocitySpace.NUM_Q_VT
        NUM_Q_VP     = params.BEVelocitySpace.NUM_Q_VP

        NUM_Q_CHI    = params.BEVelocitySpace.NUM_Q_CHI
        NUM_Q_PHI    = params.BEVelocitySpace.NUM_Q_PHI
        spec_sp      = SPEC_SPHERICAL

        num_p        = spec_sp._p +1
        sph_harm_lm  = params.BEVelocitySpace.SPH_HARM_LM 
        num_sph_harm = len(sph_harm_lm)
        [gmx,gmw]    = maxpoly.maxpolygauss(NUM_Q_VR-1)
        weight_func  = maxpoly.maxpolyweight
        
        legendre     = basis.Legendre()
        [glx,glw]    = legendre.Gauss_Pn(NUM_Q_VT)
        VTheta_q      = np.arccos(glx)
        VPhi_q        = np.linspace(0,2*np.pi,NUM_Q_VP)

        [glx_s,glw_s] = legendre.Gauss_Pn(NUM_Q_CHI)
        Chi_q         = np.arccos(glx_s)
        Phi_q         = np.linspace(0,2*np.pi,NUM_Q_PHI)
        
        assert NUM_Q_VP>1
        assert NUM_Q_PHI>1
        sq_fac_v = (2*np.pi/(NUM_Q_VP-1))
        sq_fac_s = (2*np.pi/(NUM_Q_PHI-1))

        WPhi_q   = np.ones(NUM_Q_PHI)*sq_fac_s
        WVPhi_q  = np.ones(NUM_Q_VP)*sq_fac_v

        #trap. weights
        WPhi_q[0]  = 0.5 * WPhi_q[0]
        WPhi_q[-1] = 0.5 * WPhi_q[-1]

        WVPhi_q[0]  = 0.5 * WVPhi_q[0]
        WVPhi_q[-1] = 0.5 * WVPhi_q[-1]

        energy_ev = (0.5 * collisions.MASS_ELECTRON * ((gmx*V_TH)**2))/ELE_VOLT
    
        g=collision

        scattering_mg = np.meshgrid(gmx,Chi_q,indexing='ij')
        #diff_cs       = AR_NEUTRAL_N* g.assemble_diff_cs_mat(scattering_mg[0]*V_TH , scattering_mg[1])
        #total_cs_q = 2*np.pi * np.dot(diff_cs,glw_s)
        total_cs_q = AR_NEUTRAL_N * g.total_cross_section(energy_ev)
        
        P_kr = spec_sp.Vq_r(gmx)
        #M_r  = maxwellian(gmx) * ( (gmx*V_TH)**3) * ((V_TH)/weight_func(gmx))
        M_r  = gmx* V_TH
        
        C_r  = M_r * (total_cs_q)
        Wg = gmw.reshape(-1,len(gmw))
        
        U = (P_kr * Wg ) * np.transpose(C_r) 
        U = np.dot(U,np.transpose(P_kr))
        
        num_p = spec_sp._p + 1

        #L_ij =spec_sp.create_mat()
        # num_p = spec_sp._p + 1
        # num_sph_harm = len(sph_harm_lm)
        # for pi in range(num_p):
        #     for pj in range(num_p):
        #         for lm1_idx,lm1 in enumerate(sph_harm_lm):
        #             i_id = pi*num_sph_harm + lm1_idx
        #             j_id = pj*num_sph_harm + lm1_idx
        #             L_ij[i_id,j_id] = U[pi,pj]
        #print(L_ij)
        #return L_ij
        I=np.eye(num_sph_harm)
        Lm = I
        Lm = np.kron(U,Lm).reshape(num_p*num_sph_harm,num_p*num_sph_harm)
        return Lm

    @staticmethod
    def _Lp(collision : collisions.Collisions , maxwellian):
        
        """
        \int_r^3 \int_s2 f(v0) dw
        """
        V_TH         = collisions.ELECTRON_THEMAL_VEL
        ELE_VOLT     = collisions.ELECTRON_VOLT
        MAXWELLIAN_N = collisions.MAXWELLIAN_N
        AR_NEUTRAL_N = collisions.AR_NEUTRAL_N
        
        NUM_Q_VR     = params.BEVelocitySpace.NUM_Q_VR
        NUM_Q_VT     = params.BEVelocitySpace.NUM_Q_VT
        NUM_Q_VP     = params.BEVelocitySpace.NUM_Q_VP

        NUM_Q_CHI    = params.BEVelocitySpace.NUM_Q_CHI
        NUM_Q_PHI    = params.BEVelocitySpace.NUM_Q_PHI

        spec_sp      = SPEC_SPHERICAL

        num_p        = spec_sp._p +1
        sph_harm_lm  = params.BEVelocitySpace.SPH_HARM_LM 
        num_sph_harm = len(sph_harm_lm)
        [gmx,gmw]    = maxpoly.maxpolygauss(NUM_Q_VR-1)
        weight_func  = maxpoly.maxpolyweight
        
        legendre     = basis.Legendre()
        [glx,glw]    = legendre.Gauss_Pn(NUM_Q_VT)
        VTheta_q      = np.arccos(glx)
        VPhi_q        = np.linspace(0,2*np.pi,NUM_Q_VP)

        [glx_s,glw_s] = legendre.Gauss_Pn(NUM_Q_CHI)
        Chi_q         = np.arccos(glx_s)
        Phi_q         = np.linspace(0,2*np.pi,NUM_Q_PHI)
        
        assert NUM_Q_VP>1
        assert NUM_Q_PHI>1
        sq_fac_v = (2*np.pi/(NUM_Q_VP-1))
        sq_fac_s = (2*np.pi/(NUM_Q_PHI-1))

        WPhi_q   = np.ones(NUM_Q_PHI)*sq_fac_s
        WVPhi_q  = np.ones(NUM_Q_VP)*sq_fac_v

        #trap. weights
        WPhi_q[0]  = 0.5 * WPhi_q[0]
        WPhi_q[-1] = 0.5 * WPhi_q[-1]

        WVPhi_q[0]  = 0.5 * WVPhi_q[0]
        WVPhi_q[-1] = 0.5 * WVPhi_q[-1]

        energy_ev = (0.5 * collisions.MASS_ELECTRON * ((gmx*V_TH)**2))/ELE_VOLT
        energy_ev = energy_ev.reshape(len(gmx),1)

        g=collision
        
        scattering_mg = np.meshgrid(gmx,VTheta_q,VPhi_q,Chi_q,Phi_q,indexing='ij')
        diff_cs  = g.assemble_diff_cs_mat(scattering_mg[0]*V_TH , scattering_mg[3]) * AR_NEUTRAL_N
        
        Sd    = g.compute_scattering_velocity_sp(scattering_mg[0]*V_TH,scattering_mg[1],scattering_mg[2],scattering_mg[3],scattering_mg[4])
        
        Pp_kr = spec_sp.Vq_r(Sd[0]/V_TH) 
        Yp_lm = spec_sp.Vq_sph(Sd[1],Sd[2])
        #Mp_r  = maxwellian(Sd[0]/V_TH) * ( (scattering_mg[0]*V_TH)**3) * ((V_TH)/weight_func(scattering_mg[0]))
        Mp_r  = np.exp((Sd[0]/V_TH)**2 - (scattering_mg[0])**2) *(scattering_mg[0]*V_TH)
        
        num_p  = spec_sp._p+1
        num_sh = len(spec_sp._sph_harm_lm)
        #print(num_sh)

        # Ap_klm1 = np.zeros(tuple([num_p,num_sh]) + incident_mg[0].shape)
        # for i in range(num_p):
        #     for j in range(num_sh):
        #         Ap_klm1[i,j] = diff_cs * Mp_r * Pp_kr[i] * Yp_lm[j]

        Ap_klm = np.array([diff_cs * Mp_r* Pp_kr[i] * Yp_lm[j] for i in range(num_p) for j in range(num_sh)])

        Ap_klm = Ap_klm.reshape(tuple([num_p,num_sh]) + scattering_mg[0].shape)

        Bp_klm = np.dot(Ap_klm,WPhi_q)
        Bp_klm = np.dot(Bp_klm,glw_s)

        incident_mg = np.meshgrid(gmx,VTheta_q,VPhi_q,indexing='ij')
        P_pr  = spec_sp.Vq_r(incident_mg[0])
        Y_qs  = spec_sp.Vq_sph(incident_mg[1],incident_mg[2])

        C_pqs = np.array([P_pr[i] * Y_qs[j] for i in range(num_p) for j in range(num_sh)])
        C_pqs = C_pqs.reshape(tuple([num_p,num_sh]) + incident_mg[0].shape)

        D_pqs_klm = np.array([C_pqs[pi,li] * Bp_klm[pj,lj] for pi in range(num_p) for li in range(num_sh) for pj in range(num_p) for lj in range(num_sh)])
        D_pqs_klm = D_pqs_klm.reshape(tuple([num_p,num_sh,num_p,num_sh]) + incident_mg[0].shape)

        D_pqs_klm = np.dot(D_pqs_klm,WVPhi_q)
        D_pqs_klm = np.dot(D_pqs_klm,glw)
        D_pqs_klm = np.dot(D_pqs_klm,gmw)
        
        # Lij =np.zeros((num_p*num_sh,num_p*num_sh))
        # for pi in range(num_p):
        #     for li in range(num_sh):
        #         for pj in range(num_p):
        #             for lj in range(num_sh):
        #                 Lij[pi*num_sh+li,pj*num_sh+lj] = D_pqs_klm[pi,li,pj,lj]
        # print(Lij-D_pqs_klm)

        D_pqs_klm = D_pqs_klm.reshape((num_p*num_sh,num_p*num_sh))
        return D_pqs_klm
    
    @staticmethod
    def _Lm_l(collision,maxwellian):
        """
        loop based (slow) + part of the coll op
        \int_r^3 \int_s2 f(v0) dw
        """
        V_TH         = collisions.ELECTRON_THEMAL_VEL
        ELE_VOLT     = collisions.ELECTRON_VOLT
        MAXWELLIAN_N = collisions.MAXWELLIAN_N
        AR_NEUTRAL_N = collisions.AR_NEUTRAL_N
        
        NUM_Q_VR     = params.BEVelocitySpace.NUM_Q_VR
        NUM_Q_VT     = params.BEVelocitySpace.NUM_Q_VT
        NUM_Q_VP     = params.BEVelocitySpace.NUM_Q_VP

        NUM_Q_CHI    = params.BEVelocitySpace.NUM_Q_CHI
        NUM_Q_PHI    = params.BEVelocitySpace.NUM_Q_PHI

        spec_sp      = SPEC_SPHERICAL

        num_p        = spec_sp._p +1
        sph_harm_lm  = params.BEVelocitySpace.SPH_HARM_LM 
        num_sph_harm = len(sph_harm_lm)
        [gmx,gmw]    = maxpoly.maxpolygauss(NUM_Q_VR-1)
        weight_func  = maxpoly.maxpolyweight
        
        legendre     = basis.Legendre()
        [glx,glw]    = legendre.Gauss_Pn(NUM_Q_VT)
        VTheta_q      = np.arccos(glx)
        VPhi_q        = np.linspace(0,2*np.pi,NUM_Q_VP)

        [glx_s,glw_s] = legendre.Gauss_Pn(NUM_Q_CHI)
        Chi_q         = np.arccos(glx_s)
        Phi_q         = np.linspace(0,2*np.pi,NUM_Q_PHI)
        
        assert NUM_Q_VP>1
        assert NUM_Q_PHI>1
        sq_fac_v = (2*np.pi/(NUM_Q_VP-1))
        sq_fac_s = (2*np.pi/(NUM_Q_PHI-1))

        WPhi_q   = np.ones(NUM_Q_PHI)*sq_fac_s
        WVPhi_q  = np.ones(NUM_Q_VP)*sq_fac_v

        #trap. weights
        WPhi_q[0]  = 0.5 * WPhi_q[0]
        WPhi_q[-1] = 0.5 * WPhi_q[-1]

        WVPhi_q[0]  = 0.5 * WVPhi_q[0]
        WVPhi_q[-1] = 0.5 * WVPhi_q[-1]

        energy_ev = (0.5 * collisions.MASS_ELECTRON * ((gmx*V_TH)**2))/ELE_VOLT
        energy_ev = energy_ev.reshape(len(gmx),1)

        g=collision

        L_ij =spec_sp.create_mat()
        
        for qi,v_abs in enumerate(gmx):
            m_in         = maxwellian(v_abs)
            energy_in_ev = (0.5*collisions.MASS_ELECTRON * (v_abs * V_TH)**2) / ELE_VOLT
            total_cs     = AR_NEUTRAL_N* g.total_cross_section(energy_in_ev)
            #w_factor = (((v_abs * V_TH)**3) * m_in * (V_TH)) / weight_func(v_abs)
            w_factor = (v_abs*V_TH)
            for pi in range(num_p):
                for lm2_idx,lm2 in enumerate(sph_harm_lm):
                    for pj in range(num_p):
                        for lm1_idx,lm1 in enumerate(sph_harm_lm):
                            if(lm1_idx is not lm2_idx):
                                continue
                            i_id = pi*num_sph_harm + lm2_idx
                            j_id = pj*num_sph_harm + lm1_idx
                            L_ij[i_id,j_id] += total_cs* w_factor * gmw[qi] * spec_sp.basis_eval_radial(v_abs, pi) * spec_sp.basis_eval_radial(v_abs, pj)



        return L_ij

    @staticmethod
    def _Lp_l(collision,maxwellian):
        """
        loop based (slow)  + part of the coll op
        \int_r^3 \int_s2 f(v1) dw
        """
        V_TH         = collisions.ELECTRON_THEMAL_VEL
        ELE_VOLT     = collisions.ELECTRON_VOLT
        MAXWELLIAN_N = collisions.MAXWELLIAN_N
        AR_NEUTRAL_N = collisions.AR_NEUTRAL_N
        
        NUM_Q_VR     = params.BEVelocitySpace.NUM_Q_VR
        NUM_Q_VT     = params.BEVelocitySpace.NUM_Q_VT
        NUM_Q_VP     = params.BEVelocitySpace.NUM_Q_VP

        NUM_Q_CHI    = params.BEVelocitySpace.NUM_Q_CHI
        NUM_Q_PHI    = params.BEVelocitySpace.NUM_Q_PHI

        spec_sp      = SPEC_SPHERICAL

        num_p        = spec_sp._p +1
        sph_harm_lm  = params.BEVelocitySpace.SPH_HARM_LM 
        num_sph_harm = len(sph_harm_lm)
        [gmx,gmw]    = maxpoly.maxpolygauss(NUM_Q_VR-1)
        weight_func  = maxpoly.maxpolyweight
        
        legendre     = basis.Legendre()
        [glx,glw]    = legendre.Gauss_Pn(NUM_Q_VT)
        VTheta_q      = np.arccos(glx)
        VPhi_q        = np.linspace(0,2*np.pi,NUM_Q_VP)

        [glx_s,glw_s] = legendre.Gauss_Pn(NUM_Q_CHI)
        Chi_q         = np.arccos(glx_s)
        Phi_q         = np.linspace(0,2*np.pi,NUM_Q_PHI)
        
        assert NUM_Q_VP>1
        assert NUM_Q_PHI>1
        sq_fac_v = (2*np.pi/(NUM_Q_VP-1))
        sq_fac_s = (2*np.pi/(NUM_Q_PHI-1))

        WPhi_q   = np.ones(NUM_Q_PHI)*sq_fac_s
        WVPhi_q  = np.ones(NUM_Q_VP)*sq_fac_v

        #trap. weights
        WPhi_q[0]  = 0.5 * WPhi_q[0]
        WPhi_q[-1] = 0.5 * WPhi_q[-1]

        WVPhi_q[0]  = 0.5 * WVPhi_q[0]
        WVPhi_q[-1] = 0.5 * WVPhi_q[-1]


        L_ij     = spec_sp.create_mat()
        
        for qi,v_abs in enumerate(gmx):
            energy_in_ev = (0.5*collisions.MASS_ELECTRON * (v_abs * V_TH)**2) / ELE_VOLT
            total_cs     = collision.total_cross_section(energy_in_ev)
            for v_theta_i, v_theta in enumerate(VTheta_q):
                for v_phi_i,v_phi in enumerate(VPhi_q):
                    for theta_i, theta in enumerate(Chi_q):
                        #print("in : (%f ,%f)" %(v_theta,v_phi))
                        diff_cs  = AR_NEUTRAL_N *  collision.differential_cross_section(total_cs,energy_in_ev,theta)
                        #print("diff_cs: ", diff_cs)
                        for phi_i, phi in enumerate(Phi_q):
                            vs   = collision.compute_scattering_velocity_sp(v_abs*V_TH,v_theta,v_phi,theta,phi)
                            #print(vs)
                            #m_sc = maxwellian(vs[0]/V_TH)
                            #w_factor = (((v_abs*V_TH)**3 ) * m_sc * (V_TH)) / weight_func(v_abs)
                            w_factor = np.exp((vs[0]/V_TH)**2 - (v_abs)**2) * (v_abs * V_TH)
                            #print("w_factor : ", w_factor)
                            for pi in range(num_p):
                                for lm2_idx,lm2 in enumerate(sph_harm_lm):
                                    for pj in range(num_p):
                                        for lm1_idx,lm1 in enumerate(sph_harm_lm):
                                            i_id = pi*num_sph_harm + lm2_idx
                                            j_id = pj*num_sph_harm + lm1_idx
                                            L_ij[i_id,j_id] += diff_cs * w_factor * gmw[qi] * glw[v_theta_i] * glw_s[theta_i] * WVPhi_q[v_phi_i] * WPhi_q[phi_i]* spec_sp.basis_eval_full(v_abs,v_theta,v_phi, pi,lm2[0],lm2[1]) * spec_sp.basis_eval_full((vs[0]/V_TH),vs[1],vs[2],pj,lm1[0],lm1[1])
        return L_ij

    @staticmethod
    def assemble_mat(collision : collisions.Collisions , maxwellian):
        Lij = CollisionOpSP._Lp(collision,maxwellian)-CollisionOpSP._Lm(collision,maxwellian)
        return Lij
        
        

