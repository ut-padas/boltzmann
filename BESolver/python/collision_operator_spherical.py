"""
@package : Spectral based, Petrov-Galerkin discretization of the collission integral
"""

import abc
import basis
import spec_spherical as sp
import collisions
import scipy.constants
import numpy as np
import parameters as params
import time

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
    def _Lm(collision : collisions.Collisions , maxwellian, vth):
        """
        computes the - part of the collision operator
        \int_r^3 \int_s2 f(v0) dw
        """
        V_TH         = vth
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
        [gmx,gmw]    = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR)
        weight_func  = spec_sp._basis_p.Wx()
        
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

        g=collision
        scattering_mg = np.meshgrid(gmx,Chi_q,indexing='ij')
        
        # we need to use the integrated diff cross section to better preserve mass for elastic collisions. 
        # Using the total cross section directly can cause mass error if your quadrature is not accurate enough to
        # integrate diff cross section for a given temperature. 
        diff_cs       = g.get_cross_section_scaling() * g.assemble_diff_cs_mat(scattering_mg[0]*V_TH , scattering_mg[1])
        total_cs_q    = 2*np.pi * np.dot(diff_cs,glw_s)
        #total_cs_q = AR_NEUTRAL_N * g.total_cross_section(energy_ev)
        
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
    def _Lp(collision : collisions.Collisions , maxwellian, vth):
        
        """
        \int_r^3 \int_s2 f(v0) dw
        """
        V_TH         = vth 
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
        [gmx,gmw]    = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR)
        weight_func  = spec_sp._basis_p.Wx()
        
        legendre     = basis.Legendre()
        [glx,glw]    = legendre.Gauss_Pn(NUM_Q_VT)
        VTheta_q     = np.arccos(glx)
        VPhi_q       = np.linspace(0,2*np.pi,NUM_Q_VP)

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

        g=collision
        scattering_mg = np.meshgrid(gmx,VTheta_q,VPhi_q,Chi_q,Phi_q,indexing='ij')
        diff_cs       = g.assemble_diff_cs_mat(scattering_mg[0]*V_TH , scattering_mg[3]) * g.get_cross_section_scaling()
        num_p         = spec_sp._p+1
        num_sh        = len(spec_sp._sph_harm_lm)

        # basis evaluated at incident directions. 
        incident_mg = np.meshgrid(gmx,VTheta_q,VPhi_q,indexing='ij')
        P_pr  = spec_sp.Vq_r(incident_mg[0])
        Y_qs  = spec_sp.Vq_sph(incident_mg[1],incident_mg[2])
        C_pqs = np.array([P_pr[i] * Y_qs[j] for i in range(num_p) for j in range(num_sh)])
        C_pqs = C_pqs.reshape(tuple([num_p,num_sh]) + incident_mg[0].shape)
        
        if(g._type == collisions.CollisionType.EAR_G0 or g._type == collisions.CollisionType.EAR_G1):
            Sd      = g.post_scattering_velocity_sp(scattering_mg[0]*V_TH,scattering_mg[1],scattering_mg[2],scattering_mg[3],scattering_mg[4])
            Pp_kr   = spec_sp.Vq_r(Sd[0]/V_TH) 
            Yp_lm   = spec_sp.Vq_sph(Sd[1],Sd[2])
            Mp_r    = (scattering_mg[0]*V_TH)

            Ap_klm = np.array([diff_cs * Mp_r* Pp_kr[i] * Yp_lm[j] for i in range(num_p) for j in range(num_sh)])
            Ap_klm = Ap_klm.reshape(tuple([num_p,num_sh]) + scattering_mg[0].shape)

            Bp_klm = np.dot(Ap_klm,WPhi_q)
            Bp_klm = np.dot(Bp_klm,glw_s)

            D_pqs_klm = np.array([Bp_klm[pi,li] * C_pqs[pj,lj] for pi in range(num_p) for li in range(num_sh) for pj in range(num_p) for lj in range(num_sh)])
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
        elif(g._type == collisions.CollisionType.EAR_G2):
            Sd_particles = g.post_scattering_velocity_sp(scattering_mg[0]*V_TH,scattering_mg[1],scattering_mg[2],scattering_mg[3],scattering_mg[4])
            Lp_mat       = np.zeros((num_p*num_sh,num_p*num_sh))
            for Sd in Sd_particles:
                Pp_kr   = spec_sp.Vq_r(Sd[0]/V_TH) 
                Yp_lm   = spec_sp.Vq_sph(Sd[1],Sd[2])
                Mp_r    = (scattering_mg[0]*V_TH)
                num_p   = spec_sp._p+1
                num_sh  = len(spec_sp._sph_harm_lm)
                
                Ap_klm = np.array([diff_cs * Mp_r* Pp_kr[i] * Yp_lm[j] for i in range(num_p) for j in range(num_sh)])
                Ap_klm = Ap_klm.reshape(tuple([num_p,num_sh]) + scattering_mg[0].shape)

                Bp_klm = np.dot(Ap_klm,WPhi_q)
                Bp_klm = np.dot(Bp_klm,glw_s)

                D_pqs_klm = np.array([Bp_klm[pi,li] * C_pqs[pj,lj] for pi in range(num_p) for li in range(num_sh) for pj in range(num_p) for lj in range(num_sh)])
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
                Lp_mat   += D_pqs_klm
            return Lp_mat

    @staticmethod
    def _Lm_l(collision,maxwellian, vth):
        """
        loop based (slow) + part of the coll op
        \int_r^3 \int_s2 f(v0) dw
        """
        V_TH         = vth 
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
        [gmx,gmw]    = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR)
        weight_func  = spec_sp._basis_p.Wx()
        
        legendre     = basis.Legendre()
        [glx,glw]    = legendre.Gauss_Pn(NUM_Q_VT)
        VTheta_q     = np.arccos(glx)
        VPhi_q       = np.linspace(0,2*np.pi,NUM_Q_VP)

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

        g=collision
        scattering_mg = np.meshgrid(gmx,Chi_q,indexing='ij')
        # we need to use the integrated diff cross section to better preserve mass for elastic collisions. 
        # Using the total cross section directly can cause mass error if your quadrature is not accurate enough to
        # integrate diff cross section for a given temperature. 
        diff_cs       = g.get_cross_section_scaling() * g.assemble_diff_cs_mat(scattering_mg[0]*V_TH , scattering_mg[1])
        total_cs_q    = 2*np.pi * np.dot(diff_cs,glw_s)
        #print(gmx)
        #print(total_cs_q)

        L_ij =spec_sp.create_mat()
        for qi,v_abs in enumerate(gmx):
            m_in         = maxwellian(v_abs)
            energy_in_ev = (0.5*collisions.MASS_ELECTRON * (v_abs * V_TH)**2) / ELE_VOLT
            w_factor = (v_abs*V_TH)
            for pi in range(num_p):
                for lm2_idx,lm2 in enumerate(sph_harm_lm):
                    for pj in range(num_p):
                        for lm1_idx,lm1 in enumerate(sph_harm_lm):
                            if(lm1_idx is not lm2_idx):
                                continue
                            i_id = pi*num_sph_harm + lm2_idx
                            j_id = pj*num_sph_harm + lm1_idx
                            L_ij[i_id,j_id] += total_cs_q[qi]* w_factor * gmw[qi] * spec_sp.basis_eval_radial(v_abs, pi) * spec_sp.basis_eval_radial(v_abs, pj)



        return L_ij

    @staticmethod
    def _Lp_l(collision,maxwellian, vth):
        """
        loop based (slow)  + part of the coll op
        \int_r^3 \int_s2 f(v1) dw
        """
        V_TH         = vth 
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
        [gmx,gmw]    = spec_sp._basis_p.Gauss_Pn(NUM_Q_VR)
        weight_func  = spec_sp._basis_p.Wx()
        
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
        g=collision
        scattering_mg = np.meshgrid(gmx,Chi_q,indexing='ij')
        # we need to use the integrated diff cross section to better preserve mass for elastic collisions. 
        # Using the total cross section directly can cause mass error if your quadrature is not accurate enough to
        # integrate diff cross section for a given temperature. 
        diff_cs       = g.get_cross_section_scaling() * g.assemble_diff_cs_mat(scattering_mg[0]*V_TH , scattering_mg[1])
        total_cs_q    = 2*np.pi * np.dot(diff_cs,glw_s)
        
        for qi,v_abs in enumerate(gmx):
            energy_in_ev = (0.5*collisions.MASS_ELECTRON * (v_abs * V_TH)**2) / ELE_VOLT
            total_cs     = g.get_cross_section_scaling() * g.total_cross_section(energy_in_ev) #total_cs_q[qi] #collision.total_cross_section(energy_in_ev)
            for v_theta_i, v_theta in enumerate(VTheta_q):
                for v_phi_i,v_phi in enumerate(VPhi_q):
                    for theta_i, theta in enumerate(Chi_q):
                        #print("in : (%f ,%f)" %(v_theta,v_phi))
                        diff_cs  = collision.differential_cross_section(total_cs,energy_in_ev,theta)
                        #print("diff_cs: ", diff_cs)
                        for phi_i, phi in enumerate(Phi_q):
                            vs   = collision.post_scattering_velocity_sp(v_abs*V_TH,v_theta,v_phi,theta,phi)
                            w_factor = (v_abs * V_TH)
                            for pi in range(num_p):
                                for lm2_idx,lm2 in enumerate(sph_harm_lm):
                                    for pj in range(num_p):
                                        for lm1_idx,lm1 in enumerate(sph_harm_lm):
                                            i_id = pi*num_sph_harm + lm2_idx
                                            j_id = pj*num_sph_harm + lm1_idx
                                            L_ij[i_id,j_id] += diff_cs * w_factor * gmw[qi] * glw[v_theta_i] * glw_s[theta_i] * WVPhi_q[v_phi_i] * WPhi_q[phi_i]* spec_sp.basis_eval_full(v_abs,v_theta,v_phi, pj,lm1[0],lm1[1]) * spec_sp.basis_eval_full((vs[0]/V_TH),vs[1],vs[2],pi,lm2[0],lm2[1])
        return L_ij

    @staticmethod
    def assemble_mat(collision : collisions.Collisions , maxwellian, vth):
        Lij = CollisionOpSP._Lp(collision,maxwellian,vth)-CollisionOpSP._Lm(collision,maxwellian,vth)
        return Lij
        
        

