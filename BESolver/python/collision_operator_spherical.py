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

class CollisionOpSP():
    """
    3D- velocity space, collision operator for species s-collisions
    with background heavy particles, where heavy particles obey the
    direc delta distribution function. 
    """
    
    def __init__(self,dim,p_order,q_mode=sp.QuadMode.GMX) -> None:
        self._dim  = dim
        self._p    = p_order
        self._spec = sp.SpectralExpansionSpherical(self._p,basis.Maxwell(),params.BEVelocitySpace.SPH_HARM_LM)
        self._spec._q_mode = q_mode
        self.__alloc_internal_vars()

    def __alloc_internal_vars(self):
        self._NUM_Q_VR         = params.BEVelocitySpace.NUM_Q_VR
        self._NUM_Q_VT         = params.BEVelocitySpace.NUM_Q_VT
        self._NUM_Q_VP         = params.BEVelocitySpace.NUM_Q_VP
        self._NUM_Q_CHI        = params.BEVelocitySpace.NUM_Q_CHI
        self._NUM_Q_PHI        = params.BEVelocitySpace.NUM_Q_PHI
        self._sph_harm_lm      = params.BEVelocitySpace.SPH_HARM_LM 
        spec_sp                = self._spec
        self._num_p            = spec_sp._p +1
        self._num_sh           = len(spec_sp._sph_harm_lm)

        if spec_sp._q_mode == sp.QuadMode.GMX:
            [self._gmx,self._gmw]  = spec_sp._basis_p.Gauss_Pn(self._NUM_Q_VR)
        elif spec_sp._q_mode == sp.QuadMode.SIMPSON:
            num_q                  = self._NUM_Q_VR
            [self._gmx,self._gmw]  = basis.uniform_simpson((1e-8,10),num_q)
        
        weight_func            = spec_sp._basis_p.Wx()

        legendre                  = basis.Legendre()
        [self._glx,self._glw]     = legendre.Gauss_Pn(self._NUM_Q_VT)
        self._VTheta_q            = np.arccos(self._glx)
        self._VPhi_q              = np.linspace(0,2*np.pi,self._NUM_Q_VP)

        [self._glx_s,self._glw_s] = legendre.Gauss_Pn(self._NUM_Q_CHI)
        Chi_q                     = np.arccos(self._glx_s)
        Phi_q                     = np.linspace(0,2*np.pi,self._NUM_Q_PHI)

        assert self._NUM_Q_VP>1
        assert self._NUM_Q_PHI>1
        sq_fac_v = (2*np.pi/(self._NUM_Q_VP-1))
        sq_fac_s = (2*np.pi/(self._NUM_Q_PHI-1))

        self._WPhi_q   = np.ones(self._NUM_Q_PHI)*sq_fac_s
        self._WVPhi_q  = np.ones(self._NUM_Q_VP)*sq_fac_v

        #trap. weights
        self._WPhi_q[0]  = 0.5 * self._WPhi_q[0]
        self._WPhi_q[-1] = 0.5 * self._WPhi_q[-1]

        self._WVPhi_q[0]  = 0.5 * self._WVPhi_q[0]
        self._WVPhi_q[-1] = 0.5 * self._WVPhi_q[-1]

        # basis evaluated at incident directions.  ( + term)
        self._incident_mg_plus = np.meshgrid(self._gmx,self._VTheta_q,self._VPhi_q,indexing='ij')
        self._scattering_mg_plus = np.meshgrid(self._gmx,self._VTheta_q,self._VPhi_q,Chi_q,Phi_q,indexing='ij')
        self._P_pr  = spec_sp.Vq_r(self._incident_mg_plus[0])
        self._Y_qs  = spec_sp.Vq_sph(self._incident_mg_plus[1],self._incident_mg_plus[2])
        
        self._C_pqs = np.array([self._P_pr[i] * self._Y_qs[j] for i in range(self._num_p) for j in range(self._num_sh)])
        self._C_pqs = self._C_pqs.reshape(tuple([self._num_p,self._num_sh]) + self._incident_mg_plus[0].shape)
        
        # (Lm - term quantities)
        self._scattering_mg_minus = np.meshgrid(self._gmx,Chi_q,indexing='ij')

    def _Lm(self,collision : collisions.Collisions , maxwellian, vth):
        """
        computes the - part of the collision operator
        \int_r^3 \int_s2 f(v0) dw
        """
        V_TH         = vth
        ELE_VOLT     = collisions.ELECTRON_VOLT
        MAXWELLIAN_N = collisions.MAXWELLIAN_N
        AR_NEUTRAL_N = collisions.AR_NEUTRAL_N
        
        g=collision
        scattering_mg = self._scattering_mg_minus
        glw_s         = self._glw_s
        spec_sp       = self._spec
        gmx           = self._gmx
        gmw           = self._gmw
        num_sh        = self._num_sh
        
        # we need to use the integrated diff cross section to better preserve mass for elastic collisions. 
        # Using the total cross section directly can cause mass error if your quadrature is not accurate enough to
        # integrate diff cross section for a given temperature. 
        diff_cs       = g.get_cross_section_scaling() * g.assemble_diff_cs_mat(scattering_mg[0]*V_TH , scattering_mg[1])
        total_cs_q    = 2*np.pi * np.dot(diff_cs,glw_s)
        #total_cs_q = AR_NEUTRAL_N * g.total_cross_section(energy_ev)
        
        P_kr = spec_sp.Vq_r(gmx)

        if spec_sp._q_mode == sp.QuadMode.GMX:
            M_r =  gmx * V_TH 
        elif spec_sp._q_mode == sp.QuadMode.SIMPSON:
            M_r  = np.sqrt(16/np.pi)*np.exp(-gmx**2) * (gmx**3) *V_TH 
        else:
            Mr=None
            print("Unknown quadrature mode- inner product weight function is set to none.")

        C_r  = M_r * (total_cs_q)
        Wg = gmw.reshape(-1,len(gmw))
        
        U = (P_kr * Wg ) * np.transpose(C_r) 
        U = np.dot(U,np.transpose(P_kr))
        
        num_p = spec_sp._p + 1
        I=np.eye(num_sh)
        Lm = I
        Lm = np.kron(U,Lm).reshape(num_p*num_sh,num_p*num_sh)
        return Lm

    def _Lp(self, collision : collisions.Collisions , maxwellian, vth):
        
        """
        \int_r^3 \int_s2 f(v0) dw
        """
        V_TH         = vth     
        ELE_VOLT     = collisions.ELECTRON_VOLT
        MAXWELLIAN_N = collisions.MAXWELLIAN_N
        AR_NEUTRAL_N = collisions.AR_NEUTRAL_N

        g=collision
        scattering_mg = self._scattering_mg_plus
        incident_mg   = self._incident_mg_plus
        spec_sp       = self._spec
        num_p         = self._num_p
        num_sh        = self._num_sh

        WPhi_q        = self._WPhi_q
        WVPhi_q       = self._WVPhi_q
        glw_s         = self._glw_s
        glw           = self._glw
        gmw           = self._gmw
        C_pqs         = self._C_pqs

        diff_cs      = g.assemble_diff_cs_mat(scattering_mg[0]*V_TH , scattering_mg[3]) * g.get_cross_section_scaling()
        if(g._type == collisions.CollisionType.EAR_G0 or g._type == collisions.CollisionType.EAR_G1):
            Sd      = g.post_scattering_velocity_sp(scattering_mg[0]*V_TH,scattering_mg[1],scattering_mg[2],scattering_mg[3],scattering_mg[4])
            Pp_kr   = spec_sp.Vq_r(Sd[0]/V_TH) 
            Yp_lm   = spec_sp.Vq_sph(Sd[1],Sd[2])

            if spec_sp._q_mode == sp.QuadMode.GMX:
                Mp_r    = (scattering_mg[0])*V_TH
            elif spec_sp._q_mode == sp.QuadMode.SIMPSON:
                Mp_r    = np.sqrt(16/np.pi) * np.exp(-scattering_mg[0]**2) * (scattering_mg[0]**3) * V_TH
            else:
                Mr=None
                print("Unknown quadrature mode- inner product weight function is set to none.")
            
            Ap_klm = np.array([diff_cs * Mp_r* Pp_kr[i] * Yp_lm[j] for i in range(num_p) for j in range(num_sh)])
            Ap_klm = Ap_klm.reshape(tuple([num_p,num_sh]) + scattering_mg[0].shape)

            Bp_klm = np.dot(Ap_klm,WPhi_q)
            Bp_klm = np.dot(Bp_klm,glw_s)

            D_pqs_klm = np.array([Bp_klm[pi,li] * C_pqs[pj,lj] for pi in range(num_p) for li in range(num_sh) for pj in range(num_p) for lj in range(num_sh)])
            D_pqs_klm = D_pqs_klm.reshape(tuple([num_p,num_sh,num_p,num_sh]) + incident_mg[0].shape)

            D_pqs_klm = np.dot(D_pqs_klm,WVPhi_q)
            D_pqs_klm = np.dot(D_pqs_klm,glw)
            D_pqs_klm = np.dot(D_pqs_klm,gmw)
            
            D_pqs_klm = D_pqs_klm.reshape((num_p*num_sh,num_p*num_sh))
            return D_pqs_klm

        elif(g._type == collisions.CollisionType.EAR_G2):

            Sd_particles = g.post_scattering_velocity_sp(scattering_mg[0]*V_TH,scattering_mg[1],scattering_mg[2],scattering_mg[3],scattering_mg[4])
            Lp_mat       = np.zeros((num_p*num_sh,num_p*num_sh))
            for Sd in Sd_particles:
                Pp_kr   = spec_sp.Vq_r(Sd[0]/V_TH) 
                Yp_lm   = spec_sp.Vq_sph(Sd[1],Sd[2])

                if spec_sp._q_mode == sp.QuadMode.GMX:
                    Mp_r    = (scattering_mg[0]*V_TH)
                elif spec_sp._q_mode == sp.QuadMode.SIMPSON:
                    Mp_r    = np.sqrt(16/np.pi) * np.exp(-scattering_mg[0]**2) * (scattering_mg[0]**3) * V_TH
                else:
                    Mr=None
                    print("Unknown quadrature mode- inner product weight function is set to none.")

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
                
                D_pqs_klm = D_pqs_klm.reshape((num_p*num_sh,num_p*num_sh))
                Lp_mat   += D_pqs_klm
            return Lp_mat
    
    
    def _Lm_l(self,collision,maxwellian, vth):
        """
        loop based (slow) + part of the coll op
        \int_r^3 \int_s2 f(v0) dw
        !Note: only works with gauss-maxwell quadrature points
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

        spec_sp      = self._spec

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

    
    def _Lp_l(self,collision,maxwellian, vth):
        """
        loop based (slow)  + part of the coll op
        \int_r^3 \int_s2 f(v1) dw
        !Note: only works with gauss-maxwell quadrature points
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

        spec_sp      = self._spec

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

    def assemble_mat(self,collision : collisions.Collisions , maxwellian, vth):
        Lij = self._Lp(collision,maxwellian,vth)-self._Lm(collision,maxwellian,vth)
        return Lij
        
        

