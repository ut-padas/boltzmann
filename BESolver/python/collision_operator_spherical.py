"""
@package : Spectral based, Petrov-Galerkin discretization of the collission integral
"""

import abc
import basis
import spec_spherical as sp
import collisions
import scipy
import scipy.constants
import numpy as np
import parameters as params
import time
import utils as BEUtils
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
    
    def __init__(self,dim,p_order,q_mode=sp.QuadMode.GMX,poly_type=basis.BasisType.MAXWELLIAN_POLY,k_domain=(0,10), sig_pts=None, knot_vec=None) -> None:
        self._dim  = dim
        self._p    = p_order
        self._r_basis_type = poly_type

        if self._r_basis_type == basis.BasisType.MAXWELLIAN_POLY:
            self._spec = sp.SpectralExpansionSpherical(self._p,basis.Maxwell(),params.BEVelocitySpace.SPH_HARM_LM)
            self._spec._q_mode = q_mode
        elif self._r_basis_type == basis.BasisType.MAXWELLIAN_ENERGY_POLY:
            self._spec = sp.SpectralExpansionSpherical(self._p,basis.MaxwellEnergy(),params.BEVelocitySpace.SPH_HARM_LM)
            self._spec._q_mode = q_mode
        elif self._r_basis_type == basis.BasisType.LAGUERRE:
            self._spec = sp.SpectralExpansionSpherical(self._p,basis.Laguerre(),params.BEVelocitySpace.SPH_HARM_LM)
            self._spec._q_mode = q_mode
        elif self._r_basis_type == basis.BasisType.CHEBYSHEV_POLY:
            self._spec = sp.SpectralExpansionSpherical(self._p,basis.Chebyshev(domain=(-1,1), window=k_domain) , params.BEVelocitySpace.SPH_HARM_LM)
            self._spec._q_mode = q_mode
        elif self._r_basis_type == basis.BasisType.SPLINES:
            spline_order = basis.BSPLINE_BASIS_ORDER
            splines      = basis.XlBSpline(k_domain,spline_order,self._p+1, sig_pts=sig_pts, knots_vec=knot_vec)
            self._spec   = sp.SpectralExpansionSpherical(self._p,splines,params.BEVelocitySpace.SPH_HARM_LM)
            self._spec._q_mode = q_mode
            self._spec._num_q_radial        = splines._num_knot_intervals * splines._q_per_knot
            

        self._NUM_Q_VR          = params.BEVelocitySpace.NUM_Q_VR
        self._NUM_Q_VT          = params.BEVelocitySpace.NUM_Q_VT
        self._NUM_Q_VP          = params.BEVelocitySpace.NUM_Q_VP
        self._NUM_Q_CHI         = params.BEVelocitySpace.NUM_Q_CHI
        self._NUM_Q_PHI         = params.BEVelocitySpace.NUM_Q_PHI
        self._sph_harm_lm       = params.BEVelocitySpace.SPH_HARM_LM 
        spec_sp                 = self._spec
        self._num_p             = spec_sp._p +1
        self._num_sh            = len(spec_sp._sph_harm_lm)

        self._gmx,self._gmw     = spec_sp._basis_p.Gauss_Pn(self._NUM_Q_VR)
        legendre                = basis.Legendre()
        self._glx,self._glw     = legendre.Gauss_Pn(self._NUM_Q_VT)
        self._VTheta_q          = np.arccos(self._glx)
        self._VPhi_q            = np.linspace(0,2*np.pi,self._NUM_Q_VP)

        self._glx_s,self._glw_s = legendre.Gauss_Pn(self._NUM_Q_CHI)
        self._Chi_q             = np.arccos(self._glx_s)
        self._Phi_q             = np.linspace(0,2*np.pi,self._NUM_Q_PHI)

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

        return 

    def _LOp_Eulerian(self, collision, maxwellian, vth):

        V_TH          = vth     
        ELE_VOLT      = collisions.ELECTRON_VOLT
        MAXWELLIAN_N  = collisions.MAXWELLIAN_N
        AR_NEUTRAL_N  = collisions.AR_NEUTRAL_N
        
        g             = collision
        spec_sp       = self._spec
        num_p         = self._num_p
        num_sh        = self._num_sh

        if self._r_basis_type == basis.BasisType.CHEBYSHEV_POLY:
            cc_diag = list()
            for e_id, ele_domain in enumerate(spec_sp._r_grid):
                spec_sp._basis_p     = spec_sp._r_basis_p[e_id]
                self._gmx,self._gmw  = spec_sp._basis_p.Gauss_Pn(self._NUM_Q_VR)
                self._incident_mg    = np.meshgrid(self._gmx,self._VTheta_q,self._VPhi_q,indexing='ij')
                self._scattering_mg  = np.meshgrid(self._gmx,self._VTheta_q,self._VPhi_q,self._Chi_q,self._Phi_q,indexing='ij')
                Mp_r                 = (self._scattering_mg[0] * V_TH) * self._spec._basis_p.Wx()(self._scattering_mg[0])

                g.reset_scattering_direction_sp_mat()
                diff_cs = g.assemble_diff_cs_mat(self._scattering_mg[0]*V_TH , self._scattering_mg[3]) * g.get_cross_section_scaling()
                Sd      = g.post_scattering_velocity_sp(self._scattering_mg[0]*V_TH, self._scattering_mg[1], self._scattering_mg[2],self._scattering_mg[3],self._scattering_mg[4])
                Sd[0]  /=V_TH

                sph_post   = spec_sp.Vq_sph(Sd[1],Sd[2])
                sph_pre    = spec_sp.Vq_sph(self._scattering_mg[1], self._scattering_mg[2])
                sph_pre_in = spec_sp.Vq_sph(self._incident_mg[1]  , self._incident_mg[2])

                cc_collision = np.zeros((num_p * num_sh, num_p * num_sh))
                tmp0 = np.zeros(tuple([num_p]) + self._scattering_mg[0].shape)
                tmp1 = np.zeros(tuple([num_p]) + self._incident_mg[0].shape)
                tmp2 = np.zeros(self._incident_mg[0].shape)

                if(g._type == collisions.CollisionType.EAR_G0 or g._type == collisions.CollisionType.EAR_G1):
                    for qs_idx, (q,s) in enumerate(self._sph_harm_lm):
                        tmp0[:,:,:,:,:,:]  = diff_cs * Mp_r * (spec_sp.dg_Vq_r(Sd[0], q, e_id) * sph_post[qs_idx] - spec_sp.dg_Vq_r(self._scattering_mg[0], q, e_id) * sph_pre[qs_idx])
                        tmp1[:,:,:,:]      = np.dot(np.dot(tmp0,self._WPhi_q),self._glw_s)

                        for p in range(num_p):
                            for k in range(num_p):
                                for lm_idx, (l,m) in enumerate(self._sph_harm_lm):
                                    tmp2[:,:,:] = tmp1[p,:,:,:] * spec_sp.basis_eval_radial(self._incident_mg[0], k, l) * sph_pre_in[lm_idx]
                                    cc_collision[p * num_sh + qs_idx , k * num_sh + lm_idx] = np.dot(np.dot(np.dot(tmp2, self._WVPhi_q),self._glw), self._gmw)

                elif(g._type == collisions.CollisionType.EAR_G2):
                    for qs_idx, (q,s) in enumerate(self._sph_harm_lm):
                        tmp0[:,:,:,:,:,:]  = diff_cs * Mp_r * (2 * spec_sp.dg_Vq_r(Sd[0],q,e_id) * sph_post[qs_idx] - spec_sp.dg_Vq_r(self._scattering_mg[0],q, e_id) * sph_pre[qs_idx])
                        tmp1[:,:,:,:]      = np.dot(np.dot(tmp0,self._WPhi_q),self._glw_s)

                        for p in range(num_p):
                            for k in range(num_p):
                                for lm_idx, (l,m) in enumerate(self._sph_harm_lm):
                                    tmp2[:,:,:] = tmp1[p,:,:,:] * spec_sp.basis_eval_radial(self._incident_mg[0], k, l) * sph_pre_in[lm_idx]
                                    cc_collision[p * num_sh + qs_idx , k * num_sh + lm_idx] = np.dot(np.dot(np.dot(tmp2, self._WVPhi_q),self._glw),self._gmw)
                
                cc_diag.append(cc_collision)
            
            return scipy.linalg.block_diag(*cc_diag)
        elif self._r_basis_type == basis.BasisType.SPLINES:
            self._gmx,self._gmw  = spec_sp._basis_p.Gauss_Pn(self._NUM_Q_VR)
            
            k_vec    = self._spec._basis_p._t
            dg_idx   = self._spec._basis_p._dg_idx
            sp_order = self._spec._basis_p._sp_order
            cc_collision = np.zeros((num_p * num_sh, num_p * num_sh))

            # idx_set     = np.logical_and(self._gmx>=xb, self._gmx <=xe)
            # gx_e , gw_e = self._gmx[idx_set],self._gmw[idx_set]
            gx_e , gw_e = self._gmx , self._gmw
            
            g.reset_scattering_direction_sp_mat()
            self._incident_mg    = np.meshgrid(gx_e,self._VTheta_q,self._VPhi_q,indexing='ij')
            self._scattering_mg  = np.meshgrid(gx_e,self._VTheta_q,self._VPhi_q,self._Chi_q,self._Phi_q,indexing='ij')

            diff_cs = g.assemble_diff_cs_mat(self._scattering_mg[0]*V_TH , self._scattering_mg[3]) * g.get_cross_section_scaling()
            Sd      = g.post_scattering_velocity_sp(self._scattering_mg[0]*V_TH, self._scattering_mg[1], self._scattering_mg[2],self._scattering_mg[3],self._scattering_mg[4])
            Sd[0]  /=V_TH

            sph_post   = spec_sp.Vq_sph(Sd[1],Sd[2])
            sph_pre    = spec_sp.Vq_sph(self._scattering_mg[1], self._scattering_mg[2])
            sph_pre_in = spec_sp.Vq_sph(self._incident_mg[1]  , self._incident_mg[2])
            Mp_r       = (self._scattering_mg[0] * V_TH) * (self._scattering_mg[0]**2)

            tmp0 = np.zeros(tuple([num_p]) + self._scattering_mg[0].shape)
            tmp1 = np.zeros(tuple([num_p]) + self._incident_mg[0].shape)
            tmp2 = np.zeros(self._incident_mg[0].shape)


            for e_id in range(0,len(dg_idx),2):
                ib=dg_idx[e_id]
                ie=dg_idx[e_id+1]

                xb=k_vec[ib]
                xe=k_vec[ie+sp_order+1]
                
                if(g._type == collisions.CollisionType.EAR_G0 or g._type == collisions.CollisionType.EAR_G1):
                    for qs_idx, (q,s) in enumerate(self._sph_harm_lm):
                        for p in range(ib,ie+1):
                            tmp0[p,:,:,:,:,:]  = diff_cs * Mp_r * (spec_sp.basis_eval_radial(Sd[0], p, q) * sph_post[qs_idx] - spec_sp.basis_eval_radial(self._scattering_mg[0],p,q) * sph_pre[qs_idx])
                            tmp1[p,:,:,:]      = np.dot(np.dot(tmp0[p],self._WPhi_q),self._glw_s)

                        for p in range(ib,ie+1):
                            for k in range(ib,ie+1):
                                for lm_idx, (l,m) in enumerate(self._sph_harm_lm):
                                    tmp2[:,:,:] = tmp1[p,:,:,:] * spec_sp.basis_eval_radial(self._incident_mg[0], k, l) * sph_pre_in[lm_idx]
                                    cc_collision[p * num_sh + qs_idx , k * num_sh + lm_idx] += np.dot(np.dot(np.dot(tmp2, self._WVPhi_q),self._glw), gw_e)
                                    

                elif(g._type == collisions.CollisionType.EAR_G2):
                    for qs_idx, (q,s) in enumerate(self._sph_harm_lm):
                        for p in range(ib,ie+1):
                            tmp0[p,:,:,:,:,:]  = diff_cs * Mp_r * (2 * spec_sp.Vq_r(Sd[0],q) * sph_post[qs_idx] - spec_sp.Vq_r(self._scattering_mg[0],q) * sph_pre[qs_idx])
                            tmp1[p,:,:,:]      = np.dot(np.dot(tmp0[p],self._WPhi_q),self._glw_s)

                        for p in range(ib,ie+1):
                            for k in range(ib,ie+1):
                                for lm_idx, (l,m) in enumerate(self._sph_harm_lm):
                                    tmp2[:,:,:] = tmp1[p,:,:,:] * spec_sp.basis_eval_radial(self._incident_mg[0], k, l) * sph_pre_in[lm_idx]
                                    cc_collision[p * num_sh + qs_idx , k * num_sh + lm_idx] = np.dot(np.dot(np.dot(tmp2, self._WVPhi_q),self._glw),gw_e)
            
            return cc_collision
        else:
            self._gmx,self._gmw  = spec_sp._basis_p.Gauss_Pn(self._NUM_Q_VR)
            self._incident_mg    = np.meshgrid(self._gmx,self._VTheta_q,self._VPhi_q,indexing='ij')
            self._scattering_mg  = np.meshgrid(self._gmx,self._VTheta_q,self._VPhi_q,self._Chi_q,self._Phi_q,indexing='ij')

            diff_cs = g.assemble_diff_cs_mat(self._scattering_mg[0]*V_TH , self._scattering_mg[3]) * g.get_cross_section_scaling()
            Sd      = g.post_scattering_velocity_sp(self._scattering_mg[0]*V_TH, self._scattering_mg[1], self._scattering_mg[2],self._scattering_mg[3],self._scattering_mg[4])
            Sd[0]  /=V_TH
            
            sph_post   = spec_sp.Vq_sph(Sd[1],Sd[2])
            sph_pre    = spec_sp.Vq_sph(self._scattering_mg[1], self._scattering_mg[2])
            sph_pre_in = spec_sp.Vq_sph(self._incident_mg[1]  , self._incident_mg[2])
            if self._r_basis_type == basis.BasisType.MAXWELLIAN_POLY\
                or self._r_basis_type == basis.BasisType.LAGUERRE\
                or self._r_basis_type == basis.BasisType.MAXWELLIAN_ENERGY_POLY:
                Mp_r    = (self._scattering_mg[0])*V_TH
            else:
                raise NotImplementedError
            
            cc_collision = np.zeros((num_p * num_sh, num_p * num_sh))
            tmp0 = np.zeros(tuple([num_p]) + self._scattering_mg[0].shape)
            tmp1 = np.zeros(tuple([num_p]) + self._incident_mg[0].shape)
            tmp2 = np.zeros(self._incident_mg[0].shape)

            if(g._type == collisions.CollisionType.EAR_G0 or g._type == collisions.CollisionType.EAR_G1):
                for qs_idx, (q,s) in enumerate(self._sph_harm_lm):
                    tmp0[:,:,:,:,:,:]  = diff_cs * Mp_r * (spec_sp.Vq_r(Sd[0],q) * sph_post[qs_idx] - spec_sp.Vq_r(self._scattering_mg[0],q) * sph_pre[qs_idx])
                    tmp1[:,:,:,:]      = np.dot(np.dot(tmp0,self._WPhi_q),self._glw_s)

                    for p in range(num_p):
                        for k in range(num_p):
                            for lm_idx, (l,m) in enumerate(self._sph_harm_lm):
                                tmp2[:,:,:] = tmp1[p,:,:,:] * spec_sp.basis_eval_radial(self._incident_mg[0], k, l) * sph_pre_in[lm_idx]
                                cc_collision[p * num_sh + qs_idx , k * num_sh + lm_idx] = np.dot(np.dot(np.dot(tmp2, self._WVPhi_q),self._glw), self._gmw)

            elif(g._type == collisions.CollisionType.EAR_G2):
                for qs_idx, (q,s) in enumerate(self._sph_harm_lm):
                    tmp0[:,:,:,:,:,:]  = diff_cs * Mp_r * (2 * spec_sp.Vq_r(Sd[0],q) * sph_post[qs_idx] - spec_sp.Vq_r(self._scattering_mg[0],q) * sph_pre[qs_idx])
                    tmp1[:,:,:,:]      = np.dot(np.dot(tmp0,self._WPhi_q),self._glw_s)

                    for p in range(num_p):
                        for k in range(num_p):
                            for lm_idx, (l,m) in enumerate(self._sph_harm_lm):
                                tmp2[:,:,:] = tmp1[p,:,:,:] * spec_sp.basis_eval_radial(self._incident_mg[0], k, l) * sph_pre_in[lm_idx]
                                cc_collision[p * num_sh + qs_idx , k * num_sh + lm_idx] = np.dot(np.dot(np.dot(tmp2, self._WVPhi_q),self._glw),self._gmw)
            
            return cc_collision
        
    def _LOp_reduced_memory(self, collision, maxwellian, vth, v0):
        """
        compute the collision operator without splitting, 
        more floating point friendly, computationally expensive. 
        """

        V_TH         = vth     
        ELE_VOLT     = collisions.ELECTRON_VOLT
        MAXWELLIAN_N = collisions.MAXWELLIAN_N
        AR_NEUTRAL_N = collisions.AR_NEUTRAL_N

        g=collision
        scattering_mg = self._scattering_mg
        incident_mg   = self._incident_mg
        spec_sp       = self._spec
        num_p         = self._num_p
        num_sh        = self._num_sh

        WPhi_q        = self._WPhi_q
        WVPhi_q       = self._WVPhi_q
        glw_s         = self._glw_s
        glw           = self._glw
        gmw           = self._gmw
        
        scattering_mg_cart = BEUtils.spherical_to_cartesian(scattering_mg[0],scattering_mg[1],scattering_mg[2])
        scattering_mg_cart[0] = scattering_mg_cart[0] + np.ones_like(scattering_mg_cart[0]) * v0[0]
        scattering_mg_cart[1] = scattering_mg_cart[1] + np.ones_like(scattering_mg_cart[0]) * v0[1]
        scattering_mg_cart[2] = scattering_mg_cart[2] + np.ones_like(scattering_mg_cart[0]) * v0[2]

        scattering_mg_v0    = BEUtils.cartesian_to_spherical(scattering_mg_cart[0],scattering_mg_cart[1],scattering_mg_cart[2])
        diff_cs      = g.assemble_diff_cs_mat(scattering_mg_v0[0]*V_TH , scattering_mg[3]) * g.get_cross_section_scaling()

        Sd      = g.post_scattering_velocity_sp(scattering_mg_v0[0]*V_TH,scattering_mg_v0[1],scattering_mg_v0[2],scattering_mg[3],scattering_mg[4])
        Sd      = BEUtils.spherical_to_cartesian(Sd[0]/V_TH,Sd[1],Sd[2])

        Sd[0]   = Sd[0] - np.ones_like(Sd[0]) * v0[0]
        Sd[1]   = Sd[1] - np.ones_like(Sd[0]) * v0[1]
        Sd[2]   = Sd[2] - np.ones_like(Sd[0]) * v0[2]

        Sd      = BEUtils.cartesian_to_spherical(Sd[0],Sd[1],Sd[2])

        sph_post   = spec_sp.Vq_sph(Sd[1],Sd[2])
        sph_pre    = spec_sp.Vq_sph(scattering_mg[1],scattering_mg[2])
        sph_pre_in = spec_sp.Vq_sph(incident_mg[1],incident_mg[2])
        
        if self._r_basis_type == basis.BasisType.MAXWELLIAN_POLY\
            or self._r_basis_type == basis.BasisType.LAGUERRE\
            or self._r_basis_type == basis.BasisType.MAXWELLIAN_ENERGY_POLY:
            Mp_r    = (scattering_mg_v0[0])*V_TH
        elif self._r_basis_type == basis.BasisType.SPLINES:
            Mp_r    = (scattering_mg_v0[0] * V_TH) * (scattering_mg[0]**2)
        elif self._r_basis_type == basis.BasisType.CHEBYSHEV_POLY:
            Mp_r    = (scattering_mg_v0[0] * V_TH) * self._spec._basis_p.Wx()(scattering_mg[0])
        else:
            raise NotImplementedError
        
        # more memory efficient but can be slower than the above. 
        cc_collision = np.zeros((num_p * num_sh, num_p * num_sh))
        
        tmp0 = np.zeros(tuple([num_p]) + scattering_mg[0].shape)
        tmp1 = np.zeros(tuple([num_p]) + incident_mg[0].shape)
        tmp2 = np.zeros(incident_mg[0].shape)

        if(g._type == collisions.CollisionType.EAR_G0 or g._type == collisions.CollisionType.EAR_G1):
            for qs_idx, (q,s) in enumerate(self._sph_harm_lm):
                tmp0[:,:,:,:,:,:]  = diff_cs * Mp_r * (spec_sp.Vq_r(Sd[0],q) * sph_post[qs_idx] - spec_sp.Vq_r(scattering_mg[0],q) * sph_pre[qs_idx])
                tmp1[:,:,:,:]      = np.dot(np.dot(tmp0,WPhi_q),glw_s)

                for p in range(num_p):
                    for k in range(num_p):
                        for lm_idx, (l,m) in enumerate(self._sph_harm_lm):
                            tmp2[:,:,:] = tmp1[p,:,:,:] * spec_sp.basis_eval_radial(incident_mg[0], k, l) * sph_pre_in[lm_idx]
                            cc_collision[p * num_sh + qs_idx , k * num_sh + lm_idx] = np.dot(np.dot(np.dot(tmp2, WVPhi_q),glw),gmw)

        elif(g._type == collisions.CollisionType.EAR_G2):
            for qs_idx, (q,s) in enumerate(self._sph_harm_lm):
                tmp0[:,:,:,:,:,:]  = diff_cs * Mp_r * (2 * spec_sp.Vq_r(Sd[0],q) * sph_post[qs_idx] - spec_sp.Vq_r(scattering_mg[0],q) * sph_pre[qs_idx])
                tmp1[:,:,:,:]      = np.dot(np.dot(tmp0,WPhi_q),glw_s)

                for p in range(num_p):
                    for k in range(num_p):
                        for lm_idx, (l,m) in enumerate(self._sph_harm_lm):
                            tmp2[:,:,:] = tmp1[p,:,:,:] * spec_sp.basis_eval_radial(incident_mg[0], k, l) * sph_pre_in[lm_idx]
                            cc_collision[p * num_sh + qs_idx , k * num_sh + lm_idx] = np.dot(np.dot(np.dot(tmp2, WVPhi_q),glw),gmw)


        
        return cc_collision
    
    def assemble_mat(self,collision : collisions.Collisions , maxwellian, vth,v0=np.zeros(3)):
        #Lij = self._LOp_reduced_memory(collision,maxwellian,vth,v0) 
        Lij = self._LOp_Eulerian(collision,maxwellian,vth)
        return Lij
        
        

