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
    
    def __init__(self,dim,p_order,q_mode=sp.QuadMode.GMX,poly_type=basis.BasisType.MAXWELLIAN_POLY,k_domain=(0,10), sig_pts=None) -> None:
        self._dim  = dim
        self._p    = p_order
        self._r_basis_type = poly_type

        if self._r_basis_type == basis.BasisType.MAXWELLIAN_POLY:

            self._spec = sp.SpectralExpansionSpherical(self._p,basis.Maxwell(),params.BEVelocitySpace.SPH_HARM_LM)
            self._spec._q_mode = q_mode

        elif self._r_basis_type == basis.BasisType.LAGUERRE:

            self._spec = sp.SpectralExpansionSpherical(self._p,basis.Laguerre(),params.BEVelocitySpace.SPH_HARM_LM)
            self._spec._q_mode = q_mode
        
        elif self._r_basis_type == basis.BasisType.SPLINES:
            spline_order = basis.BSPLINE_BASIS_ORDER
            splines      = basis.XlBSpline(k_domain,spline_order,self._p+1, sig_pts=sig_pts)
            self._spec   = sp.SpectralExpansionSpherical(self._p,splines,params.BEVelocitySpace.SPH_HARM_LM)
            self._spec._q_mode = q_mode

            # import matplotlib.pyplot as plt
            # gx,gw= basis.uniform_simpson((knots_vec[0],knots_vec[-1]),1001)
            # for i in range(self._p +1):
            #     plt.plot(gx,splines.Pn(i)(gx),label="i=%d"%i)
            
            # plt.grid()
            # plt.show()
            #exit(0)

        
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

        if self._r_basis_type == basis.BasisType.MAXWELLIAN_POLY or self._r_basis_type == basis.BasisType.LAGUERRE:
            [self._gmx,self._gmw]  = spec_sp._basis_p.Gauss_Pn(self._NUM_Q_VR)
        elif self._r_basis_type == basis.BasisType.SPLINES:
            [self._gmx,self._gmw]  = spec_sp._basis_p.Gauss_Pn(self._NUM_Q_VR,True)

        
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
        self._incident_mg    = np.meshgrid(self._gmx,self._VTheta_q,self._VPhi_q,indexing='ij')
        self._scattering_mg  = np.meshgrid(self._gmx,self._VTheta_q,self._VPhi_q,Chi_q,Phi_q,indexing='ij')

        return
        ## remove return for storing the quadrature matrices - high mem. footprint, but less redundant computations, when reassembling the collision matrix. 
        self._l_modes  = list(set([l for l,_ in self._sph_harm_lm]))
        self._radial_poly_l_pre       = list() # radial poly evaluated on the incident grid. 
        self._radial_poly_l_pre_on_sg = list() # radial poly evaluated on the scattering grid.          
        for l in self._l_modes:
            self._radial_poly_l_pre.append(self._spec.Vq_r(self._incident_mg[0],l))
            self._radial_poly_l_pre_on_sg.append(self._spec.Vq_r(self._scattering_mg[0],l))


        self._sph_pre       = self._spec.Vq_sph(self._incident_mg[1],self._incident_mg[2])
        self._sph_pre_on_sg = self._spec.Vq_sph(self._scattering_mg[1],self._scattering_mg[2])

        if self._r_basis_type == basis.BasisType.MAXWELLIAN_POLY or self._r_basis_type == basis.BasisType.LAGUERRE:
            self._full_basis_pre = np.array([(self._incident_mg[0]**(l)) * self._radial_poly_l_pre[self._l_modes.index(l)] *self._sph_pre[lm_idx] for lm_idx, (l,m) in enumerate(self._sph_harm_lm)])
        elif self._r_basis_type == basis.BasisType.SPLINES:
            self._full_basis_pre = np.array([self._radial_poly_l_pre[self._l_modes.index(l)] *self._sph_pre[lm_idx] for lm_idx, (l,m) in enumerate(self._sph_harm_lm)])
        
        # for each lm mode we have num_sh, num_p, mesh grid dimensions
        self._full_basis_pre=self._full_basis_pre.reshape(tuple([len(self._sph_harm_lm),self._num_p]) + self._incident_mg[0].shape)
        #print(self._full_basis_pre.shape)
        

        
    def _LOp(self, collision, maxwellian, vth, v0):
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
        
        if(g._type == collisions.CollisionType.EAR_G0 or g._type == collisions.CollisionType.EAR_G1):
            Sd      = g.post_scattering_velocity_sp(scattering_mg_v0[0]*V_TH,scattering_mg_v0[1],scattering_mg_v0[2],scattering_mg[3],scattering_mg[4])
            Sd      = BEUtils.spherical_to_cartesian(Sd[0]/V_TH,Sd[1],Sd[2])

            Sd[0]   = Sd[0] - np.ones_like(Sd[0]) * v0[0]
            Sd[1]   = Sd[1] - np.ones_like(Sd[0]) * v0[1]
            Sd[2]   = Sd[2] - np.ones_like(Sd[0]) * v0[2]

            Sd      = BEUtils.cartesian_to_spherical(Sd[0],Sd[1],Sd[2])

            # radial poly eval at post collision points. 
            radial_poly_l_post=list()
            for l in self._l_modes:
                radial_poly_l_post.append(self._spec.Vq_r(Sd[0],l))

            #print("sc mag : ", np.allclose(scattering_mg[0], Sd[0], rtol=np.finfo(float).eps/2 ,  atol=1e-14))
            # sph eval at post collission points. 
            sph_post=self._spec.Vq_sph(Sd[1],Sd[2])
            
            if self._r_basis_type == basis.BasisType.MAXWELLIAN_POLY or self._r_basis_type == basis.BasisType.LAGUERRE:
                Mp_r    = (scattering_mg_v0[0])*V_TH
            elif self._r_basis_type == basis.BasisType.SPLINES:
                Mp_r    = (scattering_mg_v0[0] * V_TH) * (scattering_mg[0]**2) 
            
            cc_collision = np.array([diff_cs * Mp_r * (radial_poly_l_post[self._l_modes.index(l)] * sph_post[lm_idx] - self._radial_poly_l_pre_on_sg[self._l_modes.index(l)] *self._sph_pre_on_sg[lm_idx]) for lm_idx, (l,m) in enumerate(self._sph_harm_lm)])
            cc_collision = cc_collision.reshape(tuple([num_sh,num_p]) + scattering_mg[0].shape)
            
            cc_collision = np.dot(cc_collision,WPhi_q)
            cc_collision = np.dot(cc_collision,glw_s)

            cc_collision =np.array([cc_collision[qs_idx,p] * self._full_basis_pre[lm_idx,k]  for p in range(num_p) for qs_idx, (q,s) in enumerate(self._sph_harm_lm) for k in range(num_p) for lm_idx, (l,m) in enumerate(self._sph_harm_lm)])
            cc_collision = cc_collision.reshape(tuple([num_p,num_sh,num_p,num_sh]) + incident_mg[0].shape)
            
            cc_collision = np.dot(cc_collision,WVPhi_q)
            cc_collision = np.dot(cc_collision,glw)
            cc_collision = np.dot(cc_collision,gmw)
            cc_collision = cc_collision.reshape((num_p*num_sh,num_p*num_sh))
            
            return cc_collision
            
        elif(g._type == collisions.CollisionType.EAR_G2):
            Sd_particles = g.post_scattering_velocity_sp(scattering_mg_v0[0]*V_TH,scattering_mg_v0[1],scattering_mg_v0[2],scattering_mg[3],scattering_mg[4])         
            Sd1          = Sd_particles[0]
            Sd2          = Sd_particles[1]

            Sd1          = BEUtils.spherical_to_cartesian(Sd1[0]/V_TH,Sd1[1],Sd1[2])
            Sd2          = BEUtils.spherical_to_cartesian(Sd2[0]/V_TH,Sd2[1],Sd2[2])

            Sd1[0]       = Sd1[0] - np.ones_like(Sd1[0]) * v0[0]
            Sd1[1]       = Sd1[1] - np.ones_like(Sd1[0]) * v0[1]
            Sd1[2]       = Sd1[2] - np.ones_like(Sd1[0]) * v0[2]

            Sd2[0]       = Sd2[0] - np.ones_like(Sd2[0]) * v0[0]
            Sd2[1]       = Sd2[1] - np.ones_like(Sd2[0]) * v0[1]
            Sd2[2]       = Sd2[2] - np.ones_like(Sd2[0]) * v0[2]

            Sd1          = BEUtils.cartesian_to_spherical(Sd1[0],Sd1[1],Sd1[2])
            Sd2          = BEUtils.cartesian_to_spherical(Sd2[0],Sd2[1],Sd2[2])
            
            radial_poly_l_post1 = list()
            radial_poly_l_post2 = list()

            for l in self._l_modes:
                radial_poly_l_post1.append(self._spec.Vq_r(Sd1[0],l))
                radial_poly_l_post2.append(self._spec.Vq_r(Sd2[0],l))

            Yp1_lm   = spec_sp.Vq_sph(Sd1[1],Sd1[2])
            Yp2_lm   = spec_sp.Vq_sph(Sd2[1],Sd2[2])

            if self._r_basis_type == basis.BasisType.MAXWELLIAN_POLY or self._r_basis_type == basis.BasisType.LAGUERRE:
                Mp_r         = (scattering_mg_v0[0])*V_TH
            
            elif self._r_basis_type == basis.BasisType.SPLINES:
                Mp_r    = (scattering_mg_v0[0] * V_TH) * (scattering_mg[0]**2)

            cc_collision = np.array([diff_cs * Mp_r * (radial_poly_l_post2[self._l_modes.index(l)] * Yp2_lm[lm_idx] + radial_poly_l_post1[self._l_modes.index(l)] * Yp1_lm[lm_idx] - self._radial_poly_l_pre_on_sg[self._l_modes.index(l)] * self._sph_pre_on_sg[lm_idx]) for lm_idx, (l,m) in enumerate(self._sph_harm_lm)])
            cc_collision = cc_collision.reshape(tuple([num_sh,num_p]) + scattering_mg[0].shape)
            
            cc_collision = np.dot(cc_collision,WPhi_q)
            cc_collision = np.dot(cc_collision,glw_s)

            cc_collision =np.array([cc_collision[qs_idx,p] * self._full_basis_pre[lm_idx,k]  for p in range(num_p) for qs_idx, (q,s) in enumerate(self._sph_harm_lm) for k in range(num_p) for lm_idx, (l,m) in enumerate(self._sph_harm_lm)])
            cc_collision = cc_collision.reshape(tuple([num_p,num_sh,num_p,num_sh]) + incident_mg[0].shape)
            
            cc_collision = np.dot(cc_collision,WVPhi_q)
            cc_collision = np.dot(cc_collision,glw)
            cc_collision = np.dot(cc_collision,gmw)
            cc_collision = cc_collision.reshape((num_p*num_sh,num_p*num_sh))
            
            # print("collision = %.8E"%np.linalg.cond(cc_collision))
            # print(cc_collision)
            #print(cc_collision.shape)
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
        
        if(g._type == collisions.CollisionType.EAR_G0 or g._type == collisions.CollisionType.EAR_G1):
            Sd      = g.post_scattering_velocity_sp(scattering_mg_v0[0]*V_TH,scattering_mg_v0[1],scattering_mg_v0[2],scattering_mg[3],scattering_mg[4])
            Sd      = BEUtils.spherical_to_cartesian(Sd[0]/V_TH,Sd[1],Sd[2])

            Sd[0]   = Sd[0] - np.ones_like(Sd[0]) * v0[0]
            Sd[1]   = Sd[1] - np.ones_like(Sd[0]) * v0[1]
            Sd[2]   = Sd[2] - np.ones_like(Sd[0]) * v0[2]

            Sd      = BEUtils.cartesian_to_spherical(Sd[0],Sd[1],Sd[2])

            sph_post   = spec_sp.Vq_sph(Sd[1],Sd[2])
            sph_pre    = spec_sp.Vq_sph(scattering_mg[1],scattering_mg[2])
            sph_pre_in = spec_sp.Vq_sph(incident_mg[1],incident_mg[2])
            
            if self._r_basis_type == basis.BasisType.MAXWELLIAN_POLY or self._r_basis_type == basis.BasisType.LAGUERRE:
                Mp_r    = (scattering_mg_v0[0])*V_TH

            elif self._r_basis_type == basis.BasisType.SPLINES:
                Mp_r    = (scattering_mg_v0[0] * V_TH) * (scattering_mg[0]**2) 
            
            cc_collision = np.array([diff_cs * Mp_r * ( spec_sp.Vq_r(Sd[0],l) * sph_post[lm_idx] - spec_sp.Vq_r(scattering_mg[0],l) * sph_pre[lm_idx]) for lm_idx, (l,m) in enumerate(self._sph_harm_lm)])
            cc_collision = cc_collision.reshape(tuple([num_sh,num_p]) + scattering_mg[0].shape)
            
            cc_collision = np.dot(cc_collision,WPhi_q)
            cc_collision = np.dot(cc_collision,glw_s)

            cc_collision =np.array([cc_collision[qs_idx,p] * spec_sp.basis_eval_radial(incident_mg[0], k, l) * sph_pre_in[lm_idx]  for p in range(num_p) for qs_idx, (q,s) in enumerate(self._sph_harm_lm) for k in range(num_p) for lm_idx, (l,m) in enumerate(self._sph_harm_lm)])

            cc_collision = cc_collision.reshape(tuple([num_p,num_sh,num_p,num_sh]) + incident_mg[0].shape)
            cc_collision = np.dot(cc_collision,WVPhi_q)
            cc_collision = np.dot(cc_collision,glw)
            cc_collision = np.dot(cc_collision,gmw)
            cc_collision = cc_collision.reshape((num_p*num_sh,num_p*num_sh))
            
            return cc_collision
            
        elif(g._type == collisions.CollisionType.EAR_G2):
            Sd_particles = g.post_scattering_velocity_sp(scattering_mg_v0[0]*V_TH,scattering_mg_v0[1],scattering_mg_v0[2],scattering_mg[3],scattering_mg[4])         
            Sd1          = Sd_particles[0]
            Sd2          = Sd_particles[1]

            Sd1          = BEUtils.spherical_to_cartesian(Sd1[0]/V_TH,Sd1[1],Sd1[2])
            Sd2          = BEUtils.spherical_to_cartesian(Sd2[0]/V_TH,Sd2[1],Sd2[2])

            Sd1[0]       = Sd1[0] - np.ones_like(Sd1[0]) * v0[0]
            Sd1[1]       = Sd1[1] - np.ones_like(Sd1[0]) * v0[1]
            Sd1[2]       = Sd1[2] - np.ones_like(Sd1[0]) * v0[2]

            Sd2[0]       = Sd2[0] - np.ones_like(Sd2[0]) * v0[0]
            Sd2[1]       = Sd2[1] - np.ones_like(Sd2[0]) * v0[1]
            Sd2[2]       = Sd2[2] - np.ones_like(Sd2[0]) * v0[2]

            Sd1          = BEUtils.cartesian_to_spherical(Sd1[0],Sd1[1],Sd1[2])
            Sd2          = BEUtils.cartesian_to_spherical(Sd2[0],Sd2[1],Sd2[2])
            
            Yp1_lm   = spec_sp.Vq_sph(Sd1[1],Sd1[2])
            Yp2_lm   = spec_sp.Vq_sph(Sd2[1],Sd2[2])
            sph_pre  = spec_sp.Vq_sph(scattering_mg[1],scattering_mg[2])
            sph_pre_in = spec_sp.Vq_sph(incident_mg[1],incident_mg[2])

            if self._r_basis_type == basis.BasisType.MAXWELLIAN_POLY or self._r_basis_type == basis.BasisType.LAGUERRE:
                Mp_r         = (scattering_mg_v0[0]) * V_TH
                
            elif self._r_basis_type == basis.BasisType.SPLINES:
                Mp_r         = (scattering_mg_v0[0] * V_TH) * (scattering_mg[0]**2) 

            cc_collision = np.array([diff_cs * Mp_r * (spec_sp.Vq_r(Sd2[0], l) * Yp2_lm[lm_idx] + spec_sp.Vq_r(Sd1[0], l) * Yp1_lm[lm_idx] - spec_sp.Vq_r(scattering_mg[0], l) * sph_pre[lm_idx]) for lm_idx, (l,m) in enumerate(self._sph_harm_lm)])
            cc_collision = cc_collision.reshape(tuple([num_sh,num_p]) + scattering_mg[0].shape)

            cc_collision = np.dot(cc_collision,WPhi_q)
            cc_collision = np.dot(cc_collision,glw_s)

            cc_collision =np.array([cc_collision[qs_idx,p] * spec_sp.basis_eval_radial(incident_mg[0], k, l) * sph_pre_in[lm_idx]  for p in range(num_p) for qs_idx, (q,s) in enumerate(self._sph_harm_lm) for k in range(num_p) for lm_idx, (l,m) in enumerate(self._sph_harm_lm)])

            cc_collision = cc_collision.reshape(tuple([num_p,num_sh,num_p,num_sh]) + incident_mg[0].shape)
            cc_collision = np.dot(cc_collision,WVPhi_q)
            cc_collision = np.dot(cc_collision,glw)
            cc_collision = np.dot(cc_collision,gmw)
            cc_collision = cc_collision.reshape((num_p*num_sh,num_p*num_sh))
            
            #print("Mr shape : ", Mp_r.shape)
            
            #print(cc_collision.shape)
            # print("collision = %.8E"%np.linalg.cond(cc_collision))
            # print(cc_collision)
            #print(cc_collision.shape)
            return cc_collision
    

    def assemble_mat(self,collision : collisions.Collisions , maxwellian, vth,v0=np.zeros(3)):
        #Lij  = self._LOp(collision,maxwellian,vth,v0) 
        Lij = self._LOp_reduced_memory(collision,maxwellian,vth,v0) 
        # print(Lij-Lij1)
        # print(np.linalg.norm(Lij-Lij1)/np.linalg.norm(Lij))
        return Lij
        
        

