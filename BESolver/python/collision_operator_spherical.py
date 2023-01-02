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
#import quadpy
import utils as BEUtils
import scipy.integrate
import scipy.sparse

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
    
    def __init__(self, spec_sp) -> None:
        
        self._spec              = spec_sp
        self._r_basis_type      = spec_sp.get_radial_basis_type()
        self._NUM_Q_VR          = params.BEVelocitySpace.NUM_Q_VR
        self._NUM_Q_VT          = params.BEVelocitySpace.NUM_Q_VT
        self._NUM_Q_VP          = params.BEVelocitySpace.NUM_Q_VP
        self._NUM_Q_CHI         = params.BEVelocitySpace.NUM_Q_CHI
        self._NUM_Q_PHI         = params.BEVelocitySpace.NUM_Q_PHI
        self._sph_harm_lm       = params.BEVelocitySpace.SPH_HARM_LM 
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

        assert self._NUM_Q_VP  > 1 
        assert self._NUM_Q_PHI > 1
        
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

    def setup_coulombic_collisions(self):
        """
        Setup the moments for columbic collisions based on Fokker-Plank approximation. 
        """

        spec_sp       = self._spec
        num_p         = spec_sp._p+1
        num_sh        = len(spec_sp._sph_harm_lm)

        gmx, gmw = spec_sp._basis_p.Gauss_Pn(self._NUM_Q_VR)

        k_vec    = spec_sp._basis_p._t
        dg_idx   = spec_sp._basis_p._dg_idx
        sp_order = spec_sp._basis_p._sp_order

        # compute the moment vectors
        def Pm(m):
            q_order       = ((sp_order + m + 1)//2) + 1
            qq_re         = np.polynomial.legendre.leggauss(q_order) #quadpy.c1.gauss_legendre(q_order)
            qq_re_points  = qq_re[0]
            qq_re_weights = qq_re[1]
            
            pm       = np.zeros((num_p, len(gmx)))
            for i in range(num_p):
                a_idx = np.where(k_vec[i] <= gmx)[0]
                b_idx = np.where(gmx <= k_vec[i + sp_order + 1])[0]
                idx   = np.sort(np.intersect1d(a_idx, b_idx))
                for j in idx:
                    xb      = k_vec[i]
                    xe      = gmx[j]
                    qx      = 0.5 * ( (xe - xb) * qq_re_points +  (xe + xb))
                    qw      = 0.5 * (xe - xb) * qq_re_weights
                    pm[i,j] =  np.dot(qw, (qx**m) * spec_sp.basis_eval_radial(qx, i , 0))

                if idx[-1] < (len(gmx) -1):
                    assert gmx[ idx[-1] + 1 ] > k_vec[i + sp_order + 1] , "moment computation fail Pm"
                    xb  = k_vec[i]
                    xe  = k_vec[i + sp_order + 1]
                    qx  = 0.5 * ( (xe - xb) * qq_re_points +  (xe + xb))
                    qw  = 0.5 * (xe - xb) * qq_re_weights
                    pm[i, idx[-1] + 1:] = np.dot(qw, (qx**m) * spec_sp.basis_eval_radial(qx, i , 0))

                    # scipy_int = scipy.integrate.quadrature(lambda qx : (qx**m) * spec_sp.basis_eval_radial(qx, i , 0), xb, xe, maxiter=100, tol=1e-16, rtol=1e-16)
                    # print("pm spline_id", i, 'm', m , 2*q_order-1,  "xe", xe, "quad : ", pm[i,j], "scipy int", scipy_int, 'rtol : ', abs(pm[i,j]-scipy_int[0])/scipy_int[0])
            
            return pm
        
        def Qm(m):
            q_order       = ((sp_order + m + 1)//2) + 1
            qq_re         = np.polynomial.legendre.leggauss(q_order) #quadpy.c1.gauss_legendre(q_order)
            qq_re_points  = qq_re[0]
            qq_re_weights = qq_re[1]
            
            qm       = np.zeros((num_p, len(gmx)))
            for i in range(num_p):
                a_idx = np.where(k_vec[i] <= gmx)[0]
                b_idx = np.where(gmx <= k_vec[i + sp_order + 1])[0]
                idx   = np.sort(np.intersect1d(a_idx, b_idx))
                for j in idx:
                    xb  = gmx[j]
                    xe  = k_vec[i + sp_order+1]
                    qx  = 0.5 * ( (xe - xb) * qq_re_points +  (xe + xb))
                    qw  = 0.5 * (xe - xb) * qq_re_weights
                    qm[i,j] =  np.dot(qw, (qx**m) * spec_sp.basis_eval_radial(qx, i , 0))

                if idx[0] > 0:
                    assert gmx[idx[0] - 1 ] < k_vec[i] , "moment computation fail Qm"
                    xb  = k_vec[i]
                    xe  = k_vec[i + sp_order+1]
                    qx  = 0.5 * ((xe - xb) * qq_re_points +  (xe + xb))
                    qw  = 0.5 * (xe - xb) * qq_re_weights
                    qm[i,0:idx[0]] =  np.dot(qw, (qx**m) * spec_sp.basis_eval_radial(qx, i , 0))

                    # scipy_int = scipy.integrate.quadrature(lambda qx : (qx**m) * spec_sp.basis_eval_radial(qx, i , 0), xb, xe, maxiter=100, tol=1e-16, rtol=1e-16)
                    # print("qm spline_id", i, 'm', m , "xe", xe, "quad : ", qm[i,j], "scipy int", scipy_int, 'rtol : ', abs(qm[i,j]-scipy_int[0])/scipy_int[0])
            
            return qm

        self._p2 = 4 * np.pi * Pm(2)
        self._p4 = 4 * np.pi * Pm(4)
        self._q1 = 4 * np.pi * Qm(1)

        self._q0 = 4 * np.pi * Qm(0)
        self._p3 = 4 * np.pi * Pm(3)
        self._p5 = 4 * np.pi * Pm(5)

        self._mass_op = BEUtils.mass_op(spec_sp, None, 64, 2, 1)
        self._temp_op = BEUtils.temp_op(spec_sp, None, 64, 2, 1)

        return

    def _LOp_eulerian(self, collision, maxwellian, vth):

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
                            tmp0[p,:,:,:,:,:]  = diff_cs * Mp_r * (2*spec_sp.basis_eval_radial(Sd[0], p, q) * sph_post[qs_idx] - spec_sp.basis_eval_radial(self._scattering_mg[0],p,q) * sph_pre[qs_idx])
                            tmp1[p,:,:,:]      = np.dot(np.dot(tmp0[p],self._WPhi_q),self._glw_s)

                        for p in range(ib,ie+1):
                            for k in range(ib,ie+1):
                                for lm_idx, (l,m) in enumerate(self._sph_harm_lm):
                                    tmp2[:,:,:] = tmp1[p,:,:,:] * spec_sp.basis_eval_radial(self._incident_mg[0], k, l) * sph_pre_in[lm_idx]
                                    cc_collision[p * num_sh + qs_idx , k * num_sh + lm_idx] += np.dot(np.dot(np.dot(tmp2, self._WVPhi_q),self._glw), gw_e)
            
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

    def _LOp_eulerian_radial_only(self, collision, maxwellian, vth):
        V_TH          = vth     
        ELE_VOLT      = collisions.ELECTRON_VOLT
        MAXWELLIAN_N  = collisions.MAXWELLIAN_N
        AR_NEUTRAL_N  = collisions.AR_NEUTRAL_N
        
        g             = collision
        spec_sp       = self._spec
        num_p         = spec_sp._p+1
        num_sh        = len(spec_sp._sph_harm_lm)
        if self._r_basis_type == basis.BasisType.CHEBYSHEV_POLY:
            raise NotImplementedError
        elif self._r_basis_type == basis.BasisType.SPLINES:

            k_vec    = spec_sp._basis_p._t
            dg_idx   = spec_sp._basis_p._dg_idx
            sp_order = spec_sp._basis_p._sp_order
            
            c_gamma      = np.sqrt(2*collisions.ELECTRON_CHARGE_MASS_RATIO)
            l_modes      = list(set([l for (l,m) in self._sph_harm_lm]))
            
            gx_e , gw_e  = spec_sp._basis_p.Gauss_Pn(self._NUM_Q_VR)
            Mp_r         = (gx_e * V_TH) * ( gx_e **2 )
            diff_cs      = collisions.Collisions.synthetic_tcs((gx_e * V_TH / c_gamma)**2, g._analytic_cross_section_type)
            cc_collision = spec_sp.create_mat()

            if(g._type == collisions.CollisionType.EAR_G0 or g._type == collisions.CollisionType.EAR_G1):
                gain_fac = 1.0
                c_mu     = 2 * collisions.MASS_R_EARGON 
                v_scale  = np.sqrt(1- c_mu)
                v_post   = gx_e * v_scale

                kappa    = (scipy.constants.Boltzmann * collisions.AR_TEMP_K * c_mu * 0.5 / scipy.constants.electron_mass) / V_TH

                for qs_idx, (q,s) in enumerate(self._sph_harm_lm):
                    tmp = np.zeros((num_p,num_p,len(gx_e)))
                    if q==0:
                        for e_id in range(0,len(dg_idx),2):
                            ib=dg_idx[e_id]
                            ie=dg_idx[e_id+1] 
                            for p in range(ib,ie+1):
                                #-0.5 * c_mu * gx_e * spec_sp.basis_derivative_eval_radial(gx_e, p, 0, 1)
                                #psi_p  = (-0.5 * c_mu * gx_e) * spec_sp.basis_derivative_eval_radial(gx_e, p, 0, 1) + (-0.5 * c_mu * gx_e)**2 * spec_sp.basis_derivative_eval_radial(gx_e, p, 0, 2) 
                                psi_p = (gain_fac * spec_sp.basis_eval_radial(v_post ,p,q) - spec_sp.basis_eval_radial(gx_e,p,q))
                                for k in range(num_p):
                                    tmp[p,k] = V_TH * gx_e**3 * diff_cs * spec_sp.basis_eval_radial(gx_e, k, q) * psi_p  - kappa * gx_e **3 * diff_cs * spec_sp.basis_derivative_eval_radial(gx_e, p, 0, 1) * spec_sp.basis_derivative_eval_radial(gx_e, k, 0, 1)
                                     
                    else:
                        for e_id in range(0,len(dg_idx),2):
                            ib=dg_idx[e_id]
                            ie=dg_idx[e_id+1] 
                            for p in range(ib,ie+1):
                                psi_p = spec_sp.basis_eval_radial(gx_e,p,q)
                                for k in range(num_p):
                                    tmp[p,k] = -V_TH * diff_cs * gx_e**3 * spec_sp.basis_eval_radial(gx_e, k, q) * psi_p
                    
                    tmp=tmp.reshape((num_p,num_p,-1))
                    tmp=np.dot(tmp, gw_e)

                    for p in range(num_p):
                        for k in range(num_p):
                            cc_collision[p * num_sh + qs_idx , k * num_sh + qs_idx] =tmp[p,k]

            elif(g._type == collisions.CollisionType.EAR_G2):
                gain_fac         = 2.0
                check_1          = (gx_e * V_TH/c_gamma)**2 > g._reaction_threshold
                v_scale          = np.zeros_like(gx_e)
                v_scale[check_1] = c_gamma * np.sqrt(0.5*((gx_e[check_1] * V_TH /c_gamma)**2  - g._reaction_threshold)) / V_TH
                v_post           = v_scale

                for qs_idx, (q,s) in enumerate(self._sph_harm_lm):
                    tmp = np.zeros((num_p,num_p,len(gx_e)))
                    if q==0:
                        for e_id in range(0,len(dg_idx),2):
                            ib=dg_idx[e_id]
                            ie=dg_idx[e_id+1] 
                            for p in range(ib,ie+1):
                                #-0.5 * c_mu * gx_e * spec_sp.basis_derivative_eval_radial(gx_e, p, 0, 1)
                                psi_p = (gain_fac * spec_sp.basis_eval_radial(v_post ,p,q) - spec_sp.basis_eval_radial(gx_e,p,q))
                                for k in range(num_p):
                                    tmp[p,k] = V_TH * gx_e**3 * diff_cs * spec_sp.basis_eval_radial(gx_e, k, q) * psi_p
                    else:
                        for e_id in range(0,len(dg_idx),2):
                            ib=dg_idx[e_id]
                            ie=dg_idx[e_id+1] 
                            for p in range(ib,ie+1):
                                psi_p = spec_sp.basis_eval_radial(gx_e,p,q)
                                for k in range(num_p):
                                    tmp[p,k] = -V_TH * diff_cs * gx_e**3 * spec_sp.basis_eval_radial(gx_e, k, q) * psi_p
                    
                    tmp=tmp.reshape((num_p,num_p,-1))
                    tmp=np.dot(tmp, gw_e)

                    for p in range(num_p):
                        for k in range(num_p):
                            cc_collision[p * num_sh + qs_idx , k * num_sh + qs_idx] =tmp[p,k]

            else:
                raise NotImplementedError
            
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
        Lij = self._LOp_eulerian_radial_only(collision,maxwellian,vth)
        #Lij = self._LOp_eulerian(collision,maxwellian,vth)
        return Lij

    def coulomb_collision_mat(self, alpha, ionization_degree, n0, fb, mw, vth):
        """
        compute the weak form of the coulomb collision operator based on fokker-plank equation
        with Rosenbluth's potentials

        Assumptions: 
            - Currently for l=0, l=1 modes only, others assumed to be zero
            - assumes azimuthal symmetry
        """

        if ionization_degree == 0:
            return 0.0, 0.0

        V_TH          = vth
        ELE_VOLT      = collisions.ELECTRON_VOLT
        MAXWELLIAN_N  = collisions.MAXWELLIAN_N
        AR_NEUTRAL_N  = collisions.AR_NEUTRAL_N
        
        spec_sp  :sp.SpectralExpansionSpherical     = self._spec
        num_p         = spec_sp._p+1
        num_sh        = len(spec_sp._sph_harm_lm)

        if self._r_basis_type != basis.BasisType.SPLINES:
            raise NotImplementedError
        
        cc_collision = spec_sp.create_mat()

        k_vec      = spec_sp._basis_p._t
        dg_idx     = spec_sp._basis_p._dg_idx
        sp_order   = spec_sp._basis_p._sp_order

        gmx , gmw  = spec_sp._basis_p.Gauss_Pn(self._NUM_Q_VR)
        
        p20        = np.dot(fb[0::num_sh], self._p2) 
        p40        = np.dot(fb[0::num_sh], self._p4) 
        q10        = np.dot(fb[0::num_sh], self._q1) 

        p31        = np.dot(fb[1::num_sh], self._p3) 
        p51        = np.dot(fb[1::num_sh], self._p5) 
        q01        = np.dot(fb[1::num_sh], self._q0) 

        
        B          = spec_sp.basis_eval_radial
        DB         = spec_sp.basis_derivative_eval_radial

        # ne_fac   = ionization_degree * n0 
        m0         = np.dot(fb,self._mass_op) * vth**3 * mw(0)
        kT         = (np.dot(fb, self._temp_op) * vth**5 * mw(0) * 0.5 * scipy.constants.electron_mass * (2./ 3) / m0) 
        eps_0      = scipy.constants.epsilon_0
        me         = scipy.constants.electron_mass
        qe         = scipy.constants.e

        ne          = n0 * ionization_degree
        T_ev        = kT / qe

        b_min       = max(qe**2 / (2 * np.pi * eps_0 * 3 * kT), scipy.constants.Planck / (np.sqrt(me * 3 * kT)))
        b_max       = np.sqrt((eps_0 * kT) / (ne * qe**2))
        c_lambda    = b_max/b_min 
        gamma_a     = (np.log(c_lambda) * (qe**4)) / (4 * np.pi * (eps_0 * me)**2) / vth**3
        
        sph_l0      = lambda l : np.sqrt((2 * l +1) / (4 * np.pi) )
        print("Coulomb logarithm %.8E \t gamma_a %.8E \t gamma_a * ne %.8E  kT %.8E" %(np.log(c_lambda), gamma_a, n0 * ionization_degree * gamma_a, kT))

        for p in range(num_p):
            for k in range(max(0, p - 2 * (sp_order+2) ), min(num_p, p + 2 * (sp_order+2))):
                tmp = -(alpha * p20 * B(gmx,k,0) + (1/(3*gmx)) * (p40 + gmx**3 * q10) * DB(gmx, k, 0, 1)) * DB(gmx, p, 0, 1)
                cc_collision[p * num_sh + 0 , k * num_sh + 0] = np.dot(tmp, gmw) * (sph_l0(0))

                # tmp  = (1 + alpha) * gmx * q01 * B(gmx, k, 0) * B(gmx, p, 0)
                # tmp -= ((1./3) * B(gmx, k, 0) * ((2 * alpha -1) * p31 - (1+alpha) * gmx**3 * q01) + (DB(gmx, k, 0, 1) / (5*gmx)) * (p51 + gmx**5 * q01 )) * (gmx * DB(gmx, p, 0, 1) - B(gmx, p, 0)) / gmx**2

                # cc_collision[p * num_sh + 1 , k * num_sh + 0] = np.dot(tmp, gmw) * (sph_l0(0)**2 / sph_l0(1))

                # tmp  =  -((1 + alpha)/gmx) * p20 * B(gmx, k , 0) * B(gmx, p, 0) + (alpha * gmx * p20 * B(gmx, k, 0) + (1./3) * gmx * (p40 + gmx**3 * q10) * ((gmx * DB(gmx, k , 0, 1) - B(gmx, k, 0))/gmx**2) ) * (((gmx * DB(gmx, p , 0, 1) - B(gmx, p, 0))/gmx**2))

                # cc_collision[p * num_sh + 1 , k * num_sh + 1] = np.dot(tmp, gmw) * (sph_l0(1))

        
        return cc_collision * gamma_a

