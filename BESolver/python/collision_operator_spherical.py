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
import sym_cc
import sympy

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
            q_order       *= 64
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
                    qx      = 0.5 * ((xe - xb) * qq_re_points +  (xe + xb))
                    qw      = 0.5 * (xe - xb) * qq_re_weights
                    pm[i,j] = np.dot(qw, (qx**m) * spec_sp.basis_eval_radial(qx, i , 0))

                # if idx[-1] < (len(gmx) -1):
                #     assert gmx[ idx[-1] + 1 ] > k_vec[i + sp_order + 1] , "moment computation fail Pm"
                #     xb  = k_vec[i]
                #     xe  = k_vec[i + sp_order + 1]
                #     qx  = 0.5 * ((xe - xb) * qq_re_points +  (xe + xb))
                #     qw  = 0.5 * (xe - xb) * qq_re_weights
                #     pm[i, idx[-1] + 1:] = np.dot(qw, (qx**m) * spec_sp.basis_eval_radial(qx, i , 0))

                a_idx = np.where(k_vec[i + sp_order + 1] < gmx)[0]
                xb  = k_vec[i]
                xe  = k_vec[i + sp_order + 1]
                qx  = 0.5 * ((xe - xb) * qq_re_points +  (xe + xb))
                qw  = 0.5 * (xe - xb) * qq_re_weights
                pm[i, a_idx] = np.dot(qw, (qx**m) * spec_sp.basis_eval_radial(qx, i , 0))




                    # scipy_int = scipy.integrate.quadrature(lambda qx : (qx**m) * spec_sp.basis_eval_radial(qx, i , 0), xb, xe, maxiter=100, tol=1e-16, rtol=1e-16)
                    # print("pm spline_id", i, 'm', m , 2*q_order-1,  "xe", xe, "quad : ", pm[i,j], "scipy int", scipy_int, 'rtol : ', abs(pm[i,j]-scipy_int[0])/scipy_int[0])
            
            return pm
        
        def Qm(m):
            q_order       = ((sp_order + m + 1)//2) + 1
            q_order       *= 64
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
                    qx  = 0.5 * ((xe - xb) * qq_re_points +  (xe + xb))
                    qw  = 0.5 * (xe - xb) * qq_re_weights
                    qm[i,j] =  np.dot(qw, (qx**m) * spec_sp.basis_eval_radial(qx, i , 0))

                # if idx[0] > 0:
                #     assert gmx[idx[0] - 1 ] < k_vec[i] , "moment computation fail Qm"
                #     xb  = k_vec[i]
                #     xe  = k_vec[i + sp_order+1]
                #     qx  = 0.5 * ((xe - xb) * qq_re_points +  (xe + xb))
                #     qw  = 0.5 * (xe - xb) * qq_re_weights
                #     qm[i,0:idx[0]] =  np.dot(qw, (qx**m) * spec_sp.basis_eval_radial(qx, i , 0))

                a_idx = np.where(gmx < k_vec[i])[0]
                xb  = k_vec[i]
                xe  = k_vec[i + sp_order + 1]
                qx  = 0.5 * ((xe - xb) * qq_re_points +  (xe + xb))
                qw  = 0.5 * (xe - xb) * qq_re_weights
                qm[i, a_idx] = np.dot(qw, (qx**m) * spec_sp.basis_eval_radial(qx, i , 0))

                    # scipy_int = scipy.integrate.quadrature(lambda qx : (qx**m) * spec_sp.basis_eval_radial(qx, i , 0), xb, xe, maxiter=100, tol=1e-16, rtol=1e-16)
                    # print("qm spline_id", i, 'm', m , "xe", xe, "quad : ", qm[i,j], "scipy int", scipy_int, 'rtol : ', abs(qm[i,j]-scipy_int[0])/scipy_int[0])
            
            return qm

        # the factor 2 comes form 4pi/ 2pi from spherical harmonics projection.

        self._p2 = Pm(2)      
        self._p4 = Pm(4)      
        self._q1 = Qm(1)

        self._q0 = Qm(0)
        self._q3 = Qm(3)
        self._p3 = Pm(3)
        self._p5 = Pm(5)

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

    def coulomb_collision_mat(self, alpha, ionization_degree, n0, fb, mw, vth, sigma_m, full_assembly=True):
        """
        compute the weak form of the coulomb collision operator based on fokker-plank equation
        with Rosenbluth's potentials

        Assumptions: 
            - Currently for l=0, l=1 modes only, others assumed to be zero
            - assumes azimuthal symmetry
        """

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

        gmx_a , gmw_a  = spec_sp._basis_p.Gauss_Pn(self._NUM_Q_VR)
        
        sph_l0     = lambda l : np.sqrt((2 * l +1) / (4 * np.pi) )

        B          = spec_sp.basis_eval_radial
        DB         = spec_sp.basis_derivative_eval_radial


        p20_a      = 2 * sph_l0(0) * np.dot(fb[0::num_sh], self._p2)
        p40_a      = 2 * sph_l0(0) * np.dot(fb[0::num_sh], self._p4) 
        q10_a      = 2 * sph_l0(0) * np.dot(fb[0::num_sh], self._q1) 

        p31_a      = 2 * sph_l0(1) * np.dot(fb[1::num_sh], self._p3)
        p51_a      = 2 * sph_l0(1) * np.dot(fb[1::num_sh], self._p5)
        q01_a      = 2 * sph_l0(1) * np.dot(fb[1::num_sh], self._q0)

        for p in range(num_p):
            for k in range(max(0, p - (sp_order+3) ), min(num_p, p + (sp_order+3))):
                
                k_min  = min(k_vec[p], k_vec[k])
                k_max  = max(k_vec[p + sp_order + 1], k_vec[k + sp_order + 1])

                qx_idx = np.logical_and(gmx_a >= k_min, gmx_a <= k_max)
                gmx    = gmx_a[qx_idx]
                gmw    = gmw_a[qx_idx]

                p20    = p20_a[qx_idx] 
                p40    = p40_a[qx_idx] 
                q10    = q10_a[qx_idx] 

                p31    = p31_a[qx_idx] 
                p51    = p51_a[qx_idx] 
                q01    = q01_a[qx_idx] 

                tmp = -(alpha * p20 * B(gmx,k,0) + (1/(3*gmx)) * (p40 + gmx**3 * q10) * DB(gmx, k, 0, 1)) * DB(gmx, p, 0, 1)
                cc_collision[p * num_sh + 0 , k * num_sh + 0] = np.dot(tmp, gmw) 
                
                if full_assembly:
                    tmp  = (1 + alpha) * gmx * q01 * B(gmx, k, 0) * B(gmx, p, 0) - (
                        (1/3) * B(gmx, k, 0) * ((2 * alpha -1) * p31 - (1+alpha) * gmx**3 * q01) 
                        + (DB(gmx, k, 0, 1) / (5 * gmx)) * (p51 + gmx**5 * q01)
                        ) * (gmx * DB(gmx, p, 0, 1) - B(gmx, p, 0)) / gmx**2
                    
                    cc_collision[p * num_sh + 1 , k * num_sh + 0] = (sph_l0(0)/ sph_l0(1)) * np.dot(tmp, gmw) 

                    tmp  =  -((1 + alpha)/gmx) * p20 * B(gmx, k , 0) * B(gmx, p, 0) - \
                    ( alpha * gmx * p20 * B(gmx, k, 0) 
                    + (1./3) * gmx * (p40 + gmx**3 * q10) * ((gmx * DB(gmx, k , 0, 1) - B(gmx, k, 0))/gmx**2) ) * (((gmx * DB(gmx, p , 0, 1) - B(gmx, p, 0))/gmx**2))

                    cc_collision[p * num_sh + 1 , k * num_sh + 1] = np.dot(tmp, gmw)


        ne           = n0 * ionization_degree
        eps_0        = scipy.constants.epsilon_0
        me           = scipy.constants.electron_mass
        qe           = scipy.constants.e
        m0           = mw(0) * np.dot(fb,self._mass_op) * vth**3 
        kT           = mw(0) * (np.dot(fb, self._temp_op) * vth**5 * 0.5 * scipy.constants.electron_mass * (2./ 3) / m0) 

        kT           = np.abs(kT) 
        
        # kp_op      = vth**5 * mw(0) * np.sqrt(4*np.pi) * (2/(3*(2 / me))) * np.array([np.dot(gmw_a, gmx_a**4 * B(gmx_a,k,0)) for k in range(num_p)])
        # kT         = np.dot(kp_op, fb[0::num_sh])        
        # print(kT)

        #kT         = ((vth**5 * p40_a[-1] * qe) / (3 * (2/me))) #/ (2 * sph_l0(0))

        #Tev        = kT/ scipy.constants.electron_volt
        #kT         = 2 * np.pi * ((2 * vth**5 * p40_a[-1] * qe) / (3 * (2/me))) / m0 
        #c_lambda   = 2 * np.pi * (2 * vth**5 * p40_a[-1] / (3 * (2/me))) #np.exp(23.5 - np.log(np.sqrt(ne* 1e-6) * Tev **(5/4) - np.sqrt(1e-5 + ((np.log(Tev)-2)**2 )/16)))
        #c_lambda   = np.exp(23 - np.log(np.sqrt(ne * 1e-6) * (kT /scipy.constants.electron_volt)**(-1.5)))

        # b_min       = max(qe**2 / (4 * np.pi * eps_0 * 3 * kT/(0.5 * me)), scipy.constants.Planck / (np.sqrt(me * 3 * kT)))
        # b_max       = np.sqrt((eps_0 * kT) / (ne * qe**2))
        # c_lambda    = b_max/b_min

        # cc_freq_op    = vth**4 * np.sqrt(4*np.pi) * n0 * np.array([np.dot(gmw_a, gmx_a**3 * sigma_m * B(gmx_a,k,0)) for k in range(num_p)])
        # cc_freq       = scipy.constants.electron_volt * np.dot(cc_freq_op, fb[0::num_sh])
        # # print(cc_freq)
        # M             = (np.sqrt(6) * cc_freq ) / (2 * np.sqrt((qe**2 * ne) / (eps_0 * me)))
        # c_lambda      = ((12 * np.pi * (eps_0 * kT)**(1.5))/(qe**3 * np.sqrt(ne)))
        # c_lambda      = (c_lambda + M) / (1 + M)
        #c_lambda      = np.exp(6.314)

        # Tev        = kT/ scipy.constants.electron_volt
        # c_lambda     = np.exp(23.5 - np.log(np.sqrt(ne* 1e-6) * Tev **(5/4) - np.sqrt(1e-5 + ((np.log(Tev)-2)**2 )/16)))
        c_lambda      = ((12 * np.pi * (eps_0 * kT)**(1.5))/(qe**3 * np.sqrt(ne)))
        gamma_a       = (np.log(c_lambda) * (qe**4)) / (4 * np.pi * (eps_0 * me)**2) / (vth)**3

        

        # print( p40[-1] * vth**5 * mw(0) * 0.5 * scipy.constants.electron_mass * (2./ 3) / m0)
        print("mass=%.8E\t Coulomb logarithm %.8E \t gamma_a %.8E \t gamma_a * ne %.8E  \t kT=%.8E temp(ev)=%.8E temp (K)=%.8E " %(m0, np.log(c_lambda) , gamma_a, n0 * ionization_degree * gamma_a, kT, kT/scipy.constants.electron_volt, kT/scipy.constants.Boltzmann))

        

        
        return cc_collision * gamma_a

    def compute_rosenbluth_potentials_op(self, mw, vth, m_ab, Minv):
        V_TH          = vth
        ELE_VOLT      = collisions.ELECTRON_VOLT
        MAXWELLIAN_N  = collisions.MAXWELLIAN_N
        AR_NEUTRAL_N  = collisions.AR_NEUTRAL_N
        
        spec_sp  :sp.SpectralExpansionSpherical     = self._spec
        num_p         = spec_sp._p+1
        sph_lm        = spec_sp._sph_harm_lm
        num_sh        = len(spec_sp._sph_harm_lm)

        if self._r_basis_type != basis.BasisType.SPLINES:
            raise NotImplementedError

        gmx, gmw = spec_sp._basis_p.Gauss_Pn(self._NUM_Q_VR)

        k_vec    = spec_sp._basis_p._t
        dg_idx   = spec_sp._basis_p._dg_idx
        sp_order = spec_sp._basis_p._sp_order


        # compute the moment vectors
        def Pm(m):
            q_order       = ((sp_order + m + 1)//2) + 1
            q_order      *= 64
            qq_re         = np.polynomial.legendre.leggauss(q_order) 
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
                    qx      = 0.5 * ((xe - xb) * qq_re_points +  (xe + xb))
                    qw      = 0.5 * (xe - xb) * qq_re_weights
                    pm[i,j] = np.dot(qw, (qx**m) * spec_sp.basis_eval_radial(qx, i , 0))

                
                a_idx = np.where(k_vec[i + sp_order + 1] < gmx)[0]
                xb  = k_vec[i]
                xe  = k_vec[i + sp_order + 1]
                qx  = 0.5 * ((xe - xb) * qq_re_points +  (xe + xb))
                qw  = 0.5 * (xe - xb) * qq_re_weights
                pm[i, a_idx] = np.dot(qw, (qx**m) * spec_sp.basis_eval_radial(qx, i , 0))

            return pm
        
        def Qm(m):
            q_order       = ((sp_order + m + 1)//2) + 1
            q_order       *= 64
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
                    qx  = 0.5 * ((xe - xb) * qq_re_points +  (xe + xb))
                    qw  = 0.5 * (xe - xb) * qq_re_weights
                    qm[i,j] =  np.dot(qw, (qx**(-m)) * spec_sp.basis_eval_radial(qx, i , 0))

                a_idx = np.where(gmx < k_vec[i])[0]
                xb  = k_vec[i]
                xe  = k_vec[i + sp_order + 1]
                qx  = 0.5 * ((xe - xb) * qq_re_points +  (xe + xb))
                qw  = 0.5 * (xe - xb) * qq_re_weights
                qm[i, a_idx] = np.dot(qw, (qx**(-m)) * spec_sp.basis_eval_radial(qx, i , 0))

            return qm


        hl_v = np.zeros((num_sh, len(gmx), num_p))
        gl_v = np.zeros((num_sh, len(gmx), num_p))
        
        for lm_idx, lm in enumerate(sph_lm):
            ll = lm[0]
            f1 = (ll-0.5) / (ll + 1.5)

            m1 = (2 / (2*ll+1) ) * (1 + m_ab) * (Pm(ll + 2) / (gmx**(ll+1)) + Qm(ll-1) * gmx** (ll))
            m2 = -( 2 / (4* ll**2 - 1)) * ((Pm(ll + 2) / gmx**(ll-1) - f1 * Pm(ll+4)/gmx**(ll+1)) + (Qm(ll - 3) * gmx**(ll)   - f1 * Qm(ll-1) * gmx**(ll+2)))

            
            # import matplotlib.pyplot as plt
            # for i in range(0,num_p,10):
            #     plt.plot(gmx, m1[i,:], label="m1 l=%d k=%d"%(lm_idx, i))
            
            # plt.legend()
            # plt.show()
            # plt.close()


            hl_v[lm_idx, : , :] = np.transpose(m1) 
            gl_v[lm_idx, : , :] = np.transpose(m2)


        # hl_op = hl_v
        # gl_op = gl_v
        hl_op = np.zeros((num_p * num_sh, num_p * num_sh))
        gl_op = np.zeros((num_p * num_sh, num_p * num_sh))

        for lm_idx, lm in enumerate(sph_lm):
            for p in range(num_p):
                bp  = spec_sp.basis_eval_radial(gmx, p, 0)
                for k in range(num_p):
                    hl_op[p * num_sh + lm_idx, k * num_sh + lm_idx] = np.dot(gmx**2 * bp * hl_v[lm_idx, : , k], gmw)
                    gl_op[p * num_sh + lm_idx, k * num_sh + lm_idx] = np.dot(gmx**2 * bp * gl_v[lm_idx, : , k], gmw)
                
        hl_op = np.dot(Minv, hl_op)
        gl_op = np.dot(Minv, gl_op)
            
        return hl_op, gl_op

    def gamma_a(self, fb, mw, vth, n0, ion_deg):
        ne           = n0 * ion_deg 
        eps_0        = scipy.constants.epsilon_0
        me           = scipy.constants.electron_mass
        qe           = scipy.constants.e
        m0           = mw(0) * np.dot(fb,self._mass_op) * vth**3 
        kT           = mw(0) * (np.dot(fb, self._temp_op) * vth**5 * 0.5 * scipy.constants.electron_mass * (2./ 3) / m0) 
        kT           = np.abs(kT)
        
        c_lambda      = ((12 * np.pi * (eps_0 * kT)**(1.5))/(qe**3 * np.sqrt(ne)))
        gamma_a       = (np.log(c_lambda) * (qe**4)) / (4 * np.pi * (eps_0 * me)**2) / (vth)**3
        
        # print("mass=%.8E\t Coulomb logarithm %.8E \t gamma_a %.8E \t gamma_a * ne %.8E  \t kT=%.8E temp(ev)=%.8E temp (K)=%.8E " %(m0, np.log(c_lambda) , gamma_a, n0 * ion_deg * gamma_a, kT, kT/scipy.constants.electron_volt, kT/scipy.constants.Boltzmann))

        return gamma_a

    def rosenbluth_potentials(self, hl_op, gl_op, fb, mw, vth):
        
        ELE_VOLT      = collisions.ELECTRON_VOLT
        MAXWELLIAN_N  = collisions.MAXWELLIAN_N
        AR_NEUTRAL_N  = collisions.AR_NEUTRAL_N
        
        spec_sp  :sp.SpectralExpansionSpherical     = self._spec
        num_p         = spec_sp._p+1
        sph_lm        = spec_sp._sph_harm_lm
        num_sh        = len(spec_sp._sph_harm_lm)
        sp_order      = spec_sp._basis_p._sp_order

        gmx, gmw = spec_sp._basis_p.Gauss_Pn(self._NUM_Q_VR)
        hl = np.zeros(num_p * num_sh)
        gl = np.zeros(num_p * num_sh)

        hl = np.dot(hl_op, fb)
        gl = np.dot(gl_op, fb)

        # for lm_idx, lm in enumerate(sph_lm):
        #     hl[lm_idx::num_sh] = np.dot(hl_op[lm_idx] ,fb[lm_idx::num_sh]) 
        #     gl[lm_idx::num_sh] = np.dot(gl_op[lm_idx] ,fb[lm_idx::num_sh]) 

        # p20_a      = (np.sqrt(4 * np.pi / (2 * 0 + 1))) * 4*np.pi * np.dot(fb[0::num_sh], self._p2)
        # p40_a      = (np.sqrt(4 * np.pi / (2 * 0 + 1))) * 4*np.pi * np.dot(fb[0::num_sh], self._p4) 
        # q10_a      = (np.sqrt(4 * np.pi / (2 * 0 + 1))) * 4*np.pi * np.dot(fb[0::num_sh], self._q1) 
        # q30_a      = (np.sqrt(4 * np.pi / (2 * 0 + 1))) * 4*np.pi * np.dot(fb[0::num_sh], self._q3)

        # p31_a      = (np.sqrt(4 * np.pi / (2 * 0 + 1))) * 4*np.pi * np.dot(fb[1::num_sh], self._p3)
        # p51_a      = (np.sqrt(4 * np.pi / (2 * 0 + 1))) * 4*np.pi * np.dot(fb[1::num_sh], self._p5)
        # q01_a      = (np.sqrt(4 * np.pi / (2 * 0 + 1))) * 4*np.pi * np.dot(fb[1::num_sh], self._q0)

        # Vqr = spec_sp.Vq_r(gmx, 0, 1)

        # import matplotlib.pyplot as plt
        # # hl_0 = np.dot(np.transpose(Vqr),hl[0::num_sh])
        # # plt.plot(gmx, hl_0/(4*np.pi), label="new h0")
        # # plt.plot(gmx, (2) * (p20_a/gmx + q10_a), label="old h0")

        # # hl_1 = np.dot(np.transpose(Vqr),hl[1::num_sh])
        # # plt.plot(gmx, hl_1/(4*np.pi), label="new h1")
        # # plt.plot(gmx, (2) * (p31_a/gmx**2/3 + q01_a * gmx/3), label="old h1")

        # gl_0 = np.dot(np.transpose(Vqr),gl[0::num_sh])
        # plt.plot(gmx, gl_0, label="new g0")
        # plt.plot(gmx, gmx * p20_a + q30_a + (1/gmx/3) * p40_a + gmx**2 * q10_a/3, label="olg g0")

        # plt.legend()
        # plt.show()
        # plt.close()

        return hl, gl
    
        
    def coulomb_collision_op_assembly(self, mw, vth, gen_code=False):
        V_TH          = vth
        ELE_VOLT      = collisions.ELECTRON_VOLT
        MAXWELLIAN_N  = collisions.MAXWELLIAN_N
        AR_NEUTRAL_N  = collisions.AR_NEUTRAL_N
        
        spec_sp  :sp.SpectralExpansionSpherical     = self._spec
        num_p         = spec_sp._p+1
        sph_lm        = spec_sp._sph_harm_lm
        num_sh        = len(spec_sp._sph_harm_lm)
        
        if self._r_basis_type != basis.BasisType.SPLINES:
            raise NotImplementedError

        v      = sympy.Symbol('vr')
        mu     = sympy.Symbol('mu')
        phi    = sympy.Symbol('phi')

        if gen_code:
            # generate integrals for evaluation
            metric = sympy.Matrix([[1,0,0], [0, v**2/(1-mu**2), 0], [0, 0, v**2 * (1-mu**2)]])
            coords = [v, mu, phi]
            Ia, Ib = sym_cc.assemble_symbolic_cc_op(metric, coords, sph_lm[-1][0])

        
        k_vec     = spec_sp._basis_p._t
        dg_idx    = spec_sp._basis_p._dg_idx
        sp_order  = spec_sp._basis_p._sp_order

        B         = lambda vr, a : spec_sp.basis_eval_radial(vr, a, 0)
        DB        = lambda vr, a, d : spec_sp.basis_derivative_eval_radial(vr, a, 0, d) 

        gmx_a , gmw_a  = spec_sp._basis_p.Gauss_Pn(self._NUM_Q_VR)
        cc_mat_a       = np.zeros((num_p * num_sh, num_p * num_sh, num_p *num_sh))
        cc_mat_b       = np.zeros((num_p * num_sh, num_p * num_sh, num_p *num_sh))
        lmax           = sph_lm[-1][0]

        import cc_terms
        
        for p in range(num_p):
            for k in range(max(0, p - (sp_order+3) ), min(num_p, p + (sp_order+3))):
                for r in range(max(0, p - (sp_order+3) ), min(num_p, p + (sp_order+3))):
                    
                    k_min  = min(min(k_vec[p], k_vec[k]), k_vec[r])
                    k_max  = max(k_vec[r + sp_order + 1] , max(k_vec[p + sp_order + 1], k_vec[k + sp_order + 1]))
            
                    qx_idx = np.logical_and(gmx_a >= k_min, gmx_a <= k_max)
                    gmx    = gmx_a[qx_idx] 
                    gmw    = gmw_a[qx_idx] 

                    B_p_vr       =  B(gmx, p)
                    DB_p_dvr     = DB(gmx, p, 1)
                    DB_p_dvr_dvr = DB(gmx, p, 2)

                    B_k_vr       =  B(gmx, k)
                    DB_k_dvr     = DB(gmx, k, 1)
                    DB_k_dvr_dvr = DB(gmx, k, 2)

                    B_r_vr       =  B(gmx, r)
                    DB_r_dvr     = DB(gmx, r, 1)
                    DB_r_dvr_dvr = DB(gmx, r, 2)
            
                    for idx in cc_terms.Ia_nz:
                        if idx[0] > lmax or idx[1] > lmax or idx[2] > lmax:
                            continue
            
                        cc_mat_a[p * num_sh +  idx[0], k * num_sh + idx[1], r * num_sh +  idx[2]] = np.dot(gmw, cc_terms.Ia(B, DB, gmx, p, k, r, idx[0], idx[1], idx[2], B_p_vr, B_k_vr, B_r_vr, DB_p_dvr, DB_k_dvr, DB_r_dvr, DB_p_dvr_dvr, DB_k_dvr_dvr, DB_r_dvr_dvr)) 

        for p in range(num_p):
            for k in range(max(0, p - (sp_order+3) ), min(num_p, p + (sp_order+3))):
                for r in range(max(0, p - (sp_order+3) ), min(num_p, p + (sp_order+3))):

                    k_min  = min(min(k_vec[p], k_vec[k]), k_vec[r])
                    k_max  = max(k_vec[r + sp_order + 1] , max(k_vec[p + sp_order + 1], k_vec[k + sp_order + 1]))
            
                    qx_idx = np.logical_and(gmx_a >= k_min, gmx_a <= k_max)
                    gmx    = gmx_a[qx_idx] 
                    gmw    = gmw_a[qx_idx] 

                    B_p_vr       =  B(gmx, p)
                    DB_p_dvr     = DB(gmx, p, 1)
                    DB_p_dvr_dvr = DB(gmx, p, 2)

                    B_k_vr       =  B(gmx, k)
                    DB_k_dvr     = DB(gmx, k, 1)
                    DB_k_dvr_dvr = DB(gmx, k, 2)

                    B_r_vr       =  B(gmx, r)
                    DB_r_dvr     = DB(gmx, r, 1)
                    DB_r_dvr_dvr = DB(gmx, r, 2)

                    for idx in cc_terms.Ib_nz:
                        
                        if idx[0] > lmax or idx[1] > lmax or idx[2] > lmax:
                            continue
                    
                        cc_mat_b[p * num_sh +  idx[0], k * num_sh + idx[1], r * num_sh +  idx[2]] = np.dot(gmw, cc_terms.Ib(B, DB, gmx, p, k, r, idx[0], idx[1], idx[2], B_p_vr, B_k_vr, B_r_vr, DB_p_dvr, DB_k_dvr, DB_r_dvr, DB_p_dvr_dvr, DB_k_dvr_dvr, DB_r_dvr_dvr)) 
        

        return cc_mat_a, cc_mat_b



        

        
