"""
@package: Generic class to store spectral discretization in spherical coordinates, 

"""
import numpy as np
import basis 
import enum
from scipy.special import sph_harm
import enum
from advection_operator_spherical_polys import *

class QuadMode(enum.Enum):
    GMX     = 0 # default
    SIMPSON = 1


class SpectralExpansionSpherical:
    """
    Handles spectral decomposition with specified orders of expansion, w.r.t. basis_p
    domain and window by default set to None. None assumes the domain is unbounded. 
    """
    def __init__(self, p_order, basis_p, sph_harm_lm, domain=None, window=None):
        """
        @param p_order : number of basis functions used in radial direction
        @param basis_p : np.polynomial object
        @param sph_harm_lm : (l,m) indices of spherical harmonics to use
        """
        self._sph_harm_lm = sph_harm_lm
        self._p = p_order
        self._num_q_radial=270
        self._domain = domain
        self._window = window

        self._basis_p  = basis_p
        self._basis_1d = list()
        self._q_mode   = QuadMode.GMX

        for deg in range(self._p+1):
            self._basis_1d.append(self._basis_p.Pn(deg))

    def get_radial_basis(self):
        return self._basis_p

    def get_radial_basis_type(self):
        return self._basis_p._basis_type   

    def _sph_harm_real(self, l, m, theta, phi):
        # in python's sph_harm phi and theta are swapped
        Y = sph_harm(abs(m), l, phi, theta)
        if m < 0:
            Y = np.sqrt(2) * (-1)**m * Y.imag
        elif m > 0:
            Y = np.sqrt(2) * (-1)**m * Y.real
        else:
            Y = Y.real

        return Y 
    
    def basis_eval_full(self,r,theta,phi,k,l,m):
        """
        Evaluates 
        """
        return self.basis_eval_radial(r,k,l) * self._sph_harm_real(l, m, theta, phi)
    
    def basis_eval_radial(self,r,k,l):
        """
        Evaluates 
        """
        return np.nan_to_num(self._basis_1d[k](r,l))
    
    def basis_derivative_eval_radial(self,r,k,l,dorder):
        """
        Evaluates 
        """
        return np.nan_to_num(self._basis_p.diff(k,dorder)(r,l))
    
    def basis_eval_spherical(self, theta, phi,l,m):
        """
        Evaluates 
        """
        return self._sph_harm_real(l, m, theta, phi)

    def create_vec(self,dtype=float):
        num_c = (self._p +1)*len(self._sph_harm_lm)
        return np.zeros((num_c,1),dtype=dtype)

    def create_mat(self,dtype=float):
        """
        Create a matrix w.r.t the number of spectral coefficients. 
        """
        num_c = (self._p +1)*len(self._sph_harm_lm)
        return np.zeros((num_c,num_c),dtype=dtype)
    
    def get_num_coefficients(self):
        """
        returns the number of coefficients in  the spectral
        representation. 
        """
        return (self._p +1)*len(self._sph_harm_lm)

    def compute_mass_matrix(self, v_th=1.0):
        """
        Compute the mass matrix w.r.t the basis polynomials
        if the chosen basis is orthogonal set is_diagonal to True. 
        Given we always use spherical harmonics we will integrate them exactly
        """
        num_p   = self._p+1
        num_sh  = len(self._sph_harm_lm)

        if self.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY\
             or self.get_radial_basis_type() == basis.BasisType.LAGUERRE\
             or self.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_ENERGY_POLY:
             
            [gx, gw] = self._basis_p.Gauss_Pn(self._num_q_radial)
            l_modes = list(set([l for l,_ in self._sph_harm_lm]))
            
            mm=np.zeros((num_p*num_sh, num_p*num_sh))
            for i,l in enumerate(l_modes):
                Vq = self.Vq_r(gx, l)
                mm_l = np.array([Vq[p,:] * Vq[k,:] for p in range(num_p) for k in range(num_p)])
                mm_l = np.dot(mm_l,gw).reshape(num_p,num_p)

                for lm_idx, (l1,m) in enumerate(self._sph_harm_lm):
                    if(l==l1):
                        for p in range(num_p):
                            for k in range(num_p):
                                idx_pqs = p * num_sh + lm_idx
                                idx_klm = k * num_sh + lm_idx
                                mm[idx_pqs, idx_klm] = mm_l[p,k]

            return mm


        elif self.get_radial_basis_type() == basis.BasisType.SPLINES:
            [gx, gw] = self._basis_p.Gauss_Pn(basis.XlBSpline.get_num_q_pts(self._p,self._basis_p._sp_order,self._basis_p._q_per_knot))
            l_modes = list(set([l for l,_ in self._sph_harm_lm]))
            
            mm=np.zeros((num_p*num_sh, num_p*num_sh))
            for i,l in enumerate(l_modes):
                Vq = self.Vq_r(gx, l)
                # below is to make things more mem. efficient
                # mm_l = np.array([(gx**2) * Vq[p,:] * Vq[k,:] for p in range(num_p) for k in range(num_p)])
                # mm_l = np.dot(mm_l,gw).reshape(num_p,num_p)
                mm_l = np.zeros((num_p,num_p))
                for p in range(num_p):
                    for k in range(num_p):
                        mm_l[p,k]= np.dot((gx**2) * Vq[p,:] * Vq[k,:], gw)

                for lm_idx, (l1,m) in enumerate(self._sph_harm_lm):
                    if(l==l1):
                        for p in range(num_p):
                            for k in range(num_p):
                                idx_pqs = p * num_sh + lm_idx
                                idx_klm = k * num_sh + lm_idx
                                mm[idx_pqs, idx_klm] = mm_l[p,k]
            return mm
        elif self.get_radial_basis_type() == basis.BasisType.CHEBYSHEV_POLY:
            [gx, gw] = self._basis_p.Gauss_Pn(self._num_q_radial)
            l_modes  = list(set([l for l,_ in self._sph_harm_lm]))
            
            mm=np.zeros((num_p*num_sh, num_p*num_sh))
            for i,l in enumerate(l_modes):
                Vq   = self.Vq_r(gx, l)
                mm_l = np.array([ self._basis_p.Wx()(gx) * Vq[p,:] * Vq[k,:] for p in range(num_p) for k in range(num_p)])
                mm_l = np.dot(mm_l,gw).reshape(num_p,num_p)

                for lm_idx, (l1,m) in enumerate(self._sph_harm_lm):
                    if(l==l1):
                        for p in range(num_p):
                            for k in range(num_p):
                                idx_pqs = p * num_sh + lm_idx
                                idx_klm = k * num_sh + lm_idx
                                mm[idx_pqs, idx_klm] = mm_l[p,k]

            return mm
            
           
    def Vq_r(self, v_r, l, scale=1):
        """
        compute the basis Vandermonde for the all the basis function
        for the specified quadrature points.  
        """
        num_p        = self._p+1
        num_q_v_r    = len(v_r)

        _shape = tuple([num_p]) + v_r.shape
        Vq = np.zeros(_shape)

        for i in range(num_p):
            Vq[i] = scale * self.basis_eval_radial(v_r,i,l)
        
        return Vq
    
    def Vdq_r(self, v_r, l, d_order=1, scale=1):
        """
        compute the basis Vandermonde for the all the basis function
        derivatives for radial polynomials. 
        """
        num_p        = self._p+1
        num_q_v_r    = len(v_r)

        _shape = tuple([num_p]) + v_r.shape
        Vq = np.zeros(_shape)

        for i in range(num_p):
            Vq[i] = scale * self.basis_derivative_eval_radial(v_r,i,l,d_order)
        
        return Vq

    def Vq_sph_mg(self, v_theta, v_phi, scale=1):
        """
        compute the basis Vandermonde for the all the basis function
        for the specified quadrature points.  
        """

        num_sph_harm = len(self._sph_harm_lm)
        num_q_v_theta   = len(v_theta)
        num_q_v_phi     = len(v_phi)

        [Vt,Vp]      = np.meshgrid(v_theta, v_phi,indexing='ij')

        Vq = np.zeros((num_sph_harm,num_q_v_theta,num_q_v_phi))

        for lm_i, lm in enumerate(self._sph_harm_lm):
            Vq[lm_i] = scale * self.basis_eval_spherical(Vt,Vp,lm[0],lm[1])
        
        return Vq

    def Vq_sph(self, v_theta, v_phi, scale=1):
        """
        compute the basis Vandermonde for the all the basis function
        for the specified quadrature points.  
        """

        num_sph_harm = len(self._sph_harm_lm)
        assert v_theta.shape == v_phi.shape, "invalid shapes, use mesh grid to get matching shapes"
        _shape = tuple([num_sph_harm]) + v_theta.shape
        Vq = np.zeros(_shape)

        for lm_i, lm in enumerate(self._sph_harm_lm):
            Vq[lm_i] = scale * self.basis_eval_spherical(v_theta,v_phi,lm[0],lm[1])
        
        return Vq

    def Vq_full(self, v_r, v_theta, v_phi, scale=1):
        """
        compute the basis Vandermonde for the all the basis function
        for the specified quadrature points.  
        """

        V_r_q   = self.Vq_r(v_r,1.0)
        V_sph_q = self.Vq_sph(v_theta,v_phi,1.0)
        Vq= np.kron(V_r_q,V_sph_q)
        
        return Vq

    def compute_advection_matix(self):
        """
        loop based assembly of advection operator 
        """
        if self.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY:
            return assemble_advection_matix_lp_max(self._p, self._sph_harm_lm)
        elif self.get_radial_basis_type() == basis.BasisType.LAGUERRE:
            return assemble_advection_matix_lp_lag(self._p, self._sph_harm_lm)
        elif self.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_ENERGY_POLY:
            return assemble_advection_matix_lp_max_energy(self._p, self._sph_harm_lm)
        elif self.get_radial_basis_type() == basis.BasisType.SPLINES:
            num_p  = self._p+1
            num_sh = len(self._sph_harm_lm)
    
            lmodes = list(set([l for l,_ in self._sph_harm_lm]))
            num_l  = len(lmodes)
            l_max  = lmodes[-1]
            
            [gx, gw] = self._basis_p.Gauss_Pn(basis.XlBSpline.get_num_q_pts(self._p,self._basis_p._sp_order,self._basis_p._q_per_knot),True)
            
            mm1=np.zeros((num_p,num_p,num_l,num_l))
            mm2=np.zeros((num_p,num_p,num_l,num_l))

            for pl in range(num_l):
                Vr_pl  = self.Vq_r(gx,pl)
                for kl in range(num_l):
                    Vr_kl  = self.Vq_r(gx,kl)
                    Vdr_kl = self.Vdq_r(gx,kl,d_order=1)
                    for p in range(num_p):
                        for k in range(num_p):
                            mm1[p,k,pl,kl] = np.dot((gx**2) * Vr_pl[p,:] * Vdr_kl[k,:],gw)
                            mm2[p,k,pl,kl] = np.dot(gx * Vr_pl[p,:] * Vr_kl[k,:],gw)

            
            advec_mat  = np.zeros((num_p,num_sh,num_p,num_sh))
            for qs_idx,qs in enumerate(self._sph_harm_lm):
                lm   =  [qs[0]+1, qs[1]]
                if lm in self._sph_harm_lm:
                    lm_idx = self._sph_harm_lm.index(lm)
                    qs_mat = qs[0]**2+qs[0]+qs[1]
                    lm_mat = lm[0]**2+lm[0]+lm[1]
                    advec_mat[:,qs_idx,:,lm_idx] = mm1[:,:,qs[0],lm[0]] * AM(lm[0],lm[1]) + AD(lm[0],lm[1]) * mm2[:,:,qs[0],lm[0]]

                lm     =  [qs[0]-1, qs[1]]
                if lm in self._sph_harm_lm:
                    lm_idx = self._sph_harm_lm.index(lm)
                    qs_mat = qs[0]**2+qs[0]+qs[1]
                    lm_mat = lm[0]**2+lm[0]+lm[1]
                    advec_mat[:,qs_idx,:,lm_idx] = mm1[:,:,qs[0],lm[0]] * BM(lm[0],lm[1]) + BD(lm[0],lm[1]) * mm2[:,:,qs[0],lm[0]]
                
            
            advec_mat = advec_mat.reshape(num_p*num_sh, num_p*num_sh)
            #print("norm adv mat = %.8E"%np.linalg.norm(advec_mat))
            return advec_mat
        else:
            raise NotImplementedError

    def compute_advection_matix_ibp(self):
        if self.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_POLY:
            return assemble_advection_matix_lp_max(self._p, self._sph_harm_lm)
        elif self.get_radial_basis_type() == basis.BasisType.LAGUERRE:
            return assemble_advection_matix_lp_lag(self._p, self._sph_harm_lm)
        elif self.get_radial_basis_type() == basis.BasisType.MAXWELLIAN_ENERGY_POLY:
            return assemble_advection_matix_lp_max_energy(self._p, self._sph_harm_lm)
        elif self.get_radial_basis_type() == basis.BasisType.SPLINES:
            num_p  = self._p+1
            num_sh = len(self._sph_harm_lm)
    
            lmodes = list(set([l for l,_ in self._sph_harm_lm]))
            num_l  = len(lmodes)
            l_max  = lmodes[-1]
            
            [gx, gw] = self._basis_p.Gauss_Pn(basis.XlBSpline.get_num_q_pts(self._p,self._basis_p._sp_order,self._basis_p._q_per_knot),True)
            
            
            D_pk=np.zeros((num_p,num_p))
            S_pk=np.zeros((num_p,num_p))
        
            for p in range(num_p):
                for k in range(num_p):
                    D_pk[p,k] = np.dot((gx**2) * self.basis_derivative_eval_radial(gx,p,0,1) * self.basis_eval_radial(gx,k,0), gw)
                    S_pk[p,k] = np.dot(gx * self.basis_eval_radial(gx,p,0) * self.basis_eval_radial(gx,k,0), gw)
            

            A_qs_lm=np.zeros((num_sh,num_sh))
            B_qs_lm=np.zeros((num_sh,num_sh))

            for qs_idx,qs in enumerate(self._sph_harm_lm):
                lm   =  [qs[0]+1, qs[1]]
                if lm in self._sph_harm_lm:
                    lm_idx = self._sph_harm_lm.index(lm)
                    A_qs_lm[qs_idx,lm_idx] = AM(lm[0],lm[1])
                    B_qs_lm[qs_idx,lm_idx] = AD(lm[0],lm[1])

                lm     =  [qs[0]-1, qs[1]]
                if lm in self._sph_harm_lm:
                    lm_idx = self._sph_harm_lm.index(lm)
                    A_qs_lm[qs_idx,lm_idx] = BM(lm[0],lm[1])
                    B_qs_lm[qs_idx,lm_idx] = BD(lm[0],lm[1])

            eA, qA = np.linalg.eig(A_qs_lm)
            eA     = np.diag(eA)
            # eA=A_qs_lm
            # qA=np.eye(A_qs_lm.shape[0])

            #print("eig: ", np.linalg.norm(A_qs_lm- np.matmul(qA, np.matmul(eA, np.transpose(qA)))))
            S_qs_lm = (2 * eA  - np.matmul(np.transpose(qA), np.matmul(B_qs_lm, qA)))
            D_qs_km = eA

            #S_qs_lm = (2 * A_qs_lm - B_qs_lm)
            #D_qs_km = A_qs_lm
            advec_mat  = np.zeros((num_p,num_sh,num_p,num_sh))
            for qs_idx,qs in enumerate(self._sph_harm_lm):
                for lm_idx,lm in enumerate(self._sph_harm_lm):
                    advec_mat[:,qs_idx,:,lm_idx] = S_pk[:,:] * S_qs_lm[qs_idx, lm_idx]  +  D_pk[:,:] * D_qs_km [qs_idx, lm_idx] 


            advec_mat = advec_mat.reshape(num_p*num_sh, num_p*num_sh)
            
            k_vec    = self._basis_p._t
            dg_idx   = self._basis_p._dg_idx
            sp_order = self._basis_p._sp_order


            if len(dg_idx)>=4:
                # flux reconstruction operators
                f_lr  = np.eye(num_p) 
                f_rl  = np.eye(num_p) 

                # left to right flux reconstruction
                f_lr[dg_idx[2]:, :]         = 0

                # right to left flux reconstruction
                f_rl[0 : dg_idx[1] + 1 , :] = 0
                # print("l to r")
                # print(f_lr)
                # print("r to l")
                # print(f_rl)

                # face loop
                flux_mat=np.zeros_like(advec_mat)
                face_ids=[(1,2)]
                for f in face_ids:
                    fx = k_vec[dg_idx[f[0]] + sp_order]
                    assert fx== k_vec[dg_idx[f[1]] + sp_order], "flux assembly face coords does not match"
                    eps= 2*np.finfo(float).eps

                    d1 = (dg_idx[f[0]-1], dg_idx[f[0]+1])
                    d2 = (dg_idx[f[1]], dg_idx[f[1]+1])

                    f_pk_left  = np.zeros(num_p) 
                    f_pk_right = np.zeros(num_p) 

                    v_fx=np.array([self.basis_eval_radial(fx-eps,p,0) for p in range(num_p)])
                    
                    v_fx[v_fx<1e-10] = 0
                    v_fx[v_fx>1-eps] = 1.0
                    v_fx=v_fx.reshape(1,num_p)
                    
                    f_pk_left = np.matmul(np.transpose(v_fx),v_fx) * fx**2

                    v_fx=np.array([self.basis_eval_radial(fx+eps,p,0) for p in range(num_p)])
                    v_fx[v_fx<1e-10] = 0
                    v_fx[v_fx>1-eps] = 1.0
                    v_fx=v_fx.reshape(1,num_p)
                    
                    f_pk_right = np.matmul(np.transpose(v_fx),v_fx) * fx**2
                    
                    flux_from_left  = np.matmul(f_pk_left, f_lr)
                    flux_from_right = np.matmul(f_pk_right, f_rl)

                    #print(flux_from_left)
                    #print(flux_from_right)
                    
                    for lm_idx,lm in enumerate(self._sph_harm_lm):
                        if eA[lm_idx, lm_idx]>0:
                            for k in range(num_p):
                                flux_mat[dg_idx[f[0]] * num_sh + lm_idx, k*num_sh + lm_idx] += eA[lm_idx, lm_idx] * flux_from_left[dg_idx[f[0]],k]
                                flux_mat[dg_idx[f[1]] * num_sh + lm_idx, k*num_sh + lm_idx] -= eA[lm_idx, lm_idx] * flux_from_left[dg_idx[f[0]],k]
                            
                        else:
                            for k in range(num_p):
                                flux_mat[dg_idx[f[1]] * num_sh + lm_idx, k*num_sh + lm_idx] -= eA[lm_idx, lm_idx] * flux_from_right[dg_idx[f[1]],k]
                                flux_mat[dg_idx[f[0]] * num_sh + lm_idx, k*num_sh + lm_idx] += eA[lm_idx, lm_idx] * flux_from_right[dg_idx[f[1]],k]

                
                # fx = k_vec[dg_idx[-1] + sp_order]
                # eps= 2*np.finfo(float).eps

                # f_pk  = np.zeros(num_p) 
                # v_fx=np.array([self.basis_eval_radial(fx-eps,p,0) for p in range(num_p)])
                
                # v_fx[v_fx<1e-10] = 0
                # v_fx[v_fx>1-eps] = 1.0
                # v_fx=v_fx.reshape(1,num_p)
                
                # f_pk = np.matmul(np.transpose(v_fx),v_fx) * fx**2

                # flux_right_bdy  = np.matmul(f_pk, f_rl)
                # #flux_from_right = np.matmul(f_pk_right, f_rl)
                # print("bdy: flux left")
                # print(flux_right_bdy)
                
                # for lm_idx,lm in enumerate(self._sph_harm_lm):
                #     if eA[lm_idx, lm_idx]>0:
                #         for k in range(num_p):
                #             flux_mat[dg_idx[-1] * num_sh + lm_idx, k*num_sh + lm_idx] += eA[lm_idx, lm_idx] * flux_right_bdy[dg_idx[-1],k]
                #     else:
                #         advec_mat[(dg_idx[-1]) * num_sh + lm_idx, :] = 0
                #         advec_mat[(dg_idx[-1]) * num_sh + lm_idx, (dg_idx[-1]-1) * num_sh + lm_idx] = -1
                #         advec_mat[(dg_idx[-1]) * num_sh + lm_idx, (dg_idx[-1]) * num_sh + lm_idx] = 1
                #         pass

                advec_mat-=flux_mat
            return advec_mat, eA, qA
        elif self.get_radial_basis_type() == basis.BasisType.CHEBYSHEV_POLY:
            num_p  = self._p+1
            num_sh = len(self._sph_harm_lm)
    
            lmodes = list(set([l for l,_ in self._sph_harm_lm]))
            num_l  = len(lmodes)
            l_max  = lmodes[-1]
            
            [gx, gw] = self._basis_p.Gauss_Pn(self._num_q_radial)
            
            D_pk=np.zeros((num_p,num_p))
            S_pk=np.zeros((num_p,num_p))
        
            for p in range(num_p):
                for k in range(num_p):
                    D_pk[p,k] = np.dot(self._basis_p.Wx()(gx) * self.basis_derivative_eval_radial(gx,p,0,1) * self.basis_eval_radial(gx,k,0), gw)
                    S_pk[p,k] = np.dot((1/gx) * self._basis_p.Wx()(gx) * self.basis_eval_radial(gx,p,0) * self.basis_eval_radial(gx,k,0), gw)
            

            A_qs_lm=np.zeros((num_sh,num_sh))
            B_qs_lm=np.zeros((num_sh,num_sh))

            for qs_idx,qs in enumerate(self._sph_harm_lm):
                lm   =  [qs[0]+1, qs[1]]
                if lm in self._sph_harm_lm:
                    lm_idx = self._sph_harm_lm.index(lm)
                    A_qs_lm[qs_idx,lm_idx] = AM(lm[0],lm[1])
                    B_qs_lm[qs_idx,lm_idx] = AD(lm[0],lm[1])

                lm     =  [qs[0]-1, qs[1]]
                if lm in self._sph_harm_lm:
                    lm_idx = self._sph_harm_lm.index(lm)
                    A_qs_lm[qs_idx,lm_idx] = BM(lm[0],lm[1])
                    B_qs_lm[qs_idx,lm_idx] = BD(lm[0],lm[1])

            advec_mat  = np.zeros((num_p,num_sh,num_p,num_sh))
            for qs_idx,qs in enumerate(self._sph_harm_lm):
                for lm_idx,lm in enumerate(self._sph_harm_lm):
                    advec_mat[:,qs_idx,:,lm_idx] = A_qs_lm[qs_idx,lm_idx] * (D_pk[:,:] + 2 * S_pk[:,:]) - B_qs_lm[qs_idx,lm_idx] * S_pk[:,:]

            advec_mat = advec_mat.reshape(num_p*num_sh, num_p*num_sh)
            return advec_mat, np.eye(num_sh) , np.eye(num_sh)

            #eA, qA = np.linalg.eig(A_qs_lm)
            #eA     = np.diag(eA)
            #eA=A_qs_lm
            #qA=np.eye(A_qs_lm.shape[0])

            #print("eig: ", np.linalg.norm(A_qs_lm- np.matmul(qA, np.matmul(eA, np.transpose(qA)))))
            # S_qs_lm = (2 * eA  - np.matmul(np.transpose(qA), np.matmul(B_qs_lm, qA)))
            # D_qs_km = eA

            # #S_qs_lm = (2 * A_qs_lm - B_qs_lm)
            # #D_qs_km = A_qs_lm
            # advec_mat  = np.zeros((num_p,num_sh,num_p,num_sh))
            # for qs_idx,qs in enumerate(self._sph_harm_lm):
            #     for lm_idx,lm in enumerate(self._sph_harm_lm):
            #         advec_mat[:,qs_idx,:,lm_idx] = S_pk[:,:] * S_qs_lm[qs_idx, lm_idx]  +  D_pk[:,:] * D_qs_km [qs_idx, lm_idx] 


            # advec_mat = advec_mat.reshape(num_p*num_sh, num_p*num_sh)
            
            # k_vec    = self._basis_p._t
            # dg_idx   = self._basis_p._dg_idx
            # sp_order = self._basis_p._sp_order
        else:
            raise NotImplementedError

