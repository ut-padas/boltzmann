"""
@package: Generic class to store spectral discretization in spherical coordinates, 

"""
import numpy as np
import scipy
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

        #radial grid domains
        self._r_grid     = [self._window]
        self._r_basis_p  = [self._basis_p]

        # for deg in range(self._p+1):
        #     self._basis_1d.append(self._basis_p.Pn(deg))

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
        return np.nan_to_num(self._basis_p.Pn(k)(r,l)) #np.nan_to_num(self._basis_1d[k](r,l))
    
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
        num_c = (self._p +1)*len(self._sph_harm_lm) * len(self._r_grid)
        return np.zeros((num_c,1),dtype=dtype)

    def create_mat(self,dtype=float):
        """
        Create a matrix w.r.t the number of spectral coefficients. 
        """
        num_c = (self._p +1)*len(self._sph_harm_lm) * len(self._r_grid)
        return np.zeros((num_c,num_c),dtype=dtype)
    
    def get_num_coefficients(self):
        """
        returns the number of coefficients in  the spectral
        representation. 
        """
        return (self._p +1) * len(self._sph_harm_lm) * len(self._r_grid)

    def get_num_radial_domains(self):
        return len(self._r_grid)
    
    def get_dof_per_radial_domain(self):
        return (self._p+1) * len(self._sph_harm_lm)

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
            num_p  = self._p+1
            num_sh = len(self._sph_harm_lm)

            k_vec    = self._basis_p._t
            dg_idx   = self._basis_p._dg_idx
            sp_order = self._basis_p._sp_order
    
            l_modes = list(set([l for l,_ in self._sph_harm_lm]))
            num_l  = len(l_modes)
            l_max  = l_modes[-1]

            [gx, gw] = self._basis_p.Gauss_Pn(self._num_q_radial)
            mm=np.zeros((num_p*num_sh, num_p*num_sh))

            for e_id in range(0,len(dg_idx),2):
                ib=dg_idx[e_id]
                ie=dg_idx[e_id+1]

                xb=k_vec[ib]
                xe=k_vec[ie+sp_order+1]
                
                idx_set     = np.logical_and(gx>=xb, gx <=xe)
                gx_e , gw_e = gx[idx_set],gw[idx_set]

                #print("velocity radial domain (v/vth) :", (xb,xe), "with basis idx: ", ib, ie, gx_e)

                Vq   = self.Vq_r(gx_e, 0)
                mm_l = np.zeros((num_p,num_p))
                for p in range(ib, ie+1):
                    for k in range(ib, ie+1):
                        mm_l[p,k]= np.dot((gx_e**2) * Vq[p,:] * Vq[k,:], gw_e)

                for lm_idx, (l,m) in enumerate(self._sph_harm_lm):
                    for p in range(ib,ie+1):
                        for k in range(ib,ie+1):
                            idx_pqs = p * num_sh + lm_idx
                            idx_klm = k * num_sh + lm_idx
                            mm[idx_pqs, idx_klm] = mm_l[p,k]
    
            return mm
        elif self.get_radial_basis_type() == basis.BasisType.CHEBYSHEV_POLY:
            l_modes  = list(set([l for l,_ in self._sph_harm_lm]))
            mm       = self.create_mat()
            for e_id , ele in enumerate(self._r_grid):
                self._basis_p  = self._r_basis_p[e_id]
                print("domain :", self._basis_p._window)
                [gx, gw] = self._basis_p.Gauss_Pn(self._num_q_radial)
                for i,l in enumerate(l_modes):
                    Vq   = self.Vq_r(gx, l)
                    mm_l = np.array([ self._basis_p.Wx()(gx) * Vq[p,:] * Vq[k,:] for p in range(num_p) for k in range(num_p)])
                    mm_l = np.dot(mm_l,gw).reshape(num_p,num_p)

                    for lm_idx, (l1,m) in enumerate(self._sph_harm_lm):
                        if(l==l1):
                            for p in range(num_p):
                                for k in range(num_p):
                                    idx_pqs = e_id * (num_p * num_sh) + p * num_sh + lm_idx
                                    idx_klm = e_id * (num_p * num_sh) + k * num_sh + lm_idx
                                    mm[idx_pqs, idx_klm] = mm_l[p,k]
            return mm

    def inverse_mass_mat(self, v_th=1, Mmat=None):
        
        if Mmat is None:
            Mmat=self.compute_mass_matrix(v_th)

        def c_inv(mmat):
            L    = np.linalg.cholesky(mmat)
            Linv = scipy.linalg.solve_triangular(L, np.identity(mmat.shape[0]),lower=True) 
            #print("cholesky solver inverse : ", np.linalg.norm(np.matmul(L,Linv)-np.eye(L.shape[0]))/np.linalg.norm(np.eye(L.shape[0])))
            return np.matmul(np.transpose(Linv),Linv)
        
        minv         = self.create_mat()
        # if self.get_radial_basis_type() == basis.BasisType.SPLINES:
        #     minv_l0  = np.zeros((self._p+1, self._p+1))
        #     k_vec    = self._basis_p._t
        #     dg_idx   = self._basis_p._dg_idx
        #     sp_order = self._basis_p._sp_order
            
        #     num_sh   = len(self._sph_harm_lm)
        #     mm_l0    = Mmat[0::num_sh, 0::num_sh]

        #     for e_id in range(0,len(dg_idx),2):
        #         ib      = (dg_idx[e_id]) 
        #         ie      = (dg_idx[e_id+1])
        #         minv_l0[ib: ie+1, ib:ie+1] = c_inv(mm_l0[ib : ie+1, ib : ie+1])

        #         for lm_idx, lm in enumerate(self._sph_harm_lm):
        #             for p in range(ib, ie+1):
        #                 for k in range(ib, ie+1):
        #                     minv[p * num_sh + lm_idx, k * num_sh + lm_idx] = minv_l0[p,k]
        # else:
        dof_per_elem = (self._p+1) * len(self._sph_harm_lm)
        for e_id, ele_domain in enumerate(self._r_grid):
            ib = (e_id) * dof_per_elem
            ie = (e_id+1) * dof_per_elem
            minv[ib:ie,ib:ie] = c_inv(Mmat[ib : ie, ib : ie])
        
        return minv

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

    def dg_Vq_r(self,v_r, l, e_i, scale=1.0):
        num_p        = self._p+1
        num_q_v_r    = len(v_r)

        _shape = tuple([num_p]) + v_r.shape
        Vq = np.zeros(_shape)

        vw  = self._r_basis_p[e_i]._window
        idx_set = np.logical_and(v_r>= vw[0], v_r <=vw[1])

        for p in range(num_p):
            Vq[p, idx_set] = self._r_basis_p[e_i].Pn(p)(v_r[idx_set],l)
        #print(Vq)
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
            
            [gx, gw] = self._basis_p.Gauss_Pn(self._num_q_radial)
            
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

    def compute_advection_matix_dg(self, advection_dir=1):
        """
        advection direction = 1  : advection left to right 
        advection direction = -1 : advection right to left
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

            k_vec    = self._basis_p._t
            dg_idx   = self._basis_p._dg_idx
            sp_order = self._basis_p._sp_order
    
            lmodes = list(set([l for l,_ in self._sph_harm_lm]))
            num_l  = len(lmodes)
            l_max  = lmodes[-1]

            [gx, gw] = self._basis_p.Gauss_Pn(self._num_q_radial)
            D_pk=np.zeros((num_p,num_p))
            S_pk=np.zeros((num_p,num_p))

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
            S_qs_lm = (np.matmul(np.transpose(qA), np.matmul(B_qs_lm, qA)) - 2 * eA)
            D_qs_km = eA

            for e_id in range(0,len(dg_idx),2):
                ib=dg_idx[e_id]
                ie=dg_idx[e_id+1]
                
                xb=k_vec[ib]
                xe=k_vec[ie+sp_order+1]

                idx_set     = np.logical_and(gx>=xb, gx <=xe)
                gx_e , gw_e = gx[idx_set],gw[idx_set]

                for p in range(ib,ie+1):
                    for k in range(ib,ie+1):
                        D_pk[p,k] = np.dot((gx_e**2) * self.basis_derivative_eval_radial(gx_e,p,0,1) * self.basis_eval_radial(gx_e,k,0), gw_e)
                        S_pk[p,k] = np.dot(gx_e * self.basis_eval_radial(gx_e,p,0) * self.basis_eval_radial(gx_e,k,0), gw_e)

            
            advec_mat  = np.zeros((num_p,num_sh,num_p,num_sh))
            for qs_idx,qs in enumerate(self._sph_harm_lm):
                for lm_idx,lm in enumerate(self._sph_harm_lm):
                    advec_mat[:,qs_idx,:,lm_idx] = S_pk[:,:] * S_qs_lm[qs_idx, lm_idx]  -  D_pk[:,:] * D_qs_km [qs_idx, lm_idx] 

            advec_mat = advec_mat.reshape(num_p*num_sh, num_p*num_sh)
            flux_mat=np.zeros_like(advec_mat)
            for f_id in range(1, len(dg_idx)-2, 2):
                f=(dg_idx[f_id], dg_idx[f_id+1])
                fx = k_vec[f[0] + sp_order]
                assert fx == k_vec[f[1] + sp_order], "flux assembly face coords does not match"
                for lm_idx,lm in enumerate(self._sph_harm_lm):
                    if advection_dir * eA[lm_idx, lm_idx]>0:
                        flux_mat[f[0] * num_sh + lm_idx, f[0]*num_sh + lm_idx] +=   advection_dir * eA[lm_idx, lm_idx] * fx**2
                        flux_mat[f[1] * num_sh + lm_idx, f[0]*num_sh + lm_idx] += - advection_dir * eA[lm_idx, lm_idx] * fx**2
                    else:
                        flux_mat[f[1] * num_sh + lm_idx, f[1]*num_sh + lm_idx] += - advection_dir * eA[lm_idx, lm_idx]  * fx**2
                        flux_mat[f[0] * num_sh + lm_idx, f[1]*num_sh + lm_idx] +=   advection_dir * eA[lm_idx, lm_idx]  * fx**2
            
            aply_bdy=False
            if(aply_bdy):
                fx = k_vec[dg_idx[-1] + sp_order]
                assert fx == k_vec[-1], "flux assembly face coords does not match at the boundary"
                for lm_idx,lm in enumerate(self._sph_harm_lm):
                    flux_mat[dg_idx[-1] * num_sh + lm_idx, dg_idx[-1]*num_sh + lm_idx] += eA[lm_idx, lm_idx]  * fx**2
                    
            advec_mat+= advection_dir * flux_mat
            return advec_mat, eA, qA
        elif self.get_radial_basis_type() == basis.BasisType.CHEBYSHEV_POLY:
            num_p  = self._p+1
            num_sh = len(self._sph_harm_lm)
    
            lmodes = list(set([l for l,_ in self._sph_harm_lm]))
            num_l  = len(lmodes)
            l_max  = lmodes[-1]

            # compute the advection coefficient matrices
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

            S_qs_lm = (2 * eA  - np.matmul(np.transpose(qA), np.matmul(B_qs_lm, qA)))
            D_qs_km = eA

            advmat_diag = list()
            for e_id, ele_domain  in enumerate(self._r_grid):
                self._basis_p = self._r_basis_p[e_id]
                [gx, gw] = self._basis_p.Gauss_Pn(self._num_q_radial)
                
                D_pk=np.zeros((num_p,num_p))
                S_pk=np.zeros((num_p,num_p))
            
                for p in range(num_p):
                    for k in range(num_p):
                        D_pk[p,k] = np.dot(self._basis_p.Wx()(gx) * self.basis_derivative_eval_radial(gx,p,0,1) * self.basis_eval_radial(gx,k,0), gw)
                        S_pk[p,k] = np.dot((1/gx) * self._basis_p.Wx()(gx) * self.basis_eval_radial(gx,p,0) * self.basis_eval_radial(gx,k,0), gw)
                
                # # without diagonalization. 
                # advec_mat  = np.zeros((num_p,num_sh,num_p,num_sh))
                # for qs_idx,qs in enumerate(self._sph_harm_lm):
                #     for lm_idx,lm in enumerate(self._sph_harm_lm):
                #         advec_mat[:,qs_idx,:,lm_idx] = A_qs_lm[qs_idx,lm_idx] * (D_pk[:,:] + 2 * S_pk[:,:]) - B_qs_lm[qs_idx,lm_idx] * S_pk[:,:]

                # advec_mat = advec_mat.reshape(num_p*num_sh, num_p*num_sh)
                advec_mat  = np.zeros((num_p,num_sh,num_p,num_sh))
                for qs_idx,qs in enumerate(self._sph_harm_lm):
                    for lm_idx,lm in enumerate(self._sph_harm_lm):
                        advec_mat[:,qs_idx,:,lm_idx] = S_qs_lm[qs_idx, lm_idx] * S_pk[:,:]   +  D_qs_km [qs_idx, lm_idx] * D_pk[:,:]

                advec_mat = advec_mat.reshape((num_p * num_sh, num_p * num_sh))
                advmat_diag.append(advec_mat)

            advec_mat      = scipy.linalg.block_diag(*advmat_diag)
            flux_mat       = self.create_mat()
            
            dof_p_d        = (self._p+1) * len(self._sph_harm_lm)
            
            for e_id in range(len(self._r_grid)-1):
                fx = self._r_grid[e_id][1]
                assert fx == self._r_grid[e_id + 1][0], "radial grid faces are not aligned. "
                fx_fac = (fx**2) * np.exp(-fx**2)
                for lm_idx,lm in enumerate(self._sph_harm_lm):
                    if eA[lm_idx, lm_idx]>0:
                        for p in range(num_p):
                            for k in range(num_p):
                                rid = e_id * dof_p_d + p * num_sh + lm_idx
                                cid = e_id * dof_p_d + k * num_sh + lm_idx
                                flux_mat[rid   ,         cid]   +=  eA[lm_idx, lm_idx] * fx_fac * self._r_basis_p[e_id].Pn(p)(fx,0)   * self._r_basis_p[e_id].Pn(k)(fx,0)   
                                flux_mat[rid + dof_p_d , cid]   += -eA[lm_idx, lm_idx] * fx_fac * self._r_basis_p[e_id+1].Pn(p)(fx,0) * self._r_basis_p[e_id].Pn(k)(fx,0)  
                    else:
                        for p in range(num_p):
                            for k in range(num_p):
                                rid = (e_id+1) * dof_p_d  + p * num_sh + lm_idx
                                cid = (e_id+1) * dof_p_d  + k * num_sh + lm_idx
                                flux_mat[rid           , cid]   += -eA[lm_idx, lm_idx] * fx_fac * self._r_basis_p[e_id+1].Pn(p)(fx,0) * self._r_basis_p[e_id+1].Pn(k)(fx,0)
                                flux_mat[rid-dof_p_d   , cid]   +=  eA[lm_idx, lm_idx] * fx_fac * self._r_basis_p[e_id].Pn(p)(fx,0)   * self._r_basis_p[e_id+1].Pn(k)(fx,0)

            e_id    = len(self._r_grid)-1
            fx      = self._r_grid[-1][1]
            fx_fac  = (fx**2) * np.exp(-fx**2)
            for lm_idx,lm in enumerate(self._sph_harm_lm):
                if eA[lm_idx, lm_idx]>0:
                    for p in range(num_p):
                        for k in range(num_p):
                            rid = e_id * dof_p_d + p * num_sh + lm_idx
                            cid = e_id * dof_p_d + k * num_sh + lm_idx
                            flux_mat[rid , cid]   +=  eA[lm_idx, lm_idx] * fx_fac * self._r_basis_p[e_id].Pn(p)(fx,0)   * self._r_basis_p[e_id].Pn(k)(fx,0)
            #     else:
            #         for p in range(num_p):
            #             for k in range(num_p):
            #                 rid = e_id * dof_p_d + p * num_sh + lm_idx
            #                 cid = e_id * dof_p_d + k * num_sh + lm_idx
            #                 flux_mat[rid , cid]   +=  eA[lm_idx, lm_idx] * fx_fac * self._r_basis_p[e_id].Pn(p)(fx,0)   * self._r_basis_p[e_id].Pn(k)(fx,0)

                            
             

            advec_mat = advec_mat - flux_mat
            #print(flux_mat)
            #print(advec_mat)
            return advec_mat, eA, qA

        else:
            raise NotImplementedError

    def diffusion_mat(self):
        if self.get_radial_basis_type() == basis.BasisType.SPLINES:
            gmx ,   gmw  = self._basis_p.Gauss_Pn(self._num_q_radial)
            
            k_vec        = self._basis_p._t
            dg_idx       = self._basis_p._dg_idx
            sp_order     = self._basis_p._sp_order
            diffusion_op = self.create_mat()

            num_sh       = len(self._sph_harm_lm)
            
            # idx_set     = np.logical_and(self._gmx>=xb, self._gmx <=xe)
            # gx_e , gw_e = self._gmx[idx_set],self._gmw[idx_set]
            gx_e , gw_e = gmx , gmw
            
            for e_id in range(0,len(dg_idx),2):
                ib=dg_idx[e_id]
                ie=dg_idx[e_id+1]

                xb=k_vec[ib]
                xe=k_vec[ie+sp_order+1]
                for p in range(ib,ie+1):
                    for k in range(ib,ie+1):
                        tmp = (gx_e**2) * self.basis_derivative_eval_radial(gx_e,p,0,1) * self.basis_derivative_eval_radial(gx_e,k,0,1) + 2 *gx_e * self.basis_derivative_eval_radial(gx_e,k,0,1) * self.basis_eval_radial(gx_e,p,0)
                        tmp = -np.dot(gw_e, tmp)

                        for qs_idx, (q,s) in enumerate(self._sph_harm_lm):
                            diffusion_op[p * num_sh + qs_idx , k * num_sh + qs_idx] +=tmp
        
            return diffusion_op
        else:
            raise NotImplementedError