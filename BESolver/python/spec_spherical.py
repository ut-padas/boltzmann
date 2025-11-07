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
import mesh 
import scipy.constants

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
        self._num_q_radial=None
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
        # theta = np.array(theta)
        # phi   = np.array(phi)
        # assert theta.shape == phi.shape, "theta and phi must have the same shape"
        Y = sph_harm(abs(m), l, phi, theta)
        if m < 0:
            Y = np.sqrt(2) * (-1)**m * Y.imag
        elif m > 0:
            Y = np.sqrt(2) * (-1)**m * Y.real
        else:
            Y = Y.real

        return Y 
    
    def _hemi_sph_harm_real(self, l, m, theta, phi, mode):
        # theta = np.array(theta)
        # phi   = np.array(phi)
        # assert theta.shape == phi.shape, "theta and phi must have the same shape"
        Y = np.zeros_like(theta)
        t = np.zeros_like(theta)
        if mode=="+":
            #assert (theta <= np.pi/2).all() and (theta >= 0).all(), "theta out of range for north pole hemisphere"
            idx      = np.logical_and(theta <= np.pi/2, theta >= 0)
            t[idx]   = np.arccos(2 * np.cos(theta[idx]) - 1)
        else:
            assert mode=="-", "invalid hemisphere mode"
            idx      = np.logical_and(theta >= np.pi/2, theta <= np.pi)
            t[idx]   = np.arccos(2 * np.cos(theta[idx]) + 1)
            
        
        Ysh  = sph_harm(abs(m), l, phi[idx], t[idx])
        if m < 0:
            Y[idx] = np.sqrt(2) * (-1)**m * Ysh.imag
        elif m > 0:
            Y[idx] = np.sqrt(2) * (-1)**m * Ysh.real
        else:
            Y[idx] = Ysh.real

        return np.sqrt(2) * Y 
    
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

    def compute_mass_matrix_supg(self, v_th=1.0):
        num_p   = self._p+1
        num_sh  = len(self._sph_harm_lm)

        if self.get_radial_basis_type() == basis.BasisType.SPLINES:
            k_vec    = self._basis_p._t
            dg_idx   = self._basis_p._dg_idx
            sp_order = self._basis_p._sp_order
    
            # l_modes = list(set([l for l,_ in self._sph_harm_lm]))
            # num_l  = len(l_modes)
            # l_max  = l_modes[-1]

            [gx, gw] = self._basis_p.Gauss_Pn(2 * (sp_order + 2) * self._basis_p._num_knot_intervals)
            mm       = np.zeros((num_p, num_p))

            for e_id in range(0,len(dg_idx),2):
                ib=dg_idx[e_id]
                ie=dg_idx[e_id+1]

                xb=k_vec[ib]
                xe=k_vec[ie+sp_order+1]
                
                idx_set     = np.logical_and(gx>=xb, gx <=xe)
                gx_e , gw_e = gx[idx_set],gw[idx_set]

                for p in range(ib, ie+1):
                    k_min   = k_vec[p]
                    k_max   = k_vec[p + sp_order + 1]
                    qx_idx  = np.logical_and(gx_e >= k_min, gx_e <= k_max)
                    gmx     = gx_e[qx_idx]
                    gmw     = gw_e[qx_idx]
                    db_p    = self.basis_derivative_eval_radial(gmx, p, 0, 1)  

                    for k in range(max(ib, p - (sp_order+3) ), min(ie+1, p + (sp_order+3))):
                        #db_k       = self.basis_derivative_eval_radial(gmx, k, 0, 1)
                        b_k       = self.basis_eval_radial(gmx, k, 0)
                        mm[p,k]   = np.dot((gmx**2) * db_p * b_k, gmw)

            return mm
    
    def supg_param(self, v_th=1.0):
        if self.get_radial_basis_type() == basis.BasisType.SPLINES:
            vg    = self._basis_p._t_unique
            dv    = vg[1:] - vg[0:-1]
            tau   = np.min(dv) / 2 / vg[-1] 
            print("tau supg : %.4E"%(tau))
            return tau
            
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

            [gx, gw] = self._basis_p.Gauss_Pn(2 * (sp_order + 2) * self._basis_p._num_knot_intervals)
            mm=np.zeros((num_p*num_sh, num_p*num_sh))

            for e_id in range(0,len(dg_idx),2):
                ib=dg_idx[e_id]
                ie=dg_idx[e_id+1]

                xb=k_vec[ib]
                xe=k_vec[ie+sp_order+1]
                
                idx_set     = np.logical_and(gx>=xb, gx <=xe)
                gx_e , gw_e = gx[idx_set],gw[idx_set]

                #print("velocity radial domain (v/vth) :", (xb,xe), "with basis idx: ", ib, ie, gx_e)

                mm_l = np.zeros((num_p,num_p))
                for p in range(ib, ie+1):
                    k_min   = k_vec[p]
                    k_max   = k_vec[p + sp_order + 1]
                    qx_idx  = np.logical_and(gx_e >= k_min, gx_e <= k_max)
                    gmx     = gx_e[qx_idx]
                    gmw     = gw_e[qx_idx]
                    b_p     = self.basis_eval_radial(gmx, p, 0)  

                    for k in range(max(ib, p - (sp_order+3) ), min(ie+1, p + (sp_order+3))):
                        b_k       = self.basis_eval_radial(gmx, k, 0)
                        mm_l[p,k] = np.dot((gmx**2) * b_p * b_k, gmw)

                # boundary correction term assuming exponential analytically integrated out. 
                # x_max = k_vec[-1]
                # mm_l[num_p-1, num_p-1]+=(1.0/16.0) * (4 * x_max * (np.sqrt(2*np.pi) * x_max + 2) + np.sqrt(2*np.pi))

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
            num_p     = self._p+1
            num_sh    = len(self._sph_harm_lm)
            sp_order  = self._basis_p._sp_order
            k_vec     = self._basis_p._t
            dg_idx    = self._basis_p._dg_idx

            assert len(dg_idx)==2 , ""
    
            lmodes = list(set([l for l,_ in self._sph_harm_lm]))
            num_l  = len(lmodes)
            l_max  = lmodes[-1]
            
            [gx_e, gw_e] = self._basis_p.Gauss_Pn((sp_order + 2) * 2 * self._basis_p._num_knot_intervals)
            
            mm1=np.zeros((num_p,num_p))
            mm2=np.zeros((num_p,num_p))

            for p in range(num_p):
                k_min   = k_vec[p]
                k_max   = k_vec[p + sp_order + 1]
                qx_idx  = np.logical_and(gx_e >= k_min, gx_e <= k_max)
                gmx     = gx_e[qx_idx]
                gmw     = gw_e[qx_idx]
                b_p     = self.basis_eval_radial(gmx, p, 0)  
                for k in range(max(0, p - (sp_order+3)), min(num_p, p + (sp_order+3))):
                    b_k  = self.basis_eval_radial(gmx, k, 0)
                    db_k = self.basis_derivative_eval_radial(gmx, k, 0, 1)
                    mm1[p,k] = np.dot((gmx ** 2) * b_p * db_k, gmw)
                    mm2[p,k] = np.dot( gmx * b_p * b_k, gmw)

            # boundary correction term assuming exponential analytically integrated out. 
            # x_max = k_vec[-1]
            # mm1[num_p-1, num_p-1] += 0.25 * (-x_max * (2 * x_max + np.sqrt(2 * np.pi)) -1)
            # mm2[num_p-1, num_p-1] += 0.25 * (np.sqrt(2 * np.pi) * x_max + 1)

            
            advec_mat  = np.zeros((num_p,num_sh,num_p,num_sh))
            for qs_idx,qs in enumerate(self._sph_harm_lm):
                lm   =  [qs[0]+1, qs[1]]
                if lm in self._sph_harm_lm:
                    lm_idx = self._sph_harm_lm.index(lm)
                    qs_mat = qs[0]**2+qs[0]+qs[1]
                    lm_mat = lm[0]**2+lm[0]+lm[1]
                    advec_mat[:,qs_idx,:,lm_idx] = mm1[:,:] * AM(lm[0],lm[1]) + AD(lm[0],lm[1]) * mm2[:,:]

                lm     =  [qs[0]-1, qs[1]]
                if lm in self._sph_harm_lm:
                    lm_idx = self._sph_harm_lm.index(lm)
                    qs_mat = qs[0]**2+qs[0]+qs[1]
                    lm_mat = lm[0]**2+lm[0]+lm[1]
                    advec_mat[:,qs_idx,:,lm_idx] = mm1[:,:] * BM(lm[0],lm[1]) + BD(lm[0],lm[1]) * mm2[:,:]
                
            
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

            [gx, gw] = self._basis_p.Gauss_Pn((sp_order + 2) * 2 * self._basis_p._num_knot_intervals)
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
            
            aply_bdy=True
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
    
    def compute_advection_matrix_ordinates(self, xp_vt, use_vt_upwinding=False):
        """
        ordinaltes v-space advection operator
        """
        assert self.get_radial_basis_type() == basis.BasisType.SPLINES, "only spline basis is implemented for v-advection with ordinates"

        xp                = np
        Nr                = self._p+1
        Nvt               = len(xp_vt)
        
        sp_order          = self._basis_p._sp_order
        k_vec             = self._basis_p._t
        dg_idx            = self._basis_p._dg_idx
        num_p             = Nr

        def __galerkin_vr__():
            k_vec    = self._basis_p._t
            dg_idx   = self._basis_p._dg_idx
            sp_order = self._basis_p._sp_order
    
            [gx, gw] = self._basis_p.Gauss_Pn((sp_order + 8) * self._basis_p._num_knot_intervals)
            mm1      = np.zeros((num_p, num_p))
            mm2      = np.zeros((num_p, num_p))
            mm3_supg = np.zeros((num_p, num_p))
            mm4_supg = np.zeros((num_p, num_p))
            
            assert len(dg_idx) == 2, "DG in vr is not implemented yet"

            for e_id in range(0,len(dg_idx),2):
                ib=dg_idx[e_id]
                ie=dg_idx[e_id+1]

                xb=k_vec[ib]
                xe=k_vec[ie+sp_order+1]
                
                idx_set     = np.logical_and(gx>=xb, gx <=xe)
                gx_e , gw_e = gx[idx_set],gw[idx_set]

                for p in range(ib, ie+1):
                    k_min   = k_vec[p]
                    k_max   = k_vec[p + sp_order + 1]
                    qx_idx  = np.logical_and(gx_e >= k_min, gx_e <= k_max)
                    gmx     = gx_e[qx_idx]
                    gmw     = gw_e[qx_idx]
                    b_p     = self.basis_eval_radial(gmx, p, 0)  
                    db_p    = self.basis_derivative_eval_radial(gmx, p, 0, 1)  
                    for k in range(max(ib, p - (sp_order+3) ), min(ie+1, p + (sp_order+3))):
                        db_k           = self.basis_derivative_eval_radial(gmx, k, 0, 1)
                        b_k            = self.basis_eval_radial(gmx, k, 0)  
                        
                        mm1[p,k]       = np.dot((gmx**2) * b_p * db_k, gmw)
                        mm2[p,k]       = np.dot((gmx) * b_p * b_k , gmw)

                        # mm3_supg[p,k]  = np.dot(gmx**2 * db_p * db_k, gmw)
                        # mm4_supg[p,k]  = np.dot(gmx**1 * db_p * b_k, gmw)
                        
            
            return mm1, mm2 #, mm3_supg, mm4_supg
        
        try:
            import cupy as cp
            cp.cuda.Device(0).use()  # Attempt to use the first CUDA device
            xp = cp 
            print("using GPUs for adv_v assembly")
    
        except Exception as e:
            print(f"CUDA device not found or accessible: {e} using CPU based assembly")

        def asnumpy(x):
            if (xp == np):
                return x            
            else:
                return cp.asnumpy(x)
                
        if (use_vt_upwinding == True):
            Dvt_LtoR          = xp.asarray(mesh.upwinded_dvt(xp_vt,1, 5, "L", use_cdx_internal=False))
            Dvt_RtoL          = xp.asarray(mesh.upwinded_dvt(xp_vt,1, 5, "R", use_cdx_internal=False))
            
            # Dvt_LtoR[-1, :]   = 0.0
            # Dvt_LtoR[-1, -1]  = 1.0

            # Dvt_RtoL[0, :]    = 0.0
            # Dvt_RtoL[0, 0]    = 1.0


            # print("L2R\n", Dvt_LtoR)
            # print("R2L\n", Dvt_RtoL)
            # Dvt_LtoR          = xp.asarray(mesh.upwinded_dvt(xp_vt, "LtoR",pw=2))
            # Dvt_RtoL          = xp.asarray(mesh.upwinded_dvt(xp_vt, "RtoL",pw=2))

            
            DvtEp             = xp.zeros((Nvt, Nvt)) # E > 0
            DvtEn             = xp.zeros((Nvt, Nvt)) # E < 0

            # DvtEp             = Dvt_LtoR
            # DvtEn             = Dvt_RtoL

            # note: this is only because, xp_vt grid is [pi, 0] (i.e., decending order)
            DvtEp       = Dvt_RtoL
            DvtEn       = Dvt_LtoR

            B, C         = __galerkin_vr__()
            print(B)
            B, C         = xp.asarray(B), xp.asarray(C) #, xp.asarray(B_SUPG), xp.asarray(C_SUPG)
            xp_cos_vt    = xp.cos(xp.asarray(xp_vt))
            xp_sin_vt    = xp.sin(xp.asarray(xp_vt))

            B           = xp.kron(B, xp.diag(xp_cos_vt)).reshape((Nr * Nvt, Nr * Nvt))
            DvtEp1      = xp.kron(C, xp.diag(xp_sin_vt) @ DvtEp).reshape((Nr * Nvt, Nr * Nvt))
            DvtEn1      = xp.kron(C, xp.diag(xp_sin_vt) @ DvtEn).reshape((Nr * Nvt, Nr * Nvt))

            op_adv_v_Ep = (B -  DvtEp1)
            op_adv_v_En = (B -  DvtEn1)

            # op_adv_v_Ep = (B -  DvtEp1 + self.supg_param() * (xp.kron(B_SUPG, xp.diag(xp_cos_vt**2)) + xp.kron(C_SUPG, xp.diag(xp_cos_vt * xp_sin_vt) @ DvtEp)))
            # op_adv_v_En = (B -  DvtEn1 + self.supg_param() * (xp.kron(B_SUPG, xp.diag(xp_cos_vt**2)) + xp.kron(C_SUPG, xp.diag(xp_cos_vt * xp_sin_vt) @ DvtEn)))

            return asnumpy(op_adv_v_Ep), asnumpy(op_adv_v_En)
        else:
            Dvt         = xp.asarray(mesh.central_dx(xp_vt, 1, 5))
            B, C        = __galerkin_vr__()
            B, C        = xp.asarray(B), xp.asarray(C)

            xp_cos_vt   = xp.cos(xp.asarray(xp_vt))
            xp_sin_vt   = xp.sin(xp.asarray(xp_vt))

            op_adv_v    = xp.kron(B, xp.diag(xp_cos_vt)).reshape((Nr * Nvt, Nr * Nvt)) - xp.kron(C, xp_sin_vt * Dvt).reshape((Nr * Nvt, Nr * Nvt))

            B           = xp.kron(B, xp.diag(xp_cos_vt)).reshape((Nr * Nvt, Nr * Nvt))
            Dvt         = xp.kron(C, xp.diag(xp_sin_vt) @ Dvt).reshape((Nr * Nvt, Nr * Nvt))
            op_adv_v    = (B - Dvt)
            return asnumpy(op_adv_v)

    def compute_advection_matrix_vrvt_fv(self, xp_vr, xp_vt, sw_vr=2, sw_vt=2, use_upwinding=True):
        """
        0-th order upwinded flux reconstruction is equivalent for 1st order FDs
        This ensures monotonicity
        """
        
        xp                = np
        try:
            import cupy as cp
            cp.cuda.Device(0).use()  # Attempt to use the first CUDA device
            xp = cp 
            print("using GPUs for adv_v assembly")
    
        except Exception as e:
            print(f"CUDA device not found or accessible: {e} using CPU based assembly")

        def asnumpy(x):
            if (xp == np):
                return x            
            else:
                return cp.asnumpy(x)
            

        assert xp_vr[0]  > self._basis_p._domain[0]
        assert xp_vr[-1] < self._basis_p._domain[1]

        assert xp_vt[0]  > 0
        assert xp_vt[-1] < np.pi
                
        num_vr            = len(xp_vr)
        num_vt            = len(xp_vt)

        xp_cos_vt         = xp.cos(xp_vt)
        xp_sin_vt         = xp.sin(xp_vt)
        vr_inv            = 1/xp_vr
        k_domain          = self._basis_p._domain

        if (use_upwinding):
            Dvt_LtoR          = xp.asarray(mesh.upwinded_dvt(xp_vt,1, sw_vt, "L", use_cdx_internal=False))
            Dvt_RtoL          = xp.asarray(mesh.upwinded_dvt(xp_vt,1, sw_vt, "R", use_cdx_internal=False))

            Dvr_LtoR          = xp.asarray(mesh.upwinded_dx(xp_vr,1 , sw_vr, "L"))
            Dvr_RtoL          = xp.asarray(mesh.upwinded_dx(xp_vr,1 , sw_vr, "R"))

            Dvr_LtoR[0,:]     = 0.0
            Dvr_LtoR[0,0]     = 1/(xp_vr[0] - k_domain[0])

            Dvr_RtoL[-1 , :]  = 0.0
            Dvr_RtoL[-1 ,-1]  = -1/(k_domain[1] - xp_vr[-1])


            Ep   = xp.zeros((num_vr, num_vt, num_vr, num_vt))
            En   = xp.zeros((num_vr, num_vt, num_vr, num_vt))
            

            mp            = num_vt // 2 
            assert (xp_cos_vt[mp:]>0).all()  == True
            assert (xp_cos_vt[0:mp]<0).all() == True

            Ep[:,  mp:, :,  mp:] = xp.kron(Dvr_RtoL, xp.diag(xp_cos_vt[mp:])) .reshape((num_vr, mp, num_vr, mp))
            Ep[:, 0:mp, :, 0:mp] = xp.kron(Dvr_LtoR, xp.diag(xp_cos_vt[0:mp])).reshape((num_vr, mp, num_vr, mp))

            En[:,  mp:, :,  mp:] = xp.kron(Dvr_LtoR, xp.diag(xp_cos_vt[mp:])) .reshape((num_vr, mp, num_vr, mp))
            En[:, 0:mp, :, 0:mp] = xp.kron(Dvr_RtoL, xp.diag(xp_cos_vt[0:mp])).reshape((num_vr, mp, num_vr, mp))

            adv_mat_Ep    = Ep.reshape((num_vr * num_vt, num_vr * num_vt)) - xp.kron(xp.diag(vr_inv), xp.diag(xp_sin_vt) @ Dvt_RtoL)
            adv_mat_En    = En.reshape((num_vr * num_vt, num_vr * num_vt)) - xp.kron(xp.diag(vr_inv), xp.diag(xp_sin_vt) @ Dvt_LtoR)

            return adv_mat_Ep, adv_mat_En
        else:
            Dvt          = xp.asarray(mesh.central_dx(xp_vt, 1, sw_vt)) 
            Dvr          = xp.asarray(mesh.central_dx(xp_vr, 1, sw_vr))

            adv_mat      = xp.kron(Dvr, xp.diag(xp_cos_vt)) - xp.kron(xp.diag(vr_inv), xp.diag(xp_sin_vt) @ Dvt)
        
            return adv_mat

    def Vq_hsph_mg(self, v_theta, v_phi, scale=1):
        """
        compute the basis Vandermonde for the all the basis function
        for the specified quadrature points.  
        """

        num_sph_harm = len(self._sph_harm_lm)
        num_q_v_theta   = len(v_theta)
        num_q_v_phi     = len(v_phi)

        [Vt,Vp]      = np.meshgrid(v_theta, v_phi,indexing='ij')

        Vq = np.zeros((2 * num_sph_harm, num_q_v_theta,num_q_v_phi))
        for didx, d in enumerate(["+", "-"]):
            for lm_i, lm in enumerate(self._sph_harm_lm):
                Vq[didx * num_sph_harm + lm_i] = scale * self._hemi_sph_harm_real(lm[0], lm[1], Vt, Vp, d)
        
        return Vq

    def Vq_hsph(self, v_theta, v_phi, scale=1):
        """
        compute the basis Vandermonde for the all the basis function
        for the specified quadrature points.  
        """

        num_sph_harm = len(self._sph_harm_lm)
        assert v_theta.shape == v_phi.shape, "invalid shapes, use mesh grid to get matching shapes"
        _shape = tuple([2 * num_sph_harm]) + v_theta.shape
        Vq = np.zeros(_shape)

        for didx, d in enumerate(["+", "-"]):
            for lm_i, lm in enumerate(self._sph_harm_lm):
                Vq[didx * num_sph_harm + lm_i] = scale * self._hemi_sph_harm_real(lm[0], lm[1], v_theta, v_phi, d)
        
        return Vq

    def sph_ords_projections_ops(self, xp_vt, xp_vt_qw, mode):
        if (mode == "hsph"):
            Ps           = np.matmul(self.Vq_hsph(xp_vt, np.zeros_like(xp_vt)), np.diag(xp_vt_qw) ) * 2 * np.pi
            Po           = np.transpose(self.Vq_hsph(xp_vt, np.zeros_like(xp_vt))) 
        else:
            assert mode == "sph"
            Ps           = np.matmul(self.Vq_sph(xp_vt, np.zeros_like(xp_vt)), np.diag(xp_vt_qw) ) * 2 * np.pi
            Po           = np.transpose(self.Vq_sph(xp_vt, np.zeros_like(xp_vt))) 
        
        # to check if the oridinates to spherical and spherical to ordinates projection is true (weak test not sufficient but necessary condition)
        #assert np.allclose(np.dot(Po, np.dot(Ps, 1 + np.cos(xp_vt))), 1 + np.cos(xp_vt))
        return Po, Ps

    def gl_vr(self, Nvr, use_bspline_qgrid=False):
        if (use_bspline_qgrid):
            return self._basis_p.Gauss_Pn(Nvr * self._basis_p._num_knot_intervals)
        else:
            k_domain     = self._basis_p._domain
            gx, gw       = np.polynomial.legendre.leggauss(Nvr)
            gx           = 0.5 * (k_domain[1]-k_domain[0]) * gx + 0.5 * (k_domain[0] + k_domain[1])
            gw           = 0.5 * (k_domain[1]-k_domain[0]) * gw
            return gx, gw
        
    def gl_vt(self, Nvt, hspace_split=True, mode="npsp"):

        if hspace_split == True:
            # if (mode == "npsp1"):
            #     gx, gw             = basis.Legendre().Gauss_Pn(Nvt//2)
            #     gx_m1_0 , gw_m1_0  = 0.5 * gx - 0.5, 0.5 * gw
            #     gx_p1_0 , gw_p1_0  = 0.5 * gx + 0.5, 0.5 * gw
            #     xp_vt              = np.append(np.arccos(gx_m1_0), np.arccos(gx_p1_0)) 
            #     xp_vt_qw           = np.append(gw_m1_0, gw_p1_0)

            # el
            if (mode == "npsp"):
                # gx, gw               = basis.gauss_radau_quadrature(Nvt//2, fixed_point=-1)
                # gx_m1_0 , gw_m1_0    = 0.5 * gx - 0.5, 0.5 * gw
                # gx, gw               = -np.flip(gx), np.flip(gw)
                # gx_p1_0 , gw_p1_0    = 0.5 * gx + 0.5, 0.5 * gw
                
                # assert np.abs(gx_m1_0[0]  +1) < 1e-12
                # assert np.abs(gx_p1_0[-1] -1) < 1e-12

                # gx_m1_0[0]  = -1.0
                # gx_p1_0[-1] =  1.0

                # xp_vt                = np.append(np.arccos(gx_m1_0), np.arccos(gx_p1_0)) 
                # xp_vt_qw             = np.append(gw_m1_0, gw_p1_0)
                # print(xp_vt)
                
                # gx_p1_0 = np.flip(np.linspace(0, 0.5 * np.pi, Nvt//2, endpoint=False))
                # gx_m1_0 = np.flip(np.linspace(0.5 * np.pi, np.pi, Nvt//2))
                # xp_vt        = np.linspace(0, np.pi, Nvt)
                # xp_vt_qw     = np.ones_like(xp_vt)

                # xp_vt_qw[0]  = 0.5
                # xp_vt_qw[-1] = 0.5
                # xp_vt_qw     = (np.pi/(Nvt-1)) * xp_vt_qw * np.sin(xp_vt)

                # xp_vt        = np.flip(xp_vt)
                # xp_vt_qw     = np.flip(xp_vt_qw)

                
                gx, gw             = basis.Legendre().Gauss_Pn(Nvt//2)
                gx_m1_0 , gw_m1_0  = 0.5 * gx - 0.5, 0.5 * gw
                gx_p1_0 , gw_p1_0  = 0.5 * gx + 0.5, 0.5 * gw
                xp_vt              = np.append(np.arccos(gx_m1_0), np.arccos(gx_p1_0)) 
                xp_vt_qw           = np.append(gw_m1_0, gw_p1_0)
            elif (mode == "np"):
                gx, gw             = basis.Legendre().Gauss_Pn(Nvt)
                gx_p1_0 , gw_p1_0  = 0.5 * gx + 0.5, 0.5 * gw
                xp_vt              = np.arccos(gx_p1_0)
                xp_vt_qw           = gw_p1_0
            elif (mode == "sp"):
                gx, gw             = basis.Legendre().Gauss_Pn(Nvt)
                gx_m1_0 , gw_m1_0  = 0.5 * gx - 0.5, 0.5 * gw
                xp_vt              = np.arccos(gx_m1_0)
                xp_vt_qw           = gw_m1_0
            else:
                return NotImplementedError
        else:
            gx, gw             = basis.Legendre().Gauss_Pn(Nvt)
            xp_vt              = np.arccos(gx)
            xp_vt_qw           = gw
   
        return xp_vt, xp_vt_qw
    
    def gl_vp(self, Nvp):
        # gp, gwp = basis.Legendre().Gauss_Pn(Nvp)
        # xp_vp   = np.pi * gp + np.pi
        # xp_vp_qw= np.pi * gwp
        # return xp_vp, xp_vp_qw

        # peridoic trapizoidal rule
        xp_vp    = np.linspace(0, 2 * np.pi, Nvp, endpoint=False)
        xp_vp_qw = (2 * np.pi / Nvp) * np.ones_like(xp_vp)
        return xp_vp, xp_vp_qw

    def radial_to_vr_projection_ops(self, xp_vr, xp_vr_w):
        assert self.get_radial_basis_type() == basis.BasisType.SPLINES, "only spline basis is implemented for v-advection with ordinates"
        num_sh                   = len(self._sph_harm_lm)
        mm_r                     = self.compute_mass_matrix()[0::num_sh, 0::num_sh]

        def c_inv(mmat):
            L    = np.linalg.cholesky(mmat)
            Linv = scipy.linalg.solve_triangular(L, np.identity(mmat.shape[0]),lower=True) 
            #print("cholesky solver inverse : ", np.linalg.norm(np.matmul(L,Linv)-np.eye(L.shape[0]))/np.linalg.norm(np.eye(L.shape[0])))
            return np.matmul(np.transpose(Linv),Linv)

        mm_r                     = c_inv(mm_r)
        

        Vqr                      = self.Vq_r(xp_vr, l=0, scale=1)
        Pvr                      = Vqr.T # Splines to vr
        Pr                       = xp_vr ** 2 * xp_vr_w * (mm_r @ Vqr)

        return Pvr, Pr
        


