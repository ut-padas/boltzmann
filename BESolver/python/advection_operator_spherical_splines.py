import enum
from random import randrange
import basis
import spec_spherical as sp
import scipy.integrate
import numpy as np
import maxpoly
import matplotlib.pyplot as plt
import profiler
from scipy.special import sph_harm
import utils as BEUtils
import parameters as params

t_adv_mat = profiler.profile_t("v_adv")
phimat = np.genfromtxt('sph_harm_del/phimat.dat',delimiter=',')
psimat = np.genfromtxt('sph_harm_del/psimat.dat',delimiter=',')

class SPEC_SPH:
    def __init__(self, p_order, basis_p, sph_harm_lm, domain=None, window=None):
            
        """
        @param p_order : number of basis functions used in radial direction
        @param basis_p : np.polynomial object
        @param sph_harm_lm : (l,m) indices of spherical harmonics to use
        """
        self._sph_harm_lm = sph_harm_lm
        self._lmodes = list(set([lm[0] for lm in self._sph_harm_lm]))
        self._p = p_order
        self._domain = domain
        self._window = window

        self._basis_p  = basis_p
        self._basis_1d = list()
        
        print(self._sph_harm_lm)
        print(self._lmodes)
        
        for deg in range(self._p+1):
            self._basis_1d.append(self._basis_p.Pn(deg,self._domain,self._window))

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
    
    def basis_eval_radial(self,r,k,l=0):
        """
        Evaluates 
        """
        return np.nan_to_num(self._basis_1d[k](l,r))

    def basis_derivative_eval_radial(self,dorder,r,k,l=0):
        """
        Evaluates 
        """
        return np.nan_to_num(self._basis_p.diff(k,dorder)(l,r))
    
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

    def Vq_r(self, v_r, scale=1):
        """
        compute the basis Vandermonde for the all the basis function
        for the specified quadrature points.  
        """
        num_p        = self._p+1
        num_l        = len(self._lmodes)

        _shape = tuple([num_l,num_p]) + v_r.shape
        Vq = np.zeros(_shape)
        for i,l in enumerate(self._lmodes):
            for j in range(num_p):
                Vq[i,j] = scale * self.basis_eval_radial(v_r,j,l)
            
            # print("l=%d"%l)
            # print(Vq[i,:])
        return Vq

    def Vdq_r(self,v_r,scale=1):
        """
        evaluate the radial derivative of the basis at v_r pts. 
        """

        num_p        = self._p+1
        num_l        = len(self._lmodes)

        _shape = tuple([num_l,num_p]) + v_r.shape
        Vq = np.zeros(_shape)

        for i,l in enumerate(self._lmodes):
            for j in range(num_p):
                Vq[i,j] = scale * self.basis_derivative_eval_radial(1,v_r,j,l)
            
            # print("l=%d"%l)
            # print(Vq[i,:])
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

    def compute_mass_matrix(self):
        """
        Compute the mass matrix w.r.t the basis polynomials
        if the chosen basis is orthogonal set is_diagonal to True. 
        Given we always use spherical harmonics we will integrate them exactly
        """
        num_p  = self._p+1
        num_sh = len(self._sph_harm_lm)
        num_l  = len(self._lmodes)
        
        print("q: ", basis.XlBSpline.get_num_q_pts(self._p,self._basis_p._sp_order,self._basis_p._q_per_knot))
        [gx, gw] = self._basis_p.Gauss_Pn(basis.XlBSpline.get_num_q_pts(self._p,self._basis_p._sp_order,self._basis_p._q_per_knot),True)
        # note that, VTH**3 should be here but discarded, since they cancel out at C operator. 
        Vr = self.Vq_r(gx)
        mr = gx**2 
        
        mm = np.array([ mr * Vr[l,i,:] * Vr[l,j,:] for l in range(num_l) for i in range(num_p) for j in range(num_p)]).reshape(tuple([num_l,num_p,num_p])+gx.shape)
        mm = np.dot(mm,gw)
        mm_full = np.zeros((num_p*num_sh,num_p*num_sh))
        
        for p in range(num_p):
            for k in range(num_p):
                for lm_idx, lm in enumerate(self._sph_harm_lm):
                    mm_full[p*num_sh + lm_idx,k*num_sh + lm_idx] = mm[lm[0],p,k]
        
        return mm_full

def create_xlbspline_spec(spline_order, k_domain, Nr, lmax):
    splines     = basis.XlBSpline(k_domain,spline_order,Nr+1)
    sph_harm_lm   = [(l,m) for l in range(lmax+1) for m in range(-l,l+1)]
    spec        = SPEC_SPH(Nr,splines,sph_harm_lm)
    return spec

# def create_bspline_spec(spline_order, k_domain, Nr, lmax):
#     num_k         = spline_order + (Nr+1) + 2
#     knots_vec1    = np.zeros(spline_order+1)
#     knots_vec2    = np.logspace(1e-2, np.log2(k_domain[1]) , num_k-2*spline_order-2,base=2)
#     #knots_vec2  = np.linspace(1e-2, k_domain[1] , num_k-2*spline_order-2)
#     knots_vec     = np.append(knots_vec1,knots_vec2)
#     knots_vec     = np.append(knots_vec,knots_vec[-1]*np.ones(spline_order+1))
#     splines       = basis.BSpline(knots_vec,spline_order,Nr+1)
#     sph_harm_lm   = [(l,m) for l in range(lmax+1) for m in range(-l,l+1)]
#     spec          = sp.SpectralExpansionSpherical(Nr,splines,sph_harm_lm)
#     return spec

def expand_in_xlsplines(spec_sp,f,qr,qt,qp,mass_inverse=None):

    num_p        = spec_sp._p +1
    sph_harm_lm  = spec_sp._sph_harm_lm
    num_sph_harm = len(sph_harm_lm)
    
    legendre     = basis.Legendre()
    [glx,glw]    = legendre.Gauss_Pn(qt)
    VTheta_q     = np.arccos(glx)
    VPhi_q       = np.linspace(0,2*np.pi,qp)

    assert qp>1
    sq_fac_v = (2*np.pi/(qp-1))
    WVPhi_q  = np.ones(qp)*sq_fac_v

    #trap. weights
    WVPhi_q[0]  = 0.5 * WVPhi_q[0]
    WVPhi_q[-1] = 0.5 * WVPhi_q[-1]
    [gmx,gmw]    = spec_sp._basis_p.Gauss_Pn(qr,True)
    weight_func  = spec_sp._basis_p.Wx()

    
    quad_grid = np.meshgrid(gmx,VTheta_q,VPhi_q,indexing='ij')
    P_kr = spec_sp.Vq_r(quad_grid[0]) 
    Y_lm = spec_sp.Vq_sph(quad_grid[1],quad_grid[2])

    hq   = f(quad_grid[0],quad_grid[1],quad_grid[2]) * np.exp(-quad_grid[0]**2)
    if mass_inverse is None:
        mass_inverse   = np.linalg.inv(spec_sp.compute_mass_matrix())
    
    MP_klm = np.array([hq * (quad_grid[0]**2) * P_kr[lm[0],i]*Y_lm[j] for i in range(num_p) for j,lm in enumerate(sph_harm_lm)])
    MP_klm = np.dot(MP_klm,WVPhi_q)
    MP_klm = np.dot(MP_klm,glw)
    MP_klm = np.dot(MP_klm,gmw)

    MP_klm = np.matmul(mass_inverse,MP_klm)
    return MP_klm


def assemble_advection_matrix_lp(spec: SPEC_SPH):

    num_p  = spec._p+1
    num_sh = len(spec._sph_harm_lm)
    num_l  = len(spec._lmodes)

    l_max = spec._lmodes[-1]
    phimat = np.genfromtxt('sph_harm_del/phimat.dat',delimiter=',')
    psimat = np.genfromtxt('sph_harm_del/psimat.dat',delimiter=',')

    psimat= np.transpose(psimat[0:(l_max+1)**2, 0:(l_max+1)**2])
    phimat= np.transpose(phimat[0:(l_max+1)**2, 0:(l_max+1)**2])

    [gx, gw] = spec._basis_p.Gauss_Pn(basis.XlBSpline.get_num_q_pts(spec._p,spec._basis_p._sp_order,spec._basis_p._q_per_knot),True)
    Vr  = spec.Vq_r(gx)
    Vdr = spec.Vdq_r(gx)
    mr = gx**2 
    mm1 = np.array([mr * Vr[pl,p,:] * Vdr[kl,k,:] for p in range(num_p) for k in range(num_p) for pl in range(num_l) for kl in range(num_l) ]).reshape(num_p,num_p,num_l,num_l,-1)
    mm1 = np.dot(mm1,gw)
    
    mr = gx
    mm2 = np.array([mr * Vr[pl,p,:] * Vr[kl,k,:] for p in range(num_p) for k in range(num_p) for pl in range(num_l) for kl in range(num_l) ]).reshape(num_p,num_p,num_l,num_l,-1)
    mm2 = np.dot(mm2,gw)

    advec_mat = np.zeros((num_p*num_sh,num_p*num_sh))
    for p in range(num_p):
        for k in range(num_p):
            for lm_idx,lm in enumerate(spec._sph_harm_lm):
                for qs_idx,qs in enumerate(spec._sph_harm_lm):
                    qs_mat = qs[0]**2+qs[0]+qs[1]
                    lm_mat = lm[0]**2+lm[0]+lm[1]
                    pqs = p*num_sh + qs_idx
                    klm = k*num_sh + lm_idx
                    advec_mat[pqs,klm] = mm1[p,k,qs[0],lm[0]] * psimat[qs_mat,lm_mat] - mm2[p,k,qs[0],lm[0]] * phimat[qs_mat,lm_mat]
    
    return advec_mat

    
def backward_euler(FOp,y0,t_end,nsteps):
    dt = t_end/nsteps
    A  = np.linalg.inv(np.eye(FOp.shape[0])+ dt* FOp)
    #A  = np.linalg.matrix_power(A,nsteps)
    for i in range(nsteps):
        y0=np.matmul(A,y0)

    return y0


NR=64
L_MODE_MAX=3
V_DOMAIN = (0,40)
VTH=1.0
spline_order  = 3
num_p = NR+1
lm_all = [(l,m) for l in range(L_MODE_MAX+1) for m in range(-l, l+1)]
num_sph=len(lm_all)
t_end  = 5e-1

basis.XLBSPLINE_NUM_Q_PTS_PER_KNOT = 2*spline_order+1
basis.BSPLINE_BASIS_ORDER=spline_order

params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER = NR
params.BEVelocitySpace.SPH_HARM_LM = [(i,j) for i in range(L_MODE_MAX+1) for j in range(-i,i+1)]

NUM_Q_VR = basis.XlBSpline.get_num_q_pts(NR,spline_order,basis.XLBSPLINE_NUM_Q_PTS_PER_KNOT)
NUM_Q_VT = 32
NUM_Q_VP = 8

params.BEVelocitySpace.NUM_Q_VR  = NUM_Q_VR
params.BEVelocitySpace.NUM_Q_VT  = NUM_Q_VT
params.BEVelocitySpace.NUM_Q_VP  = NUM_Q_VP
params.BEVelocitySpace.NUM_Q_CHI = 2
params.BEVelocitySpace.NUM_Q_PHI = 2

spec_xlbspline = create_xlbspline_spec(spline_order,V_DOMAIN,NR,L_MODE_MAX)
M    = VTH * spec_xlbspline.compute_mass_matrix()
Minv = np.linalg.inv(M)
print("%.2E"%np.linalg.cond(M))
print("|I-M^{-1} M| = %.10E " %np.linalg.norm(np.matmul(Minv,M)-np.eye(M.shape[0])))

hv         = lambda v,vt,vp : np.ones_like(v)
h_vec      = expand_in_xlsplines(spec_xlbspline,hv,NUM_Q_VR,NUM_Q_VT,NUM_Q_VP,mass_inverse=Minv)
  
coeffs_new=h_vec
coeffs = h_vec


L=assemble_advection_matrix_lp(spec_xlbspline)
advmat=np.matmul(Minv,L)
coeffs_new = backward_euler(advmat,h_vec,t_end,1000)
# func   = lambda t,a: -np.matmul(advmat,a)
# sol = scipy.integrate.solve_ivp(func, (0,t_end), h_vec,method='BDF',atol=1e-14,rtol=1e-14)
# print(sol)
# coeffs_new=sol.y[:,-1]







x = np.linspace(0,20,1000)
Vq_r   = spec_xlbspline.Vq_r(np.abs(x))
Vq_rt  = spec_xlbspline.Vq_r(np.abs(x-t_end))
f_eval_mat=np.transpose(np.array([spec_xlbspline.basis_eval_spherical(0,0,lm[0],lm[1]) * Vq_r[lm[0],k,:]   for k in range(num_p) for lm_idx, lm in enumerate(lm_all)]).reshape(num_p*num_sph,-1))
f_eval_mat_t=np.transpose(np.array([spec_xlbspline.basis_eval_spherical(0,0,lm[0],lm[1]) * Vq_rt[lm[0],k,:]   for k in range(num_p) for lm_idx, lm in enumerate(lm_all)]).reshape(num_p*num_sph,-1))

# Vq_r   = spec_bspline.Vq_r(x)
# Vq_rt  = spec_bspline.Vq_r(np.abs(x-t_end))
# f_eval_mat=np.transpose(np.array([spec_bspline.basis_eval_spherical(0,0,lm[0],lm[1]) * Vq_r[k,:]   for k in range(num_p) for lm_idx, lm in enumerate(lm_all)]).reshape(num_p*num_sph,-1))
# f_eval_mat_t=np.transpose(np.array([spec_bspline.basis_eval_spherical(0,0,lm[0],lm[1]) * Vq_rt[k,:]   for k in range(num_p) for lm_idx, lm in enumerate(lm_all)]).reshape(num_p*num_sph,-1))

f    = np.dot(f_eval_mat,coeffs_new)
f_in = np.dot(f_eval_mat,coeffs)
f_ex = np.dot(f_eval_mat_t,coeffs)

# f = np.zeros(np.shape(x))
# f_in = np.zeros(np.shape(x))
# f_ex = np.zeros(np.shape(x))
# for k in range(num_p):
#     for lm_idx, lm in enumerate(lm_all):
#         f    += coeffs_new[k*num_sph+lm_idx]* Vq_sph[lm_idx] * spec_xlbspline.basis_eval_radial(x,k,lm[0]) 
#         f_in += coeffs[k*num_sph+lm_idx]*spec_xlbspline.basis_eval_spherical(0,0, lm[0], lm[1]) * spec_xlbspline.basis_eval_radial(x,k,lm[0]) 
#         f_ex += coeffs[k*num_sph+lm_idx]*spec_xlbspline.basis_eval_spherical(0,0, lm[0], lm[1]) * spec_xlbspline.basis_eval_radial(x-t_end,k,lm[0])

plt.plot(x,f_in)
plt.plot(x,f_ex)
plt.plot(x,f)
# plt.plot(x,f-f_ex)
#plt.yscale('log')
plt.grid()
plt.legend(['Initial Conditions', 'Exact', 'Nr=%d, l_max=%d'%(NR,L_MODE_MAX)])
plt.show()


# import matplotlib.pyplot as plt
# x=np.linspace(1e-6,2,1000)
# for k in [0,1,2]:
#     for l in [0,1,2]:
#         fx=spec.basis_derivative_eval_radial(1,x,k,l)
#         plt.plot(x,fx,label="l=%d,nr=%d"%(l,k))
# plt.legend()
# plt.show()