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
import utils as BEutils
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

def create_xlbspline_spec(spline_order, k_domain, Nr, sph_harm_lm):
    splines     = basis.XlBSpline(k_domain,spline_order,Nr+1)
    #sph_harm_lm   = [(l,m) for l in range(lmax+1) for m in range(-l,l+1)]
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
            for qs_idx,qs in enumerate(spec._sph_harm_lm):
                for lm_idx,lm in enumerate([(max(qs[0]-1,0), qs[1]), (min(qs[0]+1,l_max), qs[1])]):#enumerate(spec._sph_harm_lm):
                    qs_mat = qs[0]**2+qs[0]+qs[1]
                    lm_mat = lm[0]**2+lm[0]+lm[1]
                    pqs = p*num_sh + qs_idx
                    klm = k*num_sh + spec._sph_harm_lm.index(lm)
                    advec_mat[pqs,klm] = mm1[p,k,qs[0],lm[0]] * psimat[qs_mat,lm_mat] - mm2[p,k,qs[0],lm[0]] * phimat[qs_mat,lm_mat]

    print("norm adv mat = %.8E"%np.linalg.norm(advec_mat))
    return advec_mat

    
def backward_euler(FOp,y0,t_end,nsteps):
    dt = t_end/nsteps
    A  = np.linalg.inv(np.eye(FOp.shape[0])+ dt* FOp)
    #A  = np.linalg.matrix_power(A,nsteps)
    for i in range(nsteps):
        y0=np.matmul(A,y0)

    return y0


def solve_advection(nr, sph_lm, sp_order, v_doamin,t_end=5e-1):

    NR            = nr
    L_MODE_MAX    = sph_lm[-1][0]
    V_DOMAIN      = v_doamin
    VTH           = 1.0
    spline_order  = sp_order
    basis.XLBSPLINE_NUM_Q_PTS_PER_KNOT = 11 #2*spline_order+1
    basis.BSPLINE_BASIS_ORDER=spline_order

    num_p = NR+1
    num_sph=len(sph_lm)

    params.BEVelocitySpace.VELOCITY_SPACE_POLY_ORDER = NR
    params.BEVelocitySpace.SPH_HARM_LM = sph_lm


    NUM_Q_VR = basis.XlBSpline.get_num_q_pts(NR,spline_order,basis.XLBSPLINE_NUM_Q_PTS_PER_KNOT)
    NUM_Q_VT = 128
    NUM_Q_VP = 2

    params.BEVelocitySpace.NUM_Q_VR  = NUM_Q_VR
    params.BEVelocitySpace.NUM_Q_VT  = NUM_Q_VT
    params.BEVelocitySpace.NUM_Q_VP  = NUM_Q_VP
    params.BEVelocitySpace.NUM_Q_CHI = 2
    params.BEVelocitySpace.NUM_Q_PHI = 2

    spec_xlbspline = create_xlbspline_spec(spline_order,V_DOMAIN,NR,sph_lm)
    M    = VTH * spec_xlbspline.compute_mass_matrix()
    Minv = np.linalg.inv(M)
    print(spec_xlbspline._sph_harm_lm)
    print("mass mat condition number = %.2E"%np.linalg.cond(M))
    print("|I-M^{-1} M| = %.10E " %np.linalg.norm(np.matmul(Minv,M)-np.eye(M.shape[0])))
    #print(M[0,:])


    hv         = lambda v,vt,vp : np.ones_like(v)
    h_vec      = expand_in_xlsplines(spec_xlbspline,hv,NUM_Q_VR,NUM_Q_VT,NUM_Q_VP,mass_inverse=Minv)

    coeffs_new=h_vec
    coeffs = h_vec


    L=assemble_advection_matrix_lp(spec_xlbspline)
    advmat=np.matmul(Minv,L)

    func = lambda t,a: -np.matmul(advmat,a)
    sol = scipy.integrate.solve_ivp(func, (0,t_end), coeffs, max_step=dt, method='RK45',atol=1e-12, rtol=1e-12)
    # sol = scipy.integrate.solve_ivp(func, (0,t_end), coeffs, max_step=dt, method='BDF')
    coeffs_new = sol.y[:,-1]

    
    #coeffs_new = backward_euler(advmat,h_vec,t_end,1000)
    # func   = lambda t,a: -np.matmul(advmat,a)
    # sol = scipy.integrate.solve_ivp(func, (0,t_end), h_vec,method='BDF',atol=1e-14,rtol=1e-14)
    # print(sol)
    # coeffs_new=sol.y[:,-1]

    return coeffs,coeffs_new,spec_xlbspline


nr_max=32
num_dofs_all = [(nr_max,l) for l in range(2,6)]
num_dofs_all = [(1<<l,5) for l in range(4,7)]
num_dofs_all = [(16,10), (32,10), (64,10)]
num_dofs_all = [(16,2), (32,2), (64,2)]
num_dofs_all = [(64,1), (64,2), (64,3), (64,4), (64,5), (48,5), (32,5), (16,5)]
mid_idx = 4
#num_dofs_all = [(16,0), (32,0)]
#num_dofs_all = [16]

error_linf = np.zeros(len(num_dofs_all))
error_l2 = np.zeros(len(num_dofs_all))
error_linf_2d = np.zeros(len(num_dofs_all))
error_l2_2d = np.zeros(len(num_dofs_all))

t_end = 0.2
# nsteps = 400000
nsteps = 10000
dt = t_end/nsteps

x = np.linspace(-2,2,100)
z = np.linspace(-2,2,100)
quad_grid = np.meshgrid(x,z,indexing='ij')

y = np.zeros_like(quad_grid[0])

sph_coord_init = BEutils.cartesian_to_spherical(quad_grid[0],y,quad_grid[1])
sph_coord_end  = BEutils.cartesian_to_spherical(quad_grid[0],y,quad_grid[1]-t_end)

theta = 0.5*np.pi - np.sign(x)*0.5*np.pi
f_num = np.zeros([len(num_dofs_all), len(x)])
f_initial = np.zeros([len(num_dofs_all), len(x)])
f_exact = np.zeros([len(num_dofs_all), len(x)])

f_num_2d = np.zeros([len(num_dofs_all), len(x), len(z)])
f_initial_2d = np.zeros([len(num_dofs_all), len(x), len(z)])
f_exact_2d = np.zeros([len(num_dofs_all), len(x), len(z)])

print(np.shape(sph_coord_init))
V_DOMAIN = (0,40)
SP_ORDER = 3
for num_dofs_idx, num_dofs in enumerate(num_dofs_all):
    nr = num_dofs[0]
    l_max = num_dofs[1]
    
    num_p   = nr+1
    sph_lm  = [(l,0) for l in range(l_max+1)]
    num_sph = len(sph_lm)
    print("Nr=%d sph=%s"%(nr,sph_lm))
    c,ct,spec_xlbspline = solve_advection(nr,sph_lm,SP_ORDER,V_DOMAIN,t_end)

    Vq_r_2d   = spec_xlbspline.Vq_r(sph_coord_init[0])
    Vq_rt_2d  = spec_xlbspline.Vq_r(sph_coord_end[0])

    Vq_r    = spec_xlbspline.Vq_r(np.abs(x))
    Vq_rt  = spec_xlbspline.Vq_r(np.abs(x-t_end))

    f_eval_mat_2d = np.transpose(np.array([spec_xlbspline.basis_eval_spherical(sph_coord_init[1],sph_coord_init[2],lm[0],lm[1]) * Vq_r_2d[lm[0],k,:]   for k in range(num_p) for lm_idx, lm in enumerate(sph_lm)]).reshape(num_p*num_sph,-1))

    f_eval_mat_2d_t = np.transpose(np.array([spec_xlbspline.basis_eval_spherical(sph_coord_init[1],sph_coord_init[2],lm[0],lm[1]) * Vq_rt_2d[lm[0],k,:]   for k in range(num_p) for lm_idx, lm in enumerate(sph_lm)]).reshape(num_p*num_sph,-1))
    
    f_eval_mat=np.transpose(np.array([spec_xlbspline.basis_eval_spherical(theta,0,lm[0],lm[1]) * Vq_r[lm[0],k,:]   for k in range(num_p) for lm_idx, lm in enumerate(sph_lm)]).reshape(num_p*num_sph,-1))
    f_eval_mat_t=np.transpose(np.array([spec_xlbspline.basis_eval_spherical(theta,0,lm[0],lm[1]) * Vq_rt[lm[0],k,:]   for k in range(num_p) for lm_idx, lm in enumerate(sph_lm)]).reshape(num_p*num_sph,-1))
    
    f_num[num_dofs_idx,:]     = np.dot(f_eval_mat,ct)
    f_initial[num_dofs_idx,:] = np.dot(f_eval_mat,c)
    f_exact[num_dofs_idx,:]   = np.dot(f_eval_mat_t,c)

    f_num_2d[num_dofs_idx,:]     = np.dot(f_eval_mat_2d,ct).reshape((len(x),len(z)))
    f_initial_2d[num_dofs_idx,:] = np.dot(f_eval_mat_2d,c).reshape((len(x),len(z)))
    f_exact_2d[num_dofs_idx,:]   = np.dot(f_eval_mat_2d_t,c).reshape((len(x),len(z)))
    
    error_linf[num_dofs_idx] = np.max(abs(f_num[num_dofs_idx,:]-f_exact[0,:]))
    error_l2[num_dofs_idx] = np.linalg.norm(f_num[num_dofs_idx,:]-f_exact[0,:])

    error_linf_2d[num_dofs_idx] = np.max(abs(f_num_2d[num_dofs_idx,:]-f_exact_2d[0,:]))
    error_l2_2d[num_dofs_idx] = np.linalg.norm(f_num_2d[num_dofs_idx,:]-f_exact_2d[0,:])

# plt.subplot(2,3,1)
# plt.plot(x, f_initial[0,:])
# plt.plot(x, f_exact[0,:])

# for num_dofs_idx,Nr in enumerate(num_dofs_all):
#     plt.plot(x, f_num[num_dofs_idx,:], '--')

# plt.grid()
# plt.legend(['Initial Conditions', 'Exact', 'Numerical'])
# # plt.legend(['Initial Conditions', 'Exact', '$N_r = 32, l_{max} = 2$', '$N_r = 32, l_{max} = 4$', '$N_r = 32, l_{max} = 8$', '$N_r = 32, l_{max} = 16$'])
# # plt.legend(['Initial Conditions', 'Exact', '$N_r = 8, l_{max} = 8$', '$N_r = 16, l_{max} = 8$', '$N_r = 32, l_{max} = 8$', '$N_r = 64, l_{max} = 8$'])
# plt.ylabel('Distribution function')
# plt.xlabel('$v_z$')

plt.subplot(1,4,1)
plt.semilogy(x, f_initial[mid_idx,:])
plt.plot(x, f_exact[mid_idx,:])

# for num_dofs_idx,Nr in enumerate(num_dofs_all):
    # plt.plot(x, f_num[num_dofs_idx,:], '--')
plt.plot(x, f_num[mid_idx,:], '--')

plt.grid()
plt.legend(['Initial', 'Exact', 'B-splines'])
# plt.legend(['Initial Conditions', 'Exact', '$N_r = 32, l_{max} = 2$', '$N_r = 32, l_{max} = 4$', '$N_r = 32, l_{max} = 8$', '$N_r = 32, l_{max} = 16$'])
# plt.legend(['Initial Conditions', 'Exact', '$N_r = 8, l_{max} = 8$', '$N_r = 16, l_{max} = 8$', '$N_r = 32, l_{max} = 8$', '$N_r = 64, l_{max} = 8$'])
plt.ylabel('Distribution function')
plt.xlabel('$v_z$')

plt.subplot(1,4,2)
plt.contour(quad_grid[0], quad_grid[1], f_initial_2d[mid_idx,:,:], linestyles='solid', colors='grey', linewidths=1)
plt.contour(quad_grid[0], quad_grid[1], f_exact_2d[mid_idx,:,:], linestyles='dashed', colors='red', linewidths=2)
ax = plt.contour(quad_grid[0], quad_grid[1], f_num_2d[mid_idx,:,:], linestyles='dotted', colors='blue', linewidths=2)
# ax.plot_surface(quad_grid[0], quad_grid[1], f_initial2[2,:,:], rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none')
# ax.set_title('surface');
plt.gca().set_aspect('equal')

# plt.subplot(1,3,4)

# for num_dofs_idx,Nr in enumerate(num_dofs_all):
#     plt.semilogy(x, abs(f_num[num_dofs_idx,:]-f_exact[0,:]), '--')

# plt.grid()
# # plt.legend(['Initial Conditions', 'Exact', 'Numerical'])
# # plt.legend(['$N_r = 32, l_{max} = 2$', '$N_r = 32, l_{max} = 4$', '$N_r = 32, l_{max} = 8$', '$N_r = 32, l_{max} = 16$'])
# # plt.legend(['Initial Conditions', 'Exact', '$N_r = 8, l_{max} = 8$', '$N_r = 16, l_{max} = 8$', '$N_r = 32, l_{max} = 8$', '$N_r = 64, l_{max} = 8$'])
# plt.ylabel('Error in distribution function')
# plt.xlabel('$v_z$')

plt.subplot(1,4,3)
plt.semilogy([num_dofs[0] for num_dofs in num_dofs_all[len(num_dofs_all)-1:mid_idx-1:-1]], error_linf_2d[len(num_dofs_all)-1:mid_idx-1:-1], '-o')
# plt.loglog([num_dofs[0] for num_dofs in num_dofs_all[5:10]], error_l2_2d[5:10], '-*')
plt.ylabel('Error')
plt.xlabel('$N_r$')
plt.grid()
# plt.legend(['$L_\inf$', '$L_2$'])

plt.subplot(1,4,4)
plt.semilogy([num_dofs[1] for num_dofs in num_dofs_all[0:mid_idx+1]], error_linf_2d[0:mid_idx+1], '-o')
# plt.loglog([num_dofs[0] for num_dofs in num_dofs_all[5:10]], error_l2_2d[5:10], '-*')
plt.ylabel('Error')
plt.xlabel('$l_{\max}$')
plt.grid()
# plt.legend(['$L_\inf$', '$L_2$'])




# fig = plt.gcf()
# fig.set_size_inches(16, 8)
# fig.savefig("nr_%d_to_%d_lmax_%d_to_%d.png"%(num_dofs_all[0][0],num_dofs_all[-1][0],num_dofs_all[0][1],num_dofs_all[-1][1]), dpi=100)

#plt.savefig()
plt.show()

