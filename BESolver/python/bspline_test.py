"""
@package bspline test for the radial direction. 
"""
import collisions
import utils as BEUtils
import numpy as np
import scipy.interpolate
import basis


class BSpline():
    def __init__(self,knots,spline_order, num_c_pts):
        assert len(knots) == num_c_pts + (spline_order+1) + 1, "knots vector length does not match the spline order"
        self._num_c_pts = num_c_pts
        self._sp_order = spline_order
        self._t        = knots
        self._splines = [scipy.interpolate.BSpline.basis_element(knots[i:i+spline_order+2],False) for i in range(num_c_pts)]

    def bspline_eval(self,i,x):
        return np.nan_to_num(self._splines[i](x))

    def Vx(self,x):
        return np.array([self.bspline_eval(i,x) for i in range(self._num_c_pts)])
    
    def splines_overlap(self, i, j):
        return np.abs(j-i) > self._p 
        

def assemble_mass_matrix(spline: BSpline,num_q,q_domain):
    
    
    #X2Q_TF = lambda x: (DQ/DX) * (x - x_domain[0])  + q_domain[0]
    
    
    NUM_Q_R   = num_q
    L=basis.Legendre()
    [glx,glw] = L.Gauss_Pn(NUM_Q_R)
    DQ = (q_domain[1]-q_domain[0])
    num_p = spline._num_c_pts
    
    sp_order = spline._sp_order
    mm = np.zeros((num_p,num_p))

    for i in range(num_p):
        for j in range(num_p):
            x_domain=[min(spline._t[i],spline._t[j]), max(spline._t[i+sp_order+1],spline._t[j+sp_order+1])]
            DX = (x_domain[1]-x_domain[0])
            Q2X_TF = lambda q: (DX/DQ) * (q - q_domain[0])  + x_domain[0]
            glx_domain = Q2X_TF(glx)
            
            if(np.abs(i-j)>sp_order):
                # just to check if we are skipping correct polynomials. 
                #print(np.allclose(spline.bspline_eval(i,glx_domain) * spline.bspline_eval(j,glx_domain),np.zeros_like(glx_domain)))
                continue
            
            mm[i,j] = np.dot( (DX/DQ) * spline.bspline_eval(i,glx_domain) * spline.bspline_eval(j,glx_domain), glw)
    return mm

def proj_func_to_splines(func, spline:BSpline, num_q, q_domain):
    NUM_Q_R   = num_q
    L=basis.Legendre()
    [glx,glw] = L.Gauss_Pn(NUM_Q_R)
    DQ = (q_domain[1]-q_domain[0])
    num_p = spline._num_c_pts
    
    sp_order = spline._sp_order
    c_vec    = np.zeros(num_p)
    for i in range(num_p):
        x_domain=[spline._t[i], spline._t[i+sp_order+1]]
        DX = (x_domain[1]-x_domain[0])
        Q2X_TF = lambda q: (DX/DQ) * (q - q_domain[0])  + x_domain[0]
        glx_domain = Q2X_TF(glx)
        #print(glx_domain)
        #print(func(glx_domain))
        c_vec[i]= np.dot( (DX/DQ) * spline.bspline_eval(i,glx_domain)*func(glx_domain), glw)
    
    return c_vec


T   = 1*collisions.TEMP_K_1EV
VTH = collisions.electron_thermal_velocity(T)
print("T = ",T," K")
print("V_th = ",VTH," ms-1")
Mv= lambda x: (VTH**3)*(1/(np.pi**(3/2) * VTH**3)) * np.exp(- (x/VTH)**2)

v_fac=4
vv = np.linspace(-v_fac * VTH, v_fac * VTH , 1000)



E_BOUND_EV = np.array([0,50])
#V_BOUNDS   = np.sqrt(2 * E_BOUND_EV * collisions.ELECTRON_VOLT/collisions.MASS_ELECTRON)
V_BOUNDS   = np.array([-(v_fac+1)*VTH,(v_fac+1)*VTH]) 
print("Selected energy   bounds : ", E_BOUND_EV," eV")
print("Corresponding velocity bounds : ", V_BOUNDS, " ms-1")


spline_order = 2
number_basis = 40 # control points. 
knots_vec    = np.linspace(V_BOUNDS[0],V_BOUNDS[1],spline_order + number_basis + 2)
splines      = BSpline(knots_vec,spline_order,number_basis)

NUM_Q_R   = 40
M1=assemble_mass_matrix(splines, NUM_Q_R,np.array([-1,1]))
#print("Q_PTS : ",NUM_Q_R)
#print(M1)

NUM_Q_R   = 2*NUM_Q_R
M2=assemble_mass_matrix(splines, NUM_Q_R,np.array([-1,1]))
#print("Q_PTS : ",NUM_Q_R)
#print(M2)
print("diff abs : ", np.linalg.norm(M1-M2))
print("diff |(M1-M2)|/|M1| : ", np.linalg.norm(M1-M2)/np.linalg.norm(M2))


c_vec = proj_func_to_splines(Mv,splines,NUM_Q_R,np.array([-1,1]))
c_vec = np.dot(np.linalg.inv(M2),c_vec)
print(c_vec)

sV = splines.Vx(vv).transpose()

import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 3)
axs[0].plot(vv,Mv(vv),label="Maxwellian")
axs[0].plot(vv,np.dot(sV,c_vec),label="B-Spline fit")
axs[0].set_xlabel('v')
axs[0].set_ylabel('f(v)')
axs[0].legend()

axs[1].plot(vv,np.dot(sV,c_vec)-Mv(vv),label="error abs.")
axs[1].set_xlabel('v')
axs[1].set_ylabel('Error')
axs[1].legend()

for i in range(0,10):
    spline_xx=np.linspace(knots_vec[i],knots_vec[i+spline_order+2],100)  
    axs[2].plot(spline_xx,splines.bspline_eval(i,spline_xx),label="B_%d"%i)

axs[2].set_xlabel('v')
axs[2].set_ylabel('B(v)')
axs[2].legend()

#plt.plot(vv,Mv(vv))
plt.tight_layout()
plt.show()
plt.close()





    