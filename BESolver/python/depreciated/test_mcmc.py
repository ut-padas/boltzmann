"""
@brief Test the MCMC sampling code. 
"""
import utils as bte_utils
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.interpolate
import collisions

sns.set_theme(style="whitegrid")

def prior_samples(size):
    mu  = 0
    std = 1
    return np.random.uniform(-5, 5, size)#np.random.normal(mu, std,size=size)

normal_pdf = lambda x, mu, sigma : (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x-mu)/sigma)**2 )
    
def target_pdf(x):
    mu   = 0.5
    std  = 1
    return normal_pdf(x, mu, std) 

N  = np.int64(1e6)
x  = np.linspace(-5, 5, 1000)
y  = prior_samples(N)
yp = bte_utils.mcmc_sampling(target_pdf, prior_samples, n_samples=N, burn_in=0.3)

plt.hist(y, bins=50, density=True, color="blue",label="prior")
#plt.plot(x, normal_pdf(x,0,1),label="prior")
plt.grid(visible=True)

plt.hist(yp, bins=50, density=True, color="red")
plt.plot(x,  target_pdf(x),label="target",color="red")
plt.legend()
plt.show()


mw_tmp  = 1.0 * collisions.TEMP_K_1EV
vth     = collisions.electron_thermal_velocity(mw_tmp)
c_gamma = np.sqrt(2 * collisions.ELECTRON_CHARGE_MASS_RATIO)


prior_mu  = np.zeros(3)
std       = 1 / np.sqrt(2)

prior_cov = np.zeros((3,3))
prior_cov[0,0] = (std)**2
prior_cov[1,1] = (std)**2
prior_cov[2,2] = (std)**2

def gaussian_prior_samples(size):
    np.random.seed()
    return np.random.multivariate_normal(prior_mu, prior_cov, size=size)

def uniform_prior_samples(size):
    np.random.seed()
    return np.random.uniform([-5, -5, -5], [5, 5, 5], size=(size, 3))

prior_samples = uniform_prior_samples

N     = int(1e6)
x_pr  = prior_samples(size=N)

def compute_mean_and_std(x_pr):
    s_mu  = np.mean(x_pr,axis=0)
    dd    = x_pr.shape[1]
    s_cov = np.zeros((dd,dd))
    for i in range(dd):
        for j in range(i,dd):
            s_cov[i,j] = np.mean((x_pr[:,i]-s_mu[i]) * (x_pr[:,j]-s_mu[j]))
            s_cov[j,i] = s_cov[i,j]

    return s_mu, s_cov


# print("==========================")
# print("       PRIOR STATS        ")
# print("==========================")

# np.set_printoptions(precision=4)
# print("population mean and covariance")
# print(prior_mu)
# print(prior_cov)
# print("sample mean and covariance")
# print("sample size = %.1E"%(N))
# s_mu, s_cov = compute_mean_and_std(x_pr)
# print(s_mu)
# print(s_cov)

# print("==========================")
# print("                          ")
# print("==========================")

target_mu  = np.zeros(3)
target_cov = np.zeros((3,3))
std        = 1.5/ np.sqrt(2)
target_cov[0,0] = 1 * (std)**2
# target_cov[0,1] = (std)**2
target_cov[0,2] = 0.1 * (std)**2
target_cov[2,0] = 0.1 * (std)**2

target_cov[1,1] = 2 * (std)**2
target_cov[2,2] = 3 * (std)**2

target_cov_inv = np.linalg.inv(target_cov)

def target_pdf(x):
    w = np.array([-0.5 * np.dot((x[i]-target_mu), np.dot(target_cov_inv, (x[i]-target_mu))) for i in range(x.shape[0])])
    return np.exp(w) * (2*np.pi) ** (-3/2) * np.linalg.det(target_cov)**(-0.5)

y_post = bte_utils.mcmc_sampling(target_pdf, prior_samples, N, burn_in=1, num_chains=1)
s_mu, s_cov = compute_mean_and_std(y_post)

print("==========================")
print("       TARGET STATS        ")
print("==========================")
np.set_printoptions(precision=4)
print("population mean and covariance")
print(target_mu)
print(target_cov)
print("==========================")
print("sample mean and covariance")
print("sample size = %.1E"%(N))
s_mu, s_cov = compute_mean_and_std(y_post)
print(s_mu)
print(s_cov)

print("==========================")
print("                          ")
print("==========================")

# def target_pdf(x):
#     mu     = 0
#     std    = vth / np.sqrt(2)
#     vr     = np.linalg.norm(x,axis=1)
#     cos_vt = x[:,2] / vr
#     fac    = 1.0 / (np.sqrt(2 * np.pi) * std) **3 
#     ul     = lambda l : np.sqrt((2 * l + 1) / (4 * np.pi))
#     return np.exp(-(vr / (np.sqrt(2) * std) )**2) * fac * (np.polynomial.Legendre.basis(0)(cos_vt) * ul(0) + np.polynomial.Legendre.basis(1)(cos_vt) * ul(1)  + np.polynomial.Legendre.basis(2)(cos_vt) * ul(2))


# def mcmc_lm(pts, lm, x_domain, num_bins, vth):
    
#     x_pts  = pts / vth
#     xr     = np.linalg.norm(x_pts, axis=1) 
#     x_grid = np.linspace(x_domain[0], x_domain[1], num_bins)
#     dx     = (x_domain[1] - x_domain[0])/(num_bins-1)

#     idx    = np.int64(xr/dx)
#     f_lm   = np.zeros(num_bins-1)

#     assert lm[1]==0

#     cos_vt = ( x_pts[:,2] / xr)
#     pl_x   = np.polynomial.Legendre.basis(lm[0])(cos_vt) * np.sqrt((2 * lm[0] + 1) / (4 * np.pi)) * 2 * np.pi
    
#     cnt = np.zeros(num_bins-1)
#     for i in range(len(pts)):
#         if idx[i] < num_bins-1:
#             f_lm[idx[i]]+= pl_x[idx[i]]

#     x_grid = np.array([0.5 * (x_grid[i] + x_grid[i+1]) for i in range(num_bins-1)])
#     f_lm   = scipy.interpolate.interp1d(x_grid, f_lm, kind="cubic",bounds_error=False, fill_value="extrapolate")
#     return f_lm

# def radial_component(x):
#     mu     = 0
#     std    = vth / np.sqrt(2)
#     fac    = 1.0 / (np.sqrt(2 * np.pi) * std) **3 
#     return np.exp(-(x/(np.sqrt(2) * std))**2) * fac


# N        = np.int64(1e5)
# num_bins = 50
# pts      = prior_samples(N) 
# #y_pts    = bte_utils.mcmc_sampling(target_pdf, prior_samples, 10)


# #x_domain = (0,8)
# ev_domain = (0,10)
# ev_grid   = np.linspace(ev_domain[0], ev_domain[1], 1000)
# x_grid    = np.sqrt(ev_grid) * c_gamma / vth
# x_domain  = (x_grid[0], x_grid[-1])


# f0       = mcmc_lm(pts, (0,0), x_domain = x_domain, num_bins= num_bins, vth = vth)
# f1       = mcmc_lm(pts, (1,0), x_domain = x_domain, num_bins= num_bins, vth = vth)
# f2       = mcmc_lm(pts, (2,0), x_domain = x_domain, num_bins= num_bins, vth = vth)


# f0_ev    = f0(x_grid)
# ff       = np.trapz(f0_ev, x=ev_grid)
# f0_ev    = f0(x_grid) / np.sqrt(ev_grid) / ff 
# f1_ev    = f1(x_grid) / np.sqrt(ev_grid) / ff 
# f2_ev    = f2(x_grid) / np.sqrt(ev_grid) / ff 


# y_f0       = mcmc_lm(y_pts, (0,0), x_domain = x_domain, num_bins= num_bins, vth = vth)
# y_f1       = mcmc_lm(y_pts, (1,0), x_domain = x_domain, num_bins= num_bins, vth = vth)
# y_f2       = mcmc_lm(y_pts, (2,0), x_domain = x_domain, num_bins= num_bins, vth = vth)

# y_f0_ev    = y_f0(x_grid)
# ff         = np.trapz(y_f0_ev * np.sqrt(ev_grid), x=ev_grid)
# y_f0_ev    = y_f0(x_grid) / ff
# y_f1_ev    = y_f1(x_grid) / ff
# y_f2_ev    = y_f2(x_grid) / ff


# y_rr       = radial_component(x_grid * vth)
# ww         = np.trapz(y_rr * np.sqrt(ev_grid) , x=ev_grid) 
# y_rr       = y_rr /ww


# plt.subplot(1,3,1)
# plt.semilogy(ev_grid, np.abs(f0_ev))
# #plt.semilogy(ev_grid, np.abs(y_f0_ev))
# plt.semilogy(ev_grid, np.abs(y_rr),'--')
# plt.grid(visible=True)

# plt.subplot(1,3,2)
# plt.semilogy(ev_grid, np.abs(f1_ev))
# #plt.semilogy(ev_grid, np.abs(y_f1_ev))
# plt.grid(visible=True)

# plt.subplot(1,3,3)
# plt.semilogy(ev_grid, np.abs(f2_ev))
# #plt.semilogy(ev_grid, np.abs(y_f2_ev))
# plt.grid(visible=True)

# plt.show()



# y  = prior_samples(N)
# vr = np.linalg.norm(y,axis=1)
# vt = np.arccos(y[:,2]/np.linalg.norm(y,axis=1))
# # both df1 and df2 have bivaraite normals, df1.size=200, df2.size=100
# df1 = pd.DataFrame(np.concatenate((vr.reshape((-1,1)), vt.reshape((-1,1))), axis=1), columns=['vr', 'vt'])

# yp = bte_utils.mcmc_sampling(target_pdf, prior_samples, n_samples=N, burn_in=0.3)
# vr = np.linalg.norm(yp,axis=1)
# vt = np.arccos(yp[:,2]/np.linalg.norm(yp,axis=1))
# df2 = pd.DataFrame(np.concatenate((vr.reshape((-1,1)), vt.reshape((-1,1))), axis=1), columns=['vr', 'vt'])


# # plot
# # ========================================   
# sns.jointplot(data=df1, x="vr", y="vt", kind='kde', color='r')
# sns.jointplot(data=df2, x="vr", y="vt", kind='kde', color='b')
# plt.show()

#sns.jointplot(vr, vt, kind="kde")
#plt.show()

# plt.hist(y, bins=50, density=True, color="blue")
# #plt.plot(x, normal_pdf(x,0,1),label="prior")
# plt.grid(visible=True)

# plt.hist(yp, bins=50, density=True, color="red")
# plt.plot(x,  target_pdf(x),label="target")
# plt.legend()
# plt.show()



#plt.close()

# sns.histplot(y,bins=50,stat="density",legend="prior")
# sns.lineplot(x, target_pdf(x), legend="target")
# sns.histplot(yp,bins=50,stat="density")
# plt.show()












