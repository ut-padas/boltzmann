"""
@package Utilities to visualize distribution solutions. 
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib import meshgrid
from numpy.ma import soften_mask
import spec_spherical
import utils

def plot_density_distribution_iso(f,domain,sample_points):
    """
    Plots isotropic distribution, distribution value only depends on the
    vector norm, not vector components. 
    """
    v = np.linspace(domain[0],domain[1],sample_points)
    fv= f(v) 
    plt.plot(v, fv)
    plt.xlabel('speed |v|')
    plt.ylabel('number density')
    plt.show()

def plot_density_distribution_z_slice(spec, coeff, domain, sample_points,z_val=0.0, weight_func=None,file_name=None):
    """
    Plots isotropic distribution, distribution value only depends on the
    vector norm, not vector components. 
    """
    vx = np.linspace(domain[0,0],domain[0,1],sample_points)
    vy = np.linspace(domain[1,0],domain[1,1],sample_points)

    dx = (vx[1]-vx[0])/2.
    dy = (vy[1]-vy[0])/2.
    extent = [vx[0]-dx, vx[-1]+dx, vy[0]-dy, vy[-1]+dy]

    v  = np.zeros((sample_points,sample_points))

    num_p = spec._p + 1

    for j,y in enumerate(vy):
        for i,x in enumerate(vx):
            for pk in range(num_p):
                for pj in range(num_p):
                    for pi in range(num_p):
                        v[i , j] += coeff[pk * num_p * num_p + pj * num_p + pi] * spec.basis_eval3d(x,y,z_val,(pi,pj,pk))
            
            if(weight_func is not None):
                v_abs = np.sqrt(x**2 + y**2 + z_val**2)
                v[i , j]*=weight_func(v_abs)
    
    plt.imshow(v,extent=extent)
    plt.xlabel('vx')
    plt.ylabel('vy')
    plt.colorbar()

    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()

    plt.close()


def plot_spec_coefficients(coeff,file_name=None):
    plt.plot(coeff)
    plt.xlabel('coefficient id')
    plt.ylabel('coefficient value')
    
    if file_name is not None:
        plt.savefig(file_name)
    else:
        plt.show()
    plt.close()


def plot_f_theta_slice(cf,maxwellian, spec : spec_spherical.SpectralExpansionSpherical , r_max,  r_res, phi_res,fname,theta=np.pi/2):

    phi_pts = np.linspace(0, 2*np.pi , int((2*np.pi)/phi_res))
    r_pts   = np.linspace(0, r_max   , int((r_max)/r_res))
    R,P     = np.meshgrid(r_pts,phi_pts)
    
    f_val   = np.zeros(( len(r_pts),  len(phi_pts) ))

    num_p   = spec._p + 1
    sh_lm   = spec._sph_harm_lm
    num_sh  = len(sh_lm) 

    for r_i,r in enumerate(r_pts):
        for phi_i, phi in enumerate(phi_pts):
            for pi in range(num_p):
                for lm_i, lm in enumerate(sh_lm):
                    rid = pi*num_sh + lm_i
                    f_val[r_i, phi_i] += cf[rid] * maxwellian(r) * spec.basis_eval_full(r, theta, phi,pi,lm[0],lm[1]) 


    #plt.imshow(f_val)
    #plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Express the mesh in the cartesian system.
    X, Y = R*np.cos(P), R*np.sin(P)
    #print(X.shape)
    #print(f_val.shape)
    f_val=np.transpose(f_val)
    # Plot the surface.
    ax.plot_surface(X, Y, f_val, cmap=plt.cm.YlGnBu_r)

    # Tweak the limits and add latex math labels.
    # ax.set_zlim(0, 1)
    # ax.set_xlabel(r'$\phi_\mathrm{real}$')
    # ax.set_ylabel(r'$\phi_\mathrm{im}$')
    # ax.set_zlabel(r'$V(\phi)$')
    #plt.show()
    plt.savefig(fname)
    plt.close()


def plot_f_z_slice(cf,maxwellian, spec : spec_spherical.SpectralExpansionSpherical , X , Y, fname,z_val=0.0,title=""):

    spherical_pts=list()

    for y in Y:
        for x in X:
            spherical_pts.append(utils.cartesian_to_spherical(x,y,z_val))

    num_pts = len(X)*len(Y)
    f_val   = np.zeros(num_pts)

    num_p   = spec._p + 1
    sh_lm   = spec._sph_harm_lm
    num_sh  = len(sh_lm) 

    for pt_i,pt in enumerate(spherical_pts):
        for pi in range(num_p):
            for lm_i, lm in enumerate(sh_lm):
                rid = pi*num_sh + lm_i
                f_val[pt_i] += cf[rid] * maxwellian(pt[0]) * spec.basis_eval_full(pt[0], pt[1], pt[2], pi, lm[0], lm[1])

    f_val=f_val.reshape((len(Y),len(X)))
    #plt.title(title)
    #plt.imshow(f_val)
    #plt.colorbar()
    # plt.show()

    X,Y = np.meshgrid(X,Y)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, f_val, cmap=plt.cm.YlGnBu_r)
    ax.set_title(title)

    # # Tweak the limits and add latex math labels.
    # # ax.set_zlim(0, 1)
    # ax.set_xlabel(r'$X$')
    # ax.set_ylabel(r'$Y$')
    # ax.set_ylabel(r'$\phi_\mathrm{im}$')
    # ax.set_zlabel(r'$V(\phi)$')
    #plt.colorbar()
    #plt.show()
    plt.savefig(fname)
    plt.close()

