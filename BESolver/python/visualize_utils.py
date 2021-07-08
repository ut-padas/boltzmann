"""
@package Utilities to visualize distribution solutions. 
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib import meshgrid
from numpy.ma import soften_mask

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
    plt.show()
