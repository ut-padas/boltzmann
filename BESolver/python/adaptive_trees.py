"""
linear adaptive trees
"""
import numpy as np
import collisions
import basis
import quadpy

max_depth = 15
min_depth = 4
gl=quadpy.c1.gauss_legendre(3)
ev_range=(0,30)

def adaptive_binary_tree(min_depth: int, max_depth : int, int_to_domain_coord_map, is_refine):
    leaf = [(0,0)]
    tmp  =list()

    for _ in range(min_depth):
        for node in leaf:
            tmp.append((node[0], node[1]+1))
            tmp.append((node[0] + (1<<(max_depth-node[1]-1)), node[1]+1))
        
        leaf   = tmp
        tmp    = list()
    
    tmp=list()
    while(1):
        has_changed = False
        for node in leaf:
            
            domain_coords = int_to_domain_coord_map((node[0], node[0] + (1<<(max_depth-node[1]))))
            refine        = is_refine(domain_coords)
            #print(node, (node[0], node[0] + (1<<(max_depth-node[1]))), refine)
            
            if node[1] + 1 <=max_depth  and refine:
                tmp.append((node[0], node[1]+1))
                tmp.append((node[0] + (1<<(max_depth-node[1]-1)), node[1]+1))
                has_changed=True
            else:
                tmp.append((node[0], node[1]))

        leaf = tmp
        tmp  = list()

        if not has_changed:
            break
    
    #print(leaf)
    d_coord=list()
    for node in leaf:
        domain_coords = int_to_domain_coord_map((node[0], node[0] + (1<<(max_depth-node[1]))))
        d_coord.append(domain_coords[0])
        d_coord.append(domain_coords[1])

    return np.sort(np.array(list(set(d_coord))))

coord_map = lambda x : (x[0] *  ((ev_range[-1]-ev_range[0])/(1<<max_depth) )  + ev_range[0], x[1] *  ( (ev_range[-1]-ev_range[0])/(1<<max_depth))  + ev_range[0])

g0 = collisions.eAr_G0()
g2 = collisions.eAr_G2()

tcs_g0 = lambda ev : collisions.Collisions.synthetic_tcs(ev,"g0")
tcs_g2 = lambda ev : collisions.Collisions.synthetic_tcs(ev,"g2smoothstep")

def is_refine_func(dx,tcs,rtol,atol):
    # gx,gw = 0.5 * (dx[1] - dx[0]) * gl.points + 0.5 * (dx[1] + dx[0]), 0.5 * (dx[1]-dx[0]) * gl.weights

    # q1,_  = scipy.integrate.quadrature(tcs,dx[0],dx[1],tol=atol, rtol=1e-14, maxiter=10000)
    # q2    = np.dot(tcs(gx),gw)
    
    # if(q1 == 0.0):
    #     return abs(q2-q1) > atol
    # else:
    #     return abs(q2/q1-1) > rtol

    #print(abs(0.5 * (tcs(dx[0]) + tcs(dx[1]))/tcs(0.5*(dx[0]+dx[1])) -1))
    return abs(0.5 * (tcs(dx[0]) + tcs(dx[1]))/tcs(0.5*(dx[0]+dx[1])) -1) > rtol



#knots_g0=adaptive_binary_tree(min_depth,max_depth,coord_map,is_refine_g0)
#print(knots_g0, len(knots_g0))


#knots_g2=adaptive_binary_tree(min_depth,max_depth,coord_map,is_refine_g2)
#print(knots_g2, len(knots_g2))

#knots=np.union1d(knots_g0,knots_g2)
#print(knots, len(knots))

def g0_knots(rtol, atol):
    is_refine_g0 = lambda dx : is_refine_func(dx, tcs_g0, rtol, atol)
    knots        = adaptive_binary_tree(min_depth,max_depth,coord_map,is_refine_g0)
    return knots

def g2_knots(rtol, atol):
    is_refine_g2 = lambda dx : is_refine_func(dx, tcs_g0, rtol, atol)
    knots        = adaptive_binary_tree(min_depth,max_depth,coord_map,is_refine_g2)
    return knots

def g0g2_knots(rtol, atol):
    is_refine_g0 = lambda dx : is_refine_func(dx, tcs_g0, rtol, atol)
    is_refine_g2 = lambda dx : is_refine_func(dx, tcs_g0, rtol, atol)

    knots_g0= adaptive_binary_tree(min_depth,max_depth,coord_map,is_refine_g0)
    knots_g2= adaptive_binary_tree(min_depth,max_depth,coord_map,is_refine_g2)
    knots   = np.union1d(knots_g0,knots_g2)
    return knots