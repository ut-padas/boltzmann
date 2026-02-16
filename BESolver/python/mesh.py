import numpy as np
import collocation_op
import findiff
import scipy.sparse


class grid_type():
    CHEBYSHEV_COLLOC = 0
    REGULAR_GRID     = 1

def fd_coefficients(x, order):
    xp = np
    I  = xp.eye(x.shape[0])
    L  = [scipy.interpolate.lagrange(x, I[i]) for i in range(len(x))]

    Vdx = xp.array([L[i].deriv(order)(x) for i in range(len(L))]).T
    return Vdx


def upwinded_dx(x, dorder, sw, dir):
    """
    x- collocation points
    dorder - derivative order
    dir    - "L" or "R" denoting left vs. right upwinding
    sw     - stencil width
    """
    xp = np
    Np = len(x)
    Dx = xp.zeros((Np, Np))
        
    if (dir == "L"):
        # m  = fd_coefficients(x[0:sw], dorder)
        # for i in range(sw):
        #     Dx[i, 0:sw] = m[i]

        for i in range(1, sw):
            Dx[i, 0:(i+1)] = fd_coefficients(x[0:(i+1)], dorder)[-1]

        for i in range(sw, Np):
            Dx[i, (i-(sw-1)):(i+1)] = fd_coefficients(x[(i-(sw-1)):(i+1)], dorder)[-1]
    
    else:
        assert dir == "R"
        for i in range(0, Np-(sw-1)):
            Dx[i, i:(i+sw)] = fd_coefficients(x[i:(i+sw)], dorder)[0]

        # m  = fd_coefficients(x[-sw:], dorder)
        # for i in range(sw):
        #     Dx[-(i+1), -sw:] = m[-(i+1)]

        for i in range(1, sw):
            Dx[-(i+1), -(i+1):] = fd_coefficients(x[-(i+1):], dorder)[0]



    return Dx

def central_dx(x, dorder, sw):
    xp = np
    Np = len(x)
    Dx = xp.zeros((Np, Np))

    assert sw % 2 == 1
    pw = (sw-1)//2
    sz = 2 * pw + 1

    m  = fd_coefficients(x[0:sz], dorder)
    for i in range(pw):
        Dx[i, 0:sz] = m[i]

    for i in range(pw, Np-pw):
        Dx[i, (i-pw):(i+pw+1)] = fd_coefficients(x[(i-pw):(i+pw+1)], dorder)[pw]
    
    m  = fd_coefficients(x[-sz:], dorder)
    for i in range(pw):
        Dx[-(i+1), -sz:] = m[-(i+1)]
    
    return Dx

def upwinded_dvt(x, dorder, sw, dir, use_cdx_internal=False):
    xp = np
    Np = len(x)
    Dx = xp.zeros((Np, Np))

    assert Np %2 ==0
    mp = Np//2
    if use_cdx_internal:
        Dx[0:mp, 0:mp]          = central_dx(x[0:mp], 1, 2 * sw + 1)
        Dx[mp: , mp:]           = central_dx(x[mp:] , 1, 2 * sw + 1)
    
    if (dir == "L"):
        if not use_cdx_internal:
            Dx[0:mp, 0:mp]          = upwinded_dx(x[0:mp], 1, sw, "L")
            Dx[mp: , mp:]           = upwinded_dx(x[mp:] , 1, sw, "L")

        Dx[mp]                      = 0.0
        Dx[mp, mp-1:mp+1]           = fd_coefficients(x[mp-1:mp+1], 1)[-1]
    
    else:
        assert dir == "R"
        if not use_cdx_internal:
            Dx[0:mp, 0:mp]          = upwinded_dx(x[0:mp], 1, sw, "R")
            Dx[mp: , mp:]           = upwinded_dx(x[mp:] , 1, sw, "R")

        Dx[mp-1]                    = 0.0
        Dx[mp-1, mp-1:mp+1]         = fd_coefficients(x[mp-1:mp+1], 1)[0]

    return Dx

class mesh():
    
    def __init__(self, N, dim, gtype):
        self.gtype  = gtype
        self.domain = (-1, 1)
        self.N      = N
        self.dim    = dim
        self.sw     = 3

        assert self.dim < 3
        assert self.dim > 0

        for i in range(dim):
            assert (self.N[i] > 2 * self.sw + 1)

        # 1d operators for each discretization
        self.D1     = None
        self.D2     = None

        if(self.gtype == grid_type.CHEBYSHEV_COLLOC):
            self.cheb    = [collocation_op.cheb_collocation_1d(N[i]) for i in range(dim)]
            self.D1      = [self.cheb[i].Dp for i in range(dim)]
            self.D2      = [self.cheb[i].Lp for i in range(dim)]
            self.D3      = [self.cheb[i].Lp3 for i in range(dim)]
            self.D4      = [self.cheb[i].Lp4 for i in range(dim)]
            self.xcoord  = [self.cheb[i].xp for i in range(dim)]
            self.dx      = [np.min(self.xcoord[i][1:] - self.xcoord[i][0:-1]) for i in range(dim)]

        elif(self.gtype == grid_type.REGULAR_GRID):
            sw            = self.sw
            N             = self.N
            
            self.D1       = list()
            self.D2       = list()
            for i in range(dim):
                D1 = np.zeros((N[i], N[i]))
                D2 = np.zeros((N[i], N[i]))

                for r in range(0, sw):
                    idx                           = np.array([k for k in range(-r, 2 * sw+1-r)], dtype=np.int32)
                    c1                            = findiff.coefficients(deriv=1, offsets=list(idx), symbolic=True)
                    c2                            = findiff.coefficients(deriv=2, offsets=list(idx), symbolic=True)
                    
                    D1[r, r + idx]                = c1["coefficients"]
                    D2[r, r + idx]                = c2["coefficients"]

                    c1                            = findiff.coefficients(deriv=1, offsets=list(-idx), symbolic=True)
                    c2                            = findiff.coefficients(deriv=2, offsets=list(-idx), symbolic=True)
                    D1[N[i]-1-r , N[i]-1-r - idx] = c1["coefficients"]
                    D2[N[i]-1-r , N[i]-1-r - idx] = c2["coefficients"]

                    #print(idx, -idx)


                r                = sw
                idx              = np.array([k for k in range(-r, 2 * sw+1-r)], dtype=np.int32)
                c1               = findiff.coefficients(deriv=1, offsets=list(idx), symbolic=True)
                c2               = findiff.coefficients(deriv=2, offsets=list(idx), symbolic=True)
                idx              = np.array(idx, dtype=np.int32)

                for r in range(sw, N[i]-sw):
                    D1[r, r + idx] = c1["coefficients"]
                    D2[r, r + idx] = c2["coefficients"]

                dx = 2/(N[i]-1)
                D1 = (1/dx)     * D1
                D2 = (1/dx**2)  * D2

                # self.D1.append(scipy.sparse.csr_matrix(D1))
                # self.D2.append(scipy.sparse.csr_matrix(D2))

                self.D1.append(D1)
                self.D2.append(D2)
                
                self.xcoord  = [np.linspace(-1, 1, N[i]) for i in range(dim)]
                self.dx      = [np.min(self.xcoord[i][1:] - self.xcoord[i][0:-1]) for i in range(dim)]
                
        else:
            raise NotImplementedError
    
    def create_vec(self, xp, dtype, dof):
        return xp.zeros(tuple([dof]) + tuple(self.N), dtype=dtype)
    
    
    

        