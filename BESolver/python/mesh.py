import numpy as np
import collocation_op
import findiff
import scipy.sparse


class grid_type():
    CHEBYSHEV_COLLOC = 0
    REGULAR_GRID     = 1


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

                self.D1.append(scipy.sparse.csr_matrix(D1))
                self.D2.append(scipy.sparse.csr_matrix(D2))
                self.xcoord  = [np.linspace(-1, 1, N[i]) for i in range(dim)]
                self.dx      = [np.min(self.xcoord[i][1:] - self.xcoord[i][0:-1]) for i in range(dim)]
                
        else:
            raise NotImplementedError
    
    def create_vec(self, xp, dtype, dof):
        return xp.zeros(tuple([dof]) + tuple(self.N), dtype=dtype)
    
    
    

        