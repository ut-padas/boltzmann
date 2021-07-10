"""
@package: Generic class to store spectral discretization, 

"""
import numpy as np
from numpy.lib.twodim_base import diag
import basis 

class SpectralExpansion:
    """
    Handles spectral decomposition for given dim, with specified order of expansion, w.r.t. basis_p
    domain and window by default set to None. None assumes the domain is unbounded. 
    """
    def __init__(self, dim, order, basis_p, domain=None, window=None):
        """
        @param dim : dimension (1,2,3)
        @param order : number of basis functions used in 1d
        @param basis_p : np.polynomial object
        """
        self._dim = dim
        self._p = order
        self._domain = domain
        self._window = window

        self._basis_p  = basis_p
        self._basis_1d = list()
        
        for deg in range(self._p+1):
            self._basis_1d.append(self._basis_p.Pn(deg,self._domain,self._window))
    
    def basis_eval3d(self,x,y,z,deg_p):
        """
        Evaluates the cartisian product of 1d polynomial where `deg_p` is a 
        list containing poly order for each dimension, x, y, z denote the standard cartisian coordinates (3d)
        """
        assert(self._dim == len(deg_p))
        return self._basis_1d[deg_p[0]](x) * self._basis_1d[deg_p[1]](y) * self._basis_1d[deg_p[2]](z)

    def basis_eval2d(self,x,y,deg_p):
        """
        Evaluates the cartisian product of 1d polynomial where `deg_p` is a 
        list containing poly order for each dimension, x, y denote the standard cartisian coordinates (2d)
        """
        assert(self._dim == len(deg_p))
        return self._basis_1d[deg_p[0]](x) * self._basis_1d[deg_p[1]](y)

    def basis_eval1d(self,x,deg_p):
        """
        Evaluates the cartisian product of 1d polynomial where `deg_p` is an integer,
        containing poly order for each dimension x denotes the standard cartisian coordinates (1d)
        """
        assert(self._dim == len(deg_p))
        return self._basis_1d[deg_p](x)

    def create_vec(self,dtype=float):
        num_c = (self._p +1)**self._dim
        return np.zeros((num_c,1),dtype=dtype)

    def create_mat(self,dtype=float):
        """
        Create a matrix w.r.t the number of spectral coefficients. 
        """
        num_c = (self._p +1)**self._dim
        return np.zeros((num_c,num_c),dtype=dtype)
    
    def get_num_coefficients(self):
        """
        returns the number of coefficients in  the spectral
        representation. 
        """
        return (self._p +1)**self._dim

    def compute_mass_matrix(self,is_diagonal=False):
        """
        Compute the mass matrix w.r.t the basis polynomials
        if the chosen basis is orthogonal set is_diagonal to True. 
        """
        num_p = self._p +1
        [qx_1d,qw_1d] = self._basis_p.Gauss_Pn(num_p)
        # note : r_id, c_id defined inside the loops on purpose (loops become pure nested), assuming it will allow
        # more agressive optimizations from the python
        if(is_diagonal):
            mm_diag = self.create_vec()
            if(self._dim == 3):
                for pk in range(num_p):
                    for pj in range(num_p):
                        for pi in range(num_p):
                            # quadrature loop. 
                            for qk,qz in enumerate(qx_1d):
                                for qj,qy in enumerate(qx_1d):
                                    for qi,qx in enumerate(qx_1d):
                                        r_id = pk * (num_p**2) + pj * num_p + pi
                                        mm_diag[r_id]+= (qw_1d[qk] * qw_1d[qj] * qw_1d[qi] ) * ( self._basis_1d[pk](qz) * self._basis_1d[pj](qy) * self._basis_1d[pi](qx) )**2
            elif(self._dim == 2):
                for pj in range(num_p):
                    for pi in range(num_p):
                        # quadrature loop. 
                        for qj,qy in enumerate(qx_1d):
                            for qi,qx in enumerate(qx_1d):
                                r_id = pj * num_p + pi
                                mm_diag[r_id]+= (qw_1d[qj] * qw_1d[qi] ) * (self._basis_1d[pj](qy) * self._basis_1d[pi](qx))**2
            elif(self._dim == 1):
                for pi in range(num_p):
                    # quadrature loop. 
                    for qi,qx in enumerate(qx_1d):
                        mm_diag[pi]+= (qw_1d[qi] ) * (self._basis_1d[pi](qx))**2

            return mm_diag
            
        else:
            mm = self.create_mat()
            if(self._dim == 3):
                # loop over polynomials i
                for pk in range(num_p):
                    for pj in range(num_p):
                        for pi in range(num_p):
                            # loop over polynomials j
                            for tk in range(num_p):
                                for tj in range(num_p):
                                    for ti in range(num_p):
                                        # quadrature loop. 
                                        for qk,qz in enumerate(qx_1d):
                                            for qj,qy in enumerate(qx_1d):
                                                for qi,qx in enumerate(qx_1d):
                                                    r_id = pk * (num_p**2) + pj * num_p + pi
                                                    c_id = tk * (num_p**2) + tj * num_p + ti
                                                    mm[r_id,c_id]+= (qw_1d[qk] * qw_1d[qj] * qw_1d[qi] ) * ( self._basis_1d[pk](qz) * self._basis_1d[pj](qy) * self._basis_1d[pi](qx) ) * ( self._basis_1d[tk](qz) * self._basis_1d[tj](qy) * self._basis_1d[ti](qx) )
            elif(self._dim == 2):
                # loop over polynomials i
                for pj in range(num_p):
                    for pi in range(num_p):
                        # loop over polynomials j
                        for tj in range(num_p):
                            for ti in range(num_p):
                                # quadrature loop. 
                                for qj,qy in enumerate(qx_1d):
                                    for qi,qx in enumerate(qx_1d):
                                        r_id = pj * num_p + pi
                                        c_id = tj * num_p + ti
                                        mm[r_id,c_id]+= (qw_1d[qj] * qw_1d[qi] ) * ( self._basis_1d[pj](qy) * self._basis_1d[pi](qx) ) * ( self._basis_1d[tj](qy) * self._basis_1d[ti](qx))
            elif(self._dim == 1):
                # loop over polynomials i
                for pi in range(num_p):
                    # loop over polynomials j
                    for ti in range(num_p):
                        # quadrature loop. 
                        for qi,qx in enumerate(qx_1d):
                            mm[pi,ti]+= (qw_1d[qi] ) * (self._basis_1d[pi](qx) ) * (self._basis_1d[ti](qx))
            
            return mm

    def compute_coefficients(self,func,mm_diag=None):
        """
        computes basis coefficients for a given function,
        for basis orthogonal w.r.t. weight function w(x)
        f(x) = w(x) \sum_i c_i P_i(x)
        """

        if (mm_diag is None):
            mm_diag=self.compute_mass_matrix(is_diagonal=True)
        
        num_p = self._p + 1
        [qx_1d,qw_1d] = self._basis_p.Gauss_Pn(num_p)
        poly_weight   = self._basis_p.Wx()

        coeff = np.zeros(num_p**self._dim)

        if(self._dim == 3):
            for pk in range(num_p):
                for pj in range(num_p):
                    for pi in range(num_p):
                        # quadrature loop. 
                        for qk,qz in enumerate(qx_1d):
                            for qj,qy in enumerate(qx_1d):
                                for qi,qx in enumerate(qx_1d):
                                    r_id  = pk * num_p * num_p + pj * num_p + pi
                                    q_abs = np.sqrt(qz**2 + qy**2 + qx**2)
                                    f_val = func(np.array([qx,qy,qz])) / poly_weight(q_abs)
                                    coeff[r_id]+= qw_1d[qk] * qw_1d[qj] * qw_1d[qi] * f_val * self.basis_eval3d(qx,qy,qz,(pi,pj,pk))

                        coeff[pk * num_p * num_p + pj * num_p + pi]/=mm_diag[pk * num_p * num_p + pj * num_p + pi]

        elif (self._dim==2):
            for pj in range(num_p):
                for pi in range(num_p):
                    # quadrature loop. 
                    for qj,qy in enumerate(qx_1d):
                        for qi,qx in enumerate(qx_1d):
                            r_id = pj * num_p + pi
                            q_abs = np.sqrt(qy**2 + qx**2)
                            f_val = func(np.array([qx,qy])) / poly_weight(q_abs)
                            coeff[r_id]+= qw_1d[qj] * qw_1d[qi] * func (np.array[qx,qy]) * self.basis_eval2d(qx,qy,(pi,pj))

                    coeff[pj * num_p + pi]/=mm_diag[pj * num_p + pi]

        elif (self._dim==1):
            for pi in range(num_p):
                # quadrature loop. 
                for qi,qx in enumerate(qx_1d):
                    r_id = pi
                    q_abs = np.abs(qx)
                    f_val = func(np.array([qx])) / poly_weight(q_abs)
                    coeff[r_id]+= qw_1d[qi] * func (np.array[qx]) * self.basis_eval1d(qx,(pi))

                coeff[pi]/=mm_diag[pi]
        
        return coeff


