"""
@package: Generic class to store spectral discretization, 

"""
import numpy as np
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
            self._basis_1d.append(self._basis_p(deg,self._domain,self._window))
    
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

    def create_coefficient_vec(self,dtype=float):
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
        pass
        # M=None
        # if(self._dim==3):
                        
        #     if(is_diagonal):

            



    


        






        

        
        

        
    

        


