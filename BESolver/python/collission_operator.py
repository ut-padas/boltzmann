"""
@package : Spectral based, Petrov-Galerkin discretization of the collission integral
"""
import basis
import spec as sp
import abc
import enum

class SpecWeakFormCollissionOp(abs.ABC):

    def __init__(self,dim,p_order):
        self._dim = dim
        self._p = p_order
        pass

    @abc.abstractmethod
    def assemble_matrix(self):
        pass


class CollisionOpElectronNeutral(SpecWeakFormCollissionOp):
    """
    Collission operator for electron-neutral. Density function 
    for the neutral assumed to be delta function for all times. 
    """

    def __init__(self, dim, p_order,basis_type):
        """
        dim - dimension of the problem
        p_order - polynomial order of the approximation
        basis_type - which basis to use for the expansion, 
        look BasisType
        """
        super().__init__(dim, p_order)
        self._basis_type = basis_type
        
        # currently implement for hermite only. 
        assert(self._basis_type is basis.BasisType.HERMITE_E_POLY)
        if(self._basis_type is basis.BasisType.HERMITE_E_POLY):
            self._spec       = sp.SpectralExpansion(self._dim,self._p,basis.hermite_e)

        
    def assemble_matrix(self):
        [gl_x,gl_w] = basis.gauss_legendre(2*self._p)
        [gh_x,gh_w] = basis.gauss_hermiteE(2*self._p)


        

        