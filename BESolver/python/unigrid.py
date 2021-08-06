"""
@package simple class to manage uniformly space cartiesian grid of dimension k. 
"""
import numpy as np

class UCartiesianGrid():
    """
    basic uniform cartesian grid class. 
    """
    def __init__(self, dim, grid_min, grid_max, grid_res):
        """
        @param dim : dimension of the grid
        @param grid_min : grid min point
        @param grid_min : grid max point
        @param grid_pts : resolution on each dimension
        """
        self._dim   = dim
        self._min_pt = np.array(grid_min)
        self._max_pt = np.array(grid_max)
        self._res    = np.array(grid_res)

        assert len(self._min_pt) == self._dim , "grid dimension and the grid min point are not consistent" 
        assert len(self._max_pt) == self._dim , "grid dimension and the grid max point are not consistent" 
        assert len(self._res)    == self._dim , "grid dimension and the grid resolution parameter are not consistent" 

        self._num_pts = ((self._max_pt -self._min_pt)/self._res).astype(int) + 1

    def create_vec(self,dof=1,array_type=float):
        """
        allocates a numpy array corresponding to the grid points. 
        (dim 0,dim 1,...dim N-1,dof) shape
        """
        shape = tuple(self._num_pts) + tuple([dof])
        return np.zeros(shape,dtype=array_type)
    
    def get_grid_point(self,index):
        """
        Gets the coordinates of a point.
        """
        pt = np.zeros(self._dim)
        pt = self._min_pt + self._res * index
        return pt

    def get_num_pts(self):
        """
        return the grid number of points in each dimension. 
        """
        return np.array(self._num_pts)


def init_vec(ugrid,func,dof=1,array_type=float):
    """
    initialize a ugrid array to a spatially dependent function. 
    the initialization function should match the dof the vector is created with. 
    """
    shape = tuple(ugrid._num_pts) + tuple([dof])
    fv    = np.zeros(shape,dtype=array_type)

    if (ugrid._dim==3):
        for k in range(ugrid._num_pts[2]):
            for j in range(ugrid._num_pts[1]):
                for i in range(ugrid._num_pts[0]):
                    fv[i,j,k,:] = func(ugrid.get_grid_point(np.array([i,j,k])))
    elif (ugrid._dim==2):
        for j in range(ugrid._num_pts[1]):
            for i in range(ugrid._num_pts[0]):
                fv[i,j,:] = func(ugrid.get_grid_point(np.array([i,j])))
    elif (ugrid._dim==1):
        for i in range(ugrid._num_pts[0]):
            fv[i,:] = func(ugrid.get_grid_point(np.array([i])))

    return fv

        

    






        


        


