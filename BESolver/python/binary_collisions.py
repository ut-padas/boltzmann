"""
@package : Abstract class to represents physics of the binary collisions. 
for a given binary collision  
    - pre collision (w1,w2) -> (v1,v2) post collision
    - omega - scattering angle (\theta,\phi), \theta being the polar angle. 
    - pre collision velocities  -> post collission velocities
    - post collision velocities -> pre collision velocities
    - cross section calculation
"""
import abc
import numpy as np

class BinaryCollision(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def pre_vel_to_post_vel(self,w1,w2,omega):
        """
        compute post collision velocity from pre collission velocity, 
        w1,w2 -denotes the pre collision velocity,
        omega - scattering angle (\theta,\phi)
        returns [v1,v2] post collision velocity
        """
        pass

    @abc.abstractmethod
    def post_vel_to_pre_vel(self,v1,v2,omega):
        """
        compute pre collision velocity from post collission velocity, 
        v1,v2 - denotes the post collision velocity
        omega - scattering angle (\theta,\phi)
        returns [w1,w2] pre collision velocity
        """
        pass
    
    @abc.abstractmethod
    def cross_section(self,v_re,omega):
        """
        computes the differential cross section for a collision, 
        v_re- relative velocity after collission
        omega- solid angle describing the scattering angle. 
        returns a scalar. 
        """
        pass



class BinaryElasticCollission3D(BinaryCollision):
    """
    Binary elastic collission for particles with same species
    (Equal mass particles)
    """
    def __init__(self):
        super().__init__()
    
    def pre_vel_to_post_vel(self,w1,w2,omega):
        """
        compute post collision velocity from pre collission velocity, 
        w1,w2 -denotes the pre collision velocity,
        omega - scattering angle (\theta,\phi)
        returns [v1,v2] post collision velocity
        """
        theta = omega[0]
        phi   = omega[1]
        v_omega = np.array(np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
        w_rel_l2 =np.linalg.norm(w1-w2,2)
        v1 = 0.5 * (w1-w2 + w_rel_l2 * v_omega)
        v2 = 0.5 * (w1-w2 - w_rel_l2 * v_omega)
        return [v1,v2]

    def post_vel_to_pre_vel(self,v1,v2,omega):
        """
        compute pre collision velocity from post collission velocity, 
        v1,v2 - denotes the post collision velocity
        omega - scattering angle (\theta,\phi)
        returns [w1,w2] pre collision velocity
        """
        theta = omega[0]
        phi   = omega[1]
        v_omega = np.array(np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta))
        v_rel_l2 =np.linalg.norm(v1-v2,2)
        w1 = 0.5 * (v1-v2 + v_rel_l2 * v_omega)
        w2 = 0.5 * (v1-v2 - v_rel_l2 * v_omega)
        return [w1,w2]

    def cross_section(self,v_re,omega):
        """
        computes the differential cross section for a collision, 
        v_re- relative velocity after collission
        omega- solid angle describing the scattering angle. 
        returns a scalar. 
        """
        hds_const = 1.0
        return hds_const * np.linalg.norm(v_re,2)


