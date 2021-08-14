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
from operator import contains
import numpy as np
from numpy.linalg.linalg import norm
import cross_section
from scipy import interpolate
import scipy.constants


MASS_ELECTRON       = scipy.constants.m_e
MASS_ARGON          = 6.6335209E-26 
MASS_R_EARGON       = MASS_ELECTRON/MASS_ARGON
E_AR_IONIZATION_eV  = 11.55
E_AR_EXCITATION_eV  = 15.76
ELECTRON_VOLT       = scipy.constants.electron_volt

MAXWELLIAN_TEMP_K   = 5000
MAXWELLIAN_N        = 1e3
BOLTZMANN_CONST     = scipy.constants.Boltzmann
ELECTRON_THEMAL_VEL = np.sqrt(2*BOLTZMANN_CONST*MAXWELLIAN_TEMP_K/MASS_ELECTRON)

#print(ELECTRON_THEMAL_VEL)

class Collisions(abc.ABC):

    def __init__(self)->None:
        pass
    
    def load_cross_section(self,fname)->None:
        """
        build interpolant for cross section data
        """
        self._cs_fname  = fname
        self._cs_fields = ["energy", "cross section"]
        np_data      = cross_section.lxcat_cross_section_to_numpy(self._cs_fname, self._cs_fields)
        self._energy   = np_data[0]
        self._total_cs = np_data[1]
        self._total_cs_interp1d = interpolate.interp1d(self._energy, self._total_cs, kind='linear')
        return

    @staticmethod
    def compute_scattering_direction(v0,polar_angle,azimuthal_angle):
        """
        compute scattering velocity based in input velocities
        """

        E0 = v0/np.linalg.norm(v0,2)
        ei = np.array([1.0,0.0,0.0])

        # theta computation
        theta = np.arccos(np.dot(E0,ei))
        # the below is a special case where v0 is along the ei in that case choose E1 = ej, E2 = ek
        if np.allclose(theta,0):
            E1 = np.array([0.0,1.0,0.0])
            E2 = np.array([0.0,0.0,1.0])
        else:
            E1 = np.cross(E0,ei)/np.sin(theta)
            E2 = np.cross(E0,E1)

        vs = np.cos(polar_angle) * E0 + np.sin(polar_angle)*np.sin(azimuthal_angle) * E1 + np.sin(polar_angle)*np.cos(azimuthal_angle) * E2

        # sanity check, 
        assert np.allclose(np.dot(vs,vs),1.0) , "vs scattering direction is not a unit vector %f " % np.dot(vs,vs) 
        return vs
        
    def total_cross_section(self, energy)->float:
        """
        computes the total cross section based on the experimental data. 
        """
        return self._total_cs_interp1d(energy)

    @staticmethod
    def differential_cross_section(total_cross_section : float, energy : float, scattering_angle: float ) -> float:
        """
        computes the differential cross section from total cross section. 
        """
        return total_cross_section/(4 * np.pi *  (1 + energy * (np.sin(0.5*scattering_angle)**2))  *  np.log(1+energy) )

    @abc.abstractmethod
    def compute_scattering_velocity(v0, polar_angle, azimuthal_angle):
        pass


"""
e + Ar -> e + Ar
"""
class eAr_G0(Collisions):

    def __init__(self) -> None:
        super().__init__()
        self.load_cross_section("lxcat_data/eAr_Elastic.txt")

    @staticmethod
    def compute_scattering_velocity(v0, polar_angle, azimuthal_angle):

        v1_dir   = Collisions.compute_scattering_direction(v0,polar_angle, azimuthal_angle)
        vel_fac  = np.linalg.norm(v0,2) #* np.sqrt(1- 2*MASS_R_EARGON*(1-np.cos(polar_angle)))
        v1       = vel_fac  * v1_dir
        return v1 


    

"""
e + Ar -> e + Ar^*
"""
class eAr_G1(Collisions):
    
    def __init__(self) -> None:
        super().__init__()
        self.load_cross_section("lxcat_data/eAr_Excitation.txt")

    @staticmethod
    def compute_scattering_velocity(v0, polar_angle, azimuthal_angle):
        v1_dir   = Collisions.compute_scattering_direction(v0,polar_angle, azimuthal_angle)
        vel_fac  = np.sqrt(np.linalg(v0,2) **2    - (2 * E_AR_EXCITATION_eV * ELECTRON_VOLT/MASS_ELECTRON))
        v1       =  vel_fac  * v1_dir
        return v1


"""
e + Ar -> e + Ar^+
"""
class eAr_G2(Collisions):
    
    def __init__(self) -> None:
        super().__init__()
        self.load_cross_section("lxcat_data/eAr_Ionization.txt")

    @staticmethod
    def compute_scattering_velocity(v0, polar_angle, azimuthal_angle):
        v1_dir   = Collisions.compute_scattering_direction(v0,polar_angle, azimuthal_angle)

        v1_fac   = np.sqrt(0.5 * np.linalg(v0,2)**2    - (E_AR_IONIZATION_eV * ELECTRON_VOLT/MASS_ELECTRON))
        v2_fac   = v1_fac

        v1       =  v1_fac  * v1_dir
        v2_dir   =  (v0-v1)/np.linalg.norm(v0-v1,2)
        v2       =  v2_fac * v2_dir

        return v1,v2


