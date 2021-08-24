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
from math import cos, isnan, sin
from operator import contains
import numpy as np
from numpy.linalg.linalg import norm
import cross_section
from scipy import interpolate
import scipy.constants

MASS_ELECTRON       = scipy.constants.m_e
MASS_ARGON          = 6.6335209E-26 
MASS_R_EARGON       = MASS_ELECTRON/MASS_ARGON
E_AR_IONIZATION_eV  = 15.76
E_AR_EXCITATION_eV  = 11.55
ELECTRON_VOLT       = scipy.constants.electron_volt

BOLTZMANN_CONST     = scipy.constants.Boltzmann
TEMP_K_1EV          = ELECTRON_VOLT/BOLTZMANN_CONST
MAXWELLIAN_TEMP_K   = 1*TEMP_K_1EV
AR_NEUTRAL_N        = 3.22e22 # 1/m^3
MAXWELLIAN_N        = 1.00e12 # 1/m^3
ELECTRON_THEMAL_VEL = np.sqrt(2*BOLTZMANN_CONST*MAXWELLIAN_TEMP_K/MASS_ELECTRON)

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
        self._total_cs_interp1d = interpolate.interp1d(self._energy, self._total_cs, kind='linear',bounds_error=False,fill_value=0.0)
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
        if np.allclose(theta,0.0) or np.allclose(theta,np.pi):
            E1 = np.array([0.0,1.0,0.0])
            E2 = np.array([0.0,0.0,1.0])
        else:
            E1 = np.cross(E0,ei)/np.sin(theta)
            E2 = np.cross(E0,E1)

        vs = np.cos(polar_angle) * E0 + np.sin(polar_angle)*np.sin(azimuthal_angle) * E1 + np.sin(polar_angle)*np.cos(azimuthal_angle) * E2
        # sanity check, 
        assert np.allclose(np.dot(vs,vs),1.0) , "vs scattering direction is not a unit vector %f " % np.dot(vs,vs) 
        return vs

    @staticmethod
    def compute_scattering_direction_sp(v_r,v_theta,v_phi,polar_angle,azimuthal_angle):

        check1 = np.isclose(v_theta,np.pi/2)
        check2 = np.logical_or(np.isclose(v_phi,0),np.isclose(v_phi,np.pi))
        check2 = np.logical_or(check2, np.isclose(v_phi,2*np.pi))

        check1 = np.logical_and(check1 , check1)
        
        [v_theta_0, v_theta_1] = [v_theta[check1] , v_theta[np.logical_not(check1)]]
        [v_phi_0, v_phi_1]     = [v_phi[check1]   , v_phi[np.logical_not(check1)]]
        
        [chi_0, chi_1]         = [polar_angle[check1], polar_angle[np.logical_not(check1)]]
        [az_0, az_1]           = [azimuthal_angle[check1], azimuthal_angle[np.logical_not(check1)]]
        
        t0 = np.arccos( np.cos(az_0) * np.sin(chi_0) )
        p0 = np.arctan( np.sin(az_0) * np.tan(chi_0) )

        f1 = np.sqrt(1 - (np.cos(v_phi_1)**2) * (np.sin(v_theta_1)**2))

        W= (-np.sin(az_1) * np.sin(v_theta_1) * np.sin(v_phi_1) * np.sin(chi_1) + \
             np.cos(v_theta_1) * (np.cos(chi_1) * f1 + np.cos(az_1) * np.cos(v_phi_1) *np.sin(v_theta_1) *np.sin(chi_1))) / f1

        if(np.min(W)<-1 or np.max(W)>1):
            c1=W>1
            c2=W<-1
            
            print("====resetting following to [-1,1]====")
            print(W[c1])
            print(W[c2])

            W[c1] =1.0
            W[c2]=-1.0
        
        t1 = np.arccos(W)
        p1 = np.arctan(np.cos(chi_1) * np.sin(v_theta_1) * f1 * np.sin(v_phi_1) + np.sin(chi_1) * ( np.cos(v_theta_1)* np.sin(az_1) + np.cos(az_1) * np.cos(v_phi_1) * (np.sin(v_theta_1)**2) * np.sin(v_phi_1)) / (np.cos(v_phi_1) * np.cos(chi_1) * np.sin(v_theta_1) * f1 - 
        np.cos(az_1) * ( np.cos(v_theta_1)**2 + (np.sin(v_theta_1)**2) * (np.sin(v_phi_1)**2)) * np.sin(chi_1) ))
        
        r1      = np.ones_like(v_theta)
        theta_p = np.zeros_like(v_theta)
        phi_p   = np.zeros_like(v_phi)
        
        theta_p[check1] = t0
        phi_p[check1]   = p0

        # print(np.isnan(t1).shape)
        # print(v_theta[np.isnan(t1)[0]])
        # print(v_theta[np.logical_not(check1)])

        theta_p[np.logical_not(check1)] = t1
        phi_p  [np.logical_not(check1)] = p1

        return [r1,theta_p,phi_p]        
        
    def total_cross_section(self, energy):
        """
        computes the total cross section based on the experimental data. 
        """
        return self._total_cs_interp1d(energy)

    @staticmethod
    def diff_cs_to_total_cs_ratio(energy,scattering_angle):
        return (energy)/(4 * np.pi *  (1 + energy * (np.sin(0.5*scattering_angle)**2))  *  np.log(1+energy) )

    @staticmethod
    def differential_cross_section(total_cross_section : float, energy : float, scattering_angle: float ) -> float:
        """
        computes the differential cross section from total cross section. 
        """
        return (energy*total_cross_section)/(4 * np.pi *  (1 + energy * (np.sin(0.5*scattering_angle)**2))  *  np.log(1+energy) )

    @abc.abstractmethod
    def compute_scattering_velocity(v0, polar_angle, azimuthal_angle):
        pass

    @abc.abstractmethod
    def compute_scattering_velocity_sp(vr,vt,vp, polar_angle, azimuthal_angle):
        pass
    
    @abc.abstractmethod
    def min_energy_threshold():
        """
        Gives the minimum evergy threshold for the event in ev
        """
        pass

    def assemble_diff_cs_mat(self,v,chi):
        """
        computes the differential cross section matrix. 
        v   : np array of velocity of electron particles
        chi : np array of scattering angles
        Note!! : If the energy threshold is not satisfied diff. cross section would be zero. 
        """
        #num_v = len(v)
        #v = v.reshape(num_v,1)
        energy_ev = (0.5 * MASS_ELECTRON * (v**2))/ELECTRON_VOLT
        total_cs  = self._total_cs_interp1d(energy_ev)
        diff_cs   = (total_cs*energy_ev)/(4 * np.pi * (1 + energy_ev * (np.sin(0.5*chi))**2 ) * np.log(1+energy_ev) )
        return diff_cs

    



class CollisionType():
    EAR_G0=0
    EAR_G1=1
    EAR_G2=2


"""
e + Ar -> e + Ar
Only difference is there is no energy loss
when computing the scattering velocity
"""
class eAr_G0_NoEnergyLoss(Collisions):

    def __init__(self) -> None:
        super().__init__()
        self.load_cross_section("lxcat_data/eAr_Elastic.txt")
        self._type=CollisionType.EAR_G0

    @staticmethod
    def compute_scattering_velocity(v0, polar_angle, azimuthal_angle):

        v1_dir   = Collisions.compute_scattering_direction(v0,polar_angle, azimuthal_angle)
        vel_fac  = np.linalg.norm(v0,2)
        v1       = vel_fac  * v1_dir
        return v1 

    @staticmethod
    def compute_scattering_velocity_sp(vr,vt,vp, polar_angle, azimuthal_angle):
        vs    = Collisions.compute_scattering_direction_sp(vr,vt,vp,polar_angle,azimuthal_angle)
        vs[0] = vr
        return vs


    @staticmethod
    def min_energy_threshold():
        return 0.0

"""
e + Ar -> e + Ar
"""
class eAr_G0(Collisions):

    def __init__(self) -> None:
        super().__init__()
        self.load_cross_section("lxcat_data/eAr_Elastic.txt")
        self._type=CollisionType.EAR_G0

    @staticmethod
    def compute_scattering_velocity(v0, polar_angle, azimuthal_angle):

        v1_dir   = Collisions.compute_scattering_direction(v0,polar_angle, azimuthal_angle)
        vel_fac  = np.linalg.norm(v0,2) * np.sqrt(1- 2*MASS_R_EARGON*(1-np.cos(polar_angle)))
        v1       = vel_fac  * v1_dir
        return v1 

    @staticmethod
    def compute_scattering_velocity_sp(vr,vt,vp, polar_angle, azimuthal_angle):
        vs       = Collisions.compute_scattering_direction_sp(vr,vt,vp,polar_angle,azimuthal_angle)
        vel_fac  = vr * np.sqrt(1- 2*MASS_R_EARGON*(1-np.cos(polar_angle)))
        vs[0]    = vel_fac
        return vs

    @staticmethod
    def min_energy_threshold():
        return 0.0
    

"""
e + Ar -> e + Ar^*
"""
class eAr_G1(Collisions):
    
    def __init__(self) -> None:
        super().__init__()
        self.load_cross_section("lxcat_data/eAr_Excitation.txt")
        self._type=CollisionType.EAR_G1

    @staticmethod
    def compute_scattering_velocity(v0, polar_angle, azimuthal_angle):
        v1_dir   = Collisions.compute_scattering_direction(v0,polar_angle, azimuthal_angle)
        assert (np.linalg.norm(v0,2) **2    - (2 * E_AR_EXCITATION_eV * ELECTRON_VOLT/MASS_ELECTRON)) > 0 , "collision G1 invalid velocity specified: %f "%v0
        vel_fac  = np.sqrt(np.linalg.norm(v0,2) **2    - (2 * E_AR_EXCITATION_eV * ELECTRON_VOLT/MASS_ELECTRON))
        v1       =  vel_fac  * v1_dir
        return v1

    @staticmethod
    def compute_scattering_velocity_sp(vr,vt,vp, polar_angle, azimuthal_angle):
        vs       = Collisions.compute_scattering_direction_sp(vr,vt,vp,polar_angle,azimuthal_angle)
        vel_fac  = np.sqrt(vr**2    - (2 * E_AR_EXCITATION_eV * ELECTRON_VOLT/MASS_ELECTRON))
        vs[0]    = vel_fac
        return vs

    @staticmethod
    def min_energy_threshold():
        return E_AR_EXCITATION_eV

"""
e + Ar -> e + Ar^+
"""
class eAr_G2(Collisions):
    
    def __init__(self) -> None:
        super().__init__()
        self.load_cross_section("lxcat_data/eAr_Ionization.txt")
        self._type=CollisionType.EAR_G2

    @staticmethod
    def compute_scattering_velocity(v0, polar_angle, azimuthal_angle):
        v1_dir   = Collisions.compute_scattering_direction(v0,polar_angle, azimuthal_angle)
        
        assert (0.5 * np.linalg.norm(v0,2)**2    - (E_AR_IONIZATION_eV * ELECTRON_VOLT/MASS_ELECTRON)) > 0 , "collision G2 invalid velocity specified: %f "%v0
        v1_fac   = np.sqrt(0.5 * np.linalg.norm(v0,2)**2    - (E_AR_IONIZATION_eV * ELECTRON_VOLT/MASS_ELECTRON))
        v2_fac   = v1_fac

        v1       =  v1_fac  * v1_dir
        v2_dir   =  (v0-v1)/np.linalg.norm(v0-v1,2)
        v2       =  v2_fac * v2_dir

        return v1,v2

    @staticmethod
    def compute_scattering_velocity_sp(vr,vt,vp, polar_angle, azimuthal_angle):
        pass
        vs       = Collisions.compute_scattering_direction_sp(vr,vt,vp,polar_angle,azimuthal_angle)
        v1_fac   = np.sqrt(0.5 * (vr**2)    - (E_AR_IONIZATION_eV * ELECTRON_VOLT/MASS_ELECTRON))
        v2_fac   = v1_fac

        vs[0]    = v1_fac
        v0       = np.zeros((len(vr),3))
        v1       = np.zeros((len(vr),3))
        
        v0[:,0]  = np.sin(vt) * np.cos(vp)
        v0[:,1]  = np.sin(vt) * np.sin(vp)
        v0[:,2]  = np.cos(vt)

        v1[:,0]  = np.sin(vs[1]) * np.cos(vs[2])
        v1[:,1]  = np.sin(vs[1]) * np.sin(vs[2])
        v1[:,2]  = np.cos(vs[1])

        v2       = (v0-v1)/np.linalg.norm((v0-v1), 2, axis=1)
        

        return vs

    @staticmethod
    def min_energy_threshold():
        return E_AR_IONIZATION_eV

"""
Simple test collision class to test
the collision operator computation
"""
class eAr_TestCollision(Collisions):

    def __init__(self) -> None:
        super().__init__()
        self._type=-1

    def total_cross_section(self, energy)->float:
        #print("c")
        return 1.0
    
    @staticmethod
    def differential_cross_section(total_cross_section : float, energy : float, scattering_angle: float ) -> float:
        """
        computes the differential cross section from total cross section. 
        """
        #print(total_cross_section,energy," diff/total : ", (energy)/(4 * np.pi *  (1 + energy * (np.sin(0.5*scattering_angle)**2))  *  np.log(1+energy) ))
        v0 = np.sqrt(2*energy/MASS_ELECTRON)
        return (v0 * np.cos(scattering_angle))/AR_NEUTRAL_N


    @staticmethod
    def compute_scattering_velocity(v0, polar_angle, azimuthal_angle):
        v1_dir   = Collisions.compute_scattering_direction(v0,polar_angle, azimuthal_angle)
        vel_fac  = np.linalg.norm(v0,2) 
        v1       =  vel_fac  * v1_dir

        return v1

    @staticmethod
    def min_energy_threshold():
        return 0.0