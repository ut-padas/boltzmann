# @package : Abstract class to represents physics of the binary collisions. 
# for a given binary collision  
#     - pre collision (w1,w2) -> (v1,v2) post collision
#     - omega - scattering angle (\theta,\phi), \theta being the polar angle. 
#     - pre collision velocities  -> post collission velocities
#     - post collision velocities -> pre collision velocities
#     - cross section calculation

import abc
import numpy as np
import cross_section
from scipy import interpolate
import scipy.constants
import utils as BEUtils
import scipy.ndimage
import basis
import spec_spherical as sp
import sys

def electron_thermal_velocity(T):
    """
    computes the thermal velocity
    T : temperature in K
    """
    return np.sqrt(2*BOLTZMANN_CONST*T/MASS_ELECTRON)

def electron_temperature(vth):
    """
    computes the thermal velocity
    vth : thermal velocity
    """
    return (0.5 * MASS_ELECTRON * vth**2)/BOLTZMANN_CONST

ELECTRON_CHARGE     = scipy.constants.e
MASS_ELECTRON       = scipy.constants.m_e
ELECTRON_VOLT       = scipy.constants.electron_volt
ELECTRON_CHARGE_MASS_RATIO = ELECTRON_CHARGE/MASS_ELECTRON

BOLTZMANN_CONST     = scipy.constants.Boltzmann
TEMP_K_1EV          = ELECTRON_VOLT/BOLTZMANN_CONST
MAXWELLIAN_TEMP_K   = TEMP_K_1EV
ELECTRON_THEMAL_VEL = electron_thermal_velocity(MAXWELLIAN_TEMP_K) 
#http://farside.ph.utexas.edu/teaching/plasma/Plasmahtml/node6.html
PLASMA_FREQUENCY  = np.sqrt(1 * (scipy.constants.elementary_charge**2) / (scipy.constants.epsilon_0  * scipy.constants.electron_mass))

class CollisionInterpolationType():
    USE_PICEWICE_LINEAR_INTERPOLATION = 0
    USE_BSPLINE_PROJECTION            = 1
    USE_ANALYTICAL_FUNCTION_FIT       = 2 
    
class Collisions(abc.ABC):

    def __init__(self , cross_section:str)->None:
        self._is_scattering_mat_assembled = False
        self._sc_direction_mat            = None
        self._col_name                    = cross_section
        self._reaction_threshold          = 0.0
        #self._cs_interp_type              = CollisionInterpolationType.USE_ANALYTICAL_FUNCTION_FIT
        self._cs_extrapolate              = False#True # extrapolate cs as factor of ln(e)/e
        #self._cs_interp_type              = CollisionInterpolationType.USE_BSPLINE_PROJECTION #CollisionInterpolationType.USE_ANALYTICAL_FUNCTION_FIT
        self._cs_interp_type              = CollisionInterpolationType.USE_PICEWICE_LINEAR_INTERPOLATION
        pass
    
    def load_cross_section(self, collision_str: str)->None:
        """
        build interpolant for cross section data
        """
        
        cs_data                  = cross_section.CROSS_SECTION_DATA[collision_str]
        self._energy             = cs_data["energy"]
        self._total_cs           = cs_data["cross section"]
        self._reaction_threshold = cs_data["threshold"]
        self._attachment_energy  = 0.0
        if cs_data["type"] == "ATTACHMENT":
            self._attachment_energy = float(cs_data["info"]["FORWARD"])
            
        self._mByM               = cross_section.CROSS_SECTION_DATA["E + Ar -> E + Ar"]["mass_ratio"]
        
        self._total_cs_interp1d = interpolate.interp1d(self._energy, self._total_cs, kind='linear', bounds_error=False,fill_value=(self._total_cs[0],self._total_cs[-1]))

        if self._cs_interp_type == CollisionInterpolationType.USE_BSPLINE_PROJECTION:
            sp_order        = 3
            num_p           = 255
            kd_threshold    = 1e-8
            k_domain        = (self._energy[0]- kd_threshold, self._energy[-1] + kd_threshold) 
            k_vec           = basis.BSpline.logspace_knots(k_domain, num_p, sp_order, 0.5 *(self._energy[1] + self._energy[0]) , base=2)
            bb              = basis.BSpline(k_domain, sp_order, num_p, sig_pts=None, knots_vec=k_vec, dg_splines=0, verbose=False)
            
            num_intervals   = bb._num_knot_intervals
            q_pts           = (2 * sp_order + 1) * 2
            gx, gw          = basis.Legendre().Gauss_Pn(q_pts)

            mm_mat    = np.zeros((num_p, num_p))  
            b_rhs     = np.zeros(num_p)
            for p in range(num_p):
                k_min   = bb._t[p]
                k_max   = bb._t[p + sp_order + 1]

                gmx     = 0.5 * (k_max-k_min) * gx + 0.5 * (k_min + k_max)
                gmw     = 0.5 * (k_max-k_min) * gw
                b_p     = bb.Pn(p)(gmx, 0)
                b_rhs[p] = np.dot(gmw, b_p * self._total_cs_interp1d(gmx))
                for k in range(max(0, p - (sp_order+3) ), min(num_p, p + (sp_order+3))):
                    b_k          = bb.Pn(k)(gmx, 0)
                    mm_mat[p,k]  = np.dot(gmw, b_p * b_k)

            def schur_inv(M):
                #return np.linalg.pin1v(M,rcond=1e-30)
                rtol=1e-14
                atol=1e-14

                T, Q = scipy.linalg.schur(M)
                Tinv = scipy.linalg.solve_triangular(T, np.identity(M.shape[0]),lower=False)
                #print("spline cross-section fit mass mat inverse = %.6E "%(np.linalg.norm(np.matmul(T,Tinv)-np.eye(T.shape[0]))/np.linalg.norm(np.eye(T.shape[0]))))
                return np.matmul(np.linalg.inv(np.transpose(Q)), np.matmul(Tinv, np.linalg.inv(Q)))
            
            mm_inv = schur_inv(mm_mat)
            self._sigma_k = np.dot(mm_inv, b_rhs)
            self._bb      = bb  

        # import matplotlib.pyplot as plt
        # fig=plt.figure(figsize=(16,8), dpi=300)
        # plt.subplot(1, 2, 1)
        # plt.loglog(self._energy, self._total_cs,'r-*',label=r"tabulated", markersize=0.8)
        # plt.loglog(self._energy, self.total_cross_section(self._energy),'b--^', label=r"interpolated", markersize=0.8)
        # # ext_ev = np.logspace(-4, 4, 1000, base=10)
        # # plt.loglog(ext_ev, self.total_cross_section(ext_ev),'g--^', label=r"interpolated + extended", markersize=0.8)
        
        # plt.legend()
        # plt.grid(visible=True)
        # plt.xlabel(r"energy (eV)")
        # plt.ylabel(r"cross section ($m^2$)")

        # plt.subplot(1, 2, 2)
        # plt.loglog(self._energy, np.abs(1-self.total_cross_section(self._energy)/self._total_cs))
        # #plt.loglog(self._energy, ,'b--^', label=r"interpolated", markersize=0.8)
        # plt.xlabel(r"energy (eV)")
        # plt.ylabel(r"relative error")
        # plt.grid(visible=True)
        # plt.tight_layout()

        # plt.savefig("col_%s.png"%(self._col_name))
        # plt.close()
        # cs_data=np.zeros((len(self._energy), 3))
        # cs_data[:, 0] = self._energy
        # cs_data[:, 1] = self._total_cs
        # cs_data[:, 2] = self.total_cross_section(self._energy)

        # np.savetxt("col_%s.csv"%(self._col_name), cs_data, delimiter=',', header='energy[eV], tabulated[m^2], fitted[m^2]')


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
            E2 = np.cross(E0,-E1)
        vs = np.cos(polar_angle) * E0 + np.sin(polar_angle)*np.sin(azimuthal_angle) * E1 + np.sin(polar_angle)*np.cos(azimuthal_angle) * E2
        # sanity check, 
        assert np.allclose(np.dot(vs,vs),1.0) , "vs scattering direction is not a unit vector %f " % np.dot(vs,vs) 
        return vs

    def reset_scattering_direction_sp_mat(self):
        self._is_scattering_mat_assembled=False
        self._sc_direction_mat=None
        return

    def compute_scattering_direction_sp(self,v_r,v_theta,v_phi,polar_angle,azimuthal_angle):

        # if self._is_scattering_mat_assembled :
        #     return self._sc_direction_mat
        
        # v_phi[np.isclose(v_phi,0)] +=1e-6
        # v_phi[np.isclose(v_phi,np.pi)] +=1e-6
        # v_phi[np.isclose(v_phi,2*np.pi)] -=1e-6

        E0        = np.zeros(v_r.shape+tuple([3]))
        index_set = v_r>=0

        E0[index_set,0]  = np.sin(v_theta[index_set]) * np.cos(v_phi[index_set])
        E0[index_set,1]  = np.sin(v_theta[index_set]) * np.sin(v_phi[index_set])
        E0[index_set,2]  = np.cos(v_theta[index_set])

        ei      = np.array([1.0,0.0,0.0])
        theta   = np.arccos(np.dot(E0,ei))
        s_theta = np.sin(theta[index_set])
        
        ## 04/21/22 : note this is only needed for b-spline case since it has 0 in the velocity bounds
        #s_theta[s_theta==0]+=np.finfo(float).eps

        E1    = np.cross(E0,ei)
        E1[index_set,0] = E1[index_set,0]/s_theta
        E1[index_set,1] = E1[index_set,1]/s_theta
        E1[index_set,2] = E1[index_set,2]/s_theta

        E2    = np.cross(E0,-E1)
        
        r1      = np.ones_like(v_theta)
        theta_p = np.zeros_like(v_theta)
        phi_p   = np.zeros_like(v_phi)

        v0 = np.zeros_like(E0)

        for i in range(3):
            v0[index_set,i] = np.cos(polar_angle[index_set]) * E0[index_set,i]  + np.sin(polar_angle[index_set]) * np.sin(azimuthal_angle[index_set]) * E1[index_set,i] + np.sin(polar_angle[index_set]) * np.cos(azimuthal_angle[index_set]) * E2[index_set,i]

        
        r1              = np.sqrt(v0[index_set,0]**2 + v0[index_set,1]**2 + v0[index_set,2]**2)
        r1              = r1.reshape(v_r.shape)

        theta_p         = np.arccos(np.divide(v0[index_set,2], r1[index_set], where=r1[index_set] != 0))
        phi_p           = np.arctan2(v0[index_set,1], v0[index_set,0])
        phi_p           = phi_p % (2 * np.pi)

        theta_p = theta_p.reshape(v_r.shape)
        phi_p   = phi_p.reshape(v_r.shape)

        self._sc_direction_mat=[r1,theta_p,phi_p]
        return self._sc_direction_mat
        
    def total_cross_section(self, energy):
        """
        computes the total cross section based on the experimental data. 
        """
        if self._cs_interp_type == CollisionInterpolationType.USE_BSPLINE_PROJECTION:
            bb       =  self._bb
            num_p    =  bb._num_p 
            Vq       = np.array([bb.Pn(p)(energy, 0) for p in range(num_p)]).reshape((num_p, len(energy)))
            total_cs = np.abs(np.dot(self._sigma_k, Vq))
            if self._cs_extrapolate == True:
                print("::: using crs extrapolation")
                idx           = np.where(energy > self._energy[-1])[0]
                if (len(idx)>0):
                    idx0 = max(0, idx[0]-1)
                    total_cs[idx] = total_cs[idx0] * (np.log(energy[idx]) / energy[idx]) / (np.log(energy[idx0]) / energy[idx0])
            return total_cs
        elif self._cs_interp_type == CollisionInterpolationType.USE_ANALYTICAL_FUNCTION_FIT:
            return Collisions.synthetic_tcs(energy,self._col_name)  
        else:
            #print("using linear fit")
            total_cs = self._total_cs_interp1d(energy)
            if self._cs_extrapolate == True:
                print("::: using crs extrapolation")
                idx           = np.where(energy > self._energy[-1])[0]
                if (len(idx)>0):
                    idx0 = max(0, idx[0]-1)
                    total_cs[idx] = total_cs[idx0] * (np.log(energy[idx]) / energy[idx]) / (np.log(energy[idx0]) / energy[idx0])
            return total_cs

    @staticmethod
    def diff_cs_to_total_cs_ratio(energy,scattering_angle):
        return (energy)/(4 * np.pi *  (1 + energy * (np.sin(0.5*scattering_angle)**2))  *  np.log(1+energy) )

    @staticmethod
    def differential_cross_section(total_cross_section : float, energy : float, scattering_angle: float ) -> float:
        """
        computes the differential cross section from total cross section. 
        """
        # currently disabled the diff cs. 
        return total_cross_section/(4*np.pi)
        #return (energy*total_cross_section)/(4 * np.pi *  (1 + energy * (np.sin(0.5*scattering_angle)**2))  *  np.log(1+energy) )

    # @abc.abstractmethod
    # def compute_scattering_velocity(v0, polar_angle, azimuthal_angle):
    #     pass

    # @abc.abstractmethod
    # def pre_scattering_velocity_sp(vr,vt,vp, polar_angle, azimuthal_angle):
    #     pass

    # @abc.abstractmethod
    # def post_scattering_velocity_sp(vr,vt,vp, polar_angle, azimuthal_angle):
    #     pass
    
    # @abc.abstractmethod
    # def min_energy_threshold():
    #     """
    #     Gives the minimum evergy threshold for the event in ev
    #     """
    #     pass

    @staticmethod
    def synthetic_tcs(ev, mode):
        """
        synthetic cross-sections for testing. 
        """
        if mode==0:
            return 2e-20 * np.ones_like(ev)
        
        elif mode == "E + Ar -> E + Ar":
            """
            G0 cross section data fit with analytical function
            """
            ev =     ev+1e-13
            a0 =    0.008787
            b0 =     0.07243
            c  =    0.007048
            d  =      0.9737
            a1 =        3.27
            b1 =       3.679
            x0 =      0.2347
            x1 =       11.71
            y=9.900000e-20*(a1+b1*(np.log(ev/x1))**2)/(1+b1*(np.log(ev/x1))**2)*(a0+b0*(np.log(ev/x0))**2)/(1+b0*(np.log(ev/x0))**2)/(1+c*ev**d)
            assert len(y[y<0]) == 0 , "g0 cross section is negative" 
            return  y
        
        elif mode == "E + Ar -> E + E + Ar+":
            """
            G2 cross section data fit with analytical function (ionization)
            """
            y               = np.zeros_like(ev)
            threshold_value = 15.76
            y[ev>threshold_value] = (2.860000e-20/np.log(90-threshold_value)) * np.log((ev[ev>threshold_value]-threshold_value + 1)) * np.exp(-1e-2*((ev[ev>threshold_value]-90)/90)**2)
            y[ev>=10000]=0
            return  y

        elif mode == "E + Ar -> E + E + Ar+___":
            """
            G2 cross section data fit with analytical function (ionization)
            """
            #ev =     ev+1e-8
            a = 2.84284159e-22
            b = 1.02812034e-17
            c =-1.40391999e-15
            d = 9.97783291e-14
            e =-3.82647294e-12
            f =-5.70400826e+01-0.535743163918511


            trans_width = 3
            z = ((ev+1e-15)-(15.76))/(trans_width)
            # z = (np.log(ev+1e-15)-np.log(15.76))/np.log(trans_width)

            transition = np.zeros_like(z)
            gt0 = z>0
            lt1 = z<1
            bw01 = np.logical_and(gt0, lt1)
            transition[np.logical_not(gt0)] = 0
            transition[np.logical_not(lt1)] = 1
            transition[bw01] = .5*(1.+np.tanh((2*z[bw01]-1)/np.sqrt((1.-z[bw01])*z[bw01])))

            # print(transition[transition<0])
            
            x=ev-f
            y=a + b* (1/x**1) + c * (1/x**2) + d * (1/x**3) + e * (1/x**4)
            y = y * transition
            y[ev>=10000]=0
            
            # y[ev<=15.76]=0
            # y[ev>1e3]=0
            
            return  y
        
        elif mode == "E + Ar -> E + Ar(1S5)":
            """
            G1 cross section data fit with analytical function (excitation)
            """
            #ev =     ev+1e-8
            a  = -4.06265154e-21
            b=  6.46808245e-22
            c  = -3.20434420e-23
            d  = 6.39873618e-25
            e  = -4.37947887e-27
            f  = -1.30972221e-23
            g  =  2.15683845e-19
            mixing = 1./(1+np.exp(-(ev-32)))
            y = (a +  b * ev + c * ev**2 + d * ev**3 + e * ev**4)*(1-mixing) + mixing*(f + g/ev**2)
            y[ev<=11.55] = 0
            y[y<0]       = 8e-25
            y[ev>=200]   = 0
            
            return y
        else:
            raise NotImplementedError

    # @abc.abstractmethod
    # def get_cross_section_scaling():
    #     pass

    # def assemble_diff_cs_mat(self,v,chi):
    #     raise NotImplementedError
    

class CollisionType():
    # Elastic
    EAR_G0=0
    
    # Excitation
    EAR_G1=1
    
    # Ionization
    EAR_G2=2
    
    # Attachment (Recombination)
    EAR_G3=3
    

class electron_heavy_binary_collision(Collisions):
    """
    class to manage all electron heavy collissions
    """
    def __init__(self, cross_section, collision_type) -> None:
        super().__init__(cross_section)
        self.load_cross_section(cross_section)
        
        if collision_type=="ELASTIC":
            self._type = CollisionType.EAR_G0
        elif collision_type == "EXCITATION":
            self._type = CollisionType.EAR_G1
        elif collision_type == "IONIZATION":
            self._type = CollisionType.EAR_G2
        elif collision_type == "ATTACHMENT":
            self._type = CollisionType.EAR_G3
        else:
            print("[Error]: unknown collision type")
            sys.exit(0)
            



"""
e + Ar -> e + Ar
"""
class eAr_G0(Collisions):

    def __init__(self, cross_section="g0") -> None:
        super().__init__(cross_section)
        self.load_cross_section("E + Ar -> E + Ar")
        self._type = CollisionType.EAR_G0
        

"""
e + Ar -> e + Ar^*
"""
class eAr_G1(Collisions):
    
    def __init__(self, cross_section="g1") -> None:
        super().__init__(cross_section)
        self.load_cross_section("E + Ar -> E + Ar(1S5)")
        self._type=CollisionType.EAR_G1
        
    
"""
e + Ar -> e + Ar^+
"""
class eAr_G2(Collisions):
    
    def __init__(self, cross_section="g2") -> None:
        super().__init__(cross_section)
        self.load_cross_section("E + Ar -> E + E + Ar+")
        self._type=CollisionType.EAR_G2
        self._momentum_setup = False
        

def collission_cs_test():
    """
    plot the experimental and synthetic cross section differences. 
    g0 - elastic
    g1 - excitation
    g2 - inonization
    """
    
    G    = [eAr_G0(), eAr_G1(), eAr_G2()]
    mode = ["g0", "g1", "g2"]
    #tcs  = list()
    
    import matplotlib.pyplot as plt
    for i, g in enumerate(G):
        plt.figure(figsize=(8,8),dpi=300)
        #plt.plot(g._energy, g._total_cs,linewidth=1, label="lxcat")
        ev=np.linspace(0,1000,1000000)
        tcs = Collisions.synthetic_tcs(ev,mode[i])
        plt.plot(ev,tcs, linewidth=1, label="analytical")
        plt.legend()
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r'energy (eV)')
        plt.ylabel(r'cross section ($m^2$)')
        plt.grid(True, which="both", ls="-")
        plt.savefig("%s.png"%(mode[i]))
        plt.close()
    