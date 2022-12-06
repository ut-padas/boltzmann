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
from numpy.lib.function_base import diff
import cross_section
from scipy import interpolate
import scipy.constants
import utils as BEUtils
import scipy.ndimage

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
MASS_ARGON          = 6.6335209E-26 
MASS_R_EARGON       = MASS_ELECTRON/MASS_ARGON
E_AR_IONIZATION_eV  = 15.76
E_AR_EXCITATION_eV  = 11.55
ELECTRON_VOLT       = scipy.constants.electron_volt
ELECTRON_CHARGE_MASS_RATIO = ELECTRON_CHARGE/MASS_ELECTRON

BOLTZMANN_CONST     = scipy.constants.Boltzmann
TEMP_K_1EV          = ELECTRON_VOLT/BOLTZMANN_CONST
MAXWELLIAN_TEMP_K   = TEMP_K_1EV
AR_TEMP_K           = TEMP_K_1EV
AR_NEUTRAL_N        = 3.22e22 # 1/m^3
AR_IONIZED_N        = 1.00e0
MAXWELLIAN_N        = 1.00e0 # 1/m^3
ELECTRON_THEMAL_VEL = electron_thermal_velocity(MAXWELLIAN_TEMP_K) 
#http://farside.ph.utexas.edu/teaching/plasma/Plasmahtml/node6.html
PLASMA_FREQUENCY  = np.sqrt(MAXWELLIAN_N * (scipy.constants.elementary_charge**2) / (scipy.constants.epsilon_0  * scipy.constants.electron_mass))


class Collisions(abc.ABC):

    def __init__(self)->None:
        self._is_scattering_mat_assembled=False
        self._sc_direction_mat=None
        pass
    
    def load_cross_section(self,fname)->None:
        """
        build interpolant for cross section data
        """
        self._cs_fname  = fname
        self._cs_fields = ["energy", "cross section"]
        np_data         = cross_section.lxcat_cross_section_to_numpy(self._cs_fname, self._cs_fields)
        self._energy    = np_data[0]
        self._total_cs  = np_data[1]
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
        return self._total_cs_interp1d(energy)

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

    @abc.abstractmethod
    def compute_scattering_velocity(v0, polar_angle, azimuthal_angle):
        pass

    @abc.abstractmethod
    def pre_scattering_velocity_sp(vr,vt,vp, polar_angle, azimuthal_angle):
        pass

    @abc.abstractmethod
    def post_scattering_velocity_sp(vr,vt,vp, polar_angle, azimuthal_angle):
        pass
    
    @abc.abstractmethod
    def min_energy_threshold():
        """
        Gives the minimum evergy threshold for the event in ev
        """
        pass

    @staticmethod
    def synthetic_tcs(ev,mode):
        """
        synthetic cross-sections for testing. 
        """
        #print("synthetic mode : ", mode)
        if mode==0:
            return 2e-20 * np.ones_like(ev)
        elif mode==1:
            """
            high gradient increasing
            """
            x0 = 0.0
            x1 = 1

            e0 = 1.24e-21
            e1 = 2.24e-20
            
            m  = (e1-e0)/(x1-x0)
            tcs = m * (ev-x0)  + e0
            tcs[tcs<0]=0.0
            return  tcs

        elif mode==2:
            """
            low gradient increasing
            """
            x0 = 0.0
            x1 = 1

            e0 = 1.24e-20
            e1 = 1.44e-20
            
            m  = (e1-e0)/(x1-x0)
            tcs = m * (ev-x0)  + e0
            tcs[tcs<0]=0.0
            return  tcs

        elif mode == 3:
            """
            high gradient decreasing
            """
            x0 = 0.0
            x1 = 4.0

            e0 = 1.44e-20
            e1 = 0
            
            m  = (e1-e0)/(x1-x0)
            tcs=  m * (ev-x0)  + e0
            tcs[tcs<0]=0.0
            return  tcs

        elif mode == 4:
            """
            low gradient decreasing
            """
            x0 = 0.0
            x1 = 5e1

            e0 = 1.44e-20
            e1 = 0
            
            m  = (e1-e0)/(x1-x0)
            tcs=  m * (ev-x0)  + e0
            tcs[tcs<0]=0.0
            return  tcs

        elif mode==5:
            """
            high gradient kink
            """
            tcs = np.zeros_like(ev)

            x0 = 0.0
            x1 = 4.0
            x2 = 5e1

            e0 = 3.24e-20
            e1 = 1.00e-21
            e2 = 3.24e-20

            m2= (e2-e1)/(x2-x1)
            mask = ev >= x1
            tcs[mask] = e1 + (ev[mask]-x1) * m2

            mask=np.logical_not(mask)
            m1  = (e1-e0)/(x1-x0)
            tcs[mask] = e0 + (ev[mask]-x0) * m1

            #tcs[tcs<0]=0
            return tcs

        elif mode==6:
            """
            low gradient kink
            """
            tcs = np.zeros_like(ev)

            x0 = 0.0
            x1 = 4.0
            x2 = 5e1

            e0 = 3.24e-20
            e1 = 1.00e-20
            e2 = 3.24e-20

            m2= (e2-e1)/(x2-x1)
            mask = ev >= x1
            tcs[mask] = e1 + (ev[mask]-x1) * m2

            mask=np.logical_not(mask)
            m1  = (e1-e0)/(x1-x0)
            tcs[mask] = e0 + (ev[mask]-x0) * m1

            #tcs[tcs<0]=0
            return tcs

        elif mode==7:
            """
            low gradient kink
            """
            tcs = np.zeros_like(ev)

            x0 = 0.0
            x1 = 4.0
            x2 = 1e2

            e0 = 3.24e-20
            e1 = 1.00e-20
            e2 = 3.24e-20

            m2= (e2-e1)/(x2-x1)
            mask = ev >= x1
            tcs[mask] = e1 + (ev[mask]-x1) * m2

            mask=np.logical_not(mask)
            m1  = (e1-e0)/(x1-x0)
            tcs[mask] = e0 + (ev[mask]-x0) * m1

            #tcs[tcs<0]=0
            return tcs

        elif mode==8:
            """
            quadratic smooth
            """
            tcs = np.zeros_like(ev)
            
            x0  = 0.2 
            e0  = 5.20e-26
            tcs = 3.22e-18 * (ev-x0)**2 -3.22e-24 * (ev-x0)  + e0
            return tcs

        elif mode==9:
            """
            quadratic smooth
            """
            tcs = np.zeros_like(ev)
            
            x0  = 0.2 
            e0  = 5.20e-26
            tcs = 3.22e-21 * (ev-x0)**2 + e0
            return tcs

        elif mode == 11:
            """
            Cross section data from Kevin. 
            """
            # Rydberg energy
            R0 = 13.605693122994  # eV
            # Bohr radius
            a0 = 5.29177e-11     # m
            # electron charge
            qe = 1.60217662e-19  # C
            # electron mass
            me = 9.10938356e-31  # kg
            # speed of light
            c0 = 2.99792458e8    # m/s
            # mc2 in eV
            mc2 = me * c0 * c0 / qe
            # hbar = h / 2pi
            hbar = 6.62607004e-34 / 2.0 / np.pi   # m2 kg / s
            # static polarizability
            alpha = 11.08     # a0^3
            # E (eV) from k^2 (a0^(-2))
            Efromk2 = hbar * hbar / 2.0 / me / qe / a0 / a0
            theta = [-1.4173003748675717, 62.56060131992609, -85.20199010031428, 0.7361486757602836, 0.4463721552135573, 1.6509904643360198, 2.001074057955869]
            def elastic_shifted_MERT(theta,E_input,N=10):
                NE = len(E_input)
                E  = E_input#np.copy(E_input)

                A, D, F, E1, t1, t2, t3 = theta

                E *= 0.5*(1.+t1) - 0.5 * (1. - t1) * np.tanh( (E - t2) / t3 )

                k = np.sqrt( E / Efromk2 ) # wavenumber in a0^(-1)
                crs = np.zeros((NE,))

                #     A, D, F, E1 = theta
                eta0 = - A * ( 1. + 4. / 3. * alpha * k * k * np.log(k) ) - np.pi / 3. * alpha * k + D * k**2 + F * k**3
                eta0 = np.arctan(eta0 * k)

                eta1 = np.pi / 15. * alpha * k * ( 1. - np.sqrt(E/E1) )
                eta1 = np.arctan(eta1 * k)

                crs += np.sin(eta0 - eta1)**2

                for L in range(1,N):
                    eta0 = np.copy(eta1)
                    L1 = L+1
                    eta1 = np.pi * alpha * k / (2.*L1 + 3.) / (2.*L1 + 1.) / (2.*L1 - 1.)
                    eta1 = np.arctan(eta1 * k)

                    crs += (L + 1.) * np.sin(eta0 - eta1)**2

                return crs * 4. * np.pi / k / k * a0 * a0

            return elastic_shifted_MERT(theta,ev)

        elif mode == "g0":
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
        
        elif mode == "g1":
            """
            G1 cross section data fit with analytical function (excitation)
            """
            #ev =     ev+1e-8
            a  = -4.06265154e-21
            b  =  6.46808245e-22
            c  = -3.20434420e-23
            d  = 6.39873618e-25 
            e  = -4.37947887e-27
            f  = -1.30972221e-23
            g  =  2.15683845e-19
            y = a +  b * ev**1 + c * ev**2 + d * ev**3 + e * ev **4 
            y[ev<=11.55] = 0 
            y[ev>35.00] = f + g * (1/pow(ev[ev>35.00],2))
            y[ev>=200]   = 0
            return y     

        elif mode == "g1smoother":
            """
            G1 cross section data fit with analytical function (excitation)
            """
            #ev =     ev+1e-8
            a  = -4.06265154e-21
            b  =  6.46808245e-22
            c  = -3.20434420e-23
            d  = 6.39873618e-25 
            e  = -4.37947887e-27
            f  = -1.30972221e-23
            g  =  2.15683845e-19
            mixing = 1./(1.+np.exp(-(ev-32.)))
            y = (a +  b * ev + c * ev**2 + d * ev**3 + e * ev**4)*(1.-mixing) + mixing*(f + g/ev**2)
            y[ev<=11.55] = 0 
            # y[ev>=200]   = 0
            return y    
        
        elif mode == "g2":
            """
            G2 cross section data fit with analytical function (ionization)
            """
            y=np.zeros_like(ev)
            y[ev>15.76] =(2.860000e-20/np.log(90-15.76)) * np.log((ev[ev>15.76]-15.76 + 1)) * np.exp(-1e-2*((ev[ev>15.76]-90)/90)**2)
            # y[ev>1000]=0
            # print(y[y<0])
            return  y

        elif mode == "g2step":
            """
            G2 cross section data fit with analytical function (ionization)
            """
            y=np.zeros_like(ev)
            y[ev>15.76]=5e-21
            #(2.860000e-20/np.log(90-15.76)) * np.log((ev[ev>15.76]-15.76 + 1))
            # a= 15.76
            # b= 1000.76

            # si       = np.logical_and(ev>=a, ev<=b)
            # x0 = (ev[si]-a)/(b-a)
            # y[si]    = 2.860000e-17 * (3 * (x0)**2 - 2 * (x0)**3)
            # y[ev>=b] = 2.860000e-17
            #y[ev>1000]=0
            #print(y)

            # y[ev>1e3]=0
            return  y      
        
        elif mode == "g2smoothstep":
            """
            G2 cross section data fit with analytical function (ionization)
            """
            y=np.zeros_like(ev)
            a= 15.76
            b= 50.76

            si       = np.logical_and(ev>=a, ev<=b)
            x0 = (ev[si]-a)/(b-a)
            y[si]    = 5e-20 * (6 * (x0)**5 - 15 * x0 **4  + 10 * (x0)**3)
            y[ev>=b] = 5e-20
            return  y     

        elif mode == "g2Smooth":
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


            trans_width = 50
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
            y[y<=0] = 0
            
            # y[ev<=15.76]=0
            # y[ev>1e3]=0
            
            return  y    
        
        elif mode == "g2Regul":
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
            y[y<=0] = 0
            
            # y[ev<=15.76]=0
            # y[ev>1e3]=0
            
            return  y      

        elif mode == "g2Const":

            return  9.9e-21 * np.ones_like(ev)
            
        elif mode == "g0Const" or mode=="g0ConstNoLoss":
            """
            G0 cross section data fit with analytical function
            (constant)
            """
            return np.ones_like(ev) * 9.9e-20
            #ev = ev+1e-8
            # a0 =    0.008787*0 + 1
            # b0 =     0.07243
            # c  =    0.007048*0
            # d  =      0.9737
            # a1 =        3.27*0 + 1
            # b1 =       3.679
            # x0 =      0.2347
            # x1 =       11.71
            # return  9.900000e-20*(a1+b1*(np.log(ev/x1))**2)/(1+b1*(np.log(ev/x1))**2)*(a0+b0*(np.log(ev/x0))**2)/(1+b0*(np.log(ev/x0))**2)/(1+c*ev**d)

        elif mode == "g0ConstDecay":
            """
            G0 cross section data fit with analytical function
            (constant + decay)
            """
            ev=ev+1e-8
            a0 =    0.008787*0 + 1
            b0 =     0.07243
            c  =    0.007048*1
            d  =      0.9737
            a1 =        3.27*0 + 1
            b1 =       3.679
            x0 =      0.2347
            x1 =       11.71
            return  9.900000e-20*(a1+b1*(np.log(ev/x1))**2)/(1+b1*(np.log(ev/x1))**2)*(a0+b0*(np.log(ev/x0))**2)/(1+b0*(np.log(ev/x0))**2)/(1+c*ev**d)

        elif mode == "g0ConstBumpDown":
            """
            G0 cross section data fit with analytical function
            (constant + bump down)
            """
            ev=ev+1e-8
            a0 =    0.008787*1 + 0
            b0 =     0.07243
            c  =    0.007048*0
            d  =      0.9737
            a1 =        3.27*0 + 1
            b1 =       3.679
            x0 =      0.2347
            x1 =       11.71
            return  9.900000e-20*(a1+b1*(np.log(ev/x1))**2)/(1+b1*(np.log(ev/x1))**2)*(a0+b0*(np.log(ev/x0))**2)/(1+b0*(np.log(ev/x0))**2)/(1+c*ev**d)

        elif mode == "g0ConstBumpUp":
            """
            G0 cross section data fit with analytical function
            (constant + bump up)
            """
            ev=ev+1e-8
            a0 =    0.008787*0 + 1
            b0 =     0.07243
            c  =    0.007048*0
            d  =      0.9737
            a1 =        3.27*0.5 + 0
            b1 =       3.679
            x0 =      0.2347
            x1 =       11.71
            return  9.900000e-20*(a1+b1*(np.log(ev/x1))**2)/(1+b1*(np.log(ev/x1))**2)*(a0+b0*(np.log(ev/x0))**2)/(1+b0*(np.log(ev/x0))**2)/(1+c*ev**d)

        
    @abc.abstractmethod
    def get_cross_section_scaling():
        pass

    def assemble_diff_cs_mat(self,v,chi):
        raise NotImplementedError
    

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

    def __init__(self, cross_section="g0") -> None:
        super().__init__()
        self.load_cross_section("lxcat_data/eAr_Elastic.txt")
        self._type=CollisionType.EAR_G0
        self._analytic_cross_section_type = cross_section

    @staticmethod
    def compute_scattering_velocity(v0, polar_angle, azimuthal_angle):

        v1_dir   = Collisions.compute_scattering_direction(v0,polar_angle, azimuthal_angle)
        vel_fac  = np.linalg.norm(v0,2)
        v1       = vel_fac  * v1_dir
        return v1 

    def pre_scattering_velocity_sp(self, vr,vt,vp, polar_angle, azimuthal_angle):
        vs    = self.compute_scattering_direction_sp(vr,vt,vp,polar_angle,azimuthal_angle)
        vs[0] = vr
        return vs

    def post_scattering_velocity_sp(self,vr,vt,vp, polar_angle, azimuthal_angle):
        vs    = self.compute_scattering_direction_sp(vr,vt,vp,polar_angle,azimuthal_angle)
        vs[0] = vr
        return vs

    @staticmethod
    def min_energy_threshold():
        return 0.0

    @staticmethod
    def get_cross_section_scaling():
        return 1.0
    
    def assemble_diff_cs_mat(self,v,chi):
        """
        computes the differential cross section matrix. 
        v   : np array of velocity of electron particles
        chi : np array of scattering angles
        Note!! : If the energy threshold is not satisfied diff. cross section would be zero. 
        """
        energy_ev = (0.5 * MASS_ELECTRON * (v**2))/ELECTRON_VOLT
        #total_cs  = self._total_cs_interp1d(energy_ev)
        #diff_cs   = total_cs #(total_cs*energy_ev)/(4 * np.pi * (1 + energy_ev * (np.sin(0.5*chi))**2 ) * np.log(1+energy_ev) )
        total_cs   = Collisions.synthetic_tcs(energy_ev, self._analytic_cross_section_type)
        diff_cs    = total_cs / (4*np.pi)
        return diff_cs

"""
e + Ar -> e + Ar
"""
class eAr_G0(Collisions):

    def __init__(self, cross_section="g0") -> None:
        super().__init__()
        self.load_cross_section("lxcat_data/eAr_Elastic.txt")
        self._type=CollisionType.EAR_G0
        self._v_scale=None
        self._analytic_cross_section_type = cross_section

    @staticmethod
    def compute_scattering_velocity(v0, polar_angle, azimuthal_angle):
        v1_dir   = Collisions.compute_scattering_direction(v0,polar_angle, azimuthal_angle)
        vel_fac  = np.linalg.norm(v0,2) * np.sqrt(1- 2*MASS_R_EARGON*(1-np.cos(polar_angle)))
        v1       = vel_fac  * v1_dir
        return v1 
    
    def post_scattering_velocity_sp(self,vr,vt,vp, polar_angle, azimuthal_angle):
        self._v_scale = np.sqrt(1- 2*MASS_R_EARGON*(1-np.cos(polar_angle)))
        vs       = self.compute_scattering_direction_sp(vr,vt,vp,polar_angle,azimuthal_angle)
        vel_fac  = vr * self._v_scale #-for the time disable the energy loss. 
        vs[0]    = vel_fac
        return vs

    def pre_scattering_velocity_sp(self,vr,vt,vp, polar_angle, azimuthal_angle):
        self._v_scale = np.sqrt(1- 2*MASS_R_EARGON*(1-np.cos(polar_angle)))
        vs       = self.compute_scattering_direction_sp(vr,vt,vp,polar_angle,azimuthal_angle)
        vel_fac  = vr / self._v_scale
        vs[0]    = vel_fac
        return vs

    @staticmethod
    def min_energy_threshold():
        return 0.0

    @staticmethod
    def get_cross_section_scaling():
        return 1.0
    
    def assemble_diff_cs_mat(self,v,chi):
        """
        computes the differential cross section matrix. 
        v   : np array of velocity of electron particles
        chi : np array of scattering angles
        Note!! : If the energy threshold is not satisfied diff. cross section would be zero. 
        """
        energy_ev = (0.5 * MASS_ELECTRON * (v**2))/ELECTRON_VOLT
        #total_cs  = self._total_cs_interp1d(energy_ev)
        #diff_cs   = total_cs #(total_cs*energy_ev)/(4 * np.pi * (1 + energy_ev * (np.sin(0.5*chi))**2 ) * np.log(1+energy_ev) )
        # total_cs   = Collisions.synthetic_tcs(energy_ev,"g0")
        total_cs   = Collisions.synthetic_tcs(energy_ev, self._analytic_cross_section_type)
        diff_cs    = total_cs / (4*np.pi)
        return diff_cs

"""
e + Ar -> e + Ar^*
"""
class eAr_G1(Collisions):
    
    def __init__(self, cross_section="g1", threshold=E_AR_EXCITATION_eV) -> None:
        super().__init__()
        self.load_cross_section("lxcat_data/eAr_Excitation.txt")
        self._type=CollisionType.EAR_G1
        self._analytic_cross_section_type = cross_section
        self._reaction_threshold = threshold

    @staticmethod
    def compute_scattering_velocity(v0, polar_angle, azimuthal_angle):
        v1_dir   = Collisions.compute_scattering_direction(v0,polar_angle, azimuthal_angle)
        assert (np.linalg.norm(v0,2) **2    - (2 * E_AR_EXCITATION_eV * ELECTRON_VOLT/MASS_ELECTRON)) > 0 , "collision G1 invalid velocity specified: %f "%v0
        vel_fac  = np.sqrt(np.linalg.norm(v0,2) **2    - (2 * E_AR_EXCITATION_eV * ELECTRON_VOLT/MASS_ELECTRON))
        v1       =  vel_fac  * v1_dir
        return v1

    def pre_scattering_velocity_sp(self,vr,vt,vp, polar_angle, azimuthal_angle):
        vs       = self.compute_scattering_direction_sp(vr,vt,vp,polar_angle,azimuthal_angle)
        #pre collision velocity. 
        vel_fac  = np.sqrt(vr**2    + (2 * E_AR_EXCITATION_eV * ELECTRON_VOLT/MASS_ELECTRON))
        vs[0]    = vel_fac
        return vs

    def post_scattering_velocity_sp(self,vr,vt,vp, polar_angle, azimuthal_angle):
        vs       = self.compute_scattering_direction_sp(vr,vt,vp,polar_angle,azimuthal_angle)
        # post collision velocity. - Note : the total cross-section for Ein < E_threshold is zero, this is
        # to avoid geting complex numbers in the computations. 
        check1   = vr**2 > (2 * E_AR_EXCITATION_eV * ELECTRON_VOLT/MASS_ELECTRON)
        vel_fac  = np.sqrt(vr[check1]**2    - (2 * E_AR_EXCITATION_eV * ELECTRON_VOLT/MASS_ELECTRON))
        vs[0][np.logical_not(check1)]=0
        vs[0][check1] = vel_fac
        return vs

    # @staticmethod
    def min_energy_threshold(self):
        return self._reaction_threshold

    @staticmethod
    def get_cross_section_scaling():
        return 1.0

    def assemble_diff_cs_mat(self,v,chi):
        """
        computes the differential cross section matrix. 
        v   : np array of velocity of electron particles
        chi : np array of scattering angles
        Note!! : If the energy threshold is not satisfied diff. cross section would be zero. 
        """
        energy_ev = (0.5 * MASS_ELECTRON * (v**2))/ELECTRON_VOLT
        #total_cs  = self._total_cs_interp1d(energy_ev)
        #diff_cs   = total_cs #(total_cs*energy_ev)/(4 * np.pi * (1 + energy_ev * (np.sin(0.5*chi))**2 ) * np.log(1+energy_ev) )
        total_cs   = Collisions.synthetic_tcs(energy_ev, self._analytic_cross_section_type)
        diff_cs    = total_cs / (4*np.pi)
        return diff_cs

"""
e + Ar -> e + Ar^+
"""
class eAr_G2(Collisions):
    
    def __init__(self, cross_section="g2", threshold=E_AR_IONIZATION_eV) -> None:
        super().__init__()
        self.load_cross_section("lxcat_data/eAr_Ionization.txt")
        self._type=CollisionType.EAR_G2
        self._momentum_setup = False
        self._analytic_cross_section_type = cross_section
        self._reaction_threshold = threshold
        
    def compute_scattering_velocity(self, v0, polar_angle, azimuthal_angle):
        v1_dir   = Collisions.compute_scattering_direction(v0,polar_angle, azimuthal_angle)
        
        assert (0.5 * np.linalg.norm(v0,2)**2    - (self._reaction_threshold * ELECTRON_VOLT/MASS_ELECTRON)) > 0 , "collision G2 invalid velocity specified: %f "%v0
        v1_fac   = np.sqrt(0.5 * np.linalg.norm(v0,2)**2    - (self._reaction_threshold * ELECTRON_VOLT/MASS_ELECTRON))
        v2_fac   = v1_fac

        v1       =  v1_fac  * v1_dir
        v2_dir   =  (v0-v1)/np.linalg.norm(v0-v1,2)
        v2       =  v2_fac * v2_dir

        return v1,v2

    def pre_scattering_velocity_sp(self,vr,vt,vp, polar_angle, azimuthal_angle):
        raise NotImplementedError
        return None
        
    def post_scattering_velocity_sp(self,vr,vt,vp, polar_angle, azimuthal_angle):
        c_gamma            = np.sqrt(2*ELECTRON_CHARGE_MASS_RATIO)
        vs                 = self.compute_scattering_direction_sp(vr,vt,vp,polar_angle,azimuthal_angle)
        check_1            = (vr/c_gamma)**2 > self._reaction_threshold
        vs[0][check_1]     = c_gamma * np.sqrt(0.5*((vr[check_1]/c_gamma)**2  - self._reaction_threshold))
        
        vs[0][np.logical_not(check_1)] = 0
        vs[1][np.logical_not(check_1)] = 0
        vs[2][np.logical_not(check_1)] = 0
        
        e_out = (vr/c_gamma)**2
        print("g2 energy in = (%.4E, %.4E)"%(np.min(e_out), np.max(e_out)))
        
        e_out = (vs[0]/c_gamma)**2
        print("g2 energy out = (%.4E, %.4E)"%(np.min(e_out), np.max(e_out)))
        
        return vs

    # @staticmethod
    def min_energy_threshold(self):
        return self._reaction_threshold

    @staticmethod
    def get_cross_section_scaling():
        return 1.0#AR_IONIZED_N

    def reset_scattering_direction_sp_mat(self):
        self._is_scattering_mat_assembled=False
        self._sc_direction_mat=None
        self._momentum_setup=False
        return
    
    def assemble_diff_cs_mat(self,v,chi):
        """
        computes the differential cross section matrix. 
        v   : np array of velocity of electron particles
        chi : np array of scattering angles
        Note!! : If the energy threshold is not satisfied diff. cross section would be zero. 
        """
        energy_ev = (0.5 * MASS_ELECTRON * (v**2))/ELECTRON_VOLT
        #total_cs  = self._total_cs_interp1d(energy_ev)
        #diff_cs   = total_cs #(total_cs*energy_ev)/(4 * np.pi * (1 + energy_ev * (np.sin(0.5*chi))**2 ) * np.log(1+energy_ev) )
        total_cs   = Collisions.synthetic_tcs(energy_ev, self._analytic_cross_section_type)
        diff_cs    = total_cs / (4*np.pi)
        return diff_cs



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
    