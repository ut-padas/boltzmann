import closure_models
import numpy as np
import spec_spherical as sp
import utils as bte_utils
import scipy.constants
import enum
# temporary 1d-bte code. we should replace this by a seperate standalone 1dbte solver that is decoupled from the glow1d application
import glowdischarge_boltzmann_1d
import mesh
import collisions
import matplotlib.pyplot as plt

class params():
    def __init__(self):
        self.num_moments = 4
        self.mfuncs      = []
        self.ts_type     = "RK4"        
        self.dt          = 1e-2         # []
        self.f           = 13.56e6      # Hz
        self.tau         = 1/self.f     # s
        self.L           = 2.54/1e2/2   # m
        self.Tg          = 300          # K
        self.p0          = 1            # Torr

        if self.Tg !=0: 
            self.n0    = self.p0 * scipy.constants.torr / (scipy.constants.Boltzmann * self.Tg) #m^{-3}
        else:
            self.n0    = 3.22e22                   # m^{-3}
        
        self.np0       = 8e16                      # "nominal" electron density [1/m^3]
        self.V0        = 100                       # V


class closure_type(enum.Enum):
    MAX_ENTROPY = 0
    BTE_0D3V    = 1

class gm_boundary_conditions(enum.Enum):
    ALL_MAXWELLIAN_FLUX              = 0
    ALL_MAXWELLIAN_FLUX_TE_DIRICHLET = 1


class generalized_moments():
    
    def __init__(self):
        self.params      = glowdischarge_boltzmann_1d.args_parse()
        self.bte_1d3v    = glowdischarge_boltzmann_1d.glow1d_boltzmann(self.params)
        col_list         = self.bte_1d3v.coll_list
        Te0              = self.params.Te
        self.xp_module   = np
        xp               = self.xp_module
        self.spec_sp     = self.bte_1d3v.op_spec_sp
        
        self.num_vr      = (self.spec_sp._basis_p._num_knot_intervals) * self.params.spline_qpts
        self.num_vt      = 32
        self.num_vp      = 4

        vth              = self.bte_1d3v.bs_vth
        q                = self.bte_1d3v.param
        c_gamma          = np.sqrt(2 * scipy.constants.elementary_charge / scipy.constants.electron_mass)

        scale_Te         = (vth**2) * (2/3/c_gamma**2) #(vth**2) * 0.5 * scipy.constants.electron_mass * (2/3/scipy.constants.Boltzmann) / collisions.TEMP_K_1EV

        self.bte_params  = self.bte_1d3v.param

        self.MIDX_MASS   = 0
        self.MIDX_MOM    = 1
        self.MIDX_ENERGY = 2

        self.bc_type     = None #gm_boundary_conditions.ALL_MAXWELLIAN_FLUX
        

        self.mfuncs      = [ lambda vr, vt, vp : np.ones_like(vr),
                             lambda vr, vt, vp : vth    * vr * np.cos(vt),
                             lambda vr, vt, vp : vth**2 * vr **2 * (2/3/c_gamma**2),
                             #lambda vr, vt, vp : q.n0 * q.np0  * vth * vr * np.einsum("k,l,m->klm", col_list[1].total_cross_section(np.unique(vr) **2 * Te0), np.unique(vt), xp.unique(vp))
                            ]
        
        self.mfuncs_Jx  = [lambda vr, vt, vp:  vth * vr * np.cos(vt)  * np.ones_like(vr), 
                           lambda vr, vt, vp:  vth * vr * np.cos(vt)  * vth    * vr * np.cos(vt),
                           lambda vr, vt, vp:  vth * vr * np.cos(vt)  * vth**2 * vr **2 * (2/3/c_gamma**2),
                           #lambda vr, vt, vp : vth * vr * np.cos(vt)  * q.n0 * q.np0 * vth * vr * np.einsum("k,l,m->klm", col_list[1].total_cross_section(np.unique(vr) **2 * Te0), np.unique(vt), xp.unique(vp))
                           ]

        self.mfuncs_ops    = bte_utils.assemble_moment_ops(self.spec_sp, self.num_vr, self.num_vt, self.num_vp, self.mfuncs   , scale=1.0)
        self.mfuncs_Jx_ops = bte_utils.assemble_moment_ops(self.spec_sp, self.num_vr, self.num_vt, self.num_vp, self.mfuncs_Jx, scale=1.0)

        self.mfuncs_Jx_ops_l = bte_utils.assemble_moment_ops(self.spec_sp, self.num_vr, self.num_vt, self.num_vp, self.mfuncs_Jx, 
                                                             scale=1.0, dvt=(0.5 * np.pi, np.pi))
        
        self.mfuncs_Jx_ops_r = bte_utils.assemble_moment_ops(self.spec_sp, self.num_vr, self.num_vt, self.num_vp, self.mfuncs_Jx, 
                                                             scale=1.0, dvt=(0, 0.5 * np.pi))
        # print(self.mfuncs_Jx_ops.shape)
        # print(self.mfuncs_Jx_ops)


        self.mesh          = mesh.mesh(tuple([self.params.Np]), 1, mesh.grid_type.CHEBYSHEV_COLLOC)
        assert (self.mesh.xcoord[0] == self.bte_1d3v.xp).all() == True

        self.efield        = lambda t : xp.ones_like(self.mesh.xcoord[0]) * 1e3 * xp.cos(2 * xp.pi * t)
        self.cmodel_type   = closure_type.MAX_ENTROPY
        
    def init(self):
        xp   = self.xp_module 
        u, v = self.bte_1d3v.initialize()
        
        Po          = self.bte_1d3v.op_psh2o
        Ps          = self.bte_1d3v.op_po2sh
        
        v[self.bte_1d3v.xp_vt_l, 0 ] = 0.0
        v[self.bte_1d3v.xp_vt_r, -1] = 0.0
        
        self.mae_m0 = xp.zeros((len(self.mfuncs), self.bte_1d3v.Np))
        
        m                       = self.mfuncs_ops @ (Ps @ v)
        
        return m

    def closure_model(self, m_vec: np.array, E: np.array, type:closure_type):
        xp         = self.xp_module
        spec_sp    = self.bte_1d3v.op_spec_sp
        atol, rtol = 1e-12, 1e-8
        if (type == closure_type.MAX_ENTROPY):
            m0 = xp.copy(m_vec)
            m0 = m0 / (m0[self.MIDX_MASS][xp.newaxis, :])
            
            #self.mae_m0[:, :]             = 0
            #self.mae_m0[self.MIDX_ENERGY] = -(1/xp.sqrt(m_vec[self.MIDX_ENERGY]/m_vec[self.MIDX_MASS])/c_gamma)**2
            #w, fklm = closure_models.max_entropy_reconstruction(spec_sp, self.mae_m0, self.num_vr, self.num_vt, self.num_vp, 
                                                      #m_vec, self.mfuncs, xp, rtol, atol, 100)
            
            w, fklm = closure_models.max_entropy_reconstruction(spec_sp, self.mae_m0, self.num_vr, self.num_vt, self.num_vp, 
                                                      m0, self.mfuncs, xp, rtol, atol, 100)
            
            fklm        = fklm * m_vec[self.MIDX_MASS]
            self.mae_m0 = w
            #print(fklm)
            # ev = xp.linspace(0, 100, 1000)

            # rc_comp = bte_utils.compute_radial_components(ev, self.bte_1d3v.op_spec_sp, fklm[:,10], 
            #                                     bte_utils.get_maxwellian_3d(self.bte_1d3v.bs_vth), self.bte_1d3v.bs_vth, 1)
            
            # plt.subplot(1, 3, 1)
            # plt.semilogy(ev, rc_comp[0])

            # plt.subplot(1, 3, 2)
            # plt.semilogy(ev, rc_comp[1])

            # plt.subplot(1, 3, 3)
            # plt.semilogy(ev, rc_comp[2])

            # plt.show()

            # fklm                            = self.bte_1d3v.op_psh2o @ fklm
            # fklm[self.bte_1d3v.xp_vt_l, 0 ] = 0.0
            # fklm[self.bte_1d3v.xp_vt_r, -1] = 0.0
            # fklm                            = self.bte_1d3v.op_po2sh @ fklm

            
            q   = self.bte_params
            Cop = (self.bte_1d3v.op_col_en + self.params.Tg * self.bte_1d3v.op_col_gT) * q.n0 * q.np0 
            Av  = (self.bte_1d3v.op_adv_v) 

            mJx          = xp.zeros_like(m_vec)
            mJx[:, 1:-1] = (q.tau/q.L) * self.mfuncs_Jx_ops @ fklm[:, 1:-1]
            
            # BCs. 
            mJx[:,  0]   = (q.tau/q.L) * self.mfuncs_Jx_ops_l @ fklm[:,  0]
            mJx[:, -1]   = (q.tau/q.L) * self.mfuncs_Jx_ops_r @ fklm[:, -1]

            mJv = (q.tau)     * (self.mfuncs_ops @ (Cop @ fklm + E * (Av @ fklm))) #* 0.0

            return mJx, mJv
        else:
            raise NotImplementedError
        
    def rhs(self, x, time, dt):
        xp            = self.xp_module
        E             = self.efield(time)
        c_gamma       = xp.sqrt(2 * scipy.constants.elementary_charge / scipy.constants.electron_mass)

        
        print(x)
        #print(x.shape)
        Jx, Jv = self.closure_model(x, E, self.cmodel_type)

        #Jx_l, Jx_r   = Jx[:, 0], Jx[:, -1]
        #Jx_l[Jx_l>0] = 0.0
        #Jx_r[Jx_r<0] = 0.0

        q             = self.bte_params
        
        if (self.bc_type == gm_boundary_conditions.ALL_MAXWELLIAN_FLUX):
            Teb0          = x[self.MIDX_ENERGY, 0] / x[self.MIDX_MASS, 0]
            Teb1          = x[self.MIDX_ENERGY,-1] / x[self.MIDX_MASS,-1]
            vb0           = (xp.sqrt(Teb0) * c_gamma) * (q.tau/q.L) # non-dimensionalized themeral velocity corresponds to maxwellian eedf
            vb1           = (xp.sqrt(Teb1) * c_gamma) * (q.tau/q.L) # non-dimensionalized themeral velocity corresponds to maxwellian eedf
            sqrt_pi       = (xp.pi**0.5)

            # maxwellian flux on a wall 
            Jx[self.MIDX_MASS,  0]   = -(0.5 / sqrt_pi) * vb0 * x[self.MIDX_MASS,  0]
            Jx[self.MIDX_MASS, -1]   =  (0.5 / sqrt_pi) * vb1 * x[self.MIDX_MASS, -1]

            Jx[self.MIDX_MOM,  0]    = 0.5 * sqrt_pi * vb0 * Jx[self.MIDX_MASS,  0] 
            Jx[self.MIDX_MOM, -1]    = 0.5 * sqrt_pi * vb1 * Jx[self.MIDX_MASS, -1] 

            Jx[self.MIDX_ENERGY,  0] = 2 * vb0**2 * Jx[self.MIDX_MASS,  0] 
            Jx[self.MIDX_ENERGY, -1] = 2 * vb1**2 * Jx[self.MIDX_MASS, -1] 
            
        elif (self.bc_type == gm_boundary_conditions.ALL_MAXWELLIAN_FLUX_TE_DIRICHLET):
            Teb0          = x[self.MIDX_ENERGY, 0] / x[self.MIDX_MASS, 0]
            Teb1          = x[self.MIDX_ENERGY,-1] / x[self.MIDX_MASS,-1]
            vb0           = (xp.sqrt(Teb0) * c_gamma) * (q.tau/q.L) # non-dimensionalized themeral velocity corresponds to maxwellian eedf
            vb1           = (xp.sqrt(Teb1) * c_gamma) * (q.tau/q.L) # non-dimensionalized themeral velocity corresponds to maxwellian eedf
            sqrt_pi       = (xp.pi**0.5)

            # maxwellian flux on a wall 
            Jx[self.MIDX_MASS,  0]   = -(0.5 / sqrt_pi) * vb0 * x[self.MIDX_MASS,  0]
            Jx[self.MIDX_MASS, -1]   =  (0.5 / sqrt_pi) * vb1 * x[self.MIDX_MASS, -1]

            Jx[self.MIDX_MOM,  0]    = 0.5 * sqrt_pi * vb0 * Jx[self.MIDX_MASS,  0] 
            Jx[self.MIDX_MOM, -1]    = 0.5 * sqrt_pi * vb1 * Jx[self.MIDX_MASS, -1] 
        else:
            pass
            #raise NotImplementedError

        y                       = Jv - (self.mesh.D1[0] @ Jx.T).T

        if (self.bc_type == gm_boundary_conditions.ALL_MAXWELLIAN_FLUX_TE_DIRICHLET):
            # Dirichlet BCs at boundary -- this is not physical boundary condition. Te at the wall can vary. 
            y[self.MIDX_ENERGY,  0] = (y[self.MIDX_MASS,  0] * self.bte_params.Teb0) 
            y[self.MIDX_ENERGY, -1] = (y[self.MIDX_MASS, -1] * self.bte_params.Teb1)

        return y

    def step(self, x, time, dt):

        # k1 = self.rhs(x, time, 0.5 * dt)
        # k2 = self.rhs(x + 0.5 * dt * k1, time + 0.5 * dt, dt)
        # y  = x + dt * k2

        # k1 = self.rhs(x, time, dt)
        # k2 = self.rhs(x + dt * 0.5 * k1, time + 0.5 * dt, dt)
        # k3 = self.rhs(x + dt * 0.5 * k2, time + 0.5 * dt, dt)
        # k4 = self.rhs(x + dt * 1.0 * k3, time + 1.0 * dt, dt)
        
        # y        = x + dt * ((1/6) * k1 + (1/3) * k2 + (1/3) * k3 + (1/6) * k4)
        y          = x + dt * self.rhs(x, time, dt)
        # y[self.MIDX_MASS  , y[self.MIDX_MASS]<0]   =1e-10
        # y[self.MIDX_ENERGY, y[self.MIDX_ENERGY] <0]=1e-10

        # y[2,  0] = 4 * y[0,  0] 
        # y[2, -1] = 4 * y[0, -1] 
        # y[0, y[0]<0]  = 1e-10
        # y[2, y[2]<0]  = 1e-10
        idx = (y[self.MIDX_ENERGY]/y[self.MIDX_MASS]) < 0
        y [self.MIDX_ENERGY,  idx ] = 1e-4 * y[self.MIDX_MASS, idx]
        return y


if __name__ == "__main__":
    gm     = generalized_moments()
    m_gme  = gm.init()

    #print(m0)
    params = gm.params
    
    T      = params.cycles
    dt     = params.cfl
    xx     = gm.mesh.xcoord[0]

    
    tt     = 0.0
    idx    = 0

    bte_solver = gm.bte_1d3v
    bte_solver.initialize_bte_adv_x(0.5 * dt)
    
    u, v                      = bte_solver.initialize()
    v[bte_solver.xp_vt_l, 0 ] = 0.0
    v[bte_solver.xp_vt_r, -1] = 0.0

    bte_solver.step_init(u, v, dt)
    mass_op    = bte_solver.op_mass
    temp_op    = bte_solver.op_temp

    xp         = gm.xp_module
    io_freq    = int(bte_solver.args.io_cycle_freq / dt)
    c_gamma    = np.sqrt(2 * scipy.constants.elementary_charge/ scipy.constants.electron_mass)
    

    while tt < T:
        m_bte = gm.mfuncs_ops @ (bte_solver.op_po2sh @ v)
        
        #print(tt, m_gme)

        if (idx % io_freq == 0):
            plt.figure(figsize=(12, 8), dpi=200)
            plt.subplot(1, 3, 1)
            plt.plot(xx, m_gme[gm.MIDX_MASS] * gm.bte_params.np0, label=r"GME")
            plt.plot(xx, m_bte[gm.MIDX_MASS] * gm.bte_params.np0, label=r"BTE")
            plt.xlabel(r"$\hat{x}$")
            plt.ylabel(r"$n_e [m^-3]$")
            plt.legend()
            plt.grid(visible=True)
            

            plt.subplot(1, 3, 2)
            plt.plot(xx, m_gme[gm.MIDX_MOM]/m_gme[gm.MIDX_MASS], label=r"GME")
            plt.plot(xx, m_bte[gm.MIDX_MOM]/m_bte[gm.MIDX_MASS], label=r"BTE")
            plt.xlabel(r"$\hat{x}$")
            plt.ylabel(r"$v_x [ms^{-1}]$")
            plt.legend()
            plt.grid(visible=True)

            plt.subplot(1, 3, 3)
            plt.plot(xx, m_gme[gm.MIDX_ENERGY]/m_gme[gm.MIDX_MASS], label=r"GME")
            plt.plot(xx, m_bte[gm.MIDX_ENERGY]/m_bte[gm.MIDX_MASS], label=r"BTE")
            plt.xlabel(r"$\hat{x}$")
            plt.ylabel(r"$T_e [eV]$")
            plt.legend()
            plt.grid(visible=True)
            plt.suptitle("time = %.4E [T]"%(tt))

            plt.tight_layout()
            plt.savefig("%s_%04d.png"%(bte_solver.args.fname, idx//io_freq))
            plt.close()
        
        m_gme   = gm.step(m_gme, tt, dt)

        bte_solver.bs_E = gm.efield(tt)
        v               = bte_solver.step_bte_x(v, tt, 0.5 * dt)
        v               = bte_solver.step_bte_v(v, None, tt, dt, ts_type="BE", verbose=1)
        v               = bte_solver.step_bte_x(v, tt + 0.5 * dt, 0.5 * dt)
        tt  += dt
        idx +=1
        


