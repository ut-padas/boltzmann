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

class generalized_moments():
    
    def __init__(self):
        self.params     = glowdischarge_boltzmann_1d.args_parse()
        self.bte_1d3v   = glowdischarge_boltzmann_1d.glow1d_boltzmann(self.params)
        col_list        = self.bte_1d3v.coll_list
        Te0             = self.params.Te
        self.xp_module  = np
        xp              = self.xp_module
        self.spec_sp    = self.bte_1d3v.op_spec_sp
        
        self.num_vr     = (self.spec_sp._basis_p._num_knot_intervals) * self.params.spline_qpts
        self.num_vt     = 16
        self.num_vp     = 16

        vth             = self.bte_1d3v.bs_vth
        q               = self.bte_1d3v.param

        scale_Te        = (vth**2) * 0.5 * scipy.constants.electron_mass * (2/3/scipy.constants.Boltzmann) / collisions.TEMP_K_1EV

        self.mfuncs     = [ lambda vr, vt, vp : np.ones_like(vr),
                            lambda vr, vt, vp : vth * vr * np.cos(vt),
                            lambda vr, vt, vp : scale_Te * vr **2,
                            #lambda vr, vt, vp : q.n0 * q.np0 * vth * vr * np.einsum("k,l,m->klm", col_list[1].total_cross_section(np.unique(vr) **2 * Te0), np.unique(vt), xp.unique(vp))
                            ]
        
        self.mfuncs_Jx  = [lambda vr, vt, vp: vr * np.cos(vt) * m(vr, vt, vp) for m in self.mfuncs]

        self.mfuncs_ops    = bte_utils.assemble_moment_ops(self.spec_sp, self.num_vr, self.num_vt, self.num_vp, self.mfuncs   , scale=1.0)
        self.mfuncs_Jx_ops = bte_utils.assemble_moment_ops(self.spec_sp, self.num_vr, self.num_vt, self.num_vp, self.mfuncs_Jx, scale=1.0)


        self.mesh          = mesh.mesh(tuple([self.params.Np]), 1, mesh.grid_type.CHEBYSHEV_COLLOC)
        assert (self.mesh.xcoord[0] == self.bte_1d3v.xp).all() == True

        self.efield        = lambda t : xp.ones_like(self.mesh.xcoord[0]) * 1e3 * xp.cos(2 * xp.pi * t)
        self.cmodel_type   = closure_type.MAX_ENTROPY
        
    def init(self):
        xp   = self.xp_module 
        u, v = self.bte_1d3v.initialize()
        #Po   = self.bte_1d3v.op_psh2o
        Ps          = self.bte_1d3v.op_po2sh
        self.mae_m0 = xp.zeros((len(self.mfuncs), self.bte_1d3v.Np))
        return self.mfuncs_ops @ (Ps @ v)

    def closure_model(self, m_vec: np.array, E: np.array, type:closure_type):
        xp         = self.xp_module
        spec_sp    = self.bte_1d3v.op_spec_sp
        atol, rtol = 1e-12, 1e-6
        if (type == closure_type.MAX_ENTROPY):
            w, fklm = closure_models.max_entropy_reconstruction(spec_sp, self.mae_m0, self.num_vr, self.num_vt, self.num_vp, 
                                                      m_vec, self.mfuncs, xp, rtol, atol, 300)
            
            self.mae_m0 = w
            
            q   = self.bte_1d3v.param
            Cop = (self.bte_1d3v.op_col_en + self.params.Tg * self.bte_1d3v.op_col_gT) * q.n0 * q.np0 * q.tau
            Av  = (self.bte_1d3v.op_adv_v) * q.tau

            mJx = (self.bte_1d3v.bs_vth * q.tau/q.L) * self.mfuncs_Jx_ops @ fklm
            mJv = (self.mfuncs_ops @ (Cop @ fklm + E * (Av @ fklm))) #* 0.0

            return mJx, mJv
        else:
            raise NotImplementedError
        
    def rhs(self, x, time, dt):
        xp     = self.xp_module
        E      = self.efield(time)
        Jx, Jv = self.closure_model(x, E, self.cmodel_type)
        
        return Jv - (self.mesh.D1[0] @ Jx.T).T

    def step(self, x, time, dt):
        k1 = self.rhs(x, time, dt)
        k2 = self.rhs(x + dt * 0.5 * k1, time + 0.5 * dt, dt)
        k3 = self.rhs(x + dt * 0.5 * k2, time + 0.5 * dt, dt)
        k4 = self.rhs(x + dt * 1.0 * k3, time + 1.0 * dt, dt)
        
        return x + dt * ((1/6) * k1 + (1/3) * k2 + (1/3) * k3 + (1/6) * k4)


if __name__ == "__main__":
    gm     = generalized_moments()
    m_gme  = gm.init()

    #print(m0)
    T      = 1.0
    tt     = 0.0
    idx    = 0
    params = gm.params
    dt     = params.cfl
    xx     = gm.mesh.xcoord[0]

    bte_solver = gm.bte_1d3v
    bte_solver.initialize_bte_adv_x(0.5 * dt)
    
    u, v       = bte_solver.initialize()
    bte_solver.step_init(u, v, dt)
    mass_op    = bte_solver.op_mass
    temp_op    = bte_solver.op_temp

    xp         = gm.xp_module
    io_freq    = int(bte_solver.args.io_cycle_freq / dt)
    
    

    while tt < T:
        m_bte = gm.mfuncs_ops @ (bte_solver.op_po2sh @ v)
        
        print(tt, m_gme, m_bte)

        if (idx % io_freq == 0):
            plt.figure(figsize=(8, 4), dpi=200)
            plt.subplot(1, 2, 1)
            plt.semilogy(xx, m_gme[0], label=r"GME")
            plt.semilogy(xx, m_bte[0], label=r"BTE")
            plt.legend()
            plt.grid(visible=True)
            

            plt.subplot(1, 2, 2)
            plt.plot(xx, m_gme[2]/m_gme[0], label=r"GME")
            plt.plot(xx, m_bte[2]/m_bte[0], label=r"BTE")
            plt.legend()
            plt.grid(visible=True)
            plt.savefig("%s_%04d.png"%(bte_solver.args.fname, idx//io_freq))
            plt.close()
        
        m_gme   = gm.step(m_gme, tt, dt)

        bte_solver.bs_E = gm.efield(tt)
        v               = bte_solver.step_bte_x(v, tt, 0.5 * dt)
        v               = bte_solver.step_bte_v(v, None, tt, dt, ts_type="BE", verbose=1)
        v               = bte_solver.step_bte_x(v, tt + 0.5 * dt, 0.5 * dt)
        tt  += dt
        idx +=1
        


