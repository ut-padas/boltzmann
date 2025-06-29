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
import sys
import h5py

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
    BTE_1D3V    = 2

class gm_boundary_conditions(enum.Enum):
    MAXWELLIAN                       = 0
    TE_DIRICHLET                     = 2


class generalized_moments():
    
    def __init__(self):
        self.params         = glowdischarge_boltzmann_1d.args_parse()
        self.bte_1d3v       = glowdischarge_boltzmann_1d.glow1d_boltzmann(self.params)
        self.vt, self.vt_w  = self.bte_1d3v.xp_vt, self.bte_1d3v.xp_vt_qw
        # print(self.vt)
        # print(self.vt_w, np.sum(self.vt_w))


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

        self.bc_type     = None #gm_boundary_conditions.TE_DIRICHLET
        
        vg, _            = closure_models.quadrature_grid(self.spec_sp, self.num_vr, self.num_vt, self.num_vp)
        uvt              = np.unique(vg[1])
        uvt_0            = xp.zeros(len(uvt))
        uvt_1            = xp.zeros(len(uvt))

        uvr              = np.ones_like(np.unique(vg[0]))
        uvp              = np.ones_like(np.unique(vg[2]))

        uvt_0[uvt < np.pi/2] = 1.0
        uvt_1[uvt > np.pi/2] = 1.0

        self.implicit_idx = []
        self.explicit_idx = [0, 1, 2]
        self.cfl_factor   = 0.25

        self.mfuncs       = [ lambda vr, vt, vp : np.ones_like(vr),
                              lambda vr, vt, vp : vth    * vr * np.cos(vt),
                              lambda vr, vt, vp : vth**2 * vr **2 * (2/3/c_gamma**2),
                             #lambda vr, vt, vp : np.einsum("k,l,m->klm", uvr, uvt_0, uvp),
                             #lambda vr, vt, vp : np.einsum("k,l,m->klm", uvr, uvt_1, uvp),
                            # lambda vr, vt, vp : vth    * vr * np.cos(vt)**2 ,
                             #lambda vr, vt, vp : vth**3 * vr**3 * (2/3/c_gamma**3),
                             #lambda vr, vt, vp : vth**4 * vr**4 * (2/3/c_gamma**4),
                             #lambda vr, vt, vp : vth**5 * vr**5 * (2/3/c_gamma**5),
                             #lambda vr, vt, vp : q.n0 * q.np0  * vth * vr * np.einsum("k,l,m->klm", col_list[0].total_cross_section(np.unique(vr) **2 * Te0), np.unique(vt), xp.unique(vp)),
                             #lambda vr, vt, vp : q.n0 * q.np0  * vth * vr * np.einsum("k,l,m->klm", col_list[1].total_cross_section(np.unique(vr) **2 * Te0), np.unique(vt), xp.unique(vp))
                             
                            ]
        
        self.mfuncs_Jx   = [lambda vr, vt, vp:  vth * vr * np.cos(vt)  * np.ones_like(vr), 
                            lambda vr, vt, vp:  vth * vr * np.cos(vt)  * vth    * vr * np.cos(vt),
                            lambda vr, vt, vp:  vth * vr * np.cos(vt)  * vth**2 * vr **2 * (2/3/c_gamma**2),
                            #lambda vr, vt, vp : vth * vr * np.cos(vt)  * np.einsum("k,l,m->klm", uvr, uvt_0, uvp),
                            #lambda vr, vt, vp : vth * vr * np.cos(vt)  * np.einsum("k,l,m->klm", uvr, uvt_1, uvp),
                           #lambda vr, vt, vp : vth * vr * np.cos(vt)  * vth    * vr * np.cos(vt)**2,
                           #lambda vr, vt, vp : vth * vr * np.cos(vt)  * vth**3 * vr**3 * (2/3/c_gamma**3),
                           #lambda vr, vt, vp : vth * vr * np.cos(vt)  * vth**4 * vr**4 * (2/3/c_gamma**4),
                           #lambda vr, vt, vp : vth * vr * np.cos(vt)  * vth**5 * vr**5 * (2/3/c_gamma**5),
                           #lambda vr, vt, vp : vth * vr * np.cos(vt)  * q.n0 * q.np0  * vth * vr * np.einsum("k,l,m->klm", col_list[0].total_cross_section(np.unique(vr) **2 * Te0), np.unique(vt), xp.unique(vp)),
                           #lambda vr, vt, vp : vth * vr * np.cos(vt)  * q.n0 * q.np0  * vth * vr * np.einsum("k,l,m->klm", col_list[1].total_cross_section(np.unique(vr) **2 * Te0), np.unique(vt), xp.unique(vp))
                           ]

        self.mfuncs_ops         = bte_utils.assemble_moment_ops(self.spec_sp, self.num_vr, self.num_vt, self.num_vp, self.mfuncs   , scale=1.0)
        self.mfuncs_ops_ords    = bte_utils.assemble_moment_ops_ords(self.spec_sp, self.vt, self.vt_w, self.num_vr, self.num_vp, self.mfuncs, scale=1.0)


        self.mfuncs_Jx_ops      = bte_utils.assemble_moment_ops(self.spec_sp, self.num_vr, self.num_vt, self.num_vp, self.mfuncs_Jx, scale=1.0)
        self.mfuncs_Jx_ops_ords = bte_utils.assemble_moment_ops_ords(self.spec_sp, self.vt, self.vt_w, self.num_vr, self.num_vp, self.mfuncs_Jx, scale=1.0)

        
        
        
        
        self.mfuncs_Jx_ops_l = bte_utils.assemble_moment_ops(self.spec_sp, self.num_vr, self.num_vt, self.num_vp, self.mfuncs_Jx, 
                                                             scale=1.0, dvt=(0.5 * np.pi, np.pi))
        self.mfuncs_Jx_ops_r = bte_utils.assemble_moment_ops(self.spec_sp, self.num_vr, self.num_vt, self.num_vp, self.mfuncs_Jx, 
                                                             scale=1.0, dvt=(0, 0.5 * np.pi))
        # print(self.mfuncs_Jx_ops.shape)
        # print(self.mfuncs_Jx_ops)


        self.mesh          = mesh.mesh(tuple([self.params.Np]), 1, mesh.grid_type.CHEBYSHEV_COLLOC)
        assert (self.mesh.xcoord[0] == self.bte_1d3v.xp).all() == True

        self.efield        = lambda t : xp.ones_like(self.mesh.xcoord[0]) * 1e4 * xp.cos(2 * xp.pi * t)
        #self.efield        = lambda t : xp.ones_like(self.mesh.xcoord[0]) * 1e4 * xp.cos(2 * xp.pi * t)
        self.cmodel_type   = closure_type.BTE_1D3V

        self.Ix            = xp.eye(self.mesh.N[0])
        
    def init(self):
        xp          = self.xp_module 
        u, v        = self.bte_1d3v.initialize()
        
        # q           = self.bte_1d3v.param
        # xx          = q.L * (self.bte_1d3v.xp + 1)
        # print(self.mfuncs_ops_ords[0] @ v)
        # print(self.mfuncs_ops[0] @ self.bte_1d3v.op_po2sh @ v)
        # print(1e6 * (1e7 + 1e9 * (1-0.5 * xx/q.L)**2 * (0.5 * xx/q.L)**2)/q.np0)
        # sys.exit(0)


        self.edf    = v
        
        Po          = self.bte_1d3v.op_psh2o
        Ps          = self.bte_1d3v.op_po2sh

        mw          = (Ps @ v[:,1])
        self.mw_edf = mw / (self.bte_1d3v.op_mass @ mw)
        
        # v[self.bte_1d3v.xp_vt_l, 0 ] = 0.0
        # v[self.bte_1d3v.xp_vt_r, -1] = 0.0
        
        self.mae_m0             = xp.zeros((len(self.mfuncs), self.bte_1d3v.Np))
        m                       = self.mfuncs_ops @ (Ps @ v)
        return m

    def closure_model(self, m_vec: np.array, E: np.array, type:closure_type):
        xp         = self.xp_module
        spec_sp    = self.bte_1d3v.op_spec_sp
        Ps         =  self.bte_1d3v.op_po2sh
        Po         =  self.bte_1d3v.op_psh2o

        atol, rtol = 1e-12, 1e-10
        if (type == closure_type.MAX_ENTROPY):
            m0 = xp.copy(m_vec)
            m0 = m0 / (m0[self.MIDX_MASS][xp.newaxis, :])
            
            if (self.bc_type == gm_boundary_conditions.MAXWELLIAN):

                w, fklm = closure_models.max_entropy_reconstruction(spec_sp, self.mae_m0[:, 1:-1], self.num_vr, self.num_vt, self.num_vp, 
                                                      m0[:, 1:-1], self.mfuncs, xp, rtol, atol, 100)
            
                fklm                 = fklm * m_vec[self.MIDX_MASS,1:-1]
                self.mae_m0[:, 1:-1] = w

                fklm        = xp.append((m_vec[self.MIDX_MASS,0] * self.mw_edf).reshape((-1, 1)), fklm, axis=1)
                fklm        = xp.append(fklm, (m_vec[self.MIDX_MASS,-1] * self.mw_edf).reshape((-1, 1)), axis=1)
            else:
                w, fklm = closure_models.max_entropy_reconstruction(spec_sp, self.mae_m0, self.num_vr, self.num_vt, self.num_vp, 
                                                      m0, self.mfuncs, xp, rtol, atol, 100)
            
                fklm        = fklm * m_vec[self.MIDX_MASS]
                self.mae_m0 = w
        
        elif(type == closure_type.BTE_1D3V):
            fklm        =  Ps @ m_vec

            q   = self.bte_params
            Cop = (self.bte_1d3v.op_col_en + self.params.Tg * self.bte_1d3v.op_col_gT) * q.n0 * q.np0 
            Av  = (self.bte_1d3v.op_adv_v) 


            mJx = (q.tau/q.L) * self.mfuncs_Jx_ops_ords @ m_vec
            mJv = (q.tau)     * (self.mfuncs_ops @ (Cop @ fklm + E * (Av @ fklm))) 
            m   = self.mfuncs_ops_ords @ m_vec

            # self.bte_1d3v.bs_E  = E
            # dt                  = self.params.cfl
            # v_dt                = self.bte_1d3v.step_bte_v(v, None, 0, dt, ts_type="BE")
            # mJv1                = self.mfuncs_ops_ords @ ((v_dt - v)/dt)
            # print(mJv1, mJv, xp.linalg.norm(mJv1-mJv))

            return m, mJx, mJv, fklm
        else:
            raise NotImplementedError
        
        q   = self.bte_params
        Cop = (self.bte_1d3v.op_col_en + self.params.Tg * self.bte_1d3v.op_col_gT) * q.n0 * q.np0 
        Av  = (self.bte_1d3v.op_adv_v) 

        #mJx          = (q.tau/q.L) * self.mfuncs_Jx_ops @ fklm
        

        #print(mJx[:, 0])
        #print(mJx[:, -1])
        
        # print(fklm[:, 0])
        # print(self.mw_edf * m_vec[self.MIDX_MASS, 0])
        # BCs. 

        # fl = Po @fklm[:,  0]
        # fr = Po @fklm[:, -1]

        # fl[self.bte_1d3v.xp_vt_l] = 0.0
        # fr[self.bte_1d3v.xp_vt_r] = 0.0

        # mJx[:,  0]   = (q.tau/q.L) * self.mfuncs_Jx_ops_l @ Ps @ fl
        # mJx[:, -1]   = (q.tau/q.L) * self.mfuncs_Jx_ops_r @ Ps @ fr
        

        # dt                 = self.params.cfl
        # mJx                = (q.tau/q.L) * self.mfuncs_Jx_ops @ ((self.bte_1d3v.op_po2sh @ self.bte_1d3v.step_bte_x(m_vec, tt, 0.5 * dt) -fklm)/dt/2)



        #mJx[self.MIDX_MOM, -1] *= -1
        # print(self.mfuncs_Jx_ops_l[0::3])

        # # print(mJx[:,  0])
        # # print(mJx[:,  -1])
        # Teb0          = m_vec[self.MIDX_ENERGY, 0] / m_vec[self.MIDX_MASS, 0]
        # Teb1          = m_vec[self.MIDX_ENERGY,-1] / m_vec[self.MIDX_MASS,-1]
        # vb0           = (xp.sqrt(Teb0) * c_gamma) * (q.tau/q.L) # non-dimensionalized themeral velocity corresponds to maxwellian eedf
        # vb1           = (xp.sqrt(Teb1) * c_gamma) * (q.tau/q.L) # non-dimensionalized themeral velocity corresponds to maxwellian eedf
        # sqrt_pi       = (xp.pi**0.5)

        # # maxwellian flux on a wall 
        # mJx[self.MIDX_MASS, 0 ] = -(0.5 / sqrt_pi) * vb0 * m_vec[self.MIDX_MASS,  0]
        # mJx[self.MIDX_MASS, -1] =  (0.5 / sqrt_pi) * vb1 * m_vec[self.MIDX_MASS, -1]

        # mJx[self.MIDX_MOM, 0 ]  =  0.5 * sqrt_pi * vb0 * mJx[self.MIDX_MASS,  0] * (q.L/q.tau)
        # mJx[self.MIDX_MOM, -1]  =  0.5 * sqrt_pi * vb1 * mJx[self.MIDX_MASS, -1] * (q.L/q.tau)

        # print(2 * vb0**2 * mJx[self.MIDX_MASS,  0] * (q.L/q.tau)**2) 
        # print(2 * vb1**2 * mJx[self.MIDX_MASS, -1] * (q.L/q.tau)**2)

        # sys.exit(0)

        # Jx[self.MIDX_MOM,  0]    = 0.5 * sqrt_pi * vb0 * Jx[self.MIDX_MASS,  0] 
        # Jx[self.MIDX_MOM, -1]    = 0.5 * sqrt_pi * vb1 * Jx[self.MIDX_MASS, -1] 

        # Jx[self.MIDX_ENERGY,  0] = 2 * vb0**2 * Jx[self.MIDX_MASS,  0] 
        # Jx[self.MIDX_ENERGY, -1] = 2 * vb1**2 * Jx[self.MIDX_MASS, -1]
        # mJx[3, 0]  = 0.0
        # mJx[4, 1]  = 0.0

        #mJv = (q.tau)     * (self.mfuncs_ops @ (Cop @ fklm + E * (Av @ fklm))) #* 0.0
        self.bte_1d3v.bs_E = E
        dt                 = self.params.cfl
        gklm               = self.bte_1d3v.op_psh2o @ fklm
        
        # gklm[self.bte_1d3v.xp_vt_l,  0] = 0.0
        # gklm[self.bte_1d3v.xp_vt_r, -1] = 0.0

        fklm_dt            = self.bte_1d3v.op_po2sh @ (self.bte_1d3v.step_bte_v(gklm, None, 0, dt, ts_type="BE"))
        mJv                = self.mfuncs_ops @ ((fklm_dt - fklm)/dt)
        # print(mJv)
        # print(mJv1)

        # print((self.mfuncs_ops[0] @ Cop)[0::3])
        # print((self.mfuncs_ops[0] @ Av) [0::3])
        # print(mJv[0])

        # fklm_n                 = self.bte_1d3v.bte_eedf_normalization(fklm)
        # a1                     = (self.bte_1d3v.op_rate[1] @ fklm_n[0::3]) * (q.tau * q.np0 * q.n0 * m_vec[0]) 
        # a2                     = (mJv[0])
        # print("1-m3s-1 %.8E ", a1)
        # print("2-m3s-1 %.8E ", a2)

        # print(mJv/m_vec[0][xp.newaxis, :])

        # print(self.mfuncs_ops[0][0::3])
        # print(self.bte_1d3v.op_mass[0::3])

        # sys.exit(0)

        m  = self.mfuncs_ops_ords @ m_vec

        return m, mJx, mJv, fklm
        
    def rhs(self, x, time, dt):
        xp            = self.xp_module
        E             = self.efield(time)
        c_gamma       = xp.sqrt(2 * scipy.constants.elementary_charge / scipy.constants.electron_mass)
        

        
        if (self.cmodel_type == closure_type.BTE_1D3V):
            Jx, Jv, fklm  = self.closure_model(self.edf, E, self.cmodel_type)
        else:
            Jx, Jv, fklm  = self.closure_model(x, E, self.cmodel_type)

        y             = Jv - (self.mesh.D1[0] @ Jx.T).T - 1e-3 * (self.bte_1d3v.param.tau/self.bte_1d3v.param.L**2) * (self.mesh.D2[0] @ x.T).T
        

        if (self.bc_type == gm_boundary_conditions.MAXWELLIAN or 
            self.bc_type == gm_boundary_conditions.TE_DIRICHLET):
            # Dirichlet BCs at boundary -- this is not physical boundary condition. Te at the wall can vary. 
            y[self.MIDX_ENERGY,  0] = (y[self.MIDX_MASS,  0] * self.bte_params.Teb0) 
            y[self.MIDX_ENERGY, -1] = (y[self.MIDX_MASS, -1] * self.bte_params.Teb1)

        return y

    def step(self, x, time, dt):
        q             = self.bte_params
        E             = gm.efield(time)
        if (self.cmodel_type == closure_type.BTE_1D3V):
            m, Jx, Jv, fklm  = self.closure_model(self.edf, E, self.cmodel_type)
            Jx[self.implicit_idx] = x[self.implicit_idx] * (Jx[self.implicit_idx]/m[self.implicit_idx])
            Jv[self.implicit_idx] = x[self.implicit_idx] * (Jv[self.implicit_idx]/m[self.implicit_idx])
        else:
            m, Jx, Jv, fklm  = self.closure_model(x, E, self.cmodel_type)

        y             = xp.zeros_like(x)
        jx            = xp.zeros_like(Jx)
        sv            = xp.zeros_like(Jv)
        D1            = self.mesh.D1[0]
        
        for i in self.implicit_idx:
            idx          = np.abs(x[i])>0
            jx[i, idx]   = (Jx[i, idx] / x[i, idx]) 
            sv[i, idx]   = (Jv[i, idx] / x[i, idx]) 

            # print("x", x)
            # print("m", m)
            # print("||x-m||", xp.linalg.norm(x-m))
            # print("Jx/x", (Jx[i, idx] / x[i, idx]) )
            # print("Jx/m", (Jx[i, idx] / m[i, idx]) )

            # print("Jv/x ", (Jv[i, idx] / x[i, idx]) )
            # print("Jv/m ", (Jv[i, idx] / m[i, idx]) )
            # sys.exit(0)
            
            # print(1, (Jx[i, idx] / m[i, idx]))
            # print(2, (Jx[i, idx] / x[i, idx]))

            # A            = self.Ix * (1) + dt * jx[i] * self.mesh.D1[0] 
            # b            = x[i] * (1 + dt * sv[i] - dt * self.mesh.D1[0] @ jx[i])
            # y[i]         = xp.linalg.solve(A, b)

            # a1 = D1 @ (x[i] * jx[i])
            # a2 = (D1 @ x[i]) * jx[i] + x[i] * (D1 @ jx[i])
            # print(a1)
            # print(a2)
            # print(xp.linalg.norm(a1-a2)/xp.linalg.norm(a1), xp.linalg.norm(a1))
            #sys.exit(0)

            A            = (self.Ix * (1 - 0.5 * dt * sv[i])  +  0.5 * dt * ( jx[i] * D1 +  (D1 @ jx[i]) * self.Ix))
            b            = x[i] * (1 + 0.5 * dt * sv[i]) -0.5 * dt * D1 @ (x[i] * jx[i]) #- 0.5 * dt * (jx[i] * (D1 @ x[i]) + x[i] * (D1 @ jx[i]))
            y[i]         = xp.linalg.solve(A, b)

            # A            = (self.Ix * (1 - 0.5 * dt * sv[i]) +  0.5 * dt * jx[i] * self.mesh.D1[0])
            # b            = x[i] * (1 + 0.5 * dt * sv[i]) - 0.5 * dt * jx[i] * (self.mesh.D1[0] @ x[i])
            # y[i]         = xp.linalg.solve(A, b)

        
        if (len(self.explicit_idx)>0):
            # ue   = x[1]/x[0] #(Jx[0] / x[0])  #* self.bte_1d3v.bs_vth
            # P    = Jx[1] - (q.tau/q.L) * ue* x[1] #Jx[0]

            # print("ue", ue)
            # print("Jx", Jx[1])
            # print("a", (q.tau/q.L) * ue * x[1])
            # print("Dp", xp.linalg.norm(D1 @ P))

            # A    = self.Ix + 0.5 * dt * (q.tau/q.L) * (ue * D1 + (D1 @ ue) * self.Ix)
            # b    = dt * Jv[1] - dt * D1 @ (P  + 0.5 * (q.tau/q.L) * (ue * x[1]))

            # y[1] = xp.linalg.solve(A, b)

            y[self.explicit_idx] =x[self.explicit_idx] - dt * (D1 @ Jx[self.explicit_idx].T).T + dt * Jv[self.explicit_idx]
            # w1= x[1] - dt * (D1 @ Jx[self.explicit_idx].T).T + dt * Jv[self.explicit_idx]
            # w2 =x[1] + dt * (Jv[1] - D1 @ P) - dt * (q.tau/q.L)* D1 @ (ue * x[1])
            # print(xp.linalg.norm(w1 -y[1])/xp.linalg.norm(y[1]))
            # print(xp.linalg.norm(w1 -w2)/xp.linalg.norm(w1))
            # y[1] = w2
            # dtm                  = min(self.cfl_factor * (self.mesh.dx[0]/xp.max(xp.abs(Jx[self.explicit_idx]))), 0.01/xp.max(xp.abs(Jv[self.explicit_idx])))
            # print(dtm)
            # sm                   = int(xp.ceil(dt / dtm))
            # dtm                  = (dt / sm)
            # y[self.explicit_idx] = x[self.explicit_idx]

            # for i in range(sm+1):
            #     y[self.explicit_idx] += - dtm * (self.mesh.D1[0] @ Jx[self.explicit_idx].T).T + dtm * Jv[self.explicit_idx]

        #sigma = 1e0    
        # ue_abs    = xp.abs(Jx[0]/m[0]) * (q.L/q.tau)
        # sigma     = ue_abs * 1e-4
        sigma     = 5e1
        # sigma[sigma>1e2] = 1e2
        # sigma[sigma<1]   = 1
        # print(sigma)
        Lp        = self.Ix - dt * sigma * (q.tau/q.L**2) * self.mesh.cheb[0].Lp
        Lp[0 , :] = self.Ix[0]
        Lp[-1, :] = self.Ix[-1]
        y         = xp.linalg.solve(Lp, y.T).T 
        
        #print("y", y)
        # k1 = self.rhs(x, time, 0.5 * dt)
        # k2 = self.rhs(x + 0.5 * dt * k1, time + 0.5 * dt, dt)
        # y  = x + dt * k2

        # k1 = self.rhs(x, time, dt)
        # k2 = self.rhs(x + dt * 0.5 * k1, time + 0.5 * dt, dt)
        # k3 = self.rhs(x + dt * 0.5 * k2, time + 0.5 * dt, dt)
        # k4 = self.rhs(x + dt * 1.0 * k3, time + 1.0 * dt, dt)
        
        # y        = x + dt * ((1/6) * k1 + (1/3) * k2 + (1/3) * k3 + (1/6) * k4)
        #y          = x + dt * self.rhs(x, time, dt)
        #y          = self.rhs(x,time, dt)

        # q             = self.bte_params
        # v             = self.edf
        # m             = self.mfuncs_ops_ords    @ v
        # Jx            = self.mfuncs_Jx_ops_ords @ v
        
        # #print("rhs ", time, ue)
        # #print((xp.eye(self.mesh.D1[0].shape[0]) - dt * ue * self.mesh.D1[0]).shape)

        # # print(ue[:, xp.newaxis] * self.mesh.D1[0])
        # # print(ue[0] * self.mesh.D1[0][0,:])
        # # sys.exit(0)
        # y             = xp.zeros_like(x)
        # y[0]          = xp.linalg.solve((xp.eye(self.mesh.D1[0].shape[0]) + 0.5 * dt * (q.tau/q.L) * ue * self.mesh.D1[0]), x[0])
        # y[1]          = ue * ne #* (q.L / q.tau)
        
        # y[2,  0] = 4 * y[0,  0] 
        # y[2, -1] = 4 * y[0, -1] 
        # y[0, y[0]<0]  = 1e-10
        # y[2, y[2]<0]  = 1e-10
        # idx = (y[self.MIDX_ENERGY]/y[self.MIDX_MASS]) < 0
        # y [self.MIDX_ENERGY,  idx ] = 1e-1 * y[self.MIDX_MASS, idx]
        # y[self.MIDX_MOM, (y[self.MIDX_MOM]/y[self.MIDX_MASS])> 8e5]  =  8e5 * y[self.MIDX_MASS, (y[self.MIDX_MOM]/y[self.MIDX_MASS]) > 8e5]
        # y[self.MIDX_MOM, (y[self.MIDX_MOM]/y[self.MIDX_MASS])< -8e5] = -8e5 * y[self.MIDX_MASS, (y[self.MIDX_MOM]/y[self.MIDX_MASS])< -8e5]
        # y[self.MIDX_ENERGY,  0] = (y[self.MIDX_MASS,  0] * self.bte_params.Teb0) 
        # y[self.MIDX_ENERGY, -1] = (y[self.MIDX_MASS, -1] * self.bte_params.Teb1)
        
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
    
    # v[bte_solver.xp_vt_l, 0 ] = 0.0
    # v[bte_solver.xp_vt_r, -1] = 0.0

    bte_solver.step_init(u, v, dt)
    mass_op    = bte_solver.op_mass
    temp_op    = bte_solver.op_temp

    xp         = gm.xp_module
    io_freq    = int(bte_solver.args.io_cycle_freq / dt)
    #cp_freq    = int(bte_solver.args.cp_cycle_freq / dt)
    c_gamma    = np.sqrt(2 * scipy.constants.elementary_charge/ scipy.constants.electron_mass)
    

    while tt < T:
        vlm    = (bte_solver.op_po2sh @ v)
        #m_bte  = gm.mfuncs_ops @ vlm
        m_bte   = gm.mfuncs_ops_ords @ v
        gm.edf  = v

        # print("bte ", tt, m_bte[gm.MIDX_MOM]/m_bte[gm.MIDX_MASS])
        # if(tt > 2*dt):
        #     sys.exit(0)
        if (idx % io_freq == 0):
            plt.figure(figsize=(12, 8), dpi=200)
            plt.subplot(2, 3, 1)
            plt.plot(xx, m_gme[gm.MIDX_MASS] * gm.bte_params.np0, label=r"GME")
            plt.plot(xx, m_bte[gm.MIDX_MASS] * gm.bte_params.np0, label=r"BTE")
            plt.xlabel(r"$\hat{x}$")
            plt.ylabel(r"$n_e [m^-3]$")
            plt.legend()
            plt.grid(visible=True)
            

            plt.subplot(2, 3, 2)
            plt.plot(xx, m_gme[gm.MIDX_MOM]/m_gme[gm.MIDX_MASS], label=r"GME")
            plt.plot(xx, m_bte[gm.MIDX_MOM]/m_bte[gm.MIDX_MASS], label=r"BTE")
            plt.xlabel(r"$\hat{x}$")
            plt.ylabel(r"$v_x [ms^{-1}]$")
            plt.legend()
            plt.grid(visible=True)

            plt.subplot(2, 3, 3)
            plt.plot(xx, m_gme[gm.MIDX_ENERGY]/m_gme[gm.MIDX_MASS], label=r"GME")
            plt.plot(xx, m_bte[gm.MIDX_ENERGY]/m_bte[gm.MIDX_MASS], label=r"BTE")
            plt.xlabel(r"$\hat{x}$")
            plt.ylabel(r"$T_e [eV]$")
            plt.legend()
            plt.grid(visible=True)
            

            # ev      = np.linspace(1e-3, 80, 512)
            # g       = np.array(gm.bte_1d3v.compute_radial_components(ev, gm.bte_1d3v.bte_eedf_normalization(vlm)))
            # #_, _, h = gm.closure_model(m_gme, gm.efield(tt), gm.cmodel_type)
            # _, _, _,  h = gm.closure_model(gm.edf, gm.efield(tt), gm.cmodel_type)
            # h       = np.array(gm.bte_1d3v.compute_radial_components(ev, gm.bte_1d3v.bte_eedf_normalization(h)))

            # plt.subplot(2, 3, 4)
            # for xidx in range(0, len(xx), 3):
            #     plt.semilogy(ev, np.abs(h[xidx, 0]), "--", label=r"GME (x=%.2f)"%(xx[xidx]))
            #     plt.semilogy(ev, np.abs(g[xidx, 0]), "-" , label=r"BTE (x=%.2f)"%(xx[xidx]))
            # plt.xlabel(r"energy [eV]")
            # plt.ylabel(r"$f_0$")
            # plt.legend(fontsize=6)
            # plt.grid(visible=True)

            # plt.subplot(2, 3, 5)
            # for xidx in range(0, len(xx), 3):
            #     plt.semilogy(ev, np.abs(h[xidx, 1]), "--", label=r"GME (x=%.2f)"%(xx[xidx]))
            #     plt.semilogy(ev, np.abs(g[xidx, 1]), "-" , label=r"BTE (x=%.2f)"%(xx[xidx]))
            # plt.xlabel(r"energy [eV]")
            # plt.ylabel(r"$f_1$")
            # plt.legend(fontsize=6)
            # plt.grid(visible=True)

            # plt.subplot(2, 3, 6)
            # for xidx in range(0, len(xx), 3):
            #     plt.semilogy(ev, np.abs(h[xidx, 2].T), "--", label=r"GME (x=%.2f)"%(xx[xidx]))
            #     plt.semilogy(ev, np.abs(g[xidx, 2].T), "-" , label=r"BTE (x=%.2f)"%(xx[xidx]))
            # plt.xlabel(r"energy [eV]")
            # plt.ylabel(r"$f_2$")
            # plt.legend(fontsize=6)
            # plt.grid(visible=True)
            
            plt.suptitle("time = %.4E [T]"%(tt))
            plt.tight_layout()
            plt.savefig("%s_%04d.png"%(bte_solver.args.fname, idx//io_freq))
            plt.close()

            # ff = h5py.File("%s_sol_%04d.h5"%(bte_solver.args.fname, idx//io_freq), 'w')
            # ff.create_dataset("time[T]"      , data = tt)
            # ff.create_dataset("gme"          , data = m_gme)
            # ff.create_dataset("bte"          , data = m_bte)
            # ff.create_dataset("bte_edf"      , data = vlm)
            # ff.create_dataset("gme_edf"      , data = h)
            # ff.close()

        m_gme   = gm.step(m_gme, tt, dt)

        bte_solver.bs_E = gm.efield(tt)
        v               = bte_solver.step_bte_x(v, tt, 0.5 * dt)
        v               = bte_solver.step_bte_v(v, None, tt, dt, ts_type="BE", verbose=1)
        #v               = bte_solver.op_psh2o @ (bte_solver.op_po2sh @ v)
        v               = bte_solver.step_bte_x(v, tt + 0.5 * dt, 0.5 * dt)
        tt  += dt
        idx +=1
        


