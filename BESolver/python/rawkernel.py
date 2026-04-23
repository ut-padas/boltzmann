import cupy as cp
'''
Raw kernels for the BTE solver using CuPy
'''

lower_bidiagonal_batched_code = r'''
extern "C" __global__ void lower_bidiagonal_batched(const double* diag, const double* subdiag, const double* b, double* out, int bsz, int N) {
    
    const unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= bsz) 
        return;
    
    const unsigned int idx  = gid * N     ;
    const unsigned int sidx = gid * (N-1) ;

    out[idx] = b[idx]/diag[idx];
    
    //printf("blockIdx.x = %d, threadIdx.x = %d, idx = %d\n", blockIdx.x, threadIdx.x, idx);
    for (unsigned int i = 1; i < N; i++)
        out[idx + i] = (b[idx+i] - subdiag[sidx+i-1] * out[idx + i - 1])/diag[idx+i];
    
    return;
    
}
'''

upper_bidiagonal_batched_code = r'''
extern "C" __global__ void upper_bidiagonal_batched(const double* diag, const double* subdiag, const double* b, double* out, int bsz, int N) {
    
    const unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= bsz) 
        return;
    
    const unsigned int idx  = gid * N    ;
    const unsigned int sidx = gid * (N-1);
    
    out[idx + N-1] = b[idx + N-1]/diag[idx + N-1];
    
    //printf("blockIdx.x = %d, threadIdx.x = %d, idx = %d\n", blockIdx.x, threadIdx.x, idx);
    for (unsigned int i = 1; i < N; i ++)
        out[idx + (N-1-i)] = (b[idx + (N-1-i)] - subdiag[sidx + (N-i-1)] * out[idx + (N-i)])/diag[idx + (N-1-i)];
    
    return;
    
}
'''

lower_banded_tri_batched_code = r'''
extern "C" __global__ void lower_banded_tri_batched(const double* diags, const double* b, double* out, int bsz, int ndiags, int N) {
    
    const unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= bsz) 
        return;
    
    const unsigned int idx  = gid * ndiags * N;
    const unsigned int rdx  = gid * N;

    out[rdx] = b[rdx]/diags[idx];
    
    //printf("blockIdx.x = %d, threadIdx.x = %d, idx = %d\n", blockIdx.x, threadIdx.x, idx);
    for (int i = 1; i < N; i++)
    {
        double s = 0.0;
        for (int d = 1; d < ndiags; d++)
        {
            if ((i-d) < 0)
                break;
            
            s+= diags[idx + d * N + i] * out[rdx + i - d];
        }

        out[rdx + i] = (b[rdx+i] -s)/diags[idx+i];

    }
    
    return;
}
'''

upper_banded_tri_batched_code = r'''
extern "C" __global__ void upper_banded_tri_batched(const double* diags, const double* b, double* out, int bsz, int ndiags, int N) {
    
    const unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= bsz) 
        return;
    
    const unsigned int idx  = gid * ndiags * N;
    const unsigned int rdx  = gid * N;
    
    out[rdx + N-1] = b[rdx + N-1]/diags[idx + N-1];
    
    //printf("blockIdx.x = %d, threadIdx.x = %d, idx = %d\n", blockIdx.x, threadIdx.x, idx);
    for (unsigned int i = 1; i < N; i ++)
    {
        double s = 0.0;
        for (int d = 1; d < ndiags; d++)
        {
            if((N-1-i+d) >= N)
                break;
            
            s+= diags[idx + d * N + (N-1-i)] * out[rdx + (N-1-i) + d];
        }
        
        out[rdx + (N-1-i)] = (b[rdx + (N-1-i)] - s)/diags[idx + (N-1-i)];
    }
    return;
    
}
'''

lower_banded_tri_batched_mv_code = r'''
extern "C" __global__ void lower_banded_tri_batched_mv(const double* diags, const double* b, double* out, int bsz, int ndiags, int N) {
    
    const unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= bsz) 
        return;
    
    const unsigned int idx  = gid * ndiags * N;
    const unsigned int rdx  = gid * N;

    //printf("blockIdx.x = %d, threadIdx.x = %d, idx = %d\n", blockIdx.x, threadIdx.x, idx);
    for (int i = 0; i < N; i++)
    {
        double s = 0.0;
        for (int d = 0; d < ndiags; d++)
        {
            if ((i-d) < 0)
                break;
            
            s+= diags[idx + d * N + i] * b[rdx + i - d];
        }

        out[rdx + i] = s;
    }
    return;
}
'''

upper_banded_tri_batched_mv_code = r'''
extern "C" __global__ void upper_banded_tri_batched_mv(const double* diags, const double* b, double* out, int bsz, int ndiags, int N) {
    
    const unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if (gid >= bsz) 
        return;
    
    const unsigned int idx  = gid * ndiags * N;
    const unsigned int rdx  = gid * N;
    
    //printf("blockIdx.x = %d, threadIdx.x = %d, idx = %d\n", blockIdx.x, threadIdx.x, idx);
    for (unsigned int i = 0; i < N; i ++)
    {
        double s = 0.0;
        for (int d = 0; d < ndiags; d++)
        {
            if((N-1-i+d) >= N)
                break;
            
            s+= diags[idx + d * N + (N-1-i)] * b[rdx + (N-1-i) + d];
        }
        
        out[rdx + (N-1-i)] = s;
    }
    return;
    
}
'''

## b,  out should (Nx * Nvt,  Nr) fastest changing Nr, Nvt, and Nx
adv_vr_BE_solve_code=r'''
extern "C" __global__ void adv_vr_BE_solve(const double* const D_LtoR,
                                           const double* const D_RtoL,
                                           const double* const Ex, 
                                           const double* const cos_vt, 
                                           double dt, 
                                           const double* b, double* out, int ndiags, int Nr, int Nvt, int Nx) {
    
    const unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
    const int bsz          = Nx * Nvt;
    if (gid >= bsz) 
        return;
    
    const int N=Nr;
    // Nvt as the fastest changing index since it determines the advection direction, this is for warp efficiency

    const int ix  = gid / Nvt;
    const int ivt = gid % Nvt;
    //printf("blockIdx.x = %d, threadIdx.x = %d, ix=%d, ivt=%d\n", blockIdx.x, threadIdx.x, ix, ivt);

    const double av = -Ex[ix] * cos_vt[ivt];
    const unsigned int rdx  = gid * N;
    if (av >= 0)
    {
        // up-winded from x-min to x-max (left to right)
        out[rdx]                = b[rdx] / (1.0 + dt * av * D_LtoR[0]);
        for (int i = 1; i < N; i++)
        {
            double s = 0.0;
            for (int d = 1; d < ndiags; d++)
            {
                if ((i-d) < 0)
                    break;

                s+= D_LtoR[d * N + (i-d)] * out[rdx + i - d];
            }

            out[rdx + i] = (b[rdx+i] - av * dt * s)/(1.0 + av * dt * D_LtoR[i]);
        }
    }
    else
    {
        // up-winded from x-max to x-min (right to left)
        out[rdx + N-1] = b[rdx + N-1]/(1.0 + av * dt * D_RtoL[N-1]);
        for (unsigned int i = 1; i < N; i ++)
        {
            double s = 0.0;
            for (int d = 1; d < ndiags; d++)
            {
                if((N-1-i+d) >= N)
                    break;
                
                s+= D_RtoL[d * N + (N-1-i + d)] * out[rdx + (N-1-i) + d];
            }
        
            out[rdx + (N-1-i)] = (b[rdx + (N-1-i)] - s * dt * av)/ (1.0 + av * dt * D_RtoL[N-1-i]);
        }

    }

    return;
}
'''

## b,  out should (Nx * Nr,  Nvt) fastest changing Nvt, Nr, and Nx
adv_vt_BE_solve_code=r'''
extern "C" __global__ void adv_vt_BE_solve(const double* const D_LtoR,
                                           const double* const D_RtoL,
                                           const double* const Ex, 
                                           const double* const vr,
                                           double dt, 
                                           const double* b, double* out, int ndiags, int Nr, int Nvt, int Nx) {
    
    const unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
    const int bsz          = Nx * Nr;
    if (gid >= bsz) 
        return;
    
    const int N=Nvt;
    
    const int ix  = gid / Nr;
    const int ivr = gid % Nr;
    const unsigned int rdx  = gid * N;
    //printf("blockIdx.x = %d, threadIdx.x = %d, idx = %d\n", blockIdx.x, threadIdx.x, idx);
        

    const double av = Ex[ix] / vr[ivr];
    if (-av >= 0)
    {
        // up-winded from x-min to x-max (left to right)
        out[rdx]                = b[rdx] / (1.0 + dt * av * D_LtoR[0]);
    
        for (int i = 1; i < N; i++)
        {
            double s = 0.0;
            for (int d = 1; d < ndiags; d++)
            {
                if ((i-d) < 0)
                    break;

                s+= D_LtoR[d * N + (i-d)] * out[rdx + i - d];
            }

            out[rdx + i] = (b[rdx+i] -s * av * dt)/(1.0 + av * dt * D_LtoR[i]);
        }
    }
    else
    {
        // up-winded from x-max to x-min (right to left)
        out[rdx + N-1] = b[rdx + N-1]/(1.0 + av * dt * D_RtoL[N-1]);
    
        for (unsigned int i = 1; i < N; i ++)
        {
            double s = 0.0;
            for (int d = 1; d < ndiags; d++)
            {
                if((N-1-i+d) >= N)
                    break;
                
                s+= D_RtoL[d * N + (N-1-i + d)] * out[rdx + (N-1-i) + d];
            }
        
            out[rdx + (N-1-i)] = (b[rdx + (N-1-i)] - s * dt * av )/ (1.0 + av * dt * D_RtoL[N-1-i]);
        }

    }

    return;
}
'''

## b,  out should (Nr * Nvt,  Nx) fastest changing Nx, Nvt, and Nr
adv_x_BE_solve_code=r'''
extern "C" __global__ void adv_x_BE_solve( const double* const D_LtoR,
                                           const double* const D_RtoL,
                                           const double* const vr, 
                                           const double* const cos_vt, 
                                           double dt, 
                                           const double* b, double* out, int ndiags, int Nr, int Nvt, int Nx) {
    
    const unsigned int gid = blockDim.x * blockIdx.x + threadIdx.x;
    const int bsz          = Nr * Nvt;

    if (gid >= bsz) 
        return;
    
    const int    N=Nx;
    const int ivr = gid / Nvt;
    const int ivt = gid % Nvt;
    const unsigned int rdx  = gid * N;

    const double av = vr[ivr] * cos_vt[ivt];
    /*if(gid == 0)
    {
        printf("blockIdx.x = %d, threadIdx.x = %d, gid = %d ivr=%d ivt=%d vr=%.10E cos_vt=%.10E\n", blockIdx.x, threadIdx.x, gid, ivr, ivt, vr[ivr], cos_vt[ivt]);
    
    }*/
        
    /*
    if(gid==0)
        printf("%d, %.10E\n b=%.10E\n", N-1, out[rdx + N-1],  b[rdx + N-1]);
    */    

    if (av >= 0)
    {
        // up-winded from x-min to x-max (left to right)
        out[rdx]                = b[rdx] / (1.0 + dt * av * D_LtoR[0]);
    
        for (int i = 1; i < N; i++)
        {
            double s = 0.0;
            for (int d = 1; d < ndiags; d++)
            {
                if ((i-d) < 0)
                    break;

                s+= D_LtoR[d * N + (i-d)] * out[rdx + i - d];
            }

            out[rdx + i] = (b[rdx+i] -s * av * dt)/(1.0 + av * dt * D_LtoR[i]);
        }
    }
    else
    {
        // up-winded from x-max to x-min (right to left)
        out[rdx + N-1] = b[rdx + N-1]/(1.0 + av * dt * D_RtoL[N-1]);

        for (unsigned int i = 1; i < N; i ++)
        {
            double s = 0.0;
            for (int d = 1; d < ndiags; d++)
            {
                if((N-1-i+d) >= N)
                    break;
                
                s+= D_RtoL[d * N + (N-1-i + d)] * out[rdx + (N-1-i) + d];
            }
        
            out[rdx + (N-1-i)] = (b[rdx + (N-1-i)] - s * dt * av)/ (1.0 + av * dt * D_RtoL[N-1-i]);

        }

    }

    return;
}
'''

__lower_bidiagonal_batched = cp.RawKernel(lower_bidiagonal_batched_code, 'lower_bidiagonal_batched')
__lower_banded_tri_batched = cp.RawKernel(lower_banded_tri_batched_code, 'lower_banded_tri_batched')
__upper_banded_tri_batched = cp.RawKernel(upper_banded_tri_batched_code, 'upper_banded_tri_batched')
__upper_bidiagonal_batched = cp.RawKernel(upper_bidiagonal_batched_code, 'upper_bidiagonal_batched')

__lower_banded_tri_batched_mv = cp.RawKernel(lower_banded_tri_batched_mv_code, 'lower_banded_tri_batched_mv')
__upper_banded_tri_batched_mv = cp.RawKernel(upper_banded_tri_batched_mv_code, 'upper_banded_tri_batched_mv')

__adv_vr_BE = cp.RawKernel(adv_vr_BE_solve_code, 'adv_vr_BE_solve')
__adv_vt_BE = cp.RawKernel(adv_vt_BE_solve_code, 'adv_vt_BE_solve')
__adv_x_BE  = cp.RawKernel(adv_x_BE_solve_code,   'adv_x_BE_solve')

def bidiagonal_solve_batched(diag, sdiag, b, lower, xp=cp):
    _diag  =  diag.reshape((-1,  diag.shape[-1]))
    _sdiag = sdiag.reshape((-1, sdiag.shape[-1]))
    _b     =     b.reshape((-1,      b.shape[-1]))
    
    
    assert _sdiag.shape[0] == _diag.shape[0] and _sdiag.shape[0] == _b.shape[0]
    assert _sdiag.shape[1] == _diag.shape[1] - 1 and _sdiag.shape[1] == _b.shape[1] - 1
    
    
    bsz    = _diag.shape[0]
    N      = _diag.shape[1]
    _out   = xp.zeros_like(_b)
    
    threads_per_block = 256
    blocks_per_grid   = (bsz // threads_per_block) + 1
    
    assert blocks_per_grid * threads_per_block >= bsz

    if(lower == True):
        __lower_bidiagonal_batched((blocks_per_grid,), (threads_per_block,), (_diag, _sdiag, _b, _out, bsz, N))
    else:
        __upper_bidiagonal_batched((blocks_per_grid,), (threads_per_block,), (_diag, _sdiag, _b, _out, bsz, N))
    
    return _out.reshape(diag.shape)

def banded_tri_solve_batched(diags, b, lower, xp=cp):
    _diags = diags.reshape((-1, diags.shape[-2], diags.shape[-1]))
    _b     = b.reshape((-1, b.shape[-1]))

    assert _diags.shape[0] == _b.shape[0]
    
    
    
    bsz    = _diags.shape[0]
    ndiags = _diags.shape[1]
    N      = _diags.shape[2]

    _out   = xp.zeros_like(_b)
    
    threads_per_block = 256
    blocks_per_grid   = (bsz // threads_per_block) + 1
    
    assert blocks_per_grid * threads_per_block >= bsz

    if(lower == True):
        __lower_banded_tri_batched((blocks_per_grid,), (threads_per_block,), (_diags, _b, _out, bsz, ndiags, N))
    else:
        __upper_banded_tri_batched((blocks_per_grid,), (threads_per_block,), (_diags, _b, _out, bsz, ndiags, N))
    
    return _out.reshape(b.shape)

def banded_tri_mv_batched(diags, b, lower, xp=cp):
    _diags = diags.reshape((-1, diags.shape[-2], diags.shape[-1]))
    _b     = b.reshape((-1, b.shape[-1]))

    assert _diags.shape[0] == _b.shape[0]
    
    
    
    bsz    = _diags.shape[0]
    ndiags = _diags.shape[1]
    N      = _diags.shape[2]

    _out   = xp.zeros_like(_b)
    
    threads_per_block = 256
    blocks_per_grid   = (bsz // threads_per_block) + 1
    
    assert blocks_per_grid * threads_per_block >= bsz

    if(lower == True):
        __lower_banded_tri_batched_mv((blocks_per_grid,), (threads_per_block,), (_diags, _b, _out, bsz, ndiags, N))
    else:
        __upper_banded_tri_batched_mv((blocks_per_grid,), (threads_per_block,), (_diags, _b, _out, bsz, ndiags, N))
    
    return _out.reshape(b.shape)


def adv_vr_BE(D_LtoR, D_RtoL, Ex, cos_vt, dt, b, out, ndiags, Nr, Nvt, Nx):
    """
    b,  out should (Nx * Nvt,  Nr) fastest changing Nr, Nvt, and Nx
    """
    
    bsz               = Nx * Nvt
    threads_per_block = 256
    blocks_per_grid   = (bsz // threads_per_block) + 1

    __adv_vr_BE((blocks_per_grid,), (threads_per_block,), (D_LtoR, D_RtoL, Ex, cos_vt, dt, b, out, ndiags, Nr, Nvt, Nx))

    return

def adv_vt_BE(D_LtoR, D_RtoL, Ex, vr, dt, b, out, ndiags, Nr, Nvt, Nx):
    """
    b,  out should (Nx * Nr,  Nvt) fastest changing Nvt, Nr, and Nx
    """
    
    bsz               = Nx * Nr
    threads_per_block = 256
    blocks_per_grid   = (bsz // threads_per_block) + 1

    __adv_vt_BE((blocks_per_grid,), (threads_per_block,), (D_LtoR, D_RtoL, Ex, vr, dt, b, out, ndiags, Nr, Nvt, Nx))

    return

def adv_x_BE(D_LtoR, D_RtoL, vr, cos_vt, dt, b, out, ndiags, Nr, Nvt, Nx):
    """
    b,  out should (Nr * Nvt,  Nx) fastest changing Nx, Nvt, and Nr
    """
    
    bsz               = Nr * Nvt 
    threads_per_block = 256
    blocks_per_grid   = (bsz // threads_per_block) + 1

    __adv_x_BE((blocks_per_grid,), (threads_per_block,), (D_LtoR, D_RtoL, vr, cos_vt, dt, b, out, ndiags, Nr, Nvt, Nx))

    return

if __name__ == "__main__":
    xp    = cp

    
    Nr    = 256
    Nvt   = 32
    Nx    = 100

    xx    = -xp.cos(xp.pi*xp.linspace(0,Nx-1,Nx)/(Nx-1)) #xp.linspace(-1, 1, Nx)
    xp_vr = xp.linspace(0.2, 5-0.2, Nr)
    xp_vt = xp.flip(xp.linspace(0, xp.pi, Nvt))
    Ex    = 1e3 * xx**3

    pvt   = pvr = 3
    px    = 1


    import mesh
    Dvt_LtoR              = xp.asarray(mesh.upwinded_dvt(xp_vt.get(),1, pvt+1, "L", use_cdx_internal=False))
    Dvt_RtoL              = xp.asarray(mesh.upwinded_dvt(xp_vt.get(),1, pvt+1, "R", use_cdx_internal=False))

    Dvr_LtoR              = xp.asarray(mesh.upwinded_dx(xp_vr.get(),1 , pvr+1, "L"))
    Dvr_RtoL              = xp.asarray(mesh.upwinded_dx(xp_vr.get(),1 , pvr+1, "R"))

    Dx_LtoR               = xp.asarray(mesh.upwinded_dx(xx.get(),1 , px+1, "L"))
    Dx_RtoL               = xp.asarray(mesh.upwinded_dx(xx.get(),1 , px+1, "R"))

    k_domain              = 5

    Dvr_LtoR[0,:]         = 0.0
    #Dvr_LtoR[0,0]        = 1/(xp_vr[0] - k_domain[0])
    Dvr_RtoL[-1 , :]      = 0.0
    Dvr_RtoL[-1 ,-1]      = -1/(k_domain - xp_vr[-1])


    Dvt_LtoR[0 , :]       = 0.0
    Dvt_RtoL[-1, :]       = 0.0

    Dx_LtoR[0 , :]        = 0.0
    Dx_RtoL[-1, :]        = 0.0

    xp.random.seed(0)
    b                     = xp.random.rand(Nr*Nvt, Nx)
    dt                    = 1e-1
    gmres_pc_vr           = gmres_pc_vt = 1.0

    aE                    = 1
    cos_vt                = xp.cos(xp_vt)  
    Ecos_vt               = aE * xp.einsum("i,l->il", Ex, cos_vt)
    idx_rl_vr             = Ecos_vt >= 0
    idx_lr_vr             = Ecos_vt <0

    vr_diag               = xp.zeros((Nx, Nvt, pvr+1,Nr))
    
    vr_diag [idx_rl_vr,0] =   - gmres_pc_vr * dt * xp.diagonal(Dvr_RtoL)
    vr_diag [idx_lr_vr,0] =   - gmres_pc_vr * dt * xp.diagonal(Dvr_LtoR)
        
    for k in range(1, pvr+1):
        vr_diag[idx_rl_vr, k, :-k] =   - gmres_pc_vr * dt * xp.diagonal(Dvr_RtoL, offset=k)
        vr_diag[idx_lr_vr, k, k: ] =   - gmres_pc_vr * dt * xp.diagonal(Dvr_LtoR, offset=-k)

        
    vr_diag[:, :, 0, :]  = xp.einsum("abc,ab->abc" , vr_diag[:, :, 0, :] , Ecos_vt) + 1
    vr_diag[:, :, 1:,:]  = xp.einsum("abpc,ab->abpc", vr_diag[:, :, 1:,:] , Ecos_vt)

        
    vt_diag             = xp.zeros((Nx, Nr, pvt+1, Nvt))
        
    sin_vt              = xp.sin(xp_vt)
    Ebyv                = aE * xp.einsum("i,k->ik", Ex, 1.0/xp_vr)
    idx_lr_vt           = Ebyv >=0
    idx_rl_vt           = Ebyv < 0

    Dvt_LtoR            = xp.diag(sin_vt) @ Dvt_LtoR
    Dvt_RtoL            = xp.diag(sin_vt) @ Dvt_RtoL


    vt_diag [idx_lr_vt, 0] = gmres_pc_vt * dt * xp.diagonal(Dvt_RtoL)
    vt_diag [idx_rl_vt, 0] = gmres_pc_vt * dt * xp.diagonal(Dvt_LtoR)

    for k in range(1, pvt+1):
        vt_diag[idx_lr_vt, k, :-k] = gmres_pc_vt * dt * xp.diagonal(Dvt_RtoL, offset=k)
        vt_diag[idx_rl_vt, k, k:]  = gmres_pc_vt * dt * xp.diagonal(Dvt_LtoR, offset=-k)

    vt_diag[:, :, 0, :] = xp.einsum("abc,ab->abc"  , vt_diag[:, :, 0,  :] , Ebyv) + 1
    vt_diag[:, :, 1:,:] = xp.einsum("abpc,ab->abpc", vt_diag[:, :, 1:, :] , Ebyv)

    x_diag                = xp.zeros((Nr, Nvt, px+1, Nx))
    vcos_vt               = xp.einsum("i,l->il", xp_vr, cos_vt)
    idx_rl_x              = vcos_vt < 0
    idx_lr_x              = vcos_vt >= 0

    x_diag [idx_rl_x,0] = dt * xp.diagonal(Dx_RtoL)
    x_diag [idx_lr_x,0] = dt * xp.diagonal(Dx_LtoR)
        
    for k in range(1, px+1):
        x_diag[idx_rl_x, k, :-k] =   dt * xp.diagonal(Dx_RtoL, offset=k)
        x_diag[idx_lr_x, k, k: ] =   dt * xp.diagonal(Dx_LtoR, offset=-k)

        
    x_diag[:, :, 0, :]  = xp.einsum("abc,ab->abc"  , x_diag[:, :, 0, :] , vcos_vt) + 1
    x_diag[:, :, 1:,:]  = xp.einsum("abpc,ab->abpc", x_diag[:, :, 1:,:] , vcos_vt)

    vr_l2r = xp.zeros((pvr+1, Nr))
    vr_r2l = xp.zeros((pvr+1, Nr))

    for k in range(pvr+1):
        vr_l2r[k, 0:(Nr-k)] = xp.diagonal(Dvr_LtoR, offset=-k)
        vr_r2l[k, k:]       = xp.diagonal(Dvr_RtoL, offset=k)

    vt_l2r = xp.zeros((pvt+1, Nvt))
    vt_r2l = xp.zeros((pvt+1, Nvt))

    for k in range(pvt+1):
        vt_l2r[k, 0:(Nvt-k)] = xp.diagonal(Dvt_LtoR, offset=-k)
        vt_r2l[k, k:]       = xp.diagonal(Dvt_RtoL, offset=k)

    x_l2r = xp.zeros((px+1, Nx))
    x_r2l = xp.zeros((px+1, Nx))

    for k in range(px+1):
        x_l2r[k, 0:(Nx-k)] = xp.diagonal(Dx_LtoR, offset=-k)
        x_r2l[k, k:]       = xp.diagonal(Dx_RtoL, offset=k)

    x      = xp.swapaxes(b.reshape((Nr, Nvt, Nx)), 0, 2) # Nx, Nvt, Nr
    y0     = xp.zeros_like(x.reshape((-1)))
    y1     = xp.zeros_like(x)

    adv_vr_BE(vr_l2r, vr_r2l, Ex, cos_vt, dt, x.reshape((-1)), y0, pvr+1, Nr, Nvt, Nx)
    y0     = y0.reshape((Nx * Nvt, Nr))

    y1[idx_lr_vr] = banded_tri_solve_batched(vr_diag[idx_lr_vr], x[idx_lr_vr], lower=True , xp=cp)
    y1[idx_rl_vr] = banded_tri_solve_batched(vr_diag[idx_rl_vr], x[idx_rl_vr], lower=False, xp=cp)
    y1            = y1.reshape((Nx * Nvt, Nr))

    #print(y0, "\n", y1,"\n", x.reshape((Nx * Nvt, Nr)))
    # print("y0\n", y0)
    # print("y1\n", y1)
    # print("x\n" , x.reshape((Nx * Nvt , Nr)))

    print("vr- advection : %.8E "%(xp.max(xp.abs(y1 - y0))))

    x   = xp.swapaxes(xp.swapaxes(b.reshape((Nr, Nvt, Nx)), 0, 2), 1, 2) # Nx, Nr, Nvt
    y0  = xp.zeros_like(x.reshape((-1)))
    y1  = xp.zeros_like(x)

    
    adv_vt_BE(vt_l2r, vt_r2l, Ex, xp_vr, dt, x.reshape((-1)), y0, pvt+1, Nr, Nvt, Nx)
    y0            = y0.reshape((Nx * Nr, Nvt))
    
    y1[idx_lr_vt] = banded_tri_solve_batched(vt_diag[idx_lr_vt], x[idx_lr_vt], lower=False , xp=cp)
    y1[idx_rl_vt] = banded_tri_solve_batched(vt_diag[idx_rl_vt], x[idx_rl_vt], lower=True, xp=cp)
    y1            = y1.reshape((Nx * Nr, Nvt))


    # print("y0\n", y0)
    # print("y1\n", y1)
    # print("x\n" , x.reshape((Nx * Nvt , Nr)))
    print("vt- advection : %.8E "%(xp.max(xp.abs(y1 - y0))))

    x   = b.reshape((Nr, Nvt, Nx))
    y0  = xp.zeros_like(x.reshape((-1)))
    y1  = xp.zeros_like(x)

    
    adv_x_BE(x_l2r, x_r2l, xp_vr, cos_vt, dt, x.reshape((-1)), y0, px+1, Nr, Nvt, Nx)
    y0           = y0.reshape((Nr * Nvt, Nx))
    
    y1[idx_lr_x] = banded_tri_solve_batched(x_diag[idx_lr_x], x[idx_lr_x], lower=True , xp=cp)
    y1[idx_rl_x] = banded_tri_solve_batched(x_diag[idx_rl_x], x[idx_rl_x], lower=False, xp=cp)
    y1           = y1.reshape((Nr * Nvt, Nx))

    # print("y0\n", y0)
    # print("y1\n", y1)
    # print("x\n" , x.reshape((Nx * Nvt , Nr)))

    print("x - advection : %.8E "%(xp.max(xp.abs(y1 - y0))))











    
    bsz   = 256 * 32
    N     = 100
    diag  = xp.array([1 + cp.random.rand(N)   for i in range(bsz)])
    sdiag = xp.array([cp.random.rand(N-1) for i in range(bsz)])
    b     = xp.random.rand(bsz, N)
    
    Atrl  = xp.zeros((bsz, N, N))
    xp.diagonal(Atrl, offset=0 , axis1=1, axis2=2)[:] = diag
    xp.diagonal(Atrl, offset=-1, axis1=1, axis2=2)[:] = sdiag
    
    c = bidiagonal_solve_batched(diag, sdiag, b, lower=True)
    print("rel error (lower) = %.8E"%(xp.linalg.norm(xp.einsum("ijl,il->ij", Atrl, c) - b)/xp.linalg.norm(b)))

    Atru  = xp.zeros((bsz, N, N))
    xp.diagonal(Atru, offset=0 , axis1=1, axis2=2)[:] = diag
    xp.diagonal(Atru, offset=1 , axis1=1, axis2=2)[:] = sdiag

    c = bidiagonal_solve_batched(diag, sdiag, b, lower=False)
    xp.linalg.norm(xp.einsum("ijl,il->ij", Atru, c) - b)/xp.linalg.norm(b)
    print("rel error (upper) = %.8E"%(xp.linalg.norm(xp.einsum("ijl,il->ij", Atru, c) - b)/xp.linalg.norm(b)))
    
    ndiags = 30
    diags  = xp.random.rand(bsz, ndiags, N)
    diags[:, 0, :] += 1.0
    b      = xp.random.rand(bsz, N)

    Atrl  = xp.zeros((bsz, N, N))
    for i in range(ndiags):
        xp.diagonal(Atrl, offset=-i , axis1=1, axis2=2)[:] = diags[:, i, i:]

    c = banded_tri_solve_batched(diags, b, lower=True)
    print("rel error (banded lower trigular) = %.8E"%(xp.linalg.norm(xp.einsum("ijl,il->ij", Atrl, c) - b)/xp.linalg.norm(b)))

    Atru  = xp.zeros((bsz, N, N))
    for i in range(ndiags):
        xp.diagonal(Atru, offset=i , axis1=1, axis2=2)[:] = diags[:, i, :N-i]
    
    c = banded_tri_solve_batched(diags, b, lower=False)
    print("rel error (banded upper trigular) = %.8E"%(xp.linalg.norm(xp.einsum("ijl,il->ij", Atru, c) - b)/xp.linalg.norm(b)))


    c  = banded_tri_mv_batched(diags, b, lower=True)
    c1 = xp.einsum("ijl,il->ij", Atrl, b)
    print("rel error (banded upper trigular mv) = %.8E"%(xp.linalg.norm(c1 - c)/xp.linalg.norm(c1)))

    c  = banded_tri_mv_batched(diags, b, lower=False)
    c1 = xp.einsum("ijl,il->ij", Atru, b)
    print("rel error (banded upper trigular mv) = %.8E"%(xp.linalg.norm(c1 - c)/xp.linalg.norm(c1)))

    print("done")
