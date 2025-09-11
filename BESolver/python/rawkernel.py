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


__lower_bidiagonal_batched = cp.RawKernel(lower_bidiagonal_batched_code, 'lower_bidiagonal_batched')
__upper_bidiagonal_batched = cp.RawKernel(upper_bidiagonal_batched_code, 'upper_bidiagonal_batched')

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

        
if __name__ == "__main__":
    xp    = cp
    
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
    
    print("done")
