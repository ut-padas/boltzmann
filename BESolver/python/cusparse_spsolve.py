"""
Sparse triangular solve with explicit SpSM analysis reuse — CuPy 14 / CUDA 12+
cusparseSpSM_analysis  : called ONCE before the time loop
cusparseSpSM_solve     : called every timestep, reuses the cached analysis
"""
import ctypes, ctypes.util
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cpsp
import cupyx.scipy.sparse.linalg as cpsp_lg
import scipy.sparse as sp
from time import perf_counter, sleep

_lib = ctypes.CDLL(ctypes.util.find_library("cusparse") or "libcusparse.so")

CUSPARSE_OPERATION_NON_TRANSPOSE = 0

CUSPARSE_FILL_MODE_LOWER         = 0
CUSPARSE_FILL_MODE_UPPER         = 1

CUSPARSE_DIAG_TYPE_NON_UNIT      = 0
CUSPARSE_DIAG_TYPE_UNIT          = 1

CUSPARSE_INDEX_32I               = 2
CUSPARSE_INDEX_64I               = 3

CUSPARSE_INDEX_BASE_ZERO         = 0
CUDA_R_64F                       = 1   # double

CUSPARSE_SPSM_ALG_DEFAULT        = 0
CUSPARSE_ORDER_COL               = 1   # column-major (F-order)
CUSPARSE_ORDER_ROW               = 2   # row-major    (C-order)

CUSPARSE_SPMAT_FILL_MODE         = 0
CUSPARSE_SPMAT_DIAG_TYPE         = 1

CUSPARSE_DIRECTION_ROW           = 0
CUSPARSE_DIRECTION_COLUMN        = 1

CUSPARSE_SOLVE_POLICY_NO_LEVEL   = 0
CUSPARSE_SOLVE_POLICY_USE_LEVEL  = 1

CUSPARSE_MATRIX_TYPE_GENERAL     = 0
CUSPARSE_MATRIX_TYPE_SYMMETRIC   = 1
CUSPARSE_MATRIX_TYPE_HERMITIAN   = 2
CUSPARSE_MATRIX_TYPE_TRIANGULAR  = 3


class solver_type():
    CUSPARSE_SPSM    = 0
    CUSPARSE_BSRSM2  = 1
    CUPY_SPSM_SEQ    = 2
    CUPY_SPSM_PAR    = 3
    CUDA_CODE        = 4




def _check(status):
    if status != 0:
        raise RuntimeError(f"cuSPARSE error: status={status}")


class SpSMSolver:
    """
    Wraps a single constant sparse triangular matrix for repeated solves.
    Analysis (the expensive part) runs once in __init__; each solve() call
    only runs cusparseSpSM_solve.
    """

    def __init__(self, A_sp, nrhs: int = 1, lower: bool = True, dtype=np.float64, unit_diagonal: bool = True, stype: solver_type = solver_type.CUSPARSE_BSRSM2):
        self.lower = lower
        self.dtype = np.dtype(dtype)
        self.stype = stype
        self.udiag = unit_diagonal

        assert self.dtype == np.float64, "only float64 shown here; extend for float32"
        n = A_sp.shape[0]
        self.n = n

        self.format = A_sp.format
        assert self.format in ['csr', 'bsr'], "only CSR and BSR formats supported"

        # Keep device arrays alive — their pointers are baked into descriptors
        self._data    = cp.array(A_sp.data,    dtype=np.float64)
        self._indptr  = cp.array(A_sp.indptr,  dtype=np.int32)
        self._indices = cp.array(A_sp.indices, dtype=np.int32)

        # print("indices prt \n ", self._indptr)
        # print("indices \n ", self._indices)
        #print("data \n ", self._data)

        nnz           = int(A_sp.nnz)
        self.nnz      = nnz
        

        # ── create handle 
        self._handle = ctypes.c_void_p()
        _check(_lib.cusparseCreate(ctypes.byref(self._handle)))

        # ── sparse matrix descriptor 
        self._mat_desc = ctypes.c_void_p()
        # ── SpSM descriptor 
        self._spsm_desc = ctypes.c_void_p()
        self._b        = cp.zeros((n, nrhs), dtype=np.float64)
        self._x        = cp.zeros((n, nrhs), dtype=np.float64)
        
        if(A_sp.format == 'csr'):
            self.bsz = 1
            
            _check(_lib.cusparseCreateCsr(
                ctypes.byref(self._mat_desc),
                ctypes.c_int64(n), ctypes.c_int64(n), ctypes.c_int64(nnz),
                ctypes.c_void_p(self._indptr.data.ptr),
                ctypes.c_void_p(self._indices.data.ptr),
                ctypes.c_void_p(self._data.data.ptr),
                ctypes.c_int(CUSPARSE_INDEX_32I),   # row-offsets type
                ctypes.c_int(CUSPARSE_INDEX_32I),   # col-indices type
                ctypes.c_int(CUSPARSE_INDEX_BASE_ZERO),
                ctypes.c_int(CUDA_R_64F),
            ))


            # Set fill mode and diag type on the matrix descriptor
            _check(_lib.cusparseSpMatSetAttribute(
                self._mat_desc,
                ctypes.c_int(CUSPARSE_SPMAT_FILL_MODE),
                ctypes.byref(ctypes.c_int(
                    CUSPARSE_FILL_MODE_LOWER if lower else CUSPARSE_FILL_MODE_UPPER)),
                ctypes.c_size_t(ctypes.sizeof(ctypes.c_int)),
            ))

            _check(_lib.cusparseSpMatSetAttribute(
                    self._mat_desc,
                    ctypes.c_int(CUSPARSE_SPMAT_DIAG_TYPE),
                    ctypes.byref(ctypes.c_int(CUSPARSE_DIAG_TYPE_UNIT if unit_diagonal else CUSPARSE_DIAG_TYPE_NON_UNIT)),
                    ctypes.c_size_t(ctypes.sizeof(ctypes.c_int)),
                ))
            
            # ── allocate RHS/solution dense-vector descriptors (shape n×1) ────────
            # These are placeholders; we'll update their data pointers in solve()
            self._b_desc = ctypes.c_void_p()
            self._x_desc = ctypes.c_void_p()
            
            _check(_lib.cusparseCreateDnMat(
                ctypes.byref(self._b_desc),
                ctypes.c_int64(n),
                ctypes.c_int64(nrhs),
                ctypes.c_int64(nrhs),
                ctypes.c_void_p(self._b.data.ptr),
                ctypes.c_int(CUDA_R_64F),
                ctypes.c_int(CUSPARSE_ORDER_ROW),
            ))

            _check(_lib.cusparseCreateDnMat(
                ctypes.byref(self._x_desc),
                ctypes.c_int64(n),
                ctypes.c_int64(nrhs),
                ctypes.c_int64(nrhs),
                ctypes.c_void_p(self._x.data.ptr),
                ctypes.c_int(CUDA_R_64F),
                ctypes.c_int(CUSPARSE_ORDER_ROW),
            ))

            alpha = ctypes.c_double(1.0)
            buf_size = ctypes.c_size_t(0)

            _check(_lib.cusparseSpSM_createDescr(ctypes.byref(self._spsm_desc)))

            _check(_lib.cusparseSpSM_bufferSize(
                self._handle,
                ctypes.c_int(CUSPARSE_OPERATION_NON_TRANSPOSE),
                ctypes.c_int(CUSPARSE_OPERATION_NON_TRANSPOSE),
                ctypes.byref(alpha),
                self._mat_desc,
                self._b_desc,
                self._x_desc,
                ctypes.c_int(CUDA_R_64F),
                ctypes.c_int(CUSPARSE_SPSM_ALG_DEFAULT),
                self._spsm_desc,
                ctypes.byref(buf_size),
            ))

            self._buf = cp.empty(int(buf_size.value) or 1, dtype=np.uint8)
            print(int(buf_size.value), "bytes for SpSM (CSR) analysis buffer")
            _check(_lib.cusparseSpSM_analysis(
                self._handle,
                ctypes.c_int(CUSPARSE_OPERATION_NON_TRANSPOSE),
                ctypes.c_int(CUSPARSE_OPERATION_NON_TRANSPOSE),
                ctypes.byref(alpha),
                self._mat_desc,
                self._b_desc,
                self._x_desc,
                ctypes.c_int(CUDA_R_64F),
                ctypes.c_int(CUSPARSE_SPSM_ALG_DEFAULT),
                self._spsm_desc,
                ctypes.c_void_p(self._buf.data.ptr),
            ))
            
        elif(A_sp.format == 'bsr'):
            blk_sz   = A_sp.blocksize
            assert   blk_sz[0] == blk_sz[1], "only square blocks supported"
            bsz      = blk_sz[0]
            self.bsz = bsz

            
            _check(_lib.cusparseCreateMatDescr(ctypes.byref(self._mat_desc)))
            _check(_lib.cusparseSetMatType     (self._mat_desc, CUSPARSE_MATRIX_TYPE_GENERAL))
            _check(_lib.cusparseSetMatIndexBase(self._mat_desc, CUSPARSE_INDEX_BASE_ZERO))
            _check(_lib.cusparseSetMatFillMode (self._mat_desc, CUSPARSE_FILL_MODE_UPPER if not lower else CUSPARSE_FILL_MODE_LOWER))
            _check(_lib.cusparseSetMatDiagType (self._mat_desc, CUSPARSE_DIAG_TYPE_UNIT if unit_diagonal else CUSPARSE_DIAG_TYPE_NON_UNIT))

            buf_size = ctypes.c_int(0)
            alpha    = ctypes.c_double(1.0)
            _check(_lib.cusparseCreateBsrsm2Info(ctypes.byref(self._spsm_desc)))

            _check(_lib.cusparseDbsrsm2_bufferSize(
                self._handle,
                ctypes.c_int(CUSPARSE_DIRECTION_ROW),
                ctypes.c_int(CUSPARSE_OPERATION_NON_TRANSPOSE),
                ctypes.c_int(CUSPARSE_OPERATION_NON_TRANSPOSE),
                ctypes.c_int(n//bsz),
                ctypes.c_int(nrhs),
                ctypes.c_int(nnz//(bsz*bsz)),
                self._mat_desc,
                ctypes.c_void_p(self._data.data.ptr),
                ctypes.c_void_p(self._indptr.data.ptr),
                ctypes.c_void_p(self._indices.data.ptr),
                ctypes.c_int(bsz),
                self._spsm_desc,
                ctypes.byref(buf_size)
            ))

            self._buf = cp.empty(int(buf_size.value) or 1, dtype=np.uint8)
            print(int(buf_size.value), "bytes for SpSM (BSR) analysis buffer")
            
            _check(_lib.cusparseDbsrsm2_analysis(
                self._handle,
                ctypes.c_int(CUSPARSE_DIRECTION_ROW),
                ctypes.c_int(CUSPARSE_OPERATION_NON_TRANSPOSE),
                ctypes.c_int(CUSPARSE_OPERATION_NON_TRANSPOSE),
                ctypes.c_int(n//bsz),
                ctypes.c_int(nrhs),
                ctypes.c_int(nnz//(bsz*bsz)),
                self._mat_desc,
                ctypes.c_void_p(self._data.data.ptr),
                ctypes.c_void_p(self._indptr.data.ptr),
                ctypes.c_void_p(self._indices.data.ptr),
                ctypes.c_int(bsz),
                self._spsm_desc,
                ctypes.c_int(CUSPARSE_SOLVE_POLICY_USE_LEVEL), 
                ctypes.c_void_p(self._buf.data.ptr),
            ))


        else:
            raise ValueError(f"Unsupported sparse format: {A_sp.format}")

        self.cp_stream = [cp.cuda.Stream(non_blocking=True)]  
        self.cp_ys     = cp.empty((len(self.cp_stream), self.bsz, nrhs), dtype=np.float64)

        cp.cuda.Device().synchronize()

        
        print("Analysis done — descriptor cached")
        self._alpha = ctypes.c_double(1.0)

        with open('../cuda/spsm_bsr_upper_tri_solve.cu', 'r') as f:
            spsm_cuda_code = f.read()
        
        self._spsm_cuda = cp.RawKernel(spsm_cuda_code, 'spsm_bsr_upper_tri_solve')



    def solve(self, b: cp.ndarray) -> cp.ndarray:
        """Solve A x = b. Returns x (new array). b is not modified."""
        if   (self.stype == solver_type.CUSPARSE_SPSM):
            x = cp.empty(b.shape, dtype=self.dtype)
            _check(_lib.cusparseDnMatSetValues(self._b_desc, ctypes.c_void_p(b.data.ptr)))
            _check(_lib.cusparseDnMatSetValues(self._x_desc, ctypes.c_void_p(x.data.ptr)))

            _check(_lib.cusparseSpSM_solve(
                self._handle,
                ctypes.c_int(CUSPARSE_OPERATION_NON_TRANSPOSE),
                ctypes.c_int(CUSPARSE_OPERATION_NON_TRANSPOSE),
                ctypes.byref(self._alpha),
                self._mat_desc,
                self._b_desc,
                self._x_desc,
                ctypes.c_int(CUDA_R_64F),
                ctypes.c_int(CUSPARSE_SPSM_ALG_DEFAULT),
                self._spsm_desc,
            ))

            return x
        elif (self.stype == solver_type.CUSPARSE_BSRSM2):
            bc = cp.array(b, order="F", copy=True)
            x  = cp.empty_like(b, order="F")
            _check(_lib.cusparseDbsrsm2_solve(
                self._handle,
                ctypes.c_int(CUSPARSE_DIRECTION_ROW),
                ctypes.c_int(CUSPARSE_OPERATION_NON_TRANSPOSE),
                ctypes.c_int(CUSPARSE_OPERATION_NON_TRANSPOSE),
                ctypes.c_int(self.n//self.bsz),
                ctypes.c_int(b.shape[1]),
                ctypes.c_int(self.nnz // (self.bsz*self.bsz)),
                ctypes.byref(self._alpha),
                self._mat_desc,
                ctypes.c_void_p(self._data.data.ptr),
                ctypes.c_void_p(self._indptr.data.ptr),
                ctypes.c_void_p(self._indices.data.ptr),
                ctypes.c_int(self.bsz),
                self._spsm_desc,
                ctypes.c_void_p(bc.data.ptr),
                ctypes.c_int(bc.shape[0]),
                ctypes.c_void_p(x.data.ptr),
                ctypes.c_int(x.shape[0]),
                ctypes.c_int(CUSPARSE_SOLVE_POLICY_USE_LEVEL),
                ctypes.c_void_p(self._buf.data.ptr)))
            
            return cp.array(x, order="C", copy=True)  # convert back to C order 
        elif (self.stype == solver_type.CUPY_SPSM_SEQ):
            bsz = self.bsz
            assert self.udiag == True, "unit diagonal only for now in CUPY_SPSM_SEQ mode"
            nbrows            = self.n // bsz
            x                 = cp.copy(b)  # make a copy to write the solution into
            
            #for i in range(nbrows) if self.lower else range(nbrows-1, -1, -1):
            assert self.lower == False, "only upper triangular supported for now in CUPY_SPSM_SEQ mode"
            ns = len(self.cp_stream)
            for i in range(nbrows-1, -1, -1):
                row_start = self._indptr[i]
                row_end   = self._indptr[i+1]

                #print(self._indptr, row_start, row_end, type(row_start), type(row_end), type(self._indptr))
                for idx in range(int(row_start), int(row_end)):
                    j = int(self._indices[idx])
                    if j <= i:
                        continue  # skip diagonal and lower blocks
                    
                    A_block = self._data[idx].reshape((bsz, bsz))

                    # print(i,j, "\n", A_block, "\n")

                    x_block = x[j*bsz:(j+1)*bsz, :]
                    x[i*bsz:(i+1)*bsz, :] -= A_block @ x_block

                #     with self.cp_stream[j % ns]:
                #         A_block = self._data[idx].reshape((bsz, bsz))
                #         x_block = x[j*bsz:(j+1)*bsz, :]

                #         #print(A_block.shape, x_block.shape, (A_block @ x_block).shape, self.cp_ys[j % ns].shape)
                #         self.cp_ys [j % ns] = A_block @ x_block
                
                
                # for idx in range(int(row_start), int(row_end)):
                #     j = int(self._indices[idx])
                #     if j <= i:
                #         continue  # skip diagonal and lower blocks

                #     self.cp_stream[j % ns].synchronize()  # ensure the A_block @ x_block is done
                #     x[i*bsz:(i+1)*bsz, :] -= self.cp_ys[j % ns]

            return x
        elif (self.stype == solver_type.CUPY_SPSM_PAR):
            pass
        elif (self.stype == solver_type.CUDA_CODE):
            #print("executing code code solve")
            bc = cp.array(b, order="F", copy=True)
            x  = cp.empty_like(b, order="F")
            self._spsm_cuda((bc.shape[1], ), (self.bsz, ), (self._data, self._indptr, self._indices,
                             np.int32(self.n//self.bsz), np.int32(bc.shape[1]), np.int32(self.bsz), bc, x), shared_mem=self.bsz * 8)
            return cp.array(x, order="C", copy=True)  # convert back to C order
        else:
            raise ValueError(f"Unsupported sparse format: {self.format}")

    def __del__(self):
        if (self.format == 'csr'):
            try:
                _lib.cusparseDestroyDnMat(self._b_desc)
                _lib.cusparseDestroyDnMat(self._x_desc)
                _lib.cusparseSpSM_destroyDescr(self._spsm_desc)
                _lib.cusparseDestroySpMat(self._mat_desc)
                _lib.cusparseDestroy(self._handle)
            except Exception:
                pass
        elif (self.format == 'bsr'):
            try:
                _lib.cusparseDestroyMatDescr(self._mat_desc)
                _lib.cusparseDestroyBsrsm2Info(self._spsm_desc)
                _lib.cusparseDestroy(self._handle)
            except Exception:
                pass
        else:
            pass

if __name__ == "__main__":
    # Nr   = 860
    # Nvt  = 64
    # n    = Nr * Nvt
    # Nx   = 1600
    # rng  = np.random.default_rng(0)

    # A_sp = sp.random(n, n, density=0.005, format='csr',
    #                 dtype=np.float64, random_state=rng)
    # A_sp = sp.tril(A_sp, format='csr')
    # A_sp.setdiag(rng.uniform(3.0, 5.0, n))
    # A_sp.eliminate_zeros()
    # A_sp.sum_duplicates()
    # A_sp.sort_indices()

    # L_gpu  = cpsp.csr_matrix(A_sp)
    # solver = SpSMSolver(L_gpu, lower=True, nrhs=nrhs)

    import bte_1d3v_solver
    class args:
        def __init__(self, par_file, ef_mode):
            self.par_file = par_file
            self.ef_mode  = ef_mode
    
    bte = bte_1d3v_solver.bte_1d3v(args("bte1d2v/r3_efm_2_conv.toml", 2))
    
    Nx  = bte.dof_x
    Nvt = bte.dof_vt
    Nr  = bte.dof_vr
    n   = Nr * Nvt
    
    dt   = bte.params.dt
    Lop  = sp.csr_matrix(sp.identity(n) - dt * bte.params.np0 * bte.params.n0 * bte.params.tau * bte.op_col_en)
    Dinv = np.linalg.inv(np.array([Lop[i * Nvt: (i+1) * Nvt, i * Nvt: (i+1) * Nvt].toarray() for i in range(Nr)]))
            
    Lsp  = sp.csr_matrix((Nr*Nvt, Nr * Nvt), dtype=np.float64)
    for i in range(Nr):
        Lsp[i * Nvt : (i+1)* Nvt, i * Nvt : (i+1)* Nvt] = Dinv[i]

    Lsp    = Lsp.tocsr()
    Mu     = (Lsp @ Lop).tocsr()
    #Mu.sort_indices()

    Mu_bsr = Mu.tobsr(blocksize=(Nvt, Nvt), copy=True)
    Mu_bsr.sort_indices()

    
    csr          = SpSMSolver(Mu, lower=False, nrhs=Nx, unit_diagonal=True, stype=solver_type.CUSPARSE_SPSM)
    bsr          = SpSMSolver(Mu_bsr, lower=False, nrhs=Nx, unit_diagonal=True, stype=solver_type.CUSPARSE_BSRSM2)
    cp_seq_bsr   = SpSMSolver(Mu_bsr, lower=False, nrhs=Nx, unit_diagonal=True, stype=solver_type.CUPY_SPSM_SEQ)
    cuda_seq_bsr = SpSMSolver(Mu_bsr, lower=False, nrhs=Nx, unit_diagonal=True, stype=solver_type.CUDA_CODE)
    Mu  = cpsp.csr_matrix(Mu)
    
    n_steps = 3
    for t in range(n_steps):
        b      = cp.random.rand(n, Nx)
        cp.cuda.Device().synchronize()

        t1     = perf_counter()
        x1     = bsr.solve(b)
        cp.cuda.Device().synchronize()
        t2     = perf_counter()

        print("cusparse solve time (bsr) = %.4E"%(t2-t1))

        t1     = perf_counter()
        x2     = csr.solve(b)
        cp.cuda.Device().synchronize()
        t2     = perf_counter()
        
        print("cusparse solve time (csr) = %.4E"%(t2-t1))

        t1     = perf_counter()
        x3     = cp_seq_bsr.solve(b)
        cp.cuda.Device().synchronize()
        t2     = perf_counter()
        
        print("cp seq solve time (cp_seq) = %.4E"%(t2-t1))

        t1     = perf_counter()
        x4     = cuda_seq_bsr.solve(b)
        cp.cuda.Device().synchronize()
        t2     = perf_counter()
        
        print("cp seq solve time (cuda_code) = %.4E"%(t2-t1))

        t1     = perf_counter()
        x_ref  = cpsp_lg.spsolve_triangular(Mu, b, lower=False, unit_diagonal=True)
        cp.cuda.Device().synchronize()
        t2     = perf_counter()

        print("cupyx.scipy.sparse.linalg solve time = %.4E"%(t2-t1))

        print("\n\n\n")
        err    = float(cp.max(cp.abs(x2 - cp.asarray(x_ref))))
        print(f"Max error (csr solve) vs scipy: {err:.2e}")

        err    = float(cp.max(cp.abs(x1 - cp.asarray(x_ref))))
        print(f"Max error (bsr solve) vs scipy: {err:.2e}")

        err    = float(cp.max(cp.abs(x3 - cp.asarray(x_ref))))
        print(f"Max error (cp seq) vs scipy: {err:.2e}")

        err    = float(cp.max(cp.abs(x4[:, 0] - cp.asarray(x_ref[:, 0]))))
        print(f"Max error (cuda seq) vs scipy: {err:.2e}")

        # print(b,"\n")
        # print(x,"\n")
        # print(y,"\n")
        # for i in range(Nr):
        #     print(f"Block {i}:")
        #     print("b:")
        #     print(b[i*Nvt:(i+1)*Nvt, :5])
        #     print("x (csr):")
        #     print(x[i*Nvt:(i+1)*Nvt, :5])
        #     print("y (bsr):")
        #     print(y[i*Nvt:(i+1)*Nvt, :5])
        #     print("x_ref (scipy):")
        #     print(x_ref[i*Nvt:(i+1)*Nvt, :5])
        
        # sys.exit(0)
