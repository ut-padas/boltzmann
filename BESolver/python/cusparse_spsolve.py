"""
Sparse triangular solve with explicit SpSM analysis reuse — CuPy 14 / CUDA 12+

cusparseSpSM_analysis  → called ONCE before the time loop
cusparseSpSM_solve     → called every timestep, reuses the cached analysis
"""
import ctypes, ctypes.util
import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cpsp
import cupyx.scipy.sparse.linalg as cpsp_lg
import scipy.sparse as sp
from time import perf_counter, sleep

# ── load cusparse shared library ──────────────────────────────────────────────
_lib = ctypes.CDLL(ctypes.util.find_library("cusparse") or "libcusparse.so")

# ── enums (values from cusparse.h) ────────────────────────────────────────────
CUSPARSE_OPERATION_NON_TRANSPOSE = 0
CUSPARSE_FILL_MODE_LOWER         = 0
CUSPARSE_FILL_MODE_UPPER         = 1
CUSPARSE_DIAG_TYPE_NON_UNIT      = 0
CUSPARSE_INDEX_32I               = 2
CUSPARSE_INDEX_64I               = 3
CUSPARSE_INDEX_BASE_ZERO         = 0
CUDA_R_64F                       = 1   # double
CUSPARSE_SPSM_ALG_DEFAULT        = 0
CUSPARSE_ORDER_COL               = 1   # column-major (F-order)
CUSPARSE_ORDER_ROW               = 2   # row-major    (C-order)

def _check(status):
    if status != 0:
        raise RuntimeError(f"cuSPARSE error: status={status}")


class SpSMSolver:
    """
    Wraps a single constant sparse triangular matrix for repeated solves.
    Analysis (the expensive part) runs once in __init__; each solve() call
    only runs cusparseSpSM_solve.
    """

    def __init__(self, A_csr: cpsp.csr_matrix, nrhs: int = 1, lower: bool = True, dtype=np.float64):
        self.lower = lower
        self.dtype = np.dtype(dtype)
        assert self.dtype == np.float64, "only float64 shown here; extend for float32"
        n = A_csr.shape[0]
        self.n = n

        # Keep device arrays alive — their pointers are baked into descriptors
        self._data    = cp.asarray(A_csr.data,    dtype=np.float64)
        self._indptr  = cp.asarray(A_csr.indptr,  dtype=np.int32)
        self._indices = cp.asarray(A_csr.indices, dtype=np.int32)
        nnz = int(A_csr.nnz)

        # ── create handle ─────────────────────────────────────────────────────
        self._handle = ctypes.c_void_p()
        _check(_lib.cusparseCreate(ctypes.byref(self._handle)))

        # ── sparse matrix descriptor ──────────────────────────────────────────
        self._mat_desc = ctypes.c_void_p()
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
            ctypes.c_int(0),                    # CUSPARSE_SPMAT_FILL_MODE = 0
            ctypes.byref(ctypes.c_int(
                CUSPARSE_FILL_MODE_LOWER if lower else CUSPARSE_FILL_MODE_UPPER)),
            ctypes.c_size_t(ctypes.sizeof(ctypes.c_int)),
        ))
        _check(_lib.cusparseSpMatSetAttribute(
            self._mat_desc,
            ctypes.c_int(1),                    # CUSPARSE_SPMAT_DIAG_TYPE  = 1
            ctypes.byref(ctypes.c_int(CUSPARSE_DIAG_TYPE_NON_UNIT)),
            ctypes.c_size_t(ctypes.sizeof(ctypes.c_int)),
        ))

        # ── SpSM descriptor ───────────────────────────────────────────────────
        self._spsm_desc = ctypes.c_void_p()
        _check(_lib.cusparseSpSM_createDescr(ctypes.byref(self._spsm_desc)))

        # ── allocate RHS/solution dense-vector descriptors (shape n×1) ────────
        # These are placeholders; we'll update their data pointers in solve()
        self._b_desc = ctypes.c_void_p()
        self._x_desc = ctypes.c_void_p()
        _dummy       = cp.zeros(n, dtype=np.float64)
        _dummy_m     = cp.zeros((n, nrhs), dtype=np.float64)

        _check(_lib.cusparseCreateDnMat(
            ctypes.byref(self._b_desc),
            ctypes.c_int64(n),
            ctypes.c_int64(nrhs),
            ctypes.c_int64(nrhs),
            ctypes.c_void_p(_dummy_m.data.ptr),
            ctypes.c_int(CUDA_R_64F),
            ctypes.c_int(CUSPARSE_ORDER_ROW),
        ))

        _check(_lib.cusparseCreateDnMat(
            ctypes.byref(self._x_desc),
            ctypes.c_int64(n),
            ctypes.c_int64(nrhs),
            ctypes.c_int64(nrhs),
            ctypes.c_void_p(_dummy_m.data.ptr),
            ctypes.c_int(CUDA_R_64F),
            ctypes.c_int(CUSPARSE_ORDER_ROW),
        ))

        # _check(_lib.cusparseCreateDnVec(
        #     ctypes.byref(self._b_desc),
        #     ctypes.c_int64(n),
        #     ctypes.c_void_p(_dummy.data.ptr),
        #     ctypes.c_int(CUDA_R_64F),
        # ))

        # _check(_lib.cusparseCreateDnVec(
        #     ctypes.byref(self._x_desc),
        #     ctypes.c_int64(n),
        #     ctypes.c_void_p(_dummy.data.ptr),
        #     ctypes.c_int(CUDA_R_64F),
        # ))

        # ── bufferSize then analysis (once) ───────────────────────────────────
        alpha = ctypes.c_double(1.0)
        buf_size = ctypes.c_size_t(0)
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

        print(f"Running SpSM analysis (once)... buffer={buf_size.value/1024:.1f} KB")
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
        cp.cuda.Device().synchronize()
        print("Analysis done — descriptor cached")

        self._alpha = ctypes.c_double(1.0)

    def solve(self, b: cp.ndarray) -> cp.ndarray:
        """Solve A x = b. Returns x (new array). b is not modified."""
        x = cp.empty(b.shape, dtype=self.dtype)
        
        _check(_lib.cusparseDnMatSetValues(self._b_desc, ctypes.c_void_p(b.data.ptr)))
        _check(_lib.cusparseDnMatSetValues(self._x_desc, ctypes.c_void_p(x.data.ptr)))

        # # Point dense-vector descriptors at the current b and x buffers
        # _check(_lib.cusparseSetDnVecValues(
        #     self._b_desc, ctypes.c_void_p(b.data.ptr)))
        # _check(_lib.cusparseSetDnVecValues(
        #     self._x_desc, ctypes.c_void_p(x.data.ptr)))

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

    def __del__(self):
        try:
            _lib.cusparseDestroyDnMat(self._b_desc)
            _lib.cusparseDestroyDnMat(self._x_desc)
            _lib.cusparseSpSM_destroyDescr(self._spsm_desc)
            _lib.cusparseDestroySpMat(self._mat_desc)
            _lib.cusparseDestroy(self._handle)
        except Exception:
            pass

if __name__ == "__main__":
    # ── build operator (constant in time) ─────────────────────────────────────────
    n    = 820 * 64
    nrhs = 1600
    rng  = np.random.default_rng(0)

    A_sp = sp.random(n, n, density=0.005, format='csr',
                    dtype=np.float64, random_state=rng)
    A_sp = sp.tril(A_sp, format='csr')
    A_sp.setdiag(rng.uniform(3.0, 5.0, n))
    A_sp.eliminate_zeros()
    A_sp.sum_duplicates()
    A_sp.sort_indices()

    L_gpu = cpsp.csr_matrix(A_sp)

    # ─ one-time setup ───────────────────────────────────────────────────────────
    solver = SpSMSolver(L_gpu, lower=True, nrhs=nrhs)


    # # ── time loop — only solve() runs each step ───────────────────────────────────
    n_steps = 10
    # x = None

    for t in range(n_steps):
        b      = cp.random.rand(n, nrhs)
        cp.cuda.Device().synchronize()

        t1     = perf_counter()
        x      = solver.solve(b)
        cp.cuda.Device().synchronize()
        t2     = perf_counter()

        print("cusparse solve time = %.4E"%(t2-t1))

        t1     = perf_counter()
        x_ref  = cpsp_lg.spsolve_triangular(L_gpu, b, lower=True)
        cp.cuda.Device().synchronize()
        t2     = perf_counter()

        print("cupyx.scipy.sparse.linalg solve time = %.4E"%(t2-t1))

        err    = float(cp.max(cp.abs(x - cp.asarray(x_ref))))
        print(f"Max error vs scipy: {err:.2e}")

