"""
Module for ROM helper functions
"""
import numpy as np
import scipy
import scipy.linalg


def assemble_mat(op_dim:tuple, Lop:scipy.sparse.linalg.LinearOperator, xp=np):
    assert len(op_dim) == 2
    Imat = xp.eye(op_dim[1])

    Aop  = Lop.matmat(Imat)

    assert Aop.shape == op_dim
    return Aop

def id_decomp(A,k,use_sampling=False):
    '''
    Parameters
    ----------
    A : matrix that we want to approximate
    k : target rank
    use_sampling: if True we use randomized row sampling to reduce costs


    Returns
    -------
    B : m-by-k matrix : k columns of A  that approximate range(A)
    Z : k-by-n matrix : computed such that B@Z ~ A

    '''
    m,n = A.shape
    if use_sampling:
        Om = 1/np.sqrt(k) * np.random.randn(k+8,m)
        Y= Om@A
    else:
        Y = A
    
    
    Q,R,p = scipy.linalg.qr(Y, mode='economic', pivoting=True)
    B = A[:,p[:k]]
    if use_sampling:
        Qs,Rs=scipy.linalg.qr(B,mode='economic')
    else:
        Qs = Q[:,:k]
        Rs = R[:k,:k]

    #print(Rs.shape, (Qs.T@A).shape, k, A.shape)        
    Z = scipy.linalg.solve_triangular(Rs, Qs.T@A)
    
    return B,Z

def adaptive_id_decomp(A, eps, k_init, use_sampling=False):
    k     = k_init
    B, Z  = id_decomp(A, k, use_sampling)

    normA = np.linalg.norm(A)

    rr    = np.linalg.norm(A - B@Z)/normA
    it    = 0 
    while (rr > eps):
        print("[adaptive_id_decomp] : iter = %4d k = %d rtol = %.8E"%(it, k, rr))
        k     = min(2 * k, A.shape[0])
        B, Z  = id_decomp(A, k, use_sampling)
        rr    = np.linalg.norm(A - B@Z)/normA

        if (k == A.shape[1]):
            break

        it +=1

    return B, Z

def rsvd(A, k, power_iter= 1, xp=np, rseed=0, Omega=None):
    if Omega is None:
        xp.random.seed(rseed)
        Omega = xp.random.rand(A.shape[1], k)
    
    assert Omega.shape[1] == k

    Y     = A.matmat(Omega)
    
    for q in range(power_iter):
        Y = A.matmat(A.rmatmat(Y))

    Q, _  = xp.linalg.qr(Y)
    B     = (A.rmatmat(Q)).T
    
    u_tilde, s, v = np.linalg.svd(B, full_matrices = 0)
    u = Q @ u_tilde
    return u, s, v

def sherman_morrison_woodberry(Ainv, U, V, xp=np):
    """
    computes the sherman morrion woodberry identity
    B = (A + UV) and computes B^{-1}
    """

    assert Ainv.shape[0] == Ainv.shape[1]
    assert U.shape[1]    == V.shape[1]
    assert U.shape[0]    == Ainv.shape[0]
    assert V.shape[0]    == Ainv.shape[0]

    Ik = xp.eye(U.shape[1])
    return Ainv - Ainv @ U @ xp.linalg.inv(Ik + V @ Ainv @ U) @ V @ Ainv
