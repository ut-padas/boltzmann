"""
Module for ROM helper functions
"""
import numpy as np
import scipy
import scipy.linalg
import matplotlib.pyplot as plt
import sys


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

def eigen_plot(A, labels, fname, xp=np):
    #eig_decomp = [xp.linalg.eig(a) for a in A]
    schur_decomp = [scipy.linalg.schur(a, output="complex") for a in A]

    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.figure(figsize=(8, 8), dpi=300)
    for i in range(len(schur_decomp)):
        T = schur_decomp[i][0]
        D = np.diag(T)
        print(i, " %.15E, %.15E "%(np.min(np.real(D)), np.max(np.real(D))))
        plt.scatter(np.real(D), np.imag(D), s=4, label=labels[i], facecolors='none', edgecolors=cycle[i], alpha=0.3)
    
    plt.legend()
    plt.grid(visible=True)
    plt.ylabel(r"Im")
    plt.xlabel(r"Re")
    plt.savefig(fname)
    plt.close()

def fourier_modes(qoi, tt, dt, k, xp):
    axis           = 0
    fqoi           = xp.fft.rfft(qoi, axis=axis)
    freq           = xp.fft.rfftfreq(len(tt), d=dt)

    assert qoi.shape[axis] == len(tt)

    nf             = xp.ones(len(freq)) * (2/len(tt))
    nf[0]          = 1/len(tt)
    fmodes         = nf[:, xp.newaxis] * fqoi 
    a              = xp.concatenate((xp.real(fmodes[0:k])[xp.newaxis,:, :], xp.imag(fmodes[0:k])[xp.newaxis,:,:]), axis=0)
    a              = np.swapaxes(a, 0, 1)
    
    # print(xp.real(fmodes[0:k]).shape)
    # print(xp.real(fmodes[0:k])[xp.newaxis,:,:].shape)
    # print(a.shape)
    # sys.exit(0)


    return a, freq[0:k]

def stabilize_flowmap(A:np.array, dt):
    # T, Q = scipy.linalg.schur(A, output="complex")
    # eig  = np.diag(T)

    # idx  = np.arange(A.shape[0])[np.real(eig)>0]
    # print("number of unstable modes = %d "%(len(idx)))
    # print(eig[idx])

    # # for i in range(len(idx)):
    # #     k       = idx[i] 
    # #     T[k, k] = 0 + np.imag(T[k,k]) * 1j

    # As = Q @ T @ Q.conjugate().T
    # print(np.linalg.norm(np.imag(As)))
    # As = np.real(As)
    # return As

    Ir   = np.eye(A.shape[0])
    L    = np.linalg.inv(Ir - dt * A)
    D, Q = np.linalg.eig(L)
    Qinv = np.linalg.inv(Q)

    print("||Ir - QQ^{-1}|| / ||Ir|| = %.8E"%(np.linalg.norm(Ir - Q @ Qinv)/np.linalg.norm(Ir)))
    print("||Ir - QQ^{-1}|| / ||Ir|| = %.8E"%(np.linalg.norm(Ir - Qinv @ Q)/np.linalg.norm(Ir)))
    print("||L - Q D Q^{-1}||/||L||  = %.8E"%(np.linalg.norm(L - np.real((Q * D) @ Qinv))/np.linalg.norm(L)))

    idx  = np.arange(A.shape[0])[np.real(D)>1]
    print("number of unstable modes = %d "%(len(idx)))
    print(D[idx])

    for i in range(len(idx)):
        k       = idx[i] 
        D[k]    = 0.9 + np.imag(D[k]) * 1j
    
    Ls = (Q * D) @ Qinv
    print(np.linalg.norm(np.imag(Ls)))
    return np.real(Ls)








    

    

