import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import scipy.sparse.linalg
from utils import gmres
from utils import gmres_counter

if __name__ == "__main__":
    
    

    n = 1000
    np.random.seed(0)
    A = np.eye(n) + 1e-1 * np.random.rand(n, n)#diags([1, 2, 3], [0, 1, 2], shape=(n, n)).toarray()
    M = None#np.diag(1/np.diag(A))
    b = np.random.rand(n)


    #### Note scipy gmres uses left preconditioning, while our gmres uses right preconditioning. 
    xp       = cp
    counter1 = gmres_counter(disp=True, store_residuals=True)
    x,  info = gmres(xp, xp.asarray(A), xp.asarray(b), x0=xp.zeros_like(b), atol=1e-20, rtol=1e-12, maxiter=100, restart=20, callback=counter1, M=M)
    
    print("\n\n")

    counter2= gmres_counter(disp=True, store_residuals=True)
    x1, info= scipy.sparse.linalg.gmres(A, b, x0=np.zeros_like(b), atol=1e-20, rtol=1e-12, maxiter=100, restart=20,  callback=counter2, M=M)

    # print("GMRES info : ", info)
    # print("GMRES iter : ", counter.niter)
    # print("GMRES resids : ", counter.residuals)


    plt.semilogy(counter1.residuals, 'x-', label=r"custom gmres")
    plt.semilogy(counter2.residuals, 'o-', label=r"scipy gmres")
    #plt.semilogy(np.abs(1 -np.array(counter1.residuals)/np.array(counter2.residuals)), label="custom gmres")
    plt.xlabel("GMRES iteration")
    plt.ylabel("GMRES residual")
    plt.title("GMRES convergence history")
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()




