import numpy as np
import matplotlib.pyplot as plt
from maxpoly import *
from lagpoly import *
from scipy.interpolate import interp1d
from scipy.interpolate import Rbf

# sig0 here is essentially arbitrary.  it is intended to be a
# representative cross-section value and is used in
# non-dimensionalizing the electric field
sig0 = 1e-19 # m^2

eveclist = []
dveclist = []
Etillist = []
mulist = []

with open('/home/dbochkov/Dropbox/Code/boltzmann/BESolver/bolsig-example/inputs/argon.out', 'r') as F:
    line = F.readline()
    print(line)
    EbN = 0
    mu = 0
    while (line!='Energy (eV) EEDF (eV-3/2) Anisotropy\n'):

        if (len(line)>=23):
            if (line[0:23]=='Electric field / N (Td)'):
                EbN = float(line.split(")",2)[1])*1e-21 # 1e-21 converts from Td to V*m^2
                print("Found EbN = ", EbN)

        if (len(line)>=16):
            if (line[0:16]=='Mean energy (eV)'):
                mu = float(line.split(")",2)[1]) # while it says eV, the unit here is actually V
                print("Found mu = ", mu)

        line = F.readline()


    print(EbN, mu)
    mulist.append(mu)

    Et = 1.5*EbN/sig0/mu
    Etillist.append(Et)

    while (line!=''):
        line = F.readline()
        elist = []
        dlist = []
        while (line!=' \n'):
            col = line.split()
            energy = float(col[0])
            distrib = float(col[1])

            elist.append(energy)
            dlist.append(distrib)

            line = F.readline()

        print("Adding vectors to lists for Etil = {0:.3e}...".format(Et))
        eveclist.append(np.array(elist))
        dveclist.append(np.array(dlist))

        while ( (line!='Energy (eV) EEDF (eV-3/2) Anisotropy\n') and
                (line!='') ):

            if (len(line)>=23):
                if (line[0:23]=='Electric field / N (Td)'):
                    EbN = float(line.split(")",2)[1])*1e-21 # 1e-21 converts from Td to V*m^2
                    print("  EbN = {0:.3e} [V m^2]".format(EbN))

            if (len(line)>=16):
                if (line[0:16]=='Mean energy (eV)'):
                    mu = float(line.split(")",2)[1])
                    print("  mu = {0:.3e} [eV]".format(mu))

            line = F.readline()

        mulist.append(mu)

        Et = 1.5*EbN/sig0/mu
        Etillist.append(Et)

which_all = [1, 8, 9, 10, 11, 31]
degrees_all = [3, 6, 10, 20]
# which_all = [101, 102, 103]

for which_idx,which in enumerate(which_all):

    if which < 100:
        ehat = 1.5*eveclist[which]/mulist[which]
        pehat = dveclist[which]
        maxw = 2./np.sqrt(np.pi)*(1.5/mulist[which])**(1.5)*np.exp(-ehat)
        # maxw = np.exp(-ehat)

        datax = np.sqrt(ehat)
        datay = pehat/(maxw+1e-20) - 1. 

        x = datax
        # x = np.linspace(min(datax),max(datax),num=100)
        # ftest = lambda x: np.interp(x, datax, datay)
        # ftest = Rbf(datax, datay, smoothing=10)
        ftest = interp1d(datax, datay, kind='cubic', fill_value='extrapolate')
    else:
        x = np.linspace(0,5,num=100)
        if which == 101:
            ftest = lambda x: (1.2 + np.sin(2*x))
        elif which == 102:
            ftest = lambda x: (1.2 + np.cos(2*x))
        elif which == 103:
            ftest = lambda x: (1.5 + np.sin(5*x))

    [xg, wg] = maxpolygauss(maxpoly_nmax)
    # [xg, wg] = maxpolygauss(20)
    fg = ftest(xg)

    # plt.plot(x, ftest(x))
    # plt.show()

    ftest11 = lambda y: ftest(min(x)+.5*(max(x)-min(x))*(y+1))

    a = np.zeros(maxpoly_nmax+1)
    b = np.zeros(maxpoly_nmax+1)
    c = np.polynomial.chebyshev.chebinterpolate(ftest11, maxpoly_nmax)
    # c = np.polynomial.chebyshev.chebfit(2*(x-min(x))/(max(x)-min(x))-1, ftest(x), maxpoly_nmax)
    for i in range(maxpoly_nmax+1):
        # a[i] = np.sum(maxpolyeval(xg, i)*np.exp(xg**2)*fg*wg)
        # b[i] = np.sum(lagpolyeval(xg**2, i)*np.exp(xg**2)*fg*wg)
        a[i] = np.sum(maxpolyeval(xg, i)*fg*wg)/np.sum(maxpolyeval(xg, i)**2*wg)
        b[i] = np.sum(lagpolyeval(xg**2, i)*fg*wg)/np.sum(lagpolyeval(xg**2, i)**2*wg)
        # a[i] = np.trapz(maxpolyeval(x, i)*ftest(x)*np.exp(-x**2)*x**2,x=x) / np.trapz(maxpolyeval(x, i)**2*np.exp(-x**2)*x**2,x=x)
        # b[i] = np.trapz(lagpolyeval(x**2, i)*ftest(x)*np.exp(-x**2)*x**2,x=x) / np.trapz(lagpolyeval(x**2, i)**2*ftest(x)*np.exp(-x**2)*x**2,x=x)

    f_approx_m = np.zeros(len(x))
    f_approx_l = np.zeros(len(x))
    f_approx_c = np.zeros(len(x))
    error_m = np.zeros(maxpoly_nmax+1)
    error_l = np.zeros(maxpoly_nmax+1)
    error_c = np.zeros(maxpoly_nmax+1)


    # for i in range(16):
    for i in range(maxpoly_nmax+1):
        f_approx_m += a[i]*maxpolyeval(x, i)
        f_approx_l += b[i]*lagpolyeval(x**2, i)
        f_approx_c  = np.polynomial.chebyshev.chebval(2*(x-min(x))/(max(x)-min(x))-1.,c[0:i+1])
        # error_m[i] = sum(abs(f_approx_m*np.exp(-x**2)-ftest(x)))
        # error_l[i] = sum(abs(f_approx_l*np.exp(-x**2)-ftest(x)))
        # error_m[i] = sum((f_approx_m-ftest(x))**2*np.exp(-x**2)*x**2)
        # error_l[i] = sum((f_approx_l-ftest(x))**2*np.exp(-x**2)*x**2)
        # error_c[i] = sum((f_approx_c-ftest(x))**2*np.exp(-x**2)*x**2)
        error_m[i] = max(abs((f_approx_m-ftest(x))))
        error_l[i] = max(abs((f_approx_l-ftest(x))))
        error_c[i] = max(abs((f_approx_c-ftest(x))))
        
        if i in degrees_all:
            plt.subplot(len(which_all), len(degrees_all)+1, which_idx*(len(degrees_all)+1)+degrees_all.index(i)+1)
            # plt.plot(x, f_approx_m*np.exp(-x**2)*x**2,'o--')
            # plt.plot(x, f_approx_l*np.exp(-x**2)*x**2,'x:')
            # plt.plot(x, f_approx_c*np.exp(-x**2)*x**2,'s:')
            # plt.plot(x, ftest(x)*np.exp(-x**2)*x**2,'k')
            plt.plot(x, f_approx_m,'o--')
            plt.plot(x, f_approx_l,'x:')
            plt.plot(x, f_approx_c,'s:')
            plt.plot(x, ftest(x),'k')
            # plt.xlim((0,4))

            if degrees_all.index(i) == 0:
                plt.ylabel('Density')

            if which_idx == len(which_all)-1:
                plt.xlabel('Speed')

            if which_idx == 0:
                plt.title('Num. of polys: '+str(i))

            plt.grid()
            plt.legend(['Maxwell','Laguerre','Chebyshev','Exact'])




    ax = plt.subplot(len(which_all), len(degrees_all)+1, (which_idx+1)*(len(degrees_all)+1))
    plt.semilogy(error_m)
    plt.semilogy(error_l)
    plt.semilogy(error_c)

    if which_idx == len(which_all)-1:
        plt.xlabel('Number of polynomials')

    plt.ylabel('Error')
    plt.grid()
    plt.legend(['Maxwell','Laguerre','Chebyshev'])
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

# for p in range(8,10):
#     plt.plot(maxpoly_nodes[idx_s(p):idx_e(p)], maxpoly_weights[idx_s(p):idx_e(p)], 'o')

# plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.01)
plt.show()