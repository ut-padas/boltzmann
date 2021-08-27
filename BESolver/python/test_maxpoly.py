import numpy as np
import matplotlib.pyplot as plt
from maxpoly import *
from lagpoly import *
from scipy.interpolate import interp1d
from scipy.interpolate import Rbf
from scipy.signal import savgol_filter
import scipy.fftpack

plot_convergence = True

max_dofs = 50

e_Ar_elastic = np.genfromtxt('/home/dbochkov/Dropbox/Code/boltzmann/BESolver/python/bolsig_data/e_Ar_elastic.dat',delimiter=',')
e_Ar_excitation = np.genfromtxt('/home/dbochkov/Dropbox/Code/boltzmann/BESolver/python/bolsig_data/e_Ar_excitation.dat',delimiter=',')
e_Ar_ionization = np.genfromtxt('/home/dbochkov/Dropbox/Code/boltzmann/BESolver/python/bolsig_data/e_Ar_ionization.dat',delimiter=',')

# sig0 here is essentially arbitrary.  it is intended to be a
# representative cross-section value and is used in
# non-dimensionalizing the electric field
sig0 = 1e-19 # m^2

eveclist = []
dveclist = []
Etillist = []
mulist = []
EbNlist = []

with open('bolsig_data/argon.out', 'r') as F:
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
    EbNlist.append(EbN)

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
        EbNlist.append(EbN)

degrees_all = [3, 6, 10, 20]
# degrees_all = [10, 20, 30, 40]
which_all = [0, 8, 9, 10, 11, 30]
which_all = [0, 2, 4, 6, 8, 10, 11, 35]
which_all = [0, 6, 11, 12, 35]
# which_all = [10]
# which_all = [101, 102, 103]

for which_idx,which in enumerate(which_all):

    if which < 100:
        print(EbNlist[which])
        ehat = 1.5*eveclist[which]/mulist[which]
        pehat = dveclist[which]
        maxw = 2./np.sqrt(np.pi)*(1.5/mulist[which])**(1.5)*np.exp(-ehat)
        # maxw = np.exp(-ehat)

        datax = np.sqrt(ehat)
        datay = pehat/(maxw+1e-20)

        # plt.show()
        # plt.plot(ehat, pehat)
        # plt.plot(ehat, maxw)
        # plt.show()

        print(np.trapz(datay*datax**2,x=datax))

        x_elastic = np.sqrt(1.5*e_Ar_elastic[:,0]/mulist[which])
        x_excitation = np.sqrt(1.5*e_Ar_excitation[:,0]/mulist[which])
        x_ionization = np.sqrt(1.5*e_Ar_ionization[:,0]/mulist[which])

        elastic_cf = interp1d(x_elastic, e_Ar_elastic[:,1], kind='linear', bounds_error=False, fill_value=(e_Ar_elastic[0,1], e_Ar_elastic[-1,1]))
        excitation_cf = interp1d(x_excitation, e_Ar_excitation[:,1], kind='linear', bounds_error=False, fill_value=(e_Ar_excitation[0,1], e_Ar_excitation[-1,1]))
        ionization_cf = interp1d(x_ionization, e_Ar_ionization[:,1], kind='linear', bounds_error=False, fill_value=(e_Ar_ionization[0,1], e_Ar_ionization[-1,1]))

        x = datax
        # x = np.linspace(min(datax),max(datax),num=100)
        # ftest = lambda x: np.interp(x, datax, datay)
        ftest = interp1d(datax, datay, kind='cubic', bounds_error=False, fill_value=(datay[0],datay[-1]))
        # ftest = Rbf(datax, datay, smoothing=0, kernel='gaussian')
        # ftest = interp1d(datax, datay, kind='cubic', fill_value='extrapolate')

        # x = np.linspace(0,max(datax),10000)
        # datay = ftest(x)
        # ftest = interp1d(x, datay, kind='cubic', fill_value='extrapolate')


        N = 500
        # xx = np.linspace(min(datax),max(datax),num=N)
        xx = np.linspace(0,10,num=N)
        # x = np.linspace(0,100,num=N)
        yy = ftest(xx)
        # y = np.concatenate((y, y[-1::-1]))
        # w = scipy.fftpack.rfft(y-1)
        # plt.plot(y)
        # plt.show()
        # plt.plot(abs(w),'o')
        # plt.show()
        # f = scipy.fftpack.rfftfreq(N, x[1]-x[0])
        # spectrum = w**2

        # cutoff_idx = spectrum < (spectrum.max()/10000)
        # w2 = w.copy()
        # w2[cutoff_idx] = 0
        # w2[200:-1] = 0
        # plt.plot(w,'o')
        # plt.plot(w2,'*')
        # plt.show()

        # y2 = 1+scipy.fftpack.irfft(w2)

        y2 = np.exp(savgol_filter(np.log(yy), 51, 3))
        y2 = np.exp(savgol_filter(np.log(y2), 51, 3))

        # print(x)
        # print(y2)

        # plt.semilogy(y)
        # plt.semilogy(y2)
        # plt.show()

        # y = y2[0:N]

        # ftest = lambda xx: np.interp(xx, x, y)
        ftest = interp1d(xx, y2, kind='cubic', bounds_error=False, fill_value=(y2[0],y2[-1]))
        # ftest = Rbf(x, y)
    else:
        x = np.linspace(0,5,num=100)
        if which == 101:
            ftest = lambda x: (1.2 + np.sin(2*x))
        elif which == 102:
            ftest = lambda x: (1.2 + np.cos(2*x))
        elif which == 103:
            ftest = lambda x: (1.5 + np.sin(5*x+1))

        # datay = ftest(x)
        # ftest = interp1d(x, datay, kind='cubic', fill_value='extrapolate')

    [xg, wg] = maxpolygauss(maxpoly_nmax)
    fg = ftest(xg)

    ftest11 = lambda y: ftest(min(x)+.5*(max(x)-min(x))*(y+1))

    a = np.zeros(max_dofs)
    b = np.zeros(max_dofs)
    c = np.polynomial.chebyshev.chebinterpolate(ftest11, max_dofs)
    # c = np.polynomial.chebyshev.chebfit(2*(x-min(x))/(max(x)-min(x))-1, ftest(x), maxpoly_nmax)
    # xtrapz = np.logspace(-10,2,num=50000)
    # xtrapz = np.linspace(min(x),max(x),num=50000)
    xtrapz = np.linspace(0,10,num=10000)
    # plt.plot(xtrapz, lagpolyeval(xtrapz,2))
    # plt.plot(xtrapz, lagpolyeval2(xtrapz,2),'o:')
    # plt.show()

    # plt.show()
    # plt.plot(datax, datay)
    # plt.plot(xtrapz, ftest(xtrapz))
    # plt.show()
    for i in range(max_dofs):
        # a[i] = np.sum(maxpolyeval(xg, i)*np.exp(xg**2)*fg*wg)
        # b[i] = np.sum(lagpolyeval(xg**2, i)*np.exp(xg**2)*fg*wg)
        a[i] = np.sum(maxpolyeval(xg, i)*fg*wg)
        b[i] = np.sum(lagpolyeval(xg**2, i)*fg*wg)
        # a[i] = np.trapz(maxpolyeval(xtrapz, i)*ftest(xtrapz)*np.exp(-xtrapz**2)*xtrapz**2,x=xtrapz) \
        #     / np.trapz(maxpolyeval(xtrapz, i)**2*np.exp(-xtrapz**2)*xtrapz**2,x=xtrapz)
        # b[i] = np.trapz(lagpolyeval(xtrapz**2, i)*ftest(xtrapz)*np.exp(-xtrapz**2)*xtrapz**2,x=xtrapz) \
        #     / np.trapz(lagpolyeval(xtrapz**2, i)**2*np.exp(-xtrapz**2)*xtrapz**2,x=xtrapz)
        # a[i] = np.trapz(maxpolyeval(xtrapz, i)*ftest(xtrapz)*np.exp(-xtrapz**2)*xtrapz**2,x=xtrapz) \
            # / np.sqrt(np.pi) * 4.
        # b[i] = np.trapz(lagpolyeval(xtrapz**2, i)*ftest(xtrapz)*np.exp(-xtrapz**2)*xtrapz**2,x=xtrapz) \
            # / np.sqrt(np.pi) * 4.


    xerror = np.linspace(min(x),max(x),num=10000)

    f_approx_m = np.zeros(len(x))
    f_approx_l = np.zeros(len(x))
    f_approx_c = np.zeros(len(x))
    f_approx_m_err = np.zeros(len(xerror))
    f_approx_l_err = np.zeros(len(xerror))
    f_approx_c_err = np.zeros(len(xerror))
    error_g0_m = np.zeros(max_dofs)
    error_g0_l = np.zeros(max_dofs)
    error_g0_c = np.zeros(max_dofs)
    error_g0_i = np.zeros(max_dofs)
    error_g1_m = np.zeros(max_dofs)
    error_g1_l = np.zeros(max_dofs)
    error_g1_c = np.zeros(max_dofs)
    error_g1_i = np.zeros(max_dofs)
    error_g2_m = np.zeros(max_dofs)
    error_g2_l = np.zeros(max_dofs)
    error_g2_c = np.zeros(max_dofs)
    error_g2_i = np.zeros(max_dofs)
    error_linf_m = np.zeros(max_dofs)
    error_linf_l = np.zeros(max_dofs)
    error_linf_c = np.zeros(max_dofs)
    error_linf_i = np.zeros(max_dofs)
    error_l2_m = np.zeros(max_dofs)
    error_l2_l = np.zeros(max_dofs)
    error_l2_c = np.zeros(max_dofs)
    error_l2_i = np.zeros(max_dofs)

    fttest_err = ftest(xerror)
    exp_err = np.exp(-xerror**2)
    x3exp_err = exp_err*xerror**3
    elastic_err = elastic_cf(xerror)
    excitation_err = excitation_cf(xerror)
    ionization_err = ionization_cf(xerror)

    elastic_scale = np.trapz(fttest_err*elastic_err*x3exp_err, x=xerror)
    excitation_scale = np.trapz(fttest_err*excitation_err*x3exp_err, x=xerror)
    ionization_scale = np.trapz(fttest_err*ionization_err*x3exp_err, x=xerror)

    # for i in range(16):
    for i in range(max_dofs):

        f_approx_m += a[i]*maxpolyeval(x, i)
        f_approx_l += b[i]*lagpolyeval(x**2, i)
        f_approx_c  = np.polynomial.chebyshev.chebval(2*(x-min(x))/(max(x)-min(x))-1.,c[0:i+1])
        xi = np.linspace(min(x), max(x), i+1)
        yi = ftest(xi)
        f_approx_i = np.interp(x, xi, yi)

        f_approx_m_err += a[i]*maxpolyeval(xerror, i)
        f_approx_l_err += b[i]*lagpolyeval(xerror**2, i)
        f_approx_c_err  = np.polynomial.chebyshev.chebval(2*(xerror-min(x))/(max(x)-min(x))-1.,c[0:i+1])
        f_approx_i_err = np.interp(xerror, xi, yi)

        error_g0_m[i] = np.trapz((f_approx_m_err-fttest_err)*elastic_err*x3exp_err, x=xerror)/elastic_scale
        error_g0_l[i] = np.trapz((f_approx_l_err-fttest_err)*elastic_err*x3exp_err, x=xerror)/elastic_scale
        error_g0_c[i] = np.trapz((f_approx_c_err-fttest_err)*elastic_err*x3exp_err, x=xerror)/elastic_scale
        error_g0_i[i] = np.trapz((f_approx_i_err-fttest_err)*elastic_err*x3exp_err, x=xerror)/elastic_scale

        error_g1_m[i] = np.trapz((f_approx_m_err-fttest_err)*excitation_err*x3exp_err, x=xerror)/excitation_scale
        error_g1_l[i] = np.trapz((f_approx_l_err-fttest_err)*excitation_err*x3exp_err, x=xerror)/excitation_scale
        error_g1_c[i] = np.trapz((f_approx_c_err-fttest_err)*excitation_err*x3exp_err, x=xerror)/excitation_scale
        error_g1_i[i] = np.trapz((f_approx_i_err-fttest_err)*excitation_err*x3exp_err, x=xerror)/excitation_scale

        error_g2_m[i] = np.trapz((f_approx_m_err-fttest_err)*ionization_err*x3exp_err, x=xerror)/ionization_scale
        error_g2_l[i] = np.trapz((f_approx_l_err-fttest_err)*ionization_err*x3exp_err, x=xerror)/ionization_scale
        error_g2_c[i] = np.trapz((f_approx_c_err-fttest_err)*ionization_err*x3exp_err, x=xerror)/ionization_scale
        error_g2_i[i] = np.trapz((f_approx_i_err-fttest_err)*ionization_err*x3exp_err, x=xerror)/ionization_scale

        error_linf_m[i] = max(abs((f_approx_m_err-fttest_err)*exp_err))
        error_linf_l[i] = max(abs((f_approx_l_err-fttest_err)*exp_err))
        error_linf_c[i] = max(abs((f_approx_c_err-fttest_err)*exp_err))
        error_linf_i[i] = max(abs((f_approx_i_err-fttest_err)*exp_err))
        error_l2_m[i] = np.sqrt(sum(abs((f_approx_m_err-fttest_err)*exp_err)**2)/len(xerror))
        error_l2_l[i] = np.sqrt(sum(abs((f_approx_l_err-fttest_err)*exp_err)**2)/len(xerror))
        error_l2_c[i] = np.sqrt(sum(abs((f_approx_c_err-fttest_err)*exp_err)**2)/len(xerror))
        error_l2_i[i] = np.sqrt(sum(abs((f_approx_i_err-fttest_err)*exp_err)**2)/len(xerror))

        if plot_convergence == False:
            if i in degrees_all:
                plt.subplot(len(which_all), len(degrees_all), which_idx*len(degrees_all)+degrees_all.index(i)+1)
                plt.semilogy(x, f_approx_m*np.exp(-x**2),'-')
                plt.semilogy(x, f_approx_l*np.exp(-x**2),'-')
                plt.semilogy(x, f_approx_c*np.exp(-x**2),'-')
                plt.semilogy(x, f_approx_i*np.exp(-x**2),'-')
                plt.semilogy(x, ftest(x)*np.exp(-x**2),'k')
                plt.semilogy(x, np.exp(-x**2),'k--')
                # plt.plot(x, f_approx_m*np.exp(-x**2),'-')
                # plt.plot(x, f_approx_l*np.exp(-x**2),'-')
                # plt.plot(x, f_approx_c*np.exp(-x**2),'-')
                # plt.plot(x, f_approx_i*np.exp(-x**2),'-')
                # plt.plot(x, ftest(x)*np.exp(-x**2),'k')
                # plt.plot(x, np.exp(-x**2),'k--')
                # plt.semilogy(x, f_approx_m*np.exp(-x**2)*x**2,'o--')
                # plt.semilogy(x, f_approx_l*np.exp(-x**2)*x**2,'x:')
                # plt.semilogy(x, f_approx_c*np.exp(-x**2)*x**2,'s:')
                # plt.semilogy(x, f_approx_i*np.exp(-x**2)*x**2,'s:')
                # plt.semilogy(x, ftest(x)*np.exp(-x**2)*x**2,'k')
                # plt.semilogy(x, f_approx_m,'-')
                # plt.semilogy(x, f_approx_l,'-')
                # plt.semilogy(x, f_approx_c,'-')
                # plt.semilogy(x, f_approx_i,'-')
                # plt.semilogy(x, ftest(x),'k-')
                # plt.plot(x, elastic_cf(x),'-')
                # plt.plot(x, excitation_cf(x),'-')
                # plt.plot(x, ionization_cf(x),'-')
                # plt.ylim((-0.25,1.5))
                plt.xlim((0,5))

                if degrees_all.index(i) == 0:
                    plt.ylabel('Density')

                if which_idx == len(which_all)-1:
                    plt.xlabel('Speed')

                if which_idx == 0:
                    plt.title('Num. of polys: '+str(i))

                if which_idx == 0 and degrees_all.index(i) == 0:
                    plt.legend(['Maxwell','Laguerre','Chebyshev','Linear','Bolsig','Maxwellian'], loc='upper right')

                plt.grid()


    if plot_convergence:
        ax = plt.subplot(len(which_all), 5, which_idx*5+1)
        plt.semilogy(error_linf_m)
        plt.semilogy(error_linf_l)
        plt.semilogy(error_linf_c)
        plt.semilogy(error_linf_i)
        plt.grid()
        plt.ylabel('Error')
        if which_idx == len(which_all)-1:
            plt.xlabel('Number of polynomials')
        if which_idx == 0:
            plt.title('$L_\infty$')

        ax = plt.subplot(len(which_all), 5, which_idx*5+2)
        plt.semilogy(error_l2_m)
        plt.semilogy(error_l2_l)
        plt.semilogy(error_l2_c)
        plt.semilogy(error_l2_i)
        plt.grid()
        if which_idx == len(which_all)-1:
            plt.xlabel('Number of polynomials')
        if which_idx == 0:
            plt.title('$L_2$')

        ax = plt.subplot(len(which_all), 5, which_idx*5+3)
        plt.semilogy(abs(error_g0_m))
        plt.semilogy(abs(error_g0_l))
        plt.semilogy(abs(error_g0_c))
        plt.semilogy(abs(error_g0_i))
        plt.grid()
        if which_idx == len(which_all)-1:
            plt.xlabel('Number of polynomials')
        if which_idx == 0:
            plt.title('Rate (elastic)')

        ax = plt.subplot(len(which_all), 5, which_idx*5+4)
        plt.semilogy(abs(error_g1_m))
        plt.semilogy(abs(error_g1_l))
        plt.semilogy(abs(error_g1_c))
        plt.semilogy(abs(error_g1_i))
        plt.grid()
        if which_idx == len(which_all)-1:
            plt.xlabel('Number of polynomials')
        if which_idx == 0:
            plt.title('Rate (excitation)')

        ax = plt.subplot(len(which_all), 5, which_idx*5+5)
        plt.semilogy(abs(error_g2_m))
        plt.semilogy(abs(error_g2_l))
        plt.semilogy(abs(error_g2_c))
        plt.semilogy(abs(error_g2_i))
        plt.grid()

        if which_idx == len(which_all)-1:
            plt.xlabel('Number of polynomials')
        if which_idx == 0:
            plt.legend(['Maxwell','Laguerre','Chebyshev','Linear'])
            plt.title('Rate (ionization)')

        # plt.show()

# for p in range(8,10):
#     plt.plot(maxpoly_nodes[idx_s(p):idx_e(p)], maxpoly_weights[idx_s(p):idx_e(p)], 'o')

# plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.01)
plt.show()