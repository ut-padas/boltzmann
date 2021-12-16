import numpy as np
import matplotlib.pyplot as plt
from maxpoly import *
from lagpoly import *
from scipy.interpolate import interp1d
from scipy.interpolate import Rbf
from scipy.signal import savgol_filter
import scipy.fftpack

# max dofs used for represenation
max_dofs = 250

# bolsig+ data will be extended up to this point (if needed)
ehat_ext_max = 49

# use Gaussian quadratures or trapezoidal integration for projection
use_gauss_for_proj = True

# order of Gaussian quadrature used for projection
gauss_q_order_proj = 500

# number of trapezoidal points used for projection
trapz_num_q_points_proj = 100000

# number of points to check error
error_num_points = 10000

# temperature scalling
tscale = 1.0

 # fraction of point used to extend tails
frac = 0.1

e_Ar_elastic = np.genfromtxt('/home/dbochkov/Dropbox/Code/boltzmann/BESolver/python/bolsig_data/e_Ar_elastic.dat',delimiter=',')
e_Ar_excitation = np.genfromtxt('/home/dbochkov/Dropbox/Code/boltzmann/BESolver/python/bolsig_data/e_Ar_excitation.dat',delimiter=',')
e_Ar_ionization = np.genfromtxt('/home/dbochkov/Dropbox/Code/boltzmann/BESolver/python/bolsig_data/e_Ar_ionization.dat',delimiter=',')

# parse data from bolsig+

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

# specify which of data and with what number of dofs to plot 
degrees_all = [3, 6, 10, 20]
degrees_all = [20, 40, 60, 80]
# degrees_all = [10, 20, 30, 40]
which_all = [0, 8, 9, 10, 11, 30]
which_all = [0, 2, 4, 6, 8, 10, 11, 35]
which_all = [0, 6, 11, 12, 35]
# which_all = [10]
# which_all = [101, 102, 103]

fig_data = plt.figure()
fig_data.set_size_inches(30, 20, forward=True)
fig_conv = plt.figure()
fig_conv.set_size_inches(30, 20, forward=True)
fig_vis_lin = plt.figure()
fig_vis_lin.set_size_inches(30, 20, forward=True)
fig_vis_log = plt.figure()
fig_vis_log.set_size_inches(30, 20, forward=True)

for which_idx,which in enumerate(which_all):

    # bolsig+ data denoted by indices < 100, indices >= 100 represent analytical functions
    if which < 100:

        ehat = 1.5*eveclist[which]/mulist[which]
        pehat = dveclist[which]

        # extend tail linearly on log scale to fill data up to ehat_ext_max
        startidx = int(np.ceil(len(ehat)*(1-frac)))

        taillin = np.poly1d(np.polyfit(ehat[startidx:], np.log(pehat[startidx:]), 1))

        del_e = (max(ehat)-min(ehat))/(len(ehat)-1)
        ehat_ext = np.concatenate((ehat, np.arange(max(ehat)+del_e, ehat_ext_max, del_e)))
        pehat_ext = np.concatenate((pehat, np.exp(taillin(np.arange(max(ehat)+del_e, ehat_ext_max, del_e)))))

        # scale data with maxwellian
        maxw = 2./np.sqrt(np.pi)*(1.5/mulist[which]/tscale)**(1.5)*np.exp(-ehat/tscale)

        datax = np.sqrt(ehat)
        datay = pehat/(maxw+1e-100)
        
        maxw_ext = 2./np.sqrt(np.pi)*(1.5/mulist[which]/tscale)**(1.5)*np.exp(-ehat_ext/tscale)

        datax_ext = np.sqrt(ehat_ext)
        datay_ext = pehat_ext/(maxw_ext+1e-100)

        # smooth bolsig+ data a bit (for small electric fields data seems very noisy)
        x = datax_ext
        ftest = interp1d(datax_ext, datay_ext, kind='cubic', bounds_error=False, fill_value=(datay_ext[0],datay_ext[-1]))

        N = 500
        xx = np.linspace(min(datax_ext),max(datax_ext),num=N)
        yy = ftest(xx)

        y2 = np.exp(savgol_filter(np.log(yy), 51, 3))
        y2 = np.exp(savgol_filter(np.log(y2), 51, 3))

        # create interpolation functions for perturbation of EDDF and full EDDF
        ftest = interp1d(xx, y2, kind='cubic', bounds_error=False, fill_value=(y2[0],y2[-1]))
        ftestexp = lambda x: ftest(x)*np.exp(-x**2)

        # create interpolation functions for scattering data
        x_elastic = np.sqrt(1.5*e_Ar_elastic[:,0]/mulist[which])
        x_excitation = np.sqrt(1.5*e_Ar_excitation[:,0]/mulist[which])
        x_ionization = np.sqrt(1.5*e_Ar_ionization[:,0]/mulist[which])

        elastic_cf = interp1d(x_elastic, e_Ar_elastic[:,1], kind='linear', bounds_error=False, fill_value=(e_Ar_elastic[0,1], e_Ar_elastic[-1,1]))
        excitation_cf = interp1d(x_excitation, e_Ar_excitation[:,1], kind='linear', bounds_error=False, fill_value=(e_Ar_excitation[0,1], e_Ar_excitation[-1,1]))
        ionization_cf = interp1d(x_ionization, e_Ar_ionization[:,1], kind='linear', bounds_error=False, fill_value=(e_Ar_ionization[0,1], e_Ar_ionization[-1,1]))

        # plot data
        plt.figure(fig_data)
        plt.subplot(len(which_all), 2, which_idx*2+1)
        plt.semilogy(ehat, pehat,':')
        plt.semilogy(ehat_ext, pehat_ext,'o')
        plt.subplot(len(which_all), 2, which_idx*2+2)
        plt.semilogy(np.sqrt(ehat), pehat/(maxw+1e-100),'-')
        plt.semilogy(datax, datay,'--')
        plt.semilogy(datax, ftest(datax),':')

    else:
        x = np.linspace(0,np.sqrt(ehat_ext_max),num=100)

        if which == 101:
            ftest = lambda x: (1.2 + np.sin(2*x))
        elif which == 102:
            ftest = lambda x: (1.2 + np.cos(2*x))
        elif which == 103:
            ftest = lambda x: (1.5 + np.sin(5*x+1))
        ftestexp = lambda x: ftest(x)*np.exp(-x**2)

        elastic_cf = lambda x: (1.2 + np.cos(2*x+1))
        excitation_cf = lambda x: (1.2 + np.cos(2*x+1))
        ionization_cf = lambda x: (1.2 + np.cos(2*x+1))

    # calculate projections of full EDDF and its perturbation onto Chebyshev basis
    ftest11 = lambda y: ftest(min(x)+.5*(max(x)-min(x))*(y+1))
    ftest11exp = lambda y: ftestexp(min(x)+.5*(max(x)-min(x))*(y+1))
    cheb_coeffs = np.polynomial.chebyshev.chebinterpolate(ftest11, max_dofs)
    cheb_coeffs_full = np.polynomial.chebyshev.chebinterpolate(ftest11exp, max_dofs)

    # calculate projections onto Maxwell and Laguerre bases
    maxwell_coeffs = np.zeros(max_dofs)
    laguerre_coeffs = np.zeros(max_dofs)

    if use_gauss_for_proj:
        # get gaussian quadratures
        [xg, wg] = maxpolygauss(gauss_q_order_proj)
        fg = ftest(xg)
        # plt.semilogy(xg,wg)
        # plt.show()

        for i in range(max_dofs):
            maxwell_coeffs[i] = np.sum(maxpolyeval(xg, i)*fg*wg)
            # laguerre_coeffs[i] = np.sum(maxpolyeval(xg, i)**2*wg)
        
    else:
        xtrapz = np.linspace(0, 20, num=trapz_num_q_points_proj)

        for i in range(max_dofs):
            maxwell_coeffs[i] = np.trapz(maxpolyeval(xtrapz, i)*ftest(xtrapz)*np.exp(-xtrapz**2)*xtrapz**2,x=xtrapz) \
                / np.sqrt(np.pi) * 4.
            # laguerre_coeffs[i] = np.trapz(lagpolyeval(xtrapz**2, i)*ftest(xtrapz)*np.exp(-xtrapz**2)*xtrapz**2,x=xtrapz) \
                # / np.sqrt(np.pi) * 4.

    # grid to check error on
    xerror = np.linspace(min(x),max(x),num=error_num_points)

    # 0 - maxwell, 1 - laguerre, 2 - chebyshev, 4 - linear, 3 - chebyshev full, 5 - linear full
    f_approx = np.zeros([6, len(x)])
    f_approx_at_err_pts = np.zeros([6, len(xerror)])
    error_g0 = np.zeros([6, max_dofs])
    error_g1 = np.zeros([6, max_dofs])
    error_g2 = np.zeros([6, max_dofs])
    error_linf = np.zeros([6, max_dofs])
    error_l2 = np.zeros([6, max_dofs])

    # auxiliary variables to store sampled values
    ftest_at_err_pts = ftest(xerror)
    exp_at_err_pts = np.exp(-xerror**2)
    x3exp_at_err_pts = exp_at_err_pts*xerror**3
    elastic_at_err_pts = elastic_cf(xerror)
    excitation_at_err_pts = excitation_cf(xerror)
    ionization_at_err_pts = ionization_cf(xerror)

    # 'exact' values for reaction rates
    elastic_scale = np.trapz(ftest_at_err_pts*elastic_at_err_pts*x3exp_at_err_pts, x=xerror)
    excitation_scale = np.trapz(ftest_at_err_pts*excitation_at_err_pts*x3exp_at_err_pts, x=xerror)
    ionization_scale = np.trapz(ftest_at_err_pts*ionization_at_err_pts*x3exp_at_err_pts, x=xerror)

    # error calculations
    for i in range(max_dofs):

        f_approx[0,:] += maxwell_coeffs[i]*maxpolyeval(x, i)
        # f_approx[1,:] += laguerre_coeffs[i]*lagpolyeval(x**2, i)
        f_approx[2,:] = np.polynomial.chebyshev.chebval(2*(x-min(x))/(max(x)-min(x))-1., cheb_coeffs[0:i+1])
        f_approx[4,:] = np.polynomial.chebyshev.chebval(2*(x-min(x))/(max(x)-min(x))-1., cheb_coeffs_full[0:i+1])
        x_lin = np.linspace(min(x), max(x), i+1)
        y_lin = ftest(x_lin)
        y_lin_full = ftestexp(x_lin)
        f_approx[3,:] = np.interp(x, x_lin, y_lin)
        f_approx[5,:]  = np.interp(x, x_lin, y_lin_full)

        f_approx_at_err_pts[0,:] += maxwell_coeffs[i]*maxpolyeval(xerror, i)
        # f_approx_at_err_pts[1,:] += laguerre_coeffs[i]*lagpolyeval(xerror**2, i)
        f_approx_at_err_pts[2,:] = np.polynomial.chebyshev.chebval(2*(xerror-min(x))/(max(x)-min(x))-1., cheb_coeffs[0:i+1])
        f_approx_at_err_pts[4,:] = np.polynomial.chebyshev.chebval(2*(xerror-min(x))/(max(x)-min(x))-1., cheb_coeffs_full[0:i+1])
        f_approx_at_err_pts[3,:] = np.interp(xerror, x_lin, y_lin)
        f_approx_at_err_pts[5,:] = np.interp(xerror, x_lin, y_lin_full)

        error_g0[0, i] = np.trapz((f_approx_at_err_pts[0, :]-ftest_at_err_pts)*elastic_at_err_pts*x3exp_at_err_pts, x=xerror)/elastic_scale
        error_g0[1, i] = np.trapz((f_approx_at_err_pts[1, :]-ftest_at_err_pts)*elastic_at_err_pts*x3exp_at_err_pts, x=xerror)/elastic_scale
        error_g0[2, i] = np.trapz((f_approx_at_err_pts[2, :]-ftest_at_err_pts)*elastic_at_err_pts*x3exp_at_err_pts, x=xerror)/elastic_scale
        error_g0[3, i] = np.trapz((f_approx_at_err_pts[3, :]-ftest_at_err_pts)*elastic_at_err_pts*x3exp_at_err_pts, x=xerror)/elastic_scale
        error_g0[4, i] = np.trapz((f_approx_at_err_pts[4, :]/exp_at_err_pts-ftest_at_err_pts)*elastic_at_err_pts*x3exp_at_err_pts, x=xerror)/elastic_scale
        error_g0[5, i] = np.trapz((f_approx_at_err_pts[5, :]/exp_at_err_pts-ftest_at_err_pts)*elastic_at_err_pts*x3exp_at_err_pts, x=xerror)/elastic_scale

        error_g1[0, i] = np.trapz((f_approx_at_err_pts[0, :]-ftest_at_err_pts)*excitation_at_err_pts*x3exp_at_err_pts, x=xerror)/excitation_scale
        error_g1[1, i] = np.trapz((f_approx_at_err_pts[1, :]-ftest_at_err_pts)*excitation_at_err_pts*x3exp_at_err_pts, x=xerror)/excitation_scale
        error_g1[2, i] = np.trapz((f_approx_at_err_pts[2, :]-ftest_at_err_pts)*excitation_at_err_pts*x3exp_at_err_pts, x=xerror)/excitation_scale
        error_g1[3, i] = np.trapz((f_approx_at_err_pts[3, :]-ftest_at_err_pts)*excitation_at_err_pts*x3exp_at_err_pts, x=xerror)/excitation_scale
        error_g1[4, i] = np.trapz((f_approx_at_err_pts[4, :]/exp_at_err_pts-ftest_at_err_pts)*excitation_at_err_pts*x3exp_at_err_pts, x=xerror)/excitation_scale
        error_g1[5, i] = np.trapz((f_approx_at_err_pts[5, :]/exp_at_err_pts-ftest_at_err_pts)*excitation_at_err_pts*x3exp_at_err_pts, x=xerror)/excitation_scale

        error_g2[0, i] = np.trapz((f_approx_at_err_pts[0, :]-ftest_at_err_pts)*ionization_at_err_pts*x3exp_at_err_pts, x=xerror)/ionization_scale
        error_g2[1, i] = np.trapz((f_approx_at_err_pts[1, :]-ftest_at_err_pts)*ionization_at_err_pts*x3exp_at_err_pts, x=xerror)/ionization_scale
        error_g2[2, i] = np.trapz((f_approx_at_err_pts[2, :]-ftest_at_err_pts)*ionization_at_err_pts*x3exp_at_err_pts, x=xerror)/ionization_scale
        error_g2[3, i] = np.trapz((f_approx_at_err_pts[3, :]-ftest_at_err_pts)*ionization_at_err_pts*x3exp_at_err_pts, x=xerror)/ionization_scale
        error_g2[4, i] = np.trapz((f_approx_at_err_pts[4, :]/exp_at_err_pts-ftest_at_err_pts)*ionization_at_err_pts*x3exp_at_err_pts, x=xerror)/ionization_scale
        error_g2[5, i] = np.trapz((f_approx_at_err_pts[5, :]/exp_at_err_pts-ftest_at_err_pts)*ionization_at_err_pts*x3exp_at_err_pts, x=xerror)/ionization_scale

        error_linf[0, i] = max(abs((f_approx_at_err_pts[0, :]-ftest_at_err_pts)*exp_at_err_pts))
        error_linf[1, i] = max(abs((f_approx_at_err_pts[1, :]-ftest_at_err_pts)*exp_at_err_pts))
        error_linf[2, i] = max(abs((f_approx_at_err_pts[2, :]-ftest_at_err_pts)*exp_at_err_pts))
        error_linf[3, i] = max(abs((f_approx_at_err_pts[3, :]-ftest_at_err_pts)*exp_at_err_pts))
        error_linf[4, i] = max(abs( f_approx_at_err_pts[4, :]-ftest_at_err_pts *exp_at_err_pts))
        error_linf[5, i] = max(abs( f_approx_at_err_pts[5, :]-ftest_at_err_pts *exp_at_err_pts))

        error_l2[0, i] = np.sqrt(sum(abs((f_approx_at_err_pts[0, :]-ftest_at_err_pts)*exp_at_err_pts)**2)/len(xerror))
        error_l2[1, i] = np.sqrt(sum(abs((f_approx_at_err_pts[1, :]-ftest_at_err_pts)*exp_at_err_pts)**2)/len(xerror))
        error_l2[2, i] = np.sqrt(sum(abs((f_approx_at_err_pts[2, :]-ftest_at_err_pts)*exp_at_err_pts)**2)/len(xerror))
        error_l2[3, i] = np.sqrt(sum(abs((f_approx_at_err_pts[3, :]-ftest_at_err_pts)*exp_at_err_pts)**2)/len(xerror))
        error_l2[4, i] = np.sqrt(sum(abs( f_approx_at_err_pts[4, :]-ftest_at_err_pts *exp_at_err_pts)**2)/len(xerror))
        error_l2[5, i] = np.sqrt(sum(abs( f_approx_at_err_pts[5, :]-ftest_at_err_pts *exp_at_err_pts)**2)/len(xerror))

        plt.figure(fig_vis_lin)
        if i in degrees_all:
            plt.subplot(len(which_all), len(degrees_all), which_idx*len(degrees_all)+degrees_all.index(i)+1)
            plt.plot(x, f_approx[0,:]*np.exp(-x**2),'-')
            plt.plot(x, f_approx[1,:]*np.exp(-x**2),'-')
            plt.plot(x, f_approx[2,:]*np.exp(-x**2),'-')
            plt.plot(x, f_approx[3,:]*np.exp(-x**2),'-')
            plt.plot(x, f_approx[4,:],'-')
            plt.plot(x, f_approx[5,:],'-')
            plt.plot(x, ftest(x)*np.exp(-x**2),'k')
            plt.plot(x, np.exp(-x**2),'k--')
            plt.xlim((0,np.sqrt(ehat_ext_max)))

            if degrees_all.index(i) == 0:
                plt.ylabel('Density')

            if which_idx == len(which_all)-1:
                plt.xlabel('Speed')

            if which_idx == 0:
                plt.title('Num. of polys: '+str(i))

            if which_idx == 0 and degrees_all.index(i) == 0:
                plt.legend(['Maxwell','Laguerre','Chebyshev','Chebyshev-Full','Linear','Linear-Full','Bolsig','Maxwellian'], loc='upper right')

            plt.grid()

        plt.figure(fig_vis_log)
        if i in degrees_all:
            plt.subplot(len(which_all), len(degrees_all), which_idx*len(degrees_all)+degrees_all.index(i)+1)
            plt.semilogy(x, abs(f_approx[0,:]*np.exp(-x**2)-ftest(x)*np.exp(-x**2)),'-')
            plt.semilogy(x, abs(f_approx[2,:]*np.exp(-x**2)-ftest(x)*np.exp(-x**2)),'-')
            plt.xlim((0,np.sqrt(ehat_ext_max)))

            if degrees_all.index(i) == 0:
                plt.ylabel('Density')

            if which_idx == len(which_all)-1:
                plt.xlabel('Speed')

            if which_idx == 0:
                plt.title('Num. of polys: '+str(i))

            if which_idx == 0 and degrees_all.index(i) == 0:
                plt.legend(['Maxwell','Laguerre','Chebyshev','Chebyshev-Full','Linear','Linear-Full','Bolsig','Maxwellian'], loc='upper right')

            plt.grid()

        # if i in degrees_all:
        #     plt.subplot(len(which_all), len(degrees_all), which_idx*len(degrees_all)+degrees_all.index(i)+1)
        #     plt.semilogy(x, f_approx[0,:]*np.exp(-x**2),'-')
        #     plt.semilogy(x, f_approx[1,:]*np.exp(-x**2),'-')
        #     plt.semilogy(x, f_approx[2,:]*np.exp(-x**2),'-')
        #     plt.semilogy(x, f_approx[3,:]*np.exp(-x**2),'-')
        #     plt.semilogy(x, f_approx[4,:],'-')
        #     plt.semilogy(x, f_approx[5,:],'-')
        #     plt.semilogy(x, ftest(x)*np.exp(-x**2),'k')
        #     plt.semilogy(x, np.exp(-x**2),'k--')
        #     plt.xlim((0,np.sqrt(ehat_ext_max)))

        #     if degrees_all.index(i) == 0:
        #         plt.ylabel('Density')

        #     if which_idx == len(which_all)-1:
        #         plt.xlabel('Speed')

        #     if which_idx == 0:
        #         plt.title('Num. of polys: '+str(i))

        #     if which_idx == 0 and degrees_all.index(i) == 0:
        #         plt.legend(['Maxwell','Laguerre','Chebyshev','Chebyshev-Full','Linear','Linear-Full','Bolsig','Maxwellian'], loc='upper right')

        #     plt.grid()

    # convergence plots
    plt.figure(fig_conv)
    ax = plt.subplot(len(which_all), 5, which_idx*5+1)
    for i in range(6):
        plt.semilogy(error_linf[i,:])
    plt.grid()
    plt.ylabel('Error')
    if which_idx == len(which_all)-1:
        plt.xlabel('Number of polynomials')
    if which_idx == 0:
        plt.title('$L_\infty$')

    ax = plt.subplot(len(which_all), 5, which_idx*5+2)
    for i in range(6):
        plt.semilogy(error_l2[i,:])
    plt.grid()
    if which_idx == len(which_all)-1:
        plt.xlabel('Number of polynomials')
    if which_idx == 0:
        plt.title('$L_2$')

    ax = plt.subplot(len(which_all), 5, which_idx*5+3)
    for i in range(6):
        plt.semilogy(abs(error_g0[i,:]))
    plt.grid()
    if which_idx == len(which_all)-1:
        plt.xlabel('Number of polynomials')
    if which_idx == 0:
        plt.title('Rate (elastic)')

    ax = plt.subplot(len(which_all), 5, which_idx*5+4)
    for i in range(6):
        plt.semilogy(abs(error_g1[i,:]))
    plt.grid()
    if which_idx == len(which_all)-1:
        plt.xlabel('Number of polynomials')
    if which_idx == 0:
        plt.title('Rate (excitation)')

    ax = plt.subplot(len(which_all), 5, which_idx*5+5)
    for i in range(6):
        plt.semilogy(abs(error_g2[i,:]))
    plt.grid()

    if which_idx == len(which_all)-1:
        plt.xlabel('Number of polynomials')
    if which_idx == 0:
        plt.legend(['Maxwell','Laguerre','Chebyshev','Linear','Chebyshev-Full', 'Linear-Full'])
        plt.title('Rate (ionization)')

plt.show()