clear;

ev_el = logspace(-5,4,200);
ev_ex = logspace(log10(11.55),4, 200);
ev_io = logspace(log10(15.7),4, 200);

loglog(ev_el, cs_el(ev_el), '.');
hold on
% loglog(ev, cs_ex(ev), '-');
loglog(ev_ex, cs_ex2(ev_ex), '.');
loglog(ev_io, cs_io(ev_io), '.');
hold off

% writematrix([ev_el', cs_el(ev_el)'], 'analytic_elastic.dat','Delimiter','tab');
% writematrix([ev_ex', cs_ex(ev_ex)'], 'analytic_excitation.dat','Delimiter','tab');
% writematrix([ev_io', cs_io(ev_io)'], 'analytic_ionization.dat','Delimiter','tab');

% excitation
function y = cs_ex2(ev)

a  = -4.06265154e-21;
b=  6.46808245e-22;
c  = -3.20434420e-23;
d  = 6.39873618e-25;
e  = -4.37947887e-27;
f  = -1.30972221e-23;
g  =  2.15683845e-19;
mixing = 1./(1+exp(-(ev-32)));
y = (a +  b * ev + c * ev.^2 + d * ev.^3 + e * ev.^4).*(1-mixing) + mixing.*(f + g./ev.^2);
y(ev<=11.55) = 0;
% y(ev>35.00) = f + g * (1./ev(ev>35.00).^2);
% y(ev>=200)  = 0;

end

function y = cs_ex(ev)

a  = -4.06265154e-21;
b=  6.46808245e-22;
c  = -3.20434420e-23;
d  = 6.39873618e-25;
e  = -4.37947887e-27;
f  = -1.30972221e-23;
g  =  2.15683845e-19;
y = a +  b * ev + c * ev.^2 + d * ev.^3 + e * ev.^4;
y(ev<=11.55) = 0;
y(ev>35.00) = f + g * (1./ev(ev>35.00).^2);
% y(ev>=200)  = 0;

end

% ionization
function y = cs_io(ev)

a = 2.84284159e-22;
b = 1.02812034e-17;
c =-1.40391999e-15;
d = 9.97783291e-14;
e =-3.82647294e-12;
f =-5.70400826e+01;

x=ev-f;
y=a + b* (1./x.^1) + c * (1./x.^2) + d * (1./x.^3) + e * (1./x.^4);

y(ev<=15.7) = 0;
% y(ev>1e3) = 0;

end

% momentum transfer
function y = cs_el(ev)

ev =     ev+1e-8;
a0 =    0.008787;
b0 =     0.07243;
c  =    0.007048;
d  =      0.9737;
a1 =        3.27;
b1 =       3.679;
x0 =      0.2347;
x1 =       11.71;
y = 9.900000e-20*(a1+b1*(log(ev/x1)).^2)./(1+b1*(log(ev/x1)).^2).*(a0+b0*(log(ev/x0)).^2)./(1+b0*(log(ev/x0)).^2)./(1+c*ev.^d);

end
