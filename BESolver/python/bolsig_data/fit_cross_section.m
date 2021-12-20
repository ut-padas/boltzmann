clear;

data = importdata('e_Ar_elastic.dat');
% model = fit(data(:,1), data(:,2)*1e19, ...
%     fittype('(g+h*(log((x+1.e-7)/11)).^2)./(1+h*(log((x+1.e-7)/11)).^2).*(a+b*(log((x+1.e-7)/0.225)).^2)./(1+b*(log((x+1.e-7)/0.225)).^2)./(1+c*x.^d)'),...
%     'StartPoint', [0.01, 0.08, 0.01, 0.95, 3, 2])

% model = fit(log(data(:,1)+1.e-10), log(data(:,2)*1e19), ...
%     fittype('log((g+h*(x-log(x1)).^2)./(1+h*(x-log(x1)).^2).*(a+b*(x-log(x0)).^2)./(1+b*(x-log(x0)).^2)./(1+c*exp(d*x)))'),...
%     'StartPoint', [0.01, 0.08, 0.01, 0.95, 3, 2, 0.225, 11])
% 
% plot(model, log(data(:,1)), log(data(:,2)*1e19))

% set(gca, 'XScale', 'log')
% set(gca, 'YScale', 'log')

a0 = 0.1;
b0 = 0.1;
c = 0.009;
d = 0.95;
x0 = 0.225;
a1 = 2;
b1 = 2.5;
x1 = 11;

a0 =    0.008787;
b0 =     0.07243;
c  =    0.007048;
d  =      0.9737;
a1 =        3.27;
b1 =       3.679;
x0 =      0.2347;
x1 =       11.71;

f = @(x) (a1+b1*(log(x/x1)).^2)./(1+b1*(log(x/x1)).^2).*(a0+b0*(log(x/x0)).^2)./(1+b0*(log(x/x0)).^2)./(1+c*x.^d);
x = logspace(-4, 5, 100000);
plot(data(:,1), data(:,2)*1e19, 'o', x, f(x),'-')

% plot(model,data(:,1), data(:,2)*1e19)
% hold on
% plot(data(:,1), data(:,2)*1e19,'.')
% hold off
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')

grid on