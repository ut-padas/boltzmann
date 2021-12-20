%%
kmax = 555;
krec = 5;
gamma = 2.;

a = zeros(1,kmax);
b = zeros(1,kmax);
g = zeros(1,kmax);
Y = zeros(1,kmax);

a(1) = 2./sqrt(pi);
b(1) = 0;
g(1) = -gamma/12.;

Y = 2*[0:kmax-1]+gamma;

for k = 1:krec-1
    b(k + 1) = (2*k - 1 + gamma)/2 - b(k) - a(k)^2;
    a(k + 1) = (((k + gamma/2)/2 - b(k + 1))^2 - gamma^2/16)/(a(k)*b(k + 1));
end

g(2:krec) = b(2:krec) - Y(2:krec)/12;

C0 = 1./36. - gamma^2/8.;
C1 = 23./432. - 11./48.*gamma^2 + 3./32.*gamma^4; 
C2 = 1189./2592. - 409./192.*gamma^2 + 75./64.*gamma^4 + 9./64.*gamma^6; 
C3 = 196057./20736. - 153559./3456.*gamma^2 + 7111./256.*gamma^4 + 639./128.*gamma^6 + 135./512.*gamma^8;

for k = krec:kmax-1
    Y(k + 1) = 2*k + gamma;
    g(k + 1) = C0/Y(k + 1) + C1/Y(k + 1)^3 + C2/Y(k + 1)^5 + C3/Y(k + 1)^7;
end

idx = [krec+1:kmax];

g(idx) = C0./Y(idx) + C1./Y(idx).^3 + C2./Y(idx).^5 + C3./Y(idx).^7;

plot(g,'-o')

%%
f = @(y,gn,gnm1,gnp1) ((y + 1.)/3. - gnp1 - gn).*((y - 1)/3. - gn - ...
      gnm1).*(y/12. + gn).^2 - ((y/6. - gn).^2 - 1./4.).^2;
fp = @(y,gn,gnm1,gnp1) -((5.*y)/27.) - (gnm1.*y)/18. + (gnp1.*y)/18. + ...
    (gnm1.*gnp1.*y)/6. - (7.*gnm1.*y.^2)/144. - (7.*gnp1.*y.^2)/144. + (7.*y.^3)/216.;

for i = 0:10000
 F = f(Y(2:kmax - 1), g(2:kmax - 1), g(1:kmax - 2), g(3:kmax));
 Fp = fp(Y(2:kmax - 1), g(2:kmax - 1), g(1:kmax - 2), g(3:kmax));
 g(2:kmax - 1) = g(2:kmax - 1) - F./Fp;
%  For[j = 2, j <= kmax - 1, j = j + 1,
%   If[Abs[N[Fp(j - 1), d) > 10^(-60),
%    If[Abs[N[F(j - 1)/Fp(j - 1), d) > 10^(-100),
%     g(j) = g(j) - N[F(j - 1)/Fp(j - 1), d];
%     ]
%    ]
%   ]
end

plot(g);

%%
b = g + Y/12; 
a(1:kmax - 1) = sqrt((Y(1 : kmax - 1) + 1)/3 - g(2 : kmax) - g(1 : kmax - 1));
a(kmax) = ((((kmax-1) + gamma/2)/2 - b(kmax))^2 - gamma^2/16)/(a(kmax-1)*b(kmax));

plot(b)

%%
gaussnodes = zeros(1, kmax*(kmax + 1)/2);
gaussweights2 = zeros(1, kmax*(kmax + 1)/2);

for k = 2:555
    J = diag(a(1:k)) + diag(sqrt(b(2:k)), 1) + diag(sqrt(b(2:k)), -1);
%     cn(k) = cond(J);
    [vectors, values] = eig(J);
    gaussnodes(k*(k - 1)/2 + 1 : k*(k - 1)/2 + k) = diag(values);
    gaussweights2(k*(k - 1)/2 + 1 : k*(k - 1)/2 + k) = vectors(1,:).^2;
end
%%
pnodes = zeros(kmax, kmax*(kmax + 1)/2);
pnodes(1, :) = sqrt(4./sqrt(pi));
for k = 2:555
    pnodes(k, :) = sample_poly(a,b,k,gaussnodes);
end
%%
for k = 2:kmax
    for i = 1:k
        pi = pnodes(1 : k, k*(k - 1)/2 + i);
        gaussweights(k*(k - 1)/2 + i) = 1/(pi'*pi);
    end
end

%%
k=256;
test=sort(gaussnodes(k*(k - 1)/2 + 1 : k*(k - 1)/2 + k))';

%%
testf = @(x) (sin(100*x)+cos(100*x));
k = 100;

for k=2:1:555
    x = gaussnodes(k*(k - 1)/2 + 1 : k*(k - 1)/2 + k);
    w = gaussweights(k*(k - 1)/2 + 1 : k*(k - 1)/2 + k);
    I(k) = testf(x)*(w');
end
semilogy(abs(I+2.0024036067351600382976893458605e-6))
grid on
ylabel('Integration error');
xlabel('Quadrature order');

%%
pnorm(1) = 1;
for k=2:500
    q = k;
    x = gaussnodes(q*(q - 1)/2 + 1 : q*(q - 1)/2 + q);
    f = sample_poly(a,b,k,x);
    f2 = f.^2;
    numinfs(k) = sum(isinf(f2));
    f2(isinf(f2)) = 0;
    w = gaussweights(q*(q - 1)/2 + 1 : q*(q - 1)/2 + q);
%     w(w < 1.e-30) = 0;
    pnorm(k) = f2*(w');
    xmax(k) = max(x);
end

pnorm2(1) = sqrt(pi)/4.;
for k=2:500
    pnorm2(k) = b(k)*pnorm(k-1);
end

semilogy(abs(pnorm-1),'-o')
grid on
ylabel('Error in polynomial norm');
xlabel('Polynomial order');
% hold on 
% semilogy(pnorm2)
% hold off

plot(xmax);
hold on
plot(numinfs);
hold off;


%%
x = linspace(0.01,40,40000);
f = abs(sample_poly(a,b,500,x));
semilogy(x,f)

%%
writematrix([a', b'],'maxpoly_upto555_recursive.dat');
writematrix([gaussnodes', gaussweights'],'maxpoly_upto555.dat');

%%
save('maxpoly_matlab_data');

%%
function S = sample_poly(a,b,q,x)
    Bn = zeros(q + 2, length(x));
    for k = q:-1:1
        if k == q
            Bn(k, :) = 1 + (x - a(k)).*Bn(k + 1, :)/sqrt(b(k+1)) - sqrt(b(k + 1)/b(k+2)).*Bn(k + 2, :);
        else
            Bn(k, :) = (x - a(k)).*Bn(k + 1, :)/sqrt(b(k+1)) - sqrt(b(k + 1)/b(k+2)).*Bn(k + 2, :);
        end
    end
    S = Bn(2, :).*(x-2./sqrt(pi))/sqrt(sqrt(pi)*(1.5-4./pi)/4) - sqrt(b(2)/b(3))*Bn(3, :)/sqrt(sqrt(pi)/4.);
end
