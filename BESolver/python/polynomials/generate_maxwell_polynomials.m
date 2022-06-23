clear;
digits(16)
G_all = 0.5:1:20.5;
ktarget = 300;
kl = 1000;
ku = 40000;
krec = 50;

a_all = zeros(length(G_all),ktarget);
b_all = zeros(length(G_all),ktarget);

for j = 1:length(G_all)
    
    G = G_all(j);
    
    kmax = kl + round((ku-kl)*(j-1)/length(G_all));
    
    a = (zeros(1,kmax));
    b = (zeros(1,kmax));
    g = (zeros(1,kmax));
    Y = (zeros(1,kmax));
    
    a(1) = gamma(G/2.+1.)/gamma((G+1.)/2.);
    b(1) = 0;
    g(1) = -G/12.;
    
    Y = 2*[0:kmax-1]+G;
    
    for k = 1:krec-1
        b(k + 1) = (2*k - 1 + G)/2 - b(k) - a(k)^2;
        a(k + 1) = (((k + G/2)/2 - b(k + 1))^2 - G^2/16)/(a(k)*b(k + 1));
    end
    
    g(2:krec) = b(2:krec) - Y(2:krec)/12;
    
    C0 = (1./36. - G^2/8.);
    C1 = (23./432. - 11./48.*G^2 + 3./32.*G^4);
    C2 = (1189./2592. - 409./192.*G^2 + 75./64.*G^4 + 9./64.*G^6);
    C3 = (196057./20736. - 153559./3456.*G^2 + 7111./256.*G^4 + 639./128.*G^6 + 135./512.*G^8);
    
    for k = krec:kmax-1
        Y(k + 1) = 2*k + G;
%         g(k + 1) = C0/Y(k + 1) + C1/Y(k + 1)^3 + C2/Y(k + 1)^5 + C3/Y(k + 1)^7;
        g(k + 1) = (C0 + (C1 + (C2 + C3/Y(k + 1)^2)/Y(k + 1)^2)/Y(k + 1)^2)/Y(k + 1);
    end
    
    g(2:krec) = g(1) + (g(krec+1)-g(1))/(krec)*[1:krec-1];
    
    idx = [krec+1:kmax];
    
    g(idx) = C0./Y(idx) + C1./Y(idx).^3 + C2./Y(idx).^5 + C3./Y(idx).^7;

    f = @(y,gn,gnm1,gnp1) ...
        ((y + 1.)/3. - gnp1 - gn).*((y - 1)/3. - gn - ...
        gnm1).*(y/12. + gn).^2 - ((y/6. - gn).^2 - G^2/16.).^2;
    
%     fp = @(y,gn,gnm1,gnp1) ...
%         -(y/54.) - (G^2 *y)/24. - (gnm1.*y)/18. + (gnp1.*y)/18. + (gnm1.*gnp1.*y)/6. - ...
%         (7.*gnm1.*y.^2)/144. - (7.*gnp1.*y.^2)/144. + (7.*y.^3)/216.;

    fp = @(y,gn,gnm1,gnp1) ...
        1./432. * (216.* gn.^2 .*(6.* gnm1 + 6.* gnp1 + y) + ...
        6.* gn.* (-16. + 18.* G^2 + 48.* gnp1 + 24.* gnm1.* (-2. + 6.* gnp1 - y) - ...
        24.* gnp1.* y - 23.* y.^2) + ...
        y.* (-8. - 18.* G^2 + 24.* gnp1 + 3.* gnm1.* (-8. + 24.* gnp1 - 7.* y) - ...
        21.* gnp1.* y + 14.* y.^2));
    
    g0 = g;
    
    for i = 0:300
        F = f(Y(2:kmax - 1), g(2:kmax - 1), g(1:kmax - 2), g(3:kmax));
        Fp = fp(Y(2:kmax - 1), g(2:kmax - 1), g(1:kmax - 2), g(3:kmax));
        delta = F./Fp;
%         g(2:kmax - 1) = g(2:kmax - 1) - delta;
        g(2:kmax - 1) = (g(2:kmax - 1).*Fp - F)./Fp;
%         semilogy(abs(delta))
%         drawnow
        if max(abs(delta)) < 1.e-15
            break;
        end
        delta(290)
    end
    
    semilogy(abs(g0-g)./abs(g));
    hold on 
    plot(abs(delta)./abs(g(2:kmax-1)));
    hold off
    drawnow
    
    error(j) = mean(abs(g0(end-20:end)-g(end-20:end)));
    res(j) = norm(delta);

    b = g + Y/12;
    a(1:kmax - 1) = sqrt((Y(1 : kmax - 1) + 1)/3 - g(2 : kmax) - g(1 : kmax - 1));
    a(kmax) = ((((kmax-1) + G/2)/2 - b(kmax))^2 - G^2/16)/(a(kmax-1)*b(kmax));
    
    a_all(j,:) = a(1:ktarget);
    b_all(j,:) = b(1:ktarget);

%     writematrix([a', b'],['maxpoly', num2str(G), '_upto',num2str(kmax),'_recursive.dat']);
end

semilogy(error)
hold on
semilogy(res)
hold off

writematrix(a_all,['maxpoly_frac_alpha_upto',num2str(ktarget),'.dat']);
writematrix(b_all,['maxpoly_frac_beta_upto', num2str(ktarget),'.dat']);
