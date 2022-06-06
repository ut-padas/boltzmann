clear;
kmax = 300;
krec = 5;
G_all = 0.5:1:65.5;

gaussnodes_all = zeros(length(G_all), kmax*(kmax + 1)/2);
gaussweights_all = zeros(length(G_all), kmax*(kmax + 1)/2);
a_all = zeros(length(G_all),kmax);
b_all = zeros(length(G_all),kmax);

for G_idx = 1:length(G_all)
    
    G = G_all(G_idx);

    a = zeros(1,kmax);
    b = zeros(1,kmax);
    g = zeros(1,kmax);
    Y = zeros(1,kmax);

    % compute initial and boundary values for g
    a(1) = gamma(G/2.+1.)/gamma((G+1.)/2.);
    b(1) = 0;
    g(1) = -G/12.;

    Y = 2*[0:kmax-1]+G;

    for k = 1:krec-1
        b(k + 1) = (2*k - 1 + G)/2 - b(k) - a(k)^2;
        a(k + 1) = (((k + G/2)/2 - b(k + 1))^2 - G^2/16)/(a(k)*b(k + 1));
    end

    g(2:krec) = b(2:krec) - Y(2:krec)/12;

    gcf = @(gamma, x) x.*(0.0277777777777777778-0.125000000000000000.*gamma.^2) ...
    + x.^3*(0.0532407407407407407-0.229166666666666667.*gamma.^2+0.0937500000000000000.*gamma.^4) ...
    + x.^5*(0.458719135802469136-2.13020833333333333.*gamma.^2+1.17187500000000000.*gamma.^4-0.140625000000000000.*gamma.^6) ...
    + x.^7*(9.45490933641975309-44.4325810185185185.*gamma.^2+27.7773437500000000.*gamma.^4-4.99218750000000000.*gamma.^6+0.263671875000000000.*gamma.^8) ...
    + x.^9*(346.316966199417010-1645.89297598379630.*gamma.^2+1100.04839409722222.*gamma.^4-235.115234375000000.*gamma.^6+19.4326171875000000.*gamma.^8-0.553710937500000000.*gamma.^10) ...
    + x.^11*(19884.1510530531979-95144.0103624131944.*gamma.^2+66221.7622341579861.*gamma.^4-15573.3115234375000.*gamma.^6+1572.62475585937500.*gamma.^8-71.7055664062500000.*gamma.^10+1.24584960937500000.*gamma.^12) ...
    + x.^13*(1.64621757220597595e6-7.91395398184568303e6.*gamma.^2+5.66003581377947772e6.*gamma.^4-1.41397036500379774e6.*gamma.^6+160087.115112304688.*gamma.^8-9067.52746582031250.*gamma.^10+255.409057617187500.*gamma.^12-2.93664550781250000.*gamma.^14) ...
    + x.^15*(1.85651211338693874e8-8.95574121072691141e8.*gamma.^2+6.53129084365066717e8.*gamma.^4-1.70115765115548593e8.*gamma.^6+2.07473748679250081e7.*gamma.^8-1.33448705694580078e6.*gamma.^10+47203.2979431152344.*gamma.^12-887.237731933593750.*gamma.^14+7.15807342529296875.*gamma.^16) ...
    + x.^17*(2.73598821492776861e10-1.32331698991527211e11.*gamma.^2+9.79365728700239944e10.*gamma.^4-2.63021084218291451e10.*gamma.^6+3.38075704622225698e9.*gamma.^8-2.36621961657508850e8.*gamma.^10+9.59040392912292480e6.*gamma.^12-228060.961715698242.*gamma.^14+3025.02018356323242.*gamma.^16-17.8951835632324219.*gamma.^18) ...
    + x.^19*(5.10487271208728694e12-2.47423657022406229e13.*gamma.^2+1.85229070068214495e13.*gamma.^4-5.09268772811435107e12.*gamma.^6+6.80754529586100620e11.*gamma.^8-5.06268754546066704e10.*gamma.^10+2.24961730464123344e9.*gamma.^12-6.17003785117721558e7.*gamma.^14+1.04096026873683929e6.*gamma.^16-10165.5780200958252.*gamma.^18+45.6327180862426758.*gamma.^20) ...
    + x.^21*(1.17618623138170685e15-5.71041742352206081e15.*gamma.^2+4.31460669884582915e15.*gamma.^4-1.20850000496195012e15.*gamma.^6+1.66535789449937084e14.*gamma.^8-1.29668268721289789e13.*gamma.^10+6.16092036714671559e11.*gamma.^12-1.86312720408514395e10.*gamma.^14+3.64319414863524914e8.*gamma.^16-4.54334504931879044e6.*gamma.^18+33769.6286072731018.*gamma.^20-118.230224132537842.*gamma.^22) ...
    + x.^23*(3.27966707995791839e17-1.59452098244210518e18.*gamma.^2+1.21392613611474449e18.*gamma.^4-3.45180170620637156e17.*gamma.^6+4.87395070821916333e16.*gamma.^8-3.93423281410424010e15.*gamma.^10+1.96754846507328574e14.*gamma.^12-6.39370917844195997e12.*gamma.^14+1.38460871313813934e11.*gamma.^16-2.00885736796930146e9.*gamma.^18+1.91256393415986300e7.*gamma.^20-111130.680784463882.*gamma.^22+310.354338347911835.*gamma.^24) ...
    + x.^25*(1.08847524681358190e20-5.29823521556121815e20.*gamma.^2+4.05921625072132805e20.*gamma.^4-1.16874164781901661e20.*gamma.^6+1.68349616846813380e19.*gamma.^8-1.39903145932460868e18.*gamma.^10+7.28666304472100274e16.*gamma.^12-2.50309205295153668e15.*gamma.^14+5.84779787284428493e13.*gamma.^16-9.42894308096286305e11.*gamma.^18+1.04747207459579979e10.*gamma.^20-7.81468932892654538e7.*gamma.^22+362864.816630333662.*gamma.^24-823.632667154073715.*gamma.^26) ...
    + x.^27*(4.24010150699304248e22-2.06597906710490712e23.*gamma.^2+1.59136234675286061e23.*gamma.^4-4.63031797116153297e22.*gamma.^6+6.78152447408119203e21.*gamma.^8-5.77255134031796314e20.*gamma.^10+3.10754512157954449e19.*gamma.^12-1.11591991763163184e18.*gamma.^14+2.76561553142063382e16.*gamma.^16-4.82600114781158242e14.*gamma.^18+5.97491946367792104e12.*gamma.^20-5.21364239027863858e10.*gamma.^22+3.11424749997607134e8.*gamma.^24-1.17703067777980864e6.*gamma.^26+2206.15892987698317.*gamma.^28) ...
    + x.^29*(1.91574831983850488e25-9.34254801066956108e25.*gamma.^2+7.22956525974667538e25.*gamma.^4-2.12251549947518593e25.*gamma.^6+3.15272549577564008e24.*gamma.^8-2.73828868395803969e23.*gamma.^10+1.51511188938257848e22.*gamma.^12-5.64211875373700471e20.*gamma.^14+1.46630423114731240e19.*gamma.^16-2.72219136935597804e17.*gamma.^18+3.65697547168214126e15.*gamma.^20-3.56441729757622741e13.*gamma.^22+2.49503081741580929e11.*gamma.^24-1.21496119327824255e9.*gamma.^26+3.79644484018444642e6.*gamma.^28-5956.62911066785455.*gamma.^30) ...
    + x.^31*(9.93699829726174722e27-4.84965721864988063e28.*gamma.^2+3.76788777000678516e28.*gamma.^4-1.11481344950471918e28.*gamma.^6+1.67605491564258714e27.*gamma.^8-1.48093754638504976e26.*gamma.^10+8.38615959749500693e24.*gamma.^12-3.21916423099396373e23.*gamma.^14+8.69995122610855760e21.*gamma.^16-1.69813088962048208e20.*gamma.^18+2.43287673911044611e18.*gamma.^20-2.57860994081972013e16.*gamma.^22+2.01999461929144157e14.*gamma.^24-1.15452023489033221e12.*gamma.^26+4.65397179500334717e9.*gamma.^28-1.21855032490348946e7.*gamma.^30+16194.5853946282296.*gamma.^32);

    idx = [krec+1:kmax];

    g(idx) = gcf(G,1./Y(idx));

    g(1:100) = linspace(g(1), g(100), 100);
    g_old = g;

    % iterate until g is converged
    f = @(y,gn,gnm1,gnp1) ((y + 1.)/3. - gnp1 - gn).*((y - 1)/3. - gn - ...
          gnm1).*(y/12. + gn).^2 - ((y/6. - gn).^2 - G^2/16.).^2;
    % fp = @(y,gn,gnm1,gnp1) -((5.*y)/27.) - (gnm1.*y)/18. + (gnp1.*y)/18. + ...
    %     (gnm1.*gnp1.*y)/6. - (7.*gnm1.*y.^2)/144. - (7.*gnp1.*y.^2)/144. + (7.*y.^3)/216.;
    fp = @(y,gn,gnm1,gnp1) ...
        -(y/54.) - (G^2 *y)/24. - (gnm1.*y)/18. + (gnp1.*y)/18. + (gnm1.*gnp1.*y)/6. - ...
        (7.*gnm1.*y.^2)/144. - (7.*gnp1.*y.^2)/144. + (7.*y.^3)/216.;

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

    % semilogy(-g,'-o')
    % hold on 
    % semilogy(-g_old, '-*');
    % hold off

    % semilogy(abs(g-g_old),'.');

    % compute a and b from g
    b = g + Y/12; 
    a(1:kmax - 1) = sqrt((Y(1 : kmax - 1) + 1)/3 - g(2 : kmax) - g(1 : kmax - 1));
    a(kmax) = ((((kmax-1) + G/2)/2 - b(kmax))^2 - G^2/16)/(a(kmax-1)*b(kmax));


    % compute associated gauss nodes
    gaussnodes = zeros(1, kmax*(kmax + 1)/2);
    gaussweights = zeros(1, kmax*(kmax + 1)/2);

    for k = 2:300
        J = diag(a(1:k)) + diag(sqrt(b(2:k)), 1) + diag(sqrt(b(2:k)), -1);
        values = eig(J);
        gaussnodes(k*(k - 1)/2 + 1 : k*(k - 1)/2 + k) = values;
    end

    % compute gauss weights
    pnodes = zeros(kmax, kmax*(kmax + 1)/2);
    for k = 1:kmax
        idx = k*(k - 1)/2+1;
        pnodes(k, idx:end) = sample_poly(G,[a,0,0],[b,0,0],k,gaussnodes(idx:end));
    end
    
    for k = 2:kmax
        for i = 1:k
            pi = pnodes(1 : k, k*(k - 1)/2 + i);
            gaussweights(k*(k - 1)/2 + i) = 1/(pi'*pi);
        end
    end
    
    % store results
    a_all(G_idx, :) = a;
    b_all(G_idx, :) = b;
    gaussnodes_all(G_idx, :) = gaussnodes;
    gaussweights_all(G_idx, :) = gaussweights;
    
    G
    
end

% save results
writematrix(a_all,['maxpoly_frac_alpha_',num2str(kmax),'_',num2str(G_all(end)),'.dat']);
writematrix(b_all,['maxpoly_frac_beta_',num2str(kmax),'_',num2str(G_all(end)),'.dat']);
writematrix(gaussnodes_all,['maxpoly_frac_nodes_',num2str(kmax),'_',num2str(G_all(end)),'.dat']);
writematrix(gaussweights_all,['maxpoly_frac_weights_',num2str(kmax),'_',num2str(G_all(end)),'.dat']);


% %%
% k=256;
% test=sort(gaussnodes(k*(k - 1)/2 + 1 : k*(k - 1)/2 + k))';
% 
% %%
% testf = @(x) (sin(100*x)+cos(100*x));
% k = 100;
% 
% testf = @(x) (sin(10*x)+cos(10*x)).*x.^2;
% 
% for k=2:1:300
%     x = gaussnodes(k*(k - 1)/2 + 1 : k*(k - 1)/2 + k);
%     w = gaussweights(k*(k - 1)/2 + 1 : k*(k - 1)/2 + k);
%     I(k) = testf(x)*(w');
% end
% % semilogy(abs(I+2.0024036067351600382976893458605e-6))
% semilogy(abs(I-0.0003414959490264059)) % G=4
% % semilogy(abs(I+0.002284823696325278)) % G=2
% grid on
% ylabel('Integration error');
% xlabel('Quadrature order');
% 
% %%
% pnorm(1) = 1;
% for k=2:300
%     q = k;
%     x = gaussnodes(q*(q - 1)/2 + 1 : q*(q - 1)/2 + q);
%     f = sample_poly(G,[a,0,0],[b,0,0],k,x);
%     f2 = f.^2;
%     numinfs(k) = sum(isinf(f2));
%     f2(isinf(f2)) = 0;
%     w = gaussweights(q*(q - 1)/2 + 1 : q*(q - 1)/2 + q);
% %     w(w < 1.e-30) = 0;
%     pnorm(k) = f2*(w');
%     xmax(k) = max(x);
% end
% 
% % pnorm2(1) = sqrt(pi)/4.;
% % for k=2:500
% %     pnorm2(k) = b(k)*pnorm(k-1);
% % end
% 
% semilogy(abs(pnorm-1),'-o')
% grid on
% ylabel('Error in polynomial norm');
% xlabel('Polynomial order');
% % hold on 
% % semilogy(pnorm2)
% % hold off
% 
% % plot(xmax);
% % hold on
% % plot(numinfs);
% % hold off;
% 
% 
% %%
% x = linspace(0.01,40,40000);
% f = abs(sample_poly(G,a,b,298,x));
% semilogy(x,f)
% 
% %%
% writematrix([a', b'],'maxpoly10_upto555_recursive.dat');
% % writematrix([gaussnodes', gaussweights'],'maxpoly_upto555.dat');
% 
% %%
% save('maxpoly_matlab_data');

%%
% function S = sample_poly(G,a,b,q,x)     
%     if q == 1
%         S = sqrt(2./gamma((G+1.)/2.))*ones(size(x));
%         return;
%     end
%     
%     Bn = zeros(q + 2, length(x));
%     for k = q:-1:1
%         if k == q
%             Bn(k, :) = 1 + (x - a(k)).*Bn(k + 1, :)/sqrt(b(k+1)) - sqrt(b(k + 1)/b(k+2)).*Bn(k + 2, :);
%         else
%             Bn(k, :) = (x - a(k)).*Bn(k + 1, :)/sqrt(b(k+1)) - sqrt(b(k + 1)/b(k+2)).*Bn(k + 2, :);
%         end
%     end
%     g1 = gamma((1.+G)/2.);
%     g2 = gamma((2.+G)/2.);
%     g3 = gamma((3.+G)/2.);
%     S = Bn(2, :).*(x-g2/g1)*sqrt(2)/sqrt(g3-g2^2/g1) - sqrt(b(2)/b(3))*Bn(3, :)*sqrt(2./g1);
% end
