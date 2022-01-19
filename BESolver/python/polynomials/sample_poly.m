function S = sample_poly(G,a,b,q,x)     
    if q == 1
        S = sqrt(2./gamma((G+1.)/2.))*ones(size(x));
        return;
    end
    
    Bn = zeros(q + 2, length(x));
    for k = q:-1:1
        if k == q
            Bn(k, :) = 1 + (x - a(k)).*Bn(k + 1, :)/sqrt(b(k+1)) - sqrt(b(k + 1)/b(k+2)).*Bn(k + 2, :);
        else
            Bn(k, :) = (x - a(k)).*Bn(k + 1, :)/sqrt(b(k+1)) - sqrt(b(k + 1)/b(k+2)).*Bn(k + 2, :);
        end
    end
    g1 = gamma((1.+G)/2.);
    g2 = gamma((2.+G)/2.);
    g3 = gamma((3.+G)/2.);
    S = Bn(2, :).*(x-g2/g1)*sqrt(2)/sqrt(g3-g2^2/g1) - sqrt(b(2)/b(3))*Bn(3, :)*sqrt(2./g1);
end