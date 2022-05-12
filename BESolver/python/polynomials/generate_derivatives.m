clear;

a = importdata('maxpoly_alpha_300_65.dat');
b = importdata('maxpoly_beta_300_65.dat');
nodes = importdata('maxpoly_nodes_300_65.dat');
weights = importdata('maxpoly_weights_300_65.dat');

G_all = 0:65;
kmax = 300;

d_all = zeros(length(G_all), kmax*(kmax - 1)/2);

for G_idx = 1:length(G_all)
    
    G = G_all(G_idx);
    
    if G == 0
        
        for j = 2:kmax
            for i = 1:j-1
                idx = (j-2)*(j-1)/2 + i;
                d_all(G_idx, idx) = -sample_poly(G, [a(G+1,:),0,0], [b(G+1,:),0,0], i, 0) ...
                    .*sample_poly(G, [a(G+1,:),0,0], [b(G+1,:),0,0], j, 0);
            end
            d_all(G_idx, idx) = d_all(G_idx, idx) + 2*sqrt(b(G+1, j));
        end
        
    else
        
        for j = 2:kmax
            for i = 1:j-1
                idx = (j-2)*(j-1)/2 + i;
                k_quad = ceil((i+j+1)/2);
                quad_idx = k_quad*(k_quad - 1)/2 + 1 : k_quad*(k_quad - 1)/2 + k_quad;
                d_all(G_idx, idx) = -G*sum(sample_poly(G, [a(G+1,:),0,0], [b(G+1,:),0,0], i, nodes(G, quad_idx)) ...
                    .*sample_poly(G, [a(G+1,:),0,0], [b(G+1,:),0,0], j, nodes(G, quad_idx)) ...
                    .*weights(G, quad_idx));
            end
            d_all(G_idx, idx) = d_all(G_idx, idx) + 2*sqrt(b(G+1, j));
        end
        
    end
    
    G
end


writematrix(d_all,['maxpoly_deriv_',num2str(kmax),'_',num2str(G_all(end)),'.dat']);