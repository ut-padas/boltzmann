# Attempt to generalize the Columbic collision term, 
# based on Fokker-Plank and Rosenbluth potentials. 
import sympy

def ChristoffelSymbols2ndKind(metric, coords):
    """
    Computing the Christoffel symbols 2nd kind
    for a given metric tensor
    C^k_{ij}
    """
    nn         = len(coords)
    e_idx      = range(len(coords))

    inv_metric = sympy.simplify(sympy.Inverse(metric)) 

    C_kij      = sympy.Array([sympy.simplify(sympy.Rational(1,2) * sum([inv_metric[k,m] * (sympy.diff(metric[m,i], coords[j]) + sympy.diff(metric[m,j], coords[i]) - sympy.diff(metric[i,j], coords[m]))  for m in e_idx]))   for k in e_idx for j in e_idx for i in e_idx]).reshape(nn, nn, nn)
    # sympy.pprint(C_kij)

    # print("Gamma^0_ij \n", C_kij[0,:,:])
    # print("Gamma^1_ij \n", C_kij[1,:,:])
    # print("Gamma^2_ij \n", C_kij[2,:,:])

    return C_kij

def assemble_symbolic_cc_op(metric, coords, lmax):
    inv_metric = sympy.simplify(sympy.Inverse(metric)) 
    nn         = len(coords)
    e_idx      = range(len(coords))

    v          = coords[0]
    mu         = coords[1]
    
    C_kij      = ChristoffelSymbols2ndKind(metric, coords)
    L          = sympy.functions.special.polynomials.legendre
    B          = sympy.Function("RadialPoly")
    DB         = sympy.Function("RadialPolyDeriv")

    p, q, k, l, r, s = sympy.symbols('p,q,k,l,r,s')

    psi = B(v, p) * L(q, mu) * sympy.sqrt( (2 * q + 1) / (4 * sympy.pi) )
    hh  = B(v, r) * L(s, mu) * sympy.sqrt( (2 * s + 1) / (4 * sympy.pi) )
    gg  = B(v, r) * L(s, mu) * sympy.sqrt( (2 * s + 1) / (4 * sympy.pi) )
    ff  = B(v, k) * L(l, mu) * sympy.sqrt( (2 * l + 1) / (4 * sympy.pi) )

    Dij        = lambda g : sympy.Matrix([sympy.simplify(sympy.diff(g, coords[i], coords[j]) - sum([C_kij[k,i,j] * sympy.diff(g, coords[k]) for k in e_idx])) for i in e_idx for j in e_idx]).reshape(nn,nn)
    Di         = lambda g : [sympy.simplify(sympy.diff(g, coords[i])) for i in e_idx]

    
    raise_ij   = lambda qij : sympy.Matrix([sympy.simplify(sum([inv_metric[i,a] * inv_metric[j,b] * qij[a,b] for a in e_idx for b in e_idx])) for i in e_idx for j in e_idx]).reshape(nn,nn) 
    raise_i    = lambda qi  : [sympy.simplify(sum([inv_metric[i,a] * qi[a] for a in e_idx])) for i in e_idx]

    Dij_g      = Dij(gg)
    DIJ_g      = raise_ij(Dij_g)
    Dij_psi    = Dij(psi)
    
    Di_h       = Di(hh)
    DI_h       = raise_i(Di_h)
    Di_psi     = Di(psi)

    Ia = sympy.simplify(2 * sympy.pi * v**2 * sum([Di_psi[i] * DI_h[i] * ff for i in e_idx]))
    Ib = sympy.simplify(sympy.pi * v**2  * sum([Dij_psi[i,j] * DIJ_g[i,j] * ff for i in e_idx for j in e_idx]))

    Ia_nz = dict()
    Ib_nz = dict()

    for qq in range(lmax + 1):
        for ll in range(lmax + 1):
            for ss in range(lmax + 1):
                tmp   = Ia
                tmp_r = sympy.simplify(sympy.integrate(tmp.subs({q:qq, l:ll, s:ss}), (mu,-1,1)))

                if tmp_r !=0:
                    tmp_r = tmp_r.subs([(sympy.diff(B(v,p),v,2), DB(v, p, 2)), (sympy.diff(B(v,r),v,2), DB(v, r, 2)), (sympy.diff(B(v,k),v,2), DB(v, k, 2)), (sympy.diff(B(v,p),v), DB(v, p, 1)), (sympy.diff(B(v,r),v), DB(v, r, 1)), (sympy.diff(B(v,k),v), DB(v, k, 1))])
                    Ia_nz[(qq,ll,ss)] = tmp_r
                    
                
                

    for qq in range(lmax + 1):
        for ll in range(lmax + 1):
            for ss in range(lmax + 1):
                tmp   = Ib
                tmp_r = sympy.simplify(sympy.integrate(tmp.subs({q:qq, l:ll, s:ss}), (mu,-1,1)))

                if tmp_r !=0:
                    tmp_r = tmp_r.subs([(sympy.diff(B(v,p),v,2), DB(v, p, 2)), (sympy.diff(B(v,r),v,2), DB(v, r, 2)), (sympy.diff(B(v,k),v,2), DB(v, k, 2)), (sympy.diff(B(v,p),v), DB(v, p, 1)), (sympy.diff(B(v,r),v), DB(v, r, 1)), (sympy.diff(B(v,k),v), DB(v, k, 1))])
                    Ib_nz[(qq,ll,ss)] = tmp_r
                
    """
    Generate and writ to the cc_terms.py for Columbic collision assembly
    """
    filename = "cc_terms.py"
    with open(filename, 'w') as out:
        out.write("## generated code with sympy" + '\n')
        out.write("## indexed with q-test l mode, l trial mode and s-h or g function modes" + '\n')
        out.write("import numpy\n")
        out.write("import math\n")

        tp_list=list()
        for idx, expr in Ia_nz.items():
            tp_list.append(idx)
        out.write("Ia_nz=%s\n"%(tp_list))

        tp_list=list()
        for idx, expr in Ib_nz.items():
            tp_list.append(idx)
        out.write("Ib_nz=%s\n"%(tp_list))

        out.write("def Ia(RadialPoly, RadialPolyDeriv, vr, p, k, r, q, l, s):\n")

        for idx, expr in Ia_nz.items():
            out.write("\tif (q,l,s) == (%d,%d,%d):\n"%(idx[0], idx[1], idx[2]))
            pycode = sympy.pycode(expr).split("\n")[-1]
            out.write("\t\treturn %s\n"%pycode)
                    
        out.write("\treturn %d\n"%(0))
        out.write("\n\n")
        out.write("def Ib(RadialPoly, RadialPolyDeriv, vr, p, k, r, q, l, s):\n")

        for idx, expr in Ib_nz.items():
            out.write("\tif (q,l,s) == (%d,%d,%d):\n"%(idx[0], idx[1], idx[2]))
            pycode = sympy.pycode(expr).split("\n")[-1]
            out.write("\t\treturn %s\n"%pycode)
                    
        out.write("\treturn %d\n"%(0))
                
    return Ia_nz, Ib_nz

    
if __name__ == "__main__":
    v      = sympy.Symbol('vr')
    mu     = sympy.Symbol('mu')
    phi    = sympy.Symbol('phi')

    metric = sympy.Matrix([[1,0,0], [0, v**2/(1-mu**2), 0], [0, 0, v**2 * (1-mu**2)]])
    coords = [v, mu, phi]

    # metric = sympy.Matrix([[1,0,0], [0, v**2, 0], [0, 0, v**2 * sympy.sin(mu)**2]])
    # coords = [v, mu, phi]

    Ia, Ib = assemble_symbolic_cc_op(metric,coords,4)









    




