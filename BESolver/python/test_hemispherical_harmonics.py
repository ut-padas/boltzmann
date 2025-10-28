import numpy as np
import basis
from scipy.special import sph_harm
import matplotlib.pyplot as plt
import spec_spherical as sp


def create_xlbspline_spec(spline_order, k_domain, Nr, sph_harm_lm, sig_pts=None):
    splines      = basis.BSpline(k_domain,spline_order,Nr+1, sig_pts=sig_pts, knots_vec=None, dg_splines=0)
    spec         = sp.SpectralExpansionSpherical(Nr,splines,sph_harm_lm)
    return spec


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--Nr"   , type=int, default=8,  help="B-splines in vr direction")
    parser.add_argument("--Nvt"  , type=int, default=64, help="Number of discrete ordinates in vt direction")
    parser.add_argument("--Nvp"  , type=int, default=64, help="Number of discrete ordinates in vp direction")
    parser.add_argument("--l_max", type=int, default=2,  help="Maximum spherical harmonic degree")
    args = parser.parse_args()

    Nr          = args.Nr
    Nvt         = args.Nvt
    l_max       = args.l_max
    Nvp         = args.Nvp
    #sph_harm_lm = [(l,m) for l in range(l_max+1) for m in range(-l, l+1)]
    sph_harm_lm = [(l,0) for l in range(l_max+1)]

    spec_sp             = create_xlbspline_spec(3, (0, 5), Nr, sph_harm_lm, sig_pts=None)
    xp_vt, xp_vt_qw     = spec_sp.gl_vt(Nvt, hspace_split=True)
    xp_vp, xp_vp_qw     = spec_sp.gl_vp(Nvp)

    
    mm                  = np.meshgrid(xp_vt, xp_vp, indexing='ij')
    num_sh              = len(sph_harm_lm)
    
    Vq                  = spec_sp.Vq_hsph(mm[0], mm[1])
    mm_sph              = np.einsum("pij,qij,i, j->pq", Vq, Vq, xp_vt_qw, xp_vp_qw)

    plt_cnt = 1
    plt.figure(figsize=(4 * len(sph_harm_lm), 8), dpi=100)
    for didx, d in enumerate(["+", "-"]):
        for lm_i, lm in enumerate(sph_harm_lm):
            plt.subplot(2, len(sph_harm_lm), plt_cnt)
            plt.imshow(Vq[didx * num_sh + lm_i], extent=(0, 2 * np.pi, 0, np.pi), origin='lower')
            plt.colorbar()
            plt.title(r"$Y_{%d%d}^{%s}$"%(lm[0], lm[1], d))
            plt.xlabel(r"$v_{\phi}$")
            plt.ylabel(r"$v_{\theta}$")
        
            plt_cnt += 1

    plt.tight_layout()
    plt.show()
    plt.close()

    print("||M_{++} - I||/ || I|| = %.8E" % (np.linalg.norm(mm_sph[0:num_sh, 0:num_sh] - np.eye(num_sh)) / np.linalg.norm(np.eye(num_sh))))
    print("||M_{--} - I||/ || I|| = %.8E" % (np.linalg.norm(mm_sph[num_sh:, num_sh:]   - np.eye(num_sh)) / np.linalg.norm(np.eye(num_sh))))
    print("||M_{+-}||             = %.8E" % (np.linalg.norm(mm_sph[0:num_sh, num_sh:])))
    print("||M_{-+}||             = %.8E" % (np.linalg.norm(mm_sph[num_sh:, 0:num_sh])))

    #Po, Ps = spec_sp.sph_ords_projections_ops(xp_vt, xp_vt_qw, mode="sph")
    Po, Ps = spec_sp.sph_ords_projections_ops(xp_vt, xp_vt_qw, mode="hsph")

    # fvt = np.zeros_like(xp_vt)
    # idx = xp_vt<=0.5 * np.pi
    # fvt[idx] = 1+ np.cos(xp_vt[idx])

    # plt.plot(xp_vt, fvt, '-')
    # plt.plot(xp_vt, Po @ Ps @ fvt, '--')
    
    
    fvt = np.zeros_like(xp_vt)
    idx = xp_vt>=0.5 * np.pi
    fvt[idx] = 1+ np.cos(xp_vt[idx])
    plt.plot(xp_vt, fvt, '-')
    plt.plot(xp_vt, Po @ Ps @ fvt, '--')
    plt.show()
    
    #plt.savefig("hemi_sph_harmonics.png")

    