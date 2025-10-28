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
    parser.add_argument("--Nvt"  , type=int, default=4, help="Number of discrete ordinates in vt direction")
    parser.add_argument("--Nvp"  , type=int, default=4, help="Number of discrete ordinates in vp direction")
    parser.add_argument("--Nvts" , type=int, default=4, help="Number of discrete ordinates in vt direction")
    parser.add_argument("--Nvps" , type=int, default=4, help="Number of discrete ordinates in vp direction")
    parser.add_argument("--l_max", type=int, default=1,  help="Maximum spherical harmonic degree")
    args = parser.parse_args()

    Nvt      = args.Nvt
    Nvts     = args.Nvts
    Nvp      = args.Nvp
    Nvps     = args.Nvps
    l_max    = args.l_max
    Nr       = args.Nr
    num_sh   = args.l_max + 1

    sph_harm_lm = [(l,0) for l in range(l_max+1)]
    spec_sp     = create_xlbspline_spec(3, (0, 5), Nr, sph_harm_lm, sig_pts=None)




    Wqs       = np.zeros((2 * num_sh, 2 * num_sh))
    vt, vtw   =  spec_sp.gl_vt(Nvt, mode="np")
    vp, vpw   =  spec_sp.gl_vp(Nvp)
    print(vt)

    vts, vtsw = spec_sp.gl_vt(Nvts, mode="npsp")
    vps, vpsw = spec_sp.gl_vp(Nvps)
    mg        = np.meshgrid(vt, vp, vts, vps, indexing="ij")
    vt_p      = np.acos(np.cos(mg[0]) * np.cos(mg[2]) + np.sin(mg[0]) * np.sin(mg[2]) * np.cos(mg[1] - mg[3]))
    vp_p      = 0 * vt_p

    # vts, vtsw = spec_sp.gl_vt(2 * Nvts, mode="npsp")
    # vps, vpsw = spec_sp.gl_vp(2 * Nvps)
    # mg        = np.meshgrid(vt, vp, vts, vps)
    # vt_p1     = np.acos(np.cos(mg[0]) * np.cos(mg[2]) + np.sin(mg[0]) * np.sin(mg[2]) * np.cos(mg[1] - mg[3]))

    # plt.subplot(1, 2, 1)
    # plt.imshow(vt_p[0, 0]); plt.colorbar()
    # plt.subplot(1, 2, 2)
    # plt.imshow(vt_p1[0, 0]); plt.colorbar()
    # plt.show()

    #++
    
    for didx, di in enumerate(["+", "-"]):
        for djdx, dj in enumerate(["+", "-"]):
            vt, vtw   =  spec_sp.gl_vt(Nvt, mode="np") if dj == "+" else  spec_sp.gl_vt(Nvt, mode="sp")
            vp, vpw   =  spec_sp.gl_vp(Nvp)
            
            vts, vtsw = spec_sp.gl_vt(Nvts, mode="npsp")
            vps, vpsw = spec_sp.gl_vp(Nvps)
            mg        = np.meshgrid(vt, vp, vts, vps, indexing="ij")
            vt_p      = np.acos(np.cos(mg[0]) * np.cos(mg[2]) + np.sin(mg[0]) * np.sin(mg[2]) * np.cos(mg[1] - mg[3]))
            vp_p      = 0 * vt_p

            for qs_idx, qs in enumerate(sph_harm_lm):
                Yqs = spec_sp._hemi_sph_harm_real(qs[0], qs[1], vt_p, vp_p, mode=di)
                for lm_idx, lm in enumerate(sph_harm_lm):
                    Ylm = spec_sp._hemi_sph_harm_real(lm[0], lm[1], mg[0], mg[1], mode=dj)
                    Wqs[didx * num_sh + qs_idx, djdx * num_sh + lm_idx] = np.einsum("abcd,a,b,c,d->", Yqs * Ylm, vtw, vpw, vtsw, vpsw)
                    
                    # plt.figure(figsize=(24, 24), dpi=100)
                    # plt_idx = 1
                    # for mi in range(len(vt)):
                    #     for mj in range(len(vp)):
                    #         #print(Ylm[mi, mj])
                    #         plt.subplot(len(vt), len(vp), plt_idx)
                    #         plt.title(r"$Y^{%s}_{%d%d} Y^{%s}_{%d%d} [%d, %d]$"%(di, qs[0], qs[1], dj, lm[0], lm[1], mi, mj))
                    #         plt.imshow((Ylm * Yqs)[mi, mj], extent=(0, 2*np.pi, 0, np.pi))
                    #         plt.colorbar()
                    #         #print(plt_idx)
                    #         plt_idx+=1
                    
                    # plt.tight_layout()
                    # plt.show()
                    # plt.close()

            



    #print(Wqs[0:num_sh, 0:num_sh])
    print(Wqs)




    


    

    ## ++



