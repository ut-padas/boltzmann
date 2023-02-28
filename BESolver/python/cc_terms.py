## generated code with sympy
## indexed with q-test l mode, l trial mode and s-h or g function modes
import numpy
import math
Ia_nz=[(0, 0, 0), (0, 1, 1), (0, 2, 2), (0, 3, 3), (0, 4, 4), (0, 5, 5), (0, 6, 6), (1, 0, 1), (1, 1, 0), (1, 1, 2), (1, 2, 1), (1, 2, 3), (1, 3, 2), (1, 3, 4), (1, 4, 3), (1, 4, 5), (1, 5, 4), (1, 5, 6), (1, 6, 5), (2, 0, 2), (2, 1, 1), (2, 1, 3), (2, 2, 0), (2, 2, 2), (2, 2, 4), (2, 3, 1), (2, 3, 3), (2, 3, 5), (2, 4, 2), (2, 4, 4), (2, 4, 6), (2, 5, 3), (2, 5, 5), (2, 6, 4), (2, 6, 6), (3, 0, 3), (3, 1, 2), (3, 1, 4), (3, 2, 1), (3, 2, 3), (3, 2, 5), (3, 3, 0), (3, 3, 2), (3, 3, 4), (3, 3, 6), (3, 4, 1), (3, 4, 3), (3, 4, 5), (3, 5, 2), (3, 5, 4), (3, 5, 6), (3, 6, 3), (3, 6, 5), (4, 0, 4), (4, 1, 3), (4, 1, 5), (4, 2, 2), (4, 2, 4), (4, 2, 6), (4, 3, 1), (4, 3, 3), (4, 3, 5), (4, 4, 0), (4, 4, 2), (4, 4, 4), (4, 4, 6), (4, 5, 1), (4, 5, 3), (4, 5, 5), (4, 6, 2), (4, 6, 4), (4, 6, 6), (5, 0, 5), (5, 1, 4), (5, 1, 6), (5, 2, 3), (5, 2, 5), (5, 3, 2), (5, 3, 4), (5, 3, 6), (5, 4, 1), (5, 4, 3), (5, 4, 5), (5, 5, 0), (5, 5, 2), (5, 5, 4), (5, 5, 6), (5, 6, 1), (5, 6, 3), (5, 6, 5), (6, 0, 6), (6, 1, 5), (6, 2, 4), (6, 2, 6), (6, 3, 3), (6, 3, 5), (6, 4, 2), (6, 4, 4), (6, 4, 6), (6, 5, 1), (6, 5, 3), (6, 5, 5), (6, 6, 0), (6, 6, 2), (6, 6, 4), (6, 6, 6)]
Ib_nz=[(0, 0, 0), (0, 1, 1), (0, 2, 2), (0, 3, 3), (0, 4, 4), (0, 5, 5), (0, 6, 6), (1, 0, 1), (1, 1, 0), (1, 1, 2), (1, 2, 1), (1, 2, 3), (1, 3, 2), (1, 3, 4), (1, 4, 3), (1, 4, 5), (1, 5, 4), (1, 5, 6), (1, 6, 5), (2, 0, 2), (2, 1, 1), (2, 1, 3), (2, 2, 0), (2, 2, 2), (2, 2, 4), (2, 3, 1), (2, 3, 3), (2, 3, 5), (2, 4, 2), (2, 4, 4), (2, 4, 6), (2, 5, 3), (2, 5, 5), (2, 6, 4), (2, 6, 6), (3, 0, 3), (3, 1, 2), (3, 1, 4), (3, 2, 1), (3, 2, 3), (3, 2, 5), (3, 3, 0), (3, 3, 2), (3, 3, 4), (3, 3, 6), (3, 4, 1), (3, 4, 3), (3, 4, 5), (3, 5, 2), (3, 5, 4), (3, 5, 6), (3, 6, 3), (3, 6, 5), (4, 0, 4), (4, 1, 3), (4, 1, 5), (4, 2, 2), (4, 2, 4), (4, 2, 6), (4, 3, 1), (4, 3, 3), (4, 3, 5), (4, 4, 0), (4, 4, 2), (4, 4, 4), (4, 4, 6), (4, 5, 1), (4, 5, 3), (4, 5, 5), (4, 6, 2), (4, 6, 4), (4, 6, 6), (5, 0, 5), (5, 1, 4), (5, 1, 6), (5, 2, 3), (5, 2, 5), (5, 3, 2), (5, 3, 4), (5, 3, 6), (5, 4, 1), (5, 4, 3), (5, 4, 5), (5, 5, 0), (5, 5, 2), (5, 5, 4), (5, 5, 6), (5, 6, 1), (5, 6, 3), (5, 6, 5), (6, 0, 6), (6, 1, 5), (6, 2, 4), (6, 2, 6), (6, 3, 3), (6, 3, 5), (6, 4, 2), (6, 4, 4), (6, 4, 6), (6, 5, 1), (6, 5, 3), (6, 5, 5), (6, 6, 0), (6, 6, 2), (6, 6, 4), (6, 6, 6)]
def Ia(RadialPoly, RadialPolyDeriv, vr, p, k, r, q, l, s, B_p_vr, B_k_vr, B_r_vr, DB_p_dvr, DB_k_dvr, DB_r_dvr, DB_p_dvr_dvr, DB_k_dvr_dvr, DB_r_dvr_dvr):
	if (q,l,s) == (0,0,0):
		return (1/2)*B_k_vr*DB_p_dvr*DB_r_dvr*vr**2/math.sqrt(math.pi)
	if (q,l,s) == (0,1,1):
		return (1/2)*B_k_vr*DB_p_dvr*DB_r_dvr*vr**2/math.sqrt(math.pi)
	if (q,l,s) == (0,2,2):
		return (1/2)*B_k_vr*DB_p_dvr*DB_r_dvr*vr**2/math.sqrt(math.pi)
	if (q,l,s) == (0,3,3):
		return (1/2)*B_k_vr*DB_p_dvr*DB_r_dvr*vr**2/math.sqrt(math.pi)
	if (q,l,s) == (0,4,4):
		return (1/2)*B_k_vr*DB_p_dvr*DB_r_dvr*vr**2/math.sqrt(math.pi)
	if (q,l,s) == (0,5,5):
		return (1/2)*B_k_vr*DB_p_dvr*DB_r_dvr*vr**2/math.sqrt(math.pi)
	if (q,l,s) == (0,6,6):
		return (1/2)*B_k_vr*DB_p_dvr*DB_r_dvr*vr**2/math.sqrt(math.pi)
	if (q,l,s) == (1,0,1):
		return (1/2)*B_k_vr*(2*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (1,1,0):
		return (1/2)*B_k_vr*DB_p_dvr*DB_r_dvr*vr**2/math.sqrt(math.pi)
	if (q,l,s) == (1,1,2):
		return (1/5)*math.sqrt(5)*B_k_vr*(3*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (1,2,1):
		return (1/5)*math.sqrt(5)*B_k_vr*(-B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (1,2,3):
		return (3/70)*math.sqrt(105)*B_k_vr*(4*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (1,3,2):
		return (3/70)*math.sqrt(105)*B_k_vr*(-2*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (1,3,4):
		return (2/21)*math.sqrt(21)*B_k_vr*(5*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (1,4,3):
		return (2/21)*math.sqrt(21)*B_k_vr*(-3*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (1,4,5):
		return (5/66)*math.sqrt(33)*B_k_vr*(6*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (1,5,4):
		return (5/66)*math.sqrt(33)*B_k_vr*(-4*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (1,5,6):
		return (3/143)*math.sqrt(429)*B_k_vr*(7*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (1,6,5):
		return (3/143)*math.sqrt(429)*B_k_vr*(-5*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (2,0,2):
		return (1/2)*B_k_vr*(6*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (2,1,1):
		return (1/5)*math.sqrt(5)*B_k_vr*(3*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (2,1,3):
		return (3/70)*math.sqrt(105)*B_k_vr*(8*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (2,2,0):
		return (1/2)*B_k_vr*DB_p_dvr*DB_r_dvr*vr**2/math.sqrt(math.pi)
	if (q,l,s) == (2,2,2):
		return (1/7)*math.sqrt(5)*B_k_vr*(3*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (2,2,4):
		return (3/7)*B_k_vr*(10*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (2,3,1):
		return (3/70)*math.sqrt(105)*B_k_vr*(-2*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (2,3,3):
		return (2/15)*math.sqrt(5)*B_k_vr*(3*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (2,3,5):
		return (5/231)*math.sqrt(385)*B_k_vr*(12*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (2,4,2):
		return (3/7)*B_k_vr*(-4*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (2,4,4):
		return (10/77)*math.sqrt(5)*B_k_vr*(3*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (2,4,6):
		return (15/286)*math.sqrt(65)*B_k_vr*(14*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (2,5,3):
		return (5/231)*math.sqrt(385)*B_k_vr*(-6*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (2,5,5):
		return (5/39)*math.sqrt(5)*B_k_vr*(3*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (2,6,4):
		return (15/286)*math.sqrt(65)*B_k_vr*(-8*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (2,6,6):
		return (7/55)*math.sqrt(5)*B_k_vr*(3*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (3,0,3):
		return (1/2)*B_k_vr*(12*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (3,1,2):
		return (3/70)*math.sqrt(105)*B_k_vr*(8*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (3,1,4):
		return (2/21)*math.sqrt(21)*B_k_vr*(15*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (3,2,1):
		return (3/70)*math.sqrt(105)*B_k_vr*(4*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (3,2,3):
		return (2/15)*math.sqrt(5)*B_k_vr*(9*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (3,2,5):
		return (5/231)*math.sqrt(385)*B_k_vr*(18*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (3,3,0):
		return (1/2)*B_k_vr*DB_p_dvr*DB_r_dvr*vr**2/math.sqrt(math.pi)
	if (q,l,s) == (3,3,2):
		return (2/15)*math.sqrt(5)*B_k_vr*(3*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (3,3,4):
		return (3/11)*B_k_vr*(10*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (3,3,6):
		return (50/429)*math.sqrt(13)*B_k_vr*(21*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (3,4,1):
		return (2/21)*math.sqrt(21)*B_k_vr*(-3*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (3,4,3):
		return (3/11)*B_k_vr*(2*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (3,4,5):
		return (30/1001)*math.sqrt(77)*B_k_vr*(11*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (3,5,2):
		return (5/231)*math.sqrt(385)*B_k_vr*(-6*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (3,5,4):
		return (30/1001)*math.sqrt(77)*B_k_vr*(B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (3,5,6):
		return (7/858)*math.sqrt(1001)*B_k_vr*(12*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (3,6,3):
		return (50/429)*math.sqrt(13)*B_k_vr*(-9*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (3,6,5):
		return (7/858)*math.sqrt(1001)*B_k_vr*DB_p_dvr*DB_r_dvr*vr**2/math.sqrt(math.pi)
	if (q,l,s) == (4,0,4):
		return (1/2)*B_k_vr*(20*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (4,1,3):
		return (2/21)*math.sqrt(21)*B_k_vr*(15*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (4,1,5):
		return (5/66)*math.sqrt(33)*B_k_vr*(24*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (4,2,2):
		return (3/7)*B_k_vr*(10*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (4,2,4):
		return (10/77)*math.sqrt(5)*B_k_vr*(17*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (4,2,6):
		return (15/286)*math.sqrt(65)*B_k_vr*(28*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (4,3,1):
		return (2/21)*math.sqrt(21)*B_k_vr*(5*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (4,3,3):
		return (3/11)*B_k_vr*(10*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (4,3,5):
		return (30/1001)*math.sqrt(77)*B_k_vr*(19*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (4,4,0):
		return (1/2)*B_k_vr*DB_p_dvr*DB_r_dvr*vr**2/math.sqrt(math.pi)
	if (q,l,s) == (4,4,2):
		return (10/77)*math.sqrt(5)*B_k_vr*(3*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (4,4,4):
		return (243/1001)*B_k_vr*(10*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (4,4,6):
		return (10/143)*math.sqrt(13)*B_k_vr*(21*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (4,5,1):
		return (5/66)*math.sqrt(33)*B_k_vr*(-4*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (4,5,3):
		return (30/1001)*math.sqrt(77)*B_k_vr*(B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (4,5,5):
		return (3/13)*B_k_vr*(10*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (4,6,2):
		return (15/286)*math.sqrt(65)*B_k_vr*(-8*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (4,6,4):
		return (10/143)*math.sqrt(13)*B_k_vr*(-B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (4,6,6):
		return (42/187)*B_k_vr*(10*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (5,0,5):
		return (1/2)*B_k_vr*(30*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (5,1,4):
		return (5/66)*math.sqrt(33)*B_k_vr*(24*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (5,1,6):
		return (3/143)*math.sqrt(429)*B_k_vr*(35*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (5,2,3):
		return (5/231)*math.sqrt(385)*B_k_vr*(18*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (5,2,5):
		return (5/39)*math.sqrt(5)*B_k_vr*(27*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (5,3,2):
		return (5/231)*math.sqrt(385)*B_k_vr*(12*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (5,3,4):
		return (30/1001)*math.sqrt(77)*B_k_vr*(19*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (5,3,6):
		return (7/858)*math.sqrt(1001)*B_k_vr*(30*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (5,4,1):
		return (5/66)*math.sqrt(33)*B_k_vr*(6*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (5,4,3):
		return (30/1001)*math.sqrt(77)*B_k_vr*(11*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (5,4,5):
		return (3/13)*B_k_vr*(20*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (5,5,0):
		return (1/2)*B_k_vr*DB_p_dvr*DB_r_dvr*vr**2/math.sqrt(math.pi)
	if (q,l,s) == (5,5,2):
		return (5/39)*math.sqrt(5)*B_k_vr*(3*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (5,5,4):
		return (3/13)*B_k_vr*(10*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (5,5,6):
		return (40/663)*math.sqrt(13)*B_k_vr*(21*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (5,6,1):
		return (3/143)*math.sqrt(429)*B_k_vr*(-5*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (5,6,3):
		return (7/858)*math.sqrt(1001)*B_k_vr*DB_p_dvr*DB_r_dvr*vr**2/math.sqrt(math.pi)
	if (q,l,s) == (5,6,5):
		return (40/663)*math.sqrt(13)*B_k_vr*(9*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (6,0,6):
		return (1/2)*B_k_vr*(42*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (6,1,5):
		return (3/143)*math.sqrt(429)*B_k_vr*(35*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (6,2,4):
		return (15/286)*math.sqrt(65)*B_k_vr*(28*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (6,2,6):
		return (7/55)*math.sqrt(5)*B_k_vr*(39*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (6,3,3):
		return (50/429)*math.sqrt(13)*B_k_vr*(21*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (6,3,5):
		return (7/858)*math.sqrt(1001)*B_k_vr*(30*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (6,4,2):
		return (15/286)*math.sqrt(65)*B_k_vr*(14*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (6,4,4):
		return (10/143)*math.sqrt(13)*B_k_vr*(21*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (6,4,6):
		return (42/187)*B_k_vr*(32*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (6,5,1):
		return (3/143)*math.sqrt(429)*B_k_vr*(7*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (6,5,3):
		return (7/858)*math.sqrt(1001)*B_k_vr*(12*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (6,5,5):
		return (40/663)*math.sqrt(13)*B_k_vr*(21*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (6,6,0):
		return (1/2)*B_k_vr*DB_p_dvr*DB_r_dvr*vr**2/math.sqrt(math.pi)
	if (q,l,s) == (6,6,2):
		return (7/55)*math.sqrt(5)*B_k_vr*(3*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (6,6,4):
		return (42/187)*B_k_vr*(10*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (6,6,6):
		return (200/3553)*math.sqrt(13)*B_k_vr*(21*B_p_vr*B_r_vr + DB_p_dvr*DB_r_dvr*vr**2)/math.sqrt(math.pi)
	return 0


def Ib(RadialPoly, RadialPolyDeriv, vr, p, k, r, q, l, s, B_p_vr, B_k_vr, B_r_vr, DB_p_dvr, DB_k_dvr, DB_r_dvr, DB_p_dvr_dvr, DB_k_dvr_dvr, DB_r_dvr_dvr):
	if (q,l,s) == (0,0,0):
		return (1/4)*B_k_vr*(2*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2)/math.sqrt(math.pi)
	if (q,l,s) == (0,1,1):
		return (1/4)*B_k_vr*(-2*B_r_vr*DB_p_dvr + vr*(2*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2))/(math.sqrt(math.pi)*vr)
	if (q,l,s) == (0,2,2):
		return (1/4)*B_k_vr*(-6*B_r_vr*DB_p_dvr + vr*(2*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2))/(math.sqrt(math.pi)*vr)
	if (q,l,s) == (0,3,3):
		return (1/4)*B_k_vr*(-12*B_r_vr*DB_p_dvr + vr*(2*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2))/(math.sqrt(math.pi)*vr)
	if (q,l,s) == (0,4,4):
		return (1/4)*B_k_vr*(-20*B_r_vr*DB_p_dvr + vr*(2*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2))/(math.sqrt(math.pi)*vr)
	if (q,l,s) == (0,5,5):
		return (1/4)*B_k_vr*(-30*B_r_vr*DB_p_dvr + vr*(2*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2))/(math.sqrt(math.pi)*vr)
	if (q,l,s) == (0,6,6):
		return (1/4)*B_k_vr*(-42*B_r_vr*DB_p_dvr + vr*(2*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2))/(math.sqrt(math.pi)*vr)
	if (q,l,s) == (1,0,1):
		return (1/4)*B_k_vr*(6*B_p_vr*B_r_vr + vr**2*(6*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) - 6*vr*(B_p_vr*DB_r_dvr + B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (1,1,0):
		return (1/4)*B_k_vr*(-2*B_p_vr*DB_r_dvr + vr*(2*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2))/(math.sqrt(math.pi)*vr)
	if (q,l,s) == (1,1,2):
		return (1/10)*math.sqrt(5)*B_k_vr*(12*B_p_vr*B_r_vr - 8*B_p_vr*DB_r_dvr*vr - 12*B_r_vr*DB_p_dvr*vr + 8*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (1,2,1):
		return (1/10)*math.sqrt(5)*B_k_vr*DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2/math.sqrt(math.pi)
	if (q,l,s) == (1,2,3):
		return (3/140)*math.sqrt(105)*B_k_vr*(20*B_p_vr*B_r_vr + vr**2*(10*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) - 10*vr*(B_p_vr*DB_r_dvr + 2*B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (1,3,2):
		return (3/140)*math.sqrt(105)*B_k_vr*(2*B_p_vr*B_r_vr + vr**2*(-2*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) + 2*vr*(B_p_vr*DB_r_dvr - B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (1,3,4):
		return (1/21)*math.sqrt(21)*B_k_vr*(30*B_p_vr*B_r_vr - 12*B_p_vr*DB_r_dvr*vr - 30*B_r_vr*DB_p_dvr*vr + 12*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (1,4,3):
		return (1/21)*math.sqrt(21)*B_k_vr*(6*B_p_vr*B_r_vr + vr**2*(-4*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) + 2*vr*(2*B_p_vr*DB_r_dvr - 3*B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (1,4,5):
		return (5/132)*math.sqrt(33)*B_k_vr*(42*B_p_vr*B_r_vr + vr**2*(14*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) - 14*vr*(B_p_vr*DB_r_dvr + 3*B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (1,5,4):
		return (5/132)*math.sqrt(33)*B_k_vr*(12*B_p_vr*B_r_vr + vr**2*(-6*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) + 6*vr*(B_p_vr*DB_r_dvr - 2*B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (1,5,6):
		return (3/286)*math.sqrt(429)*B_k_vr*(56*B_p_vr*B_r_vr - 16*B_p_vr*DB_r_dvr*vr - 56*B_r_vr*DB_p_dvr*vr + 16*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (1,6,5):
		return (3/286)*math.sqrt(429)*B_k_vr*(20*B_p_vr*B_r_vr + vr**2*(-8*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) + 4*vr*(2*B_p_vr*DB_r_dvr - 5*B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (2,0,2):
		return (1/4)*B_k_vr*(42*B_p_vr*B_r_vr + vr**2*(14*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) - 18*vr*(B_p_vr*DB_r_dvr + B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (2,1,1):
		return (1/10)*math.sqrt(5)*B_k_vr*(12*B_p_vr*B_r_vr - 12*B_p_vr*DB_r_dvr*vr - 8*B_r_vr*DB_p_dvr*vr + 8*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (2,1,3):
		return (3/140)*math.sqrt(105)*B_k_vr*(72*B_p_vr*B_r_vr - 22*B_p_vr*DB_r_dvr*vr - 28*B_r_vr*DB_p_dvr*vr + 18*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (2,2,0):
		return (1/4)*B_k_vr*(-6*B_p_vr*DB_r_dvr + vr*(2*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2))/(math.sqrt(math.pi)*vr)
	if (q,l,s) == (2,2,2):
		return (1/14)*math.sqrt(5)*B_k_vr*(12*B_p_vr*B_r_vr + vr**2*(8*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) - 12*vr*(B_p_vr*DB_r_dvr + B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (2,2,4):
		return (3/14)*B_k_vr*(110*B_p_vr*B_r_vr - 26*B_p_vr*DB_r_dvr*vr - 40*B_r_vr*DB_p_dvr*vr + 22*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (2,3,1):
		return (3/140)*math.sqrt(105)*B_k_vr*(2*B_p_vr*B_r_vr + vr**2*(-2*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) + 2*vr*(-B_p_vr*DB_r_dvr + B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (2,3,3):
		return (1/15)*math.sqrt(5)*B_k_vr*(12*B_p_vr*B_r_vr - 12*B_p_vr*DB_r_dvr*vr - 18*B_r_vr*DB_p_dvr*vr + 8*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (2,3,5):
		return (5/462)*math.sqrt(385)*B_k_vr*(156*B_p_vr*B_r_vr - 30*B_p_vr*DB_r_dvr*vr - 54*B_r_vr*DB_p_dvr*vr + 26*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (2,4,2):
		return (3/14)*B_k_vr*(12*B_p_vr*B_r_vr + vr**2*(-6*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) + 2*vr*(B_p_vr*DB_r_dvr + B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (2,4,4):
		return (5/77)*math.sqrt(5)*B_k_vr*(12*B_p_vr*B_r_vr - 12*B_p_vr*DB_r_dvr*vr - 26*B_r_vr*DB_p_dvr*vr + 8*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (2,4,6):
		return (15/572)*math.sqrt(65)*B_k_vr*(210*B_p_vr*B_r_vr - 34*B_p_vr*DB_r_dvr*vr - 70*B_r_vr*DB_p_dvr*vr + 30*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (2,5,3):
		return (5/462)*math.sqrt(385)*B_k_vr*(30*B_p_vr*B_r_vr + 6*B_p_vr*DB_r_dvr*vr + vr**2*(-10*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (2,5,5):
		return (5/78)*math.sqrt(5)*B_k_vr*(12*B_p_vr*B_r_vr + vr**2*(8*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) - 12*vr*(B_p_vr*DB_r_dvr + 3*B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (2,6,4):
		return (15/572)*math.sqrt(65)*B_k_vr*(56*B_p_vr*B_r_vr + vr**2*(-14*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) + 2*vr*(5*B_p_vr*DB_r_dvr - 2*B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (2,6,6):
		return (7/110)*math.sqrt(5)*B_k_vr*(12*B_p_vr*B_r_vr + vr**2*(8*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) - 12*vr*(B_p_vr*DB_r_dvr + 4*B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (3,0,3):
		return (1/4)*B_k_vr*(156*B_p_vr*B_r_vr + vr**2*(26*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) - 36*vr*(B_p_vr*DB_r_dvr + B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (3,1,2):
		return (3/140)*math.sqrt(105)*B_k_vr*(72*B_p_vr*B_r_vr - 28*B_p_vr*DB_r_dvr*vr - 22*B_r_vr*DB_p_dvr*vr + 18*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (3,1,4):
		return (1/21)*math.sqrt(21)*B_k_vr*(240*B_p_vr*B_r_vr - 42*B_p_vr*DB_r_dvr*vr - 50*B_r_vr*DB_p_dvr*vr + 32*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (3,2,1):
		return (3/140)*math.sqrt(105)*B_k_vr*(20*B_p_vr*B_r_vr + vr**2*(10*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) - 10*vr*(2*B_p_vr*DB_r_dvr + B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (3,2,3):
		return (1/15)*math.sqrt(5)*B_k_vr*(90*B_p_vr*B_r_vr + vr**2*(20*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) - 30*vr*(B_p_vr*DB_r_dvr + B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (3,2,5):
		return (5/462)*math.sqrt(385)*B_k_vr*(342*B_p_vr*B_r_vr - 48*B_p_vr*DB_r_dvr*vr - 66*B_r_vr*DB_p_dvr*vr + 38*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (3,3,0):
		return (1/4)*B_k_vr*(-12*B_p_vr*DB_r_dvr + vr*(2*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2))/(math.sqrt(math.pi)*vr)
	if (q,l,s) == (3,3,2):
		return (1/15)*math.sqrt(5)*B_k_vr*(12*B_p_vr*B_r_vr - 18*B_p_vr*DB_r_dvr*vr - 12*B_r_vr*DB_p_dvr*vr + 8*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (3,3,4):
		return (3/22)*B_k_vr*(110*B_p_vr*B_r_vr - 32*B_p_vr*DB_r_dvr*vr - 40*B_r_vr*DB_p_dvr*vr + 22*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (3,3,6):
		return (25/429)*math.sqrt(13)*B_k_vr*(462*B_p_vr*B_r_vr - 54*B_p_vr*DB_r_dvr*vr - 84*B_r_vr*DB_p_dvr*vr + 44*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (3,4,1):
		return (1/21)*math.sqrt(21)*B_k_vr*(6*B_p_vr*B_r_vr + vr**2*(-4*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) + 2*vr*(-3*B_p_vr*DB_r_dvr + 2*B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (3,4,3):
		return (3/22)*B_k_vr*(6*B_p_vr*B_r_vr + vr**2*(6*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) - 16*vr*(B_p_vr*DB_r_dvr + B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (3,4,5):
		return (15/1001)*math.sqrt(77)*B_k_vr*(132*B_p_vr*B_r_vr - 34*B_p_vr*DB_r_dvr*vr - 52*B_r_vr*DB_p_dvr*vr + 24*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (3,5,2):
		return (5/462)*math.sqrt(385)*B_k_vr*(30*B_p_vr*B_r_vr + 6*B_r_vr*DB_p_dvr*vr + vr**2*(-10*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (3,5,4):
		return (15/1001)*math.sqrt(77)*B_k_vr*(2*B_p_vr*B_r_vr - 14*B_p_vr*DB_r_dvr*vr - 22*B_r_vr*DB_p_dvr*vr + 4*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (3,5,6):
		return (7/1716)*math.sqrt(1001)*B_k_vr*(156*B_p_vr*B_r_vr - 36*B_p_vr*DB_r_dvr*vr - 66*B_r_vr*DB_p_dvr*vr + 26*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (3,6,3):
		return (25/429)*math.sqrt(13)*B_k_vr*(72*B_p_vr*B_r_vr + vr**2*(-16*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) + 6*vr*(B_p_vr*DB_r_dvr + B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (3,6,5):
		return (7/1716)*math.sqrt(1001)*B_k_vr*(-12*B_p_vr*DB_r_dvr - 30*B_r_vr*DB_p_dvr + vr*(2*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2))/(math.sqrt(math.pi)*vr)
	if (q,l,s) == (4,0,4):
		return (1/4)*B_k_vr*(420*B_p_vr*B_r_vr + vr**2*(42*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) - 60*vr*(B_p_vr*DB_r_dvr + B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (4,1,3):
		return (1/21)*math.sqrt(21)*B_k_vr*(240*B_p_vr*B_r_vr - 50*B_p_vr*DB_r_dvr*vr - 42*B_r_vr*DB_p_dvr*vr + 32*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (4,1,5):
		return (5/132)*math.sqrt(33)*B_k_vr*(600*B_p_vr*B_r_vr - 68*B_p_vr*DB_r_dvr*vr - 78*B_r_vr*DB_p_dvr*vr + 50*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (4,2,2):
		return (3/14)*B_k_vr*(110*B_p_vr*B_r_vr - 40*B_p_vr*DB_r_dvr*vr - 26*B_r_vr*DB_p_dvr*vr + 22*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (4,2,4):
		return (5/77)*math.sqrt(5)*B_k_vr*(306*B_p_vr*B_r_vr + vr**2*(36*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) - 54*vr*(B_p_vr*DB_r_dvr + B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (4,2,6):
		return (15/572)*math.sqrt(65)*B_k_vr*(812*B_p_vr*B_r_vr - 76*B_p_vr*DB_r_dvr*vr - 98*B_r_vr*DB_p_dvr*vr + 58*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (4,3,1):
		return (1/21)*math.sqrt(21)*B_k_vr*(30*B_p_vr*B_r_vr - 30*B_p_vr*DB_r_dvr*vr - 12*B_r_vr*DB_p_dvr*vr + 12*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (4,3,3):
		return (3/22)*B_k_vr*(110*B_p_vr*B_r_vr - 40*B_p_vr*DB_r_dvr*vr - 32*B_r_vr*DB_p_dvr*vr + 22*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (4,3,5):
		return (15/1001)*math.sqrt(77)*B_k_vr*(380*B_p_vr*B_r_vr - 58*B_p_vr*DB_r_dvr*vr - 68*B_r_vr*DB_p_dvr*vr + 40*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (4,4,0):
		return (1/4)*B_k_vr*(-20*B_p_vr*DB_r_dvr + vr*(2*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2))/(math.sqrt(math.pi)*vr)
	if (q,l,s) == (4,4,2):
		return (5/77)*math.sqrt(5)*B_k_vr*(12*B_p_vr*B_r_vr - 26*B_p_vr*DB_r_dvr*vr - 12*B_r_vr*DB_p_dvr*vr + 8*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (4,4,4):
		return (243/2002)*B_k_vr*(110*B_p_vr*B_r_vr + vr**2*(22*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) - 40*vr*(B_p_vr*DB_r_dvr + B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (4,4,6):
		return (5/143)*math.sqrt(13)*B_k_vr*(462*B_p_vr*B_r_vr - 62*B_p_vr*DB_r_dvr*vr - 84*B_r_vr*DB_p_dvr*vr + 44*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (4,5,1):
		return (5/132)*math.sqrt(33)*B_k_vr*(12*B_p_vr*B_r_vr + vr**2*(-6*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) + 6*vr*(-2*B_p_vr*DB_r_dvr + B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (4,5,3):
		return (15/1001)*math.sqrt(77)*B_k_vr*(2*B_p_vr*B_r_vr - 22*B_p_vr*DB_r_dvr*vr - 14*B_r_vr*DB_p_dvr*vr + 4*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (4,5,5):
		return (3/26)*B_k_vr*(110*B_p_vr*B_r_vr - 40*B_p_vr*DB_r_dvr*vr - 50*B_r_vr*DB_p_dvr*vr + 22*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (4,6,2):
		return (15/572)*math.sqrt(65)*B_k_vr*(56*B_p_vr*B_r_vr + vr**2*(-14*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) + 2*vr*(-2*B_p_vr*DB_r_dvr + 5*B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (4,6,4):
		return (5/143)*math.sqrt(13)*B_k_vr*(-18*B_p_vr*DB_r_dvr - 18*B_r_vr*DB_p_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**3)/(math.sqrt(math.pi)*vr)
	if (q,l,s) == (4,6,6):
		return (21/187)*B_k_vr*(110*B_p_vr*B_r_vr - 40*B_p_vr*DB_r_dvr*vr - 62*B_r_vr*DB_p_dvr*vr + 22*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (5,0,5):
		return (1/4)*B_k_vr*(930*B_p_vr*B_r_vr + vr**2*(62*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) - 90*vr*(B_p_vr*DB_r_dvr + B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (5,1,4):
		return (5/132)*math.sqrt(33)*B_k_vr*(600*B_p_vr*B_r_vr - 78*B_p_vr*DB_r_dvr*vr - 68*B_r_vr*DB_p_dvr*vr + 50*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (5,1,6):
		return (3/286)*math.sqrt(429)*B_k_vr*(1260*B_p_vr*B_r_vr - 100*B_p_vr*DB_r_dvr*vr - 112*B_r_vr*DB_p_dvr*vr + 72*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (5,2,3):
		return (5/462)*math.sqrt(385)*B_k_vr*(342*B_p_vr*B_r_vr - 66*B_p_vr*DB_r_dvr*vr - 48*B_r_vr*DB_p_dvr*vr + 38*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (5,2,5):
		return (5/78)*math.sqrt(5)*B_k_vr*(756*B_p_vr*B_r_vr + vr**2*(56*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) - 84*vr*(B_p_vr*DB_r_dvr + B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (5,3,2):
		return (5/462)*math.sqrt(385)*B_k_vr*(156*B_p_vr*B_r_vr - 54*B_p_vr*DB_r_dvr*vr - 30*B_r_vr*DB_p_dvr*vr + 26*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (5,3,4):
		return (15/1001)*math.sqrt(77)*B_k_vr*(380*B_p_vr*B_r_vr - 68*B_p_vr*DB_r_dvr*vr - 58*B_r_vr*DB_p_dvr*vr + 40*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (5,3,6):
		return (7/1716)*math.sqrt(1001)*B_k_vr*(930*B_p_vr*B_r_vr - 90*B_p_vr*DB_r_dvr*vr - 102*B_r_vr*DB_p_dvr*vr + 62*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (5,4,1):
		return (5/132)*math.sqrt(33)*B_k_vr*(42*B_p_vr*B_r_vr + vr**2*(14*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) - 14*vr*(3*B_p_vr*DB_r_dvr + B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (5,4,3):
		return (15/1001)*math.sqrt(77)*B_k_vr*(132*B_p_vr*B_r_vr - 52*B_p_vr*DB_r_dvr*vr - 34*B_r_vr*DB_p_dvr*vr + 24*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (5,4,5):
		return (3/26)*B_k_vr*(420*B_p_vr*B_r_vr + vr**2*(42*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) - 70*vr*(B_p_vr*DB_r_dvr + B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (5,5,0):
		return (1/4)*B_k_vr*(-30*B_p_vr*DB_r_dvr + vr*(2*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2))/(math.sqrt(math.pi)*vr)
	if (q,l,s) == (5,5,2):
		return (5/78)*math.sqrt(5)*B_k_vr*(12*B_p_vr*B_r_vr + vr**2*(8*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) - 12*vr*(3*B_p_vr*DB_r_dvr + B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (5,5,4):
		return (3/26)*B_k_vr*(110*B_p_vr*B_r_vr - 50*B_p_vr*DB_r_dvr*vr - 40*B_r_vr*DB_p_dvr*vr + 22*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (5,5,6):
		return (20/663)*math.sqrt(13)*B_k_vr*(462*B_p_vr*B_r_vr - 72*B_p_vr*DB_r_dvr*vr - 84*B_r_vr*DB_p_dvr*vr + 44*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (5,6,1):
		return (3/286)*math.sqrt(429)*B_k_vr*(20*B_p_vr*B_r_vr + vr**2*(-8*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) + 4*vr*(-5*B_p_vr*DB_r_dvr + 2*B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (5,6,3):
		return (7/1716)*math.sqrt(1001)*B_k_vr*(-30*B_p_vr*DB_r_dvr - 12*B_r_vr*DB_p_dvr + vr*(2*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2))/(math.sqrt(math.pi)*vr)
	if (q,l,s) == (5,6,5):
		return (20/663)*math.sqrt(13)*B_k_vr*(90*B_p_vr*B_r_vr + vr**2*(20*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) - 48*vr*(B_p_vr*DB_r_dvr + B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (6,0,6):
		return (1/4)*B_k_vr*(1806*B_p_vr*B_r_vr + vr**2*(86*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) - 126*vr*(B_p_vr*DB_r_dvr + B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (6,1,5):
		return (3/286)*math.sqrt(429)*B_k_vr*(1260*B_p_vr*B_r_vr - 112*B_p_vr*DB_r_dvr*vr - 100*B_r_vr*DB_p_dvr*vr + 72*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (6,2,4):
		return (15/572)*math.sqrt(65)*B_k_vr*(812*B_p_vr*B_r_vr - 98*B_p_vr*DB_r_dvr*vr - 76*B_r_vr*DB_p_dvr*vr + 58*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (6,2,6):
		return (7/110)*math.sqrt(5)*B_k_vr*(1560*B_p_vr*B_r_vr + vr**2*(80*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) - 120*vr*(B_p_vr*DB_r_dvr + B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (6,3,3):
		return (25/429)*math.sqrt(13)*B_k_vr*(462*B_p_vr*B_r_vr - 84*B_p_vr*DB_r_dvr*vr - 54*B_r_vr*DB_p_dvr*vr + 44*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (6,3,5):
		return (7/1716)*math.sqrt(1001)*B_k_vr*(930*B_p_vr*B_r_vr - 102*B_p_vr*DB_r_dvr*vr - 90*B_r_vr*DB_p_dvr*vr + 62*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (6,4,2):
		return (15/572)*math.sqrt(65)*B_k_vr*(210*B_p_vr*B_r_vr - 70*B_p_vr*DB_r_dvr*vr - 34*B_r_vr*DB_p_dvr*vr + 30*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (6,4,4):
		return (5/143)*math.sqrt(13)*B_k_vr*(462*B_p_vr*B_r_vr - 84*B_p_vr*DB_r_dvr*vr - 62*B_r_vr*DB_p_dvr*vr + 44*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (6,4,6):
		return (21/187)*B_k_vr*(1056*B_p_vr*B_r_vr + vr**2*(66*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) - 106*vr*(B_p_vr*DB_r_dvr + B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (6,5,1):
		return (3/286)*math.sqrt(429)*B_k_vr*(56*B_p_vr*B_r_vr - 56*B_p_vr*DB_r_dvr*vr - 16*B_r_vr*DB_p_dvr*vr + 16*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (6,5,3):
		return (7/1716)*math.sqrt(1001)*B_k_vr*(156*B_p_vr*B_r_vr - 66*B_p_vr*DB_r_dvr*vr - 36*B_r_vr*DB_p_dvr*vr + 26*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (6,5,5):
		return (20/663)*math.sqrt(13)*B_k_vr*(462*B_p_vr*B_r_vr - 84*B_p_vr*DB_r_dvr*vr - 72*B_r_vr*DB_p_dvr*vr + 44*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (6,6,0):
		return (1/4)*B_k_vr*(-42*B_p_vr*DB_r_dvr + vr*(2*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2))/(math.sqrt(math.pi)*vr)
	if (q,l,s) == (6,6,2):
		return (7/110)*math.sqrt(5)*B_k_vr*(12*B_p_vr*B_r_vr + vr**2*(8*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) - 12*vr*(4*B_p_vr*DB_r_dvr + B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (6,6,4):
		return (21/187)*B_k_vr*(110*B_p_vr*B_r_vr - 62*B_p_vr*DB_r_dvr*vr - 40*B_r_vr*DB_p_dvr*vr + 22*DB_p_dvr*DB_r_dvr*vr**2 + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**4)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (6,6,6):
		return (100/3553)*math.sqrt(13)*B_k_vr*(462*B_p_vr*B_r_vr + vr**2*(44*DB_p_dvr*DB_r_dvr + DB_p_dvr_dvr*DB_r_dvr_dvr*vr**2) - 84*vr*(B_p_vr*DB_r_dvr + B_r_vr*DB_p_dvr))/(math.sqrt(math.pi)*vr**2)
	return 0
