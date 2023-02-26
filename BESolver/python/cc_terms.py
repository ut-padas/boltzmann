## generated code with sympy
## indexed with q-test l mode, l trial mode and s-h or g function modes
import numpy
import math
Ia_nz=[(0, 0, 0), (0, 1, 1), (0, 2, 2), (0, 3, 3), (0, 4, 4), (1, 0, 1), (1, 1, 0), (1, 1, 2), (1, 2, 1), (1, 2, 3), (1, 3, 2), (1, 3, 4), (1, 4, 3), (2, 0, 2), (2, 1, 1), (2, 1, 3), (2, 2, 0), (2, 2, 2), (2, 2, 4), (2, 3, 1), (2, 3, 3), (2, 4, 2), (2, 4, 4), (3, 0, 3), (3, 1, 2), (3, 1, 4), (3, 2, 1), (3, 2, 3), (3, 3, 0), (3, 3, 2), (3, 3, 4), (3, 4, 1), (3, 4, 3), (4, 0, 4), (4, 1, 3), (4, 2, 2), (4, 2, 4), (4, 3, 1), (4, 3, 3), (4, 4, 0), (4, 4, 2), (4, 4, 4)]
Ib_nz=[(0, 0, 0), (0, 1, 1), (0, 2, 2), (0, 3, 3), (0, 4, 4), (1, 0, 1), (1, 1, 0), (1, 1, 2), (1, 2, 1), (1, 2, 3), (1, 3, 2), (1, 3, 4), (1, 4, 3), (2, 0, 2), (2, 1, 1), (2, 1, 3), (2, 2, 0), (2, 2, 2), (2, 2, 4), (2, 3, 1), (2, 3, 3), (2, 4, 2), (2, 4, 4), (3, 0, 3), (3, 1, 2), (3, 1, 4), (3, 2, 1), (3, 2, 3), (3, 3, 0), (3, 3, 2), (3, 3, 4), (3, 4, 1), (3, 4, 3), (4, 0, 4), (4, 1, 3), (4, 2, 2), (4, 2, 4), (4, 3, 1), (4, 3, 3), (4, 4, 0), (4, 4, 2), (4, 4, 4)]
def Ia(RadialPoly, RadialPolyDeriv, vr, p, k, r, q, l, s):
	if (q,l,s) == (0,0,0):
		return (1/2)*vr**2*RadialPoly(vr, k)*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)/math.sqrt(math.pi)
	if (q,l,s) == (0,1,1):
		return (1/2)*vr**2*RadialPoly(vr, k)*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)/math.sqrt(math.pi)
	if (q,l,s) == (0,2,2):
		return (1/2)*vr**2*RadialPoly(vr, k)*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)/math.sqrt(math.pi)
	if (q,l,s) == (0,3,3):
		return (1/2)*vr**2*RadialPoly(vr, k)*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)/math.sqrt(math.pi)
	if (q,l,s) == (0,4,4):
		return (1/2)*vr**2*RadialPoly(vr, k)*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)/math.sqrt(math.pi)
	if (q,l,s) == (1,0,1):
		return (1/2)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 2*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (1,1,0):
		return (1/2)*vr**2*RadialPoly(vr, k)*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)/math.sqrt(math.pi)
	if (q,l,s) == (1,1,2):
		return (1/5)*math.sqrt(5)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 3*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (1,2,1):
		return (1/5)*math.sqrt(5)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) - RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (1,2,3):
		return (3/70)*math.sqrt(105)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 4*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (1,3,2):
		return (3/70)*math.sqrt(105)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) - 2*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (1,3,4):
		return (2/21)*math.sqrt(21)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 5*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (1,4,3):
		return (2/21)*math.sqrt(21)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) - 3*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (2,0,2):
		return (1/2)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 6*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (2,1,1):
		return (1/5)*math.sqrt(5)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 3*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (2,1,3):
		return (3/70)*math.sqrt(105)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 8*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (2,2,0):
		return (1/2)*vr**2*RadialPoly(vr, k)*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)/math.sqrt(math.pi)
	if (q,l,s) == (2,2,2):
		return (1/7)*math.sqrt(5)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 3*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (2,2,4):
		return (3/7)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 10*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (2,3,1):
		return (3/70)*math.sqrt(105)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) - 2*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (2,3,3):
		return (2/15)*math.sqrt(5)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 3*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (2,4,2):
		return (3/7)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) - 4*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (2,4,4):
		return (10/77)*math.sqrt(5)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 3*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (3,0,3):
		return (1/2)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 12*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (3,1,2):
		return (3/70)*math.sqrt(105)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 8*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (3,1,4):
		return (2/21)*math.sqrt(21)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 15*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (3,2,1):
		return (3/70)*math.sqrt(105)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 4*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (3,2,3):
		return (2/15)*math.sqrt(5)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 9*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (3,3,0):
		return (1/2)*vr**2*RadialPoly(vr, k)*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)/math.sqrt(math.pi)
	if (q,l,s) == (3,3,2):
		return (2/15)*math.sqrt(5)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 3*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (3,3,4):
		return (3/11)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 10*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (3,4,1):
		return (2/21)*math.sqrt(21)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) - 3*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (3,4,3):
		return (3/11)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 2*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (4,0,4):
		return (1/2)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 20*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (4,1,3):
		return (2/21)*math.sqrt(21)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 15*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (4,2,2):
		return (3/7)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 10*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (4,2,4):
		return (10/77)*math.sqrt(5)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 17*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (4,3,1):
		return (2/21)*math.sqrt(21)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 5*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (4,3,3):
		return (3/11)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 10*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (4,4,0):
		return (1/2)*vr**2*RadialPoly(vr, k)*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)/math.sqrt(math.pi)
	if (q,l,s) == (4,4,2):
		return (10/77)*math.sqrt(5)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 3*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (4,4,4):
		return (243/1001)*(vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 10*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/math.sqrt(math.pi)
	return 0


def Ib(RadialPoly, RadialPolyDeriv, vr, p, k, r, q, l, s):
	if (q,l,s) == (0,0,0):
		return (1/4)*(vr**2*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1))*RadialPoly(vr, k)/math.sqrt(math.pi)
	if (q,l,s) == (0,1,1):
		return (1/4)*(vr*(vr**2*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)) - 2*RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr)
	if (q,l,s) == (0,2,2):
		return (1/4)*(vr*(vr**2*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)) - 6*RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr)
	if (q,l,s) == (0,3,3):
		return (1/4)*(vr*(vr**2*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)) - 12*RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr)
	if (q,l,s) == (0,4,4):
		return (1/4)*(vr*(vr**2*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)) - 20*RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr)
	if (q,l,s) == (1,0,1):
		return (1/4)*(vr**2*(vr**2*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 6*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)) - 6*vr*(RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) + RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1)) + 6*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (1,1,0):
		return (1/4)*(vr*(vr**2*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)) - 2*RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr)
	if (q,l,s) == (1,1,2):
		return (1/10)*math.sqrt(5)*(vr**4*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 8*vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) - 8*vr*RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) - 12*vr*RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1) + 12*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (1,2,1):
		return (1/10)*math.sqrt(5)*vr**2*RadialPoly(vr, k)*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2)/math.sqrt(math.pi)
	if (q,l,s) == (1,2,3):
		return (3/140)*math.sqrt(105)*(vr**2*(vr**2*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 10*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)) - 10*vr*(RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) + 2*RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1)) + 20*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (1,3,2):
		return (3/140)*math.sqrt(105)*(vr**2*(vr**2*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) - 2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)) + 2*vr*(RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) - RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1)) + 2*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (1,3,4):
		return (1/21)*math.sqrt(21)*(vr**4*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 12*vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) - 12*vr*RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) - 30*vr*RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1) + 30*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (1,4,3):
		return (1/21)*math.sqrt(21)*(vr**2*(vr**2*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) - 4*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)) + 2*vr*(2*RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) - 3*RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1)) + 6*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (2,0,2):
		return (1/4)*(vr**2*(vr**2*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 14*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)) - 18*vr*(RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) + RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1)) + 42*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (2,1,1):
		return (1/10)*math.sqrt(5)*(vr**4*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 8*vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) - 12*vr*RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) - 8*vr*RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1) + 12*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (2,1,3):
		return (3/140)*math.sqrt(105)*(vr**4*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 18*vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) - 22*vr*RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) - 28*vr*RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1) + 72*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (2,2,0):
		return (1/4)*(vr*(vr**2*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)) - 6*RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr)
	if (q,l,s) == (2,2,2):
		return (1/14)*math.sqrt(5)*(vr**2*(vr**2*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 8*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)) - 12*vr*(RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) + RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1)) + 12*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (2,2,4):
		return (3/14)*(vr**4*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 22*vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) - 26*vr*RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) - 40*vr*RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1) + 110*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (2,3,1):
		return (3/140)*math.sqrt(105)*(vr**2*(vr**2*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) - 2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)) + 2*vr*(-RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) + RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1)) + 2*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (2,3,3):
		return (1/15)*math.sqrt(5)*(vr**4*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 8*vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) - 12*vr*RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) - 18*vr*RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1) + 12*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (2,4,2):
		return (3/14)*(vr**2*(vr**2*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) - 6*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)) + 2*vr*(RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) + RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1)) + 12*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (2,4,4):
		return (5/77)*math.sqrt(5)*(vr**4*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 8*vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) - 12*vr*RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) - 26*vr*RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1) + 12*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (3,0,3):
		return (1/4)*(vr**2*(vr**2*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 26*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)) - 36*vr*(RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) + RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1)) + 156*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (3,1,2):
		return (3/140)*math.sqrt(105)*(vr**4*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 18*vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) - 28*vr*RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) - 22*vr*RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1) + 72*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (3,1,4):
		return (1/21)*math.sqrt(21)*(vr**4*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 32*vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) - 42*vr*RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) - 50*vr*RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1) + 240*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (3,2,1):
		return (3/140)*math.sqrt(105)*(vr**2*(vr**2*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 10*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)) - 10*vr*(2*RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) + RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1)) + 20*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (3,2,3):
		return (1/15)*math.sqrt(5)*(vr**2*(vr**2*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 20*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)) - 30*vr*(RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) + RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1)) + 90*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (3,3,0):
		return (1/4)*(vr*(vr**2*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)) - 12*RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr)
	if (q,l,s) == (3,3,2):
		return (1/15)*math.sqrt(5)*(vr**4*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 8*vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) - 18*vr*RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) - 12*vr*RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1) + 12*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (3,3,4):
		return (3/22)*(vr**4*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 22*vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) - 32*vr*RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) - 40*vr*RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1) + 110*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (3,4,1):
		return (1/21)*math.sqrt(21)*(vr**2*(vr**2*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) - 4*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)) + 2*vr*(-3*RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) + 2*RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1)) + 6*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (3,4,3):
		return (3/22)*(vr**2*(vr**2*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 6*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)) - 16*vr*(RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) + RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1)) + 6*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (4,0,4):
		return (1/4)*(vr**2*(vr**2*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 42*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)) - 60*vr*(RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) + RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1)) + 420*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (4,1,3):
		return (1/21)*math.sqrt(21)*(vr**4*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 32*vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) - 50*vr*RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) - 42*vr*RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1) + 240*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (4,2,2):
		return (3/14)*(vr**4*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 22*vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) - 40*vr*RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) - 26*vr*RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1) + 110*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (4,2,4):
		return (5/77)*math.sqrt(5)*(vr**2*(vr**2*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 36*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)) - 54*vr*(RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) + RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1)) + 306*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (4,3,1):
		return (1/21)*math.sqrt(21)*(vr**4*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 12*vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) - 30*vr*RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) - 12*vr*RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1) + 30*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (4,3,3):
		return (3/22)*(vr**4*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 22*vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) - 40*vr*RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) - 32*vr*RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1) + 110*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (4,4,0):
		return (1/4)*(vr*(vr**2*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)) - 20*RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr)
	if (q,l,s) == (4,4,2):
		return (5/77)*math.sqrt(5)*(vr**4*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 8*vr**2*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) - 26*vr*RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) - 12*vr*RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1) + 12*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (4,4,4):
		return (243/2002)*(vr**2*(vr**2*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + 22*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1)) - 40*vr*(RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) + RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1)) + 110*RadialPoly(vr, p)*RadialPoly(vr, r))*RadialPoly(vr, k)/(math.sqrt(math.pi)*vr**2)
	return 0
