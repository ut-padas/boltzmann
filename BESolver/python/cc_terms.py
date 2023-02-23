## generated code with sympy
## indexed with q-test l mode, l trial mode and s-h or g function modes
import numpy
import math
Ia_nz=[(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
Ib_nz=[(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
def Ia(RadialPoly, RadialPolyDeriv, vr, p, k, r, q, l, s):
	if (q,l,s) == (0,0,0):
		return (1/2)*vr**2*(RadialPoly(vr, k)*RadialPolyDeriv(vr, r, 1) + RadialPoly(vr, r)*RadialPolyDeriv(vr, k, 1))*RadialPolyDeriv(vr, p, 1)/math.sqrt(math.pi)
	if (q,l,s) == (0,1,1):
		return (1/2)*vr**2*(RadialPoly(vr, k)*RadialPolyDeriv(vr, r, 1) + RadialPoly(vr, r)*RadialPolyDeriv(vr, k, 1))*RadialPolyDeriv(vr, p, 1)/math.sqrt(math.pi)
	if (q,l,s) == (1,0,1):
		return (1/2)*(vr**2*RadialPoly(vr, k)*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + vr**2*RadialPoly(vr, r)*RadialPolyDeriv(vr, k, 1)*RadialPolyDeriv(vr, p, 1) + 2*RadialPoly(vr, k)*RadialPoly(vr, p)*RadialPoly(vr, r))/math.sqrt(math.pi)
	if (q,l,s) == (1,1,0):
		return (1/2)*(vr**2*RadialPoly(vr, k)*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + vr**2*RadialPoly(vr, r)*RadialPolyDeriv(vr, k, 1)*RadialPolyDeriv(vr, p, 1) + 2*RadialPoly(vr, k)*RadialPoly(vr, p)*RadialPoly(vr, r))/math.sqrt(math.pi)
	return 0


def Ib(RadialPoly, RadialPolyDeriv, vr, p, k, r, q, l, s):
	if (q,l,s) == (0,0,0):
		return (1/4)*(vr**2*RadialPoly(vr, k)*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + vr**2*RadialPoly(vr, r)*RadialPolyDeriv(vr, k, 2)*RadialPolyDeriv(vr, p, 2) + 2*vr**2*RadialPolyDeriv(vr, k, 1)*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 1) + 2*RadialPoly(vr, k)*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 2*RadialPoly(vr, r)*RadialPolyDeriv(vr, k, 1)*RadialPolyDeriv(vr, p, 1))/math.sqrt(math.pi)
	if (q,l,s) == (0,1,1):
		return (1/4)*(vr**2*RadialPoly(vr, k)*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + vr**2*RadialPoly(vr, r)*RadialPolyDeriv(vr, k, 2)*RadialPolyDeriv(vr, p, 2) + 2*vr**2*RadialPolyDeriv(vr, k, 1)*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 1) + 2*RadialPoly(vr, k)*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 2*RadialPoly(vr, r)*RadialPolyDeriv(vr, k, 1)*RadialPolyDeriv(vr, p, 1))/math.sqrt(math.pi)
	if (q,l,s) == (1,0,1):
		return (1/4)*(vr**2*(vr**2*RadialPoly(vr, k)*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + vr**2*RadialPoly(vr, r)*RadialPolyDeriv(vr, k, 2)*RadialPolyDeriv(vr, p, 2) + 2*vr**2*RadialPolyDeriv(vr, k, 1)*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 1) + 6*RadialPoly(vr, k)*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 6*RadialPoly(vr, r)*RadialPolyDeriv(vr, k, 1)*RadialPolyDeriv(vr, p, 1)) - 6*vr*(RadialPoly(vr, k)*RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) + RadialPoly(vr, k)*RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1) + RadialPoly(vr, p)*RadialPoly(vr, r)*RadialPolyDeriv(vr, k, 1)) + 6*RadialPoly(vr, k)*RadialPoly(vr, p)*RadialPoly(vr, r))/(math.sqrt(math.pi)*vr**2)
	if (q,l,s) == (1,1,0):
		return (1/4)*(vr**2*(vr**2*RadialPoly(vr, k)*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 2) + vr**2*RadialPoly(vr, r)*RadialPolyDeriv(vr, k, 2)*RadialPolyDeriv(vr, p, 2) + 2*vr**2*RadialPolyDeriv(vr, k, 1)*RadialPolyDeriv(vr, p, 2)*RadialPolyDeriv(vr, r, 1) + 6*RadialPoly(vr, k)*RadialPolyDeriv(vr, p, 1)*RadialPolyDeriv(vr, r, 1) + 6*RadialPoly(vr, r)*RadialPolyDeriv(vr, k, 1)*RadialPolyDeriv(vr, p, 1)) - 6*vr*(RadialPoly(vr, k)*RadialPoly(vr, p)*RadialPolyDeriv(vr, r, 1) + RadialPoly(vr, k)*RadialPoly(vr, r)*RadialPolyDeriv(vr, p, 1) + RadialPoly(vr, p)*RadialPoly(vr, r)*RadialPolyDeriv(vr, k, 1)) + 6*RadialPoly(vr, k)*RadialPoly(vr, p)*RadialPoly(vr, r))/(math.sqrt(math.pi)*vr**2)
	return 0
