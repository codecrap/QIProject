import numpy as np
from sympy import Symbol, Matrix, init_printing, pprint, sin, cos, simplify, pi
from sympy.physics.quantum import TensorProduct as TP
from sympy.physics.quantum.dagger import Dagger as Dag


b = Symbol('b')
g = Symbol('g')

Id = Matrix([[1,0],
    [0,1]])

Z = Matrix([[1,0],
    [0,-1]])

X = Matrix([[0,1],
    [1,0]])

Pl = Matrix([1,1,1,1,1,1,1,1])/(8**(1/2))

X1 = (b*1j*(TP(Id, TP(X,Id)))).exp()
X0 = (b*1j*TP(X, TP(Id,Id))).exp()
U01 = (g*1j/2*(TP(Id,TP(Id,Id)) - TP(Z,TP(Z,Id)))).exp()
U02 = (g*1j/2*(TP(Id,TP(Id,Id)) - TP(Z,TP(Id,Z)))).exp()
U21 = (g*1j/2*(TP(Id,TP(Id,Id)) - TP(Id,TP(Z,Z)))).exp()

Z0 = TP(Z,TP(Id,Id))
Z1 = TP(Id,TP(Z,Id))
Zf = Z0*Z1


U = U21*U02*U01
Xf = X0*X1

F = U*Xf
FH = Dag(F)

T = F*Zf*FH


RM = Dag(Pl) * T * Pl

R = 0.5*(1-RM[0,0])

r = simplify(R)

f = 0.5*(sin(4*g)*sin(4*b)+sin(2*b)**2*sin(2*g)**2)

print(R.evalf(subs={b:0.2, g:1.7}))

print(f.evalf(subs={b:0.2, g:1.7}))
