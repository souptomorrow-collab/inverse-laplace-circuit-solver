# -*- coding: utf-8 -*-
"""
Validate the s-domain companion models for INITIAL CONDITIONS and MUTUAL
INDUCTANCE that will be added to engine.js. Cross-checked against SymPy /
closed-form analytics.

Companion models (derived):
  Capacitor (+,-), value C, initial voltage v0 = v+(0)-v-(0):
     I_C(s) = sC*V(s) - C*v0   ->  stamp admittance sC ; RHS[+] += C*v0, RHS[-] -= C*v0
  Inductor (+,-), value L, initial current i0 (from + to -):
     I_L(s) = (1/sL)V(s) + i0/s ->  stamp admittance 1/(sL) ; RHS[+] -= i0/s, RHS[-] += i0/s
  Coupled inductors (matrix L, initial current vector i0):
     I(s) = (1/s) L^{-1} V(s) + i0/s  ->  stamp (1/s)L^{-1} as a coupled 2-port ;
     RHS companion = i0/s per inductor (same as uncoupled).
"""
import numpy as np, sympy as sp, math, cmath, random
from proto import inverse_laplace_rational, eval_ft
from proto_circuit import R, ptrim, solve_rational

random.seed(3)
s_sym = sp.symbols('s'); t_sym = sp.symbols('t', positive=True)

# ---------------- generic MNA with IC + mutual ----------------
def mna(nodes, comps, mutuals=None, nodeIC=None):
    """
    comps: list of dicts:
      {type:'R'|'C'|'L', a, b, value, v0?(C), i0?(L)}
      {type:'V', a, b, Vs:R}    {type:'I', a, b, Is:R}
    mutuals: list of (idxLa, idxLb, M)  indices into comps (must be type 'L')
    nodeIC: dict node->initial voltage (used to derive C v0 if not explicit)
    Returns: dict node->R (node voltages), and helper to get inductor current.
    """
    mutuals = mutuals or []
    nodeIC = nodeIC or {}
    nz = sorted(n for n in nodes if n != 0)
    idx = {n: i for i, n in enumerate(nz)}
    nv = len(nz)
    vsrc = [c for c in comps if c['type'] == 'V']
    m = len(vsrc)
    size = nv + m
    A = [[R.const(0) for _ in range(size)] for _ in range(size)]
    b = [R.const(0) for _ in range(size)]
    def addA(i, j, val):
        A[i][j] = A[i][j].add(val)
    def addB(node, val):
        if node != 0:
            b[idx[node]] = b[idx[node]].add(val)
    def stampY(a, bb, Y):
        if a != 0: addA(idx[a], idx[a], Y)
        if bb != 0: addA(idx[bb], idx[bb], Y)
        if a != 0 and bb != 0:
            addA(idx[a], idx[bb], Y.neg()); addA(idx[bb], idx[a], Y.neg())
    def cap_v0(c):
        if 'v0' in c and c['v0'] is not None:
            return c['v0']
        return nodeIC.get(c['a'], 0.0) - nodeIC.get(c['b'], 0.0)

    coupled = set()
    for (ia, ib, M) in mutuals:
        coupled.add(ia); coupled.add(ib)

    # passive stamps
    for k, c in enumerate(comps):
        if c['type'] == 'R':
            stampY(c['a'], c['b'], R.const(1.0 / c['value']))
        elif c['type'] == 'C':
            stampY(c['a'], c['b'], R([0, c['value']]))   # sC
            v0 = cap_v0(c)
            if abs(v0) > 1e-15:
                addB(c['a'], R.const(c['value'] * v0))    # +C*v0 into a
                addB(c['b'], R.const(-c['value'] * v0))
        elif c['type'] == 'L' and k not in coupled:
            stampY(c['a'], c['b'], R([1.0], [0, c['value']]))  # 1/(sL)
            i0 = c.get('i0', 0.0) or 0.0
            if abs(i0) > 1e-15:
                addB(c['a'], R([-i0], [0, 1.0]))   # -i0/s into a
                addB(c['b'], R([i0], [0, 1.0]))
    # mutual coupled stamps
    for (ia, ib, M) in mutuals:
        ca, cb = comps[ia], comps[ib]
        La, Lb = ca['value'], cb['value']
        D = La * Lb - M * M
        # G = L^{-1} = [[Lb,-M],[-M,La]]/D ; admittance block = (1/s) G
        g11, g12, g21, g22 = Lb / D, -M / D, -M / D, La / D
        def Yof(g):
            return R([g], [0, 1.0])  # g/s
        # branch a = (a1,a2), branch b = (b1,b2)
        a1, a2 = ca['a'], ca['b']; b1, b2 = cb['a'], cb['b']
        # self a
        stampY(a1, a2, Yof(g11))
        # self b
        stampY(b1, b2, Yof(g22))
        # cross: Ia depends on Vb (g12), Ib depends on Va (g21)
        def addcross(p, q, val):  # current into p proportional to V at q (with node 0 handling)
            if p != 0 and q != 0:
                addA(idx[p], idx[q], val)
        Y12 = Yof(g12); Y21 = Yof(g21)
        # Ia = ... + g12/s*(Vb1-Vb2): contributes to KCL at a1(+),a2(-)
        addcross(a1, b1, Y12); addcross(a1, b2, Y12.neg())
        addcross(a2, b1, Y12.neg()); addcross(a2, b2, Y12)
        addcross(b1, a1, Y21); addcross(b1, a2, Y21.neg())
        addcross(b2, a1, Y21.neg()); addcross(b2, a2, Y21)
        # IC companion i0/s for coupled inductors
        for c in (ca, cb):
            i0 = c.get('i0', 0.0) or 0.0
            if abs(i0) > 1e-15:
                addB(c['a'], R([-i0], [0, 1.0])); addB(c['b'], R([i0], [0, 1.0]))

    # current sources
    for c in comps:
        if c['type'] == 'I':
            addB(c['a'], c['Is']); addB(c['b'], c['Is'].neg())
    # voltage sources
    for kk, c in enumerate(vsrc):
        row = nv + kk
        if c['a'] != 0: addA(idx[c['a']], row, R.const(1)); addA(row, idx[c['a']], R.const(1))
        if c['b'] != 0: addA(idx[c['b']], row, R.const(-1)); addA(row, idx[c['b']], R.const(-1))
        b[row] = c['Vs']
    x = solve_rational(A, b)
    V = {0: R.const(0)}
    for n, i in idx.items():
        V[n] = x[i]
    V['__Iv'] = {kk: x[nv + kk] for kk in range(m)}  # voltage-source branch currents
    return V

def ratR(r):
    return [c.real for c in r.num], [c.real for c in r.den]

def ft(rnum, rden, tv):
    dl, pf = inverse_laplace_rational(list(map(complex, rnum)), list(map(complex, rden)))
    return eval_ft(dl, pf, tv).real

def check(name, Vr, fn, tvals=(0.2,0.7,1.5,3.0)):
    num, den = ratR(Vr)
    err = 0.0
    for tv in tvals:
        err = max(err, abs(ft(num, den, tv) - fn(tv)))
    print(f"[{'OK' if err<1e-6 else 'FAIL'}] {name:36s} maxerr={err:.2e}")
    return err < 1e-6

print("="*72); print("VALIDATION: initial conditions + mutual inductance"); print("="*72)
ok = True

# 1) RC discharge: C=0.5 v0=2, R=2 across it. no source -> v(t)=2 e^{-t}
V = mna({0,1}, [
    {'type':'C','a':1,'b':0,'value':0.5,'v0':2.0},
    {'type':'R','a':1,'b':0,'value':2.0},
])
ok &= check("RC discharge (cap IC) -> 2e^-t", V[1], lambda t: 2*math.exp(-t))

# 2) RL decay: L=1 i0=3, R=2 across. node voltage = -6 e^{-2t}; check current = 3 e^{-2t}
V = mna({0,1}, [
    {'type':'L','a':1,'b':0,'value':1.0,'i0':3.0},
    {'type':'R','a':1,'b':0,'value':2.0},
])
ok &= check("RL decay node V (ind IC) -> -6e^-2t", V[1], lambda t: -6*math.exp(-2*t))
# inductor current i_L = (1/sL)V + i0/s
iL = R([1.0],[0,1.0]).mul(V[1]).add(R([3.0],[0,1.0]))
ok &= check("RL decay inductor current -> 3e^-2t", iL, lambda t: 3*math.exp(-2*t))

# 3) node IC derives cap IC: same RC discharge via nodeIC instead of explicit v0
V = mna({0,1}, [
    {'type':'C','a':1,'b':0,'value':0.5},
    {'type':'R','a':1,'b':0,'value':2.0},
], nodeIC={1:2.0})
ok &= check("RC discharge via NODE IC -> 2e^-t", V[1], lambda t: 2*math.exp(-t))

# 4) RLC source-free with IC vs sympy:
#    series R-L-C loop, cap IC v0, ind IC i0. node1 - R - node2 - L - node3, C node3-?
#    Use parallel RLC: R,L,C all between node1 and gnd. cap v0=1, ind i0=0.5
Rv,Lv,Cv,v0,i0 = 2.0, 1.0, 0.5, 1.0, 0.5
V = mna({0,1}, [
    {'type':'R','a':1,'b':0,'value':Rv},
    {'type':'L','a':1,'b':0,'value':Lv,'i0':i0},
    {'type':'C','a':1,'b':0,'value':Cv,'v0':v0},
])
# analytic: node eq (1/R + 1/(sL) + sC)V = C*v0 - i0/s   (cap inj +C v0, ind inj -i0/s)
Vexp = (Cv*v0 - i0/s_sym) / (1/Rv + 1/(s_sym*Lv) + s_sym*Cv)
fexp = sp.inverse_laplace_transform(sp.simplify(Vexp), s_sym, t_sym)
num,den = ratR(V[1])
err=0
for tv in [0.2,0.7,1.5,3.0]:
    err=max(err, abs(ft(num,den,tv) - complex(fexp.subs(t_sym,tv).evalf()).real))
print(f"[{'OK' if err<1e-6 else 'FAIL'}] {'parallel RLC source-free (IC)':36s} maxerr={err:.2e}"); ok &= err<1e-6

# 5) MUTUAL: transformer. Vs step at node1; La node1-gnd; secondary Lb node2-gnd, R node2-gnd.
#    Compare V2 to analytic V2 = -(g21/s)Vs / (g22/s + 1/R)
La,Lb,M,Rload = 1.0, 2.0, 1.0, 3.0
Vs = R([1.0],[0,1.0])  # step 1/s
comps = [
    {'type':'V','a':1,'b':0,'Vs':Vs},
    {'type':'L','a':1,'b':0,'value':La},   # idx1 primary
    {'type':'L','a':2,'b':0,'value':Lb},   # idx2 secondary
    {'type':'R','a':2,'b':0,'value':Rload},
]
V = mna({0,1,2}, comps, mutuals=[(1,2,M)])
D = La*Lb - M*M
g21, g22 = -M/D, La/D
V2exp = (-(g21/s_sym)*(1/s_sym)) / (g22/s_sym + 1/Rload)
f2 = sp.inverse_laplace_transform(sp.simplify(V2exp), s_sym, t_sym)
num,den = ratR(V[2])
err=0
for tv in [0.2,0.7,1.5,3.0]:
    err=max(err, abs(ft(num,den,tv) - complex(f2.subs(t_sym,tv).evalf()).real))
print(f"[{'OK' if err<1e-6 else 'FAIL'}] {'mutual transformer V2 vs sympy':36s} maxerr={err:.2e}"); ok &= err<1e-6

# 6) sanity: M=0 should reduce to independent inductors (V2 -> 0 since no primary coupling)
V0 = mna({0,1,2}, comps, mutuals=[(1,2,0.0)])
num,den = ratR(V0[2])
val = ft(num,den,1.0)
print(f"[{'OK' if abs(val)<1e-9 else 'FAIL'}] {'M=0 -> secondary V2=0':36s} v(1)={val:.2e}"); ok &= abs(val)<1e-9

print("="*72); print("ALL OK" if ok else "SOME FAILED"); print("="*72)
