# -*- coding: utf-8 -*-
"""
Prototype + validation of the s-domain Modified Nodal Analysis (MNA) solver
that will be ported to JS. Uses a hand-rolled RationalFunction-in-s class
(mirroring the JS port) so we validate the ALGORITHM, then cross-check the
resulting transfer function / response against sympy.
"""
import numpy as np
import sympy as sp
import math, cmath, random
from proto import inverse_laplace_rational, eval_ft, poly_mul, cluster_roots, _rebuild_from_groups, refine_root

# ---------------- Rational function in s: num/den (ascending complex coeffs) -------------
def ptrim(a):
    a = list(a)
    while len(a) > 1 and abs(a[-1]) < 1e-12:
        a.pop()
    return a

def padd(a, b):
    n = max(len(a), len(b))
    return [ (a[i] if i < len(a) else 0j) + (b[i] if i < len(b) else 0j) for i in range(n) ]

def pscale(a, k):
    return [c * k for c in a]

def _roots_of(coeffs_asc):
    if len(coeffs_asc) <= 1:
        return []
    return [complex(r) for r in np.roots(list(reversed(coeffs_asc)))]

def _deflate_asc(coeffs_asc, root):
    desc = list(reversed(coeffs_asc))
    n = len(desc)
    q = [0j] * (n - 1)
    prev = 0j
    for i in range(n - 1):
        cur = desc[i] + prev * root
        q[i] = cur; prev = cur
    return list(reversed(q))

def _reduce(num, den, tol=8e-3):
    """Robust pole-zero cancellation (cluster num & den roots separately, cancel
    min multiplicity at matched locations, rebuild clean). Ascending in/out."""
    num = ptrim(num); den = ptrim(den)
    leadN = num[-1] if num else 0j
    if abs(leadN) < 1e-13:
        return [0j], [1j]
    leadD = den[-1]
    nroots = _roots_of(num)
    droots = _roots_of(den)
    ngroups = [(refine_root(num, rep, m), m) for (rep, m) in cluster_roots(nroots)] if nroots else []
    dgroups = [[refine_root(den, rep, m), m] for (rep, m) in cluster_roots(droots)] if droots else []
    total_cancel = 0
    new_n = []
    for (rep, mult) in ngroups:
        matched = -1
        for j in range(len(dgroups)):
            if dgroups[j][1] > 0 and abs(dgroups[j][0] - rep) < tol * (1 + abs(rep)):
                matched = j; break
        if matched >= 0:
            cancel = min(mult, dgroups[matched][1])
            total_cancel += cancel
            new_n.append((rep, mult - cancel))
            dgroups[matched][1] -= cancel
        else:
            new_n.append((rep, mult))
    if total_cancel == 0:
        return ptrim([c / leadD for c in num]), ptrim([c / leadD for c in den])
    numpoly = _rebuild_from_groups([(r, m) for (r, m) in new_n if m > 0], leadN)
    denpoly = _rebuild_from_groups([(g[0], g[1]) for g in dgroups if g[1] > 0], leadD)
    ld = denpoly[-1]
    return ptrim([c / ld for c in numpoly]), ptrim([c / ld for c in denpoly])

class R:
    def __init__(self, num, den=None, reduce=True):
        num = ptrim([complex(c) for c in num])
        den = ptrim([complex(c) for c in (den if den is not None else [1])])
        if reduce:
            num, den = _reduce(num, den)
        self.num = num
        self.den = den
    @staticmethod
    def const(c):
        return R([c], [1])
    @staticmethod
    def zero():
        return R([0], [1])
    def add(self, o):
        # a/b + c/d = (a d + c b)/(b d)
        num = padd(poly_mul(self.num, o.den), poly_mul(o.num, self.den))
        den = poly_mul(self.den, o.den)
        return R(num, den)
    def sub(self, o):
        return self.add(R(pscale(o.num, -1), o.den))
    def mul(self, o):
        return R(poly_mul(self.num, o.num), poly_mul(self.den, o.den))
    def div(self, o):
        return R(poly_mul(self.num, o.den), poly_mul(self.den, o.num))
    def neg(self):
        return R(pscale(self.num, -1), self.den)
    def is_zero(self):
        return all(abs(c) < 1e-12 for c in self.num)
    def eval(self, x):
        def pe(c, x):
            r = 0j
            for v in reversed(c):
                r = r * x + v
            return r
        return pe(self.num, x) / pe(self.den, x)

# ---------------- Gaussian elimination over rational functions -------------
def solve_rational(A, b):
    n = len(A)
    # build augmented
    M = [[A[i][j] for j in range(n)] + [b[i]] for i in range(n)]
    for col in range(n):
        # pivot: pick a row >= col whose entry is non-zero (prefer simplest by evaluating at random point)
        piv = None
        for r in range(col, n):
            if not M[r][col].is_zero():
                piv = r
                break
        if piv is None:
            raise ValueError("singular matrix")
        M[col], M[piv] = M[piv], M[col]
        pivval = M[col][col]
        # normalize row
        M[col] = [x.div(pivval) for x in M[col]]
        # eliminate
        for r in range(n):
            if r != col and not M[r][col].is_zero():
                factor = M[r][col]
                M[r] = [ M[r][k].sub(factor.mul(M[col][k])) for k in range(n + 1) ]
    return [M[i][n] for i in range(n)]

# ---------------- MNA builder -------------
class Circuit:
    def __init__(self):
        self.nodes = set([0])
        self.comps = []  # (type, a, b, value-as-R-admittance-or-source)
        self.vsrc = []   # (a,b,Vs as R)
        self.isrc = []   # (a,b,Is as R)  current from b->a (injected into a)
    def node(self, n):
        self.nodes.add(n); return n
    def R_(self, a, b, val):  # resistor
        self.comps.append(('Y', a, b, R.const(1.0 / val))); self.nodes.update([a,b])
    def C_(self, a, b, val):  # capacitor admittance sC
        self.comps.append(('Y', a, b, R([0, val]))); self.nodes.update([a,b])
    def L_(self, a, b, val):  # inductor admittance 1/(sL)
        self.comps.append(('Y', a, b, R([1.0], [0, val]))); self.nodes.update([a,b])
    def V_(self, a, b, Vs):   # voltage source a(+) b(-), Vs is R
        self.vsrc.append((a, b, Vs)); self.nodes.update([a,b])
    def I_(self, a, b, Is):   # current source, current Is injected into a, out of b
        self.isrc.append((a, b, Is)); self.nodes.update([a,b])

    def solve(self):
        nn = sorted(self.nodes)
        # map non-ground nodes to indices
        idx = {n: i for i, n in enumerate(n_ for n_ in nn if n_ != 0)}
        nv = len(idx)
        m = len(self.vsrc)
        size = nv + m
        A = [[R.zero() for _ in range(size)] for _ in range(size)]
        b = [R.zero() for _ in range(size)]
        def addA(i, j, val):
            A[i][j] = A[i][j].add(val)
        # stamp admittances
        for (t, a, bb, Y) in self.comps:
            if a != 0:
                ia = idx[a]; addA(ia, ia, Y)
            if bb != 0:
                ib = idx[bb]; addA(ib, ib, Y)
            if a != 0 and bb != 0:
                ia = idx[a]; ib = idx[bb]
                addA(ia, ib, Y.neg()); addA(ib, ia, Y.neg())
        # current sources
        for (a, bb, Is) in self.isrc:
            if a != 0: b[idx[a]] = b[idx[a]].add(Is)
            if bb != 0: b[idx[bb]] = b[idx[bb]].sub(Is)
        # voltage sources
        for k, (a, bb, Vs) in enumerate(self.vsrc):
            row = nv + k
            if a != 0:
                addA(idx[a], row, R.const(1)); addA(row, idx[a], R.const(1))
            if bb != 0:
                addA(idx[bb], row, R.const(-1)); addA(row, idx[bb], R.const(-1))
            b[row] = Vs
        x = solve_rational(A, b)
        # node voltages
        Vnode = {0: R.zero()}
        for n, i in idx.items():
            Vnode[n] = x[i]
        return Vnode

# ---------------- pole-zero cancellation / reduce to F(s) -------------
def ratfunc_to_NDreal(rf):
    # return ascending real coeffs (num, den), normalized so it's clean-ish
    num = [c.real for c in rf.num]
    den = [c.real for c in rf.den]
    return num, den

# ================= TESTS =================
s = sp.symbols('s')
t = sp.symbols('t', positive=True)

def peval_real(c, x):
    r = 0j
    for v in reversed(c):
        r = r * x + v
    return r

def compare_ft(name, num, den, sympy_expr, debug=False):
    # (1) numeric F(s) check: MNA result vs expected, at random complex points
    Ffun = sp.lambdify(s, sympy_expr, 'numpy')
    fserr = 0.0
    for _ in range(10):
        sv = complex(random.uniform(0.5, 3), random.uniform(-2, 2))
        mine = peval_real(num, sv) / peval_real(den, sv)
        ref = complex(Ffun(sv))
        fserr = max(fserr, abs(mine - ref) / (1 + abs(ref)))
    # (2) time-domain check vs sympy inverse transform
    delta, pf = inverse_laplace_rational(list(map(complex, num)), list(map(complex, den)))
    f_sym = sp.inverse_laplace_transform(sympy_expr, s, t)
    maxerr = 0.0
    for tv in [0.2, 0.7, 1.3, 2.5, 4.2]:
        mine = eval_ft(delta, pf, tv).real
        ref = complex(f_sym.subs(t, tv).evalf()).real
        maxerr = max(maxerr, abs(mine - ref))
    if debug:
        print("   raw num:", [round(c,6) for c in num])
        print("   raw den:", [round(c,6) for c in den])
    status = "OK" if (maxerr < 1e-5 and fserr < 1e-6) else "FAIL"
    print(f"[{status}] {name:34s} F(s)_numerr={fserr:.2e}  maxerr_ft={maxerr:.2e}")
    return status == "OK"

print("=" * 78)
print("VALIDATION: MNA circuit solver vs sympy")
print("=" * 78)
all_ok = True

# --- Test 1: RC low-pass, Vin step across series R then C to ground. Output = Vc.
# Vin(node1) --R-- node2 --C-- gnd. Vin = step (1/s). R=2, C=0.5 -> RC=1
c = Circuit()
c.V_(1, 0, R([1.0], [0, 1.0]))   # 1/s  step
c.R_(1, 2, 2.0)
c.C_(2, 0, 0.5)
V = c.solve()
num, den = ratfunc_to_NDreal(V[2])
# expected Vc(s) = (1/s)*(1/(1+sRC)) = 1/(s(1+s)) since RC=1
F_exp = (sp.Rational(1)/s) * (1/(1 + s*2*sp.Rational(1,2)))
all_ok &= compare_ft("RC lowpass Vc (step)", num, den, F_exp)

# --- Test 2: series RLC, output across C. Vin step. R=2,L=1,C=0.5
# node1 --R-- node2 --L-- node3 --C-- gnd ; Vin at node1
c = Circuit()
c.V_(1, 0, R([1.0], [0, 1.0]))
c.R_(1, 2, 2.0)
c.L_(2, 3, 1.0)
c.C_(3, 0, 0.5)
V = c.solve()
num, den = ratfunc_to_NDreal(V[3])
# H(s)=Vc/Vin = (1/sC)/(R+sL+1/sC) = 1/(s^2 LC + sRC +1)
LC=1*0.5; RC=2*0.5
F_exp = (sp.Rational(1)/s) * (1/(s**2*sp.Rational(1,2) + s*sp.Rational(1) + 1))
all_ok &= compare_ft("RLC series Vc (step)", num, den, F_exp, debug=True)

# --- Test 3: current source into RC parallel. Is impulse=1 into node1, R||C to gnd.
# R=4, C=0.25 -> tau=1. Vnode = Is * (R || 1/sC) = Is * R/(1+sRC)
c = Circuit()
c.I_(1, 0, R.const(1.0))         # impulse current = 1
c.R_(1, 0, 4.0)
c.C_(1, 0, 0.25)
V = c.solve()
num, den = ratfunc_to_NDreal(V[1])
F_exp = 1 * (4/(1 + s*4*sp.Rational(1,4)))
all_ok &= compare_ft("RC parallel V (impulse I)", num, den, F_exp)

# --- Test 4: voltage divider two resistors (sanity, no dynamics). Vin step, R1,R2
c = Circuit()
c.V_(1,0, R([1.0],[0,1.0]))
c.R_(1,2,3.0)
c.R_(2,0,1.0)
V=c.solve()
num,den = ratfunc_to_NDreal(V[2])
F_exp = (sp.Rational(1)/s)*(sp.Rational(1)/(3+1))
all_ok &= compare_ft("Resistive divider V2 (step)", num, den, F_exp)

# --- Test 5: RL circuit current. Vin step, R series L, output = current through L
# = node current. We'll measure Vnode across L? Instead measure inductor current via (V1-V2)/...
# Simpler: series R-L to ground, output voltage across L. node1 --R-- node2 --L-- gnd
c = Circuit()
c.V_(1,0, R([1.0],[0,1.0]))
c.R_(1,2,2.0)
c.L_(2,0,1.0)
V=c.solve()
num,den = ratfunc_to_NDreal(V[2])
# VL = Vin * sL/(R+sL) = (1/s)*s/(2+s) = 1/(s+2)
F_exp = (sp.Rational(1)/s)*(s*1/(2 + s*1))
all_ok &= compare_ft("RL series VL (step)", num, den, F_exp)

print("=" * 78)
print("ALL OK" if all_ok else "SOME FAILED")
print("=" * 78)
