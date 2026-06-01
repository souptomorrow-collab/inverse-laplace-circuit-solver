# -*- coding: utf-8 -*-
"""
Prototype + validation of the inverse-Laplace algorithm that will be ported to JS.

Core algorithm (rational F(s) = N(s)/D(s)):
  1. If deg N >= deg D  -> polynomial long division. Quotient -> delta(t) terms.
  2. Find roots of D (numpy.roots) -> poles.
  3. Cluster near-equal roots -> multiplicities.
  4. Partial fractions via LINEAR SYSTEM:
       basis_{j,k}(s) = 1/(s-p_j)^k ,  k=1..m_j
       B_{j,k}(s) = D(s)/(s-p_j)^k   (a polynomial, computed by deflation)
       N(s) = sum c_{j,k} B_{j,k}(s)  -> equate coeffs -> solve M c = n
  5. Inverse Laplace of basis term:
       L^{-1}{1/(s-p)^k} = t^{k-1}/(k-1)! * e^{p t}
     Combine complex-conjugate pairs into damped sinusoids for a real answer.

Validation:
  A) Recombine partial fractions, check == F(s) at random complex points.
  B) Evaluate f(t) numerically and compare against sympy's inverse_laplace_transform.
"""
import numpy as np
import sympy as sp
import math, cmath, random

random.seed(12345)
np.random.seed(12345)

# ---------- polynomial helpers (ascending coeff: c[i] is coeff of s^i) ----------
def poly_mul(a, b):
    out = [0j] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    return out

def poly_eval(coeffs, x):
    # coeffs ascending
    r = 0j
    for c in reversed(coeffs):
        r = r * x + c
    return r

def poly_trim(a, tol=1e-9):
    a = list(a)
    while len(a) > 1 and abs(a[-1]) < tol:
        a.pop()
    return a

def poly_deriv(a):
    """derivative of ascending-coeff polynomial."""
    if len(a) <= 1:
        return [0j]
    return [a[i] * i for i in range(1, len(a))]

def refine_root(poly_asc, p, m):
    """Newton-refine a root of multiplicity m: it is a SIMPLE root of poly^(m-1),
    so Newton converges quadratically to machine precision."""
    g = poly_asc
    for _ in range(m - 1):
        g = poly_deriv(g)
    gp = poly_deriv(g)
    for _ in range(12):
        gv = poly_eval(g, p); gpv = poly_eval(gp, p)
        if abs(gpv) < 1e-300:
            break
        step = gv / gpv; p = p - step
        if abs(step) < 1e-15 * (1 + abs(p)):
            break
    return p

def deflate(coeffs_desc, root):
    """Divide polynomial (descending coeffs) by (s - root). Returns (quotient_desc, remainder)."""
    n = len(coeffs_desc)
    q = [0j] * (n - 1)
    rem = 0j
    prev = 0j
    for i in range(n):
        cur = coeffs_desc[i] + prev * root
        if i < n - 1:
            q[i] = cur
            prev = cur
        else:
            rem = cur
    return q, rem

# ---------- main algorithm ----------
def cluster_roots(roots, tol=8e-3):
    """Group near-equal roots (relative tol). Representative = cluster mean
    (errors of a multiple root are ~symmetric, so the mean is accurate even
    when individual DK estimates spread by ~eps^(1/m))."""
    used = [False] * len(roots)
    groups = []
    for i in range(len(roots)):
        if used[i]:
            continue
        cluster = [roots[i]]
        used[i] = True
        for j in range(i + 1, len(roots)):
            if not used[j] and abs(roots[j] - roots[i]) < tol * (1 + abs(roots[i])):
                cluster.append(roots[j])
                used[j] = True
        rep = sum(cluster) / len(cluster)
        groups.append((rep, len(cluster)))
    return groups

def partial_fractions(N_asc, D_asc):
    """
    N_asc, D_asc: ascending complex coeff lists, deg N < deg D.
    Returns list of dicts: {'pole': p, 'order': k, 'coeff': c}
    """
    D_desc = list(reversed(D_asc))
    roots = np.roots(D_desc)
    groups = cluster_roots(list(roots))

    # refine each pole to machine precision (Newton on D^(m-1))
    groups = [(refine_root(D_asc, p, m), m) for (p, m) in groups]

    # build basis polynomials B_{j,k} = D(s)/(s-p_j)^k  (ascending)
    basis = []  # (pole, k, Bcoeffs_asc)
    for (p, m) in groups:
        # repeatedly deflate D by (s - p)
        cur_desc = list(D_desc)
        for k in range(1, m + 1):
            q_desc, rem = deflate(cur_desc, p)
            cur_desc = q_desc  # D/(s-p)^k
            Bk_asc = list(reversed(cur_desc))
            basis.append((p, k, Bk_asc))

    degD = len(D_asc) - 1
    # Matrix: rows = power 0..degD-1, cols = each basis poly coeff at that power
    M = np.zeros((degD, len(basis)), dtype=complex)
    for col, (_, _, Bk) in enumerate(basis):
        for power, coeff in enumerate(Bk):
            if power < degD:
                M[power, col] = coeff
    rhs = np.zeros(degD, dtype=complex)
    for power, coeff in enumerate(N_asc):
        if power < degD:
            rhs[power] = coeff
    c = np.linalg.solve(M, rhs)
    out = []
    for (p, k, _), ci in zip(basis, c):
        out.append({'pole': p, 'order': k, 'coeff': ci})
    return out

def deflate_asc(coeffs_asc, root):
    """Divide ascending-coeff polynomial by (s - root). Returns quotient (ascending)."""
    desc = list(reversed(coeffs_asc))
    q, _rem = deflate(desc, root)
    return list(reversed(q))

def _rebuild_from_groups(groups, lead):
    """groups: list of (rep, mult) with mult>0. Returns ascending coeffs lead*prod(s-rep)^mult."""
    poly = [1.0 + 0j]
    for (rep, m) in groups:
        for _ in range(m):
            poly = poly_mul(poly, [-rep, 1.0 + 0j])
    return [c * lead for c in poly]

def reduce_rational(num, den, tol=8e-3):
    """Pole-zero cancellation robust to repeated roots: cluster num-roots and
    den-roots SEPARATELY (generous tol, mean reps), cancel min(multiplicity) at
    each matched location, rebuild with clean repeated factors. If nothing
    cancels, keep the exact original coeffs (monic-normalized). Ascending in/out."""
    num = poly_trim(num); den = poly_trim(den)
    leadN = num[-1]; leadD = den[-1]
    if abs(leadN) < 1e-14:
        return [0j], [1j]
    nroots = [complex(r) for r in np.roots(list(reversed(num)))] if len(num) > 1 else []
    droots = [complex(r) for r in np.roots(list(reversed(den)))] if len(den) > 1 else []
    # refine each cluster representative on the ORIGINAL polynomial so repeated
    # factors are rebuilt at machine precision (not the DK-spread cluster mean).
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
        return poly_trim([c / leadD for c in num]), poly_trim([c / leadD for c in den])
    numpoly = _rebuild_from_groups([(r, m) for (r, m) in new_n if m > 0], leadN)
    denpoly = _rebuild_from_groups([(g[0], g[1]) for g in dgroups if g[1] > 0], leadD)
    ld = denpoly[-1]
    return poly_trim([c / ld for c in numpoly]), poly_trim([c / ld for c in denpoly])

def inverse_laplace_rational(N_asc, D_asc, reduce=True):
    """Returns (poly_part_desc_for_delta, terms) where terms describe f(t)."""
    N_asc = poly_trim(N_asc)
    D_asc = poly_trim(D_asc)
    if reduce:
        N_asc, D_asc = reduce_rational(N_asc, D_asc)
    degN = len(N_asc) - 1
    degD = len(D_asc) - 1
    delta_part = []  # coefficients [q0, q1, ...] -> q0*delta + q1*delta' + ...
    if degN >= degD:
        # long division: N/D = Q + R/D  (work in descending)
        Nd = list(reversed(N_asc))
        Dd = list(reversed(D_asc))
        q = [0j] * (len(Nd) - len(Dd) + 1)
        rem = list(Nd)
        for i in range(len(q)):
            coef = rem[i] / Dd[0]
            q[i] = coef
            for j in range(len(Dd)):
                rem[i + j] -= coef * Dd[j]
        # remainder ascending
        R_asc = poly_trim(list(reversed(rem)))
        delta_part = list(reversed(q))  # ascending: q0*s^0 ... actually q is descending
        # q descending -> ascending delta coeffs
        delta_part = list(reversed(q))
        N_asc = R_asc
        degN = len(N_asc) - 1
        if len(N_asc) == 1 and abs(N_asc[0]) < 1e-12:
            pf = []
        else:
            pf = partial_fractions(N_asc, D_asc)
    else:
        pf = partial_fractions(N_asc, D_asc)
    return delta_part, pf

def eval_ft(delta_part, pf, t):
    """Numeric f(t) for t>0 (ignores delta at t=0, t>0)."""
    val = 0j
    for term in pf:
        p = term['pole']
        k = term['order']
        c = term['coeff']
        val += c * (t ** (k - 1)) / math.factorial(k - 1) * cmath.exp(p * t)
    return val

# ---------- validation A: recombination ----------
def recombine_check(N_asc, D_asc, pf, delta_part):
    ok = True
    for _ in range(8):
        s = complex(random.uniform(-3, 3), random.uniform(-3, 3))
        # avoid being near a pole
        Dval = poly_eval(D_asc, s)
        if abs(Dval) < 1e-3:
            continue
        Fval = poly_eval(N_asc, s) / Dval
        rec = 0j
        for term in pf:
            rec += term['coeff'] / ((s - term['pole']) ** term['order'])
        for power, q in enumerate(delta_part):
            rec += q * (s ** power)
        if abs(Fval - rec) > 1e-6 * (1 + abs(Fval)):
            ok = False
    return ok

# ---------- validation B: compare to sympy ----------
s_sym, t_sym = sp.symbols('s t', positive=True)

def sympy_ft(N_coeffs_asc, D_coeffs_asc, t_val):
    N = sum(sp.nsimplify(c.real) * s_sym**i for i, c in enumerate(N_coeffs_asc))
    D = sum(sp.nsimplify(c.real) * s_sym**i for i, c in enumerate(D_coeffs_asc))
    F = N / D
    f = sp.inverse_laplace_transform(F, s_sym, t_sym)
    return complex(f.subs(t_sym, t_val).evalf())

# ---------- test cases: (name, N_asc(real), D_asc(real)) ----------
def from_factors(poles_with_mult, numer_asc):
    """Build D from poles, return ascending real coeffs."""
    D = [1.0]
    for (p, m) in poles_with_mult:
        for _ in range(m):
            D = poly_mul(D, [-p, 1.0])  # (s - p)
    D = [c.real for c in D]
    return numer_asc, D

cases = []
# 1/(s+2)
cases.append(("1/(s+2)", [1.0], [2.0, 1.0]))
# s/(s^2+4)  -> cos(2t)
cases.append(("s/(s^2+4)", [0.0, 1.0], [4.0, 0.0, 1.0]))
# 2/(s^2+4) -> sin(2t)
cases.append(("2/(s^2+4)", [2.0], [4.0, 0.0, 1.0]))
# (s+3)/(s^2+2s+5) damped sinusoid
cases.append(("(s+3)/(s^2+2s+5)", [3.0, 1.0], [5.0, 2.0, 1.0]))
# 1/(s+1)^2 -> t e^{-t}
cases.append(("1/(s+1)^2", [1.0], [1.0, 2.0, 1.0]))
# 1/(s(s+1)) -> 1 - e^{-t}
cases.append(("1/(s(s+1))", [1.0], [0.0, 1.0, 1.0]))
# (2s+1)/((s+1)(s+2)^2) repeated
n,d = from_factors([(-1,1),(-2,2)], [1.0, 2.0])
cases.append(("(2s+1)/((s+1)(s+2)^2)", n, d))
# 1/s^3 -> t^2/2
cases.append(("1/s^3", [1.0], [0.0,0.0,0.0,1.0]))
# improper: (s^2+1)/(s+1) -> delta'+... let sympy handle
cases.append(("(s^2+1)/(s+1)", [1.0,0.0,1.0], [1.0,1.0]))
# 6/(s^2+9) -> 2 sin(3t)
cases.append(("6/(s^2+9)", [6.0], [9.0,0.0,1.0]))
# 1/((s^2+1)^2) repeated complex poles
n2 = [1.0]
d2 = poly_mul([1.0,0.0,1.0],[1.0,0.0,1.0])
d2 = [c.real for c in d2]
cases.append(("1/(s^2+1)^2", n2, d2))
# (s+1)/(s^2(s+2))
n,d = from_factors([(0,2),(-2,1)], [1.0,1.0])
cases.append(("(s+1)/(s^2(s+2))", n, d))
# STRESS: higher multiplicities at shifted locations
n,d = from_factors([(-2,3)], [1.0])              # 1/(s+2)^3
cases.append(("1/(s+2)^3", n, d))
n,d = from_factors([(-3,4)], [3.0, 1.0])         # (s+3)/(s+3)^4 style -> (s+3) in num? use (s+1)/(s+3)^4
n,d = from_factors([(-3,4)], [1.0, 1.0])         # (s+1)/(s+3)^4
cases.append(("(s+1)/(s+3)^4", n, d))
n,d = from_factors([(-1,3),(-2,1)], [2.0,1.0])   # (s+2)/((s+1)^3(s+2))
cases.append(("(s+2)/((s+1)^3(s+2))", n, d))
# repeated complex: (s+1)/((s^2+2s+2)^2)
dc = poly_mul([2.0,2.0,1.0],[2.0,2.0,1.0]); dc=[c.real for c in dc]
cases.append(("(s+1)/(s^2+2s+2)^2", [1.0,1.0], dc))

print("=" * 70)
print("VALIDATION: my algorithm vs sympy")
print("=" * 70)
all_ok = True
for (name, N, D) in cases:
    delta_part, pf = inverse_laplace_rational(list(map(complex,N)), list(map(complex,D)))
    rc = recombine_check(list(map(complex,N)), list(map(complex,D)), pf, delta_part)
    # numeric compare at several t
    has_delta = any(abs(q) > 1e-9 for q in delta_part)
    maxerr = 0.0
    tvals = [0.3, 0.8, 1.5, 2.7, 4.0]
    for tv in tvals:
        mine = eval_ft(delta_part, pf, tv).real
        try:
            ref = sympy_ft(N, D, tv).real
            err = abs(mine - ref)
            maxerr = max(maxerr, err)
        except Exception as e:
            ref = float('nan')
    status = "OK" if (rc and (maxerr < 1e-5)) else "FAIL"
    if status != "OK":
        all_ok = False
    poles_str = ", ".join(f"{t['pole']:.3g}^{t['order']}(c={t['coeff']:.3g})" for t in pf)
    print(f"[{status}] {name:28s} recombine={rc} maxerr_vs_sympy={maxerr:.2e} delta={has_delta}")

print("=" * 70)
print("ALL OK" if all_ok else "SOME FAILED")
print("=" * 70)
