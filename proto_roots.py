# -*- coding: utf-8 -*-
"""
Validate the Durand-Kerner root finder (to be ported to JS, replacing numpy.roots).
Strategy: implement DK exactly as JS will, compare to numpy.roots, THEN monkeypatch
np.roots := dk and re-run all three validated suites. If they still pass, the JS
port (same DK) is trustworthy.
"""
import numpy as np, cmath

def poly_eval_desc(a, x):
    r = 0j
    for c in a:
        r = r * x + c
    return r

def dk_roots(coeffs_desc, max_iter=600, tol=1e-14):
    """Durand-Kerner (Weierstrass). coeffs_desc: highest degree first (numpy-like)."""
    a = [complex(c) for c in coeffs_desc]
    # strip leading zeros
    while len(a) > 1 and abs(a[0]) < 1e-300:
        a.pop(0)
    n = len(a) - 1
    if n <= 0:
        return []
    # monic
    a = [c / a[0] for c in a]
    # Cauchy bound for initial radius
    R = 1.0 + max(abs(c) for c in a[1:]) if n >= 1 else 1.0
    radius = min(R, 1e6) ** (1.0)
    radius = max(0.1, min(radius, 100.0))
    # initial guesses spread on a circle, off the real axis
    seed = complex(0.4, 0.9)
    roots = []
    cur = complex(1.0, 0.0)
    for k in range(n):
        roots.append(radius * 0.5 * cur)
        cur *= seed
    for it in range(max_iter):
        maxdelta = 0.0
        for i in range(n):
            xi = roots[i]
            num = poly_eval_desc(a, xi)
            den = complex(1.0, 0.0)
            for j in range(n):
                if j != i:
                    den *= (xi - roots[j])
            if abs(den) < 1e-300:
                den = complex(1e-300, 0)
            delta = num / den
            roots[i] = xi - delta
            d = abs(delta)
            if d > maxdelta:
                maxdelta = d
        if maxdelta < tol:
            break
    # optional Newton polish (helps simple roots)
    deriv = [a[k] * (n - k) for k in range(n)]  # derivative coeffs desc, length n
    for i in range(n):
        for _ in range(2):
            p = poly_eval_desc(a, roots[i])
            dp = poly_eval_desc(deriv, roots[i])
            if abs(dp) > 1e-300:
                step = p / dp
                if abs(step) < 1.0:
                    roots[i] -= step
    return roots

# ---------- compare to numpy ----------
def maxmatch_err(r1, r2):
    """max over r1 of min distance to some r2 (greedy match)."""
    r2 = list(r2)
    err = 0.0
    for r in r1:
        best = min(range(len(r2)), key=lambda k: abs(r2[k] - r))
        err = max(err, abs(r2[best] - r))
        r2.pop(best)
    return err

print("="*70); print("Durand-Kerner vs numpy.roots"); print("="*70)
tests = [
    ("(s+2)",        [1, 2]),
    ("s^2+2s+5",     [1, 2, 5]),
    ("(s+1)^2",      [1, 2, 1]),
    ("(s+1)^3",      [1, 3, 3, 1]),
    ("s^2+4",        [1, 0, 4]),
    ("(s^2+1)^2",    np.poly([1j,-1j,1j,-1j]).real.tolist()),
    ("s(s+1)(s+2)",  [1, 3, 2, 0]),
    ("(s+2)^2(s+1)", [1, 5, 8, 4]),
    ("0.5s^3+s^2+s", [0.5, 1, 1, 0]),
    ("high deg",     np.poly([-1,-2,-3,-4+1j,-4-1j,0]).real.tolist()),
    ("(s+1)^4",      [1,4,6,4,1]),
]
worst_simple = 0.0
worst_rep = 0.0
for name, c in tests:
    mine = dk_roots(c)
    ref = list(np.roots(c))
    err = maxmatch_err(mine, ref)
    repeated = len(set(np.round(ref, 3))) < len(ref)
    tag = "rep" if repeated else "simple"
    if repeated: worst_rep = max(worst_rep, err)
    else: worst_simple = max(worst_simple, err)
    print(f"  {name:18s} ({tag:6s}) maxerr={err:.2e}  n={len(c)-1}")
print(f"\nworst simple-root err = {worst_simple:.2e}")
print(f"worst repeated-root err = {worst_rep:.2e}")

# ---------- monkeypatch np.roots and re-run all suites ----------
print("="*70); print("Re-running ALL suites with np.roots := Durand-Kerner"); print("="*70)
_orig = np.roots
def patched_roots(c):
    return np.array(dk_roots(list(np.asarray(c).tolist())), dtype=complex)
np.roots = patched_roots
import importlib
import proto, proto_circuit, proto_parser
importlib.reload(proto)
proto.np.roots = patched_roots
importlib.reload(proto_circuit)
proto_circuit.np.roots = patched_roots
# run proto suite
import runpy
print("--- inverse Laplace suite ---")
runpy.run_path("proto.py", run_name="__main__")
print("--- circuit suite ---")
runpy.run_path("proto_circuit.py", run_name="__main__")
print("--- parser suite ---")
runpy.run_path("proto_parser.py", run_name="__main__")
