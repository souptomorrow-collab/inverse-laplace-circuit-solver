# -*- coding: utf-8 -*-
"""
Validate the PIECEWISE-LAPLACE framework (basis for switches & diodes):
solve one LTI interval, capture each capacitor voltage and inductor current at
the boundary time t_s, then re-solve the next interval (possibly different
topology) using those captured values as initial conditions. The full response
is the concatenation of the (time-shifted) per-interval solutions.
"""
import math, sympy as sp
from proto import inverse_laplace_rational, eval_ft
from proto_circuit import R
from proto_ic_mutual import mna, ratR, ft

def cap_voltage_fn(V, c):
    """time function of capacitor voltage v_a - v_b."""
    Vab = V[c['a']].sub(V[c['b']])
    num, den = ratR(Vab)
    return lambda t: ft(num, den, t)

def ind_current_fn(V, c):
    """time function of inductor current (a->b): (1/sL)(Va-Vb) + i0/s."""
    i0 = c.get('i0', 0.0) or 0.0
    Vab = V[c['a']].sub(V[c['b']])
    I = R([1.0], [0, c['value']]).mul(Vab).add(R([i0], [0, 1.0]))
    num, den = ratR(I)
    return lambda t: ft(num, den, t)

def node_voltage_fn(V, node):
    num, den = ratR(V[node])
    return lambda t: ft(num, den, t)

# ---------------- two-phase switched RC ----------------
# Phase1 [0,ts): source step V=1, R1=1 (1-2), C=1 (2-0). switch (R2 branch) OPEN.
# Phase2 [ts,inf): switch CLOSED -> R2=1 (2-0) added. cap continues from v_C(ts).
print("="*72); print("VALIDATION: piecewise-Laplace switch (IC hand-off)"); print("="*72)
ts = 1.0
Vs = R([1.0], [0, 1.0])  # unit step
nodes = {0, 1, 2}

# phase 1
comps1 = [
    {'type':'V','a':1,'b':0,'Vs':Vs},
    {'type':'R','a':1,'b':2,'value':1.0},
    {'type':'C','a':2,'b':0,'value':1.0,'v0':0.0},
]
V1 = mna(nodes, comps1)
vC1 = cap_voltage_fn(V1, comps1[2])
vc_at_ts = vC1(ts)
print(f"phase1 v_C(ts=1) = {vc_at_ts:.6f}   (analytic 1-e^-1 = {1-math.exp(-1):.6f})")

# phase 2 (switch closed: add R2), cap IC = captured value
comps2 = [
    {'type':'V','a':1,'b':0,'Vs':Vs},
    {'type':'R','a':1,'b':2,'value':1.0},
    {'type':'C','a':2,'b':0,'value':1.0,'v0':vc_at_ts},
    {'type':'R','a':2,'b':0,'value':1.0},   # switch closed
]
V2 = mna(nodes, comps2)
vC2 = cap_voltage_fn(V2, comps2[2])

def piecewise(t):
    return vC1(t) if t < ts else vC2(t - ts)

# analytic: phase1 1-e^-t ; phase2 0.5 + (v(ts)-0.5) e^{-2(t-ts)}
def analytic(t):
    if t < ts: return 1 - math.exp(-t)
    return 0.5 + (1 - math.exp(-1) - 0.5) * math.exp(-2 * (t - ts))

err = 0.0
for t in [0.3, 0.7, 0.99, 1.01, 1.5, 2.5, 4.0]:
    err = max(err, abs(piecewise(t) - analytic(t)))
print(f"[{'OK' if err<1e-6 else 'FAIL'}] switched RC piecewise vs analytic   maxerr={err:.2e}")

# ---------------- switched RL: capture inductor current across boundary ----------------
# Phase1 [0,ts): V=1 step, R1=1 (1-2), L=1 (2-0). i_L charges: i_L=1-e^-t (since R=1,L=1).
# Phase2 [ts): source shorted (switch disconnects source, node1->gnd via wire); L decays through R1.
# Model phase2: remove source, tie node1 to gnd (switch), so R1 from gnd to node2, L node2-gnd.
ts2 = 1.0
comps1b = [
    {'type':'V','a':1,'b':0,'Vs':Vs},
    {'type':'R','a':1,'b':2,'value':1.0},
    {'type':'L','a':2,'b':0,'value':1.0,'i0':0.0},
]
V1b = mna({0,1,2}, comps1b)
iL1 = ind_current_fn(V1b, comps1b[2])
iL_at = iL1(ts2)
print(f"phase1 i_L(ts=1) = {iL_at:.6f}   (analytic 1-e^-1 = {1-math.exp(-1):.6f})")
# phase2: source removed, node1 shorted to gnd (switch). R1 from gnd(1=0) to node2 -> R1 2-0. L 2-0 with i0.
comps2b = [
    {'type':'R','a':2,'b':0,'value':1.0},
    {'type':'L','a':2,'b':0,'value':1.0,'i0':iL_at},
]
V2b = mna({0,2}, comps2b)
iL2 = ind_current_fn(V2b, comps2b[1])
def iLpiece(t):
    return iL1(t) if t < ts2 else iL2(t-ts2)
def iLanalytic(t):
    if t < ts2: return 1-math.exp(-t)
    return (1-math.exp(-1))*math.exp(-(t-ts2))   # decay through R=1,L=1 -> tau=1
err2=0
for t in [0.3,0.99,1.01,1.5,3.0]:
    err2=max(err2, abs(iLpiece(t)-iLanalytic(t)))
print(f"[{'OK' if err2<1e-6 else 'FAIL'}] switched RL inductor current piecewise  maxerr={err2:.2e}")

print("="*72)
print("ALL OK" if (err<1e-6 and err2<1e-6) else "SOME FAILED")
print("="*72)
