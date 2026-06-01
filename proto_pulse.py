# -*- coding: utf-8 -*-
"""
Validate the PULSE source via LTI superposition of shifted step responses:
  pulse(t) = A * sum_n [ u(t-nTs) - u(t-nTs-D*Ts) ]
  output(t) = A * sum_n [ g(t-nTs) - g(t-nTs-D*Ts) ],  g = unit-step response.
Reference: direct numerical integration of the RC ODE driven by the square wave.
"""
import math
from proto import inverse_laplace_rational, eval_ft
from proto_circuit import R
from proto_ic_mutual import mna, ratR, ft

# RC lowpass: Vin - R(1) - node2 - C(1) - gnd. H(s)=Vc/Vin = 1/(1+s). Step response g=1-e^-t.
Vstep = R([1.0], [0, 1.0])
V = mna({0, 1, 2}, [
    {'type': 'V', 'a': 1, 'b': 0, 'Vs': Vstep},
    {'type': 'R', 'a': 1, 'b': 2, 'value': 1.0},
    {'type': 'C', 'a': 2, 'b': 0, 'value': 1.0},
])
gn, gd = ratR(V[2])
g = lambda t: ft(gn, gd, t) if t >= 0 else 0.0   # unit-step response

A, Ts, D = 1.0, 2.0, 0.5     # square wave: on 1s, off 1s, period 2
def pulse_response(t, horizon):
    v = 0.0
    n = 0
    while n * Ts <= t + 1e-12:
        t1 = n * Ts; t2 = n * Ts + D * Ts
        if t >= t1: v += A * g(t - t1)
        if t >= t2: v -= A * g(t - t2)
        n += 1
    return v

def vin(t):
    ph = (t % Ts)
    return A if ph < D * Ts else 0.0

# numerical reference: dv/dt = (vin - v)/(RC), RC=1
def numeric_ref(T, dt=1e-5):
    v = 0.0; t = 0.0; out = {}
    targets = sorted([0.5, 1.0, 1.5, 2.3, 3.0, 3.7, 5.0, 6.2, 8.0])
    ti = 0
    steps = int(T / dt)
    for k in range(steps + 1):
        t = k * dt
        while ti < len(targets) and t >= targets[ti] - 1e-9:
            out[targets[ti]] = v; ti += 1
        v += dt * (vin(t) - v)   # RC=1
    return out

print("="*72); print("VALIDATION: pulse source (superposition vs numerical ODE)"); print("="*72)
ref = numeric_ref(8.5)
err = 0.0
for t in sorted(ref):
    mine = pulse_response(t, 8.5)
    err = max(err, abs(mine - ref[t]))
print(f"[{'OK' if err < 2e-4 else 'FAIL'}] RC square-wave response (superposition)  maxerr={err:.2e}  (numeric dt=1e-5)")
print("="*72)
print("ALL OK" if err < 2e-4 else "SOME FAILED")
print("="*72)
