# -*- coding: utf-8 -*-
"""
Validate the IDEAL DIODE via state-assumption + event detection on top of the
piecewise framework. An ideal diode is a switch whose state is auto-determined:
  ON  (conducting) modeled as a 0 V source  -> read branch current i_D; valid while i_D >= 0
  OFF (blocking)   modeled as open          -> read v_D = v_anode - v_cathode; valid while v_D <= 0
Event = earliest time a constraint is violated -> toggle that diode, re-solve.
"""
import math
from proto import inverse_laplace_rational, eval_ft
from proto_circuit import R
from proto_ic_mutual import mna, ratR, ft

def tf(r):
    n, d = ratR(r); return lambda t: ft(n, d, t)

def source_shifted(c, t0):
    """Laplace transform of the source waveform re-expressed in interval-local
    time tau = t - t0 (so a continuing source keeps its phase/level)."""
    w = c.get('wave', 'step'); A = c.get('amp', 1.0); f = c.get('freq', 1.0)
    if w in ('step', 'dc'):
        return R([A], [0, 1.0])
    if w == 'impulse':
        return R.const(A) if abs(t0) < 1e-12 else R.const(0.0)
    if w == 'ramp':
        return R([A * t0], [0, 1.0]).add(R([A], [0, 0, 1.0]))   # A*t0/s + A/s^2
    if w == 'exp':
        return R([A * math.exp(-f * t0)], [f, 1.0])
    if w == 'sin':
        # A sin(w(t0+tau)) = A sin(w t0) cos + A cos(w t0) sin
        return R([A * math.cos(f * t0) * f, A * math.sin(f * t0)], [f * f, 0, 1.0])
    if w == 'cos':
        return R([-A * math.sin(f * t0) * f, A * math.cos(f * t0)], [f * f, 0, 1.0])
    return R([A], [0, 1.0])

def solve_with_diodes(nodes, base_comps, diodes, source_fns, horizon, dt=0.002):
    """
    diodes: list of {anode, cathode, state(initial 'on'/'off')}
    base_comps: non-diode comps (R/L/C/V/I with Laplace sources)
    Returns sampled (ts, out_fn_values) by walking events; also returns a probe
    function value list for node `probe_node` voltage. For validation we return
    a function out(t) for a given node via closure list of segments.
    """
    segments = []  # (t_start, t_end, V_dict, t_offset_for_eval)
    states = [d['state'] for d in diodes]
    t0 = 0.0
    guard = 0
    while t0 < horizon - 1e-9 and guard < 2000:
        guard += 1
        # rebuild base comps with sources re-expressed in interval-local time
        comps = []
        for c in base_comps:
            if c['type'] in ('V', 'I'):
                cc = dict(c); cc['Vs' if c['type'] == 'V' else 'Is'] = source_shifted(c, t0)
                comps.append(cc)
            else:
                comps.append(c)
        # add diode models per current state
        diode_vsrc_index = {}
        vcount = sum(1 for c in comps if c['type'] == 'V')
        for di, d in enumerate(diodes):
            if states[di] == 'on':
                diode_vsrc_index[di] = vcount
                comps.append({'type': 'V', 'a': d['anode'], 'b': d['cathode'], 'Vs': R.const(0)})
                vcount += 1
            # off: omit (open)
        V = mna(nodes, comps)
        # build monitors
        def make_monitor(di):
            d = diodes[di]
            if states[di] == 'on':
                k = diode_vsrc_index[di]
                # branch current of that 0V source; sign: + means anode->cathode (forward)
                return ('on', tf(V['__Iv'][k]))
            else:
                return ('off', tf(V[d['anode']].sub(V[d['cathode']])))
        mons = [make_monitor(di) for di in range(len(diodes))]
        # find earliest violation in (t0, horizon]
        event_t = None; event_di = None
        t = t0 + dt
        prev = {}
        while t <= horizon + 1e-12:
            for di, (kind, fn) in enumerate(mons):
                val = fn(t - t0)  # interval-local time
                viol = (val < -1e-9) if kind == 'on' else (val > 1e-9)
                if viol:
                    if event_t is None or t < event_t:
                        event_t = t; event_di = di
            if event_t is not None:
                break
            t += dt
        end = event_t if event_t is not None else horizon
        segments.append((t0, end, V, t0))
        if event_t is None:
            break
        states[event_di] = 'off' if states[event_di] == 'on' else 'on'
        t0 = event_t
    return segments

def node_out(segments, node):
    fns = [(s0, s1, tf(V[node]), off) for (s0, s1, V, off) in segments]
    def out(t):
        for (s0, s1, fn, off) in fns:
            if s0 - 1e-12 <= t < s1 + 1e-12:
                return fn(t - off)
        return fns[-1][2](t - fns[-1][3])
    return out

print("="*72); print("VALIDATION: ideal diode (event detection)"); print("="*72)

# Half-wave rectifier: sin source at node1, diode 1->2, R load 2->0.  R=1.
# Expected v_2(t) = max(0, sin(t)).
w = 1.0
base = [
    {'type': 'V', 'a': 1, 'b': 0, 'wave': 'sin', 'amp': 1.0, 'freq': w},
    {'type': 'R', 'a': 2, 'b': 0, 'value': 1.0},
]
diodes = [{'anode': 1, 'cathode': 2, 'state': 'on'}]
segs = solve_with_diodes({0,1,2}, base, diodes, None, horizon=10.0, dt=0.005)
out = node_out(segs, 2)
err = 0.0
for t in [0.5, 1.2, 2.0, 3.0, 3.3, 4.0, 5.0, 6.0, 6.4, 7.0, 8.0, 9.0]:
    ref = max(0.0, math.sin(t))
    err = max(err, abs(out(t) - ref))
print(f"diode events found: {len(segs)-1}  (expect ~3 in [0,10]: pi,2pi,3pi)")
print(f"[{'OK' if err<5e-3 else 'FAIL'}] half-wave rectifier v_R = max(0,sin t)   maxerr={err:.2e}")

# Diode with step source: positive step -> diode ON -> v_R = step ; reverse step -> OFF -> 0
basep = [
    {'type': 'V', 'a': 1, 'b': 0, 'wave': 'step', 'amp': 1.0},  # +1 step
    {'type': 'R', 'a': 2, 'b': 0, 'value': 2.0},
]
segp = solve_with_diodes({0,1,2}, basep, [{'anode':1,'cathode':2,'state':'on'}], None, 5.0, dt=0.01)
outp = node_out(segp, 2)
errp = max(abs(outp(t) - 1.0) for t in [0.5, 2.0, 4.0])  # forward step passes (ideal, v_R=1)
print(f"[{'OK' if errp<1e-6 else 'FAIL'}] forward step through diode -> v_R=1   maxerr={errp:.2e}")

basen = [
    {'type': 'V', 'a': 1, 'b': 0, 'wave': 'step', 'amp': -1.0},  # -1 step (reverse)
    {'type': 'R', 'a': 2, 'b': 0, 'value': 2.0},
]
segn = solve_with_diodes({0,1,2}, basen, [{'anode':1,'cathode':2,'state':'off'}], None, 5.0, dt=0.01)
outn = node_out(segn, 2)
errn = max(abs(outn(t) - 0.0) for t in [0.5, 2.0, 4.0])  # blocked -> 0
print(f"[{'OK' if errn<1e-6 else 'FAIL'}] reverse step blocked -> v_R=0   maxerr={errn:.2e}")

print("="*72)
print("ALL OK" if (err<5e-3 and errp<1e-6 and errn<1e-6) else "SOME FAILED")
print("="*72)
