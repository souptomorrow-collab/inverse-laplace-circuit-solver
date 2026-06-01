/* =====================================================================
 * engine.js — 反拉普拉斯轉換 / 電路 s 域求解 核心引擎 (no DOM)
 * 演算法已用 Python + sympy 完整驗證 (見 proto*.py)：
 *   - 有理函數 F(s)=N(s)/D(s) 反轉換：求根→重根分群→Newton 精修→
 *     線性方程組部分分式→留數對時域 (含 δ 衝激、時間延遲 e^{-as})
 *   - 電路 Modified Nodal Analysis (MNA) s 域符號求解
 * ===================================================================== */
(function (global) {
'use strict';

/* ---------- Complex ---------- */
class C {
  constructor(re, im) { this.re = re; this.im = im || 0; }
  add(o) { return new C(this.re + o.re, this.im + o.im); }
  sub(o) { return new C(this.re - o.re, this.im - o.im); }
  mul(o) { return new C(this.re * o.re - this.im * o.im, this.re * o.im + this.im * o.re); }
  div(o) { const d = o.re * o.re + o.im * o.im;
           return new C((this.re * o.re + this.im * o.im) / d, (this.im * o.re - this.re * o.im) / d); }
  neg() { return new C(-this.re, -this.im); }
  conj() { return new C(this.re, -this.im); }
  scale(k) { return new C(this.re * k, this.im * k); }
  abs() { return Math.hypot(this.re, this.im); }
}
function cx(re, im) { return new C(re, im || 0); }
function cexp(z) { const e = Math.exp(z.re); return new C(e * Math.cos(z.im), e * Math.sin(z.im)); }

/* ---------- Polynomials: ascending arrays of C (index i = coeff of s^i) ---------- */
function pmul(a, b) {
  const out = new Array(a.length + b.length - 1);
  for (let i = 0; i < out.length; i++) out[i] = cx(0);
  for (let i = 0; i < a.length; i++)
    for (let j = 0; j < b.length; j++)
      out[i + j] = out[i + j].add(a[i].mul(b[j]));
  return out;
}
function padd(a, b) { const n = Math.max(a.length, b.length), o = [];
  for (let i = 0; i < n; i++) o.push((a[i] || cx(0)).add(b[i] || cx(0))); return o; }
function psub(a, b) { const n = Math.max(a.length, b.length), o = [];
  for (let i = 0; i < n; i++) o.push((a[i] || cx(0)).sub(b[i] || cx(0))); return o; }
function peval(c, x) { let r = cx(0); for (let i = c.length - 1; i >= 0; i--) r = r.mul(x).add(c[i]); return r; }
function ptrim(a, tol) { tol = tol || 1e-9; a = a.slice();
  while (a.length > 1 && a[a.length - 1].abs() < tol) a.pop(); return a; }
function pderiv(a) { if (a.length <= 1) return [cx(0)]; const o = [];
  for (let i = 1; i < a.length; i++) o.push(a[i].scale(i)); return o; }
function deflateAsc(coeffs, root) { // divide by (s - root), return quotient ascending
  const desc = coeffs.slice().reverse(), n = desc.length, q = [];
  let prev = cx(0);
  for (let i = 0; i < n - 1; i++) { const cur = desc[i].add(prev.mul(root)); q.push(cur); prev = cur; }
  return q.reverse();
}

/* ---------- Durand-Kerner root finder (validated vs numpy.roots) ---------- */
function rootsOf(coeffsAsc) {
  let a = ptrim(coeffsAsc);
  let desc = a.slice().reverse();             // descending
  const n = desc.length - 1;
  if (n <= 0) return [];
  const lead = desc[0];
  desc = desc.map(c => c.div(lead));          // monic
  let maxc = 0; for (let i = 1; i < desc.length; i++) maxc = Math.max(maxc, desc[i].abs());
  const radius = Math.max(0.1, Math.min(1 + maxc, 100));
  const seed = cx(0.4, 0.9);
  const roots = []; let cur = cx(1, 0);
  for (let k = 0; k < n; k++) { roots.push(cur.scale(radius * 0.5)); cur = cur.mul(seed); }
  const pe = (coef, x) => { let r = cx(0); for (const c of coef) r = r.mul(x).add(c); return r; };
  for (let it = 0; it < 600; it++) {
    let maxd = 0;
    for (let i = 0; i < n; i++) {
      const xi = roots[i];
      const num = pe(desc, xi);
      let den = cx(1, 0);
      for (let j = 0; j < n; j++) if (j !== i) den = den.mul(xi.sub(roots[j]));
      if (den.abs() < 1e-300) den = cx(1e-300, 0);
      const delta = num.div(den);
      roots[i] = xi.sub(delta);
      maxd = Math.max(maxd, delta.abs());
    }
    if (maxd < 1e-14) break;
  }
  // Newton polish (simple roots)
  const deriv = []; for (let i = 0; i < n; i++) deriv.push(desc[i].scale(n - i));
  for (let i = 0; i < n; i++)
    for (let t = 0; t < 2; t++) {
      const p = pe(desc, roots[i]), dp = pe(deriv, roots[i]);
      if (dp.abs() > 1e-300) { const step = p.div(dp); if (step.abs() < 1) roots[i] = roots[i].sub(step); }
    }
  return roots;
}

/* Newton-refine a root of multiplicity m: simple root of poly^(m-1) -> machine precision */
function refineRoot(poly, p, m) {
  let g = poly; for (let i = 0; i < m - 1; i++) g = pderiv(g);
  const gp = pderiv(g);
  for (let it = 0; it < 12; it++) {
    const gv = peval(g, p), gpv = peval(gp, p);
    if (gpv.abs() < 1e-300) break;
    const step = gv.div(gpv); p = p.sub(step);
    if (step.abs() < 1e-15 * (1 + p.abs())) break;
  }
  return p;
}
function rebuildFromGroups(groups, lead) { // groups: [[rep, mult>0]] -> lead*prod(s-rep)^mult
  let poly = [cx(1)];
  for (const g of groups) for (let k = 0; k < g[1]; k++) poly = pmul(poly, [g[0].neg(), cx(1)]);
  return poly.map(c => c.mul(lead));
}

/* ---------- robust pole-zero cancellation (handles repeated roots) ----------
 * Cluster num-roots and den-roots SEPARATELY (generous tol, Newton-refined
 * reps), cancel min(multiplicity) at each matched location, rebuild with clean
 * repeated factors. If nothing cancels, keep exact original (monic-normalized). */
function reduceND(num, den, tol) {
  tol = tol || 8e-3;
  num = ptrim(num); den = ptrim(den);
  const leadN = num[num.length - 1], leadD = den[den.length - 1];
  if (leadN.abs() < 1e-14) return [[cx(0)], [cx(1)]];
  const nroots = num.length > 1 ? rootsOf(num) : [];
  const droots = den.length > 1 ? rootsOf(den) : [];
  const ngroups = nroots.length ? clusterRoots(nroots).map(g => [refineRoot(num, g[0], g[1]), g[1]]) : [];
  const dgroups = droots.length ? clusterRoots(droots).map(g => [refineRoot(den, g[0], g[1]), g[1]]) : [];
  let totalCancel = 0; const newN = [];
  for (const g of ngroups) {
    const rep = g[0], mult = g[1]; let matched = -1;
    for (let j = 0; j < dgroups.length; j++)
      if (dgroups[j][1] > 0 && dgroups[j][0].sub(rep).abs() < tol * (1 + rep.abs())) { matched = j; break; }
    if (matched >= 0) {
      const cancel = Math.min(mult, dgroups[matched][1]);
      totalCancel += cancel; newN.push([rep, mult - cancel]); dgroups[matched][1] -= cancel;
    } else newN.push([rep, mult]);
  }
  if (totalCancel === 0)
    return [ptrim(num.map(c => c.div(leadD))), ptrim(den.map(c => c.div(leadD)))];
  const numpoly = rebuildFromGroups(newN.filter(g => g[1] > 0), leadN);
  const denpoly = rebuildFromGroups(dgroups.filter(g => g[1] > 0), leadD);
  const ld = denpoly[denpoly.length - 1];
  return [ptrim(numpoly.map(c => c.div(ld))), ptrim(denpoly.map(c => c.div(ld)))];
}

/* ---------- cluster roots (relative tol; representative = mean) ---------- */
function clusterRoots(roots, tol) {
  tol = tol || 8e-3;
  const used = new Array(roots.length).fill(false), groups = [];
  for (let i = 0; i < roots.length; i++) {
    if (used[i]) continue;
    const cluster = [roots[i]]; used[i] = true;
    for (let j = i + 1; j < roots.length; j++)
      if (!used[j] && roots[j].sub(roots[i]).abs() < tol * (1 + roots[i].abs())) { cluster.push(roots[j]); used[j] = true; }
    let rep = cx(0); for (const c of cluster) rep = rep.add(c); rep = rep.scale(1 / cluster.length);
    groups.push([rep, cluster.length]);
  }
  return groups;
}

function factorial(n) { let f = 1; for (let i = 2; i <= n; i++) f *= i; return f; }

/* ---------- complex linear solve (Gaussian elim, partial pivot) ---------- */
function solveComplexLinear(M, rhs) {
  const n = rhs.length;
  const A = M.map((row, i) => row.slice().concat([rhs[i]]));
  for (let col = 0; col < n; col++) {
    let piv = col, best = A[col][col].abs();
    for (let r = col + 1; r < n; r++) { const v = A[r][col].abs(); if (v > best) { best = v; piv = r; } }
    if (best < 1e-300) throw new Error('部分分式線性系統奇異');
    const tmp = A[col]; A[col] = A[piv]; A[piv] = tmp;
    const pv = A[col][col];
    for (let k = col; k <= n; k++) A[col][k] = A[col][k].div(pv);
    for (let r = 0; r < n; r++) if (r !== col) {
      const f = A[r][col];
      if (f.abs() > 1e-300) for (let k = col; k <= n; k++) A[r][k] = A[r][k].sub(f.mul(A[col][k]));
    }
  }
  return A.map(row => row[n]);
}

/* ---------- partial fractions via linear system ---------- */
function partialFractions(N, D) {
  const degD = D.length - 1;
  let groups = clusterRoots(rootsOf(D));
  // refine every pole to machine precision (Newton on D^(m-1))
  groups = groups.map(g => [refineRoot(D, g[0], g[1]), g[1]]);
  const basis = [];
  for (const g of groups) {
    let cur = D.slice();
    for (let k = 1; k <= g[1]; k++) { cur = deflateAsc(cur, g[0]); basis.push({ pole: g[0], order: k, B: cur.slice() }); }
  }
  const M = [];
  for (let row = 0; row < degD; row++) M.push(basis.map(b => b.B[row] || cx(0)));
  const rhs = []; for (let row = 0; row < degD; row++) rhs.push(N[row] || cx(0));
  const c = solveComplexLinear(M, rhs);
  return basis.map((b, i) => ({ pole: b.pole, order: b.order, coeff: c[i] }));
}

/* ---------- inverse Laplace of rational N/D ---------- */
function inverseLaplaceRational(N, D, doReduce) {
  if (doReduce === undefined) doReduce = true;
  N = ptrim(N); D = ptrim(D);
  if (doReduce) { const r = reduceND(N, D); N = r[0]; D = r[1]; }
  let degN = N.length - 1; const degD = D.length - 1;
  let delta = [], pf = [];
  if (degN >= degD) {
    const Nd = N.slice().reverse(), Dd = D.slice().reverse();
    const qlen = Nd.length - Dd.length + 1;
    const q = new Array(qlen).fill(cx(0)); const rem = Nd.slice();
    for (let i = 0; i < qlen; i++) {
      const coef = rem[i].div(Dd[0]); q[i] = coef;
      for (let j = 0; j < Dd.length; j++) rem[i + j] = rem[i + j].sub(coef.mul(Dd[j]));
    }
    const Rasc = ptrim(rem.slice().reverse());
    delta = q.slice().reverse().map(c => c.re);
    N = Rasc; degN = N.length - 1;
    if (!(N.length === 1 && N[0].abs() < 1e-12)) pf = partialFractions(N, D);
  } else {
    pf = partialFractions(N, D);
  }
  return { delta: delta, terms: pf };
}

/* ---------- {delta, terms} -> real-valued time-domain term objects ---------- */
function toTimeDomain(res) {
  const reals = [], comps = [], deltas = [];
  (res.delta || []).forEach((c, i) => { if (Math.abs(c) > 1e-9) deltas.push({ order: i, coeff: c }); });
  for (const t of res.terms) {
    const p = t.pole, k = t.order, c = t.coeff, fact = factorial(k - 1);
    if (Math.abs(p.im) < 1e-6 * (1 + p.abs())) {
      reals.push({ A: c.re / fact, polyDeg: k - 1, sigma: p.re });
    } else if (p.im > 0) {
      comps.push({ Ccos: 2 * c.re / fact, Csin: -2 * c.im / fact, sigma: p.re, omega: p.im, polyDeg: k - 1 });
    } // p.im < 0 handled by conjugate partner
  }
  return { reals: reals, comps: comps, deltas: deltas };
}

/* evaluate continuous part g(tau) (deltas excluded) */
function evalG(td, tau) {
  let v = 0;
  for (const r of td.reals) v += r.A * Math.pow(tau, r.polyDeg) * Math.exp(r.sigma * tau);
  for (const c of td.comps)
    v += Math.exp(c.sigma * tau) * Math.pow(tau, c.polyDeg) *
         (c.Ccos * Math.cos(c.omega * tau) + c.Csin * Math.sin(c.omega * tau));
  return v;
}

/* ---------- number formatting ---------- */
function fmtNum(x, opts) {
  opts = opts || {};
  if (Math.abs(x) < 1e-10) return '0';
  const r = Math.round(x);
  if (Math.abs(x - r) < 1e-7) return String(r);
  // simple fractions
  for (const den of [2, 3, 4, 5, 6, 8]) {
    const n = x * den; const nr = Math.round(n);
    if (Math.abs(n - nr) < 1e-7 && nr !== 0) return nr + '/' + den;
  }
  let s = x.toFixed(opts.prec || 4);
  s = s.replace(/\.?0+$/, '');
  return s;
}
function fmtCoeffMul(x) { // coefficient that multiplies a term: "", "-", "3", "-2.5", "1/2"
  if (Math.abs(x - 1) < 1e-9) return '';
  if (Math.abs(x + 1) < 1e-9) return '-';
  return fmtNum(x);
}

/* ---------- format time-domain to HTML / plain text ----------
 * Build a list of {neg, body} items, then join with proper +/− signs. */
function coefT(k) { // format k·t  ->  "t" / "-t" / "2t" / "-1.5t"
  const s = fmtNum(k);
  if (s === '1') return 't';
  if (s === '-1') return '-t';
  return s + 't';
}
function eFactor(sigma, html) {
  const a = coefT(sigma);
  return html ? ('e<sup>' + a + '</sup>') : ('e^(' + a + ')');
}
function tFactor(d, html) {
  if (d === 1) return 't';
  return html ? ('t<sup>' + d + '</sup>') : ('t^' + d);
}
// hide a leading coefficient of 1 (tie the test to fmtNum's own rounding)
function magPrefix(mag) { const s = fmtNum(mag); return (s === '1') ? '' : (s + '·'); }

function formatTD(td, html) {
  const items = []; // {neg:bool, body:string}
  // real terms:  A · t^d · e^{σt}
  for (const r of td.reals) {
    if (Math.abs(r.A) < 1e-10) continue;
    const neg = r.A < 0, mag = Math.abs(r.A), factors = [];
    if (r.polyDeg >= 1) factors.push(tFactor(r.polyDeg, html));
    if (Math.abs(r.sigma) > 1e-9) factors.push(eFactor(r.sigma, html));
    const body = (factors.length === 0) ? fmtNum(mag) : (magPrefix(mag) + factors.join('·'));
    items.push({ neg: neg, body: body });
  }
  // complex pairs:  e^{σt} · t^d · (Ccos·cos(ωt) + Csin·sin(ωt))
  for (const c of td.comps) {
    if (Math.hypot(c.Ccos, c.Csin) < 1e-10) continue;
    const pre = [];
    if (Math.abs(c.sigma) > 1e-9) pre.push(eFactor(c.sigma, html));
    if (c.polyDeg >= 1) pre.push(tFactor(c.polyDeg, html));
    const preStr = pre.length ? (pre.join('·') + '·') : '';
    const wt = coefT(c.omega);
    const inner = [];
    if (Math.abs(c.Ccos) > 1e-10) inner.push({ neg: c.Ccos < 0, s: magPrefix(Math.abs(c.Ccos)) + 'cos(' + wt + ')' });
    if (Math.abs(c.Csin) > 1e-10) inner.push({ neg: c.Csin < 0, s: magPrefix(Math.abs(c.Csin)) + 'sin(' + wt + ')' });
    if (inner.length === 1) {
      items.push({ neg: inner[0].neg, body: preStr + inner[0].s });
    } else {
      let s = (inner[0].neg ? '−' : '') + inner[0].s + (inner[1].neg ? ' − ' : ' + ') + inner[1].s;
      items.push({ neg: false, body: preStr + '(' + s + ')' });
    }
  }
  // impulses:  c · δ^(n)(t)
  for (const d of td.deltas) {
    if (Math.abs(d.coeff) < 1e-10) continue;
    const name = d.order === 0 ? 'δ(t)' : (d.order === 1 ? "δ'(t)" : 'δ' + superscript(d.order) + '(t)');
    items.push({ neg: d.coeff < 0, body: magPrefix(Math.abs(d.coeff)) + name });
  }
  if (items.length === 0) return '0';
  let out = (items[0].neg ? '−' : '') + items[0].body;
  for (let i = 1; i < items.length; i++) out += (items[i].neg ? ' − ' : ' + ') + items[i].body;
  return out;
}
function superscript(n) { const m = { '0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹' };
  return String(n).split('').map(d => m[d] || d).join(''); }

/* ---------- format a rational F(s) for display ---------- */
function formatPolyReal(coeffsAsc, html) {
  // coeffsAsc: array of C (use .re), variable s
  const terms = [];
  for (let i = coeffsAsc.length - 1; i >= 0; i--) {
    const a = coeffsAsc[i].re;
    if (Math.abs(a) < 1e-9) continue;
    let coeff = (i === 0) ? fmtNum(a) : fmtCoeffMul(a);
    let pw = i === 0 ? '' : (i === 1 ? 's' : (html ? 's<sup>' + i + '</sup>' : 's^' + i));
    let term;
    if (i === 0) term = fmtNum(a);
    else if (coeff === '') term = pw;
    else if (coeff === '-') term = '-' + pw;
    else term = coeff + pw;
    terms.push(term);
  }
  if (terms.length === 0) return '0';
  let out = terms[0];
  for (let i = 1; i < terms.length; i++) {
    const p = terms[i];
    if (p[0] === '-') out += ' − ' + p.slice(1); else out += ' + ' + p;
  }
  return out;
}

/* =====================================================================
 * Expression parser:  string -> SVal (list of {delay, rat})
 * ===================================================================== */
class Rat {
  constructor(num, den, reduce) {
    num = ptrim(num); den = den ? ptrim(den) : [cx(1)];
    if (reduce !== false) { const r = reduceND(num, den); num = r[0]; den = r[1]; }
    this.num = num; this.den = den;
  }
  static cR(k) { return new Rat([cx(k)], [cx(1)]); }
  add(o) { return new Rat(padd(pmul(this.num, o.den), pmul(o.num, this.den)), pmul(this.den, o.den)); }
  sub(o) { return new Rat(psub(pmul(this.num, o.den), pmul(o.num, this.den)), pmul(this.den, o.den)); }
  mul(o) { return new Rat(pmul(this.num, o.num), pmul(this.den, o.den)); }
  div(o) { return new Rat(pmul(this.num, o.den), pmul(this.den, o.num)); }
  neg() { return new Rat(this.num.map(c => c.neg()), this.den, false); }
  isZero() { return this.num.every(c => c.abs() < 1e-12); }
  eval(x) { return peval(this.num, x).div(peval(this.den, x)); }
}

const CONSTS = { pi: Math.PI };
const FUNCS = { exp: 1 };

function tokenize(src) {
  const s = src.replace(/\s+/g, '');
  const toks = []; let i = 0;
  while (i < s.length) {
    const ch = s[i];
    if (/[0-9.]/.test(ch)) {
      let j = i; while (j < s.length && /[0-9.]/.test(s[j])) j++;
      if (j < s.length && /[eE]/.test(s[j]) && j + 1 < s.length && /[0-9+\-]/.test(s[j + 1])) {
        j++; if (/[+\-]/.test(s[j])) j++; while (j < s.length && /[0-9]/.test(s[j])) j++;
      }
      toks.push({ t: 'num', v: parseFloat(s.slice(i, j)) }); i = j;
    } else if (/[a-zA-Z_]/.test(ch)) {
      let j = i; while (j < s.length && /[a-zA-Z0-9_]/.test(s[j])) j++;
      toks.push({ t: 'id', v: s.slice(i, j) }); i = j;
    } else if ('+-*/^(),'.indexOf(ch) >= 0) {
      toks.push({ t: ch, v: ch }); i++;
    } else throw new Error('無法解析的字元: ' + ch);
  }
  // implicit multiplication
  const out = [];
  for (let k = 0; k < toks.length; k++) {
    const tk = toks[k];
    if (out.length) {
      const prev = out[out.length - 1];
      const prevEnd = (prev.t === 'num' || prev.t === 'id' || prev.t === ')');
      const curStart = (tk.t === 'num' || tk.t === 'id' || tk.t === '(');
      const isFuncCall = (prev.t === 'id' && tk.t === '(' && FUNCS[prev.v]);
      if (prevEnd && curStart && !isFuncCall) out.push({ t: '*', v: '*' });
    }
    out.push(tk);
  }
  return out;
}

class SVal {
  constructor(terms) { this.terms = terms; } // [{delay, rat}]
  static cR(k) { return new SVal([{ delay: 0, rat: Rat.cR(k) }]); }
  static svar() { return new SVal([{ delay: 0, rat: new Rat([cx(0), cx(1)], [cx(1)]) }]); }
  add(o) { return new SVal(this.terms.concat(o.terms)); }
  neg() { return new SVal(this.terms.map(t => ({ delay: t.delay, rat: t.rat.neg() }))); }
  mul(o) { const out = [];
    for (const a of this.terms) for (const b of o.terms) out.push({ delay: a.delay + b.delay, rat: a.rat.mul(b.rat) });
    return new SVal(out); }
  combine() {
    const keys = [], groups = {};
    for (const t of this.terms) {
      const key = Math.round(t.delay * 1e9) / 1e9;
      if (groups[key]) groups[key] = groups[key].add(t.rat); else { groups[key] = t.rat; keys.push(key); }
    }
    keys.sort((a, b) => a - b);
    return new SVal(keys.map(k => ({ delay: k, rat: groups[k] })));
  }
  div(o) {
    const oc = o.combine();
    if (oc.terms.length !== 1 || Math.abs(oc.terms[0].delay) > 1e-12) throw new Error('分母不可含時間延遲 e^{-as}');
    const r0 = oc.terms[0].rat;
    return new SVal(this.terms.map(t => ({ delay: t.delay, rat: t.rat.div(r0) })));
  }
  powi(n) {
    if (n === 0) return SVal.cR(1);
    if (n < 0) return SVal.cR(1).div(this.powi(-n));
    let res = this; for (let i = 1; i < n; i++) res = res.mul(this); return res;
  }
  isConst() { const c = this.combine();
    return c.terms.length === 1 && Math.abs(c.terms[0].delay) < 1e-12 &&
           c.terms[0].rat.den.length === 1 && c.terms[0].rat.num.length <= 1; }
  constVal() { const c = this.combine(); const n = c.terms[0].rat.num; return n.length ? n[0] : cx(0); }
}

class Parser {
  constructor(toks) { this.toks = toks; this.pos = 0; }
  peek() { return this.pos < this.toks.length ? this.toks[this.pos] : { t: null }; }
  next() { return this.toks[this.pos++]; }
  expect(t) { const x = this.next(); if (!x || x.t !== t) throw new Error('語法錯誤：缺少 ' + t); return x; }
  parse() { const v = this.expr(); if (this.pos !== this.toks.length) throw new Error('多餘的符號'); return v.combine(); }
  expr() { let v = this.term();
    while (this.peek().t === '+' || this.peek().t === '-') { const op = this.next().t; const r = this.term(); v = (op === '+') ? v.add(r) : v.add(r.neg()); }
    return v; }
  term() { let v = this.factor();
    while (this.peek().t === '*' || this.peek().t === '/') { const op = this.next().t; const r = this.factor(); v = (op === '*') ? v.mul(r) : v.div(r); }
    return v; }
  factor() {
    if (this.peek().t === '-') { this.next(); return this.factor().neg(); }
    if (this.peek().t === '+') { this.next(); return this.factor(); }
    let base = this.base();
    if (this.peek().t === '^') {
      this.next(); const e = this.factor();
      if (!e.isConst()) throw new Error('指數必須是常數');
      const n = e.constVal().re;
      if (Math.abs(n - Math.round(n)) > 1e-9) throw new Error('僅支援整數次方');
      base = base.powi(Math.round(n));
    }
    return base;
  }
  base() {
    const tk = this.peek();
    if (tk.t === '(') { this.next(); const v = this.expr(); this.expect(')'); return v; }
    if (tk.t === 'num') { this.next(); return SVal.cR(tk.v); }
    if (tk.t === 'id') {
      this.next();
      if (tk.v === 's') return SVal.svar();
      if (tk.v === 'exp') { this.expect('('); const arg = this.expr(); this.expect(')'); return this.doExp(arg); }
      // Euler's number: allow "e^(-2s)" as a delay (treat e^x as exp(x))
      if (tk.v === 'e') {
        if (this.peek().t === '^') { this.next(); const ex = this.factor(); return this.doExp(ex); }
        return SVal.cR(Math.E);
      }
      if (CONSTS[tk.v] !== undefined) return SVal.cR(CONSTS[tk.v]);
      throw new Error('未知符號: ' + tk.v + '（變數請用 s）');
    }
    throw new Error('語法錯誤');
  }
  doExp(arg) {
    arg = arg.combine();
    if (arg.terms.length !== 1 || Math.abs(arg.terms[0].delay) > 1e-12) throw new Error('exp 參數過於複雜');
    const r = arg.terms[0].rat;
    if (r.den.length !== 1) throw new Error('exp 參數必須是 s 的多項式');
    const co = r.num;
    const c0 = co.length >= 1 ? co[0].re : 0;
    const c1 = co.length >= 2 ? co[1].re : 0;
    for (let i = 2; i < co.length; i++) if (co[i].abs() > 1e-12) throw new Error('exp 參數必須是 s 的一次式（時間延遲）');
    if (c1 > 1e-12) throw new Error('不支援 e^{+as}（非因果）');
    return new SVal([{ delay: -c1, rat: Rat.cR(Math.exp(c0)) }]);
  }
}
function parseExpr(src) { return new Parser(tokenize(src)).parse(); }

/* =====================================================================
 * Top-level: invert a formula string -> structured result
 * ===================================================================== */
function invertFormula(src) {
  // try special non-rational table first
  const sp = specialTransform(src);
  if (sp) return { special: true, html: sp.html, text: sp.text, evalFn: sp.evalFn, blocks: null, note: sp.note };
  const sv = parseExpr(src.toLowerCase());
  const blocks = [];
  for (const t of sv.terms) {
    const res = inverseLaplaceRational(t.rat.num, t.rat.den);
    const td = toTimeDomain(res);
    blocks.push({ delay: t.delay, td: td, res: res, num: t.rat.num, den: t.rat.den });
  }
  return { special: false, blocks: blocks };
}

/* build a result (same shape as invertFormula) directly from a rational F(s)
   given as REAL coefficient arrays (ascending). Used by the circuit solver. */
function invertRational(numReal, denReal) {
  const N = numReal.map(v => cx(v)), D = denReal.map(v => cx(v));
  const res = inverseLaplaceRational(N, D);
  const td = toTimeDomain(res);
  return { special: false, blocks: [{ delay: 0, td: td, res: res, num: N, den: D }] };
}

/* poles (with multiplicity) across all blocks — for display */
function polesOf(result) {
  if (result.special || !result.blocks) return [];
  const out = [];
  for (const b of result.blocks) {
    const seen = [];
    for (const t of b.res.terms) {
      let f = seen.find(x => x.p.sub(t.pole).abs() < 1e-7);
      if (!f) { f = { p: t.pole, m: 0 }; seen.push(f); }
      f.m = Math.max(f.m, t.order);
    }
    for (const x of seen) out.push({ pole: x.p, mult: x.m, delay: b.delay });
  }
  return out;
}

/* evaluate f(t) for a formula result (continuous part; deltas excluded) */
function evalFormula(result, t) {
  if (result.special) return result.evalFn ? result.evalFn(t) : NaN;
  let v = 0;
  for (const b of result.blocks) {
    if (t >= b.delay - 1e-12) v += evalG(b.td, t - b.delay);
  }
  return v;
}

/* collect all deltas (with time location = delay) for plotting/markers */
function collectDeltas(result) {
  const out = [];
  if (result.special || !result.blocks) return out;
  for (const b of result.blocks)
    for (const d of b.td.deltas) out.push({ at: b.delay, order: d.order, coeff: d.coeff });
  return out;
}

/* render formula result to HTML (full f(t) = ...) */
function renderFormula(result) {
  if (result.special) return result.html;
  const segs = [];
  for (const b of result.blocks) {
    let g = formatTD(b.td, true);
    if (Math.abs(b.delay) < 1e-9) segs.push(g);
    else {
      const a = fmtNum(b.delay);
      const shifted = g.replace(/t/g, '(t−' + a + ')');
      segs.push('[ ' + shifted + ' ]·u(t−' + a + ')');
    }
  }
  let out = segs[0] || '0';
  for (let i = 1; i < segs.length; i++) out += '<br>&nbsp;&nbsp;+ ' + segs[i];
  return out;
}

/* =====================================================================
 * Special (non-rational) transform table — common research forms
 * ===================================================================== */
function specialTransform(src) {
  const s = src.replace(/\s+/g, '').toLowerCase();
  const table = [
    { re: /^1\/sqrt\(s\)$/, html: '1 / √(π·t)', text: '1/sqrt(pi*t)', evalFn: t => 1 / Math.sqrt(Math.PI * t), note: '非有理：L⁻¹{1/√s}' },
    { re: /^1\/s\^\(1\/2\)$/, html: '1 / √(π·t)', text: '1/sqrt(pi*t)', evalFn: t => 1 / Math.sqrt(Math.PI * t), note: '非有理' },
    { re: /^1\/s\^\(3\/2\)$/, html: '2·√(t/π)', text: '2*sqrt(t/pi)', evalFn: t => 2 * Math.sqrt(t / Math.PI), note: '非有理：L⁻¹{s^(-3/2)}' },
    { re: /^sqrt\(s\)$/, html: '−1 / (2·√π·t<sup>3/2</sup>)', text: '-1/(2 sqrt(pi) t^(3/2))', evalFn: t => -1 / (2 * Math.sqrt(Math.PI) * Math.pow(t, 1.5)), note: '分布意義下' },
    { re: /^1\/\(s\^2\+([0-9.]+)\)\^\(1\/2\)$|^1\/sqrt\(s\^2\+([0-9.]+)\)$/, special: 'besselJ0', note: 'L⁻¹{1/√(s²+a²)} = J₀(a t)' },
  ];
  for (const e of table) {
    const m = s.match(e.re);
    if (m) {
      if (e.special === 'besselJ0') {
        const a = Math.sqrt(parseFloat(m[1] || m[2]));
        return { html: 'J₀(' + fmtNum(a) + '·t)　<span style="opacity:.7">(第一類零階貝索函數)</span>',
                 text: 'J0(' + fmtNum(a) + ' t)', evalFn: t => besselJ0(a * t), note: e.note };
      }
      return { html: e.html, text: e.text, evalFn: e.evalFn, note: e.note };
    }
  }
  return null;
}
function besselJ0(x) { // series for |x|<8, Numerical-Recipes asymptotic otherwise
  const ax = Math.abs(x);
  if (ax < 8) {
    let sum = 1, term = 1; const x2 = -(x * x) / 4;
    for (let k = 1; k < 40; k++) { term *= x2 / (k * k); sum += term; if (Math.abs(term) < 1e-16) break; }
    return sum;
  }
  const z = 8 / ax, y = z * z, xx = ax - 0.785398164;
  const p1 = 1 + y * (-0.1098628627e-2 + y * (0.2734510407e-4 + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
  const p2 = -0.1562499995e-1 + y * (0.1430488765e-3 + y * (-0.6911147651e-5 + y * (0.7621095161e-6 + y * (-0.934935152e-7))));
  return Math.sqrt(0.636619772 / ax) * (Math.cos(xx) * p1 - z * Math.sin(xx) * p2);
}

/* =====================================================================
 * Circuit MNA solver (s-domain) — supports initial conditions, mutual
 * inductance, ideal diodes & switches (states), and source phase-shifting.
 * component: {type:'R'|'L'|'C'|'V'|'I'|'D'|'SW'|'W'|'GND', a, b, value,
 *             wave, amp, freq, v0(C), i0(L), state(SW), tSwitch(SW)}
 * ===================================================================== */
function admittance(type, value) {
  if (type === 'R') return Rat.cR(1 / value);                       // 1/R
  if (type === 'C') return new Rat([cx(0), cx(value)], [cx(1)]);    // sC
  if (type === 'L') return new Rat([cx(1)], [cx(0), cx(value)]);    // 1/(sL)
  throw new Error('非被動元件');
}

/* source Laplace transform re-expressed in interval-local time tau=t-t0 (so a
   continuing source keeps its phase/level across piecewise intervals) */
function sourceLaplace(spec, t0) {
  t0 = t0 || 0;
  const A = (spec.amp !== undefined) ? spec.amp : (spec.value !== undefined ? spec.value : 1);
  const f = (spec.freq !== undefined) ? spec.freq : 1;
  const w = spec.wave || 'step';
  switch (w) {
    case 'step': case 'dc': return new Rat([cx(A)], [cx(0), cx(1)]);                 // A/s
    case 'impulse': return Math.abs(t0) < 1e-12 ? Rat.cR(A) : Rat.cR(0);
    case 'ramp': return new Rat([cx(A * t0)], [cx(0), cx(1)]).add(new Rat([cx(A)], [cx(0), cx(0), cx(1)]));
    case 'exp': return new Rat([cx(A * Math.exp(-f * t0))], [cx(f), cx(1)]);
    case 'sin': return new Rat([cx(A * Math.cos(f * t0) * f), cx(A * Math.sin(f * t0))], [cx(f * f), cx(0), cx(1)]);
    case 'cos': return new Rat([cx(-A * Math.sin(f * t0) * f), cx(A * Math.cos(f * t0))], [cx(f * f), cx(0), cx(1)]);
    default: return new Rat([cx(A)], [cx(0), cx(1)]);
  }
}
function sourceS(wave, amp, param) { return sourceLaplace({ wave: wave, amp: amp, freq: param }, 0); }

function solveCircuit(components, opts) {
  opts = opts || {};
  const t0 = opts.t0 || 0;
  const nodeIC = opts.nodeIC || {};                  // 'x,y' -> initial voltage
  const mutuals = opts.mutuals || [];                // [{a:compRef,b:compRef,M}]
  const stateOf = opts.stateOf || (c => c.state);    // diode/switch state resolver
  const uf = {};
  function find(k) { if (uf[k] === undefined) uf[k] = k; while (uf[k] !== k) { uf[k] = uf[uf[k]]; k = uf[k]; } return k; }
  function union(a, b) { uf[find(a)] = find(b); }
  const key = p => p.x + ',' + p.y;
  for (const c of components) { find(key(c.a)); if (c.b) find(key(c.b)); }
  for (const c of components) if (c.type === 'W') union(key(c.a), key(c.b));
  const groundKeys = new Set();
  for (const c of components) if (c.type === 'GND') groundKeys.add(find(key(c.a)));
  if (groundKeys.size === 0) throw new Error('電路缺少接地 (GND)');
  const repSet = new Set();
  for (const c of components) { repSet.add(find(key(c.a))); if (c.b) repSet.add(find(key(c.b))); }
  const isGround = r => groundKeys.has(find(r));
  const nodeIndex = {}; let idx = 0;
  for (const r of repSet) { const rr = find(r); if (isGround(rr)) nodeIndex[rr] = 0; }
  const nonGround = [...repSet].map(find).filter((v, i, arr) => arr.indexOf(v) === i).filter(r => !isGround(r));
  for (const r of nonGround) nodeIndex[r] = ++idx;
  const nv = idx;
  const nodeOf = p => nodeIndex[find(key(p))];
  const icAt = p => (nodeIC[key(p)] || 0);

  // voltage-defining branches: real V sources + ON diodes (V=Vf) + closed switches (0V).
  // Non-ideal options: a closed switch with Ron>0 is stamped as a resistor (Rds_on)
  // rather than an ideal 0V source; an ON diode carries its forward drop Vf.
  const vbranches = [];
  const onRes = [];                                       // closed switches modeled by on-resistance
  for (const c of components) {
    if (c.type === 'V') vbranches.push({ comp: c, kind: 'V', a: c.a, b: c.b, Vs: sourceLaplace(c, t0) });
    else if (c.type === 'D' && stateOf(c) === 'on') vbranches.push({ comp: c, kind: 'D', a: c.a, b: c.b, Vs: (c.Vf ? new Rat([cx(c.Vf)], [cx(0), cx(1)]) : Rat.cR(0)) });   // Vf/s (a DC forward drop)
    else if (c.type === 'SW' && (stateOf(c) || 'open') === 'closed') {
      if (c.Ron > 0) onRes.push({ comp: c, a: c.a, b: c.b, R: c.Ron });
      else vbranches.push({ comp: c, kind: 'SW', a: c.a, b: c.b, Vs: Rat.cR(0) });
    }
  }
  const m = vbranches.length;
  const size = nv + m;
  if (size === 0) throw new Error('電路沒有可解的節點');
  const A = []; for (let i = 0; i < size; i++) { A.push([]); for (let j = 0; j < size; j++) A[i].push(Rat.cR(0)); }
  const bb = []; for (let i = 0; i < size; i++) bb.push(Rat.cR(0));
  const addA = (i, j, val) => { A[i][j] = A[i][j].add(val); };
  const addB = (node, val) => { if (node !== 0) bb[node - 1] = bb[node - 1].add(val); };
  const stampY = (na, nb, Y) => {
    if (na !== 0) addA(na - 1, na - 1, Y);
    if (nb !== 0) addA(nb - 1, nb - 1, Y);
    if (na !== 0 && nb !== 0) { addA(na - 1, nb - 1, Y.neg()); addA(nb - 1, na - 1, Y.neg()); }
  };
  const capV0 = c => (c.v0 !== undefined && c.v0 !== null) ? c.v0 : (icAt(c.a) - icAt(c.b));
  // which inductors are coupled
  const coupledSet = new Set();
  for (const mu of mutuals) { coupledSet.add(mu.a); coupledSet.add(mu.b); }

  // passive stamps with IC companion sources
  for (const c of components) {
    if (c.type === 'R') stampY(nodeOf(c.a), nodeOf(c.b), admittance('R', c.value));
    else if (c.type === 'C') {
      stampY(nodeOf(c.a), nodeOf(c.b), admittance('C', c.value));
      const v0 = capV0(c);
      if (Math.abs(v0) > 1e-15) { addB(nodeOf(c.a), Rat.cR(c.value * v0)); addB(nodeOf(c.b), Rat.cR(-c.value * v0)); }
    } else if (c.type === 'L' && !coupledSet.has(c)) {
      stampY(nodeOf(c.a), nodeOf(c.b), admittance('L', c.value));
      const i0 = c.i0 || 0;
      if (Math.abs(i0) > 1e-15) { addB(nodeOf(c.a), new Rat([cx(-i0)], [cx(0), cx(1)])); addB(nodeOf(c.b), new Rat([cx(i0)], [cx(0), cx(1)])); }
    }
  }
  // mutual coupled inductor pairs: stamp (1/s)L^{-1} as a coupled 2-port
  for (const mu of mutuals) {
    const ca = mu.a, cb = mu.b, M = mu.M, La = ca.value, Lb = cb.value, D = La * Lb - M * M;
    const Yof = g => new Rat([cx(g)], [cx(0), cx(1)]);               // g/s
    const g11 = Lb / D, g12 = -M / D, g21 = -M / D, g22 = La / D;
    const a1 = nodeOf(ca.a), a2 = nodeOf(ca.b), b1 = nodeOf(cb.a), b2 = nodeOf(cb.b);
    stampY(a1, a2, Yof(g11)); stampY(b1, b2, Yof(g22));
    const cross = (p, q, val) => { if (p !== 0 && q !== 0) addA(p - 1, q - 1, val); };
    const Y12 = Yof(g12), Y21 = Yof(g21);
    cross(a1, b1, Y12); cross(a1, b2, Y12.neg()); cross(a2, b1, Y12.neg()); cross(a2, b2, Y12);
    cross(b1, a1, Y21); cross(b1, a2, Y21.neg()); cross(b2, a1, Y21.neg()); cross(b2, a2, Y21);
    for (const c of [ca, cb]) {
      const i0 = c.i0 || 0;
      if (Math.abs(i0) > 1e-15) { addB(nodeOf(c.a), new Rat([cx(-i0)], [cx(0), cx(1)])); addB(nodeOf(c.b), new Rat([cx(i0)], [cx(0), cx(1)])); }
    }
  }
  // closed switches with on-resistance (Rds_on): plain conductance 1/Ron
  for (const r of onRes) stampY(nodeOf(r.a), nodeOf(r.b), Rat.cR(1 / r.R));
  // current sources (shifted)
  for (const c of components) if (c.type === 'I') {
    const Is = sourceLaplace(c, t0);
    addB(nodeOf(c.a), Is); addB(nodeOf(c.b), Is.neg());
  }
  // voltage-defining branches
  vbranches.forEach((vb, kk) => {
    const row = nv + kk, na = nodeOf(vb.a), nb = nodeOf(vb.b);
    if (na !== 0) { addA(na - 1, row, Rat.cR(1)); addA(row, na - 1, Rat.cR(1)); }
    if (nb !== 0) { addA(nb - 1, row, Rat.cR(-1)); addA(row, nb - 1, Rat.cR(-1)); }
    bb[row] = vb.Vs;
  });

  // gmin: tiny conductance to ground on every node — resolves floating-node
  // singularities (open switches / OFF diodes). 0 by default (exact).
  if (opts.gmin) for (let i = 0; i < nv; i++) addA(i, i, Rat.cR(opts.gmin));

  const x = solveRational(A, bb);
  const Vnode = {};
  for (const r of repSet) { const rr = find(r); Vnode[rr] = isGround(rr) ? Rat.cR(0) : x[nodeIndex[rr] - 1]; }
  const Vof = p => Vnode[find(key(p))] || Rat.cR(0);
  const branchIndex = comp => vbranches.findIndex(vb => vb.comp === comp);
  const Iof = i => x[nv + i];
  const couplingOf = c => mutuals.find(mu => mu.a === c || mu.b === c);
  const indCurrent = c => {                                          // inductor current a->b
    const i0 = c.i0 || 0;
    const mu = couplingOf(c);
    if (!mu) return new Rat([cx(1)], [cx(0), cx(c.value)]).mul(Vof(c.a).sub(Vof(c.b))).add(new Rat([cx(i0)], [cx(0), cx(1)]));
    const La = mu.a.value, Lb = mu.b.value, M = mu.M, D = La * Lb - M * M;
    const Va = Vof(mu.a.a).sub(Vof(mu.a.b)), Vb = Vof(mu.b.a).sub(Vof(mu.b.b));
    const sInv = g => new Rat([cx(g)], [cx(0), cx(1)]);
    if (c === mu.a) return sInv(Lb / D).mul(Va).add(sInv(-M / D).mul(Vb)).add(new Rat([cx(i0)], [cx(0), cx(1)]));
    return sInv(-M / D).mul(Va).add(sInv(La / D).mul(Vb)).add(new Rat([cx(i0)], [cx(0), cx(1)]));
  };
  return {
    nv: nv, vbranches: vbranches,
    voltageAt: Vof,
    voltageDiff: (p, q) => Vof(p).sub(Vof(q)),
    capVoltage: c => Vof(c.a).sub(Vof(c.b)),
    indCurrent: indCurrent,
    branchCurrent: comp => { const k = branchIndex(comp); if (k < 0) throw new Error('該元件非導通分支'); return Iof(k); },
    currentThrough: comp => {
      if (comp.type === 'R' || comp.type === 'C') return admittance(comp.type, comp.value).mul(Vof(comp.a).sub(Vof(comp.b)));
      if (comp.type === 'L') return indCurrent(comp);
      if (comp.type === 'SW') {
        const k = branchIndex(comp); if (k >= 0) return Iof(k);              // ideal closed (0V source)
        if (comp.Ron > 0 && (stateOf(comp) || 'open') === 'closed') return admittance('R', comp.Ron).mul(Vof(comp.a).sub(Vof(comp.b)));
        return Rat.cR(0);                                                     // open
      }
      if (comp.type === 'V' || comp.type === 'D') { const k = branchIndex(comp); if (k < 0) return Rat.cR(0); return Iof(k); }
      if (comp.type === 'I') return sourceLaplace(comp, t0);
      throw new Error('無法量測此元件電流');
    },
    nodeLabel: p => { const n = nodeOf(p); return n === 0 ? 'GND' : 'N' + n; }
  };
}

function solveRational(A, b) {
  const n = A.length;
  const M = A.map((row, i) => row.map(x => x).concat([b[i]]));
  for (let col = 0; col < n; col++) {
    let piv = -1; for (let r = col; r < n; r++) if (!M[r][col].isZero()) { piv = r; break; }
    if (piv < 0) throw new Error('電路矩陣奇異（可能是浮接節點或缺少接地路徑）');
    const tmp = M[col]; M[col] = M[piv]; M[piv] = tmp;
    const pv = M[col][col];
    M[col] = M[col].map(x => x.div(pv));
    for (let r = 0; r < n; r++) if (r !== col && !M[r][col].isZero()) {
      const f = M[r][col]; M[r] = M[r].map((x, k) => x.sub(f.mul(M[col][k])));
    }
  }
  return M.map(row => row[n]);
}

/* convert a Rat (F(s)) into real coeff arrays for the inverse engine */
function ratToReal(rat) {
  return { num: rat.num.map(c => c.re), den: rat.den.map(c => c.re) };
}

/* =====================================================================
 * Transient orchestration: linear / pulse-superposition / piecewise(diode,switch)
 * Produces a uniform "Signal": {evalAt, deltas, tEndHint, renderHTML, Fs?, segments?}
 * ===================================================================== */
function ratTimeFn(rat) { const F = ratToReal(rat); const res = invertRational(F.num, F.den); return t => evalFormula(res, t); }
/* limit of num(s)/den(s) as s→+∞ — captures the impulsive (t=0⁺) behaviour of a
   signal, e.g. the δ-spike in an ideal inductor's terminal voltage when its branch
   is suddenly opened. Needed so ideal-diode turn-on detection sees that impulsive
   forward bias (which lives in the s→∞ part, not at t=eps>0). */
function ratLimInf(rat) {
  const deg = arr => { let n = arr.length - 1; while (n > 0 && Math.abs(arr[n].re) < 1e-13) n--; return n; };
  const dn = deg(rat.num), dd = deg(rat.den);
  const ld = rat.den[dd].re;
  if (Math.abs(ld) < 1e-300) return 0;
  const ratio = rat.num[dn].re / ld;
  if (dn > dd) return ratio >= 0 ? Infinity : -Infinity;
  if (dn === dd) return ratio;
  return 0;
}
function solveSafe(components, opts) {
  try { return solveCircuit(components, opts); }
  catch (e) {
    if (/奇異|singular/.test(e.message)) return solveCircuit(components, Object.assign({}, opts, { gmin: (opts && opts.gmin) || 1e-9 }));
    throw e;
  }
}
function probeRat(sol, probe) {
  if (probe.kind === 'V') return sol.voltageDiff(probe.a, probe.b);          // V(a)−V(b)
  if (probe.kind === 'I') { const r = sol.currentThrough(probe.comp); return probe.dir === -1 ? r.neg() : r; }   // current a→b (dir=−1 reverses)
  throw new Error('未知探針類型');
}
function resolveMutuals(components, couplings) {
  // couplings: [{aRef, bRef, M}] where aRef/bRef identify inductors (by object). Returns [{a,b,M}]
  if (!couplings) return [];
  return couplings.map(c => ({ a: c.a, b: c.b, M: c.M })).filter(c => c.a && c.b);
}
function tEndForResult(result) {
  if (!result || !result.blocks) return 10;
  let maxTau = 0, minOmega = Infinity, maxDelay = 0, hasOsc = false, unstable = false;
  for (const b of result.blocks) {
    maxDelay = Math.max(maxDelay, b.delay);
    for (const r of b.td.reals) { if (r.sigma < -1e-6) maxTau = Math.max(maxTau, 1 / -r.sigma); else if (r.sigma > 1e-6) unstable = true; }
    for (const c of b.td.comps) { if (c.sigma < -1e-6) maxTau = Math.max(maxTau, 1 / -c.sigma); else if (c.sigma > 1e-6) unstable = true; if (c.omega > 1e-6) { hasOsc = true; minOmega = Math.min(minOmega, c.omega); } }
  }
  let T = 8;
  if (maxTau > 0) T = Math.max(T, 6 * maxTau);
  if (hasOsc && isFinite(minOmega)) T = Math.max(maxTau > 0 ? 6 * maxTau : 0, 4 * (2 * Math.PI / minOmega));
  if (unstable) T = Math.min(maxTau > 0 ? 6 * maxTau : 8, 8);
  return Math.max(0.5, Math.min(T + maxDelay, 200));
}
/* characteristic time scales from component values (so horizon/dt auto-adapt to
   µs-scale switching circuits as well as ~1s circuits) */
function circuitTimeScale(components) {
  const Rs = components.filter(c => c.type === 'R' && c.value > 0).map(c => c.value);
  const Ls = components.filter(c => c.type === 'L' && c.value > 0).map(c => c.value);
  const Cs = components.filter(c => c.type === 'C' && c.value > 0).map(c => c.value);
  const slow = [], fast = [];
  for (const C of Cs) for (const R of Rs) { slow.push(R * C); fast.push(R * C); }
  for (const L of Ls) for (const R of Rs) { slow.push(L / R); fast.push(L / R); }
  for (const L of Ls) for (const C of Cs) { const w = 2 * Math.PI * Math.sqrt(L * C); slow.push(w); fast.push(w); }
  const swT = [], srcT = [];
  for (const c of components) {
    if (c.type === 'SW' && c.tSwitch > 0) swT.push(c.tSwitch);
    if (c.type === 'SW' && c.mode === 'pwm' && c.Tsw > 0) swT.push(c.Tsw);
    if ((c.type === 'V' || c.type === 'I') && c.wave === 'pulse' && c.Ts > 0) swT.push(c.Ts);
    if ((c.type === 'V' || c.type === 'I') && (c.wave === 'sin' || c.wave === 'cos') && c.freq > 0) { slow.push(2 * Math.PI / c.freq); srcT.push(2 * Math.PI / c.freq); }
  }
  for (const t of swT) { fast.push(t); }                  // switching intervals are "fast" events
  let tSlow = slow.length ? Math.max.apply(null, slow) : 1;
  let tFast = fast.length ? Math.min.apply(null, fast) : tSlow;
  // for periodic switching, the meaningful window must cover settling (slow) — keep tSlow.
  if (!isFinite(tSlow) || tSlow <= 0) tSlow = 1;
  if (!isFinite(tFast) || tFast <= 0) tFast = tSlow;
  return { tFast: tFast, tSlow: tSlow, switching: swT.length > 0 };
}
function circuitHorizon(components) {
  const ts = circuitTimeScale(components);
  let T = 8 * ts.tSlow;
  // when periodically switching, cap to a sensible number of switching periods so
  // the sim stays responsive (piecewise-Laplace makes ~2 segments per period).
  const periods = components.filter(c => (c.type === 'SW' && c.mode === 'pwm' && c.Tsw > 0)).map(c => c.Tsw)
    .concat(components.filter(c => (c.type === 'V' || c.type === 'I') && c.wave === 'pulse' && c.Ts > 0).map(c => c.Ts));
  if (periods.length) {
    const Tp = Math.min.apply(null, periods);
    T = Math.min(T, 400 * Tp);          // at most ~400 switching periods
    T = Math.max(T, 20 * Tp);           // but show at least ~20 periods
  }
  return T;
}
function hasEvents(components) {
  return components.some(c => c.type === 'D') ||
         components.some(c => c.type === 'SW' && ((c.tSwitch > 0) || (c.mode === 'pwm' && c.Tsw > 0))) ||
         components.some(c => (c.type === 'V' || c.type === 'I') && c.wave === 'pulse');
}

/* ---- linear (no events) ---- */
function linearAnalyze(components, probe, mutuals, nodeIC) {
  const sol = solveSafe(components, { nodeIC: nodeIC, mutuals: mutuals, stateOf: c => c.state });
  const rat = probeRat(sol, probe);
  const F = ratToReal(rat);
  const result = invertRational(F.num, F.den);
  return {
    piecewise: false, Fs: F, result: result,
    evalAt: t => evalFormula(result, t),
    deltas: collectDeltas(result),
    tEndHint: tEndForResult(result),
    renderHTML: renderFormula(result),
    segments: null,
  };
}

/* ---- pulse source via superposition of shifted step responses ---- */
function pulseAnalyze(components, probe, mutuals, nodeIC, pulseComp) {
  const A = pulseComp.amp !== undefined ? pulseComp.amp : (pulseComp.value !== undefined ? pulseComp.value : 1);
  const Ts = pulseComp.Ts > 1e-9 ? pulseComp.Ts : 1;
  const D = (pulseComp.duty !== undefined ? pulseComp.duty : 0.5);
  const Don = Math.max(0, Math.min(1, D)) * Ts;
  // g(t): unit-step response of pulse source alone (others deactivated, IC zeroed)
  const compsG = components.map(c => {
    if (c === pulseComp) return Object.assign({}, c, { wave: 'step', amp: 1, value: 1 });
    if (c.type === 'V' || c.type === 'I') return Object.assign({}, c, { wave: 'step', amp: 0, value: 0 });
    if (c.type === 'C') return Object.assign({}, c, { v0: 0 });
    if (c.type === 'L') return Object.assign({}, c, { i0: 0 });
    return c;
  });
  const gFn = ratTimeFn(probeRat(solveSafe(compsG, { nodeIC: {}, mutuals: mutuals, stateOf: c => c.state }), probe));
  // rest(t): pulse source deactivated, others + IC active
  const compsR = components.map(c => c === pulseComp ? Object.assign({}, c, { wave: 'step', amp: 0, value: 0 }) : c);
  const rFn = ratTimeFn(probeRat(solveSafe(compsR, { nodeIC: nodeIC, mutuals: mutuals, stateOf: c => c.state }), probe));
  const horizon = Math.max(circuitHorizon(components), 5 * Ts);
  const evalAt = t => {
    let v = rFn(t);
    let n = 0;
    while (n * Ts <= t + 1e-12) {
      const t1 = n * Ts, t2 = n * Ts + Don;
      if (t >= t1) v += A * gFn(t - t1);
      if (t >= t2) v -= A * gFn(t - t2);
      n++;
      if (n > 100000) break;
    }
    return v;
  };
  return {
    piecewise: true, Fs: null, result: null,
    evalAt: evalAt, deltas: [], tEndHint: horizon,
    renderHTML: '脈波響應（步階響應疊加）：A=' + fmtNum(A) + ', T<sub>s</sub>=' + fmtNum(Ts) + ', duty=' + fmtNum(D),
    segments: null,
  };
}

/* ---- piecewise: ideal diodes (event detection) + timed/PWM switches + pulse edges ----
 * dt, horizon and segment count auto-scale to the circuit's time constants so
 * µs-scale switching circuits work as well as ~1s circuits. */
function piecewiseAnalyze(components, probe, mutuals, nodeIC, opts) {
  const horizon = (opts && opts.horizon) || circuitHorizon(components);
  const ts = circuitTimeScale(components);
  const dt = (opts && opts.dt) || Math.max(ts.tFast / 20, horizon / 2e5, 1e-15);
  const MAXSEG = (opts && opts.maxSeg) || 4000;
  const caps = components.filter(c => c.type === 'C');
  const inds = components.filter(c => c.type === 'L');
  const diodes = components.filter(c => c.type === 'D');
  const switches = components.filter(c => c.type === 'SW');
  const pulses = components.filter(c => (c.type === 'V' || c.type === 'I') && c.wave === 'pulse');
  const dstate = new Map(diodes.map(d => [d, d.init || 'off']));
  const capVal = new Map(), indVal = new Map();
  for (const c of caps) capVal.set(c, (c.v0 !== undefined && c.v0 !== null) ? c.v0 : ((nodeIC[c.a.x + ',' + c.a.y] || 0) - (nodeIC[c.b.x + ',' + c.b.y] || 0)));
  for (const c of inds) indVal.set(c, c.i0 || 0);
  // PSS hook: seed the state from a supplied vector (caps first, then inductors)
  if (opts && opts.x0) { caps.forEach((c, i) => capVal.set(c, opts.x0[i])); inds.forEach((c, i) => indVal.set(c, opts.x0[caps.length + i])); }

  // Phase of a periodic signal, evaluated JUST AFTER t. The +eps resolves the
  // edge ambiguity at a segment that starts exactly on a period boundary k·T:
  // without it, float error in (t % T) can return ≈T instead of 0, flipping the
  // switch to OPEN during a half-period that should be CLOSED (skipped charge
  // cycles → the converter sags then jumps). eps ≫ float drift, ≪ half-period.
  const phaseOf = (t, T) => { const e = T * 1e-9; return ((((t + e) % T) + T) % T); };
  const pulseLevel = (p, t) => {
    const A = p.amp !== undefined ? p.amp : (p.value !== undefined ? p.value : 1);
    const Ts = p.Ts > 1e-12 ? p.Ts : 1, D = (p.duty !== undefined ? p.duty : 0.5);
    return phaseOf(t, Ts) < D * Ts ? A : 0;
  };
  const swState = (sw, t) => {
    if (sw.mode === 'pwm' && sw.Tsw > 1e-12) {
      let closed = phaseOf(t, sw.Tsw) < (sw.duty !== undefined ? sw.duty : 0.5) * sw.Tsw;
      if (sw.inv) closed = !closed;                       // complementary gate (synchronous rectifier)
      return closed ? 'closed' : 'open';
    }
    let st = sw.state || 'open';
    if (sw.tSwitch > 0 && t >= sw.tSwitch - 1e-12) st = (st === 'open' ? 'closed' : 'open');
    return st;
  };
  const stateOf = (t) => (c => c.type === 'D' ? dstate.get(c) : (c.type === 'SW' ? swState(c, t) : c.state));
  const nextEdgeAfter = (period, duty, t0) => {
    const k = Math.floor((t0 + 1e-12) / period);
    for (const e of [k * period, k * period + duty * period, (k + 1) * period, (k + 1) * period + duty * period]) if (e > t0 + period * 1e-9) return e;
    return Infinity;
  };

  const segs = [];
  let t0 = 0, guard = 0, truncated = false;
  // One segment iteration. Returns true if more segments remain, false when done.
  // Factored out of the while-loop so an async driver can yield to the UI between
  // segments (for a live progress bar) without duplicating the solve logic.
  function step() {
    if (!(t0 < horizon - horizon * 1e-12)) return false;
    if (guard++ >= MAXSEG) { truncated = true; return false; }
    // Build this segment's component snapshot ONCE: carried IC (i0/v0) and the
    // current pulse level are baked into fresh clones of L/C/pulse sources.
    // Extraction below MUST read inductor current / cap voltage through these
    // clones (they hold the right i0/v0) — reading through the originals would
    // drop the inductor's DC current each segment (the IC-handoff bug).
    const segComps = components.map(c => {
      if (c.type === 'C') return Object.assign({}, c, { v0: capVal.get(c) });
      if (c.type === 'L') return Object.assign({}, c, { i0: indVal.get(c) });
      if ((c.type === 'V' || c.type === 'I') && c.wave === 'pulse') { const lv = pulseLevel(c, t0 + dt * 0.5); return Object.assign({}, c, { wave: 'step', amp: lv, value: lv }); }
      return c;
    });
    const cloneOf = new Map(); components.forEach((c, i) => cloneOf.set(c, segComps[i]));
    const cl = c => cloneOf.get(c) || c;
    const segMutuals = mutuals.map(mu => ({ a: cl(mu.a), b: cl(mu.b), M: mu.M }));
    const segProbe = (probe && probe.kind === 'I') ? { kind: 'I', comp: cl(probe.comp) } : probe;
    const eps = Math.min(dt * 0.1, ts.tFast * 0.05);
    let sol, tries = 0;
    while (true) {
      tries++;
      sol = solveSafe(segComps, { t0: t0, nodeIC: nodeIC, mutuals: segMutuals, stateOf: stateOf(t0) });
      let changed = false;
      for (const d of diodes) {
        if (dstate.get(d) === 'on') {
          const bi = sol.branchCurrent(d);                 // current anode→cathode
          if (ratLimInf(bi) < -1e-9 || ratTimeFn(bi)(eps) < -1e-9) { dstate.set(d, 'off'); changed = true; }
        } else {
          const vd = sol.voltageDiff(d.a, d.b);            // anode−cathode voltage
          const vth = (d.Vf || 0) + 1e-9;                  // conducts once forward bias exceeds Vf
          // turn ON if forward-biased at t=0⁺ — including an impulsive forward spike
          // (ratLimInf), which occurs when an inductor's only path is this diode.
          if (ratLimInf(vd) > vth || ratTimeFn(vd)(eps) > vth) { dstate.set(d, 'on'); changed = true; }
        }
      }
      if (!changed) break;                                 // converged: sol matches dstate
      if (tries > 2 * diodes.length + 3) {                 // chattering at a threshold: stop, but
        sol = solveSafe(segComps, { t0: t0, nodeIC: nodeIC, mutuals: segMutuals, stateOf: stateOf(t0) });
        break;                                             // re-solve so sol is consistent with dstate
      }
    }
    // scheduled events: timed/PWM switch toggles and pulse edges strictly after t0
    let nextSched = Infinity;
    for (const sw of switches) {
      if (sw.mode === 'pwm' && sw.Tsw > 1e-12) nextSched = Math.min(nextSched, nextEdgeAfter(sw.Tsw, (sw.duty !== undefined ? sw.duty : 0.5), t0));
      else if (sw.tSwitch > t0 + 1e-12) nextSched = Math.min(nextSched, sw.tSwitch);
    }
    for (const p of pulses) nextSched = Math.min(nextSched, nextEdgeAfter(p.Ts > 1e-12 ? p.Ts : 1, (p.duty !== undefined ? p.duty : 0.5), t0));
    // diode event scan (bounded number of sub-steps per segment)
    const mons = diodes.map(d => dstate.get(d) === 'on'
      ? { d: d, kind: 'on', fn: ratTimeFn(sol.branchCurrent(d)) }
      : { d: d, kind: 'off', fn: ratTimeFn(sol.voltageDiff(d.a, d.b)) });
    let eventT = null, eventDiode = null;
    if (mons.length) {
      const scanEnd = Math.min(horizon, nextSched);
      const span = scanEnd - t0;
      const nstep = Math.min(3000, Math.max(4, Math.ceil(span / dt)));
      const ddt = span / nstep;
      const fired = (mon, v) => mon.kind === 'on' ? v < -1e-9 : v > (mon.d.Vf || 0) + 1e-9;
      const prev = mons.map(mon => mon.fn(0));             // monitor values at segment start (not fired)
      for (let i = 1; i <= nstep; i++) {
        const tau = i * ddt;
        for (let mi = 0; mi < mons.length; mi++) {
          const mon = mons[mi], v = mon.fn(tau);
          if (fired(mon, v) && !fired(mon, prev[mi])) {     // crossing in ((i-1)·ddt, tau]: bisect to it
            let lo = (i - 1) * ddt, hi = tau;
            for (let b = 0; b < 50; b++) { const mid = (lo + hi) / 2; if (fired(mon, mon.fn(mid))) hi = mid; else lo = mid; }
            if (eventT === null || t0 + hi < eventT) { eventT = t0 + hi; eventDiode = mon.d; }
          }
          prev[mi] = v;
        }
        if (eventT !== null) break;
      }
    }
    let kind = 'diode';
    if (eventT === null && isFinite(nextSched) && nextSched <= horizon) { eventT = nextSched; kind = 'sched'; }
    const end = eventT === null ? horizon : eventT;
    const F = ratToReal(probeRat(sol, segProbe));
    segs.push({ t0: t0, t1: end, result: invertRational(F.num, F.den), offset: t0 });
    if (opts && opts.__trace) opts.__trace.push({ t0: t0, t1: end, kind: kind, sw: switches.map(s => swState(s, t0)), ds: diodes.map(d => dstate.get(d)), indIn: inds.map(c => indVal.get(c)), capIn: caps.map(c => capVal.get(c)) });
    if (eventT === null) return false;
    const tau = eventT - t0;
    for (const c of caps) capVal.set(c, ratTimeFn(sol.capVoltage(cl(c)))(tau));
    for (const c of inds) indVal.set(c, ratTimeFn(sol.indCurrent(cl(c)))(tau));
    if (opts && opts.__trace) { const tr = opts.__trace[opts.__trace.length - 1]; tr.indOut = inds.map(c => indVal.get(c)); tr.capOut = caps.map(c => capVal.get(c)); }
    if (kind === 'diode') dstate.set(eventDiode, dstate.get(eventDiode) === 'on' ? 'off' : 'on');
    t0 = eventT;
    return true;
  }
  function finalize() {
    const lastT = segs.length ? segs[segs.length - 1].t1 : horizon;
    const evalAt = t => {
      for (const sg of segs) if (t >= sg.t0 - 1e-12 && t < sg.t1 + Math.abs(sg.t1) * 1e-9 + 1e-15) return evalFormula(sg.result, t - sg.offset);
      const last = segs[segs.length - 1]; return last ? evalFormula(last.result, t - last.offset) : 0;
    };
    let note = '分段解：' + segs.length + ' 段';
    if (truncated) note += '（已達上限 ' + MAXSEG + ' 段，僅顯示到 t≈' + fmtNum(lastT) + 's；切換太快/視窗太長，請調小 t_max 或降低切換頻率）';
    return {
      piecewise: true, Fs: null, result: null, truncated: truncated,
      evalAt: evalAt, deltas: collectDeltas(segs[0] ? segs[0].result : { special: false, blocks: [] }),
      tEndHint: truncated ? lastT : horizon,
      renderHTML: note, segments: segs.map(s => ({ t0: s.t0, t1: s.t1 })),
      stateEnd: caps.map(c => capVal.get(c)).concat(inds.map(c => indVal.get(c))),   // [Vc...; iL...] at end (PSS)
    };
  }
  // progress estimate in [0,1): how far t0 has advanced through the time window
  const progressFrac = () => (horizon > 0 ? Math.min(0.999, Math.max(0, t0 / horizon)) : 0);
  // async driver path: hand back the stepper so the caller can yield between segments
  if (opts && opts.__stepper) return { step: step, finalize: finalize, progressFrac: progressFrac };
  while (step()) { /* run all segments synchronously */ }
  return finalize();
}

/* ---- real (float) linear solve, for the periodic-steady-state fixed point ---- */
function solveLinearReal(A, b) {
  const n = A.length, M = A.map((row, i) => row.slice().concat([b[i]]));
  for (let col = 0; col < n; col++) {
    let piv = col; for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) < 1e-300) throw new Error('PSS 線性系統奇異');
    const tmp = M[col]; M[col] = M[piv]; M[piv] = tmp;
    const pv = M[col][col];
    for (let r = 0; r < n; r++) if (r !== col) { const f = M[r][col] / pv; for (let k = col; k <= n; k++) M[r][k] -= f * M[col][k]; }
  }
  const x = new Array(n);
  for (let i = 0; i < n; i++) x[i] = M[i][n] / M[i][i];      // diagonalised: x_i = rhs_i / pivot_i
  return x;
}

/* fundamental switching period = max of the periodic source/switch periods */
function switchingPeriod(components) {
  const ps = [];
  for (const c of components) {
    if (c.type === 'SW' && c.mode === 'pwm' && c.Tsw > 0) ps.push(c.Tsw);
    if ((c.type === 'V' || c.type === 'I') && c.wave === 'pulse' && c.Ts > 0) ps.push(c.Ts);
  }
  return ps.length ? Math.max.apply(null, ps) : null;
}

/* =====================================================================
 * Periodic steady-state (PSS) via shooting.
 *
 * A switching converter's long startup transient is impractical to reach by
 * piecewise time-stepping (a near-lossless LC tank accumulates error over the
 * hundreds of cycles needed to settle). Instead we solve for the periodic state
 * x* directly: x* is the fixed point of the one-period state map P(x) (the cap
 * voltages & inductor currents after exactly one switching period, starting from
 * x). Newton on F(x)=P(x)−x converges in ONE step for switch-only (affine)
 * circuits and a few steps with diodes — then we display a few clean periods.
 * Returns null if the circuit has no periodic switching or no state. ===== */
function steadyStateAnalyze(components, probe, mutuals, nodeIC, opts) {
  const Tp = switchingPeriod(components);
  const caps = components.filter(c => c.type === 'C'), inds = components.filter(c => c.type === 'L');
  const n = caps.length + inds.length;
  if (!Tp || !isFinite(Tp) || Tp <= 0 || n === 0) return null;
  const periodMap = x => piecewiseAnalyze(components, probe, mutuals, nodeIC,
    Object.assign({}, opts, { horizon: Tp, x0: x, maxSeg: 400, steadyState: false })).stateEnd;
  // Newton shooting on F(x) = P(x) − x
  let x = new Array(n).fill(0);
  let ok = false;
  for (let it = 0; it < 12; it++) {
    const Px = periodMap(x);
    const F = Px.map((v, i) => v - x[i]);
    let fn = 0; for (const v of F) fn += v * v; fn = Math.sqrt(fn);
    if (!isFinite(fn)) return null;                        // map blew up → fall back to transient
    if (fn < 1e-7 * (1 + norm(x))) { ok = true; break; }
    const J = [];                                          // J[i][j] = ∂F_i/∂x_j
    for (let i = 0; i < n; i++) J.push(new Array(n).fill(0));
    for (let j = 0; j < n; j++) {
      const h = 1e-6 * (Math.abs(x[j]) + 1), xp = x.slice(); xp[j] += h;
      const Pp = periodMap(xp);
      for (let i = 0; i < n; i++) J[i][j] = (Pp[i] - Px[i]) / h - (i === j ? 1 : 0);
    }
    let dx; try { dx = solveLinearReal(J, F.map(v => -v)); } catch (e) { return null; }
    let step = 0; for (const v of dx) step += v * v;
    x = x.map((v, i) => v + dx[i]);
    if (Math.sqrt(step) < 1e-10) { ok = true; break; }
  }
  if (!ok) return null;
  // display: a handful of clean steady-state periods starting from x*
  const nP = Math.max(4, Math.min(24, Math.round((opts.horizon || 12 * Tp) / Tp)));
  const r = piecewiseAnalyze(components, probe, mutuals, nodeIC,
    Object.assign({}, opts, { horizon: nP * Tp, x0: x, maxSeg: 4000, steadyState: false }));
  r.renderHTML = '週期穩態解（PSS）：顯示 ' + nP + ' 個切換週期（T=' + fmtNum(Tp) + 's）的穩態波形';
  r.steadyState = true;
  return r;
  function norm(v) { let s = 0; for (const e of v) s += e * e; return Math.sqrt(s); }
}

function analyzeCircuitRaw(components, probe, opts) {
  opts = opts || {};
  const mutuals = resolveMutuals(components, opts.couplings);
  const nodeIC = opts.nodeIC || {};
  const events = hasEvents(components);
  const pulse = components.find(c => (c.type === 'V' || c.type === 'I') && c.wave === 'pulse');
  if (!events) return linearAnalyze(components, probe, mutuals, nodeIC);
  if (opts.steadyState) { const r = steadyStateAnalyze(components, probe, mutuals, nodeIC, opts); if (r) return r; }
  const onlyPulse = pulse && !components.some(c => c.type === 'D') &&
    !components.some(c => c.type === 'SW' && ((c.tSwitch > 0) || (c.mode === 'pwm' && c.Tsw > 0)));
  if (onlyPulse) return pulseAnalyze(components, probe, mutuals, nodeIC, pulse);
  return piecewiseAnalyze(components, probe, mutuals, nodeIC, opts);
}

/* =====================================================================
 * Time-scale normalization.
 *
 * µs-scale switching circuits (Buck etc.) put the poles at ~1e5–1e6, and
 * the root finder / partial-fraction step — tuned on O(1) circuits — loses
 * precision there, producing NaN / non-physical garbage. We fix this with an
 * exact change of variable t = T0·τ:
 *   - in normalized time, C → C/T0, L → L/T0, R unchanged (from i=C·dv/dt etc.)
 *   - source waveforms re-expressed in τ (ω→ω·T0, decay→decay·T0, times→t/T0)
 *   - the solution obeys  v_real(t) = v_norm(t/T0)  EXACTLY (same amplitudes)
 * Choosing T0 = √(t_fast·t_slow) centres the pole spread near 1, so the
 * normalized circuit is well-conditioned. Only markedly ill-scaled circuits
 * are normalized; ~O(1) circuits are left byte-for-byte unchanged.
 * ===================================================================== */
function chooseTimeScale(ts) {
  const T0 = Math.sqrt(ts.tFast * ts.tSlow);
  return (isFinite(T0) && T0 > 0) ? T0 : 1;
}
function normalizeComponents(components, T0) {
  const map = new Map();
  const comps = components.map(c => {
    let n;
    if (c.type === 'L' || c.type === 'C') n = Object.assign({}, c, { value: c.value / T0 });
    else if (c.type === 'V' || c.type === 'I') {
      n = Object.assign({}, c);
      const w = c.wave || 'step';
      if (w === 'impulse') { if (n.amp !== undefined) n.amp = c.amp / T0; if (n.value !== undefined) n.value = c.value / T0; }
      else if (w === 'ramp') { if (n.amp !== undefined) n.amp = c.amp * T0; if (n.value !== undefined) n.value = c.value * T0; }
      else if (w === 'sin' || w === 'cos' || w === 'exp') { if (n.freq !== undefined) n.freq = c.freq * T0; }
      if (w === 'pulse' && n.Ts !== undefined) n.Ts = c.Ts / T0;
    }
    else if (c.type === 'SW') {
      n = Object.assign({}, c);
      if (n.tSwitch) n.tSwitch = c.tSwitch / T0;
      if (n.Tsw) n.Tsw = c.Tsw / T0;
    }
    else n = Object.assign({}, c);
    map.set(c, n);
    return n;
  });
  return { comps: comps, map: map };
}
function unscaleFs(Fs, T0) {
  // H_real(s) = H_norm(p = s·T0): scale coeff of s^k by T0^k (ascending order)
  const sc = arr => arr.map((v, k) => v * Math.pow(T0, k));
  return { num: sc(Fs.num), den: sc(Fs.den) };
}
function wrapNormalizedResult(r, T0) {
  if (T0 === 1) return r;
  const evalN = r.evalAt;
  const out = Object.assign({}, r);
  out.evalAt = t => evalN(t / T0);
  out.tEndHint = r.tEndHint * T0;
  if (r.segments) out.segments = r.segments.map(s => ({ t0: s.t0 * T0, t1: s.t1 * T0 }));
  if (r.Fs) out.Fs = unscaleFs(r.Fs, T0);
  // δ^(n)(τ−a) → T0^(n+1)·δ^(n)(t−a·T0): scale impulse location and strength
  if (r.deltas && r.deltas.length) out.deltas = r.deltas.map(d => Object.assign({}, d, { at: d.at * T0, coeff: d.coeff * Math.pow(T0, (d.order || 0) + 1) }));
  return out;
}
// decide whether/how to normalize; returns the (possibly rescaled) problem + an unwrap fn
function normPrep(components, probe, opts) {
  const ts = circuitTimeScale(components);
  const T0 = chooseTimeScale(ts);
  if (!(T0 > 0 && (T0 < 0.1 || T0 > 10))) return { components: components, probe: probe, opts: opts, T0: 1, wrap: r => r };
  const nc = normalizeComponents(components, T0), map = nc.map;
  const probeN = (probe && probe.kind === 'I') ? { kind: 'I', comp: map.get(probe.comp) || probe.comp, dir: probe.dir } : probe;
  const optsN = Object.assign({}, opts);
  if (opts.couplings) optsN.couplings = opts.couplings.map(c => ({ a: map.get(c.a) || c.a, b: map.get(c.b) || c.b, M: c.M / T0 }));
  if (opts.horizon && opts.horizon > 0) optsN.horizon = opts.horizon / T0;
  return { components: nc.comps, probe: probeN, opts: optsN, T0: T0, wrap: r => wrapNormalizedResult(r, T0) };
}

function analyzeCircuit(components, probe, opts) {
  opts = opts || {};
  const p = normPrep(components, probe, opts);
  return p.wrap(analyzeCircuitRaw(p.components, p.probe, p.opts));
}

/* async variant: yields to the event loop between piecewise segments so the UI
   can paint a live progress bar. opts.onProgress(frac 0..1), opts.yieldTo()→Promise. */
function analyzeCircuitAsync(components, probe, opts) {
  opts = opts || {};
  const onProgress = typeof opts.onProgress === 'function' ? opts.onProgress : function () {};
  const yieldTo = typeof opts.yieldTo === 'function' ? opts.yieldTo : function () { return new Promise(function (res) { setTimeout(res, 0); }); };
  const p = normPrep(components, probe, opts);
  const comps = p.components, prb = p.probe, o = p.opts;
  const mutuals = resolveMutuals(comps, o.couplings);
  const nodeIC = o.nodeIC || {};
  const events = hasEvents(comps);
  const pulse = comps.find(c => (c.type === 'V' || c.type === 'I') && c.wave === 'pulse');
  if (!events) { onProgress(1); return Promise.resolve(p.wrap(linearAnalyze(comps, prb, mutuals, nodeIC))); }
  if (o.steadyState) { const r = steadyStateAnalyze(comps, prb, mutuals, nodeIC, o); if (r) { onProgress(1); return Promise.resolve(p.wrap(r)); } }
  const onlyPulse = pulse && !comps.some(c => c.type === 'D') &&
    !comps.some(c => c.type === 'SW' && ((c.tSwitch > 0) || (c.mode === 'pwm' && c.Tsw > 0)));
  if (onlyPulse) { onProgress(1); return Promise.resolve(p.wrap(pulseAnalyze(comps, prb, mutuals, nodeIC, pulse))); }
  const handle = piecewiseAnalyze(comps, prb, mutuals, nodeIC, Object.assign({}, o, { __stepper: true }));
  return (async function () {
    let i = 0; onProgress(0);
    while (handle.step()) { if ((++i % 3) === 0) { onProgress(handle.progressFrac()); await yieldTo(); } }
    onProgress(1);
    return p.wrap(handle.finalize());
  })();
}

/* =====================================================================
 * Self-test (runs in browser to catch any porting errors)
 * ===================================================================== */
function asC(arr) { return arr.map(v => cx(v)); }
function runSelfTest() {
  const cases = [
    { name: '1/(s+2) → e^{-2t}', N: [1], D: [2, 1], chk: t => Math.exp(-2 * t) },
    { name: 's/(s²+4) → cos2t', N: [0, 1], D: [4, 0, 1], chk: t => Math.cos(2 * t) },
    { name: '2/(s²+4) → sin2t', N: [2], D: [4, 0, 1], chk: t => Math.sin(2 * t) },
    { name: '(s+3)/(s²+2s+5)', N: [3, 1], D: [5, 2, 1], chk: t => Math.exp(-t) * (Math.cos(2 * t) + Math.sin(2 * t)) },
    { name: '1/(s+1)² → t·e^{-t}', N: [1], D: [1, 2, 1], chk: t => t * Math.exp(-t) },
    { name: '1/(s(s+1)) → 1−e^{-t}', N: [1], D: [0, 1, 1], chk: t => 1 - Math.exp(-t) },
    { name: '1/s³ → t²/2', N: [1], D: [0, 0, 0, 1], chk: t => t * t / 2 },
    { name: '6/(s²+9) → 2sin3t', N: [6], D: [9, 0, 1], chk: t => 2 * Math.sin(3 * t) },
    { name: '1/(s+2)³ → t²/2·e^{-2t}', N: [1], D: [8, 12, 6, 1], chk: t => t * t / 2 * Math.exp(-2 * t) },
    { name: '1/(s²+1)² ', N: [1], D: [1, 0, 2, 0, 1], chk: t => 0.5 * (Math.sin(t) - t * Math.cos(t)) },
  ];
  const results = [];
  for (const c of cases) {
    try {
      const res = inverseLaplaceRational(asC(c.N), asC(c.D));
      const td = toTimeDomain(res);
      let maxerr = 0;
      for (const t of [0.3, 0.8, 1.5, 2.7, 4.0]) maxerr = Math.max(maxerr, Math.abs(evalG(td, t) - c.chk(t)));
      results.push({ name: c.name, ok: maxerr < 1e-6, err: maxerr });
    } catch (e) { results.push({ name: c.name, ok: false, err: NaN, error: e.message }); }
  }
  // circuit test: RC lowpass step, RC=1 -> 1-e^{-t}
  try {
    const comps = [
      { type: 'V', a: { x: 0, y: 0 }, b: { x: 0, y: 1 }, wave: 'step', value: 1 },
      { type: 'R', a: { x: 0, y: 0 }, b: { x: 1, y: 0 }, value: 2 },
      { type: 'C', a: { x: 1, y: 0 }, b: { x: 1, y: 1 }, value: 0.5 },
      { type: 'W', a: { x: 0, y: 1 }, b: { x: 1, y: 1 } },
      { type: 'GND', a: { x: 0, y: 1 } },
    ];
    const sol = solveCircuit(comps);
    const F = ratToReal(sol.voltageAt({ x: 1, y: 0 }));
    const res = inverseLaplaceRational(asC(F.num), asC(F.den));
    const td = toTimeDomain(res);
    let maxerr = 0;
    for (const t of [0.3, 1, 2, 4]) maxerr = Math.max(maxerr, Math.abs(evalG(td, t) - (1 - Math.exp(-t))));
    results.push({ name: '電路 RC 低通階躍 → 1−e^{-t}', ok: maxerr < 1e-6, err: maxerr });
  } catch (e) { results.push({ name: '電路 RC 低通', ok: false, err: NaN, error: e.message }); }
  return results;
}

/* ---------- exports ---------- */
const API = {
  C, cx, cexp, pmul, padd, psub, peval, ptrim, rootsOf, reduceND,
  partialFractions, inverseLaplaceRational, toTimeDomain, evalG,
  Rat, parseExpr, invertFormula, invertRational, polesOf, evalFormula, collectDeltas, renderFormula,
  formatTD, formatPolyReal, fmtNum, specialTransform,
  solveCircuit, ratToReal, sourceS, sourceLaplace, admittance, runSelfTest,
  analyzeCircuit, analyzeCircuitAsync, tEndForResult, circuitHorizon, circuitTimeScale, ratTimeFn,
};
global.LaplaceEngine = API;
if (typeof module !== 'undefined' && module.exports) module.exports = API;

})(typeof window !== 'undefined' ? window : globalThis);
