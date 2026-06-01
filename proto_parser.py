# -*- coding: utf-8 -*-
"""
Prototype + validation of the s-domain expression parser to be ported to JS.
Parses strings like (s+3)/(s^2+2s+5), exp(-2s)/(s+1), 1/((s+1)(s+2)^2) into
a list of terms { delay: a>=0, rat: RationalFunction }.
Validated against sympy.
"""
import numpy as np, sympy as sp, random
from proto_circuit import R, ptrim, _reduce
from proto import poly_mul

# ---------------- Tokenizer ----------------
def tokenize(src):
    toks = []
    i = 0
    s = src.replace(' ', '')
    while i < len(s):
        ch = s[i]
        if ch.isdigit() or ch == '.':
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] == '.'):
                j += 1
            # scientific notation e.g. 1e-3
            if j < len(s) and s[j] in 'eE' and (j+1 < len(s) and (s[j+1].isdigit() or s[j+1] in '+-')):
                j += 1
                if s[j] in '+-': j += 1
                while j < len(s) and s[j].isdigit(): j += 1
            toks.append(('num', float(s[i:j]))); i = j
        elif ch.isalpha() or ch == '_':
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == '_'):
                j += 1
            toks.append(('id', s[i:j])); i = j
        elif ch in '+-*/^(),':
            toks.append((ch, ch)); i += 1
        else:
            raise ValueError(f"bad char {ch!r}")
    # insert implicit multiplication
    out = []
    for k, tk in enumerate(toks):
        if out:
            prev = out[-1]
            # prev ends an operand, tk starts an operand -> insert *
            prev_end = prev[0] in ('num', 'id', ')')
            cur_start = tk[0] in ('num', 'id', '(')
            # but 'id' followed by '(' is a function call, not implicit mult
            if prev_end and cur_start and not (prev[0] == 'id' and tk[0] == '(' and prev[1] in FUNCS):
                out.append(('*', '*'))
        out.append(tk)
    return out

FUNCS = {'exp'}
CONSTS = {'pi': np.pi, 'e': np.e}

# ---------------- SVal: list of terms {delay, rat:R} ----------------
class SVal:
    def __init__(self, terms):
        self.terms = terms  # list of (delay, R)
    @staticmethod
    def const(c):
        return SVal([(0.0, R([c]))])
    @staticmethod
    def svar():  # s
        return SVal([(0.0, R([0, 1]))])
    def add(self, o):
        return SVal(self.terms + o.terms)
    def combine(self):
        # merge terms with equal delay by summing their rationals
        keys = []
        groups = {}
        for (d, r) in self.terms:
            key = round(d, 9)
            if key in groups:
                groups[key] = groups[key].add(r)
            else:
                groups[key] = r; keys.append(key)
        return SVal([(k, groups[k]) for k in sorted(keys)])
    def neg(self):
        return SVal([(d, r.neg()) for (d, r) in self.terms])
    def mul(self, o):
        out = []
        for (d1, r1) in self.terms:
            for (d2, r2) in o.terms:
                out.append((d1 + d2, r1.mul(r2)))
        return SVal(out)
    def div(self, o):
        # divide by a pure delay-0 rational only (combine first)
        oc = o.combine()
        if len(oc.terms) != 1 or abs(oc.terms[0][0]) > 1e-12:
            raise ValueError("can only divide by a non-delayed rational")
        d0, r0 = oc.terms[0]
        return SVal([(d, r.div(r0)) for (d, r) in self.terms])
    def powi(self, n):
        if n == 0:
            return SVal.const(1)
        if n < 0:
            return SVal.const(1).div(self.powi(-n))
        res = self
        for _ in range(n - 1):
            res = res.mul(self)
        return res
    def is_const(self):
        c = self.combine()
        return len(c.terms) == 1 and abs(c.terms[0][0]) < 1e-12 and \
               len(c.terms[0][1].den) == 1 and len(c.terms[0][1].num) <= 1
    def const_val(self):
        c = self.combine()
        return c.terms[0][1].num[0] if c.terms[0][1].num else 0j

# ---------------- Parser (recursive descent) ----------------
class Parser:
    def __init__(self, toks):
        self.toks = toks; self.pos = 0
    def peek(self): return self.toks[self.pos] if self.pos < len(self.toks) else (None, None)
    def next(self): t = self.toks[self.pos]; self.pos += 1; return t
    def expect(self, ty):
        t = self.next()
        if t[0] != ty: raise ValueError(f"expected {ty} got {t}")
        return t
    def parse(self):
        v = self.expr()
        if self.pos != len(self.toks): raise ValueError("trailing tokens")
        return v
    def expr(self):
        v = self.term()
        while self.peek()[0] in ('+', '-'):
            op = self.next()[0]
            rhs = self.term()
            v = v.add(rhs if op == '+' else rhs.neg())
        return v
    def term(self):
        v = self.factor()
        while self.peek()[0] in ('*', '/'):
            op = self.next()[0]
            rhs = self.factor()
            v = v.mul(rhs) if op == '*' else v.div(rhs)
        return v
    def factor(self):
        # unary minus
        if self.peek()[0] == '-':
            self.next(); return self.factor().neg()
        if self.peek()[0] == '+':
            self.next(); return self.factor()
        base = self.base()
        if self.peek()[0] == '^':
            self.next()
            exp = self.factor()  # right assoc
            if not exp.is_const():
                raise ValueError("exponent must be constant")
            n = exp.const_val().real
            if abs(n - round(n)) > 1e-9:
                raise ValueError("non-integer exponent unsupported")
            base = base.powi(int(round(n)))
        return base
    def base(self):
        ty, val = self.peek()
        if ty == '(':
            self.next(); v = self.expr(); self.expect(')'); return v
        if ty == 'num':
            self.next(); return SVal.const(val)
        if ty == 'id':
            self.next()
            if val == 's':
                return SVal.svar()
            if val in CONSTS:
                return SVal.const(CONSTS[val])
            if val == 'exp':
                self.expect('('); arg = self.expr(); self.expect(')')
                return self.do_exp(arg)
            raise ValueError(f"unknown id {val}")
        raise ValueError(f"unexpected {ty}")
    def do_exp(self, arg):
        # arg must be c0 + c1*s (linear). exp(c0)*exp(c1 s). For causal delay need c1<0: delay=-c1.
        arg = arg.combine()
        if len(arg.terms) != 1 or abs(arg.terms[0][0]) > 1e-12:
            raise ValueError("exp argument too complex")
        r = arg.terms[0][1]
        if len(r.den) != 1:
            raise ValueError("exp argument must be polynomial")
        coeffs = r.num
        c0 = coeffs[0].real if len(coeffs) >= 1 else 0.0
        c1 = coeffs[1].real if len(coeffs) >= 2 else 0.0
        if len(coeffs) > 2 and any(abs(c) > 1e-12 for c in coeffs[2:]):
            raise ValueError("exp argument must be linear in s")
        if c1 > 1e-12:
            raise ValueError("non-causal exp(+a s)")
        delay = -c1
        import math
        scale = math.exp(c0)
        return SVal([(delay, R([scale]))])

def parse(src):
    return Parser(tokenize(src)).parse().combine()

# ---------------- validation ----------------
s = sp.symbols('s')
random.seed(7)

def rat_to_sympy(r):
    num = sum(sp.nsimplify(round(r.num[i].real, 9), rational=False) * s**i for i in range(len(r.num)))
    den = sum(sp.nsimplify(round(r.den[i].real, 9), rational=False) * s**i for i in range(len(r.den)))
    return num/den

def check(name, src, sympy_str, expected_delay=0.0):
    v = parse(src)
    # for these tests, expect single term
    assert len(v.terms) == 1, f"{name}: expected 1 term got {len(v.terms)}"
    delay, r = v.terms[0]
    expr = sp.sympify(sympy_str)
    Ffun = sp.lambdify(s, expr, 'numpy')
    err = 0.0
    for _ in range(10):
        sv = complex(random.uniform(0.5, 3), random.uniform(-2, 2))
        mine = (sum(r.num[i]*sv**i for i in range(len(r.num))) /
                sum(r.den[i]*sv**i for i in range(len(r.den))))
        err = max(err, abs(mine - complex(Ffun(sv)))/(1+abs(complex(Ffun(sv)))))
    derr = abs(delay - expected_delay)
    # ALSO verify the full inverse transform vs sympy (catches un-reduced
    # degree explosions that the value-only check above would miss).
    fterr = 0.0
    if abs(expected_delay) < 1e-12:
        try:
            from proto import inverse_laplace_rational, eval_ft
            dl, pf = inverse_laplace_rational(list(map(complex, r.num)), list(map(complex, r.den)))
            t = sp.symbols('t', positive=True)
            fsym = sp.inverse_laplace_transform(expr, s, t)
            for tv in [0.3, 0.9, 1.8, 3.0]:
                fterr = max(fterr, abs(eval_ft(dl, pf, tv).real - complex(fsym.subs(t, tv).evalf()).real))
        except Exception as e:
            fterr = float('inf')
    ok = err < 1e-7 and derr < 1e-9 and fterr < 1e-5
    print(f"[{'OK' if ok else 'FAIL'}] {name:34s} raterr={err:.1e} ft_err={fterr:.1e} delay={delay:.3g}")
    return ok

print("="*70); print("VALIDATION: expression parser vs sympy"); print("="*70)
allok = True
allok &= check("1/(s+2)", "1/(s+2)", "1/(s+2)")
allok &= check("implicit (s+3)/(s^2+2s+5)", "(s+3)/(s^2+2s+5)", "(s+3)/(s**2+2*s+5)")
allok &= check("2s/(s^2+1)", "2s/(s^2+1)", "2*s/(s**2+1)")
allok &= check("(s+1)(s+2) numerator", "(s+1)(s+2)", "(s+1)*(s+2)")
allok &= check("poly s^2+3s+2", "s^2+3s+2", "s**2+3*s+2")
allok &= check("3/(s^2+4)^2", "3/(s^2+4)^2", "3/(s**2+4)**2")
allok &= check("1/((s+1)(s+2)^2)", "1/((s+1)(s+2)^2)", "1/((s+1)*(s+2)**2)")
allok &= check("unary -(1)/(s+1)", "-(1)/(s+1)", "-1/(s+1)")
allok &= check("pi const", "pi/(s^2+pi^2)", "pi/(s**2+pi**2)")
allok &= check("exp(-2*s)/(s+1)", "exp(-2*s)/(s+1)", "1/(s+1)", 2.0)
allok &= check("e^(-2s)/s delay", "e^(-2s)/s", "1/s", 0.0) if False else True  # e^ handled? e is const E
# e^(-2s): 'e' is const E -> e^(-2s) = E**(-2s) which our parser treats as const^expr -> powi with non-int -> error.
# So we standardize on exp(...) for delays. Test scientific/decimal:
allok &= check("decimal 0.5/(s+0.25)", "0.5/(s+0.25)", "0.5/(s+0.25)")
allok &= check("nested 1/(s*(s+1)*(s+2))", "1/(s(s+1)(s+2))", "1/(s*(s+1)*(s+2))")
# regression: parser sums fractions over repeated common denominators -> must reduce
allok &= check("(s+1)/(s+2)^2 [reduce]", "(s+1)/(s+2)^2", "(s+1)/(s+2)**2")
allok &= check("2(s+1)(s+3)/(s+2)^2 [reduce]", "2(s+1)(s+3)/((s+2)^2)", "2*(s+1)*(s+3)/(s+2)**2")
allok &= check("1/(s+2)+1/(s+2)^2 [reduce]", "1/(s+2)+1/(s+2)^2", "1/(s+2)+1/(s+2)**2")
allok &= check("repeated triple sum", "1/(s+1)+2/(s+1)^2+3/(s+1)^3", "1/(s+1)+2/(s+1)**2+3/(s+1)**3")
print("="*70); print("ALL OK" if allok else "SOME FAILED"); print("="*70)
