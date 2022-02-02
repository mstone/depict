import cvxpy as cp
l = cp.Variable(2, pos=True)
r = cp.Variable(2, pos=True)
s = cp.Variable(2, pos=True)
t = cp.Variable(2, pos=True)
# eps = cp.Constant(0.05)
eps = cp.Variable(pos=True)
constr = [
    r[0] >= l[0], 
    r[1] >= l[1],

    s[0] >= l[0] + eps,
    s[1] >= s[0] + eps,
    s[1] <= r[0] - eps,

    t[0] >= l[1] + eps,
    t[1] >= t[0] + eps,
    t[1] <= r[1] - eps,

    r[0] <= 1.0, 
    r[1] <= 0.5,
    ]
# obj = cp.square(s[0] - t[0]) + cp.square(s[1] - t[1]) + cp.square(s[0] - 0.25 * r[0]) + cp.square(s[1] - 0.75 * r[1])
obj = 100 * cp.square(s[0] - t[0]) + 100 * cp.square(s[1] - t[1]) + cp.square(r[1] - s[1]) + cp.square(s[1] - s[0]) + cp.square(s[0] - l[0]) - 1000 * eps
# obj = cp.square(s[0] - t[0]) + cp.square(s[1] - t[1]) - eps
# obj = t[0] - s[0] + t[1] - s[1] - eps
prb = cp.Problem(cp.Minimize(obj), constr)
assert prb.is_dcp()
prb.solve()
print(f"l: {l.value}\nr: {r.value}\ns: {s.value}\nt: {t.value}\nε: {eps.value}\n")
f = [(v-l.value[i]) / (r.value[i] - l.value[i])  for (i,v) in enumerate(s.value)]
g = [(v-l.value[i]) / (r.value[i] - l.value[i])  for (i,v) in enumerate(t.value)]
print(f"f: {f}\ng: {g}\n")