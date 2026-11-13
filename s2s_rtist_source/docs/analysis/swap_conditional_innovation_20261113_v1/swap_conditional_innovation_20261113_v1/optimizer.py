import numpy as np
from scipy.special import logsumexp
RADIUS, REG, TAU, PG_TOL, MAX_STEPS = .25, .01, .05, 1e-6, 100000
KINDS=("static","feedback")
def need(ok,msg):
    if not ok:raise ValueError(msg)

def project(theta, radius=RADIUS):
    return theta * np.minimum(1., radius / np.maximum(np.linalg.norm(theta, axis=1, keepdims=True), 1e-300))


class Objective:
    def __init__(self, T, features, off, target, weights, kind, radius):
        need(kind in KINDS, 'Fit kind')
        self.kind = kind
        self.radius = float(radius)
        need(0 < self.radius <= RADIUS, 'Radius domain')
        self.x = np.asarray(features, float)
        need(self.x.ndim == 2 and np.max(np.linalg.norm(self.x, axis=1)) <= 1+1e-12, 'Feature ball')
        self.shape = (6, self.x.shape[1])
        self.D = np.einsum('nak,nj->nakj', T, self.x).reshape(len(features), 8, -1)
        self.off, self.target, self.w = np.asarray(off), np.asarray(target), np.asarray(weights)
        need(self.off.shape == self.target.shape == (len(features), 8) and self.w.shape == (len(features),), 'Objective shape')
        need(all(np.isfinite(a).all() for a in [self.D, self.off, self.target, self.w]), 'Objective finite')
        need(np.all(self.w > 0) and abs(self.w.sum() - 1) < 1e-12, 'Episode weights')
        self.mu = 2 * REG
        norms = np.sum(self.D ** 2, axis=2)
        if kind.endswith('mse'):
            # Exact quadratic acceleration; L is the safe Hessian trace bound.
            self.H = 2 * np.einsum('n,nai,naj->ij', self.w / 8, self.D, self.D) + self.mu * np.eye(self.D.shape[-1])
            self.b = 2 * np.einsum('n,nai,na->i', self.w / 8, self.D, self.off - self.target)
            self.c = float(np.sum(self.w[:, None] * (self.off - self.target) ** 2) / 8)
            self.L = float(2 * (self.w * norms.mean(1)).sum() + self.mu)
        else:
            # H = (4/tau) D' Cov(softmax) D; <= (4/tau) max_a ||D_a||^2 I.
            self.L = float(4 / TAU * (self.w * norms.max(1)).sum() + self.mu)
            self.best = self.target.argmax(1)
        self.beta = (np.sqrt(self.L) - np.sqrt(self.mu)) / (np.sqrt(self.L) + np.sqrt(self.mu))

    def __call__(self, theta):
        flat = theta.ravel()
        if self.kind.endswith('mse'):
            g = self.H @ flat + self.b
            f = .5 * flat @ self.H @ flat + self.b @ flat + self.c
        else:
            pred = self.off + self.D @ flat
            u = (2 * pred - self.target) / TAU
            ls = logsumexp(u, axis=1)
            f = np.sum(self.w * (TAU * ls - 2 * pred[np.arange(len(pred)), self.best] + self.target[np.arange(len(pred)), self.best])) + REG * np.sum(flat ** 2)
            q = np.exp(u - ls[:, None]); q[np.arange(len(q)), self.best] -= 1
            g = 2 * np.einsum('n,na,nak->k', self.w, q, self.D) + self.mu * flat
        return float(f), g.reshape(self.shape)

    def mapping(self, theta, gradient):
        return self.L * (theta - project(theta - gradient / self.L, self.radius))


def optimize(obj, binding, resume=None, checkpoint=None, max_steps=MAX_STEPS):
    initial = np.zeros(obj.shape)
    x, y, it = initial.copy(), initial.copy(), 0
    initial_f = obj(initial)[0]
    if resume is not None:
        need(resume['binding'].item() == binding and resume['kind'].item() == obj.kind, 'Checkpoint input binding')
        need(float(resume['L']) == obj.L and float(resume['beta']) == obj.beta and float(resume['radius']) == obj.radius, 'Checkpoint optimizer binding')
        x, y, it = resume['theta'].copy(), resume['momentum'].copy(), int(resume['iteration'])
        need(x.shape == y.shape == obj.shape and np.isfinite(x).all() and np.isfinite(y).all(), 'Checkpoint state')
        need(0 <= it <= MAX_STEPS and np.max(np.linalg.norm(x, axis=1)) <= obj.radius + 1e-12, 'Checkpoint feasible state')
    start_it = it
    def emit():
        f, g = obj(x)
        state = dict(theta=x.copy(), momentum=y.copy(), iteration=np.asarray(it), binding=np.asarray(binding), kind=np.asarray(obj.kind),
                     radius=np.asarray(obj.radius), L=np.asarray(obj.L), beta=np.asarray(obj.beta), objective=np.asarray(f), gradient=g,
                     projected_gradient=obj.mapping(x, g), initial_objective=np.asarray(initial_f))
        if checkpoint:
            checkpoint(state)
        return state
    emit()
    while True:
        f, g = obj(x)
        pg = float(np.max(abs(obj.mapping(x, g))))
        need(np.isfinite(f) and np.isfinite(g).all(), 'Nonfinite optimizer')
        if pg <= PG_TOL or it >= max_steps:
            break
        _, gy = obj(y)
        nxt = project(y - gy / obj.L, obj.radius)
        y = nxt + obj.beta * (nxt - x)
        x = nxt; it += 1
        if it % 100 == 0:
            emit()
    state = emit()
    accepted = pg <= PG_TOL
    report = dict(accepted=accepted, kind=obj.kind, iterations=it, resumed_iteration=start_it, initial_objective=initial_f,
                  final_objective=f, projected_gradient_inf=pg, threshold=PG_TOL, L=obj.L, mu=obj.mu, beta=obj.beta,
                  row_norm_max=float(np.max(np.linalg.norm(x, axis=1))), binding=binding,
                  objective='normalized_gain_mse' if obj.kind.endswith('mse') else 'smooth_SPO_plus', tau=TAU,
                  regularization=REG, radius=obj.radius, solver='strongly_convex_accelerated_projected_gradient')
    return x, report, state
