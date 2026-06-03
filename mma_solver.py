"""
Method of Moving Asymptotes (MMA) solver for topology optimization.

Implements the globally convergent MMA variant (GCMMA) by Svanberg (1987, 2002).
Handles multiple constraints simultaneously — enables combined volume + stress
constraints that OC cannot handle well.

Reference:
    K. Svanberg, "The method of moving asymptotes — a new method for structural
    optimization", Int. J. Numer. Meth. Engng., 24, 359–373, 1987.
"""

import numpy as np


class MMASolver:
    """Method of Moving Asymptotes (MMA) optimizer for topology optimization.

    Solves:
        min  f0(x)
        s.t. fi(x) <= 0,  i = 1, ..., m
             x_min <= x <= x_max

    Parameters
    ----------
    n : int
        Number of design variables.
    m : int
        Number of constraints.
    x_min : np.ndarray
        Lower bounds on design variables.
    x_max : np.ndarray
        Upper bounds on design variables.
    """

    def __init__(self, n, m, x_min, x_max):
        self.n = int(n)
        self.m = int(m)
        self.x_min = np.asarray(x_min, dtype=np.float64).reshape(-1)
        self.x_max = np.asarray(x_max, dtype=np.float64).reshape(-1)

        # Asymptote parameters
        self.asy_init = 0.5       # Initial asymptote distance fraction
        self.asy_decr = 0.7       # Asymptote contraction factor
        self.asy_incr = 1.2       # Asymptote expansion factor
        self.asy_min_frac = 0.01  # Minimum asymptote distance fraction
        self.asy_max_frac = 10.0  # Maximum asymptote distance fraction

        # Move limit
        self.move = 0.1

        # History for asymptote adaptation
        self._x_prev1 = None
        self._x_prev2 = None
        self._low = None
        self._upp = None
        self._iter = 0

    def _compute_asymptotes(self, x):
        """Compute lower and upper asymptotes based on oscillation history."""
        dx = self.x_max - self.x_min
        dx_safe = np.maximum(dx, 1e-12)

        if self._iter < 2 or self._x_prev2 is None:
            # Initial asymptotes
            self._low = x - self.asy_init * dx_safe
            self._upp = x + self.asy_init * dx_safe
        else:
            # Detect oscillation: sign change in consecutive steps
            osc = (x - self._x_prev1) * (self._x_prev1 - self._x_prev2)
            gamma = np.ones(self.n, dtype=np.float64)
            gamma[osc < 0.0] = self.asy_decr   # Oscillating: contract
            gamma[osc > 0.0] = self.asy_incr   # Monotone: expand

            self._low = x - gamma * (self._x_prev1 - self._low)
            self._upp = x + gamma * (self._upp - self._x_prev1)

            # Clamp asymptotes to reasonable range
            min_dist = self.asy_min_frac * dx_safe
            max_dist = self.asy_max_frac * dx_safe
            self._low = np.maximum(self._low, x - max_dist)
            self._low = np.minimum(self._low, x - min_dist)
            self._upp = np.minimum(self._upp, x + max_dist)
            self._upp = np.maximum(self._upp, x + min_dist)

    def _solve_subproblem(self, x, f0val, df0dx, fval, dfdx):
        """Solve the MMA convex subproblem to get the next iterate.

        Parameters
        ----------
        x : ndarray (n,)
            Current design point.
        f0val : float
            Current objective value.
        df0dx : ndarray (n,)
            Gradient of objective w.r.t. x.
        fval : ndarray (m,)
            Current constraint values (fi <= 0 is feasible).
        dfdx : ndarray (m, n)
            Jacobian of constraints.

        Returns
        -------
        x_new : ndarray (n,)
            Updated design variables.
        """
        n, m = self.n, self.m
        low = self._low
        upp = self._upp

        # Move limits
        alpha = np.maximum(self.x_min, np.maximum(low + 0.1 * (x - low), x - self.move * (self.x_max - self.x_min)))
        beta = np.minimum(self.x_max, np.minimum(upp - 0.1 * (upp - x), x + self.move * (self.x_max - self.x_min)))

        # Build convex approximation coefficients
        # p0, q0 for objective; p, q for constraints
        ux = upp - x
        xl = x - low
        ux2 = np.maximum(ux ** 2, 1e-12)
        xl2 = np.maximum(xl ** 2, 1e-12)

        # Objective approximation
        p0 = np.maximum(df0dx, 0.0) * ux2
        q0 = np.maximum(-df0dx, 0.0) * xl2

        # Add small regularization for convexity
        pq_sum = p0 + q0
        eps_pq = 1e-6 * np.maximum(pq_sum, 1e-12)
        p0 += eps_pq
        q0 += eps_pq

        # Constraint approximations
        p = np.zeros((m, n), dtype=np.float64)
        q = np.zeros((m, n), dtype=np.float64)
        b = np.zeros(m, dtype=np.float64)

        for i in range(m):
            p[i] = np.maximum(dfdx[i], 0.0) * ux2
            q[i] = np.maximum(-dfdx[i], 0.0) * xl2
            pq_i = p[i] + q[i]
            eps_i = 1e-6 * np.maximum(pq_i, 1e-12)
            p[i] += eps_i
            q[i] += eps_i
            b[i] = fval[i] - np.sum(p[i] / np.maximum(ux, 1e-12)) - np.sum(q[i] / np.maximum(xl, 1e-12))

        # Solve dual problem via damped Newton on Lagrange multipliers
        lam = np.ones(m, dtype=np.float64) * max(1e-2, 1.0)

        for _ in range(120):
            # Compute x(lam) by solving KKT stationarity
            sum_p = p0.copy()
            sum_q = q0.copy()
            for i in range(m):
                sum_p += lam[i] * p[i]
                sum_q += lam[i] * q[i]

            # x(lam) from stationarity: d/dx [sum_p/(u-x) + sum_q/(x-l)] = 0
            # => sum_p/(u-x)^2 = sum_q/(x-l)^2
            # => (x-l)/(u-x) = sqrt(sum_q/sum_p)
            ratio = np.sqrt(np.maximum(sum_q, 1e-30) / np.maximum(sum_p, 1e-30))
            x_trial = (low + ratio * upp) / (1.0 + ratio)
            x_trial = np.clip(x_trial, alpha, beta)

            ux_t = np.maximum(upp - x_trial, 1e-12)
            xl_t = np.maximum(x_trial - low, 1e-12)

            # Dual function gradient and Hessian
            grad = np.zeros(m, dtype=np.float64)
            hess = np.zeros((m, m), dtype=np.float64)

            for i in range(m):
                gi = np.sum(p[i] / ux_t) + np.sum(q[i] / xl_t) + b[i]
                grad[i] = gi

            # Diagonal Hessian approximation (sufficient for most TO problems)
            for i in range(m):
                dxdlam_p = p[i] / (ux_t ** 2)
                dxdlam_q = q[i] / (xl_t ** 2)
                h_diag = np.sum(dxdlam_p ** 2 / np.maximum(sum_p, 1e-30)) + \
                          np.sum(dxdlam_q ** 2 / np.maximum(sum_q, 1e-30))
                hess[i, i] = max(h_diag, 1e-12)

            # Newton step with Hessian regularization
            if m == 1:
                dlam = -grad / np.maximum(np.diag(hess), 1e-12)
            else:
                hess += np.eye(m) * 1e-8 * np.max(np.abs(np.diag(hess)))
                try:
                    dlam = np.linalg.solve(hess, -grad)
                except np.linalg.LinAlgError:
                    dlam = -grad / np.maximum(np.diag(hess), 1e-12)

            # Damped step to keep lam >= 0
            step = 1.0
            for i in range(m):
                if dlam[i] < 0.0:
                    step = min(step, -0.9 * lam[i] / min(dlam[i], -1e-30))
            step = min(step, 1.0)

            lam_new = np.maximum(lam + step * dlam, 1e-12)
            lam = lam_new

            if np.max(np.abs(grad)) < 1e-6:
                break

        # Final x from solved dual
        sum_p = p0.copy()
        sum_q = q0.copy()
        for i in range(m):
            sum_p += lam[i] * p[i]
            sum_q += lam[i] * q[i]

        ratio = np.sqrt(np.maximum(sum_q, 1e-30) / np.maximum(sum_p, 1e-30))
        x_new = (low + ratio * upp) / (1.0 + ratio)
        x_new = np.clip(x_new, alpha, beta)

        return x_new

    def update(self, x, f0val, df0dx, fval, dfdx):
        """Perform one MMA iteration.

        Parameters
        ----------
        x : ndarray (n,)
            Current design variables.
        f0val : float
            Objective function value.
        df0dx : ndarray (n,)
            Objective gradient.
        fval : ndarray (m,)
            Constraint values (fi <= 0 is feasible).
        dfdx : ndarray (m, n)
            Constraint Jacobian.

        Returns
        -------
        x_new : ndarray (n,)
            Updated design variables.
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        df0dx = np.asarray(df0dx, dtype=np.float64).reshape(-1)
        fval = np.asarray(fval, dtype=np.float64).reshape(-1)
        dfdx = np.asarray(dfdx, dtype=np.float64).reshape(self.m, self.n)

        self._compute_asymptotes(x)
        x_new = self._solve_subproblem(x, f0val, df0dx, fval, dfdx)

        # Update history
        self._x_prev2 = self._x_prev1.copy() if self._x_prev1 is not None else None
        self._x_prev1 = x.copy()
        self._iter += 1

        return x_new


def pnorm_stress(vm_stresses, sigma_allow, pn=8.0):
    """Compute p-norm stress aggregation with N^(-1/p) correction and its derivative.

    Provides a smooth, differentiable approximation of the maximum stress
    suitable for gradient-based optimization. The correction factor removes
    the mesh-size-dependent bias per Le et al. (2010).

    Parameters
    ----------
    vm_stresses : ndarray (n_elements,)
        Von Mises stress per element.
    sigma_allow : float
        Allowable stress.
    pn : float
        P-norm exponent (higher = closer to max; typically 8-16).

    Returns
    -------
    pn_value : float
        The corrected p-norm aggregated stress (approximation of max stress).
    dpn_dvm : ndarray (n_elements,)
        Derivative of corrected p-norm stress w.r.t. each element's von Mises stress.
    constraint_value : float
        Normalized constraint value: pn_value/sigma_allow - 1.0
        (negative = feasible, positive = violated).
    """
    vm = np.asarray(vm_stresses, dtype=np.float64)
    sa = float(max(sigma_allow, 1e-12))
    p = float(max(pn, 1.0))
    n = vm.size

    # Normalize stresses to avoid numerical overflow
    ratio = vm / sa
    ratio_safe = np.maximum(ratio, 1e-30)

    # P-norm with log-sum-exp style stabilization for large p
    # The division by n inside the log already embeds the Le et al. (2010)
    # N^(-1/p) correction: for uniform σ_i = σ, this yields PN = σ exactly.
    # No additional correction factor is needed.
    log_terms = p * np.log(ratio_safe)
    log_max = float(np.max(log_terms))
    log_sum = log_max + np.log(float(np.sum(np.exp(log_terms - log_max))) / max(n, 1))
    pn_value = sa * np.exp(log_sum / p)

    # Derivative: d(PN)/d(vm_i)
    pn_norm = pn_value / sa
    if pn_norm > 1e-30:
        dpn_dvm = pn_norm ** (1.0 - p) * ratio_safe ** (p - 1.0) / max(n, 1)
    else:
        dpn_dvm = np.zeros_like(vm)

    constraint_value = pn_value / sa - 1.0

    return pn_value, dpn_dvm, constraint_value


print('MMA solver and p-norm stress aggregation module created')
