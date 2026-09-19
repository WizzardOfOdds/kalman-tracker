"""Kalman filters for a constant-velocity target.

State x = [px, py, vx, vy]. The motion model says velocity stays constant,
and anything else (turns, gusts) is treated as random acceleration with
spectral density `q`. That is the one tuning knob: small `q` trusts the
model and smooths hard, large `q` trusts the measurements and follows turns.

Two measurement models:
  * CartesianKF: the radar's range and bearing are converted to x, y first,
    then a plain linear Kalman filter runs.
  * RadarEKF: the filter takes range and bearing directly and linearises
    the measurement function around its current estimate.
"""

from __future__ import annotations

import numpy as np


def cv_model(dt: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Transition matrix F and process noise Q for white-noise acceleration."""
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    q1 = np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]]) * q
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] = q1
    Q[np.ix_([1, 3], [1, 3])] = q1
    return F, Q


def polar_to_xy(z: np.ndarray, range_sd: float, bearing_sd: float) -> tuple[np.ndarray, np.ndarray]:
    """Convert one range/bearing measurement to x, y with its covariance.

    The covariance is not a circle. It is long across the line of sight,
    because a small bearing error at long range is many metres sideways.
    """
    r, b = z
    xy = np.array([r * np.cos(b), r * np.sin(b)])
    J = np.array([[np.cos(b), -r * np.sin(b)], [np.sin(b), r * np.cos(b)]])
    R = J @ np.diag([range_sd**2, bearing_sd**2]) @ J.T
    return xy, R


class _Filter:
    def __init__(self, x0: np.ndarray, P0: np.ndarray, dt: float, q: float):
        self.x, self.P = x0.astype(float), P0.astype(float)
        self.F, self.Q = cv_model(dt, q)

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def _correct(self, innovation: np.ndarray, H: np.ndarray, R: np.ndarray) -> float:
        """Kalman update. Returns the likelihood of the measurement (used by the IMM)."""
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        # Joseph form: stays symmetric and positive definite despite rounding.
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        d2 = innovation @ np.linalg.solve(S, innovation)
        return float(np.exp(-0.5 * d2) / np.sqrt(np.linalg.det(2 * np.pi * S)))

    def run(self, zs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Filter a whole sequence. Missed detections (NaN) only predict."""
        xs, Ps = [self.x.copy()], [self.P.copy()]
        for z in zs[1:]:
            self.predict()
            if not np.isnan(z).any():
                self.update(z)
            xs.append(self.x.copy())
            Ps.append(self.P.copy())
        return np.array(xs), np.array(Ps)


class CartesianKF(_Filter):
    H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])

    def __init__(self, x0, P0, dt, q, range_sd, bearing_sd):
        super().__init__(x0, P0, dt, q)
        self.range_sd, self.bearing_sd = range_sd, bearing_sd

    def update(self, z: np.ndarray) -> float:
        xy, R = polar_to_xy(z, self.range_sd, self.bearing_sd)
        return self._correct(xy - self.H @ self.x, self.H, R)


class RadarEKF(_Filter):
    def __init__(self, x0, P0, dt, q, range_sd, bearing_sd):
        super().__init__(x0, P0, dt, q)
        self.R = np.diag([range_sd**2, bearing_sd**2])

    def update(self, z: np.ndarray) -> float:
        px, py = self.x[:2]
        r = np.hypot(px, py)
        predicted = np.array([r, np.arctan2(py, px)])
        H = np.array([[px / r, py / r, 0, 0], [-py / r**2, px / r**2, 0, 0]])
        innovation = z - predicted
        innovation[1] = (innovation[1] + np.pi) % (2 * np.pi) - np.pi   # keep bearing in [-pi, pi)
        return self._correct(innovation, H, self.R)


class IMM:
    """Interacting multiple model filter: several filters, blended by how well each explains the data.

    Here: a "cruise" EKF with small q and a "manoeuvre" EKF with large q. The
    target is assumed to switch between the two behaviours with a small
    probability each step. After every measurement, each model's probability
    is updated with its likelihood, so the output leans on the cruise model
    in straight flight and on the manoeuvre model in a turn. No hand-tuned
    compromise is needed.
    """

    def __init__(self, x0, P0, dt, range_sd, bearing_sd, qs=(0.1, 30.0), p_stay=0.95):
        self.filters = [RadarEKF(x0, P0, dt, q, range_sd, bearing_sd) for q in qs]
        m = len(qs)
        self.T = np.full((m, m), (1 - p_stay) / (m - 1))
        np.fill_diagonal(self.T, p_stay)
        self.mu = np.full(m, 1 / m)
        self.history: list[np.ndarray] = []

    def _mix(self) -> None:
        c = self.T.T @ self.mu                        # predicted model probabilities
        w = self.T * self.mu[:, None] / c[None, :]    # w[i, j] = P(was i | now j)
        xs = [f.x for f in self.filters]
        Ps = [f.P for f in self.filters]
        for j, f in enumerate(self.filters):
            x = sum(w[i, j] * xs[i] for i in range(len(xs)))
            P = sum(w[i, j] * (Ps[i] + np.outer(xs[i] - x, xs[i] - x)) for i in range(len(xs)))
            f.x, f.P = x, P
        self.mu = c

    @property
    def x(self) -> np.ndarray:
        return sum(m * f.x for m, f in zip(self.mu, self.filters))

    @property
    def P(self) -> np.ndarray:
        x = self.x
        return sum(m * (f.P + np.outer(f.x - x, f.x - x)) for m, f in zip(self.mu, self.filters))

    def run(self, zs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs, Ps = [self.x], [self.P]
        self.history = [self.mu.copy()]
        for z in zs[1:]:
            self._mix()
            for f in self.filters:
                f.predict()
            if not np.isnan(z).any():
                lik = np.array([f.update(z) for f in self.filters])
                self.mu = self.mu * lik + 1e-300
                self.mu /= self.mu.sum()
            xs.append(self.x)
            Ps.append(self.P)
            self.history.append(self.mu.copy())
        return np.array(xs), np.array(Ps)


def initialise(zs: np.ndarray, range_sd: float, bearing_sd: float, dt: float = 1.0):
    """Two-point start: position from the first detection, velocity from the difference to the second."""
    hits = [i for i, z in enumerate(zs) if not np.isnan(z).any()][:2]
    (p1, R1), (p2, R2) = (polar_to_xy(zs[i], range_sd, bearing_sd) for i in hits)
    gap = (hits[1] - hits[0]) * dt
    x0 = np.concatenate([p1, (p2 - p1) / gap])
    P0 = np.zeros((4, 4))
    P0[:2, :2] = R1
    P0[2:, 2:] = (R1 + R2) / gap**2
    return x0, P0, hits[0]
