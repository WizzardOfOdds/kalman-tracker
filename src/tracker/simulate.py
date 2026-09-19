"""A target flying straight, turning, and flying straight again, seen by a ground radar.

The radar sits at the origin and measures range and bearing once per
second. It misses the target now and then (probability of detection < 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Scenario:
    t: np.ndarray          # seconds
    truth: np.ndarray      # (n, 4): x, y, vx, vy in metres and m/s
    z: np.ndarray          # (n, 2): range [m], bearing [rad]; NaN where missed
    range_sd: float
    bearing_sd: float


def trajectory(n: int = 180, dt: float = 1.0, speed: float = 25.0,
               turn: tuple[int, int] = (70, 110), turn_rate_deg: float = 3.0) -> np.ndarray:
    """Constant speed, with a coordinated turn between the two given steps."""
    x = np.zeros((n, 4))
    heading = np.deg2rad(20.0)
    x[0] = [-1500.0, 800.0, speed * np.cos(heading), speed * np.sin(heading)]
    for k in range(1, n):
        if turn[0] <= k < turn[1]:
            heading += np.deg2rad(turn_rate_deg) * dt
        v = speed * np.array([np.cos(heading), np.sin(heading)])
        x[k, :2] = x[k - 1, :2] + v * dt
        x[k, 2:] = v
    return x


def radar(truth: np.ndarray, range_sd: float = 15.0, bearing_sd_deg: float = 0.5,
          p_detect: float = 0.9, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r = np.hypot(truth[:, 0], truth[:, 1]) + rng.normal(0, range_sd, len(truth))
    b = np.arctan2(truth[:, 1], truth[:, 0]) + rng.normal(0, np.deg2rad(bearing_sd_deg), len(truth))
    z = np.column_stack([r, b])
    z[rng.random(len(truth)) > p_detect] = np.nan
    return z


def make(seed: int = 0, p_detect: float = 0.9) -> Scenario:
    truth = trajectory()
    return Scenario(np.arange(len(truth), dtype=float), truth,
                    radar(truth, p_detect=p_detect, seed=seed), 15.0, np.deg2rad(0.5))
