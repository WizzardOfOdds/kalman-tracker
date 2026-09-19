"""Monte Carlo comparison of the filters.

    python -m tracker.evaluate          # tables
    python -m tracker.evaluate --plot   # also docs/track.png and docs/tuning.png

Two questions:
  1. Accuracy: how far is the estimate from the truth (RMSE)?
  2. Honesty: does the filter *know* how far off it is? The normalised
     estimation error squared (NEES) compares the actual error with the
     covariance the filter reports. For a 4-state filter that is telling the
     truth, NEES averages 4. Much more than 4 means the filter is
     overconfident, which is dangerous: gating and track fusion downstream
     trust that covariance.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import chi2

from .filters import IMM, CartesianKF, RadarEKF, initialise, polar_to_xy
from .simulate import make

RUNS = 100
SKIP = 5          # let the filters settle before scoring
TURN = slice(70, 110)
STRAIGHT = np.r_[SKIP:70, 115:180]


def run_filter(kind, sc, q):
    x0, P0, start = initialise(sc.z, sc.range_sd, sc.bearing_sd)
    if kind == "IMM":
        f = IMM(x0, P0, 1.0, sc.range_sd, sc.bearing_sd)
    else:
        cls = {"KF (converted)": CartesianKF, "EKF (polar)": RadarEKF}[kind]
        f = cls(x0, P0, 1.0, q, sc.range_sd, sc.bearing_sd)
    xs, Ps = f.run(sc.z[start:])
    pad = np.full((start, 4), np.nan)
    return np.vstack([pad, xs]), np.concatenate([np.full((start, 4, 4), np.nan), Ps])


def errors(sc, xs, Ps):
    e = xs - sc.truth
    pos = np.hypot(e[:, 0], e[:, 1])
    nees = np.array([ei @ np.linalg.solve(P, ei) if not np.isnan(ei).any() else np.nan
                     for ei, P in zip(e, Ps)])
    return pos, nees


def raw_position_error(sc):
    xy = np.array([polar_to_xy(z, sc.range_sd, sc.bearing_sd)[0] if not np.isnan(z).any()
                   else [np.nan, np.nan] for z in sc.z])
    return np.hypot(*(xy - sc.truth[:, :2]).T)


def monte_carlo(kind, q, runs=RUNS):
    pos, nees = [], []
    for seed in range(runs):
        sc = make(seed)
        p, n = errors(sc, *run_filter(kind, sc, q))
        pos.append(p)
        nees.append(n)
    return np.array(pos), np.array(nees)


def rms(a, idx):
    return float(np.sqrt(np.nanmean(a[:, idx] ** 2)))


def compare(q: float = 1.0) -> pd.DataFrame:
    raw = np.array([raw_position_error(make(s)) for s in range(RUNS)])
    rows = [{"method": "raw radar fixes", "RMSE straight (m)": rms(raw, STRAIGHT),
             "RMSE turn (m)": rms(raw, TURN), "mean NEES": np.nan}]
    for kind in ("KF (converted)", "EKF (polar)", "IMM"):
        pos, nees = monte_carlo(kind, q)
        rows.append({"method": kind if kind == "IMM" else f"{kind}, q = {q:g}", "RMSE straight (m)": rms(pos, STRAIGHT),
                     "RMSE turn (m)": rms(pos, TURN), "mean NEES": float(np.nanmean(nees[:, SKIP:]))})
    return pd.DataFrame(rows)


def tuning(qs=(0.01, 0.1, 1.0, 10.0, 100.0), kind="EKF (polar)") -> pd.DataFrame:
    rows = []
    for q in qs:
        pos, nees = monte_carlo(kind, q)
        rows.append({"q": q, "RMSE straight (m)": rms(pos, STRAIGHT), "RMSE turn (m)": rms(pos, TURN),
                     "NEES straight": float(np.nanmean(nees[:, STRAIGHT])),
                     "NEES turn": float(np.nanmean(nees[:, TURN]))})
    return pd.DataFrame(rows)


def plot(out="docs") -> None:
    import matplotlib.pyplot as plt

    sc = make(seed=1)
    xy = np.array([polar_to_xy(z, sc.range_sd, sc.bearing_sd)[0] if not np.isnan(z).any()
                   else [np.nan, np.nan] for z in sc.z])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(*sc.truth[:, :2].T, color="0.2", lw=1.5, label="truth")
    ax.scatter(*xy.T, s=6, color="0.6", label="radar fixes")
    for kind, q, color in [("EKF (polar)", 0.01, "tab:red"), ("IMM", None, "tab:blue")]:
        xs, _ = run_filter(kind, sc, q)
        ax.plot(*xs[:, :2].T, color=color, lw=1.2, label="IMM" if kind == "IMM" else f"EKF, q = {q:g}")
    ax.scatter([0], [0], marker="^", color="k")
    ax.annotate("radar", (0, 0), textcoords="offset points", xytext=(6, -12), fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{out}/track.png", dpi=130)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    lo, hi = chi2.ppf([0.025, 0.975], 4 * RUNS) / RUNS
    for kind, q, color in [("EKF (polar)", 0.01, "tab:red"), ("EKF (polar)", 1.0, "tab:blue"),
                           ("EKF (polar)", 100.0, "tab:green"), ("IMM", None, "k")]:
        pos, nees = monte_carlo(kind, q)
        label = "IMM" if kind == "IMM" else f"EKF q = {q:g}"
        axes[0].plot(np.sqrt(np.nanmean(pos**2, axis=0)), color=color, lw=1.2, label=label)
        axes[1].plot(np.nanmean(nees, axis=0), color=color, lw=1.2)
    axes[1].axhspan(lo, hi, color="0.85", label="95 % band for an honest filter")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.axvspan(TURN.start, TURN.stop, color="tab:orange", alpha=0.08)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("position RMSE (m)")
    axes[0].set_ylim(0, None)
    axes[0].legend(frameon=False, ncol=4)
    axes[0].text(TURN.start + 1, axes[0].get_ylim()[1] * 0.92, "turn", color="tab:orange", fontsize=9)
    axes[1].set_ylabel("mean NEES")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{out}/tuning.png", dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    fmt = "{:.1f}".format
    print(compare().to_string(index=False, float_format=fmt))
    print()
    print(tuning().to_string(index=False, float_format=lambda v: f"{v:.2f}" if v < 1 else f"{v:.1f}"))
    if args.plot:
        plot()


if __name__ == "__main__":
    main()
