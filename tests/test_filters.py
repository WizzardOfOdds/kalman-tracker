import numpy as np
import pytest

from tracker.filters import IMM, CartesianKF, RadarEKF, cv_model, initialise, polar_to_xy
from tracker.simulate import make, trajectory


def test_cv_model_moves_position_by_velocity():
    F, Q = cv_model(2.0, 1.0)
    assert F @ np.array([0, 0, 3, -1]) == pytest.approx([6, -2, 3, -1])
    assert np.allclose(Q, Q.T)
    assert np.all(np.linalg.eigvalsh(Q) >= 0)


def test_trajectory_keeps_speed_and_turns_120_degrees():
    x = trajectory()
    speed = np.hypot(x[:, 2], x[:, 3])
    assert speed == pytest.approx(np.full(len(x), 25.0))
    h0 = np.arctan2(x[0, 3], x[0, 2])
    h1 = np.arctan2(x[-1, 3], x[-1, 2])
    assert np.rad2deg(h1 - h0) % 360 == pytest.approx(120)


def test_polar_conversion_is_exact_without_noise():
    xy, R = polar_to_xy(np.array([1000.0, np.pi / 2]), 10.0, 0.01)
    assert xy == pytest.approx([0, 1000], abs=1e-9)
    # Cross-range (x here) uncertainty is r * bearing_sd = 10 m, down-range is 10 m too.
    assert np.sqrt(np.diag(R)) == pytest.approx([10.0, 10.0])


@pytest.mark.parametrize("cls", [CartesianKF, RadarEKF])
def test_filter_beats_raw_measurements_on_a_straight_track(cls):
    sc = make(seed=4)
    x0, P0, start = initialise(sc.z, sc.range_sd, sc.bearing_sd)
    xs, _ = cls(x0, P0, 1.0, 1.0, sc.range_sd, sc.bearing_sd).run(sc.z[start:])
    truth = sc.truth[start:]
    err = np.hypot(*(xs[10:60, :2] - truth[10:60, :2]).T)
    raw = np.array([polar_to_xy(z, sc.range_sd, sc.bearing_sd)[0] for z in sc.z[start:][10:60]
                    if not np.isnan(z).any()])
    raw_err = np.hypot(*(raw - truth[10:60][~np.isnan(sc.z[start:][10:60, 0])][:, :2]).T)
    assert np.sqrt(np.mean(err**2)) < 0.7 * np.sqrt(np.mean(raw_err**2))


def test_missed_detections_only_predict():
    sc = make(seed=0, p_detect=1.0)
    z = sc.z.copy()
    z[50:55] = np.nan
    x0, P0, _ = initialise(z, sc.range_sd, sc.bearing_sd)
    xs, Ps = RadarEKF(x0, P0, 1.0, 1.0, sc.range_sd, sc.bearing_sd).run(z)
    assert np.trace(Ps[54]) > np.trace(Ps[49])          # uncertainty grows while blind
    assert xs[52, :2] == pytest.approx(xs[51, :2] + xs[51, 2:], abs=1e-9)


def test_imm_leans_on_manoeuvre_model_in_the_turn():
    sc = make(seed=2, p_detect=1.0)
    x0, P0, _ = initialise(sc.z, sc.range_sd, sc.bearing_sd)
    imm = IMM(x0, P0, 1.0, sc.range_sd, sc.bearing_sd)
    imm.run(sc.z)
    mu = np.array(imm.history)
    straight, turn = mu[20:65, 1].mean(), mu[75:110, 1].mean()
    assert straight < 0.3                 # straight flight: mostly the cruise model
    assert turn > 2 * straight            # the gentle 3 deg/s turn shifts weight to the manoeuvre model
    assert np.allclose(mu.sum(axis=1), 1)
