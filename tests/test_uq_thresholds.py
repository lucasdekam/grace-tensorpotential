"""Regression tests for the per-cluster robust sigma threshold estimator.

The build threshold is ``median + k*(1.4826*MAD)`` (k=3), read off each cluster's
sigma histogram via a piecewise-linear (uniform-within-bin) CDF. This guards:
  - the value formula (median + k robust-sigma, not a tail quantile),
  - bin-size independence (the whole point of interpolating within the bin),
  - the reliability gate / backfill (still p99-midpoint-based, so the reliable
    SET is unchanged from the midpoint era — only the cell VALUES are robust),
  - saturation detection in the info CLI,
  - the opt-in --threshold-percentile estimator, and that selecting it leaves
    the default path and the reliability gate untouched.
"""
import numpy as np

from tensorpotential.uq.cli.build.thresholds import (
    _MAD_TO_STD,
    _ROBUST_K,
    _compute_dual_thresholds,
    _percentile_threshold_from_hist,
    _robust_threshold_from_hist,
)
from tensorpotential.uq.cli.info import _is_saturated


BINS = np.linspace(0.0, 100.0, 250)  # 250 left edges, as the build writes them
BW = float(BINS[1] - BINS[0])

# For a histogram whose mass is a single bin b (a pure spike), the uniform-
# within-bin CDF gives median = bins[b] + bw/2 and MAD = bw/4, so:
#     threshold = bins[b] + bw*(1/2 + k*1.4826/4)
_SPIKE_MARGIN = 0.5 + _ROBUST_K * _MAD_TO_STD / 4.0  # ~1.612 for k=3


def _hist_in_bin(b, n=200, nbins=250):
    """A length-`nbins` histogram with all `n` counts in bin `b`."""
    h = np.zeros(nbins, dtype=np.int64)
    h[b] = n
    return h


def test_threshold_is_median_plus_k_robust_sigma_on_a_spike():
    # Pure spike in bin 50 -> threshold = bins[50] + bw*(1/2 + k*1.4826/4).
    thr = _robust_threshold_from_hist(_hist_in_bin(50), BINS, BW)
    assert np.isclose(thr, BINS[50] + _SPIKE_MARGIN * BW)
    # strictly above the bin (it adds a positive k-sigma margin), never a clamp
    assert thr > BINS[50] + BW / 2.0


def test_threshold_matches_analytic_uniform():
    # Bin-aligned Uniform[10,20]: median=15, MAD=2.5 -> 15 + 3*1.4826*2.5 = 26.12.
    bins = np.arange(250) * 0.4          # left edges 0, 0.4, ... ; bin 25 = [10,10.4]
    h = np.zeros(250)
    h[25:50] = 1000.0                     # bins 25..49 cover [10.0, 20.0] exactly
    thr = _robust_threshold_from_hist(h, bins, 0.4)
    assert np.isclose(thr, 15.0 + _ROBUST_K * _MAD_TO_STD * 2.5, atol=1e-2)


def test_threshold_is_bin_size_independent():
    """An exact 2x bin refinement represents the SAME distribution, so the
    uniform-within-bin CDF — and the robust threshold — must be identical. (The
    old midpoint estimator snapped to bin centers and would shift by ~bw/2.)"""
    N, bwc = 250, 0.4
    coarse = np.arange(N) * bwc
    fine = np.arange(2 * N) * (bwc / 2.0)
    hc = np.zeros(N)
    hc[20:60] = 1000.0    # a uniform block
    hc[120] = 50000.0     # plus a spike
    hf = np.zeros(2 * N)  # split each coarse bin into 2 equal-density fine bins
    for i in range(N):
        hf[2 * i] = hc[i] / 2.0
        hf[2 * i + 1] = hc[i] / 2.0
    tc = _robust_threshold_from_hist(hc, coarse, bwc)
    tf = _robust_threshold_from_hist(hf, fine, bwc / 2.0)
    assert np.isclose(tc, tf, atol=1e-6)


def test_reliable_cluster_takes_its_own_robust_threshold():
    hists = {0: _hist_in_bin(50)[None, :]}
    raw, eff, _, _ = _compute_dual_thresholds(hists, hists, BINS, 1, 128, 1)
    expected = _robust_threshold_from_hist(_hist_in_bin(50), BINS, BW)
    assert np.isclose(raw[0, 0], expected)
    assert np.isclose(eff[0, 0], expected)


def test_bin0_eff_threshold_is_lifted_off_zero():
    """Raw p99 healthy (cell reliable) but the weighted histogram piles in bin 0.

    The robust eff threshold of a bin-0 spike is bins[0] + margin*bw > 0 — the
    estimator can never produce the zero threshold the old left-edge p99 did.
    """
    raw_hist = {0: _hist_in_bin(50)[None, :]}   # reliable (gate sees bin 50)
    eff_hist = {0: _hist_in_bin(0)[None, :]}    # weighted mass collapsed to bin 0
    raw, eff, _, _ = _compute_dual_thresholds(raw_hist, eff_hist, BINS, 1, 128, 1)
    assert eff[0, 0] > 0.0
    assert np.isclose(eff[0, 0], _SPIKE_MARGIN * BW)  # bins[0]=0


def test_reliability_gate_still_rejects_bins_0_and_1():
    """Gate must reject a cluster whose p99 is in bin 0/1 (too concentrated) and
    backfill it with a reliable sibling's robust threshold — same SET as before,
    just a robust fill value."""
    h = np.stack([_hist_in_bin(50), _hist_in_bin(1)])  # [2, nbins]
    raw, eff, _, _ = _compute_dual_thresholds({0: h}, {0: h}, BINS, 2, 128, 1)
    reliable_val = _robust_threshold_from_hist(_hist_in_bin(50), BINS, BW)
    assert np.isclose(raw[0, 0], reliable_val)
    assert np.isclose(raw[0, 1], reliable_val)  # bin-1 cluster took the backfill


def test_all_degenerate_element_backfills_to_populated_max():
    """Every cluster degenerate (p99 in bin 0): no reliable cluster, so backfill
    uses the populated max — which under the robust estimator is bins[0]+margin*bw,
    never 0."""
    h = np.stack([_hist_in_bin(0), _hist_in_bin(0)])
    raw, eff, _, _ = _compute_dual_thresholds({0: h}, {0: h}, BINS, 2, 128, 1)
    assert np.allclose(raw[0], _SPIKE_MARGIN * BW)
    assert np.all(raw[0] > 0.0)


def test_absent_element_row_defaults_to_one():
    hists = {0: _hist_in_bin(50)[None, :]}
    raw, eff, _, _ = _compute_dual_thresholds(hists, hists, BINS, 1, 128, 2)
    assert np.allclose(raw[1], 1.0)


def test_saturation_detection_handles_ceiling():
    # A threshold below the ceiling is not saturated; at/above bins[-1] it is.
    assert not _is_saturated(50.0, BINS)
    assert not _is_saturated(float(BINS[-1]) - 1.0, BINS)
    assert _is_saturated(float(BINS[-1]), BINS)
    assert _is_saturated(float(BINS[-1]) + BW, BINS)


# ---------------------------------------------------------------------------
# --threshold-percentile: opt-in tail-quantile estimator
# ---------------------------------------------------------------------------


def _spike_plus_tail(nbins=250):
    """Sharp bulk plus a thin heavy tail — the shape that makes median+k*MAD
    flag a large slice of the TRAINING set (MAD sees only the spike)."""
    h = np.zeros(nbins, dtype=np.float64)
    h[2] = 10_000.0          # the bulk
    h[10:60] = 20.0          # 1000 counts smeared over a long tail
    return h


def test_percentile_threshold_matches_analytic_uniform():
    # Bin-aligned Uniform[10,20]: p99 = 10 + 0.99*10 = 19.9.
    bins = np.arange(250) * 0.4
    h = np.zeros(250)
    h[25:50] = 1000.0
    thr = _percentile_threshold_from_hist(h, bins, 0.4, 99.0)
    assert np.isclose(thr, 19.9, atol=1e-2)


def test_percentile_threshold_is_bin_size_independent():
    """Same guarantee as the robust estimator: an exact 2x refinement describes
    the same distribution, so the quantile must not move."""
    N, bwc = 250, 0.4
    coarse = np.arange(N) * bwc
    hc = _spike_plus_tail(N)
    fine = np.arange(2 * N) * (bwc / 2)
    hf = np.repeat(hc, 2) / 2.0
    assert np.isclose(
        _percentile_threshold_from_hist(hc, coarse, bwc, 99.0),
        _percentile_threshold_from_hist(hf, fine, bwc / 2, 99.0),
        atol=1e-9,
    )


def test_percentile_is_above_robust_on_a_heavy_tail():
    """The motivating case: the robust fence sits just above the bulk, so much of
    the training distribution reads gamma > 1. p99 rides the tail instead, which
    is exactly the trade --threshold-percentile makes."""
    h = _spike_plus_tail()
    robust = _robust_threshold_from_hist(h, BINS, BW)
    p99 = _percentile_threshold_from_hist(h, BINS, BW, 99.0)
    assert p99 > robust
    # by construction ~1% of the mass sits above the p99 threshold ...
    above_p99 = h[BINS + BW / 2.0 > p99].sum() / h.sum()
    assert above_p99 < 0.02
    # ... while the robust fence leaves far more of the TRAINING mass above 1
    above_robust = h[BINS + BW / 2.0 > robust].sum() / h.sum()
    assert above_robust > 5 * above_p99


def test_default_path_is_unchanged_by_the_new_parameter():
    """Omitting threshold_percentile and passing None explicitly must be the same
    computation — the option must not perturb existing artifacts."""
    h = _spike_plus_tail()
    hists = {0: np.stack([h, h * 2]), 1: np.stack([h * 3, np.zeros_like(h)])}
    default = _compute_dual_thresholds(hists, hists, BINS, 2, 50, 2)
    explicit = _compute_dual_thresholds(
        hists, hists, BINS, 2, 50, 2, threshold_percentile=None
    )
    for a, b in zip(default[:2], explicit[:2]):
        assert np.array_equal(a, b, equal_nan=True)
    assert default[2:] == explicit[2:]


def _cluster_hist(spike_bin, n=10_000.0, tail=None, tail_h=20.0, nbins=250):
    """Sharp bulk in `spike_bin`, optionally smeared with a thin heavy tail."""
    h = np.zeros(nbins, dtype=np.float64)
    h[spike_bin] = n
    if tail is not None:
        h[tail[0]:tail[1]] = tail_h
    return h


def _backfilled_mask(mat):
    """Cells that took the row fill (the element-wide max of reliable cells).

    `_backfill` leaves no NaN behind, so comparing NaN patterns compares two
    all-False arrays and pins nothing. Backfilled cells are exactly those equal
    to the row fill, which IS observable in the returned matrix — provided the
    fixture gives the reliable clusters DIFFERENT thresholds, otherwise every
    cell equals the max and the mask is all-True and equally useless.
    """
    return np.isclose(mat, np.nanmax(mat, axis=1, keepdims=True))


def test_percentile_changes_values_but_not_the_reliability_set():
    """Constraint: the gate stays on the p99 midpoint whichever estimator runs,
    so the reliable-cluster SET and every backfill position are identical.

    Fixture: two well-populated clusters with distinct thresholds plus one
    under-populated cluster that must backfill — so the mask is [F, T, T] and
    actually discriminates.
    """
    hists = {
        0: np.stack([
            _cluster_hist(2, tail=(10, 60)),    # reliable, low threshold
            _cluster_hist(30, tail=(40, 90)),   # reliable, high threshold
            _cluster_hist(2, n=10.0),           # 10 atoms < min_atoms=50 -> backfilled
        ])
    }
    robust = _compute_dual_thresholds(hists, hists, BINS, 3, 50, 1)
    pctl = _compute_dual_thresholds(
        hists, hists, BINS, 3, 50, 1, threshold_percentile=99.0
    )

    # the fixture must actually exercise backfill AND discriminate, or this
    # test would pass no matter what the gate did
    assert robust[2] == [(0, 2, 10)], robust[2]
    assert not _backfilled_mask(robust[0]).all()

    assert not np.allclose(robust[0], pctl[0])                 # values move ...
    for a, b in zip(robust[:2], pctl[:2]):                     # ... positions do not
        assert np.array_equal(_backfilled_mask(a), _backfilled_mask(b))
    # underpop warnings carry (elem, cluster, n_calib) — counts, so estimator-free.
    # elementwise_fallback_warnings carry a fill VALUE and legitimately differ.
    assert robust[2] == pctl[2]


def test_backfill_mask_fixture_detects_a_gate_change():
    """Guard on the guard: if the reliability gate moved, the mask above must
    change — otherwise the invariant test is decorative."""
    hists = {
        0: np.stack([
            _cluster_hist(2, tail=(10, 60)),
            _cluster_hist(30, tail=(40, 90)),
            _cluster_hist(2, n=10.0),
        ])
    }
    base = _compute_dual_thresholds(hists, hists, BINS, 3, 50, 1)
    stricter = _compute_dual_thresholds(hists, hists, BINS, 3, 50_000, 1)
    assert not np.array_equal(
        _backfilled_mask(base[0]), _backfilled_mask(stricter[0])
    )


def test_percentile_rejects_out_of_range_q():
    """Q outside (0, 100] gives a degenerate threshold (0 -> gamma=inf, >100 ->
    gamma~0), so it must raise rather than silently produce a broken artifact."""
    h = _spike_plus_tail()
    for bad in (0.0, -1.0, 100.5, float("nan"), float("inf")):
        try:
            _percentile_threshold_from_hist(h, BINS, BW, bad)
        except ValueError:
            continue
        raise AssertionError(f"percentile {bad} was accepted")
