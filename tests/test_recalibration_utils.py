"""Tests for ml4h.calibration."""

import os
import tempfile

import numpy as np
import pytest

from ml4h.recalibration_utils import IsotonicRecalibrator, LinearRecalibrator




def _make_biased_predictions(n: int = 500, scale: float = 0.001, seed: int = 0):
    """Simulate a UKB model applied to mV-scale ECGs (outputs compressed ~1000x)."""
    rng = np.random.default_rng(seed)
    targets = rng.normal(loc=150, scale=40, size=n).clip(50, 400)  # LV mass in grams
    noise = rng.normal(0, 5, size=n)
    # Model was trained on µV data; at inference it sees mV -> outputs ~1/1000 of true value
    predictions = targets * scale + noise * scale
    return predictions, targets




class TestLinearRecalibrator:
    def test_fit_reduces_mace(self):
        preds, targets = _make_biased_predictions(scale=0.001)
        cal = LinearRecalibrator(degree=1)
        mace_before = cal.calibration_error(preds, targets)
        cal.fit(preds, targets)
        mace_after = cal.calibration_error(cal.transform(preds), targets)
        assert mace_after < mace_before * 0.1, "Linear calibration should reduce MACE by >90%"

    def test_transform_shape(self):
        preds, targets = _make_biased_predictions()
        cal = LinearRecalibrator().fit(preds, targets)
        out = cal.transform(preds)
        assert out.shape == preds.shape

    def test_raises_before_fit(self):
        with pytest.raises(RuntimeError):
            LinearRecalibrator().transform(np.array([1.0, 2.0]))

    def test_save_load_roundtrip(self):
        preds, targets = _make_biased_predictions()
        cal = LinearRecalibrator(degree=2)
        cal.fit(preds, targets)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            cal.save(path)
            restored = LinearRecalibrator.load(path)
            np.testing.assert_allclose(
                cal.transform(preds),
                restored.transform(preds),
                rtol=1e-5,
            )
        finally:
            os.unlink(path)

    def test_degree2_fits_nonlinear(self):
        rng = np.random.default_rng(42)
        preds = rng.uniform(0, 1, 300)
        targets = 100 * preds ** 2 + rng.normal(0, 1, 300)
        cal_linear = LinearRecalibrator(degree=1).fit(preds, targets)
        cal_poly = LinearRecalibrator(degree=2).fit(preds, targets)
        mace_linear = cal_linear.calibration_error(cal_linear.transform(preds), targets)
        mace_poly = cal_poly.calibration_error(cal_poly.transform(preds), targets)
        assert mace_poly <= mace_linear, "Degree-2 should fit quadratic bias as well as or better than degree-1"




class TestIsotonicRecalibrator:
    def test_fit_reduces_mace(self):
        preds, targets = _make_biased_predictions(scale=0.001)
        cal = IsotonicRecalibrator()
        mace_before = cal.calibration_error(preds, targets)
        cal.fit(preds, targets)
        mace_after = cal.calibration_error(cal.transform(preds), targets)
        assert mace_after < mace_before * 0.1

    def test_monotonicity(self):
        preds, targets = _make_biased_predictions()
        cal = IsotonicRecalibrator().fit(preds, targets)
        test_pts = np.linspace(preds.min(), preds.max(), 100)
        out = cal.transform(test_pts)
        assert np.all(np.diff(out) >= -1e-10), "Isotonic calibration must be non-decreasing"

    def test_clip_out_of_bounds(self):
        preds, targets = _make_biased_predictions()
        cal = IsotonicRecalibrator(out_of_bounds="clip").fit(preds, targets)
        extreme = np.array([-1e9, 1e9])
        result = cal.transform(extreme)
        assert np.all(np.isfinite(result)), "Clipped out-of-bounds should not produce inf/nan"

    def test_save_load_roundtrip(self):
        preds, targets = _make_biased_predictions()
        cal = IsotonicRecalibrator().fit(preds, targets)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            cal.save(path)
            restored = IsotonicRecalibrator.load(path)
            np.testing.assert_allclose(
                cal.transform(preds),
                restored.transform(preds),
                rtol=1e-5,
            )
        finally:
            os.unlink(path)

    def test_raises_before_fit(self):
        with pytest.raises(RuntimeError):
            IsotonicRecalibrator().transform(np.array([1.0]))




class TestCalibrationError:
    def test_perfect_calibration(self):
        cal = LinearRecalibrator()
        x = np.linspace(0, 100, 200)
        assert cal.calibration_error(x, x) < 1e-10

    def test_constant_bias(self):
        cal = LinearRecalibrator()
        x = np.linspace(0, 100, 200)
        bias = 30.0
        mace = cal.calibration_error(x, x + bias)
        assert abs(mace - bias) < 1.0, "MACE should roughly equal the constant bias"
