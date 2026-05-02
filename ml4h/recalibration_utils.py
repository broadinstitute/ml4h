"""
Recalibration utilities for ml4h regression models.

Addresses Issue #609 (cardiovascular model small output values) and the
related DROID-LV miscalibration (Issue #550).  Models trained on UK Biobank
ECGs normalised with Standardize(mean=0, std=2000) produce systematically
biased outputs when applied to external cohorts that store voltages in
different units.  The classes here implement lightweight, post-hoc
recalibration that corrects this without retraining the encoder.

Usage
-----
    from ml4h.calibration import IsotonicRecalibrator, LinearRecalibrator

    cal = IsotonicRecalibrator()
    cal.fit(raw_predictions, ground_truth)
    corrected = cal.transform(new_predictions)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger(__name__)


class _BaseRecalibrator:
    """Shared serialisation and diagnostic interface."""

    def save(self, path: str) -> None:
        """Persist fitted calibration parameters to a .npz file."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "_BaseRecalibrator":
        """Restore a calibrator from a .npz file."""
        raise NotImplementedError

    def calibration_error(
        self, predictions: np.ndarray, targets: np.ndarray, n_bins: int = 10,
    ) -> float:
        """
        Mean absolute calibration error (MACE) computed over equal-frequency
        bins of the prediction distribution.

        Parameters
        ----------
        predictions : shape (N,)
        targets     : shape (N,)
        n_bins      : number of quantile bins

        Returns
        -------
        float – MACE in the same units as predictions/targets
        """
        order = np.argsort(predictions)
        p_sorted = predictions[order]
        t_sorted = targets[order]
        bins = np.array_split(np.arange(len(p_sorted)), n_bins)
        errors = [
            abs(p_sorted[b].mean() - t_sorted[b].mean())
            for b in bins if len(b) > 0
        ]
        return float(np.mean(errors))


class LinearRecalibrator(_BaseRecalibrator):
    """
    Fits  y_cal = a * y_raw + b  on a calibration set.

    This is the minimal fix for the unit-mismatch problem (Issue #609):
    when a UKB-trained model is applied to a cohort with ECGs in mV rather
    than µV, the slope `a` will be close to the µV/mV ratio (~1000) and `b`
    captures any constant offset.

    Parameters
    ----------
    degree : int
        Polynomial degree.  degree=1 (default) is a simple affine correction.
        Use degree=2 to additionally capture mean-reversion artefacts.
    """

    def __init__(self, degree: int = 1):
        self.degree = degree
        self._poly = PolynomialFeatures(degree=degree, include_bias=True)
        self._model = LinearRegression(fit_intercept=False)
        self._fitted = False

    def fit(self, predictions: np.ndarray, targets: np.ndarray) -> "LinearRecalibrator":
        X = self._poly.fit_transform(predictions.reshape(-1, 1))
        self._model.fit(X, targets)
        self._fitted = True
        mace_before = self.calibration_error(predictions, targets)
        mace_after = self.calibration_error(self.transform(predictions), targets)
        logger.info(
            "LinearRecalibrator (degree=%d) fitted: MACE %.4f → %.4f",
            self.degree, mace_before, mace_after,
        )
        return self

    def transform(self, predictions: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        X = self._poly.transform(predictions.reshape(-1, 1))
        return self._model.predict(X)

    def save(self, path: str) -> None:
        np.savez(
            path,
            coefficients=self._model.coef_,
            degree=np.array([self.degree]),
        )
        logger.info("LinearRecalibrator saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "LinearRecalibrator":
        data = np.load(path)
        obj = cls(degree=int(data["degree"][0]))
        obj._poly = PolynomialFeatures(degree=obj.degree, include_bias=True)
        obj._poly.fit(np.zeros((1, 1)))  # initialise transformer state
        obj._model = LinearRegression(fit_intercept=False)
        obj._model.coef_ = data["coefficients"]
        obj._model.intercept_ = 0.0
        obj._fitted = True
        return obj


class IsotonicRecalibrator(_BaseRecalibrator):
    """
    Monotone recalibration using isotonic regression.

    Isotonic regression is a non-parametric method that finds the best
    monotone-non-decreasing function mapping model outputs to observed
    targets.  It handles both affine bias and non-linear compression
    artefacts (common in models trained with asymmetric loss).

    Suitable for LVM-AI (trained with the asymmetric loss described in
    CIRC:CIMAGING.120.012281) where the output distribution is skewed.

    Parameters
    ----------
    out_of_bounds : {'clip', 'nan', 'raise'}
        Passed to sklearn.isotonic.IsotonicRegression.  'clip' is safest
        for deployment: predictions outside the calibration range are
        clipped to the boundary calibrated value.
    """

    def __init__(self, out_of_bounds: str = "clip"):
        self._ir = IsotonicRegression(out_of_bounds=out_of_bounds)
        self._fitted = False

    def fit(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "IsotonicRecalibrator":
        self._ir.fit(predictions, targets, sample_weight=sample_weight)
        self._fitted = True
        mace_before = self.calibration_error(predictions, targets)
        mace_after = self.calibration_error(self.transform(predictions), targets)
        logger.info(
            "IsotonicRecalibrator fitted: MACE %.4f → %.4f",
            mace_before, mace_after,
        )
        return self

    def transform(self, predictions: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        return self._ir.transform(predictions)

    def save(self, path: str) -> None:
        # Only save the two public fitted arrays; boundaries are derived on load.
        np.savez(
            path,
            X_thresholds=self._ir.X_thresholds_,
            y_thresholds=self._ir.y_thresholds_,
        )
        logger.info("IsotonicRecalibrator saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "IsotonicRecalibrator":
        from scipy.interpolate import interp1d

        data = np.load(path)
        X_thresh = data["X_thresholds"]
        y_thresh = data["y_thresholds"]

        obj = cls()
        obj._ir = IsotonicRegression(out_of_bounds="clip")
        obj._ir.X_thresholds_ = X_thresh
        obj._ir.y_thresholds_ = y_thresh
        # X_min_/X_max_ are the only boundary attributes still present in >=1.3
        obj._ir.X_min_ = float(X_thresh[0])
        obj._ir.X_max_ = float(X_thresh[-1])
        # Reconstruct the internal interpolator that sklearn uses in transform()
        obj._ir.f_ = interp1d(
            X_thresh,
            y_thresh,
            kind="linear",
            bounds_error=False,
            fill_value=(float(y_thresh[0]), float(y_thresh[-1])),
        )
        obj._fitted = True
        return obj


def recalibrate_predictions_csv(
    predictions_path: str,
    targets_path: str,
    output_path: str,
    prediction_col: str = "prediction_0",
    target_col: str = "lv_mass",
    method: str = "isotonic",
    calibration_fraction: float = 0.5,
    seed: int = 42,
) -> None:
    """
    End-to-end recalibration pipeline for a predictions CSV/parquet.

    Splits the provided data into a calibration set and a held-out evaluation
    set, fits the chosen recalibrator on the calibration split, applies it to
    the full dataset, and writes the corrected predictions to `output_path`.

    Parameters
    ----------
    predictions_path      : path to model output file (.csv or .parquet)
    targets_path          : path to ground-truth file (.csv or .parquet)
    output_path           : where to write the recalibrated output
    prediction_col        : column name of raw model predictions
    target_col            : column name of ground-truth labels
    method                : 'isotonic' (default) or 'linear'
    calibration_fraction  : fraction of samples used to fit the calibrator
    seed                  : random seed for the train/cal split
    """
    def _read(path: str) -> pd.DataFrame:
        return pd.read_parquet(path) if path.endswith(".pq") or path.endswith(".parquet") \
            else pd.read_csv(path)

    preds_df = _read(predictions_path)
    targets_df = _read(targets_path)

    merged = preds_df.merge(targets_df[["sample_id", target_col]], on="sample_id", how="inner")
    merged = merged.dropna(subset=[prediction_col, target_col])

    rng = np.random.default_rng(seed)
    mask = rng.random(len(merged)) < calibration_fraction
    cal_set = merged[mask]
    eval_set = merged[~mask]

    logger.info(
        "Calibration set: %d samples, evaluation set: %d samples",
        len(cal_set), len(eval_set),
    )

    if method == "isotonic":
        cal = IsotonicRecalibrator()
    elif method == "linear":
        cal = LinearRecalibrator()
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'isotonic' or 'linear'.")

    cal.fit(cal_set[prediction_col].values, cal_set[target_col].values)

    merged["prediction_recalibrated"] = cal.transform(merged[prediction_col].values)

    mace_raw = cal.calibration_error(eval_set[prediction_col].values, eval_set[target_col].values)
    mace_cal = cal.calibration_error(
        cal.transform(eval_set[prediction_col].values), eval_set[target_col].values,
    )
    logger.info(
        "Held-out MACE: raw=%.4f  recalibrated=%.4f  (Δ=%.4f)",
        mace_raw, mace_cal, mace_raw - mace_cal,
    )

    out = Path(output_path)
    if out.suffix in {".pq", ".parquet"}:
        merged.to_parquet(output_path, index=False)
    else:
        merged.to_csv(output_path, index=False)

    logger.info("Recalibrated predictions written to %s", output_path)
