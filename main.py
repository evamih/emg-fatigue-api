"""
EMG Muscle Fatigue Analysis API  v3.0
--------------------------------------
Single-channel endpoint:
  POST /analyze  — unchanged from v2, backwards compatible

Dual-channel endpoint (new):
  POST /analyze_dual — bicep + tricep CSV → DualFatigueReport JSON
    · Independent MDF/MNF fatigue analysis per channel
    · Co-activation ratio (CAR) time series and trend
    · Cross-channel correlation metrics
    · Multi-modal fatigue verdict combining spectral + CAR indices

GET /health — liveness check

CSV format for /analyze_dual:
  time_s, emg1_adc, emg2_adc [, mvc1, mvc2]
  (mvc columns optional — pass mvc1_adc / mvc2_adc as form fields instead)
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, welch
from typing import Optional
import io

# ── constants ──────────────────────────────────────────────────────────────────
ADC_MAX         = 4095.0
ADC_VREF        = 3.3
BP_LOW          = 20
BP_HIGH         = 450
NOTCH_F0        = 50
NOTCH_Q         = 30
ENV_WIN_MS      = 250
REST_BASELINE_S = 3.0
NOTCH_SPIKE_DB  = 6.0

# CAR thresholds — based on elbow flexion literature
CAR_LOW      = 0.15   # below = efficient antagonist control
CAR_HIGH     = 0.30   # above = high co-activation / fatigue
CAR_SLOPE_THR = 0.002  # /s — rising faster than this = neuromuscular fatigue

COLUMN_ALIASES = {
    "time":  ["time_s", "time", "t", "timestamp", "Time", "T"],
    "emg":   ["emg", "emgData", "emg_data", "EMG", "value", "adc",
               "emg1_adc", "emg_adc"],
    "emg1":  ["emg1_adc", "emg1", "EMG1"],
    "emg2":  ["emg2_adc", "emg2", "EMG2"],
    "mvc":   ["mvc", "MVC", "mvc_ref", "mvc1", "mvc1_adc"],
    "mvc2":  ["mvc2", "mvc2_adc", "MVC2"],
}

app = FastAPI(title="EMG Fatigue API", version="3.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"],
)


# ── response models ────────────────────────────────────────────────────────────

class FrequencyPoint(BaseModel):
    time_s: float
    mdf_hz: Optional[float] = None
    mnf_hz: Optional[float] = None


class ChannelReport(BaseModel):
    """Fatigue analysis for a single EMG channel."""
    channel:       str    # "bicep" | "tricep"
    duration_s:    float
    fs_hz:         float
    n_samples:     int
    notch_applied: bool
    mvc_source:    str
    mvc_v:         float
    threshold_adc: float
    peak_mvc_pct:  float
    mean_mvc_pct:  float
    global_mdf_hz: float
    global_mnf_hz: float
    mdf_start_hz:  float
    mdf_end_hz:    float
    mdf_slope_hz_per_s:       float
    mdf_slope_norm_pct_per_s: float
    mnf_start_hz:  float
    mnf_end_hz:    float
    mnf_slope_hz_per_s:       float
    mnf_slope_norm_pct_per_s: float
    active_windows: int
    total_windows:  int
    fatigue_detected: bool
    fatigue_level:    str
    summary:          str
    envelope_norm_pct:  list[float]
    envelope_trend_pct: list[float]
    time_envelope:      list[float]
    frequency_series:   list[FrequencyPoint]
    regression_mdf: list[float]
    regression_mnf: list[float]
    regression_t:   list[float]


class CarPoint(BaseModel):
    time_s:    float
    car:       float    # co-activation ratio 0–1
    bicep_rms: float
    tricep_rms: float


class FatigueReport(BaseModel):
    """Single-channel report — unchanged from v2."""
    duration_s:    float
    fs_hz:         float
    n_samples:     int
    notch_applied: bool
    mvc_source:    str
    mvc_v:         float
    threshold_adc: float
    peak_mvc_pct:  float
    mean_mvc_pct:  float
    global_mdf_hz: float
    global_mnf_hz: float
    mdf_start_hz:  float
    mdf_end_hz:    float
    mdf_slope_hz_per_s:       float
    mdf_slope_norm_pct_per_s: float
    mnf_start_hz:  float
    mnf_end_hz:    float
    mnf_slope_hz_per_s:       float
    mnf_slope_norm_pct_per_s: float
    active_windows: int
    total_windows:  int
    fatigue_detected: bool
    fatigue_level:    str
    summary:          str
    envelope_norm_pct:  list[float]
    envelope_trend_pct: list[float]
    time_envelope:      list[float]
    frequency_series:   list[FrequencyPoint]
    regression_mdf: list[float]
    regression_mnf: list[float]
    regression_t:   list[float]


class DualFatigueReport(BaseModel):
    """
    Full dual-channel analysis: independent spectral fatigue per channel
    + co-activation ratio + multi-modal verdict.
    """
    # Per-channel reports
    bicep:  ChannelReport
    tricep: ChannelReport

    # Co-activation ratio time series
    car_series:   list[CarPoint]
    avg_car:      float   # mean CAR over active windows
    car_slope:    float   # linear trend slope (/s), positive = rising co-activation
    car_start:    float   # regression start value
    car_end:      float   # regression end value
    car_level:    str     # low | moderate | high
    car_trending: bool    # True if slope exceeds CAR_SLOPE_THR

    # Cross-channel correlation
    # Pearson r between normalised bicep and tricep envelopes over active windows
    envelope_correlation: float

    # Multi-modal fatigue verdict
    spectral_fatigue_agonist:    str   # none | mild | moderate | severe
    spectral_fatigue_antagonist: str
    kinematic_fatigue:           str   # not_assessed (IMU processed client-side)
    overall_verdict:             str   # none | mild | moderate | severe
    conclusions:                 str   # human-readable multi-line analysis


# ── shared signal processing ───────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, key: str) -> Optional[str]:
    for alias in COLUMN_ALIASES[key]:
        if alias in df.columns:
            return alias
    return None


def _detect_fs(t: np.ndarray) -> float:
    diffs = np.diff(t)
    return 1.0 / float(np.median(diffs[diffs > 0]))


def _to_volts(emg: np.ndarray) -> np.ndarray:
    return emg / ADC_MAX * ADC_VREF if np.max(np.abs(emg)) > 10 else emg.copy()


def _bandpass(emg: np.ndarray, fs: float) -> np.ndarray:
    fnyq = fs / 2
    high = min(BP_HIGH, fnyq * 0.95)
    b, a = butter(4, [BP_LOW / fnyq, high / fnyq], "bandpass")
    return filtfilt(b, a, emg)


def _check_notch_needed(emg_filt: np.ndarray, fs: float) -> bool:
    nperseg = min(len(emg_filt), int(fs))
    f, pxx  = welch(emg_filt, fs=fs, nperseg=nperseg)
    idx     = np.argmin(np.abs(f - NOTCH_F0))
    if idx < 2 or idx > len(pxx) - 3:
        return False
    local_noise = np.median(pxx[max(0, idx - 5): idx + 6])
    spike_ratio = 10 * np.log10(pxx[idx] / (local_noise + 1e-30))
    return bool(spike_ratio > NOTCH_SPIKE_DB)


def _apply_notch(emg: np.ndarray, fs: float) -> np.ndarray:
    fnyq = fs / 2
    b, a = iirnotch(NOTCH_F0 / fnyq, NOTCH_Q)
    return filtfilt(b, a, emg)


def _envelope(emg_filt: np.ndarray, fs: float) -> np.ndarray:
    win = max(1, int(ENV_WIN_MS / 1000 * fs))
    return np.convolve(np.abs(emg_filt), np.ones(win) / win, mode="same")


def _data_driven_threshold(env: np.ndarray, t: np.ndarray, fs: float) -> float:
    rest_mask = t < REST_BASELINE_S
    if rest_mask.sum() > int(0.5 * fs):
        rest      = env[rest_mask]
        candidate = float(rest.mean() + 2.5 * rest.std())
        if float(np.mean(env > candidate)) >= 0.20:
            return candidate
    p10 = float(np.percentile(env, 10))
    p90 = float(np.percentile(env, 90))
    return p10 + 0.30 * (p90 - p10)


def _spectral_metrics(segment: np.ndarray, fs: float):
    nperseg = min(len(segment), max(64, int(fs)))
    if len(segment) < nperseg:
        return np.nan, np.nan
    f, pxx = welch(segment, fs=fs, window="hamming",
                   nperseg=nperseg, noverlap=nperseg // 2)
    band  = (f >= BP_LOW) & (f <= BP_HIGH)
    f_b, p_b = f[band], pxx[band]
    if p_b.sum() == 0:
        return np.nan, np.nan
    cum = np.cumsum(p_b)
    mdf = float(f_b[np.searchsorted(cum, cum[-1] / 2)])
    mnf = float(np.sum(f_b * p_b) / np.sum(p_b))
    return mdf, mnf


def _medfilt1_nan(arr: np.ndarray, k: int = 3) -> np.ndarray:
    out  = arr.copy()
    half = k // 2
    for i in range(len(arr)):
        win   = arr[max(0, i - half): i + half + 1]
        valid = win[~np.isnan(win)]
        if len(valid):
            out[i] = float(np.median(valid))
    return out


def _fatigue_level(slope: float) -> tuple[bool, str]:
    if slope >= -0.05:   return False, "none"
    elif slope >= -0.15: return True,  "mild"
    elif slope >= -0.30: return True,  "moderate"
    return True, "severe"


def _ds(arr: np.ndarray, max_pts: int = 500) -> np.ndarray:
    if len(arr) > max_pts:
        idx = np.linspace(0, len(arr) - 1, max_pts, dtype=int)
        return arr[idx]
    return arr


# ── single-channel analysis (reused by both endpoints) ────────────────────────

def _analyze_channel(
    t:        np.ndarray,
    raw:      np.ndarray,
    fs:       float,
    mvc_v:    float,
    mvc_source: str,
    channel:  str,
    window_s: float = 2.0,
    step_s:   float = 0.5,
) -> tuple[ChannelReport, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run full fatigue analysis on one EMG channel.
    Returns (ChannelReport, emg_filt, env, env_norm) for cross-channel use.
    """
    emg_v    = _to_volts(raw)
    emg_filt = _bandpass(emg_v, fs)

    notch_applied = _check_notch_needed(emg_filt, fs)
    if notch_applied:
        emg_filt = _apply_notch(emg_filt, fs)

    env      = _envelope(emg_filt, fs)
    mvc_v    = max(mvc_v, 1e-6)
    env_norm = env / mvc_v * 100.0
    th_v     = _data_driven_threshold(env, t, fs)
    th_adc   = th_v / ADC_VREF * ADC_MAX

    # global PSD
    npg   = min(len(emg_filt), int(fs))
    f_g, pxx_g = welch(emg_filt, fs=fs, window="hamming",
                       nperseg=npg, noverlap=npg // 2)
    band  = (f_g >= BP_LOW) & (f_g <= BP_HIGH)
    f_b, p_b = f_g[band], pxx_g[band]
    cum   = np.cumsum(p_b)
    global_mdf = float(f_b[np.searchsorted(cum, cum[-1] / 2)])
    global_mnf = float(np.sum(f_b * p_b) / np.sum(p_b))

    # spectral series
    win_n  = int(window_s * fs)
    step_n = int(step_s   * fs)
    freq_series: list[FrequencyPoint] = []

    for k in range(0, len(emg_filt) - win_n, step_n):
        seg      = emg_filt[k: k + win_n]
        t_center = float(t[k + win_n // 2])
        if float(env[k: k + win_n].mean()) < th_v:
            freq_series.append(FrequencyPoint(time_s=t_center))
            continue
        mdf, mnf = _spectral_metrics(seg, fs)
        freq_series.append(FrequencyPoint(
            time_s=t_center,
            mdf_hz=round(mdf, 2) if np.isfinite(mdf) else None,
            mnf_hz=round(mnf, 2) if np.isfinite(mnf) else None,
        ))

    total_windows  = len(freq_series)
    active_windows = sum(1 for p in freq_series if p.mdf_hz is not None)

    if active_windows < 4:
        raise HTTPException(
            422,
            f"Channel {channel}: only {active_windows} active windows "
            f"(need ≥4). Check electrode placement or reduce window_s."
        )

    mdf_arr = np.array([p.mdf_hz if p.mdf_hz is not None else np.nan
                        for p in freq_series])
    mnf_arr = np.array([p.mnf_hz if p.mnf_hz is not None else np.nan
                        for p in freq_series])
    t_arr   = np.array([p.time_s for p in freq_series])

    mdf_arr = _medfilt1_nan(mdf_arr, 3)
    mnf_arr = _medfilt1_nan(mnf_arr, 3)

    valid = ~np.isnan(mdf_arr)
    t_v   = t_arr[valid]
    mdf_v = mdf_arr[valid]
    mnf_v = mnf_arr[valid]

    p_mdf = np.polyfit(t_v, mdf_v, 1)
    p_mnf = np.polyfit(t_v, mnf_v, 1)

    mdf_start = float(np.polyval(p_mdf, t_v[0]))
    mdf_end   = float(np.polyval(p_mdf, t_v[-1]))
    mnf_start = float(np.polyval(p_mnf, t_v[0]))
    mnf_end   = float(np.polyval(p_mnf, t_v[-1]))

    mdf_slope      = float(p_mdf[0])
    mnf_slope      = float(p_mnf[0])
    mdf_slope_norm = mdf_slope / mdf_start * 100.0 if mdf_start > 0 else 0.0
    mnf_slope_norm = mnf_slope / mnf_start * 100.0 if mnf_start > 0 else 0.0

    fatigue_detected, fatigue_level = _fatigue_level(mdf_slope)

    trend_win = min(len(env_norm), int(2.0 * fs))
    env_trend = np.convolve(env_norm, np.ones(trend_win) / trend_win,
                            mode="same")

    env_out   = _ds(env_norm)
    trend_out = _ds(env_trend)
    t_out     = _ds(t)

    summary_data = dict(
        fatigue_level=fatigue_level,
        mdf_slope_hz_per_s=mdf_slope,
        mdf_slope_norm_pct_per_s=mdf_slope_norm,
        mdf_start_hz=mdf_start,
        mdf_end_hz=mdf_end,
        duration_s=round(float(t[-1] - t[0]), 2),
        peak_mvc_pct=round(float(env_norm.max()), 1),
        active_windows=active_windows,
        total_windows=total_windows,
    )

    report = ChannelReport(
        channel        = channel,
        duration_s     = round(float(t[-1] - t[0]), 2),
        fs_hz          = round(fs, 1),
        n_samples      = int(len(raw)),
        notch_applied  = notch_applied,
        mvc_source     = mvc_source,
        mvc_v          = round(mvc_v, 5),
        threshold_adc  = round(th_adc, 1),
        peak_mvc_pct   = round(float(env_norm.max()), 1),
        mean_mvc_pct   = round(float(
            env_norm[env_norm > 5].mean()) if (env_norm > 5).any() else 0.0, 1),
        global_mdf_hz  = round(global_mdf, 2),
        global_mnf_hz  = round(global_mnf, 2),
        mdf_start_hz   = round(mdf_start, 1),
        mdf_end_hz     = round(mdf_end,   1),
        mdf_slope_hz_per_s       = round(mdf_slope,      4),
        mdf_slope_norm_pct_per_s = round(mdf_slope_norm, 4),
        mnf_start_hz   = round(mnf_start, 1),
        mnf_end_hz     = round(mnf_end,   1),
        mnf_slope_hz_per_s       = round(mnf_slope,      4),
        mnf_slope_norm_pct_per_s = round(mnf_slope_norm, 4),
        active_windows = active_windows,
        total_windows  = total_windows,
        fatigue_detected = fatigue_detected,
        fatigue_level    = fatigue_level,
        summary          = _channel_summary(summary_data, channel),
        envelope_norm_pct  = [round(v, 2) for v in env_out.tolist()],
        envelope_trend_pct = [round(v, 2) for v in trend_out.tolist()],
        time_envelope      = [round(v, 3) for v in t_out.tolist()],
        frequency_series   = freq_series,
        regression_mdf     = [round(mdf_start, 2), round(mdf_end, 2)],
        regression_mnf     = [round(mnf_start, 2), round(mnf_end, 2)],
        regression_t       = [round(float(t_v[0]), 2), round(float(t_v[-1]), 2)],
    )

    return report, emg_filt, env, env_norm


# ── co-activation analysis ─────────────────────────────────────────────────────

def _compute_car(
    env_bicep:  np.ndarray,
    env_tricep: np.ndarray,
    t:          np.ndarray,
    th_bicep:   float,
    th_tricep:  float,
    fs:         float,
    step_s:     float = 0.5,
) -> tuple[list[CarPoint], float, float, float, float, str, bool]:
    """
    Compute co-activation ratio (CAR) over time.

    CAR = tricep_rms / (bicep_rms + tricep_rms)

    Uses the same step_s windows as the spectral analysis.
    Only windows where BOTH channels are above their respective thresholds
    are included in the CAR trend — this avoids noisy ratio estimates during
    rest periods.

    Returns:
        car_series, avg_car, car_slope, car_start, car_end, car_level, car_trending
    """
    step_n = max(1, int(step_s * fs))
    series: list[CarPoint] = []

    for k in range(0, min(len(env_bicep), len(env_tricep)) - step_n, step_n):
        b_rms = float(np.sqrt(np.mean(env_bicep [k: k + step_n] ** 2)))
        t_rms = float(np.sqrt(np.mean(env_tricep[k: k + step_n] ** 2)))
        total = b_rms + t_rms
        t_c   = float(t[min(k + step_n // 2, len(t) - 1)])

        # Only include windows where both channels are meaningfully active
        b_active = float(env_bicep [k: k + step_n].mean()) > th_bicep
        t_active = float(env_tricep[k: k + step_n].mean()) > th_tricep

        if not (b_active or t_active) or total < 1e-9:
            continue

        car = t_rms / total if total > 0 else 0.0
        series.append(CarPoint(
            time_s    = round(t_c, 3),
            car       = round(car, 4),
            bicep_rms = round(b_rms, 4),
            tricep_rms= round(t_rms, 4),
        ))

    if len(series) < 4:
        # Not enough active windows — return neutral values
        return series, 0.0, 0.0, 0.0, 0.0, "low", False

    cars    = np.array([p.car    for p in series])
    times   = np.array([p.time_s for p in series])
    avg_car = float(np.mean(cars))

    p_car     = np.polyfit(times, cars, 1)
    car_slope = float(p_car[0])
    car_start = float(np.polyval(p_car, times[0]))
    car_end   = float(np.polyval(p_car, times[-1]))

    if avg_car < CAR_LOW:
        car_level = "low"
    elif avg_car < CAR_HIGH:
        car_level = "moderate"
    else:
        car_level = "high"

    car_trending = bool(car_slope > CAR_SLOPE_THR)

    return series, avg_car, car_slope, car_start, car_end, car_level, car_trending


def _envelope_correlation(
    env_norm_bicep:  np.ndarray,
    env_norm_tricep: np.ndarray,
) -> float:
    """
    Pearson r between the two normalised envelopes.
    High positive r: both muscles activate together (efficient coupling).
    Low or negative r: asynchronous activation (possible coordination deficit).
    """
    n = min(len(env_norm_bicep), len(env_norm_tricep))
    if n < 10:
        return 0.0
    b = env_norm_bicep[:n]
    c = env_norm_tricep[:n]
    corr = np.corrcoef(b, c)[0, 1]
    return round(float(corr) if np.isfinite(corr) else 0.0, 4)


# ── verdict and conclusions ────────────────────────────────────────────────────

def _channel_summary(d: dict, channel: str) -> str:
    level = d["fatigue_level"]
    slope = d["mdf_slope_hz_per_s"]
    norm  = d["mdf_slope_norm_pct_per_s"]
    drop  = d["mdf_start_hz"] - d["mdf_end_hz"]
    dur   = d["duration_s"]
    act   = d["active_windows"]
    tot   = d["total_windows"]

    lines = [f"{channel.capitalize()}: {dur:.0f}s, {act}/{tot} windows active."]
    if level == "none":
        lines.append(f"No significant fatigue — {channel} frequencies stable.")
    elif level == "mild":
        lines.append(
            f"Mild fatigue (MDF {slope:.3f} Hz/s, {norm:.3f}%/s). "
            f"{drop:.1f} Hz downshift — early metabolic accumulation."
        )
    elif level == "moderate":
        lines.append(
            f"Moderate fatigue (MDF {slope:.3f} Hz/s, {norm:.3f}%/s). "
            f"Clear {drop:.1f} Hz spectral downshift."
        )
    else:
        lines.append(
            f"Severe fatigue (MDF {slope:.3f} Hz/s, {norm:.3f}%/s). "
            f"{drop:.1f} Hz downshift — significant neuromuscular fatigue."
        )
    return " ".join(lines)


def _overall_verdict(
    agonist_level:    str,
    antagonist_level: str,
    car_level:        str,
    car_trending:     bool,
) -> str:
    """
    Combine agonist spectral, antagonist spectral, and CAR into one verdict.
    The agonist spectral result dominates; antagonist and CAR modulate it.
    """
    severity_rank = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}
    agonist_rank  = severity_rank.get(agonist_level,    0)
    antag_rank    = severity_rank.get(antagonist_level, 0)

    # Boost verdict if antagonist is also fatiguing (both sides affected)
    combined = max(agonist_rank, antag_rank - 1)  # antagonist counts less

    # CAR trending up adds one level of concern
    if car_trending or car_level == "high":
        combined = min(combined + 1, 3)

    return ["none", "mild", "moderate", "severe"][combined]


def _conclusions(
    agonist:        ChannelReport,
    antagonist:     ChannelReport,
    avg_car:        float,
    car_slope:      float,
    car_level:      str,
    car_trending:   bool,
    envelope_corr:  float,
    overall:        str,
    exercise_name:  str = "",
) -> str:
    """
    Generate a multi-paragraph conclusions block suitable for the
    Analysis tab in the Flutter app.
    """
    lines: list[str] = []

    # ── Overall verdict ────────────────────────────────────────────────────
    verdict_map = {
        "none":     "No significant fatigue was detected across any index.",
        "mild":     "Mild fatigue was detected — early signs of metabolic "
                    "accumulation without performance compromise.",
        "moderate": "Moderate fatigue detected — meaningful neuromuscular "
                    "stress that warrants monitoring.",
        "severe":   "Severe fatigue detected — significant neuromuscular "
                    "stress. Allow adequate recovery before the next session.",
    }
    lines.append(verdict_map.get(overall, ""))

    # ── Spectral ────────────────────────────────────────────────────────────
    if agonist.fatigue_level != "none" and antagonist.fatigue_level != "none":
        lines.append(
            f"Both the agonist (MDF slope {agonist.mdf_slope_hz_per_s:.3f} Hz/s) "
            f"and antagonist (MDF slope {antagonist.mdf_slope_hz_per_s:.3f} Hz/s) "
            "showed spectral fatigue. Co-fatigue of both muscles in an "
            "antagonist pair suggests whole-joint neuromuscular fatigue "
            "rather than isolated peripheral fatigue."
        )
    elif agonist.fatigue_level != "none":
        lines.append(
            f"The agonist showed spectral fatigue "
            f"(MDF slope {agonist.mdf_slope_hz_per_s:.3f} Hz/s) while "
            "the antagonist remained stable — consistent with isolated "
            "agonist peripheral fatigue."
        )
    elif antagonist.fatigue_level != "none":
        lines.append(
            "Unusually, the antagonist showed spectral fatigue while the "
            "agonist remained stable. This may indicate compensatory "
            "antagonist co-contraction to maintain joint stability."
        )
    else:
        lines.append(
            "Both channels showed stable spectral frequencies — "
            "no peripheral fatigue detected."
        )

    # ── Co-activation ───────────────────────────────────────────────────────
    car_pct = round(avg_car * 100, 1)
    if car_level == "low" and not car_trending:
        lines.append(
            f"Co-activation was low (avg CAR {car_pct}%) and stable — "
            "efficient antagonist control throughout the session."
        )
    elif car_level == "low" and car_trending:
        lines.append(
            f"Co-activation started low (CAR {car_pct}%) but increased "
            f"over time (slope {car_slope:.4f}/s) — early neuromuscular "
            "fatigue pattern emerging."
        )
    elif car_level == "moderate" and not car_trending:
        lines.append(
            f"Moderate co-activation (avg CAR {car_pct}%) — "
            "the tricep was providing joint stability throughout."
        )
    elif car_level == "moderate" and car_trending:
        lines.append(
            f"Moderate co-activation (avg CAR {car_pct}%) with a rising "
            f"trend (slope {car_slope:.4f}/s) — neuromuscular fatigue "
            "progressively reducing joint control efficiency."
        )
    else:
        lines.append(
            f"High co-activation throughout (avg CAR {car_pct}%) — "
            "significant antagonist recruitment consistent with joint "
            "fatigue or instability."
        )

    # ── Envelope correlation ────────────────────────────────────────────────
    if envelope_corr > 0.7:
        lines.append(
            f"The two muscle envelopes were highly correlated "
            f"(r = {envelope_corr:.2f}), suggesting synchronised "
            "recruitment — typical of well-coordinated movement."
        )
    elif envelope_corr > 0.3:
        lines.append(
            f"Moderate envelope correlation (r = {envelope_corr:.2f}) — "
            "the muscles activate in a broadly similar pattern with "
            "some independent modulation."
        )
    else:
        lines.append(
            f"Low envelope correlation (r = {envelope_corr:.2f}) — "
            "bicep and tricep are activating asynchronously, which may "
            "reflect complex coordination demands or technique variability."
        )

    # ── Convergence note ────────────────────────────────────────────────────
    indices_agree = (
        (agonist.fatigue_level != "none") == car_trending
    )
    if indices_agree and overall != "none":
        lines.append(
            "Spectral and co-activation indices converge — "
            "this increases confidence in the fatigue assessment."
        )
    elif not indices_agree and overall != "none":
        lines.append(
            "The spectral and co-activation indices diverge. "
            "This is an interesting finding: consider whether fatigue "
            "is primarily peripheral (spectral) or central/coordinative "
            "(co-activation) in nature — a useful discussion point for "
            "the thesis."
        )

    return "\n\n".join(lines)


# ── old /analyze summary (unchanged) ──────────────────────────────────────────

def _summary(d: dict) -> str:
    level = d["fatigue_level"]
    slope = d["mdf_slope_hz_per_s"]
    norm  = d["mdf_slope_norm_pct_per_s"]
    drop  = d["mdf_start_hz"] - d["mdf_end_hz"]
    dur   = d["duration_s"]
    act   = d["active_windows"]
    tot   = d["total_windows"]

    lines = [
        f"Session: {dur:.0f}s, {act}/{tot} windows active.",
        f"Peak activation: {d['peak_mvc_pct']:.0f}% MVC.",
    ]
    if level == "none":
        lines.append("No significant fatigue detected.")
    elif level == "mild":
        lines.append(
            f"Mild fatigue (MDF slope {slope:.3f} Hz/s, {norm:.3f}%/s). "
            f"MDF dropped {drop:.1f} Hz — early metabolic accumulation."
        )
    elif level == "moderate":
        lines.append(
            f"Moderate fatigue (MDF slope {slope:.3f} Hz/s, {norm:.3f}%/s). "
            f"Clear {drop:.1f} Hz spectral downshift — good training stimulus."
        )
    else:
        lines.append(
            f"Severe fatigue (MDF slope {slope:.3f} Hz/s, {norm:.3f}%/s). "
            f"MDF dropped {drop:.1f} Hz. Allow adequate recovery."
        )
    return " ".join(lines)


# ── endpoints ──────────────────────────────────────────────────────────────────

@app.post("/analyze", response_model=FatigueReport)
async def analyze(
    file:     UploadFile      = File(...),
    mvc_adc:  Optional[float] = Form(None),
    window_s: float           = Form(2.0),
    step_s:   float           = Form(0.5),
):
    """Single-channel analysis — unchanged from v2."""
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Cannot parse CSV: {e}")

    t_col   = _find_col(df, "time")
    emg_col = _find_col(df, "emg")
    if not t_col or not emg_col:
        raise HTTPException(400,
            f"Cannot find time/emg columns. Found: {list(df.columns)}")

    t   = df[t_col].to_numpy(dtype=float)
    raw = df[emg_col].to_numpy(dtype=float)
    mask = np.isfinite(t) & np.isfinite(raw)
    t, raw = t[mask], raw[mask]

    if len(t) < 500:
        raise HTTPException(400, "Signal too short (< 500 samples).")

    fs = _detect_fs(t)
    if not (100 <= fs <= 10000):
        raise HTTPException(400, f"Implausible sampling rate: {fs:.1f} Hz")

    mvc_col = _find_col(df, "mvc")
    if mvc_adc is not None:
        mvc_v      = mvc_adc / ADC_MAX * ADC_VREF
        mvc_source = "provided"
    elif mvc_col:
        mvc_v      = float(df[mvc_col].dropna().iloc[0]) / ADC_MAX * ADC_VREF
        mvc_source = "provided"
    else:
        mvc_v      = float(_envelope(
            _bandpass(_to_volts(raw), fs), fs).max())
        mvc_source = "estimated_from_session"

    report, _, _, _ = _analyze_channel(
        t, raw, fs, mvc_v, mvc_source, "agonist", window_s, step_s)

    # Re-pack as FatigueReport (flat v2 shape)
    return FatigueReport(
        duration_s   = report.duration_s,
        fs_hz        = report.fs_hz,
        n_samples    = report.n_samples,
        notch_applied= report.notch_applied,
        mvc_source   = report.mvc_source,
        mvc_v        = report.mvc_v,
        threshold_adc= report.threshold_adc,
        peak_mvc_pct = report.peak_mvc_pct,
        mean_mvc_pct = report.mean_mvc_pct,
        global_mdf_hz= report.global_mdf_hz,
        global_mnf_hz= report.global_mnf_hz,
        mdf_start_hz = report.mdf_start_hz,
        mdf_end_hz   = report.mdf_end_hz,
        mdf_slope_hz_per_s       = report.mdf_slope_hz_per_s,
        mdf_slope_norm_pct_per_s = report.mdf_slope_norm_pct_per_s,
        mnf_start_hz = report.mnf_start_hz,
        mnf_end_hz   = report.mnf_end_hz,
        mnf_slope_hz_per_s       = report.mnf_slope_hz_per_s,
        mnf_slope_norm_pct_per_s = report.mnf_slope_norm_pct_per_s,
        active_windows  = report.active_windows,
        total_windows   = report.total_windows,
        fatigue_detected= report.fatigue_detected,
        fatigue_level   = report.fatigue_level,
        summary         = report.summary,
        envelope_norm_pct  = report.envelope_norm_pct,
        envelope_trend_pct = report.envelope_trend_pct,
        time_envelope      = report.time_envelope,
        frequency_series   = report.frequency_series,
        regression_mdf = report.regression_mdf,
        regression_mnf = report.regression_mnf,
        regression_t   = report.regression_t,
    )


@app.post("/analyze_dual", response_model=DualFatigueReport)
async def analyze_dual(
    file:      UploadFile      = File(...),
    mvc1_adc:  Optional[float] = Form(None),   # bicep MVC in ADC units
    mvc2_adc:  Optional[float] = Form(None),   # tricep MVC in ADC units
    exercise:  str             = Form("Bicep Curl"),
    window_s:  float           = Form(2.0),
    step_s:    float           = Form(0.5),
):
    """
    Dual-channel fatigue analysis.

    Parameters
    ----------
    file      : CSV with columns time_s, emg1_adc, emg2_adc [, mvc1, mvc2]
    mvc1_adc  : Bicep MVC in raw ADC units from calibration
    mvc2_adc  : Tricep MVC in raw ADC units from calibration
    exercise  : "Bicep Curl" | "Tricep Pushdown"
    window_s  : Welch window in seconds
    step_s    : Step between windows in seconds
    """
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Cannot parse CSV: {e}")

    t_col    = _find_col(df, "time")
    emg1_col = _find_col(df, "emg1")
    emg2_col = _find_col(df, "emg2")

    if not t_col or not emg1_col or not emg2_col:
        raise HTTPException(400,
            f"Expected columns: time_s, emg1_adc, emg2_adc. "
            f"Found: {list(df.columns)}")

    t    = df[t_col].to_numpy(dtype=float)
    raw1 = df[emg1_col].to_numpy(dtype=float)
    raw2 = df[emg2_col].to_numpy(dtype=float)

    mask = np.isfinite(t) & np.isfinite(raw1) & np.isfinite(raw2)
    t, raw1, raw2 = t[mask], raw1[mask], raw2[mask]

    if len(t) < 500:
        raise HTTPException(400, "Signal too short (< 500 samples).")

    fs = _detect_fs(t)
    if not (100 <= fs <= 10000):
        raise HTTPException(400, f"Implausible sampling rate: {fs:.1f} Hz")

    # ── determine which channel is agonist ───────────────────────────────────
    is_bicep_curl = "bicep" in exercise.lower() or exercise == "Bicep Curl"
    # EMG1 = A0 = bicep sensor, EMG2 = A1 = tricep sensor (hardware fixed)
    # For Bicep Curl:      bicep = agonist  (EMG1), tricep = antagonist (EMG2)
    # For Tricep Pushdown: tricep = agonist (EMG2), bicep  = antagonist (EMG1)
    # Channel labels are always bicep/tricep regardless of agonist role

    # ── MVC values ───────────────────────────────────────────────────────────
    mvc1_col = _find_col(df, "mvc")   # column in CSV for bicep MVC
    mvc2_col = _find_col(df, "mvc2")  # column in CSV for tricep MVC

    def _get_mvc(form_val, col, raw_ch, label):
        if form_val is not None:
            return form_val / ADC_MAX * ADC_VREF, "provided"
        if col and col in df.columns:
            v = df[col].dropna()
            if not v.empty:
                return float(v.iloc[0]) / ADC_MAX * ADC_VREF, "provided"
        env = _envelope(_bandpass(_to_volts(raw_ch), fs), fs)
        return float(env.max()), f"estimated_{label}"

    mvc1_v, src1 = _get_mvc(mvc1_adc, mvc1_col, raw1, "bicep")
    mvc2_v, src2 = _get_mvc(mvc2_adc, mvc2_col, raw2, "tricep")

    # ── per-channel analysis ─────────────────────────────────────────────────
    bicep_report,  _, env1, env_norm1 = _analyze_channel(
        t, raw1, fs, mvc1_v, src1, "bicep",  window_s, step_s)
    tricep_report, _, env2, env_norm2 = _analyze_channel(
        t, raw2, fs, mvc2_v, src2, "tricep", window_s, step_s)

    # ── co-activation ────────────────────────────────────────────────────────
    th1 = bicep_report.threshold_adc  / ADC_MAX * ADC_VREF
    th2 = tricep_report.threshold_adc / ADC_MAX * ADC_VREF

    (car_series, avg_car, car_slope,
     car_start, car_end,
     car_level, car_trending) = _compute_car(
        env1, env2, t, th1, th2, fs, step_s)

    # ── envelope correlation ─────────────────────────────────────────────────
    envelope_corr = _envelope_correlation(env_norm1, env_norm2)

    # ── overall verdict ──────────────────────────────────────────────────────
    agonist_report    = bicep_report  if is_bicep_curl else tricep_report
    antagonist_report = tricep_report if is_bicep_curl else bicep_report

    overall = _overall_verdict(
        agonist_report.fatigue_level,
        antagonist_report.fatigue_level,
        car_level,
        car_trending,
    )

    conclusions_text = _conclusions(
        agonist        = agonist_report,
        antagonist     = antagonist_report,
        avg_car        = avg_car,
        car_slope      = car_slope,
        car_level      = car_level,
        car_trending   = car_trending,
        envelope_corr  = envelope_corr,
        overall        = overall,
        exercise_name  = exercise,
    )

    return DualFatigueReport(
        bicep  = bicep_report,
        tricep = tricep_report,

        car_series   = car_series,
        avg_car      = round(avg_car,   4),
        car_slope    = round(car_slope, 6),
        car_start    = round(car_start, 4),
        car_end      = round(car_end,   4),
        car_level    = car_level,
        car_trending = car_trending,

        envelope_correlation = envelope_corr,

        spectral_fatigue_agonist    = agonist_report.fatigue_level,
        spectral_fatigue_antagonist = antagonist_report.fatigue_level,
        kinematic_fatigue           = "not_assessed",
        overall_verdict             = overall,
        conclusions                 = conclusions_text,
    )


@app.get("/health")
def health():
    return {"status": "ok", "version": "3.0.0"}
