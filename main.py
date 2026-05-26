"""
EMG Muscle Fatigue Analysis API  v2.0
--------------------------------------
Mirrors the refined MATLAB pipeline:
  - Butterworth bandpass 20–450 Hz
  - 50 Hz notch only if power spike detected
  - Data-driven threshold from rest baseline (first 3 s)
  - Welch PSD on filtered signal restricted to [20, 450] Hz
  - MDF / MNF computed with polyfit linear regression over active windows
  - Normalised slope (%/s) for cross-subject comparison
  - Robust to intermittent bicep curl sessions (multiple contractions)

POST /analyze  — CSV upload → FatigueReport JSON
GET  /health   — liveness check

Run locally:
    venv\Scripts/activate
    uvicorn main:app --reload --port 8000
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
ADC_MAX  = 4095.0
ADC_VREF = 3.3
BP_LOW   = 20
BP_HIGH  = 450
NOTCH_F0 = 50
NOTCH_Q  = 30
ENV_WIN_MS     = 250   # moving-average envelope window
REST_BASELINE_S = 3.0  # seconds used for threshold estimation
NOTCH_SPIKE_DB  = 6.0  # minimum dB above neighbours to trigger notch

COLUMN_ALIASES = {
    "time": ["time", "t", "timestamp", "Time", "T"],
    "emg":  ["emg", "emgData", "emg_data", "EMG", "value", "adc"],
    "mvc":  ["mvc", "MVC", "mvc_ref"],
}

app = FastAPI(title="EMG Fatigue API", version="2.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"]
)

# ── response models ────────────────────────────────────────────────────────────

class FrequencyPoint(BaseModel):
    time_s: float
    mdf_hz: Optional[float] = None
    mnf_hz: Optional[float] = None

class FatigueReport(BaseModel):
    # recording info
    duration_s: float
    fs_hz: float
    n_samples: int
    notch_applied: bool

    # MVC
    mvc_source: str
    mvc_v: float
    threshold_adc: float

    # amplitude
    peak_mvc_pct: float
    mean_mvc_pct: float

    # global spectral metrics (over whole signal)
    global_mdf_hz: float
    global_mnf_hz: float

    # fatigue regression
    mdf_start_hz: float
    mdf_end_hz:   float
    mdf_slope_hz_per_s: float
    mdf_slope_norm_pct_per_s: float   # normalised slope

    mnf_start_hz: float
    mnf_end_hz:   float
    mnf_slope_hz_per_s: float
    mnf_slope_norm_pct_per_s: float

    active_windows: int
    total_windows:  int

    # verdict
    fatigue_detected: bool
    fatigue_level: str    # none | mild | moderate | severe
    summary: str

    # time-series (downsampled ≤500 pts each)
    envelope_norm_pct: list[float]
    envelope_trend_pct: list[float]   # 2 s smoothed amplitude trend
    time_envelope: list[float]

    frequency_series: list[FrequencyPoint]

    # regression line endpoints for drawing in Flutter
    regression_mdf: list[float]   # [y_start, y_end]
    regression_mnf: list[float]
    regression_t:   list[float]   # [t_start, t_end]


# ── signal processing ──────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, key: str) -> Optional[str]:
    for alias in COLUMN_ALIASES[key]:
        if alias in df.columns:
            return alias
    return None


def _detect_fs(t: np.ndarray) -> float:
    diffs = np.diff(t)
    return 1.0 / float(np.median(diffs[diffs > 0]))


def _to_volts(emg: np.ndarray) -> np.ndarray:
    """Auto-detect ADC vs volt and convert to volts."""
    return emg / ADC_MAX * ADC_VREF if np.max(np.abs(emg)) > 10 else emg.copy()


def _bandpass(emg: np.ndarray, fs: float) -> np.ndarray:
    fnyq = fs / 2
    b, a = butter(4, [BP_LOW / fnyq, BP_HIGH / fnyq], "bandpass")
    return filtfilt(b, a, emg)


def _check_notch_needed(emg_filt: np.ndarray, fs: float) -> bool:
    """Return True only if there's a real 50 Hz power spike."""
    nperseg = min(len(emg_filt), int(fs))
    f, pxx = welch(emg_filt, fs=fs, nperseg=nperseg)
    idx = np.argmin(np.abs(f - NOTCH_F0))
    if idx < 2 or idx > len(pxx) - 3:
        return False
    local_noise = np.median(pxx[max(0, idx-5):idx+6])
    spike_ratio = 10 * np.log10(pxx[idx] / (local_noise + 1e-30))
    return bool(spike_ratio > NOTCH_SPIKE_DB)


def _apply_notch(emg: np.ndarray, fs: float) -> np.ndarray:
    fnyq = fs / 2
    wo = NOTCH_F0 / fnyq
    b, a = iirnotch(wo, NOTCH_Q)
    return filtfilt(b, a, emg)


def _envelope(emg_filt: np.ndarray, fs: float) -> np.ndarray:
    win = max(1, int(ENV_WIN_MS / 1000 * fs))
    return np.convolve(np.abs(emg_filt), np.ones(win) / win, mode="same")


def _data_driven_threshold(env: np.ndarray, t: np.ndarray, fs: float) -> float:
    """
    Data-driven activity threshold.

    Priority 1 — if the first REST_BASELINE_S seconds look like genuine rest
    (low mean, low std relative to session peak), use mean + 2.5*std of that
    window. This is the ideal case: subject at rest before first contraction.

    Priority 2 — if the signal starts contracting immediately (no clear rest),
    use a robust percentile spread: p10 + 0.3*(p90-p10). This sits well below
    active peaks while staying above low-level background noise.
    """
    rest_mask = t < REST_BASELINE_S
    if rest_mask.sum() > int(0.5 * fs):
        rest = env[rest_mask]
        candidate = float(rest.mean() + 2.5 * rest.std())
        # sanity check: threshold must leave at least 20% of windows active
        active_frac = float(np.mean(env > candidate))
        if active_frac >= 0.20:
            return candidate

    # fallback: robust spread ignoring outliers
    p10 = float(np.percentile(env, 10))
    p90 = float(np.percentile(env, 90))
    return p10 + 0.30 * (p90 - p10)


def _spectral_metrics(segment: np.ndarray, fs: float):
    """(MDF, MNF) restricted to [BP_LOW, BP_HIGH]. Returns (nan, nan) if bad."""
    nperseg = min(len(segment), max(64, int(fs)))
    if len(segment) < nperseg:
        return np.nan, np.nan
    f, pxx = welch(segment, fs=fs, window="hamming",
                   nperseg=nperseg, noverlap=nperseg // 2)
    band = (f >= BP_LOW) & (f <= BP_HIGH)
    f_b, p_b = f[band], pxx[band]
    if p_b.sum() == 0:
        return np.nan, np.nan
    cum = np.cumsum(p_b)
    mdf = float(f_b[np.searchsorted(cum, cum[-1] / 2)])
    mnf = float(np.sum(f_b * p_b) / np.sum(p_b))
    return mdf, mnf


def _medfilt1_nan(arr: np.ndarray, k: int = 3) -> np.ndarray:
    """Median filter ignoring NaNs."""
    out = arr.copy()
    half = k // 2
    for i in range(len(arr)):
        win = arr[max(0, i - half): i + half + 1]
        valid = win[~np.isnan(win)]
        if len(valid):
            out[i] = float(np.median(valid))
    return out


def _fatigue_level(slope: float) -> tuple[bool, str]:
    if slope >= -0.05:
        return False, "none"
    elif slope >= -0.15:
        return True, "mild"
    elif slope >= -0.30:
        return True, "moderate"
    return True, "severe"


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
        lines.append(
            "No significant fatigue detected — spectral frequencies "
            "remained stable. Consider longer or heavier holds."
        )
    elif level == "mild":
        lines.append(
            f"Mild fatigue (MDF slope {slope:.3f} Hz/s, "
            f"{norm:.3f}%/s normalised). "
            f"MDF dropped {drop:.1f} Hz — early metabolic accumulation."
        )
    elif level == "moderate":
        lines.append(
            f"Moderate fatigue (MDF slope {slope:.3f} Hz/s, "
            f"{norm:.3f}%/s normalised). "
            f"Clear {drop:.1f} Hz spectral downshift — good training stimulus."
        )
    else:
        lines.append(
            f"Severe fatigue (MDF slope {slope:.3f} Hz/s, "
            f"{norm:.3f}%/s normalised). "
            f"MDF dropped {drop:.1f} Hz — significant neuromuscular fatigue. "
            "Allow adequate recovery before next session."
        )
    return " ".join(lines)


# ── endpoint ───────────────────────────────────────────────────────────────────

@app.post("/analyze", response_model=FatigueReport)
async def analyze(
    file:      UploadFile       = File(...),
    mvc_adc:   Optional[float]  = Form(None),
    window_s:  float            = Form(2.0),
    step_s:    float            = Form(0.5),
):
    """
    Analyze a post-session EMG CSV for muscle fatigue.

    Parameters
    ----------
    file      : CSV with 'time' and 'emg' columns (ADC int16 or volts)
    mvc_adc   : MVC reference in raw ADC units from calibration session
    window_s  : Welch window in seconds (default 2.0)
    step_s    : Step between windows in seconds (default 0.5)
    """
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Cannot parse CSV: {e}")

    t_col   = _find_col(df, "time")
    emg_col = _find_col(df, "emg")
    if not t_col or not emg_col:
        raise HTTPException(
            400,
            f"Cannot find time/emg columns. Found: {list(df.columns)}. "
            f"Expected one of: {COLUMN_ALIASES}"
        )

    t   = df[t_col].to_numpy(dtype=float)
    raw = df[emg_col].to_numpy(dtype=float)
    mask = np.isfinite(t) & np.isfinite(raw)
    t, raw = t[mask], raw[mask]

    if len(t) < 500:
        raise HTTPException(400, "Signal too short (< 500 samples).")

    fs = _detect_fs(t)
    if not (100 <= fs <= 10000):
        raise HTTPException(400, f"Implausible sampling rate: {fs:.1f} Hz")

    # ── preprocessing ──────────────────────────────────────────────────────────
    emg_v    = _to_volts(raw)
    emg_filt = _bandpass(emg_v, fs)

    notch_applied = _check_notch_needed(emg_filt, fs)
    if notch_applied:
        emg_filt = _apply_notch(emg_filt, fs)

    env = _envelope(emg_filt, fs)

    # ── MVC ────────────────────────────────────────────────────────────────────
    mvc_col = _find_col(df, "mvc")
    if mvc_adc is not None:
        mvc_v      = mvc_adc / ADC_MAX * ADC_VREF
        mvc_source = "provided"
    elif mvc_col:
        mvc_v      = float(df[mvc_col].dropna().iloc[0]) / ADC_MAX * ADC_VREF
        mvc_source = "provided"
    else:
        mvc_v      = float(env.max())
        mvc_source = "estimated_from_session"

    mvc_v = max(mvc_v, 1e-6)
    env_norm = env / mvc_v * 100.0

    # ── threshold ──────────────────────────────────────────────────────────────
    th_v   = _data_driven_threshold(env, t, fs)
    th_adc = th_v / ADC_VREF * ADC_MAX   # back to ADC for reporting

    # ── global PSD metrics ─────────────────────────────────────────────────────
    nperseg_global = min(len(emg_filt), int(fs))
    f_g, pxx_g = welch(emg_filt, fs=fs, window="hamming",
                       nperseg=nperseg_global, noverlap=nperseg_global // 2)
    band = (f_g >= BP_LOW) & (f_g <= BP_HIGH)
    f_b, p_b = f_g[band], pxx_g[band]
    cum = np.cumsum(p_b)
    global_mdf = float(f_b[np.searchsorted(cum, cum[-1] / 2)])
    global_mnf = float(np.sum(f_b * p_b) / np.sum(p_b))

    # ── spectral fatigue series ────────────────────────────────────────────────
    win_n  = int(window_s * fs)
    step_n = int(step_s * fs)
    freq_series: list[FrequencyPoint] = []

    for k in range(0, len(emg_filt) - win_n, step_n):
        seg      = emg_filt[k: k + win_n]
        t_center = float(t[k + win_n // 2])
        # use envelope mean for activity check (more stable than raw RMS)
        env_mean = float(env[k: k + win_n].mean())

        if env_mean < th_v:
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
            f"Only {active_windows} active windows found (need ≥4). "
            "Check that the signal has muscle contractions above rest level, "
            f"or lower window_s (current: {window_s}s)."
        )

    # ── median-filter series and regression ───────────────────────────────────
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

    # ── amplitude trend (2 s smoothing for Flutter panel 1) ───────────────────
    trend_win = min(len(env_norm), int(2.0 * fs))
    env_trend = np.convolve(env_norm, np.ones(trend_win) / trend_win, mode="same")

    # ── downsample timelines to ≤500 pts ──────────────────────────────────────
    MAX_PTS = 500
    def _ds(arr):
        if len(arr) > MAX_PTS:
            idx = np.linspace(0, len(arr) - 1, MAX_PTS, dtype=int)
            return arr[idx]
        return arr

    env_out   = _ds(env_norm)
    trend_out = _ds(env_trend)
    t_out     = _ds(t)

    report_data = dict(
        duration_s   = round(float(t[-1] - t[0]), 2),
        fs_hz        = round(fs, 1),
        n_samples    = int(len(raw)),
        notch_applied= notch_applied,

        mvc_source   = mvc_source,
        mvc_v        = round(mvc_v, 5),
        threshold_adc= round(th_adc, 1),

        peak_mvc_pct = round(float(env_norm.max()), 1),
        mean_mvc_pct = round(float(env_norm[env_norm > 5].mean())
                             if (env_norm > 5).any() else 0.0, 1),

        global_mdf_hz= round(global_mdf, 2),
        global_mnf_hz= round(global_mnf, 2),

        mdf_start_hz = round(mdf_start, 1),
        mdf_end_hz   = round(mdf_end, 1),
        mdf_slope_hz_per_s       = round(mdf_slope, 4),
        mdf_slope_norm_pct_per_s = round(mdf_slope_norm, 4),

        mnf_start_hz = round(mnf_start, 1),
        mnf_end_hz   = round(mnf_end, 1),
        mnf_slope_hz_per_s       = round(mnf_slope, 4),
        mnf_slope_norm_pct_per_s = round(mnf_slope_norm, 4),

        active_windows= active_windows,
        total_windows = total_windows,

        fatigue_detected= fatigue_detected,
        fatigue_level   = fatigue_level,
        summary         = "",

        envelope_norm_pct  = [round(v, 2) for v in env_out.tolist()],
        envelope_trend_pct = [round(v, 2) for v in trend_out.tolist()],
        time_envelope      = [round(v, 3) for v in t_out.tolist()],

        frequency_series= freq_series,

        regression_mdf= [round(mdf_start, 2), round(mdf_end, 2)],
        regression_mnf= [round(mnf_start, 2), round(mnf_end, 2)],
        regression_t  = [round(float(t_v[0]), 2), round(float(t_v[-1]), 2)],
    )
    report_data["summary"] = _summary(report_data)
    return FatigueReport(**report_data)


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}
