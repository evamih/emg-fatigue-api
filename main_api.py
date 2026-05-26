"""
EMG Muscle Fatigue Analysis API
--------------------------------
POST /analyze  — upload a CSV file, get back fatigue metrics + summary
GET  /health   — sanity check

Expected CSV columns (flexible, see COLUMN_ALIASES below):
    time (s), emg (ADC int16 or volts)

Optional CSV columns:
    mvc  — if present, first non-NaN value is used as MVC reference

Run locally:
    pip install fastapi uvicorn scipy numpy pandas python-multipart
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, welch
from typing import Optional
import io

# ── constants ──────────────────────────────────────────────────────────────

ADC_MAX   = 4095.0
ADC_VREF  = 3.3          # volts
BP_LOW    = 20           # Hz  bandpass low
BP_HIGH   = 450          # Hz  bandpass high
NOTCH_F0  = 50           # Hz  power line
NOTCH_Q   = 30
ENV_WIN_MS = 250         # ms  moving average envelope window

# column name aliases — we accept several naming conventions
COLUMN_ALIASES = {
    "time": ["time", "t", "timestamp", "Time", "T"],
    "emg":  ["emg", "emgData", "emg_data", "EMG", "value", "adc"],
    "mvc":  ["mvc", "MVC", "mvc_ref"],
}

app = FastAPI(title="EMG Fatigue API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ── response models ────────────────────────────────────────────────────────

class FrequencyPoint(BaseModel):
    time_s: float
    mdf_hz: Optional[float]
    mnf_hz: Optional[float]

class FatigueReport(BaseModel):
    # recording info
    duration_s: float
    fs_hz: float
    n_samples: int

    # MVC
    mvc_source: str          # "provided" | "estimated_from_session"
    mvc_v: float

    # amplitude
    peak_mvc_pct: float
    mean_mvc_pct: float

    # spectral fatigue indices
    mdf_start_hz: float
    mdf_end_hz: float
    mdf_slope_hz_per_s: float
    mnf_start_hz: float
    mnf_end_hz: float
    mnf_slope_hz_per_s: float

    # fatigue verdict
    fatigue_detected: bool
    fatigue_level: str       # "none" | "mild" | "moderate" | "severe"
    summary: str

    # time-series for plotting
    envelope_norm_pct: list[float]   # downsampled to max 500 pts
    time_envelope: list[float]
    frequency_series: list[FrequencyPoint]


# ── signal processing helpers ──────────────────────────────────────────────

def _find_column(df: pd.DataFrame, key: str) -> Optional[str]:
    for alias in COLUMN_ALIASES[key]:
        if alias in df.columns:
            return alias
    return None


def _detect_fs(t: np.ndarray) -> float:
    diffs = np.diff(t)
    median_dt = np.median(diffs[diffs > 0])
    return 1.0 / median_dt


def _to_volts(emg: np.ndarray) -> np.ndarray:
    """Auto-detect ADC vs volt signal and convert to volts."""
    if np.max(np.abs(emg)) > 10:          # clearly ADC range
        return emg / ADC_MAX * ADC_VREF
    return emg.copy()                      # already volts


def _bandpass(emg_v: np.ndarray, fs: float) -> np.ndarray:
    fnyq = fs / 2
    b, a = butter(4, [BP_LOW / fnyq, BP_HIGH / fnyq], "bandpass")
    return filtfilt(b, a, emg_v)


def _notch_chain(emg_filt: np.ndarray, fs: float) -> np.ndarray:
    fnyq = fs / 2
    for i in range(1, 9):
        wo = NOTCH_F0 * i / fnyq
        if wo >= 1.0:
            break
        b_n, a_n = iirnotch(wo, NOTCH_Q)
        emg_filt = filtfilt(b_n, a_n, emg_filt)
    return emg_filt


def _envelope(emg_filt: np.ndarray, fs: float) -> np.ndarray:
    win = max(1, int(ENV_WIN_MS / 1000 * fs))
    return np.convolve(np.abs(emg_filt), np.ones(win) / win, mode="same")


def _activity_threshold(env: np.ndarray, t: np.ndarray) -> float:
    """
    Use first 3 s as rest baseline if available, else fall back to
    mean + 1.5*std of the full envelope.
    """
    rest_mask = t < 3.0
    if rest_mask.sum() > 50:
        rest = env[rest_mask]
        return rest.mean() + 2 * rest.std()
    return env.mean() * 0.3   # fallback


def _spectral_metrics(segment: np.ndarray, fs: float):
    """Return (MDF, MNF) for one segment. Returns (nan, nan) if too short."""
    nperseg = min(len(segment), max(64, int(fs)))
    if len(segment) < nperseg:
        return np.nan, np.nan
    f, pxx = welch(segment, fs=fs, window="hamming",
                   nperseg=nperseg, noverlap=nperseg // 2)
    if pxx.sum() == 0:
        return np.nan, np.nan
    cum = np.cumsum(pxx)
    mdf = float(f[np.searchsorted(cum, cum[-1] / 2)])
    mnf = float(np.sum(f * pxx) / np.sum(pxx))
    return mdf, mnf


def _fatigue_level(slope_hz_per_s: float) -> tuple[bool, str]:
    """
    Classify fatigue from MDF slope.
    Thresholds are conservative for a 60s isometric contraction.
    """
    if slope_hz_per_s >= -0.05:
        return False, "none"
    elif slope_hz_per_s >= -0.15:
        return True, "mild"
    elif slope_hz_per_s >= -0.30:
        return True, "moderate"
    else:
        return True, "severe"


def _build_summary(report_data: dict) -> str:
    level = report_data["fatigue_level"]
    mdf_s = report_data["mdf_slope_hz_per_s"]
    peak  = report_data["peak_mvc_pct"]
    dur   = report_data["duration_s"]

    intro = (
        f"Session lasted {dur:.0f}s. "
        f"Peak muscle activation reached {peak:.0f}% MVC. "
    )

    if level == "none":
        trend = (
            "No significant fatigue was detected — "
            "spectral frequencies remained stable throughout the session. "
            "Consider increasing load or duration for a more demanding stimulus."
        )
    elif level == "mild":
        trend = (
            f"Mild fatigue was detected (MDF slope: {mdf_s:.3f} Hz/s). "
            "A small downward shift in median frequency indicates early-stage "
            "fatigue. The muscle was still well within its working capacity."
        )
    elif level == "moderate":
        trend = (
            f"Moderate fatigue was detected (MDF slope: {mdf_s:.3f} Hz/s). "
            "A clear downward spectral shift occurred over the session, "
            "consistent with accumulating metabolic byproducts and progressive "
            "motor unit substitution. Good training stimulus."
        )
    else:
        trend = (
            f"Severe fatigue was detected (MDF slope: {mdf_s:.3f} Hz/s). "
            "The median frequency dropped sharply — the muscle reached "
            "significant fatigue. Monitor recovery time before the next session."
        )

    return intro + trend


# ── main endpoint ──────────────────────────────────────────────────────────

@app.post("/analyze", response_model=FatigueReport)
async def analyze(
    file: UploadFile = File(...),
    mvc_adc: Optional[float] = Form(None),   # pass MVC from app if known
    window_s: float = Form(2.0),             # spectral window length
    step_s:   float = Form(0.5),             # step between windows
):
    """
    Analyze an EMG session CSV file for muscle fatigue.

    Parameters
    ----------
    file     : CSV with columns for time and EMG (ADC or volts)
    mvc_adc  : MVC reference in raw ADC units (from calibration session).
               If omitted, the session peak envelope is used (less accurate).
    window_s : Welch window length in seconds (default 2.0)
    step_s   : Step between windows in seconds (default 0.5)
    """
    # ── load CSV ──
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    t_col   = _find_column(df, "time")
    emg_col = _find_column(df, "emg")
    if not t_col or not emg_col:
        raise HTTPException(
            400,
            f"Could not find time/emg columns. "
            f"Found columns: {list(df.columns)}. "
            f"Expected one of: {COLUMN_ALIASES}"
        )

    t   = df[t_col].to_numpy(dtype=float)
    raw = df[emg_col].to_numpy(dtype=float)

    # drop NaN / inf
    mask = np.isfinite(t) & np.isfinite(raw)
    t, raw = t[mask], raw[mask]

    if len(t) < 200:
        raise HTTPException(400, "Signal too short (< 200 samples).")

    fs = _detect_fs(t)
    if fs < 100 or fs > 10000:
        raise HTTPException(400, f"Implausible sampling rate detected: {fs:.1f} Hz")

    # ── preprocessing ──
    emg_v    = _to_volts(raw)
    emg_filt = _bandpass(emg_v, fs)
    emg_filt = _notch_chain(emg_filt, fs)
    env      = _envelope(emg_filt, fs)

    # ── MVC ──
    mvc_col = _find_column(df, "mvc")
    if mvc_adc is not None:
        mvc_v      = mvc_adc / ADC_MAX * ADC_VREF
        mvc_source = "provided"
    elif mvc_col:
        mvc_raw    = df[mvc_col].dropna().iloc[0]
        mvc_v      = float(mvc_raw) / ADC_MAX * ADC_VREF
        mvc_source = "provided"
    else:
        mvc_v      = float(env.max())
        mvc_source = "estimated_from_session"

    mvc_v = max(mvc_v, 1e-6)   # avoid division by zero

    env_norm = env / mvc_v * 100.0

    # ── spectral fatigue metrics ──
    win_n  = int(window_s * fs)
    step_n = int(step_s * fs)
    th     = _activity_threshold(env, t)

    freq_series: list[FrequencyPoint] = []
    for k in range(0, len(emg_filt) - win_n, step_n):
        seg      = emg_filt[k : k + win_n]
        t_center = float(t[k + win_n // 2])
        rms_seg  = float(np.sqrt(np.mean(seg ** 2)))

        if rms_seg < th * 0.5:   # rest period
            freq_series.append(FrequencyPoint(
                time_s=t_center, mdf_hz=None, mnf_hz=None))
            continue

        mdf, mnf = _spectral_metrics(seg, fs)
        freq_series.append(FrequencyPoint(
            time_s=t_center,
            mdf_hz=round(mdf, 2) if np.isfinite(mdf) else None,
            mnf_hz=round(mnf, 2) if np.isfinite(mnf) else None,
        ))

    # active points only for slope
    active = [(p.time_s, p.mdf_hz, p.mnf_hz)
              for p in freq_series if p.mdf_hz is not None]

    if len(active) < 4:
        raise HTTPException(
            422,
            "Not enough active windows for fatigue analysis. "
            "Check that the signal contains muscle contractions above rest level."
        )

    at, am, an = zip(*active)
    at = np.array(at); am = np.array(am); an = np.array(an)

    p_mdf = np.polyfit(at, am, 1)
    p_mnf = np.polyfit(at, an, 1)

    mdf_start = float(np.polyval(p_mdf, at[0]))
    mdf_end   = float(np.polyval(p_mdf, at[-1]))
    mnf_start = float(np.polyval(p_mnf, at[0]))
    mnf_end   = float(np.polyval(p_mnf, at[-1]))

    fatigue_detected, fatigue_level = _fatigue_level(float(p_mdf[0]))

    # ── downsample envelope for response ──
    MAX_PTS = 500
    if len(env_norm) > MAX_PTS:
        idx = np.linspace(0, len(env_norm) - 1, MAX_PTS, dtype=int)
        env_out = env_norm[idx].tolist()
        t_out   = t[idx].tolist()
    else:
        env_out = env_norm.tolist()
        t_out   = t.tolist()

    report_data = dict(
        duration_s         = round(float(t[-1] - t[0]), 2),
        fs_hz              = round(fs, 1),
        n_samples          = len(raw),
        mvc_source         = mvc_source,
        mvc_v              = round(mvc_v, 5),
        peak_mvc_pct       = round(float(env_norm.max()), 1),
        mean_mvc_pct       = round(float(env_norm[env_norm > 5].mean())
                                   if (env_norm > 5).any() else 0.0, 1),
        mdf_start_hz       = round(mdf_start, 1),
        mdf_end_hz         = round(mdf_end, 1),
        mdf_slope_hz_per_s = round(float(p_mdf[0]), 4),
        mnf_start_hz       = round(mnf_start, 1),
        mnf_end_hz         = round(mnf_end, 1),
        mnf_slope_hz_per_s = round(float(p_mnf[0]), 4),
        fatigue_detected   = fatigue_detected,
        fatigue_level      = fatigue_level,
        summary            = "",
        envelope_norm_pct  = [round(v, 2) for v in env_out],
        time_envelope      = [round(v, 3) for v in t_out],
        frequency_series   = freq_series,
    )
    report_data["summary"] = _build_summary(report_data)
    return FatigueReport(**report_data)


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}
