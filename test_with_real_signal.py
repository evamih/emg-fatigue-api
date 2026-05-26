"""
Integration test — uses semnal_alex_1.mat against the running API.

Usage:
    # Terminal 1
    uvicorn main:app --reload --port 8000

    # Terminal 2
    python test_with_real_signal.py [path/to/semnal_alex_1.mat]
"""

import sys
import io
import json
import numpy as np
import pandas as pd
import requests

# ── load .mat if scipy available, else expect a CSV path ──────────────────────
mat_path = sys.argv[1] if len(sys.argv) > 1 else "semnal_alex_1.mat"
mvc_adc  = float(sys.argv[2]) if len(sys.argv) > 2 else None

try:
    import scipy.io as sio
    mat = sio.loadmat(mat_path)
    emg = mat["emgData"].flatten().astype(float)
    t   = mat["t"].flatten()
    if mvc_adc is None and "MVC" in mat:
        mvc_adc = float(mat["MVC"].flat[0])
    df = pd.DataFrame({"time": t, "emgData": emg})
    print(f"Loaded .mat: {len(emg)} samples, fs≈{1/float(np.median(np.diff(t))):.0f} Hz")
except Exception as e:
    print(f"Could not load .mat ({e}), trying as CSV…")
    df = pd.read_csv(mat_path)

csv_bytes = df.to_csv(index=False).encode()
print(f"MVC reference: {mvc_adc:.1f} ADC units")

# ── health check ──────────────────────────────────────────────────────────────
try:
    r = requests.get("http://localhost:8000/health", timeout=5)
    print(f"Health: {r.json()}")
except Exception as e:
    print(f"Server not reachable: {e}")
    sys.exit(1)

# ── POST /analyze ─────────────────────────────────────────────────────────────
fields = {"window_s": "2.0", "step_s": "0.5"}
if mvc_adc is not None:
    fields["mvc_adc"] = str(mvc_adc)

resp = requests.post(
    "http://localhost:8000/analyze",
    files={"file": ("session.csv", io.BytesIO(csv_bytes), "text/csv")},
    data=fields,
    timeout=60,
)

if resp.status_code != 200:
    print(f"ERROR {resp.status_code}: {resp.text}")
    sys.exit(1)

r = resp.json()

# ── print report ──────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  EMG FATIGUE REPORT")
print("="*55)
print(f"  Duration:      {r['duration_s']}s  |  fs: {r['fs_hz']} Hz")
print(f"  Notch applied: {r['notch_applied']}")
print(f"  MVC:           {r['mvc_v']:.5f} V  ({r['mvc_source']})")
print(f"  Threshold:     {r['threshold_adc']:.1f} ADC")
print(f"  Active windows:{r['active_windows']} / {r['total_windows']}")
print(f"  Peak %MVC:     {r['peak_mvc_pct']}%")
print()
print(f"  Global MDF:    {r['global_mdf_hz']} Hz")
print(f"  Global MNF:    {r['global_mnf_hz']} Hz")
print()
print(f"  MDF slope:     {r['mdf_slope_hz_per_s']} Hz/s"
      f"  ({r['mdf_slope_norm_pct_per_s']} %/s normalised)")
print(f"  MDF range:     {r['mdf_start_hz']} → {r['mdf_end_hz']} Hz")
print(f"  MNF slope:     {r['mnf_slope_hz_per_s']} Hz/s")
print()
print(f"  Fatigue level: {r['fatigue_level'].upper()}"
      f"  (detected={r['fatigue_detected']})")
print()
print(f"  Summary:")
for line in r['summary'].split('. '):
    if line.strip():
        print(f"    {line.strip()}.")
print("="*55)
print(f"\n  Frequency series points: {len(r['frequency_series'])}")
print(f"  Envelope points:         {len(r['envelope_norm_pct'])}")

# ── save full JSON for inspection ─────────────────────────────────────────────
with open("last_report.json", "w") as f:
    json.dump(r, f, indent=2)
print("\n  Full report saved to last_report.json")
