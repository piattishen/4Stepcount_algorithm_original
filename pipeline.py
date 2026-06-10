"""
Four-Algorithm Step Count Pipeline
====================================
Reads an ActiGraph GT3X+ CSV file and runs four step counting algorithms:

  1. Verisense  – Python port of R verisense_count_steps.R (peak-detection)
  2. ADEPT      – Adaptive pattern segmentation via pyadept Python package
  3. OAK        – Continuous Wavelet Transform-based (forest-oak base.py)
  4. Oxford     – SSL/RF model via stepcount CLI

Input format (ActiGraph GT3X+):
  ------------ Data File Created By ActiGraph GT3X+ ActiLife v6.13.4 ...
  Start Time 18:00:00
  Start Date 11/1/2021
  ...
  Accelerometer X,Accelerometer Y,Accelerometer Z
  0.000,0.000,0.000
  ...

Output format per algorithm  (Time,Steps  at 10-second intervals):
  Time,Steps
  2021-11-01 18:00:00,0.0
  2021-11-01 18:00:10,5.0
  ...

Usage:
  python pipeline.py <input.csv> [--output-dir <dir>] [--algorithms verisense adept oak oxford]
"""

import argparse
import glob
import os
import subprocess
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Locate original algorithm source directories (no modifications to originals)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_algo_dir(start: str) -> str:
    """Locate the 'algorithms_original_code' folder.

    Searches this file's own directory first, then walks up the parent
    directories. This lets pipeline.py live in a subfolder (e.g.
    pipeline_for_original/) while the shared algorithms_original_code/ sits
    higher up in the repo. An explicit path in the ALGORITHMS_ORIGINAL_CODE
    environment variable overrides the search.
    """
    env = os.environ.get("ALGORITHMS_ORIGINAL_CODE")
    if env and os.path.isdir(env):
        return env
    d = start
    while True:
        cand = os.path.join(d, "algorithms_original_code")
        if os.path.isdir(cand):
            return cand
        parent = os.path.dirname(d)
        if parent == d:            # reached the filesystem root
            return os.path.join(start, "algorithms_original_code")
        d = parent


_ALGO_DIR    = _find_algo_dir(_HERE)
_ADEPT_DIR   = os.path.join(_ALGO_DIR, "ADEPT")
_OAK_DIR     = os.path.join(_ALGO_DIR, "oak")

# pyadept package root: ADEPT/pyadept-main/pyadept-main/ contains pyadept/
# This makes  "from pyadept.sliding_functions import ..."  work inside the
# algorithms_original_code/ADEPT/ files, which were written as pyadept internals.
_PYADEPT_PKG = os.path.join(_HERE, "ADEPT", "pyadept-main", "pyadept-main")

# Prepend so that project-local versions take priority over any installed packages
for _p in (_ADEPT_DIR, _PYADEPT_PKG, _OAK_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ===========================================================================
# SHARED UTILITIES
# ===========================================================================

def read_actigraph(filepath: str, default_hz: int = 80) -> tuple:
    """Parse an ActiGraph GT3X+ CSV file.

    Reads the header lines to extract start date/time and sample rate,
    then reads the raw x, y, z accelerometer data.

    Parameters
    ----------
    filepath   : path to the ActiGraph CSV file
    default_hz : fallback sample rate if header line not found

    Returns
    -------
    data     : pd.DataFrame with columns x, y, z  (float)
    start_dt : datetime  (recording start timestamp)
    hz       : int       (actual sample rate from header)
    """
    with open(filepath, "r", encoding="utf-8-sig", errors="replace") as f:
        lines = f.readlines()

    start_date = start_time_str = ""
    hz = default_hz
    header_row = None

    for i, line in enumerate(lines):
        s = line.strip().lstrip("\ufeff")
        if s.startswith("Start Time"):
            start_time_str = s.split()[-1].strip().strip('"')
        elif s.startswith("Start Date"):
            start_date = s.split()[-1].strip().strip('"')
        elif "Hz" in s and "at" in s:
            try:
                parts  = s.split()
                hz_idx = next(j for j, p in enumerate(parts) if p == "Hz")
                hz     = int(parts[hz_idx - 1])
            except Exception:
                pass
        elif s.startswith("Accelerometer X"):
            header_row = i
            break

    if header_row is None:
        raise ValueError(f"Cannot find 'Accelerometer X' header in {filepath}")

    # Robustly parse M/d/yyyy (ActiGraph omits leading zeros)
    dt_str   = f"{start_date} {start_time_str}".strip()
    start_dt = None
    for fmt in ("%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
        try:
            start_dt = datetime.strptime(dt_str, fmt)
            break
        except ValueError:
            pass
    if start_dt is None:
        try:
            d_part, t_part = dt_str.split(" ", 1)
            m, day, y = d_part.split("/")
            start_dt = datetime.strptime(
                f"{int(m):02d}/{int(day):02d}/{y} {t_part}", "%m/%d/%Y %H:%M:%S"
            )
        except Exception as exc:
            raise ValueError(f"Cannot parse datetime '{dt_str}': {exc}")

    data = pd.read_csv(
        filepath, skiprows=header_row, header=0,
        usecols=[0, 1, 2], names=["x", "y", "z"],
        encoding="utf-8-sig", on_bad_lines="skip"
    )
    data = data.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    return data, start_dt, hz


def build_output(steps_per_sec: np.ndarray, start_dt: datetime,
                 interval: int = 10) -> pd.DataFrame:
    """Aggregate per-second step counts into fixed-length time bins.

    Per-bin total is rounded (not truncated) before cast to int. For
    integer-count algorithms (Verisense, ADEPT) this is a no-op. For OAK,
    which produces fractional cadence (steps/s) per second, rounding the
    *bin sum* matches the paper: "The total number of steps in a signal
    is calculated as a rounded sum of all 1-second counts in that signal"
    (Straczkiewicz 2023, Fig 1E).

    Parameters
    ----------
    steps_per_sec : 1-D array, one entry per second of recording
    start_dt      : recording start timestamp
    interval      : bin width in seconds (default 10)

    Returns
    -------
    pd.DataFrame with columns  Time (str), Steps (float)
    """
    n    = len(steps_per_sec)
    rows = []
    for i in range(int(np.ceil(n / interval))):
        s  = i * interval
        e  = min(s + interval, n)
        t0 = start_dt + timedelta(seconds=s)
        rows.append({
            "Time":  t0.strftime("%Y-%m-%d %H:%M:%S"),
            "Steps": float(int(round(float(np.nansum(steps_per_sec[s:e])))))
        })
    return pd.DataFrame(rows)


# ===========================================================================
# ALGORITHM 1 – VERISENSE
# ===========================================================================
# Original code: algorithms_original_code/verisense/verisense_count_steps.R
#
# The R function verisense_count_steps() is called via myscript.R with
# parameters c(3, 5, 15, -0.5, 3, 4, 0.001, 1.2) at fs=15 Hz.
#
# This section reimplements the identical algorithm in pure Python so the
# pipeline runs without an R dependency.  The logic mirrors the R source
# line-by-line:
#   1. Compute RMS acceleration
#   2. Find local maxima in each k-sample window
#   3. Filter by magnitude threshold
#   4. Filter by periodicity (inter-peak distance)
#   5. Filter by similarity (smoothness of peak magnitudes)
#   6. Filter by continuity (variance within inter-peak intervals)
#   7. Bin surviving step locations into per-second counts
# ===========================================================================

def _verisense_count_steps(xyz: np.ndarray, fs: int = 15) -> np.ndarray:
    """Python reimplementation of verisense_count_steps.R.

    Parameters match myscript.R: coeffs = c(3, 5, 15, -0.5, 3, 4, 0.001, 1.2)

    Parameters
    ----------
    xyz : (N, 3) array of x, y, z acceleration in g at ``fs`` Hz
    fs  : sample rate (must be 15 Hz to match original algorithm)

    Returns
    -------
    steps_per_sec : 1-D int array, one count per second
    """
    # Parameters from myscript.R
    k             = 3       # peak window length (samples)
    period_min    = 5       # minimum inter-peak distance (samples)
    period_max    = 15      # maximum inter-peak distance (samples)
    sim_thres     = -0.5    # similarity threshold
    cont_win_size = 3       # continuity window size
    cont_thres    = 4       # continuity threshold (ddof=1 in R)
    var_thres     = 0.001   # variance threshold
    mag_thres     = 1.2     # magnitude threshold (g)

    acc   = np.sqrt(np.sum(xyz ** 2, axis=1))
    n_sec = int(np.round(len(acc) / fs))

    # Early exit: stationary signal
    if np.std(acc, ddof=1) < 0.025:
        return np.zeros(n_sec, dtype=float)

    # Step 1: find local maximum in each k-sample segment
    half_k = int(round(k / 2))
    n      = len(acc)
    n_segs = int(np.floor(n / k))

    locs, mags = [], []
    for i in range(n_segs):
        s  = i * k
        e  = s + k
        lm = int(np.argmax(acc[s:e]))
        b  = s + lm
        # Accept only if this sample is the max in the wider ±half_k window
        sc = max(0, b - half_k)
        ec = min(n, b + half_k + 1)
        if int(np.argmax(acc[sc:ec])) == (b - sc):
            locs.append(b)
            mags.append(float(np.max(acc[s:e])))

    if len(locs) < 2:
        return np.zeros(n_sec, dtype=float)

    # peak_info: columns [location, magnitude, periodicity, similarity, continuity]
    pi      = np.full((len(locs), 5), np.nan)
    pi[:, 0] = locs
    pi[:, 1] = mags

    # Step 2: filter by magnitude threshold
    pi = pi[pi[:, 1] > mag_thres]
    if len(pi) < 3:
        return np.zeros(n_sec, dtype=float)

    # Step 3: periodicity filter (inter-peak distance in samples)
    pi[:-1, 2] = np.diff(pi[:, 0])
    pi = pi[(pi[:, 2] > period_min) & (pi[:, 2] < period_max)]
    if len(pi) < 3:
        return np.zeros(n_sec, dtype=float)

    # Step 4: similarity filter
    pi[:-2, 3] = -np.abs(pi[2:, 1] - pi[:-2, 1])
    pi = pi[~np.isnan(pi[:, 3]) & (pi[:, 3] > sim_thres)]
    if len(pi) == 0:
        return np.zeros(n_sec, dtype=float)

    # Step 5: continuity filter (variance in inter-peak windows)
    if len(pi) < cont_thres + 1:
        return np.zeros(n_sec, dtype=float)

    cont = np.zeros(len(pi))
    for i in range(cont_thres - 1, len(pi) - 1):
        v = 0
        for x in range(1, cont_thres + 1):
            i0  = int(pi[i - x,     0])
            i1  = min(int(pi[i - x + 1, 0]) + 1, n)
            seg = acc[i0:i1]
            if len(seg) > 1 and np.var(seg, ddof=1) > var_thres:
                v += 1
        cont[i] = 1 if v >= cont_win_size else 0

    step_locs = pi[cont == 1, 0]
    step_locs = step_locs[~np.isnan(step_locs)]

    if len(step_locs) == 0:
        return np.zeros(n_sec, dtype=float)

    # Step 6: bin step locations into per-second counts
    idx    = np.clip((np.array(step_locs) // fs).astype(int), 0, n_sec - 1)
    counts = np.zeros(n_sec, dtype=float)
    for i in idx:
        counts[i] += 1
    return counts


def run_verisense(data: pd.DataFrame, start_dt: datetime,
                  hz: int, interval: int = 10) -> pd.DataFrame:
    """Resample ActiGraph data to 15 Hz and run the Verisense algorithm.

    The original algorithm (verisense_count_steps.R) requires exactly 15 Hz
    input.  Data recorded at ``hz`` Hz is resampled to 15 Hz using pandas
    time-based mean interpolation before being passed to the algorithm.
    """
    # Attach a DatetimeIndex at the original sample rate
    step_td   = timedelta(seconds=1.0 / hz)
    ts        = [start_dt + i * step_td for i in range(len(data))]
    df        = data.copy()
    df.index  = pd.DatetimeIndex(ts)

    # Resample to 15 Hz
    target_us = int(round(1_000_000 / 15))
    resampled = df.resample(f"{target_us}us").mean().interpolate(method="time")
    xyz       = resampled[["x", "y", "z"]].values

    steps_per_sec = _verisense_count_steps(xyz, fs=15)
    return build_output(steps_per_sec, start_dt, interval)


# ===========================================================================
# ALGORITHM 2 – ADEPT  (R implementation via Rscript subprocess)
# ===========================================================================
# Original R files: ADEPT/segmentWalking.R  (+ segmentPattern.R, etc.)
#
# adept_runner.R (project root) sources those R files, loads templates from
# the adeptdata R package (or falls back to cosine templates), calls
# segmentWalking(), and writes per-second step counts to a CSV that
# run_adept() reads back.
#
# Requirements: R must be on PATH (or RSCRIPT_PATH env var set).
#               R packages: pracma  (required)
#                           adeptdata, dvmisc, dplyr (recommended)
# ===========================================================================

def _find_rscript() -> str:
    """Return the path to Rscript, searching common Windows install locations."""
    import shutil
    # 1. Explicit override via environment variable
    env_path = os.environ.get("RSCRIPT_PATH", "")
    if env_path and os.path.isfile(env_path):
        return env_path
    # 2. Already on PATH
    if shutil.which("Rscript"):
        return "Rscript"
    # 3. Scan C:\Program Files\R\ for any installed version
    pf = os.environ.get("PROGRAMFILES", r"C:\Program Files")
    r_root = os.path.join(pf, "R")
    if os.path.isdir(r_root):
        # Sort so newest version (highest name) comes last → pick last
        candidates = sorted(
            [os.path.join(r_root, d, "bin", "Rscript.exe")
             for d in os.listdir(r_root)]
        )
        for c in reversed(candidates):
            if os.path.isfile(c):
                return c
    # 4. RStudio ships its own R on some installs
    rstudio_r = r"C:\Program Files\RStudio\resources\app\bin\quarto\bin\tools\rscript.exe"
    if os.path.isfile(rstudio_r):
        return rstudio_r
    return "Rscript"   # fall back; will fail with FileNotFoundError


def run_adept(data: pd.DataFrame, start_dt: datetime,
              hz: int, interval: int = 10,
              input_name: str = "xyz.csv") -> pd.DataFrame:
    """Run ADEPT stride segmentation via the R implementation.

    Writes x,y,z to a temp CSV, calls Rscript adept_runner.R, reads back
    the per-second step counts.
    """
    import tempfile

    n_sec = int(np.ceil(len(data) / hz))

    # Locate Rscript executable (auto-detect if not on PATH)
    rscript = _find_rscript()

    # Locate adept_runner.R (same directory as pipeline.py)
    runner_path = os.path.join(_HERE, "adept_runner.R")
    if not os.path.isfile(runner_path):
        print(f"  [ADEPT] adept_runner.R not found at {runner_path}")
        return build_output(np.zeros(n_sec), start_dt, interval)

    with tempfile.TemporaryDirectory(prefix="adept_") as tmp:
        in_csv  = os.path.join(tmp, input_name)
        out_csv = os.path.join(tmp, "steps.csv")

        # Write x,y,z data
        data[["x", "y", "z"]].to_csv(in_csv, index=False)

        cmd = [rscript, "--vanilla", runner_path, in_csv, out_csv, str(hz)]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd=_HERE
            )
        except FileNotFoundError:
            pf = os.environ.get("PROGRAMFILES", r"C:\Program Files")
            print(
                f"  [ADEPT] Rscript not found (tried: {rscript}).\n"
                "  Fix options:\n"
                f"    set RSCRIPT_PATH={pf}\\R\\R-4.x.x\\bin\\Rscript.exe\n"
                "  or pass --rscript to the pipeline, e.g.:\n"
                f"    --rscript \"{pf}\\R\\R-4.x.x\\bin\\Rscript.exe\""
            )
            return build_output(np.zeros(n_sec), start_dt, interval)

        # Print R messages/warnings so user can diagnose package issues
        if result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                print(f"  [ADEPT/R] {line}")
        if result.stderr.strip():
            for line in result.stderr.strip().splitlines():
                print(f"  [ADEPT/R] {line}")

        if result.returncode != 0:
            print(f"  [ADEPT] Rscript exited with code {result.returncode}")
            return build_output(np.zeros(n_sec), start_dt, interval)

        if not os.path.isfile(out_csv):
            print("  [ADEPT] R script did not produce output CSV.")
            return build_output(np.zeros(n_sec), start_dt, interval)

        steps_df = pd.read_csv(out_csv)

    # steps_df has columns: second (0-based offset), steps
    steps_per_sec = np.zeros(n_sec, dtype=float)
    for _, row in steps_df.iterrows():
        idx = int(row["second"])
        if 0 <= idx < n_sec:
            steps_per_sec[idx] = float(row["steps"])

    return build_output(steps_per_sec, start_dt, interval)


# ===========================================================================
# ALGORITHM 3 – OAK (Open Access sigKINetics)
# ===========================================================================
# Original code: algorithms_original_code/oak/base.py
#
# The original base.py exposes two functions that this pipeline calls:
#   preprocess_bout(t, x, y, z, fs=10) -> (t_interp, vm)
#   find_walking(vm, fs, ...)           -> cadence array (steps/s per second)
#
# Input adaptation:
#   ActiGraph records at 80 Hz.  OAK expects 10 Hz.  Raw triaxial data is
#   passed *directly* to preprocess_bout with proper Unix timestamps; that
#   function performs linear interpolation to 10 Hz internally, matching
#   the paper's Methods section ("Data Preprocessing"):
#     "we used linear interpolation to impose a uniform sampling frequency
#      of 10 Hz across triaxial accelerometer data".
#   Pre-downsampling by block-averaging is intentionally avoided so the
#   pipeline reproduces the paper's preprocessing exactly.
# ===========================================================================

def _load_oak_base():
    """Load algorithms_original_code/oak/base.py, stubbing out the 'forest'
    package if it is not installed (forest is only used by the study-level
    batch function, not by preprocess_bout / find_walking).
    """
    import importlib
    import importlib.util
    import types

    # Inject lightweight stub modules so 'from forest.x import y' succeeds
    # even when beiwe-forest is not installed in this environment.
    if "forest" not in sys.modules:
        # Minimal Frequency enum matching what base.py references
        import enum
        class _Frequency(enum.Enum):
            HOURLY_AND_DAILY = "HOURLY_AND_DAILY"
            HOURLY           = "HOURLY"
            DAILY            = "DAILY"
            MINUTE           = "MINUTE"

        forest_pkg          = types.ModuleType("forest")
        forest_constants    = types.ModuleType("forest.constants")
        forest_utils        = types.ModuleType("forest.utils")
        forest_constants.Frequency = _Frequency
        forest_utils.get_ids       = lambda *a, **kw: []
        forest_pkg.constants       = forest_constants
        forest_pkg.utils           = forest_utils

        sys.modules["forest"]          = forest_pkg
        sys.modules["forest.constants"] = forest_constants
        sys.modules["forest.utils"]    = forest_utils

    spec = importlib.util.spec_from_file_location(
        "oak_base", os.path.join(_OAK_DIR, "base.py")
    )
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError as exc:
        pkg = str(exc).split("'")[1] if "'" in str(exc) else str(exc)
        raise ImportError(
            f"OAK requires '{pkg}' in the active Python environment.\n"
            f"  Active Python: {sys.executable}\n"
            f"  Install:       pip install {pkg}"
        )
    return mod


def run_oak(data: pd.DataFrame, start_dt: datetime,
            hz: int, interval: int = 10) -> pd.DataFrame:
    """Run OAK CWT-based step counting using the original base.py.

    Imports preprocess_bout and find_walking from
    algorithms_original_code/oak/base.py unchanged.
    The 'forest' package is stubbed out if not installed — it is only
    used by the study-level batch function, not the functions called here.
    """
    _base_mod       = _load_oak_base()
    preprocess_bout = _base_mod.preprocess_bout
    find_walking    = _base_mod.find_walking

    # Pass raw signal (at source Hz) directly to preprocess_bout, which
    # linearly interpolates to 10 Hz. This matches the paper's preprocessing.
    target_fs = 10
    n         = len(data)
    t_raw     = start_dt.timestamp() + np.arange(n) / float(hz)
    t_i, vm   = preprocess_bout(
        t_raw,
        data["x"].values.astype("float64"),
        data["y"].values.astype("float64"),
        data["z"].values.astype("float64"),
        fs=target_fs,
    )

    n_sec = int(np.ceil(len(data) / hz))
    if len(t_i) == 0:
        return build_output(np.zeros(n_sec), start_dt, interval)

    # find_walking returns cadence in Hz (steps/s) for each 1-second window
    cad = find_walking(
        vm, fs=target_fs,
        min_amp=0.3,
        step_freq=(1.4, 2.3),
        alpha=0.6,
        beta=2.5,
        min_t=3,
        delta=20,
    )

    # Keep cadence as float (steps/s per 1-second window). build_output
    # rounds the per-bin SUM, matching the paper. Pre-rounding each second
    # would inflate counts (e.g. cadence 1.78 -> 2 instead of contributing
    # 1.78 toward the bin total).
    steps_per_sec = np.asarray(cad, dtype=float)
    return build_output(steps_per_sec, start_dt, interval)


# ===========================================================================
# ALGORITHM 4 – OXFORD (stepcount)
# ===========================================================================
# Original code: algorithms_original_code/stepcount/
#
# The Oxford stepcount package is called through its CLI entry point
# ("stepcount") rather than imported directly.  This avoids interfering
# with any installed package version and keeps the calling convention
# identical to what the authors document.
#
# Input adaptation:
#   The CLI expects a CSV with at minimum a 'time' column and three
#   acceleration columns.  _prepare_oxford_input() writes such a file from
#   the ActiGraph data, inserting a datetime timestamp column computed from
#   start_dt and the source sample rate.
# ===========================================================================

def _prepare_oxford_input(data: pd.DataFrame, start_dt: datetime,
                          hz: int, out_path: str) -> None:
    """Write a time-stamped CSV for the stepcount CLI.

    The CLI requires: time (datetime string), x, y, z
    Timestamps are constructed at the original sample rate.
    """
    step_td    = timedelta(seconds=1.0 / hz)
    timestamps = [start_dt + i * step_td for i in range(len(data))]
    df         = data.copy()
    df.insert(0, "time", timestamps)
    df.to_csv(out_path, index=False)


def run_oxford(data: pd.DataFrame, start_dt: datetime,
               hz: int, interval: int = 10,
               work_dir: str = None,
               oxford_exe: str = None) -> pd.DataFrame:
    """Run Oxford stepcount CLI on the data.

    Parameters
    ----------
    oxford_exe : full path to the stepcount executable, e.g.
                 "C:/Users/you/miniconda3/envs/stepcount/Scripts/stepcount.exe"
                 If None, 'stepcount' must be on PATH.
    """
    n_sec = int(np.ceil(len(data) / hz))

    if work_dir is None:
        import tempfile
        work_dir = tempfile.mkdtemp(prefix="oxford_stepcount_")
    # Use absolute paths so stepcount can find files regardless of cwd
    work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    csv_path = os.path.join(work_dir, "input.csv")
    _prepare_oxford_input(data, start_dt, hz, csv_path)

    exe = oxford_exe if oxford_exe else "stepcount"
    cmd = [
        exe, csv_path,
        "--outdir", work_dir,
        "--txyz", "time,x,y,z",
        "--quiet",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)
    except FileNotFoundError:
        print(
            f"  [Oxford] '{exe}' not found.\n"
            "  Option 1: activate your stepcount conda env before running:\n"
            "            conda activate stepcount\n"
            "  Option 2: pass the full exe path with --oxford-exe, e.g.:\n"
            "            --oxford-exe \"C:/Users/you/miniconda3/envs/stepcount/Scripts/stepcount.exe\""
        )
        return build_output(np.zeros(n_sec), start_dt, interval)

    # Locate *-Steps.csv.gz (default) or *-Steps.csv
    matches = glob.glob(os.path.join(work_dir, "**", "*-Steps.csv.gz"),
                        recursive=True)
    if not matches:
        matches = glob.glob(os.path.join(work_dir, "**", "*-Steps.csv"),
                            recursive=True)

    if not matches:
        err_tail = (result.stderr or "")[-400:]
        print(f"  [Oxford] stepcount output not found.\n"
              f"  Return code: {result.returncode}\n"
              f"  stderr tail: {err_tail}")
        return build_output(np.zeros(n_sec), start_dt, interval)

    sc_df = pd.read_csv(matches[0], parse_dates=["time"])
    sc_df = sc_df.sort_values("time").reset_index(drop=True)
    sc_df["Time"]  = sc_df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    sc_df["Steps"] = pd.to_numeric(sc_df["Steps"], errors="coerce").fillna(0.0)
    sc_df["Steps"] = sc_df["Steps"].apply(lambda v: float(int(round(v))))
    return sc_df[["Time", "Steps"]]


# ===========================================================================
# PIPELINE ORCHESTRATOR
# ===========================================================================

def run_pipeline(input_file: str, output_dir: str,
                 algorithms: list = None,
                 interval: int = 10,
                 oxford_exe: str = None) -> None:
    """Run all (or selected) step counting algorithms on one ActiGraph file.

    Parameters
    ----------
    input_file : path to ActiGraph GT3X+ CSV
    output_dir : directory where per-algorithm output CSVs are saved
    algorithms : algorithms to run; default ["verisense", "adept", "oak", "oxford"]
    interval   : output bin width in seconds (default 10)
    oxford_exe : full path to stepcount executable (for conda envs not on PATH)

    Output files (one per algorithm):
      <output_dir>/<stem>_verisense_steps.csv
      <output_dir>/<stem>_adept_steps.csv
      <output_dir>/<stem>_oak_steps.csv
      <output_dir>/<stem>_oxford_steps.csv
    """
    if algorithms is None:
        algorithms = ["verisense", "adept", "oak", "oxford"]

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nReading: {input_file}")
    data, start_dt, hz = read_actigraph(input_file)
    print(f"  Start     : {start_dt}")
    print(f"  Sample Hz : {hz}")
    print(f"  Samples   : {len(data):,}  ({len(data)/hz/3600:.2f} h)")

    stem = os.path.splitext(os.path.basename(input_file))[0]

    for algo in algorithms:
        print(f"\n[{algo.upper()}] running...", flush=True)
        try:
            if algo == "verisense":
                out_df = run_verisense(data, start_dt, hz, interval)

            elif algo == "adept":
                out_df = run_adept(data, start_dt, hz, interval,
                                   input_name=os.path.basename(input_file))

            elif algo == "oak":
                out_df = run_oak(data, start_dt, hz, interval)

            elif algo == "oxford":
                oxford_work = os.path.join(output_dir, f"{stem}_oxford_work")
                out_df = run_oxford(data, start_dt, hz, interval,
                                    oxford_work, oxford_exe)

            else:
                print(f"  Unknown algorithm '{algo}' – skipping.")
                continue

            total    = int(out_df["Steps"].sum())
            out_path = os.path.join(output_dir, f"{stem}_{algo}_steps.csv")
            out_df.to_csv(out_path, index=False)
            print(f"  Total steps : {total:,}")
            print(f"  Output      : {out_path}")

        except Exception as exc:
            import traceback
            print(f"  ERROR in [{algo}]: {exc}")
            traceback.print_exc()

    print("\nDone.")


# ===========================================================================
# CLI ENTRY POINT
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Four-algorithm step count pipeline for ActiGraph GT3X+ CSV files.\n"
            "Runs Verisense, ADEPT, OAK, and Oxford (stepcount) and writes\n"
            "one Time,Steps CSV per algorithm."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        help="Path to the ActiGraph GT3X+ CSV file"
    )
    parser.add_argument(
        "output_dir_pos",
        nargs="?",
        default=None,
        help="Output directory (positional shortcut, same as --output-dir)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="stepcount_results",
        help="Directory for output files (default: stepcount_results/)"
    )
    parser.add_argument(
        "--algorithms", "-a",
        nargs="+",
        choices=["verisense", "adept", "oak", "oxford"],
        default=["verisense", "adept", "oak", "oxford"],
        help="Algorithms to run (default: all four)"
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=10,
        help="Output bin size in seconds (default: 10)"
    )
    parser.add_argument(
        "--oxford-exe",
        default=None,
        metavar="PATH",
        help=(
            "Full path to the stepcount executable when it is not on PATH "
            "(e.g. inside a conda environment). "
            "Example: --oxford-exe \"C:/Users/you/miniconda3/envs/stepcount/Scripts/stepcount.exe\""
        )
    )
    parser.add_argument(
        "--rscript",
        default=None,
        metavar="PATH",
        help=(
            "Full path to Rscript when R is not on PATH. "
            "Example: --rscript \"C:/Program Files/R/R-4.4.1/bin/Rscript.exe\""
        )
    )
    args = parser.parse_args()
    # Positional output_dir overrides --output-dir if both provided
    out_dir = args.output_dir_pos if args.output_dir_pos else args.output_dir

    # Allow overriding Rscript path via CLI argument
    if args.rscript:
        os.environ["RSCRIPT_PATH"] = args.rscript

    run_pipeline(
        input_file = args.input,
        output_dir = out_dir,
        algorithms = args.algorithms,
        interval   = args.interval,
        oxford_exe = args.oxford_exe,
    )


if __name__ == "__main__":
    main()