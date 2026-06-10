"""
pipeline_cluster_v2.py — Merged four-algorithm cluster pipeline
================================================================
Combines pipeline_cluster.py (Verisense/OAK/Oxford/ADEPT) with the
location-aware ADEPT logic from pipeline_adept_freeliving.py.

Per-algorithm sensor location filtering (per user spec, 2026-05):

    adept     -> any sensor location (wrist / waist / ankle / thigh / phone)
    oak       -> thigh, waist  (paper validation locations)
    oxford    -> wrist
    verisense -> wrist

For each input CSV the pipeline:
  1. Auto-detects sensor location from filename
       wrist  : "wrist"
       waist  : "waist" / "hip"
       ankle  : "ankle"
       thigh  : "thigh"
       phone  : "phone" / "pocket" / "backpack"
  2. Skips any algorithm whose allowed-location set doesn't include the
     detected location (prints "[<algo>] skipped: location=<loc>").
  3. For ADEPT, passes the location to adept_runner_cluster_v2.R so it
     picks the right per-body-location template + amplitude filter preset.

Output (one CSV per algorithm per input):
  <output_dir>/<stem>_verisense_steps.csv
  <output_dir>/<stem>_adept_steps.csv
  <output_dir>/<stem>_oak_steps.csv
  <output_dir>/<stem>_oxford_steps.csv

Usage examples:
  # Single CSV
  python pipeline_cluster_v2.py file.csv /scratch/out/

  # Directory
  python pipeline_cluster_v2.py /scratch/PAAWS/DS_87/accel/ /scratch/out/DS_87/

  # Only ADEPT and OAK
  python pipeline_cluster_v2.py file.csv /scratch/out/ --algorithms adept oak

  # Force a location (overrides filename auto-detect — useful for
  # ambiguous filenames or one-off testing)
  python pipeline_cluster_v2.py file.csv /scratch/out/ --location wrist
"""

import argparse
import glob as glob_module
import os
import re
import shutil
import subprocess
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CLUSTER PATHS — edit if your layout changes
# ---------------------------------------------------------------------------
_HERE     = os.path.dirname(os.path.abspath(__file__))
# Absolute path to the original algorithm code on the cluster. Override with
# the ALGO_ORIGINAL_CODE_DIR environment variable without editing this file.
_ALGO_DIR = os.environ.get(
    "ALGO_ORIGINAL_CODE_DIR", "/home/wang.yichen8/4_algo_original_code")
_OAK_DIR  = os.path.join(_ALGO_DIR, "oak")

_ROCKER_IMAGE    = os.environ.get(
    "ADEPT_ROCKER_IMAGE",
    "/shared/container_repository/explorer/rstudio/4.4.2/rocker-geospatial-4.4.2.sif",
)
_APPTAINER_BINDS = os.environ.get(
    "ADEPT_APPTAINER_BINDS",
    "/scratch:/scratch,/home:/home,/projects:/projects,/tmp:/tmp",
)

_ADEPT_RUNNER_R     = os.path.join(_HERE, "adept_runner_cluster_v2.R")
_VERISENSE_RUNNER_R = os.path.join(_HERE, "verisense_runner_cluster.R")

if _OAK_DIR not in sys.path:
    sys.path.insert(0, _OAK_DIR)


# ---------------------------------------------------------------------------
# Per-algorithm allowed sensor locations.
# None = run on any location. Set membership = only run if file matches.
# ---------------------------------------------------------------------------
_ALGO_LOC_ALLOWED = {
    "adept":     None,                # all sensor locations
    "oak":       None,                # all locations (cross-body in the paper, incl. phone)
    "oxford":    {"wrist", "waist"},  # wrist + hip  (a 'hip' file is detected as 'waist')
    "verisense": {"wrist"},           # wrist only
}


# ===========================================================================
# ActiGraph CSV parsing (shared)
# ===========================================================================

def read_actigraph(filepath: str, default_hz: int = 80):
    with open(filepath, "r", encoding="utf-8-sig", errors="replace") as f:
        lines = f.readlines()

    start_date = start_time_str = ""
    hz = default_hz
    header_row = None
    for i, line in enumerate(lines[:50]):
        s = line.strip().lstrip("﻿")
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

    dt_str = f"{start_date} {start_time_str}".strip()
    start_dt = None
    for fmt in ("%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
        try:
            start_dt = datetime.strptime(dt_str, fmt)
            break
        except ValueError:
            pass
    if start_dt is None:
        d_part, t_part = dt_str.split(" ", 1)
        m, day, y = d_part.split("/")
        start_dt = datetime.strptime(
            f"{int(m):02d}/{int(day):02d}/{y} {t_part}", "%m/%d/%Y %H:%M:%S"
        )

    data = pd.read_csv(
        filepath, skiprows=header_row, header=0,
        usecols=[0, 1, 2], names=["x", "y", "z"],
        encoding="utf-8-sig", on_bad_lines="skip"
    )
    data = data.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    return data, start_dt, hz


def build_output(steps_per_sec: np.ndarray, start_dt: datetime,
                 interval: int = 5) -> pd.DataFrame:
    n = len(steps_per_sec)
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
# Sensor location auto-detect
# ===========================================================================

_LOC_PATTERNS = [
    ("wrist", re.compile(r"wrist",                 re.IGNORECASE)),
    ("ankle", re.compile(r"ankle",                 re.IGNORECASE)),
    ("thigh", re.compile(r"thigh",                 re.IGNORECASE)),
    ("waist", re.compile(r"waist|hip",             re.IGNORECASE)),
    ("phone", re.compile(r"phone|pocket|backpack", re.IGNORECASE)),
]


def infer_location(path: str, default: str = "waist") -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    flat = stem.replace("-", "").replace("_", "")
    for name, pat in _LOC_PATTERNS:
        if pat.search(flat):
            return name
    return default


# ===========================================================================
# ALGORITHM 1 — VERISENSE (R via Apptainer)
# ===========================================================================

def run_verisense(data: pd.DataFrame, start_dt: datetime,
                  hz: int, interval: int = 5) -> pd.DataFrame:
    import tempfile
    n_sec = int(np.ceil(len(data) / hz))

    if not os.path.isfile(_VERISENSE_RUNNER_R):
        print(f"  [Verisense] runner not found: {_VERISENSE_RUNNER_R}")
        return build_output(np.zeros(n_sec), start_dt, interval)
    if not os.path.isfile(_ROCKER_IMAGE):
        print(f"  [Verisense] image not found: {_ROCKER_IMAGE}")
        return build_output(np.zeros(n_sec), start_dt, interval)

    with tempfile.TemporaryDirectory(prefix="verisense_") as tmp:
        in_csv  = os.path.join(tmp, "xyz.csv")
        out_csv = os.path.join(tmp, "steps.csv")
        data[["x", "y", "z"]].to_csv(in_csv, index=False)

        container_exe = "apptainer" if shutil.which("apptainer") else (
                        "singularity" if shutil.which("singularity") else None)
        if container_exe is None:
            print("  [Verisense] no apptainer/singularity on PATH")
            return build_output(np.zeros(n_sec), start_dt, interval)

        cmd = [container_exe, "exec", "-B", _APPTAINER_BINDS, _ROCKER_IMAGE,
               "Rscript", "--vanilla",
               _VERISENSE_RUNNER_R, in_csv, out_csv, str(hz)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            print(f"  [Verisense] '{container_exe}' missing — module load singularity?")
            return build_output(np.zeros(n_sec), start_dt, interval)

        for line in (result.stdout or "").strip().splitlines():
            print(f"  [Verisense/R] {line}")
        for line in (result.stderr or "").strip().splitlines():
            print(f"  [Verisense/R] {line}")

        if result.returncode != 0 or not os.path.isfile(out_csv):
            print(f"  [Verisense] Rscript exit={result.returncode} — zeros")
            return build_output(np.zeros(n_sec), start_dt, interval)

        steps_df = pd.read_csv(out_csv)

    steps_per_sec = np.zeros(n_sec, dtype=float)
    for _, row in steps_df.iterrows():
        idx = int(row["second"])
        if 0 <= idx < n_sec:
            steps_per_sec[idx] = float(row["steps"])
    return build_output(steps_per_sec, start_dt, interval)


# ===========================================================================
# ALGORITHM 2 — ADEPT (location-aware, R via Apptainer)
# ===========================================================================

def run_adept(data: pd.DataFrame, start_dt: datetime,
              hz: int, location: str,
              interval: int = 5, n_cores: int = 4,
              input_name: str = "xyz.csv") -> pd.DataFrame:
    import tempfile
    n_sec = int(np.ceil(len(data) / hz))

    if not os.path.isfile(_ADEPT_RUNNER_R):
        print(f"  [ADEPT] runner not found: {_ADEPT_RUNNER_R}")
        return build_output(np.zeros(n_sec), start_dt, interval)
    if not os.path.isfile(_ROCKER_IMAGE):
        print(f"  [ADEPT] image not found: {_ROCKER_IMAGE}")
        return build_output(np.zeros(n_sec), start_dt, interval)

    with tempfile.TemporaryDirectory(prefix="adept_") as tmp:
        in_csv  = os.path.join(tmp, input_name)
        out_csv = os.path.join(tmp, "steps.csv")
        data[["x", "y", "z"]].to_csv(in_csv, index=False)

        container_exe = "apptainer" if shutil.which("apptainer") else (
                        "singularity" if shutil.which("singularity") else None)
        if container_exe is None:
            print("  [ADEPT] no apptainer/singularity on PATH")
            return build_output(np.zeros(n_sec), start_dt, interval)

        cmd = [container_exe, "exec", "-B", _APPTAINER_BINDS, _ROCKER_IMAGE,
               "Rscript", "--vanilla",
               _ADEPT_RUNNER_R,
               in_csv, out_csv, str(hz), location, str(n_cores)]
        print(f"  [ADEPT] location={location}  n_cores={n_cores}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            print(f"  [ADEPT] '{container_exe}' missing — module load singularity?")
            return build_output(np.zeros(n_sec), start_dt, interval)

        for line in (result.stdout or "").strip().splitlines():
            print(f"  [ADEPT/R] {line}")
        for line in (result.stderr or "").strip().splitlines():
            print(f"  [ADEPT/R] {line}")

        if result.returncode != 0 or not os.path.isfile(out_csv):
            print(f"  [ADEPT] Rscript exit={result.returncode} — zeros")
            return build_output(np.zeros(n_sec), start_dt, interval)

        steps_df = pd.read_csv(out_csv)

    steps_per_sec = np.zeros(n_sec, dtype=float)
    for _, row in steps_df.iterrows():
        idx = int(row["second"])
        if 0 <= idx < n_sec:
            steps_per_sec[idx] = float(row["steps"])
    return build_output(steps_per_sec, start_dt, interval)


# ===========================================================================
# ALGORITHM 3 — OAK (Python)
# ===========================================================================

def _load_oak_base():
    import importlib.util
    import types
    import enum

    if "forest" not in sys.modules:
        class _Frequency(enum.Enum):
            HOURLY_AND_DAILY = "HOURLY_AND_DAILY"
            HOURLY = "HOURLY"; DAILY = "DAILY"; MINUTE = "MINUTE"
        forest_pkg       = types.ModuleType("forest")
        forest_constants = types.ModuleType("forest.constants")
        forest_utils     = types.ModuleType("forest.utils")
        forest_constants.Frequency = _Frequency
        forest_utils.get_ids       = lambda *a, **kw: []
        forest_pkg.constants = forest_constants
        forest_pkg.utils     = forest_utils
        sys.modules["forest"]           = forest_pkg
        sys.modules["forest.constants"] = forest_constants
        sys.modules["forest.utils"]     = forest_utils

    base_path = os.path.join(_OAK_DIR, "base.py")
    if not os.path.isfile(base_path):
        raise ImportError(f"OAK base.py not found at {base_path}")
    spec = importlib.util.spec_from_file_location("oak_base", base_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_oak(data: pd.DataFrame, start_dt: datetime,
            hz: int, interval: int = 5) -> pd.DataFrame:
    _base_mod       = _load_oak_base()
    preprocess_bout = _base_mod.preprocess_bout
    find_walking    = _base_mod.find_walking

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

    cad = find_walking(
        vm, fs=target_fs,
        min_amp=0.3, step_freq=(1.4, 2.3),
        alpha=0.6, beta=2.5, min_t=3, delta=20,
    )
    return build_output(np.asarray(cad, dtype=float), start_dt, interval)


# ===========================================================================
# ALGORITHM 4 — OXFORD (stepcount CLI)
# ===========================================================================

def _prepare_oxford_input(data: pd.DataFrame, start_dt: datetime,
                          hz: int, out_path: str) -> None:
    step_td    = timedelta(seconds=1.0 / hz)
    timestamps = [start_dt + i * step_td for i in range(len(data))]
    df         = data.copy()
    df.insert(0, "time", timestamps)
    df.to_csv(out_path, index=False)


def run_oxford(data: pd.DataFrame, start_dt: datetime,
               hz: int, interval: int = 5,
               work_dir: str = None,
               oxford_exe: str = None) -> pd.DataFrame:
    n_sec = int(np.ceil(len(data) / hz))
    if work_dir is None:
        import tempfile
        work_dir = tempfile.mkdtemp(prefix="oxford_stepcount_")
    work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    csv_path = os.path.join(work_dir, "input.csv")
    _prepare_oxford_input(data, start_dt, hz, csv_path)

    exe = oxford_exe if oxford_exe else "stepcount"
    cmd = [exe, csv_path, "--outdir", work_dir, "--txyz", "time,x,y,z", "--quiet"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)
    except FileNotFoundError:
        print(f"  [Oxford] '{exe}' not found — source activate stepcount?")
        return build_output(np.zeros(n_sec), start_dt, interval)

    matches = glob_module.glob(os.path.join(work_dir, "**", "*-Steps.csv.gz"), recursive=True)
    if not matches:
        matches = glob_module.glob(os.path.join(work_dir, "**", "*-Steps.csv"), recursive=True)
    if not matches:
        err_tail = (result.stderr or "")[-400:]
        print(f"  [Oxford] no output. rc={result.returncode}  stderr tail: {err_tail}")
        return build_output(np.zeros(n_sec), start_dt, interval)

    sc_df = pd.read_csv(matches[0], parse_dates=["time"])
    sc_df = sc_df.sort_values("time").reset_index(drop=True)
    sc_df["Time"]  = sc_df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    sc_df["Steps"] = pd.to_numeric(sc_df["Steps"], errors="coerce").fillna(0.0)
    sc_df["Steps"] = sc_df["Steps"].apply(lambda v: float(int(round(v))))
    return sc_df[["Time", "Steps"]]


# ===========================================================================
# ORCHESTRATOR
# ===========================================================================

def run_pipeline_on_file(input_file: str, output_dir: str,
                         algorithms: list,
                         interval: int = 5,
                         location_override: str = None,
                         n_cores: int = 4,
                         oxford_exe: str = None) -> None:
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nReading: {input_file}")
    data, start_dt, hz = read_actigraph(input_file)
    loc = location_override or infer_location(input_file)
    print(f"  start    : {start_dt}")
    print(f"  hz       : {hz}")
    print(f"  samples  : {len(data):,}  ({len(data)/hz/3600:.2f} h)")
    print(f"  location : {loc}  ({'forced' if location_override else 'auto'})")

    stem = os.path.splitext(os.path.basename(input_file))[0]

    for algo in algorithms:
        allowed = _ALGO_LOC_ALLOWED.get(algo)
        if allowed is not None and loc not in allowed:
            print(f"\n[{algo.upper()}] skipped: location={loc} not in {sorted(allowed)}")
            continue

        print(f"\n[{algo.upper()}] running on location={loc} ...", flush=True)
        try:
            if algo == "verisense":
                out_df = run_verisense(data, start_dt, hz, interval)
            elif algo == "adept":
                out_df = run_adept(data, start_dt, hz, loc, interval, n_cores,
                                   input_name=os.path.basename(input_file))
            elif algo == "oak":
                out_df = run_oak(data, start_dt, hz, interval)
            elif algo == "oxford":
                oxford_work = os.path.join(output_dir, f"{stem}_oxford_work")
                out_df = run_oxford(data, start_dt, hz, interval,
                                    oxford_work, oxford_exe)
            else:
                print(f"  unknown algorithm '{algo}' — skipping")
                continue

            total    = int(out_df["Steps"].sum())
            out_path = os.path.join(output_dir, f"{stem}_{algo}_steps.csv")
            out_df.to_csv(out_path, index=False)
            print(f"  total steps : {total:,}")
            print(f"  output      : {out_path}")
        except Exception as exc:
            import traceback
            print(f"  ERROR in [{algo}]: {exc}")
            traceback.print_exc()


def collect_input_files(input_arg: str) -> list:
    if os.path.isdir(input_arg):
        return sorted(glob_module.glob(os.path.join(input_arg, "*.csv")))
    if input_arg.lower().endswith(".txt"):
        files = []
        with open(input_arg, "r") as fh:
            for line in fh:
                p = line.strip().strip('"').strip("'")
                if p: files.append(p)
        return files
    return [input_arg]


# ===========================================================================
# CLI
# ===========================================================================

def main():
    p = argparse.ArgumentParser(
        description="Merged four-algorithm cluster pipeline with per-algorithm sensor filtering.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input",
                   help="CSV file, .txt list of CSV paths, or directory of CSVs")
    p.add_argument("output_dir", help="Output directory")
    p.add_argument("--algorithms", "-a",
                   nargs="+",
                   choices=["verisense", "adept", "oak", "oxford"],
                   default=["verisense", "adept", "oak", "oxford"],
                   help="Algorithms to attempt (default: all four)")
    p.add_argument("--location", "-l",
                   choices=["wrist", "waist", "ankle", "thigh", "phone"],
                   default=None,
                   help="Force sensor location (overrides filename auto-detect)")
    p.add_argument("--interval", "-i", type=int, default=5,
                   help="Output bin size in seconds (default 5)")
    p.add_argument("--n-cores", type=int, default=4,
                   help="Parallel cores for ADEPT (default 4)")
    p.add_argument("--oxford-exe", default=None,
                   help="Path to 'stepcount' executable if not on PATH")
    args = p.parse_args()

    files = collect_input_files(args.input)
    if not files:
        print("No input files. Exiting."); sys.exit(1)

    print(f"Input files  : {len(files)}")
    print(f"Output dir   : {args.output_dir}")
    print(f"Algorithms   : {', '.join(args.algorithms)}")
    print(f"Interval     : {args.interval} s   ADEPT cores: {args.n_cores}")
    _loc_str = ",  ".join(
        f"{a}={'any' if s is None else '/'.join(sorted(s))}"
        for a, s in _ALGO_LOC_ALLOWED.items())
    print(f"Allowed locs : {_loc_str}")

    for fp in files:
        if not os.path.isfile(fp):
            print(f"\nSKIPPING (not found): {fp}"); continue
        run_pipeline_on_file(
            input_file        = fp,
            output_dir        = args.output_dir,
            algorithms        = args.algorithms,
            interval          = args.interval,
            location_override = args.location,
            n_cores           = args.n_cores,
            oxford_exe        = args.oxford_exe,
        )
    print("\nAll done.")


if __name__ == "__main__":
    main()