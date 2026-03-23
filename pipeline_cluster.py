"""
Four-Algorithm Step Count Pipeline — SLURM CLUSTER VERSION
============================================================
Adapted for the wang.yichen8 cluster environment.

Key differences from pipeline.py (local version):
  - Hard-coded cluster paths for algorithm source files
  - ADEPT (R-based) runs via Apptainer with the Rocker container
  - Flexible input: single CSV, .txt list file, or directory of CSVs
  - Oxford uses the active conda environment (stepcount)

Algorithm paths on cluster:
  Oxford   : /home/wang.yichen8/4_algo_original_code/oxford/
  OAK      : /home/wang.yichen8/4_algo_original_code/oak/base.py
  ADEPT    : /home/wang.yichen8/4_algo_original_code/ADEPT/*.R  (via Apptainer)
  Verisense: /home/wang.yichen8/4_algo_original_code/verisense/*.R  (via Apptainer)

Usage:
  # Single CSV file
  python pipeline_cluster.py DS_138-Free-RightWaist.csv /scratch/.../results/

  # List file (one quoted CSV path per line, as produced by generate_lists.py)
  python pipeline_cluster.py list5.txt /scratch/.../results/

  # Directory of CSV files
  python pipeline_cluster.py /scratch/.../DS_138/accel/ /scratch/.../results/

  # With specific algorithms only
  python pipeline_cluster.py list5.txt /scratch/.../results/ --algorithms verisense oak

Output (one CSV per algorithm per input file):
  <output_dir>/<stem>_verisense_steps.csv
  <output_dir>/<stem>_adept_steps.csv
  <output_dir>/<stem>_oak_steps.csv
  <output_dir>/<stem>_oxford_steps.csv
"""

import argparse
import glob as glob_module
import os
import subprocess
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CLUSTER PATHS — edit if your directory layout changes
# ---------------------------------------------------------------------------
_HERE     = os.path.dirname(os.path.abspath(__file__))

# Algorithm source directories on cluster
_ALGO_DIR = "/home/wang.yichen8/4_algo_original_code"
_OAK_DIR  = os.path.join(_ALGO_DIR, "oak")

# Apptainer container used for R (ADEPT)
_ROCKER_IMAGE    = "/shared/container_repository/explorer/rstudio/4.4.2/rocker-geospatial-4.4.2.sif"
# Bind mounts: make /scratch, /home, /projects, /tmp accessible inside container
_APPTAINER_BINDS = "/scratch:/scratch,/home:/home,/projects:/projects,/tmp:/tmp"

# adept_runner_cluster.R and verisense_runner_cluster.R must live in the same directory as this script
_ADEPT_RUNNER_R      = os.path.join(_HERE, "adept_runner_cluster.R")
_VERISENSE_RUNNER_R  = os.path.join(_HERE, "verisense_runner_cluster.R")

# Add OAK to Python path so base.py can be imported
if _OAK_DIR not in sys.path:
    sys.path.insert(0, _OAK_DIR)


# ===========================================================================
# SHARED UTILITIES
# ===========================================================================

def read_actigraph(filepath: str, default_hz: int = 80) -> tuple:
    """Parse an ActiGraph GT3X+ CSV file.

    Returns
    -------
    data     : pd.DataFrame with columns x, y, z
    start_dt : datetime
    hz       : int
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
    """Aggregate per-second step counts into fixed-length time bins."""
    n    = len(steps_per_sec)
    rows = []
    for i in range(int(np.ceil(n / interval))):
        s  = i * interval
        e  = min(s + interval, n)
        t0 = start_dt + timedelta(seconds=s)
        rows.append({
            "Time":  t0.strftime("%Y-%m-%d %H:%M:%S"),
            "Steps": float(int(np.nansum(steps_per_sec[s:e])))
        })
    return pd.DataFrame(rows)


# ===========================================================================
# ALGORITHM 1 — VERISENSE (R via Apptainer)
# ===========================================================================
# verisense_runner_cluster.R (same directory as this script) sources
# verisense_count_steps.R from /home/wang.yichen8/4_algo_original_code/verisense/,
# resamples to 15 Hz, and writes per-second step counts to a CSV.
#
# Requires:
#   - Apptainer on PATH
#   - _ROCKER_IMAGE pointing to rocker-geospatial-4.4.2.sif
#   - verisense_runner_cluster.R in the same directory as pipeline_cluster.py
# ===========================================================================

def run_verisense(data: pd.DataFrame, start_dt: datetime,
                  hz: int, interval: int = 10) -> pd.DataFrame:
    """Run Verisense step detection via the original R script inside Apptainer."""
    import tempfile

    n_sec = int(np.ceil(len(data) / hz))

    if not os.path.isfile(_VERISENSE_RUNNER_R):
        print(f"  [Verisense] verisense_runner_cluster.R not found at {_VERISENSE_RUNNER_R}")
        return build_output(np.zeros(n_sec), start_dt, interval)

    if not os.path.isfile(_ROCKER_IMAGE):
        print(f"  [Verisense] Rocker image not found: {_ROCKER_IMAGE}")
        return build_output(np.zeros(n_sec), start_dt, interval)

    with tempfile.TemporaryDirectory(prefix="verisense_") as tmp:
        in_csv  = os.path.join(tmp, "xyz.csv")
        out_csv = os.path.join(tmp, "steps.csv")

        data[["x", "y", "z"]].to_csv(in_csv, index=False)

        import shutil
        container_exe = "apptainer" if shutil.which("apptainer") else (
                        "singularity" if shutil.which("singularity") else None)
        if container_exe is None:
            print("  [Verisense] Neither 'apptainer' nor 'singularity' found on PATH.")
            print("  [Verisense] Run:  module load apptainer  (then retry)")
            return build_output(np.zeros(n_sec), start_dt, interval)

        cmd = [
            container_exe, "exec",
            "-B", _APPTAINER_BINDS,
            _ROCKER_IMAGE,
            "Rscript", "--vanilla",
            _VERISENSE_RUNNER_R, in_csv, out_csv, str(hz)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            print(f"  [Verisense] '{container_exe}' not found. Run: module load apptainer")
            return build_output(np.zeros(n_sec), start_dt, interval)

        if result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                print(f"  [Verisense/R] {line}")
        if result.stderr.strip():
            for line in result.stderr.strip().splitlines():
                print(f"  [Verisense/R] {line}")

        if result.returncode != 0:
            print(f"  [Verisense] Rscript exited with code {result.returncode}")
            return build_output(np.zeros(n_sec), start_dt, interval)

        if not os.path.isfile(out_csv):
            print("  [Verisense] R script did not produce output CSV.")
            return build_output(np.zeros(n_sec), start_dt, interval)

        steps_df = pd.read_csv(out_csv)

    steps_per_sec = np.zeros(n_sec, dtype=float)
    for _, row in steps_df.iterrows():
        idx = int(row["second"])
        if 0 <= idx < n_sec:
            steps_per_sec[idx] = float(row["steps"])

    return build_output(steps_per_sec, start_dt, interval)


# ===========================================================================
# ALGORITHM 2 — ADEPT (R via Apptainer)
# ===========================================================================
# adept_runner_cluster.R (same directory as this script) sources the ADEPT R files
# from /home/wang.yichen8/4_algo_original_code/ADEPT/ and writes per-second
# step counts to a CSV that run_adept() reads back.
#
# Requires:
#   - Apptainer on PATH
#   - _ROCKER_IMAGE pointing to rocker-geospatial-4.4.2.sif
#   - adept_runner_cluster.R in the same directory as pipeline_cluster.py
#   - R packages: pracma, dplyr, dvmisc, assertthat (+ optionally adeptdata)
#     Pre-install these once: apptainer exec <image> Rscript -e "install.packages(...)"
# ===========================================================================

def run_adept(data: pd.DataFrame, start_dt: datetime,
              hz: int, interval: int = 10,
              n_cores: int = 4) -> pd.DataFrame:
    """Run ADEPT stride segmentation via R inside a Singularity/Apptainer container.

    Parameters
    ----------
    n_cores : parallel cores passed to R's segmentPattern (default 4).
              Match this to --cpus-per-task in your SLURM script.
              Results are identical to single-core; only speed differs.
    """
    import tempfile

    n_sec = int(np.ceil(len(data) / hz))

    if not os.path.isfile(_ADEPT_RUNNER_R):
        print(f"  [ADEPT] adept_runner_cluster.R not found at {_ADEPT_RUNNER_R}")
        return build_output(np.zeros(n_sec), start_dt, interval)

    if not os.path.isfile(_ROCKER_IMAGE):
        print(f"  [ADEPT] Rocker image not found: {_ROCKER_IMAGE}")
        return build_output(np.zeros(n_sec), start_dt, interval)

    # Use /tmp for temp files (Singularity binds /tmp from host by default)
    with tempfile.TemporaryDirectory(prefix="adept_") as tmp:
        in_csv  = os.path.join(tmp, "xyz.csv")
        out_csv = os.path.join(tmp, "steps.csv")

        data[["x", "y", "z"]].to_csv(in_csv, index=False)

        # Try apptainer first, fall back to singularity (same syntax, different name)
        import shutil
        container_exe = "apptainer" if shutil.which("apptainer") else (
                        "singularity" if shutil.which("singularity") else None)
        if container_exe is None:
            print("  [ADEPT] Neither 'apptainer' nor 'singularity' found on PATH.")
            print("  [ADEPT] Run:  module load apptainer  (then retry)")
            return build_output(np.zeros(n_sec), start_dt, interval)

        cmd = [
            container_exe, "exec",
            "-B", _APPTAINER_BINDS,
            _ROCKER_IMAGE,
            "Rscript", "--vanilla",
            _ADEPT_RUNNER_R, in_csv, out_csv, str(hz), str(n_cores)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            print(f"  [ADEPT] '{container_exe}' not found. Run: module load apptainer")
            return build_output(np.zeros(n_sec), start_dt, interval)

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

    steps_per_sec = np.zeros(n_sec, dtype=float)
    for _, row in steps_df.iterrows():
        idx = int(row["second"])
        if 0 <= idx < n_sec:
            steps_per_sec[idx] = float(row["steps"])

    return build_output(steps_per_sec, start_dt, interval)


# ===========================================================================
# ALGORITHM 3 — OAK (CWT-based, Python)
# ===========================================================================
# Loads base.py from _OAK_DIR; stubs out the 'forest' package if absent.
# ===========================================================================

def _load_oak_base():
    """Load oak/base.py from the cluster algorithm directory."""
    import importlib.util
    import types
    import enum

    if "forest" not in sys.modules:
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
        sys.modules["forest"]           = forest_pkg
        sys.modules["forest.constants"] = forest_constants
        sys.modules["forest.utils"]     = forest_utils

    base_path = os.path.join(_OAK_DIR, "base.py")
    if not os.path.isfile(base_path):
        raise ImportError(f"OAK base.py not found at {base_path}")

    spec = importlib.util.spec_from_file_location("oak_base", base_path)
    mod  = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError as exc:
        pkg = str(exc).split("'")[1] if "'" in str(exc) else str(exc)
        raise ImportError(
            f"OAK requires '{pkg}' — install it in the active conda env:\n"
            f"  pip install {pkg}"
        )
    return mod


def run_oak(data: pd.DataFrame, start_dt: datetime,
            hz: int, interval: int = 10) -> pd.DataFrame:
    """Run OAK CWT-based step counting using the original base.py."""
    _base_mod       = _load_oak_base()
    preprocess_bout = _base_mod.preprocess_bout
    find_walking    = _base_mod.find_walking

    target_fs = 10
    factor    = hz // target_fs
    n_full    = len(data) // factor * factor
    x10 = data["x"].values[:n_full].reshape(-1, factor).mean(axis=1)
    y10 = data["y"].values[:n_full].reshape(-1, factor).mean(axis=1)
    z10 = data["z"].values[:n_full].reshape(-1, factor).mean(axis=1)

    n10 = len(x10)
    t10 = np.array([start_dt.timestamp() + i * (1.0 / target_fs)
                    for i in range(n10)])

    t_i, vm = preprocess_bout(t10, x10, y10, z10, fs=target_fs)

    n_sec = int(np.ceil(len(data) / hz))
    if len(t_i) == 0:
        return build_output(np.zeros(n_sec), start_dt, interval)

    cad = find_walking(
        vm, fs=target_fs,
        min_amp=0.3,
        step_freq=(1.4, 2.3),
        alpha=0.6,
        beta=2.5,
        min_t=3,
        delta=20,
    )

    steps_per_sec = np.round(cad).astype(float)
    return build_output(steps_per_sec, start_dt, interval)


# ===========================================================================
# ALGORITHM 4 — OXFORD (stepcount CLI, active conda env)
# ===========================================================================

def _prepare_oxford_input(data: pd.DataFrame, start_dt: datetime,
                          hz: int, out_path: str) -> None:
    step_td    = timedelta(seconds=1.0 / hz)
    timestamps = [start_dt + i * step_td for i in range(len(data))]
    df         = data.copy()
    df.insert(0, "time", timestamps)
    df.to_csv(out_path, index=False)


def run_oxford(data: pd.DataFrame, start_dt: datetime,
               hz: int, interval: int = 10,
               work_dir: str = None,
               oxford_exe: str = None) -> pd.DataFrame:
    """Run Oxford stepcount CLI (uses active conda env by default)."""
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
        print(
            f"  [Oxford] '{exe}' not found.\n"
            "  Make sure the stepcount conda env is active:\n"
            "    module load anaconda3/2024.06 && source activate stepcount"
        )
        return build_output(np.zeros(n_sec), start_dt, interval)

    matches = glob_module.glob(os.path.join(work_dir, "**", "*-Steps.csv.gz"), recursive=True)
    if not matches:
        matches = glob_module.glob(os.path.join(work_dir, "**", "*-Steps.csv"), recursive=True)

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

def run_pipeline_on_file(input_file: str, output_dir: str,
                         algorithms: list = None,
                         interval: int = 10,
                         oxford_exe: str = None) -> None:
    """Run all selected algorithms on a single ActiGraph CSV file."""
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
                out_df = run_adept(data, start_dt, hz, interval)

            elif algo == "oak":
                out_df = run_oak(data, start_dt, hz, interval)

            elif algo == "oxford":
                oxford_work = os.path.join(output_dir, f"{stem}_oxford_work")
                out_df = run_oxford(data, start_dt, hz, interval,
                                    oxford_work, oxford_exe)
            else:
                print(f"  Unknown algorithm '{algo}' — skipping.")
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


def collect_input_files(input_arg: str) -> list:
    """Resolve input_arg to a list of CSV file paths.

    Accepts:
      - A single .csv file path
      - A .txt list file with one (optionally quoted) CSV path per line
      - A directory — all *.csv files inside are returned
    """
    if os.path.isdir(input_arg):
        files = sorted(glob_module.glob(os.path.join(input_arg, "*.csv")))
        if not files:
            print(f"WARNING: No .csv files found in directory {input_arg}")
        return files

    if input_arg.lower().endswith(".txt"):
        files = []
        with open(input_arg, "r") as fh:
            for line in fh:
                path = line.strip().strip('"').strip("'")
                if path:
                    files.append(path)
        return files

    # Treat as a single CSV path
    return [input_arg]


# ===========================================================================
# CLI ENTRY POINT
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Four-algorithm step count pipeline (cluster version).\n"
            "Runs Verisense, ADEPT, OAK, and Oxford on ActiGraph GT3X+ CSV files\n"
            "and writes one Time,Steps CSV per algorithm per input file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        help=(
            "One of:\n"
            "  (1) path to a single ActiGraph CSV file\n"
            "  (2) path to a .txt list file (one CSV path per line)\n"
            "  (3) path to a directory of CSV files"
        )
    )
    parser.add_argument(
        "output_dir",
        help="Directory where output CSVs are written"
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
            "Full path to the stepcount executable when not on PATH. "
            "Usually not needed if the stepcount conda env is active."
        )
    )
    args = parser.parse_args()

    csv_files = collect_input_files(args.input)
    if not csv_files:
        print("No input files found. Exiting.")
        sys.exit(1)

    print(f"Input files  : {len(csv_files)}")
    print(f"Output dir   : {args.output_dir}")
    print(f"Algorithms   : {', '.join(args.algorithms)}")
    print(f"Bin interval : {args.interval} s")

    for csv_path in csv_files:
        if not os.path.isfile(csv_path):
            print(f"\nSKIPPING (file not found): {csv_path}")
            continue
        run_pipeline_on_file(
            input_file = csv_path,
            output_dir = args.output_dir,
            algorithms = args.algorithms,
            interval   = args.interval,
            oxford_exe = args.oxford_exe,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
