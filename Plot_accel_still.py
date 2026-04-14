#!/usr/bin/env python3
"""
Generate accelerometer data plots for 'Still' physical activity labels.
Runs in parallel across all subjects in PAAWS_FreeLiving directory.
Only saves plots where the accelerometer signal is highly intensive
(high standard deviation) during the labeled still period.

Usage:
    python plot_accel_still.py
    # or with SLURM:  sbatch run_plot.sh
"""

import os
import re
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ========================== CONFIG ==========================
BASE_DIR = "/scratch/wang.yichen8/PAAWS_FreeLiving"
OUTPUT_DIR = "/scratch/wang.yichen8/PAAWS_FreeLiving/still_activity_plots"

TARGET_LABELS = [
    "Kneeling_Still",
    "Sitting_Still",
    "Standing_Still",
    "Lying_Still",
]

SENSOR_FILES = [
    "RightWrist",
    "RightWaist",
]

# Intensity threshold: only plot segments where the mean std across
# 3 axes exceeds this value (in g). Adjust as needed.
INTENSITY_STD_THRESHOLD = 0.5
# ============================================================


def parse_accel_header(filepath):
    """
    Parse the ActiGraph header to extract sampling rate, start date/time.
    Returns (sampling_rate_hz, start_datetime, header_end_line).
    """
    sampling_rate = 80  # default
    start_time = None
    start_date = None
    header_end_line = 0

    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            line_stripped = line.strip()

            # Extract sampling rate: "at 80 Hz"
            match = re.search(r'at\s+(\d+)\s+Hz', line_stripped)
            if match:
                sampling_rate = int(match.group(1))

            # Extract start time: "Start Time 18:00:00"
            match = re.search(r'Start Time\s+(\d+:\d+:\d+)', line_stripped)
            if match:
                start_time = match.group(1)

            # Extract start date: "Start Date 11/1/2021"
            match = re.search(r'Start Date\s+(\S+)', line_stripped)
            if match:
                start_date = match.group(1)

            # Find data header line
            if 'Accelerometer X' in line_stripped:
                header_end_line = i
                break

            if i > 30:
                break

    # Build start datetime
    start_dt = None
    if start_date and start_time:
        for fmt in ['%m/%d/%Y', '%Y-%m-%d', '%d/%m/%Y']:
            try:
                start_dt = datetime.strptime(
                    f"{start_date} {start_time}", f"{fmt} %H:%M:%S"
                )
                break
            except ValueError:
                continue

    return sampling_rate, start_dt, header_end_line


def read_accel_csv(filepath):
    """
    Read ActiGraph accelerometer CSV file.
    Returns (DataFrame with X,Y,Z, sampling_rate, start_datetime).
    """
    sampling_rate, start_dt, header_line = parse_accel_header(filepath)

    df = pd.read_csv(
        filepath,
        skiprows=header_line,
        sep=r'[\t,]+',
        engine='python',
        names=['X', 'Y', 'Z'],
        header=0,
        dtype={'X': np.float32, 'Y': np.float32, 'Z': np.float32},
        on_bad_lines='skip',
    )
    df = df.dropna().reset_index(drop=True)

    print(f"    Accel: {len(df)} samples, {sampling_rate} Hz, start={start_dt}")
    return df, sampling_rate, start_dt


def read_label_csv(filepath):
    """
    Read the label CSV file. Timestamps are full datetimes like
    '2021-11-01 18:24:11.012'.
    """
    df = pd.read_csv(filepath, on_bad_lines='skip')
    df.columns = [c.strip() for c in df.columns]

    # Strip whitespace from string columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()

    # Parse datetime columns
    df['start_dt'] = pd.to_datetime(df['START_TIME'], errors='coerce')
    df['stop_dt'] = pd.to_datetime(df['STOP_TIME'], errors='coerce')

    df = df.dropna(subset=['start_dt', 'stop_dt'])
    return df


def datetime_to_sample(dt, accel_start_dt, sampling_rate):
    """Convert a datetime to a sample index relative to accel recording start."""
    delta = (dt - accel_start_dt).total_seconds()
    return int(delta * sampling_rate)


def compute_segment_intensity(accel_segment):
    """Mean std across X, Y, Z axes."""
    if len(accel_segment) < 10:
        return 0.0
    return accel_segment[['X', 'Y', 'Z']].std().mean()


def plot_accel_segment(accel_df, start_sample, stop_sample, subject_id,
                       sensor_name, label_name, seg_idx, output_dir,
                       sampling_rate, label_start_dt, label_stop_dt):
    """
    Generate a 3-axis accelerometer plot matching the reference style.
    """
    # Context: 5 seconds before/after
    context_samples = int(5 * sampling_rate)
    plot_start = max(0, start_sample - context_samples)
    plot_stop = min(len(accel_df), stop_sample + context_samples)

    segment = accel_df.iloc[plot_start:plot_stop].copy().reset_index(drop=True)

    highlight_start = start_sample - plot_start
    highlight_stop = stop_sample - plot_start

    time_axis = np.arange(len(segment)) / sampling_rate

    # Downsample if very long (>50k points)
    if len(segment) > 50000:
        step = len(segment) // 50000 + 1
        idx = np.arange(0, len(segment), step)
        t_plot = time_axis[idx]
        x_plot = segment['X'].values[idx]
        y_plot = segment['Y'].values[idx]
        z_plot = segment['Z'].values[idx]
    else:
        t_plot = time_axis
        x_plot = segment['X'].values
        y_plot = segment['Y'].values
        z_plot = segment['Z'].values

    hl_start_t = highlight_start / sampling_rate
    hl_stop_t = highlight_stop / sampling_rate

    # ---- Plot ----
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    duration_str = (f"{label_start_dt.strftime('%H:%M:%S')} - "
                    f"{label_stop_dt.strftime('%H:%M:%S')}")
    fig.suptitle(
        f"Accelerometer Sensor Data\n"
        f"{subject_id} | {sensor_name} | {label_name} (Seg {seg_idx}) | {duration_str}",
        fontsize=13, fontweight='bold', y=0.98
    )

    colors = ['red', 'green', 'blue']
    bg_colors = ['#FFCCCC', '#CCFFCC', '#CCCCFF']
    axis_labels = ['X', 'Y', 'Z']
    data_arrays = [x_plot, y_plot, z_plot]

    for i, ax in enumerate(axes):
        ax.axvspan(hl_start_t, hl_stop_t, color=bg_colors[i], alpha=0.5, zorder=0)
        ax.plot(t_plot, data_arrays[i], color=colors[i], linewidth=0.4, zorder=1)
        ax.set_ylabel(axis_labels[i], fontsize=16, fontweight='bold',
                       rotation=0, labelpad=20)
        ax.set_ylim(-20, 20)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel("Time (seconds)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    fname = f"{subject_id}_{sensor_name}_{label_name}_seg{seg_idx}.png"
    fpath = os.path.join(output_dir, fname)
    fig.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fpath


def process_subject(subject_dir):
    """Process a single subject: read labels + accel, filter, plot."""
    subject_id = os.path.basename(subject_dir)
    results = []

    # --- Find label file ---
    label_files = glob.glob(
        os.path.join(subject_dir, "label", f"{subject_id}-Free-label.csv")
    )
    if not label_files:
        label_files = glob.glob(os.path.join(subject_dir, "label", "*label*.csv"))
    if not label_files:
        print(f"[{subject_id}] No label file found, skipping.")
        return results

    try:
        labels_df = read_label_csv(label_files[0])
    except Exception as e:
        print(f"[{subject_id}] Error reading labels: {e}")
        return results

    # --- Filter for target PA_TYPE ---
    pa_col = None
    for col in ['PA_TYPE', 'PA Type', 'pa_type']:
        if col in labels_df.columns:
            pa_col = col
            break
    if pa_col is None:
        print(f"[{subject_id}] No PA_TYPE column. Columns: {list(labels_df.columns)}")
        return results

    still_labels = labels_df[labels_df[pa_col].isin(TARGET_LABELS)].copy()
    if still_labels.empty:
        sample_vals = labels_df[pa_col].dropna().unique()[:5]
        print(f"[{subject_id}] No still-activity labels. Sample: {list(sample_vals)}")
        return results

    print(f"[{subject_id}] Found {len(still_labels)} still-activity segments.")

    # --- Process each sensor ---
    for sensor_name in SENSOR_FILES:
        accel_files = glob.glob(
            os.path.join(subject_dir, "accel", f"{subject_id}-Free-{sensor_name}.csv")
        )
        if not accel_files:
            accel_files = glob.glob(
                os.path.join(subject_dir, "accel", f"*{sensor_name}*.csv")
            )
        if not accel_files:
            print(f"  [{subject_id}][{sensor_name}] No accel file, skipping.")
            continue

        try:
            accel_df, sampling_rate, accel_start_dt = read_accel_csv(accel_files[0])
        except Exception as e:
            print(f"  [{subject_id}][{sensor_name}] Error reading accel: {e}")
            continue

        if accel_start_dt is None:
            print(f"  [{subject_id}][{sensor_name}] Could not parse accel start datetime.")
            continue

        total_samples = len(accel_df)

        # --- Plot each still segment ---
        for seg_idx, (_, row) in enumerate(still_labels.iterrows()):
            label_start_dt = row['start_dt'].to_pydatetime()
            label_stop_dt = row['stop_dt'].to_pydatetime()
            label_name = row[pa_col]

            start_sample = datetime_to_sample(
                label_start_dt, accel_start_dt, sampling_rate
            )
            stop_sample = datetime_to_sample(
                label_stop_dt, accel_start_dt, sampling_rate
            )

            # Bounds check
            if start_sample >= total_samples or stop_sample <= 0:
                continue
            start_sample = max(0, start_sample)
            stop_sample = min(total_samples, stop_sample)
            if stop_sample - start_sample < sampling_rate:  # skip < 1 sec
                continue

            # Compute intensity
            seg_data = accel_df.iloc[start_sample:stop_sample]
            intensity = compute_segment_intensity(seg_data)

            if intensity < INTENSITY_STD_THRESHOLD:
                continue

            print(f"    Seg {seg_idx}: {label_name} | "
                  f"{label_start_dt} -> {label_stop_dt} | "
                  f"intensity={intensity:.3f} (PLOTTING)")

            subj_output_dir = os.path.join(OUTPUT_DIR, subject_id)
            os.makedirs(subj_output_dir, exist_ok=True)

            try:
                fpath = plot_accel_segment(
                    accel_df, start_sample, stop_sample,
                    subject_id, sensor_name, label_name, seg_idx,
                    subj_output_dir, sampling_rate,
                    label_start_dt, label_stop_dt
                )
                results.append(fpath)
            except Exception as e:
                print(f"    Error plotting seg {seg_idx}: {e}")

    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    subject_dirs = sorted(glob.glob(os.path.join(BASE_DIR, "DS_*")))
    if not subject_dirs:
        print(f"No subject directories found in {BASE_DIR}")
        return

    print(f"Found {len(subject_dirs)} subjects.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Intensity threshold (std): {INTENSITY_STD_THRESHOLD}")
    print(f"Target labels: {TARGET_LABELS}")
    print(f"Sensors: {SENSOR_FILES}")
    print("=" * 60)

    n_workers = min(cpu_count(), len(subject_dirs), 16)
    print(f"Using {n_workers} parallel workers.\n")

    with Pool(processes=n_workers) as pool:
        all_results = pool.map(process_subject, subject_dirs)

    all_plots = [p for sublist in all_results for p in sublist]

    print("\n" + "=" * 60)
    print(f"Done! Generated {len(all_plots)} high-intensity plots.")
    print(f"Saved to: {OUTPUT_DIR}")

    if all_plots:
        summary_path = os.path.join(OUTPUT_DIR, "plot_summary.csv")
        with open(summary_path, 'w') as f:
            f.write("plot_path\n")
            for p in all_plots:
                f.write(f"{p}\n")
        print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()