#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║         STEP ALGORITHM COMPARISON ANALYSIS  —  Paper Final          ║
╠══════════════════════════════════════════════════════════════════════╣
║  Categories:                                                         ║
║    A) Move+Step      : locomotion (walking, running, frisbee)       ║
║    B) Move+NoStep    : arm/body movement, no foot stepping          ║
║    C) NoMove+NoStep  : no movement, no stepping                     ║
║    D) Move+MaybeStep : ambiguous (cycling, chores, resistance)      ║
║                                                                      ║
║  Outputs:                                                            ║
║    Table 1 — Activity-level step rates (mean ± std) per algorithm   ║
║    Table 2 — Category-level summary per algorithm × sensor          ║
║    Fig 1   — FP rates: NoMove+NoStep vs Move+NoStep                ║
║    Fig 2a  — Algo-vs-algo scatter, restricted to one wear location  ║
║    Fig 2b  — LeftWrist vs RightWrist scatter, per algorithm         ║
║    Fig 3   — Step attribution stacked bar (4 categories)            ║
║    Fig 4   — Sensitivity vs specificity scatter                     ║
║    Fig 5   — Pseudo-ROC threshold sweep                             ║
║                                                                      ║
║  Sensors (PAAWS v2): LeftWrist, RightWrist, RightWaist,             ║
║                     RightThigh, RightAnkle, Phone                   ║
║  Comparisons are run within a sensor only; the left/right wrist     ║
║  pair gets an additional cross-sensor scatter per algorithm.        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, re, glob, argparse, warnings, gzip, io
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ROOT   = "/scratch/wang.yichen8/PAAWS_results_v3"
DEFAULT_OUTPUT = "/scratch/wang.yichen8/step_analysis_output_v3"
DEFAULT_LABELS = "/scratch/wang.yichen8/PAAWS_FreeLiving"
ALGORITHMS     = ["adept", "oak", "oxford", "verisense"]
SENSORS        = ["LeftWrist", "RightWrist", "RightWaist",
                  "RightThigh", "RightAnkle", "Phone"]

# Algorithms expected per wear location.  Only algorithms listed for a
# sensor are valid comparisons on that location — oak/oxford/verisense
# each only ship for the sensors they were trained for.  Comparisons
# (tables, FP-rate, attribution, ROC) are run independently per sensor
# using just that sensor's allowed algorithm set.
SENSOR_ALGO_MAP = {
    "LeftWrist":  ["adept", "oxford", "verisense", "oak"],
    "RightWrist": ["adept", "oxford", "verisense", "oak"],
    "RightWaist": ["adept", "oak", "oxford"],
    "RightThigh": ["adept", "oak"],
    "RightAnkle": ["adept", "oak"],
    "Phone":      ["adept", "oak"],
}

# Wrist locations get an additional Left-vs-Right scatter per algorithm
WRIST_SENSORS = ["LeftWrist", "RightWrist"]

LABEL_PATTERNS = [
    "{ds_root}/labels/{ds_id}-Free-label.csv",
    "{ds_root}/labels/{ds_id}-Free-label.csv.gz",
    "{ds_root}/{ds_id}-Free-label.csv",
    "{ds_root}/{ds_id}-Free-label.csv.gz",
]

# ── 4-Category Definitions ──────────────────────────────────────────────────

MOVE_STEP_ACTIVITIES = [
    "Walking", "Walking_Fast", "Walking_Slow", "Walking_Treadmill",
    "Walking_Up_Stairs", "Walking_Down_Stairs",
    "Playing_Frisbee", "Running_Non-Treadmill", "Running_Treadmill",
]

MOVE_NOSTEP_ACTIVITIES = [
    "Kneeling_With_Movement", "Bathing",
    "Sitting_With_Movement", "Lying_With_Movement",
    "Applying_Makeup", "Blowdrying_Hair",
    "Washing_Face", "Brushing_Teeth", "Brushing/Combing/Tying_Hair",
    "Flossing_Teeth", "Washing_Hands",
]

NOMOVE_NOSTEP_ACTIVITIES = [
    "Kneeling_Still", "Sitting_Still", "Standing_Still", "Lying_Still",
]

MOVE_MAYBESTEP_ACTIVITIES = [
    "Cycling_Active_Pedaling_Regular_Bicycle",
    "Cycling_Active_Pedaling_Stationary_Bike",
    "Doing_Resistance_Training_Free_Weights",
    "Doing_Resistance_Training_Other",
    "Puttering_Around", "Loading/Unloading_Washing_Machine/Dryer",
    "Standing_With_Movement", "Showering",
    "Putting_Clothes_Away", "Organizing_Shelf/Cabinet"
    "Watering_Plants", "Dusting", "Dry_Mopping", "Sweeping",
    "Vacuuming", "Wet_Mopping", "Folding_Clothes", "Ironing"
]

EXCLUDE_ACTIVITIES = [
    "PA_Type_Video_Unavailable/Indecipherable",
    "Posture_Video_Unavailable/Indecipherable",
    "Synchronizing_Sensors", "PA_Type_Too_Complex",
    "PA_Type_Other", "PA_Type_Unlabeled",
]

CATEGORY_MAP = {
    "Move+Step":      MOVE_STEP_ACTIVITIES,
    "Move+NoStep":    MOVE_NOSTEP_ACTIVITIES,
    "NoMove+NoStep":  NOMOVE_NOSTEP_ACTIVITIES,
    "Move+MaybeStep": MOVE_MAYBESTEP_ACTIVITIES,
}

CATEGORY_ORDER = ["Move+Step", "Move+NoStep", "NoMove+NoStep",
                  "Move+MaybeStep"]

CATEGORY_DISPLAY = {
    "Move+Step":      "Movement & Step",
    "Move+NoStep":    "Movement & No Step",
    "NoMove+NoStep":  "No Movement & No Step",
    "Move+MaybeStep": "Movement & Maybe Step",
}

MIN_SEGMENT_SEC = 5

# ─────────────────────────────────────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────────────────────────────────────

ALGO_COLORS = {"oxford": "#2E86AB", "oak": "#E84855",
               "verisense": "#57CC99", "adept": "#F4A261"}
EXTRA_COLORS = ["#9B5DE5", "#F15BB5", "#FEE440", "#00BBF9"]

DARK_BG     = "#FFFFFF"
DARK_AX     = "#FFFFFF"
GRID_COL    = "#E0E0E0"
TEXT_COL    = "#111111"
SPINE_COL   = "#BDBDBD"
LEGEND_FACE = "#FFFFFF"
LEGEND_EDGE = "#CCCCCC"
LEGEND_TEXT = TEXT_COL

CATEGORY_COLORS = {
    "Move+Step":      "#57CC99",
    "Move+NoStep":    "#E84855",
    "NoMove+NoStep":  "#2E86AB",
    "Move+MaybeStep": "#F4A261",
}

def algo_color(name):
    for key, col in ALGO_COLORS.items():
        if key in name.lower():
            return col
    return EXTRA_COLORS[hash(name) % len(EXTRA_COLORS)]


def get_algos_for_sensor(df_long, sensor):
    """Return sorted list of algorithms present for a given sensor in
    df_long, intersected with that sensor's allowed algorithms in
    SENSOR_ALGO_MAP (so unexpected combos are dropped)."""
    expected = set(SENSOR_ALGO_MAP.get(sensor, ALGORITHMS))
    present  = set(df_long[df_long["sensor"] == sensor]
                   ["algorithm"].unique())
    return sorted(expected & present)

# ─────────────────────────────────────────────────────────────────────────────
# ACTIVITY → CATEGORY
# ─────────────────────────────────────────────────────────────────────────────

def get_category(pa_type):
    for cat, acts in CATEGORY_MAP.items():
        if pa_type in acts:
            return cat
    return "Other"

# ─────────────────────────────────────────────────────────────────────────────
# SAFE FORMATTING HELPER  (fixes NaN and encoding)
# ─────────────────────────────────────────────────────────────────────────────

def fmt_mean_std(vals):
    """Return 'mean +/- std' with safe handling of n<=1 (NaN std)."""
    if len(vals) == 0:
        return "—"
    m = vals.mean()
    s = vals.std()
    if np.isnan(m):
        return "—"
    if np.isnan(s) or len(vals) == 1:
        # single observation → std is undefined, report mean only
        return f"{m:.1f} +/- 0.0"
    return f"{m:.1f} +/- {s:.1f}"

# ─────────────────────────────────────────────────────────────────────────────
# FILE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def discover_ds_folders(root):
    folders = sorted([d for d in glob.glob(os.path.join(root, "DS_*"))
                      if os.path.isdir(d)])
    if not folders:
        raise FileNotFoundError(f"No DS_* folders found under: {root}")
    return folders


def find_label_file(ds_root, ds_id, label_root=None):
    if label_root:
        for sub in [
            os.path.join(label_root, ds_id, "label",
                         f"{ds_id}-Free-label.csv"),
            os.path.join(label_root, ds_id, "label",
                         f"{ds_id}-Free-label.csv.gz"),
            os.path.join(label_root, ds_id,
                         f"{ds_id}-Free-label.csv"),
            os.path.join(label_root, ds_id,
                         f"{ds_id}-Free-label.csv.gz"),
        ]:
            if os.path.isfile(sub):
                return sub
    for pat in LABEL_PATTERNS:
        path = pat.format(ds_root=ds_root, ds_id=ds_id)
        if os.path.isfile(path):
            return path
    for ext in ["csv", "csv.gz"]:
        hits = glob.glob(os.path.join(ds_root, "**", f"*label*.{ext}"),
                         recursive=True)
        if hits:
            return hits[0]
    return None


def find_steps_files(ds_root, ds_id, algorithm):
    hits = []
    for ext in ["csv.gz", "csv"]:
        hits.extend(glob.glob(
            os.path.join(ds_root,
                         f"*-Free-*_{algorithm}_steps.{ext}")))
    if not hits:
        return []
    seen = {}
    for h in hits:
        stem = re.sub(r"\.csv(\.gz)?$", "", os.path.basename(h))
        if stem not in seen or h.endswith(".gz"):
            seen[stem] = h
    results = []
    for stem, path in seen.items():
        m = re.search(r"-Free-(.+?)_" + re.escape(algorithm), stem)
        sensor = m.group(1) if m else stem
        results.append((sensor, path))
    return sorted(results)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def read_csv_auto(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return pd.read_csv(io.BytesIO(f.read()))
    return pd.read_csv(path)


def load_labels(path):
    df = read_csv_auto(path)
    df["START_TIME"] = pd.to_datetime(df["START_TIME"])
    df["STOP_TIME"]  = pd.to_datetime(df["STOP_TIME"])
    df["duration_sec"] = (
        df["STOP_TIME"] - df["START_TIME"]).dt.total_seconds()
    df = df[df["duration_sec"] >= MIN_SEGMENT_SEC].copy()
    df = df[~df["PA_TYPE"].isin(EXCLUDE_ACTIVITIES)].copy()
    return df.dropna(subset=["PA_TYPE"]).reset_index(drop=True)


def load_steps_series(path):
    df = read_csv_auto(path)
    df.columns = [c.strip().lower() for c in df.columns]
    time_col = next(
        (c for c in df.columns if "time" in c or "date" in c),
        df.columns[0])
    step_col = next(
        (c for c in df.columns if "step" in c), df.columns[1])
    df[time_col] = pd.to_datetime(df[time_col])
    return df.set_index(time_col)[step_col].sort_index()

# ─────────────────────────────────────────────────────────────────────────────
# CORE AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_segments(labels_df, series, ds_id, algorithm, sensor):
    idx    = series.index.values.astype("int64")
    sv     = series.values.astype(float)
    starts = labels_df["START_TIME"].values.astype("int64")
    stops  = labels_df["STOP_TIME"].values.astype("int64")
    durs   = labels_df["duration_sec"].values
    pa_types = labels_df["PA_TYPE"].values

    lo = np.searchsorted(idx, starts, side="left")
    hi = np.searchsorted(idx, stops,  side="left")

    records = []
    for i in range(len(labels_df)):
        total = float(sv[lo[i]:hi[i]].sum())
        dur   = durs[i]
        spm   = (total / dur * 60) if dur > 0 else 0.0
        records.append({
            "ds_id":        ds_id,
            "algorithm":    algorithm,
            "sensor":       sensor,
            "PA_TYPE":      pa_types[i],
            "category":     get_category(pa_types[i]),
            "duration_sec": dur,
            "total_steps":  total,
            "steps_per_min": spm,
        })
    return records

# ─────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _style(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(DARK_AX)
    ax.tick_params(colors=TEXT_COL)
    for sp in ax.spines.values():
        sp.set_color(SPINE_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.5, linestyle="--",
            axis="both")
    ax.set_axisbelow(True)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT_COL, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=10)
    if title:  ax.set_title(title,   color=TEXT_COL, fontsize=11,
                            fontweight="bold", pad=10)


def _save(fig, path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"    Saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
#  TABLE 1 — Activity-level step rates per algorithm
# ═════════════════════════════════════════════════════════════════════════════

def table1_activity_step_rates(df_long, out_dir):
    sensors = [s for s in SENSORS if s in df_long["sensor"].unique()]

    ordered_acts = []
    for cat in CATEGORY_ORDER:
        acts = CATEGORY_MAP.get(cat, [])
        present = [a for a in acts if a in df_long["PA_TYPE"].unique()]
        ordered_acts.extend([(cat, a) for a in present])

    # ── Per-sensor tables: columns = only that sensor's algorithms ──
    for sensor in sensors:
        s_df  = df_long[df_long["sensor"] == sensor]
        algos = get_algos_for_sensor(df_long, sensor)
        if not algos:
            continue
        rows = []
        for cat, act in ordered_acts:
            row = {
                "Category": CATEGORY_DISPLAY.get(cat, cat),
                "Activity": act.replace("_", " "),
            }
            for algo in algos:
                vals = s_df[(s_df["algorithm"] == algo) &
                            (s_df["PA_TYPE"] == act)]["steps_per_min"]
                row[algo] = fmt_mean_std(vals)
            rows.append(row)

        paper_cols = ["Category", "Activity"] + algos
        pd.DataFrame(rows)[paper_cols].to_csv(
            os.path.join(out_dir,
                         f"table1_{sensor}_activity_step_rates.csv"),
            index=False, encoding="utf-8-sig")
        print(f"    Saved: table1_{sensor}_activity_step_rates.csv")

    # ── Combined table: one column per (sensor, algo) pair that exists ──
    sensor_algo_pairs = [
        (sensor, algo)
        for sensor in sensors
        for algo in get_algos_for_sensor(df_long, sensor)
    ]
    rows_all = []
    for cat, act in ordered_acts:
        row = {
            "Category": CATEGORY_DISPLAY.get(cat, cat),
            "Activity": act.replace("_", " "),
        }
        for sensor, algo in sensor_algo_pairs:
            vals = df_long[(df_long["sensor"]    == sensor) &
                           (df_long["algorithm"] == algo)   &
                           (df_long["PA_TYPE"]   == act)]["steps_per_min"]
            row[f"{sensor}_{algo}"] = fmt_mean_std(vals)
        rows_all.append(row)
    pd.DataFrame(rows_all).to_csv(
        os.path.join(out_dir,
                     "table1_all_sensors_activity_step_rates.csv"),
        index=False, encoding="utf-8-sig")
    print("    Saved: table1_all_sensors_activity_step_rates.csv")


# ═════════════════════════════════════════════════════════════════════════════
#  TABLE 2 — Category-level summary per algorithm × sensor
# ═════════════════════════════════════════════════════════════════════════════

def table2_category_summary(df_long, out_dir):
    sensors = [s for s in SENSORS if s in df_long["sensor"].unique()]

    for sensor in sensors:
        s_df  = df_long[df_long["sensor"] == sensor]
        algos = get_algos_for_sensor(df_long, sensor)
        if not algos:
            continue
        rows = []
        for algo in algos:
            a_df = s_df[s_df["algorithm"] == algo]
            row = {"Algorithm": algo}
            for cat in CATEGORY_ORDER:
                vals = a_df[a_df["category"] == cat]["steps_per_min"]
                disp = CATEGORY_DISPLAY.get(cat, cat)
                if len(vals) > 0:
                    m = vals.mean()
                    s = vals.std()
                    med = vals.median()
                    n = len(vals)
                    if np.isnan(s) or n == 1:
                        s = 0.0
                    row[disp] = (
                        f"{m:.1f} +/- {s:.1f} "
                        f"(med={med:.1f}, n={n})")
                else:
                    row[disp] = "—"
            rows.append(row)

        paper_cols = (["Algorithm"] +
                      [CATEGORY_DISPLAY[c] for c in CATEGORY_ORDER])
        pd.DataFrame(rows)[paper_cols].to_csv(
            os.path.join(out_dir,
                         f"table2_{sensor}_category_summary.csv"),
            index=False, encoding="utf-8-sig")
        print(f"    Saved: table2_{sensor}_category_summary.csv")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — FP rates: NoMove+NoStep vs Move+NoStep
# ═════════════════════════════════════════════════════════════════════════════

def fig1_fp_rates(df_long, out_dir):
    sensors = [s for s in SENSORS if s in df_long["sensor"].unique()]

    neg_cats   = ["NoMove+NoStep", "Move+NoStep"]
    neg_colors = {"NoMove+NoStep": "#2E86AB", "Move+NoStep": "#E84855"}
    neg_labels = {
        "NoMove+NoStep": f"No Movement & No Step ({len(NOMOVE_NOSTEP_ACTIVITIES)} activities)",
        "Move+NoStep":   f"Movement & No Step ({len(MOVE_NOSTEP_ACTIVITIES)} activities)",
    }

    rows = []
    for sensor in sensors:
        s_df  = df_long[df_long["sensor"] == sensor]
        algos = get_algos_for_sensor(df_long, sensor)
        for algo in algos:
            a_df = s_df[s_df["algorithm"] == algo]
            for cat in neg_cats:
                sub = a_df[a_df["category"] == cat]
                fp = ((sub["total_steps"] > 0).mean() * 100
                      if len(sub) else np.nan)
                rows.append({
                    "sensor": sensor, "algorithm": algo,
                    "category": cat, "fp_rate_pct": fp,
                    "n_segments": len(sub),
                })

    fp_df = pd.DataFrame(rows)
    fp_df.to_csv(os.path.join(out_dir, "fig1_fp_rates.csv"),
                 index=False)

    sensors_with_data = [s for s in sensors
                         if get_algos_for_sensor(df_long, s)]
    n_sensors = len(sensors_with_data)
    if n_sensors == 0:
        print("    [fig1] No sensor data — skipped"); return

    # ── Wrap into 2 rows when there are many sensors ──
    n_cols = min(n_sensors, 3)
    n_rows = int(np.ceil(n_sensors / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6.5 * n_cols, 5.5 * n_rows),
                             squeeze=False)
    fig.patch.set_facecolor(DARK_BG)

    y_max = max(fp_df["fp_rate_pct"].max() * 1.2, 10)
    bar_w = 0.35
    offsets = [-bar_w / 2, bar_w / 2]

    for si, sensor in enumerate(sensors_with_data):
        ax = axes[si // n_cols, si % n_cols]
        ax.set_facecolor(DARK_AX)
        algos = get_algos_for_sensor(df_long, sensor)
        sf = fp_df[fp_df["sensor"] == sensor]
        x  = np.arange(len(algos))

        for ci, cat in enumerate(neg_cats):
            vals = np.array([
                sf[(sf["algorithm"] == a) & (sf["category"] == cat)]
                ["fp_rate_pct"].values[0]
                if len(sf[(sf["algorithm"] == a) &
                          (sf["category"] == cat)]) else 0.0
                for a in algos
            ])
            bars = ax.bar(x + offsets[ci], vals, bar_w,
                          label=neg_labels[cat],
                          color=neg_colors[cat], alpha=0.85,
                          edgecolor="#444", linewidth=0.6)
            for bar in bars:
                h = bar.get_height()
                if not np.isnan(h):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            h + 0.8,
                            f"{h:.1f}%", ha="center", va="bottom",
                            fontsize=8, color=TEXT_COL,
                            fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(algos, color=TEXT_COL, fontsize=10)
        ax.set_ylim(0, y_max)
        ax.legend(framealpha=0.9, labelcolor=LEGEND_TEXT,
                  facecolor=LEGEND_FACE, edgecolor=LEGEND_EDGE,
                  fontsize=8, loc="upper left")
        _style(ax,
               ylabel="False Positive Rate\n"
                      "(% segments with steps > 0)",
               title=f"False Positive Rate — {sensor}")
        ax.spines[["top", "right"]].set_visible(False)

    # Blank out unused subplot cells
    for k in range(n_sensors, n_rows * n_cols):
        axes[k // n_cols, k % n_cols].axis("off")

    _save(fig, os.path.join(out_dir, "fig1_fp_rates.png"))


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 1b — FP rates per activity within non-step categories
# ═════════════════════════════════════════════════════════════════════════════

def fig1b_fp_rates_by_activity(df_long, out_dir):
    """Break down FP rate by individual activity within NoMove+NoStep
    and Move+NoStep so we can see which activities drive false positives."""
    sensors = [s for s in SENSORS if s in df_long["sensor"].unique()]

    neg_cats = ["NoMove+NoStep", "Move+NoStep"]

    # ── Collect per-activity FP data (per-sensor algorithm set) ──
    rows = []
    for sensor in sensors:
        s_df  = df_long[df_long["sensor"] == sensor]
        algos = get_algos_for_sensor(df_long, sensor)
        for algo in algos:
            a_df = s_df[s_df["algorithm"] == algo]
            for cat in neg_cats:
                activities = CATEGORY_MAP[cat]
                for act in activities:
                    sub = a_df[a_df["PA_TYPE"] == act]
                    if len(sub) == 0:
                        continue
                    fp = (sub["total_steps"] > 0).mean() * 100
                    mean_spm = sub["steps_per_min"].mean()
                    rows.append({
                        "sensor": sensor, "algorithm": algo,
                        "category": cat,
                        "activity": act.replace("_", " "),
                        "fp_rate_pct": fp,
                        "mean_steps_per_min": round(mean_spm, 2),
                        "n_segments": len(sub),
                    })

    act_fp_df = pd.DataFrame(rows)
    act_fp_df.to_csv(os.path.join(out_dir,
                     "fig1b_fp_rates_by_activity.csv"), index=False)

    # ── One figure per sensor ──
    cat_colors = {"NoMove+NoStep": "#2E86AB", "Move+NoStep": "#E84855"}
    cat_display = {
        "NoMove+NoStep":  "No Movement & No Step",
        "Move+NoStep":    "Movement & No Step",
    }

    for sensor in sensors:
        sf = act_fp_df[act_fp_df["sensor"] == sensor]
        if sf.empty:
            continue

        algos   = get_algos_for_sensor(df_long, sensor)
        n_algos = len(algos)
        if n_algos == 0:
            continue

        # Get unique activities present, ordered by category then by
        # mean FP rate (averaged across algorithms) descending
        act_order = (
            sf.groupby(["category", "activity"])["fp_rate_pct"]
            .mean().reset_index()
            .sort_values(["category", "fp_rate_pct"],
                         ascending=[True, False])
        )
        activities = act_order["activity"].tolist()
        act_cats   = act_order.set_index("activity")["category"].to_dict()
        n_acts     = len(activities)

        if n_acts == 0:
            continue

        fig_h = max(6, n_acts * 0.45 + 2)
        fig, ax = plt.subplots(figsize=(12, fig_h))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(DARK_AX)

        bar_h   = 0.8 / n_algos
        y_pos   = np.arange(n_acts)

        for ai, algo in enumerate(algos):
            fp_vals = []
            for act in activities:
                row = sf[(sf["algorithm"] == algo) &
                         (sf["activity"] == act)]
                fp_vals.append(row["fp_rate_pct"].values[0]
                               if len(row) else 0.0)

            offset = (ai - (n_algos - 1) / 2) * bar_h
            bars = ax.barh(y_pos + offset, fp_vals, bar_h * 0.9,
                           label=algo, color=algo_color(algo),
                           alpha=0.85, edgecolor="#444", linewidth=0.5)

            for bar in bars:
                w = bar.get_width()
                if w > 0:
                    ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2,
                            f"{w:.1f}%", va="center", ha="left",
                            fontsize=7, color=TEXT_COL)

        ax.set_yticks(y_pos)
        # Colour y-tick labels by category
        ax.set_yticklabels(activities, fontsize=9)
        for i, act in enumerate(activities):
            cat = act_cats.get(act, "")
            ax.get_yticklabels()[i].set_color(cat_colors.get(cat, TEXT_COL))

        ax.invert_yaxis()
        ax.set_xlim(0, max(sf["fp_rate_pct"].max() * 1.15, 10))

        # Category colour legend patches
        patches = [mpatches.Patch(color=cat_colors[c],
                                  label=cat_display[c])
                   for c in neg_cats]
        leg1 = ax.legend(handles=patches, loc="lower right",
                         framealpha=0.9, fontsize=8,
                         facecolor=LEGEND_FACE, edgecolor=LEGEND_EDGE,
                         labelcolor=LEGEND_TEXT,
                         title="Category (label colour)",
                         title_fontsize=8)
        ax.add_artist(leg1)

        # Algorithm legend
        ax.legend(loc="upper right", framealpha=0.9, fontsize=8,
                  facecolor=LEGEND_FACE, edgecolor=LEGEND_EDGE,
                  labelcolor=LEGEND_TEXT)

        _style(ax,
               xlabel="False Positive Rate (% segments with steps > 0)",
               title=f"FP Rate by Activity (Non-Step Categories) — {sensor}")
        ax.spines[["top", "right"]].set_visible(False)

        _save(fig, os.path.join(out_dir,
              f"fig1b_fp_by_activity_{sensor}.png"))


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Pairwise scatter  (labels removed from points)
# ═════════════════════════════════════════════════════════════════════════════

def _scatter_compare(x_vals, y_vals, categories, xlabel, ylabel,
                     title, out_path, cat_patches):
    """Helper: scatter + regression + 1:1 line, used for both algo-vs-algo
    (within a sensor) and sensor-vs-sensor (within an algorithm)."""
    from scipy import stats as sp_stats

    fig, ax = plt.subplots(figsize=(7.5, 6))
    fig.patch.set_facecolor(DARK_BG)

    if len(x_vals) < 3:
        _style(ax, title=f"{title}\ninsufficient data")
        _save(fig, out_path); return

    colors = [CATEGORY_COLORS.get(c, "#888") for c in categories]
    ax.scatter(x_vals, y_vals, c=colors, s=60, alpha=0.85,
               linewidths=0.5, edgecolors="#333", zorder=3)

    slope, intercept, r_val, p_val, _ = sp_stats.linregress(
        x_vals, y_vals)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
    ax.plot(x_line, slope * x_line + intercept, color=TEXT_COL,
            linewidth=1.8, linestyle="--", alpha=0.9, zorder=4,
            label=f"y = {slope:.2f}x + {intercept:.2f}\n"
                  f"r = {r_val:.3f},  R² = {r_val**2:.3f}")
    lim = max(x_vals.max(), y_vals.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#777", linewidth=1.0,
            linestyle=":", alpha=0.7, zorder=2, label="1:1 line")
    ax.legend(framealpha=0.9, labelcolor=LEGEND_TEXT,
              facecolor=LEGEND_FACE, edgecolor=LEGEND_EDGE,
              fontsize=8, loc="upper left")

    p_str = "<0.001" if p_val < 0.001 else f"{p_val:.3f}"
    _style(ax, xlabel=xlabel, ylabel=ylabel,
           title=f"{title}\n"
                 f"r = {r_val:.3f},  p = {p_str},  "
                 f"R² = {r_val**2:.3f}  (n = {len(x_vals)} labels)")
    ax.spines[["top", "right"]].set_visible(False)

    if cat_patches:
        fig.legend(handles=cat_patches, loc="lower center",
                   ncol=min(len(cat_patches), 4), framealpha=0.9,
                   labelcolor=LEGEND_TEXT, facecolor=LEGEND_FACE,
                   edgecolor=LEGEND_EDGE, fontsize=8,
                   bbox_to_anchor=(0.5, -0.06))
    _save(fig, out_path)


def fig2_pairwise_scatter(df_long, out_dir):
    """Pairwise scatter plots restricted to comparisons that share a wear
    location:
      (a) Within each sensor: every algo-pair available at that sensor.
      (b) Between LeftWrist and RightWrist: for each algorithm present
          on both, scatter LeftWrist mean vs RightWrist mean across
          activities (this is the only cross-sensor comparison we make
          since left/right wrist are the only pair that share their
          full algorithm set)."""
    sensors = [s for s in SENSORS if s in df_long["sensor"].unique()]

    # Category legend patches (shared across all panels)
    cats_in_data = set(df_long["category"].dropna().unique())
    cat_patches = [
        mpatches.Patch(color=CATEGORY_COLORS.get(c, "#888"), label=c)
        for c in CATEGORY_ORDER if c in cats_in_data
    ]

    # ── (a) Within-sensor algo-vs-algo scatters ───────────────────────────
    for sensor in sensors:
        s_df  = df_long[df_long["sensor"] == sensor]
        algos = get_algos_for_sensor(df_long, sensor)
        if len(algos) < 2:
            print(f"    [fig2] {sensor}: only {len(algos)} algo(s), "
                  "skipping within-sensor pairs")
            continue

        # Pivot to mean spm per (PA_TYPE, algorithm) for this sensor
        label_algo_mean = (
            s_df.groupby(["PA_TYPE", "category", "algorithm"])
            ["steps_per_min"].mean().reset_index()
        )
        pivot = label_algo_mean.pivot_table(
            index=["PA_TYPE", "category"],
            columns="algorithm", values="steps_per_min"
        ).reset_index()

        pairs = [(algos[i], algos[j])
                 for i in range(len(algos))
                 for j in range(i + 1, len(algos))]

        for (a, b) in pairs:
            if a not in pivot.columns or b not in pivot.columns:
                continue
            sub = pivot.dropna(subset=[a, b])
            out_path = os.path.join(
                out_dir, f"fig2a_{sensor}_scatter_{a}_vs_{b}.png")
            _scatter_compare(
                x_vals=sub[a].values,
                y_vals=sub[b].values,
                categories=sub["category"].values,
                xlabel=f"{a}  (mean spm)",
                ylabel=f"{b}  (mean spm)",
                title=f"{sensor} — {a} vs {b}",
                out_path=out_path,
                cat_patches=cat_patches,
            )

    # ── (b) Cross-wrist (Left vs Right) scatter per algorithm ─────────────
    left, right = "LeftWrist", "RightWrist"
    if left in sensors and right in sensors:
        left_algos  = set(get_algos_for_sensor(df_long, left))
        right_algos = set(get_algos_for_sensor(df_long, right))
        shared_algos = sorted(left_algos & right_algos)

        for algo in shared_algos:
            wrist_mean = (
                df_long[(df_long["algorithm"] == algo) &
                        (df_long["sensor"].isin(WRIST_SENSORS))]
                .groupby(["PA_TYPE", "category", "sensor"])
                ["steps_per_min"].mean().reset_index()
            )
            pivot = wrist_mean.pivot_table(
                index=["PA_TYPE", "category"],
                columns="sensor", values="steps_per_min"
            ).reset_index()
            if left not in pivot.columns or right not in pivot.columns:
                continue
            sub = pivot.dropna(subset=[left, right])
            out_path = os.path.join(
                out_dir, f"fig2b_wrist_LvR_{algo}.png")
            _scatter_compare(
                x_vals=sub[left].values,
                y_vals=sub[right].values,
                categories=sub["category"].values,
                xlabel=f"LeftWrist  {algo}  (mean spm)",
                ylabel=f"RightWrist {algo}  (mean spm)",
                title=f"Left vs Right Wrist — {algo}",
                out_path=out_path,
                cat_patches=cat_patches,
            )
    else:
        print("    [fig2] Skipping wrist L-vs-R: both wrists not present")


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Step attribution stacked bar  (legend at bottom)
# ═════════════════════════════════════════════════════════════════════════════

def fig3_step_attribution(df_long, out_dir):
    sensors = [s for s in SENSORS if s in df_long["sensor"].unique()
               and get_algos_for_sensor(df_long, s)]

    agg_rows = []
    for sensor in sensors:
        s_df  = df_long[df_long["sensor"] == sensor]
        algos = get_algos_for_sensor(df_long, sensor)
        for algo in algos:
            a_df  = s_df[s_df["algorithm"] == algo]
            grand = a_df["total_steps"].sum()
            for cat in CATEGORY_ORDER:
                cat_total = a_df[a_df["category"] == cat][
                    "total_steps"].sum()
                pct = (cat_total / grand * 100) if grand > 0 else 0.0
                agg_rows.append({
                    "sensor": sensor, "algorithm": algo,
                    "category": cat, "total_steps": cat_total,
                    "pct_of_total": pct,
                })
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(os.path.join(out_dir, "fig3_step_attribution.csv"),
                  index=False)

    n_sensors = len(sensors)
    if n_sensors == 0:
        print("    [fig3] No sensor data — skipped"); return

    n_cols = min(n_sensors, 3)
    n_rows = int(np.ceil(n_sensors / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 5.5 * n_rows),
                             squeeze=False)
    fig.patch.set_facecolor(DARK_BG)

    legend_handles = []
    for si, sensor in enumerate(sensors):
        ax = axes[si // n_cols, si % n_cols]
        ax.set_facecolor(DARK_AX)
        sf     = agg_df[agg_df["sensor"] == sensor]
        algos  = get_algos_for_sensor(df_long, sensor)
        x      = np.arange(len(algos))
        bottom = np.zeros(len(algos))

        for cat in CATEGORY_ORDER:
            vals = np.array([
                sf[(sf["algorithm"] == a) & (sf["category"] == cat)]
                ["pct_of_total"].values[0]
                if len(sf[(sf["algorithm"] == a) &
                          (sf["category"] == cat)]) else 0.0
                for a in algos
            ])
            col = CATEGORY_COLORS.get(cat, "#888")
            ax.bar(x, vals, 0.6, bottom=bottom,
                   color=col, alpha=0.85, edgecolor="#444",
                   linewidth=0.6)
            # Collect legend handles only once
            if si == 0:
                legend_handles.append(
                    mpatches.Patch(color=col, alpha=0.85,
                                  label=CATEGORY_DISPLAY.get(cat, cat)))
            for i, (v, b) in enumerate(zip(vals, bottom)):
                if v > 3:
                    ax.text(i, b + v / 2, f"{v:.1f}%",
                            ha="center", va="center",
                            fontsize=8, color="#111", fontweight="bold")
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(algos, color=TEXT_COL, fontsize=10)
        ax.set_ylim(0, 105)
        _style(ax, ylabel="% of Total Detected Steps",
               title=f"Step Attribution by Category — {sensor}")
        ax.spines[["top", "right"]].set_visible(False)

    # Hide empty subplot cells
    for k in range(n_sensors, n_rows * n_cols):
        axes[k // n_cols, k % n_cols].axis("off")

    # ── Shared legend at the bottom ──
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(CATEGORY_ORDER), framealpha=0.9,
               labelcolor=LEGEND_TEXT, facecolor=LEGEND_FACE,
               edgecolor=LEGEND_EDGE, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))

    _save(fig, os.path.join(out_dir, "fig3_step_attribution.png"))


# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Sensitivity vs Specificity scatter
#  FIGURE 5 — Pseudo-ROC threshold sweep
#           Positive = Move+Step
#           Negative = NoMove+NoStep ∪ Move+NoStep
#
#  NOTE: The 45-degree diagonal from (0,0)→(1,1) has been REMOVED.
#        In specificity-vs-sensitivity space, (0,0) is NOT the ideal
#        nor random baseline.  A random classifier would follow the
#        line from (1,0) to (0,1)  (since specificity = 1 − FPR).
# ═════════════════════════════════════════════════════════════════════════════

def fig4_sensitivity_specificity(df_long, out_dir):
    sensors = [s for s in SENSORS if s in df_long["sensor"].unique()
               and get_algos_for_sensor(df_long, s)]

    binary_df = df_long[
        df_long["category"].isin(
            ["Move+Step", "NoMove+NoStep", "Move+NoStep"])
    ].copy()

    ss_rows = []
    for sensor in sensors:
        s_df  = binary_df[binary_df["sensor"] == sensor]
        algos = get_algos_for_sensor(df_long, sensor)
        for algo in algos:
            a_df = s_df[s_df["algorithm"] == algo]
            pos = a_df[a_df["category"] == "Move+Step"]
            neg = a_df[a_df["category"].isin(
                ["NoMove+NoStep", "Move+NoStep"])]
            sens = ((pos["total_steps"] > 0).mean()
                    if len(pos) else np.nan)
            spec = ((neg["total_steps"] == 0).mean()
                    if len(neg) else np.nan)
            ss_rows.append({
                "sensor": sensor, "algorithm": algo,
                "sensitivity": sens, "specificity": spec,
                "n_pos": len(pos), "n_neg": len(neg),
            })

    ss_df = pd.DataFrame(ss_rows)
    ss_df.to_csv(os.path.join(out_dir,
                               "fig4_sensitivity_specificity.csv"),
                 index=False)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)
    marker_pool = ["o", "s", "D", "^", "v", "P", "X", "*"]
    marker_map  = {s: marker_pool[i % len(marker_pool)]
                   for i, s in enumerate(sensors)}
    all_algos   = sorted({a for s in sensors
                          for a in get_algos_for_sensor(df_long, s)})

    for _, row in ss_df.iterrows():
        col = algo_color(row["algorithm"])
        mkr = marker_map.get(row["sensor"], "o")
        ax.scatter(row["specificity"], row["sensitivity"],
                   c=[col], s=140, marker=mkr, alpha=0.9,
                   edgecolors="#333", linewidths=0.8, zorder=3)
        ax.annotate(
            f"{row['algorithm']}\n({row['sensor']})",
            (row["specificity"], row["sensitivity"]),
            textcoords="offset points", xytext=(8, -5),
            fontsize=7.5, color=TEXT_COL, alpha=0.85)

    # ── Perfect point only; no misleading diagonal ──
    ax.scatter(1, 1, marker="*", s=200, color="#FFD700",
               edgecolors="#333", linewidths=0.8, zorder=4,
               label="Perfect")

    handles = (
        [mpatches.Patch(color=algo_color(a), label=a) for a in all_algos]
        + [plt.Line2D([0], [0], marker=marker_map[s], color="#888",
                      markersize=8, linestyle="None", label=s)
           for s in sensors]
    )
    ax.legend(handles=handles, framealpha=0.9, labelcolor=LEGEND_TEXT,
              facecolor=LEGEND_FACE, edgecolor=LEGEND_EDGE,
              fontsize=8, loc="lower left")
    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.02, 1.08)
    _style(ax,
           xlabel="Specificity\n(NoMove+NoStep ∪ Move+NoStep "
                  "correctly rejected)",
           ylabel="Sensitivity\n(Move+Step correctly detected)",
           title="Sensitivity vs Specificity\n"
                 "(Pos = Move+Step,  "
                 "Neg = NoMove+NoStep ∪ Move+NoStep)")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, os.path.join(out_dir,
                            "fig4_sensitivity_specificity.png"))


def fig5_pseudo_roc(df_long, out_dir):
    sensors = [s for s in SENSORS if s in df_long["sensor"].unique()
               and get_algos_for_sensor(df_long, s)]
    if not sensors:
        print("    [fig5] No sensor data — skipped"); return

    binary_df = df_long[
        df_long["category"].isin(
            ["Move+Step", "NoMove+NoStep", "Move+NoStep"])
    ].copy()

    thresholds = [0, 0.5, 1, 2, 5, 10, 20, 50]

    n_cols = min(len(sensors), 3)
    n_rows = int(np.ceil(len(sensors) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6.5 * n_cols, 5.5 * n_rows),
                             squeeze=False)
    fig.patch.set_facecolor(DARK_BG)

    roc_rows = []
    for si, sensor in enumerate(sensors):
        ax = axes[si // n_cols, si % n_cols]
        ax.set_facecolor(DARK_AX)
        s_df  = binary_df[binary_df["sensor"] == sensor]
        algos = get_algos_for_sensor(df_long, sensor)

        for algo in algos:
            a_df = s_df[s_df["algorithm"] == algo]
            pos = a_df[a_df["category"] == "Move+Step"]
            neg = a_df[a_df["category"].isin(
                ["NoMove+NoStep", "Move+NoStep"])]

            sens_list, spec_list = [], []
            for thr in thresholds:
                sn = ((pos["steps_per_min"] > thr).mean()
                      if len(pos) else np.nan)
                sp = ((neg["steps_per_min"] <= thr).mean()
                      if len(neg) else np.nan)
                sens_list.append(sn)
                spec_list.append(sp)
                roc_rows.append({
                    "sensor": sensor, "algorithm": algo,
                    "threshold_spm": thr,
                    "sensitivity": sn, "specificity": sp,
                })

            col = algo_color(algo)
            ax.plot(spec_list, sens_list, marker="o", markersize=5,
                    color=col, linewidth=2, alpha=0.85, label=algo)
            for thr, sp, sn in zip(thresholds, spec_list, sens_list):
                if not np.isnan(sp) and not np.isnan(sn):
                    ax.annotate(f"{thr}", (sp, sn),
                                textcoords="offset points",
                                xytext=(4, 4), fontsize=6,
                                color=col, alpha=0.7)

        # ── Perfect point only; no misleading diagonal ──
        ax.scatter(1, 1, marker="*", s=150, color="#FFD700",
                   edgecolors="#333", linewidths=0.8, zorder=4)
        ax.set_xlim(-0.02, 1.08)
        ax.set_ylim(-0.02, 1.08)
        ax.legend(framealpha=0.9, labelcolor=LEGEND_TEXT,
                  facecolor=LEGEND_FACE, edgecolor=LEGEND_EDGE,
                  fontsize=8, loc="lower left")
        _style(ax, xlabel="Specificity", ylabel="Sensitivity",
               title=f"Pseudo-ROC — {sensor}\n"
                     f"(numbers = spm threshold)")
        ax.spines[["top", "right"]].set_visible(False)

    # Hide empty subplot cells
    for k in range(len(sensors), n_rows * n_cols):
        axes[k // n_cols, k % n_cols].axis("off")

    pd.DataFrame(roc_rows).to_csv(
        os.path.join(out_dir, "fig5_pseudo_roc_data.csv"), index=False)
    _save(fig, os.path.join(out_dir, "fig5_pseudo_roc.png"))


# ═════════════════════════════════════════════════════════════════════════════
#  PARALLEL PROCESSING & MAIN
# ═════════════════════════════════════════════════════════════════════════════

_print_lock = None

def _pool_init(lock):
    global _print_lock
    _print_lock = lock


def _process_ds_worker(args):
    ds_root, ds_id, algorithms, label_root = args
    records, log_lines = process_ds(ds_root, ds_id, algorithms,
                                    label_root)
    block = "\n".join(log_lines)
    if _print_lock is not None:
        with _print_lock:
            print(block, flush=True)
    else:
        print(block, flush=True)
    return records


def process_ds(ds_root, ds_id, algorithms, label_root=None):
    log = []
    log.append(f"\n  ┌─ {ds_id}")
    label_path = find_label_file(ds_root, ds_id, label_root)
    if label_path is None:
        log.append(f"  └─ [{ds_id}] [SKIP] No label file found.")
        return [], log
    log.append(f"  │  [{ds_id}] Labels: {label_path}")
    labels = load_labels(label_path)
    log.append(f"  │  [{ds_id}] {len(labels)} segments after filtering")

    records = []
    for algo in algorithms:
        sensor_files = find_steps_files(ds_root, ds_id, algo)
        if not sensor_files:
            log.append(
                f"  │  [{ds_id}] [SKIP] No steps files for {algo}")
            continue
        seen = {}
        for sensor, fpath in sensor_files:
            if sensor not in seen:
                seen[sensor] = fpath
        for sensor, fpath in seen.items():
            try:
                series = load_steps_series(fpath)
                recs   = aggregate_segments(
                    labels, series, ds_id, algo, sensor)
                records.extend(recs)
                log.append(
                    f"  │  [{ds_id}] ✓ {algo:12s}  {sensor:22s}  "
                    f"{len(recs)} segs  sum={series.sum():.0f} steps")
            except Exception as e:
                log.append(
                    f"  │  [{ds_id}] [ERROR] {algo}/{sensor}: {e}")
    log.append(f"  └─ [{ds_id}] {len(records)} records")
    return records, log


def main():
    parser = argparse.ArgumentParser(
        description="Step algorithm comparison — paper final")
    parser.add_argument("--root",    default=DEFAULT_ROOT)
    parser.add_argument("--labels",  default=DEFAULT_LABELS)
    parser.add_argument("--ds",      nargs="*", default=None)
    parser.add_argument("--algos",   nargs="*", default=ALGORITHMS)
    parser.add_argument("--out",     default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("╔══════════════════════════════════════════════════════╗")
    print("║  STEP ALGORITHM COMPARISON — PAPER FINAL             ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Root   : {args.root}")
    print(f"  Labels : {args.labels}")
    print(f"  Algos  : {args.algos}")
    print(f"  Workers: {args.workers}")
    print(f"  Out    : {args.out}")
    print(f"  Min segment: {MIN_SEGMENT_SEC}s")

    all_ds = discover_ds_folders(args.root)
    ds_folders = ([d for d in all_ds if os.path.basename(d) in args.ds]
                  if args.ds else all_ds)
    print(f"\n  {len(ds_folders)} DS folder(s): "
          f"{[os.path.basename(d) for d in ds_folders]}")

    n_workers = min(len(ds_folders), args.workers)
    worker_args = [
        (ds_root, os.path.basename(ds_root), args.algos, args.labels)
        for ds_root in ds_folders
    ]

    all_records = []
    if n_workers > 1:
        print(f"\n  Using {n_workers} parallel workers...")
        lock = mp.Manager().Lock()
        with mp.Pool(processes=n_workers,
                     initializer=_pool_init,
                     initargs=(lock,)) as pool:
            results = pool.map(_process_ds_worker, worker_args)
        for recs in results:
            all_records.extend(recs)
    else:
        for wa in worker_args:
            all_records.extend(_process_ds_worker(wa))

    if not all_records:
        print("\nERROR: No data collected."); return

    df_long = pd.DataFrame(all_records)
    df_long = df_long[df_long["sensor"].isin(SENSORS)].copy()

    # Drop (sensor, algorithm) combinations that aren't in our wear-location
    # whitelist — keeps comparisons restricted to algos validated for each
    # location (e.g. oak only on hip/thigh).
    valid_pairs = set()
    for sensor, algos in SENSOR_ALGO_MAP.items():
        for algo in algos:
            valid_pairs.add((sensor, algo))
    df_long = df_long[
        df_long.apply(
            lambda r: (r["sensor"], r["algorithm"]) in valid_pairs,
            axis=1)
    ].copy()

    df_long.to_csv(os.path.join(args.out, "segments_long.csv"),
                   index=False)
    print(f"\n  {len(df_long)} total segment records "
          f"(sensors: {SENSORS})")

    print("\n  Per-sensor algorithm coverage:")
    for sensor in SENSORS:
        if sensor in df_long["sensor"].unique():
            present = get_algos_for_sensor(df_long, sensor)
            print(f"    {sensor:12s}: {present}")
        else:
            print(f"    {sensor:12s}: (no data)")

    for cat in CATEGORY_ORDER:
        n = (df_long["category"] == cat).sum()
        print(f"    {cat:20s}: {n:>7d} segments")

    print("\n  ══ Tables ══")
    print("\n  [Table 1] Activity-level step rates ...")
    table1_activity_step_rates(df_long, args.out)
    print("\n  [Table 2] Category-level summary ...")
    table2_category_summary(df_long, args.out)

    print("\n  ══ Figures ══")
    print("\n  [Figure 1] FP rates ...")
    fig1_fp_rates(df_long, args.out)
    print("\n  [Figure 1b] FP rates by activity (non-step breakdown) ...")
    fig1b_fp_rates_by_activity(df_long, args.out)
    print("\n  [Figure 2] Pairwise scatter ...")
    fig2_pairwise_scatter(df_long, args.out)
    print("\n  [Figure 3] Step attribution ...")
    fig3_step_attribution(df_long, args.out)
    print("\n  [Figure 4] Sensitivity vs specificity ...")
    fig4_sensitivity_specificity(df_long, args.out)
    print("\n  [Figure 5] Pseudo-ROC ...")
    fig5_pseudo_roc(df_long, args.out)

    print(f"\n  ✓ Done! Output in: {args.out}")
    print()
    for f in sorted(os.listdir(args.out)):
        kb = os.path.getsize(os.path.join(args.out, f)) / 1024
        print(f"    {f:<60} {kb:6.1f} KB")


if __name__ == "__main__":
    main()