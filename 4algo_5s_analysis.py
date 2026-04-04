#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║         STEP ALGORITHM COMPARISON ANALYSIS  —  Cluster Version      ║
║                    v2: Step-Based Ground Truth Plots                 ║
╠══════════════════════════════════════════════════════════════════════╣
║  USAGE                                                               ║
║    python step_analysis.py                                           ║
║    python step_analysis.py --root /scratch/wang.yichen8/PAAWS_results║
║    python step_analysis.py --ds DS_10 DS_12                          ║
║    python step_analysis.py --out ./my_results                        ║
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
import matplotlib.dates
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ROOT   = "/scratch/wang.yichen8/PAAWS_results"
DEFAULT_OUTPUT = "/scratch/wang.yichen8/step_analysis_output"
DEFAULT_LABELS = "/scratch/wang.yichen8/PAAWS_FreeLiving"
ALGORITHMS     = ["adept", "oak", "oxford", "verisense"]
SENSORS        = ["RightWaist", "RightWrist"]

LABEL_PATTERNS = [
    "{ds_root}/labels/{ds_id}-Free-label.csv",
    "{ds_root}/labels/{ds_id}-Free-label.csv.gz",
    "{ds_root}/{ds_id}-Free-label.csv",
    "{ds_root}/{ds_id}-Free-label.csv.gz",
]

# ── Movement-based category definitions (original) ──────────────────────────

NON_MOVEMENT_ACTIVITIES = [
    "Kneeling_Still", "Sitting_Still", "Standing_Still", "Lying_Still",
]
MOVEMENT_ACTIVITIES = [
    "Walking", "Walking_Fast", "Walking_Slow", "Walking_Treadmill",
    "Walking_Up_Stairs", "Walking_Down_Stairs",
    "Cycling_Active_Pedaling_Regular_Bicycle",
    "Cycling_Active_Pedaling_Stationary_Bike",
    "Doing_Resistance_Training_Free_Weights",
    "Doing_Resistance_Training_Other",
    "Playing_Frisbee", "Running_Non-Treadmill", "Running_Treadmill",
    "Puttering_Around",
]
MAYBE_MOVING_ACTIVITIES = [
    "Kneeling_With_Movement",
    "Sitting_With_Movement", "Lying_With_Movement",
    "Standing_With_Movement", "Applying_Makeup", "Bathing",
    "Blowdrying_Hair", "Showering",
    "Washing_Face", "Brushing_Teeth", "Brushing/Combing/Tying_Hair",
    "Organizing_Shelf/Cabinet", "Folding_Clothes", "Ironing",
    "Flossing_Teeth", "Washing_Hands", "Putting_Clothes_Away",
    "Loading/Unloading_Washing_Machine/Dryer",
    "Watering_Plants", "Dusting", "Dry_Mopping", "Sweeping",
    "Vacuuming", "Wet_Mopping",
]
EXCLUDE_ACTIVITIES = [
    "PA_Type_Video_Unavailable/Indecipherable",
    "Posture_Video_Unavailable/Indecipherable",
    "Synchronizing_Sensors", "PA_Type_Too_Complex",
    "PA_Type_Other", "PA_Type_Unlabeled",
]

# ── Step-based category definitions (NEW) ───────────────────────────────────
# Categorised by whether a human judges the feet to be stepping.

NO_STEP_ACTIVITIES = [
    "Kneeling_Still", "Sitting_Still", "Standing_Still", "Lying_Still",
    "Kneeling_With_Movement",
    "Sitting_With_Movement", "Lying_With_Movement",
    "Applying_Makeup", "Bathing", "Blowdrying_Hair",
    "Washing_Face", "Brushing_Teeth", "Brushing/Combing/Tying_Hair",
    "Organizing_Shelf/Cabinet", "Folding_Clothes", "Ironing",
    "Flossing_Teeth", "Washing_Hands",
    "Loading/Unloading_Washing_Machine/Dryer",
]
STEP_ACTIVITIES = [
    "Walking", "Walking_Fast", "Walking_Slow", "Walking_Treadmill",
    "Walking_Up_Stairs", "Walking_Down_Stairs",
    "Cycling_Active_Pedaling_Regular_Bicycle",
    "Cycling_Active_Pedaling_Stationary_Bike",
    "Doing_Resistance_Training_Free_Weights",
    "Doing_Resistance_Training_Other",
    "Playing_Frisbee", "Running_Non-Treadmill", "Running_Treadmill",
    "Puttering_Around",
]
MAYBE_STEP_ACTIVITIES = [
    "Standing_With_Movement", "Showering",
    "Putting_Clothes_Away",
    "Watering_Plants", "Dusting", "Dry_Mopping", "Sweeping",
    "Vacuuming", "Wet_Mopping",
]

# ── Activities that are Maybe-Moving (body moves) but No_Step (feet don't) ──
# These are the *discrimination-critical* activities.
RECLASSIFIED_ACTIVITIES = sorted(
    set(MAYBE_MOVING_ACTIVITIES) & set(NO_STEP_ACTIVITIES)
)

# ── Category ordering and maps ──────────────────────────────────────────────

CATEGORY_ORDER = ["Non-Movement", "Movement", "Maybe-Moving"]
CATEGORY_ACTIVITY_MAP = {
    "Non-Movement": NON_MOVEMENT_ACTIVITIES,
    "Movement":     MOVEMENT_ACTIVITIES,
    "Maybe-Moving": MAYBE_MOVING_ACTIVITIES,
}

STEP_CATEGORY_ORDER = ["No_Step", "Step", "Maybe_Step"]
STEP_CATEGORY_MAP = {
    "No_Step":    NO_STEP_ACTIVITIES,
    "Step":       STEP_ACTIVITIES,
    "Maybe_Step": MAYBE_STEP_ACTIVITIES,
}

MIN_SEGMENT_SEC = 5   # ← changed from 10 to 5

# ─────────────────────────────────────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────────────────────────────────────

ALGO_COLORS  = {"oxford": "#2E86AB", "oak": "#E84855",
                "verisense": "#57CC99", "adept": "#F4A261"}
EXTRA_COLORS = ["#9B5DE5", "#F15BB5", "#FEE440", "#00BBF9"]

DARK_BG    = "#FFFFFF"
DARK_AX    = "#FFFFFF"
GRID_COL   = "#E0E0E0"
TEXT_COL   = "#111111"
SPINE_COL  = "#BDBDBD"
LEGEND_FACE = "#FFFFFF"
LEGEND_EDGE = "#CCCCCC"
LEGEND_TEXT = TEXT_COL

CATEGORY_COLORS = {
    "Non-Movement": "#57CC99",
    "Movement":     "#2E86AB",
    "Maybe-Moving": "#F4A261",
    "Other":        "#888888",
}

STEP_CATEGORY_COLORS = {
    "No_Step":    "#E84855",   # red — false positives here are bad
    "Step":       "#57CC99",   # green — true positives here are good
    "Maybe_Step": "#F4A261",   # orange — ambiguous
}

def algo_color(name):
    for key, col in ALGO_COLORS.items():
        if key in name.lower():
            return col
    return EXTRA_COLORS[hash(name) % len(EXTRA_COLORS)]

# ─────────────────────────────────────────────────────────────────────────────
# ACTIVITY → CATEGORY MAPPING
# ─────────────────────────────────────────────────────────────────────────────

def activity_category(pa_type):
    """Return the movement-based category name."""
    for cat, acts in CATEGORY_ACTIVITY_MAP.items():
        if pa_type in acts:
            return cat
    return "Other"


def step_category(pa_type):
    """Return the step-based category name (No_Step / Step / Maybe_Step)."""
    for cat, acts in STEP_CATEGORY_MAP.items():
        if pa_type in acts:
            return cat
    return "Other"

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
        candidates = [
            os.path.join(label_root, ds_id, "label", f"{ds_id}-Free-label.csv"),
            os.path.join(label_root, ds_id, "label", f"{ds_id}-Free-label.csv.gz"),
            os.path.join(label_root, ds_id, f"{ds_id}-Free-label.csv"),
            os.path.join(label_root, ds_id, f"{ds_id}-Free-label.csv.gz"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
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
        pattern = os.path.join(ds_root, f"*-Free-*_{algorithm}_steps.{ext}")
        matches = glob.glob(pattern)
        hits.extend(matches)
    if not hits:
        return []
    seen = {}
    for h in hits:
        basename = os.path.basename(h)
        stem = re.sub(r"\.csv(\.gz)?$", "", basename)
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
    df["duration_sec"] = (df["STOP_TIME"] - df["START_TIME"]).dt.total_seconds()
    df = df[df["duration_sec"] >= MIN_SEGMENT_SEC].copy()
    df = df[~df["PA_TYPE"].isin(EXCLUDE_ACTIVITIES)].copy()
    return df.dropna(subset=["PA_TYPE"]).reset_index(drop=True)


def load_steps_series(path):
    df = read_csv_auto(path)
    df.columns = [c.strip().lower() for c in df.columns]
    time_col = next((c for c in df.columns if "time" in c or "date" in c),
                    df.columns[0])
    step_col = next((c for c in df.columns if "step" in c), df.columns[1])
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

    lo_arr = np.searchsorted(idx, starts, side="left")
    hi_arr = np.searchsorted(idx, stops,  side="left")

    records = []
    for i in range(len(labels_df)):
        total = float(sv[lo_arr[i]:hi_arr[i]].sum())
        dur   = durs[i]
        spm   = (total / dur * 60) if dur > 0 else 0.0
        records.append({
            "ds_id":             ds_id,
            "algorithm":         algorithm,
            "sensor":            sensor,
            "PA_TYPE":           pa_types[i],
            "activity_category": activity_category(pa_types[i]),
            "step_category":     step_category(pa_types[i]),
            "duration_sec":      dur,
            "total_steps":       total,
            "steps_per_min":     spm,
        })
    return records

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLES
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(df_long):
    grp = df_long.groupby(
        ["ds_id", "sensor", "algorithm", "PA_TYPE",
         "activity_category", "step_category"])
    return grp["steps_per_min"].agg(
        mean_spm="mean", std_spm="std", median_spm="median",
        n_segments="count"
    ).reset_index()


def build_ranking(df_long):
    rows = []
    for (ds_id, sensor, algo), sub in df_long.groupby(
            ["ds_id", "sensor", "algorithm"]):
        w  = sub[sub["activity_category"] == "Movement"]["steps_per_min"]
        nm = sub[sub["activity_category"] == "Non-Movement"]
        rows.append({
            "ds_id": ds_id, "sensor": sensor, "algorithm": algo,
            "movement_mean_spm":   round(w.mean(), 2) if len(w) else None,
            "movement_cv_pct":     round(w.std() / w.mean() * 100, 1)
                                   if len(w) > 1 and w.mean() > 0 else None,
            "nonmovement_mean_spm": round(nm["steps_per_min"].mean(), 2)
                                    if len(nm) else None,
            "fp_rate_pct":        round((nm["total_steps"] > 0).mean() * 100, 1)
                                  if len(nm) else None,
            "n_movement_segments":    int(
                (sub["activity_category"] == "Movement").sum()),
            "n_nonmovement_segments": int(
                (sub["activity_category"] == "Non-Movement").sum()),
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _style(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(DARK_AX)
    ax.tick_params(colors=TEXT_COL)
    for sp in ax.spines.values():
        sp.set_color(SPINE_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.5, linestyle="--", axis="both")
    ax.set_axisbelow(True)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT_COL, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=10)
    if title:  ax.set_title(title,   color=TEXT_COL, fontsize=11,
                            fontweight="bold", pad=10)

def _save(fig, path):
    plt.tight_layout()
    plt.savefig(path, dpi=80, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"    Saved: {path}")

def _legend(ax, algos):
    ax.legend(
        handles=[mpatches.Patch(color=algo_color(a), label=a) for a in algos],
        framealpha=0.9, labelcolor=LEGEND_TEXT, facecolor=LEGEND_FACE,
        edgecolor=LEGEND_EDGE, fontsize=9)


def _ordered_activities_by_category(pa_types_present):
    ordered = []
    boundaries = []
    for cat in CATEGORY_ORDER:
        cat_acts = CATEGORY_ACTIVITY_MAP.get(cat, [])
        present  = [a for a in cat_acts if a in pa_types_present]
        if present:
            start = len(ordered)
            ordered.extend(present)
            boundaries.append((cat, start, len(ordered) - 1))
    return ordered, boundaries


# ═════════════════════════════════════════════════════════════════════════════
#  ORIGINAL PLOTS 1–5  (unchanged except aggregate_segments adds step_category)
# ═════════════════════════════════════════════════════════════════════════════

def plot1_label_boxplot_by_category(df_long, out_dir):
    algos = sorted(df_long["algorithm"].unique())
    pa_types_present = set(df_long["PA_TYPE"].unique())
    activities, boundaries = _ordered_activities_by_category(pa_types_present)
    if not activities:
        print("  [SKIP plot1] No activities found"); return

    MAX_FIG_HEIGHT = 400

    for algo in algos:
        sub = df_long[df_long["algorithm"] == algo]
        data = [sub[sub["PA_TYPE"] == a]["steps_per_min"].dropna().values
                for a in activities]
        n_acts = len(activities)
        chunk_size = max(1, int(MAX_FIG_HEIGHT / max(0.45 + 3 / max(n_acts, 1), 0.1)))
        chunks = [activities[i:i+chunk_size]
                  for i in range(0, n_acts, chunk_size)]

        for chunk_idx, act_chunk in enumerate(chunks):
            chunk_data = [data[activities.index(a)] for a in act_chunk]
            n_chunk = len(act_chunk)
            fig_h = min(max(8, n_chunk * 0.45 + 3), MAX_FIG_HEIGHT)
            fig, ax = plt.subplots(figsize=(13, fig_h))
            fig.patch.set_facecolor(DARK_BG)
            y_pos = np.arange(n_chunk)

            for i, (act, d) in enumerate(zip(act_chunk, chunk_data)):
                cat = activity_category(act)
                col = CATEGORY_COLORS.get(cat, "#888888")
                if len(d) == 0:
                    ax.plot(0, i, marker="|", color="#555", markersize=8)
                    continue
                bp = ax.boxplot(
                    d, positions=[i], vert=False, widths=0.6,
                    patch_artist=True, notch=False,
                    medianprops={"color": TEXT_COL, "linewidth": 2},
                    whiskerprops={"color": "#aaa"},
                    capprops={"color": "#aaa"},
                    flierprops={"marker": "o", "markersize": 2.5,
                                "markerfacecolor": "#666",
                                "linestyle": "none"},
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(col); patch.set_alpha(0.78)

            for cat_name, _, _ in boundaries:
                col = CATEGORY_COLORS.get(cat_name, "#888888")
                local = [i for i, a in enumerate(act_chunk)
                         if activity_category(a) == cat_name]
                if not local: continue
                if local[0] > 0:
                    ax.axhline(local[0] - 0.5, color="#555", linewidth=1.0,
                               linestyle="--", alpha=0.7)
                mid = (local[0] + local[-1]) / 2
                ax.text(1, mid, f"  {cat_name}", color=col, fontsize=8,
                        fontweight="bold", va="center", ha="left",
                        transform=ax.get_yaxis_transform())

            ax.set_yticks(y_pos)
            ax.set_yticklabels([a.replace("_", " ") for a in act_chunk],
                               fontsize=8, color=TEXT_COL)
            chunk_label = (f" (part {chunk_idx+1}/{len(chunks)})"
                           if len(chunks) > 1 else "")
            _style(ax, xlabel="Steps / Min",
                   title=f"Steps/Min Distribution per Activity — {algo}"
                         f"{chunk_label}")
            ax.spines[["top", "right"]].set_visible(False)
            legend_handles = [
                mpatches.Patch(color=CATEGORY_COLORS.get(c, "#888"), label=c)
                for c, _, _ in boundaries
            ]
            ax.legend(handles=legend_handles, framealpha=0.9,
                      labelcolor=LEGEND_TEXT, facecolor=LEGEND_FACE,
                      edgecolor=LEGEND_EDGE, fontsize=8, loc="lower right")
            suffix = f"_part{chunk_idx+1}" if len(chunks) > 1 else ""
            _save(fig, os.path.join(
                out_dir, f"plot1_{algo}_activity_boxplot{suffix}.png"))

    df_long[["PA_TYPE", "activity_category", "step_category", "algorithm",
             "sensor", "ds_id", "steps_per_min"]].to_csv(
        os.path.join(out_dir, "plot1_segments_long.csv"), index=False)


def plot2_fp_heatmap(df_long, out_dir):
    algos   = sorted(df_long["algorithm"].unique())
    sensors = sorted(df_long["sensor"].unique())
    nm = df_long[df_long["activity_category"] == "Non-Movement"].copy()
    if nm.empty:
        print("  [SKIP plot2] No Non-Movement segments"); return

    nm["detected"] = (nm["total_steps"] > 0).astype(float)
    fp = (nm.groupby(["sensor", "algorithm"])["detected"]
            .mean().mul(100).reset_index(name="fp_pct"))
    mat = pd.DataFrame(index=sensors, columns=algos, dtype=float)
    for _, row in fp.iterrows():
        mat.loc[row["sensor"], row["algorithm"]] = row["fp_pct"]
    best = mat.idxmin(axis=1)

    fig, ax = plt.subplots(
        figsize=(max(6, len(algos)*2+2), max(5, len(sensors)*0.9+2)))
    fig.patch.set_facecolor(DARK_BG)
    cmap = LinearSegmentedColormap.from_list(
        "fp", ["#57CC99", "#F4A261", "#E84855"])
    mv = mat.values.astype(float)
    im = ax.imshow(mv, aspect="auto", cmap=cmap, vmin=0, vmax=100,
                   interpolation="nearest")
    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels(algos, color=TEXT_COL, fontsize=10)
    ax.set_yticks(range(len(sensors)))
    ax.set_yticklabels(sensors, color=TEXT_COL, fontsize=9)
    for r, sensor in enumerate(sensors):
        for c, algo in enumerate(algos):
            v = mv[r, c]
            if not np.isnan(v):
                is_best = (algo == best[sensor])
                ax.text(c, r, f"{v:.1f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color=("#FFF" if v > 50 else "#111"))
                if is_best:
                    ax.add_patch(plt.Rectangle(
                        (c-0.48, r-0.48), 0.96, 0.96,
                        fill=False, edgecolor="white", linewidth=2.5))
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("False Positive Rate (%)", color=TEXT_COL, fontsize=9)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    ax.set_title("FP Rate per Sensor × Algorithm\n"
                 "(Non-Movement activities as standard)",
                 color=TEXT_COL, fontsize=11, fontweight="bold", pad=10)
    for sp in ax.spines.values(): sp.set_color(SPINE_COL)
    _save(fig, os.path.join(out_dir, "plot2_sensor_algorithm_fp_heatmap.png"))
    mat_out = mat.copy().round(1)
    mat_out["best_algorithm"] = best
    mat_out.index.name = "sensor"
    mat_out.to_csv(os.path.join(out_dir, "plot2_sensor_fp_rates.csv"))


def plot3_allDS_per_algorithm(df_long, out_dir):
    algos   = sorted(df_long["algorithm"].unique())
    sensors = sorted(df_long["sensor"].unique())
    if not algos:
        print("  [SKIP plot3] No algorithms"); return
    df_long = df_long.copy()
    df_long["activity_category"] = df_long["PA_TYPE"].apply(activity_category)
    df_long["step_category"]     = df_long["PA_TYPE"].apply(step_category)

    mov_cats_present  = [c for c in CATEGORY_ORDER
                         if c in df_long["activity_category"].unique()]
    step_cats_present = [c for c in STEP_CATEGORY_ORDER
                         if c in df_long["step_category"].unique()]

    if not mov_cats_present and not step_cats_present:
        print("  [SKIP plot3] No categories"); return

    # Combined label list: movement cats, then a gap, then step cats
    all_labels = mov_cats_present + [""] + step_cats_present
    n_total    = len(all_labels)
    gap_idx    = len(mov_cats_present)  # index of the spacer

    for algo in algos:
        for sensor in sensors:
            sub = df_long[(df_long["algorithm"] == algo) &
                          (df_long["sensor"] == sensor)]

            # Build data arrays — one per position (None for the gap)
            data_arrays = []
            color_list  = []
            for cat in mov_cats_present:
                d = sub[sub["activity_category"] == cat][
                    "steps_per_min"].dropna().values
                data_arrays.append(d)
                color_list.append(CATEGORY_COLORS.get(cat, "#888"))
            data_arrays.append(None)   # gap
            color_list.append(None)
            for cat in step_cats_present:
                d = sub[sub["step_category"] == cat][
                    "steps_per_min"].dropna().values
                data_arrays.append(d)
                color_list.append(STEP_CATEGORY_COLORS.get(cat, "#888"))

            fig, ax = plt.subplots(
                figsize=(max(12, n_total * 1.6 + 2), 6))
            fig.patch.set_facecolor(DARK_BG)
            x_pos = np.arange(n_total)

            for i, (label, d, col) in enumerate(
                    zip(all_labels, data_arrays, color_list)):
                if d is None:
                    continue  # spacer
                if len(d) == 0:
                    ax.text(i, 0, "N/A", color="#555", ha="center",
                            va="bottom", fontsize=8)
                    continue
                bp = ax.boxplot(
                    d, positions=[i], widths=0.55, patch_artist=True,
                    notch=False,
                    medianprops={"color": TEXT_COL, "linewidth": 2.5},
                    whiskerprops={"color": "#aaa"},
                    capprops={"color": "#aaa"},
                    flierprops={"marker": "o", "markersize": 2.5,
                                "markerfacecolor": "#555",
                                "linestyle": "none"},
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(col); patch.set_alpha(0.80)
                med = np.median(d)
                ax.text(i, np.percentile(d, 75) + 0.5, f"{med:.1f}",
                        color=TEXT_COL, ha="center", va="bottom",
                        fontsize=7)

            # Vertical divider between the two classification systems
            ax.axvline(gap_idx, color="#999", linewidth=1.5,
                       linestyle="--", alpha=0.6)

            # Group headers
            mov_mid  = (0 + len(mov_cats_present) - 1) / 2
            step_mid = (gap_idx + 1 + n_total - 1) / 2
            ylim_top = ax.get_ylim()[1]
            ax.text(mov_mid, ylim_top * 0.97,
                    "Movement-Based Categories",
                    ha="center", va="top", fontsize=9,
                    color="#555", fontstyle="italic", fontweight="bold")
            ax.text(step_mid, ylim_top * 0.97,
                    "Step-Based Categories",
                    ha="center", va="top", fontsize=9,
                    color="#555", fontstyle="italic", fontweight="bold")

            # X-axis labels
            display_labels = []
            for lbl in all_labels:
                if lbl == "":
                    display_labels.append("")
                elif len(lbl) > 8:
                    display_labels.append(lbl.replace("-", "\n")
                                              .replace("_", "\n"))
                else:
                    display_labels.append(lbl)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(display_labels, color=TEXT_COL, fontsize=9)
            ax.axhline(0, color="#555", linewidth=0.8, linestyle="--")

            _style(ax, ylabel="Steps / Min",
                   title=f"All DS — Steps/Min by Category\n"
                         f"Algorithm: {algo}  |  Sensor: {sensor}  "
                         f"(left = movement-based, right = step-based)")
            ax.spines[["top", "right"]].set_visible(False)

            # n-count annotation below each box
            for i, (label, d) in enumerate(
                    zip(all_labels, data_arrays)):
                if d is None or len(d) == 0:
                    continue
                ax.text(i, ax.get_ylim()[0], f"n={len(d)}",
                        color="#888", ha="center", va="top", fontsize=7)

            # Combined legend
            legend_handles = (
                [mpatches.Patch(color=CATEGORY_COLORS.get(c, "#888"),
                                label=f"Mov: {c}")
                 for c in mov_cats_present]
                + [mpatches.Patch(color=STEP_CATEGORY_COLORS.get(c, "#888"),
                                  label=f"Step: {c}")
                   for c in step_cats_present]
            )
            ax.legend(handles=legend_handles, framealpha=0.9,
                      labelcolor=LEGEND_TEXT, facecolor=LEGEND_FACE,
                      edgecolor=LEGEND_EDGE, fontsize=8,
                      loc="upper right", ncol=2)

            _save(fig, os.path.join(
                out_dir,
                f"plot3_{algo}_{sensor}_allDS_category_boxplot.png"))


def plot4_pairwise_scatter(df_long, out_dir, **_kwargs):
    from scipy import stats as sp_stats
    algos_avail = sorted(df_long["algorithm"].unique())
    pairs = [(algos_avail[i], algos_avail[j])
             for i in range(len(algos_avail))
             for j in range(i+1, len(algos_avail))]
    if not pairs:
        print("  [plot4] Not enough algorithms"); return

    label_algo_mean = (
        df_long
        .groupby(["PA_TYPE", "activity_category", "algorithm"])
        ["steps_per_min"].mean().reset_index()
    )
    pivot = label_algo_mean.pivot_table(
        index=["PA_TYPE", "activity_category"],
        columns="algorithm", values="steps_per_min"
    ).reset_index()

    cats_in_data = sorted(set(pivot["activity_category"].dropna()))
    cat_patches = [
        mpatches.Patch(color=CATEGORY_COLORS.get(c, "#888"), label=c)
        for c in CATEGORY_ORDER if c in cats_in_data
    ]

    for (a, b) in pairs:
        fig, ax = plt.subplots(figsize=(7.5, 6))
        fig.patch.set_facecolor(DARK_BG)
        sub = pivot.dropna(subset=[a, b])
        if len(sub) < 3:
            ax.text(0.5, 0.5, "Insufficient data", color=TEXT_COL,
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=11)
            _style(ax, xlabel=f"{a} (spm)", ylabel=f"{b} (spm)",
                   title=f"{a} vs {b}")
            _save(fig, os.path.join(
                out_dir,
                f"plot4_allDS_pairwise_scatter_{a}_vs_{b}.png"))
            continue

        x_vals = sub[a].values; y_vals = sub[b].values
        categories = sub["activity_category"].values
        pa_types   = sub["PA_TYPE"].values
        point_colors = [CATEGORY_COLORS.get(c, "#888") for c in categories]
        ax.scatter(x_vals, y_vals, c=point_colors, s=60, alpha=0.85,
                   linewidths=0.5, edgecolors="#333", zorder=3)
        for xi, yi, pa in zip(x_vals, y_vals, pa_types):
            short = pa.replace("_", " ")
            if len(short) > 22: short = short[:20] + "…"
            ax.annotate(short, (xi, yi), textcoords="offset points",
                        xytext=(4, 3), fontsize=5.5, color=TEXT_COL,
                        alpha=0.75)
        slope, intercept, r_val, p_val, _ = sp_stats.linregress(
            x_vals, y_vals)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
        ax.plot(x_line, slope*x_line + intercept, color=TEXT_COL,
                linewidth=1.8, linestyle="--", alpha=0.9, zorder=4,
                label=f"y = {slope:.2f}x + {intercept:.2f}\n"
                      f"r = {r_val:.3f},  R² = {r_val**2:.3f}")
        lim_max = max(x_vals.max(), y_vals.max()) * 1.05
        ax.plot([0, lim_max], [0, lim_max], color="#777", linewidth=1.0,
                linestyle=":", alpha=0.7, zorder=2, label="1:1 line")
        ax.legend(framealpha=0.9, labelcolor=LEGEND_TEXT,
                  facecolor=LEGEND_FACE, edgecolor=LEGEND_EDGE, fontsize=8,
                  loc="upper left")
        p_str = "<0.001" if p_val < 0.001 else f"{p_val:.3f}"
        _style(ax, xlabel=f"{a}  (mean spm)", ylabel=f"{b}  (mean spm)",
               title=f"All DS — {a}  vs  {b}\n"
                     f"r = {r_val:.3f},  p = {p_str},  "
                     f"R² = {r_val**2:.3f}  (n = {len(sub)} labels)")
        ax.spines[["top", "right"]].set_visible(False)
        if cat_patches:
            fig.legend(handles=cat_patches, loc="lower center",
                       ncol=min(len(cat_patches), 6), framealpha=0.9,
                       labelcolor=LEGEND_TEXT, facecolor=LEGEND_FACE,
                       edgecolor=LEGEND_EDGE, fontsize=8,
                       bbox_to_anchor=(0.5, -0.06))
        _save(fig, os.path.join(
            out_dir, f"plot4_allDS_pairwise_scatter_{a}_vs_{b}.png"))


def plot5_per_person_distribution(df_long, out_dir):
    algos   = sorted(df_long["algorithm"].unique())
    sensors = sorted(df_long["sensor"].unique())
    pa_types_present = set(df_long["PA_TYPE"].unique())
    activities, boundaries = _ordered_activities_by_category(pa_types_present)
    if not activities:
        print("  [SKIP plot5] No activities found"); return
    MAX_FIG_HEIGHT = 400

    for sensor in sensors:
        for algo in algos:
            sub = df_long[(df_long["algorithm"] == algo) &
                          (df_long["sensor"] == sensor)].copy()
            if sub.empty: continue
            person_agg = (
                sub.groupby(["ds_id", "PA_TYPE"])
                .agg(total_dur_sec=("duration_sec", "sum"),
                     total_steps=("total_steps", "sum"))
                .reset_index()
            )
            person_totals = (
                person_agg.groupby("ds_id")
                .agg(grand_dur=("total_dur_sec", "sum"),
                     grand_steps=("total_steps", "sum"))
                .reset_index()
            )
            person_agg = person_agg.merge(person_totals, on="ds_id")
            person_agg["pct_dur"] = (
                person_agg["total_dur_sec"]
                / person_agg["grand_dur"].replace(0, np.nan) * 100)
            person_agg["pct_steps"] = (
                person_agg["total_steps"]
                / person_agg["grand_steps"].replace(0, np.nan) * 100)

            n_acts = len(activities)
            fig_h  = min(max(10, n_acts * 0.5 + 4), MAX_FIG_HEIGHT)
            fig, axes = plt.subplots(1, 2, figsize=(22, fig_h))
            fig.patch.set_facecolor(DARK_BG)
            y_pos = np.arange(n_acts)
            plot_configs = [
                (axes[0], "total_dur_sec", "pct_dur",
                 "Duration (min)", 60.0),
                (axes[1], "total_steps", "pct_steps",
                 "Step Count", 1.0),
            ]
            for ax, val_col, pct_col, xlabel, divisor in plot_configs:
                ax.set_facecolor(DARK_AX)
                for i, act in enumerate(activities):
                    act_data = person_agg[person_agg["PA_TYPE"] == act]
                    cat = activity_category(act)
                    col = CATEGORY_COLORS.get(cat, "#888888")
                    raw_vals = act_data[val_col].dropna().values / divisor
                    pct_vals = act_data[pct_col].dropna().values
                    if len(raw_vals) == 0:
                        ax.plot(0, i, marker="|", color="#555", markersize=8)
                        continue
                    bp = ax.boxplot(
                        raw_vals, positions=[i], vert=False, widths=0.6,
                        patch_artist=True, notch=False,
                        medianprops={"color": TEXT_COL, "linewidth": 2},
                        whiskerprops={"color": "#aaa"},
                        capprops={"color": "#aaa"},
                        flierprops={"marker": "o", "markersize": 2.5,
                                    "markerfacecolor": "#666",
                                    "linestyle": "none"},
                    )
                    for patch in bp["boxes"]:
                        patch.set_facecolor(col); patch.set_alpha(0.78)
                    med_raw = np.median(raw_vals)
                    med_pct = (np.median(pct_vals) if len(pct_vals)
                               else 0.0)
                    ax.text(med_raw, i + 0.38, f"{med_pct:.1f}%",
                            color=TEXT_COL, ha="center", va="bottom",
                            fontsize=6.5)

                for cat_name, _, _ in boundaries:
                    col = CATEGORY_COLORS.get(cat_name, "#888888")
                    local = [i for i, a in enumerate(activities)
                             if activity_category(a) == cat_name]
                    if not local: continue
                    if local[0] > 0:
                        ax.axhline(local[0] - 0.5, color="#555",
                                   linewidth=1.0, linestyle="--", alpha=0.7)
                    mid = (local[0] + local[-1]) / 2
                    ax.text(1, mid, f"  {cat_name}", color=col, fontsize=8,
                            fontweight="bold", va="center", ha="left",
                            transform=ax.get_yaxis_transform())
                ax.set_yticks(y_pos)
                ax.set_yticklabels(
                    [a.replace("_", " ") for a in activities],
                    fontsize=8, color=TEXT_COL)
                _style(ax, xlabel=xlabel)
                ax.spines[["top", "right"]].set_visible(False)

            legend_handles = [
                mpatches.Patch(color=CATEGORY_COLORS.get(c, "#888"), label=c)
                for c, _, _ in boundaries
            ]
            fig.legend(handles=legend_handles, loc="lower center",
                       ncol=min(len(legend_handles), 4), framealpha=0.9,
                       labelcolor=LEGEND_TEXT, facecolor=LEGEND_FACE,
                       edgecolor=LEGEND_EDGE, fontsize=9,
                       bbox_to_anchor=(0.5, -0.01))
            fig.suptitle(
                f"Per-Person Distribution — {sensor}  |  {algo}",
                color=TEXT_COL, fontsize=12, fontweight="bold")
            _save(fig, os.path.join(
                out_dir,
                f"plot5_{sensor}_{algo}_per_person_distribution.png"))

    summary = (
        df_long.groupby(["ds_id", "sensor", "algorithm", "PA_TYPE",
                         "activity_category", "step_category"])
        .agg(total_dur_min=("duration_sec",
                            lambda x: round(x.sum() / 60, 2)),
             total_steps=("total_steps", "sum"))
        .reset_index()
    )
    summary.to_csv(
        os.path.join(out_dir, "plot5_per_person_activity_summary.csv"),
        index=False)


# ═════════════════════════════════════════════════════════════════════════════
#  NEW PLOTS 6–9  —  Step-Based Ground Truth Analysis
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6 — FP rate shift: old (Non-Movement) vs new (No_Step) definition
#
#   RQ: How does redefining the ground truth from "movement" to "stepping"
#       change false-positive rates?
#
#   Grouped bar chart: for each algo × sensor, show two bars side by side.
#     Bar 1 = FP rate under old definition (Non-Movement, 4 activities)
#     Bar 2 = FP rate under new definition (No_Step, 19 activities)
# ─────────────────────────────────────────────────────────────────────────────

def plot6_fp_rate_shift(df_long, out_dir):
    """
    Grouped bar chart comparing FP rates under old vs new ground truth.
    One group per algorithm, faceted by sensor.
    """
    algos   = sorted(df_long["algorithm"].unique())
    sensors = sorted(df_long["sensor"].unique())

    rows = []
    for sensor in sensors:
        s_df = df_long[df_long["sensor"] == sensor]
        for algo in algos:
            a_df = s_df[s_df["algorithm"] == algo]

            # Old definition: Non-Movement only (4 activities)
            old_nm = a_df[a_df["activity_category"] == "Non-Movement"]
            old_fp = ((old_nm["total_steps"] > 0).mean() * 100
                      if len(old_nm) else np.nan)

            # New definition: No_Step (19 activities)
            new_ns = a_df[a_df["step_category"] == "No_Step"]
            new_fp = ((new_ns["total_steps"] > 0).mean() * 100
                      if len(new_ns) else np.nan)

            rows.append({
                "sensor": sensor, "algorithm": algo,
                "fp_old_pct": old_fp, "fp_new_pct": new_fp,
                "n_old": len(old_nm), "n_new": len(new_ns),
            })

    fp_df = pd.DataFrame(rows)
    fp_df.to_csv(os.path.join(out_dir, "plot6_fp_rate_shift.csv"),
                 index=False)

    n_sensors = len(sensors)
    fig, axes = plt.subplots(1, n_sensors,
                             figsize=(7 * n_sensors, 6), squeeze=False)
    fig.patch.set_facecolor(DARK_BG)

    bar_w = 0.35
    for si, sensor in enumerate(sensors):
        ax = axes[0, si]
        ax.set_facecolor(DARK_AX)
        sf = fp_df[fp_df["sensor"] == sensor]
        x = np.arange(len(algos))

        bars_old = ax.bar(x - bar_w / 2, sf["fp_old_pct"], bar_w,
                          label="Old (Non-Movement, 4 activities)",
                          color="#2E86AB", alpha=0.85, edgecolor="#1a5276",
                          linewidth=0.8)
        bars_new = ax.bar(x + bar_w / 2, sf["fp_new_pct"], bar_w,
                          label="New (No_Step, 19 activities)",
                          color="#E84855", alpha=0.85, edgecolor="#9b1c24",
                          linewidth=0.8)

        # Value labels on bars
        for bar_set in [bars_old, bars_new]:
            for bar in bar_set:
                h = bar.get_height()
                if not np.isnan(h):
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                            f"{h:.1f}%", ha="center", va="bottom",
                            fontsize=8, color=TEXT_COL, fontweight="bold")

        # Annotate the delta on top
        for i, row in sf.iterrows():
            idx = algos.index(row["algorithm"])
            delta = row["fp_new_pct"] - row["fp_old_pct"]
            top = max(row["fp_old_pct"], row["fp_new_pct"])
            if not np.isnan(delta):
                ax.text(idx, top + 4.5,
                        f"Δ = +{delta:.1f}pp" if delta >= 0
                        else f"Δ = {delta:.1f}pp",
                        ha="center", va="bottom", fontsize=7,
                        color="#E84855" if delta > 0 else "#57CC99",
                        fontstyle="italic")

        ax.set_xticks(x)
        ax.set_xticklabels(algos, color=TEXT_COL, fontsize=10)
        ax.set_ylim(0, max(fp_df["fp_new_pct"].max(),
                           fp_df["fp_old_pct"].max()) * 1.25)
        ax.legend(framealpha=0.9, labelcolor=LEGEND_TEXT,
                  facecolor=LEGEND_FACE, edgecolor=LEGEND_EDGE,
                  fontsize=8, loc="upper left")
        _style(ax, ylabel="False Positive Rate (%)",
               title=f"FP Rate Shift — {sensor}\n"
                     f"(old: Non-Movement standard  →  "
                     f"new: No_Step standard)")
        ax.spines[["top", "right"]].set_visible(False)

    _save(fig, os.path.join(out_dir, "plot6_fp_rate_shift.png"))


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 7 — Step attribution stacked bar + per-person box plots
#
#   RQ: How many total steps come from No_Step activities, and what share
#       of each person's daily total do they represent?
#
#   (a) Stacked bar: % of total steps from Step / No_Step / Maybe_Step
#       per algorithm × sensor.
#   (b) Per-person box plots: distribution across participants of % total
#       steps from No_Step, one box per algorithm, faceted by sensor.
# ─────────────────────────────────────────────────────────────────────────────

def plot7_step_attribution(df_long, out_dir):
    """
    (a) Stacked bars of step attribution by step_category.
    (b) Per-person box plots of No_Step step share.
    """
    algos   = sorted(df_long["algorithm"].unique())
    sensors = sorted(df_long["sensor"].unique())
    cats    = STEP_CATEGORY_ORDER   # No_Step, Step, Maybe_Step

    # ── (a) Aggregate stacked bar ────────────────────────────────────────────

    agg_rows = []
    for sensor in sensors:
        s_df = df_long[df_long["sensor"] == sensor]
        for algo in algos:
            a_df = s_df[s_df["algorithm"] == algo]
            grand_total = a_df["total_steps"].sum()
            for cat in cats:
                cat_total = a_df[a_df["step_category"] == cat][
                    "total_steps"].sum()
                pct = (cat_total / grand_total * 100
                       if grand_total > 0 else 0.0)
                agg_rows.append({
                    "sensor": sensor, "algorithm": algo,
                    "step_category": cat,
                    "total_steps": cat_total, "pct_of_total": pct,
                })
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(os.path.join(out_dir, "plot7_step_attribution.csv"),
                  index=False)

    n_sensors = len(sensors)
    fig, axes = plt.subplots(1, n_sensors,
                             figsize=(6 * n_sensors, 6), squeeze=False)
    fig.patch.set_facecolor(DARK_BG)

    for si, sensor in enumerate(sensors):
        ax = axes[0, si]
        ax.set_facecolor(DARK_AX)
        sf = agg_df[agg_df["sensor"] == sensor]
        x  = np.arange(len(algos))
        bottom = np.zeros(len(algos))

        for cat in cats:
            vals = []
            for algo in algos:
                row = sf[(sf["algorithm"] == algo) &
                         (sf["step_category"] == cat)]
                vals.append(row["pct_of_total"].values[0]
                            if len(row) else 0.0)
            vals = np.array(vals)
            col  = STEP_CATEGORY_COLORS.get(cat, "#888")
            bars = ax.bar(x, vals, 0.6, bottom=bottom, label=cat,
                          color=col, alpha=0.85, edgecolor="#444",
                          linewidth=0.6)
            # Label inside each segment
            for i, (v, b) in enumerate(zip(vals, bottom)):
                if v > 3:   # only label if > 3% for readability
                    ax.text(i, b + v / 2, f"{v:.1f}%",
                            ha="center", va="center",
                            fontsize=8, color="#111", fontweight="bold")
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(algos, color=TEXT_COL, fontsize=10)
        ax.set_ylim(0, 105)
        ax.legend(framealpha=0.9, labelcolor=LEGEND_TEXT,
                  facecolor=LEGEND_FACE, edgecolor=LEGEND_EDGE,
                  fontsize=9, loc="upper right")
        _style(ax, ylabel="% of Total Detected Steps",
               title=f"Step Attribution by Category — {sensor}\n"
                     f"(all DS combined)")
        ax.spines[["top", "right"]].set_visible(False)

    _save(fig, os.path.join(out_dir, "plot7a_step_attribution_stacked.png"))

    # ── (b) Per-person box plots of No_Step share ────────────────────────────

    person_agg = (
        df_long.groupby(["ds_id", "sensor", "algorithm", "step_category"])
        ["total_steps"].sum().reset_index()
    )
    person_total = (
        df_long.groupby(["ds_id", "sensor", "algorithm"])
        ["total_steps"].sum().reset_index(name="grand_steps")
    )
    person_agg = person_agg.merge(person_total,
                                  on=["ds_id", "sensor", "algorithm"])
    person_agg["pct"] = (
        person_agg["total_steps"]
        / person_agg["grand_steps"].replace(0, np.nan) * 100
    )
    nostep_pct = person_agg[
        person_agg["step_category"] == "No_Step"
    ].copy()
    nostep_pct.to_csv(
        os.path.join(out_dir, "plot7_per_person_nostep_pct.csv"),
        index=False)

    fig, axes = plt.subplots(1, n_sensors,
                             figsize=(6 * n_sensors, 6), squeeze=False)
    fig.patch.set_facecolor(DARK_BG)

    for si, sensor in enumerate(sensors):
        ax = axes[0, si]
        ax.set_facecolor(DARK_AX)
        sf = nostep_pct[nostep_pct["sensor"] == sensor]
        data = [sf[sf["algorithm"] == algo]["pct"].dropna().values
                for algo in algos]
        x = np.arange(len(algos))
        for i, (algo, d) in enumerate(zip(algos, data)):
            col = algo_color(algo)
            if len(d) == 0: continue
            bp = ax.boxplot(
                d, positions=[i], widths=0.55, patch_artist=True,
                notch=False,
                medianprops={"color": TEXT_COL, "linewidth": 2},
                whiskerprops={"color": "#aaa"},
                capprops={"color": "#aaa"},
                flierprops={"marker": "o", "markersize": 3,
                            "markerfacecolor": "#666",
                            "linestyle": "none"},
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(col); patch.set_alpha(0.80)
            med = np.median(d)
            ax.text(i, np.percentile(d, 75) + 0.5, f"{med:.1f}%",
                    color=TEXT_COL, ha="center", va="bottom",
                    fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(algos, color=TEXT_COL, fontsize=10)
        ax.axhline(0, color="#555", linewidth=0.8, linestyle="--")
        _style(ax,
               ylabel="% of Participant's Total Steps from No_Step",
               title=f"Per-Person No_Step Step Share — {sensor}\n"
                     f"(distribution across participants)")
        ax.spines[["top", "right"]].set_visible(False)

    _save(fig, os.path.join(out_dir,
                            "plot7b_per_person_nostep_boxplot.png"))


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 8 — Reclassified activities: Maybe-Moving BUT No_Step
#
#   RQ: Which algorithms best distinguish stepping from non-stepping movement?
#
#   (a) Box plots of steps/min for the 15 reclassified activities,
#       one figure per algorithm.
#   (b) Discrimination gap dumbbell chart: for each algo × sensor,
#       median spm of Step activities vs median spm of reclassified
#       No_Step-but-Moving activities.
# ─────────────────────────────────────────────────────────────────────────────

def plot8_reclassified_activities(df_long, out_dir):
    """
    (a) Box plots for activities that are Maybe-Moving AND No_Step.
    (b) Discrimination gap dumbbell chart.
    """
    algos   = sorted(df_long["algorithm"].unique())
    sensors = sorted(df_long["sensor"].unique())
    reclass = RECLASSIFIED_ACTIVITIES   # pre-computed at module level

    if not reclass:
        print("  [SKIP plot8] No reclassified activities found"); return

    reclass_df = df_long[df_long["PA_TYPE"].isin(reclass)].copy()

    # ── (a) Box plots per algorithm — all sensors combined ───────────────────

    n_acts = len(reclass)
    for algo in algos:
        sub = reclass_df[reclass_df["algorithm"] == algo]
        data = [sub[sub["PA_TYPE"] == a]["steps_per_min"].dropna().values
                for a in reclass]

        fig_h = min(max(8, n_acts * 0.5 + 3), 400)
        fig, ax = plt.subplots(figsize=(13, fig_h))
        fig.patch.set_facecolor(DARK_BG)
        y_pos = np.arange(n_acts)

        for i, (act, d) in enumerate(zip(reclass, data)):
            if len(d) == 0:
                ax.plot(0, i, marker="|", color="#555", markersize=8)
                continue
            bp = ax.boxplot(
                d, positions=[i], vert=False, widths=0.6,
                patch_artist=True, notch=False,
                medianprops={"color": TEXT_COL, "linewidth": 2},
                whiskerprops={"color": "#aaa"},
                capprops={"color": "#aaa"},
                flierprops={"marker": "o", "markersize": 2.5,
                            "markerfacecolor": "#666",
                            "linestyle": "none"},
            )
            # Colour: red because these are No_Step — any detected steps = FP
            for patch in bp["boxes"]:
                patch.set_facecolor(STEP_CATEGORY_COLORS["No_Step"])
                patch.set_alpha(0.75)
            # Annotate median
            med = np.median(d)
            ax.text(np.percentile(d, 75) + 0.3, i,
                    f"med={med:.1f}", color=TEXT_COL,
                    va="center", ha="left", fontsize=7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([a.replace("_", " ") for a in reclass],
                           fontsize=8, color=TEXT_COL)
        ax.axvline(0, color="#555", linewidth=0.8, linestyle="--")
        _style(ax, xlabel="Steps / Min",
               title=f"Reclassified Activities — {algo}\n"
                     f"(body moves BUT feet don't step → "
                     f"ideal = 0 spm)")
        ax.spines[["top", "right"]].set_visible(False)

        note = ("These activities are Maybe-Moving (body moves) but "
                "No_Step (no foot stepping).\nAny detected steps are "
                "false positives under the step-based ground truth.")
        ax.text(0.98, 0.02, note, transform=ax.transAxes,
                fontsize=7, color="#777", ha="right", va="bottom",
                fontstyle="italic")

        _save(fig, os.path.join(
            out_dir, f"plot8a_{algo}_reclassified_boxplot.png"))

    # Save underlying data
    reclass_df[["PA_TYPE", "activity_category", "step_category",
                "algorithm", "sensor", "ds_id",
                "steps_per_min"]].to_csv(
        os.path.join(out_dir, "plot8_reclassified_segments.csv"),
        index=False)

    # ── (b) Discrimination gap dumbbell chart ────────────────────────────────

    gap_rows = []
    for sensor in sensors:
        s_df = df_long[df_long["sensor"] == sensor]
        for algo in algos:
            a_df = s_df[s_df["algorithm"] == algo]

            # Median spm for Step activities (sensitivity proxy)
            step_spm = a_df[a_df["step_category"] == "Step"][
                "steps_per_min"]
            med_step = step_spm.median() if len(step_spm) else 0.0

            # Median spm for reclassified (No_Step but Maybe-Moving)
            recl_spm = a_df[a_df["PA_TYPE"].isin(reclass)][
                "steps_per_min"]
            med_recl = recl_spm.median() if len(recl_spm) else 0.0

            gap_rows.append({
                "sensor": sensor, "algorithm": algo,
                "median_step_spm": med_step,
                "median_reclassified_spm": med_recl,
                "gap": med_step - med_recl,
            })
    gap_df = pd.DataFrame(gap_rows)
    gap_df.to_csv(os.path.join(out_dir, "plot8_discrimination_gap.csv"),
                  index=False)

    n_sensors = len(sensors)
    fig, axes = plt.subplots(1, n_sensors,
                             figsize=(7 * n_sensors, 5), squeeze=False)
    fig.patch.set_facecolor(DARK_BG)

    for si, sensor in enumerate(sensors):
        ax = axes[0, si]
        ax.set_facecolor(DARK_AX)
        sf = gap_df[gap_df["sensor"] == sensor]
        y  = np.arange(len(algos))

        for i, algo in enumerate(algos):
            row = sf[sf["algorithm"] == algo].iloc[0]
            col = algo_color(algo)

            # Connecting line
            ax.plot([row["median_reclassified_spm"],
                     row["median_step_spm"]],
                    [i, i], color=col, linewidth=2.5, alpha=0.7,
                    zorder=2)
            # Reclassified dot (left, red-ish)
            ax.scatter(row["median_reclassified_spm"], i,
                       color=STEP_CATEGORY_COLORS["No_Step"],
                       s=100, zorder=3, edgecolors="#333", linewidths=0.8)
            # Step dot (right, green)
            ax.scatter(row["median_step_spm"], i,
                       color=STEP_CATEGORY_COLORS["Step"],
                       s=100, zorder=3, edgecolors="#333", linewidths=0.8)
            # Gap annotation
            ax.text((row["median_reclassified_spm"]
                     + row["median_step_spm"]) / 2,
                    i + 0.3,
                    f"gap = {row['gap']:.1f}",
                    ha="center", va="bottom", fontsize=8,
                    color=TEXT_COL, fontweight="bold")

        ax.set_yticks(y)
        ax.set_yticklabels(algos, fontsize=10, color=TEXT_COL)
        ax.axvline(0, color="#555", linewidth=0.8, linestyle="--")

        legend_handles = [
            mpatches.Patch(color=STEP_CATEGORY_COLORS["Step"],
                           label="Step activities (median spm)"),
            mpatches.Patch(color=STEP_CATEGORY_COLORS["No_Step"],
                           label="Reclassified No_Step (median spm)"),
        ]
        ax.legend(handles=legend_handles, framealpha=0.9,
                  labelcolor=LEGEND_TEXT, facecolor=LEGEND_FACE,
                  edgecolor=LEGEND_EDGE, fontsize=8, loc="lower right")
        _style(ax, xlabel="Median Steps / Min",
               title=f"Discrimination Gap — {sensor}\n"
                     f"(Step vs body-moves-but-no-step)")
        ax.spines[["top", "right"]].set_visible(False)

    _save(fig, os.path.join(out_dir, "plot8b_discrimination_gap.png"))


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 9 — Sensitivity vs Specificity under step-based ground truth
#
#   RQ: What is the sensitivity-specificity trade-off when using
#       step-based labels as ground truth?
#
#   Binary classification: Step = positive, No_Step = negative.
#   Detection = total_steps > 0 in a segment.
#   One point per algorithm × sensor.
#   Also: threshold sweep producing pseudo-ROC curves.
# ─────────────────────────────────────────────────────────────────────────────

def plot9_sensitivity_specificity(df_long, out_dir):
    """
    (a) Scatter plot: sensitivity vs (1 - FP rate).
    (b) Pseudo-ROC: sweep steps/min threshold.
    """
    algos   = sorted(df_long["algorithm"].unique())
    sensors = sorted(df_long["sensor"].unique())

    # Exclude Maybe_Step segments — binary evaluation only
    binary_df = df_long[
        df_long["step_category"].isin(["Step", "No_Step"])
    ].copy()

    # ── (a) Single-threshold scatter (threshold = steps > 0) ─────────────────

    ss_rows = []
    for sensor in sensors:
        s_df = binary_df[binary_df["sensor"] == sensor]
        for algo in algos:
            a_df = s_df[s_df["algorithm"] == algo]

            positives = a_df[a_df["step_category"] == "Step"]
            negatives = a_df[a_df["step_category"] == "No_Step"]

            sensitivity = ((positives["total_steps"] > 0).mean()
                           if len(positives) else np.nan)
            specificity = ((negatives["total_steps"] == 0).mean()
                           if len(negatives) else np.nan)

            ss_rows.append({
                "sensor": sensor, "algorithm": algo,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "n_step": len(positives),
                "n_nostep": len(negatives),
            })

    ss_df = pd.DataFrame(ss_rows)
    ss_df.to_csv(os.path.join(out_dir, "plot9_sensitivity_specificity.csv"),
                 index=False)

    # Scatter: one point per algo × sensor
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_AX)

    marker_map = {s: m for s, m in zip(sensors, ["o", "s", "D", "^"])}

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

    # Reference lines
    ax.axhline(1, color="#ccc", linewidth=0.6, linestyle=":")
    ax.axvline(1, color="#ccc", linewidth=0.6, linestyle=":")
    ax.plot([0, 1], [0, 1], color="#bbb", linewidth=0.8,
            linestyle="--", alpha=0.6, label="Random classifier")

    # Perfect corner marker
    ax.scatter(1, 1, marker="*", s=200, color="#FFD700",
               edgecolors="#333", linewidths=0.8, zorder=4,
               label="Perfect classifier")

    # Legend for sensors
    sensor_handles = [
        plt.Line2D([0], [0], marker=marker_map[s], color="#888",
                   markersize=8, linestyle="None", label=s)
        for s in sensors
    ]
    algo_handles = [
        mpatches.Patch(color=algo_color(a), label=a) for a in algos
    ]
    ax.legend(
        handles=algo_handles + sensor_handles +
                [plt.Line2D([0], [0], marker="*", color="#FFD700",
                            markersize=10, linestyle="None",
                            label="Perfect"),
                 plt.Line2D([0], [0], color="#bbb", linestyle="--",
                            label="Random")],
        framealpha=0.9, labelcolor=LEGEND_TEXT, facecolor=LEGEND_FACE,
        edgecolor=LEGEND_EDGE, fontsize=8, loc="lower left")

    ax.set_xlim(-0.02, 1.08)
    ax.set_ylim(-0.02, 1.08)
    _style(ax,
           xlabel="Specificity (1 − FP rate, No_Step correctly rejected)",
           ylabel="Sensitivity (Step correctly detected)",
           title="Sensitivity vs Specificity — Step-Based Ground Truth\n"
                 "(detection = total_steps > 0 per segment)")
    ax.spines[["top", "right"]].set_visible(False)

    _save(fig, os.path.join(out_dir, "plot9a_sensitivity_specificity.png"))

    # ── (b) Pseudo-ROC: sweep threshold on steps_per_min ─────────────────────

    thresholds = [0, 0.5, 1, 2, 5, 10, 20, 50]

    fig, axes = plt.subplots(1, len(sensors),
                             figsize=(7 * len(sensors), 6), squeeze=False)
    fig.patch.set_facecolor(DARK_BG)

    roc_rows = []
    for si, sensor in enumerate(sensors):
        ax = axes[0, si]
        ax.set_facecolor(DARK_AX)
        s_df = binary_df[binary_df["sensor"] == sensor]

        for algo in algos:
            a_df = s_df[s_df["algorithm"] == algo]
            positives = a_df[a_df["step_category"] == "Step"]
            negatives = a_df[a_df["step_category"] == "No_Step"]

            sens_list, spec_list = [], []
            for thr in thresholds:
                sens = ((positives["steps_per_min"] > thr).mean()
                        if len(positives) else np.nan)
                spec = ((negatives["steps_per_min"] <= thr).mean()
                        if len(negatives) else np.nan)
                sens_list.append(sens)
                spec_list.append(spec)
                roc_rows.append({
                    "sensor": sensor, "algorithm": algo,
                    "threshold_spm": thr,
                    "sensitivity": sens, "specificity": spec,
                })

            col = algo_color(algo)
            ax.plot(spec_list, sens_list, marker="o", markersize=5,
                    color=col, linewidth=2, alpha=0.85, label=algo)

            # Label each threshold point
            for thr, sp, sn in zip(thresholds, spec_list, sens_list):
                if not np.isnan(sp) and not np.isnan(sn):
                    ax.annotate(f"{thr}", (sp, sn),
                                textcoords="offset points",
                                xytext=(4, 4), fontsize=6,
                                color=col, alpha=0.7)

        ax.plot([0, 1], [0, 1], color="#bbb", linewidth=0.8,
                linestyle="--", alpha=0.6)
        ax.scatter(1, 1, marker="*", s=150, color="#FFD700",
                   edgecolors="#333", linewidths=0.8, zorder=4)

        ax.set_xlim(-0.02, 1.08)
        ax.set_ylim(-0.02, 1.08)
        ax.legend(framealpha=0.9, labelcolor=LEGEND_TEXT,
                  facecolor=LEGEND_FACE, edgecolor=LEGEND_EDGE,
                  fontsize=8, loc="lower left")
        _style(ax,
               xlabel="Specificity",
               ylabel="Sensitivity",
               title=f"Pseudo-ROC (threshold sweep) — {sensor}\n"
                     f"(numbers = steps/min threshold)")
        ax.spines[["top", "right"]].set_visible(False)

    pd.DataFrame(roc_rows).to_csv(
        os.path.join(out_dir, "plot9_pseudo_roc_data.csv"), index=False)
    _save(fig, os.path.join(out_dir, "plot9b_pseudo_roc.png"))


# ═════════════════════════════════════════════════════════════════════════════
#  PARALLEL PROCESSING & MAIN
# ═════════════════════════════════════════════════════════════════════════════

_print_lock = None

def _pool_init(lock):
    global _print_lock
    _print_lock = lock


def _process_ds_worker(args):
    ds_root, ds_id, algorithms, label_root = args
    records, log_lines = process_ds(ds_root, ds_id, algorithms, label_root)
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
                f"  │  [{ds_id}] [SKIP] No steps files for algo={algo}")
            continue
        seen_sensors = {}
        for sensor, fpath in sensor_files:
            if sensor not in seen_sensors:
                seen_sensors[sensor] = fpath
        for sensor, fpath in seen_sensors.items():
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
        description="Step algorithm comparison (v2 with step-based GT)")
    parser.add_argument("--root",    default=DEFAULT_ROOT)
    parser.add_argument("--labels",  default=DEFAULT_LABELS)
    parser.add_argument("--ds",      nargs="*", default=None)
    parser.add_argument("--algos",   nargs="*", default=ALGORITHMS)
    parser.add_argument("--out",     default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("╔══════════════════════════════════════════════════════╗")
    print("║  STEP ALGORITHM COMPARISON v2 — STEP-BASED GT      ║")
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

    n_workers   = min(len(ds_folders), args.workers)
    worker_args = [
        (ds_root, os.path.basename(ds_root), args.algos, args.labels)
        for ds_root in ds_folders
    ]

    all_records = []
    if n_workers > 1:
        print(f"\n  Using {n_workers} parallel workers...")
        lock = mp.Manager().Lock()
        with mp.Pool(processes=n_workers,
                     initializer=_pool_init, initargs=(lock,)) as pool:
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
    df_long.to_csv(os.path.join(args.out, "segments_long.csv"), index=False)
    print(f"\n  {len(df_long)} total segment records (sensors: {SENSORS})")

    # Print step category coverage
    for cat in STEP_CATEGORY_ORDER:
        n = (df_long["step_category"] == cat).sum()
        print(f"    {cat:12s}: {n:>7d} segments")
    n_recl = df_long["PA_TYPE"].isin(RECLASSIFIED_ACTIVITIES).sum()
    print(f"    {'Reclassified':12s}: {n_recl:>7d} segments "
          f"(Maybe-Moving ∩ No_Step)")

    print("\n  ══ Original Plots (1–5) ══")

    print("\n  [1/9] Activity box plots per algorithm...")
    plot1_label_boxplot_by_category(df_long, args.out)

    print("\n  [2/9] Sensor × algorithm FP heatmap (Non-Movement)...")
    plot2_fp_heatmap(df_long, args.out)

    print("\n  [3/9] All-DS category box plots per algorithm...")
    plot3_allDS_per_algorithm(df_long, args.out)

    print("\n  [4/9] Pairwise scatter with regression...")
    plot4_pairwise_scatter(df_long, args.out)

    print("\n  [5/9] Per-person time & step distribution...")
    plot5_per_person_distribution(df_long, args.out)

    print("\n  ══ New Step-Based Plots (6–9) ══")

    print("\n  [6/9] FP rate shift: old (Non-Movement) vs "
          "new (No_Step) definition...")
    plot6_fp_rate_shift(df_long, args.out)

    print("\n  [7/9] Step attribution stacked bar + "
          "per-person No_Step share...")
    plot7_step_attribution(df_long, args.out)

    print("\n  [8/9] Reclassified activities (Maybe-Moving ∩ No_Step) "
          "box plots + discrimination gap...")
    plot8_reclassified_activities(df_long, args.out)

    print("\n  [9/9] Sensitivity vs specificity + pseudo-ROC...")
    plot9_sensitivity_specificity(df_long, args.out)

    print(f"\n  ✓ Done! Output in: {args.out}")
    print()
    for f in sorted(os.listdir(args.out)):
        kb = os.path.getsize(os.path.join(args.out, f)) / 1024
        print(f"    {f:<60} {kb:6.1f} KB")


if __name__ == "__main__":
    main()