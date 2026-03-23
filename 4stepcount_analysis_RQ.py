#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║         STEP ALGORITHM COMPARISON ANALYSIS  —  Cluster Version      ║
╠══════════════════════════════════════════════════════════════════════╣
║  Auto-discovers all DS_xx folders, all sensors, all 4 algorithms    ║
║  (oxford / oak / verisense / adept), reads .csv.gz files, joins     ║
║  with activity labels, and outputs summary CSVs + comparison plots. ║
║                                                                      ║
║  Research Questions:                                                 ║
║    RQ1: Do algorithms generally agree?                               ║
║      Exp1: Pairwise Pearson + Bland-Altman (overall)                 ║
║      Exp2: Same, stratified by activity category and sensor          ║
║    RQ2: Do algorithms make mistakes?                                 ║
║      FP analysis per sensor + severity (daily inflation)             ║
║    RQ3: Where do algorithms most disagree?                           ║
║      Per-activity CV across algorithms, filtered to highlight        ║
║      higher-than-normal disagreement                                 ║
╠══════════════════════════════════════════════════════════════════════╣
║  USAGE                                                               ║
║    python 4stepcount_analysis.py                                     ║
║    python 4stepcount_analysis.py --root /scratch/.../PAAWS_results   ║
║    python 4stepcount_analysis.py --ds DS_10 DS_12                    ║
║    python 4stepcount_analysis.py --out ./my_results                  ║
╠══════════════════════════════════════════════════════════════════════╣
║  EXPECTED DIRECTORY LAYOUT                                           ║
║    {root}/                                                           ║
║      DS_10/                                                          ║
║        DS_10-Free-LeftWrist_adept_steps.csv                          ║
║        DS_10-Free-LeftWrist_oak_steps.csv                            ║
║        DS_10-Free-LeftWrist_oxford_steps.csv                         ║
║        DS_10-Free-LeftWrist_verisense_steps.csv                      ║
║      DS_12/                                                          ║
║        ...                                                           ║
║    {labels}/                                                         ║
║      DS_10/label/DS_10-Free-label.csv                                ║
║      DS_12/label/DS_12-Free-label.csv                                ║
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
from scipy import stats as sp_stats
from itertools import combinations
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ROOT   = "/scratch/wang.yichen8/PAAWS_results"
DEFAULT_OUTPUT = "/scratch/wang.yichen8/step_analysis_output"
DEFAULT_LABELS = "/scratch/wang.yichen8/PAAWS_FreeLiving"
ALGORITHMS     = ["adept", "oak", "oxford", "verisense"]

LABEL_PATTERNS = [
    "{ds_root}/labels/{ds_id}-Free-label.csv",
    "{ds_root}/labels/{ds_id}-Free-label.csv.gz",
    "{ds_root}/{ds_id}-Free-label.csv",
    "{ds_root}/{ds_id}-Free-label.csv.gz",
]

# ── Activity category definitions ────────────────────────────────────────────
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
    "Blowdrying_Hair", "Showering", "Washing_Face", "Brushing_Teeth",
    "Brushing/Combing/Tying_Hair", "Organizing_Shelf/Cabinet",
    "Folding_Clothes", "Ironing", "Flossing_Teeth", "Washing_Hands",
    "Putting_Clothes_Away", "Loading/Unloading_Washing_Machine/Dryer",
    "Watering_Plants", "Dusting", "Dry_Mopping", "Sweeping",
    "Vacuuming", "Wet_Mopping",
]
EXCLUDE_ACTIVITIES = [
    "PA_Type_Video_Unavailable/Indecipherable",
    "Posture_Video_Unavailable/Indecipherable",
    "Synchronizing_Sensors", "PA_Type_Too_Complex",
    "PA_Type_Other", "PA_Type_Unlabeled",
]

CATEGORY_ORDER = ["Non-Movement", "Movement", "Maybe-Moving"]

CATEGORY_ACTIVITY_MAP = {
    "Non-Movement": NON_MOVEMENT_ACTIVITIES,
    "Movement":     MOVEMENT_ACTIVITIES,
    "Maybe-Moving": MAYBE_MOVING_ACTIVITIES,
}

MIN_SEGMENT_SEC = 10

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

def algo_color(name):
    for key, col in ALGO_COLORS.items():
        if key in name.lower():
            return col
    return EXTRA_COLORS[hash(name) % len(EXTRA_COLORS)]

# ─────────────────────────────────────────────────────────────────────────────
# ACTIVITY → CATEGORY MAPPING
# ─────────────────────────────────────────────────────────────────────────────

def activity_category(pa_type):
    for cat, acts in CATEGORY_ACTIVITY_MAP.items():
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
        pattern = os.path.join(ds_root,
                               f"*-Free-*_{algorithm}_steps.{ext}")
        hits.extend(glob.glob(pattern))
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
            "ds_id": ds_id, "algorithm": algorithm, "sensor": sensor,
            "PA_TYPE": pa_types[i],
            "activity_category": activity_category(pa_types[i]),
            "duration_sec": dur, "total_steps": total,
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
    ax.grid(True, color=GRID_COL, linewidth=0.5, linestyle="--", axis="both")
    ax.set_axisbelow(True)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT_COL, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT_COL, fontsize=10)
    if title:  ax.set_title(title, color=TEXT_COL, fontsize=11,
                            fontweight="bold", pad=10)

def _save(fig, path):
    plt.tight_layout()
    plt.savefig(path, dpi=80, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"    Saved: {path}")

def _ordered_activities_by_category(pa_types_present):
    ordered, boundaries = [], []
    for cat in CATEGORY_ORDER:
        cat_acts = CATEGORY_ACTIVITY_MAP.get(cat, [])
        present  = [a for a in cat_acts if a in pa_types_present]
        if present:
            start = len(ordered)
            ordered.extend(present)
            boundaries.append((cat, start, len(ordered) - 1))
    return ordered, boundaries


# ═══════════════════════════════════════════════════════════════════════
# PLOT 1 — Box plots of steps/min per activity, grouped by category
# ═══════════════════════════════════════════════════════════════════════

def plot1_label_boxplot_by_category(df_long, out_dir):
    algos = sorted(df_long["algorithm"].unique())
    pa_types_present = set(df_long["PA_TYPE"].unique())
    activities, boundaries = _ordered_activities_by_category(pa_types_present)
    if not activities:
        print("  [SKIP plot1] No activities found"); return

    for algo in algos:
        sub = df_long[df_long["algorithm"] == algo]
        data = [sub[sub["PA_TYPE"] == a]["steps_per_min"].dropna().values
                for a in activities]
        n_acts = len(activities)
        fig_h = min(max(8, n_acts * 0.45 + 3), 400)
        fig, ax = plt.subplots(figsize=(13, fig_h))
        fig.patch.set_facecolor(DARK_BG)

        for i, (act, d) in enumerate(zip(activities, data)):
            cat = activity_category(act)
            col = CATEGORY_COLORS.get(cat, "#888888")
            if len(d) == 0:
                ax.plot(0, i, marker="|", color="#555", markersize=8)
                continue
            bp = ax.boxplot(d, positions=[i], vert=False, widths=0.6,
                            patch_artist=True,
                            medianprops={"color": TEXT_COL, "linewidth": 2},
                            whiskerprops={"color": "#aaa"},
                            capprops={"color": "#aaa"},
                            flierprops={"marker": "o", "markersize": 2.5,
                                        "markerfacecolor": "#666",
                                        "linestyle": "none"})
            for patch in bp["boxes"]:
                patch.set_facecolor(col); patch.set_alpha(0.78)

        for cat_name, start, end in boundaries:
            col = CATEGORY_COLORS.get(cat_name, "#888888")
            local = [i for i, a in enumerate(activities)
                     if activity_category(a) == cat_name]
            if not local: continue
            if local[0] > 0:
                ax.axhline(local[0] - 0.5, color="#555", linewidth=1.0,
                           linestyle="--", alpha=0.7)
            mid = (local[0] + local[-1]) / 2
            ax.text(1, mid, f"  {cat_name}", color=col, fontsize=8,
                    fontweight="bold", va="center", ha="left",
                    transform=ax.get_yaxis_transform())

        ax.set_yticks(range(n_acts))
        ax.set_yticklabels([a.replace("_", " ") for a in activities],
                           fontsize=8, color=TEXT_COL)
        _style(ax, xlabel="Steps / Min",
               title=f"Steps/Min per Activity — {algo}\n"
                     "(box=IQR, whiskers=1.5×IQR, line=median)")
        ax.spines[["top", "right"]].set_visible(False)
        legend_handles = [mpatches.Patch(color=CATEGORY_COLORS.get(c, "#888"),
                          label=c) for c, _, _ in boundaries]
        ax.legend(handles=legend_handles, framealpha=0.9,
                  labelcolor=LEGEND_TEXT, facecolor=LEGEND_FACE,
                  edgecolor=LEGEND_EDGE, fontsize=8, loc="lower right")
        _save(fig, os.path.join(out_dir,
              f"plot1_{algo}_activity_boxplot.png"))

    df_long[["PA_TYPE", "activity_category", "algorithm", "sensor",
             "ds_id", "steps_per_min"]].to_csv(
        os.path.join(out_dir, "plot1_segments_long.csv"), index=False)


# ═══════════════════════════════════════════════════════════════════════
# PLOT 2 — False-positive heatmap
# ═══════════════════════════════════════════════════════════════════════

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
    cmap = LinearSegmentedColormap.from_list("fp",
           ["#57CC99", "#F4A261", "#E84855"])
    mv = mat.values.astype(float)
    im = ax.imshow(mv, aspect="auto", cmap=cmap, vmin=0, vmax=100)
    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels(algos, color=TEXT_COL, fontsize=10)
    ax.set_yticks(range(len(sensors)))
    ax.set_yticklabels(sensors, color=TEXT_COL, fontsize=9)
    for r, sensor in enumerate(sensors):
        for c, algo in enumerate(algos):
            v = mv[r, c]
            if not np.isnan(v):
                ax.text(c, r, f"{v:.1f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="#FFF" if v > 50 else "#111")
                if algo == best[sensor]:
                    ax.add_patch(plt.Rectangle((c-.48, r-.48), .96, .96,
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


# ═══════════════════════════════════════════════════════════════════════
# PLOT 3 — All-DS category box plots per algorithm
# ═══════════════════════════════════════════════════════════════════════

def plot3_allDS_per_algorithm(df_long, out_dir):
    algos = sorted(df_long["algorithm"].unique())
    df_long = df_long.copy()
    df_long["activity_category"] = df_long["PA_TYPE"].apply(activity_category)
    cats_present = [c for c in CATEGORY_ORDER
                    if c in df_long["activity_category"].unique()]
    if not cats_present:
        print("  [SKIP plot3] No categories"); return

    for algo in algos:
        sub = df_long[df_long["algorithm"] == algo]
        data = [sub[sub["activity_category"] == cat]["steps_per_min"]
                .dropna().values for cat in cats_present]
        fig, ax = plt.subplots(figsize=(max(10, len(cats_present)*1.6+2), 6))
        fig.patch.set_facecolor(DARK_BG)
        for i, (cat, d) in enumerate(zip(cats_present, data)):
            col = CATEGORY_COLORS.get(cat, algo_color(algo))
            if len(d) == 0: continue
            bp = ax.boxplot(d, positions=[i], widths=0.55, patch_artist=True,
                            medianprops={"color": TEXT_COL, "linewidth": 2.5},
                            whiskerprops={"color": "#aaa"},
                            capprops={"color": "#aaa"},
                            flierprops={"marker": "o", "markersize": 2.5,
                                        "markerfacecolor": "#555"})
            for patch in bp["boxes"]:
                patch.set_facecolor(col); patch.set_alpha(0.80)
            med = np.median(d)
            ax.text(i, np.percentile(d, 75)+0.5, f"{med:.1f}",
                    color=TEXT_COL, ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(cats_present)))
        ax.set_xticklabels(cats_present, color=TEXT_COL, fontsize=9)
        ax.axhline(0, color="#555", linewidth=0.8, linestyle="--")
        for i, (cat, d) in enumerate(zip(cats_present, data)):
            ax.text(i, ax.get_ylim()[0], f"n={len(d)}",
                    color="#888", ha="center", va="top", fontsize=7)
        _style(ax, ylabel="Steps / Min",
               title=f"All DS — Steps/Min by Category — {algo}")
        ax.spines[["top", "right"]].set_visible(False)
        _save(fig, os.path.join(out_dir,
              f"plot3_{algo}_allDS_category_boxplot.png"))


# ═══════════════════════════════════════════════════════════════════════
# PLOT 4 — Pairwise scatter plots
# ═══════════════════════════════════════════════════════════════════════

def plot4_pairwise_scatter(df_long, out_dir):
    algos_avail = sorted(df_long["algorithm"].unique())
    pairs = list(combinations(algos_avail, 2))
    if not pairs:
        print("  [plot4] Need ≥2 algorithms"); return

    label_algo_mean = (
        df_long.groupby(["PA_TYPE", "activity_category", "algorithm"])
        ["steps_per_min"].mean().reset_index())
    pivot = label_algo_mean.pivot_table(
        index=["PA_TYPE", "activity_category"],
        columns="algorithm", values="steps_per_min").reset_index()

    for (a, b) in pairs:
        fig, ax = plt.subplots(figsize=(7.5, 6))
        fig.patch.set_facecolor(DARK_BG)
        sub = pivot.dropna(subset=[a, b])
        if len(sub) < 3:
            _save(fig, os.path.join(out_dir,
                  f"plot4_scatter_{a}_vs_{b}.png")); continue
        x, y = sub[a].values, sub[b].values
        cats = sub["activity_category"].values
        colors = [CATEGORY_COLORS.get(c, "#888") for c in cats]
        ax.scatter(x, y, c=colors, s=60, alpha=0.85, edgecolors="#333",
                   linewidths=0.5, zorder=3)
        for xi, yi, pa in zip(x, y, sub["PA_TYPE"].values):
            short = pa.replace("_", " ")
            if len(short) > 22: short = short[:20] + "…"
            ax.annotate(short, (xi, yi), textcoords="offset points",
                        xytext=(4, 3), fontsize=5.5, color=TEXT_COL, alpha=0.75)
        slope, intercept, r_val, p_val, _ = sp_stats.linregress(x, y)
        xl = np.linspace(x.min(), x.max(), 200)
        ax.plot(xl, slope*xl+intercept, color=TEXT_COL, linewidth=1.8,
                linestyle="--", alpha=0.9, zorder=4,
                label=f"y={slope:.2f}x+{intercept:.2f}\nr={r_val:.3f}")
        lm = max(x.max(), y.max()) * 1.05
        ax.plot([0, lm], [0, lm], color="#777", linewidth=1, linestyle=":",
                alpha=0.7, label="1:1 line")
        ax.legend(fontsize=8, facecolor=LEGEND_FACE, edgecolor=LEGEND_EDGE,
                  labelcolor=LEGEND_TEXT, loc="upper left")
        p_str = "<0.001" if p_val < 0.001 else f"{p_val:.3f}"
        _style(ax, xlabel=f"{a} (mean spm)", ylabel=f"{b} (mean spm)",
               title=f"{a} vs {b}  r={r_val:.3f} p={p_str} n={len(sub)}")
        ax.spines[["top", "right"]].set_visible(False)
        _save(fig, os.path.join(out_dir,
              f"plot4_scatter_{a}_vs_{b}.png"))


# ═══════════════════════════════════════════════════════════════════════
# PLOT 5 — Per-person distribution
# ═══════════════════════════════════════════════════════════════════════

def plot5_per_person_distribution(df_long, out_dir):
    algos = sorted(df_long["algorithm"].unique())
    pa_types_present = set(df_long["PA_TYPE"].unique())
    activities, boundaries = _ordered_activities_by_category(pa_types_present)
    if not activities:
        print("  [SKIP plot5] No activities"); return

    for algo in algos:
        sub = df_long[df_long["algorithm"] == algo].copy()
        person_agg = (sub.groupby(["ds_id", "sensor", "PA_TYPE"])
                      .agg(total_dur_sec=("duration_sec", "sum"),
                           total_steps=("total_steps", "sum")).reset_index())
        person_totals = (person_agg.groupby(["ds_id", "sensor"])
                         .agg(grand_dur=("total_dur_sec", "sum"),
                              grand_steps=("total_steps", "sum")).reset_index())
        person_agg = person_agg.merge(person_totals, on=["ds_id", "sensor"])
        person_agg["pct_dur"] = (person_agg["total_dur_sec"]
                                 / person_agg["grand_dur"].replace(0, np.nan)
                                 * 100)
        person_agg["pct_steps"] = (person_agg["total_steps"]
                                   / person_agg["grand_steps"].replace(0, np.nan)
                                   * 100)
        n_acts = len(activities)
        fig_h = min(max(10, n_acts * 0.5 + 4), 400)
        fig, axes = plt.subplots(1, 2, figsize=(22, fig_h))
        fig.patch.set_facecolor(DARK_BG)

        configs = [
            (axes[0], "total_dur_sec", "pct_dur", "Duration (min)", 60.0),
            (axes[1], "total_steps", "pct_steps", "Step Count", 1.0),
        ]
        for ax, val_col, pct_col, xlabel, divisor in configs:
            ax.set_facecolor(DARK_AX)
            for i, act in enumerate(activities):
                act_data = person_agg[person_agg["PA_TYPE"] == act]
                cat = activity_category(act)
                col = CATEGORY_COLORS.get(cat, "#888888")
                raw = act_data[val_col].dropna().values / divisor
                pct = act_data[pct_col].dropna().values
                if len(raw) == 0: continue
                bp = ax.boxplot(raw, positions=[i], vert=False, widths=0.6,
                                patch_artist=True,
                                medianprops={"color": TEXT_COL, "linewidth": 2},
                                whiskerprops={"color": "#aaa"},
                                capprops={"color": "#aaa"},
                                flierprops={"marker": "o", "markersize": 2.5,
                                            "markerfacecolor": "#666"})
                for patch in bp["boxes"]:
                    patch.set_facecolor(col); patch.set_alpha(0.78)
                med_pct = np.median(pct) if len(pct) else 0.0
                ax.text(np.median(raw), i+0.38, f"{med_pct:.1f}%",
                        color=TEXT_COL, ha="center", va="bottom", fontsize=6.5)
            for cat_name, _, _ in boundaries:
                local = [i for i, a in enumerate(activities)
                         if activity_category(a) == cat_name]
                if local and local[0] > 0:
                    ax.axhline(local[0]-0.5, color="#555", linewidth=1,
                               linestyle="--", alpha=0.7)
            ax.set_yticks(range(n_acts))
            ax.set_yticklabels([a.replace("_", " ") for a in activities],
                               fontsize=8, color=TEXT_COL)
            _style(ax, xlabel=xlabel)
            ax.spines[["top", "right"]].set_visible(False)

        fig.suptitle(f"Per-Person Distribution — {algo}",
                     color=TEXT_COL, fontsize=12, fontweight="bold")
        _save(fig, os.path.join(out_dir,
              f"plot5_{algo}_per_person_distribution.png"))


# ═══════════════════════════════════════════════════════════════════════
# RQ1 EXPERIMENT 1 — Overall pairwise Pearson + Bland-Altman
# ═══════════════════════════════════════════════════════════════════════

def rq1_exp1_overall_agreement(df_long, out_dir):
    algos = sorted(df_long["algorithm"].unique())
    sensors = sorted(df_long["sensor"].unique())
    pairs = list(combinations(algos, 2))
    if not pairs:
        print("  [RQ1-Exp1] Need ≥2 algorithms"); return

    df = df_long.copy()
    df["seg_key"] = (df["ds_id"] + "|" + df["sensor"] + "|" +
                     df["PA_TYPE"] + "|" + df["duration_sec"].astype(str))

    summary_rows = []

    for (a, b) in pairs:
        da = df[df["algorithm"] == a][["seg_key", "steps_per_min"]].rename(
            columns={"steps_per_min": "spm_a"})
        db = df[df["algorithm"] == b][["seg_key", "steps_per_min"]].rename(
            columns={"steps_per_min": "spm_b"})
        merged = da.merge(db, on="seg_key", how="inner")
        if len(merged) < 10: continue
        merged["sensor"] = merged["seg_key"].str.split("|").str[1]

        # Overall
        x, y = merged["spm_a"].values, merged["spm_b"].values
        r_all, p_all = sp_stats.pearsonr(x, y)
        diff_all = x - y
        bias_all, sd_all = np.mean(diff_all), np.std(diff_all, ddof=1)
        summary_rows.append({
            "algo_a": a, "algo_b": b, "sensor": "ALL",
            "n": len(merged), "pearson_r": round(r_all, 4), "pearson_p": p_all,
            "ba_bias": round(bias_all, 3),
            "ba_loa_lo": round(bias_all - 1.96*sd_all, 3),
            "ba_loa_hi": round(bias_all + 1.96*sd_all, 3),
            "ba_sd": round(sd_all, 3),
        })

        # Per sensor
        sensor_stats = {}
        for sensor, grp in merged.groupby("sensor"):
            if len(grp) < 5: continue
            xs, ys = grp["spm_a"].values, grp["spm_b"].values
            r_s, p_s = sp_stats.pearsonr(xs, ys)
            diff_s = xs - ys
            bias_s, sd_s = np.mean(diff_s), np.std(diff_s, ddof=1)
            sensor_stats[sensor] = {"r": r_s, "bias": bias_s, "sd": sd_s,
                                    "n": len(grp)}
            summary_rows.append({
                "algo_a": a, "algo_b": b, "sensor": sensor,
                "n": len(grp), "pearson_r": round(r_s, 4), "pearson_p": p_s,
                "ba_bias": round(bias_s, 3),
                "ba_loa_lo": round(bias_s - 1.96*sd_s, 3),
                "ba_loa_hi": round(bias_s + 1.96*sd_s, 3),
                "ba_sd": round(sd_s, 3),
            })

        # Bland-Altman figure
        n_s = len(sensor_stats)
        if n_s == 0: continue
        n_cols = min(n_s, 3)
        n_rows = int(np.ceil(n_s / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(6*n_cols, 5*n_rows), squeeze=False)
        fig.patch.set_facecolor(DARK_BG)
        for idx, (sensor, ss) in enumerate(sorted(sensor_stats.items())):
            ax = axes[idx // n_cols, idx % n_cols]
            ax.set_facecolor(DARK_AX)
            grp = merged[merged["sensor"] == sensor]
            mean_pair = (grp["spm_a"] + grp["spm_b"]).values / 2
            diff_pair = (grp["spm_a"] - grp["spm_b"]).values
            ax.scatter(mean_pair, diff_pair, s=12, alpha=0.4,
                       color=algo_color(a), edgecolors="none")
            ax.axhline(ss["bias"], color="#E84855", linewidth=1.5,
                       label=f"Bias: {ss['bias']:.2f}")
            ax.axhline(ss["bias"]+1.96*ss["sd"], color="#F4A261",
                       linewidth=1, linestyle="--",
                       label=f"+1.96SD: {ss['bias']+1.96*ss['sd']:.2f}")
            ax.axhline(ss["bias"]-1.96*ss["sd"], color="#F4A261",
                       linewidth=1, linestyle="--",
                       label=f"−1.96SD: {ss['bias']-1.96*ss['sd']:.2f}")
            ax.axhline(0, color="#888", linewidth=0.5, linestyle=":")
            _style(ax, xlabel="Mean (spm)", ylabel=f"{a}−{b} (spm)",
                   title=f"{sensor} (r={ss['r']:.3f}, n={ss['n']})")
            ax.legend(fontsize=7, facecolor=LEGEND_FACE, edgecolor=LEGEND_EDGE,
                      labelcolor=LEGEND_TEXT, loc="upper right")
        for idx in range(n_s, n_rows*n_cols):
            axes[idx//n_cols, idx%n_cols].set_visible(False)
        fig.suptitle(f"Bland-Altman: {a} vs {b}", color=TEXT_COL,
                     fontsize=13, fontweight="bold")
        _save(fig, os.path.join(out_dir,
              f"rq1_exp1_bland_altman_{a}_vs_{b}.png"))

    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        sdf.to_csv(os.path.join(out_dir, "rq1_exp1_agreement_summary.csv"),
                    index=False)
        low = sdf[(sdf["pearson_r"] < 0.7) & (sdf["sensor"] != "ALL")]
        if len(low):
            print(f"    ⚠ {len(low)} sensor×pair with r < 0.7:")
            for _, r in low.iterrows():
                print(f"      {r['algo_a']} vs {r['algo_b']} @ "
                      f"{r['sensor']}: r={r['pearson_r']:.3f}")


# ═══════════════════════════════════════════════════════════════════════
# RQ1 EXPERIMENT 2 — Agreement stratified by category + sensor
# ═══════════════════════════════════════════════════════════════════════

def rq1_exp2_category_agreement(df_long, out_dir):
    algos = sorted(df_long["algorithm"].unique())
    sensors = sorted(df_long["sensor"].unique())
    pairs = list(combinations(algos, 2))
    cats = [c for c in CATEGORY_ORDER
            if c in df_long["activity_category"].unique()]
    if not pairs or not cats:
        print("  [RQ1-Exp2] Need ≥2 algos and ≥1 category"); return

    df = df_long.copy()
    df["seg_key"] = (df["ds_id"] + "|" + df["sensor"] + "|" +
                     df["PA_TYPE"] + "|" + df["duration_sec"].astype(str))
    rows = []

    for sensor_filter in ["ALL"] + sensors:
        sub = df if sensor_filter == "ALL" else df[df["sensor"]==sensor_filter]
        for cat in cats:
            cat_sub = sub[sub["activity_category"] == cat]
            for (a, b) in pairs:
                da = cat_sub[cat_sub["algorithm"]==a][
                    ["seg_key", "steps_per_min"]].rename(
                    columns={"steps_per_min": "spm_a"})
                db = cat_sub[cat_sub["algorithm"]==b][
                    ["seg_key", "steps_per_min"]].rename(
                    columns={"steps_per_min": "spm_b"})
                m = da.merge(db, on="seg_key", how="inner")
                if len(m) < 5: continue
                x, y = m["spm_a"].values, m["spm_b"].values
                r_val, p_val = sp_stats.pearsonr(x, y)
                diff = x - y
                bias, sd = np.mean(diff), np.std(diff, ddof=1)
                rows.append({
                    "sensor": sensor_filter, "category": cat,
                    "algo_a": a, "algo_b": b,
                    "pair": f"{a} vs {b}", "n": len(m),
                    "pearson_r": round(r_val, 4), "pearson_p": p_val,
                    "ba_bias": round(bias, 3),
                    "ba_loa_lo": round(bias-1.96*sd, 3),
                    "ba_loa_hi": round(bias+1.96*sd, 3),
                    "ba_sd": round(sd, 3),
                })

    if not rows:
        print("  [RQ1-Exp2] No data"); return
    rdf = pd.DataFrame(rows)
    rdf.to_csv(os.path.join(out_dir, "rq1_exp2_category_agreement.csv"),
               index=False)

    # Heatmaps
    pair_labels = [f"{a} vs {b}" for (a, b) in pairs]
    for sf in ["ALL"] + sensors:
        sub = rdf[rdf["sensor"] == sf]
        if sub.empty: continue
        mat = sub.pivot_table(index="category", columns="pair",
                              values="pearson_r")
        mat = mat.reindex(index=[c for c in CATEGORY_ORDER if c in mat.index],
                          columns=[p for p in pair_labels if p in mat.columns])
        if mat.empty: continue

        fig, ax = plt.subplots(
            figsize=(max(6, len(mat.columns)*2.2+2),
                     max(3, len(mat.index)*1.0+2)))
        fig.patch.set_facecolor(DARK_BG)
        cmap = LinearSegmentedColormap.from_list("rq1",
               ["#E84855", "#F4A261", "#57CC99"])
        mv = mat.values.astype(float)
        im = ax.imshow(mv, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels(mat.columns, color=TEXT_COL, fontsize=9,
                           rotation=30, ha="right")
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index, color=TEXT_COL, fontsize=10)
        for r in range(mv.shape[0]):
            for c in range(mv.shape[1]):
                v = mv[r, c]
                if not np.isnan(v):
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                            fontsize=10, fontweight="bold",
                            color="#FFF" if v < 0.5 else "#111")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Pearson r", color=TEXT_COL, fontsize=9)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COL)
        label = "All sensors" if sf == "ALL" else sf
        ax.set_title(f"RQ1-Exp2: Pearson r by Category × Pair\n{label}",
                     color=TEXT_COL, fontsize=11, fontweight="bold", pad=10)
        for sp in ax.spines.values(): sp.set_color(SPINE_COL)
        _save(fig, os.path.join(out_dir,
              f"rq1_exp2_category_r_heatmap_{sf}.png"))

    poor = rdf[(rdf["pearson_r"] < 0.5) & (rdf["sensor"] == "ALL")]
    if len(poor):
        print(f"    ⚠ {len(poor)} category×pair with r < 0.5:")
        for _, r in poor.iterrows():
            print(f"      {r['category']:15s} {r['pair']:25s} "
                  f"r={r['pearson_r']:.3f} n={r['n']}")


# ═══════════════════════════════════════════════════════════════════════
# RQ2 — False-positive severity
# ═══════════════════════════════════════════════════════════════════════

def rq2_false_positive_severity(df_long, out_dir):
    nm = df_long[df_long["activity_category"] == "Non-Movement"].copy()
    if nm.empty:
        print("  [RQ2] No Non-Movement segments"); return
    algos = sorted(df_long["algorithm"].unique())

    # Part A: FP rate table
    nm["is_fp"] = (nm["total_steps"] > 0).astype(int)
    fp_rate = (nm.groupby(["sensor", "algorithm"])
               .agg(n_segments=("is_fp", "count"),
                    n_fp=("is_fp", "sum"),
                    fp_rate_pct=("is_fp", "mean"),
                    fp_total_steps=("total_steps", "sum"),
                    fp_mean_spm=("steps_per_min", "mean"),
                    fp_median_spm=("steps_per_min", "median"))
               .reset_index())
    fp_rate["fp_rate_pct"] = (fp_rate["fp_rate_pct"] * 100).round(1)
    fp_rate.to_csv(os.path.join(out_dir, "rq2_fp_rate_by_sensor.csv"),
                   index=False)

    # Part B: Per-participant severity
    sev_rows = []
    for (ds_id, sensor, algo), grp in df_long.groupby(
            ["ds_id", "sensor", "algorithm"]):
        total_steps = grp["total_steps"].sum()
        total_dur_hr = grp["duration_sec"].sum() / 3600
        nm_grp = grp[grp["activity_category"] == "Non-Movement"]
        fp_steps = nm_grp["total_steps"].sum()
        nm_dur_hr = nm_grp["duration_sec"].sum() / 3600
        pct_fp = (fp_steps / total_steps * 100) if total_steps > 0 else 0.0
        n_days = max(1, total_dur_hr / 16)
        sev_rows.append({
            "ds_id": ds_id, "sensor": sensor, "algorithm": algo,
            "total_steps": round(total_steps),
            "fp_steps": round(fp_steps),
            "fp_pct_of_total": round(pct_fp, 2),
            "fp_steps_per_day_est": round(fp_steps / n_days),
            "nm_hours_total": round(nm_dur_hr, 2),
            "nm_hours_per_day_est": round(nm_dur_hr / n_days, 2),
            "nm_segments": len(nm_grp),
            "nm_fp_segments": int((nm_grp["total_steps"] > 0).sum()),
        })
    sev_df = pd.DataFrame(sev_rows)
    sev_df.to_csv(os.path.join(out_dir,
                  "rq2_fp_severity_per_participant.csv"), index=False)

    # Part C: Box plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(DARK_BG)

    for ax_idx, (ax, col_name, ylabel, title_suffix) in enumerate([
        (axes[0], "fp_pct_of_total", "FP Steps as % of Total",
         "False-Positive Step Inflation (%)"),
        (axes[1], "fp_steps_per_day_est", "Estimated FP Steps/Day",
         "Estimated Daily FP Steps"),
    ]):
        ax.set_facecolor(DARK_AX)
        for i, algo in enumerate(algos):
            vals = sev_df[sev_df["algorithm"]==algo][col_name].dropna().values
            if len(vals) == 0: continue
            bp = ax.boxplot(vals, positions=[i], widths=0.55,
                            patch_artist=True,
                            medianprops={"color": TEXT_COL, "linewidth": 2},
                            whiskerprops={"color": "#aaa"},
                            capprops={"color": "#aaa"},
                            flierprops={"marker": "o", "markersize": 3,
                                        "markerfacecolor": "#666"})
            for patch in bp["boxes"]:
                patch.set_facecolor(algo_color(algo)); patch.set_alpha(0.8)
            med = np.median(vals)
            offset = 0.3 if ax_idx == 0 else 5
            ax.text(i, med+offset, f"{med:.1f}" if ax_idx==0 else f"{med:.0f}",
                    ha="center", va="bottom", color=TEXT_COL, fontsize=8)
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(algos, color=TEXT_COL)
        _style(ax, ylabel=ylabel, title=f"RQ2: {title_suffix}")
        ax.spines[["top", "right"]].set_visible(False)

    _save(fig, os.path.join(out_dir, "rq2_fp_severity_boxplots.png"))

    for algo in algos:
        asub = sev_df[sev_df["algorithm"] == algo]
        if asub.empty: continue
        print(f"    {algo:12s}  median FP%={asub['fp_pct_of_total'].median():.1f}%  "
              f"median FP/day≈{asub['fp_steps_per_day_est'].median():.0f}")


# ═══════════════════════════════════════════════════════════════════════
# RQ3 — Inter-algorithm disagreement analysis
# ═══════════════════════════════════════════════════════════════════════

def rq3_disagreement_analysis(df_long, out_dir):
    algos = sorted(df_long["algorithm"].unique())
    sensors = sorted(df_long["sensor"].unique())
    if len(algos) < 2:
        print("  [RQ3] Need ≥2 algorithms"); return

    # Mean spm per (activity, sensor, algorithm)
    agg = (df_long.groupby(
        ["PA_TYPE", "activity_category", "sensor", "algorithm"])
        .agg(mean_spm=("steps_per_min", "mean"),
             n_segments=("steps_per_min", "count")).reset_index())
    wide = agg.pivot_table(
        index=["PA_TYPE", "activity_category", "sensor"],
        columns="algorithm", values="mean_spm").reset_index()
    seg_counts = (agg.groupby(["PA_TYPE", "sensor"])["n_segments"]
                  .sum().reset_index().rename(
                      columns={"n_segments": "total_segments"}))
    wide = wide.merge(seg_counts, on=["PA_TYPE", "sensor"], how="left")

    algo_cols = [c for c in wide.columns if c in algos]
    av = wide[algo_cols].values
    with np.errstate(divide="ignore", invalid="ignore"):
        rm = np.nanmean(av, axis=1)
        rs = np.nanstd(av, axis=1, ddof=1)
        rcv = np.where(rm > 0, rs / rm * 100, np.nan)
        rrange = np.nanmax(av, axis=1) - np.nanmin(av, axis=1)
    wide["algo_mean_spm"] = np.round(rm, 2)
    wide["algo_sd_spm"] = np.round(rs, 2)
    wide["algo_cv_pct"] = np.round(rcv, 1)
    wide["algo_range_spm"] = np.round(rrange, 2)
    wide = wide[wide["algo_mean_spm"] > 0.1].copy()

    wide.sort_values("algo_cv_pct", ascending=False).to_csv(
        os.path.join(out_dir, "rq3_disagreement_full.csv"), index=False)

    median_cv = wide["algo_cv_pct"].median()
    high = wide[wide["algo_cv_pct"] > median_cv].sort_values(
        "algo_cv_pct", ascending=False)
    high.to_csv(os.path.join(out_dir, "rq3_disagreement_above_median.csv"),
                index=False)
    print(f"    Median CV = {median_cv:.1f}%  |  "
          f"{len(high)}/{len(wide)} above median")

    # Heatmap: activity × sensor
    pa_present = set(wide["PA_TYPE"].unique())
    activities, boundaries = _ordered_activities_by_category(pa_present)
    if not activities or not sensors: return

    mat = pd.DataFrame(index=activities, columns=sensors, dtype=float)
    for _, row in wide.iterrows():
        pa, s = row["PA_TYPE"], row["sensor"]
        if pa in mat.index and s in mat.columns:
            mat.loc[pa, s] = row["algo_cv_pct"]

    fig, ax = plt.subplots(
        figsize=(max(8, len(sensors)*2.5+3),
                 max(8, len(activities)*0.45+3)))
    fig.patch.set_facecolor(DARK_BG)
    mv = mat.values.astype(float)
    cmap = LinearSegmentedColormap.from_list("rq3",
           ["#57CC99", "#FEE440", "#F4A261", "#E84855"])
    vmax = min(np.nanpercentile(mv[~np.isnan(mv)], 95), 200) \
        if np.any(~np.isnan(mv)) else 100
    im = ax.imshow(mv, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(range(len(sensors)))
    ax.set_xticklabels(sensors, color=TEXT_COL, fontsize=9, rotation=30,
                       ha="right")
    ax.set_yticks(range(len(activities)))
    ax.set_yticklabels([a.replace("_", " ") for a in activities],
                       fontsize=7, color=TEXT_COL)
    for cat_name, start, _ in boundaries:
        if start > 0:
            ax.axhline(start-0.5, color="#555", linewidth=1, linestyle="--")
    for r in range(mv.shape[0]):
        for c in range(mv.shape[1]):
            v = mv[r, c]
            if not np.isnan(v):
                ax.text(c, r, f"{v:.0f}", ha="center", va="center",
                        fontsize=6, color="#FFF" if v > vmax*0.6 else "#111")
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("CV (%) across algorithms", color=TEXT_COL, fontsize=9)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT_COL)
    ax.set_title(f"RQ3: Inter-Algorithm Disagreement (CV%)\n"
                 f"Activity × Sensor  (median CV={median_cv:.1f}%)",
                 color=TEXT_COL, fontsize=11, fontweight="bold", pad=10)
    for sp in ax.spines.values(): sp.set_color(SPINE_COL)
    _save(fig, os.path.join(out_dir, "rq3_disagreement_heatmap.png"))

    # Top-N bar chart (all sensors combined)
    agg_all = (df_long.groupby(
        ["PA_TYPE", "activity_category", "algorithm"])
        .agg(mean_spm=("steps_per_min", "mean")).reset_index())
    wide_all = agg_all.pivot_table(
        index=["PA_TYPE", "activity_category"],
        columns="algorithm", values="mean_spm").reset_index()
    av2 = wide_all[algo_cols].values
    with np.errstate(divide="ignore", invalid="ignore"):
        m2 = np.nanmean(av2, axis=1)
        s2 = np.nanstd(av2, axis=1, ddof=1)
        cv2 = np.where(m2 > 0.1, s2/m2*100, np.nan)
    wide_all["cv_pct"] = np.round(cv2, 1)
    wide_all = wide_all.dropna(subset=["cv_pct"]).sort_values(
        "cv_pct", ascending=True)

    top_n = min(20, len(wide_all))
    top = wide_all.tail(top_n)

    fig, ax = plt.subplots(figsize=(12, max(6, top_n*0.4+2)))
    fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_AX)
    colors = [CATEGORY_COLORS.get(top.iloc[i]["activity_category"], "#888")
              for i in range(top_n)]
    ax.barh(range(top_n), top["cv_pct"].values, color=colors, alpha=0.8,
            edgecolor="#333", linewidth=0.5)

    for i in range(top_n):
        row = top.iloc[i]
        parts = [f"{a[:3]}:{row.get(a, np.nan):.1f}"
                 for a in algo_cols if not np.isnan(row.get(a, np.nan))]
        ax.text(row["cv_pct"]+1, i, "  "+", ".join(parts),
                va="center", fontsize=6, color=TEXT_COL, alpha=0.8)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(
        [top.iloc[i]["PA_TYPE"].replace("_", " ") for i in range(top_n)],
        fontsize=8, color=TEXT_COL)
    ax.axvline(median_cv, color="#E84855", linewidth=1.5, linestyle="--")

    legend_handles = [
        mpatches.Patch(color=CATEGORY_COLORS.get(c, "#888"), label=c)
        for c in CATEGORY_ORDER if c in set(top["activity_category"])
    ] + [plt.Line2D([0], [0], color="#E84855", linewidth=1.5, linestyle="--",
                    label=f"Median CV={median_cv:.1f}%")]
    ax.legend(handles=legend_handles, fontsize=8, facecolor=LEGEND_FACE,
              edgecolor=LEGEND_EDGE, labelcolor=LEGEND_TEXT, loc="lower right")

    _style(ax, xlabel="CV (%) across algorithms",
           title=f"RQ3: Top-{top_n} Activities by Disagreement\n"
                 "(all sensors; annotations = per-algo mean spm)")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, os.path.join(out_dir, "rq3_top_disagreement_activities.png"))

    print(f"\n    Top-10 disagreement (all sensors):")
    for _, row in wide_all.tail(10).iloc[::-1].iterrows():
        parts = [f"{a}={row[a]:.1f}" for a in algo_cols
                 if not np.isnan(row.get(a, np.nan))]
        print(f"      CV={row['cv_pct']:5.1f}%  "
              f"{row['PA_TYPE']:40s} [{row['activity_category']}]  "
              f"{', '.join(parts)}")


# ═══════════════════════════════════════════════════════════════════════
# DS PROCESSING
# ═══════════════════════════════════════════════════════════════════════

_print_lock = None

def _pool_init(lock):
    global _print_lock
    _print_lock = lock

def _process_ds_worker(args):
    ds_root, ds_id, algorithms, label_root = args
    records, log_lines = process_ds(ds_root, ds_id, algorithms, label_root)
    block = "\n".join(log_lines)
    if _print_lock is not None:
        with _print_lock: print(block, flush=True)
    else:
        print(block, flush=True)
    return records

def process_ds(ds_root, ds_id, algorithms, label_root=None):
    log = [f"\n  ┌─ {ds_id}"]
    label_path = find_label_file(ds_root, ds_id, label_root)
    if label_path is None:
        log.append(f"  └─ [{ds_id}] [SKIP] No label file found")
        return [], log
    log.append(f"  │  [{ds_id}] Labels: {label_path}")
    labels = load_labels(label_path)
    log.append(f"  │  [{ds_id}] {len(labels)} segments after filtering")

    records = []
    for algo in algorithms:
        sensor_files = find_steps_files(ds_root, ds_id, algo)
        if not sensor_files:
            log.append(f"  │  [{ds_id}] [SKIP] No files for {algo}")
            continue
        seen = {}
        for sensor, fpath in sensor_files:
            if sensor not in seen: seen[sensor] = fpath
        for sensor, fpath in seen.items():
            try:
                series = load_steps_series(fpath)
                recs = aggregate_segments(labels, series, ds_id, algo, sensor)
                records.extend(recs)
                log.append(f"  │  [{ds_id}] ✓ {algo:12s} {sensor:22s} "
                           f"{len(recs)} segs sum={series.sum():.0f}")
            except Exception as e:
                log.append(f"  │  [{ds_id}] [ERROR] {algo}/{sensor}: {e}")
    log.append(f"  └─ [{ds_id}] {len(records)} records")
    return records, log


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Step algorithm comparison analysis")
    parser.add_argument("--root",    default=DEFAULT_ROOT)
    parser.add_argument("--labels",  default=DEFAULT_LABELS)
    parser.add_argument("--ds",      nargs="*", default=None)
    parser.add_argument("--algos",   nargs="*", default=ALGORITHMS)
    parser.add_argument("--out",     default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("╔══════════════════════════════════════════════╗")
    print("║   STEP ALGORITHM COMPARISON — CLUSTER RUN   ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"  Root   : {args.root}")
    print(f"  Labels : {args.labels}")
    print(f"  Algos  : {args.algos}")
    print(f"  Workers: {args.workers}")
    print(f"  Out    : {args.out}")

    all_ds = discover_ds_folders(args.root)
    ds_folders = ([d for d in all_ds if os.path.basename(d) in args.ds]
                  if args.ds else all_ds)
    print(f"\n  {len(ds_folders)} DS folder(s): "
          f"{[os.path.basename(d) for d in ds_folders]}")

    n_workers = min(len(ds_folders), args.workers)
    worker_args = [(ds_root, os.path.basename(ds_root), args.algos, args.labels)
                   for ds_root in ds_folders]

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
    df_long.to_csv(os.path.join(args.out, "segments_long.csv"), index=False)
    print(f"\n  {len(df_long)} total segment records collected")

    # ── Original plots (1–5) ─────────────────────────────────────────────
    print("\n  Generating plots...")

    print("\n  [1/9] Activity box plots per algorithm...")
    plot1_label_boxplot_by_category(df_long, args.out)

    print("\n  [2/9] Sensor × algorithm FP-rate heatmap...")
    plot2_fp_heatmap(df_long, args.out)

    print("\n  [3/9] All-DS category box plots per algorithm...")
    plot3_allDS_per_algorithm(df_long, args.out)

    print("\n  [4/9] Pairwise scatter plots...")
    plot4_pairwise_scatter(df_long, args.out)

    print("\n  [5/9] Per-person distribution...")
    plot5_per_person_distribution(df_long, args.out)

    # ── RQ analyses (6–9) ────────────────────────────────────────────────
    print("\n  [6/9] RQ1-Exp1: Pairwise Pearson + Bland-Altman...")
    rq1_exp1_overall_agreement(df_long, args.out)

    print("\n  [7/9] RQ1-Exp2: Agreement by category × sensor...")
    rq1_exp2_category_agreement(df_long, args.out)

    print("\n  [8/9] RQ2: False-positive severity...")
    rq2_false_positive_severity(df_long, args.out)

    print("\n  [9/9] RQ3: Inter-algorithm disagreement...")
    rq3_disagreement_analysis(df_long, args.out)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n  ✓ Done! Output in: {args.out}")
    print()
    for f in sorted(os.listdir(args.out)):
        kb = os.path.getsize(os.path.join(args.out, f)) / 1024
        print(f"    {f:<55} {kb:6.1f} KB")


if __name__ == "__main__":
    main()