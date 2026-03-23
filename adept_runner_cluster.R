#!/usr/bin/env Rscript
# =============================================================================
# adept_runner_cluster.R  —  CLUSTER VERSION
# Called by pipeline_cluster.py via:
#   apptainer exec -B ... <rocker.sif> Rscript --vanilla adept_runner_cluster.R \
#       <in_csv> <out_csv> <hz>
#
# Key differences from adept_runner.R (local version):
#   - Sources ADEPT R files from the cluster path instead of a relative path
#   - Extends .libPaths() to ~/R/library for user-installed packages
#   - No auto-install (cluster jobs should not write to network filesystems)
#
# Required R packages — pre-install once:
#   apptainer exec -B /home:/home <rocker.sif> Rscript -e "
#     dir.create('~/R/library', recursive=TRUE, showWarnings=FALSE)
#     .libPaths('~/R/library')
#     install.packages(
#       c('pracma', 'dvmisc', 'assertthat', 'adeptdata'),
#       repos = 'https://cloud.r-project.org',
#       lib   = '~/R/library'
#     )
#   "
# =============================================================================

# ---- User library -----------------------------------------------------------
user_lib <- file.path(Sys.getenv("HOME"), "R", "library")
if (dir.exists(user_lib)) .libPaths(c(user_lib, .libPaths()))

# ---- Parse arguments --------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript adept_runner_cluster.R <in_csv> <out_csv> <hz> [n_cores]")
}
in_csv  <- args[1]
out_csv <- args[2]
hz      <- as.integer(args[3])
n_cores <- if (length(args) >= 4) as.integer(args[4]) else 1L

message("[ADEPT] in_csv  = ", in_csv)
message("[ADEPT] out_csv = ", out_csv)
message("[ADEPT] hz      = ", hz)
message("[ADEPT] n_cores = ", n_cores)

# ---- Load required packages -------------------------------------------------
required_pkgs <- c("pracma", "assertthat", "dplyr", "magrittr", "dvmisc")
for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    stop(paste0(
      "[ADEPT] Required package '", pkg, "' is not installed.\n",
      "  Pre-install into ~/R/library (see comment block at top of this script)."
    ))
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

# ---- Source ADEPT R files from cluster path ---------------------------------
# Edit this path if the algorithm directory moves.
adept_dir <- "/home/wang.yichen8/4_algo_original_code/ADEPT"

r_files <- c(
  "windowSmooth.R",
  "scaleTemplate.R",
  "similarityMatrix.R",
  "maxAndTune.R",
  "segmentPattern.R",
  "segmentWalking.R"
)

for (f in r_files) {
  full_path <- file.path(adept_dir, f)
  if (!file.exists(full_path)) {
    stop(paste("[ADEPT] Cannot find:", full_path))
  }
  source(full_path)
}

message("[ADEPT] ADEPT R source files loaded from: ", adept_dir)

# ---- Load walking templates -------------------------------------------------
template <- tryCatch({
  if (requireNamespace("adeptdata", quietly = TRUE)) {
    suppressPackageStartupMessages(library(adeptdata))
    tmpl_mat <- adeptdata::stride_template$left_wrist[[3]]
    tpl <- lapply(seq_len(nrow(tmpl_mat)), function(i) as.numeric(tmpl_mat[i, ]))
    message("[ADEPT] Using adeptdata stride templates (left_wrist, 3-template set).")
    tpl
  } else {
    stop("adeptdata not available")
  }
}, error = function(e) {
  message("[ADEPT] adeptdata not available (", conditionMessage(e), ").")
  message("[ADEPT] Falling back to cosine templates.")
  n_pts <- 200
  lapply(0:2, function(i) {
    phase <- i * (pi / 6)
    cos(seq(0, 2 * pi, length.out = n_pts) + phase)
  })
})

# ---- Read accelerometer data ------------------------------------------------
xyz_df <- read.csv(in_csv)
if (!all(c("x", "y", "z") %in% names(xyz_df))) {
  stop(paste("Input CSV must have columns x, y, z. Found:",
             paste(names(xyz_df), collapse = ", ")))
}
xyz_mat   <- as.matrix(xyz_df[, c("x", "y", "z")])
n_samples <- nrow(xyz_mat)
n_sec     <- ceiling(n_samples / hz)

message(sprintf("[ADEPT] Running segmentWalking: %d samples at %d Hz (%d seconds).",
                n_samples, hz, n_sec))

# ---- Run ADEPT walking segmentation ----------------------------------------
out <- tryCatch(
  segmentWalking(
    xyz                = xyz_mat,
    xyz.fs             = hz,
    template           = template,
    run.parallel       = (n_cores > 1L),
    run.parallel.cores = n_cores,
    verbose            = FALSE
  ),
  error = function(e) {
    message("[ADEPT] segmentWalking error: ", conditionMessage(e))
    data.frame(
      tau_i        = integer(0),
      T_i          = integer(0),
      sim_i        = numeric(0),
      template_i   = integer(0),
      is_walking_i = integer(0)
    )
  }
)

n_walking <- if (nrow(out) > 0) sum(out$is_walking_i == 1, na.rm = TRUE) else 0L
message("[ADEPT] Total strides detected: ", nrow(out),
        "  |  Walking strides: ", n_walking)

# ---- Convert walking strides to per-second step counts ---------------------
# 1 walking stride = 2 steps.
# Steps are placed at the second containing the stride start (tau_i).
steps_per_sec <- rep(0.0, n_sec)

if (nrow(out) > 0) {
  walking_rows <- out[!is.na(out$is_walking_i) & out$is_walking_i == 1L, , drop = FALSE]
  if (nrow(walking_rows) > 0) {
    for (i in seq_len(nrow(walking_rows))) {
      tau_i   <- as.integer(walking_rows$tau_i[i])   # 1-based sample index
      sec_idx <- floor((tau_i - 1L) / hz) + 1L       # 1-based second
      sec_idx <- min(max(sec_idx, 1L), n_sec)
      steps_per_sec[sec_idx] <- steps_per_sec[sec_idx] + 2.0
    }
  }
}

total_steps <- sum(steps_per_sec)
message(sprintf("[ADEPT] Done.  Total steps: %.0f", total_steps))

# ---- Write output -----------------------------------------------------------
# All seconds written (pipeline_cluster.py expects second + steps columns).
result_df <- data.frame(
  second = seq_len(n_sec) - 1L,   # 0-based second offset
  steps  = steps_per_sec
)
write.csv(result_df, out_csv, row.names = FALSE)
message("[ADEPT] Output written to: ", out_csv)
