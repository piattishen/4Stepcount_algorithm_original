#!/usr/bin/env Rscript
# =============================================================================
# verisense_runner_cluster.R  —  CLUSTER VERSION
# Called by pipeline_cluster.py via:
#   apptainer exec -B ... <rocker.sif> Rscript --vanilla verisense_runner_cluster.R \
#       <in_csv> <out_csv> <hz>
#
# Sources verisense_count_steps.R from the cluster algorithm directory,
# resamples input to 15 Hz, runs the algorithm, and writes per-second counts.
#
# Required R packages (pre-installed in ~/R/library):
#   signal  — for resampling when input Hz != 15
# =============================================================================

# ---- User library -----------------------------------------------------------
user_lib <- file.path(Sys.getenv("HOME"), "R", "library")
if (dir.exists(user_lib)) .libPaths(c(user_lib, .libPaths()))

# ---- Parse arguments --------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript verisense_runner_cluster.R <in_csv> <out_csv> <hz>")
}
in_csv  <- args[1]
out_csv <- args[2]
hz      <- as.integer(args[3])

message("[Verisense] in_csv  = ", in_csv)
message("[Verisense] out_csv = ", out_csv)
message("[Verisense] hz      = ", hz)

# ---- Source Verisense R file ------------------------------------------------
verisense_dir <- "/home/wang.yichen8/4_algo_original_code/verisense"
verisense_r   <- file.path(verisense_dir, "verisense_count_steps.R")
if (!file.exists(verisense_r)) {
  stop(paste("[Verisense] Cannot find:", verisense_r))
}
source(verisense_r)
message("[Verisense] Loaded: ", verisense_r)

# ---- Read accelerometer data ------------------------------------------------
xyz_df <- read.csv(in_csv)
if (!all(c("x", "y", "z") %in% names(xyz_df))) {
  stop(paste("[Verisense] Input CSV must have columns x, y, z. Found:",
             paste(names(xyz_df), collapse = ", ")))
}

# ---- Resample to 15 Hz if needed -------------------------------------------
target_hz <- 15L

if (hz != target_hz) {
  message(sprintf("[Verisense] Resampling from %d Hz to %d Hz ...", hz, target_hz))
  n_in  <- nrow(xyz_df)
  t_in  <- seq(0, (n_in - 1) / hz, length.out = n_in)
  n_out <- round(n_in * target_hz / hz)
  t_out <- seq(0, t_in[n_in], length.out = n_out)
  xyz_df <- data.frame(
    x = approx(t_in, xyz_df$x, xout = t_out)$y,
    y = approx(t_in, xyz_df$y, xout = t_out)$y,
    z = approx(t_in, xyz_df$z, xout = t_out)$y
  )
  message(sprintf("[Verisense] Resampled: %d -> %d samples.", n_in, nrow(xyz_df)))
}

# ---- Run Verisense ----------------------------------------------------------
# Parameters match myscript.R: c(3, 5, 15, -0.5, 3, 4, 0.001, 1.2)
coeffs <- c(3, 5, 15, -0.5, 3, 4, 0.001, 1.2)
n_samples <- nrow(xyz_df)
message(sprintf("[Verisense] Running on %d samples at %d Hz ...", n_samples, target_hz))

steps_per_sec <- tryCatch(
  verisense_count_steps(input_data = xyz_df, coeffs = coeffs),
  error = function(e) {
    message("[Verisense] Error: ", conditionMessage(e))
    n_sec <- round(n_samples / target_hz)
    rep(0.0, n_sec)
  }
)

total_steps <- sum(steps_per_sec)
message(sprintf("[Verisense] Done. Total steps: %.0f", total_steps))

# ---- Write output -----------------------------------------------------------
result_df <- data.frame(
  second = seq_along(steps_per_sec) - 1L,   # 0-based second offset
  steps  = steps_per_sec
)
write.csv(result_df, out_csv, row.names = FALSE)
message("[Verisense] Output written to: ", out_csv)
