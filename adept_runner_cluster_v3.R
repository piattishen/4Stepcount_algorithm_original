# adept_runner_cluster_v2.R
# =============================================================================
# CLUSTER-SAFE build of the TRUSTED local adept_runner.R.
#
# The walking-detection algorithm is IDENTICAL to the local runner you
# validated with (same location detection, same stride_template[[loc]][[3]],
# same default segmentWalking() call, same 2-steps-per-stride conversion and
# output format), so it reproduces your local results. Only two things changed,
# both about *environment*, not the algorithm:
#
#   1. Packages install into / load from a WRITABLE user library, because the
#      container's /usr/local/lib/R/site-library is read-only. CRAN-archived
#      dvmisc is fetched via remotes::install_version.
#   2. The ADEPT *.R source files are located via the ADEPT_R_DIR env var (and
#      the cluster algorithms dir), in addition to the original "walk up from
#      this script" search, so it works no matter where the runner sits.
#
# Called by pipeline_cluster_v3.py via:
#   apptainer exec -B ... <rocker.sif> Rscript --vanilla \
#       adept_runner_cluster_v2.R <in_csv> <out_csv> <hz> [location] [n_cores]
# The extra <location>/<n_cores> args are accepted and ignored: location is
# detected from the input filename, exactly as in the local runner.
#
# Optional environment variables:
#   R_ADEPT_LIB  writable R library     (default: ~/R/adept_lib)
#   ADEPT_R_DIR  folder with ADEPT *.R  (default: cluster algorithms dir below)
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript adept_runner_cluster_v2.R <input.csv> <output.csv> <hz> [location] [n_cores]")
}

input_csv  <- args[1]
output_csv <- args[2]
xyz_fs     <- as.numeric(args[3])
# args[4] (location) and args[5] (n_cores) are intentionally ignored: this
# runner detects location from the filename, matching the validated behavior.

# ---- Writable user library (container site-library is read-only) -------------
user_lib <- Sys.getenv("R_ADEPT_LIB",
                       unset = file.path(Sys.getenv("HOME"), "R", "adept_lib"))
dir.create(user_lib, recursive = TRUE, showWarnings = FALSE)
# Put the user library first, and also pick up an existing ~/R/library if you
# populated one earlier, so already-installed packages are found and reused.
extra_libs <- c(user_lib, file.path(Sys.getenv("HOME"), "R", "library"))
.libPaths(c(extra_libs[dir.exists(extra_libs)], .libPaths()))
message("ADEPT R library paths: ", paste(.libPaths(), collapse = " : "))

# ---- Ensure required packages are present (installing into user_lib) ---------
repos_url     <- "https://cloud.r-project.org"
required_pkgs <- c("pracma", "assertthat", "dplyr", "magrittr",
                   "dvmisc", "adeptdata")
have <- function(p) requireNamespace(p, quietly = TRUE)
missing_pkgs <- required_pkgs[!vapply(required_pkgs, have, logical(1))]

if (length(missing_pkgs) > 0) {
  message("Installing missing R packages into ", user_lib, ": ",
          paste(missing_pkgs, collapse = ", "))
  if (!have("remotes")) {
    install.packages("remotes", lib = user_lib, repos = repos_url, quiet = TRUE)
  }
  for (p in missing_pkgs) {
    if (identical(p, "dvmisc")) {
      # dvmisc is archived on CRAN -> install latest archived version
      remotes::install_version("dvmisc", lib = user_lib, repos = repos_url,
                               upgrade = "never", quiet = TRUE)
    } else {
      install.packages(p, lib = user_lib, repos = repos_url,
                       quiet = TRUE, dependencies = TRUE)
    }
  }
  still_missing <- required_pkgs[!vapply(required_pkgs, have, logical(1))]
  if (length(still_missing) > 0) {
    stop("Could not install: ", paste(still_missing, collapse = ", "),
         ". Pre-install them into ", user_lib, " inside the container.")
  }
}

suppressPackageStartupMessages({
  library(pracma)      # cart2sph (segmentWalking)
  library(assertthat)  # input validation (segmentPattern)
  library(dplyr)       # data manipulation (segmentPattern)
  library(magrittr)    # pipe operator (segmentPattern)
})

# ---- Source algorithm R files ------------------------------------------------
# Determine the directory this script lives in.
script_dir <- tryCatch({
  all_args   <- commandArgs(trailingOnly = FALSE)
  file_flag  <- grep("^--file=", all_args, value = TRUE)
  if (length(file_flag) > 0) {
    dirname(normalizePath(sub("^--file=", "", file_flag[1]), mustWork = FALSE))
  } else {
    getwd()
  }
}, error = function(e) getwd())

r_files <- c("windowSmooth.R", "scaleTemplate.R",
              "similarityMatrix.R", "maxAndTune.R",
              "segmentPattern.R", "segmentWalking.R")

# Roots to search, in priority order:
#   1. ADEPT_R_DIR (and its parent) if set
#   2. this script's dir and up to three parents (original local behavior)
#   3. the cluster algorithms directory
adept_env <- Sys.getenv("ADEPT_R_DIR", unset = "")
cluster_algo_root <- "/home/wang.yichen8/4_algo_original_code"
candidate_roots <- unique(c(
  if (nzchar(adept_env)) c(adept_env, dirname(adept_env)) else character(0),
  script_dir,
  dirname(script_dir),
  dirname(dirname(script_dir)),
  dirname(dirname(dirname(script_dir))),
  cluster_algo_root
))
# "." lets a root point directly at the folder holding the .R files.
subdirs <- c(".", "ADEPT", file.path("algorithms_original_code", "ADEPT"))

find_r_file <- function(f) {
  for (root in candidate_roots) for (sd in subdirs) {
    p <- file.path(root, sd, f)
    if (file.exists(p)) return(p)
  }
  NULL
}

for (f in r_files) {
  path <- find_r_file(f)
  if (is.null(path)) {
    stop(sprintf("Cannot find R file '%s'. Searched roots %s under subdirs %s. ",
                 f, paste(candidate_roots, collapse=', '),
                 paste(subdirs, collapse=', ')),
         "Set ADEPT_R_DIR to the folder containing the ADEPT *.R files.")
  }
  source(path)
}
message("ADEPT source files loaded.")

# ---- Load optional packages for dvmisc (used by similarityMatrix / windowSmooth)
if (requireNamespace("dvmisc", quietly = TRUE)) {
  suppressPackageStartupMessages(library(dvmisc))
}

# ---- Read accelerometer data -------------------------------------------------
xyz <- read.csv(input_csv)
if (!all(c("x", "y", "z") %in% names(xyz))) {
  stop(paste("Input CSV must have columns x, y, z. Found:", paste(names(xyz), collapse = ", ")))
}
xyz <- as.matrix(xyz[, c("x", "y", "z")])

n_samples <- nrow(xyz)
n_sec     <- ceiling(n_samples / xyz_fs)

# ---- Detect sensor location from input filename ------------------------------
detect_location <- function(path) {
  base <- basename(path)
  known <- c("left_wrist", "left_hip", "left_ankle", "right_ankle")
  hit <- known[sapply(known, function(k) grepl(k, base, fixed = TRUE))]
  if (length(hit) >= 1) return(hit[1])
  return("left_wrist")
}

sensor_loc <- detect_location(input_csv)
message("ADEPT template location: ", sensor_loc)

# ---- Build stride templates --------------------------------------------------
template <- tryCatch({
  if (requireNamespace("adeptdata", quietly = TRUE)) {
    suppressPackageStartupMessages(library(adeptdata))
    tmpl_mat <- adeptdata::stride_template[[sensor_loc]][[3]]   # 3 templates
    lapply(seq_len(nrow(tmpl_mat)), function(i) as.numeric(tmpl_mat[i, ]))
  } else {
    stop("adeptdata not available")
  }
}, error = function(e) {
  message("adeptdata not available - using cosine templates")
  n_pts <- 200
  lapply(0:2, function(i) {
    phase <- i * (pi / 6)
    cos(seq(0, 2 * pi, length.out = n_pts) + phase)
  })
})

# ---- Run segmentWalking ------------------------------------------------------
message(sprintf("Running ADEPT on %d samples at %.0f Hz ...", n_samples, xyz_fs))

out <- tryCatch(
  segmentWalking(xyz, xyz.fs = xyz_fs, template = template, verbose = FALSE),
  error = function(e) {
    message("segmentWalking error: ", conditionMessage(e))
    NULL
  }
)

# ---- Convert strides to per-second step counts (1 stride = 2 steps) ---------
steps_per_sec <- rep(0.0, n_sec)

if (!is.null(out) && nrow(out) > 0) {
  walking_rows <- out[out$is_walking_i == 1, , drop = FALSE]
  if (nrow(walking_rows) > 0) {
    for (i in seq_len(nrow(walking_rows))) {
      tau_i   <- as.integer(walking_rows$tau_i[i])   # 1-based index in R
      sec_idx <- floor((tau_i - 1) / xyz_fs) + 1     # 1-based second
      sec_idx <- min(max(sec_idx, 1L), n_sec)
      steps_per_sec[sec_idx] <- steps_per_sec[sec_idx] + 2.0
    }
  }
}

total_steps <- sum(steps_per_sec)
message(sprintf("ADEPT done.  Walking strides: %d  Total steps: %.0f",
                if (!is.null(out)) sum(out$is_walking_i) else 0L,
                total_steps))

# ---- Write output ------------------------------------------------------------
result_df <- data.frame(
  second = seq_len(n_sec) - 1L,   # 0-based second offset
  steps  = steps_per_sec
)
write.csv(result_df, output_csv, row.names = FALSE)
message("Output written to: ", output_csv)