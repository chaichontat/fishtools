#!/usr/bin/env bash
#
# Sequential GAM runs with hardcoded inputs/outputs.
# Uses conda env `seq` and disables conda plugins to avoid seccomp hangs.
#
# Runs:
# 1) all_excit_r300.h5ad, leiden==7
# 2) all_excit_r300.h5ad, leiden==4
# 3) all_neurons.h5ad (full)
#
# Notes:
# - Export drops non-finite rows by default; no t/x filtering is applied unless you pass flags.
# - Fit uses --threads for process count and --bam-threads for mgcv::bam(nthreads).

set -euo pipefail

_die() {
    echo "ERROR: $*" >&2
    exit 1
}

_run() {
    echo "$(date -Is) > $*" >&2
    command "$@" || _die "command failed: $*"
}

export CONDA_NO_PLUGINS=true
export OMP_NUM_THREADS=1
export OMP_THREAD_LIMIT=1
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

repo_dir=/fast2/cs_outputs/fishtools2
out_root=/fast2/cs_outputs/fishtools2/_out/gam_runs
export TMPDIR=/fast2/cs_outputs/fishtools2/_out/tmp

h5ad_excit=/fast2/cs_outputs/all_excit_r300.h5ad
h5ad_neurons=/fast2/cs_outputs/all_neurons.h5ad

threads=4
bam_threads=4

cd "$repo_dir" || _die "could not cd to $repo_dir"

_run mkdir -p "$out_root"
_run mkdir -p "$TMPDIR"

run7="$out_root/gam_all_excit_r300_leiden7"
_run conda run -n seq python -u scripts/gam/export_panel_from_pooled_h5ad.py \
    "$h5ad_excit" \
    --out-dir "$run7/panel" \
    --subset-col leiden \
    --subset-values 7 \
    --genes all
_run conda run -n seq Rscript scripts/gam/fit_inm_panel.R \
    "$run7/panel" \
    "$run7/fit_results.tsv" \
    --no-pos \
    --threads "$threads" \
    --bam-threads "$bam_threads"

run4="$out_root/gam_all_excit_r300_leiden4"
_run conda run -n seq python -u scripts/gam/export_panel_from_pooled_h5ad.py \
    "$h5ad_excit" \
    --out-dir "$run4/panel" \
    --subset-col leiden \
    --subset-values 4 \
    --genes all
_run conda run -n seq Rscript scripts/gam/fit_inm_panel.R \
    "$run4/panel" \
    "$run4/fit_results.tsv" \
    --no-pos \
    --threads "$threads" \
    --bam-threads "$bam_threads"

# runN="$out_root/gam_all_neurons"
# _run conda run -n seq python -u scripts/gam/export_panel_from_pooled_h5ad.py \
#     "$h5ad_neurons" \
#     --out-dir "$runN/panel" \
#     --no-theta \
#     --genes all
# _run conda run -n seq Rscript scripts/gam/fit_inm_panel.R \
#     "$runN/panel" \
#     "$runN/fit_results.tsv" \
#     --no-pos \
#     --no-theta \
#     --threads "$threads" \
#     --bam-threads "$bam_threads"

# echo "$(date -Is) DONE" >&2
# echo "cluster7: $run7"
# echo "cluster4: $run4"
# echo "all_neurons: $runN"
