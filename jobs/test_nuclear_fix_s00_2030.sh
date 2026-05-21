#!/bin/bash
#BSUB -J nuc-test-s00-2030
#BSUB -q hpc
#BSUB -n 8
#BSUB -W 06:00
#BSUB -R "rusage[mem=15000]"
#BSUB -R "span[hosts=1]"
#BSUB -o /work3/s240459/pypsa-eur-thesis/logs/supply_curve/nuc-test-s00-2030_%J.out
#BSUB -e /work3/s240459/pypsa-eur-thesis/logs/supply_curve/nuc-test-s00-2030_%J.err

set -euo pipefail

module purge
module load python3/3.10.18
module load gurobi/12.0.3

REPO_ROOT="/work3/s240459/pypsa-eur-thesis"
cd "$REPO_ROOT"

export PATH="$HOME/.pixi/bin:$HOME/.local/bin:$PATH"
PIXI_BIN="$(command -v pixi || true)"
[[ -z "$PIXI_BIN" && -x "$HOME/.pixi/bin/pixi" ]] && PIXI_BIN="$HOME/.pixi/bin/pixi"

export TMPDIR="${TMPDIR:-/work3/$USER/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/work3/$USER/.cache}"
export SNAKEMAKE_OUTPUT_CACHE="${SNAKEMAKE_OUTPUT_CACHE:-/work3/$USER/.snakemake_cache}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$SNAKEMAKE_OUTPUT_CACHE"

echo "Host:    $(hostname)"
echo "Started: $(date)"

CONFIG="$REPO_ROOT/config/Myruns/supply_curve_test/config.S00-seq-000eur.yaml"
RUN_NAME="S00t-cdr-000eur-seq-168seg"

# Delete the 2030 solved network to force re-solve (brownfield already patched)
rm -f "$REPO_ROOT/results/$RUN_NAME/networks/base_s_96__168seg_2030.nc"
echo "Cleared solved 2030 network — will re-solve from patched brownfield"

set +e
"$PIXI_BIN" run snakemake \
  -j 8 \
  --nolock \
  --rerun-incomplete \
  --rerun-triggers mtime \
  --keep-going \
  --printshellcmds \
  --configfile "$CONFIG" \
  2>&1
STATUS=$?
set -e

echo "Completed: $(date) (exit $STATUS)"
exit $STATUS
