#!/bin/bash
#BSUB -J sector-L6-myopic-2030-2050
#BSUB -q hpc
#BSUB -n 8
#BSUB -W 48:00
#BSUB -R "rusage[mem=10000]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

set -euo pipefail

module purge
module load python3/3.10.18
module load gurobi/12.0.3

RUN_DIR="${LS_SUBCWD:-$PWD}"
cd "$RUN_DIR"

if [[ ! -f "pixi.toml" ]]; then
  FALLBACK_DIR="/work3/s240459/pypsa-eur-thesis"
  if [[ -f "$FALLBACK_DIR/pixi.toml" ]]; then
    cd "$FALLBACK_DIR"
  else
    echo "Error: no pixi.toml found in '$RUN_DIR' and fallback '$FALLBACK_DIR'."
    echo "Submit from the repository root, e.g.: cd /work3/s240459/pypsa-eur-thesis && bsub < run_L6.sh"
    exit 2
  fi
fi

CFG="config/Myruns/config.L6.yaml"
if [[ ! -s "$CFG" ]]; then
  echo "Error: '$CFG' is missing or empty."
  exit 2
fi

echo "Host: $(hostname)"
echo "CWD:  $(pwd)"
echo "User: $(whoami)"

export PATH="$HOME/.pixi/bin:$HOME/.local/bin:$PATH"

PIXI_BIN="$(command -v pixi || true)"
if [[ -z "$PIXI_BIN" ]] && [[ -x "$HOME/.pixi/bin/pixi" ]]; then
  PIXI_BIN="$HOME/.pixi/bin/pixi"
fi

if [[ -z "$PIXI_BIN" ]]; then
  echo "Error: pixi not found in PATH for user '$USER'."
  exit 127
fi

echo "Pixi: $PIXI_BIN"
"$PIXI_BIN" --version

WORK_ROOT="${WORK_ROOT:-/work3/$USER}"
export TMPDIR="${TMPDIR:-$WORK_ROOT/tmp}"
mkdir -p "$TMPDIR"

echo "Starting snakemake..."

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORK_ROOT/.cache}"
export SNAKEMAKE_OUTPUT_CACHE="${SNAKEMAKE_OUTPUT_CACHE:-$WORK_ROOT/.snakemake_cache}"
mkdir -p "$XDG_CACHE_HOME"
mkdir -p "$SNAKEMAKE_OUTPUT_CACHE"

"$PIXI_BIN" run snakemake --version

run_snakemake() {
  "$PIXI_BIN" run snakemake \
    -j 8 \
    --rerun-incomplete \
    --keep-going \
    --printshellcmds \
    --configfile "$CFG"
}

tmp_log="$(mktemp)"
set +e
run_snakemake 2>&1 | tee "$tmp_log"
status=${PIPESTATUS[0]}
set -e

if [[ "$status" -ne 0 ]] && grep -q "Directory cannot be locked" "$tmp_log"; then
  echo "Detected stale Snakemake lock. Running --unlock and retrying once..."
  "$PIXI_BIN" run snakemake --unlock --configfile "$CFG"
  run_snakemake
elif [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

echo "L6 country-storage and expanded-generation run completed successfully."
