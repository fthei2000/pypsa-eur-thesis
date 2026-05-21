#!/bin/bash
#BSUB -J test-cdr
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

# fallback if not in repo root
if [[ ! -f "pixi.toml" ]]; then
  FALLBACK_DIR="/work3/s240459/pypsa-eur-thesis"
  if [[ -f "$FALLBACK_DIR/pixi.toml" ]]; then
    cd "$FALLBACK_DIR"
  else
    echo "Error: no pixi.toml found."
    exit 2
  fi
fi

# 👉 YOUR TEST CONFIG HERE
CFG="config/Myruns/config.test_cdr_philipp.yaml"

if [[ ! -s "$CFG" ]]; then
  echo "Error: '$CFG' missing or empty."
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
  echo "Error: pixi not found."
  exit 127
fi

echo "Pixi: $PIXI_BIN"
"$PIXI_BIN" --version

WORK_ROOT="${WORK_ROOT:-/work3/$USER}"
export TMPDIR="${TMPDIR:-$WORK_ROOT/tmp}"
mkdir -p "$TMPDIR"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORK_ROOT/.cache}"
export SNAKEMAKE_OUTPUT_CACHE="${SNAKEMAKE_OUTPUT_CACHE:-$WORK_ROOT/.snakemake_cache}"
mkdir -p "$XDG_CACHE_HOME"
mkdir -p "$SNAKEMAKE_OUTPUT_CACHE"

echo "Starting snakemake..."

"$PIXI_BIN" run snakemake \
  -j 8 \
  --rerun-incomplete \
  --keep-going \
  --printshellcmds \
  --configfile "$CFG"

echo "Test CDR run completed successfully."