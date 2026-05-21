#!/bin/bash
#BSUB -J myrun
#BSUB -q hpc
#BSUB -n 8
#BSUB -W 06:00
#BSUB -R "rusage[mem=4000]"
#BSUB -o /work3/s240459/logs/%J.out
#BSUB -e /work3/s240459/logs/%J.err

set -euo pipefail

module purge
module load python3/3.10.18
module load gurobi/12.0.3

cd /work3/s240459/pypsa-eur-thesis

echo "Host: $(hostname)"
echo "CWD:  $(pwd)"
echo "User: $(whoami)"
echo "PATH: $PATH"

# Make pixi available if needed:
export PATH="$HOME/.pixi/bin:$PATH"
export PATH="$HOME/.local/bin:$PATH"

echo "Pixi: $(command -v pixi || true)"
pixi --version

echo "Starting snakemake..."
pixi run snakemake --version

run_snakemake() {
  pixi run snakemake -j 8 --rerun-incomplete --keep-going --printshellcmds \
    --configfile config/test/config.electricity.yaml
}

tmp_log="$(mktemp)"
set +e
run_snakemake 2>&1 | tee "$tmp_log"
status=${PIPESTATUS[0]}
set -e

if [[ "$status" -ne 0 ]] && grep -q "Directory cannot be locked" "$tmp_log"; then
  echo "Detected stale Snakemake lock. Running --unlock and retrying once..."
  pixi run snakemake --unlock --configfile config/test/config.electricity.yaml
  run_snakemake
elif [[ "$status" -ne 0 ]]; then
  exit "$status"
fi