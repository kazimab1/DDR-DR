#!/usr/bin/env bash
# Train every Track A config (G0-G5) one after another. Finished runs are skipped.
#   bash scripts/run_grading_all.sh                      # seed 42
#   SEEDS="42 43 44" bash scripts/run_grading_all.sh     # several seeds (each gets its own folder)
#   bash scripts/run_grading_all.sh --set train.epochs=2 # extra arguments go to src.train
set -euo pipefail
cd "$(dirname "$0")/.."
for cfg in configs/grading/g*.yaml; do
  for seed in ${SEEDS:-42}; do
    python -m src.train --config "$cfg" --seed "$seed" "$@"
  done
done
