#!/usr/bin/env bash
# End-to-end dry run on tiny synthetic data (CPU is fine, no downloads, ~minutes).
# Run it once after installing, to check that your environment and the code work:
#   bash scripts/smoke_test.sh
# Everything is written to data/smoke/ and never touches your real data/ or outputs/.
set -euo pipefail
cd "$(dirname "$0")/.."
S=data/smoke
rm -rf "$S"

python scripts/make_fake_data.py --out $S/raw

P="--out-dir $S/processed --splits-dir $S/splits"
python -m src.prepare grading      --src $S/raw/ddr_grading --size 64  $P
python -m src.prepare segmentation --src $S/raw/ddr_seg     --size 128 $P
python -m src.prepare grading      --src $S/raw/aptos       --size 64  $P --name aptos --external
python -m src.prepare segmentation --src $S/raw/idrid_seg   --size 128 $P --name idrid --external

COMMON=(train.out_dir=$S/runs train.epochs=2 train.num_workers=0 train.amp=false model.pretrained=false
        train.batch_size=8 eval.batch_size=8)
bash scripts/run_grading_all.sh --set "${COMMON[@]}" data.image_size=64 \
     data.train_csv=$S/splits/grading_train.csv data.val_csv=$S/splits/grading_val.csv
bash scripts/run_seg_all.sh --set "${COMMON[@]}" data.crop_size=64 eval.batch_size=2 \
     data.train_csv=$S/splits/seg_train.csv data.val_csv=$S/splits/seg_val.csv

python -m src.compare --track grading      --runs $S/runs --out $S/out
python -m src.compare --track segmentation --runs $S/runs --out $S/out

python -m src.evaluate --run $S/runs/g2_oversample --split test
python -m src.evaluate --run $S/runs/g2_oversample --csv $S/splits/ext_aptos_grading.csv --tag aptos
python -m src.evaluate --run $S/runs/s5_lesion_crops --split test
python -m src.evaluate --run $S/runs/s5_lesion_crops --csv $S/splits/ext_idrid_seg.csv --tag idrid
echo "SMOKE TEST PASSED"
