#!/usr/bin/env bash
# run_all.sh — every dataset, every stage, then the paper tables.
#
#   ./run_all.sh                                  # defaults below
#   DATASETS="mitbih baker" WORKERS=16 ./run_all.sh
#   GRID=configs/grid_quick.json RESULTS=results_quick ./run_all.sh   # pipeline smoke test
#   SELFTEST=1 ./run_all.sh                       # run every self-test first
#   SELFTEST=1 DATASETS="" ./run_all.sh           # self-tests only
#
# Data must already be prepared into the split dirs named in configs/datasets.json
# (python prepare.py <dataset> ...). Every stage is resumable: re-running skips finished work.
set -euo pipefail
cd "$(dirname "$0")"
# keep caches/config/temp files off the home directory
[[ -f scratch_env.sh ]] && source scratch_env.sh

DATASETS=${DATASETS-"mitbih baker gw synthetic memory"}   # DATASETS="" → self-tests only
STAGES=${STAGES:-"select train score baselines teardown report"}
WORKERS=${WORKERS:-8}
GRID=${GRID:-configs/grid.json}
RESULTS=${RESULTS:-results}
PAPER=${PAPER:-paper}
STRICT=${STRICT:-}                       # STRICT=1 → refuse to train on a dirty git tree
LOG=${LOG:-logs}
mkdir -p "$LOG"

if [[ -n "${SELFTEST:-}" ]]; then
  for m in qvr.circuit qvr.data qvr.train qvr.infer qvr.stats qvr.baselines qvr.teardown; do
    echo "== self-test $m"; python -m "$m" | tail -1
  done
  python prepare.py --self-test | tail -1
  python run.py --self-test | tail -1
fi

for ds in $DATASETS; do
  dir=$(python -c "import json;print(json.load(open('configs/datasets.json'))['$ds']['data_dir'])")
  if [[ ! -f "$dir/Xtr.pickle" ]]; then
    echo "== $ds: $dir not prepared — skipping (python prepare.py $ds ...)"; continue
  fi
  echo "== $ds  ($(date '+%F %T'))"
  python -u run.py --dataset "$ds" --grid "$GRID" --out "$RESULTS/$ds" --workers "$WORKERS" \
      --stages $STAGES ${STRICT:+--strict-git} 2>&1 | tee "$LOG/$ds.log"
done

[[ -z "${DATASETS// /}" ]] && exit 0
python collate.py --results "$RESULTS" --out "$PAPER"
echo "== done: $PAPER/tables/ALL.md"
