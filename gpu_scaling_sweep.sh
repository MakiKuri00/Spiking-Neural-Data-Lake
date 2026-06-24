#!/usr/bin/env bash
# GPU scaling-law sweep — accuracy vs neuron count for the conductance Diehl & Cook model.
# Uses the SCALE-AWARE recipe (lower inhibition + higher theta_plus as neurons grow), not
# the defaults — defaults collapse at scale (6400/default measured 47.8%).
#   measured: 400 -> 86.4% | 1600 -> 90.0% (verified here)
#   target:   6400 -> ~95% (Diehl & Cook 2015; needs the recipe + multi-epoch, day-scale)
#
# Requires a CUDA box (cu128 torch + bindsnet). `python -u` = unbuffered live progress.
#   bash gpu_scaling_sweep.sh
set -e

for M in 400 1600 6400; do
  case $M in
    400)  INH=120; TP=0.05; EP=1 ;;   # defaults work at this size
    1600) INH=60;  TP=0.20; EP=1 ;;   # verified -> 90.0%
    6400) INH=40;  TP=0.30; EP=3 ;;   # extrapolated recipe + 3 epochs (~day-scale)
  esac
  echo "================ M=$M  INH=$INH THETA_PLUS=$TP EPOCHS=$EP ================"
  NORD_M=$M NORD_INH=$INH NORD_THETA_PLUS=$TP NORD_EPOCHS=$EP \
    NORD_TRAIN=60000 NORD_TEST=10000 \
    python -u eth_mnist_bindsnet.py --gpu 2>&1 \
    | grep -E "neurons=|step [0-9]+/|TEST ACCURACY|paper:"
done
echo "scaling sweep complete — accuracy-vs-neurons charted."
