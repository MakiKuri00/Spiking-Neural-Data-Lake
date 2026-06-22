#!/usr/bin/env bash
# GPU scaling-law sweep — accuracy vs neuron count for the conductance Diehl & Cook
# model (the path to the literature ~95%). Charts how accuracy scales with neurons:
#   400 -> ~87% | 1600 -> ~92% | 6400 -> ~95%   (Diehl & Cook 2015)
#
# Requires a CUDA box with the cu128 torch build + bindsnet (see README "GPU").
# Run AFTER any long single run frees the GPU. `python -u` = UNBUFFERED, so per-step
# progress and the final accuracy print live (the 6400 run was invisible due to buffering).
#
#   conda/venv-activate first, then:  bash gpu_scaling_sweep.sh
set -e

for M in 400 1600 6400; do
  echo "================ M=$M neurons ================"
  NORD_M=$M NORD_TRAIN=60000 NORD_TEST=10000 \
    python -u eth_mnist_bindsnet.py --gpu 2>&1 \
    | grep -E "neurons=|step [0-9]+/|TEST ACCURACY|paper:"
done
echo "scaling sweep complete — accuracy-vs-neurons charted."
