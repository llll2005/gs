#!/bin/bash
# Find the true test camera->image pairing. Needs the GPU to itself (7.18M-primitive merged model).
#
# tools/eval_official_test.py only searches offsets -1/0/+1 and reported 16.60/16.63/16.42 -- a
# 0.03 dB spread. I twice explained that away (first "the indexing formula is wrong", then "dense
# flight paths make neighbouring frames similar"). The user looked at a side-by-side and settled
# it: left and right are different places. All three offsets are wrong, and the tool's refusal to
# certify was correct.
#
# The naming makes the assumption of a SMALL offset unjustified:
#   cameras (sparse/0)   0000.png .. 0740.png   4-digit, 0-based
#   images_1.2/ symlinks 0001.png .. 0741.png   4-digit, 1-based -> input/000001.png
# The camera names do not exist in the image directory at all, so the correspondence is being
# guessed from ordering alone. It may be a large constant offset, or not a constant offset -- the
# camera order may simply differ from the file order.
#
# So: render a handful of probe cameras once each, score them against EVERY image, and look at the
# argmax. A constant offset shows up as the same delta for every probe; anything else means the
# ordering assumption itself has to go, and pairing must come from the poses (match each test
# camera to the training camera nearest it and inspect) rather than from filenames.
set -u
cd "$(dirname "$0")/.." || exit 1
PYTHONPATH=. conda run -n gspl --no-capture-output python tools/find_test_pairing.py \
  --ckpt outputs/b3x3_full/checkpoints/merged.ckpt --probes 6
