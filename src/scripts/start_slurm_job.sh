#!/bin/bash
# Deprecated: replaced by src/scripts/submit.sh and the train.sh / visualize.sh /
# evaluate.sh wrappers. Kept only so in-flight references don't dangle; do not
# add new jobs here.
#
# Use instead, e.g.:
#   src/scripts/train.sh training=fast data=cityscapes
#   src/scripts/visualize.sh <run_dir>
#   src/scripts/evaluate.sh <checkpoint> evaluate=consistency data=pascal_voc
echo "src/scripts/start_slurm_job.sh is deprecated; use train.sh/visualize.sh/evaluate.sh instead." >&2
exit 1
