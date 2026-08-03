#!/bin/bash
# Launch the full cross-domain transfer grid (run AFTER transfer_smoke passes).
# scratch arms have no dependency; transfer/zeroshot wait for the pretrain ckpt.
set -e
cd /share/hpcproject/zzhang66/nfm
PRE=$(sbatch --parsable transfer_pre.sub)
echo "pretrain job: $PRE"
for PD in 0.25 1 2 7; do
  for S in 0 1 2; do
    sbatch --job-name=tr_sc_${PD}_${S} --export=ALL,MODE=scratch,PREDAYS=$PD,SEED=$S transfer_ft.sub
    sbatch --job-name=tr_tf_${PD}_${S} --dependency=afterok:$PRE --export=ALL,MODE=transfer,PREDAYS=$PD,SEED=$S transfer_ft.sub
  done
done
sbatch --job-name=tr_zs --dependency=afterok:$PRE --export=ALL,MODE=zeroshot,PREDAYS=0.25,SEED=0 transfer_ft.sub
squeue -u zzhang66 -o "%.10i %.14j %.8T %.10M %R"
