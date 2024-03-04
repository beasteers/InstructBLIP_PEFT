#!/bin/bash

#SBATCH --job-name=instructblip
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=1-12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err

#GREENE GREENE_GPU_MPS=YES

benchmark=${1}
experiment=${2}

[[ -z $benchmark ]] && exit 1;
[[ -z $experiment ]] && exit 1;

CONFIG=lavis/projects/instructblip/train/${benchmark}/finetune_instructblip_${benchmark}_${experiment}.yaml
./sing <<< "
python -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path $CONFIG
"