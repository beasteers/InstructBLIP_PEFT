

Batch 1: Train

```bash


# ir
# 0 emp
# 1 nj
# 2 oj
# 3 WAI
# 4 WAI
# 5 WAI
# 6 nj
# 7 oj
# 8 oj
# 9 WAI
# 10 WAI
# 11 WAI

# bs
# 0 nj
# 1 nj
# 2 WAI
# 3 WAI
# 4 WAI
# 5 j
# 6 j
# 7 j
# 8 j
# 9 WAI
# 10 emp


# 3 frame
# bs 5 | 6 | 7
# python -m torch.distributed.run --nproc_per_node=1 --master_port=25678 train.py --cfg-path lavis/projects/instructblip/train/epic-kitchens/finetune_instructblip_epic_kitchens_4.yaml
python -m torch.distributed.run --nproc_per_node=1 --master_port=25678 train.py --cfg-path lavis/projects/instructblip/train/epic-kitchens/finetune_instructblip_epic_kitchens_4_ho_masks.yaml
python -m torch.distributed.run --nproc_per_node=1 --master_port=25678 train.py --cfg-path lavis/projects/instructblip/train/epic-kitchens/finetune_instructblip_epic_kitchens_4_ho_boxes.yaml
python -m torch.distributed.run --nproc_per_node=1 --master_port=25678 train.py --cfg-path lavis/projects/instructblip/train/epic-kitchens/finetune_instructblip_epic_kitchens_2.yaml
# python -m torch.distributed.run --nproc_per_node=1 --master_port=25678 train.py --cfg-path lavis/projects/instructblip/train/epic-kitchens/finetune_instructblip_epic_kitchens_2_nocls.yaml

# 1 frame
# bs 0 | 1, ir 1
# python -m torch.distributed.run --nproc_per_node=1 --master_port=25678 train.py --cfg-path lavis/projects/instructblip/train/epic-kitchens/finetune_instructblip_epic_kitchens_3.yaml
python -m torch.distributed.run --nproc_per_node=1 --master_port=25678 train.py --cfg-path lavis/projects/instructblip/train/epic-kitchens/finetune_instructblip_epic_kitchens_3_ho_masks.yaml
python -m torch.distributed.run --nproc_per_node=1 --master_port=25678 train.py --cfg-path lavis/projects/instructblip/train/epic-kitchens/finetune_instructblip_epic_kitchens_3_ho_boxes.yaml
python -m torch.distributed.run --nproc_per_node=1 --master_port=25678 train.py --cfg-path lavis/projects/instructblip/train/epic-kitchens/finetune_instructblip_epic_kitchens_1.yaml
# python -m torch.distributed.run --nproc_per_node=1 --master_port=25678 train.py --cfg-path lavis/projects/instructblip/train/epic-kitchens/finetune_instructblip_epic_kitchens_2_nocls.yaml

# SSv2
# ir 6
python -m torch.distributed.run --nproc_per_node=1 --master_port=25678 train.py --cfg-path lavis/projects/instructblip/train/s20bn/finetune_instructblip_s20bn_1.yaml
```

Batch 2: Eval
```bash
# 
python -m torch.distributed.run --nproc_per_node=1 --master_port=25678 train.py --cfg-path lavis/projects/instructblip/train/epic-kitchens/finetune_instructblip_epic_kitchens_4.yaml
python -m torch.distributed.run --nproc_per_node=1 --master_port=25678 train.py --cfg-path lavis/projects/instructblip/train/epic-kitchens/finetune_instructblip_epic_kitchens_4_ho_boxes.yaml
python -m torch.distributed.run --nproc_per_node=1 --master_port=25678 train.py --cfg-path lavis/projects/instructblip/train/epic-kitchens/finetune_instructblip_epic_kitchens_2.yaml

# 
python -m torch.distributed.run --nproc_per_node=1 --master_port=25678 train.py --cfg-path lavis/projects/instructblip/train/epic-kitchens/finetune_instructblip_epic_kitchens_4.yaml
python -m torch.distributed.run --nproc_per_node=1 --master_port=25678 train.py --cfg-path lavis/projects/instructblip/train/epic-kitchens/finetune_instructblip_epic_kitchens_4_ho_boxes.yaml
python -m torch.distributed.run --nproc_per_node=1 --master_port=25678 train.py --cfg-path lavis/projects/instructblip/train/epic-kitchens/finetune_instructblip_epic_kitchens_2.yaml

```
























 - lavis/projects/instructblip/train/epic-kitchens/finetune_instructblip_epic_kitchens_1.yaml

```bash
sbatch -J s20bn run_scripts/instructblip/train/run_finetune_instructblip_experiments.sbatch s20bn 1
sbatch -J ek1 run_scripts/instructblip/train/run_finetune_instructblip_experiments.sbatch epic_kitchens 1
sbatch -J ek2 run_scripts/instructblip/train/run_finetune_instructblip_experiments.sbatch epic_kitchens 2
sbatch -J ek3 run_scripts/instructblip/train/run_finetune_instructblip_experiments.sbatch epic_kitchens 3
sbatch -J ek4 run_scripts/instructblip/train/run_finetune_instructblip_experiments.sbatch epic_kitchens 4
```






### Original ScienceQA Training
 - dataset config path: lavis/configs/datasets/scienceqa/defaults.yaml
 - model config path: lavis/configs/models/blip2/blip2_instruct_flant5xl_qformer_lora.yaml
 - experiment path: lavis/projects/instructblip/train/scienceqa/finetune_instructblip_scienceqa_15.yaml
 - model path: lavis/models/blip2_models/blip2_t5_instruct_qformer_lora.py
 - dataset path: lavis/datasets/datasets/scienceqa_datasets.py

 - runner class: lavis.runners.runner_base.RunnerBase
 - model class: lavis.models.blip2_models.blip2_t5_instruct_qformer_lora.Blip2T5InstructQformerLoRA
 - datasets dict: {scienceqa: {'test': lavis.datasets.datasets.scienceqa_datasets.ScienceQADataset, 'train', 'val'}}
 - task class: lavis.tasks.vqa.ScienceQATask

 - predict_class:
    ```
    answer_list = ['(a)', '(b)', '(c)', '(d)', '(e)']
    all_losses = torch.cat(all_losses, dim=-1)
    output_class_ranks = torch.argsort(all_losses, dim=-1)
    top_predicted_classes = [candidates[idx] for idx in output_class_ranks[:, 0].tolist()]
    ```
 
dataset format:
```

```

### Something-Something Training
 - dataset config: lavis/configs/datasets/s20bn/defaults.yaml

 - dataset: lavis/datasets/datasets/vqa_datasets.py

 - create new experiment configs
 - format dataset

dataset format:
```

```

 - image, text_in, text_out -> text_input is same for qformer/llm
 - loss
 - 


TODO:
 - New Task
 - New Dataset


### EKG Training

dataset
 - builder: lavis/datasets/builders/vqa_builder.py
 - dataset class: lavis/datasets/datasets/epic_kitchens_datasets.py
 - default config: lavis/configs/datasets/epic-kitchens/defaults.yaml

model

job
 - experiment configs: lavis/projects/instructblip/train/epic-kitchens
 - sbatch: run_scripts/instructblip/train/run_finetune_instructblip_experiments.sbatch