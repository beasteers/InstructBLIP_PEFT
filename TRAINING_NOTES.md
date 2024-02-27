


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