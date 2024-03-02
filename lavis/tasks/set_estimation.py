"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import json
import os
from collections import defaultdict
from re import L
import numpy as np
import torch
# from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, f1_score, multilabel_confusion_matrix
import matplotlib.pyplot as plt
from lavis.common.registry import registry
import lavis.common.dist_utils as dist_utils
from lavis.common.vqa_tools.vqa_clean import VQACleaner
from lavis.tasks.base_task import BaseTask

import wandb


@registry.register_task("epic_kitchens")
class EpicKitchensTask(BaseTask):
    clean = VQACleaner()
    TRUTH = {
        'true': 'yes',
        'false': 'no',
        'y': 'yes',
        'n': 'no',
    }

    def __init__(self, num_beams, max_len, min_len):
        super().__init__()
        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg
        return cls(
            num_beams=run_cfg.num_beams,
            max_len=run_cfg.max_len,
            min_len=run_cfg.min_len,
        )

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)
        # for name, dataset in datasets.items():
        #     dataset
        first = datasets[list(datasets)[0]]
        self.classes = first['val'].classes
        return datasets

    def valid_step(self, model, samples):
        results = []

        answer_pred = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )

        cls_pred = None
        if samples.get('targets') is not None and model.fixed_cls is not None:
            cls_pred = model.class_head(samples)                

        for i in range(len(answer_pred)):
            r = {
                "question": samples["text_input"][i],
                "answer_pred": answer_pred[i], 
                "answer_true": samples["text_output"][i],
                "image_id": samples["image_id"][i].item(), 
                "narration_id": samples["narration_id"][i], 
                "noun": samples["noun"][i], 
            }
            if cls_pred is not None:
                r.update({
                    "cls_pred": cls_pred[i].cpu().numpy().tolist(),
                    "cls_true": samples['targets'][i].cpu().numpy().tolist(),
                    # "cls_labels": samples['class_labels'][i].cpu().numpy().tolist(),
                })
            # yt=samples['targets'][i].cpu().numpy()
            # yp=cls_pred[i].cpu().numpy()
            # print(np.array(self.classes)[yt!=-1])
            # print(yt[yt!=-1])
            # print((yp[yt!=-1] >= 0.5).astype(int))
            # print(np.round(yp[yt!=-1], 3))
            # if input('>?'):from IPython import embed;embed()
            results.append(r)
            if samples['sample_id'][i].item() in self.sample_index:
                print("Sample:", samples['sample_id'][i], samples["narration_id"][i], samples["narration"][i])
                print("in:", samples['text_input'][i])
                print("pred:", answer_pred[i])
                print("true:", samples["text_output"][i])
                yt=samples['targets'][i].cpu().numpy()
                yp=cls_pred[i].cpu().numpy()
                print(np.array(self.classes)[yt!=-1])
                print(yt[yt!=-1])
                print(np.round(yp[yt!=-1], 1))
                self.result_table.add_data(
                    samples['sample_id'][i],
                    wandb.Video(norm_video(samples["image"][i]).cpu().numpy(), fps=3) 
                    if samples["image"].ndim == 5 else
                    wandb.Image(samples["image"][i].cpu()),
                    samples["text_input"][i],
                    answer_pred[i],  # Predicted answer
                    samples["text_output"][i],  # True answer
                    samples["narration_id"][i],  # Narration ID
                    samples["narration"][i]  # Narration text
                )
        
        return results

    def before_evaluation(self, model, dataset, **kwargs):
        super().before_evaluation(model, dataset, **kwargs)
        self.sample_index = np.random.choice(len(dataset), min(60, len(dataset)), replace=False)
        print("eval samples:", self.sample_index, len(dataset))
        self.result_table = wandb.Table(columns=["index", "image", "question", "answer_pred", "answer_true", "narration_id", "narration"])

    def after_evaluation(self, val_result, split_name, **kwargs):
        # Log the table
        print("write table", len(self.result_table.data))
        wandb.log({"predictions": self.result_table})

        result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_epic_kitchens_result",
            remove_duplicate="", 
        )
        metrics = None
        if split_name == 'val':
            metrics = self._report_metrics(result_file=result_file, split=split_name)
        return metrics

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        results = json.load(open(result_file, "r"))

        # calculate metrics
        errors = []
        acc = []
        for res in results:
            try:
                # acc.append(self._yes_no_statement_jaccard(res["answer_pred"], res["answer_true"]))
                acc.append(self._jaccard(res["answer_pred"], res["answer_true"]))
            except Exception as e:
                print("Could not parse", res["answer_pred"], e)
                errors.append(res["answer_pred"])
                acc.append(0)
        
        cls_metrics = {}
        if results[0].get('cls_true') is not None:
            try:
                y_true = np.array([d['cls_true'] for d in results]).astype(int)
                y_pred = np.array([d['cls_pred'] for d in results]).astype(float)
                cls_metrics = compute_cls_metrics(y_true, y_pred)
                plot_ml_cm(y_true, y_pred, self.classes)
            except Exception:
                import traceback
                traceback.print_exc()

        # report metrics
        accuracy = np.mean(acc)
        metrics = {"agg_metrics": accuracy, "accuracy": accuracy, "split": split, **cls_metrics}

        with open(os.path.join(registry.get_path("output_dir"), f"log.txt"), "a") as f:
            f.write(json.dumps(metrics) + "\n")
        with open(os.path.join(registry.get_path("output_dir"), f"parse_error.txt"), "a") as f:
            for x in errors:
                f.write(f'{x}\n')

        logging.info(metrics)

        return metrics
    
    def _jaccard(self, y_pred, y_true):
        # XXX: repetition/contradiction
        y_pred = {self.clean(x) for x in split_text(y_pred, '.', ',', n=-1)}
        y_true = {self.clean(x) for x in split_text(y_true, '.', ',', n=-1)}
        return len(y_pred & y_true) / (len(y_pred | y_true) or 1)
    
    # def _yes_no_statement_jaccard(self, y_pred, y_true):
    #     ans_pred = self._extract_yes_no_answer_format(y_pred)
    #     # print('pred:', ans_pred, y_pred)
    #     ans_true = self._extract_yes_no_answer_format(y_true)
    #     # print('true:', ans_true)
    #     intersection = sum(ans_pred[k] == ans_true[k] for k in set(ans_pred) & set(ans_true))
    #     union = len(set(ans_pred) | set(ans_true))
    #     return intersection / (union or 1)
    
    # def _extract_yes_no_answer_format(self, text):
    #     # group answers by key
    #     answers = defaultdict(lambda: set())
    #     for x in split_text(text, '.', n=-1):
    #         try:
    #             key, value = split_text(x, ':', n=1)
    #             value, *_ = split_text(value, ',', ' ', n=1)
    #             key = self.clean(key)
    #             value = self.clean(value)
    #             value = self.TRUTH.get(value, value)
    #             answers[key].add(value)
    #         except Exception as e:
    #             print("Could not parse substring", x, 'from', text, e)
        
    #     # filter answers
    #     final_answers = {}
    #     for k, values in answers.items():
    #         values = [v for v in values if v in ('yes', 'no')]
    #         if len(values) > 1:
    #             continue
    #         if values:
    #             final_answers[k] = values[0]
    #     return final_answers

    # def _compare_text(self, y_pred, y_true):
    #     y_pred = self.clean(y_pred)
    #     y_true = self.clean(y_true)
    #     return y_pred == y_true

    # def _assignment_accuracy(self, txt_pred, txt_true, delim=','):
    #     txt_pred = split_text(txt_pred, delim)
    #     txt_true = split_text(txt_true, delim)
    #     if not txt_pred or not txt_true:
    #         return int(len(txt_true) == len(txt_pred))

    #     accuracy_matrix = np.array([
    #         [self._compare_text(yp, yt) for yp in txt_pred]
    #         for yt in txt_true
    #     ])
    #     row, col = linear_sum_assignment(accuracy_matrix, maximize=True)
    #     accuracy = accuracy_matrix[row, col]
    #     return np.mean(accuracy)

def split_text(text, *delims, n=-1):
    for d in delims:
        if d in text:
            return [y.strip() for y in text.split(d, n) if y.strip()]
    return ([text] + ['']*n) if n else [text]


def norm_video(img):
    img = img.float()
    for t in img:
        low, high = float(t.min()), float(t.max())
        t.clamp_(min=low, max=high)
        t.sub_(low).div_(max(high - low, 1e-5))
    img = img.mul(255).clamp(0, 255).byte()
    return img


def compute_cls_metrics(y_true, y_pred, threshold=0.5):
    y_true = y_true.astype(int)
    y_pred = (y_pred > threshold).astype(int)
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    y_pred = y_pred[y_true != -1]
    y_true = y_true[y_true != -1]
    return {
        'cls_acc': accuracy_score(y_true, y_pred), 
        'cls_f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=1), 
        'cls_f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=1),
    }


def plot_ml_cm(y_true, y_pred, labels, ncols=8, threshold=0.5, s=3):
    # Compute multilabel confusion matrix
    y_pred = (y_pred > threshold).astype(int)
    # mask = y_true == -1
    # y_true_masked = np.where(mask, y_pred, y_true)
    # mcm = multilabel_confusion_matrix(y_true_masked, y_pred)
    # mcm = np.stack([
    #     np.stack([(y_true == 0) & (y_pred == 0), (y_true == 0) & (y_pred == 1)], axis=1),
    #     np.stack([(y_true == 1) & (y_pred == 0), (y_true == 1) & (y_pred == 1)], axis=1),
    # ], axis=1)
    mcm = np.zeros((len(labels), 2, 2))
    for yt, yp in zip(y_true, y_pred):
        for j, (yti, ypi) in enumerate(zip(yt, yp)):
            if yti != -1:
                # print(labels[j], yti, ypi)
                mcm[j, yti, ypi] += 1
    mcm = mcm.astype(float) / np.maximum(1, mcm.sum(-1, keepdims=True))

    # Plotting the confusion matrices for each label
    nrows = int(np.ceil(len(mcm)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(s * ncols, s * nrows))
    for i, (label, ax) in enumerate(zip(labels, axes.flat)):
        # Confusion matrix for each label
        cm = mcm[i]
        
        # Display the confusion matrix
        cax = ax.matshow(cm, cmap='bone_r', vmin=0, vmax=1)
        # fig.colorbar(cax, ax=ax)
        
        # Annotate the matrix with text
        for (j, k), val in np.ndenumerate(cm):
            ax.text(k, j, f'{val:.0%}', ha='center', va='center', color='red')

        # Set labels and titles
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(f'{label}')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

    # fig.colorbar(im, ax=axes.ravel().tolist())
    plt.tight_layout()

    plt.savefig("confusion_matrix.png")
    wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})