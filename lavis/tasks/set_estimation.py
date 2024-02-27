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


    def valid_step(self, model, samples):
        results = []

        answer_pred = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=self.num_beams,
            max_length=self.max_len,
            min_length=self.min_len,
        )

        for i in range(len(answer_pred)):
            results.append({
                "answer_pred": answer_pred[i], 
                "answer_true": samples["text_output"][i],
                "image_id": samples["image_id"][i].item(), 
                "narration_id": samples["narration_id"][i], 
                "noun": samples["noun"][i], 
            })
            if samples['sample_id'][i].item() in self.sample_index:
                print("Sample:", samples['sample_id'][i], samples["narration_id"][i], samples["narration"][i])
                print("pred:", answer_pred[i])
                print("true:", samples["text_output"][i])
                self.result_table.add_data(
                    samples['sample_id'][i],
                    wandb.Video(norm_video(samples["image"][i]).cpu().numpy(), fps=3) 
                    if samples["image"].ndim == 5 else
                    wandb.Image(samples["image"][i].cpu()),
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
        self.result_table = wandb.Table(columns=["index", "image", "answer_pred", "answer_true", "narration_id", "narration"])

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

        # report metrics
        accuracy = np.mean(acc)
        metrics = {"agg_metrics": accuracy, "accuracy": accuracy, "split": split}

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