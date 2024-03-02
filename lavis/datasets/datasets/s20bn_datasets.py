import os
import json
import tqdm
import random
import numpy as np
import pandas as pd
from lavis.common.predicate_utils.predicates import load_pddl_yaml, Predicate
from lavis.common.predicate_utils.prompts import get_prompt_function
from lavis.datasets.datasets.base_video_detection_prompt_datasets import VideoFrameDataset
from lavis.datasets.datasets.base_dataset import BaseDataset


class SSVQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, split=None, **kw):
        super().__init__(vis_processor, text_processor, vis_root=vis_root, ann_paths=ann_paths)
        self.dataset = S20bnDataset(ann_paths[0], **kw)
        self.annotations = self.dataset.annotations
        self.split = split
        self._add_instance_ids()

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i):
        return self.dataset[i]


class S20bnDataset(VideoFrameDataset):
    def __init__(self, annotation_path, json_dir, get_prompt=None, fake_downsample_count=None, outer_buffer=60, inner_buffer=30, **kw):
        annotations = load_epic_kitchens_dataset(annotation_path, fake_downsample_count, outer_buffer, inner_buffer)
        super().__init__(annotations, get_prompt_function(get_prompt), json_dir, **kw)

    def load_detections(self, ann):
        return load_detections(self.json_dir, ann["narration_id"])


# 


def load_epic_kitchens_dataset(annotation_path, count=None, outer_buffer=60, inner_buffer=30):
    df = load_annotation_csv(annotation_path)

    annotation_dir = os.path.dirname(annotation_path)
    actions, predicates = load_pddl_yaml(f"{annotation_dir}/EPIC_100_conditions.yaml")
    
    # if count:
    #     count = count * len(df) if count <= 1 else count
    #     df = df.sample(min(count, len(df)))
    
    annotations = []
    for ann in tqdm.tqdm(df.to_dict('records'), desc=annotation_path.split(os.sep)[-1], leave=False):
        for verb in [ann['verb'], ann['verb_norm']]:
            if verb not in actions:
                continue
            action = actions[verb]
            pre = action.get_state(action.vars[0], 'pre')
            post = action.get_state(action.vars[0], 'post')

            duration = ann['stop_frame'] - ann['start_frame']
            inner_buffer = int(min(inner_buffer, duration // 4))
            if pre:
                annotations.append(dict(
                    ann, 
                    start_frame=ann['start_frame'] - outer_buffer,
                    stop_frame=ann['start_frame'] + inner_buffer,
                    pre_post='pre', 
                    action=action,
                    state=pre,
                    unknown_state=set(random.sample(predicates, len(predicates))) - set(pre),
                ))
            if post:
                annotations.append(dict(
                    ann, 
                    start_frame=ann['stop_frame'] - inner_buffer,
                    stop_frame=ann['stop_frame'] + outer_buffer,
                    pre_post='post', 
                    action=action,
                    state=post,
                    unknown_state=set(random.sample(predicates, len(predicates))) - set(post),
                ))
            if pre or post:
                break
        if count and len(annotations) >= count:
            break
    
    return annotations, predicates





def load_annotation_csv(annotation_path):
    df = pd.read_csv(annotation_path)
    df = df.sort_values(['video_id', 'start_frame'])
    
    # fix errors
    df.loc[df.narration == 'continue transferring rice from saucepan to plate to plate', 'all_nouns'] = '["rice","saucepan","plate"]'
    df.loc[df.narration == 'continue transferring rice from saucepan to plate to plate', 'narration'] = 'continue transferring rice from saucepan to plate'
    
    # # convert to timedelta
    # df['start_timestamp'] = pd.to_timedelta(df['start_timestamp'])
    # df['stop_timestamp'] = pd.to_timedelta(df['stop_timestamp'])
    # df['duration'] = (df['stop_timestamp']-df['start_timestamp']).dt.total_seconds()
    
    # parse list strings
    df['all_nouns'] = df.all_nouns.apply(eval)
    df['all_noun_classes'] = df.all_noun_classes.apply(eval)

    annotation_dir = os.path.dirname(annotation_path)
    verb_df = load_verbs(annotation_dir)
    noun_df = load_nouns(annotation_dir)
    df['verb_norm'] = verb_df.loc[df.verb_class]
    df['noun_norm'] = noun_df.loc[df.noun_class]
    return df


def load_verbs(annotation_dir):
    verb_df = pd.read_csv(os.path.join(annotation_dir, 'EPIC_100_verb_classes.csv')).set_index('key')
    verb_df['instances'] = verb_df['instances'].apply(eval)
    if 'use' in verb_df.index: verb_df.loc['use'].instances.append('use-to')
    if 'finish' in verb_df.index: verb_df.loc['finish'].instances.append('end-of')
    if 'carry' in verb_df.index: verb_df.loc['carry'].instances.append('bring-into')
    return verb_df


def load_nouns(annotation_dir):
    noun_df = pd.read_csv(os.path.join(annotation_dir, 'EPIC_100_noun_classes_v2.csv')).set_index('key')
    noun_df['instances'] = noun_df['instances'].apply(eval)
    return noun_df


def fix_colon(x):
    xs = x.split(':')
    return ' '.join(xs[1:] + xs[:1])



# Masks

def load_detections(annotation_dir, narration_id):  # TODO: slow. pre-dump masks?
    json_path = os.path.join(annotation_dir, f'{narration_id}.json')
    with open(json_path) as f:
        data = json.load(f)
    return {
        int(d['image']['image_path'].split('_')[-1].split('.')[0]): d
        for d in data
    }