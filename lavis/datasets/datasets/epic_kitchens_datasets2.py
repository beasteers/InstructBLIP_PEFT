import os
import json
import orjson
import tqdm
import random
import numpy as np
import pandas as pd
from lavis.common.predicate_utils.predicates import load_pddl_yaml, Predicate
from lavis.datasets.datasets.base_video_detection_prompt_datasets import VideoFrameDataset
from lavis.datasets.datasets.base_dataset import BaseDataset


class EKVQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, json_dir, split=None, train_samples_portion=None, **kw):
        super().__init__(vis_processor, text_processor, vis_root=vis_root, ann_paths=ann_paths)
        self.annotation_path = ann_paths[0]
        split_file = self.annotation_path.split('_')[-1].split('.')[0].replace('validation', 'val')
        # json_dir = os.path.join('/vast/irr2020/EKU/FINAL', split_file)
        # json_dir = os.path.join('/scratch/work/ptg/Something_ek_labels', split_file)
        assert split and json_dir
        self.dataset = EpicKitchensDataset(
            self.annotation_path, vis_root, 
            json_dir=f'{json_dir}/{split_file}',
            vis_processor=vis_processor, 
            **kw)
        self.annotations = self.dataset.annotations
        self.split = split
        self._add_instance_ids()

    @property
    def classes(self):
        return self.dataset.classes

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, i):
        return self.dataset[i]


# import pyinstrument
# prof = pyinstrument.Profiler()

class EpicKitchensDataset(VideoFrameDataset):
    def __init__(self, annotation_path, vis_root, json_dir, vis_processor, 
                 downsample_count=None, fake_duplicate_count=None, 
                 outer_buffer=60, inner_buffer=2, 
                 include_video_ids=None, exclude_video_ids=None, 
                 filter_verbs=None, shuffle=True, predicate_freq_balancing=True, **kw):
        annotations, predicates, predicate_counts = load_epic_kitchens_dataset(
            annotation_path, downsample_count, outer_buffer, inner_buffer, include_video_ids, exclude_video_ids, filter_verbs, shuffle, predicate_freq_balancing)
        if fake_duplicate_count:
            annotations = annotations * fake_duplicate_count
        self.json_dir = json_dir
        self.classes = [str(p) for p in predicates]
        super().__init__(annotations, vis_root, vis_processor=vis_processor, classes=predicates, **kw)
        self.prompt_kw['predicate_freq'] = predicate_counts
        self.prof_count = 0

    def load_detections(self, ann):
        # import time
        # t0=time.time()
        # try:
            return load_detections(self.json_dir, ann["narration_id"])
        # finally:
        #     print("t=", time.time()-t0)

#     def __getitem__(self, i):
#         try:
#             with prof:
#                 return super().__getitem__(i)
#         finally:
#             self.prof_count += 1
#             if not self.prof_count % 32:
#                 prof.print()
# # 


def load_epic_kitchens_dataset(annotation_path, count=None, outer_buffer=60, inner_buffer=2, include_video_ids=None, exclude_video_ids=None, filter_verbs=None, shuffle=True, predicate_freq_balancing=True):
    df = load_annotation_csv(annotation_path)

    # use video split list
    if exclude_video_ids is not None:
        if isinstance(exclude_video_ids, str):
            exclude_video_ids = pd.read_csv(exclude_video_ids).video_id.tolist()
        print(exclude_video_ids)
        print("excluding", len(exclude_video_ids), 'files', len(df))
        df = df[~df.video_id.isin(exclude_video_ids)]
        print(len(df))
        if not len(df):
            print(df.video_id.unique())
            print(exclude_video_ids)
    elif include_video_ids is not None:
        if isinstance(include_video_ids, str):
            include_video_ids = pd.read_csv(include_video_ids).video_id.tolist()
        print("including", len(include_video_ids), 'files', len(df))
        print(include_video_ids)
        df = df[df.video_id.isin(include_video_ids)]
        if not len(df):
            print(df.video_id.unique())
            print(include_video_ids)
        print(len(df))
    
    if filter_verbs is not None:
        print("filtering", filter_verbs, 'verbs', len(df))
        df = df[df.verb.isin(filter_verbs)]
        print(len(df))

    annotation_dir = os.path.dirname(annotation_path)
    actions, predicates = load_pddl_yaml(f"{annotation_dir}/EPIC_100_conditions.yaml")

    predicate_counts = {p.norm_vars(): 0 for p in predicates + [p.flip(False) for p in predicates]}
    for name, act in actions.items():
        for p in act.pre + act.post:
            p = p.norm_vars()
            if p not in predicate_counts:
                print("WARNING:", p, f"is in action {name} but not in the predicate class list")
                predicate_counts[p] = 0
    #         predicate_counts[p] += 1
    # with open('predicate_counts.json', 'w') as f:  # debug
    #     json.dump({str(k): c for k, c in predicate_counts.items()}, f)
    # total = sum(predicate_counts.values())
    # predicate_counts = {k: 5 * np.log(total / (c+1)) for k, c in predicate_counts.items()}
    
    # print(predicate_counts)

    
    if count:
        count = int(count * len(df) if count <= 1 else count)
    if shuffle:
        print("SHUFFLING")
        df = df.sample(frac=1, random_state=12345)
    
    annotations = []
    for ann in tqdm.tqdm(df.to_dict('records'), desc=annotation_path.split(os.sep)[-1], leave=False):
        for verb in [ann['verb'], ann.get('verb_norm')]:
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
                for p in pre:
                    p = p.norm_vars()
                    predicate_counts[p] += 1
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
                for p in post:
                    p = p.norm_vars()
                    predicate_counts[p] += 1
            if pre or post:
                break
        if count and len(annotations) >= count:
            break

    # if 'validation' in annotation_path:
    #     with open('val_annotations.json', 'w') as f:  # debug
    #         json.dump(annotations, indent=2)
    print([d['narration_id'] for d in annotations[:10]])
    print([d['narration_id'] for d in annotations[-10:]])
    # input()

    # predicates = [p for p in predicates if predicate_counts.get(p.norm_vars()) and predicate_counts.get(p.flip(False).norm_vars())]

    with open('predicate_counts.json', 'w') as f:  # debug
        json.dump({str(k): c for k, c in predicate_counts.items()}, f, indent=2)
    total = sum(predicate_counts.values())
    predicate_counts = {k: total / (c+1) for k, c in predicate_counts.items()}
    if predicate_freq_balancing is True:
        predicate_counts = None
    return annotations, predicates, predicate_counts





def load_annotation_csv(annotation_path):
    df = pd.read_csv(annotation_path)
    df = df.sort_values(['video_id', 'start_frame'])
    if 'verb' not in df:
        raise RuntimeError(f"verb not in {annotation_path}")

    df = df.dropna(how='any', subset=['start_frame', 'stop_frame', 'verb'])
    df['start_frame'] = df['start_frame'].astype(int)
    df['stop_frame'] = df['stop_frame'].astype(int)

    df['narration_id'] = df['narration_id'].astype(str)
    df['video_id'] = df['video_id'].astype(str)
    
    # # fix errors
    # df.loc[df.narration == 'continue transferring rice from saucepan to plate to plate', 'all_nouns'] = '["rice","saucepan","plate"]'
    # df.loc[df.narration == 'continue transferring rice from saucepan to plate to plate', 'narration'] = 'continue transferring rice from saucepan to plate'
    
    # # convert to timedelta
    # df['start_timestamp'] = pd.to_timedelta(df['start_timestamp'])
    # df['stop_timestamp'] = pd.to_timedelta(df['stop_timestamp'])
    # df['duration'] = (df['stop_timestamp']-df['start_timestamp']).dt.total_seconds()
    
    # parse list strings
    df['all_nouns'] = df.all_nouns.apply(eval)
    if 'all_noun_classes' in df.columns:
        df['all_noun_classes'] = df.all_noun_classes.apply(eval)

    annotation_dir = os.path.dirname(annotation_path)
    try:
        verb_df = load_verbs(annotation_dir)
        df['verb_norm'] = verb_df.key.loc[df.verb_class].values
    except FileNotFoundError:
        df['verb_norm'] = df.verb
    try:
        noun_df = load_nouns(annotation_dir)
        df['noun_norm'] = noun_df.key.loc[df.noun_class].values
    except FileNotFoundError:
        df['noun_norm'] = df.noun

    df['noun'] = df.noun.apply(fix_colon)
    df['noun_norm'] = df.noun_norm.apply(fix_colon)
    df['all_nouns'] = df.all_nouns.apply(lambda xs: [fix_colon(x) for x in xs])
    return df


def load_verbs(annotation_dir):
    verb_df = pd.read_csv(os.path.join(annotation_dir, 'EPIC_100_verb_classes.csv')).set_index('id')
    verb_df['instances'] = verb_df['instances'].apply(eval)
    if 'use' in verb_df.columns: verb_df.loc['use'].instances.append('use-to')
    if 'finish' in verb_df.columns: verb_df.loc['finish'].instances.append('end-of')
    if 'carry' in verb_df.columns: verb_df.loc['carry'].instances.append('bring-into')
    return verb_df


def load_nouns(annotation_dir):
    noun_df = pd.read_csv(os.path.join(annotation_dir, 'EPIC_100_noun_classes_v2.csv')).set_index('id')
    noun_df['instances'] = noun_df['instances'].apply(eval)
    return noun_df


def fix_colon(x):
    xs = x.split(':')
    return ' '.join(xs[1:] + xs[:1])



# Masks

def load_detections(annotation_dir, narration_id):  # TODO: slow. pre-dump masks?
    json_path = os.path.join(annotation_dir, f'{narration_id}.json')
    if not os.path.isfile(json_path): 
        print(json_path, "doesn't exist")
        return {}
    # with open(json_path) as f:
    #     data = json.load(f)
    with open(json_path, 'rb') as f:
        data = orjson.loads(f.read())
    return {
        int(d['image']['image_path'].split('/')[-1].split('_')[-1].split('.')[0]): d
        for d in data
    }