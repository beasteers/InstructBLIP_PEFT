"""
"""

import os
import re
import json
import tqdm
import random
from collections import OrderedDict
from PIL import Image
import torch
import numpy as np
import pandas as pd
import supervision as sv
import cv2

from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.common.predicate_utils import load_pddl_yaml, Predicate
# from torchvision import datasets


# single image input
FRAME_TASK_1 = (
    'Describe the state of the {noun}:',
    '{state_csv}'
)
# pre/post image input
FRAME_TASK_2 = (
    'Describe the changes to {noun}.',
    '{state_diff_csv}'
)


def qa_1(ann):
    text_input = 'Describe the state of the {noun}:'.format(**ann)
    text_output = ', '.join(sorted(ann['state']))
    return text_input, text_output


# def qa_2(ann, translate, detection_labels=None):
#     noun = ann['noun']
#     all_nouns = ann['all_nouns']
#     state = ann['state']
#     # unknown = random.sample(list(ann['unknown_state']), max(10-len(state), 0))
#     # unknown = [Predicate(x, unknown=True) for x in unknown]
#     candidate_state = state# + unknown
#     candidate_state = [Predicate(x) for x in candidate_state]
#     # print(candidate_state)
#     # print([type(x) for x in candidate_state])
#     # print([str(x) for x in candidate_state])
#     random.shuffle(candidate_state)
#     # candidate_state_text = [translate[x.true().norm()] for x in ann['state']]
#     # candidate_state_value = [x.known_state for x in ann['state']]
#     state_list = " ".join([f'{translate[str(x.switch(True).rename_vars())].format(all_nouns[1:])}.' for x in candidate_state])
# #     state_example_list = " ".join([f'{translate[str(x.true().norm())]}: [choose one: yes|no|unknown].' for x in candidate_state])
# #     text_input = f'''
# # Evaluate the following statements about the "{}". 

# # For each statement, determine whether it is true, false, or cannot be determined with the information provided.

# # Statements:
# # {state_list}

# # State each statement then answer "yes", "no", or "unknown" based on your belief.

# # {state_example_list}
# # '''
    
#     # ANSWER = {True: 'yes', False: 'no', None: 'unknown'}

#     text_output = ' '.join(f'{translate[str(x.rename_vars())].format(all_nouns[1:])}.' for x in candidate_state)

#     objs = ''
#     if detection_labels is not None:
#         objs = f"Objects in scene: {', '.join(f'{i}: {o}' for i, o in enumerate(detection_labels))}.  "

#     text_input = random.choice([
#         f'{objs}Based on the image, which of the following predicates apply to the "{noun}"? {state_list}. Answer: ',
#         f'{objs}Which of the following predicates apply to the "{noun}"? {state_list}  Answer: ',
#         f'{objs}Which of the following predicates describe the "{noun}"? {state_list}  Answer: ',
#         f'{objs}Which of the following apply to the "{noun}"? {state_list}  Answer: ',
#         f'{objs}Which of the following describe the "{noun}"? {state_list}  Answer: ',
#     ])
    
#     return text_input.strip(), text_output.strip()

def qa_2(ann, detection_labels=None):
    noun = ann['noun']
    all_nouns = ann['all_nouns']
    # state = ann['state']
    action = ann['action']
    candidate_state = action.get_state(action.vars[0], ann['pre_post'])
    random.shuffle(candidate_state)
    noun_dict = action.var_dict(all_nouns)

    state_input = []
    state_output = []
    for s in candidate_state:
        state_input.append(s.switch(True).format(**noun_dict))
        state_output.append(s.format(**noun_dict))

    state_list = " ".join(set(f'{x}.' for x in state_input))
    text_output = " ".join(set(f'{x}.' for x in state_output))

    objs = ''
    if detection_labels is not None:
        objs = f"Objects in scene: {', '.join(f'{i}: {o}' for i, o in enumerate(detection_labels))}.  "

    text_input = random.choice([
        f'{objs}Based on the image, which of the following predicates apply to the "{noun}"? {state_list}. Answer: ',
        f'{objs}Which of the following predicates apply to the "{noun}"? {state_list}  Answer: ',
        f'{objs}Which of the following predicates describe the "{noun}"? {state_list}  Answer: ',
        f'{objs}Which of the following apply to the "{noun}"? {state_list}  Answer: ',
        f'{objs}Which of the following describe the "{noun}"? {state_list}  Answer: ',
    ])
    
    return text_input.strip(), text_output.strip()

# def qa_3(ann):
#     text_input = 'What is the user doing?'.format(**ann)
#     text_output = ann['narration']
#     return text_input, text_output



QA = {
    "state_v1": qa_1,
    "state_v2": qa_2,

}



class EKVQADataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, split=None, 
                 include_detections=False, n_frames=None, outer_buffer=60, inner_buffer=30, downsample_count=None, fake_duplicate_count=None, qa_format='state_v2', **kw):
        super().__init__(vis_processor, text_processor, vis_root=vis_root, ann_paths=ann_paths)
        self.annotation_path = ann_paths[0]
        self.annotation_dir = os.path.dirname(self.annotation_path)

        # self.split = split or self.annotation_path.split('_')[-1].split('.')[0].replace('validation', 'val')
        # self.json_dir = os.path.join('/vast/irr2020/EKU/FINAL', self.split)
        # self.json_dir = os.path.join('/scratch/work/ptg/EPIC-KITCHENS/EKU/HOSCL_not_in_VISOR_buffer_XMemHOS_2', self.split)
        split_file = self.annotation_path.split('_')[-1].split('.')[0].replace('validation', 'val')
        self.split = split or split_file
        self.json_dir = os.path.join('/vast/irr2020/EKU/FINAL', split_file)

        self.include_detections = include_detections
        self.n_frames = n_frames
        self.outer_buffer = outer_buffer
        self.inner_buffer = inner_buffer
        self.downsample_count = downsample_count
        self.fake_duplicate_count = fake_duplicate_count

        self._prepare_annotations(self.annotation_path)
        self._add_instance_ids()

        self.frame_file_format = 'frame_{:010d}.jpg'
        self._get_qa = QA[qa_format]

        # for i in tqdm.tqdm(range(len(self))):
        #     d = self[i]

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        # get potential frame list
        frame_fnames = self._existing_frames(ann['video_id'], ann['start_frame'], ann['stop_frame'])
        if not len(frame_fnames):
            print(f"WARNING: no frames - {index} {ann['video_id']} {ann['start_frame']} {ann['stop_frame']}")
            return self.__getitem__((index + 1)%len(self))

        # select frames
        sampled_frame_ids = list(frame_fnames)
        if self.n_frames != 'all':
            sampled_frame_ids = random.sample(list(frame_fnames), min(self.n_frames or 1, len(frame_fnames)))
        sampled_frame_ids = sorted(sampled_frame_ids)
        
        # load frames
        frames = [Image.open(frame_fnames[i]) for i in sampled_frame_ids]

        # state_lookup = dict(zip(ann['state'], ann['state']))
        # state_vector = np.array([bool(state_lookup[p]) if p in state_lookup else -1 for p in self.predicates])

        if self.include_detections:
            # load detection frames
            detections = load_detections(self.json_dir, ann["narration_id"], sampled_frame_ids, np.array(frames[0]))
            det_index = list({l for d in detections for l in d.data['labels']})
            det_frames = [draw_masks(x, d, det_index) for x, d in zip(frames, detections)]
            # interleave frames
            frames = [x for xs in zip(frames, det_frames) for x in xs]
            text_input, text_output = self._get_qa(ann, det_index)
            if any(len(d) for d in detections):
                for i, f in enumerate(frames):
                    f.save(f'demo{i}.png')
        else:
            # load question answer
            text_input, text_output = self._get_qa(ann)

        # if any(len(d) for d in detections):
        #     print(detections)
            # for i, f in enumerate(frames):
            #     f.save(f'demo{i}.png')
        #     print({
        #         # metadata
        #         "narration": ann["narration"],
        #         "noun": ann["noun"],

        #         # ID
        #         "image_id": index,
        #         "narration_id": ann["narration_id"],
        #         "instance_id": ann["instance_id"],
        #         "question_id": ann["instance_id"],
        #         "sample_id": index,

        #         # QA pair
        #         "prompt": text_input,
        #         "text_input": text_input,
        #         "text_output": text_output,
        #     })
        #     input()
        image = torch.stack([self.vis_processor(x) for x in frames], dim=0)
        if self.n_frames is None and image.size(0) == 1:
            # single image? for compatability
            image = image[0]
            assert image.size() == (3, 224, 224)
        return {
            # load a random frame from that range
            "image": image,
            # "label": ann["label"],
            # "classes": 

            # metadata
            "narration": ann["narration"],
            "noun": ann["noun"],

            # ID
            "image_id": index,
            "narration_id": ann["narration_id"],
            "instance_id": ann["instance_id"],
            "question_id": ann["instance_id"],
            "sample_id": index,

            # QA pair
            "prompt": text_input,
            "text_input": text_input,
            "text_output": text_output,
        }

    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict({
            "file": ann["image"],
            # "label": self.classnames[ann["label"]],
            # "image": sample["image"].shape,
            "text_input": sample['text_input'],
            "text_output": sample['text_output'],
        })


    # utils

    def _prepare_annotations(self, annotation_path):
        df = load_annotation_csv(annotation_path)
        actions, predicates = load_pddl_yaml(os.path.join(os.path.dirname(annotation_path), "EPIC_100_conditions.yaml"))
        self.predicates = predicates
        self.verbs = list(actions)

        annotations = []
        
        count = self.downsample_count
        count = count * len(df) if count and count <= 1 else count
        # if count:
        #     df = df.sample(frac=1)
        
        for ann in tqdm.tqdm(df.to_dict('records'), desc=self.annotation_path.split(os.sep)[-1]):
            for verb in [ann['verb'], ann['verb_norm']]:
                if verb not in actions:
                    continue
                action = actions[verb]
                pre = action.get_state(action.vars[0], 'pre')
                post = action.get_state(action.vars[0], 'post')

                duration = ann['stop_frame'] - ann['start_frame']
                inner_buffer = int(min(self.inner_buffer, duration // 4))
                if pre:
                    annotations.append(dict(
                        ann, 
                        start_frame=ann['start_frame'] - self.outer_buffer,
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
                        stop_frame=ann['stop_frame'] + self.outer_buffer,
                        pre_post='post', 
                        action=action,
                        state=post,
                        unknown_state=set(random.sample(predicates, len(predicates))) - set(post),
                    ))
                if pre or post:
                    break
            if count and len(annotations) >= count:
                break
        self.annotation = annotations
        if self.fake_duplicate_count:
            self.annotation = self.annotation * self.fake_duplicate_count

    def _frame_name(self, video_id, i):
        return os.path.join(self.vis_root, video_id, self.frame_file_format.format(i))
    
    def _existing_frames(self, video_id, start_frame, stop_frame):
        frames = {f: self._frame_name(video_id, f) for f in range(start_frame, stop_frame+1)}
        frames = {i: f for i, f in frames.items() if os.path.isfile(f)}
        return frames



def sample_frames(frames, n=None, random=False):
    frames = np.asarray(frames)
    if random:
        assert n
        return np.sort(np.random.choice(frames, min(n, len(frames)), replace=False))
    if n:
        i = np.linspace(0, len(frames)-1, min(n, len(frames))).round().astype(int)
        return frames[i]
    return frames



# Narrations


def load_annotation_csv(annotation_path):
    # load train/test
    df = pd.read_csv(annotation_path)
    # fix errors
    df.loc[df.narration == 'continue transferring rice from saucepan to plate to plate', 'all_nouns'] = '["rice","saucepan","plate"]'
    df.loc[df.narration == 'continue transferring rice from saucepan to plate to plate', 'narration'] = 'continue transferring rice from saucepan to plate'
    # sort
    df = df.sort_values(['video_id', 'start_timestamp'])
    # convert to timedelta
    df['start_timestamp'] = pd.to_timedelta(df['start_timestamp'])
    df['stop_timestamp'] = pd.to_timedelta(df['stop_timestamp'])
    df['duration'] = (df['stop_timestamp']-df['start_timestamp']).dt.total_seconds()
    # parse list strings
    df['all_nouns'] = df.all_nouns.apply(lambda x: eval(x))

    annotation_dir = os.path.dirname(annotation_path)
    verb_df = load_verbs(annotation_dir)
    noun_df = load_nouns(annotation_dir)
    df = norm_df(df, verb_df, noun_df)
    return df

def load_verbs(annotation_dir):
    verb_df = pd.read_csv(os.path.join(annotation_dir, 'EPIC_100_verb_classes.csv')).set_index('key')
    verb_df['instances'] = verb_df['instances'].apply(lambda x: eval(x))
    verb_df.loc['use'].instances.append('use-to')
    verb_df.loc['finish'].instances.append('end-of')
    verb_df.loc['carry'].instances.append('bring-into')
    return verb_df


def load_nouns(annotation_dir):
    noun_df = pd.read_csv(os.path.join(annotation_dir, 'EPIC_100_noun_classes_v2.csv')).set_index('key')
    noun_df['instances'] = noun_df['instances'].apply(lambda x: eval(x))
    return noun_df

def norm_df(df, verb_df, noun_df):
    # norm verbs
    verb_norm = {v: k for k,row in verb_df.iterrows() for v in row.instances}
    df['verb_norm'] = df.verb.apply(lambda x: verb_norm[x])
    # norm nouns
    noun_norm = {v: k for k,row in noun_df.iterrows() for v in row.instances}
    df['all_nouns_norm'] = df.all_nouns.apply(lambda xs: [noun_norm[x] for x in xs])
    return df



# Masks


def load_detections(annotation_dir, narration_id, frame_ids, sample_frame):
    json_path = os.path.join(annotation_dir, f'{narration_id}.json')
    with open(json_path) as f:
        data = json.load(f)
    data = {
        int(d['image']['image_path'].split('_')[-1].split('.')[0]): d
        for d in data
    }
    return [
        get_dets(data.get(i, {}), sample_frame)
        for i in frame_ids
    ]

def get_dets(frame_data, frame, scale=None):
    anns = frame_data.get('annotations') or []

    # extract annotation data
    xyxy = np.array([
        np.asarray(d['bounding_box']) if 'bounding_box' in d else 
        seg2box(d['segments'], scale) if 'segments' in d else 
        np.zeros(4)
        for d in anns
    ])
    mask = np.array([
        seg2mask(d['segments'], frame.shape, scale) if 'segments' in d else 
        np.zeros(frame.shape[:2], dtype=bool)
        for d in anns
    ], dtype=bool).reshape(-1, *frame.shape[:2]) if any('segments' in d for d in anns) else None
    confidence = [d.get('confidence', 1) for d in anns]
    class_id = [d['class_id'] for d in anns]
    track_id = [d.get('track_id', -1) for d in anns]
    labels = [d['name'] for d in anns]

    # create detections
    detections = sv.Detections(
        xyxy=xyxy.reshape(-1, 4),
        mask=mask,
        class_id=np.array(class_id, dtype=int),
        tracker_id=np.array(track_id, dtype=int),
        confidence=np.array(confidence),
        data={'labels': np.array(labels, dtype=str)},
    )
    return detections

TARGET = np.array([456, 256])
VISOR_SCALE = np.array([854, 480])
SCALE = list(VISOR_SCALE / TARGET)
def prepare_poly(points, scale=None):
    X = np.array([x for x in points]).reshape(-1, 2)
    if scale is not False:
        scale = SCALE if scale is None or scale is True else scale
        X = X / np.asarray(scale)
    return X.astype(int)


def seg2poly(segments, scale=None, min_points=3):
    xs = [prepare_poly(x, scale) for x in segments]
    return [x for x in xs if len(x) >= min_points]


def seg2box(segments, scale=None):
    X = np.concatenate(seg2poly(segments, scale) or [np.zeros((0, 2))])
    if X.shape[0] == 0:
        return np.array([0, 0, 0, 0])
    return np.concatenate([np.min(X, axis=0), np.max(X, axis=0)])


def seg2mask(segments, shape, scale=None):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    xs = seg2poly(segments, scale)
    if len(xs):
        mask = cv2.fillPoly(mask, xs, [255])
    return mask > 0


ma = sv.MaskAnnotator()
ba = sv.BoxCornerAnnotator()
la = sv.LabelAnnotator(text_position=sv.Position.CENTER)
def draw_masks(image, detections, det_index):
    color_lookup = np.array([det_index.index(l) for l in detections.data['labels']])
    image = np.array(image)[:, :, ::-1]
    image = image.copy()
    is_ma = detections.mask.any((1,2)) if detections.mask is not None else np.zeros(len(detections), dtype=bool)
    is_ba = ~is_ma & detections.xyxy.any(1)
    image = ma.annotate(scene=image, detections=detections[is_ma], custom_color_lookup=color_lookup)
    image = ba.annotate(scene=image, detections=detections[is_ba], custom_color_lookup=color_lookup)
    image = la.annotate(scene=image, detections=detections, labels=color_lookup.astype(str), custom_color_lookup=color_lookup)
    return Image.fromarray(image[:, :, ::-1])



# Predicate 


# import os
# import re
# class Predicate:
#     def __init__(self, text, unknown=None, positive=None, negative=None):
#         self.positive = positive
#         self.negative = negative
#         if isinstance(text, Predicate):
#             self.neg, self.name = text.neg, text.name
#             self.var = list(text.var)
#             self.unknown = text.unknown if unknown is None else unknown
#         else:
#             m = re.findall(r'\((?:(not) \()?([=\w-]+)\s((?:\??\w+\s*)+)\)?\)', text)
#             if not m:
#                 raise ValueError(f"could not parse {text}")
#             self.neg, self.name, var = m[0]
#             self.unknown = unknown or False
#             self.var = var.split()

#     @property
#     def known_state(self):
#         return not bool(self.neg) if not self.unknown else None

#     def __hash__(self):
#         return hash(str((self.name, len(self.var))))
    
#     def __eq__(self, other):
#         return (self.neg, self.name, len(self.var)) == (other.neg, other.name, len(other.var))
    
#     def __nonzero__(self):
#         return not self.neg

#     def __str__(self):
#         p = f'({self.name} {" ".join(self.var)})'
#         return f'(not {p})' if self.neg else p
#     __repr__ = __str__
    
#     def translate_vars(self, ref, *others):
#         """
#         (above b a) .translate_vars(
#             (above x y), 
#             (below y x), (not (behind x y))
#         ) -> 
#         (below a b), (not (behind b a))
#         """
#         trans = dict(zip(self.var, ref.var))
#         others = [Predicate(o) for o in others]
#         for o in others:
#             o.var = [trans.get(x,x) for x in o.var]
#         return others
    
#     def format(self, **nouns):
#         nouns = [nouns.get(x, x) for x in self.var]
#         if self.neg and self.positive:
#             return self.positive.format(*nouns)
#         if self.neg and self.negative:
#             return self.negative.format(*nouns)
#         return str(self.rename_vars(nouns))
    
#     def rename_vars(self, *vars):
#         p=Predicate(self)
#         p.var = vars or [f"?{chr(ord('a')+i)}" for i,x in enumerate(p.var)]
#         assert len(p.var) == len(self.var), f"vars lengths must match {self.var} {vars}"
#         return p
    
#     def switch(self, on):
#         p=Predicate(self)
#         p.neg='' if on else 'not'
#         return p


# def add_axioms(current, axioms):
#     for x in list(current):
#         for a in axioms:
#             if a['context'] == x:
#                 current.extend(a['context'].translate_vars(x, *a['implies']))
#     return list(set(current))

# def load_prepost_conditions(annotation_dir):
#     import yaml
#     with open(os.path.join(annotation_dir, 'EPIC_100_conditions.yaml'), 'r') as f:
#         data = yaml.safe_load(f)
    
#     predicates = [Predicate(x) for x in data['predicates']]
#     predicates += [p.switch(False) for p in predicates]

#     translations: dict[str, str] = data['translations']
#     actions: list[dict] = data['definitions']
#     axioms: list[dict] = data['axioms']
#     for d in axioms:
#         d['context'] = Predicate(d['context'])
#         d['implies'] = [Predicate(x) for x in d['implies']]
#     for d in actions:
#         p = [Predicate(x) for x in d['preconditions']]
#         e = [Predicate(x) for x in d['effects']]
#         p = [x for x in add_axioms(p, axioms) if x in predicates]
#         e = [x for x in add_axioms(e, axioms) if x in predicates]
#         # p = [translations.get(str(x.norm()), x) for x in p]
#         # e = [translations.get(str(x.norm()), x) for x in e]
#         d['preconditions'] = p
#         d['effects'] = e

#     prepost = {
#         d['name']: [d['preconditions'], d['effects']]
#         for d in actions
#     }
#     # predicates = [translations.get(str(x.norm()), x) for x in predicates]
#     return prepost, translations, predicates
# # load_prepost_conditions('../epic-kitchens-100-annotations')


# class Action:
#     def __init__(self, name, params, pre, post):
#         self.name = name
#         self.params = params
#         self.pre = pre
#         self.post = post

#     def preconditions(self, *nouns):
#         nouns = dict(zip(self.params, nouns))
#         return [p.format(**nouns) for p in self.pre]
    
#     def postconditions(self, *nouns):
#         nouns = dict(zip(self.params, nouns))
#         return [p.format(**nouns) for p in self.post]
    

# #   - actions:
# #     - open
# #     description: open X
# #     effects:
# #     - (is-openable ?a)
# #     - (within-reach ?a)
# #     - (not (touching ?a hand))
# #     - (opened ?a)
# #     generated: 0
# #     id: 2
# #     name: open
# #     params:
# #     - X
# #     preconditions:
# #     - (is-openable ?a)
# #     - (within-reach ?a)
# #     - (touching ?a hand)
# #     - (not (opened ?a))

# # Action('open', ['X', 'Y'], [], [])