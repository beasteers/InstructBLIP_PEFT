import os
import tqdm
import random
from lavis.common.predicate_utils.predicates import Predicate
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from PIL import Image
import json
import supervision as sv
import cv2
from PIL import Image
from lavis.common.predicate_utils.masks import get_detections, draw_detections, get_detections_h5
from lavis.common.predicate_utils.prompts import get_prompt_function


def PIL_load(f, shape=None): 
    im = Image.open(f) 
    if shape is not None: 
        im.draft('RGB',tuple(shape)) #(1008,756)
    return im


class VideoFrameDataset(Dataset):
    def __init__(self, 
                 annotations, vis_root, vis_processor, classes, h5_file=None,
                 qa_prompt=None, include_detections=False, filename_format='{video_id}/frame_{i:010d}.jpg', 
                 n_frames=1, boxes_only=False, included_object_class_ids=None, main_object_only=False,
                 prompt_kw=None, image_load_shape=None, return_masks=False,
    ):
        super().__init__()
        self.vis_processor = vis_processor
        self.annotations = annotations
        self.classes = classes
        for i, a in enumerate(annotations):
            a['instance_id'] = i
        self.get_prompt = get_prompt_function(qa_prompt)
        self.prompt_kw = {}
        self.n_frames = n_frames
        self.vis_root = vis_root
        self.filename_format = filename_format
        self.include_detections = include_detections
        self.boxes_only = boxes_only
        self.included_object_class_ids = included_object_class_ids
        self.main_object_only = main_object_only
        self.image_load_shape = image_load_shape
        self.return_masks = return_masks
        self.h5_file = None
        if self.include_detections and h5_file is not None:
            print("Using", h5_file)
            self.h5_file = h5py.File(h5_file, 'r', libver='latest')
            idxs = []
            for i, a in enumerate(annotations):
                if a['narration_id'] not in self.h5_file:
                    print(a['narration_id'], 'missing from h5')
                    idxs.append(i)
            self.annotations = [a for i,a in enumerate(annotations) if i not in idxs]
            print(f"dropping samples as they are missing from h5: {len(idxs)}/{len(self.annotations)}")


        # self.im_transform = transforms.ToTensor()
        # self.im_resize = transforms.Resize(size,
        #                                    interpolation=InterpolationMode.BILINEAR,
        #                                    antialias=True)
        # self.mask_resize = transforms.Resize(size,
        #                                      interpolation=InterpolationMode.NEAREST,
        #                                      antialias=True)

    def __len__(self):
        return len(self.annotations)
    
    def load_detections(self, ann):
        raise NotImplementedError()
    
    def __getitem__(self, i, _recursion=0):
        if _recursion > 50: raise RuntimeError("I tried to find some data, I really did... but idk. too many missing.")
        try:
            return self._load_ann(i)
        except Exception as e:
            print("WARNING:", e)
            return self.__getitem__((i + 1)%len(self), _recursion=_recursion+1)

    def _load_ann(self, i):
        ann = self.annotations[i]

        # list frames
        files = self._existing_frames(ann, ann['start_frame'], ann['stop_frame'])
        if not len(files):
            print(f"WARNING: no frames for {self._frame_name(video_id=ann['video_id'], i=1)} - {i} {ann['video_id']} {ann['start_frame']} {ann['stop_frame']}")
            return self.__getitem__((i + 1)%len(self))

        # load detection frames
        detections = None
        if self.include_detections and self.h5_file is None:
            detections = self.load_detections(ann)

        masks = None
        if self.include_detections:
            # load frames
            frame_ids = list(files)
            if self.h5_file is not None:
                group = self.h5_file[ann['narration_id']]
                frame_index = group['frame_index'][()]
                has_detection = np.array([i in frame_index for i in frame_ids])
            else:
                has_detection = np.array([i in detections for i in frame_ids])
            # frame_ids = self._sorted_sample(files, has_detection * 10 + 1)
            # frames = [Image.open(files[i]) for i in frame_ids]
            frames, frame_ids = self._load_frames(files, frame_ids, has_detection * 10 + 1)
            if self.h5_file is not None:
                dets = get_detections_h5(group, frame_ids, frames[0])
            else:
                dets = [get_detections(detections.get(i, {}), x) for x, i in zip(frames, frame_ids)]

            # draw detection frames
            included_object_class_ids = []
            if self.included_object_class_ids is not None:
                included_object_class_ids.extend(self.included_object_class_ids)
            if self.main_object_only:
                included_object_class_ids.extend(ann['all_noun_classes'])
            if included_object_class_ids:
                dets = [
                    ds[np.isin(ds.class_id, included_object_class_ids)]
                    for ds in dets
                ]
            object_index = list({l for d in dets for l in d.data['labels']})

            if self.return_masks:
                masks = detections.mask
            else:
                det_frames = [
                    draw_detections(x, d, object_index, boxes_only=self.boxes_only) 
                    for x, i, d in zip(frames, frame_ids, dets)
                ]
                # interleave frames
                frames = [x for xs in zip(frames, det_frames) for x in xs]
                # frames = det_frames

            # load question answer
            prompt, target = self.get_prompt(ann, object_index, **self.prompt_kw)
            # if any(len(d) for d in dets):
            #     for i, f in enumerate(frames):
            #         f.save(f'demo{i}.png')
            # with open('qa.txt', 'w') as fh:
            #     fh.write(f'{prompt}\n\n{target}')
        else:
            # load frames
            # frame_ids = self._sorted_sample(files)
            # frames = [Image.open(files[i]) for i in frame_ids]
            frames, frame_ids = self._load_frames(files)

            # load question answer
            prompt, target = self.get_prompt(ann, **self.prompt_kw)

        # print(ann['narration'])
        # print(ann['verb'])
        # print(ann['pre_post'])
        # print(ann['action'])

        class_targets = class_labels = None
        if self.classes is not None:
            act = ann['action']
            states = act.get_state('?a', ann['pre_post'])
            states = [s.norm_vars() for s in states]
            # class_labels = [str(c) for c in self.classes]
            predicates = [Predicate(c).norm_vars() for c in self.classes]
            class_targets = torch.as_tensor([1 if c.flip(True) in states else 0 if c.flip(False) in states else -1 for c in predicates])
            # print(act.name)
            # print(list(map(str, predicates)))
            # print(class_targets)
            # print(states)
            # # print(prompt)
            # # print(target)
            # print(np.array(list(map(str, predicates)))[class_targets!=-1])
            # print(class_targets[class_targets!=-1])
            # input()

        video = torch.stack([self.vis_processor(x) for x in frames], dim=0)
        # video = self.vis_processor(frames[1])
        return {
            # **ann,
            'image': video,
            "prompt": prompt,
            "text_input": prompt,
            "text_output": target,
            # metadata
            "narration": ann["narration"],
            "noun": ann["noun"],

            # ID
            "image_id": i,
            "narration_id": ann["narration_id"],
            "instance_id": ann["instance_id"],
            "question_id": ann["instance_id"],
            "sample_id": i,

            "targets": class_targets,
            # "class_labels": class_labels,
            **({'masks': masks} if masks is not None else {})
        }

    
    def _sorted_sample(self, frame_fnames, weights=None):
        if weights is not None:
            weights = np.asarray(weights)
            weights = weights / weights.sum()
        if self.n_frames != 'all':
            frame_fnames = np.random.choice(
                list(frame_fnames), 
                min(self.n_frames or 1, len(frame_fnames)),
                replace=False,
                p=weights)
        return sorted(frame_fnames)

    def _load_frames(self, fs, frame_ids=None, weights=None):
        if frame_ids is None:
            frame_ids = list(fs)
        n = len(fs) if self.n_frames == 'all' else self.n_frames
        n = min(n or 1, len(fs))

        if weights is not None:
            weights = np.asarray(weights)
            weights = weights / weights.sum()

        samples = np.random.choice(frame_ids, len(frame_ids), replace=False, p=weights)
        
        frames = []
        out_frame_ids = []
        for fid in samples:
            for _ in range(3):
                try:
                    frames.append(PIL_load(fs[fid], self.image_load_shape))
                    out_frame_ids.append(fid)
                    break
                except OSError:
                    import time
                    time.sleep(0.1)
            if len(frames) == n:
                break
        
        sort = np.argsort(out_frame_ids)
        out_frame_ids = [out_frame_ids[i] for i in sort]
        frames = [frames[i] for i in sort]
        return frames, out_frame_ids
    
    def _frame_name(self, video_id, i):
        return os.path.join(self.vis_root, self.filename_format.format(video_id=video_id, i=i))

    def _existing_frames(self, ann, start_frame, stop_frame):
        # print(start_frame, stop_frame+1)
        frames = {i: self._frame_name(ann['video_id'], i) for i in range(start_frame, stop_frame+1)}
        # print(frames)
        frames = {i: f for i, f in frames.items() if os.path.isfile(f)}
        # print(frames)
        return frames
