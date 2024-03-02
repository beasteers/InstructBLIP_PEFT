import os
import json
import tqdm
import random
import numpy as np

'''

- Object ID: what is object number 3?
- Direct yes/no: is object 4 on a surface?
- Affordances: What numbered object is openable?
- Relational: What numbered objects are holding other numbered objects?
- Negative attributes: What of the following predicates do not apply to the fridge?
- Contrastive: What numbered objects are on a surface but not active?
- Composite: What numbered objects are on a surface and active?

'''

def sample_states(states, freq, n=None):
    weights = np.array([freq.get(k, 1) for k in states])
    weights = weights / weights.sum()
    n = min(n, len(states)) if n else None
    return np.random.choice(list(states), n, replace=False, p=weights)

def join_and(states):
    if len(states) == 2:
        return ' and '.join(states)
    if len(states) > 2:
        states = states[:-1] + [f'and {states[-1]}']
    return ', '.join(map(str, states))


def QA_describe_predicates(ann, detection_labels=None, predicate_freq=None):
    noun = ann['noun']
    all_nouns = ann['all_nouns']
    # state = ann['state']
    action = ann['action']
    candidate_state = action.get_state(action.vars[0], ann['pre_post'])
    noun_dict = action.var_dict(*all_nouns)

    # use at most 5 states
    # random.shuffle(candidate_state)
    # candidate_state = candidate_state[:8]
    candidate_state = sample_states(candidate_state, predicate_freq, 5)

    state_input = []
    state_output = []
    for s in candidate_state:
        # if len(s.vars) > len(noun_dict):
        #     continue
        if '?' in s.format(**noun_dict):
            continue
        state_input.append(s.flip(np.random.rand() < 0.3).format(**noun_dict))
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



def QA_object_id(ann, detection_labels=None, predicate_freq=None):
    object_index = np.random.randint(len(detection_labels))
    
    text_input = random.choice([
        f'What is object number {object_index}?',
        f'What object is labeled as {object_index}?',
        f'What is object #{object_index}?',
    ])

    text_output = ann['noun']

    return text_input.strip(), text_output.strip()



def QA_yes_no_predicate(ann, detection_labels=None, predicate_freq=None):
    noun = ann['noun']
    all_nouns = ann['all_nouns']
    action = ann['action']
    states = action.get_state(action.vars[0], ann['pre_post'])
    noun_dict = action.var_dict(*all_nouns)
    
    picked_state = sample_states(states, predicate_freq)

    # flip positive to negative
    flipped = np.random.rand() < 0.5
    if flipped:
        picked_state = picked_state.flip(not picked_state)

    # answer
    text_output = 'yes' if bool(picked_state) != flipped else 'no'

    # Get object description list
    objs = ''
    noun_index = -1
    if detection_labels is not None:
        objs = f"Objects in scene: {', '.join(f'{i}: {o}' for i, o in enumerate(detection_labels))}.  "
        if noun in detection_labels:
            noun_index = detection_labels.index(noun)
    
    state_str = picked_state.format(**noun_dict)
    text_input = random.choice([
        f'{objs}is {noun} {state_str}?',
        f'{objs}is {noun} {state_str}? (yes/no)',
        f'{objs}Does "{state_str}" describe the {noun}?',
        *([
            f'{objs}is #{noun_index} {state_str}?',
            f'{objs}is object #{noun_index} {state_str}?',
            f'{objs}Does "{state_str}" apply to object #{noun_index}?',
        ] if noun_index >= 0 else []),
    ])
    return text_input.strip(), text_output.strip()


def QA_object_from_predicate(ann, detection_labels=None, predicate_freq=None):
    noun = ann['noun']
    all_nouns = ann['all_nouns']
    action = ann['action']
    states = action.get_state(action.vars[0], ann['pre_post'])
    states = sample_states(states, predicate_freq, np.random.choice([2, 3], p=[0.75, 0.25]))
    noun_dict = action.var_dict(*all_nouns)

    # Get object description list
    objs = ''
    if detection_labels is not None and np.random.rand() < 0.1:
        objs = f"Objects in scene: {', '.join(f'{i}: {o}' for i, o in enumerate(detection_labels))}.  "
    
    text_output = 'none'
    if noun in detection_labels:
        text_output = str(detection_labels.index(noun))

    states_str = join_and([s.format(**noun_dict) for s in states])
    text_input = random.choice([
        f"{objs}What numbered objects are {states_str}?",
        f"{objs}Which detected objects are {states_str}?",
        f"{objs}Which listed objects are {states_str}?",
        f"{objs}Which shown objects are {states_str}?",
        f"{objs}Which detected objects are {states_str}?",
    ])
    return text_input.strip(), text_output.strip()



def QA_mix(ann, detection_labels=None, **kw):
    qas = [QA_describe_predicates, QA_yes_no_predicate,  QA_object_from_predicate]

    # required numbered main object
    if detection_labels and ann['noun'] in detection_labels:
        qas.extend([QA_object_id])

    return random.choice(qas)(ann, detection_labels, **kw)


PROMPTS = {
    "qa2": QA_describe_predicates
}


def get_prompt_function(get_prompt):
    if callable(get_prompt):
        return get_prompt
    if get_prompt:
        return PROMPTS[get_prompt]
    return QA_mix