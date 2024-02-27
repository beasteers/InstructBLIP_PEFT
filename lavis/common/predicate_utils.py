import os
import re


class Predicate:
    def __init__(self, text, unknown=None, positive=None, negative=None):
        self.unknown = unknown
        self.positive = positive
        self.negative = negative
        if isinstance(text, Predicate):
            self.name = text.name
            self.neg = text.neg
            self.vars = list(text.vars)
            self.positive = positive or text.positive
            self.negative = negative or text.negative
            self.unknown = unknown or text.unknown
        else:
            m = re.findall(r'\((?:(not) \()?([=\w-]+)\s((?:\??\w+\s*)+)\)?\)', text)
            if not m:
                raise ValueError(f"could not parse {text}")
            self.neg, self.name, var = m[0]
            self.vars = var.split()
        
        self.neg = bool(self.neg)

    def __hash__(self):
        return hash((self.name, len(self.vars)))

    def __eq__(self, other):
        return (self.name, len(self.vars)) == (other.name, len(other.vars))

    def __str__(self):
        p = f'({self.name} {" ".join(self.vars)})'
        return f'(not {p})' if self.neg else p
    __repr__ = __str__

    def format(self, **nouns):
        nouns = [nouns.get(x.strip('?'), x) for x in self.vars]
        if not self.neg and self.positive:
            return self.positive.format(*nouns)
        if self.neg and self.negative:
            return self.negative.format(*nouns)
        p=Predicate(self)
        p.vars = nouns
        return str(self)

    def translate_vars(self, ref, *others):
        """
        (above b a) .translate_vars(
            (above x y),
            (below y x), (not (behind x y))
        ) ->
        (below a b), (not (behind b a))
        """
        trans = dict(zip(self.vars, ref.vars))
        others = [Predicate(o) for o in others]
        for o in others:
            o.vars = [trans.get(x,x) for x in o.vars]
        return others

    def switch(self, on):
        p=Predicate(self)
        p.neg=not on
        return p


class Action:
    def __init__(self, name, vars, pre, post, skip_nouns=1):
        self.name = name
        self.vars = vars
        self.pre = pre
        self.post = post
        self.skip_nouns = skip_nouns

    def __str__(self):
        return f'{self.name}({" ".join(self.vars)})\n  pre: {", ".join(map(str, self.pre))}.\n  post: {", ".join(map(str, self.post))}.'

    def var_dict(self, *nouns):
        return dict(zip([x.strip('?') for x in self.vars], nouns))

    def get_state(self, var, when):
        if isinstance(var, int): var = self.vars[var]
        return [p for p in (self.pre if when == 'pre' else self.post) if p.vars[0] == var]

    def get_state_text(self, var, when, *nouns):
        nouns = self.var_dict(*nouns)
        return [p.format(**nouns) for p in self.get_state(var)]

    def postconditions(self, var, *nouns):
        return format_predicates(self.post, var, self.vars, nouns)

    @classmethod
    def from_dict(cls, data, translations):
        pre = [Predicate(p) for p in data['preconditions']]
        post = [Predicate(p) for p in data['effects']]
        for p in pre + post:
            t = translations[p]
            if t:
                p.positive = t['positive']
                p.negative = t['negative']
        return Action(data['name'], data['params'], pre, post)


def format_predicates(predicates, var, vars, nouns):
    assert var in vars, f'{var} not in {vars}'
    nouns = dict(zip([x.strip('?') for x in vars], nouns))
    return [p.format(**nouns) for p in predicates if p.vars[0] == var]


def add_axioms(current, axioms):
    for x in list(current):
        for a in axioms:
            if a['context'] == x:
                current.extend(a['context'].translate_vars(x, *a['implies']))
    return list(set(current))



def load_pddl_yaml(fname):
    import yaml
    with open(fname, 'r') as f:
        data = yaml.safe_load(f)
    
    predicates = [Predicate(x) for x in data['predicates']]
    predicates += [p.switch(False) for p in predicates]

    translations: dict[str, str] = data['translations']
    actions: list[dict] = data['definitions']
    axioms: list[dict] = data['axioms']
    for d in axioms:
        d['context'] = Predicate(d['context'])
        d['implies'] = [Predicate(x) for x in d['implies']]
    for d in actions:
        p = [Predicate(x) for x in d['preconditions']]
        e = [Predicate(x) for x in d['effects']]
        p = [x for x in add_axioms(p, axioms) if x in predicates]
        e = [x for x in add_axioms(e, axioms) if x in predicates]
        d['preconditions'] = p
        d['effects'] = e

    translations = {Predicate(p): t for p, t in data['translations'].items()}
    actions = {
        d['name']: Action.from_dict(d, translations)
        for d in actions
    }
    return actions, predicates
