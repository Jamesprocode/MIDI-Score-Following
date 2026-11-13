"""Named followers, as data: which cues a follower listens to, with which weight, and any HMM setting that differs
from B5's.

A cue is named in CUES, with a factory that builds it for one take. A follower is cue: weight pairs, in the order
their evidence is added up (harmony, rhythm, melody, as the old runners added them: the order changes the last bit of
the sums, so keep it to reproduce old runs bit for bit). A cue with weight 0 is not built.

To try a new chord model or beat tracker: write its cue (cues/), add one line to CUES, then name it in a follower,
here or on run.py's command line:

  B5+v2w1                                      a named follower
  "B5new=harmony_new:1,rhythm:1"               written out: label=cue:weight,...
  "B5fix=harmony:1,rhythm:1;exact_beat_crossing=true"   HMM settings after ";" (hmm.HMMConfig fields)
  "B5z=harmony@z:1,rhythm@z:1"                a scaling rule per cue after "@" (scaling.py; default "max")
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
import functools
import json
from pathlib import Path

from .cues.harmony import b5_harmony
from .cues.melody import melody_v1, melody_v2
from .cues.rhythm import network_downbeat
from .hmm import HMMConfig, Plugged
from .scaling import FITTED, RULES, resolve


@dataclass(frozen=True)
class Resources:
    """Files the cues load, chosen once per run (run.py's arguments) and recorded in run.json. Nothing is picked
    from a default or an environment variable here."""
    checkpoint: str = ""        # the timing network (rhythm cue)
    melody_params: str = ""     # a folder of fitted note-HMM tables (melody v2); see note_hmm.NoteHMMEvidence
    lead_sheets: str = ""       # the lead-sheet melodies (melody v1 and v2); "" = MelodyConfig's default
    scaling_params: str = ""    # fit_scaling.py's file: each cue's mean and std (the "z" and "zraw" rules)

    def model(self):
        if not self.checkpoint:
            raise ValueError("the rhythm cue needs the timing network: --checkpoint")
        return _load_model(self.checkpoint)

    def params(self):
        if not self.melody_params:
            raise ValueError("melody v2 needs its fitted tables: --melody-params FOLDER")
        return self.melody_params


@functools.lru_cache(maxsize=None)
def _load_model(checkpoint):
    """One copy per process (each worker loads its own)."""
    from . import legacy  # noqa: F401
    from yamaha_timing.training import load_checkpoint
    return load_checkpoint(checkpoint, "cpu")[0]


@dataclass
class Inputs:
    """What a cue factory gets for one take."""
    data: dict          # follower_data's plan and truth
    take: object        # the piano notes (yamaha_timing take)
    times: object       # the update grid (takes.update_times)
    samba: Path
    resources: Resources


CUES = {  # name: factory(Inputs) -> cue
    "harmony": lambda x: b5_harmony(x.data, x.times),
    "rhythm": lambda x: network_downbeat(x.resources.model(), x.take, x.data["beats_per_bar"]),
    "melody_v1": lambda x: melody_v1(x.data, x.samba, x.resources.lead_sheets or None),
    "melody_v2": lambda x: melody_v2(x.data, x.samba, x.resources.params(), x.resources.lead_sheets or None),
    "melody_v2max": lambda x: melody_v2(x.data, x.samba, x.resources.params(), x.resources.lead_sheets or None,
                                        combine="max"),
}
NEEDS = {"rhythm": "checkpoint", "melody_v2": "melody_params", "melody_v2max": "melody_params"}


@dataclass
class Spec:
    cues: dict                                       # cue name -> weight, in the order the evidence is added
    hmm: dict = field(default_factory=dict)          # HMMConfig fields that differ from B5's
    scaling: dict = field(default_factory=dict)      # cue name -> scaling rule (default "max")

    def __post_init__(self):
        unknown = [c for c in self.cues if c not in CUES]
        if unknown:
            raise ValueError(f"unknown cue(s) {unknown}; known: {sorted(CUES)}")
        bad = {c: s for c, s in self.scaling.items() if s not in RULES or c not in self.cues}
        if bad:
            raise ValueError(f"bad scaling {bad}; rules: {list(RULES)}")
        replace(HMMConfig(), **self.hmm)  # raises on an unknown field

    def config(self):
        return replace(HMMConfig(), **self.hmm)

    def record(self):
        """As written into run.json."""
        return {"cues": {c: float(w) for c, w in self.cues.items()},
                "scaling": {c: self.scaling.get(c, "max") for c, w in self.cues.items() if w},
                "hmm": asdict(self.config())}


def _b5(**melody):
    return {"harmony": 1., "rhythm": 1., **melody}


FOLLOWERS = {
    "none": Spec({}),                                     # the tempo model alone, counting from the start
    "harmony": Spec({"harmony": 1.}),
    "rhythm": Spec({"rhythm": 1.}),
    "B5": Spec(_b5()),                                    # Maezawa (2023) as built here: harmony x network downbeat
    "v1": Spec({"melody_v1": 1.}),
    "v2": Spec({"melody_v2": 1.}),
    "v2max": Spec({"melody_v2max": 1.}),
    "v2+rhythm": Spec({"rhythm": 1., "melody_v2": 1.}),
    "v2max+rhythm": Spec({"rhythm": 1., "melody_v2max": 1.}),
    "B5+v2": Spec(_b5(melody_v2=.5)),
    "B5+v2max": Spec(_b5(melody_v2max=.5)),
    "B5+v1w1": Spec(_b5(melody_v1=1.)),                   # equal weights (11-05)
    "B5+v2w1": Spec(_b5(melody_v2=1.)),                   # "B5+H1" in run_melody_follower.py
    "B5+v2maxw1": Spec(_b5(melody_v2max=1.)),
    # per-cue z-score, mean and std from the 60 training songs (11-13; scaling.py, fit_scaling.py): weights 1, so
    # each cue's evidence is its score over its own std
    "B5z": Spec(_b5(), scaling={"harmony": "z", "rhythm": "z"}),
    "B5+v2z": Spec(_b5(melody_v2=1.), scaling={"harmony": "z", "rhythm": "z", "melody_v2": "z"}),
    "B5+v2zraw": Spec(_b5(melody_v2=1.), scaling={"harmony": "zraw", "rhythm": "zraw", "melody_v2": "zraw"}),
    "B5+v2zw": Spec(_b5(melody_v2=1.), scaling={"harmony": "zw", "rhythm": "zw", "melody_v2": "zw"}),
    # control for B5+v2z: B5+v2's weights all times 0.838, which gives the same total evidence as B5+v2z (sum over
    # cues of weight x within-moment spread, params/zscore_train.json: 1.886 vs B5+v2's 2.249), so z's effect on the
    # balance between cues can be told apart from its effect on the total
    "B5+v2s": Spec({"harmony": .838, "rhythm": .838, "melody_v2": .419}),
}


def _value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse(text):
    """(label, Spec) for a follower named in FOLLOWERS or written out (see the module docstring)."""
    if text in FOLLOWERS:
        return text, FOLLOWERS[text]
    label, eq, body = text.partition("=")
    if not eq or not label or not body:
        raise ValueError(f"unknown follower {text!r}: name one of {sorted(FOLLOWERS)} or write label=cue:weight,...")
    cue_part, *settings = body.split(";")
    cues, scaling = {}, {}
    for item in filter(None, cue_part.split(",")):
        name, colon, weight = item.partition(":")
        name, at, rule = name.partition("@")
        cues[name.strip()] = float(weight) if colon else 1.
        if at:
            scaling[name.strip()] = rule.strip()
    hmm = {}
    for setting in filter(None, settings):
        key, _, value = setting.partition("=")
        hmm[key.strip()] = _value(value.strip())
    names = {f.name for f in fields(HMMConfig)}
    if set(hmm) - names:
        raise ValueError(f"unknown HMM setting(s) {sorted(set(hmm) - names)}; known: {sorted(names)}")
    return label, Spec(cues, hmm, scaling)


def build(spec, inputs):
    """The follower's plugged cues for one take, its HMM config, and what each cue loaded (for the frame file)."""
    plugged, info = [], {}
    for name, weight in spec.cues.items():
        if not weight:
            continue
        cue = CUES[name](inputs)
        rule = spec.scaling.get(name, "max")
        scaling = resolve(rule, name, inputs.resources.scaling_params)
        plugged.append(Plugged(cue, float(weight), scaling))
        params_file = getattr(getattr(cue, "matcher", None), "params_file", None)
        if params_file is not None:
            info.setdefault(name, {})["params_file"] = str(params_file)
        if rule in FITTED:
            info.setdefault(name, {})["scaling"] = {"rule": rule, "mean": scaling.mean, "std": scaling.std,
                                                    "base": scaling.base}
    return plugged, spec.config(), info


def needs(specs):
    """Which Resources fields the followers need."""
    specs = list(specs)
    fitted = any(s.scaling.get(c) in FITTED for s in specs for c, w in s.cues.items() if w)
    return sorted({NEEDS[c] for s in specs for c, w in s.cues.items() if w and c in NEEDS}
                  | ({"scaling_params"} if fitted else set()))
