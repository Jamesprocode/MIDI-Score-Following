"""Melody cue v2: a note-level HMM that explains each horn note as a tune note or an extra note (2026-11-04).

After Pardo & Birmingham (2005, AAAI) and Nakamura, Ono, Sagayama & Watanabe (2015, JNMR); both in Papers/.
The clock is the horn's notes. The hidden state of a note is the lead-sheet note it belongs to (k) and its role:

  PRE   an extra note before tune note k's main note (approach, grace note; Nakamura's short appoggiatura)
  MAIN  the horn playing tune note k
  POST  an extra note after it (turn, passing note; Nakamura's after note, Pardo's self-loop)

plus, with probability ``noise``, a note unrelated to the tune (transcription error, harmony part); the state then
stays where it was. From MAIN or POST of k the next note can be MAIN/PRE of k+1 (move on), POST of k (another extra
note), or MAIN/PRE of k+2 (tune note k+1 left out). From PRE of k: PRE of k again, MAIN of k, or MAIN/PRE of k+1. With
probability ``jump``, any note can instead re-enter at MAIN or PRE of any tune note near where it lands (the horn left
out several tune notes; Pardo's jumps, which our timing keeps local).
Each role has its own chance of the pitch's distance to tune note k (0-6 semitones, octave-free), of the onset's
offset from the written onset (16 bins of 1/8 beat over [-1, 1)), and of the time since the horn's previous note
(4 bins, seconds): Nakamura's clue that ornaments are fast.

Unlike both papers, the follower here stays time-driven (yamaha filter over position and tempo). The HMM is only the
melody cue: for each candidate place q, the notes of the last ``window_beats`` land at q - lag (lag from the
follower's tempo), and the cue is log P(those notes | q) per note (forward algorithm, summed over explanations), each
horn on its own.

The parameters are counted, not trained: ``fit`` aligns the head notes of one protocol (click takes or live takes,
which are different songs) to the lead sheet at the annotated true place, by Viterbi, counts the roles, moves, pitches,
offsets and gaps, and repeats (Viterbi training). The cue on a take uses the parameters counted on the other
protocol (cross-fitting). The starting values and the bins are our choices.

  ../yamaha_timing/.venv/bin/python note_hmm.py --samba "../../Dataset/SAMBA dataset" --out params/<horn set>
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from melody_evidence import MelodyConfig, _band_notes, section_kinds, written_tune
import follower_data

PRE, MAIN, POST = 0, 1, 2
ROLES = ("pre", "main", "post")
AFTER_MOVES = ("main+1", "pre+1", "post+0", "main+2", "pre+2")  # from MAIN or POST of k
PRE_MOVES = ("pre+0", "main+0", "main+1", "pre+1")              # from PRE of k
MOVE = {name: ({"pre": PRE, "main": MAIN, "post": POST}[name[:-2]], int(name[-1])) for name in AFTER_MOVES + PRE_MOVES}
TIME_BINS, TIME_SPAN = 16, 1.0          # onset offset from the written onset: 1/8-beat bins over [-1, 1)
IOI_EDGES = np.array([.1, .2, .4])      # seconds since the horn's previous note: <0.1, 0.1-0.2, 0.2-0.4, >=0.4
N_IOI = len(IOI_EDGES) + 1
MULTIPLICITY = np.array([1, 2, 2, 2, 2, 2, 1])  # pitch classes at each distance 0..6
TINY = 1e-300


def start_params():
    """Hand-set starting values for Viterbi training (our choice; fitting replaces them)."""
    time = np.ones((3, TIME_BINS))
    time[PRE, TIME_BINS // 2:] = .05
    time[POST, :TIME_BINS // 2] = .05
    centre = (np.arange(TIME_BINS) + .5) / TIME_BINS * 2 * TIME_SPAN - TIME_SPAN
    time[MAIN] = np.exp(-.5 * (centre / .25) ** 2)
    return {"after": dict(zip(AFTER_MOVES, (.7, .1, .1, .07, .03))), "pre": dict(zip(PRE_MOVES, (.2, .7, .05, .05))),
            "pitch": {"pre": [.2, .3, .3, .05, .05, .05, .05], "main": [.8, .05, .05, .03, .03, .02, .02],
                      "post": [.2, .3, .3, .05, .05, .05, .05]},
            "time": {r: list(time[i] / time[i].sum()) for i, r in enumerate(ROLES)},
            "ioi": {r: [.25] * N_IOI for r in ROLES + ("noise",)}, "noise": .1, "jump": .05, "jump_role": [.2, .8]}


@dataclass(frozen=True)
class NoteHMMConfig:
    params: str = ""                 # folder with fit_on_sync.json and fit_on_async.json
    window_beats: float = 8.0        # as v1
    min_notes: int = 2               # as v1
    max_tune_notes: int = 48         # tune notes looked at per place (the window plus a beat each side)
    lead_sheets: str = MelodyConfig.lead_sheets


class Tables:
    """Parameters as arrays, in probabilities."""

    def __init__(self, p):
        self.after = np.array([p["after"][m] for m in AFTER_MOVES])
        self.pre = np.array([p["pre"][m] for m in PRE_MOVES])
        self.pitch = np.array([np.asarray(p["pitch"][r]) / MULTIPLICITY for r in ROLES])  # per pitch class
        self.time = np.array([p["time"][r] for r in ROLES])
        self.ioi = np.array([p["ioi"][r] for r in ROLES])
        self.noise_ioi = np.array(p["ioi"]["noise"])
        self.noise = float(p["noise"])
        self.jump, self.jump_role = float(p["jump"]), np.asarray(p["jump_role"])  # [pre, main]
        self.noise_emission = 1. / 12 / TIME_BINS  # pitch and offset uniform; times the gap below


def ioi_bins(onsets):
    gaps = np.diff(onsets, prepend=-np.inf)
    return np.searchsorted(IOI_EDGES, gaps, side="right")


def time_bins(offset):
    offset = np.asarray(offset, dtype=float)
    inside = np.isfinite(offset) & (offset >= -TIME_SPAN) & (offset < TIME_SPAN)
    b = np.floor((np.where(inside, offset, 0.) + TIME_SPAN) / (2 * TIME_SPAN) * TIME_BINS).astype(int)
    return np.where(inside, np.minimum(b, TIME_BINS - 1), -1)


def distance(pitch_class, tune_class):
    d = np.abs(np.asarray(pitch_class) - np.asarray(tune_class)) % 12
    return np.minimum(d, 12 - d)


def tune_notes(data, lead_sheets):
    """The lead-sheet melody laid along the performance: onset (beats from the take's start), pitch class, and the
    beats of bars whose measure has a melody (as v1's ``known``)."""
    protocol, song, _ = data["performance_id"].split("/")
    ref = written_tune(lead_sheets, protocol, song)
    measures = [int(b["measure"]) for b in data["bars"]]
    beats = data["beats_per_bar"]
    onset, pc = [], []
    for i, m in enumerate(measures):
        here = ref["measure"] == m
        onset.append(i * beats + ref["beat"][here])
        pc.append(ref["pitch_class"][here])
    onset, pc = np.concatenate(onset), np.concatenate(pc).astype(int)
    order = np.argsort(onset, kind="stable")
    known_bar = np.array([m in ref["covered"] for m in measures])
    return onset[order], pc[order], known_bar


def horn_parts(root, performance_id):
    """Per horn: (onsets s, pitch classes) sorted by onset."""
    protocol, song, take = performance_id.split("/")
    parts, present = _band_notes(Path(root), protocol, song, take, ("horns",))
    out = []
    for notes in parts:
        notes = notes[np.argsort(notes[:, 0], kind="stable")]
        out.append((notes[:, 0], notes[:, 2].astype(int) % 12))
    return out, present


# ---- fitting: Viterbi alignment of head notes at the true place ------------------------------------------------

def viterbi(x, pcs, ioi, tune_onset, tune_pc, t, beam=40):
    """Best explanation of one horn's notes in one head. x: true place of each onset (beats). Returns per note
    (kind, role, tune index, move): kind "noise" or "tune"; move is the AFTER/PRE move name (None for the first)."""
    log = lambda v: np.log(np.maximum(v, TINY))
    # (role, delta) -> (move, log probability), per group of the previous role
    from_after = {MOVE[m]: (m, lp) for m, lp in zip(AFTER_MOVES, log(t.after))}
    from_pre = {MOVE[m]: (m, lp) for m, lp in zip(PRE_MOVES, log(t.pre))}
    l_keep, l_noise = np.log(1 - t.noise), np.log(t.noise)
    l_chain, l_jump = np.log(1 - t.jump), np.log(t.jump) + log(t.jump_role)
    steps, back = [], []
    prev = {}
    for n in range(len(x)):
        lo, hi = np.searchsorted(tune_onset, [x[n] - TIME_SPAN, x[n] + TIME_SPAN])
        emis = {}
        for k in range(lo, hi):
            tb = time_bins(np.array([x[n] - tune_onset[k]]))[0]
            if tb < 0:
                continue
            d = distance(pcs[n], tune_pc[k])
            for r in (PRE, MAIN, POST):
                emis[(k, r)] = log(t.pitch[r, d] * t.time[r, tb] * t.ioi[r, ioi[n]])
        cur, ptr = {}, {}
        if not prev:
            first = {PRE: t.after[1] + t.after[4], MAIN: t.after[0] + t.after[3], POST: t.after[2]}
            for (k, r), e in emis.items():
                cur[(k, r)] = e + log(first[r]) - np.log(max(hi - lo, 1))
                ptr[(k, r)] = (None, None)
        else:
            top_state = max(prev, key=prev.get)
            top_value = prev[top_state]
            for (k, r), e in emis.items():
                best, arg = -np.inf, None
                for delta in (0, 1, 2):
                    for r0 in (PRE, MAIN, POST):
                        v = prev.get((k - delta, r0))
                        move = (from_pre if r0 == PRE else from_after).get((r, delta))
                        if v is not None and move is not None and v + move[1] > best:
                            best, arg = v + move[1], ((k - delta, r0), move[0])
                if arg is not None:
                    best += l_chain
                if r in (PRE, MAIN):
                    v = top_value + l_jump[0 if r == PRE else 1] - np.log(max(hi - lo, 1))
                    if v > best:
                        best, arg = v, (top_state, "jump")
                if arg is not None:
                    cur[(k, r)], ptr[(k, r)] = best + l_keep + e, arg
            noise = l_noise + log(t.noise_emission * t.noise_ioi[ioi[n]])
            for state, v in prev.items():
                if v + noise > cur.get(state, -np.inf):
                    cur[state], ptr[state] = v + noise, (state, "noise")
        if not cur:  # nothing explains this note and nothing came before: start again after it
            steps.append({}), back.append({})
            prev = {}
            continue
        top = sorted(cur, key=cur.get, reverse=True)[:beam]
        prev = {s: cur[s] for s in top}
        steps.append(prev), back.append({s: ptr[s] for s in top})
    # trace back each stretch separately
    out = [None] * len(x)
    n = len(x) - 1
    while n >= 0:
        if not steps[n]:
            out[n] = ("noise", None, None, None)
            n -= 1
            continue
        state = max(steps[n], key=steps[n].get)
        while n >= 0 and steps[n]:
            source, move = back[n][state]
            if move == "noise":
                out[n] = ("noise", None, None, None)
            else:
                out[n] = ("tune", state[1], state[0], move)
            n -= 1
            if source is None:
                break
            state = source
    return out


def fit(takes, root, lead_sheets, iterations=6, log=print):
    """Viterbi training over the head notes of ``takes``. Returns the parameters and a description of the last
    alignment (the measurement)."""
    heads = []
    for data, _ in takes:
        onset, pc, _ = tune_notes(data, lead_sheets)
        parts, present = horn_parts(root, data["performance_id"])
        for (on, pcs), horn in zip(parts, present):
            if not len(on):
                continue
            is_head = section_kinds(root, data["performance_id"], on) == "head"
            x = np.interp(on, data["times"], data["truth_position_beats"])
            ioi = ioi_bins(on)
            # one head at a time: split where the head flag changes
            edges = np.flatnonzero(np.diff(np.r_[0, is_head.astype(int), 0]))
            for a, b in zip(edges[::2], edges[1::2]):
                heads.append((data["performance_id"], horn, x[a:b], pcs[a:b], ioi[a:b], on[a:b], onset, pc))
    params = start_params()
    for it in range(iterations):
        t = Tables(params)
        after, pre = np.ones(len(AFTER_MOVES)), np.ones(len(PRE_MOVES))  # add-one smoothing
        pitch, time, ioi_c = np.ones((3, 7)), np.ones((3, TIME_BINS)), np.ones((4, N_IOI))
        noise = tune = jumps = moves = 0
        jump_role = np.ones(2)
        roles = np.zeros(3)
        main_lag = []
        for pid, horn, x, pcs, ioi, on, onset, pc in heads:
            path = viterbi(x, pcs, ioi, onset, pc, t)
            prev_role = None
            for n, (kind, role, k, move) in enumerate(path):
                if kind == "noise":
                    noise += 1
                    ioi_c[3, ioi[n]] += 1
                    continue
                tune += 1
                roles[role] += 1
                if move == "jump":
                    jumps += 1
                    moves += 1
                    jump_role[0 if role == PRE else 1] += 1
                elif move is not None and move != "noise":
                    moves += 1
                    if move in AFTER_MOVES and prev_role in (MAIN, POST):
                        after[AFTER_MOVES.index(move)] += 1
                    elif move in PRE_MOVES and prev_role == PRE:
                        pre[PRE_MOVES.index(move)] += 1
                pitch[role, distance(pcs[n], pc[k])] += 1
                time[role, time_bins(np.array([x[n] - onset[k]]))[0]] += 1
                ioi_c[role, ioi[n]] += 1
                if role == MAIN:
                    main_lag.append(x[n] - onset[k])
                prev_role = role
        params = {"after": dict(zip(AFTER_MOVES, (after / after.sum()).round(5).tolist())),
                  "pre": dict(zip(PRE_MOVES, (pre / pre.sum()).round(5).tolist())),
                  "pitch": {r: (pitch[i] / pitch[i].sum()).round(5).tolist() for i, r in enumerate(ROLES)},
                  "time": {r: (time[i] / time[i].sum()).round(5).tolist() for i, r in enumerate(ROLES)},
                  "ioi": {r: (ioi_c[i] / ioi_c[i].sum()).round(5).tolist() for i, r in enumerate(ROLES + ("noise",))},
                  "noise": round(noise / max(noise + tune, 1), 5),
                  "jump": round((jumps + 1) / (moves + 2), 5), "jump_role": (jump_role / jump_role.sum()).round(5).tolist()}
        log(f"iteration {it + 1}: {noise + tune} notes, noise {params['noise']:.3f}, roles "
            + " ".join(f"{r} {v / max(tune, 1):.3f}" for r, v in zip(ROLES, roles)) + f", jump {params['jump']:.3f}")
    main_lag = np.array(main_lag)
    params["measured"] = {
        "heads": len(heads), "notes": int(noise + tune),
        "share": {"noise": round(noise / (noise + tune), 3),
                  **{r: round(v / (noise + tune), 3) for r, v in zip(ROLES, roles)}},
        "main_right_pitch": round(float(params["pitch"]["main"][0]), 3),
        "main_offset_beats": {"median": round(float(np.median(main_lag)), 3),
                              "median_abs": round(float(np.median(np.abs(main_lag))), 3)},
        "skip_after_main_or_post": round(params["after"]["main+2"] + params["after"]["pre+2"], 3),
        "jump": params["jump"],
        "fast_under_0.1s": {r: round(params["ioi"][r][0], 3) for r in ROLES + ("noise",)}}
    return params


# ---- the cue ----------------------------------------------------------------------------------------------------

def _shift(a, s):
    """a[:, j - s] at j (zeros where j < s), along axis 1."""
    out = np.zeros_like(a)
    out[:, s:] = a[:, :a.shape[1] - s]
    return out


class NoteHMMEvidence:
    """Same interface as melody_evidence.MelodyEvidence: ``scores`` (here: log P(recent notes | place) per note, higher
    is better) and ``log_likelihood``, on the follower's grid."""

    def __init__(self, data, root, config=NoteHMMConfig(), bins_per_beat=8):
        protocol = data["performance_id"].split("/")[0]
        other = {"sync": "async", "async": "sync"}[protocol]
        self.params_file = Path(config.params) / f"fit_on_{other}.json"  # cross-fitting: never this protocol's songs
        self.t = Tables(json.loads(self.params_file.read_text()))
        self.config, self.step = config, 1. / bins_per_beat
        self.tune_onset, self.tune_pc, known_bar = tune_notes(data, config.lead_sheets)
        per_bar = int(round(data["beats_per_bar"] * bins_per_beat))
        self.known = np.repeat(known_bar, per_bar)
        # Places whose surrounding measures are the same (every chorus) get the same score: compute it once, at the
        # first such place. The context is the place in its bar and the measures from the window's start to a bar on.
        measures = [int(b["measure"]) for b in data["bars"]]
        back = int(np.ceil((config.window_beats + TIME_SPAN) / data["beats_per_bar"])) + 1
        first, self.rep = {}, np.empty(len(self.known), dtype=int)
        for g in range(len(self.known)):
            b = g // per_bar
            key = (g % per_bar,) + tuple(measures[i] if 0 <= i < len(measures) else -1 for i in range(b - back, b + 2))
            self.rep[g] = first.setdefault(key, g)
        self.tune = self.known  # grid length, as v1's ``tune``
        self.reference = {"beat": self.tune_onset}
        parts, self.horns = horn_parts(root, data["performance_id"])
        self.parts = [(on, pcs, ioi_bins(on)) for on, pcs in parts]

    def _horn(self, x, pcs, ioi):
        """log P(notes | place) summed over explanations, for every place (rows of x: landing places of the notes)."""
        t, Y, C = self.t, self.tune_onset, self.tune_pc
        base = np.clip(np.searchsorted(Y, x.min(axis=1) - TIME_SPAN) - 1, 0, None)
        top = np.searchsorted(Y, x.max(axis=1) + TIME_SPAN)
        K = int(min(max((top - base).max(), 1) + 1, self.config.max_tune_notes))
        J = base[:, None] + np.arange(K)
        valid = J < len(Y)
        Jc = np.minimum(J, len(Y) - 1)
        Yw, Cw = np.where(valid, Y[Jc], np.inf), C[Jc]
        alpha = np.zeros((len(x), K, 3))
        alpha[:, :, MAIN] = valid / np.maximum(valid.sum(axis=1, keepdims=True), 1)  # "just after tune note j"
        logp = np.zeros(len(x))
        for n in range(x.shape[1]):
            tb = time_bins(x[:, n:n + 1] - Yw)
            ok = (tb >= 0) & valid
            d = distance(pcs[n], Cw)
            emission = t.pitch[:, d] * np.where(ok, t.time[:, np.maximum(tb, 0)], 0.) * t.ioi[:, ioi[n], None, None]
            after, pre = alpha[:, :, MAIN] + alpha[:, :, POST], alpha[:, :, PRE]
            a, p = t.after, t.pre
            new = np.empty_like(alpha)
            new[:, :, MAIN] = _shift(after, 1) * a[0] + _shift(after, 2) * a[3] + pre * p[1] + _shift(pre, 1) * p[2]
            new[:, :, PRE] = _shift(after, 1) * a[1] + _shift(after, 2) * a[4] + pre * p[0] + _shift(pre, 1) * p[3]
            new[:, :, POST] = after * a[2]
            new *= 1 - t.jump
            reenter = t.jump * ok / np.maximum(ok.sum(axis=1, keepdims=True), 1)  # alpha sums to 1 per place
            new[:, :, PRE] += reenter * t.jump_role[0]
            new[:, :, MAIN] += reenter * t.jump_role[1]
            new *= np.moveaxis(emission, 0, -1)
            total = (1 - t.noise) * new + t.noise * t.noise_emission * t.noise_ioi[ioi[n]] * alpha
            c = np.maximum(total.sum(axis=(1, 2)), TINY)
            alpha = total / c[:, None, None]
            logp += np.log(c)
        return logp

    def scores(self, now, bpm, n_positions):
        window = self.config.window_beats * 60. / bpm
        n = min(n_positions, len(self.known))
        places = np.flatnonzero(self.known[:n])
        if not len(places):
            return None
        reps, back = np.unique(self.rep[places], return_inverse=True)
        places_all, places = places, reps
        total, count = np.zeros(len(places)), 0
        for on, pcs, ioi in self.parts:
            lo, hi = np.searchsorted(on, [now - window, now], side="right")
            if hi - lo == 0:
                continue
            lags = (now - on[lo:hi]) * bpm / 60.
            x = places[:, None] * self.step - lags[None, :]
            total += self._horn(x, pcs[lo:hi], ioi[lo:hi])
            count += hi - lo
        if count < self.config.min_notes:
            return None
        per_note = (total / count)[back]
        out = np.full(n_positions, per_note.mean())  # no melody here: the average (no opinion), as v1
        out[places_all] = per_note
        return out

    def log_likelihood(self, now, bpm, n_positions):
        return self.scores(now, bpm, n_positions)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samba", type=Path, required=True)
    parser.add_argument("--lead-sheets", default=MelodyConfig.lead_sheets)
    parser.add_argument("--out", type=Path, required=True, help="folder for fit_on_sync.json and fit_on_async.json")
    parser.add_argument("--iterations", type=int, default=6)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for protocol in ("async", "sync"):
        takes = follower_data.samba_follow_takes(args.samba, protocols=(protocol,))
        print(f"== fit on {protocol}: {len(takes)} takes", flush=True)
        params = fit(takes, args.samba, args.lead_sheets, args.iterations, log=lambda s: print(s, flush=True))
        (args.out / f"fit_on_{protocol}.json").write_text(json.dumps(params, indent=1))
        print(json.dumps(params["measured"], indent=1), flush=True)


if __name__ == "__main__":
    main()
