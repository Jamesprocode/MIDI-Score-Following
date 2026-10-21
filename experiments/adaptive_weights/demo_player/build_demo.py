"""Build the demo page's data: the B7 follower with fixed cue weights on selected JazzSAMBA takes.

A copy of yamaha_follower/player (unchanged there) with a lead-sheet view. Each
follower is B7 (per-beat tempo walk with the 0.02 pull toward the played tempo,
tempo feedback, 50 ms updates, 32 positions per bar), run with fixed weights on
the two cues, exactly as in sweep_fixed_weights.py:

  fixed     harmony 1 : rhythm 1, Maezawa's product p_h * p_d
  rhythm    rhythm only (harmony weight 0)
  harmony   harmony only (rhythm weight 0)

Shown at the 200 ms evaluation instants. The tempo-only clock on the page runs
at the played tempo, the same tempo the followers start from.

  ../../yamaha_timing/.venv/bin/python build_demo.py --samba "../../../Dataset/SAMBA dataset" \\
      --checkpoint ../../yamaha_timing/runs/lakh_r3b/best_model.pt
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "yamaha_follower" / "player"))
sys.path.insert(0, str(HERE.parent.parent / "yamaha_follower"))
import follower_data  # noqa: E402  (sets sys.path for the benchmark modules)
from build_player import annotation_rows, midi_with_velocity, numeric  # noqa: E402
import run_study  # noqa: E402
from evidence import NetworkPhase  # noqa: E402
from run_follow import position_metrics  # noqa: E402
from run_follow_b import STEP, played_bpm  # noqa: E402
from yamaha_filter import YamahaConfig, follow  # noqa: E402

B7 = YamahaConfig(beat_tempo_reversion=0.02)
FOLLOWERS = {
    "fixed": ((1.0, 1.0), "Fixed weights (Yamaha): harmony 1 : rhythm 1",
              "Harmony evidence times the network's downbeat phase, both weights fixed at 1 (Maezawa 2023's "
              "product), for every song and every moment."),
    "rhythm": ((0.0, 1.0), "Rhythm only: harmony 0 : rhythm 1",
               "The network's downbeat phase alone; harmony is ignored."),
    "harmony": ((1.0, 0.0), "Harmony only: harmony 1 : rhythm 0",
                "Harmony evidence alone; the timing network is ignored."),
}
TAKES = {
    "green_dolphin": ("sync/67-on_green_dolphin_street/preferred", "On Green Dolphin Street"),
    "amazing_grace": ("sync/51-amazing_grace/preferred", "Amazing Grace"),
    "samba_de_orfeu": ("sync/70-samba_de_orfeu/preferred", "Samba de Orfeu"),
    "dont_fence_me_in": ("sync/58-dont_fence_me_in/preferred", "Don't Fence Me In"),
    "la_vie_en_rose": ("sync/63-la_vie_en_rose/preferred", "La Vie en Rose"),
    "sunny_side": ("sync/68-on_the_sunny_side_of_the_street/alternate", "On the Sunny Side of the Street"),
    "invitation": ("sync/61-invitation/preferred", "Invitation"),
    "almost_like": ("sync/50-almost_like_being_in_love/preferred", "Almost Like Being in Love"),
    "blackbird": ("sync/55-bye_bye_blackbird/preferred", "Bye Bye Blackbird"),
}


def run_followers(data, take, model):
    sample = np.arange(0, len(data["times"]), 2)
    eval_times = data["times"][sample]
    fine = np.round(np.arange(eval_times[0], data["times"][-1] + 1e-9, STEP), 6)
    at = np.round((eval_times - fine[0]) / STEP).astype(int)
    case = run_study.get_features(dict(data, times=fine), {"triad"})["triad"]
    scores, available, mass = case["features"]["active_pooled"]
    probabilities, _ = run_study.observation(scores, available, mass, {"scaling": "zscore", "reliability": "none"})
    network = NetworkPhase(model, take)
    bpm = played_bpm(data)
    runs = {}
    for fid, ((w_h, w_d), _, _) in FOLLOWERS.items():
        config = replace(B7, emission_power_per_second=w_h, downbeat_power_per_second=w_d)
        out = follow(fine, probabilities, available, data["chord_starts_beats"], data["chord_ends_beats"],
                     case["targets"]["ids"], data["bar_starts_beats"], data["beats_per_bar"], bpm,
                     start_time=data["start_time"], network=network if w_d else None, config=config)
        runs[fid] = {k: np.asarray(out[k])[at] for k in ("position_beats", "tempo_bpm", "position_confidence",
                                                         "bar_index")}
    return sample, eval_times, np.asarray(available)[at], runs, bpm


def build(root, model, key, performance_id, title, takes_by_id, catalog):
    data, take = takes_by_id[performance_id]
    protocol, song, name = performance_id.split("/")
    sample, times, available, runs, bpm = run_followers(data, take, model)
    first_bar, bpb = data["first_annotation_bar"], data["beats_per_bar"]
    bars = []
    for row in annotation_rows(root, protocol, song, name, "bars"):
        bar = numeric(row)
        bar["padding"] = bar["measure"] == 0
        bar["startBeat"] = (bar["bar"] - first_bar) * bpb
        bar["endBeat"] = bar["startBeat"] + bpb
        bar["referenceBpm"] = bpb * 60 / (bar["end_time"] - bar["start_time"]) if not bar["padding"] else None
        bars.append(bar)
    chords = [{"label": r["chord"], "noChord": r["is_no_chord"] == "True",
               "startBeat": (float(r["start_bar"]) - first_bar + float(r["start_fraction"])) * bpb,
               "endBeat": (float(r["end_bar"]) - first_bar + float(r["end_fraction"])) * bpb}
              for r in annotation_rows(root, protocol, song, name, "chords")]
    musicians = {}
    for r in annotation_rows(root, protocol, song, name, "section_musicians"):
        musicians.setdefault(int(r["section_index"]), []).append(r["instrument"])
    sections = [{**numeric(r), "label": r["section"], "players": musicians.get(int(r["section_index"]), []),
                 "startBeat": (float(r["start_bar"]) - first_bar + float(r["start_fraction"])) * bpb,
                 "endBeat": (float(r["end_bar"]) - first_bar + float(r["end_fraction"])) * bpb}
                for r in annotation_rows(root, protocol, song, name, "sections")]
    musical = [b for b in bars if not b["padding"]]
    bar_tempi = np.array([b["referenceBpm"] for b in musical])
    reference_bpm = bar_tempi[data["truth_bar_index"][sample]]
    truth = data["truth_position_beats"][sample]
    scale, shift = take.info.get("scale", 1.0), take.info.get("shift_s", 0.0)
    take_dir = root / f"{protocol}_audio/songs/{song}/{name}"
    midi_dir = root / f"{protocol}_midi/songs/{song}/{name}"
    tracks = [{"id": "piano", "label": "Piano (follower input)",
               "notes": midi_with_velocity(take_dir / "piano.mid", scale, shift)}]
    for track_id, label, path in (("bass", "Bass (transcribed)", midi_dir / "bass.transcribed.derived.mid"),
                                  ("drums", "Drums", take_dir / "drums.mid"),
                                  ("drums", "Drums (transcribed)", midi_dir / "drums.transcribed.derived.mid")):
        if path.exists() and all(t["id"] != track_id for t in tracks):
            tracks.append({"id": track_id, "label": label, "notes": midi_with_velocity(path)})
    followers = {}
    for fid, ((w_h, w_d), label, description) in FOLLOWERS.items():
        run = runs[fid]
        samples = [{"time": round(float(t), 6), "truthPositionBeats": float(truth[i]),
                    "estimatedPositionBeats": float(run["position_beats"][i]),
                    "estimatedBpm": float(run["tempo_bpm"][i]), "referenceBpm": float(reference_bpm[i]),
                    "truthBar": int(data["truth_bar_index"][sample[i]]) + first_bar,
                    "estimatedBar": int(run["bar_index"][i]) + first_bar,
                    "harmonyAvailable": bool(available[i])} for i, t in enumerate(times)]
        metrics = position_metrics({"position_beats": run["position_beats"], "bar_index": run["bar_index"]},
                                   data, sample, times)
        followers[fid] = {"label": label, "description": description, "instruments": ["piano"], "samples": samples,
                          "weights": {"harmony": w_h, "rhythm": w_d},
                          "metrics": metrics, "evidenceCoverage": float(np.mean(available))}
    meta = catalog[song]
    end = max(n[1] for t in tracks for n in t["notes"])
    payload = {
        "metadata": {
            "title": title, "key": meta.get("key_signature", ""), "form": meta.get("form", ""), "recordingKey": key,
            "performanceId": performance_id, "protocol": "Synchronous" if protocol == "sync" else "Asynchronous",
            "take": name.capitalize(), "bpm": round(bpm, 1), "writtenBpm": data["initial_bpm"],
            "meter": meta["time_signature"], "beatsPerBar": bpb, "updateSeconds": 0.2,
            "musicalStartSeconds": data["start_time"], "annotationEndSeconds": data["end_time"],
            "durationSeconds": end, "scoreEndBeats": data["score_end_beats"], "musicalBarCount": len(musical),
            "firstAnnotationBar": first_bar,
            "groundTruthDescription": "JazzSAMBA bar annotations; position within each bar is interpolated linearly.",
            "playbackDescription": "Piano: recorded MIDI, the followers' only input" +
                                   (" (sync timing repaired: 175 BPM session clock and a start shift fitted from "
                                    "the piano audio)" if protocol == "sync" else "") +
                                   ". Bass and drums: dataset MIDI for listening only. Browser sound is synthesized.",
            "sampleDescription": "Followers update every 50 ms and are shown at the 200 ms checks."},
        "tracks": tracks, "bars": bars, "chords": chords, "sections": sections, "followers": followers,
        "sources": {"piano": {"path": str(take_dir / "piano.mid")}},
    }
    (HERE / f"{key}.json").write_text(json.dumps(payload))
    return {key: {"file": f"{key}.json", "title": title, "protocol": payload["metadata"]["protocol"],
                  "metrics": {fid: followers[fid]["metrics"]["bar_accuracy"] for fid in followers}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samba", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--takes", nargs="*", default=list(TAKES), help=f"keys from {list(TAKES)}")
    args = parser.parse_args()
    import torch
    from yamaha_timing.training import load_checkpoint
    torch.set_num_threads(2)
    model = load_checkpoint(args.checkpoint, "cpu")[0]
    catalog = {r["song_subdir_basename"]: r for r in csv.DictReader(open(args.samba / "JazzSAMBA-catalog/songs.csv"))}
    takes_by_id = {d["performance_id"]: (d, t) for d, t in follower_data.samba_follow_takes(args.samba)}
    menu_path = HERE / "recordings.json"
    recordings = json.loads(menu_path.read_text())["recordings"] if menu_path.exists() else {}
    for key in args.takes:
        performance_id, title = TAKES[key]
        recordings.update(build(args.samba, model, key, performance_id, title, takes_by_id, catalog))
        print(json.dumps({key: recordings[key]["metrics"]}), flush=True)
    menu_path.write_text(json.dumps({
        "recordings": {k: recordings[k] for k in TAKES if k in recordings},
        "followers": {fid: label for fid, (_, label, _) in FOLLOWERS.items()},
        "defaultFollower": "fixed"}, indent=1))


if __name__ == "__main__":
    main()
