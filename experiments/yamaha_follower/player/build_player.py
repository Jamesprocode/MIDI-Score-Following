"""Build replay data for the follower player (a copy of ../../ground_truth_player).

For each selected JazzSAMBA take, runs four causal followers on the recorded
piano MIDI and stores their positions at the 200 ms evaluation instants:

  harmony    harmony only (benchmark g00 filter, 32 positions per bar)
  onset      harmony + the rule-based onset-phase cue (Stage A)
  downbeat   harmony + the timing network's downbeat phase (Stage A)
  paper      Maezawa (2023) as described: per-beat tempo random walk, tempo fed
             back as the network's BPM condition, 50 ms updates (Stage B)

and writes <take>.json in the player's format plus recordings.json (the menu).
Playback tracks: the piano (the follower's only input; sync takes with the
timing repair of yamaha_timing/real_data.py), and the bass and drums MIDI where
the dataset has them, for listening only.

  python build_player.py --samba "../../../Dataset/SAMBA dataset" --checkpoint best_model.pt
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import follower_data  # noqa: E402  (sets sys.path)
import adaptive_grid  # noqa: E402
import run_study  # noqa: E402
from evidence import DownbeatEvidence, NetworkPhase, timing_log_probs  # noqa: E402
from rhythm import prepare_rhythm  # noqa: E402
from run_follow import position_metrics  # noqa: E402
from yamaha_filter import YamahaConfig, follow  # noqa: E402

TAKES = {
    "green_dolphin": ("sync/67-on_green_dolphin_street/preferred", "On Green Dolphin Street"),
    "dont_fence_me_in": ("sync/58-dont_fence_me_in/preferred", "Don't Fence Me In"),
    "samba_de_orfeu": ("sync/70-samba_de_orfeu/preferred", "Samba de Orfeu"),
    "blue_bossa": ("async/07-blue_bossa/preferred", "Blue Bossa"),
}
FOLLOWERS = {
    "paper": ("Paper version (Stage B)", "Maezawa (2023) as described: harmony x network downbeat phase, tempo drawn "
              "as a random walk at each beat, tracked tempo low-pass filtered and fed back as the network's BPM "
              "condition, updates every 50 ms (shown every 200 ms)."),
    "downbeat": ("Harmony + network downbeat (Stage A)", "The benchmark's harmonic grid filter with the timing "
                 "network's downbeat-phase probability multiplied in; written tempo as a fixed BPM condition; "
                 "continuous tempo drift within +-30%; updates every 200 ms."),
    "onset": ("Harmony + rule-based onset cue", "The same filter with the earlier rule-based onset-phase cue "
              "(beat phase only, no bar information) instead of the network."),
    "harmony": ("Harmony only", "The same filter with harmonic evidence only (the benchmark's Maezawa-style "
                "control, on piano input)."),
}


def midi_with_velocity(path, scale=1.0, shift=0.0):
    import mido
    t, active, notes = 0.0, {}, []
    for m in mido.MidiFile(path):
        t += m.time
        if m.type not in ("note_on", "note_off"):
            continue
        key = (m.channel, m.note)
        if key in active:
            start, velocity = active.pop(key)
            notes.append([start * scale + shift, t * scale + shift, m.note, velocity])
        if m.type == "note_on" and m.velocity > 0:
            active[key] = (t, m.velocity)
    return sorted([n for n in notes if n[1] > n[0]], key=lambda n: n[0])


def annotation_rows(root, protocol, song, take, name):
    with open(root / f"{protocol}_audio/songs/{song}/{take}/annotations/{name}.csv", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(row):
    out = {}
    for k, v in row.items():
        try:
            out[k] = float(v) if "." in v or "e" in v.lower() else int(v)
        except (ValueError, AttributeError):
            out[k] = {"True": True, "False": False}.get(v, v)
    return out


def traces(data, take, model):
    sample = np.arange(0, len(data["times"]), 2)
    times = data["times"][sample]
    case = run_study.get_features(data, {"triad"})["triad"]
    scores, available, mass = [a[sample] for a in case["features"]["active_pooled"]]
    probabilities, _ = run_study.observation(scores, available, mass, {"scaling": "zscore", "reliability": "none"})
    bpm = data["initial_bpm"]
    tempi = bpm * np.asarray(adaptive_grid.FilterConfig().tempo_ratios)
    _, bar_lp = timing_log_probs(model, take, bpm)
    rhythm = {"harmony": None, "onset": prepare_rhythm(data["notes_by_instrument"], times, tempi, bpm),
              "downbeat": DownbeatEvidence(bar_lp, model.config.frame_rate, times, data["beats_per_bar"], len(tempi))}
    strength = {"harmony": 0., "onset": 3., "downbeat": 1.}
    out = {}
    for key in ("harmony", "onset", "downbeat"):
        out[key] = adaptive_grid.follow_positions(
            times, probabilities, available, data["chord_starts_beats"], data["chord_ends_beats"],
            case["targets"]["ids"], data["bar_starts_beats"], bpm, start_time=data["start_time"],
            config={"bins_per_beat": 8, "rhythm_strength": strength[key]}, rhythm=rhythm[key])
    fine = np.round(np.arange(times[0], data["times"][-1] + 1e-9, 0.05), 6)
    at = np.round((times - fine[0]) / 0.05).astype(int)
    case50 = run_study.get_features(dict(data, times=fine), {"triad"})["triad"]
    s50, a50, m50 = case50["features"]["active_pooled"]
    p50, _ = run_study.observation(s50, a50, m50, {"scaling": "zscore", "reliability": "none"})
    paper = follow(fine, p50, a50, data["chord_starts_beats"], data["chord_ends_beats"], case50["targets"]["ids"],
                   data["bar_starts_beats"], data["beats_per_bar"], bpm, start_time=data["start_time"],
                   network=NetworkPhase(model, take), config=YamahaConfig())
    out["paper"] = {key: paper[key][at] for key in ("position_beats", "tempo_bpm", "position_confidence")}
    out["paper"]["bar_index"] = paper["bar_index"][at]
    return sample, times, available, out


def build(root, model, key, performance_id, title, takes_by_id, catalog):
    data, take = takes_by_id[performance_id]
    protocol, song, name = performance_id.split("/")
    sample, times, available, runs = traces(data, take, model)
    first_bar = data["first_annotation_bar"]
    bpb = data["beats_per_bar"]
    bars = []
    for row in annotation_rows(root, protocol, song, name, "bars"):
        bar = numeric(row)
        bar["padding"] = bar["measure"] == 0
        bar["startBeat"] = (bar["bar"] - first_bar) * bpb
        bar["endBeat"] = bar["startBeat"] + bpb
        bar["referenceBpm"] = bpb * 60 / (bar["end_time"] - bar["start_time"]) if not bar["padding"] else None
        bars.append(bar)
    chords = [{**numeric(r), "label": r["chord"],
               "startBeat": (float(r["start_bar"]) - first_bar + float(r["start_fraction"])) * bpb,
               "endBeat": (float(r["end_bar"]) - first_bar + float(r["end_fraction"])) * bpb}
              for r in annotation_rows(root, protocol, song, name, "chords")]
    sections = [{**numeric(r), "label": r["section"],
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
    for fid, (label, description) in FOLLOWERS.items():
        run = runs[fid]
        samples = [{"time": round(float(t), 6), "truthPositionBeats": float(truth[i]),
                    "estimatedPositionBeats": float(run["position_beats"][i]),
                    "estimatedBpm": float(run["tempo_bpm"][i]), "referenceBpm": float(reference_bpm[i]),
                    "truthBar": int(data["truth_bar_index"][sample[i]]) + first_bar,
                    "estimatedBar": int(run["bar_index"][i]) + first_bar,
                    "harmonyAvailable": bool(available[i])} for i, t in enumerate(times)]
        metrics = position_metrics({"position_beats": np.asarray(run["position_beats"]),
                                    "bar_index": np.asarray(run["bar_index"])}, data, sample, times)
        followers[fid] = {"label": label, "description": description, "instruments": ["piano"], "samples": samples,
                          "metrics": metrics, "evidenceCoverage": float(np.mean(available))}
    meta = catalog[song]
    end = max(n[1] for t in tracks for n in t["notes"])
    payload = {
        "metadata": {
            "title": title, "key": meta.get("key_signature", ""), "recordingKey": key, "performanceId": performance_id,
            "protocol": "Synchronous" if protocol == "sync" else "Asynchronous", "take": name.capitalize(),
            "bpm": data["initial_bpm"], "meter": meta["time_signature"], "beatsPerBar": bpb, "updateSeconds": 0.2,
            "musicalStartSeconds": data["start_time"], "annotationEndSeconds": data["end_time"],
            "durationSeconds": end, "scoreEndBeats": data["score_end_beats"], "musicalBarCount": len(musical),
            "firstAnnotationBar": first_bar,
            "groundTruthDescription": "JazzSAMBA bar annotations; position within each bar is interpolated linearly.",
            "playbackDescription": "Piano: recorded MIDI, the followers' only input" +
                                   (" (sync timing repaired: 175 BPM session clock and a start shift fitted from "
                                    "the piano audio)" if protocol == "sync" else "") +
                                   ". Bass and drums: dataset MIDI for listening only. Browser sound is synthesized.",
            "sampleDescription": "Stage A followers update every 200 ms; the paper version updates every 50 ms and "
                                 "is shown at the same 200 ms checks."},
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
    args = parser.parse_args()
    from yamaha_timing.training import load_checkpoint
    model = load_checkpoint(args.checkpoint, "cpu")[0]
    catalog = {r["song_subdir_basename"]: r for r in csv.DictReader(open(args.samba / "JazzSAMBA-catalog/songs.csv"))}
    takes_by_id = {d["performance_id"]: (d, t) for d, t in follower_data.samba_follow_takes(args.samba)}
    recordings = {}
    for key, (performance_id, title) in TAKES.items():
        recordings.update(build(args.samba, model, key, performance_id, title, takes_by_id, catalog))
        print(json.dumps({key: recordings[key]["metrics"]}), flush=True)
    (HERE / "recordings.json").write_text(json.dumps({
        "recordings": recordings, "followers": {fid: label for fid, (label, _) in FOLLOWERS.items()},
        "defaultFollower": "paper"}, indent=1))


if __name__ == "__main__":
    main()
