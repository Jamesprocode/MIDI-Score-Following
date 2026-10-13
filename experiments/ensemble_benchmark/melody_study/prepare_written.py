"""Extract local JAZZMUS written notes and conservatively align untimed charts.

Run with /Users/james/miniconda3/envs/jazzomr/bin/python (pyarrow/music21).
No performance MIDI, audio, performance pitches, or annotation timestamps are
used. Title matching only creates candidates; usable references must pass the
independent score-chord agreement/coverage and structural checks below.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import warnings
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
import music21 as m21
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
sys.path.insert(0, str(BENCH))
import common

SOURCE = Path('/Users/james/Documents/Workspace/OMR/JAZZMUS/data/train-00000-of-00001.original.parquet')
SOURCE_URL = 'https://huggingface.co/datasets/PRAIG/JAZZMUS/blob/main/data/train-00000-of-00001.parquet'
BASE_KIND = {
    'major': 'maj', 'major-seventh': 'maj', 'major-sixth': 'maj', 'major-ninth': 'maj',
    'major-13th': 'maj', 'dominant': 'maj', 'dominant-seventh': 'maj', 'dominant-ninth': 'maj',
    'dominant-11th': 'maj', 'dominant-13th': 'maj', 'minor': 'min', 'minor-seventh': 'min',
    'minor-sixth': 'min', 'minor-ninth': 'min', 'minor-11th': 'min', 'minor-13th': 'min',
    'major-minor': 'min', 'diminished': 'dim', 'diminished-seventh': 'dim',
    'half-diminished': 'dim', 'half-diminished-seventh': 'dim',
    'augmented': 'aug', 'augmented-seventh': 'aug',
    'suspended-fourth': 'sus4', 'suspended-second': 'sus2', 'power': 'power',
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def normalize(title):
    return re.sub('[^a-z0-9]', '', title.lower())

def aliases(title):
    return {normalize(title), normalize(title.split('(')[0])}

def score_events(xml):
    tree = ET.fromstring(xml)
    parts = tree.findall('part')
    if len(parts) != 1:
        raise ValueError('Only single-part written references are supported')
    # Honor explicit XML transposition only. Instrument names can make music21
    # infer a transposition even for files without a <transpose> declaration.
    # Chord-based transposition later aligns editions without performance notes.
    explicit = []
    for value in tree.findall('.//transpose'):
        explicit.append(float(value.findtext('chromatic', '0')) + 12 * float(value.findtext('octave-change', '0')))
    if len(set(explicit)) > 1:
        raise ValueError('Changing explicit instrument transposition is unsupported')
    semitones = int(explicit[0]) if explicit else 0
    if explicit and semitones != explicit[0]:
        raise ValueError('Microtonal explicit transposition is unsupported')
    with warnings.catch_warnings(record=True) as caught:
        score = m21.converter.parseData(xml, format='musicxml')
        if semitones:
            score = score.transpose(semitones, inPlace=False)
        try:
            score = score.expandRepeats()
            repeat_status, repeat_error = 'expanded', None
        except Exception as exc:
            repeat_status, repeat_error = 'unexpanded', str(exc)
    part = score.parts[0]
    measures = list(part.getElementsByClass(m21.stream.Measure))
    notes, chords, measure_rows = [], [], []
    unsupported = Counter()
    for index, measure in enumerate(measures):
        begin = float(measure.offset)
        measure_name = str(measure.number) + str(getattr(measure, 'numberSuffix', None) or '')
        end = float(measures[index + 1].offset) if index + 1 < len(measures) else float(part.highestTime)
        meter = measure.getContextByClass(m21.meter.TimeSignature)
        length = float(measure.barDuration.quarterLength)
        measure_rows.append({'expanded_measure_index': index, 'score_measure': measure_name,
            'onset_beats': begin, 'end_beats': end, 'nominal_bar_beats': length,
            'complete_bar': abs(end - begin - length) < 1e-6,
            'meter': str(meter.ratioString) if meter else None})
        for element in measure.recurse().notes:
            if isinstance(element, m21.harmony.Harmony):
                continue
            duration = float(element.duration.quarterLength)
            if duration <= 0:
                continue
            pitches = element.pitches if isinstance(element, m21.chord.Chord) else [element.pitch]
            for pitch in pitches:
                if abs(float(pitch.ps) - round(float(pitch.ps))) > 1e-6:
                    raise ValueError('Microtonal note cannot be represented as integer MIDI')
                notes.append({'pitch': int(round(float(pitch.ps))),
                    'onset_beats': begin + float(element.getOffsetInHierarchy(measure)),
                    'duration_beats': duration, 'score_measure': measure_name,
                    'expanded_measure_index': index,
                    'tie': element.tie.type if element.tie else None})
        for chord in measure.recurse().getElementsByClass(m21.harmony.Harmony):
            kind = getattr(chord, 'chordKind', 'none')
            if isinstance(chord, m21.harmony.NoChord) or kind == 'none':
                root, quality = None, None
            elif kind in BASE_KIND:
                root, quality = int(chord.root().pitchClass), BASE_KIND[kind]
            else:
                root, quality = None, None
                unsupported[kind] += 1
            chords.append({'onset_beats': begin + float(chord.getOffsetInHierarchy(measure)),
                'root': root, 'base_quality': quality, 'source_kind': kind,
                'score_measure': measure_name, 'expanded_measure_index': index})
    # Tie continuations are one sustained note, not fresh attacks.
    merged, open_ties = [], {}
    for note in sorted(notes, key=lambda r: (r['onset_beats'], r['pitch'])):
        previous = open_ties.get(note['pitch'])
        if note['tie'] in ('continue', 'stop') and previous is not None and abs(previous['onset_beats'] + previous['duration_beats'] - note['onset_beats']) < 1e-6:
            previous['duration_beats'] += note['duration_beats']
        else:
            merged.append(dict(note))
            previous = merged[-1]
        if note['tie'] in ('start', 'continue'):
            open_ties[note['pitch']] = previous
        else:
            open_ties.pop(note['pitch'], None)
    for note in merged:
        note.pop('tie')
    return {'notes': merged, 'chords': sorted(chords, key=lambda r: r['onset_beats']), 'measures': measure_rows,
        'duration_beats': float(part.highestTime), 'repeat_status': repeat_status, 'repeat_error': repeat_error,
        'explicit_instrument_transposition_semitones': semitones,
        'pitch_basis': 'MusicXML pitches plus explicit transpose only; missing transpose treated as concert. Instrument-name inferred transpositions ignored.',
        'unsupported_chord_kinds': dict(unsupported), 'parser_warnings': [str(w.message) for w in caught]}

def first_head_chart(song):
    # Deliberately never access start_time/end_time or performance MIDI paths.
    annotation = json.loads(Path(song['annotation_path']).read_text())
    bars = [r for r in annotation['bars'] if int(r['measure']) != 0]
    first_bar = min(int(r['bar']) for r in bars)
    num, den = map(int, song['catalog']['time_signature'].split('/'))
    bpb = num * 4. / den
    heads = [r for r in annotation['sections'] if common.truth(r.get('is_head_in'))]
    start = min((float(r['start_bar']) - first_bar + float(r['start_fraction'])) * bpb for r in heads)
    end = max((float(r['end_bar']) - first_bar + float(r['end_fraction'])) * bpb for r in heads)
    chords = []
    for row in annotation['chords']:
        a = (float(row['start_bar']) - first_bar + float(row['start_fraction'])) * bpb
        z = (float(row['end_bar']) - first_bar + float(row['end_fraction'])) * bpb
        if z <= a or a >= end or z <= start:
            continue
        key = common.quality_id(row, 'triad')
        chords.append({'start': max(a, start) - start, 'end': min(z, end) - start,
            'root': key[0] if key else None, 'quality': key[1] if key else None})
    return {'start_q': start, 'end_q': end, 'length_beats': end - start, 'beats_per_bar': bpb,
        'chords': sorted(chords, key=lambda r: r['start'])}

def chord_segments(events, end):
    # Same-time declarations are last-wins, matching document order; retain gaps
    # before the first declaration as unknown rather than filling from the future.
    unique = {}
    for row in events:
        unique[row['onset_beats']] = row
    ordered = [unique[t] for t in sorted(unique)]
    return [{'start': r['onset_beats'], 'end': ordered[i + 1]['onset_beats'] if i + 1 < len(ordered) else end,
             'root': r['root'], 'quality': r['base_quality']} for i, r in enumerate(ordered)]

def agreement(source_segments, chart, start, shift):
    target_total = sum(r['end'] - r['start'] for r in chart['chords'] if r['root'] is not None)
    covered = exact = roots = 0.
    for target in chart['chords']:
        if target['root'] is None:
            continue
        for source in source_segments:
            overlap = min(target['end'], source['end'] - start) - max(target['start'], source['start'] - start)
            if overlap <= 0 or source['root'] is None:
                continue
            covered += overlap
            same_root = (source['root'] + shift) % 12 == target['root']
            roots += overlap * same_root
            exact += overlap * (same_root and source['quality'] == target['quality'])
    return {'pitched_target_beats': target_total, 'coverage': covered / target_total if target_total else 0.,
            'exact_base_chord_agreement': exact / target_total if target_total else 0.,
            'root_agreement': roots / target_total if target_total else 0.}

def align(events, chart, min_agreement):
    segments = chord_segments(events['chords'], events['duration_beats'])
    candidates = []
    for measure in events['measures']:
        start = measure['onset_beats']
        end = start + chart['length_beats']
        if not measure['complete_bar'] or end > events['duration_beats'] + 1e-6:
            continue
        included = [r for r in events['measures'] if start <= r['onset_beats'] < end - 1e-6]
        if any(not r['complete_bar'] or abs(r['nominal_bar_beats'] - chart['beats_per_bar']) > 1e-6 for r in included):
            continue
        for shift in range(12):
            candidates.append({'source_start_beats': start, 'source_end_beats': end, 'pitch_shift_mod12': shift,
                               **agreement(segments, chart, start, shift)})
    candidates.sort(key=lambda r: (-r['exact_base_chord_agreement'], -r['coverage'], r['source_start_beats'], r['pitch_shift_mod12']))
    diagnostics = {'candidate_count': len(candidates), 'minimum_exact_agreement': min_agreement,
        'minimum_coverage': .99, 'chart': chart, 'best_candidates': candidates[:12],
        'method': 'Contiguous complete-measure span, matching full first-head length/meter; choose shift and span from exact duration-weighted base-chord agreement only. No nonlinear warping or performance notes.'}
    if not candidates:
        return 'no_complete_matching_length_span', None, diagnostics
    best = candidates[0]
    if events['repeat_status'] != 'expanded':
        return 'unverified_repeat_expansion', None, diagnostics
    if best['coverage'] < .99:
        return 'insufficient_chord_coverage', None, diagnostics
    if best['exact_base_chord_agreement'] < min_agreement:
        return 'poor_score_chord_agreement', None, diagnostics
    def mapped_notes(candidate):
        begin, end = candidate['source_start_beats'], candidate['source_end_beats']
        shift = candidate['pitch_shift_mod12']
        shift = shift if shift <= 6 else shift - 12
        # Pickups before the selected complete head are retained in raw events,
        # but omitted here rather than assigned to a preceding chart occurrence.
        return [{'pitch': r['pitch'] + shift, 'onset_beats': r['onset_beats'] - begin + chart['start_q'],
                 'duration_beats': min(r['duration_beats'], end - r['onset_beats']),
                 'score_onset_beats': r['onset_beats'], 'score_duration_beats': r['duration_beats'],
                 'score_measure': r['score_measure'], 'expanded_measure_index': r['expanded_measure_index']}
                for r in events['notes'] if begin <= r['onset_beats'] < end and r['onset_beats'] + r['duration_beats'] > begin]
    notes = mapped_notes(best)
    def signature(values):
        return tuple((r['pitch'], round(r['onset_beats'], 6), round(r['duration_beats'], 6)) for r in values)
    equivalent = [r for r in candidates if abs(r['exact_base_chord_agreement'] - best['exact_base_chord_agreement']) < 1e-9 and abs(r['coverage'] - best['coverage']) < 1e-9]
    diagnostics['tied_best_candidates'] = len(equivalent)
    diagnostics['distinct_tied_note_mappings'] = len({signature(mapped_notes(r)) for r in equivalent})
    if diagnostics['distinct_tied_note_mappings'] > 1:
        return 'ambiguous_score_only_mapping', None, diagnostics
    if not notes or any(not 0 <= r['pitch'] <= 127 for r in notes):
        return 'invalid_or_empty_mapped_notes', None, diagnostics
    diagnostics['selected'] = best
    diagnostics['pickup_policy'] = 'Exclude notes starting before selected full head; retain them in original XML/raw events.'
    return 'usable', notes, diagnostics

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=HERE/'references'/'written')
    parser.add_argument('--minimum-agreement', type=float, default=.85)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(HERE):
        raise ValueError('Outputs must stay inside melody_study')
    for name in ('originals', 'events', 'aligned'):
        (output/name).mkdir(parents=True, exist_ok=True)
    dataset = pq.read_table(args.source, columns=['id', 'annotation']).to_pylist()
    manifest_path = BENCH/'data'/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    matches = {}
    for row in dataset:
        xml = ast.literal_eval(row['annotation'])['encodings']['musicxml']
        tree = ET.fromstring(xml)
        titles = [r.text for key in ('work-title', 'movement-title') for r in tree.iter(key) if r.text]
        for song in manifest['songs']:
            if any(aliases(song['catalog']['title']) & aliases(title) for title in titles):
                matches.setdefault(song['song'], []).append({'row_id': row['id'], 'titles': titles, 'xml': xml})
    results = []
    for song in manifest['songs']:
        if song['song'] not in matches:
            continue
        candidates = sorted(matches[song['song']], key=lambda r: int(r['row_id']))
        candidate = candidates[0]
        name = song['protocol']+'__'+song['song']
        xml_path = output/'originals'/f'{name}__row{candidate["row_id"]}.musicxml'
        xml_path.write_text(candidate['xml'])
        event_path = output/'events'/f'{name}.json'
        row = {'song': song['song'], 'protocol': song['protocol'], 'title': song['catalog']['title'],
            'source_title': candidate['titles'], 'source_row_id': candidate['row_id'],
            'matched_row_ids': [r['row_id'] for r in candidates], 'row_selection': 'lowest numeric matched row ID; never selected by performance results',
            'source_url': SOURCE_URL, 'source_local_path': str(args.source.resolve()),
            'xml_path': str(xml_path), 'xml_sha256': sha(xml_path), 'event_path': str(event_path),
            'annotation_path': song['annotation_path'], 'annotation_sha256': sha(Path(song['annotation_path']))}
        try:
            events = score_events(candidate['xml'])
            save(event_path, {**row, **events})
            status, notes, diagnostic = align(events, first_head_chart(song), args.minimum_agreement)
            row.update(status=status, diagnostics=diagnostic, event_sha256=sha(event_path),
                       repeat_status=events['repeat_status'], repeat_error=events['repeat_error'],
                       source_note_count=len(events['notes']), source_chord_count=len(events['chords']))
            aligned_path = output/'aligned'/f'{name}.json'
            if status == 'usable':
                save(aligned_path, {'song': song['song'], 'protocol': song['protocol'], 'status': 'usable',
                    'notes': notes, 'source': row,
                    'coordinate_system': 'Quarter-note beats q from first musical bar in the supplied untimed JazzSAMBA plan; notes occupy FIRST head-in only.',
                    'reference_type': 'Independent written MusicXML from local JAZZMUS; not a transcription of the evaluated performance.'})
                row.update(aligned_path=str(aligned_path), aligned_sha256=sha(aligned_path), aligned_note_count=len(notes))
            elif aligned_path.exists():
                raise ValueError(f'Stale usable reference exists; use a new output directory: {aligned_path}')
        except Exception as exc:
            row.update(status='preparation_error', error=str(exc))
        results.append(row)
        print(song['song'], row['status'], row.get('aligned_note_count'), flush=True)
    save(output/'manifest.json', {'source_path': str(args.source.resolve()), 'source_sha256': sha(args.source),
        'source_url': SOURCE_URL, 'source_dataset_rows': len(dataset), 'catalog_manifest_sha256': sha(manifest_path),
        'script_sha256': sha(Path(__file__)), 'music21_version': m21.__version__,
        'title_matched_songs': len(matches), 'title_matched_rows': sum(len(v) for v in matches.values()),
        'usable_songs': sum(r['status']=='usable' for r in results), 'status_counts': dict(Counter(r['status'] for r in results)),
        'minimum_exact_base_chord_agreement': args.minimum_agreement, 'minimum_coverage': .99,
        'query_data_used': False, 'annotation_timestamp_fields_used': False,
        'limitations': ['Written editions can differ in melody and substitutions.',
            'A title match is not an accepted reference.', 'Chord agreement does not prove note-level equivalence.',
            'No rhythmic swing transformation or ornament correction applied.',
            'Only explicit XML transposition is applied before score-only alignment.',
            'Ties merged; grace notes and pickup attacks before the mapped first full head omitted from aligned notes.',
            'Only contiguous full-head spans are accepted; incompatible or ambiguous forms are rejected.'],
        'songs': results})
    lines = ['# Written melody reference audit\n',
        'These are independent written JAZZMUS MusicXML encodings already present in the local OMR workspace. They are not first-head AMT transcriptions. '
        'Only the six (or otherwise explicitly counted) usable files under `aligned/` should enter the written-reference experiment; `events/` also includes rejected title matches.\n',
        '## Source and preparation\n',
        f'Local source: `{args.source.resolve()}`. Public dataset record: {SOURCE_URL}. '
        'The Parquet `annotation` string contains `encodings.musicxml`; original XML is preserved under `originals/`. '
        'All 14 matching rows are catalogued; the lowest row ID is chosen for the three Blue Skies variants without inspecting performance accuracy.\n',
        'Music21 parses backup/forward voices, note lengths and notated repeats. Ties are merged. Grace notes are omitted. '
        'Only explicit XML transpose declarations are applied; instrument-name inferred transpositions are ignored. '
        'Missing declarations are provisionally interpreted as concert notation, then the edition is transposed using source/target chord agreement only.\n',
        f'Acceptance was fixed at exact root-plus-base-quality agreement ≥{100*args.minimum_agreement:.0f}% and pitched-chord coverage ≥99%, '
        'with a complete contiguous first-head span of the same length/meter, verified repeat expansion, and no unequal-note tied-best mapping. '
        'These are source-comparison criteria, not performance accuracy. No evaluated performance MIDI, audio, pitches or annotation timestamp fields are read.\n',
        '| Song | Status | Base-chord agreement | Coverage | Added pitch shift | Notes |\n|---|---|---:|---:|---:|---:|']
    for row in results:
        best = row.get('diagnostics', {}).get('best_candidates', [])
        b = best[0] if best else {}
        agreement_text = f"{100*b['exact_base_chord_agreement']:.2f}%" if b else '—'
        coverage_text = f"{100*b['coverage']:.2f}%" if b else '—'
        shift = b.get('pitch_shift_mod12')
        shift = shift if shift is None or shift <= 6 else shift - 12
        lines.append(f"| {row['song']} | {row['status']} | {agreement_text} | {coverage_text} | {shift if shift is not None else '—'} | {row.get('aligned_note_count', '—')} |")
    lines += ['\n## Coordinates and interpretation\n',
        '`aligned/{protocol}__{song}.json` contains `notes` with concert-oriented MIDI pitch, `onset_beats` and `duration_beats`. '
        'Aligned onset is q in the supplied untimed JazzSAMBA plan, beginning at the first musical bar, and occupies FIRST head-in only. '
        '`score_onset_beats` and `score_duration_beats` preserve the expanded written score coordinates. '
        'Alignment applies one constant beat offset and pitch-class shift; it does not stretch rhythm or perform nonlinear warping. '
        'Pickups before the selected full head are retained only in raw events; durations crossing the head end are clipped. '
        'Written repeats generate distinct legitimate beat positions; no new notes are copied by measure-number matching here.\n',
        'Title overlap covers 12 songs, not 12 validated charts. Chart agreement does not establish note-for-note agreement with the JazzSAMBA internal chart. '
        'Octave placement and swing/ornament differences remain possible. Compare written and learned-head methods on the same usable subset; '
        'the 75-song learned-head result is a separate cohort. Scores and source metadata are preserved unchanged outside this output directory.\n']
    (output/'README.md').write_text('\n'.join(lines))
    print(json.dumps({'output': str(output), 'title_matches': len(results), 'statuses': dict(Counter(r['status'] for r in results))}, indent=2))

if __name__ == '__main__':
    main()
