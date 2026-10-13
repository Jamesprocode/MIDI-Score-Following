import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { buildTempoClock } from './tempo_clock.js';

const readJson = async file => JSON.parse(await readFile(new URL(file, import.meta.url), 'utf8'));
const metadata = { musicalStartSeconds: 1, bpm: 120, beatsPerBar: 4,
  scoreEndBeats: 8, musicalBarCount: 2, firstAnnotationBar: 1 };
const sample = (time, truthPositionBeats) => ({ time, truthPositionBeats });
const source = [sample(1, 0), sample(2.999999, 3.999998), sample(3, 4), sample(5, 8), sample(6, 10)];
const original = JSON.stringify(source);
const result = buildTempoClock(source, metadata);
assert.deepEqual(result.samples.map(s => s.tempoBar), [1, 1, 2, 3, 3]);
assert.equal(result.samples.at(-1).tempoPositionBeats, 10, 'Overflow must not clamp position');
assert.equal(result.metrics.bar_accuracy, 1);
assert.equal(JSON.stringify(source), original, 'Inputs must not be mutated');

// Truth and follower outputs cannot affect clock generation.
const misleading = source.map(s => ({ ...s, truthPositionBeats: 10000,
  truthBar: 999, estimatedPositionBeats: -1234, estimatedBpm: 999 }));
assert.deepEqual(buildTempoClock(misleading, metadata).samples.map(s => s.tempoPositionBeats),
  result.samples.map(s => s.tempoPositionBeats));
assert.deepEqual(buildTempoClock(source.slice(0, 3), metadata).samples, result.samples.slice(0, 3),
  'A prefix must be independent of future samples');
assert.equal(buildTempoClock([{ time: 1 }, { time: 2 }], metadata).metrics.frames, 0,
  'Clock generation must work without ground truth');
assert.throws(() => buildTempoClock([{ time: 2 }, { time: 1 }], metadata));
assert.throws(() => buildTempoClock([{ time: .9 }], metadata));

// Expected values independently computed using position_filter.clock_positions
// and run_benchmark metric definitions on the same serialized playback samples.
const expected = {
  blue_bossa: { frames: 1793, bar_accuracy: 1, within2beat_accuracy: 1,
    mean_abs_beats: 4.243445156516607e-13, median_abs_beats: 3.979039320256561e-13 },
  green_dolphin: { frames: 1226, bar_accuracy: 0.0432300163132137,
    within2beat_accuracy: 0.04812398042414356, mean_abs_beats: 44.789964923094246,
    median_abs_beats: 44.12472448165906 },
  samba_de_orfeu: { frames: 2050, bar_accuracy: 0.03219512195121951,
    within2beat_accuracy: 0.021463414634146343, mean_abs_beats: 21.257643226489815,
    median_abs_beats: 14.442773888920954 },
};
const diagnostics = await readJson('diagnostic_controls.json');
const reports = [];
for (const file of ['data.json', 'green_dolphin.json', 'samba_de_orfeu.json']) {
  const data = await readJson(file);
  const bass = buildTempoClock(data.followers.bass.samples, data.metadata);
  const tempoFields = s => [s.time, s.tempoPositionBeats, s.tempoBpm, s.tempoBar];
  assert.ok(data.followers.both, `${file}: combined bass + horns run must exist`);
  for (const [input, follower] of Object.entries(data.followers)) {
    const clock = buildTempoClock(follower.samples, data.metadata);
    assert.deepEqual(bass.samples.map(tempoFields), clock.samples.map(tempoFields),
      `${file}/${input}: the fixed clock must be identical across evidence inputs`);
    assert.deepEqual(bass.metrics, clock.metrics);
  }
  for (const [key, value] of Object.entries(expected[data.metadata.recordingKey])) {
    assert.ok(Math.abs(bass.metrics[key] - value) < 1e-9, `${file}: Python comparison ${key}`);
  }
  const prior = diagnostics.performances.find(p => p.performance_id === data.metadata.performanceId);
  if (prior) {
    const control = prior.controls.find(c => c.method === 'fixed_catalog_clock').all_frames;
    for (const key of Object.keys(control)) {
      assert.ok(Math.abs(bass.metrics[key] - control[key]) < 1e-9, `${file}: saved diagnostic ${key}`);
    }
  }
  reports.push({ recording: data.metadata.title, ...bass.metrics,
    final_error_beats: bass.samples.at(-1).tempoPositionBeats - bass.samples.at(-1).truthPositionBeats });
}
console.log(JSON.stringify({ status: 'passed', reports }, null, 2));
