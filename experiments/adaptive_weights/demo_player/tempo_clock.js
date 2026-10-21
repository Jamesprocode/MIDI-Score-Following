/** Independent fixed-tempo clock: known start + supplied initial BPM only.
 * No MIDI evidence, estimated tempo/position, or annotation timestamps enter
 * the clock. Ground-truth sample fields are used only to calculate metrics.
 */
export function buildTempoClock(samples, metadata) {
  const { musicalStartSeconds: start, bpm, beatsPerBar, scoreEndBeats,
    musicalBarCount, firstAnnotationBar = 1 } = metadata;
  if (!Array.isArray(samples) || !Number.isFinite(start) ||
      !Number.isFinite(bpm) || bpm <= 0 || !Number.isFinite(beatsPerBar) || beatsPerBar <= 0 ||
      !Number.isFinite(scoreEndBeats) || scoreEndBeats <= 0 ||
      !Number.isInteger(musicalBarCount) || musicalBarCount <= 0) {
    throw new TypeError('A sample array, known start, positive tempo, and valid bar plan are required.');
  }
  // Matches position_filter.bar_indices: boundaries belong to the next bar;
  // every position past score end has one explicit overflow index. No epsilon.
  function barAt(position) {
    if (position < 0) return firstAnnotationBar - 1;
    if (position >= scoreEndBeats) return firstAnnotationBar + musicalBarCount;
    return firstAnnotationBar + Math.floor(position / beatsPerBar);
  }
  let previousTime = -Infinity;
  const clockSamples = samples.map(source => {
    if (!Number.isFinite(source.time) || source.time < start || source.time <= previousTime) {
      throw new TypeError('Sample times must increase and must not precede the known musical start.');
    }
    previousTime = source.time;
    const position = (source.time - start) * bpm / 60;
    return { ...source, tempoPositionBeats: position, tempoBpm: bpm, tempoBar: barAt(position) };
  });

  let correctBar = 0, withinTwo = 0;
  const errors = [];
  for (const sample of clockSamples) {
    if (!Number.isFinite(sample.truthPositionBeats)) continue;
    const truthBar = Number.isFinite(sample.truthBar) ? sample.truthBar : barAt(sample.truthPositionBeats);
    const error = Math.abs(sample.tempoPositionBeats - sample.truthPositionBeats);
    errors.push(error);
    correctBar += Number(sample.tempoBar === truthBar);
    withinTwo += Number(error <= 2);
  }
  const sortedErrors = [...errors].sort((a, b) => a - b);
  const n = errors.length, middle = Math.floor(n / 2);
  const median = n ? (n % 2 ? sortedErrors[middle] : (sortedErrors[middle - 1] + sortedErrors[middle]) / 2) : null;
  return {
    samples: clockSamples,
    metrics: {
      bar_accuracy: n ? correctBar / n : null,
      within2beat_accuracy: n ? withinTwo / n : null,
      mean_abs_beats: n ? errors.reduce((sum, error) => sum + error, 0) / n : null,
      median_abs_beats: median,
      frames: n,
    },
    description: `Fixed clock at the supplied initial tempo of ${bpm} BPM, starting at ${start} seconds. No harmonic updates or tempo adaptation; positions continue beyond the score without clamping.`,
  };
}
