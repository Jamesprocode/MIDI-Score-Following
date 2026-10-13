import {buildTempoClock} from './tempo_clock.js';
const $ = id => document.getElementById(id);
const SVG = 'http://www.w3.org/2000/svg';
let data, notes = [], audio, master, compressor, analyser, audioSamples, noiseBuffer, trackGains = {}, waves = {}, voices = new Set();
let playing = false, offset = 0, anchor = 0, nextNote = 0, sampleIndex = -2, lastBar = -2;
let duration = 0, checkedTime = null, tickHistory = [];
let chartScale, graphElements, graphWindowKey, tempoClock;
const enabled = new Map();
const recordings = {green_dolphin: {file:'green_dolphin.json',audit:'green_dolphin_audit.json'}, blue_bossa:{file:'data.json',audit:'data_audit.json'}, samba_de_orfeu:{file:'samba_de_orfeu.json',audit:'samba_de_orfeu_audit.json'}};
const requestedEvidence = new URLSearchParams(location.search).get('evidence');
let selectedRecording = 'green_dolphin', selectedEvidence = ['bass','horn','both'].includes(requestedEvidence)?requestedEvidence:'bass';
const currentFollower = () => data.followers[selectedEvidence];
const currentSamples = () => tempoClock.samples;
function updateUrl(){history.replaceState(null,'',`?song=${selectedRecording}&evidence=${selectedEvidence}`);}
const fmt = t => `${Math.floor(t / 60)}:${(t % 60).toFixed(2).padStart(5, '0')}`;
const prettySection = s => (s || '—').replaceAll('saxophone_alto', 'alto sax').replaceAll('saxophone_tenor', 'tenor sax').replaceAll('solos-', '').replaceAll('head-in', 'Head in').replaceAll('head-out', 'Head out').replaceAll('/', ' / ').replace(/^./, x => x.toUpperCase());
const num = n => Number.isFinite(n) ? n.toFixed(2) : '—';
// Display-only time equivalent at the reference tempo of each annotated bar.
// Musical positions and the original beat-based benchmark metrics stay intact.
const timingErrorMs = (sample, field) => (sample[field] - sample.truthPositionBeats) * 60000 / sample.referenceBpm;
const meanTimingErrorMs = (samples, field) => samples.reduce((sum, sample) => sum + Math.abs(timingErrorMs(sample, field)), 0) / samples.length;
const fmtMs = value => Math.round(Math.abs(value)).toLocaleString('en-US');
function upperBound(array, value, key) { let lo = 0, hi = array.length; while (lo < hi) { const mid = (lo + hi) >>> 1; if (key(array[mid]) <= value) lo = mid + 1; else hi = mid; } return lo; }
function intervalAt(array, t, start = 'startBeat', end = 'endBeat') { const i = upperBound(array, t, x => x[start]) - 1; const item = array[i]; return item && t >= item[start] && t < item[end] ? item : null; }
function now() { return Math.max(0, Math.min(duration, playing ? offset + Math.max(0,audio.currentTime-anchor) : offset)); }
function svgEl(tag, attrs, text) { const el = document.createElementNS(SVG, tag); for (const [key, value] of Object.entries(attrs || {})) el.setAttribute(key, value); if (text !== undefined) el.textContent = text; return el; }

function initAudio() {
  if (audio) return;
  audio = new AudioContext({latencyHint: 'interactive'});
  master = audio.createGain(); master.gain.value = Number($('volume').value);
  compressor = audio.createDynamicsCompressor(); compressor.threshold.value = -14; compressor.ratio.value = 5;
  analyser=audio.createAnalyser();analyser.fftSize=1024;audioSamples=new Float32Array(1024);
  master.connect(compressor); compressor.connect(analyser);analyser.connect(audio.destination);
  noiseBuffer = audio.createBuffer(1, audio.sampleRate, audio.sampleRate);
  const noise = noiseBuffer.getChannelData(0); for(let i=0;i<noise.length;i++)noise[i]=Math.random()*2-1;
  for (const track of data.tracks) {
    const gain = audio.createGain(); gain.gain.value = enabled.get(track.id) ? 1 : 0; gain.connect(master); trackGains[track.id] = gain;
    const harmonics = track.id === 'bass' ? [0, 1, .35, .12, .045] : track.id === 'trumpet' ? [0, .8, .7, .4, .18, .08] : [0, 1, .5, .25, .1, .06];
    waves[track.id] = audio.createPeriodicWave(new Float32Array(harmonics.length), new Float32Array(harmonics));
  }
}
function stopVoices() { for (const voice of voices) { try { voice.osc.stop(); } catch {} voice.osc.disconnect(); voice.gain.disconnect(); voice.filter?.disconnect(); } voices.clear(); }
function soundDrum(note, when) {
  const kick=[35,36].includes(note.pitch), cymbal=[42,44,46,49,51,52,53,55,57,59].includes(note.pitch);
  const decay=kick?.18:cymbal?(note.pitch===42?.055:.24):.13;
  const osc=kick?audio.createOscillator():audio.createBufferSource(),gain=audio.createGain();
  let filter;
  if(kick){osc.frequency.setValueAtTime(130,when);osc.frequency.exponentialRampToValueAtTime(42,when+.12);osc.connect(gain);}
  else{osc.buffer=noiseBuffer;filter=audio.createBiquadFilter();filter.type=cymbal?'highpass':'bandpass';filter.frequency.value=cymbal?6500:1800;filter.Q.value=.7;osc.connect(filter);filter.connect(gain);}
  const level=(kick?.23:cymbal?.055:.09)*note.velocity/127;
  gain.gain.setValueAtTime(.0001,when);gain.gain.linearRampToValueAtTime(level,when+.003);gain.gain.exponentialRampToValueAtTime(.0001,when+decay);
  gain.connect(trackGains[note.track]);const voice={osc,gain,filter};voices.add(voice);
  osc.onended=()=>{voices.delete(voice);osc.disconnect();gain.disconnect();filter?.disconnect();};osc.start(when);osc.stop(when+decay+.01);
}
function soundNote(note, when, remaining) {
  if (remaining <= .001) return;
  if(note.track==='drums'){soundDrum(note,when);return;}
  const osc = audio.createOscillator(), gain = audio.createGain();
  osc.setPeriodicWave(waves[note.track]); osc.frequency.value = 440 * 2 ** ((note.pitch - 69) / 12);
  const level = (note.track === 'bass' ? .13 : .055) * note.velocity / 127;
  const attack = Math.min(.015, remaining / 3), end = when + remaining;
  gain.gain.setValueAtTime(0, when); gain.gain.linearRampToValueAtTime(level, when + attack);
  gain.gain.setValueAtTime(level * .82, Math.max(when + attack, end - .01)); gain.gain.linearRampToValueAtTime(0, end + .025);
  osc.connect(gain); gain.connect(trackGains[note.track]);
  const voice = {osc, gain}; voices.add(voice);
  osc.onended = () => { voices.delete(voice); osc.disconnect(); gain.disconnect(); };
  osc.start(when); osc.stop(end + .03);
}
function schedule() {
  if (!playing || audio.state !== 'running') return;
  const horizon = now() + .13;
  while (nextNote < notes.length && notes[nextNote].start <= horizon) {
    const n = notes[nextNote++], intended = anchor + n.start - offset;
    const when = Math.max(audio.currentTime + .003, intended);
    soundNote(n, when, n.end - n.start - Math.max(0, when - intended));
  }
}
async function startPlayback() {
  initAudio(); await audio.resume();
  if (offset >= duration - .01) offset = 0;
  anchor = audio.currentTime + .04; playing = true;
  nextNote = upperBound(notes, offset - 1e-8, x => x.start);
  for (const n of notes) if (n.start < offset - 1e-8 && n.end > offset) soundNote(n, anchor, n.end - offset);
  schedule(); $('play').textContent = 'Pause'; $('status').textContent = 'Playing the original MIDI events. Positions update every 200 ms.';
}
function pausePlayback() {
  offset = now(); playing = false; stopVoices(); $('play').textContent = 'Play MIDI';
  $('status').textContent = 'Paused. Seek or step 200 ms to inspect all three positions.';
}
async function seekTo(t) {
  const resume = playing; if (playing) pausePlayback(); else stopVoices();
  offset = Math.max(0, Math.min(duration, t)); sampleIndex = -2;
  refreshPosition(); refreshTransport(); if (resume) await startPlayback();
}
function refreshTransport() {
  if (!data) return;
  const t = now(); $('seek').value = t; $('time').textContent = `${fmt(t)} / ${fmt(duration)}`;
  let rms=0;if(analyser){analyser.getFloatTimeDomainData(audioSamples);rms=Math.sqrt(audioSamples.reduce((sum,x)=>sum+x*x,0)/audioSamples.length);}
  document.body.dataset.playback=JSON.stringify({recording:selectedRecording,evidence:selectedEvidence,playing,time:t,checkedTime,audioState:audio?.state??'not-started',activeVoices:voices.size,rms,tickHistory:tickHistory.slice(-20)});
  if (playing && t >= duration) { pausePlayback(); offset = duration; refreshPosition(); $('status').textContent = 'Playback finished.'; }
}

function chartWindow(t) { return $('whole').checked ? [0,duration] : [Math.max(0,Math.min(duration-24,Math.floor(t/5)*5-5)),Math.min(duration,Math.max(24,Math.min(duration,Math.floor(t/5)*5+19)))]; }
function buildChart() {
  const svg = $('chart'); svg.replaceChildren();
  const width=Math.max(300,svg.clientWidth),left = 44, right = width-16, top = 25, bottom = 263;
  svg.setAttribute('viewBox',`0 0 ${width} 300`);
  const [from,to]=chartWindow(now()); graphWindowKey=`${from}:${to}`;
  const shown=currentSamples().filter(s=>s.time>=from&&s.time<=to);
  const positionBar=q=>q/data.metadata.beatsPerBar+data.metadata.firstAnnotationBar;
  const bars=shown.flatMap(s=>[positionBar(s.truthPositionBeats),positionBar(s.estimatedPositionBeats),positionBar(s.tempoPositionBeats)]);
  const step=$('whole').checked?40:Math.max(2,Math.ceil((Math.max(...bars)-Math.min(...bars))/6/2)*2);
  const minBar=$('whole').checked?0:Math.floor((Math.min(...bars)-.5)/step)*step;
  const maxBar=Math.ceil((Math.max(...bars)+.5)/step)*step;
  chartScale = {x: t => left + (t-from)/(to-from)*(right-left), y: q => bottom - (positionBar(q)-minBar)/(maxBar-minBar)*(bottom-top), left, right, top, bottom,from,to};
  const {x, y} = chartScale;
  for (let bar=minBar;bar<=maxBar;bar+=step) { const yp = bottom-(bar-minBar)/(maxBar-minBar)*(bottom-top); svg.append(svgEl('line', {x1:left,y1:yp,x2:right,y2:yp,class:'grid'}), svgEl('text', {x:left - 12,y:yp + 4,'text-anchor':'end'}, bar)); }
  const timeStep=$('whole').checked?60:(width<500?10:5);
  for(let t=Math.ceil(from/timeStep)*timeStep;t<=to;t+=timeStep) { svg.append(svgEl('line', {x1:x(t),y1:top,x2:x(t),y2:bottom,class:'grid'}), svgEl('text', {x:x(t),y:bottom + 23,'text-anchor':'middle'}, `${Math.floor(t/60)}:${String(t%60).padStart(2,'0')}`)); }
  svg.append(svgEl('text', {x:left,y:12,class:'axis-title'}, 'Performance position (bars)'));
  const defs = svgEl('defs'), clip = svgEl('clipPath', {id:'played-clip'}), rect = svgEl('rect',{x:left - 5,y:0,width:0,height:300}); clip.append(rect); defs.append(clip); svg.append(defs);
  const path = field => shown.map((s,i) => `${i ? 'L' : 'M'}${x(s.time).toFixed(2)},${y(s[field]).toFixed(2)}`).join(' ');
  const truthPath = path('truthPositionBeats'), estimatePath = path('estimatedPositionBeats'), tempoPath = path('tempoPositionBeats');
  svg.append(svgEl('path',{d:truthPath,class:'truth-line future'}),svgEl('path',{d:estimatePath,class:'estimate-line future'}),svgEl('path',{d:tempoPath,class:'tempo-line future'}));
  svg.append(svgEl('path',{d:truthPath,class:'truth-line','clip-path':'url(#played-clip)'}),svgEl('path',{d:estimatePath,class:'estimate-line','clip-path':'url(#played-clip)'}),svgEl('path',{d:tempoPath,class:'tempo-line','clip-path':'url(#played-clip)'}));
  const cursor = svgEl('line',{x1:left,x2:left,y1:top,y2:bottom,class:'cursor'});
  const truthDot = svgEl('circle',{r:5,class:'truth-dot',visibility:'hidden'}), estimateDot = svgEl('circle',{r:5,class:'estimate-dot',visibility:'hidden'}), tempoDot = svgEl('circle',{r:8,class:'tempo-dot',visibility:'hidden'});
  svg.append(cursor, tempoDot, truthDot, estimateDot); graphElements = {rect,cursor,truthDot,estimateDot,tempoDot};
  svg.onclick=e=>{const point=svg.createSVGPoint();point.x=e.clientX;point.y=e.clientY;const p=point.matrixTransform(svg.getScreenCTM().inverse());seekTo(from+(p.x-left)/(right-left)*(to-from));};
}
function updateChart(sample, t) {
  const {x,y,left} = chartScale, {rect,cursor,truthDot,estimateDot,tempoDot} = graphElements;
  rect.setAttribute('width', Math.max(0, x(t) - left + 5)); cursor.setAttribute('x1',x(t)); cursor.setAttribute('x2',x(t));
  for (const [dot,field] of [[truthDot,'truthPositionBeats'],[estimateDot,'estimatedPositionBeats'],[tempoDot,'tempoPositionBeats']]) {
    dot.setAttribute('visibility',sample ? 'visible' : 'hidden');
    if (sample) { dot.setAttribute('cx',x(sample.time)); dot.setAttribute('cy',y(sample[field])); }
  }
}
function showPosition(prefix, q, bpm) {
  const bar = intervalAt(data.bars.filter(b=>!b.padding),q), chord = intervalAt(data.chords,q), section = intervalAt(data.sections,q);
  if (!bar) { $(prefix+'-bar').textContent = 'Outside plan'; $(prefix+'-chord').textContent = '—'; $(prefix+'-detail').textContent = `Position ${num(q)} beats`; $(prefix+'-section').textContent = 'No annotated location'; return; }
  const beat = q - bar.startBeat + 1;
  $(prefix+'-bar').textContent = `Bar ${bar.bar}`;
  $(prefix+'-chord').textContent = chord?.label || 'N.C.';
  $(prefix+'-detail').textContent = `Chart ${bar.measure} · beat ${beat.toFixed(2)} · ${Number(bpm).toFixed(1)} BPM${prefix==='truth'?' (bar average)':prefix==='tempo'?' (fixed)':''}`;
  $(prefix+'-section').textContent = prettySection(section?.label);
}
function renderBars(truthQ, estimateQ, tempoQ) {
  const musical = data.bars.filter(b => !b.padding), bar = intervalAt(musical, truthQ ?? 0), estBar = intervalAt(musical, estimateQ), tempoBar=intervalAt(musical,tempoQ);
  const index = truthQ===null&&now()>=data.metadata.annotationEndSeconds?musical.length-1:bar?musical.indexOf(bar):0, first = Math.max(0, Math.min(musical.length - 8,index - 2));
  if (lastBar !== first) {
    $('bar-strip').replaceChildren();
    for (const b of musical.slice(first,first+8)) {
      const cell = document.createElement('div'); cell.className = 'bar-cell'; cell.dataset.bar = b.bar; cell.tabIndex = 0; cell.setAttribute('role','button'); cell.setAttribute('aria-label',`Seek to performance bar ${b.bar}, chart measure ${b.measure}`);
      const n = document.createElement('div'); n.className='bar-number'; n.textContent=`Bar ${b.bar}`;
      const c = document.createElement('div'); c.className='bar-chord'; c.textContent=[...new Set(data.chords.filter(ch=>ch.startBeat < b.endBeat && ch.endBeat > b.startBeat).map(ch=>ch.label))].join(' / ') || 'N.C.';
      const m=document.createElement('div'); m.className='chart-number'; m.textContent=`Chart ${b.measure}`;
      cell.append(n,c,m); cell.onclick=()=>seekTo(b.start_time); cell.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();seekTo(b.start_time);}}; $('bar-strip').append(cell);
    }
    lastBar=first;
  }
  for(const cell of $('bar-strip').children) { cell.classList.toggle('is-truth',Number(cell.dataset.bar)===bar?.bar && truthQ !== null); cell.classList.toggle('is-estimate',Number(cell.dataset.bar)===estBar?.bar && estimateQ !== null);cell.classList.toggle('is-tempo',Number(cell.dataset.bar)===tempoBar?.bar && tempoQ !== null); }
}
function refreshPosition() {
  if(!data) return;
  const samples=currentSamples();
  const t=now(), inRange=t >= data.metadata.musicalStartSeconds && t < data.metadata.annotationEndSeconds;
  const idx=inRange ? upperBound(samples,t,s=>s.time)-1 : -1;
  if(idx === sampleIndex) return; sampleIndex=idx;
  const sample=idx>=0 ? samples[idx] : null;
  if(chartWindow(t).join(':')!==graphWindowKey)buildChart();
  if(!sample){
    const before=t<data.metadata.musicalStartSeconds;
    const waiting=!before&&t<samples[0].time;
    for(const prefix of ['truth','estimate','tempo']) { $(prefix+'-bar').textContent=before?`Before bar ${data.metadata.firstAnnotationBar}`:waiting?'First check pending':'Annotations ended'; $(prefix+'-chord').textContent='—'; $(prefix+'-detail').textContent=before?`First musical bar at ${fmt(data.metadata.musicalStartSeconds)}`:waiting?`First update at ${fmt(samples[0].time)}`:`Reference ends at ${fmt(data.metadata.annotationEndSeconds)}`; $(prefix+'-section').textContent=before?'Leading padding':waiting?'Musical start':'MIDI tail'; }
    $('tempo-error').textContent='';
    $('difference').textContent=before?'Waiting for the first musical bar':waiting?'Waiting for the first 200 ms check':`No reference beyond ${fmt(data.metadata.annotationEndSeconds)}`; $('difference').classList.remove('off'); $('check-time').textContent='Last position check: —'; checkedTime=null;
    updateChart(null,t); renderBars(null,null,null); return;
  }
  const referenceBar=intervalAt(data.bars,sample.time,'start_time','end_time');
  const referenceBpm=sample.referenceBpm??(60*data.metadata.beatsPerBar/(referenceBar.end_time-referenceBar.start_time));
  checkedTime=sample.time; showPosition('truth',sample.truthPositionBeats,referenceBpm); showPosition('estimate',sample.estimatedPositionBeats,sample.estimatedBpm);showPosition('tempo',sample.tempoPositionBeats,sample.tempoBpm);
  const diff=timingErrorMs(sample,'estimatedPositionBeats'), sameBar=Math.floor(sample.estimatedPositionBeats/data.metadata.beatsPerBar)===Math.floor(sample.truthPositionBeats/data.metadata.beatsPerBar);
  $('difference').textContent=Math.abs(diff)<.5?'Harmony: ≈0 ms offset':`Harmony: ≈${fmtMs(diff)} ms ${diff>=0?'ahead':'behind'}`; $('difference').classList.toggle('off',!sameBar);
  const tempoError=timingErrorMs(sample,'tempoPositionBeats');
  $('tempo-error').textContent=Math.abs(tempoError)<.5?'≈0 ms from the reference':`≈${fmtMs(tempoError)} ms ${tempoError>=0?'ahead of':'behind'} the reference`;
  const evidenceName={bass:'bass',horn:'horn',both:'bass + horn'}[selectedEvidence];
  $('check-time').textContent=`Checked at ${fmt(sample.time)} · ${sample.harmonyAvailable?`${evidenceName} evidence present`:`no active ${evidenceName} notes; clock continues`}`;
  updateChart(sample,sample.time); renderBars(sample.truthPositionBeats,sample.estimatedPositionBeats,sample.tempoPositionBeats);
  tickHistory.push({wall:performance.now(),time:sample.time,truth:sample.truthPositionBeats,estimate:sample.estimatedPositionBeats,tempo:sample.tempoPositionBeats}); if(tickHistory.length>80)tickHistory.shift();
}
function setEvidence(value){
  selectedEvidence=value;const follower=currentFollower();$('evidence').value=value;updateUrl();
  tempoClock=buildTempoClock(follower.samples,data.metadata);
  $('evidence-hint').textContent=follower.label;
  $('evidence-instruments').textContent=follower.instruments.map(id=>data.tracks.find(t=>t.id===id)?.label||id).join(' + ');
  const m=follower.metrics;
  $('run-metrics').textContent=`${follower.label} · ${(100*m.bar_accuracy).toFixed(1)}% correct bar · ≈${fmtMs(meanTimingErrorMs(currentSamples(),'estimatedPositionBeats'))} ms mean absolute error · evidence present at ${(100*follower.evidenceCoverage).toFixed(1)}% of checks`;
  $('follower-method').textContent=`The orange line replays a separate causal position-and-tempo filter using only the selected instruments as harmonic evidence. ${follower.description} Its trace is precomputed; playback reveals each estimate at its original update time. It receives no later annotation timestamps, melody matches, or beat-detector outputs.`;
  const cm=tempoClock.metrics;
  $('clock-metrics').textContent=`Green fixed-tempo clock · ${(100*cm.bar_accuracy).toFixed(1)}% correct bar · ≈${fmtMs(meanTimingErrorMs(currentSamples(),'tempoPositionBeats'))} ms mean absolute error`;
  $('tempo-hint').textContent=`Fixed ${data.metadata.bpm} BPM`;
  $('clock-method').textContent=`The green dotted line starts at the same known musical start and advances continuously at the supplied ${data.metadata.bpm} BPM. It uses elapsed time only, with no MIDI observations, harmonic corrections, or later tempo updates. Switching tracking input leaves this line unchanged. Its chord and section labels are looked up at the clock's position; they do not update it.`;
  sampleIndex=-2;tickHistory=[];buildChart();refreshPosition();refreshTransport();
}
function bind() {
  $('play').onclick=async()=>{try{if(playing)pausePlayback();else await startPlayback();}catch(e){$('status').textContent=`Playback could not start: ${e.message}`;$('status').classList.add('error');}};
  $('restart').onclick=()=>seekTo(0); $('seek').oninput=e=>seekTo(Number(e.target.value));
  $('back').onclick=()=>{if(playing)pausePlayback();seekTo(now()-.2);}; $('forward').onclick=()=>{if(playing)pausePlayback();seekTo(now()+.2);};
  $('volume').oninput=e=>{if(master)master.gain.setTargetAtTime(Number(e.target.value),audio.currentTime,.02);};
  $('sections').onchange=e=>seekTo(Number(e.target.value));
  $('song').onchange=e=>loadRecording(e.target.value).catch(showLoadError);
  $('evidence').onchange=e=>setEvidence(e.target.value);
  $('whole').onchange=()=>{buildChart();sampleIndex=-2;refreshPosition();};
  let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{if(data){buildChart();sampleIndex=-2;refreshPosition();}},120);});
  document.addEventListener('keydown',e=>{if(e.code==='Space'&&e.target===document.body){e.preventDefault();$('play').click();}});
  setInterval(schedule,25);
  // Poll the audio clock frequently; refreshPosition only changes the display
  // when the next 200 ms source sample is reached. This avoids a second 200 ms lag.
  setInterval(()=>{refreshPosition();refreshTransport();},50);
  window.addEventListener('pagehide',()=>{if(playing)pausePlayback();});
}
async function loadRecording(id) {
  if(playing)pausePlayback();else stopVoices();
  data=undefined;tempoClock=undefined;playing=false;offset=0;sampleIndex=-2;lastBar=-2;checkedTime=null;tickHistory=[];notes=[];enabled.clear();
  for(const control of ['play','restart','back','forward','seek','song','evidence','sections','whole'])$(control).disabled=true;
  $('status').classList.remove('error');$('status').textContent='Loading the MIDI notes and annotations…';
  if(audio)await audio.close();audio=undefined;master=undefined;analyser=undefined;trackGains={};waves={};
  selectedRecording=id;const recording=recordings[id];
  const response=await fetch(recording.file); if(!response.ok)throw new Error(`Data request returned ${response.status}`); data=await response.json(); duration=data.metadata.durationSeconds;
  document.title=`${data.metadata.title} — following the performance`;
  $('piece-title').textContent=data.metadata.title;$('piece-tempo').textContent=`${data.metadata.bpm} BPM supplied`;$('piece-meter').textContent=data.metadata.meter;$('piece-key').textContent=data.metadata.key||'';
  $('song').value=id;updateUrl();
  const tempi=data.bars.filter(b=>!b.padding).map(b=>60*data.metadata.beatsPerBar/(b.end_time-b.start_time)).sort((a,b)=>a-b);
  const variable=tempi.at(-1)-tempi[0]>.01;
  $('tempo-note').textContent=variable?`Changing tempo · bar averages ${tempi[Math.floor(tempi.length*.1)].toFixed(0)}–${tempi[Math.floor(tempi.length*.9)].toFixed(0)} BPM (middle 80%)`:`Fixed annotation grid · ${tempi[0].toFixed(0)} BPM`;
  $('truth-hint').textContent=variable?'Variable bar timing':'Steady tempo';
  $('playback-note').textContent=`${data.metadata.playbackDescription} Track switches mute playback only.`;
  $('reference-method').textContent=`The blue line follows the downloaded annotations. ${data.metadata.groundTruthDescription} The displayed reference BPM is an average over the current annotated bar.`;
  $('sampling-method').textContent=`All three position displays use the latest available 200 ms check. ${data.metadata.sampleDescription}`;
  $('piece-footer').textContent=`${data.metadata.title} / JazzSAMBA ${data.metadata.protocol.toLowerCase()} preferred take`;
  $('seek').max=duration;
  $('tracks').replaceChildren();$('sections').replaceChildren(new Option('Beginning',0));$('sources').replaceChildren();
  for(const track of data.tracks){
    enabled.set(track.id,true); for(const [start,end,pitch,velocity] of track.notes)notes.push({start,end,pitch,velocity,track:track.id});
    const label=document.createElement('label'),box=document.createElement('input'); box.type='checkbox';box.checked=true;box.setAttribute('aria-label',`Play ${track.label}`);
    box.onchange=()=>{enabled.set(track.id,box.checked);if(trackGains[track.id])trackGains[track.id].gain.setTargetAtTime(box.checked?1:0,audio.currentTime,.015);};label.append(box,document.createTextNode(track.label));$('tracks').append(label);
  }
  notes.sort((a,b)=>a.start-b.start);
  let previous='';for(const section of data.sections){const group=section.label.split('/')[0];if(group!==previous){const option=document.createElement('option');option.value=section.start_time;option.textContent=prettySection(group);$('sections').append(option);previous=group;}}
  const sourceTitle=document.createElement('p');sourceTitle.innerHTML='<strong>Source files</strong>';$('sources').append(sourceTitle);
  const walkSources=value=>{if(!value||typeof value!=='object')return;if(value.path){const p=document.createElement('p');p.textContent=value.path;$('sources').append(p);}else for(const item of Object.values(value))walkSources(item);};walkSources(data.sources);
  const sourceLink=document.createElement('a');sourceLink.href=recording.audit;sourceLink.textContent='Open source hashes and data checks';$('sources').append(sourceLink);
  setEvidence(selectedEvidence);for(const control of ['play','restart','back','forward','seek','song','evidence','sections','whole'])$(control).disabled=false;
  $('status').textContent='Ready. Press Play MIDI, or click a bar to start there.';refreshPosition();refreshTransport();
  // Read-only diagnostics for checking that playback and the 200 ms display use the same timeline.
  window.playerDiagnostics=()=>({playing,evidence:selectedEvidence,time:now(),checkedTime,sampleIndex,audioState:audio?.state??'not-started',activeVoices:voices.size,noteCount:notes.length,duration,tickHistory:tickHistory.slice(),sample:sampleIndex>=0?currentSamples()[sampleIndex]:null});
}
function showLoadError(e){data=undefined;$('song').disabled=false;$('status').textContent=`Could not load this replay: ${e.message}. Open the page through the local server.`;$('status').classList.add('error');console.error(e);}
bind();
const initial=new URLSearchParams(location.search).get('song');
loadRecording(recordings[initial]?initial:'green_dolphin').catch(showLoadError);
