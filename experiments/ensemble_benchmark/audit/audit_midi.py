"""Read-only source MIDI/FLAC audit; output stays in this audit directory."""
from pathlib import Path
from collections import Counter,defaultdict,deque
import struct,json,statistics,hashlib
root=Path(__file__).resolve().parents[3]
def vlq(b,p):
    n=0
    while True:
        x=b[p];p+=1;n=(n<<7)|(x&127)
        if x<128:return n,p

def parse(path):
    b=path.read_bytes(); assert b[:4]==b'MThd'
    hl=struct.unpack('>I',b[4:8])[0];fmt,ntracks,division=struct.unpack('>HHH',b[8:14]);assert fmt in (0,1) and 0<division<32768
    p=8+hl; tracks=[];allnotes=[];alltempos=[];allmeta=[];messages=Counter();programs=[];channels=Counter();dangling=0;unmatched=0
    for k in range(ntracks):
        assert b[p:p+4]==b'MTrk',(path,k,p);ln=struct.unpack('>I',b[p+4:p+8])[0];d=b[p+8:p+8+ln];p+=8+ln
        q=0;tick=0;status=None;active=defaultdict(deque);ns=[];met=[];cnt=Counter()
        while q<len(d):
            delta,q=vlq(d,q);tick+=delta
            if d[q]>=128:status=d[q];q+=1
            assert status is not None
            if status==255:
                typ=d[q];q+=1;sz,q=vlq(d,q);v=d[q:q+sz];q+=sz
                val=int.from_bytes(v,'big') if typ==81 else list(v) if typ in (88,89) else v.decode('utf-8',errors='replace') if typ in range(1,10) else v.hex()
                met.append({'tick':tick,'type':typ,'value':val});allmeta.append((typ,val));cnt['meta_'+str(typ)]+=1
                if typ==81:alltempos.append((tick,val))
                status=None
            elif status in (240,247):
                sz,q=vlq(d,q);q+=sz;cnt['sysex']+=1;status=None
            else:
                typ=status>>4;ch=status&15;size=1 if typ in(12,13) else 2
                v=d[q:q+size];assert len(v)==size and all(x<128 for x in v),(path,q,status,list(v));q+=size;cnt[str(typ)]+=1
                if typ==12:programs.append((k,tick,ch,v[0]))
                if typ==9 and v[1]>0:
                    active[(ch,v[0])].append((tick,v[1]));channels[ch]+=1
                elif typ==8 or (typ==9 and v[1]==0):
                    if active[(ch,v[0])]:
                        onset,velocity=active[(ch,v[0])].popleft();ns.append((onset,tick,v[0],velocity,ch,k))
                    else:unmatched+=1
        dangling+=sum(len(v) for v in active.values());allnotes.extend(ns);messages.update(cnt)
        tracks.append({'index':k,'end_tick':tick,'note_count':len(ns),'meta':met,'messages':dict(cnt)})
    assert p==len(b)
    tempos=sorted(alltempos);
    def seconds(t):
        last=0;us=500000;s=0
        for pos,newus in tempos:
            if pos>t:break
            s+=(pos-last)*us/division/1000000;last=pos;us=newus
        return s+(t-last)*us/division/1000000
    notes=sorted(allnotes)
    events=[]
    for onset,end,pitch,vel,ch,k in notes:
        events.extend([(seconds(onset),1),(seconds(end),-1)])
    events.sort();level=maxpoly=0
    for t,delta in events:level+=delta;maxpoly=max(maxpoly,level)
    end_tick=max(t['end_tick'] for t in tracks)
    return {'path':str(path.relative_to(root)),'format':fmt,'tracks':tracks,'division':division,'tempos':alltempos,'programs':programs,'note_channels':dict(channels),'notes':len(notes),'first_note_s':seconds(notes[0][0]) if notes else None,'last_note_s':max(seconds(n[1]) for n in notes) if notes else None,'file_duration_s':seconds(end_tick),'pitch_range':[min(n[2] for n in notes),max(n[2] for n in notes)] if notes else None,'velocities':dict(Counter(n[3] for n in notes)),'duration_s_summary':[min(seconds(n[1])-seconds(n[0]) for n in notes),statistics.median(seconds(n[1])-seconds(n[0]) for n in notes),max(seconds(n[1])-seconds(n[0]) for n in notes)] if notes else None,'notes_over_10s':sum(seconds(n[1])-seconds(n[0])>10 for n in notes),'notes_over_60s':sum(seconds(n[1])-seconds(n[0])>60 for n in notes),'zero_duration_notes':sum(n[0]==n[1] for n in notes),'outside_88key_notes':sum(not 21<=n[2]<=108 for n in notes),'max_polyphony':maxpoly,'unmatched_off':unmatched,'dangling_on':dangling,'messages':dict(messages),'first_notes':[{'onset_s':round(seconds(n[0]),5),'offset_s':round(seconds(n[1]),5),'pitch':n[2],'velocity':n[3],'channel':n[4],'track':n[5]} for n in notes[:5]],'sha256':hashlib.sha256(b).hexdigest()}

import csv
OUT=Path(__file__).resolve().parent

def flac_info(path):
    with path.open('rb') as f:
        if f.read(4)!=b'fLaC':raise ValueError('Missing FLAC marker')
        while True:
            h=f.read(4)
            if len(h)!=4:raise ValueError('Truncated FLAC metadata')
            typ=h[0]&127;size=int.from_bytes(h[1:],'big')
            block=f.read(size)
            if len(block)!=size:raise ValueError('Truncated FLAC block')
            if typ==0:
                if len(block)!=34:raise ValueError('Invalid STREAMINFO')
                x=int.from_bytes(block[10:18],'big');sr=x>>44;ch=((x>>41)&7)+1;bits=((x>>36)&31)+1;n=x&((1<<36)-1)
                return {'path':str(path.relative_to(root)),'sample_rate':sr,'channels':ch,'bits_per_sample':bits,'samples':n,'duration_s':n/sr}
            if h[0]&128:raise ValueError('No STREAMINFO')

rows=[];errors=[];audio=[];inventory={};groups=defaultdict(list)
for protocol in ('async','sync'):
    base=root/protocol
    files=[p for p in base.rglob('*') if p.is_file()]
    for path in sorted(p for p in files if p.suffix.lower() in ('.mid','.midi')):
        parts=path.relative_to(base).parts
        try:
            r=parse(path);r.update(protocol=protocol,song=parts[1],take=parts[2],instrument=path.name.split('.')[0].split('-')[0]);rows.append(r)
            groups[(protocol,r['song'],r['take'])].append(r)
        except Exception as exc:errors.append({'path':str(path.relative_to(root)),'error_type':type(exc).__name__,'error':str(exc)})
    for path in sorted(p for p in files if p.suffix.lower()=='.flac'):
        try:audio.append(flac_info(path))
        except Exception as exc:errors.append({'path':str(path.relative_to(root)),'error_type':type(exc).__name__,'error':str(exc)})
    rs=[r for r in rows if r['protocol']==protocol]
    inventory[protocol]={'files':len(files),'bytes':sum(p.stat().st_size for p in files),'extensions':dict(Counter(p.suffix.lower() for p in files)),
        'song_directories':sorted(p.name for p in (base/'songs').iterdir() if p.is_dir()),'midi_files':len(rs),
        'midi_instruments':dict(Counter(r['instrument'] for r in rs)),'midi_take_counts':dict(Counter(r['take'] for r in rs)),
        'native_midi_filenames': [r['path'] for r in rs if '.transcribed.derived.' not in r['path']],
        'piano_midi_files':[r['path'] for r in rs if 'piano' in r['instrument']]}

summary={'scope':'Local dataset audit; no edits to source data or filtering applied. Does not establish transcription correctness or audio/annotation alignment.',
    'total_midi_files':len(rows),'total_local_song_directories':sum(len(x['song_directories']) for x in inventory.values()),
    'core_song_take_groups':sum(k[2] in ('preferred','alternate') for k in groups),
    'pitched_core_files':sum(r['take']!='extras' and r['instrument']!='drums' for r in rows),
    'percussion_core_files':sum(r['take']!='extras' and r['instrument']=='drums' for r in rows),'extra_files':sum(r['take']=='extras' for r in rows),
    'errors':errors,'formats':dict(Counter(r['format'] for r in rows)),'track_counts':dict(Counter(len(r['tracks']) for r in rows)),
    'ppq':dict(Counter(r['division'] for r in rows)),'tempo_event_counts':dict(Counter(len(r['tempos']) for r in rows)),
    'unmatched_note_offs':sum(r['unmatched_off'] for r in rows),'dangling_note_ons':sum(r['dangling_on'] for r in rows),
    'zero_duration_notes':sum(r['zero_duration_notes'] for r in rows),'completed_notes':sum(r['notes'] for r in rows),
    'programs':dict(Counter(str(r['programs']) for r in rows)),'midi_messages':dict(sum((Counter(r['messages']) for r in rows),Counter())),
    'channels_by_instrument':{inst:dict(Counter(str(sorted(r['note_channels'])) for r in rows if r['instrument']==inst)) for inst in sorted({r['instrument'] for r in rows})},
    'meters':dict(Counter(str(m['value']) for r in rows for t in r['tracks'] for m in t['meta'] if m['type']==88)),
    'velocities':dict(sum((Counter(r['velocities']) for r in rows),Counter())),
    'method_independent_quality_flags':{'definition':'Pitched files only: max polyphony >4, any note >10 seconds, or pitches outside MIDI21..108. Diagnostic flags, not automatic exclusions.'}}
for protocol in inventory:
    rs=[r for r in rows if r['protocol']==protocol]
    summary[protocol]={'completed_notes':sum(r['notes'] for r in rs),'midi_file_duration_s_min_median_max':[min(r['file_duration_s'] for r in rs),statistics.median(r['file_duration_s'] for r in rs),max(r['file_duration_s'] for r in rs)],
        'longest_note_seconds':max(r['duration_s_summary'][2] for r in rs if r['notes']),
        'notes_over_10s':sum(r['notes_over_10s'] for r in rs),'notes_over_60s':sum(r['notes_over_60s'] for r in rs),
        'max_polyphony_over16_files':sum(r['max_polyphony']>16 for r in rs),'notes_outside_88key':sum(r['outside_88key_notes'] for r in rs),
        'tempo_bpm_range':[min(60000000/r['tempos'][0][1] for r in rs),max(60000000/r['tempos'][0][1] for r in rs)]}

flagged=[]
for r in rows:
    flags=[]
    if r['instrument']!='drums':
        if r['max_polyphony']>4:flags.append('polyphony_gt4')
        if r['notes_over_10s']:flags.append('notes_gt10s')
        if r['outside_88key_notes']:flags.append('outside_88keys')
    if r['unmatched_off'] or r['dangling_on']:flags.append('unpaired_events')
    if flags:flagged.append({'path':r['path'],'protocol':r['protocol'],'take':r['take'],'instrument':r['instrument'],'flags':';'.join(flags),'notes':r['notes'],'max_polyphony':r['max_polyphony'],'max_note_duration_s':r['duration_s_summary'][2],'notes_over_10s':r['notes_over_10s'],'notes_over_60s':r['notes_over_60s'],'pitch_min':r['pitch_range'][0],'pitch_max':r['pitch_range'][1],'outside_88key_notes':r['outside_88key_notes']})
summary['quality_flagged_files']=len(flagged)
summary['quality_flagged_core_files']=sum(x['take']!='extras' for x in flagged)
summary['quality_flagged_files_by_protocol']=dict(Counter(x['protocol'] for x in flagged))
summary['quality_flagged_core_groups']=len({tuple(Path(x['path']).parts[:4]) for x in flagged if x['take']!='extras'})
summary['hash_duplicate_sets']=[[r['path'] for r in rows if r['sha256']==sha] for sha,count in Counter(r['sha256'] for r in rows).items() if count>1]

group_rows=[]
for (protocol,song,take),rs in sorted(groups.items()):
    af=[a for a in audio if Path(a['path']).parts[2:4]==(song,take)]
    group_rows.append({'protocol':protocol,'song':song,'take':take,'midi_instruments':[r['instrument'] for r in rs],
        'midi_files':[r['path'] for r in rs],'pitched_instruments':[r['instrument'] for r in rs if r['instrument']!='drums'],
        'tempo_maps_consistent':len({str(r['tempos']) for r in rs})==1,
        'last_note_off_min_max_s':[min(r['last_note_s'] for r in rs),max(r['last_note_s'] for r in rs)],
        'first_note_on_min_max_s':[min(r['first_note_s'] for r in rs),max(r['first_note_s'] for r in rs)],
        'audio_files':len(af),'audio_samples_distinct':sorted({a['samples'] for a in af}),
        'audio_duration_min_max_s':[min(a['duration_s'] for a in af),max(a['duration_s'] for a in af)] if af else None})
summary['group_instrument_sets']={protocol:dict(Counter(','.join(sorted(r['midi_instruments'])) for r in group_rows if r['protocol']==protocol and r['take']!='extras')) for protocol in inventory}
summary['groups_with_inconsistent_tempo_maps']=[(r['protocol'],r['song'],r['take']) for r in group_rows if not r['tempo_maps_consistent']]
summary['sync_audio_groups_with_different_sample_lengths']=[(r['song'],r['take']) for r in group_rows if r['protocol']=='sync' and len(r['audio_samples_distinct'])>1]
summary['flac_metadata']={'files':len(audio),'sample_rates':dict(Counter(a['sample_rate'] for a in audio)),'channels':dict(Counter(a['channels'] for a in audio)),'bits_per_sample':dict(Counter(a['bits_per_sample'] for a in audio))}
for name,data in [('inventory.json',inventory),('midi_files.json',rows),('audio_headers.json',audio),('song_take_groups.json',group_rows),('summary.json',summary)]:
    (OUT/name).write_text(json.dumps(data,indent=2)+'\n')
with (OUT/'quality_flags.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(flagged[0]));writer.writeheader();writer.writerows(flagged)

# Prefer instrument-stem lengths to mixture length when assessing export durations.
nonmixture_unequal=[]
for r in group_rows:
    if r['protocol']!='sync':continue
    af=[a for a in audio if Path(a['path']).parts[2:4]==(r['song'],r['take']) and not Path(a['path']).name.startswith('mixture.')]
    if len({a['samples'] for a in af})>1:nonmixture_unequal.append([r['song'],r['take']])
summary['sync_instrument_stems_with_different_sample_lengths']=nonmixture_unequal
catalog_path=OUT.parent/'data/catalog.json'
timing=[]
if catalog_path.exists():
    catalog=json.loads(catalog_path.read_text())['songs']
    for c in catalog:
        protocol=c['protocol'];song=c['song_subdir_basename']
        annpath=OUT.parent/'data/annotations'/protocol/(song+'.json')
        rs=[r for r in rows if r['protocol']==protocol and r['song']==song]
        if not rs or not annpath.exists():continue
        ann=json.loads(annpath.read_text())
        durations=[float(b['end_time'])-float(b['start_time']) for b in ann['bars'] if b['bar']!='0' and float(b['end_time'])>float(b['start_time'])]
        timing.append({'protocol':protocol,'song':song,'catalog_meter':c['time_signature'],'midi_meter':'4/4',
            'catalog_bpm':c['bpm'],'midi_metadata_bpm':60000000/rs[0]['tempos'][0][1],
            'bar_rows_excluding_padding':len(durations),'annotation_bar_s_min':min(durations),'annotation_bar_s_median':statistics.median(durations),
            'annotation_bar_s_max':max(durations),'annotation_bar_s_distinct_rounded_3dp':len({round(x,3) for x in durations}),
            'annotation_end_s':max(float(b['end_time']) for b in ann['bars'])})
    with (OUT/'annotation_timing.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(timing[0]));writer.writeheader();writer.writerows(timing)
summary['annotation_timing_rows']=len(timing)
summary['catalog_midi_meter_mismatches']=[{'protocol':t['protocol'],'song':t['song'],'catalog_meter':t['catalog_meter']} for t in timing if t['catalog_meter']!='4/4']
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ['total_midi_files','total_local_song_directories','quality_flagged_core_files','errors','annotation_timing_rows','catalog_midi_meter_mismatches','sync_instrument_stems_with_different_sample_lengths']},indent=2))

