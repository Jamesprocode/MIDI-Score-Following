"""Build concise meeting notes and SVG diagnostics from saved study results."""
from pathlib import Path
import csv
import html
import json
import xml.etree.ElementTree as ET
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
BENCH = HERE.parent
summary = json.loads((OUT / "summary.json").read_text())
details = json.loads((OUT / "case_detail.json").read_text())
cases = [c for c in summary["cases"] if c["song"] in ("59-groovin_high", "70-samba_de_orfeu")]
COLORS = {"Original": "#1763ac", "Equal initial": "#d56412", "Clock only": "#888888", "Reference": "#333333"}


def plot_case(case):
    song = case["song"]
    folder = OUT / "traces" / song
    ev = np.load(folder / "evidence.npz")
    a = np.load(folder / "A_gaussian.npz")
    b = np.load(folder / "B_equal_initial.npz")
    clock = np.load(folder / "clock.npz")
    annotation = json.loads((BENCH / "data/annotations/sync" / (song + ".json")).read_text())
    width, height, left, right = 1160, 630, 82, 24
    xmax = float(ev["time"][-1])
    x = lambda value: left + value / xmax * (width-left-right)
    elements = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(case["title"])} tracking diagnostic">',
                f'<rect width="{width}" height="{height}" fill="white"/>',
                '<g font-family="Arial,sans-serif" font-size="13" fill="#222">',
                f'<text x="{left}" y="27" font-size="20" font-weight="bold">{html.escape(case["title"])}</text>']
    solos = [s for s in annotation["soloists"] if s["section"] in ("solos-piano", "solos-drums")]
    for i,(label,color) in enumerate(COLORS.items()):
        elements.extend([f'<line x1="{left+i*220}" x2="{left+24+i*220}" y1="51" y2="51" stroke="{color}" stroke-width="2"/>',
                         f'<text x="{left+31+i*220}" y="56">{label}</text>'])
    errors = {"Original": a["estimated_beats"]-ev["reference_beats"],
              "Equal initial": b["estimated_beats"]-ev["reference_beats"],
              "Clock only": clock["estimated_beats"]-ev["reference_beats"]}
    tempo = {"Original": a["mean_bpm"], "Equal initial": b["mean_bpm"], "Reference": ev["reference_bar_bpm"]}
    for top,values,title in [(95, errors, "Position error in beats: negative means behind"),
                             (350, tempo, "Tempo distribution mean; reference BPM comes from annotated bars")]:
        bottom = top+185
        ymin = min(float(np.min(v)) for v in values.values())
        ymax = max(float(np.max(v)) for v in values.values())
        if values is errors: ymin=min(ymin,-2);ymax=max(ymax,2)
        pad=(ymax-ymin)*.08 or 1
        ymin-=pad;ymax+=pad
        y=lambda value: bottom-(value-ymin)/(ymax-ymin)*(bottom-top)
        elements.append(f'<text x="{left}" y="{top-12}" font-weight="bold">{title}</text>')
        for s in solos:
            start,end=float(s["start_time"]),float(s["end_time"])
            color="#fff3d3" if s["section"]=="solos-piano" else "#e8edf3"
            elements.append(f'<rect x="{x(start):.2f}" y="{top}" width="{x(end)-x(start):.2f}" height="185" fill="{color}"/>')
            elements.append(f'<text x="{x(start)+4:.2f}" y="{top+15}" font-size="11">{s["section"].replace("solos-", "")} solo</text>')
        for value in np.linspace(ymin,ymax,5):
            elements.extend([f'<line x1="{left}" x2="{width-right}" y1="{y(value):.2f}" y2="{y(value):.2f}" stroke="#ddd"/>',
                             f'<text x="{left-9}" y="{y(value)+4:.2f}" text-anchor="end">{value:.0f}</text>'])
        if ymin<=0<=ymax:
            elements.append(f'<line x1="{left}" x2="{width-right}" y1="{y(0):.2f}" y2="{y(0):.2f}" stroke="#666" stroke-dasharray="3 3"/>')
        for label,values_y in values.items():
            take = np.unique(np.r_[np.arange(0,len(ev["time"]),2), len(ev["time"])-1])
            points=" ".join(f'{x(ev["time"][i]):.2f},{y(values_y[i]):.2f}' for i in take)
            dash=' stroke-dasharray="5 4"' if label in ("Equal initial", "Clock only") else ''
            elements.append(f'<polyline fill="none" stroke="{COLORS[label]}" stroke-width="1.5"{dash} points="{points}"/>')
        elements.append(f'<line x1="{left}" x2="{width-right}" y1="{bottom}" y2="{bottom}" stroke="#555"/>')
        for value in np.linspace(0,xmax,6):
            elements.append(f'<text x="{x(value):.2f}" y="{bottom+19}" text-anchor="middle">{value:.0f}</text>')
        elements.append(f'<text x="{width/2}" y="{bottom+39}" text-anchor="middle">Recording time (seconds)</text>')
    elements.append(f'<text x="{left}" y="602" font-size="12">Shaded solos are annotation context only. Piano is excluded from the tested input. Initial-only weights become identical in reported position after {details[song]["last_A_B_position_difference_seconds"]:.1f} s.</text>')
    elements.append('</g></svg>')
    svg="\n".join(elements)
    ET.fromstring(svg)
    name=song+"_diagnostic.svg"
    (OUT/name).write_text(svg)
    return name


def table(headers, rows):
    cells=lambda values, tag: '<tr>'+''.join(f'<{tag}>{html.escape(str(v))}</{tag}>' for v in values)+'</tr>'
    return '<table><thead>'+cells(headers,'th')+'</thead><tbody>'+''.join(cells(row,'td') for row in rows)+'</tbody></table>'


def main():
    svg_names={c['song']:plot_case(c) for c in cases}
    labels={"A_gaussian":"Original: Gaussian initialization and dynamics", "B_equal_initial":"Equal initialization only",
            "C_equal_dynamics":"Equal dynamics only", "D_equal_both":"Equal initialization and dynamics",
            "no_harmony":"Same original filter, without harmony", "clock":"Constant metadata-tempo clock"}
    aggregate=table(["Setting", "Correct bar", "Within two beats", "Mean absolute error (beats)"],
                    [[labels[r['method']],f"{100*r['bar_accuracy']:.2f}%",f"{100*r['within2beat_accuracy']:.2f}%",f"{r['mean_abs_beats']:.2f}"] for r in summary['aggregate']])
    case_table=table(["Piece", "Original correct bar", "Equal initial correct bar", "Original mean error"],
                     [[c['title'],f"{100*c['variants']['A_gaussian']['bar_accuracy']:.2f}%",f"{100*c['variants']['B_equal_initial']['bar_accuracy']:.2f}%",
                       f"{c['variants']['A_gaussian']['mean_abs_beats']:.2f} beats"] for c in cases])
    content=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Live-band harmonic follower: tempo-prior test</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;color:#222;max-width:1160px;margin:36px auto;padding:0 22px}}h1{{font-size:28px}}h2{{font-size:21px;margin-top:32px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px 12px;text-align:left;border-bottom:1px solid #ddd}}th{{background:#f3f5f7}}img{{width:100%;height:auto;border:1px solid #ddd}}p{{max-width:1000px}}a{{color:#1763ac}}.small{{font-size:14px;color:#555}}</style>
<h1>Live-band harmonic follower: what the first experiment shows</h1>
<p><strong>Scope:</strong> 26 simultaneous-band preferred takes, using the original bass-and-horn MIDI transcriptions. Overdubbed recordings are excluded. This is Maezawa-inspired chord matching inside our own position–tempo filter, a first step toward a baseline. A neural beat/downbeat tracker is not included.</p>
<h2>Equal starting tempo probabilities barely change the average</h2>
<p>The primary comparison changes only initial probabilities: 11.11% per tempo instead of a Gaussian preference around metadata BPM. The nine tempo values, chord matching, transition rule and 200-ms updates stay fixed. Ten pieces improve, fourteen worsen and two are unchanged. Probabilities are free to change after initialization.</p>
{aggregate}
<p class="small">Equal dynamics is a separate sensitivity: it changes the ongoing tempo-transition acceptance rule, including directional preference and effective mobility. It still permits tempo changes. All variants retain the 0.7–1.3× metadata tempo range and a known synchronized start. Metrics average pieces equally.</p>
<h2>One strong and one weak case</h2>{case_table}
<p>These are the best and worst original cases, selected before the equal-weight runs. They are explanatory examples, not representative samples.</p>
<h2>Groovin’ High: harmony corrects a substantial clock mismatch</h2>
<p>Metadata is 185 BPM; reference bar-derived tempo is approximately 210 BPM overall. Original tracking reaches 86.81% correct bars and 0.66-beat mean absolute error; the constant clock has about 48.74-beat mean error. The follower increasingly supports faster tempo candidates. A short error entering the bass solo is followed by recovery. During the drum solo it maintains 89.3% correct bars despite only 21.5% pitched-input coverage.</p>
<img src="{svg_names['59-groovin_high']}" alt="Groovin' High signed position error and tempo diagnostic">
<h2>Samba de Orfeu: ambiguous evidence and missing parts accompany drift</h2>
<p>Metadata is 200 BPM, close to the overall reference of about 208 BPM and within the supported range. Nevertheless, original mean error reaches 44.04 beats, worse than the clock’s 21.26 beats. From about 106 to 378.4 seconds it remains more than four beats wrong.</p>
<p>During the piano solo (152–217 s), piano is absent from the tested input; bass dominates, and 97.8% of active snapshots tie several chord templates. Mean posterior tempo is about 173 BPM versus 206 BPM reference. During the drum solo (280–343 s), only 8.6% of snapshots contain pitched evidence; the already incorrect state persists. These associations identify failure conditions, not a controlled proof of one cause.</p>
<p>At 120 s, bass and trumpet both play G while the annotation is D minor 7. The binary D-minor template scores zero while several templates containing G score two. This is an evidence mismatch; listening or cleaner MIDI is needed to determine whether it reflects improvisation, transcription error or annotation timing.</p>
<img src="{svg_names['70-samba_de_orfeu']}" alt="Samba de Orfeu signed position error and tempo diagnostic">
<p>Equal initialization produces exactly the same reported positions after 76 s, so it does not repair the later sustained failure. Repeated harmony leaves many locations plausible; the error is not simply a persistent wrong-chorus lock. Late abrupt corrections are switches between surviving position modes.</p>
<h2>What this supports next</h2>
<ul><li>Keep this as a transparent first baseline: harmonic evidence can improve an uncertain clock on live-band data, but it can also worsen position error.</li><li>Study correction behavior when observations are sparse or tied, rather than optimizing initial tempo weights alone.</li><li>Use the newly available native piano MIDI in a separate input comparison before attributing missing-part failures to the fusion method.</li><li>Compare a simple ambiguity-aware correction rule with fixed harmonic strength before proposing learned weighting.</li></ul>
<p class="small">Reference positions interpolate within annotated bars; reference BPM is evaluation-only, not independent beat annotations. The known performance plan is unfolded from annotation bar/fraction coordinates. No unknown starts, unplanned repeats/skips, latency or human interaction are tested. Native piano and alternate sync annotations now exist locally but are excluded to preserve the original comparison. The results do not establish performance of Maezawa’s complete system or prove adaptive weighting will solve the failures.</p>
<p><strong>Verification:</strong> all 26 original traces reproduced exactly; equal initial weights checked; all saved tempo probabilities finite and normalized; original files unchanged.</p>
<p><a href="meeting_notes.md">Advisor bullets</a> · <a href="metrics.csv">All per-piece results</a> · <a href="per_song_comparison.csv">Paired changes</a> · <a href="case_sections.csv">Section diagnostics</a> · <a href="summary.json">Summary and checks</a></p></html>'''
    (OUT/'report.html').write_text(content)
    notes='''- First baseline step: Maezawa-inspired chord templates plus a probabilistic timing model; full neural beat/downbeat tracking is still to be implemented.
- Focus this analysis on 26 live-band performances. Keep MIDI inputs and tracker settings identical across comparisons.
- Harmony adds useful information: average correct-bar accuracy is 54.11%, versus 7.45% for the same filter without harmony and 7.72% for the constant clock.
- Strong case — Groovin’ High: 86.81% correct bars, 0.66-beat mean error; follows despite metadata 185 BPM versus approximately 210 BPM performance.
- Weak case — Samba de Orfeu: 12.88% correct bars, 44.04-beat mean error; bass-only piano-solo evidence is highly ambiguous, followed by almost no pitched evidence during the drum solo. Harmony can worsen the error.
- Equal initial tempo probabilities barely change the average: 54.11%→53.98%. Samba’s later failure is unchanged after 76 seconds.
- Removing the ongoing tempo preference as well gives 52.32%; this also changes the tempo-transition behavior.
- Next: test the newly available native piano MIDI separately, then examine when to limit harmonic corrections during sparse or ambiguous passages. These results motivate that question; they do not prove learned weighting is required.
- Evaluation uses annotated bar timing and a known unfolded plan/start; this is exploratory causal replay, not a complete real-time accompaniment evaluation.
'''
    (OUT/'meeting_notes.md').write_text(notes)
    print('Saved report.html, meeting_notes.md and two validated SVG diagnostics')


if __name__=='__main__':
    main()
