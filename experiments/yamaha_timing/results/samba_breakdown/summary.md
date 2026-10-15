# Timing network on JazzSAMBA: breakdown

BPM condition at the played tempo unless marked; mir_eval F-measure (±70 ms).

| subset | takes | beat F | downbeat F | beat CMLt | downbeat CMLt | beat F, written tempo | downbeat F, written tempo |
|---|---|---|---|---|---|---|---|
| all | 152 | 0.532 | 0.337 | 0.301 | 0.325 | 0.519 | 0.325 |
| async (click) | 100 | 0.505 | 0.303 | 0.272 | 0.279 | 0.506 | 0.304 |
| sync (live) | 52 | 0.584 | 0.404 | 0.356 | 0.412 | 0.544 | 0.366 |

## By style

| genre | takes | beat F | downbeat F | beat CMLt |
|---|---|---|---|---|
| waltz | 10 | 0.515 | 0.441 | 0.201 |
| calypso | 2 | 0.617 | 0.401 | 0.355 |
| latin | 14 | 0.556 | 0.373 | 0.304 |
| bossa nova | 24 | 0.515 | 0.371 | 0.297 |
| samba | 10 | 0.530 | 0.349 | 0.208 |
| ballad | 18 | 0.527 | 0.322 | 0.456 |
| swing | 58 | 0.545 | 0.310 | 0.310 |
| bebop | 16 | 0.497 | 0.292 | 0.210 |

## By what the piano is doing (sections, time-weighted)

| piano | sections | minutes | beat F | downbeat F |
|---|---|---|---|---|
| piano solo | 597 | 142 | 0.484 | 0.289 |
| comping | 3347 | 791 | 0.526 | 0.330 |
| laid out | 141 | 30 | 0.153 | 0.074 |

## Is the downbeat wrong, or just on the wrong beat?

For each take: the beat of the bar the network most often calls "1", and downbeat F if that
offset were the only error (predicted downbeats scored against that beat of every bar).

| subset | takes | calls beat 1 | calls another beat | downbeat F | downbeat F, offset fixed |
|---|---|---|---|---|---|
| all | 152 | 149 | 3 | 0.337 | 0.339 |
| async (click) | 100 | 98 | 2 | 0.303 | 0.305 |
| sync (live) | 52 | 51 | 1 | 0.404 | 0.406 |

## What goes with a good downbeat F (Spearman, per take)

Pooled over both protocols; live takes score higher and vary more in tempo, so check within protocol.

| take property | rho | p | rho async | p | rho sync | p |
|---|---|---|---|---|---|---|
| played_bpm | +0.14 | 0.092 | +0.12 | 0.242 | +0.09 | 0.509 |
| tempo_range_4bar | +0.35 | 0.000 | +0.14 | 0.173 | -0.09 | 0.539 |
| bar_to_bar_change | +0.22 | 0.006 | -0.16 | 0.120 | -0.21 | 0.135 |
| piano_notes_per_s | +0.21 | 0.009 | +0.08 | 0.456 | +0.20 | 0.160 |
| laid_out_share | -0.03 | 0.745 | -0.31 | 0.002 | -0.08 | 0.565 |
| swung | -0.25 | 0.002 | -0.39 | 0.000 | +0.02 | 0.868 |

## Best and worst takes (downbeat F)

| take | genre | played BPM | 4-bar tempo range | piano notes/s | laid out | beat F | downbeat F | calls as "1" |
|---|---|---|---|---|---|---|---|---|
| 51-amazing_grace/preferred | swing | 225 | 7% | 9.3 | 4% | 0.84 | 0.76 | beat 1 (82%) |
| 37-song_for_my_father/alternate | latin | 125 | 0% | 7.2 | 0% | 0.86 | 0.64 | beat 1 (75%) |
| 51-amazing_grace/alternate | swing | 205 | 15% | 9.8 | 11% | 0.69 | 0.62 | beat 1 (87%) |
| 57-chega_de_saudade/preferred | bossa nova | 189 | 10% | 8.6 | 5% | 0.69 | 0.62 | beat 1 (80%) |
| 74-waltz_for_debby/alternate | waltz | 205 | 105% | 6.8 | 15% | 0.65 | 0.60 | beat 1 (89%) |
| ... | | | | | | | | |
| 49-woodchoppers_ball/alternate | swing | 170 | 0% | 8.7 | 0% | 0.56 | 0.12 | beat 4 (33%) |
| 70-samba_de_orfeu/alternate | samba | 230 | 24% | 10.9 | 9% | 0.37 | 0.15 | beat 2 (43%) |
| 20-impressions/preferred | bebop | 250 | 0% | 7.6 | 7% | 0.56 | 0.15 | beat 3 (27%) |
| 18-i_wish_you_love/preferred | ballad | 110 | 0% | 5.8 | 0% | 0.38 | 0.15 | beat 1 (38%) |
| 20-impressions/alternate | bebop | 250 | 0% | 8.4 | 7% | 0.57 | 0.15 | beat 1 (30%) |

Figures: best `51-amazing_grace/preferred`, median `10-desafinado/preferred`, worst `49-woodchoppers_ball/alternate` (timeline and 30 s detail each), and overview.png.
