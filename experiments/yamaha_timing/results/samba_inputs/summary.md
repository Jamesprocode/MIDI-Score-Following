# JazzSAMBA: piano only vs more of the band as input

mir_eval F-measure (±70 ms); bass and horns are dataset transcriptions; no drums.

## BPM condition: written tempo

| input | takes | beat F all | downbeat F all | beat F async | downbeat F async | beat F sync | downbeat F sync | beat CMLt all | downbeat CMLt all |
|---|---|---|---|---|---|---|---|---|---|
| piano | 152 | 0.519 | 0.325 | 0.506 | 0.304 | 0.544 | 0.366 | 0.280 | 0.293 |
| piano+bass | 152 | 0.606 | 0.387 | 0.590 | 0.341 | 0.637 | 0.476 | 0.384 | 0.352 |
| piano+bass+horns | 152 | 0.596 | 0.373 | 0.579 | 0.331 | 0.629 | 0.455 | 0.370 | 0.335 |
| bass | 150 | 0.517 | 0.260 | 0.479 | 0.202 | 0.587 | 0.370 | 0.323 | 0.236 |

By what the piano is doing (sections, time-weighted, beat F / downbeat F):

| input | piano solo | comping | piano laid out |
|---|---|---|---|
| piano | 0.47 / 0.28 | 0.52 / 0.32 | 0.15 / 0.07 |
| piano+bass | 0.60 / 0.37 | 0.59 / 0.37 | 0.25 / 0.14 |
| piano+bass+horns | 0.59 / 0.35 | 0.58 / 0.36 | 0.26 / 0.13 |
| bass | 0.53 / 0.26 | 0.46 / 0.22 | 0.21 / 0.12 |

## BPM condition: played tempo

| input | takes | beat F all | downbeat F all | beat F async | downbeat F async | beat F sync | downbeat F sync | beat CMLt all | downbeat CMLt all |
|---|---|---|---|---|---|---|---|---|---|
| piano | 152 | 0.532 | 0.337 | 0.505 | 0.303 | 0.584 | 0.404 | 0.301 | 0.325 |
| piano+bass | 152 | 0.621 | 0.401 | 0.589 | 0.340 | 0.682 | 0.519 | 0.410 | 0.389 |
| piano+bass+horns | 152 | 0.611 | 0.388 | 0.578 | 0.331 | 0.675 | 0.496 | 0.396 | 0.370 |
| bass | 150 | 0.533 | 0.276 | 0.479 | 0.203 | 0.634 | 0.413 | 0.348 | 0.265 |

By what the piano is doing (sections, time-weighted, beat F / downbeat F):

| input | piano solo | comping | piano laid out |
|---|---|---|---|
| piano | 0.48 / 0.29 | 0.53 / 0.33 | 0.15 / 0.07 |
| piano+bass | 0.61 / 0.38 | 0.60 / 0.38 | 0.26 / 0.14 |
| piano+bass+horns | 0.60 / 0.36 | 0.59 / 0.37 | 0.27 / 0.14 |
| bass | 0.54 / 0.27 | 0.48 / 0.23 | 0.22 / 0.13 |

## Paired, played tempo: piano+bass+horns minus piano

- takes: 152; downbeat F better in 107, worse in 43 (median change +0.043); beat F better in 123, worse in 27 (median change +0.065)
