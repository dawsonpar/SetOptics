# Rally detection with a small temporal convolutional network

A research write-up of a rally detector that runs on CPU, uses no LLM at
inference, and matches or beats a tuned Gemini pipeline on the games we
could score. It explains how the detector was built and tested so you can
rebuild the idea on your own footage.

## Summary

The problem: a volleyball recording is mostly downtime (warm-ups, walking
back to serve, timeouts). We want the start and end of every rally.

The approach: compute cheap per-moment signals from the video (motion,
optical flow, player pose, encoder bit cost, a frozen image embedding and a
serve probability), feed them to a small dilated temporal convolutional
network (TCN) that outputs P(in play) ten times a second, then turn that
curve into segments with a few fixed rules.

Result, IoU-F1 (defined in [Evaluation](#evaluation-protocol)), mean per
game, leave-one-game-out, 3 seeds averaged:

| | Beach | Grass | 9-man | Indoor | Mean of 4 | 3 unseen indoor games |
|---|---|---|---|---|---|---|
| TCN (shipped features) | 0.928 | 0.919 | 0.966 | 0.977 | 0.948 | 0.957 |
| Gemini baseline | 0.931 | 0.906 | 0.902 | 0.921 | 0.915 | 0.917 |

The TCN costs about $0.15 per hour of video on a 4 vCPU CPU-only cloud
container, against $1.36 for the Gemini baseline.

Released here: the method, the evaluation protocol and the numbers.
Not released: the model weights and the dataset (videos, labels, features).

## Why a TCN

A rally is defined by context, not by one frame. A single frame of players
standing still could be a serve about to happen or a timeout. Seconds later
the answer is obvious.

A TCN builds that context in layers. The first layer looks at each moment
and its immediate neighbours (0.1 s either side). The next layer looks at
neighbours of neighbours, twice as far apart. Each layer doubles the
spacing (the "dilation"), so eight layers of width 3 see about 51 seconds of
video in total, roughly 25 seconds before and after the moment being
judged. That is enough to see a serve, the rally and the walk back.

Why not ask an LLM about every second? We did, and it works (that is the
baseline below), but it has three costs:

- Money and time: ten labelling passes over the video per detection.
- Short rallies: on the Gemini baseline, rallies under 6 seconds failed 44
  to 58% of the time on every surface, against 7 to 12% overall.
- No learning from corrections: a prompt does not improve when a user fixes
  a boundary. A trained model does, on the next retrain.

A TCN is also tiny. Training one fold took about 8 seconds on a laptop.

## Data

Counts and categories only.

| Set | Games | Labels |
|---|---|---|
| Beach | 4 | Ground-truth rallies |
| Grass | 4 | Ground-truth rallies |
| 9-man (indoor variant) | 15 | 8 hand-corrected; 7 user-corrected detector output from videos whose owners allow training |
| Indoor (6-a-side) | 7 | Corrected annotations |
| Unseen indoor test | 3 | Hand-corrected from Gemini drafts (53, 46 and 29 rallies) |
| Gemini-labeled indoor | 14 | No human labels; Gemini baseline output |
| Serve labels | 10 of the games above | Serve timestamps, for the serve spotter only |

- The 30 games in the first four rows are the scored set for every
  cross-validation below. The 3 unseen games were added after every design
  choice was made and appear in no training set.
- A user-corrected video counts as ground truth: the user reviewed and fixed
  the detector's output, which is the standard the product is judged by.
- The 14 Gemini-labeled videos are unannotated indoor recordings labelled by
  the Gemini baseline. They join training at a loss weight of 0.3 and are
  never scored or used to fit the decoder. Four candidate videos turned out
  to be the same footage as scored games (frame-embedding similarity 0.998 to
  1.0) and were excluded; no remaining video matches a scored game at 0.99
  or above.
- Older indoor games were annotated under a looser start/end convention.
  They are scored forgiving up to 3.5 s of outward overhang (see
  [Evaluation](#evaluation-protocol)); newer games use plain IoU.

## Features

Everything is resampled to one 10 Hz grid (one row per 0.1 s) and
standardised per video (each column z-scored using that video's own values,
which needs no labels).

| Group | Columns | Native rate | What it is | Why |
|---|---|---|---|---|
| Motion | 11 | 10 Hz | Frame-difference energy, global plus a 3x3 grid, and its spatial concentration | Rallies move; downtime drifts |
| Flow | 18 | 10 Hz | Dense optical flow (Farneback) with the camera removed: camera dx, dy, magnitude; residual magnitude stats; up/down split; concentration; 3x3 grid | Separates a camera pan from player and ball motion, which matters on handheld footage |
| Pose | 14 | 5 Hz, held to 10 Hz | Scene aggregates from a small pose model (YOLO11n-pose), normalised by torso length or frame size | Describes what the group is doing (one player isolated behind the line, everyone static) without caring about jerseys or lighting |
| Codec | 6 | per frame, to 10 Hz | H.264 bits spent per frame, z-scored within each picture type (I, P, B) | The encoder already measured "how surprising is this frame". Free, one ffprobe pass, no decoding |
| DINOv2 | 64 | 2 Hz, held to 10 Hz | DINOv2-small CLS token plus mean patch token (768 dims), reduced to 64 by PCA fitted on the training videos only | A general sense of the scene (players on court vs crowd milling) that the hand-made features miss |
| Serve | 1 | 10 Hz | P(serve) from a separate serve spotter | A rally starts with a serve |

The serve spotter is itself a small TCN ensemble: the geometric mean of four
members trained on the 10 serve-labelled videos with pose, flow, codec and
motion inputs. Its own F1 at plus or minus 1.5 s is 0.889 without audio. Its
output enters the rally TCN as a raw channel (not standardised).

Audio was computed (log-mel bands, spectral flux, RMS, band ratios) and
removed. See [What did not work](#what-did-not-work).

Total input: 114 channels per 0.1 s.

One practical detail cost real accuracy: the production extractor must hand
each model the same frame times it trained on. Taking every 2nd and 5th
frame of a 10 Hz stream for pose and DINOv2 shifted them by 0.05 s and 0.2 s
and cost 0.027 IoU-F1 on the unseen games. Decoding once and splitting into
separate 10, 5 and 2 fps outputs recovered most of it.

## Architecture and training

A stack of dilated 1D convolutions over time. Non-causal (each layer pads
both sides, so a moment sees the past and the future). No residual
connections.

| Layer | Op | Kernel | Dilation | Channels out |
|---|---|---|---|---|
| 1 | Conv1d, BatchNorm, GELU, Dropout 0.1 | 3 | 1 | 32 |
| 2 | same | 3 | 2 | 32 |
| 3 | same | 3 | 4 | 32 |
| 4 | same | 3 | 8 | 32 |
| 5 | same | 3 | 16 | 32 |
| 6 | same | 3 | 32 | 32 |
| 7 | same | 3 | 64 | 32 |
| 8 | same | 3 | 128 | 32 |
| head | Conv1d | 1 | | 1 (logit of P(in play)) |

Receptive field: each layer with kernel 3 and dilation d adds 2d frames, so
the total is 1 + 2 x (1 + 2 + 4 + ... + 128) = 1 + 2 x 255 = 511 frames, or
51.1 s at 10 Hz.

Training:

- Target: 1 inside a ground-truth rally, else 0, per 0.1 s frame.
- Loss: binary cross-entropy, unweighted per frame. Each Gemini-labeled
  video carries a loss weight of 0.3; human-labelled games carry 1.0.
- Optimiser: AdamW, learning rate 1e-3, weight decay 1e-4.
- Schedule: 600 steps. Each step is a batch of 16 random windows of 1024
  frames (102.4 s). The video for each window is drawn in proportion to its
  length; the start offset is uniform.
- Seeds: 0, 1 and 2. The three models' probabilities are averaged and the
  average is decoded once.
- The architecture and step count were set once, before any score was seen,
  and never tuned.

The shipped model is trained on all 30 scored games plus the 14
Gemini-labeled videos, with the PCA fitted on the same set. The 3 unseen
games stay out.

## Decoding probabilities into segments

1. Smooth the 10 Hz curve with a Gaussian (sigma 1 s in the shipped model).
2. Hysteresis threshold: a rally is a run where P stays at or above `lo`
   and reaches `hi` at least once (shipped: hi 0.6, lo 0.4). This avoids
   flicker at the edges.
3. Merge rallies separated by gaps under 2 s.
4. Drop rallies shorter than 2 s.

During cross-validation the decoder is chosen from a grid (sigma 0 to 4 s,
hi 0.4 to 0.7, lo = hi minus 0.1 or 0.2, plus an optional serve-anchor
rule). Chosen values always landed at sigma 0 to 3 s, and the serve-anchor
rule was never selected in any fold.

## Evaluation protocol

### IoU-F1

For each ground-truth rally in time order, take the unused predicted rally
with the highest temporal IoU (overlap divided by union). If that IoU is at
least 0.5 it is a match. Precision is matches over predictions, recall is
matches over ground-truth rallies, and IoU-F1 is their harmonic mean. Scores
are computed per game and then averaged, so a long game does not dominate.

For the older indoor games, the IoU forgives up to 3.5 s of a prediction
starting early or ending late (and nothing else). Starting late, ending
early, missing a rally and inventing one are always penalised.

### Cross-validation

- Leave-one-game-out (LOGO): 30 folds, each trained on 29 games.
- Leave-one-surface-out (LOSO): train on three surfaces, test on the fourth.
- Leave-one-venue-out (LOVO): games clustered into venues by DINOv2 frame
  similarity, then hand-corrected; one model per held-out venue.
- Unseen: train on all 30, test on 3 indoor games recorded later.

The decoder is never fitted on the game it scores. Games are dealt into 5
surface-stratified blocks. For every pair of blocks, a model is trained on
the other three. The decoder for a held-out game in block B is the grid
point with the best mean IoU-F1 over games outside B, each predicted by a
model that also excluded B. LOSO and LOVO do the same with surfaces or
venues as the units. The PCA is refitted inside every fold on training
videos only.

### Gemini baseline

The baseline (protocol `binary_pooled_v2`) asks Gemini 3.1 Flash-Lite, for
each second of a clip, whether play is live. It runs 5 passes over 15 s
clips and 5 over 20 s clips (the two grids make partly independent boundary
errors), keeps a second only if all 10 passes agree, closes gaps under 8 s,
then shifts starts 2 s earlier and ends 4 s later. Thinking budget is 0,
which was the key control for run-to-run variance. It was tuned on the
indoor games and run unchanged on the other surfaces. Its scores above are
from the same games and the same scorer.

## Results and ablations

### How the model got here

LOGO, 3 seeds, IoU-F1 per surface:

| Step | Beach | Grass | 9-man | Indoor | Mean of 4 |
|---|---|---|---|---|---|
| v1: motion, flow, pose, codec, audio, serve | 0.862 | 0.878 | 0.946 | 0.974 | 0.915 |
| + DINOv2 channel | 0.939 | 0.907 | 0.963 | 0.984 | 0.948 |
| minus audio | 0.898 | 0.930 | 0.969 | 0.983 | 0.945 |
| + 14 Gemini-labeled videos (candidate) | 0.941 | 0.926 | 0.966 | 0.988 | 0.955 |
| Candidate, features re-extracted by the production extractor (shipped) | 0.928 | 0.919 | 0.966 | 0.977 | 0.948 |
| Gemini baseline | 0.931 | 0.906 | 0.902 | 0.921 | 0.915 |

DINOv2 was the largest single gain. Beach game-level scores swing a lot:
the weakest beach game went from 0.706 (v1) to 0.861 with DINOv2.

### Generalisation (candidate features)

| Test | Beach | Grass | 9-man | Indoor | Mean of 4 |
|---|---|---|---|---|---|
| LOGO | 0.941 | 0.926 | 0.966 | 0.988 | 0.955 |
| LOSO | 0.922 | 0.923 | 0.935 | 0.984 | 0.941 |
| LOVO (no Gemini-labeled videos) | 0.908 | 0.873 | 0.965 | 0.979 | 0.931 |

Grass is where venue holdout bites: it has two venues of two games each.
LOSO and LOVO for 9-man and indoor are optimistic; see
[Limitations](#limitations-and-caveats).

### Unseen indoor games

| | Game 1 | Game 2 | Game 3 | Mean | Precision | Recall |
|---|---|---|---|---|---|---|
| Gemini baseline | 0.911 | 0.944 | 0.897 | 0.917 | 0.944 | 0.892 |
| TCN, candidate features | 0.971 | 0.966 | 0.983 | 0.973 | 0.989 | 0.959 |
| TCN, shipped features | 0.991 | 0.967 | 0.912 | 0.957 | | |
| TCN, shipped, through the production code path | 0.991 | 0.967 | 0.931 | 0.963 | | |

Game 3 has 29 rallies and ranges 0.909 to 0.949 across single seeds, so
three games cannot separate 0.957 from 0.973. Treat the two TCN rows as the
same result.

### Does it catch what Gemini misses?

Of the distinct rallies Gemini misses (atlas of 55 Gemini runs), the
candidate matches at IoU 0.5 or better: beach 13 of 19, grass 13 of 19,
9-man 60 of 76. On the unseen indoor games, 10 of Gemini's 14 misses. The
two detectors often fail on different rallies.

### Which inputs matter

Remove one group at a time from the candidate, LOGO, 3 seeds, mean of 4
surfaces. Cost is how much the mean drops.

| Removed | Mean of 4 | Cost |
|---|---|---|
| nothing | 0.955 | |
| DINOv2 | 0.921 | 0.035 |
| Pose | 0.943 | 0.013 |
| Gemini-labeled videos | 0.945 | 0.010 |
| Serve channel | 0.946 | 0.009 |
| Flow | 0.946 | 0.009 |
| Motion | 0.951 | 0.004 |
| Codec | 0.952 | 0.003 |

Motion and codec each fell under a 0.005 keep bar, but removing both
together cost 0.012 LOGO and 0.009 LOSO, and codec alone cost 0.023 on the
unseen games, so both stay.

### What did not work

- Audio. Worth +0.003 on the LOGO mean and -0.002 on LOSO, under a 0.01 bar
  everywhere, and negative on grass in every reading. It looked like +0.041
  on beach LOGO, but that fell to +0.002 LOSO and sits inside beach's seed
  spread. Removed. It was also the noisiest signal to begin with.
- DINOv3 instead of DINOv2 (ViT-S/16, same pooling and PCA). +0.006 LOGO
  without the Gemini-labeled videos, -0.003 with them, and -0.010 LOSO. No
  reliable gain, so DINOv2 stays.
- DINOv2 plus serve alone, without the hand-made features: 0.902, 0.902,
  0.946, 0.972 (beach, grass, 9-man, indoor). Worse on every surface than
  DINOv2 plus the hand-made features.
- Serve-anchor rule in the decoder (keep a rally only if a serve peak sits
  just before its start). Available in every decoder grid and never chosen.
  The serve spotter misses serves, so requiring one throws away real
  rallies.
- Serve-anchored correction of Gemini output. Adding rallies after
  confident serves helped by about +0.01. Dropping Gemini rallies with no
  serve failed badly: at its lowest threshold it still removed 11 to 39
  real rallies per run.
- A warm-up gate based on rally rhythm. 51 of 56 false rallies outside play
  fall 2 to 40 s before the first real rally at normal rally spacing. That
  is pre-game play, and timing alone cannot tell it from a match.
- Mac profiling as a cost estimate. An old torch on a laptop ran pose 5x
  slower than a current torch on the same frames. Measure on the target CPU
  before buying GPUs or retraining at lower rates.

## Cost and compute

Measured on a CPU-only container at 4 vCPU and 16 GiB, 1080p input:

- Runtime: 0.44x real time at 60 fps and 0.39x at 30 fps, so a 20-minute
  video takes about 8 to 9 minutes. Peak memory 3.0 GB. No GPU.
- The largest remaining cost is the ffprobe pass for codec features, which
  overlaps the decode.
- At list price for that container (about $0.37 per wall-clock hour), about
  $0.15 per video-hour. The Gemini baseline costs $1.36 per video-hour.
- Pose at 2 Hz would save about 0.1x but needs a retrain. OpenVINO saved
  under 0.05x and doubled memory. Neither was worth it.
- Feature extraction on Apple Silicon ran at 0.14x real time; DINOv2-small
  alone runs at about 14 frames/s there.

## Limitations and caveats

These come from the experiment log and from isolated reviews of its code
and numbers.

- Small dataset. Beach and grass rest on 4 games each, and the same
  configuration moves by up to 0.05 between one seed and three. On those
  two surfaces the TCN is level with Gemini within noise, not better.
- 9-man and indoor footage look alike. Every indoor game has a 9-man game
  at 0.86 to 0.91 frame similarity, so LOSO for those two is not a truly
  unseen surface. "Holds on an unseen surface" is supported for beach
  (0.924) and grass (0.903) only.
- The serve channel leaks under LOSO and LOVO. Serve tracks were built
  once, from serve models trained on labelled videos that include the
  held-out surface or venue. LOSO and LOVO numbers for 9-man and indoor
  are optimistic by an unknown amount. LOGO sees only a second-order effect.
- Venue overlap inflates LOGO within a surface: some 9-man games are 0.96
  to 0.97 similar to each other.
- "Unseen" is weaker than it sounds. Two of the three unseen games are more
  similar than the venue cut (0.870) to training material (0.906 and 0.893).
- Gemini shaped some labels. The unseen ground truth started as Gemini
  drafts (this favours Gemini on unchanged rallies), the Gemini-labeled
  training videos come from the same baseline, and the 7 user-corrected
  9-man games began as detector output. Part of the TCN's lead with
  Gemini-labeled videos may be learning Gemini's boundaries.
- Some choices were made on test scores: the best variant was picked on
  LOGO, and the 0.005 keep bar sits inside the seed spread.
- Warm-up and long stoppages still cause false rallies, for both detectors.
- The shipped features differ slightly from the research features
  (frame-embedding precision and frame timing), which is why the shipped
  row is a little lower than the candidate.

## How to try it yourself

This repo does not ship the TCN, its weights or its features. These public
pieces cover the rest:

1. Label rallies. Use the desktop annotation tool in
   [`tools/annotation-ui/`](../tools/annotation-ui/README.md), or draft
   labels with the Gemini pipeline in `tools/annotation/` and correct them
   (see [`annotation.md`](annotation.md)). Aim for several games per surface
   and several venues: holding out a venue is where scores dropped most.
2. Extract features at 10 Hz. Everything above uses public tools: ffmpeg
   and ffprobe (frames and per-frame bit cost), OpenCV Farneback flow,
   YOLO11n-pose from ultralytics, and `facebook/dinov2-small` from Hugging
   Face. Decode once, hand each model the frame times it was trained on,
   and z-score each column per video.
3. Train the TCN in the table above with BCE on a 0/1 in-play target. Fit
   any PCA inside each fold. Average three seeds.
4. Decode with smoothing, hysteresis, a 2 s merge gap and a 2 s minimum.
5. Score with [`tools/shared/eval/temporal.py`](../tools/shared/eval/temporal.py)
   at IoU 0.5, per game, leave-one-game-out at minimum. Its default matcher
   pairs the highest-IoU pairs first rather than walking ground truth in
   order, so numbers can differ slightly from the ones here. Fit decoder
   thresholds without the held-out game, and hold out whole venues before
   you trust a number.
6. Cut the rallies into a video with `scripts/export_rallies.py` (it reads
   any `{start_ms, end_ms, type: "in-play"}` JSON).

If you skip the serve spotter and the Gemini-labeled videos, expect to lose
about 0.01 each on the mean (see [Which inputs matter](#which-inputs-matter)).
DINOv2 is the one input not to skip.
