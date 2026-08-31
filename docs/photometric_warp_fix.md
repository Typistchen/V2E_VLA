# Photometric warp halo fix

## Problem

The legacy `b_primary` compositor used the warped future keyframe throughout a
keyframe gap. HDR shading therefore changed abruptly between the real previous
frame and the first synthesized frame. A complete multi-threshold event model
correctly converted that artificial contrast step into dense level-set rings.

## Implementation

`DVSCamera` now defaults to `composite="log_blend"` and `mv_dilate=1`.

- Co-visible, depth/flow-consistent pixels interpolate linearly in log luminance.
- Depth and warped-motion disagreement creates a conservative validity mask.
- Holes and unreliable occlusion boundaries do not generate signal events.
- Pixels recovering from an invalid interval silently re-anchor their event
  reference instead of emitting the unknown accumulated change.
- `b_primary` remains available for legacy comparisons.

The initial conservative checkpoint is `8ac5418`. It used depth/flow agreement
as a hard validity gate and removed too many events from accelerating/rotating
task objects. The balanced checkpoint `e90175b` keeps agreement as a compositing
cue and hard-masks only true bidirectional coverage holes.

## Seed-2 DOM validation

All runs use the same DOM scene, task, threshold (`0.15`), HDR input, 360x480
cameras, and 158 simulation frames.

| Run | Total events | Wrist events | 10 ms phase imbalance (opst/side/wrist) |
| --- | ---: | ---: | --- |
| Legacy warp=4 | 15,361,356 | 8,765,062 | 2.96 / 3.71 / 1.41 |
| Conservative warp=4 (`8ac5418`) | 6,847,164 | 2,862,544 | 1.23 / 1.25 / 1.37 |
| Balanced warp=4 (`e90175b`) | 16,645,741 | 9,802,022 | 1.36 / 1.43 / 1.17 |
| No-warp control | 11,195,602 | 6,277,052 | 1.34 / 1.40 / 1.31 |

At 2 seconds the legacy wrist stream changes from 14,392 events in one 10 ms
bin to 2,818 in the next. The conservative stream stays between 4,046 and
4,899, but visibly loses the cup. The balanced stream retains the cup and ranges
from 8,629 to 12,658 over the same four bins. All streams have monotonic
timestamps.

The large periodic burst is reduced, but the no-warp control still contains
weaker rings around the HDR illumination spot. Those remaining events come from
real keyframe radiometry (view-dependent lighting/reflection) rather than the
warp compositor. The conservative checkpoint reduces the wrist stream to about
46% of the no-warp count and erases the manipulated object in some frames; it
must not be used as the default. The balanced checkpoint is the VLA baseline.

## Next evaluation

Train/evaluate VLA inputs with polarity-separated temporal voxels. If the
remaining low-frequency illumination rings still dominate, compare this
baseline against photoreceptor bandwidth filtering or local illumination
normalization as a separate, explicitly non-ideal sensor variant.
