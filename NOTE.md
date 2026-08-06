# Week Ending 5 Aug

## Benchmark and Compute

Most of the time went into setting up the sketch benchmark — specifically figuring out
and configuring the benchmarks to run on the ICF compute cluster.

- Smaller benchmarks run correctly on the cluster, and results/checkpoints import into the UI.
- Larger benchmarks consist of many jobs, and the variable nature of the ICF Teaching
  partition nodes makes it difficult to keep tasks running consistently.

### Teaching partition (have access)

Good VRAM available:

| GPU type                   | free | used | total | unusable | nodes |
| -------------------------- | ---: | ---: | ----: | -------: | ----: |
| nvidia_geforce_rtx_2080_ti |   81 |    9 |    90 |        0 |    12 |
| h200_1g.18gb               |   35 |    0 |    35 |        0 |     1 |
| nvidia_rtx_a6000           |    0 |    8 |     8 |        0 |     1 |
| h200                       |    0 |    1 |     1 |        0 |     1 |
| h200_3g.71gb               |    0 |    4 |     4 |        0 |     1 |

### ICF-Free partition (no access right now)

More capacity, but currently unavailable:

| GPU type                   | free | used | total | unusable | nodes |
| -------------------------- | ---: | ---: | ----: | -------: | ----: |
| nvidia_geforce_rtx_2080_ti |   81 |    7 |    88 |        0 |    11 |
| h200_1g.18gb               |   35 |    0 |    35 |        0 |     1 |
| a40                        |   18 |   10 |    28 |        0 |     7 |
| nvidia_l40s                |   11 |   21 |    32 |        0 |     8 |
| nvidia_h200                |    0 |    8 |     8 |        0 |     1 |
| nvidia_rtx_a6000           |    0 |    8 |    12 |        4 |     2 |
| h200                       |    0 |    1 |     1 |        0 |     1 |
| h200_3g.71gb               |    0 |    4 |     4 |        0 |     1 |

### Local

Most tasks can be run on my RX 7800XT (16GB), excluding TRELLIS 2 (24GB). TRELLIS 1 is fine.
The AMD GPU makes some tasks need fixes e.g. TRELLIS-1 has the open-source fork TRELLIS-AMD.

---

## Directions

### 1. Agentic

I prompted Claude (Opus 5, medium effort) three times in the same context to surface the
excavator sketch model with regular primitives (sphere, cylinder, rectangle, any regular prism).
I did not mention specifics on how to surface the sketch model, Claude self-directed its own
iterative process and made the supporting scripts.

**First prompt - surface entire model**

Claude produced a rough primitive fit. It demonstrated that Claude was able to optimize the
primitives to fit the sketch, but the output was very blocky and imprecise.

Completely unprompted, Claude devised an iterative process: fit the primitives, then render to
matplotlib (1: the sketch, 2: the fit primitives, 3: highlighted overlaps) and take reference
from the rendered plot.

This suggested Claude was able to reason about and fit to the sketch data, but that optimizing
across the whole model at once was imprecise.

**Second prompt - focus on a single part**

I prompted it to optimize the tracks and duplicate across all 4 tracks on the model. I did
mention to focus on the fact that there are ridges on the tracks.

Claude made the primitive geometry more precise, added ridges, and attempted to provide
primitives solely for semantic detail rather than geometric structure.

**Third prompt — fix the kinematic/articulation error in the joint chain to the tracks**

I mentioned that there is a kinematic issue, and told Claude to find the error and create
geometry that worked across the entire joint range without semantic inconsistencies or large
overlap.

Claude found the kinematic issue, which spanned multiple parts, and noticed the vague user
intent in the sketch that actually described the hole a connector went through. It made a script
to test the model across snapshots of the kinematic range, and created geometry that was
kinematically stable and worked semantically with the sketches.

There are issues in the primitive fit for the kinematics, but these are largely due to the lack
of tools given to Claude (no boolean cutouts makes it hard to indent into the track) and the
imprecision of the prompt.

All prompts were done in under 100k tokens.

**Findings**

Claude can optimize fit to the sketches, generate geometry with fidelity, reason about kinematic
ranges, and create geometry that was reasonable under these terms.

Making Claude more accurate comes down to:

1. Optimizing and enlarging the context
2. Precise prompts / iterative flow
3. A good toolset to operate within

Essentially, it is an engineering problem. The scientific contribution in this direction is hard
to find - the results get better the better the model is, and the longer it thinks for. It is
also very possible that engineering a good pipeline to optimize all of this gets replaced by a
better model that does it in a way it finds more intuitive itself.

### 2. TRELLIS voxel hacking

The idea is that the voxel occupancy stage in TRELLIS can be conditioned via inpainting with the
3D strokes.

Coming experiments could be:

- Multi-view image conditions of 3D strokes -> TRELLIS -> geometry
- Augmented multi-view image conditions (potentially passed through a generative image model)
- Image conds + sketch voxelisation in voxel occ -> TRELLIS -> geometry
  - Sketch voxelisation and generic/simple topology filling in voxel occ
  - Sketch voxelisation and potentially probabilistic surface voxels from NeuralSketch2Surf in
    voxel occ

Moving forwards as ablations. Will be looking for metrics like fit to sketch specifications and
semantic/structural identity/consistency.

Experiments for articulations are possible - for example, using the same inpainting logic to
guide generation away from kinematic overlap regions.

**On the literature**

Although the surfacing literature works in second-order terms to optimize curvature, structural
sketches (and curvature/flow sketches) are akin to first-order boundaries. The professional
sketches from SketchLab contain sketches that specify strictly:

1. The geometric boundary of the object
2. Semantic details that are impossible to represent as filling in geometry

The second-order optimization exists to fill in the variable surface between sketches. This
literature standard does not discount using a first-order process like TRELLIS structured flow
matching to specify against the 3D sketches, since, as said, they are first-order (or
zeroth-order) boundaries.

This direction seems much more plausible than the agentic solution. Depending on how the
experiments go, the contribution could be either surfacing only, or articulation as the main
contribution.

### Other directions still available

1. Parent-child joint-knowing surfacer (medium/easy)
2. Neural surfacer with differentiable definition of articulation constraints (hard)
