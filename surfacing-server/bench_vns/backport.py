"""Backport the two benchmark-harness features into the older bench forks.

The three bench forks (see bench_vns.py) differ only in the performance work
being measured, but to run the *same* protocol on all of them the two older
ones need two features that only exist further up the stack:

  * ``--slim_output``  (from the fork's "Slim output" commit) so no fork writes
    checkpoints / grid dumps / samplings during the timed run. Step 2 and 3
    already have it; only step 1 needs it.
  * ``--iso_after`` / ``--smooth_after`` (from the losses_torch branch) so the
    initialization stage and the point where L_smooth enters the loss can be
    moved off their hardcoded 200 / 100. Steps 1 and 2 both need these.

Everything else is left alone on purpose: the k-NN search, the open3d distance
mask, the numpy round-trip of the grid and the skimage/open3d isosurface
sampling all stay as they are in the older forks, because removing them *is*
the step-3 change under test.

Run via ``bench_vns.py setup``; every replacement below is asserted to match
exactly once, so a silent no-op is impossible.
"""

from __future__ import annotations

from pathlib import Path

SCHEDULE_ARGS = """    parser.add_argument('--morse_near', action='store_true')

    # --- staged loss schedule (backported from the losses_torch branch) ------
    # "after N" == inactive while iter <= N, active from N+1. The defaults are
    # the constants this fork hardcodes (200 in models/losses.py, 100 in
    # recon_dataset.py), so leaving them alone reproduces this fork exactly.
    parser.add_argument('--iso_after', type=int, default=200,
                        help='length of the initialization stage: until this many '
                             'iterations have passed the zero-isosurface sample set S '
                             'is not extracted, and the Poisson solve and the Eikonal '
                             'term use the stroke points Gamma alone')
    parser.add_argument('--smooth_after', type=int, default=None,
                        help='iteration after which L_smooth enters the loss; defaults '
                             'to iso_after (they start together, as in the paper) and '
                             'is clamped up to it, since L_smooth needs S')
"""

SLIM_OUTPUT_ARG = """    parser.add_argument('--output_any', action='store_true')
    parser.add_argument('--slim_output', action='store_true',
                        help='only write meshes: skip model checkpoints, grid values, '
                             'ras_p, samplings and source-file backups')
"""

# (description, old, new) applied to every fork that gets the backport
COMMON: list[tuple[str, str, str, str]] = [
    (
        "surface_reconstruction/surface_recon_args.py",
        "schedule flags",
        "    parser.add_argument('--morse_near', action='store_true')\n",
        SCHEDULE_ARGS,
    ),
    (
        "models/losses.py",
        "FlowLoss signature",
        "                 div_type='l1', bidirectional_morse=True, udf=False):",
        "                 div_type='l1', bidirectional_morse=True, udf=False,\n"
        "                 iso_after=200, smooth_after=None):",
    ),
    (
        "models/losses.py",
        "FlowLoss schedule fields",
        "        self.bidirectional_morse = bidirectional_morse\n        self.udf = udf\n",
        "        self.bidirectional_morse = bidirectional_morse\n"
        "        self.udf = udf\n"
        "        # staged loss schedule; iso_after replaces the hardcoded 200 below and\n"
        "        # L_smooth can be delayed past it (never before: it needs S).\n"
        "        self.iso_after = iso_after\n"
        "        self.smooth_after = iso_after if smooth_after is None else max(smooth_after, iso_after)\n",
    ),
    (
        "models/losses.py",
        "iso stage gate",
        "        if iso_points is None or iter <= 200:",
        "        if iso_points is None or iter <= self.iso_after:",
    ),
    (
        "models/losses.py",
        "L_smooth gate",
        "        if iso_points is not None:\n"
        "            # print(\"Iso preds NaN: \", torch.isnan(iso_pred).any().item())",
        "        if iso_points is not None and iter > self.smooth_after:\n"
        "            # print(\"Iso preds NaN: \", torch.isnan(iso_pred).any().item())",
    ),
    (
        "surface_reconstruction/recon_dataset.py",
        "SuperDataset signature",
        "                 requires_dist=False, requires_curvatures=False, grid_range=2, evaluator=None):\n"
        "        self.file_path = file_path\n"
        "        self.n_points = n_points\n"
        "        self.n_samples = n_samples\n",
        "                 requires_dist=False, requires_curvatures=False, grid_range=2, evaluator=None,\n"
        "                 iso_after=200):\n"
        "        self.file_path = file_path\n"
        "        self.n_points = n_points\n"
        "        self.n_samples = n_samples\n"
        "        # matches the loss-side gate: no isosurface is extracted during the\n"
        "        # initialization stage (this fork hardcoded 100 here and 200 there)\n"
        "        self.iso_after = iso_after\n",
    ),
    (
        "surface_reconstruction/recon_dataset.py",
        "isosurface extraction gate",
        "        if self.evaluator.grid_values is None or self.iteration <= 100:",
        "        if self.evaluator.grid_values is None or self.iteration <= self.iso_after:",
    ),
    (
        "surface_reconstruction/train_surface_reconstruction.py",
        "dataset wiring",
        "train_set = dataset.SuperDataset(args.data_path, args.n_points, args.n_samples, args.grid_res, grid_range=args.grid_size/2)",
        "train_set = dataset.SuperDataset(args.data_path, args.n_points, args.n_samples, args.grid_res, grid_range=args.grid_size/2,\n"
        "                                 iso_after=args.iso_after)",
    ),
    (
        "surface_reconstruction/train_surface_reconstruction.py",
        "FlowLoss wiring",
        "                      div_type=args.morse_type, bidirectional_morse=args.bidirectional_morse, udf=args.udf)",
        "                      div_type=args.morse_type, bidirectional_morse=args.bidirectional_morse, udf=args.udf,\n"
        "                      iso_after=args.iso_after, smooth_after=args.smooth_after)",
    ),
]

# step 1 only: the slim-output plumbing from the fork's "Slim output" commit,
# minus its performance hunks (the cached DPSR module and the on-GPU ras_p),
# which are step 2's contribution and must stay out of the baseline.
SLIM: list[tuple[str, str, str, str]] = [
    (
        "surface_reconstruction/surface_recon_args.py",
        "slim_output flag",
        "    parser.add_argument('--output_any', action='store_true')\n",
        SLIM_OUTPUT_ARG,
    ),
    (
        "surface_reconstruction/train_surface_reconstruction.py",
        "source backups",
        "os.system('cp %s %s' % (__file__, logdir))  # backup the current training file\n"
        "os.system('cp %s %s' % ('recon_dataset.py', logdir))  # backup the current training file\n"
        "os.system('cp %s %s' % ('../models/overfit_network.py', logdir))  # backup the models files\n"
        "os.system('cp %s %s' % ('../models/losses.py', logdir))  # backup the losses files\n",
        "if not args.slim_output:\n"
        "    os.system('cp %s %s' % (__file__, logdir))  # backup the current training file\n"
        "    os.system('cp %s %s' % ('recon_dataset.py', logdir))  # backup the current training file\n"
        "    os.system('cp %s %s' % ('../models/overfit_network.py', logdir))  # backup the models files\n"
        "    os.system('cp %s %s' % ('../models/losses.py', logdir))  # backup the losses files\n",
    ),
    (
        "surface_reconstruction/train_surface_reconstruction.py",
        "checkpoint / grid / ras_p writes",
        "            torch.save(net.state_dict(), os.path.join(model_outdir, str(batch_idx) + '.pth'))\n"
        "\n"
        "            os.makedirs(os.path.join(logdir, 'grid_values'), exist_ok=True)\n"
        "            evaluator.save_grid_values(path=os.path.join(logdir, 'grid_values', str(batch_idx) + '.npy'))\n"
        "            evaluator.plot_grid_values(path=os.path.join(logdir, 'mesh'), batch_id=batch_idx, show=False)\n"
        "            os.makedirs(os.path.join(logdir, 'ras_p'), exist_ok=True)\n"
        "            evaluator.save_ras_p(path=os.path.join(logdir, 'ras_p', str(batch_idx) + '.npy'))\n",
        "            if not args.slim_output:\n"
        "                torch.save(net.state_dict(), os.path.join(model_outdir, str(batch_idx) + '.pth'))\n"
        "\n"
        "                os.makedirs(os.path.join(logdir, 'grid_values'), exist_ok=True)\n"
        "                evaluator.save_grid_values(path=os.path.join(logdir, 'grid_values', str(batch_idx) + '.npy'))\n"
        "            evaluator.plot_grid_values(path=os.path.join(logdir, 'mesh'), batch_id=batch_idx, show=False)\n"
        "            if not args.slim_output:\n"
        "                os.makedirs(os.path.join(logdir, 'ras_p'), exist_ok=True)\n"
        "                evaluator.save_ras_p(path=os.path.join(logdir, 'ras_p', str(batch_idx) + '.npy'))\n",
    ),
    (
        "surface_reconstruction/train_surface_reconstruction.py",
        "sampling dumps",
        "        if (batch_idx % 100 == 0 or batch_idx == len(train_dataloader) - 1):\n"
        "\n"
        "            os.makedirs(os.path.join(logdir, 'samplings'), exist_ok=True)",
        "        if (batch_idx % 100 == 0 or batch_idx == len(train_dataloader) - 1) and not args.slim_output:\n"
        "\n"
        "            os.makedirs(os.path.join(logdir, 'samplings'), exist_ok=True)",
    ),
]


def _apply(root: Path, edits: list[tuple[str, str, str, str]]) -> None:
    for rel, what, old, new in edits:
        path = root / rel
        text = path.read_text()
        count = text.count(old)
        if count != 1:
            raise SystemExit(
                f"{root.name}: {rel}: {what}: expected exactly 1 match, found {count}"
            )
        path.write_text(text.replace(old, new))


def backport(fork: Path, slim: bool) -> None:
    """Apply the harness backport in-place to a checked-out fork."""
    _apply(fork, SLIM + COMMON if slim else COMMON)
