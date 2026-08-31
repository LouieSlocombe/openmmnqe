# Enhanced sampling with PLUMED

Every production stage takes a `plumed_script_path` argument. Point it at a
PLUMED input file and the stage attaches the bias it describes; leave it out and
the stage runs unbiased. That one argument is the whole integration surface.

The collective variables themselves are **not** built here. Writing PLUMED
inputs, turning a steered trajectory into a reference path, and plotting the
free-energy surface that comes out all live in
[reactiontools](https://github.com/LouieSlocombe/reactiontools). Its builders
take the `openmm.app.Modeller` and `openmm.unit.Quantity` objects this package
already works in, so they can be called with what is to hand.

## Metadynamics and OPES

```{literalinclude} ../../examples/opes.py
:pyobject: main
:language: python
```

## Multiple independent walkers

{func}`~openmmnqe.walkers.run_openmm_walkers` runs N copies of a production
stage as parallel processes — one working directory, one derived seed, and one
copy of the PLUMED script per walker, so the CWD-relative outputs and PLUMED's
`COLVAR`, `STATE` and `KERNELS` land per walker instead of colliding.

The walkers are independent rather than bias-sharing, and deliberately so:
`OPES_METAD` shares bias between walkers only through `WALKERS_MPI`, which
needs an MPI-enabled PLUMED this project's build does not have — and OPES has
no file-based walker scheme the way `METAD` does. Independent runs are still
combined into one estimate at analysis time, because every `COLVAR` sample
carries the bias its own walker collected it under:
`reactiontools.combine_colvar_files` merges the files and
`reactiontools.run_opes_reweighting` reweights the merged samples into a
surface, with `blocks` giving error bars from the cross-walker scatter.

```{literalinclude} ../../examples/walkers.py
:pyobject: main
:language: python
```

Each walker's stdout and stderr go to `screen.out` in its directory, and a
completed walker leaves a `walker.json` record. `resume=True` skips recorded
walkers and reruns the rest, so a failed or interrupted launch continues from
where it stopped — and it refuses a directory recorded under a different
master seed or stage, which would silently mix ensembles. Runs on a single
GPU lose little to `max_workers=1`, since concurrent contexts time-slice the
card anyway; on the CPU platform, `cpu_threads_per_walker` divides the
machine instead of oversubscribing it.

The RPMD production stages orchestrate identically — each walker executes the
whole driver, so the centroid-bias contraction described below happens per
walker unchanged. Ship the restart archive into every directory with
`shared_inputs=["rpmd_ready.chk"]`.

## Steered MD and path collective variables

{func}`~openmmnqe.openmm.run_openmm_steered` sits outside the normal stage
sequence. It drags a collective variable from one value to another, and the
trajectory that comes out is the raw material for a reference path:

```{literalinclude} ../../examples/steered_path.py
:pyobject: main
:language: python
```

The path is returned, so it does not matter which format wrote it — but a
pulling run is short and frequently reported, which is exactly the case PDB
handles well, and it is the only format
{func}`reactiontools.path_from_steered_md` reads unaided. Pass
`trajectory='dcd'` only if the run is long enough to need it, and then hand
`<prefix>_topology.pdb` to that function as `top=`.

A `PATHMSD` collective variable aligns each frame against a reference structure
containing only the atoms that define the path.
{func}`~openmmnqe.io.save_only_index_atoms` writes exactly that file — the
`index_atoms.pdb` the CV aligns against — from a `Modeller`.

Once the path exists, biasing along it is an ordinary production run with a
different PLUMED script:

```{literalinclude} ../../examples/workflows.py
:pyobject: run_malonaldehyde_pathmsd
:language: python
```

## Combining with nuclear quantum effects

The RPMD production stages take `plumed_script_path` like any other, so a biased
ring-polymer run needs nothing special. This is usually the point of the
exercise: a proton transfer barrier that is too high to cross on its own, in a
system where the proton's zero-point energy is precisely what you are trying to
measure.

The bias acts on the **centroid**, and it takes work to make that true.
`RPMDIntegrator` evaluates a force group on every bead unless its contractions
map says otherwise, so a bias merely added to the System is applied once per
bead, at each bead's own coordinates -- which is not the centroid potential of
mean force, and which hands PLUMED, a stateful engine, `n_beads` coordinate
sets per step. Both production stages therefore give the bias a force group of
its own and contract that group to a single copy, so it is evaluated on the
contracted position and the force transformed back onto every bead.

Pass `centroid_bias=False` only to reproduce a run made before that was so; it
warns, because the surface it produces is not a centroid free energy.

See [](reporters.md) for reading the bead spread back out along the biased
coordinate.
