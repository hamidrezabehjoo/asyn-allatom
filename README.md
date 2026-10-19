# α-synuclein all-atom pipeline (DESRES a99SB-disp protocol)

Self-hosted replacement for the DESRES 73 µs α-synuclein reference:
same force field (a99SB-disp), same water (TIP4P-Disp), same 160 Å
(16 nm) cubic box, NPT 300 K / 1 atm — run as an ensemble of independent
replicas with per-replica random-coil starts (start-class control), with
checkpoint resume at every stage and one replica per GPU.

## Files

| script | role |
|---|---|
| `build_start.py` | sequence → expanded random coil (PeptideBuilder, seeded per replica) → implicit-solvent collapse (OpenMM) until the chain fits the box → `start.pdb` |
| `setup_asyn_aa.py` | `start.pdb` → pdb2gmx (a99SB-disp + TIP4P-Disp, auto-detected) → 16 nm box → solvate → neutralize + 150 mM NaCl; writes all `.mdp` files |
| `run_asyn_aa.py` | em → NVT → NPT×2 → production, with resume: stage `.done` sentinels + production runs in `--maxh` wallclock chunks, each resuming from `prod.cpt` (`-cpi -append`) |
| `launch_asyn_aa.py` | multi-GPU queue: replicas over GPUs, in-place retry on failure, nonzero exit if any replica fails |

## Prerequisites

1. GROMACS (2021+; 2024+ if you want HMR/4 fs — see comment in `prod.mdp`).
2. The `a99SBdisp.ff` GROMACS port — not pip-installable; get it from the
   DESRES resources page or a Kragelund-group IDP repository, and place it
   next to these scripts (or pass `--ff-dir`). The TIP4P-Disp water model
   is auto-detected from the port's `watermodels.dat`.
3. Python deps (start builder + analysis): `openmm PeptideBuilder biopython
   numpy mdtraj` — see the conda section below (a GPU node is recommended
   for the collapse, CPU works but is slow).

## Conda environment (install into scratch)

Create a dedicated env under `/scratch` (keeps home quota clean; adjust the
prefix to your scratch path):

```bash
# one-time: keep the package cache off the home quota too
mkdir -p /scratch/users/industry/ism/behjoo/conda-pkgs
conda config --add pkgs_dirs /scratch/users/industry/ism/behjoo/conda-pkgs

# create + activate (prefix install, not a named env)
conda create -p /scratch/users/industry/ism/behjoo/conda-envs/asyn-aa \
    -c conda-forge python=3.11 openmm mdtraj numpy biopython pip -y
conda activate /scratch/users/industry/ism/behjoo/conda-envs/asyn-aa

# start builder's one pip-only dependency
pip install PeptideBuilder
```

Optional: `conda install -p ... -c conda-forge gromacs` gives a **CPU-only**
GROMACS — handy for `--setup-only` work and smoke tests on a login node.
For GPU production use the cluster's GPU build instead:

```bash
conda activate /scratch/users/industry/ism/behjoo/conda-envs/asyn-aa
module load gromacs/2024.x-gpu     # your cluster's GPU module
which gmx                          # must point at the GPU build
gmx --version | grep -i gpu
```

Order matters: activate conda first, then `module load` (modules prepend
PATH), and confirm with `which gmx`. In a SLURM script, put those same
three lines before the `python3 launch_asyn_aa.py ...` call.

## Quick start

```bash
# 8 replicas x 2.5 us on 4 GPUs (setup + run + resume + retry):
python3 launch_asyn_aa.py --replicas 8 --gpus 0,1,2,3 --us 2.5

# PBS (ASPIRE 2A): submit_asyn_aa.pbs -- smoke first, then the campaign:
qsub -v PHASE=smoke,REPO="$PWD" submit_asyn_aa.pbs
qsub -v "PHASE=run,REPO=$PWD,REPLICAS=1-8" submit_asyn_aa.pbs

# SLURM: set --maxh just under the walltime limit; if preempted,
# re-submit the identical command -- everything resumes.
python3 launch_asyn_aa.py --replicas 8 --gpus 0,1,2,3 --maxh 23.5

# systems only (login node), run later:
python3 launch_asyn_aa.py --replicas 8 --gpus 0 --setup-only
```

Per-replica manual control: `python3 setup_asyn_aa.py --replica 3` then
`python3 run_asyn_aa.py --replica 3`.

## Resume semantics (the point of this rewrite)

- Every stage (`build/pdb2gmx/box/solvate/ions`, `em/nvt/npt1/npt2/prod`)
  is idempotent: re-running any script skips completed stages.
- Production runs in wallclock chunks; each chunk resumes exactly
  (positions, velocities, RNG state) from `prod.cpt`. An interrupted chunk
  costs at most `--maxh` hours.
- A replica crash makes the launcher retry in place (≤ `--retries`,
  default 2); persistent failures are listed at the end and the launcher
  exits nonzero, so chained shell commands stop instead of marching
  through broken runs.

## Protocol notes and honest caveats

- **Conditions**: 300 K, 1 atm, **150 mM NaCl** (chosen to match the
  Mpipi-GG CG ladder, Debye 0.795 nm — change with `--salt-M`);
  1.2 nm cutoff with LJ force-switch 1.0→1.2 nm, PME, v-rescale +
  C-rescale, dt = 2 fs, constraints h-bonds. These mirror published
  a99SB-disp GROMACS protocols (e.g. Greener 2024); DESRES's Anton
  internals are not public, so "same protocol" means same force field,
  water, box, and ensemble — not bit-identical numerics.
- **Seeds**: `seed = 1000 + replica_index`. No string-length formulas.
- **Starts**: each replica gets an independent seeded random coil,
  collapsed in implicit solvent (amber14 + GBn2). The collapse force
  field only shapes the start; explicit-solvent equilibration and the
  first ns wash it out. Distinct seeds = distinct start classes.
- **Termini/protonation**: NH3+/COO− termini; H50 protonation is chosen
  by pdb2gmx (use `pdb2gmx -his` interactively if you want a specific
  tautomer). For physiological N-acetylation, cap the start structure
  before setup.
- **Ensemble, not kinetics**: N × 2.5 µs replicas give ensemble coverage
  and replica-spread convergence diagnostics; they do not reproduce the
  single-chain kinetic continuity of the DESRES 73 µs run.
- **Cost**: ~400k atoms in a 16 nm box; expect ~100–200 ns/day/GPU
  (A100/H100, GPU-resident update). 8 × 2.5 µs = 20 µs ≈ 100–200
  GPU-days, i.e. roughly 2–4 weeks on 8 GPUs. Verify throughput with a
  short chunk before committing.

## CPU-only mode

`run_asyn_aa.py --cpu` drops all GPU offload flags; `--mpi-launcher "srun"`
routes mdrun through MPI with `gmx_mpi` for multi-node runs. Expect
~20–30 ns/day per modern 32–64-core node at ~400k atoms — a 2.5 µs replica
is 3–4 node-months. Practical CPU-only scope: a few shorter replicas
(e.g. 4 × 1 µs, `--us 1`) spread over nodes with a SLURM array, not one
long run. The start-structure collapse (`build_start.py`) already supports
`--platform CPU` (slow but one-time; OpenMM CPU scales with cores).

## Output (per replica, `runs/repNN/`)

`prod.xtc` (frames every `--frame-ps`, default 1 ns), `prod.log`,
`prod.cpt`, `progress.json`, `state.json`, plus all setup artifacts.
Analysis: strip water and load with mdtraj (`md.load('prod.xtc',
top='start.pdb')`); the start PDB is mdtraj-parseable by construction
(verified).
