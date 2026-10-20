# asyn-allatom

Checkpoint-resumable, multi-node all-atom MD campaigns of α-synuclein
(a99SBdisp / TIP4P-Disp, explicit solvent) on PBS Pro GPU clusters.

Built for long IDP campaigns where the scheduler *will* kill you mid-run,
shared filesystems *will* delay metadata, and a multi-node fan-out *will*
find ways to race itself. Every stage is idempotent, every completion
sentinel is grounded in evidence (step counts, not log strings), and every
failure is loud.

Default campaign: **8 replicas × 2.5 µs** of human α-synuclein (140 aa) in a
16 nm cubic box, 150 mM NaCl, 300 K, 1 atm — the design used for rare-event
studies of IDP conformational ensembles. Tunable to any intrinsically
disordered protein by swapping the sequence and the start structures.

## Features

- **Stage-level resume.** `build → pdb2gmx → box → solvate → ions → em →
  nvt → npt1 → npt2 → prod`, each stage checkpointed in `state.json` plus
  `<stage>.done` sentinels. Re-run any command after a crash or walltime
  kill and it continues where it stopped.
- **Chunked production with exact resume.** `mdrun -cpi prod.cpt -append
  -maxh H` in a loop until the target step count is reached; an interrupted
  chunk costs at most the checkpoint interval.
- **Evidence-grounded completion.** A replica is "done" only when the last
  step in `prod.log` reaches the target — never because a log string or a
  stale sentinel says so. (A graceful `-maxh` exit also prints
  `Finished mdrun`; trusting that string silently truncates campaigns.)
- **Multi-node fan-out without ssh.** One launcher per allocated node via
  `pbsdsh`, replica blocks split contiguously, one replica per GPU.
- **Cluster-proof plumbing.** Environment handoff by file (TM scrubs
  `PATH`/`LD_LIBRARY_PATH`), per-node logs (pbsdsh stdout relay is lossy),
  per-task sentinels (pbsdsh exits 0 even when every task failed),
  `place=scatter` + per-host lockfiles (packed chunks get duplicate node
  ordinals and will race your topology files).
- **Fail loud, retry thin.** No silent fallbacks at the setup layer; a
  production chunk that advances zero steps aborts instead of spinning.

## Requirements

- GROMACS ≥ 2021 (developed on 2026.2-gpu), GPU build recommended
- PBS Pro (or adapt `submit_asyn_aa.pbs`; the Python layer is
  scheduler-agnostic)
- Python 3.11 with OpenMM + PeptideBuilder — **only** needed for the
  `build_start.py` fallback chain builder; skipped entirely when replicas
  inherit start structures (the recommended route)
- The a99SBdisp force-field port (`a99SBdisp.ff/`) — **not redistributed
  here**; see `docs/PROTOCOL.md` for where to get it and what it must
  contain

## Quickstart

```bash
git clone <this-repo> && cd asyn-allatom
md5sum -c md5sums.txt                        # intact checkout: every line OK
conda env create -f environment.yml          # python 3.11 + openmm + PeptideBuilder

# site setup: your allocation code and the force field
sed -i 's/#PBS -P CHANGE_ME/#PBS -P <your-allocation>/' submit_asyn_aa.pbs
# (that edit makes the pbs line of md5sum -c report FAILED from now on --
#  expected; the deployed copy is supposed to differ from the committed one)
cp -r /path/to/a99SBdisp.ff .                # see docs/PROTOCOL.md

# 1. smoke test FIRST: 1 replica x 1 ns, short/priority queue
qsub -l walltime=02:00:00 -v PHASE=smoke,REPO=$PWD submit_asyn_aa.pbs

# 2. build all systems (topology only, no MD)
qsub -v PHASE=setup,REPLICAS=1-8,US=2.5,REPO=$PWD submit_asyn_aa.pbs

# 3. production: disjoint replica blocks per job, never share a replica
qsub -l select=2:ncpus=48:ngpus=2 -l walltime=120:00:00 \
     -v PHASE=run,REPLICAS=1-4,US=2.5,MAXH=119,REPO=$PWD submit_asyn_aa.pbs
qsub -l select=2:ncpus=48:ngpus=2 -l walltime=120:00:00 \
     -v PHASE=run,REPLICAS=5-8,US=2.5,MAXH=119,REPO=$PWD submit_asyn_aa.pbs

# 4. when a job hits its wall: resubmit the IDENTICAL command.
#    Every replica resumes from prod.cpt. Repeat until all prod.done exist.
```

Knobs (all via `qsub -v`): `REPLICAS` (e.g. `4` or `1-8`), `US` (µs per
replica), `MAXH` (chunk hours, just under walltime), `RETRIES`, `NTOMP`,
`FF_DIR`, `GMX_MODULE`, `ENV_PREFIX`, `SCRATCH`, `REPO`.

> **Cardinal rule:** never let two jobs share a replica number. Same run
> directory + same `prod.cpt` + two appending mdruns = corrupted checkpoint.

## Start structures: inherit, don't build

Hand-built polypeptide chains (random coils *and* fully extended) can carry
local force singularities — finite energy, but ~10⁶–10⁸ kJ/mol/nm forces
from sequence-neighbor contacts that minimizers cannot resolve — and no
timestep or warm-up survives them. The field's answer is inheritance:
start replicas from an experimentally anchored or already-equilibrated
conformation and discard burn-in.

```bash
# seed a new replica from a production-proven start (adopt-check: 140
# residues, fits the box with 2 nm margin) -- setup then skips the builder
mkdir -p runs/rep05 && cp runs/rep01/start.pdb runs/rep05/start.pdb
```

Diversity comes from distinct start classes × per-replica velocity seeds
(`seed = 1000 + replica index`) × a burn-in discard (e.g. 0.5 µs) at
analysis time. `build_start.py` remains as a fallback: it screens
candidates by post-minimization maximum force under the real Hamiltonian
and prints the offending atom pair on rejection.

## Repository layout

```
submit_asyn_aa.pbs   PBS job: preflight, env capture, pbsdsh fan-out
pbsdsh_task.sh       per-node task: env replay, hostlock, launch
launch_asyn_aa.py    node-local orchestrator: setup queue, GPU->replica map, retries
setup_asyn_aa.py     build one replica: FF, box, solvent, ions, mdp files, state.json
run_asyn_aa.py       run one replica: equilibration chain, chunked production
build_start.py       fallback chain builder (OpenMM collapse + force screen)
md5sums.txt          integrity check for cluster copies: md5sum -c md5sums.txt
docs/PROTOCOL.md     the scientific protocol and its literature grounding
docs/CLUSTER_NOTES.md the failure modes this pipeline is hardened against
environment.yml      conda environment (builder fallback only)
```

Per replica, everything lives in `runs/repNN/`: topology (`topol.top`),
coordinates, `mdp/`, `state.json`, stage sentinels, `prod.{tpr,cpt,xtc,log,
gro}`, `run.log` (every command + full output), `progress.json`.

## Performance reference

Measured on NVIDIA H200 (141 GB), GROMACS 2026.2, 537,716 atoms
(16 nm box, TIP4P-Disp), 2 fs, `-nb/-pme/-bonded gpu`, `-ntomp 24`:
**~64 ns/day/replica/GPU**. At that rate one 2.5 µs replica ≈ 8 × 119 h
wallclock chunks ≈ 39 days of cadence on one GPU; 8 replicas on 8 GPUs
finish together in that span.

## Data hygiene

- `prod.xtc` contains **full solvent** (~3.7 MB/frame at 1 ns stride).
  Strip water with `gmx trjconv` before archiving (~90% smaller).
- Scratch filesystems purge (ASPIRE 2B: ~30 days). Copy `runs/` home after
  every chunk.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| grompp/mdrun exit 127 in pbsdsh tasks | TM scrubbed `LD_LIBRARY_PATH` — the env file replays it; check `pbsdsh_node<N>.log` |
| `[task] FATAL: gmx not found` | module system unreachable from task; `GMX_BIN` fallback failed |
| topol.top atom count ≫ coordinate file | two writers on one replica (packed chunks or a shared replica number) |
| replica "done" suspiciously fast | stale `prod.done` — current `run_asyn_aa.py` auto-removes it |
| `build_start` NaN on every platform | force-singular start; use inheritance instead |

Deep dive on each: `docs/CLUSTER_NOTES.md`.

## Citing

If you use this pipeline, cite the force field and engines (see
`docs/PROTOCOL.md` for full references): Robustelli, Piana & Shaw, *PNAS*
115:E4758 (2018); Abraham et al., *SoftwareX* 1–2:19 (2015); Eastman et
al., *PLoS Comput. Biol.* 13:e1005659 (2017).

## License & author

MIT (see LICENSE). Developed by Hamidreza Behjoo, Institute of Systems
Medicine, CAMS Suzhou, on NSCC ASPIRE 2B.
