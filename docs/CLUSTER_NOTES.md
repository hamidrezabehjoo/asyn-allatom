# Cluster notes: the failure modes this pipeline is hardened against

Each of these cost at least one full job on NSCC ASPIRE 2B (PBS Pro,
H200 nodes, October 2026). They are written down so they cost you none.

## 1. pbsdsh/TM tasks inherit almost nothing

Tasks spawned by `pbsdsh` receive only the `qsub -v` environment.
Variables the job script computes and exports at runtime do **not**
cross; TM additionally scrubs `PATH` and `LD_LIBRARY_PATH`.

- Symptom A: `FileNotFoundError: 'gmx'` deep in a Python subprocess,
  although the head node resolved gmx fine.
- Symptom B: exit code **127** from an existing absolute-path binary.
  127 here means the dynamic loader cannot find shared libraries — it
  is not "command not found".

**Fix (implemented):** the head node writes `.pbsdsh_task_env`
(`CONDA_BASE`, `ENV_PREFIX`, `GMX_MODULE`, `GMX_BIN` resolved from the
live `gmx`, full `PATH_SAVE`, `LD_LIBRARY_PATH_SAVE`, all knobs); the
task sources it and replays both paths before touching conda or gmx,
then hard-fails with exit 3 if `gmx` still does not resolve. Never
fail soft at the setup layer — a silent `|| true` on `module load`
cost us a 4-node job.

## 2. pbsdsh exits 0 even when every task failed, and its stdout relay
is lossy

**Fix (implemented):** per-task sentinels (`.task_<N>.ok`) counted by
the head node; per-node logs `pbsdsh_node<N>.log` on the shared
filesystem. Never diagnose from the relayed stdout.

## 3. Packed chunks produce duplicate node ordinals → replica races

PBS Pro's default placement is *pack*. With
`select=4:ncpus=48:ngpus=2`, the scheduler once placed **three chunks
on one physical host**; on this site `PBS_NODENUM` is the *host*
ordinal, so three tasks believed they were the same node and ran the
same replica block concurrently. Three racing `gmx solvate` processes
read-modify-wrote one `topol.top` and produced a molecule count of
exactly 3× the coordinate file's — grompp then aborted on the
mismatch. Diagnosis tip: when topology/coordinate atom counts differ
by an exact integer ratio, count your writers.

**Fix (implemented):** `#PBS -l place=scatter` (one chunk per physical
host) plus a per-host `mkdir .hostlock-$(hostname)` mutex in the task —
the first task on each host proceeds, duplicates exit 0. Atomic `mkdir`
is the mutex because it is the one POSIX operation that is reliably
atomic on shared filesystems.

Related: multiple concurrent jobs share the `$REPO` namespace for
`.pbsdsh_task_env`, sentinels and node logs. Run jobs own disjoint
replica ranges; node logs are per job, so expect the newest job's logs
to have overwritten older ones.

## 4. Sentinels must be grounded in evidence

Twice, a completion marker diverged from reality:

- A 1 ns smoke test left `prod.done` + a 1 ns `prod.tpr` behind; the
  2.5 µs campaign would have skipped that replica forever.
- GROMACS prints `Finished mdrun` at **every** graceful exit, including
  `-maxh` chunk-boundary exits. A completion check keyed on that string
  marks a replica done after its first chunk (~12% of target).

**Fix (implemented):** production completion = last energy-table step
in `prod.log` ≥ target steps from `state.json`. A `prod.done` that
contradicts the step count is deleted and the replica resumes. A chunk
that advances zero steps aborts loudly.

## 5. Hand-built start structures can carry force singularities

Random-coil and fully-extended chains (PeptideBuilder) repeatedly
passed geometric and *energy* screens (Epot ~ +7×10⁴ kJ/mol) yet NaN'd
on every platform, timestep and warm-up — because finite energy does
not imply finite forces. Sequence-neighbor (1–4/1–5) contacts produced
~10⁶–10⁸ kJ/mol/nm forces; L-BFGS stepped into the r⁻¹² wall.

**Fix (implemented):** start-structure inheritance from
production-proven PDBs (the field's standard practice; see
PROTOCOL.md), with a maximum-force screen in the fallback builder that
turns the failure into a named atom pair.

## 6. Miscellaneous, briefly

- `/scratch` purges at ~30 days: copy `runs/` home after every chunk;
  strip water (`gmx trjconv`) before archiving (~90% smaller).
- OpenMM CUDA on this site is broken (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`);
  OpenCL and CPU work. The builder's force screen runs on CPU, which is
  deterministic — that is a feature for reproducibility.
- Edited on Windows? `sed -i 's/\r$//' submit_asyn_aa.pbs` before qsub.
