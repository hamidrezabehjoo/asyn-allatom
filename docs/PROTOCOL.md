# Scientific protocol

## System

- **Protein:** human α-synuclein, 140 residues (sequence in
  `build_start.py: SEQ`). Net charge ≈ −9 at pH 7.
- **Force field:** a99SBdisp protein parameters with TIP4P-Disp water
  (Robustelli, Piana & Shaw, *PNAS* 115:E4758–E4766, 2018), the force
  field of the 73 µs D. E. Shaw Research α-synuclein benchmark
  trajectory. The port (`a99SBdisp.ff/`) is **not redistributed** with
  this repo. It is circulated with D. E. Shaw Research's a99SBdisp
  resources and appears in several public IDP-simulation repositories;
  the directory must contain the TIP4P-Disp water model
  (`a99SBdisp_water` entry in `watermodels.dat`,
  `a99SBdisp_water.gro`, and the matching `tip4p_disp.itp`). Place it
  next to the scripts or point `FF_DIR` at it.
- **Cell:** cubic box, 16.0 nm edge (`--box-nm`), matching the benchmark
  geometry; ≈134k waters, ≈538k atoms.
- **Ionic strength:** 150 mM NaCl, neutralized (`genion -neutral -conc
  0.15`; expect ~379 Na⁺ / ~370 Cl⁻ replacing ~749 waters).

## Simulation settings

- 2 fs timestep, constraints on H-bonds (LINCS order 4), Verlet lists,
  PME with 1.2 nm real-space cutoff. (Hydrogen-mass repartitioning at
  4 fs is documented as an option in `mdp/prod.mdp` comments but off by
  default; TIP4P-Disp's virtual M-site forbids `-update gpu`, and the
  pipeline does not request it.)
- 300 K, 1 atm (C-rescale barostat).
- Equilibration chain: energy minimization → 100 ps NVT with protein
  position restraints → 100 ps NPT restrained → 200 ps NPT free, each
  stage grompp'd from the previous stage's checkpoint.
- Production: 2.5 µs per replica (`--us`), 1 ns frame stride
  (`--frame-ps`), run as `-maxh`-bounded wallclock chunks with
  `-cpi -append` exact continuation (positions, velocities, RNG state).

## Replica design

- **Starts by inheritance.** New replicas copy an experimentally anchored
  or production-proven `start.pdb` (the adopt-check requires 140 residues
  and a maximum atomic displacement at least 2 nm inside the box).
  Seeding all replicas from one equilibrated monomer conformation and
  discarding burn-in is established practice for this system — the
  Kragelund group's α-synuclein studies seed every replica from the PDB
  **2N0A** fibril monomer conformation and discard ~1 µs burn-in;
  extended-start variants pre-collapse in implicit solvent before
  explicit-solvent production.
- **Diversity:** distinct start classes × per-replica velocity seeds
  (`gen-seed = 1000 + replica index`) × burn-in discard at analysis time.
- **Fallback builder** (`build_start.py`): PeptideBuilder chains
  collapsed in implicit solvent (GB-OBC2 default; GBn2 documented as
  unstable on extended charged starts), unconditionally pre-conditioned
  (minimize → 100 K → 200 K ramp → 300 K), then screened by **maximum
  post-minimization force** (< 10⁵ kJ/mol/nm) under the real
  amber14/GB Hamiltonian on CPU — finite energy does not imply finite
  forces, and force-singular starts NaN on every platform. On rejection
  the screen prints the offending atom and its nearest nonbonded
  neighbor.

## Completion semantics

A replica is complete only when the last energy-table step in `prod.log`
reaches `state.json: prod_steps`. Log strings (`Finished mdrun` appears
at *every* graceful exit, including `-maxh` chunk boundaries) and bare
sentinel files are never trusted; a `prod.done` that disagrees with the
step count is deleted and the replica resumes.

## References

1. Robustelli P, Piana S, Shaw DE. Developing a molecular dynamics force
   field for both folded and disordered protein states. *PNAS*
   115:E4758–E4766 (2018). — a99SBdisp + TIP4P-Disp; 73 µs α-synuclein
   benchmark.
2. Piana S, Donchev AG, Robustelli P, Shaw DE. Water dispersion
   interactions strongly influence simulated structural properties of
   disordered protein states. *J. Phys. Chem. B* 119:5113–5123 (2015).
3. Abraham MJ et al. GROMACS: high performance molecular simulations
   through multi-level parallelism. *SoftwareX* 1–2:19–25 (2015).
4. Eastman P et al. OpenMM 7: rapid development of high performance
   algorithms for molecular dynamics. *PLoS Comput. Biol.*
   13:e1005659 (2017).
5. Tien MZ, Sydykova DK, Meyer AG, Wilke CO. PeptideBuilder: a simple
   Python library to generate model peptides. *PeerJ* 1:e80 (2013).
