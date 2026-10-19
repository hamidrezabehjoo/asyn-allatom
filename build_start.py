#!/usr/bin/env python3
"""
build_start.py -- build a compact, box-fitting alpha-synuclein starting
structure for all-atom explicit-solvent MD (DESRES-protocol a99SB-disp).

Strategy (standard for long IDPs; cf. Greener 2024, PMC10966991):
  1. Build an expanded random coil from sequence with ideal peptide
     geometry (PeptideBuilder), phi/psi sampled from extended-beta /
     PPII / alpha Ramachandran basins, per-replica seeded.
  2. Add the C-terminal OXT oxygen (needed for C-terminal templates).
  3. Collapse in implicit solvent (OpenMM, amber14-all + GBn2 by
     default) in chunks until the chain fits the target box
     (max CA-CA distance < fit_nm) -- a fully extended 140-mer spans
     ~30-50 nm and would crash through its own periodic image in the
     DESRES 160 A (16 nm) box.

The collapsed conformer is only a *starting structure*: explicit-solvent
equilibration and the first ns of production erase memory of the
collapse force field. Different seeds give different start classes --
use one seed per replica (this is your start-class control).

Usage:
  python3 build_start.py --seed 101 --out start.pdb
  python3 build_start.py --seed 102 --out start.pdb --box-nm 16 --max-collapse-ns 2

Deps: pip install PeptideBuilder biopython numpy openmm
"""
import argparse, io, os, sys

import numpy as np
from PeptideBuilder import Geometry, PeptideBuilder
import Bio.PDB

# Human alpha-synuclein, 140 aa (UniProt P37840)
ASYN_SEQ = ("MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGG"
            "AVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA")

# Ramachandran basins for the expanded coil: (phi range, psi range, weight)
BASINS = [((-160.0, -100.0), (100.0, 170.0), 0.65),   # extended beta
          ((-95.0, -55.0),   (120.0, 170.0), 0.20),   # PPII
          ((-80.0, -50.0),   (-55.0, -25.0), 0.15)]   # alpha


def _sample_phipsi(rng):
    u = rng.random(); c = 0.0
    for (plo, phi), (pslo, pshi), w in BASINS:
        c += w
        if u <= c:
            return rng.uniform(plo, phi), rng.uniform(pslo, pshi)
    return -120.0, 130.0


def build_coil(seq, seed):
    """Expanded random-coil chain with ideal peptide geometry (heavy atoms)."""
    rng = np.random.default_rng(seed)
    phi, psi = _sample_phipsi(rng)
    g = Geometry.geometry(seq[0]); g.phi, g.psi, g.omega = phi, psi, 180.0
    st = PeptideBuilder.initialize_res(g)
    for aa in seq[1:]:
        phi, psi = _sample_phipsi(rng)
        g = Geometry.geometry(aa); g.phi, g.psi, g.omega = phi, psi, 180.0
        PeptideBuilder.add_residue(st, g)
    return st


def _pdb_text(structure):
    out = Bio.PDB.PDBIO(); out.set_structure(structure)
    buf = io.StringIO(); out.save(buf)
    return buf.getvalue()


def add_oxt(pdb_txt):
    """Add the C-terminal OXT oxygen by reflecting C->O across the C->CA axis
    (standard planar carboxyl geometry). Without it, template matching fails
    on the last residue in both OpenMM and pdb2gmx."""
    lines = pdb_txt.splitlines()
    atoms = [(i, l) for i, l in enumerate(lines) if l.startswith("ATOM")]
    last_res = atoms[-1][1][22:26]
    ra = [(i, l) for i, l in atoms if l[22:26] == last_res]
    pos = {}
    for i, l in ra:
        pos[l[12:16].strip()] = (i, np.array([float(l[30:38]),
                                              float(l[38:46]),
                                              float(l[46:54])]))
    a = pos['CA'][1] - pos['C'][1]
    b = pos['O'][1] - pos['C'][1]
    a_hat = a / np.linalg.norm(a)
    d = b / np.linalg.norm(b)
    refl = 2.0 * np.dot(d, a_hat) * a_hat - d
    oxt = pos['C'][1] + 1.25 * refl / np.linalg.norm(refl)
    serial = int(atoms[-1][1][6:11]) + 1
    iC, _ = pos['C']
    lC = lines[iC]
    oxt_line = (f"ATOM  {serial:5d}  OXT {lC[17:27]}   "
                f"{oxt[0]:8.3f}{oxt[1]:8.3f}{oxt[2]:8.3f}  1.00  0.00           O  ")
    out = []
    for i, l in enumerate(lines):
        out.append(l)
        if i == pos['CB'][0]:
            out.append(oxt_line)
    return "\n".join(out) + "\n"


def ca_geometry(pdb_path):
    """Return (Rg nm, max CA-CA distance nm, n_CA)."""
    xyz = []
    with open(pdb_path) as fh:
        for l in fh:
            if l.startswith("ATOM") and l[12:16].strip() == "CA":
                xyz.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
    xyz = np.array(xyz) / 10.0
    rg = float(np.sqrt(((xyz - xyz.mean(0)) ** 2).sum(1).mean()))
    # pairwise max without scipy
    d2max = 0.0
    for i in range(len(xyz)):
        d2 = ((xyz[i + 1:] - xyz[i]) ** 2).sum(1)
        if len(d2):
            d2max = max(d2max, float(d2.max()))
    return rg, float(np.sqrt(d2max)), len(xyz)


def collapse(pdb_in, pdb_out, seed, chunk_ps, gb_model, platform_name):
    """Run one implicit-solvent collapse chunk; returns None.

    Platform handling is fail-soft: 'auto' tries CUDA -> OpenCL -> CPU, but a
    platform is only accepted after a real Context has been built AND a chunk
    step has run -- a registered-but-broken plugin (e.g. conda-forge OpenMM
    whose CUDA kernels are newer than the node driver ->
    CUDA_ERROR_UNSUPPORTED_PTX_VERSION) falls through to the next candidate
    instead of killing the setup."""
    from openmm import app, unit, LangevinMiddleIntegrator, Platform
    candidates = (['CUDA', 'OpenCL', 'CPU'] if platform_name == 'auto'
                  else [platform_name])
    pdb = app.PDBFile(pdb_in)
    ff = app.ForceField('amber14-all.xml', f'implicit/{gb_model}.xml')
    mod = app.Modeller(pdb.topology, pdb.positions)
    if not getattr(collapse, '_hydrated', False):
        mod.addHydrogens(ff, pH=7.0)
        collapse._hydrated = True
    system = ff.createSystem(mod.topology, nonbondedMethod=app.NoCutoff,
                             constraints=app.HBonds)
    errs = []
    for cand in candidates:
        try:
            platform = Platform.getPlatformByName(cand)
        except Exception as e:
            errs.append(f'{cand}: not registered ({e})')
            continue
        try:
            integ = LangevinMiddleIntegrator(300 * unit.kelvin,
                                             1.0 / unit.picosecond,
                                             2.0 * unit.femtoseconds)
            integ.setRandomNumberSeed(seed)
            sim = app.Simulation(mod.topology, system, integ, platform)
            sim.context.setPositions(mod.positions)
            sim.context.setVelocitiesToTemperature(300 * unit.kelvin, seed)
            sim.minimizeEnergy(maxIterations=500)
            sim.step(int(chunk_ps * 1000 / 2.0))
        except Exception as e:
            errs.append(f'{cand}: {type(e).__name__} '
                        f'{str(e).strip().splitlines()[-1] if str(e).strip() else e}')
            print(f'[build_start] platform {cand} unusable '
                  f'({errs[-1]}); trying next')
            continue
        st = sim.context.getState(getPositions=True)
        with open(pdb_out, 'w') as f:
            app.PDBFile.writeFile(mod.topology, st.getPositions(), f)
        if cand != candidates[0] or len(candidates) > 1:
            print(f'[build_start] collapse running on platform: {cand}')
        return
    sys.exit('[build_start] no usable OpenMM platform: ' + ' | '.join(errs))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, required=True,
                    help='per-replica seed (use the replica index)')
    ap.add_argument('--out', required=True, help='output PDB (compact start)')
    ap.add_argument('--box-nm', type=float, default=16.0,
                    help='target cubic box edge (DESRES protocol: 16 nm = 160 A)')
    ap.add_argument('--fit-margin-nm', type=float, default=2.0,
                    help='required clearance: max CA-CA must be < box - margin')
    ap.add_argument('--chunk-ps', type=float, default=200.0,
                    help='implicit-solvent collapse chunk between geometry checks')
    ap.add_argument('--max-collapse-ns', type=float, default=2.0,
                    help='give up if not collapsed after this much implicit MD')
    ap.add_argument('--gb-model', default='gbn2',
                    choices=['gbn2', 'gbn', 'obc2', 'obc1', 'hct'],
                    help='implicit model for the collapse (start only; gbn2 = GBNeck-family)')
    ap.add_argument('--platform', default='auto',
                    choices=['auto', 'CUDA', 'OpenCL', 'CPU'],
                    help='OpenMM platform for the collapse (default auto: '
                         'CUDA -> OpenCL -> CPU)')
    ap.add_argument('--keep-coil', action='store_true',
                    help='also write the pre-collapse expanded coil to <out>.coil.pdb')
    args = ap.parse_args()

    fit_nm = args.box_nm - args.fit_margin_nm
    coil_txt = add_oxt(_pdb_text(build_coil(ASYN_SEQ, args.seed)))
    coil_path = args.out + '.coil.pdb'
    with open(coil_path, 'w') as f:
        f.write(coil_txt)
    rg, dmax, n = ca_geometry(coil_path)
    print(f'[build_start] expanded coil: {n} residues, Rg={rg:.2f} nm, '
          f'maxDist={dmax:.2f} nm (target fit < {fit_nm:.2f} nm)')

    done = dmax < fit_nm
    elapsed_ps = 0.0
    cur = coil_path
    while not done and elapsed_ps < args.max_collapse_ns * 1000.0:
        collapse(cur, args.out, seed=args.seed + int(elapsed_ps),
                 chunk_ps=args.chunk_ps, gb_model=args.gb_model,
                 platform_name=args.platform)
        elapsed_ps += args.chunk_ps
        rg, dmax, n = ca_geometry(args.out)
        print(f'[build_start] collapse {elapsed_ps:7.0f} ps: Rg={rg:.2f} nm, '
              f'maxDist={dmax:.2f} nm')
        done = dmax < fit_nm
        cur = args.out

    if not done:
        sys.exit(f'[build_start] chain still spans {dmax:.1f} nm after '
                 f'{args.max_collapse_ns} ns of collapse; increase '
                 f'--max-collapse-ns or use a larger --box-nm.')

    if not args.keep_coil and os.path.exists(coil_path):
        os.remove(coil_path)
    rg, dmax, n = ca_geometry(args.out)
    print(f'[build_start] wrote {args.out}: {n} residues, Rg={rg:.2f} nm, '
          f'maxDist={dmax:.2f} nm -- fits {args.box_nm:.0f} nm box.')


if __name__ == '__main__':
    main()
