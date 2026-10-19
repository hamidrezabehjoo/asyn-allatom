#!/usr/bin/env python3
"""
setup_asyn_aa.py -- build the GROMACS system for one alpha-synuclein
all-atom replica following the DESRES a99SB-disp protocol
(Robustelli, Piana & Shaw 2018; 160 A box, as used for the 73 us
reference trajectory).

Protocol encoded here:
  * force field: a99SB-disp (GROMACS port -- obtain the a99SBdisp.ff
    directory from the DESRES resources page or a Kragelund-group IDP
    repository; place it next to this script or pass --ff-dir)
  * water: the a99SB-disp TIP4P-Disp variant shipped inside the port
    (auto-detected from watermodels.dat; override with --water)
  * box: cubic 16.0 nm
  * ions: neutralize + 150 mM NaCl  (matches the Mpipi-GG CG ladder,
    Debye length 0.795 nm; change with --salt-M)
  * 300 K / 1 atm, PME, 1.2 nm cutoff with LJ force-switch 1.0->1.2 nm
    (mirrors published a99SB-disp GROMACS protocols, e.g. Greener 2024)
  * dt = 2 fs, constraints h-bonds  (HMR/4 fs: see prod.mdp comment)

Layout created per replica:
  runs/repNN/{start.pdb, conf.gro, topol.top, posre.itp, boxed.gro,
              solv.gro, ions.gro, mdp/{ions,minim,nvt,npt1,npt2,prod}.mdp,
              state.json}

Stages (each idempotent -- safe to re-run; completed stages are skipped):
  build    (build_start.py)   -> start.pdb
  pdb2gmx                     -> conf.gro topol.top posre.itp
  box                         -> boxed.gro   (editconf -bt cubic -box 16)
  solvate                     -> solv.gro
  ions                        -> ions.gro    (genion -neutral -conc 0.15)

Usage:
  python3 setup_asyn_aa.py --replica 1
  python3 setup_asyn_aa.py --replica 1 --dry-run     # print commands only
"""
import argparse, json, os, re, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# mdp templates (GROMACS 2021+ syntax)
# --------------------------------------------------------------------------
def _mdp_common(T=300.0):
    return f"""\
; --- common (a99SB-disp explicit-solvent protocol) ---
integrator              = md
dt                      = 0.002          ; 2 fs (HMR/4 fs: see prod.mdp)
constraints             = h-bonds
constraint-algorithm    = lincs
lincs-order             = 4
cutoff-scheme           = Verlet
coulombtype             = PME
rcoulomb                = 1.2
fourierspacing          = 0.12
pme-order               = 4
vdwtype                 = cutoff
vdw-modifier            = force-switch
rvdw-switch             = 1.0
rvdw                    = 1.2
DispCorr                = no
pbc                     = xyz
tcoupl                  = v-rescale
tc-grps                 = Protein Water_and_ions
tau-t                   = 1.0 1.0          ; one value PER tc-grp (grompp errors on count mismatch)
ref-t                   = {T} {T}
comm-mode               = Linear
comm-grps               = Protein Water_and_ions
nstcomm                 = 100
"""

MDP_IONS = """\
integrator = steep
nsteps     = 0
cutoff-scheme = Verlet
coulombtype   = PME
rcoulomb      = 1.2
rvdw          = 1.2
pbc           = xyz
"""

MDP_MINIM = """\
integrator  = steep
emtol       = 500.0
emstep      = 0.01
nsteps      = 50000
cutoff-scheme = Verlet
coulombtype   = PME
rcoulomb      = 1.2
rvdw-switch   = 1.0
rvdw          = 1.2
vdwtype       = cutoff
vdw-modifier  = force-switch
pbc           = xyz
"""

def mdp_nvt():
    return _mdp_common() + """\
; --- NVT equilibration, 100 ps, position restraints (POSRES) ---
define                  = -DPOSRES
nsteps                  = 50000
gen_vel                 = yes
gen-temp                = 300
gen-seed                = {gen_seed}
pcoupl                  = no
nstxout-compressed      = 5000
nstenergy               = 2500
"""

def mdp_npt(restrained, gen_seed):
    posres = "define                  = -DPOSRES" if restrained else \
             "; no position restraints"
    nsteps = 50000 if restrained else 100000
    return _mdp_common() + f"""\
; --- NPT equilibration, {'100 ps restrained' if restrained else '200 ps free'} ---
{posres}
nsteps                  = {nsteps}
gen_vel                 = no
pcoupl                  = C-rescale
ref-p                   = 1.0
tau-p                   = 5.0
compressibility         = 4.5e-5
nstxout-compressed      = 5000
nstenergy               = 2500
"""

def mdp_prod(total_steps, frame_steps, gen_seed):
    return _mdp_common() + f"""\
; --- production ---
; For 4 fs timesteps on GROMACS 2024+, generate the tpr with hydrogen
; mass repartitioning (grompp does HMR automatically when the mdp sets
;   dt = 0.004
; and the version supports it); keep 2 fs otherwise. DESRES used the
; equivalent a99SB-disp settings on Anton at ~2.5 fs.
nsteps                  = {total_steps}
gen_vel                 = no
gen-seed                = {gen_seed}
pcoupl                  = C-rescale
ref-p                   = 1.0
tau-p                   = 5.0
compressibility         = 4.5e-5
nstxout-compressed      = {frame_steps}   ; trajectory frame interval
nstenergy               = 2500            ; 5 ps
nstcheckpoint           = 250000          ; state.cpt every 0.5 ns
"""


# --------------------------------------------------------------------------
def sh(cmd, cwd, dry, log, env=None):
    """Run a command (list of str), echo it, abort on failure."""
    log.write('$ ' + ' '.join(cmd) + '\n'); log.flush()
    print('$ ' + ' '.join(cmd))
    if dry:
        return
    r = subprocess.run(cmd, cwd=cwd, env=env, stdout=log,
                       stderr=subprocess.STDOUT)
    if r.returncode != 0:
        sys.exit(f'[setup] command failed ({r.returncode}): {" ".join(cmd)}\n'
                 f'        see {log.name}')


def _ca_span_nm(pdb_path):
    """(Rg nm, max CA-CA distance nm, n_CA) -- stdlib only, so the
    adopt-existing-start check cannot die on missing heavy deps."""
    import math
    xyz = []
    with open(pdb_path) as fh:
        for ln in fh:
            if ln.startswith('ATOM') and ln[12:16].strip() == 'CA':
                xyz.append((float(ln[30:38]), float(ln[38:46]),
                            float(ln[46:54])))
    n = len(xyz)
    cx = sum(p[0] for p in xyz) / n
    cy = sum(p[1] for p in xyz) / n
    cz = sum(p[2] for p in xyz) / n
    rg2 = sum((p[0] - cx) ** 2 + (p[1] - cy) ** 2 + (p[2] - cz) ** 2
              for p in xyz) / n
    d2max = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d2 = ((xyz[i][0] - xyz[j][0]) ** 2 + (xyz[i][1] - xyz[j][1]) ** 2
                  + (xyz[i][2] - xyz[j][2]) ** 2)
            if d2 > d2max:
                d2max = d2
    return math.sqrt(rg2) / 10.0, math.sqrt(d2max) / 10.0, n


def find_ff(ff_dir):
    """Locate a99SBdisp.ff and auto-detect its TIP4P-Disp water keyword."""
    if not os.path.isdir(ff_dir):
        sys.exit(f'[setup] force-field directory not found: {ff_dir}\n'
                 f'        Download the a99SB-disp GROMACS port (DESRES resources\n'
                 f'        page / Kragelund-group IDP repos) and place a99SBdisp.ff here\n'
                 f'        or pass --ff-dir /path/to/a99SBdisp.ff')
    wm = os.path.join(ff_dir, 'watermodels.dat')
    waters = []
    if os.path.isfile(wm):
        with open(wm) as fh:
            for ln in fh:
                ln = ln.strip()
                if ln and not ln.startswith(('#', ';')):
                    waters.append(ln.split()[0])
    disp = [w for w in waters if re.search(r'tip4p.*dis|dis.*tip4p|disp', w, re.I)]
    return disp[0] if disp else (waters[0] if waters else None), waters


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--replica', type=int, required=True)
    ap.add_argument('--ff-dir', default=os.path.join(HERE, 'a99SBdisp.ff'))
    ap.add_argument('--water', default=None,
                    help='pdb2gmx water keyword (default: auto-detect TIP4P-Disp)')
    ap.add_argument('--box-nm', type=float, default=16.0)
    ap.add_argument('--salt-M', type=float, default=0.15)
    ap.add_argument('--us', type=float, default=2.5,
                    help='production length per replica (microseconds)')
    ap.add_argument('--frame-ps', type=float, default=1000.0,
                    help='trajectory frame interval (ps)')
    ap.add_argument('--runs-dir', default=os.path.join(HERE, 'runs'))
    ap.add_argument('--build-platform', default='auto',
                    choices=['auto', 'CUDA', 'OpenCL', 'CPU'],
                    help='OpenMM platform for the start-structure collapse '
                         '(GPU node recommended; auto detects)')
    ap.add_argument('--max-collapse-ns', type=float, default=8.0,
                    help='implicit-solvent collapse budget per replica; some '
                         'seeds collapse to a dumbbell whose tail needs '
                         '>2 ns to fit the box (default 8.0)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    gmx = shutil.which('gmx') or shutil.which('gmx_mpi')
    if gmx is None and not args.dry_run:
        sys.exit('[setup] gmx not found on PATH (load the gromacs module first)')
    gmx = gmx or 'gmx'

    rep = os.path.join(args.runs_dir, f'rep{args.replica:02d}')
    os.makedirs(rep, exist_ok=True)
    mdpdir = os.path.join(rep, 'mdp')
    os.makedirs(mdpdir, exist_ok=True)
    state_path = os.path.join(rep, 'state.json')
    state = json.load(open(state_path)) if os.path.isfile(state_path) else {}
    log = open(os.path.join(rep, 'setup.log'), 'a')

    def save_state():
        """Persist stage checkpoints after EVERY stage, so a retry after a
        late failure (e.g. pdb2gmx) does not redo the expensive collapse."""
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

    seed = 1000 + args.replica  # per-replica seed = f(index) only (no string games)
    dt_ps = 0.002
    total_steps = int(round(args.us * 1e6 / dt_ps))
    frame_steps = int(round(args.frame_ps / dt_ps))

    # ---- write mdp files ------------------------------------------------
    for name, txt in [('ions', MDP_IONS), ('minim', MDP_MINIM),
                      ('nvt', mdp_nvt().format(gen_seed=seed)),
                      ('npt1', mdp_npt(True, seed)),
                      ('npt2', mdp_npt(False, seed)),
                      ('prod', mdp_prod(total_steps, frame_steps, seed))]:
        with open(os.path.join(mdpdir, name + '.mdp'), 'w') as f:
            f.write(txt)

    # ---- stage: build start structure ------------------------------------
    start_pdb = os.path.join(rep, 'start.pdb')
    if state.get('build') and os.path.isfile(start_pdb):
        print('[setup] build: done, skipping')
    else:
        if not os.path.isfile(start_pdb):
            state['build'] = False  # checkpoint lied; structure is gone
        if os.path.isfile(start_pdb):
            # adopt a leftover start structure if it is COMPLETE and fits
            # the box. The residue-count check matters: a NaN crash during
            # PDBFile.writeFile can leave a truncated start.pdb whose
            # partial chain trivially "fits" -- adopting it would build a
            # silently shortened protein.
            try:
                rg0, dmax0, n0 = _ca_span_nm(start_pdb)
            except Exception:
                rg0, dmax0, n0 = 0.0, None, 0
            if n0 == 140 and dmax0 is not None and dmax0 < args.box_nm - 2.0:
                print(f'[setup] adopting existing start.pdb '
                      f'(Rg={rg0:.2f} nm, maxDist={dmax0:.2f} nm)')
                state['build'] = True
        if not state.get('build'):
            sh([sys.executable, '-u', os.path.join(HERE, 'build_start.py'),
                '--seed', str(seed), '--out', start_pdb,
                '--box-nm', str(args.box_nm), '--platform', args.build_platform,
                '--max-collapse-ns', str(args.max_collapse_ns)],
               rep, args.dry_run, log)
            state['build'] = True
        save_state()

    # ---- stage: pdb2gmx ---------------------------------------------------
    water, waters = find_ff(args.ff_dir)
    water = args.water or water
    if water is None:
        sys.exit(f'[setup] could not detect water models in {args.ff_dir}; '
                 f'pass --water explicitly')
    print(f'[setup] force field: {args.ff_dir}; water: {water} '
          f'(choices in port: {waters})')
    # Make the custom FF visible to every gmx call from here on (pdb2gmx,
    # ions grompp): topol.top includes the FF by relative name, resolved via
    # cwd/GMXLIB/GMXDATA -- and cwd is the replica dir, so GMXLIB is required.
    ff_parent = os.path.dirname(os.path.abspath(args.ff_dir))
    # NOTE: never leave empty components in GMXLIB (e.g. a trailing colon
    # when GMXLIB was unset) -- GROMACS >= 2026 aborts with
    # "Attempted to open empty/null directory path" when it hits one.
    _gmxlib = [ff_parent] + [p for p in
                             os.environ.get('GMXLIB', '').split(os.pathsep) if p]
    os.environ['GMXLIB'] = os.pathsep.join(_gmxlib)
    if state.get('pdb2gmx') and os.path.isfile(os.path.join(rep, 'conf.gro')):
        print('[setup] pdb2gmx: done, skipping')
    else:
        ff_name = os.path.basename(os.path.abspath(args.ff_dir)).split('.')[0]
        env = os.environ.copy()
        base = [gmx, 'pdb2gmx', '-f', 'start.pdb', '-o', 'conf.gro',
                '-p', 'topol.top', '-ff', ff_name, '-ignh']
        # GROMACS >= 2026 validates -water against the *stock* water list
        # only: the custom port's watermodels.dat never reaches the
        # command-line enum even with GMXLIB set (-ff IS accepted, -water is
        # NOT). Try, in order:
        #   1. -water <keyword>          (works where the enum includes GMXLIB)
        #   2. interactive stdin selection (menu lists the FF's own models)
        #   3. -water tip4p, then patch the topol.top include to the port's itp
        water_idx = waters.index(water) + 1 if water in waters else 1
        water_itp = water + '.itp'
        attempts = [([*base, '-water', water], None),
                    (base, f'{water_idx}\n')]
        if os.path.isfile(os.path.join(args.ff_dir, water_itp)):
            attempts.append(([*base, '-water', 'tip4p'], 'PATCH'))
        ok = False
        for cmd, mode in attempts:
            note = ('' if mode is None else
                    '   [stdin: water choice]' if mode != 'PATCH'
                    else '   [+ patch topol.top water include]')
            log.write('$ ' + ' '.join(cmd) + note + '\n'); log.flush()
            print('$ ' + ' '.join(cmd) + note)
            if args.dry_run:
                ok = True
                break
            r = subprocess.run(cmd, cwd=rep, env=env,
                               input=(mode if mode not in (None, 'PATCH')
                                      else None),
                               text=True, stdout=log,
                               stderr=subprocess.STDOUT)
            if r.returncode != 0:
                log.write(f'[setup] pdb2gmx attempt failed (rc='
                          f'{r.returncode}); trying next strategy\n')
                log.flush()
                continue
            if mode == 'PATCH':
                tp = os.path.join(rep, 'topol.top')
                txt = open(tp).read()
                # unanchored on purpose: pdb2gmx may write the include with
                # a relative or an absolute ff path depending on how the FF
                # was found; tip4p.itp appears only on the water line
                new = txt.replace('tip4p.itp', water_itp)
                if new != txt:
                    open(tp, 'w').write(new)
                    log.write(f'[setup] patched water include -> '
                              f'{water_itp}\n')
                else:
                    log.write('[setup] WARNING: tip4p include not found in '
                              'topol.top; check the water topology\n')
                log.flush()
            ok = True
            break
        if not ok:
            sys.exit(f'[setup] pdb2gmx failed with all water-selection '
                     f'strategies; see {log.name}')
        state['pdb2gmx'] = True
        save_state()

    # ---- stage: box --------------------------------------------------------
    if state.get('box') and os.path.isfile(os.path.join(rep, 'boxed.gro')):
        print('[setup] box: done, skipping')
    else:
        sh([gmx, 'editconf', '-f', 'conf.gro', '-o', 'boxed.gro',
            '-bt', 'cubic', '-box', f'{args.box_nm:.1f}'], rep, args.dry_run, log)
        state['box'] = True
        save_state()

    # ---- stage: solvate ----------------------------------------------------
    # 4-point water geometry: prefer the .gro shipped in the FF port
    # (e.g. a99SBdisp_water.gro in the Robustelli port), else the stock
    # GROMACS tip4p.gro (identical geometry to TIP4P-Disp).
    tip4p_gro = None
    for cand in os.listdir(args.ff_dir):
        if re.search(r'(tip4p|disp).*\.gro$', cand, re.I):
            tip4p_gro = os.path.join(args.ff_dir, cand); break
    if tip4p_gro is None:
        gmxdata = os.environ.get('GMXDATA') or '/usr/local/gromacs/share/gromacs'
        guess = os.path.join(gmxdata, 'top', 'tip4p.gro')
        tip4p_gro = guess if os.path.isfile(guess) else 'tip4p.gro'
    if state.get('solvate') and os.path.isfile(os.path.join(rep, 'solv.gro')):
        print('[setup] solvate: done, skipping')
    else:
        sh([gmx, 'solvate', '-cp', 'boxed.gro', '-cs', tip4p_gro,
            '-o', 'solv.gro', '-p', 'topol.top'], rep, args.dry_run, log)
        state['solvate'] = True
        save_state()

    # ---- stage: ions --------------------------------------------------------
    if state.get('ions') and os.path.isfile(os.path.join(rep, 'ions.gro')):
        print('[setup] ions: done, skipping')
    else:
        sh([gmx, 'grompp', '-f', 'mdp/ions.mdp', '-c', 'solv.gro',
            '-p', 'topol.top', '-o', 'ions.tpr', '-maxwarn', '1'],
           rep, args.dry_run, log)
        # genion reads an index group from stdin: pick "SOL"
        cmd = f'echo SOL | {gmx} genion -s ions.tpr -o ions.gro -p topol.top ' \
              f'-neutral -conc {args.salt_M} -pname NA -nname CL'
        log.write('$ ' + cmd + '\n'); log.flush(); print('$ ' + cmd)
        if not args.dry_run:
            r = subprocess.run(cmd, cwd=rep, shell=True,
                               stdout=log, stderr=subprocess.STDOUT)
            if r.returncode != 0:
                sys.exit(f'[setup] genion failed; see {log.name}')
        state['ions'] = True
        save_state()

    state.update({'replica': args.replica, 'seed': seed,
                  'box_nm': args.box_nm, 'salt_M': args.salt_M,
                  'prod_us': args.us, 'prod_steps': total_steps,
                  'frame_ps': args.frame_ps, 'water': water,
                  'ff_parent': ff_parent})
    save_state()
    print(f'[setup] replica {args.replica} ready in {rep} '
          f'({args.us} us = {total_steps} steps). Next: run_asyn_aa.py')


if __name__ == '__main__':
    main()
