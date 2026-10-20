#!/usr/bin/env python3
"""
run_asyn_aa.py -- run one alpha-synuclein all-atom replica to completion
with stage-level resume: if interrupted (preemption, crash, power), re-run
the same command and it continues from the last completed stage / last
checkpoint -- never from the start.

Stages (each guarded by a <stage>.done sentinel):
  em    energy minimization
  nvt   100 ps NVT, position restraints on protein heavy atoms
  npt1  100 ps NPT, position restraints
  npt2  200 ps NPT, free
  prod  production NPT (default 2.5 us), run in wallclock chunks:
        `gmx mdrun -cpi prod.cpt -append -maxh H` in a loop until the
        target step count is reached. Every chunk resumes exactly from
        the last checkpoint (positions, velocities, RNG state), so an
        interrupted chunk costs at most --maxh hours.

Usage:
  python3 run_asyn_aa.py --replica 1
  python3 run_asyn_aa.py --replica 1 --maxh 23.5 --ntomp 8
"""
import argparse, datetime, json, os, re, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))


def logline(msg):
    print(f'[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--replica', type=int, required=True)
    ap.add_argument('--runs-dir', default=os.path.join(HERE, 'runs'))
    ap.add_argument('--maxh', type=float, default=23.5,
                    help='wallclock per production chunk (hours); set just under '
                         'your scheduler limit')
    ap.add_argument('--ntomp', type=int, default=8)
    ap.add_argument('--gpu-update', action='store_true', default=False,
                    help='use -update gpu (default OFF: GPU update does not '
                         'support virtual sites, and TIP4P-Disp carries one '
                         'M-site per water -- GROMACS 2026 aborts if asked)')
    ap.add_argument('--cpu', action='store_true',
                    help='CPU-only node: drop all GPU offload flags (nb/pme/bonded/update)')
    ap.add_argument('--mpi-launcher', default='',
                    help='e.g. "srun" or "mpirun -np 64" for multi-node CPU runs '
                         '(use with gmx_mpi; launcher is prepended to mdrun)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    rep = os.path.join(args.runs_dir, f'rep{args.replica:02d}')
    state_path = os.path.join(rep, 'state.json')
    if not os.path.isfile(state_path):
        sys.exit(f'[run] {state_path} missing -- run setup_asyn_aa.py first')
    state = json.load(open(state_path))
    if not state.get('ions'):
        sys.exit(f'[run] setup incomplete for replica {args.replica} '
                 f'(ions stage not done) -- re-run setup_asyn_aa.py')

    # topol.top includes the custom FF by relative name; make it resolvable
    # for every grompp below (older state.json may lack ff_parent -> setup
    # must be re-run once to record it)
    ff_parent = state.get('ff_parent')
    if ff_parent:
        # no empty components -- GROMACS aborts on them (see setup script)
        _gmxlib = [ff_parent] + [p for p in
                                 os.environ.get('GMXLIB', '').split(os.pathsep)
                                 if p]
        os.environ['GMXLIB'] = os.pathsep.join(_gmxlib)
    else:
        print('[run] WARNING: state.json has no ff_parent; if grompp cannot '
              'find the force field, re-run setup_asyn_aa.py once', flush=True)

    if args.mpi_launcher:
        gmx = shutil.which('gmx_mpi') or shutil.which('gmx')
    else:
        gmx = shutil.which('gmx') or shutil.which('gmx_mpi')
    if gmx is None:
        # Never fall back to a bare 'gmx' string: a scrubbed PATH (pbsdsh
        # task shells) turns that into a cryptic FileNotFoundError deep in
        # subprocess instead of an actionable message here.
        sys.exit('[run] gmx not found on PATH -- the job environment lost '
                 'the GROMACS install (pbsdsh task? check GMX_BIN in '
                 '.pbsdsh_task_env and the task log)')
    logf = open(os.path.join(rep, 'run.log'), 'a', buffering=1)
    prog_path = os.path.join(rep, 'progress.json')
    t_start = time.time()

    def prog(**kw):
        p = json.load(open(prog_path)) if os.path.isfile(prog_path) else {}
        p.update(kw); p['updated'] = f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S}'
        json.dump(p, open(prog_path, 'w'), indent=2)

    def sh(cmd):
        line = '$ ' + ' '.join(cmd)
        print(line, flush=True); logf.write(line + '\n')
        if args.dry_run:
            return
        env = dict(os.environ)
        # mdrun aborts if OMP_NUM_THREADS disagrees with -ntomp; the cluster
        # environment may export an arbitrary value, so pin it here
        env['OMP_NUM_THREADS'] = str(args.ntomp)
        r = subprocess.run(cmd, cwd=rep, env=env,
                           stdout=logf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            sys.exit(f'[run] FAILED ({r.returncode}): {" ".join(cmd)} '
                     f'-- see {logf.name}; re-run this script to resume')

    gpu_flags = [] if args.cpu else \
                ['-nb', 'gpu', '-pme', 'gpu', '-bonded', 'gpu'] + \
                (['-update', 'gpu'] if args.gpu_update else [])

    mpi = args.mpi_launcher.split()
    # One replica = one thread-MPI rank on its pinned GPU. GROMACS 2026.2
    # refuses '-ntomp N' without an explicit rank count whenever a GPU is
    # visible; an external MPI launcher (srun/mpirun) sets ranks instead.
    ntmpi = [] if args.mpi_launcher else ['-ntmpi', '1']

    def mdrun(deffnm, maxh=None):
        cmd = mpi + [gmx, 'mdrun', '-deffnm', deffnm] + ntmpi + \
              ['-ntomp', str(args.ntomp), '-pin', 'on'] + gpu_flags
        if os.path.isfile(os.path.join(rep, deffnm + '.cpt')):
            cmd += ['-cpi', deffnm + '.cpt', '-append']  # resume in-stage
        if maxh is not None:
            cmd += ['-maxh', str(maxh)]
        sh(cmd)

    def stage_done(name):
        return os.path.isfile(os.path.join(rep, name + '.done'))

    def mark_done(name):
        if args.dry_run:
            return
        open(os.path.join(rep, name + '.done'), 'w').write(
            f'{datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n')
        prog(stage=name)

    # ---- em ----------------------------------------------------------------
    if stage_done('em'):
        logline('[run] em done, skipping')
    else:
        sh([gmx, 'grompp', '-f', 'mdp/minim.mdp', '-c', 'ions.gro',
            '-p', 'topol.top', '-o', 'em.tpr', '-maxwarn', '1'])
        sh([gmx, 'mdrun', '-deffnm', 'em'] + ntmpi +
           ['-ntomp', str(args.ntomp)])
        if not args.dry_run and not os.path.isfile(os.path.join(rep, 'em.gro')):
            sys.exit('[run] em.gro missing after minimization')
        mark_done('em')

    # ---- equilibration chain ------------------------------------------------
    equil = [('nvt',  'em.gro',  'em.gro',  'em.cpt'),
             ('npt1', 'nvt.gro', 'nvt.gro', 'nvt.cpt'),
             ('npt2', 'npt1.gro', None,     'npt1.cpt')]
    for name, conf, ref, cpt in equil:
        if stage_done(name):
            logline(f'[run] {name} done, skipping')
            continue
        cmd = [gmx, 'grompp', '-f', f'mdp/{name}.mdp', '-c', conf,
               '-p', 'topol.top', '-o', f'{name}.tpr', '-maxwarn', '1']
        if ref:
            cmd += ['-r', ref]
        if os.path.isfile(os.path.join(rep, cpt)):
            cmd += ['-t', cpt]            # exact continuation of velocities/state
        sh(cmd)
        mdrun(name)
        if not args.dry_run and not os.path.isfile(os.path.join(rep, name + '.gro')):
            sys.exit(f'[run] {name}.gro missing after mdrun')
        mark_done(name)

    # ---- production (chunked, checkpoint-resumed) ---------------------------
    # Completion is grounded in the STEP COUNT parsed from prod.log, never
    # in the string 'Finished mdrun': GROMACS prints that at EVERY graceful
    # exit, including the -maxh chunk-boundary exits that are supposed to
    # trigger the NEXT chunk. A string check marks every replica "done"
    # after one ~maxh chunk (a fraction of the target) and the sentinel
    # then skips the replica forever (2026-10-20 review).
    def prod_step():
        """Best estimate of completed production steps (0 if nothing
        reports). Format drift killed the naive table regex (GROMACS
        2026.2 dropped the Lambda column for non-FEP runs, dropped the
        'will finish' heartbeat, and row trailing whitespace is not
        guaranteed), so use THREE independent sources and take the max:
        1. energy-table rows, anchored on the 'Step ... Time' header and
           matched to end-of-line,
        2. 'step N, will finish' heartbeat lines (older GROMACS),
        3. the edr trailer via `gmx check` (time / dt).
        """
        last = 0
        logf_path = os.path.join(rep, 'prod.log')
        if os.path.isfile(logf_path):
            want_row = False
            for line in open(logf_path, errors='ignore'):
                if re.match(r'^\s*Step\s+Time\b', line):
                    want_row = True
                    continue
                if want_row:
                    m = re.match(r'^\s*(\d+)\s+\d+\.\d+(?:\s.*)?$', line)
                    if m:
                        last = int(m.group(1))
                    want_row = False
                m = re.match(r'^\s*step\s+(\d+),?\s+will finish', line, re.I)
                if m:
                    last = max(last, int(m.group(1)))
        edr_path = os.path.join(rep, 'prod.edr')
        if os.path.isfile(edr_path):
            dt = 0.002
            mdp_path = os.path.join(rep, 'mdp', 'prod.mdp')
            if os.path.isfile(mdp_path):
                m = re.search(r'^\s*dt\s*=\s*([\d.]+)',
                              open(mdp_path, errors='ignore').read(), re.M)
                if m:
                    dt = float(m.group(1))
            try:
                r = subprocess.run([gmx, 'check', '-e', 'prod.edr'],
                                   cwd=rep, capture_output=True, text=True,
                                   timeout=120)
                m = re.findall(r'Last\s+(?:energy\s+)?frame(?:\s+read)?'
                               r'\s+\S+\s+time\s+([\d.]+)',
                               r.stdout + r.stderr)
                if m:
                    last = max(last, int(float(m[-1]) / dt))
            except Exception:
                pass  # edr may be mid-write; the log sources still hold
        return last

    target = state['prod_steps']
    if stage_done('prod') and prod_step() >= target:
        logline('[run] prod done already -- nothing to do')
        return
    if stage_done('prod'):
        logline(f'[run] WARNING: prod.done present but only {prod_step()}/'
                f'{target} steps in prod.log -- removing the stale sentinel '
                f'and resuming')
        os.remove(os.path.join(rep, 'prod.done'))
    if not os.path.isfile(os.path.join(rep, 'prod.tpr')):
        sh([gmx, 'grompp', '-f', 'mdp/prod.mdp', '-c', 'npt2.gro',
            '-t', 'npt2.cpt', '-p', 'topol.top', '-o', 'prod.tpr',
            '-maxwarn', '1'])
    chunk = 0
    while prod_step() < target:
        before = prod_step()
        chunk += 1
        logline(f'[run] prod chunk {chunk} ({before}/{target} steps done)')
        prog(stage='prod', chunk=chunk, target_steps=target, steps_done=before)
        mdrun('prod', maxh=args.maxh)
        if args.dry_run:
            break
        if prod_step() <= before:
            sys.exit(f'[run] prod chunk {chunk} made no progress '
                     f'({before}/{target} steps before and after mdrun) '
                     f'-- see {logf.name}')
        if chunk > 10000:
            sys.exit('[run] absurd chunk count -- something is wrong')
    logline(f'[run] prod target reached: {prod_step()}/{target} steps')
    mark_done('prod')
    hours = (time.time() - t_start) / 3600.0
    prog(stage='done', walltime_h=round(hours, 2))
    logline(f'[run] replica {args.replica} COMPLETE '
            f'({state["prod_us"]} us) in {hours:.1f} h wall')


if __name__ == '__main__':
    main()
