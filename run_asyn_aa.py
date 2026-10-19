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
import argparse, datetime, json, os, shutil, subprocess, sys, time

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
    ap.add_argument('--gpu-update', action='store_true', default=True,
                    help='use -update gpu (default on; --no-gpu-update to disable)')
    ap.add_argument('--no-gpu-update', dest='gpu_update', action='store_false')
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
        os.environ['GMXLIB'] = (ff_parent + os.pathsep
                                + os.environ.get('GMXLIB', ''))
    else:
        print('[run] WARNING: state.json has no ff_parent; if grompp cannot '
              'find the force field, re-run setup_asyn_aa.py once', flush=True)

    if args.mpi_launcher:
        gmx = shutil.which('gmx_mpi') or shutil.which('gmx') or 'gmx_mpi'
    else:
        gmx = shutil.which('gmx') or shutil.which('gmx_mpi') or 'gmx'
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
        r = subprocess.run(cmd, cwd=rep, stdout=logf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            sys.exit(f'[run] FAILED ({r.returncode}): {" ".join(cmd)} '
                     f'-- see {logf.name}; re-run this script to resume')

    gpu_flags = [] if args.cpu else \
                ['-nb', 'gpu', '-pme', 'gpu', '-bonded', 'gpu'] + \
                (['-update', 'gpu'] if args.gpu_update else [])

    mpi = args.mpi_launcher.split()

    def mdrun(deffnm, maxh=None):
        cmd = mpi + [gmx, 'mdrun', '-deffnm', deffnm,
                     '-ntomp', str(args.ntomp), '-pin', 'on'] + gpu_flags
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
        sh([gmx, 'mdrun', '-deffnm', 'em', '-ntomp', str(args.ntomp)])
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
    if stage_done('prod'):
        logline('[run] prod done already -- nothing to do')
        return
    if not os.path.isfile(os.path.join(rep, 'prod.tpr')):
        sh([gmx, 'grompp', '-f', 'mdp/prod.mdp', '-c', 'npt2.gro',
            '-t', 'npt2.cpt', '-p', 'topol.top', '-o', 'prod.tpr',
            '-maxwarn', '1'])
    target = state['prod_steps']
    chunk = 0
    while True:
        logf_path = os.path.join(rep, 'prod.log')
        finished = (os.path.isfile(logf_path) and
                    'Finished mdrun' in open(logf_path, errors='ignore').read())
        if finished:
            break
        chunk += 1
        logline(f'[run] prod chunk {chunk} (target {target} steps)')
        prog(stage='prod', chunk=chunk, target_steps=target)
        mdrun('prod', maxh=args.maxh)
        if args.dry_run:
            break
        if chunk > 10000:
            sys.exit('[run] absurd chunk count -- something is wrong')
    mark_done('prod')
    hours = (time.time() - t_start) / 3600.0
    prog(stage='done', walltime_h=round(hours, 2))
    logline(f'[run] replica {args.replica} COMPLETE '
            f'({state["prod_us"]} us) in {hours:.1f} h wall')


if __name__ == '__main__':
    main()
