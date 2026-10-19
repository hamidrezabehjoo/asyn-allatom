#!/usr/bin/env python3
"""
launch_asyn_aa.py -- multi-GPU launcher for alpha-synuclein all-atom replicas.

One replica per GPU (GROMACS single-GPU offload is near-optimal at this
system size; replicas-per-GPU beats multi-GPU-per-replica for ensemble
generation). Maintains a queue: each free GPU picks up the next replica;
each replica runs with CUDA_VISIBLE_DEVICES pinned to its GPU.

Failure policy (the zgpu01 lesson): a failed replica is retried in place
up to --retries times -- the runner resumes from its last checkpoint, so
a retry costs at most one chunk. Replicas that still fail are listed at
the end and the launcher exits nonzero, so a surrounding shell loop can
stop instead of marching through 30 broken runs.

Usage:
  python3 launch_asyn_aa.py --replicas 8 --gpus 0,1,2,3
  python3 launch_asyn_aa.py --replicas 1-12 --gpus 0,1 --us 2.5 --maxh 23.5
  python3 launch_asyn_aa.py --replicas 4 --gpus 0 --setup-only
"""
import argparse, datetime, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))


def logline(msg):
    print(f'[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}', flush=True)


def parse_replicas(spec):
    if '-' in spec:
        a, b = spec.split('-')
        return list(range(int(a), int(b) + 1))
    return list(range(1, int(spec) + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--replicas', required=True, help='N (=> 1..N) or A-B')
    ap.add_argument('--gpus', required=True, help='comma list, e.g. 0,1,2,3')
    ap.add_argument('--us', type=float, default=2.5)
    ap.add_argument('--frame-ps', type=float, default=1000.0)
    ap.add_argument('--salt-M', type=float, default=0.15)
    ap.add_argument('--ff-dir', default=os.path.join(HERE, 'a99SBdisp.ff'))
    ap.add_argument('--maxh', type=float, default=23.5)
    ap.add_argument('--ntomp', type=int, default=8)
    ap.add_argument('--retries', type=int, default=2)
    ap.add_argument('--poll-s', type=int, default=60)
    ap.add_argument('--setup-only', action='store_true',
                    help='only build systems, do not run production')
    ap.add_argument('--node-index', type=int, default=0,
                    help='index of this node within a multi-node job '
                         '(the pbs script passes PBS_NODENUM via pbsdsh)')
    ap.add_argument('--node-count', type=int, default=1,
                    help='number of nodes in the job; the replica list is '
                         'split into contiguous blocks, one per node')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    replicas = parse_replicas(args.replicas)
    gpus = [g.strip() for g in args.gpus.split(',')]
    if args.node_count > 1:
        if not 0 <= args.node_index < args.node_count:
            sys.exit(f'[launch] --node-index {args.node_index} out of range '
                     f'for --node-count {args.node_count}')
        base, rem = divmod(len(replicas), args.node_count)
        lo = args.node_index * base + min(args.node_index, rem)
        hi = lo + base + (1 if args.node_index < rem else 0)
        replicas = replicas[lo:hi]
    logf = open(os.path.join(HERE, 'launcher.log'), 'a', buffering=1)

    # Truncated copies (e.g. an interrupted scp leaving a 0-byte script)
    # make Python exit 0 with no output -- indistinguishable from success
    # downstream. Refuse to fly with empty/missing scripts.
    for script in ('setup_asyn_aa.py', 'run_asyn_aa.py', 'build_start.py'):
        p = os.path.join(HERE, script)
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            sys.exit(f'[launch] {p} is missing or EMPTY (truncated copy?) '
                     f'-- re-copy it and verify with md5sum')

    def log(msg):
        logline(msg); logf.write(f'[{datetime.datetime.now():%H:%M:%S}] {msg}\n')

    if args.node_count > 1:
        log(f'node {args.node_index}/{args.node_count}: replicas {replicas}')
    if not replicas:
        log('nothing to do on this node')
        return

    # ---- setup phase (sequential; fast, CPU-bound) -------------------------
    for r in replicas:
        marker = os.path.join(HERE, 'runs', f'rep{r:02d}', 'state.json')
        need = True
        if os.path.isfile(marker):
            import json
            need = not json.load(open(marker)).get('ions')
        if need:
            log(f'setup replica {r}')
            rc = subprocess.call(
                [sys.executable, '-u', os.path.join(HERE, 'setup_asyn_aa.py'),
                 '--replica', str(r), '--us', str(args.us),
                 '--frame-ps', str(args.frame_ps), '--salt-M', str(args.salt_M),
                 '--ff-dir', args.ff_dir]
                + (['--dry-run'] if args.dry_run else []))
            if rc != 0:
                sys.exit(f'[launch] setup failed for replica {r} -- fix before '
                         f'running anything (stopping, as requested by || logic)')
        else:
            log(f'setup replica {r}: already done')
    if args.setup_only:
        log('setup-only: done')
        return

    # ---- production queue ---------------------------------------------------
    pending = [{'replica': r, 'attempts': 0} for r in replicas]
    running = {}   # gpu -> (proc, item)
    failed, done = [], []

    def launch(gpu, item):
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu
        cmd = [sys.executable, '-u', os.path.join(HERE, 'run_asyn_aa.py'),
               '--replica', str(item['replica']),
               '--maxh', str(args.maxh), '--ntomp', str(args.ntomp)]
        if args.dry_run:
            cmd.append('--dry-run')
        item['attempts'] += 1
        p = subprocess.Popen(cmd, env=env)
        log(f'GPU {gpu} <- replica {item["replica"]} '
            f'(attempt {item["attempts"]})')
        return p

    while pending or running:
        for gpu in gpus:
            if gpu in running:
                p, item = running[gpu]
                rc = p.poll()
                if rc is None:
                    continue
                del running[gpu]
                # rc==0 alone is not proof of completion (a truncated
                # script exits 0 silently): require the prod.done sentinel.
                sentinel = os.path.join(HERE, 'runs',
                                        f'rep{item["replica"]:02d}',
                                        'prod.done')
                really_done = (rc == 0 and
                               (args.dry_run or os.path.isfile(sentinel)))
                if really_done:
                    done.append(item['replica'])
                    log(f'replica {item["replica"]} DONE '
                        f'({len(done)}/{len(replicas)})')
                elif item['attempts'] <= args.retries:
                    why = ('exited 0 but prod.done MISSING -- suspicious'
                           if rc == 0 else f'exited {rc}')
                    log(f'replica {item["replica"]} {why} -- '
                        f'requeuing (resume from checkpoint)')
                    pending.append(item)
                else:
                    failed.append(item['replica'])
                    log(f'replica {item["replica"]} FAILED after '
                        f'{item["attempts"]} attempts')
            if pending and gpu not in running:
                item = pending.pop(0)
                running[gpu] = (launch(gpu, item), item)
        time.sleep(args.poll_s)

    log(f'finished: {len(done)} done, failed={failed}')
    if failed:
        sys.exit(f'[launch] FAILED replicas: {failed} -- inspect their run.log; '
                 f'relaunch to retry')


if __name__ == '__main__':
    main()
