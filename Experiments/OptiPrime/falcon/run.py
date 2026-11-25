"""Run both parties on loopback and write per-direction application bytes to CSV."""
import argparse
import csv
import socket
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('binary', type=Path)
parser.add_argument('plan', type=Path)
parser.add_argument('degree', type=int)
parser.add_argument('output', type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
with socket.socket() as probe:
    probe.bind(('127.0.0.1', 0))
    port = probe.getsockname()[1]
jobs, logs = [], []
try:
    for party in (1, 2):
        log = (args.output / f'party{party}.log').open('w')
        logs.append(log)
        jobs.append(subprocess.Popen([str(args.binary.resolve()), str(party), str(port),
                                      str(args.degree), str(args.plan.resolve())],
                                     stdout=log, stderr=subprocess.STDOUT))
    deadline = time.monotonic() + 900
    while any(p.poll() is None for p in jobs):
        if any(p.poll() not in (0, None) for p in jobs) or time.monotonic() > deadline:
            raise RuntimeError(f'Failed two-party run; see {args.output}')
        time.sleep(0.2)
    if any(p.returncode for p in jobs):
        raise RuntimeError(f'Failed two-party run; see {args.output}')
finally:
    for p in jobs:
        if p.poll() is None:
            p.kill()
        p.wait()
    for log in logs:
        log.close()
records = []
for party in (1, 2):
    path = args.output / f'party{party}.log'
    records.append([s.split()[1:] for s in path.read_text().splitlines() if s.startswith('ROW ')])
expected = len(args.plan.read_text().splitlines()) * 2
if [len(r) for r in records] != [expected, expected]:
    raise RuntimeError('Missing layer records')
with (args.output / 'layers.csv').open('w') as out:
    writer = csv.writer(out)
    writer.writerow('layer method height channels stride inputs outputs dense_CPmult dense_auto s2c c2s checked'.split())
    for server, client in zip(*records):
        if server[:-2] != client[:-2] or server[-1] != client[-1]:
            raise RuntimeError('Party records disagree')
        writer.writerow([*server[:-2], server[-2], client[-2], server[-1]])
