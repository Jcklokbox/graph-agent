"""Background runs without polling the agent: start -> log + done marker; wait = one blocking
call with a bounded timeout and a bounded output (tail).

    python3 -m harness.bg start NAME [--cpu 0|1|0,1] -- CMD ...   (runs CMD in a fresh snapshot)
    python3 -m harness.bg wait NAME [TIMEOUT<=170] [BYTES<=2000]
"""
import os
import shlex
import subprocess
import sys
import time

from harness.snap import SNAP_ROOT, snapshot

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # development/
RUNS = os.path.join(HERE, "results", "runs")
TAU_MAX, B_MAX = 170, 2000


def start(name, cmd, cpu="0,1", tree=None):
    d = os.path.join(RUNS, name)
    if os.path.exists(d):
        raise SystemExit(f"run {name} exists")
    if tree:                                  # resume on an existing immutable snapshot
        h, snap = tree, os.path.join(SNAP_ROOT, tree)
        if not os.path.isdir(snap):
            raise SystemExit(f"no snapshot {tree}")
    os.makedirs(d)
    if not tree:
        h, snap = snapshot(HERE)
    env = dict(os.environ, GRAPH_AGENT_MACHINE="1", PYTHONDONTWRITEBYTECODE="1", PG_TREE=h, PG_SNAP=snap,
               PG_RUN=d, PG_RUNS=RUNS, PYTHONPATH=os.pathsep.join(
                   [snap, os.environ.get("PYTHONPATH", "/tmp/graph-agent-deps")]))
    inner = " ".join(shlex.quote(c) for c in cmd)
    sh = (f"nice -n 19 taskset -c {cpu} {inner} > {shlex.quote(d)}/log 2>&1; "
          f"echo $? > {shlex.quote(d)}/done.tmp && mv {shlex.quote(d)}/done.tmp {shlex.quote(d)}/done")
    with open(os.path.join(d, "cmd"), "w") as f:
        f.write(f"tree {h}\ncpu {cpu}\n{inner}\n")
    subprocess.Popen(["sh", "-c", sh], cwd=d, env=env, start_new_session=True,  # private writable cwd; code from snap
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)
    print(f"started {name} tree={h} cpu={cpu}")


def wait(name, timeout=TAU_MAX, nbytes=B_MAX):
    d = os.path.join(RUNS, name)
    timeout, nbytes = min(float(timeout), TAU_MAX), min(int(nbytes), B_MAX)
    t0 = time.monotonic()
    done = os.path.join(d, "done")
    while not os.path.exists(done) and time.monotonic() - t0 < timeout:
        time.sleep(0.5)
    status = open(done).read().strip() if os.path.exists(done) else "running"
    log = os.path.join(d, "log")
    tail = b""
    if os.path.exists(log):
        with open(log, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - nbytes))
            tail = f.read()
    print(f"[{name}] exit={status} waited={time.monotonic() - t0:.0f}s")
    sys.stdout.write(tail.decode(errors="replace")[-nbytes:])
    return status


if __name__ == "__main__":
    from harness.beacon import cli

    with cli():
        a = sys.argv[1:]
        if a[0] == "start":
            cpu = a[a.index("--cpu") + 1] if "--cpu" in a else "0,1"
            pre = a[:a.index("--")]
            tree = pre[pre.index("--tree") + 1] if "--tree" in pre else None
            start(a[1], a[a.index("--") + 1:], cpu, tree)
        elif a[0] == "wait":
            wait(a[1], *(a[2:4]))
