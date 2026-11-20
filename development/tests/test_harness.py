import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from harness import journal
from harness.snap import snapshot, tree_hash

DEV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
           PYTHONPATH=os.pathsep.join([DEV, os.environ.get("PYTHONPATH", "")]))
ARGS = ["t2", "--agents", "rw", "cfg", "--set", "v1", "--rows",
        "blocking,renewal-ABA,savings", "--stages", "0:0-3", "--procs", "1"]


def run_staged(jpath, cwd=DEV, env=ENV, extra=()):
    return subprocess.run([sys.executable, "-m", "harness.staged", *ARGS, "--journal", jpath,
                           *extra], cwd=cwd, env=env, capture_output=True, text=True, timeout=160)


def metrics(jpath):
    recs, dup, bad = journal.load(jpath)
    return {k[1:]: journal.dumps(r["m"]) for k, r in recs.items()}, dup, bad


class TestJournal(unittest.TestCase):
    def test_truncated_line_and_duplicates(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "j.jsonl")
        journal.append(p, {"k": ["h", 0, "a"], "m": 1})
        journal.append(p, {"k": ["h", 0, "a"], "m": 2})
        with open(p, "ab") as f:
            f.write(b'{"k":["h",1,"a"],"m"')         # crash mid-record
        recs, dup, bad = journal.load(p)
        self.assertEqual((len(recs), dup, bad), (1, 1, 1))
        self.assertEqual(recs[("h", 0, "a")]["m"], 1)


class TestStaged(unittest.TestCase):
    def test_kill9_resume_no_dup_no_loss(self):
        d = tempfile.mkdtemp()
        ref, killed = os.path.join(d, "ref.jsonl"), os.path.join(d, "k.jsonl")
        self.assertEqual(run_staged(ref).returncode, 0)
        p = subprocess.Popen([sys.executable, "-m", "harness.staged", *ARGS, "--journal", killed],
                             cwd=DEV, env=ENV, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        t0 = time.time()
        while time.time() - t0 < 60:
            if os.path.exists(killed) and sum(1 for _ in open(killed)) >= 5:
                break
            time.sleep(0.05)
        os.kill(p.pid, signal.SIGKILL)
        p.wait()
        with open(killed) as f:
            n_before = sum(1 for _ in f)
        self.assertEqual(run_staged(killed).returncode, 0)
        a, dup_a, _ = metrics(ref)
        b, dup_b, bad_b = metrics(killed)
        self.assertGreater(n_before, 0)
        self.assertLess(n_before - 1, len(a))        # killed mid-run (last line may be partial)
        self.assertEqual(dup_b, 0)
        self.assertEqual(a, b)                       # no loss, bitwise equal metrics

    def test_probe_is_prefix(self):
        d = tempfile.mkdtemp()
        probe, full = os.path.join(d, "p.jsonl"), os.path.join(d, "f.jsonl")
        run_staged(probe, extra=("--stages", "0"))   # the last --stages wins
        a, _, _ = metrics(probe)
        run_staged(full)
        b, _, _ = metrics(full)
        self.assertTrue(a)
        for k, v in a.items():
            self.assertEqual(b[k], v)               # s=0 records bitwise identical


class TestIsolation(unittest.TestCase):
    def test_mutating_working_copy_does_not_change_run(self):
        d = tempfile.mkdtemp()
        work = os.path.join(d, "dev")
        shutil.copytree(DEV, work, ignore=shutil.ignore_patterns("results", "__pycache__"))
        for dp, _, fns in os.walk(work):                  # DEV may be a read-only snapshot
            for fn in fns:
                os.chmod(os.path.join(dp, fn), 0o644)
        h, snap = snapshot(work)
        env = dict(ENV, PYTHONPATH=os.pathsep.join([snap, os.environ.get("PYTHONPATH", "")]),
                   PG_TREE=h)
        ja, jb = os.path.join(d, "a.jsonl"), os.path.join(d, "b.jsonl")
        f = os.path.join(work, "eval", "phenomena.py")   # mutate the working copy, then run
        src = open(f).read()
        open(f, "w").write(src.replace("def run_group(", "def _x(): pass\n\n\ndef run_group(", 1)
                           + "\nraise SystemExit('mutated')\n")
        self.assertNotEqual(tree_hash(work), h)
        p = subprocess.Popen([sys.executable, "-m", "harness.staged", *ARGS, "--journal", ja],
                             cwd=snap, env=env, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        open(os.path.join(work, "models", "factor.py"), "a").write("\nX = 1\n")  # mid-run
        self.assertEqual(p.wait(timeout=160), 0)
        shutil.copy(os.path.join(snap, "models", "factor.py"), os.path.join(work, "models"))
        os.chmod(os.path.join(work, "models", "factor.py"), 0o644)
        open(f, "w").write(src)
        self.assertEqual(tree_hash(work), h)
        self.assertEqual(run_staged(jb, cwd=snap, env=env).returncode, 0)
        self.assertEqual(metrics(ja)[0], metrics(jb)[0])


class TestVectorizedEquivalence(unittest.TestCase):
    """eps-test: the cached OnlineHMM.T (hot path) equals the reference formula, fixed seed."""

    def test_cached_T_bitwise(self):
        import numpy as np
        from eval.phenomena import run_group
        from models import online_hmm

        def t_ref(self, a):
            K = self.K
            pool = self.ntr.sum(0) + self.alpha + self.sticky * np.eye(K)
            pool /= pool.sum(1, keepdims=True)
            n = self.ntr[a]
            return (n + self.shrink * pool) / (n.sum(1, keepdims=True) + self.shrink)

        args = ("factor-L4-P-C-N25", "savings", "one", 0, "v1")
        y_vec = run_group(*args, trace=True)
        cached = online_hmm.OnlineHMM.T
        online_hmm.OnlineHMM.T = t_ref
        try:
            y_ref = run_group(*args, trace=True)
        finally:
            online_hmm.OnlineHMM.T = cached
        self.assertEqual(journal.dumps(y_vec), journal.dumps(y_ref))   # eps = 0


if __name__ == "__main__":
    unittest.main()
