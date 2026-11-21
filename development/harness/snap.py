"""Immutable code snapshot keyed by the content hash of development/ (h_tree)."""
import hashlib
import json
import os
import shutil
import stat

SNAP_ROOT = "/tmp/gsnap"
SKIP_DIRS = {"results", "__pycache__", ".pytest_cache"}
CODE_EXT = (".py", ".json", ".txt")     # code + data the code reads (sigma.json, fingerprints)


def tree_hash(root):
    h = hashlib.sha256()
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d not in SKIP_DIRS)
        for fn in sorted(fns):
            if not fn.endswith(CODE_EXT):
                continue
            p = os.path.join(dp, fn)
            h.update(os.path.relpath(p, root).encode() + b"\0")
            with open(p, "rb") as f:
                h.update(f.read() + b"\0")
    return h.hexdigest()[:12]


def snapshot(root):
    """Copy root's code tree to SNAP_ROOT/<h> (read-only) once; -> (h, path)."""
    root = os.path.abspath(root)
    h = tree_hash(root)
    dst = os.path.join(SNAP_ROOT, h)
    if not os.path.isdir(dst):
        os.makedirs(SNAP_ROOT, exist_ok=True)
        tmp = f"{dst}.tmp{os.getpid()}"
        shutil.copytree(root, tmp, ignore=lambda d, names: [
            n for n in names if n in SKIP_DIRS or (os.path.isfile(os.path.join(d, n))
                                                   and not n.endswith(CODE_EXT))])
        if tree_hash(tmp) != h:
            raise RuntimeError("snapshot hash mismatch (tree changed while copying)")
        for dp, dns, fns in os.walk(tmp):
            for fn in fns:
                os.chmod(os.path.join(dp, fn), stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        try:
            os.rename(tmp, dst)
        except OSError:                      # a concurrent snapshot won the race
            shutil.rmtree(tmp, ignore_errors=True)
    return h, dst


def program_digest(program):
    """Digest a registered stage program using the original JSON representation."""
    return hashlib.sha256(json.dumps(program, sort_keys=True).encode()).hexdigest()
