"""Append-only JSONL journal. One record = one os.write(O_APPEND) + fsync, so a crash leaves at
most one truncated last line; load() drops unparseable lines and duplicate keys (first wins)."""
import json
import os


def dumps(rec):
    return json.dumps(rec, sort_keys=True, separators=(",", ":"))


def append(path, rec):
    data = (dumps(rec) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def load(path):
    """-> (records by key tuple, n_duplicates, n_bad_lines)."""
    recs, dup, bad = {}, 0, 0
    if not os.path.exists(path):
        return recs, 0, 0
    with open(path, "rb") as f:
        for line in f:
            try:
                r = json.loads(line)
                k = tuple(r["k"])
            except (ValueError, KeyError, TypeError):
                bad += 1
                continue
            if k in recs:
                dup += 1
            else:
                recs[k] = r
    return recs, dup, bad


def load_strict(path):
    """Load a qualification journal, rejecting malformed rows or duplicate keys."""
    rows, dup, bad = load(path)
    if dup or bad:
        raise ValueError(f'corrupt or duplicate journal: {os.path.basename(path)}')
    return rows
