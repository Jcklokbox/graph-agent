"""Check English Markdown and exact historical sources, or read a bounded English excerpt."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
PAGE_LINES, PAGE_BYTES = 32, 4200


def git(*argv):
    return subprocess.check_output(['git', *argv], cwd=ROOT)


def cyrillic_count(text):
    return sum('CYRILLIC' in unicodedata.name(ch, '') for ch in text)


def source(manifest, path):
    return git('show', f"{manifest['source_commit']}:{path}")


def check(manifest):
    old = {p for p in git('ls-tree', '-rz', '--name-only', manifest['source_commit']).decode().split('\0')
           if p.lower().endswith('.md')}
    if old != set(manifest['files']):
        raise ValueError('source manifest does not cover exactly the old Markdown inventory')
    for p, m in manifest['files'].items():
        raw = source(manifest, p)
        lines = raw.decode().splitlines()
        if (hashlib.sha256(raw).hexdigest() != m['sha256'] or len(raw) != m['bytes'] or len(lines) != m['lines']
                or cyrillic_count(raw.decode()) != m['cyrillic']
                or [i for i, line in enumerate(lines, 1) if line.startswith('#')] != m['section_lines']):
            raise ValueError(f'historical source mismatch: {p}')
        for module in m['modules']:
            if not (ROOT / module).is_file():
                raise ValueError(f'missing curated module: {module}')
    count, size, violations = 0, 0, []
    for p in git('ls-files', '-z').decode().split('\0'):
        if not p or not (ROOT / p).is_file():
            continue
        raw = (ROOT / p).read_bytes()
        if p.lower().endswith('.md'):
            count, size = count + 1, size + len(raw)
            raw.decode()  # require valid UTF-8 for Markdown
        try:
            n = cyrillic_count(raw.decode())
        except UnicodeDecodeError:
            continue  # compressed/binary evidence is not run-facing text
        if n:
            violations.append(f'{p}: {n}')
    if violations:
        raise ValueError('Cyrillic in tracked UTF-8 files: ' + ', '.join(violations))
    for p in ('STATE.md', 'docs/INDEX.md'):
        if (ROOT / p).stat().st_size > 2048:
            raise ValueError(f'bootstrap document exceeds 2 KiB: {p}')
    previous = sum(m['bytes'] for m in manifest['files'].values())
    print(f'{len(old)} historical Markdown sources verified byte-for-byte; '
          f'{count} current Markdown files, {previous} -> {size} bytes; 0 Cyrillic in tracked UTF-8 text.')


def paged(lines, page):
    pages, block, size = [], [], 0
    for line in lines:
        while len(line.encode()) > PAGE_BYTES:
            prefix = line.encode()[:PAGE_BYTES].decode(errors='ignore')
            if block:
                pages.append(block)
                block, size = [], 0
            pages.append([prefix])
            line = line[len(prefix):]
        n = len(line.encode()) + 1
        if block and (len(block) >= PAGE_LINES or size+n > PAGE_BYTES):
            pages.append(block)
            block, size = [], 0
        block.append(line)
        size += n
    if block:
        pages.append(block)
    if not 0 <= page < len(pages):
        raise ValueError(f'page must be 0..{len(pages)-1}')
    print(f'Page {page}/{len(pages)-1}; select --page explicitly.')
    print('\n'.join(pages[page]))


def excerpt(manifest, ident, part, page, path, line):
    if ident:
        if path or line is not None or not re.fullmatch(r'D-[0-9]{1,3}', ident):
            raise ValueError('use D-<number> or --source PATH --line NUMBER')
        n = int(ident[2:])
        path = 'archive/research/decisions.md' if n <= 38 else 'archive/research/runs/AR-5.md'
    elif path not in manifest['files'] or line is None:
        raise ValueError('choose a source from docs/source-map.json and a one-based --line')
    text = source(manifest, path).decode()
    # The archived historical ledger has one quoted assent. Keep the semantic assent in English.
    if path == 'archive/research/decisions.md':
        text = re.sub('"[\u0400-\u052f]+, [\u0400-\u052f]+"', '"Yes, do it"', text)
    if cyrillic_count(text):
        raise ValueError('raw non-English source is not emitted; see its English modules in docs/source-map.json')
    lines = text.splitlines()
    if ident and n <= 38:
        starts = [i for i, value in enumerate(lines) if re.match(r'^#{2,3}\s+' + ident + r'(?!\d)', value)]
        bounds = [(i, next((j for j in range(i+1, len(lines)) if re.match(r'^#{2,3} ', lines[j])), len(lines)))
                  for i in starts]
    elif ident:
        rx = re.compile(r'(?<!\w)D[- ]?' + str(n) + r'(?!\d)')
        starts = [i for i, value in enumerate(lines) if rx.search(value)]
        bounds = [(i, next((j for j in range(i+1, len(lines)) if not lines[j].strip()), len(lines)))
                  for i in starts]
    else:
        if not 1 <= line <= len(lines):
            raise ValueError('line outside source')
        bounds = [(line-1, len(lines))]
    if not 0 <= part < len(bounds):
        raise ValueError(f'part must be 0..{len(bounds)-1}; use --source/--line to inspect related passages')
    start, stop = bounds[part]
    print(f"{manifest['source_commit'][:12]}:{path}:{start+1}; part {part}/{len(bounds)-1}")
    paged(lines[start:stop], page)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    sub.add_parser('check')
    ex = sub.add_parser('excerpt')
    ex.add_argument('ident', nargs='?')
    ex.add_argument('--part', type=int, default=0)
    ex.add_argument('--page', type=int, default=0)
    ex.add_argument('--source')
    ex.add_argument('--line', type=int)
    a = p.parse_args()
    manifest = json.loads((ROOT / 'docs/source-map.json').read_text())
    try:
        if a.mode == 'check':
            check(manifest)
        else:
            excerpt(manifest, a.ident, a.part, a.page, a.source, a.line)
    except (ValueError, subprocess.CalledProcessError) as err:
        p.exit(2, f'{err}\n')


if __name__ == '__main__':
    main()
