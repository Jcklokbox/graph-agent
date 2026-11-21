"""Check live documentation and the English-only repository text convention."""
import argparse
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
CYRILLIC = re.compile('[\u0400-\u052f\u1c80-\u1c8f\u2de0-\u2dff\ua640-\ua69f]')
LINK = re.compile(r'\]\(([^()\s]+)\)')


def check():
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    count = 0
    errors = []
    for name in filter(None, paths):
        path = ROOT / name
        if not path.is_file():
            errors.append(f'missing tracked file: {name}')
            continue
        is_markdown = path.suffix.lower() == '.md'
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            if is_markdown:
                errors.append(f'invalid UTF-8: {name}')
            continue
        if CYRILLIC.search(text):
            errors.append(f'Cyrillic in tracked UTF-8 text: {name}')
        if is_markdown:
            count += 1
            for target in LINK.findall(text):
                url = urlsplit(target)
                if not url.scheme and not url.netloc and url.path:
                    if not (path.parent / unquote(url.path)).exists():
                        errors.append(f'broken local link: {name} -> {target}')
    for name in ('STATE.md', 'docs/INDEX.md'):
        if (ROOT / name).stat().st_size > 2048:
            errors.append(f'bootstrap document exceeds 2 KiB: {name}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'{count} Markdown files checked; no broken local links or Cyrillic text.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check'])
    parser.parse_args()
    check()
