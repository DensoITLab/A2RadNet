"""Check release source for machine paths, credentials and accidental artifacts.

Runtime outputs and locally exported checkpoints are excluded from the source
review. Inspect their contents separately before distributing those artifacts.
"""
import argparse
from pathlib import Path
import re

SKIP = {'outputs', 'checkpoints', '.git', '.venv', '__pycache__', '.pytest_cache', 'build', 'dist'}
PATTERNS = [
    re.compile(r'/(?:home|Users|teranex(?:_local)?)/[^\s\"\']+'),
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(r'\b(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16})\b'),
    re.compile(r'https?://[^/\s:@]+:[^/\s@]+@'),
]


def check(root):
    failures = []
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if any(part in SKIP or part.endswith('.egg-info') for part in relative.parts):
            continue
        if path.is_symlink():
            failures.append(str(relative) + ': symbolic link')
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() in {'.ipynb', '.pdf', '.pth', '.pt', '.npy', '.pickle', '.pyc', '.log', '.sif'} or path.name in {'.DS_Store', '.env'}:
            failures.append(str(relative) + ': unexpected release artifact')
            continue
        try:
            content = path.read_text()
        except UnicodeDecodeError:
            failures.append(str(relative) + ': unexpected binary')
            continue
        for number, line in enumerate(content.splitlines(), 1):
            if any(pattern.search(line) for pattern in PATTERNS):
                failures.append(f'{relative}:{number}: review path or credential pattern')
    return failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    failures = check(args.root.resolve())
    for failure in failures:
        print(failure)
    print('Source audit:', 'FAIL' if failures else 'PASS')
    raise SystemExit(bool(failures))
