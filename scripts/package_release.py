"""Package an allowlisted release candidate without runtime logs or Git history."""
import argparse
import io
from pathlib import Path
import tarfile

from check_release import check

ROOT_FILES = ['LICENSE', 'README.md', 'NOTICE.txt', 'CITATION.cff', 'requirements.txt', 'pyproject.toml',
              '.gitignore', 'train.py', 'evaluate.py']
SOURCE_DIRS = ['a2radnet', 'configs', 'reference_results', 'docs', 'scripts', 'tests', 'LICENSES']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-weights', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('dist/A2RadNet-source.tar.gz'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    failures = check(root)
    if failures:
        raise SystemExit('Source audit failed: ' + '; '.join(failures))
    files = [root / name for name in ROOT_FILES]
    files.extend(root / name for name in ['Dockerfile', '.dockerignore'] if (root / name).is_file())
    for folder in SOURCE_DIRS:
        files.extend(p for p in (root / folder).rglob('*') if p.is_file() and
                     '__pycache__' not in p.parts and p.suffix != '.pyc')
    if args.include_weights:
        for dataset in ['radial', 'raddet']:
            files += [root / 'checkpoints' / f'{dataset}_seed3{suffix}' for suffix in ['.pth', '.json']]
    output = args.output.resolve()
    if output.exists():
        parser.error('Output exists; choose a new archive filename.')
    if output in [p.resolve() for p in files]:
        parser.error('Archive output must not replace a source file.')
    for path in files:
        if not path.is_file() or path.is_symlink():
            raise SystemExit('Missing file or symbolic link: ' + str(path.relative_to(root)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, 'w:gz') as archive:
        for path in sorted(files):
            data = path.read_bytes()
            info = tarfile.TarInfo('A2RadNet/' + path.relative_to(root).as_posix())
            info.size = len(data)
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ''
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
    print('Release candidate archive:', output)
    print('ITLAB non-commercial terms apply to original contributions and pretrained models; see licensing.md.')


if __name__ == '__main__':
    main()
