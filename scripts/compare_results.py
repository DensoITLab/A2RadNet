"""Compare every numeric evaluation field against a recorded reference."""
import argparse
import json
import math
from pathlib import Path


def differences(expected, actual, atol, path=''):
    problems = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            return [path + ': result fields differ']
        for key in expected:
            problems.extend(differences(expected[key], actual[key], atol, path + '/' + key))
    elif isinstance(expected, (int, float)):
        if not isinstance(actual, (int, float)) or not math.isfinite(actual) or abs(expected-actual) > atol:
            problems.append(f'{path}: expected {expected}, actual {actual}')
    elif expected != actual:
        problems.append(path + ': values differ')
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--actual', type=Path, required=True)
    parser.add_argument('--atol', type=float, default=1e-5)
    args = parser.parse_args()
    if args.atol < 0:
        parser.error('--atol must be non-negative')
    problems = differences(json.loads(args.reference.read_text()), json.loads(args.actual.read_text()), args.atol)
    for problem in problems:
        print(problem)
    print('Reference comparison:', 'FAIL' if problems else 'PASS', '(absolute tolerance', args.atol, ')')
    raise SystemExit(bool(problems))


if __name__ == '__main__':
    main()
