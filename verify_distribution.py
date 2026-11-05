"""Check the curated release's integrity without importing the model runtime."""
from pathlib import Path
import argparse
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parent
IGNORED = {'.git', '.venv', '.cache', 'lits-llm', 'tmp', '__pycache__'}
PATTERNS = {
    'API-key-shaped text': rb'\bsk-[A-Za-z0-9_-]{20,}',
    'GitHub-token-shaped text': rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})',
    'private-key header': rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', action='store_true',
                        help='also reject a local .env, symlinks and unexpected release files')
    args = parser.parse_args()
    inventory = json.loads((ROOT / 'EVIDENCE_INVENTORY.json').read_text())
    rows = inventory['copied_files']
    seen = set()
    problems = []
    for row in rows:
        rel = Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts or str(rel) in seen:
            problems.append('Invalid or repeated inventory path')
            continue
        seen.add(str(rel))
        path = ROOT / rel
        if path.is_symlink() or not path.is_file():
            problems.append(f'Missing file or symlink: {rel}')
            continue
        if rel.name == '.env' or any(p in IGNORED for p in rel.parts):
            problems.append(f'Forbidden distribution path: {rel}')
            continue
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != row['sha256']:
            problems.append(f'Hash mismatch: {rel}')
        for label, pattern in PATTERNS.items():
            if re.search(pattern, payload):
                # Never display the matching text or a secret value.
                problems.append(f'{label}: {rel}')
    if args.release:
        if (ROOT / '.env').exists():
            problems.append('Local .env exists; exclude it before release')
        for path in ROOT.rglob('*'):
            rel = path.relative_to(ROOT)
            if path.is_symlink():
                problems.append(f'Symlink must be removed before release: {rel}')
                continue
            if any(p in IGNORED for p in rel.parts) or path.suffix == '.pyc':
                continue
            if path.is_file() and str(rel) not in seen and rel.name != 'EVIDENCE_INVENTORY.json':
                problems.append(f'Uninventoried release file: {rel}')
    if problems:
        print(json.dumps({'ok': False, 'problems': problems}, indent=2))
        raise SystemExit(1)
    print(json.dumps({'ok': True, 'files_verified': len(seen),
                      'model_api_calls': 0, 'release_check': args.release}, indent=2))

if __name__ == '__main__':
    main()
