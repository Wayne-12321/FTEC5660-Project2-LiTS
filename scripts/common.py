"""Shared paths and CLI translation; importing this module never contacts a service."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'lits-llm'
DATA = UPSTREAM / 'demos/blocksworld'

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_manifest():
    manifest = read(ROOT / 'data/manifest.json')
    current = subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()
    if current != manifest['upstream_commit']:
        raise ValueError('Upstream commit differs from frozen manifest')
    if subprocess.check_output(['git','-C',str(UPSTREAM),'diff','HEAD','--'],text=True).strip():
        raise ValueError('Upstream tracked files modified; review and record changes before running')
    for relative, expected in manifest['file_hashes'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f'Frozen file changed: {relative}')
    return manifest

def environment(offline=True):
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(UPSTREAM / 'demos'), str(UPSTREAM), str(ROOT / 'scripts')])
    env['PYTHONHASHSEED'] = '42'
    env['HF_HUB_OFFLINE'] = '1'
    env['HF_HUB_DISABLE_TELEMETRY'] = '1'
    env['AWS_EC2_METADATA_DISABLED'] = 'true'
    env['TOKENIZERS_PARALLELISM'] = 'false'
    env['KMP_USE_SHM'] = '0'
    env['OMP_NUM_THREADS'] = '1'
    env['MPLCONFIGDIR'] = str(ROOT / '.cache/matplotlib')
    env['HF_HOME'] = str(ROOT / '.cache/huggingface')
    env['LITS_OFFLINE'] = '1' if offline else '0'
    # No ambient credential or .env use during preflight/evaluation.
    if offline:
        for key in list(env):
            if any(s in key for s in ('API_KEY', 'ACCESS_KEY', 'SECRET', 'TOKEN')):
                env.pop(key, None)
        env['PYTHON_DOTENV_DISABLED'] = '1'
    return env

def dataset_kwargs(split):
    return dict(config_file=str(DATA / 'bw_data_bw_config.yaml'),
                domain_file=str(DATA / 'bw_data_generated_domain.pddl'),
                data_file=str(ROOT / 'data' / f'{split}.json'), base_dir=str(DATA))

def cli_args(config, split, index, output, model, dry=False, original=False):
    args = ['--include', 'lits_benchmark.blocksworld', 'lits_fixes', '--dataset', 'blocksworld',
            '--policy', 'blocksworld_unique',
            '--transition', 'blocksworld', '--policy-model', model, '--eval-model', model,
            '--cfg', 'search_algorithm=mcts_frontier', '--var', f'offset={index}', 'limit=1',
            '--output-dir', str(output), '--dataset-arg']
    args += [f'{k}={v}' for k, v in dataset_kwargs(split).items()]
    args += ['--search-arg'] + [f'{k}={str(v).lower() if isinstance(v, bool) else v}' for k,v in config['search_args'].items()]
    args[args.index('search_algorithm=mcts_frontier')] = 'search_algorithm='+config.get('search_algorithm','mcts_frontier')
    args += ['--component-arg', f"max_new_tokens={config['max_output_tokens']}"]
    if original:
        args.remove('lits_fixes')
        at = args.index('--policy')
        del args[at:at+2]
        args[args.index('search_algorithm='+config.get('search_algorithm','mcts_frontier'))] = 'search_algorithm=mcts'
    if dry:
        args += ['--dry-run']
    return args
