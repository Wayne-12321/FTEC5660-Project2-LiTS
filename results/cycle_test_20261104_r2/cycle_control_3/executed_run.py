"""Default: offline upstream dry run. Paid calls require explicit --execute-api."""
import argparse
import os
import subprocess
import sys
import time
from common import ROOT, UPSTREAM, read, write, sha, environment, cli_args, verify_manifest

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', choices=['baseline_10','modified_3','cycle_control_3','cycle_pruned_3'], required=True)
    p.add_argument('--split', choices=['dev','test'], default='dev')
    p.add_argument('--repeats', type=int, choices=[1,2,3,4,5], default=1)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--task-index', type=int, help='Run one selected index within the frozen split')
    p.add_argument('--continue-on-budget', action='store_true', help='Continue batch after a task exhausts its declared budget')
    p.add_argument('--original', action='store_true', help='Use frozen upstream without local fixes')
    p.add_argument('--execute-api', action='store_true')
    p.add_argument('--output', type=__import__('pathlib').Path)
    a = p.parse_args()
    cfg = read(ROOT / 'configs' / f'{a.config}.json')
    manifest = verify_manifest()
    current = subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'], text=True).strip()
    if current != manifest['upstream_commit']:
        p.error('Upstream commit changed; review and regenerate manifest explicitly')
    if sha(ROOT / 'data' / f'{a.split}.json') != manifest['splits'][a.split + '_sha256']:
        p.error('Frozen dataset hash mismatch')
    tasks = manifest['splits'][a.split]
    if a.task_index is not None:
        if not 0 <= a.task_index < len(tasks):p.error('Invalid task index')
        tasks=[tasks[a.task_index]]
    if a.limit is not None:
        if not 1 <= a.limit <= len(tasks):
            p.error('--limit outside split size')
        tasks = tasks[:a.limit]
    env = environment(offline=not a.execute_api)
    model = 'openai/offline-placeholder'
    if a.execute_api:
        from dotenv import dotenv_values
        values = {k:v for k,v in dotenv_values(ROOT / '.env').items() if v}
        for key in ('OPENAI_API_KEY','OPENAI_API_BASE','LITS_MODEL'):
            value = os.environ.get(key) or values.get(key)
            if not value or 'REPLACE_' in value:
                p.error(f'Set {key} in Project2/.env before API execution')
            env[key] = value
        model = env['LITS_MODEL']
        if not model.startswith('openai/'):
            p.error('This bounded runner supports openai/ provider-compatible models only')
        env['PYTHON_DOTENV_DISABLED'] = '1'
        # Do not let upstream perform an unrelated Hugging Face login.
        env.pop('HF_TOKEN', None)
    output = (a.output or ROOT / ('results' if a.execute_api else 'reports/dry_runs') /
              f'{a.config}_{a.split}_{time.time_ns()}').resolve()
    if output.exists():
        p.error('Output directory exists; choose a new directory to avoid mixing runs')
    output.mkdir(parents=True)
    write(output / 'experiment.json', dict(config=cfg, variant=a.config, split=a.split,
        repeats=a.repeats, tasks=tasks, model=model, api_execution=a.execute_api,
        endpoint=env.get('OPENAI_API_BASE') if a.execute_api else None,
        upstream_commit=current, implementation='upstream' if a.original else cfg.get('search_algorithm','mcts_frontier'),
        extension_sha256=sha(ROOT/'scripts/lits_fixes.py'), config_sha256=sha(ROOT/'configs'/f'{a.config}.json'),
        data_sha256=sha(ROOT/'data'/f'{a.split}.json')))
    for source in ('lits_fixes.py','cycle_state.py','entry.py','common.py','run.py','evaluate.py'):
        (output/('executed_'+source)).write_bytes((ROOT/'scripts'/source).read_bytes())
    failures = 0
    # Separate process per task: fresh state, reliable timeout and failure accounting.
    for repeat in range(a.repeats):
        for task in tasks:
            dest = output / f"r{repeat}_task{task['index']:03d}"
            dest.mkdir()
            seed = 42 + repeat
            env.update(LITS_SEED=str(seed), LITS_MAX_CALLS=str(cfg['limits']['max_api_calls_per_task']),
                       LITS_MAX_TOKENS=str(cfg['limits']['max_total_tokens_per_task']),
                       LITS_MAX_OUTPUT=str(cfg['max_output_tokens']), LITS_CALL_LEDGER=str(dest/'api_calls.jsonl'))
            args = cli_args(cfg,a.split,task['index'],dest,model,dry=not a.execute_api,original=a.original)
            write(dest/'invocation.json',dict(args=args, seed=seed, task=task, repeat=repeat))
            start = time.monotonic()
            status, code = 'completed', None
            try:
                with (dest/'console.log').open('w') as log:
                    proc = subprocess.run([sys.executable,str(ROOT/'scripts/entry.py'),'search',*args],
                        env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                        timeout=cfg['limits']['timeout_seconds_per_task'])
                code = proc.returncode
                if code:
                    status = 'error'
            except subprocess.TimeoutExpired:
                status = 'timeout'
            if 'Experiment API budget exhausted' in (dest/'console.log').read_text(errors='replace'):
                status = 'budget_exhausted'
            write(dest/'status.json',dict(status=status,returncode=code,elapsed_seconds=time.monotonic()-start))
            print(f'{dest.name}: {status}', flush=True)
            if status != 'completed':
                failures += 1
                if status=='budget_exhausted' and a.continue_on_budget:
                    continue
                # Stop on first infrastructure/API failure; remaining planned tasks stay in denominator.
                print(f'Stopped; inspect {dest / "console.log"}')
                return 1
    print(f'Output: {output}')
    return bool(failures)

if __name__ == '__main__':
    sys.exit(main())
