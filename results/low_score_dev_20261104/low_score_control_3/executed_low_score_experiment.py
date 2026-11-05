"""Independent two-development-task trial; frozen cycle experiment remains intact.

Default: guarded offline CLI dry runs. Paid mode requires matching regression pass.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.parse import urlparse
from common import ROOT,read,write,sha,verify_manifest,environment,cli_args

VARIANTS=('low_score_control_3','low_score_expand_3')
NEW_SOURCES=('lits_low_score.py','low_score_experiment.py','check_low_score.py')
SHARED_SOURCES=('lits_fixes.py','cycle_state.py','common.py','entry.py','evaluate.py','symbolic.py','deepseek_cost.py')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute-api',action='store_true')
    a=p.parse_args()
    manifest=verify_manifest()
    old=read(ROOT/'reports/cycle_protocol_20261104.json')
    assert all(sha(ROOT/f)==v for f,v in old['fingerprints'].items()),'Original frozen source/config changed'
    configs={v:read(ROOT/'configs'/f'{v}.json') for v in VARIANTS}
    c=dict(configs[VARIANTS[0]]);c['search_algorithm']=configs[VARIANTS[1]]['search_algorithm']
    assert c==configs[VARIANTS[1]],'Unexpected second experimental variable'
    for cfg in configs.values():
        base=read(ROOT/'configs/cycle_pruned_3.json');base['search_algorithm']=cfg['search_algorithm']
        assert cfg==base,'Original search or spending limits changed'
    if a.execute_api:
        passed=read(ROOT/'reports/low_score_checks_20261104.json')
        assert passed['passed'] and passed['paid_api_calls']==0
        assert passed['implementation_sha256']==sha(ROOT/'scripts/lits_low_score.py')
        assert passed['test_sha256']==sha(ROOT/'scripts/check_low_score.py')
    env=environment(offline=not a.execute_api)
    model='openai/offline-placeholder';endpoint=None
    if a.execute_api:
        from dotenv import dotenv_values
        values=dotenv_values(ROOT/'.env')
        for key in ('OPENAI_API_KEY','OPENAI_API_BASE','LITS_MODEL'):
            value=os.environ.get(key) or values.get(key)
            if not value:p.error('Missing local API setting: '+key)
            env[key]=value
        model=env['LITS_MODEL'];endpoint=env['OPENAI_API_BASE']
        if model!='openai/deepseek-flash' or urlparse(endpoint).hostname!='api.deepseek.com':
            p.error('This paired trial expects official DeepSeek Flash')
        env['PYTHON_DOTENV_DISABLED']='1';env.pop('HF_TOKEN',None)
    out=a.output.resolve()
    if out.exists():p.error('Output directory exists')
    sources=NEW_SOURCES+SHARED_SOURCES
    hashes={f'scripts/{f}':sha(ROOT/'scripts'/f) for f in sources}
    hashes.update({f'configs/{v}.json':sha(ROOT/'configs'/f'{v}.json') for v in VARIANTS})
    protocol=dict(date='2026-11-04',fingerprints=hashes,original_fingerprints=old['fingerprints'],
        tasks=manifest['splits']['dev'],split='dev',local_seed=42,provider_seed_sent=False,
        workers=2,paired_runs_per_task=1,api_execution=a.execute_api,
        treatment='simulation singleton score<=0: add at most one candidate, at most twice/task; retain original candidate and selection/backprop/cycle rules',
        admission='at least 4 unused API slots and 10000 returned-token headroom; conservative precheck, original guard is final next-call protection',
        primary_metric='Independent symbolic replay on both planned development tasks; incomplete tasks count as failure',
        no_test_tasks=True,no_reference_plans=True)
    out.mkdir(parents=True);write(out/'protocol.json',protocol)
    for variant,cfg in configs.items():
        arm=out/variant
        write(arm/'experiment.json',dict(variant=variant,config=cfg,split='dev',repeats=1,
            tasks=manifest['splits']['dev'],api_execution=a.execute_api,model=model,endpoint=endpoint,
            pricing_verified_date='2026-11-04',upstream_commit=manifest['upstream_commit'],
            data_sha256=sha(ROOT/'data/dev.json'),source_fingerprints=hashes))
        for src in sources:shutil.copyfile(ROOT/'scripts'/src,arm/('executed_'+src))
    def task(variant,record):
        cfg=configs[variant];dest=out/variant/f"r0_task{record['index']:03d}";dest.mkdir()
        e=env.copy();e.update(LITS_SEED='42',LITS_MAX_CALLS=str(cfg['limits']['max_api_calls_per_task']),
            LITS_MAX_TOKENS=str(cfg['limits']['max_total_tokens_per_task']),LITS_MAX_OUTPUT=str(cfg['max_output_tokens']),
            LITS_CALL_LEDGER=str(dest/'api_calls.jsonl'))
        args=cli_args(cfg,'dev',record['index'],dest,model,dry=not a.execute_api)
        # Add to the existing nargs+ module list instead of overriding --include.
        args.insert(args.index('--dataset'),'lits_low_score')
        write(dest/'invocation.json',dict(args=args,seed=42,task=record,repeat=0))
        start=time.monotonic();status='completed';code=None
        try:
            with (dest/'console.log').open('w') as log:
                proc=subprocess.run([sys.executable,str(ROOT/'scripts/entry.py'),'search',*args],
                    env=e,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=cfg['limits']['timeout_seconds_per_task'])
            code=proc.returncode
            if code:status='error'
        except subprocess.TimeoutExpired:status='timeout'
        if 'Experiment API budget exhausted' in (dest/'console.log').read_text(errors='replace'):status='budget_exhausted'
        write(dest/'status.json',dict(status=status,returncode=code,elapsed_seconds=time.monotonic()-start))
        result=dict(variant=variant,task_id=record['task_id'],index=record['index'],status=status)
        print(result,flush=True);return result
    jobs=[(v,t) for i,t in enumerate(manifest['splits']['dev']) for v in (VARIANTS if i%2==0 else VARIANTS[::-1])]
    results=[];fatal=False
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pending={};queue=iter(jobs)
        def add():
            try:v,t=next(queue)
            except StopIteration:return
            pending[pool.submit(task,v,t)]=(v,t)
        for _ in range(2):add()
        while pending:
            done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                pending.pop(f);r=f.result();results.append(r)
                if r['status'] not in ('completed','budget_exhausted'):fatal=True
            write(out/'batch_status.json',dict(planned_jobs=4,completed_jobs=results,fatal=fatal))
            if not fatal:
                for _ in done:add()
    assert all(sha(ROOT/f)==h for f,h in hashes.items()),'Trial source/config changed during execution'
    print('Trial ended:',len(results),'/4',flush=True)
    return int(fatal)

if __name__=='__main__':sys.exit(main())
