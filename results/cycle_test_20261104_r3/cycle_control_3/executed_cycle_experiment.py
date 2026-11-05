"""Paired four-worker development/test experiment. Default mode is offline dry-run."""
import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
from common import ROOT,read,write,sha,verify_manifest,UPSTREAM,environment

VARIANTS=['cycle_control_3','cycle_pruned_3']
SOURCES=['lits_fixes.py','cycle_state.py','common.py','run.py','entry.py','evaluate.py','cycle_experiment.py']
FREEZE=ROOT/'reports/cycle_protocol_20261104.json'

def fingerprints():
    return {f'scripts/{name}':sha(ROOT/'scripts'/name) for name in SOURCES} | {
        f'configs/{name}.json':sha(ROOT/'configs'/f'{name}.json') for name in VARIANTS}

def freeze(dev_root):
    if FREEZE.exists():raise ValueError('Protocol already frozen; do not overwrite after test access')
    for variant in VARIANTS:
        assert (dev_root/variant/'summary.json').exists(), 'Evaluate both development arms first'
    write(FREEZE,dict(date='2026-11-04',fingerprints=fingerprints(),dev_root=str(dev_root),
        development_summaries={v:read(dev_root/v/'summary.json') for v in VARIANTS},
        test_rule='Next 20 frozen tasks, one run each arm, local seed 42, 4 concurrent tasks, alternate arm order by task. No test-driven tuning.',
        primary_metric='Independent symbolic replay success / all 20 planned tasks; incomplete runs count as failures',
        secondary_metrics=['API calls','returned tokens','estimated CNY off-peak/peak','pruned nodes','ancestor revisits','budget interruptions'],
        only_experimental_difference='search_algorithm: mcts_frontier versus mcts_cycle',
        model='openai/deepseek-flash',pricing_verified_date='2026-11-04',
        no_reference_plans=True,no_symbolic_transition_or_planner=True))
    print('Protocol frozen:',FREEZE)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=['dev','test'],default='dev')
    p.add_argument('--output',type=__import__('pathlib').Path)
    p.add_argument('--freeze-dev',type=__import__('pathlib').Path)
    p.add_argument('--execute-api',action='store_true')
    a=p.parse_args()
    manifest=verify_manifest()
    b=read(ROOT/'configs/cycle_control_3.json');c=read(ROOT/'configs/cycle_pruned_3.json')
    b['search_algorithm']=c['search_algorithm'];assert b==c
    if a.freeze_dev:freeze(a.freeze_dev.resolve());return
    if a.split=='test':
        assert read(FREEZE)['fingerprints']==fingerprints(), 'Frozen source/config changed'
    if not a.output:p.error('--output required')
    output=a.output.resolve()
    if output.exists():p.error('Output already exists')
    output.mkdir(parents=True)
    tasks=manifest['splits'][a.split]
    jobs=[]
    for i,task in enumerate(tasks):
        for variant in (VARIANTS if i%2==0 else VARIANTS[::-1]):jobs.append((variant,task))
    for variant in VARIANTS:
        cfg=read(ROOT/'configs'/f'{variant}.json')
        write(output/variant/'experiment.json',dict(config=cfg,variant=variant,split=a.split,repeats=1,
            tasks=tasks,model='openai/deepseek-flash' if a.execute_api else 'openai/offline-placeholder',
            endpoint='https://api.deepseek.com',api_execution=a.execute_api,
            pricing_verified_date='2026-11-04',upstream_commit=manifest['upstream_commit'],
            data_sha256=sha(ROOT/'data'/f'{a.split}.json'),source_fingerprints=fingerprints()))
        for source in SOURCES:shutil.copyfile(ROOT/'scripts'/source,output/variant/('executed_'+source))
    def job(variant,task):
        index=task['index'];dest=output/'.jobs'/f'{variant}_{index:03d}'
        args=[sys.executable,str(ROOT/'scripts/run.py'),'--config',variant,'--split',a.split,
              '--task-index',str(index),'--output',str(dest)]
        if a.execute_api:args.append('--execute-api')
        env=os.environ.copy()
        for key in ('OPENAI_API_KEY','OPENAI_API_BASE','LITS_MODEL'):env.pop(key,None)
        proc=subprocess.run(args,cwd=ROOT,env=env,capture_output=True,text=True)
        src=dest/f'r0_task{index:03d}'
        if src.exists():
            target=output/variant/src.name
            shutil.move(src,target)
            status=read(target/'status.json')['status']
        else:
            status='startup_error'
            write(dest/'startup_error.json',dict(returncode=proc.returncode))
        # Keep all job manifests/source snapshots, but never copy .env or credentials.
        return dict(variant=variant,index=index,status=status,returncode=proc.returncode)
    results=[];fatal=False
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending={};iterator=iter(jobs)
        def add_job():
            try:v,t=next(iterator)
            except StopIteration:return False
            pending[pool.submit(job,v,t)]=(v,t);return True
        for _ in range(4):add_job()
        while pending:
            done,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                r=future.result();results.append(r);print(r,flush=True)
                if r['status'] not in ('completed','budget_exhausted'):fatal=True
                write(output/'batch_status.json',dict(completed_jobs=results,planned_jobs=len(jobs),fatal=fatal))
            if not fatal:
                for _ in done:add_job()
    print('Batch ended:',len(results),'/',len(jobs),flush=True)
    return int(fatal)

if __name__=='__main__':sys.exit(main())
