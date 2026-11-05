"""Offline summary. Missing/failed tasks remain in the planned denominator."""
import argparse
import csv
import statistics
from pathlib import Path
from common import read, write
from symbolic import replay

def evaluate_task(folder, task, example):
    status = read(folder/'status.json') if (folder/'status.json').exists() else {'status':'not_run','elapsed_seconds':None}
    console = folder/'console.log'
    if console.exists() and 'Experiment API budget exhausted' in console.read_text(errors='replace'):
        status['status'] = 'budget_exhausted'
    row = dict(task_id=task['task_id'],status=status['status'],
        upstream_success=False, symbolic_success=False, evaluation_error='',
        elapsed_seconds=status['elapsed_seconds'],input_tokens=0,output_tokens=0,
        completed_api_calls=0,unaccounted_api_calls=0)
    ledger = folder/'api_calls.jsonl'
    if ledger.exists():
        records=[__import__('json').loads(s) for s in ledger.read_text().splitlines() if s.strip()]
        started={x['call'] for x in records if x['status']=='started'}
        completed={x['call'] for x in records if x['status']=='completed' and x.get('usage')}
        row['unaccounted_api_calls']=len(started-completed)
        for x in records:
            if x['status']=='completed' and x.get('usage'):
                row['completed_api_calls']+=1
                row['input_tokens']+=x['usage']['prompt_tokens']
                row['output_tokens']+=x['usage']['completion_tokens']
    files = list((folder/'terminal_nodes').glob('terminal_nodes_*.json'))
    if not files:
        row['evaluation_error']='missing terminal output'
        return row
    try:
        if len(files)!=1:
            raise ValueError('Expected one task per result directory')
        from lits.cli.eval_search import load_terminal_nodes_from_file
        from lits_benchmark.blocksworld import BlocksWorldTransition
        data=load_terminal_nodes_from_file(files[0])
        if data['query'] != example['query_or_goals'] or data['query_idx'] != task['index']:
            raise ValueError('Task/query does not match frozen manifest')
        nodes=data['terminal_nodes']
        if nodes:
            # Identical selection rule to upstream env-grounded evaluator.
            best=max(nodes,key=lambda n:max(n.cum_rewards) if getattr(n,'cum_rewards',None) else -float('inf'))
            state=best.state
            row['upstream_success']=BlocksWorldTransition.goal_check(data['query'],state.env_state)[0]
            if state.init_state != example['init_state_str']:
                raise ValueError('Initial state mismatch')
            row['symbolic_success']=replay(example['init_state_str'],[str(s.action) for s in state],data['query'])
        if status['status']!='completed':
            # Partial trajectories retained for audit, but incomplete runs are not counted as success.
            row['upstream_success']=row['symbolic_success']=False
    except Exception as exc:
        row['evaluation_error']=f'{type(exc).__name__}: {exc}'
        row['symbolic_success']=False
    if status['status']!='completed':
        row['upstream_success']=row['symbolic_success']=False
    return row

def summarize(root):
    from common import dataset_kwargs, verify_manifest, ROOT, sha
    from lits_benchmark.blocksworld import load_blocksworld
    exp=read(root/'experiment.json')
    if not exp['api_execution'] and not exp.get('fixture'):
        raise ValueError('Dry runs are not experimental results')
    manifest=verify_manifest()
    if not exp.get('fixture'):
        if exp['data_sha256'] != sha(ROOT/'data'/f"{exp['split']}.json") or exp['upstream_commit'] != manifest['upstream_commit']:
            raise ValueError('Result provenance differs from local code/data')
    examples=load_blocksworld(**dataset_kwargs(exp['split']))
    rows=[]
    for repeat in range(exp['repeats']):
        for task in exp['tasks']:
            row=evaluate_task(root/f"r{repeat}_task{task['index']:03d}",task,examples[task['index']])
            rows.append(dict(repeat=repeat,**row))
    rates=[sum(r['symbolic_success'] for r in rows if r['repeat']==i)/len(exp['tasks']) for i in range(exp['repeats'])]
    summary=dict(variant=exp['variant'],fixture=exp.get('fixture',False),planned_tasks=len(rows),
        completed_tasks=sum(r['status']=='completed' for r in rows),
        upstream_success_rate=sum(r['upstream_success'] for r in rows)/len(rows),
        symbolic_success_rate=statistics.mean(rates),
        symbolic_success_std=statistics.stdev(rates) if len(rates)>1 else None,
        repeat_symbolic_rates=rates,input_tokens=sum(r['input_tokens'] for r in rows),
        output_tokens=sum(r['output_tokens'] for r in rows),
        unaccounted_api_calls=sum(r['unaccounted_api_calls'] for r in rows),
        cost=None,cost_note='Use provider billing/usage details; no prices assumed. Failed requests may be billed.')
    write(root/'summary.json',summary)
    with (root/'per_task.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return summary

if __name__=='__main__':
    import os,sys
    from common import UPSTREAM
    sys.path[:0]=[str(UPSTREAM),str(UPSTREAM/'demos')]
    os.environ['HF_HUB_OFFLINE']='1'
    from entry import offline_guard
    offline_guard()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('result_dir',type=Path)
    args=p.parse_args()
    print(__import__('json').dumps(summarize(args.result_dir.resolve()),indent=2))
