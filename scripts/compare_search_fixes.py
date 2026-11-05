"""Offline, auditable before/after comparison for the same frozen dev sample."""
import os, sys, json
from collections import Counter
from common import ROOT, UPSTREAM, read, write, environment
os.environ.update(environment())
sys.path[:0]=[str(UPSTREAM),str(UPSTREAM/'demos')]
from entry import offline_guard
offline_guard()
from evaluate import summarize
from deepseek_cost import estimate

def metrics(folder):
    task=folder/'r0_task000'
    tree_path=task/'full_tree/full_tree_0.json'
    tree=read(tree_path) if tree_path.exists() else None
    stats=dict(nodes=0, expanded_parents=0, duplicate_sibling_edges=0)
    def visit(node):
        stats['nodes']+=1
        children=node.get('children',[])
        if children:
            stats['expanded_parents']+=1
            actions=[json.dumps(c['action'],sort_keys=True) for c in children]
            stats['duplicate_sibling_edges']+=len(actions)-len(set(actions))
        for child in children: visit(child)
    if tree is not None:
        visit(tree)
    else:
        stats={key:None for key in stats}
        stats['tree_note']='No full tree saved on budget interruption; checkpoint paths remain available'
    logs=[json.loads(x) for x in (task/'inferencelogger.log').read_text().splitlines()]
    calls=Counter(str(x['iteration']) for x in logs if 'iteration' in x)
    stats['inference_records_per_iteration']=dict(sorted(calls.items()))
    traces=[json.loads(x) for x in (task/'treetojsonl.jsonl').read_text().splitlines() if x.strip()]
    # File holds one JSON array of per-iteration paths per line.
    paths=[path for batch in traces for path in batch]
    # to_paths prepends the best trace; it is not an extra search iteration.
    if paths and read(task/'status.json')['status']=='completed':
        paths=paths[1:]
    if not paths:
        paths=[read(p) for p in sorted((task/'checkpoints').glob('0_*.json')) if p.stem != '0_result']
    ends=[p[-1]['id'] for p in paths if p]
    stats['trace_end_node_ids']=ends
    stats['repeated_trace_endpoints']=len(ends)-len(set(ends))
    return stats

specs=[('before','modified_3','deepseek_dev_modified_01'),
       ('after','modified_3','deepseek_dev_modified_fixed_02'),
       ('before','baseline_10','deepseek_dev_baseline_01'),
       ('after','baseline_10','deepseek_dev_baseline_fixed_01')]
rows=[]
for phase,variant,name in specs:
    folder=ROOT/'results'/name
    exp=read(folder/'experiment.json')
    assert exp['tasks'][0]['task_id']=='instance-11'
    summary=summarize(folder)
    cost=estimate(folder)
    status=read(folder/'r0_task000/status.json')
    rows.append(dict(phase=phase,variant=variant,result_dir=str(folder.relative_to(ROOT)),
        status=status,summary=summary,cost=cost,search=metrics(folder)))
assert len({read(ROOT/r['result_dir']/'experiment.json')['data_sha256'] for r in rows})==1
write(ROOT/'reports/search_fix_comparison.json',dict(rows=rows,
    note='Single development example, one run per arm; not an accuracy estimate. Cache rates and service latency vary. Earlier modified_fixed_01 initialization failed before any API call. Executed extension snapshots are stored in both result directories. Two additional post-run guards (unknown edge at max depth, partial tree on RuntimeError) passed offline tests; no extra paid runs. Neither alters the measured pre-interruption paths, whose selected depths were below 6.'))
print(json.dumps(rows,indent=2))
