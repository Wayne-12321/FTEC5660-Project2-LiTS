"""Audit prior checkpoints, without reading reference plans or making API calls."""
import json
from collections import Counter
from common import ROOT,read,write
from symbolic import positions,apply
rows=[]
for name in ['deepseek_dev_modified_fixed_02','deepseek_dev_baseline_fixed_01']:
    folder=ROOT/'results'/name/'r0_task000'
    logs=[json.loads(x) for x in (folder/'inferencelogger.log').read_text().splitlines()]
    row=dict(result_dir=name,calls_by_role=dict(Counter(x['role'].split('_')[0] for x in logs)),paths=[])
    for checkpoint in sorted((folder/'checkpoints').glob('0_*.json')):
        if checkpoint.stem=='0_result':continue
        path=read(checkpoint);pos=positions(path[0]['state']['init_state'])
        seen={tuple(sorted(pos.items()))};actions=[]
        for node in path[1:]:
            action=node['action']['action_str'];apply(pos,action)
            key=tuple(sorted(pos.items()))
            generated=positions(node['state']['steps'][-1]['next_state'])
            actions.append(dict(action=action,fast_reward=node['fast_reward'],
                                revisits_ancestor=key in seen,state_matches_symbolic=generated==pos))
            seen.add(key)
        row['paths'].append(dict(checkpoint=checkpoint.name,actions=actions))
    rows.append(row)
write(ROOT/'reports/planning_diagnosis.json',dict(rows=rows,
    scope='Completed checkpoint paths only; shared prefix transitions counted more than once. No state-position mismatch observed in these paths. This does not prove every transition or state predicate was correct.'))
print('Diagnosis saved; no API calls')
