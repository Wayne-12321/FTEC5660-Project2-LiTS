"""Bounded local BFS repair; explicit symbolic alternative, not LiTS MCTS.

Uses only initial positions, goal relations and operator rules; never reads gold plans.
LLM proposal optionally orders successors, but cannot bypass legality/goal checks.
"""
import argparse
import json
import re
import time
import sys
from collections import deque
from pathlib import Path
from common import ROOT,UPSTREAM,read,write,sha,verify_manifest,dataset_kwargs
from symbolic import positions,apply,replay
from plan_verify import verify_plan

def legal_successors(pos, preferred=()):
    blocks=sorted(pos)
    actions=[]
    for block in blocks:
        actions.extend([f'pick up the {block} block',f'put down the {block} block'])
        for support in blocks:
            if support!=block:
                actions.extend([f'unstack the {block} block from on top of the {support} block',
                                f'stack the {block} block on top of the {support} block'])
    ranking={a:i for i,a in enumerate(dict.fromkeys(preferred))}
    actions.sort(key=lambda a:(ranking.get(a,len(ranking)),a))
    for action in actions:
        nxt=dict(pos)
        try: apply(nxt,action)
        except ValueError:continue
        yield action,nxt

def repair(initial,goal,max_steps=6,max_states=2000,preferred=()):
    initial_pos=positions(initial)
    targets=re.findall(r'the (\w+) block is on top of the (\w+) block',goal)
    if not initial_pos or not targets:raise ValueError('Unparsed state/goal')
    goal_met=lambda p:all(p.get(b)==s for b,s in targets)
    start=tuple(sorted(initial_pos.items()))
    queue=deque([(initial_pos,[])])
    seen={start};expanded=0;duplicates=0
    while queue:
        pos,path=queue.popleft()
        if goal_met(pos):
            assert replay(initial,path,goal)
            return dict(success=True,actions=path,expanded_states=expanded,
                        discovered_states=len(seen),duplicates_pruned=duplicates,stop_reason='verified_solution')
        if len(path)>=max_steps:continue
        expanded+=1
        for action,nxt in legal_successors(pos,preferred):
            key=tuple(sorted(nxt.items()))
            if key in seen:
                duplicates+=1;continue
            if len(seen)>=max_states:
                return dict(success=False,expanded_states=expanded,discovered_states=len(seen),
                            duplicates_pruned=duplicates,stop_reason='state_budget')
            seen.add(key);queue.append((nxt,path+[action]))
    return dict(success=False,expanded_states=expanded,discovered_states=len(seen),
                duplicates_pruned=duplicates,stop_reason='depth_exhausted')

def render_state(pos):
    facts=[]
    held=next((b for b,s in pos.items() if s=='hand'),None)
    facts.append(f'the hand is holding the {held} block' if held else 'the hand is empty')
    for b,support in sorted(pos.items()):
        if support=='table':facts.append(f'the {b} block is on the table')
        elif support=='hand':facts.append(f'the {b} block is in the hand')
        else:facts.append(f'the {b} block is on top of the {support} block')
        if b not in pos.values() and support!='hand':facts.append(f'the {b} block is clear')
    return ', '.join(facts)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proposal-result',type=Path)
    p.add_argument('--index',type=int,default=0)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    from entry import offline_guard
    offline_guard()
    manifest=verify_manifest()
    if not 0 <= a.index < len(manifest['splits']['dev']):p.error('Index outside dev split')
    sys.path[:0]=[str(UPSTREAM),str(UPSTREAM/'demos')]
    from lits_benchmark.blocksworld import load_blocksworld,BlocksWorldTransition
    example=load_blocksworld(**dataset_kwargs('dev'))[a.index]
    result=dict(initial=example['init_state_str'],goal=example['query_or_goals'],
                task_id=manifest['splits']['dev'][a.index]['task_id'],attempts=[])
    if sha(ROOT/'data/dev.json')!=manifest['splits']['dev_sha256']:p.error('Changed dataset')
    if a.proposal_result:
        proposal=read(a.proposal_result/'result.json')
        for key in ('initial','goal','task_id'):assert proposal[key]==result[key]
        result['attempts']=proposal['attempts']
    output=a.output.resolve()
    if output.exists():p.error('Output already exists')
    output.mkdir(parents=True)
    preference=[]
    for attempt in result['attempts']:
        try:
            actions=json.loads(attempt['response'])['actions']
            preference.extend(x for x in actions if isinstance(x,str))
        except (ValueError,KeyError,TypeError):pass
    rows=[]
    variants=[('pure_symbolic_control',[])]
    if a.proposal_result: variants.insert(0,('proposal_ordered_symbolic_repair',preference))
    for name,preferred in variants:
        start=time.monotonic()
        solved=repair(result['initial'],result['goal'],max_steps=6,max_states=2000,preferred=preferred)
        solved.update(variant=name,elapsed_seconds=time.monotonic()-start,new_api_calls=0)
        if solved['success']:
            checked=verify_plan(json.dumps({'actions':solved['actions']}),result['initial'],result['goal'],6)
            assert checked['success'];solved['verification']=checked
            # Cross-check action membership with upstream's separately implemented operator rules.
            for action,pos in zip(solved['actions'],checked['states'][:-1]):
                assert action in BlocksWorldTransition.generate_actions(render_state(pos))
            assert BlocksWorldTransition.goal_check(result['goal'],render_state(checked['states'][-1]))[0]
            solved['upstream_operator_goal_check']=True
        rows.append(solved)
    write(output/'result.json',dict(task_id=result['task_id'],initial=result['initial'],goal=result['goal'],
        rows=rows,max_steps=6,max_states=2000,reference_plan_used=False,api_calls=0,
        proposal_source=str(a.proposal_result) if a.proposal_result else None,
        proposal_result_sha256=sha(a.proposal_result/'result.json') if a.proposal_result else None,
        implementation_sha256=sha(__file__),verifier_sha256=sha(ROOT/'scripts/symbolic.py'),
        upstream_commit=manifest['upstream_commit'],data_sha256=sha(ROOT/'data/dev.json')))
    (output/'executed_local_plan_repair.py').write_text(Path(__file__).read_text())
    print(json.dumps(rows,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
