"""Network-disabled audit of round-three paired losses; never loads gold plans.

Read only frozen experiment results. No search/policy change, no model client.
"""
import json
import math
from collections import Counter
from pathlib import Path
from common import ROOT, read, write, sha, verify_manifest
from entry import offline_guard
offline_guard()
from symbolic import positions, apply, replay
from cycle_state import canonical_state

ROUND=ROOT/'results/cycle_test_20261104_r3'
ARMS=('cycle_control_3','cycle_pruned_3')

def state_text(state):
    steps=state.get('steps') or []
    return steps[-1].get('next_state') if steps else state['init_state']

def action(node):
    value=node['action']
    return value.get('action_str') if isinstance(value,dict) else value

def indexed(tree):
    nodes={};paths={}
    def visit(node,path):
        nodes[node['id']]=node;paths[node['id']]=path+[node]
        for child in node.get('children',[]):visit(child,path+[node])
    visit(tree,[])
    return nodes,paths

def tree_replay(tree,nodes,paths):
    initial=tree['state']['init_state'];errors=[];keys={};checked=0
    for ident,node in nodes.items():
        if node['state'] is None:continue
        pos=positions(initial)
        try:
            for n in paths[ident][1:]:apply(pos,action(n))
            if pos!=positions(state_text(node['state'])):
                errors.append(dict(node_id=ident,error='LLM position mismatch'))
            keys[ident]=tuple(sorted(pos.items()));checked+=1
        except ValueError as exc:errors.append(dict(node_id=ident,error=str(exc)))
    return dict(known_nodes_checked=checked,errors=errors),keys

def policy_records(folder):
    return [dict(line_no=i+1,**json.loads(line)) for i,line in enumerate((folder/'llm_calls.jsonl').read_text().splitlines())]

def node_summary(n):
    return dict(id=n['id'],action=action(n),fast_reward=n['fast_reward'],
                known=n['state'] is not None,visit_count=n.get('visit_count',0),
                cum_rewards=n.get('cum_rewards',[]))

def arm_details(folder,index):
    tree=read(folder/'full_tree'/f'full_tree_{index}.json');nodes,paths=indexed(tree)
    audit,keys=tree_replay(tree,nodes,paths)
    logs=[json.loads(line) for line in (folder/'inferencelogger.log').read_text().splitlines()]
    iteration_calls={}
    for log in logs:
        it=str(log['iteration']);counter=iteration_calls.setdefault(it,Counter())
        counter[log['role'].split('_')[0]]+=1
    traces=[]
    for f in sorted((folder/'checkpoints').glob(f'{index}_*.json')):
        if f.stem.endswith('_result'):continue
        path=read(f);traces.append(dict(file=str(f.relative_to(ROOT)),
            node_ids=[n['id'] for n in path],depth=len(path)-1,
            actions=[action(n) for n in path[1:]],fast_rewards=[n['fast_reward'] for n in path[1:]]))
    return dict(tree=tree,nodes=nodes,paths=paths,keys=keys,report=dict(
        folder=str(folder.relative_to(ROOT)),status=read(folder/'status.json'),position_replay=audit,
        iteration_api_records={it:dict(c) for it,c in iteration_calls.items()},traces=traces))

def main():
    manifest=verify_manifest();protocol=read(ROOT/'reports/cycle_protocol_20261104.json')
    assert all(sha(ROOT/name)==h for name,h in protocol['fingerprints'].items())
    source_report=read(ROUND/'cycle_comparison.json')
    losses=[p for p in source_report['paired_tasks'] if p['outcome']=='loss']
    rows=[];inputs={}
    for paired in losses:
        task=next(t for t in manifest['splits']['test'] if t['task_id']==paired['task_id'])
        index=task['index'];data={}
        for arm in ARMS:
            folder=ROUND/arm/f'r0_task{index:03d}';data[arm]=arm_details(folder,index)
            for file in [folder/'status.json',folder/'full_tree'/f'full_tree_{index}.json',
                         folder/'llm_calls.jsonl',folder/'inferencelogger.log',folder/'cycle_audit.json']:
                if file.exists():inputs[str(file.relative_to(ROOT))]=sha(file)
        c,p=(data[a] for a in ARMS)
        terminals=[n for n in c['nodes'].values() if n['is_terminal']]
        best=max(terminals,key=lambda n:max(n.get('cum_rewards') or [-float('inf')]))
        solution=c['paths'][best['id']];acts=[action(n) for n in solution[1:]]
        initial=c['tree']['state']['init_state'];goal=c['tree']['action']
        assert replay(initial,acts,goal)
        solution_keys=[c['keys'][n['id']] for n in solution]
        assert len(solution_keys)==len(set(solution_keys)), 'Control solution contains a cycle'
        # How far the logged pruned tree follows the successful control action sequence.
        node=p['tree'];prefix=[]
        for a in acts:
            match=next((ch for ch in node.get('children',[]) if action(ch)==a),None)
            if match is None:break
            prefix.append(match['id']);node=match
            if node['state'] is None:break
        missing=acts[len(prefix)] if len(prefix)<len(acts) else None
        children=[node_summary(n) for n in node.get('children',[])]
        events=[]
        for event in read(ROUND/ARMS[1]/f'r0_task{index:03d}'/'cycle_audit.json')['pruned_nodes']:
            n,a=event['node_id'],event['ancestor_id']
            ancestor_ids={x['id'] for x in p['paths'][n][:-1]}
            events.append(dict(**event,ancestor_is_on_path=a in ancestor_ids,
                independently_same_positions=p['keys'][n]==p['keys'][a],
                canonical_llm_states_equal=canonical_state(state_text(p['nodes'][n]['state']))==canonical_state(state_text(p['nodes'][a]['state'])),
                complete_goal_met=replay(initial,[action(x) for x in p['paths'][n][1:]],goal),
                descendants=len(p['nodes'][n].get('children',[]))))
        cp=policy_records(ROUND/ARMS[0]/f'r0_task{index:03d}')
        pp=policy_records(ROUND/ARMS[1]/f'r0_task{index:03d}')
        same_prompt_different_outputs=[]
        for left in cp:
            for right in pp:
                if (left['prompt_hash']==right['prompt_hash'] and left['temperature']==right['temperature']
                        and left['output']!=right['output']):
                    same_prompt_different_outputs.append(dict(prompt_hash=left['prompt_hash'],temperature=left['temperature'],
                        control_line=left['line_no'],pruned_line=right['line_no'],control_output=left['output'],pruned_output=right['output']))
        row=dict(task=task,paired_result=paired,initial=initial,goal=goal,
            arms={a:data[a]['report'] for a in ARMS},control_successful_node_ids=[n['id'] for n in solution],
            control_successful_actions=acts,control_success_path_cycle_free=True,
            pruned_tree_solution_prefix_node_ids=prefix,missing_next_solution_action=missing,
            unresolved_decision_node_id=node['id'],children_at_unresolved_decision=children,
            cycle_checks=events,same_prompt_different_policy_outputs=same_prompt_different_outputs,
            causal_limit='A cycle-free control solution is not directly cut by ancestor equality. Independently sampled arms do not isolate the pruning intervention; missing responses cannot be synthesized by replay.')
        if task['task_id']=='instance-69':
            folder=ROUND/ARMS[1]/f'r0_task{index:03d}'
            first=read(folder/'checkpoints'/f'{index}_0.json')
            second=read(folder/'checkpoints'/f'{index}_1.json')
            root_visits=second[0]['visit_count'];orange=first[1];red=second[1]
            def uct(n):return max(n['cum_rewards'])+math.sqrt(math.log(max(2,root_visits))/n['visit_count'])
            row['uct_before_iteration_2']=dict(w_exp=1,root_visits=root_visits,
                orange=dict(node_id=orange['id'],q=max(orange['cum_rewards']),visits=orange['visit_count'],uct=uct(orange)),
                red=dict(node_id=red['id'],q=max(red['cum_rewards']),visits=red['visit_count'],uct=uct(red)),
                hypothetical_if_cycle_rollout_value_zero=dict(orange_q=0,orange_uct=math.sqrt(math.log(max(2,root_visits))/orange['visit_count']),red_uct=uct(red),root_preference='red',not_a_full_search_replay=True),
                note='Existing positive mean-of-path reward from a no-progress loop makes orange the next selected root branch. This calculation alone does not predict a solution after changing backpropagation.')
        rows.append(row)
    report=dict(date='2026-11-04',round=str(ROUND.relative_to(ROOT)),paid_api_calls=0,network_disabled=True,
        frozen_sources_unchanged=True,reference_plans_used=False,source_hashes=inputs,cases=rows,
        summary=dict(loss_cases=len(rows),cycle_events=sum(len(r['cycle_checks']) for r in rows),
            all_cycle_events_independently_valid=all(e['ancestor_is_on_path'] and e['independently_same_positions'] and e['canonical_llm_states_equal'] and not e['complete_goal_met'] for r in rows for e in r['cycle_checks']),
            replay_errors=sum(len(a['position_replay']['errors']) for r in rows for a in r['arms'].values()),
            control_solution_next_action_missing_in_all_pruned_trees=all(r['missing_next_solution_action'] is not None for r in rows)))
    write(ROOT/'reports/cycle_round3_loss_analysis_20261104.json',report)
    print(json.dumps(report['summary'],ensure_ascii=False,indent=2))
    for row in rows:
        print(row['task']['task_id'],'critical node',row['unresolved_decision_node_id'],'missing',row['missing_next_solution_action'])

if __name__=='__main__':main()
