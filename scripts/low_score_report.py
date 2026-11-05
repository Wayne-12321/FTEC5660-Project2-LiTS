"""Offline paired report for the two-task low-score-widening development trial.

Only development results are read. Success requires independent symbolic replay;
completion/cost alone is not an effectiveness result. No model client is created.
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from common import ROOT,UPSTREAM,read,write,sha,verify_manifest,environment

VARIANTS=('low_score_control_3','low_score_expand_3')
EXPECTED_RULE=dict(threshold=0,max_triggers_per_task=2,extra_actions_per_trigger=1,
    reserve_calls=4,reserve_returned_tokens=10000,phase='simulate',requires_total_children=1)


def require(ok,message):
    if not ok:raise ValueError(message)


def ledger_report(path):
    rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    starts=[r['call'] for r in rows if r['status']=='started']
    require(len(starts)==len(set(starts)),f'Duplicate request start: {path}')
    completed=[r for r in rows if r['status']=='completed' and r.get('usage')]
    end_ids=[r['call'] for r in completed]
    require(len(end_ids)==len(set(end_ids)),f'Duplicate request completion: {path}')
    require(set(end_ids)<=set(starts),f'Completion without request start: {path}')
    hits=misses=output=0
    for record in completed:
        usage=record['usage']
        require(usage['prompt_cache_hit_tokens']+usage['prompt_cache_miss_tokens']==usage['prompt_tokens'],
                f'Invalid cache accounting: {path}')
        hits+=usage['prompt_cache_hit_tokens'];misses+=usage['prompt_cache_miss_tokens']
        output+=usage['completion_tokens']
    low=(hits*.02+misses+output*4)/1_000_000
    return dict(calls=len(completed),unaccounted_calls=len(set(starts)-set(end_ids)),
        cache_hit_input_tokens=hits,cache_miss_input_tokens=misses,input_tokens=hits+misses,
        output_tokens=output,total_tokens=hits+misses+output,
        estimated_cny_off_peak=round(low,8),estimated_cny_peak=round(low*2,8))


def validate(root):
    manifest=verify_manifest();protocol=read(root/'protocol.json')
    require(protocol.get('api_execution') is True,'Dry runs are not experimental results')
    require(protocol.get('split')=='dev' and protocol['tasks']==manifest['splits']['dev'] and len(protocol['tasks'])==2,
            'Expected exactly the two frozen development tasks')
    require(protocol.get('local_seed')==42 and protocol.get('provider_seed_sent') is False,
            'Unexpected random seed protocol')
    for f,h in protocol['fingerprints'].items():
        require(sha(ROOT/f)==h,f'Trial source/config changed: {f}')
    for f,h in protocol['original_fingerprints'].items():
        require(sha(ROOT/f)==h,f'Original frozen source/config changed: {f}')
    batch=read(root/'batch_status.json');jobs=batch.get('completed_jobs',[])
    expected={(v,t['index']) for v in VARIANTS for t in protocol['tasks']}
    actual=[(j['variant'],j['index']) for j in jobs]
    require(batch.get('planned_jobs')==4 and len(actual)==4 and len(set(actual))==4 and set(actual)==expected,
            'Both complete paired task runs are required')
    experiments={}
    for variant in VARIANTS:
        folder=root/variant;exp=read(folder/'experiment.json');experiments[variant]=exp
        require(exp['variant']==variant and exp['split']=='dev' and exp['repeats']==1 and exp['api_execution'] is True,
                'Unexpected experiment metadata')
        require(exp['tasks']==protocol['tasks'] and exp['data_sha256']==manifest['splits']['dev_sha256'] and
                exp['upstream_commit']==manifest['upstream_commit'],'Task/data provenance mismatch')
        require(exp['source_fingerprints']==protocol['fingerprints'],'Arm source fingerprint mismatch')
        require(exp['config']==read(ROOT/'configs'/f'{variant}.json'),'Saved config differs from trial source')
        for name,h in protocol['fingerprints'].items():
            if name.startswith('scripts/'):
                require(sha(folder/('executed_'+Path(name).name))==h,'Executed source snapshot mismatch')
        for task in exp['tasks']:
            dest=folder/f"r0_task{task['index']:03d}"
            inv=read(dest/'invocation.json');status=read(dest/'status.json')
            require(inv['seed']==42 and inv['repeat']==0 and inv['task']==task,f'Invocation mismatch: {dest}')
            job=next(j for j in jobs if j['variant']==variant and j['index']==task['index'])
            require(job['status']==status['status'],f'Batch/task status mismatch: {dest}')
            require((dest/'api_calls.jsonl').exists(),f'Missing actual API ledger: {dest}')
    c,p=(experiments[v] for v in VARIANTS)
    for key in ('tasks','split','repeats','model','endpoint','data_sha256','upstream_commit','source_fingerprints'):
        require(c[key]==p[key],f'Paired arms differ in {key}')
    cfg=dict(c['config']);cfg['search_algorithm']=p['config']['search_algorithm']
    require(cfg==p['config'],'Unexpected extra experimental difference')
    require(c['config']['search_algorithm']=='mcts_cycle' and
            p['config']['search_algorithm']=='mcts_cycle_low_score','Unexpected search algorithms')
    return protocol,experiments


def widening_report(folder):
    path=folder/'low_score_audit.json'
    require(path.exists(),f'Missing widening audit: {folder}')
    audit=read(path)
    require(audit['rule']==EXPECTED_RULE,'Unexpected widening rule')
    events=audit['events'];count=audit['trigger_attempts']
    tree_files=list((folder/'full_tree').glob('full_tree_*.json'))
    nodes={}
    if len(tree_files)==1:
        def visit(node):
            nodes[node['id']]=node
            for child in node.get('children',[]):visit(child)
        visit(read(tree_files[0]))
    elif read(folder/'status.json')['status']=='completed':
        raise ValueError('Completed widening task lacks one full tree')
    require(count==len(events) and 0<=count<=2,'Per-task trigger cap violated')
    require(len({e['node_id'] for e in events})==len(events),'Node widened more than once')
    require([e['trigger'] for e in events]==list(range(1,count+1)),'Trigger sequence mismatch')
    for event in events:
        require(math.isfinite(event['original_fast_reward']) and event['original_fast_reward']<=0 and
                event['original_fast_reward']!=-1,'Invalid trigger score')
        require(event['budget_before']['allowed'] and event['budget_before']['calls_remaining']>=4 and
                event['budget_before']['tokens_remaining']>=10000,'Insufficient pre-admission reserve')
        added=event.get('new_candidates',[])
        require(len(added)<=1,'More than one added candidate in an event')
        require(all(n['action']!=event['original_action'] for n in added),'Duplicate added candidate')
        if nodes:
            require(event['node_id'] in nodes,'Audited widening node missing from tree')
            children=nodes[event['node_id']].get('children',[])
            def action_text(node):
                value=node['action']
                return value['action_str'] if isinstance(value,dict) else value
            require(any(action_text(c)==event['original_action'] for c in children),'Original candidate was removed')
            for new in added:
                require(any(c['id']==new['node_id'] and action_text(c)==new['action'] for c in children),
                        'Audited new candidate missing from tree')
                saved=nodes[new['node_id']]
                new['modeled_in_final_tree']=saved.get('state') is not None
                new['visit_count']=saved.get('visit_count',0)
        else:
            for new in added:
                new['modeled_in_final_tree']=None
                new['visit_count']=None
        if event['status']=='completed':require(len(added)==1,'Completed widening did not add exactly one candidate')
        calls=event.get('additional_api_calls')
        if calls is not None:require(0<=calls<=4,'Widening request reservation exceeded')
        tokens=event.get('additional_returned_tokens')
        if tokens is not None:require(tokens>=0,'Negative widening token delta')
    return dict(candidate_retention_verified=bool(nodes),trigger_attempts=count,new_candidates=sum(len(e.get('new_candidates',[])) for e in events),
        new_candidates_modeled=sum(n.get('modeled_in_final_tree') is True for e in events for n in e.get('new_candidates',[])),
        new_candidates_unmodeled=sum(n.get('modeled_in_final_tree') is False for e in events for n in e.get('new_candidates',[])),
        new_candidates_modeling_unknown=sum(n.get('modeled_in_final_tree') is None for e in events for n in e.get('new_candidates',[])),
        direct_widening_api_calls=sum(e.get('additional_api_calls',0) for e in events),
        direct_widening_returned_tokens=sum(e.get('additional_returned_tokens',0) for e in events),
        unaccounted_widening_events=sum('additional_api_calls' not in e for e in events),
        skips_by_reason=dict(Counter(e['reason'] for e in audit['skips'])),events=events,
        note='Direct widening deltas exclude any later rollout-path change; they are not the arm total cost difference.')


def compare(root):
    protocol,experiments=validate(root)
    from evaluate import summarize
    from deepseek_cost import estimate
    arms={}
    for variant in VARIANTS:
        folder=root/variant;summary=summarize(folder);cost=estimate(folder)
        with (folder/'per_task.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
        require(len(rows)==2,'Unexpected evaluation denominator')
        tasks=[]
        for row in rows:
            record=next(t for t in protocol['tasks'] if t['task_id']==row['task_id'])
            dest=folder/f"r0_task{record['index']:03d}";usage=ledger_report(dest/'api_calls.jsonl')
            require(usage['calls']==int(row['completed_api_calls']) and usage['input_tokens']==int(row['input_tokens']) and
                    usage['output_tokens']==int(row['output_tokens']),'Per-task usage differs from evaluation')
            widened=widening_report(dest) if variant==VARIANTS[1] else None
            tasks.append(dict(task_id=record['task_id'],index=record['index'],status=row['status'],
                symbolic_success=row['symbolic_success'].lower()=='true',
                upstream_success=row['upstream_success'].lower()=='true',evaluation_error=row['evaluation_error'],
                elapsed_seconds=float(row['elapsed_seconds']) if row['elapsed_seconds'] else None,
                usage=usage,widening=widened))
        require(sum(t['usage']['calls'] for t in tasks)==cost['completed_calls'],'Arm API ledger total mismatch')
        for field in ('input_tokens','output_tokens'):
            require(sum(t['usage'][field] for t in tasks)==summary[field],f'Arm {field} accounting mismatch')
        require(abs(sum(t['usage']['estimated_cny_peak'] for t in tasks)-cost['estimated_cny_peak'])<1e-7,
                'Per-task fees differ from full arm estimate')
        low=[t['widening'] for t in tasks if t['widening'] is not None]
        arms[variant]=dict(summary=summary,cost=cost,tasks=tasks,
            independent_successes=sum(t['symbolic_success'] for t in tasks),
            status_counts=dict(Counter(t['status'] for t in tasks)),
            budget_interruptions=sum(t['status']=='budget_exhausted' for t in tasks),
            widening_totals={k:sum(e[k] for e in low) for k in ('trigger_attempts','new_candidates','new_candidates_modeled','new_candidates_unmodeled','new_candidates_modeling_unknown',
                'direct_widening_api_calls','direct_widening_returned_tokens','unaccounted_widening_events')})
    pairs=[]
    for task in protocol['tasks']:
        c,p=(next(t for t in arms[v]['tasks'] if t['task_id']==task['task_id']) for v in VARIANTS)
        before,after=c['symbolic_success'],p['symbolic_success']
        outcome=('both_success' if before else 'gain') if after else ('loss' if before else 'both_failure')
        pairs.append(dict(task_id=task['task_id'],control_success=before,expanded_success=after,outcome=outcome,
            total_api_call_difference=p['usage']['calls']-c['usage']['calls'],
            total_returned_token_difference=p['usage']['total_tokens']-c['usage']['total_tokens'],
            total_estimated_cny_peak_difference=p['usage']['estimated_cny_peak']-c['usage']['estimated_cny_peak']))
    c,p=(arms[v] for v in VARIANTS);triggers=p['widening_totals']['trigger_attempts']
    report=dict(date='2026-11-04',split='dev',planned_paired_tasks=2,api_analysis_calls=0,
        rule=EXPECTED_RULE,arms=arms,paired_tasks=pairs,paired_counts=dict(Counter(t['outcome'] for t in pairs)),
        mechanism_triggered=triggers>0,
        call_difference=p['cost']['completed_calls']-c['cost']['completed_calls'],
        success_difference=p['independent_successes']-c['independent_successes'],
        estimated_total_cny_off_peak=sum(a['cost']['estimated_cny_off_peak'] for a in arms.values()),
        estimated_total_cny_peak=sum(a['cost']['estimated_cny_peak'] for a in arms.values()),
        source_fingerprints=protocol['fingerprints'],report_script_sha256=sha(__file__),
        limitations=['Two familiar development tasks, one run per arm, do not establish generalization or stable accuracy improvement.',
            'Local seed is 42; provider seed is not sent, so independently sampled arms can diverge before widening.',
            'Both arms retain cycle pruning, original candidate selection and backpropagation; no backtracking or cycle-specific reward change.',
            'Low-score original candidates are retained; widening is not a hard filter and the original argmax may still select them.',
            'Widening accounting measures generation/evaluation overhead only; later path changes can add or save calls separately.',
            'The 10000-token headroom check is conservative admission, not an exact prediction; the original guard checks returned usage before each next request.',
            'Fees use actual cache usage at the verified snapshot and are not provider bills. Unknown requests may incur additional charges.',
            'No test task was run in this trial; the existing test protocol/results were not changed.'])
    write(root/'low_score_comparison.json',report)
    (root/'low_score_comparison.txt').write_text(render(report))
    return report


def render(report):
    lines=['2026-11-04 低分单候选补充：两个开发样本的小额配对报告','',
        '控制组继承原CycleMCTS；实验组仅在模拟阶段对低分单候选有条件补一个候选。',
        '两组均使用同一开发任务、local seed42、3次搜索、6步深度及原预算；API未指定seed。',
        '每任务至多触发两次，每节点至多一次；原候选保留，并沿用原argmax选择/回传/剪枝。',
        '补充前预留至少4请求槽和10000已返回token余量；原逐调用guard仍是最终保护。','',
        '逐题独立回放结果与实际用量（人民币费用为估算）：']
    for variant in VARIANTS:
        arm=report['arms'][variant];cost=arm['cost']
        lines.append(variant)
        for task in arm['tasks']:
            u=task['usage'];w=task['widening']
            lines.append(f"  {task['task_id']}: 成功={task['symbolic_success']}；状态={task['status']}；API={u['calls']}；输入/输出tokens={u['input_tokens']}/{u['output_tokens']}；费用¥{u['estimated_cny_off_peak']:.8f}–{u['estimated_cny_peak']:.8f}；未知请求={u['unaccounted_calls']}。")
            if w is not None:
                lines.append(f"    触发{w['trigger_attempts']}次，新增候选{w['new_candidates']}个，其中最终树已转移/探索{w['new_candidates_modeled']}个、本轮未转移/探索{w['new_candidates_unmodeled']}个、缺树无法判断{w['new_candidates_modeling_unknown']}个；直接补候选开销{w['direct_widening_api_calls']}请求/{w['direct_widening_returned_tokens']}tokens；skip={json.dumps(w['skips_by_reason'],ensure_ascii=False)}。")
                for e in w['events']:
                    lines.append(f"    节点{e['node_id']} depth={e['depth']}，原动作={e['original_action']}（分数{e['original_fast_reward']}）；新增={json.dumps(e.get('new_candidates',[]),ensure_ascii=False)}；状态={e['status']}。")
        lines.append(f"  合计独立成功{arm['independent_successes']}/2；API {cost['completed_calls']}；输入/输出tokens {cost['input_tokens']}/{cost['output_tokens']}；估算费用¥{cost['estimated_cny_off_peak']:.8f}–{cost['estimated_cny_peak']:.8f}；预算中断{arm['budget_interruptions']}；未知请求{cost['unaccounted_calls']}。")
    pair=report['paired_counts'];w=report['arms'][VARIANTS[1]]['widening_totals']
    lines+=['',f"配对改善{pair.get('gain',0)}、退步{pair.get('loss',0)}、均成功{pair.get('both_success',0)}、均失败{pair.get('both_failure',0)}。成功数差{report['success_difference']:+d}，总调用差{report['call_difference']:+d}。",
        f"实验组补候选机制共触发{w['trigger_attempts']}次，新增{w['new_candidates']}个动作，其中最终树已转移/探索{w['new_candidates_modeled']}个；直接开销{w['direct_widening_api_calls']}次请求/{w['direct_widening_returned_tokens']}tokens。",
        '直接补候选开销仅含生成/评分；总费用差还包含后续轨迹改变，两者不能等同。',
        '机制正确触发、候选被探索、成功率提升是不同事实。state=None仅表示本轮未探索，不能据此断定该候选无用；同分时原argmax仍可能优先原候选。',
        f"本次四任务合计估算¥{report['estimated_total_cny_off_peak']:.8f}–{report['estimated_total_cny_peak']:.8f}；实际以平台账单为准。"]
    if not report['mechanism_triggered']:
        lines.append('机制未触发，不能衡量补候选的效果；两组差异只能视为独立采样/服务波动的观察。')
    else:
        lines.append('机制已触发，但仅两个开发样本各一次，不能把成功差归因于补候选或声称稳定提升。')
    lines+=['','局限：低分原动作仍可获选；必要的0分动作不被硬过滤。未加入循环回退或循环失败回传。',
        '未运行测试集。原冻结方案与历史实验保持原样。本报告不代表未见任务上的泛化结论。',
        '预算不足/账本不确定时保守拒绝补候选；token预留不是精确未来usage预测，末次响应仍可能使原累计阈值略超。',
        '完整节点事件、预留检查和逐题数据见同名JSON。离线重算此报告不调用API。']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',nargs='?',type=Path,default=ROOT/'results/low_score_dev_20261104')
    a=parser.parse_args()
    os.environ.update(environment())
    sys.path[:0]=[str(UPSTREAM),str(UPSTREAM/'demos')]
    from entry import offline_guard
    offline_guard()
    report=compare(a.root.resolve())
    print('Saved low_score_comparison.json/.txt; paired successes delta',report['success_difference'])

if __name__=='__main__':main()
