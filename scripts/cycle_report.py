"""Offline paired MCTS cycle-pruning report; success requires symbolic replay.

Usage: .venv/bin/python scripts/cycle_report.py results/cycle_dev_20261104
Costs use every actual ledger record, including interrupted tasks. Missing/failed
planned tasks remain failures in the primary denominator.
"""
import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from common import ROOT, UPSTREAM, read, write, environment

VARIANTS = ('cycle_control_3', 'cycle_pruned_3')


def state_text(state):
    if not isinstance(state, dict):
        return None
    steps = state.get('steps') or []
    if steps and isinstance(steps[-1], dict) and steps[-1].get('next_state'):
        return steps[-1]['next_state']
    return state.get('init_state')


def tree_audit(path):
    """Count repeated modeled states on a path; this is NOT a success evaluator."""
    from cycle_state import canonical_state
    if not path.exists():
        return dict(present=False)
    tree = read(path)
    root_key = canonical_state(state_text(tree.get('state')))
    blocks = {b for b, _ in root_key} if root_key is not None else None
    result = dict(present=True, root_parse_valid=root_key is not None,
                  modeled_nodes=0, unmodeled_nodes=0, parse_failures=0,
                  ancestor_revisit_nodes=0, ancestor_revisit_nodes_with_children=0, revisits=[])
    def visit(node, ancestors):
        state = node.get('state')
        if state is None:
            result['unmodeled_nodes'] += 1
            key = None
        else:
            result['modeled_nodes'] += 1
            key = canonical_state(state_text(state), blocks) if blocks is not None else None
            if key is None:
                result['parse_failures'] += 1
        if key is not None:
            matches = [ident for ident, previous in ancestors if previous == key]
            if matches:
                result['ancestor_revisit_nodes'] += 1
                result['ancestor_revisit_nodes_with_children'] += bool(node.get('children'))
                result['revisits'].append(dict(node_id=node['id'], ancestor_id=matches[-1]))
        ancestors = ancestors + [(node['id'], key)] if key is not None else ancestors
        for child in node.get('children') or []:
            visit(child, ancestors)
    visit(tree, [])
    return result


def per_task_rows(folder):
    rows = []
    with (folder/'per_task.csv').open(newline='') as stream:
        for record in csv.DictReader(stream):
            record['repeat'] = int(record['repeat'])
            record['symbolic_success'] = record['symbolic_success'].lower() == 'true'
            record['upstream_success'] = record['upstream_success'].lower() == 'true'
            for name in ('completed_api_calls', 'unaccounted_api_calls', 'input_tokens', 'output_tokens'):
                record[name] = int(record[name])
            record['elapsed_seconds'] = float(record['elapsed_seconds']) if record['elapsed_seconds'] else None
            rows.append(record)
    return rows


def arm_report(folder):
    from evaluate import summarize
    from deepseek_cost import estimate
    experiment = read(folder/'experiment.json')
    summary = summarize(folder)
    cost = estimate(folder)
    rows = per_task_rows(folder)
    tasks = {t['task_id']: t for t in experiment['tasks']}
    status_counts = dict(Counter(r['status'] for r in rows))
    for row in rows:
        task = tasks[row['task_id']]
        dest = folder/f"r{row['repeat']}_task{task['index']:03d}"
        files = list((dest/'full_tree').glob('full_tree_*.json'))
        if len(files) > 1:
            raise ValueError(f'Multiple full trees in single-task directory: {dest}')
        row['tree'] = tree_audit(files[0]) if files else dict(present=False)
        audit_file = dest/'cycle_audit.json'
        if audit_file.exists():
            audit = read(audit_file)
            events = audit.get('pruned_nodes', [])
            row['pruning'] = dict(audit_present=True,
                root_parse_valid=audit.get('root_parse_valid'),
                pruned_nodes=len({e['node_id'] for e in events}),
                parse_failures=len(set(audit.get('parse_failure_node_ids', []))),
                events=events)
        else:
            row['pruning'] = dict(audit_present=False, pruned_nodes=0, parse_failures=None)
    elapsed = [r['elapsed_seconds'] for r in rows if r['elapsed_seconds'] is not None]
    trees = [r['tree'] for r in rows if r['tree']['present']]
    audits = [r['pruning'] for r in rows if r['pruning']['audit_present']]
    result = dict(variant=experiment['variant'], summary=summary, cost=cost,
        status_counts=status_counts, budget_interruptions=status_counts.get('budget_exhausted', 0),
        independent_successes=sum(r['symbolic_success'] for r in rows),
        upstream_claimed_successes=sum(r['upstream_success'] for r in rows),
        mean_task_elapsed_seconds=statistics.mean(elapsed) if elapsed else None,
        sum_task_elapsed_seconds=sum(elapsed),
        elapsed_note='Task times overlap when workers run concurrently; their sum is not batch wall time.',
        full_trees_available=len(trees), full_trees_missing=len(rows)-len(trees),
        modeled_nodes=sum(t['modeled_nodes'] for t in trees),
        llm_state_ancestor_revisit_nodes=sum(t['ancestor_revisit_nodes'] for t in trees),
        ancestor_revisit_nodes_with_children=sum(t['ancestor_revisit_nodes_with_children'] for t in trees),
        tree_state_parse_failures=sum(t['parse_failures'] for t in trees),
        cycle_audits_available=len(audits),
        pruned_nodes=sum(a['pruned_nodes'] for a in audits),
        cycle_parse_failure_nodes=sum(a['parse_failures'] for a in audits),
        invalid_root_audits=sum(a['root_parse_valid'] is False for a in audits), tasks=rows)
    if cost['completed_calls'] != sum(r['completed_api_calls'] for r in rows):
        raise ValueError('Ledger total differs from per-task evaluation accounting')
    if cost['input_tokens'] != summary['input_tokens'] or cost['output_tokens'] != summary['output_tokens']:
        raise ValueError('Cost ledger tokens differ from evaluation summary')
    return experiment, result


def compare(root):
    experiments = {}
    arms = {}
    for variant in VARIANTS:
        experiments[variant], arms[variant] = arm_report(root/variant)
    control, pruned = (experiments[v] for v in VARIANTS)
    for field in ('split', 'tasks', 'repeats', 'model', 'endpoint', 'data_sha256', 'upstream_commit'):
        if control[field] != pruned[field]:
            raise ValueError(f'Paired arms differ in {field}')
    first = { (r['repeat'], r['task_id']):r for r in arms[VARIANTS[0]]['tasks'] }
    second = { (r['repeat'], r['task_id']):r for r in arms[VARIANTS[1]]['tasks'] }
    if set(first) != set(second):
        raise ValueError('Unmatched planned task keys')
    pairings = []
    for key, before in first.items():
        after = second[key]
        a, b = before['symbolic_success'], after['symbolic_success']
        outcome = ('both_success' if a else 'gain') if b else ('loss' if a else 'both_failure')
        pairings.append(dict(repeat=key[0],task_id=key[1],outcome=outcome,
            control_success=a,pruned_success=b,control_status=before['status'],pruned_status=after['status'],
            api_call_difference=after['completed_api_calls']-before['completed_api_calls'],
            token_difference=(after['input_tokens']+after['output_tokens'])-(before['input_tokens']+before['output_tokens'])))
    count = len(pairings)
    pairing_counts = dict(Counter(p['outcome'] for p in pairings))
    net = pairing_counts.get('gain',0)-pairing_counts.get('loss',0)
    report = dict(split=control['split'],planned_paired_tasks=count,
        pricing_verified_date=control.get('pricing_verified_date'),
        primary_metric='Independent symbolic replay successes / all planned tasks; incomplete/missing are failures.',
        arms=arms,paired_counts=pairing_counts,paired_tasks=pairings,
        net_success_gain=net,success_rate_difference=net/count if count else None,
        estimated_total_cny_off_peak=sum(a['cost']['estimated_cny_off_peak'] for a in arms.values()),
        estimated_total_cny_peak=sum(a['cost']['estimated_cny_peak'] for a in arms.values()),
        limitations=[
            'One run per arm/task: results do not measure run-to-run variance or prove statistical improvement.',
            'These paired tasks use the same frozen examples and local random seed; model sampling and service behavior can still vary.',
            'Ancestor revisits describe parseable LLM state text in saved trees, not verified ground-truth transitions.',
            'Missing trees and parse failures limit cycle diagnostics. Invalid/incomplete state text is conservatively not pruned.',
            'Fees estimate actual recorded successful ledger usage at the verified rate snapshot, including prompt cache usage; they are not provider bills.',
            'Unknown/errored ledger requests may incur additional charges; cache and concurrent execution can affect costs/latency.',
            'Development observations tune the method; held-out test observations must not be used for further tuning.'])
    write(root/'cycle_comparison.json', report)
    lines = [f"MCTS 路径循环剪枝对照：{control['split']}",
             f"计划配对任务：{count}；每组分母均为全部计划任务。",
             f"价格核对日期：{report['pricing_verified_date']}；费用为人民币估算，非账单。",'']
    for variant in VARIANTS:
        a=arms[variant];c=a['cost']
        lines.extend([variant,
            f"  独立回放成功：{a['independent_successes']}/{count} ({a['summary']['symbolic_success_rate']:.1%})；完成任务：{a['summary']['completed_tasks']}/{count}",
            f"  API 成功请求：{c['completed_calls']}；未记账/不确定请求：{c['unaccounted_calls']}",
            f"  输入/输出 tokens：{c['input_tokens']}/{c['output_tokens']}；合计 {c['input_tokens']+c['output_tokens']}",
            f"  估算费用：¥{c['estimated_cny_off_peak']:.8f}–{c['estimated_cny_peak']:.8f}",
            f"  预算中断：{a['budget_interruptions']}；状态：{json.dumps(a['status_counts'],ensure_ascii=False)}",
            f"  保存树：{a['full_trees_available']}；缺失树：{a['full_trees_missing']}；LLM 状态祖先重访节点：{a['llm_state_ancestor_revisit_nodes']}",
            f"  重访后仍有子节点：{a['ancestor_revisit_nodes_with_children']}",
            f"  循环审计：{a['cycle_audits_available']}；剪枝节点：{a['pruned_nodes']}；审计解析失败节点：{a['cycle_parse_failure_nodes']}；无效根状态：{a['invalid_root_audits']}",
            f"  树中无法规范化的已转移节点：{a['tree_state_parse_failures']}；平均任务耗时：{a['mean_task_elapsed_seconds']} 秒（并发任务耗时不可相加当作批次耗时）",''])
    lines.extend([f"配对 gains={pairing_counts.get('gain',0)}，losses={pairing_counts.get('loss',0)}，both_success={pairing_counts.get('both_success',0)}，both_failure={pairing_counts.get('both_failure',0)}。",
        f"成功数净变化：{net:+d}；成功率差：{report['success_rate_difference']:+.1%}。",
        f"两组合计估算：¥{report['estimated_total_cny_off_peak']:.8f}–{report['estimated_total_cny_peak']:.8f}。",'',
        '单次运行不能证明稳定提升；所有未完成任务都计入失败，成功仅由独立符号回放认定。',
        '循环指标来自保存的 LLM 状态文本，受解析失败和树缺失影响；剪枝节点不会被标成解。',
        '缓存命中和服务状态会影响费用/延迟；请求错误仍可能计费，最终以平台账单为准。',
        '开发集用于确定方案，测试集结果不得用于后续调参。'])
    (root/'cycle_comparison.txt').write_text('\n'.join(lines)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    os.environ.update(environment())
    sys.path[:0]=[str(UPSTREAM),str(UPSTREAM/'demos')]
    from entry import offline_guard
    offline_guard()
    report=compare(args.root.resolve())
    print(f"Saved comparison: {args.root}/cycle_comparison.json and .txt")
    print(f"Paired tasks={report['planned_paired_tasks']}, net success gain={report['net_success_gain']:+d}")

if __name__=='__main__':main()
