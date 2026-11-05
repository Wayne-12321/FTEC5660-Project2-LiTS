"""Offline aggregation of three frozen paired tests on the same 20 tasks.

Only reporting is added: frozen experiment sources, configuration and data are
never changed. Missing rounds or task result directories are rejected. Budget
interruptions remain failures, and their ledger usage remains in the costs.
"""
import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from common import ROOT, UPSTREAM, environment, read, sha, verify_manifest, write

VARIANTS = ('cycle_control_3', 'cycle_pruned_3')
DEFAULT_ROOTS = [ROOT/'results'/name for name in (
    'cycle_test_20261104', 'cycle_test_20261104_r2', 'cycle_test_20261104_r3')]
PROTOCOL = ROOT/'reports/cycle_protocol_20261104.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_inputs(roots):
    """Reject incomplete or incomparable runs before refreshing any reports."""
    require(len(roots) == 3 and len(set(roots)) == 3, 'Exactly three distinct rounds are required')
    protocol = read(PROTOCOL)
    manifest = verify_manifest()
    expected_tasks = manifest['splits']['test']
    require(len(expected_tasks) == 20, 'Expected frozen 20-task test set')
    for name, expected in protocol['fingerprints'].items():
        require(sha(ROOT/name) == expected, f'Frozen local source/config changed: {name}')
    baseline = {}
    for round_no, root in enumerate(roots, 1):
        require(root.is_dir(), f'Missing round directory: {root}')
        batch_path = root/'batch_status.json'
        require(batch_path.exists(), f'Missing batch status: {root}')
        batch = read(batch_path)
        jobs = batch.get('completed_jobs', [])
        expected_keys = {(v, t['index']) for v in VARIANTS for t in expected_tasks}
        job_keys = [(j['variant'], j['index']) for j in jobs]
        require(batch.get('planned_jobs') == 40 and len(jobs) == 40 and
                len(set(job_keys)) == 40 and set(job_keys) == expected_keys,
                f'Incomplete or duplicate paired jobs in round {round_no}')
        by_job = {(j['variant'], j['index']):j for j in jobs}
        for variant in VARIANTS:
            folder = root/variant
            exp = read(folder/'experiment.json')
            require(exp.get('variant') == variant and exp.get('split') == 'test' and
                    exp.get('api_execution') is True and exp.get('repeats') == 1,
                    f'Invalid test experiment in {folder}')
            require(exp['tasks'] == expected_tasks, f'Different test tasks/order in {folder}')
            require(exp['source_fingerprints'] == protocol['fingerprints'],
                    f'Frozen fingerprints differ in {folder}')
            require(exp['model'] == protocol['model'], f'Model differs in {folder}')
            require(exp['data_sha256'] == manifest['splits']['test_sha256'] and
                    exp['data_sha256'] == sha(ROOT/'data/test.json') and
                    exp['upstream_commit'] == manifest['upstream_commit'],
                    f'Data/upstream provenance differs in {folder}')
            require(exp['config'] == read(ROOT/'configs'/f'{variant}.json'),
                    f'Configuration differs from freeze in {folder}')
            if round_no == 1:
                baseline[variant] = exp
            else:
                for field in ('config', 'tasks', 'repeats', 'model', 'endpoint', 'split',
                              'upstream_commit', 'data_sha256', 'source_fingerprints',
                              'pricing_verified_date'):
                    require(exp.get(field) == baseline[variant].get(field),
                            f'Round {round_no} differs in {variant}.{field}')
            for task in expected_tasks:
                dest = folder/f"r0_task{task['index']:03d}"
                require(dest.is_dir() and (dest/'status.json').exists(),
                        f'Missing task result/status: {dest}')
                require((dest/'invocation.json').exists(), f'Missing invocation: {dest}')
                invoke = read(dest/'invocation.json')
                require(invoke.get('seed') == 42 and invoke.get('repeat') == 0 and
                        invoke.get('task') == task, f'Local seed/task differs: {dest}')
                status = read(dest/'status.json')
                require(status['status'] == by_job[(variant, task['index'])]['status'],
                        f'Batch/task status differs: {dest}')
                require(status['status'] not in ('not_run', 'startup_error'),
                        f'Task not actually run: {dest}')
                ledger = dest/'api_calls.jsonl'
                require(ledger.exists(), f'Missing API ledger: {dest}')
    return protocol, expected_tasks


def mean_sd(values):
    return dict(values=values, mean=statistics.mean(values),
                sample_sd=statistics.stdev(values) if len(values)>1 else None)


def direction(values):
    return dict(positive=sum(v > 0 for v in values), zero=sum(v == 0 for v in values),
                negative=sum(v < 0 for v in values))


def aggregate(roots):
    protocol, tasks = validate_inputs(roots)
    from cycle_report import compare
    rounds = []
    for i, root in enumerate(roots, 1):
        raw = compare(root)
        require(raw['planned_paired_tasks'] == 20, 'Unexpected evaluation denominator')
        arms = {}
        for variant in VARIANTS:
            a = raw['arms'][variant]
            cost = a['cost']
            require(a['summary']['planned_tasks'] == 20 and len(a['tasks']) == 20,
                    'Unexpected arm evaluation denominator')
            arms[variant] = dict(successes=a['independent_successes'], denominator=20,
                success_rate=a['summary']['symbolic_success_rate'],
                completed_tasks=a['summary']['completed_tasks'], status_counts=a['status_counts'],
                calls=cost['completed_calls'], unaccounted_calls=cost['unaccounted_calls'],
                input_tokens=cost['input_tokens'], output_tokens=cost['output_tokens'],
                total_tokens=cost['input_tokens']+cost['output_tokens'],
                cny_off_peak=cost['estimated_cny_off_peak'], cny_peak=cost['estimated_cny_peak'],
                budget_interruptions=a['budget_interruptions'], pruned_nodes=a['pruned_nodes'],
                ancestor_revisits=a['llm_state_ancestor_revisit_nodes'],
                revisits_with_children=a['ancestor_revisit_nodes_with_children'],
                parse_failure_nodes=a['tree_state_parse_failures'],
                cycle_audits_available=a['cycle_audits_available'],
                mean_task_elapsed_seconds=a['mean_task_elapsed_seconds'])
        c, p = (arms[v] for v in VARIANTS)
        rounds.append(dict(round=i, root=str(root.relative_to(ROOT)) if root.is_relative_to(ROOT) else str(root),
            arms=arms, paired_counts=raw['paired_counts'], paired_tasks=raw['paired_tasks'],
            net_success_gain=raw['net_success_gain'], success_rate_difference=raw['success_rate_difference'],
            call_reduction=1-p['calls']/c['calls'] if c['calls'] else None,
            token_reduction=1-p['total_tokens']/c['total_tokens'] if c['total_tokens'] else None,
            cost_reduction=1-p['cny_peak']/c['cny_peak'] if c['cny_peak'] else None,
            estimated_total_cny_off_peak=raw['estimated_total_cny_off_peak'],
            estimated_total_cny_peak=raw['estimated_total_cny_peak']))
    totals = {}
    for variant in VARIANTS:
        arms = [r['arms'][variant] for r in rounds]
        additive = ('successes', 'completed_tasks', 'calls', 'unaccounted_calls', 'input_tokens',
                    'output_tokens', 'total_tokens', 'cny_off_peak', 'cny_peak', 'budget_interruptions',
                    'pruned_nodes', 'ancestor_revisits', 'revisits_with_children', 'parse_failure_nodes')
        totals[variant] = {name:sum(a[name] for a in arms) for name in additive}
        totals[variant].update(denominator=60, unique_tasks=20,
            pooled_success_rate=totals[variant]['successes']/60,
            round_statistics={name:mean_sd([a[name] for a in arms]) for name in (
                'success_rate', 'successes', 'calls', 'total_tokens', 'cny_off_peak', 'cny_peak', 'pruned_nodes')})
    pairs = Counter()
    for r in rounds:pairs.update(r['paired_counts'])
    per_task=[]
    for task in tasks:
        row=dict(task_id=task['task_id'], index=task['index'])
        outcomes=[]
        for variant, flag in [(VARIANTS[0], 'control_success'), (VARIANTS[1], 'pruned_success')]:
            values=[]; statuses=[]
            for r in rounds:
                matches=[p for p in r['paired_tasks'] if p['task_id']==task['task_id']]
                require(len(matches)==1, 'Task pairing missing/duplicated')
                values.append(bool(matches[0][flag]));statuses.append(matches[0][flag.replace('_success','_status')])
            row[variant]=dict(successes=sum(values), denominator=3, by_round=values, statuses=statuses)
        for r in rounds:outcomes.append(next(p['outcome'] for p in r['paired_tasks'] if p['task_id']==task['task_id']))
        row['paired_outcomes']=outcomes
        row['pruned_minus_control_successes']=row[VARIANTS[1]]['successes']-row[VARIANTS[0]]['successes']
        per_task.append(row)
    task_consistency={}
    for variant in VARIANTS:
        counts=[row[variant]['successes'] for row in per_task]
        task_consistency[variant]=dict(all_3_success=sum(n==3 for n in counts),
            all_3_failure=sum(n==0 for n in counts),mixed_outcomes=sum(0<n<3 for n in counts))
    task_success_difference_distribution={str(n):sum(
        row['pruned_minus_control_successes']==n for row in per_task) for n in range(-3,4)}
    call_reductions=[r['call_reduction'] for r in rounds]
    gains=[r['net_success_gain'] for r in rounds]
    c,p=(totals[v] for v in VARIANTS)
    report=dict(date='2026-11-04', protocol=str(PROTOCOL.relative_to(ROOT)),
        round_count=3, unique_tasks=20, observations_per_arm=60,
        repeated_observations_not_independent_tasks=True,
        local_seed_each_round=42, api_seed_sent=False,
        frozen_fingerprints=protocol['fingerprints'], model=protocol['model'],
        rounds=rounds, totals=totals, paired_counts=dict(pairs), per_task=per_task,
        task_consistency=task_consistency,
        task_success_difference_distribution=task_success_difference_distribution,
        stability=dict(net_success_gains=gains, accuracy_difference_direction_counts=direction(gains),
            call_reduction_each_round=call_reductions,
            calls_reduced_every_round=all(v is not None and v>0 for v in call_reductions),
            cost_reduced_every_round=all(r['cost_reduction'] is not None and r['cost_reduction']>0 for r in rounds),
            pooled_call_reduction=1-p['calls']/c['calls'],
            pooled_token_reduction=1-p['total_tokens']/c['total_tokens'],
            pooled_cost_reduction=1-p['cny_peak']/c['cny_peak'],
            success_rate_differences=mean_sd([r['success_rate_difference'] for r in rounds])),
        spending=dict(new_rounds=[2,3], new_calls=sum(r['arms'][v]['calls'] for r in rounds[1:] for v in VARIANTS),
            new_cny_off_peak=sum(r['estimated_total_cny_off_peak'] for r in rounds[1:]),
            new_cny_peak=sum(r['estimated_total_cny_peak'] for r in rounds[1:]),
            all_test_calls=sum(r['arms'][v]['calls'] for r in rounds for v in VARIANTS),
            all_test_cny_off_peak=sum(r['estimated_total_cny_off_peak'] for r in rounds),
            all_test_cny_peak=sum(r['estimated_total_cny_peak'] for r in rounds),
            scope='Test runs only; development and earlier diagnostic trials excluded.'),
        limitations=[
            'The same 20 consecutive six-step tasks are repeated three times: 60 observations per arm are not 60 independent tasks.',
            'Python/NumPy/Torch local seed stays 42; no provider API seed is sent. Repetition captures service/model sampling variation under that protocol.',
            'Three rounds provide descriptive stability only, not evidence of causal or statistically significant accuracy improvement.',
            'Control is engineering-fixed FrontierMCTS/UniquePolicy, not unmodified upstream LiTS.',
            'Costs are estimates from recorded successful usage, including prompt cache effects; unaccounted requests may also be charged.',
            'The response model is an alias; provider weight snapshot is unknown. Parallel scheduling/cache/service conditions can affect latency and cost.',
            'Failed or budget-interrupted tasks remain failures in the 20-task round denominator. No post-test tuning was performed.'])
    return report


def render(report):
    lines=['2026-11-04 MCTS 路径循环剪枝：三轮重复验证报告','',
        '方案、配置及20个冻结测试样本不变；原一轮加新两轮，各组每轮运行20任务。',
        '每轮本地随机种子均为42，未给DeepSeek API发送seed。共60次/组重复观测，不是60个独立样本。',
        '控制组是工程修复版FrontierMCTS；剪枝组只增加祖先状态循环剪枝。','',
        '逐轮结果（成功须独立动作回放；费用为人民币估算）：']
    for r in report['rounds']:
        lines.append(f"第{r['round']}轮：")
        for v in VARIANTS:
            a=r['arms'][v]
            lines.append(f"  {v}: 成功{a['successes']}/20 ({a['success_rate']:.1%})；API {a['calls']}次；tokens {a['total_tokens']}；费用¥{a['cny_off_peak']:.8f}–{a['cny_peak']:.8f}；剪枝{a['pruned_nodes']}；预算中断{a['budget_interruptions']}；未记账请求{a['unaccounted_calls']}。")
        lines.append(f"  配对改善{r['paired_counts'].get('gain',0)}、退步{r['paired_counts'].get('loss',0)}；净成功数变化{r['net_success_gain']:+d}；调用下降{r['call_reduction']:.1%}、费用下降{r['cost_reduction']:.1%}。")
    lines+=['','三轮汇总：']
    for v in VARIANTS:
        a=report['totals'][v];s=a['round_statistics']
        lines.append(f"{v}: 成功{a['successes']}/60 ({a['pooled_success_rate']:.1%})；逐轮成功率均值{s['success_rate']['mean']:.1%}，样本SD {100*s['success_rate']['sample_sd']:.2f} 个百分点；调用总计{a['calls']}，逐轮均值{s['calls']['mean']:.2f}、样本SD {s['calls']['sample_sd']:.2f}；tokens总计{a['total_tokens']}；费用¥{a['cny_off_peak']:.8f}–{a['cny_peak']:.8f}。")
        lines.append(f"  剪枝总数{a['pruned_nodes']}；重访后仍有子节点{a['revisits_with_children']}；预算中断{a['budget_interruptions']}；未记账请求{a['unaccounted_calls']}。")
    st=report['stability'];sp=report['spending']
    lines.append(f"配对结果合计：{json.dumps(report['paired_counts'],ensure_ascii=False)}；三轮净成功数变化：{st['net_success_gains']}。")
    lines.append(f"三轮合计调用下降{st['pooled_call_reduction']:.1%}、tokens下降{st['pooled_token_reduction']:.1%}、费用下降{st['pooled_cost_reduction']:.1%}。")
    if st['calls_reduced_every_round']:
        lines.append('调用节省在三轮中均出现，支持本测试集内重复观测下的效率收益；仍不能保证对其他任务或未来服务稳定成立。')
    else:
        lines.append('调用节省未在全部三轮中出现，效率收益在本重复实验中不一致。')
    d=st['accuracy_difference_direction_counts']
    if d['positive']==3:
        lines.append('三轮成功率差均为正，但仅20个相同任务、三轮重复，不能据此声称统计显著或因果性的准确率提高。')
    elif d['negative']==3:
        lines.append('三轮成功率差均为负；本重复实验没有支持准确率收益。')
    elif d['positive'] and d['negative']:
        lines.append('成功率差在不同轮次有正有负，准确率提升方向不一致。')
    else:
        lines.append('成功率差包含持平轮次，尚不能认定稳定的准确率提升。')
    lines+=['','相同20任务的跨轮一致性（不是独立新样本）：']
    for variant in VARIANTS:
        tc=report['task_consistency'][variant]
        lines.append(f"  {variant}: 三轮均成功{tc['all_3_success']}个，三轮均失败{tc['all_3_failure']}个，成功/失败混合{tc['mixed_outcomes']}个。")
    lines.append('逐任务成功次数差（剪枝−控制，范围−3至3）的任务数分布：'+json.dumps(report['task_success_difference_distribution'],ensure_ascii=False))
    lines+=['',f"新增两轮：{sp['new_calls']}次API请求，估算¥{sp['new_cny_off_peak']:.8f}–{sp['new_cny_peak']:.8f}。",
        f"三轮测试合计：{sp['all_test_calls']}次请求，估算¥{sp['all_test_cny_off_peak']:.8f}–{sp['all_test_cny_peak']:.8f}；不含开发与早期诊断费用。",
        '费用受缓存命中/并发顺序影响；未知请求可能另外收费，实际以平台账单为准。','',
        '逐样本三轮成功次数（控制组 / 剪枝组）：']
    for row in report['per_task']:
        lines.append(f"  {row['task_id']}: {row[VARIANTS[0]]['successes']}/3 / {row[VARIANTS[1]]['successes']}/3；{row['paired_outcomes']}")
    lines+=['','完整明细与每轮均值/样本SD见同名JSON。未完成或预算中断任务均计失败；未改变测试集或用测试结果调参。']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots',nargs=3,type=Path,default=DEFAULT_ROOTS)
    parser.add_argument('--output-prefix',type=Path,default=ROOT/'reports/cycle_stability_20261104')
    args=parser.parse_args()
    os.environ.update(environment())
    sys.path[:0]=[str(UPSTREAM),str(UPSTREAM/'demos')]
    from entry import offline_guard
    offline_guard()
    report=aggregate([p.resolve() for p in args.roots])
    prefix=args.output_prefix.resolve()
    write(prefix.with_suffix('.json'),report)
    prefix.with_suffix('.txt').write_text(render(report))
    print('Saved:',prefix.with_suffix('.json'),prefix.with_suffix('.txt'))

if __name__=='__main__':main()
