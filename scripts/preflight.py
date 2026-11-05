"""Network-disabled data, configuration, component and evaluator checks."""
import os
import subprocess
import sys
from pathlib import Path
from common import ROOT, UPSTREAM, DATA, read, write, environment, dataset_kwargs, cli_args

def main():
    sys.path[:0]=[str(UPSTREAM),str(UPSTREAM/'demos')]
    os.environ.update(environment())
    from entry import offline_guard
    offline_guard()
    import socket
    try:
        socket.create_connection(('api.deepseek.com',443))
        raise AssertionError('Network guard failed')
    except RuntimeError as e:
        assert 'OFFLINE CHECK' in str(e)
    from lits_benchmark.blocksworld import load_blocksworld, BlocksWorldTransition
    import lits_fixes
    from lits.config import ExperimentConfig
    from lits.components.factory import create_components
    from lits.lm.base import LanguageModel
    from lits.structures.env_grounded import EnvState, EnvAction, EnvStep
    from lits.agents.tree.node import MCTSNode
    from symbolic import replay, reference_actions
    from evaluate import summarize

    # No credentials, clients or weights are created by this sentinel model.
    class NeverCall(LanguageModel):
        def __init__(self):
            self.model_name='offline-sentinel'
            self.sys_prompt=None
        def __call__(self,*a,**kw):
            raise AssertionError('Preflight must never invoke a model')

    baseline=read(ROOT/'configs/baseline_10.json')
    modified=read(ROOT/'configs/modified_3.json')
    b=__import__('copy').deepcopy(baseline)
    b['search_args']['n_iters']=3
    assert b==modified, 'Unexpected extra experimental variable'
    report={'network_disabled':True,'paid_api_calls':0,'checks':[]}
    for name,cfg in [('baseline_10',baseline),('modified_3',modified)]:
        config=ExperimentConfig(dataset='blocksworld',policy='blocksworld_unique',search_algorithm='mcts_frontier',policy_model_name='openai/offline-placeholder',
            eval_model_name='openai/offline-placeholder',search_args=cfg['search_args'],
            component_args={'max_new_tokens':cfg['max_output_tokens']})
        search=config.create_search_config()
        search.verify()
        world,policy,reward=create_components(task_type='env_grounded',task_name='blocksworld',
            base_model=NeverCall(),eval_base_model=NeverCall(),terminal_model=None,tool_use_spec=None,config=config)
        assert policy.usr_prompt_spec is not None
        assert policy.__class__.__name__ == 'UniquePolicy'
        resolved=dict(search=search.to_dict(), policy_temperature=policy.temperature,
            reward_alpha=reward.reward_alpha, goal_reached_reward=reward.goal_reached_reward,
            reward_temperature=0.6, transition_temperature=1e-6)
        write(ROOT/'reports'/f'{name}_resolved.json',resolved)
        report['checks'].append(f'{name}: config instantiated, verified, components constructed')
        cmd=[sys.executable,str(ROOT/'scripts/entry.py'),'search',*cli_args(cfg,'test',0,
            ROOT/'reports/unused','openai/offline-placeholder',dry=True)]
        with (ROOT/'reports'/f'{name}_dry_run.log').open('w') as log:
            subprocess.run(cmd,env=environment(),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        report['checks'].append(f'{name}: official CLI dry-run passed')

    full=load_blocksworld(config_file=str(DATA/'bw_data_bw_config.yaml'),
        domain_file=str(DATA/'bw_data_generated_domain.pddl'),data_file=str(DATA/'bw_data_step_6.json'))
    raw=read(DATA/'bw_data_step_6.json')
    for example,record in zip(full,raw):
        assert replay(example['init_state_str'],reference_actions(record[1]),example['query_or_goals'])
        assert BlocksWorldTransition.generate_actions(example['init_state_str'])
    report['checks'].append(f'{len(full)} source tasks loaded; every reference plan passed independent replay')
    manifest=read(ROOT/'data/manifest.json')
    dev_ids={r['task_id'] for r in manifest['splits']['dev']}
    test_ids={r['task_id'] for r in manifest['splits']['test']}
    assert len(dev_ids)==2 and len(test_ids)==20 and not dev_ids & test_ids
    report['checks'].append('Frozen development/test sets disjoint (2/20)')

    # Fixture deliberately reports a goal state with no supporting actions:
    # upstream goal check accepts it; independent replay must reject it.
    fixture=ROOT/'reports/evaluator_fixture'
    task=manifest['splits']['dev'][0]
    example=load_blocksworld(**dataset_kwargs('dev'))[0]
    write(fixture/'experiment.json',dict(config=baseline,variant='FIXTURE_NOT_AN_EXPERIMENT',
        split='dev',repeats=3,tasks=[task],api_execution=False,fixture=True))
    for repeat in range(2):
        dest=fixture/f'r{repeat}_task000'
        state=EnvState(init_state=example['init_state_str'])
        if repeat==0:
            for action in reference_actions(raw[0][1]):
                state.append(EnvStep(action=EnvAction(action),next_state=example['query_or_goals']))
        else:
            state.append(EnvStep(action=EnvAction('put down the purple block'),next_state=example['query_or_goals']))
        node=MCTSNode(state,EnvAction('fixture'),cross_rollout_q_func=max,is_terminal=True)
        node.cum_rewards=[100.0]
        write(dest/'terminal_nodes/terminal_nodes_0.json',dict(query=example['query_or_goals'],query_idx=0,terminal_nodes=[node.to_dict()]))
        write(dest/'status.json',dict(status='completed',elapsed_seconds=0.0))
        write(dest/'config.json',dict(dataset='blocksworld',policy_model_name='openai/offline-placeholder',
            import_modules=['lits_benchmark.blocksworld']))
    summary=summarize(fixture)
    assert summary['planned_tasks']==3 and summary['completed_tasks']==2
    assert summary['symbolic_success_rate']==1/3
    assert summary['upstream_success_rate']==2/3
    with (fixture/'official_eval.log').open('w') as log:
        subprocess.run([sys.executable,str(ROOT/'scripts/entry.py'),'eval','--result_dir',
            str(fixture/'r0_task000')],env=environment(),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    report['checks'].append('Official evaluator passed synthetic fixture; independent replay rejects illegal plan; missing task stays in denominator')
    write(ROOT/'reports/preflight.json',report)
    print(__import__('json').dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__':
    main()
