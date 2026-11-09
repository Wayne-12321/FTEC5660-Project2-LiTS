"""Export compact, independently checked presentation trajectories from saved runs.

This reads existing measured artifacts only. It never imports a model client,
starts a search, reads reference plans, or writes frozen run artifacts.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from symbolic import positions, apply, replay


def read(path):
    return json.loads(path.read_text())


def source(path):
    return {'path': str(path.relative_to(ROOT)),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def action(node):
    raw = node['action']
    return raw['action_str'] if isinstance(raw, dict) else raw


def state_text(node):
    state = node['state']
    assert state is not None, 'Only already modeled nodes may be replayed'
    return state['steps'][-1]['next_state'] if state['steps'] else state['init_state']


def index_tree(tree):
    nodes, paths = {}, {}
    def walk(node, path):
        assert node['id'] not in nodes
        nodes[node['id']] = node
        paths[node['id']] = path + [node]
        for child in node['children']:
            walk(child, path + [node])
    walk(tree, [])
    return nodes, paths


def export_example(arm, index, example_id, label, cycle_node=None):
    folder = ROOT / 'results/cycle_test_20261104_r3' / arm / f'r0_task{index:03d}'
    paths_used = [folder / 'full_tree' / f'full_tree_{index}.json',
                  folder / 'terminal_nodes' / f'terminal_nodes_{index}.json',
                  folder / 'status.json', folder / 'api_calls.jsonl']
    tree, terminals = (read(path) for path in paths_used[:2])
    status = read(paths_used[2])
    assert status['status'] == 'completed'
    assert terminals['query_idx'] == index
    nodes, paths = index_tree(tree)
    audit_file = folder / 'cycle_audit.json'
    cuts = {}
    if audit_file.exists():
        paths_used.append(audit_file)
        cuts = {row['node_id']: row for row in read(audit_file)['pruned_nodes']}
    goal_text = terminals['query']
    goal_relations = re.findall(r'the (\w+) block is on top of the (\w+) block', goal_text)
    assert goal_relations
    initial = tree['state']['init_state']
    initial_positions = positions(initial)
    assert initial_positions and 'hand' not in initial_positions.values()

    # The same selected-terminal rule as the measured evaluator, not a gold plan.
    selected = None
    if terminals['terminal_nodes']:
        selected = max(terminals['terminal_nodes'],
                       key=lambda node: max(node['cum_rewards']) if node['cum_rewards'] else -float('inf'))
    run_success = bool(selected and replay(initial, [step['action'] for step in selected['state']['steps']], goal_text))

    if cycle_node is None:
        assert selected is not None and run_success
        end = selected['id']
        trajectory = paths[end]
        assert nodes[end]['is_terminal']
        assert [action(n) for n in trajectory[1:]] == [s['action'] for s in selected['state']['steps']]
        assert len(trajectory) == 7, 'Presentation success should contain six logged actions'
    else:
        assert cycle_node in cuts
        end = cycle_node
        trajectory = paths[end]
        assert not nodes[end]['children'] and not nodes[end]['is_terminal']
        checkpoint = folder / 'checkpoints' / f'{index}_result.json'
        assert [n['id'] for n in read(checkpoint)] == [n['id'] for n in trajectory]
        paths_used.append(checkpoint)
        assert not run_success

    steps, current = [], dict(initial_positions)
    by_node_step = {node['id']: i for i, node in enumerate(trajectory)}
    for i, node in enumerate(trajectory):
        if i:
            apply(current, action(node))
        model_positions = positions(state_text(node))
        assert current == model_positions, f'Modeled state differs from legal replay at node {node["id"]}'
        assert set(current) == set(initial_positions)
        goal = all(current.get(block) == support for block, support in goal_relations)
        cut = cuts.get(node['id'])
        ancestor_node = cut['ancestor_id'] if cut else None
        ancestor_step = by_node_step[ancestor_node] if cut else None
        if cut:
            assert ancestor_step < i
            assert tuple(sorted(current.items())) == tuple(tuple(pair) for pair in cut['state_key'])
            assert current == steps[ancestor_step]['positions']
            assert not goal and not node['children']
        steps.append({'step': i, 'node_id': node['id'],
                      'action': action(node) if i else None,
                      'positions': dict(sorted(current.items())),
                      'state_text': state_text(node),
                      'goal_satisfied': goal,
                      'legal_action': True if i else None,
                      'modeled_state_matches': True,
                      'closed_as_cycle': bool(cut),
                      'cycle_ancestor_node_id': ancestor_node,
                      'cycle_ancestor_step': ancestor_step})
    if cycle_node is None:
        assert steps[-1]['goal_satisfied']
    else:
        assert steps[-1]['closed_as_cycle']

    ledger = [json.loads(line) for line in paths_used[3].read_text().splitlines() if line.strip()]
    starts = {row['call'] for row in ledger if row['status'] == 'started'}
    completed = [row for row in ledger if row['status'] == 'completed' and row.get('usage')]
    assert starts == {row['call'] for row in completed}
    task = next(t for t in read(ROOT / 'data/manifest.json')['splits']['test'] if t['index'] == index)
    return {'id': example_id, 'label': label, 'task_id': task['task_id'],
            'task_index': index, 'arm': arm, 'round': 3,
            'run_success': run_success, 'run_calls': len(completed),
            'run_returned_tokens': sum(row['usage']['total_tokens'] for row in completed),
            'run_status': status['status'],
            'trajectory_kind': 'audited_cycle' if cycle_node is not None else 'selected_solution',
            'initial_positions': dict(sorted(initial_positions.items())),
            'goal_text': goal_text, 'goal_relations': [list(pair) for pair in goal_relations],
            'steps': steps, 'sources': [source(path) for path in paths_used]}


def build():
    examples = [
        export_example('cycle_pruned_3', 3, 'pruned-success-35',
                       'Pruned search succeeds | instance-35'),
        export_example('cycle_control_3', 12, 'control-solution-69',
                       'Control solution | instance-69'),
        export_example('cycle_pruned_3', 12, 'pruned-cycle-69',
                       'Pruned branch closes a cycle | instance-69', cycle_node=9),
    ]
    assert examples[1]['initial_positions'] == examples[2]['initial_positions']
    assert examples[1]['goal_relations'] == examples[2]['goal_relations']
    return {'version': 1, 'source_type': 'measured_saved_runs',
            'new_model_calls': 0, 'new_search_runs': 0,
            'transition_validation': 'independent four-operator action replay matched every displayed modeled state',
            'selection': 'two selected successful terminals and one audited selected cycle-ending trajectory',
            'validation_sources': [source(ROOT / 'scripts/symbolic.py'),
                                   source(ROOT / 'data/manifest.json'),
                                   source(Path(__file__).resolve())],
            'examples': examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'tmp/blocks_replay_data.json')
    args = parser.parse_args()
    data = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'examples': [
        {'id': row['id'], 'steps': len(row['steps']) - 1,
         'run_calls': row['run_calls'], 'run_success': row['run_success'],
         'cycle_ancestor_step': row['steps'][-1]['cycle_ancestor_step']}
        for row in data['examples']], 'new_model_calls': 0, 'new_search_runs': 0}, indent=2))


if __name__ == '__main__':
    main()
