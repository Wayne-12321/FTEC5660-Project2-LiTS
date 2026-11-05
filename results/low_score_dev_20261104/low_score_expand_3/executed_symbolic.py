"""Independent four-action BlocksWorld replay; never trusts LLM-generated states."""
import re

def positions(text):
    pos = {}
    for block in re.findall(r'the (\w+) block is on the table', text):
        pos[block] = 'table'
    for block, support in re.findall(r'the (\w+) block is on top of the (\w+) block', text):
        pos[block] = support
    for block in re.findall(r'the (\w+) block is in the hand', text):
        pos[block] = 'hand'
    return pos

def apply(pos, action):
    a = action.strip().lower().rstrip('.')
    patterns = [('pick', r'pick up the (\w+) block'),
                ('put', r'put down the (\w+) block'),
                ('unstack', r'unstack the (\w+) block from on top of the (\w+) block'),
                ('stack', r'stack the (\w+) block on top of the (\w+) block')]
    for kind, pattern in patterns:
        m = re.fullmatch(pattern, a)
        if m:
            break
    else:
        raise ValueError(f'Unrecognized action: {a}')
    b = m[1]
    if b not in pos:
        raise ValueError('Unknown block')
    clear = b not in pos.values()
    empty = 'hand' not in pos.values()
    if kind == 'pick':
        valid = empty and clear and pos[b] == 'table'
        target = 'hand'
    elif kind == 'put':
        valid = pos[b] == 'hand'
        target = 'table'
    elif kind == 'unstack':
        valid = empty and clear and pos[b] == m[2]
        target = 'hand'
    else:
        support = m[2]
        valid = (pos[b] == 'hand' and support in pos and b != support
                 and support not in pos.values() and pos[support] != 'hand')
        target = support
    if not valid:
        raise ValueError(f'Illegal action: {a}')
    pos[b] = target

def replay(initial, actions, goal):
    pos = positions(initial)
    if not pos:
        raise ValueError('No blocks parsed')
    for action in actions:
        apply(pos, action)
    goals = re.findall(r'the (\w+) block is on top of the (\w+) block', goal)
    if not goals:
        raise ValueError('No goals parsed')
    return all(pos.get(b) == s for b,s in goals)

def reference_actions(plan):
    actions=[]
    for line in plan.strip().splitlines():
        parts=line.strip('()').split()
        kind,b,*rest=parts
        actions.append({'pick-up':lambda:f'pick up the {b} block',
            'put-down':lambda:f'put down the {b} block',
            'stack':lambda:f'stack the {b} block on top of the {rest[0]} block',
            'unstack':lambda:f'unstack the {b} block from on top of the {rest[0]} block'}[kind]())
    return actions
