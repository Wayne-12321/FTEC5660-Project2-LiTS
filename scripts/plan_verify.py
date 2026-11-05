"""Low-cost alternative: whole-plan LLM proposal + deterministic verification.

No gold plans or symbolic search; separate from the frozen LiTS/MCTS experiment.
Default mode only writes the prompt and performs no API call.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from common import ROOT, UPSTREAM, read, write, sha, verify_manifest, dataset_kwargs, environment
from symbolic import positions, apply, replay

RULES = '''Plan a BlocksWorld task using at most {max_steps} actions.
Only a clear block may be picked up or unstacked, and only with an empty hand.
Pick up only from the table; unstack only from its actual supporting block.
Put down or stack only the block held in the hand. Stack only onto another clear block.
Actions must use exactly one of these forms (replace X/Y with block names):
pick up the X block
put down the X block
unstack the X block from on top of the Y block
stack the X block on top of the Y block
Plan the entire sequence before answering. Resolve the dependencies between goal relations:
a support block must be in its final place before placing another goal block onto it.
Do not undo actions or return to a prior state. Temporary placements may be needed.
Return ONLY a JSON object {{"actions": ["action", ...]}}. Do not output predicted states,
explanations, markdown, or a success claim. The plan will be checked by an executable verifier.
'''

def goal_order(goal):
    """Topological order of goal relations, not an action plan or state-space search."""
    pairs=re.findall(r'the (\w+) block is on top of the (\w+) block',goal)
    if not pairs: raise ValueError('No goal relations parsed')
    targets=dict(pairs)
    if len(targets)!=len(pairs): raise ValueError('Duplicate/conflicting goal relations')
    ordered=[];done=set();active=set()
    def visit(block):
        if block in done: return
        if block in active: raise ValueError('Cyclic goal stack')
        active.add(block)
        support=targets[block]
        if support in targets: visit(support)
        active.remove(block);done.add(block);ordered.append((block,support))
    for block in targets: visit(block)
    return ordered

def task_prompt(initial,goal,cfg):
    order=goal_order(goal)
    return (RULES.format(max_steps=cfg['max_steps'])+'\nInitial state: '+initial+'\nGoal: '+goal+
        '\nBottom-up goal dependency order computed from the goal ONLY: '+json.dumps(order)+
        '\nEach pair is [block, support]. Finish the earlier relation before placing a goal block '+
        'on its support in a later relation. Do not disturb an already correctly placed support. '+
        'Do not move later-goal blocks early unless required to clear the earlier goal. '+
        'Track which block is held after every action; you cannot move a block with another on top. '+
        'All listed goals must hold simultaneously within the step limit.')

def verify_plan(response, initial, goal, max_steps):
    states=[]
    try:
        payload=json.loads(response)
        actions=payload['actions']
        if not isinstance(actions,list) or not all(isinstance(a,str) for a in actions):
            raise ValueError('actions must be a list of strings')
        if not 1 <= len(actions) <= max_steps:
            raise ValueError(f'Plan must contain 1..{max_steps} actions')
        pos=positions(initial)
        if not pos: raise ValueError('No initial blocks parsed')
        states=[dict(pos)]
        seen={tuple(sorted(pos.items()))}
        for i,action in enumerate(actions):
            try: apply(pos,action)
            except ValueError as exc: raise ValueError(f'Step {i+1}: {exc}; actual positions: {pos}') from exc
            key=tuple(sorted(pos.items()))
            if key in seen: raise ValueError(f'Step {i+1}: returns to an earlier state')
            seen.add(key);states.append(dict(pos))
        if not replay(initial,actions,goal):
            raise ValueError('Unsatisfied goal relations: '+str([(b,s) for b,s in goal_order(goal) if pos.get(b)!=s])+'; final positions: '+str(pos))
        return dict(success=True,actions=actions,states=states,error=None)
    except (ValueError,TypeError,KeyError) as exc:
        return dict(success=False,actions=[],states=states,error=str(exc))

def solve(initial,goal,cfg,call):
    prompt=task_prompt(initial,goal,cfg)
    attempts=[]
    for i in range(cfg['max_attempts']):
        if len(prompt)>cfg['max_prompt_chars']:
            return dict(success=False,attempts=attempts,stop_reason='prompt_budget')
        response=call(prompt)
        checked=verify_plan(response,initial,goal,cfg['max_steps'])
        attempts.append(dict(attempt=i+1,prompt=prompt,response=response,verification=checked))
        if checked['success']:
            return dict(success=True,attempts=attempts,actions=checked['actions'],stop_reason='verified_solution')
        prompt=(task_prompt(initial,goal,cfg)+
                '\nPrevious rejected proposal: '+response+'\nVerifier feedback: '+checked['error']+
                '\nReturn a corrected complete plan from the ORIGINAL initial state.')
    return dict(success=False,attempts=attempts,stop_reason='attempt_budget')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-api',action='store_true')
    parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    manifest=verify_manifest()
    cfg=read(ROOT/'configs/plan_verify.json')
    if not 0 <= args.index < len(manifest['splits']['dev']): parser.error('Index outside dev split')
    if sha(ROOT/'data/dev.json')!=manifest['splits']['dev_sha256']: parser.error('Data hash mismatch')
    out=args.output.resolve()
    if out.exists(): parser.error('Output exists; use a fresh directory')
    # Only standard task loader fields are passed to planning, never the source reference plan.
    sys.path[:0]=[str(UPSTREAM),str(UPSTREAM/'demos')]
    os.environ.update({k:v for k,v in environment(offline=not args.execute_api).items()
                       if k in ('HF_HUB_OFFLINE','HF_HOME','OMP_NUM_THREADS','KMP_USE_SHM','MPLCONFIGDIR')})
    os.environ['PYTHON_DOTENV_DISABLED']='1'
    from lits_benchmark.blocksworld import load_blocksworld
    example=load_blocksworld(**dataset_kwargs('dev'))[args.index]
    initial,goal=example['init_state_str'],example['query_or_goals']
    out.mkdir(parents=True)
    model='openai/offline-placeholder';endpoint=None
    call=None
    if args.execute_api:
        from dotenv import dotenv_values
        values=dotenv_values(ROOT/'.env')
        values={k:os.environ.get(k) or values.get(k) for k in ('OPENAI_API_KEY','OPENAI_API_BASE','LITS_MODEL')}
        if not all(values.values()): parser.error('Missing local API configuration')
        model=values['LITS_MODEL'];endpoint=values['OPENAI_API_BASE']
        from urllib.parse import urlparse
        if urlparse(endpoint).hostname!='api.deepseek.com' or model!='openai/deepseek-flash':
            parser.error('This bounded experiment expects official DeepSeek Flash')
        os.environ.update(LITS_MAX_CALLS=str(cfg['max_attempts']),LITS_MAX_TOKENS=str(cfg['max_total_tokens']),
                          LITS_MAX_OUTPUT=str(cfg['max_output_tokens']),LITS_CALL_LEDGER=str(out/'r0_task000/api_calls.jsonl'))
        (out/'r0_task000').mkdir()
        from entry import api_guard
        api_guard()
        from openai import OpenAI
        client=OpenAI(api_key=values['OPENAI_API_KEY'],base_url=endpoint)
        def call(prompt):
            result=client.chat.completions.create(model=model.removeprefix('openai/'),
                messages=[{'role':'user','content':prompt}],temperature=cfg['temperature'],
                max_tokens=cfg['max_output_tokens'],response_format={'type':'json_object'})
            return result.choices[0].message.content or ''
    else:
        from entry import offline_guard
        offline_guard()
    write(out/'experiment.json',dict(variant='plan_verify',config=cfg,model=model,endpoint=endpoint,
        split='dev',tasks=[manifest['splits']['dev'][args.index]],api_execution=args.execute_api,
        data_sha256=sha(ROOT/'data/dev.json'),upstream_commit=manifest['upstream_commit'],
        implementation_sha256=sha(__file__),verifier_sha256=sha(ROOT/'scripts/symbolic.py')))
    (out/'executed_plan_verify.py').write_text(Path(__file__).read_text())
    if not args.execute_api:
        write(out/'prompt.json',dict(prompt=task_prompt(initial,goal,cfg)))
        print('Offline prompt prepared; API calls: 0');return
    start=time.monotonic()
    try:
        result=solve(initial,goal,cfg,call)
        result.update(elapsed_seconds=time.monotonic()-start,task_id=manifest['splits']['dev'][args.index]['task_id'],
                      initial=initial,goal=goal)
        write(out/'result.json',result)
        print(json.dumps({k:result[k] for k in ('success','stop_reason','elapsed_seconds','actions') if k in result},ensure_ascii=False))
    except Exception as exc:
        write(out/'error.json',dict(type=type(exc).__name__,elapsed_seconds=time.monotonic()-start))
        print('Run stopped; error type: '+type(exc).__name__)
        return 1
    finally:
        from deepseek_cost import estimate
        estimate(out)
    return 0

if __name__=='__main__': sys.exit(main())
