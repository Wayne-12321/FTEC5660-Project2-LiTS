"""Independent low-score singleton widening, layered on frozen CycleMCTS.

No backtracking/reward changes. Adds at most one action twice per task.
"""
import json
import logging
import math
import os
from pathlib import Path
from lits.agents.registry import register_search
from lits.agents.tree.mcts import MCTSConfig
from lits_fixes import CycleMCTS
from common import write

log=logging.getLogger(__name__)
RULE=dict(threshold=0,max_triggers_per_task=2,extra_actions_per_trigger=1,
          reserve_calls=4,reserve_returned_tokens=10000,
          phase='simulate',requires_total_children=1)

@register_search('mcts_cycle_low_score',config_class=MCTSConfig)
class LowScoreMCTS(CycleMCTS):
    def _setup(self,query,query_idx):
        super()._setup(query,query_idx)
        self._widen_count=0
        self._widen_nodes=set()
        self._widen_events=[]
        self._widen_skips=[]

    def _budget_snapshot(self):
        try:
            ledger=Path(os.environ['LITS_CALL_LEDGER'])
            max_calls=int(os.environ['LITS_MAX_CALLS'])
            max_tokens=int(os.environ['LITS_MAX_TOKENS'])
            if not ledger.exists():return dict(allowed=False,reason='budget_unavailable')
            rows=[json.loads(s) for s in ledger.read_text().splitlines() if s.strip()]
            if any(not isinstance(r,dict) or r.get('status') not in ('started','completed') for r in rows):
                return dict(allowed=False,reason='unaccounted_request')
            starts=[r['call'] for r in rows if r['status']=='started']
            ends=[r['call'] for r in rows if r['status']=='completed']
            started=set(starts);completed=set(ends)
            if (started!=completed or len(starts)!=len(started) or len(ends)!=len(completed)
                    or any(not r.get('usage') for r in rows if r['status']=='completed')): return dict(allowed=False,reason='unaccounted_request')
            tokens=sum(r['usage']['total_tokens'] for r in rows if r['status']=='completed' and r.get('usage'))
            if tokens<0 or max_calls<=0 or max_tokens<=0:return dict(allowed=False,reason='budget_unavailable')
            info=dict(used_calls=len(started),used_tokens=tokens,
                      calls_remaining=max_calls-len(started),tokens_remaining=max_tokens-tokens)
            info['allowed']=(info['calls_remaining']>=RULE['reserve_calls'] and
                             info['tokens_remaining']>=RULE['reserve_returned_tokens'])
            info['reason']='admitted' if info['allowed'] else 'insufficient_reserve'
            return info
        except (KeyError,ValueError,OSError,TypeError):
            return dict(allowed=False,reason='budget_unavailable')

    def _do_expand(self,query,query_idx,node,policy,n_actions,**kwargs):
        super()._do_expand(query,query_idx,node,policy,n_actions,**kwargs)
        if (kwargs.get('from_phase')!='simulate' or not kwargs.get('assign_rewards',True)
                or kwargs.get('transition_before_evaluate',False)
                or getattr(kwargs.get('reward_model'),'n_sample',None)!=1
                or node.id in self._widen_nodes or len(node.children)!=1 or not self._open(node)):
            return
        only=node.children[0]
        score=only.fast_reward
        if (not self._open(only) or score==-1 or not math.isfinite(score)
                or score>RULE['threshold']):return
        legal=set(policy.generate_all_actions(node.state.env_state))
        existing={c.action.action_str for c in node.children}
        if len(legal)<=1 or not legal-existing:return
        if self._widen_count>=RULE['max_triggers_per_task']:
            self._widen_skips.append(dict(node_id=node.id,reason='task_trigger_cap'));return
        budget=self._budget_snapshot()
        if not budget['allowed']:
            self._widen_skips.append(dict(node_id=node.id,**budget));return
        self._widen_count+=1
        self._widen_nodes.add(node.id)
        before=list(node.children)
        event=dict(trigger=self._widen_count,node_id=node.id,depth=node.depth,
                   original_action=only.action.action_str,original_fast_reward=score,
                   budget_before=budget,status='started')
        self._widen_events.append(event)
        log.info('LOW_SCORE widen node=%s depth=%s original_score=%s trigger=%s',node.id,node.depth,score,self._widen_count)
        try:
            # Calls the frozen expansion directly; no recursive trigger or sibling deletion.
            super()._do_expand(query,query_idx,node,policy,n_actions=2,**kwargs)
            assert len(node.children)<=2 and node.children[:1]==before
            event['status']='completed'
        except Exception as exc:
            event.update(status='error',error_type=type(exc).__name__)
            raise
        finally:
            event['new_candidates']=[dict(node_id=c.id,action=c.action.action_str,fast_reward=c.fast_reward)
                                     for c in node.children if c not in before]
            event['budget_after']=self._budget_snapshot()
            if 'used_calls' in event['budget_after']:
                event['additional_api_calls']=event['budget_after']['used_calls']-budget['used_calls']
                event['additional_returned_tokens']=event['budget_after']['used_tokens']-budget['used_tokens']

    def search(self,query,query_idx):
        try:return super().search(query,query_idx)
        finally:
            if self.checkpoint_dir:
                write(Path(self.checkpoint_dir)/'low_score_audit.json',dict(rule=RULE,
                    trigger_attempts=self._widen_count,events=self._widen_events,skips=self._widen_skips,
                    note='Token reserve is conservative admission, not a strict prediction; original SDK guard enforces each next-call threshold. Low-score original candidate is retained.'))
