"""Network-disabled semantic regressions for bounded low-score widening."""
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from common import ROOT, UPSTREAM, environment, read, write, sha
os.environ.update(environment())
sys.path[:0]=[str(UPSTREAM),str(UPSTREAM/'demos')]
from entry import offline_guard
offline_guard()
from lits_low_score import LowScoreMCTS, RULE
from lits_fixes import CycleMCTS, UniquePolicy, MCTSSearch
from lits.agents.tree.mcts import MCTSConfig
from lits.agents.tree.node import MCTSNode
from lits.structures.env_grounded import EnvState, EnvAction

TABLE=('the hand is empty, the cyan block is on the table, the magenta block is on the table, '
       'the cyan block is clear, the magenta block is clear')
HELD=('the hand is holding the cyan block, the cyan block is in the hand, '
      'the magenta block is on the table, the magenta block is clear')
A='pick up the cyan block';B='pick up the magenta block'
CHECKS=[]

def fixture(score=0.0):
    root=MCTSNode(EnvState(init_state=TABLE),None)
    child=MCTSNode(None,EnvAction(A),parent=root,fast_reward=score)
    root.children=[child]
    s=LowScoreMCTS(MCTSConfig(max_steps=6,force_terminating_on_depth_limit=True),None,None,None)
    s.root=root;s._cycle_events=[];s._cycle_parse_failures=set()
    s._widen_count=0;s._widen_nodes=set();s._widen_events=[];s._widen_skips=[]
    policy=SimpleNamespace(generate_all_actions=lambda text:[A,B],n_actions=3)
    reward=SimpleNamespace(n_sample=1)
    return s,root,child,policy,reward

def kw(reward,**updates):
    result=dict(from_phase='simulate',assign_rewards=True,reward_model=reward)
    result.update(updates);return result

def fake_expand(self,q,i,node,policy,n_actions,**kwargs):
    if n_actions==2:
        assert len(node.children)==1
        node.children.append(MCTSNode(None,EnvAction(B),parent=node,fast_reward=0.0))

def rows(used=1,total_tokens=100):
    out=[]
    for i in range(used):
        out.extend([dict(call=i+1,status='started'),
                    dict(call=i+1,status='completed',usage={'total_tokens':total_tokens})])
    return out

with tempfile.TemporaryDirectory(prefix='low-score-check-') as temp:
    ledger=Path(temp)/'ledger.jsonl'
    def ledger_rows(records):ledger.write_text(''.join(json.dumps(r)+'\n' for r in records))
    ledger_rows(rows())
    with patch.dict(os.environ,{'LITS_CALL_LEDGER':str(ledger),'LITS_MAX_CALLS':'150',
                                 'LITS_MAX_TOKENS':'150000','LITS_MAX_OUTPUT':'512'}):
        # Trigger once and preserve the original object/action; a second visit adds nothing.
        s,node,original,policy,reward=fixture()
        with patch.object(CycleMCTS,'_do_expand',autospec=True,side_effect=fake_expand) as expand:
            s._do_expand('fixture',0,node,policy,1,**kw(reward))
            assert len(node.children)==2 and node.children[0] is original
            assert [str(n.action) for n in node.children]==[A,B]
            assert s._widen_count==1 and len(s._widen_events)==1
            assert s._widen_events[0]['status']=='completed'
            assert [x.args[5] if len(x.args)>5 else x.kwargs['n_actions'] for x in expand.call_args_list]==[1,2]
            # Simulate returning to a node where only the same original child remains.
            node.children=[original]
            s._do_expand('fixture',0,node,policy,1,**kw(reward))
            assert s._widen_count==1 and len(node.children)==1
        CHECKS.append('zero-score singleton adds exactly one distinct candidate and original is retained; node attempted once')

        # All non-trigger conditions are semantic, not inferred from trigger counts alone.
        for label,alter in [
            ('positive',lambda s,n,c,p,r:(setattr(c,'fast_reward',.5),{})),
            ('unscored',lambda s,n,c,p,r:(setattr(c,'fast_reward',-1),{})),
            ('nan',lambda s,n,c,p,r:(setattr(c,'fast_reward',float('nan')),{})),
            ('one legal action',lambda s,n,c,p,r:(setattr(p,'generate_all_actions',lambda text:[A]),{})),
            ('no remaining distinct legal action',lambda s,n,c,p,r:(setattr(p,'generate_all_actions',lambda text:[A,A]),{})),
            ('multiple children',lambda s,n,c,p,r:(n.children.append(MCTSNode(None,EnvAction(B),parent=n,fast_reward=0)),{})),
            ('closed child',lambda s,n,c,p,r:((setattr(c,'is_terminal',True),setattr(c,'state',EnvState(init_state=HELD))),{})),
            ('multiple prm samples',lambda s,n,c,p,r:(setattr(r,'n_sample',2),{})),
            ('expand phase',lambda s,n,c,p,r:(None,{'from_phase':'expand'})),
            ('unassigned rewards',lambda s,n,c,p,r:(None,{'assign_rewards':False})),
            ('transition-before-score mode',lambda s,n,c,p,r:(None,{'transition_before_evaluate':True})),
        ]:
            s,node,original,policy,reward=fixture()
            _,overrides=alter(s,node,original,policy,reward)
            with patch.object(CycleMCTS,'_do_expand',autospec=True,side_effect=fake_expand) as expand:
                s._do_expand('fixture',0,node,policy,1,**kw(reward,**overrides))
            assert s._widen_count==0 and expand.call_count==1,label
        CHECKS.append('positive/unscored/nonfinite, one legal action, duplicates, multiple children, closed child, non-simulate/PRM modes do not trigger')

        s,node,original,policy,reward=fixture()
        with patch.object(CycleMCTS,'_do_expand',autospec=True,side_effect=fake_expand):
            for _ in range(3):
                fresh,_,_,_,_=fixture()
                n=fresh.root
                s._do_expand('fixture',0,n,policy,1,**kw(reward))
            assert s._widen_count==2 and len(s._widen_events)==2
            assert s._widen_skips[-1]['reason']=='task_trigger_cap'
        CHECKS.append('at most two widening attempts per task across distinct nodes')

        # Boundary admission and insufficient reserves.
        ledger_rows(rows(1,100))
        s,node,original,policy,reward=fixture()
        with patch.dict(os.environ,{'LITS_MAX_CALLS':'5','LITS_MAX_TOKENS':'10100'}):
            assert s._budget_snapshot()['allowed'] # exactly 4 calls and 10000 tokens remain
        for limits in [{'LITS_MAX_CALLS':'4'}, {'LITS_MAX_TOKENS':'10099'}]:
            s,node,original,policy,reward=fixture()
            with patch.dict(os.environ,limits),patch.object(CycleMCTS,'_do_expand',autospec=True,side_effect=fake_expand) as expand:
                s._do_expand('fixture',0,node,policy,1,**kw(reward))
            assert s._widen_count==0 and expand.call_count==1
            assert s._widen_skips[-1]['reason']=='insufficient_reserve'
        CHECKS.append('four-call/10000-token reserve exact boundary; insufficient reserve blocks extra expansion')

        for label,records in [
            ('pending',[dict(call=1,status='started')]),
            ('missing usage',[dict(call=1,status='started'),dict(call=1,status='completed')]),
            ('incomplete usage',[dict(call=1,status='started'),dict(call=1,status='completed',usage={'prompt_tokens':100})]),
            ('duplicate start',[dict(call=1,status='started'),dict(call=1,status='started'),dict(call=1,status='completed',usage={'total_tokens':100})]),
            ('duplicate completion',[dict(call=1,status='started'),dict(call=1,status='completed',usage={'total_tokens':100}),dict(call=1,status='completed',usage={'total_tokens':100})]),
            ('bad status',[dict(call=1,status='unexpected')]),
            ('bad row type',[42]),
        ]:
            ledger_rows(records)
            s,node,original,policy,reward=fixture()
            with patch.object(CycleMCTS,'_do_expand',autospec=True,side_effect=fake_expand) as expand:
                s._do_expand('fixture',0,node,policy,1,**kw(reward))
            assert s._widen_count==0 and expand.call_count==1,label
        ledger.write_text('{bad JSON\n')
        s,*_=fixture();assert not s._budget_snapshot()['allowed']
        with patch.dict(os.environ,{'LITS_MAX_CALLS':'not a number'}):
            assert not s._budget_snapshot()['allowed']
        with patch.dict(os.environ,{},clear=True):assert not s._budget_snapshot()['allowed']
        CHECKS.append('pending/missing usage/duplicate records/bad row or status/malformed ledger or limits deny widening')

        ledger.unlink()
        s,*_=fixture()
        assert not s._budget_snapshot()['allowed'],'Missing ledger must be conservatively rejected'
        CHECKS.append('missing ledger rejected rather than assumed zero spent')
        ledger_rows(rows())

        # Exercise the real upstream expansion (no policy or reward API). The sentinel
        # model is absent; finite enumeration must exclude the existing action.
        s,node,original,_,_=fixture()
        policy=UniquePolicy(None,generate_all_actions=lambda text:[A,B],n_actions=3)
        reward=SimpleNamespace(n_sample=1,fast_reward=lambda *args,**kwargs:(0.0,{}))
        s.policy=policy;s.reward_model=reward
        s._do_expand('fixture',0,node,policy,1,**kw(reward))
        assert len(node.children)==2 and node.children[0] is original
        assert {str(n.action) for n in node.children}=={A,B}
        assert node.children[1].fast_reward==0 and s._widen_count==1
        assert int(s.config.simulate_choice([n.fast_reward for n in node.children]))==0
        CHECKS.append('real upstream expansion excludes existing sibling and zero-score original wins an equal-score tie')

        # A full real rollout with fake transition confirms original zero-score action
        # may still be executed; this change does not hard-filter nonpositive actions.
        s,node,original,_,_=fixture();s.policy=policy;s.reward_model=reward
        s.config.max_steps=1
        def transition(q,i,child,**kwargs):
            assert child is original
            child.state=EnvState(init_state=HELD);child.is_terminal=True
        path=[node]
        with patch('lits_fixes._world_modeling',side_effect=transition):
            s._do_simulate('fixture',0,path,s.config,None,policy,reward)
        assert path[-1] is original and len(node.children)==2
        CHECKS.append('real rollout retains and executes the original zero-score action under tie')

        # A failure during the additional expansion consumes the attempt, propagates
        # unchanged, and is audited by the actual inherited search wrappers.
        s,node,original,policy,reward=fixture();s.checkpoint_dir=temp;s.save_full_tree=Mock()
        def explode(self,q,i,n,p,n_actions,**kwargs):
            if n_actions==2:raise RuntimeError('budget fixture')
        def search_body(self,q,i):self._do_expand(q,i,node,policy,1,**kw(reward))
        with patch.object(CycleMCTS,'_do_expand',autospec=True,side_effect=explode), \
             patch.object(MCTSSearch,'search',autospec=True,side_effect=search_body):
            try:s.search('fixture',0)
            except RuntimeError as exc:assert str(exc)=='budget fixture'
            else:raise AssertionError('Budget exception was swallowed')
        assert s._widen_count==1 and node.id in s._widen_nodes
        audit=read(Path(temp)/'low_score_audit.json')
        assert audit['events'][0]['status']=='error' and audit['events'][0]['error_type']=='RuntimeError'
        assert audit['trigger_attempts']==1 and (Path(temp)/'cycle_audit.json').exists()
        s.save_full_tree.assert_called_once_with(0)
        CHECKS.append('extra expansion error consumes one attempt, propagates, preserves tree and both audits')

write(ROOT/'reports/low_score_checks_20261104.json',dict(passed=True,paid_api_calls=0,
    network_disabled=True,implementation_sha256=sha(ROOT/'scripts/lits_low_score.py'),
    test_sha256=sha(__file__),checks=CHECKS))
print('Low-score widening regressions passed; paid API calls: 0; checks:',len(CHECKS))
