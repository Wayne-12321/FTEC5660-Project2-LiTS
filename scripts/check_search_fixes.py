"""Offline regression checks for finite actions and MCTS frontier."""
import os, sys
from types import SimpleNamespace
from unittest.mock import Mock, patch
from common import ROOT, UPSTREAM, environment, write
os.environ.update(environment())
sys.path[:0]=[str(UPSTREAM),str(UPSTREAM/'demos')]
from entry import offline_guard
offline_guard()
from lits_fixes import UniquePolicy, FrontierMCTS, FrontierExhausted
from lits.agents.tree.mcts import MCTSConfig
from lits.agents.tree.node import MCTSNode
from lits.structures.env_grounded import EnvState, EnvAction
state=EnvState(init_state='fixture')
p=UniquePolicy(None,generate_all_actions=lambda s:['a','b'])
p.base_model=object()
p._build_prompt=Mock(side_effect=lambda s,q,actions:','.join(actions))
p._call_model=Mock(return_value=SimpleNamespace(text='a'))
assert [str(s.action) for s in p._get_actions(state,3,.7)]==['a','b']
assert p._call_model.call_count==1
p._existing_actions=['a'];p._call_model.reset_mock()
assert [str(s.action) for s in p._get_actions(state,3,.7)]==['b']
assert p._call_model.call_count==0
p._existing_actions=['a','b']
assert p._get_actions(state,3,.7)==[]
p._existing_actions=[]
p._call_model.return_value=SimpleNamespace(text='invalid')
assert [str(s.action) for s in p._get_actions(state,3,.7)]==['a','b']
s=FrontierMCTS(MCTSConfig(max_steps=1,force_terminating_on_depth_limit=True),None,p,None)
s.set_log_field=Mock()
root=MCTSNode(state,None);root._fully_expanded=True
capped=MCTSNode(state,EnvAction('a'),parent=root,fast_reward=1)
unknown=MCTSNode(None,EnvAction('b'),parent=root,fast_reward=0)
root.children=[capped,unknown];s.root=root
assert s._do_select_path('',0,0) == [root]
unknown.state=state
try:
 s._do_select_path('',0,1)
 raise AssertionError('Exhausted frontier selected again')
except FrontierExhausted: pass
s.config.max_steps=3;root._fully_expanded=False
assert s._do_select_path('',0,2)==[root]
root.children=[capped];p.base_model=None
s._do_expand('',0,root,p,3,reward_model=None,assign_rewards=False,from_phase='expand')
assert [str(c.action) for c in root.children]==['a','b']
assert root._fully_expanded
s.config.max_steps=1;root.children[1].fast_reward=0
path=[root]
def transition(q,i,node,**kwargs): node.state=state
with patch('lits_fixes._world_modeling',side_effect=transition) as world:
 s._do_simulate('',0,path,s.config,None,p,None)
assert path[-1] is root.children[1] and world.call_count==1
# Runtime interruptions must preserve evidence and remain failures.
s.save_full_tree=Mock()
with patch('lits_fixes.MCTSSearch.search', side_effect=RuntimeError('budget fixture')):
 try:
  s.search('',0)
  raise AssertionError('Runtime error swallowed')
 except RuntimeError: pass
s.save_full_tree.assert_called_once_with(0)
write(ROOT/'reports/search_fix_checks.json',dict(paid_api_calls=0,passed=True,checks=[
 'finite action exhaustion','exclude siblings','singleton no call','bounded invalid fallback',
 'skip capped branches','clean exhaustion','widen rollout node','preserve existing children',
 'rollout checks unknown depth-limit goal','budget exception saves tree and propagates']))
print('Search fix regression checks passed; paid calls: 0')
