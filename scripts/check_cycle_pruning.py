"""Network-disabled semantic regressions for path-local MCTS cycle pruning."""
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from common import ROOT, UPSTREAM, environment, read
os.environ.update(environment())
sys.path[:0]=[str(UPSTREAM),str(UPSTREAM/'demos')]
from entry import offline_guard
offline_guard()
from cycle_state import canonical_state
from lits_fixes import CycleMCTS, FrontierMCTS, MCTSSearch
from lits.agents.tree.mcts import MCTSConfig
from lits.agents.tree.node import MCTSNode
from lits.structures.env_grounded import EnvState,EnvAction

TABLE=('the hand is empty, the red block is on the table, the blue block is on the table, '
       'the red block is clear, the blue block is clear')
HELD=('the hand is holding the red block, the red block is in the hand, '
      'the blue block is on the table, the blue block is clear')
STACK=('the hand is empty, the red block is on top of the blue block, '
       'the blue block is on the table, the red block is clear')
blocks={'red','blue'}
key=canonical_state(TABLE,blocks)
assert key==(('blue','table'),('red','table'))
assert canonical_state(', '.join(reversed(TABLE.split(', '))).upper()+'.',blocks)==key
bad=[None,'',TABLE.replace('the blue block is on the table, ',''),
 TABLE+', the red block is on top of the blue block',
 TABLE.replace('hand is empty','hand is holding the red block'),
 HELD+', the hand is empty',HELD.replace('the blue block is on the table','the blue block is in the hand'),
 TABLE.replace('the red block is clear','the green block is clear'),
 STACK.replace('on top of the blue block','on top of the green block'),
 'the hand is empty, the red block is on top of the blue block, the blue block is on top of the red block',
 'the hand is empty, the red block is on top of the red block, the blue block is on the table, the blue block is clear']
for text in bad:assert canonical_state(text,blocks) is None,text
multi=('the hand is empty, the red block is on top of the blue block, '
       'the green block is on top of the blue block, the blue block is on the table, '
       'the red block is clear, the green block is clear')
assert canonical_state(multi,{'red','blue','green'}) is None

def node(text,parent=None,terminal=False):
    obj=MCTSNode(EnvState(init_state=text) if text is not None else None,
                 EnvAction('fixture') if parent else None,parent=parent,is_terminal=terminal,fast_reward=0)
    if parent:parent.children.append(obj)
    return obj

def search(root,max_steps=6):
    cfg=MCTSConfig(max_steps=max_steps,force_terminating_on_depth_limit=True)
    s=CycleMCTS(cfg,None,SimpleNamespace(n_actions=3),None)
    s.root=root;s._cycle_events=[];s._cycle_parse_failures=set()
    return s

root=node(TABLE);held=node(HELD,root);loop=node(TABLE,held)
s=search(root)
assert not s._open(loop) and loop._cycle_pruned and not loop.is_terminal
assert s._cycle_events[0]['ancestor_id']==root.id
assert not s._open(loop) and len(s._cycle_events)==1
# Terminal means actual upstream goal; never relabel a cycle as a solution.
goal=node(TABLE,held,True)
assert not s._cyclic(goal) and not hasattr(goal,'_cycle_pruned')
unknown=node(None,root)
assert s._open(unknown)
unknown.state=EnvState(init_state=TABLE)
assert not s._open(unknown),'unknown state was permanently cached'
# Matching a sibling/other branch is not an ancestor cycle.
branch=node(STACK,root);cross=node(HELD,branch)
assert s._open(cross) and not hasattr(cross,'_cycle_pruned')
malformed=node('the hand is empty',held)
assert s._open(malformed) and malformed.id in s._cycle_parse_failures
bad_root=node('incomplete');bad_child=node(TABLE,bad_root);bad_loop=node(TABLE,bad_child)
assert search(bad_root)._open(bad_loop),'invalid root should disable pruning'
# The selected leaf is modeled by upstream before expansion; expansion must stop here.
with patch.object(FrontierMCTS,'_do_expand') as upstream_expand:
    s._do_expand('',0,loop,s.policy,3)
assert not upstream_expand.called

# Exercise the real rollout loop with mock transitions only. The second transition
# returns to root. No policy/reward/world calls may follow that discovery, including
# when the discovery is exactly the depth/rollout boundary.
for max_steps,rollout_steps in [(6,6),(2,2)]:
    r=node(TABLE);first=node(None,r);second=node(None,first)
    r._fully_expanded=True;first._fully_expanded=True
    simulated=search(r,max_steps)
    expanded=[];modeled=[]
    def expand(self,q,i,n,p,n_actions,**kw):
        expanded.append(n.id)
        assert n is not second,'expanded a detected cycle'
    def transition(q,i,n,**kw):
        modeled.append(n.id)
        n.state=EnvState(init_state=HELD if n is first else TABLE)
    path=[r]
    with patch.object(FrontierMCTS,'_do_expand',autospec=True,side_effect=expand), \
         patch('lits_fixes._world_modeling',side_effect=transition):
        simulated._do_simulate('',0,path,simulated.config,None,simulated.policy,None,
                               roll_out_steps=rollout_steps)
    assert modeled==[first.id,second.id] and expanded==[r.id,first.id]
    assert second._cycle_pruned and len(simulated._cycle_events)==1
    assert path[-1] is second and not second.is_terminal

# Budget exhaustion must propagate unchanged, preserve the partial tree, and write
# a separate audit; it cannot quietly produce a successful completed run.
with tempfile.TemporaryDirectory(prefix='cycle-test-') as temp:
    r=node(TABLE);s=search(r);s.checkpoint_dir=temp;s.save_full_tree=Mock()
    with patch.object(MCTSSearch,'search',side_effect=RuntimeError('budget fixture')):
        try:s.search('',0)
        except RuntimeError as exc:assert str(exc)=='budget fixture'
        else:raise AssertionError('budget exception swallowed')
    s.save_full_tree.assert_called_once_with(0)
    audit=read(Path(temp)/'cycle_audit.json')
    assert audit['root_parse_valid'] and audit['pruned_nodes']==[]
print('Cycle pruning regressions passed; network disabled; paid API calls: 0')
