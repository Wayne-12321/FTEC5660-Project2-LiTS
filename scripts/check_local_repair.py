from entry import offline_guard
offline_guard()
from local_plan_repair import repair
from symbolic import replay
from common import ROOT,write
initial='the red block is on the table and the blue block is on the table'
goal='the red block is on top of the blue block'
a=repair(initial,goal,max_steps=2)
assert a['success'] and len(a['actions'])==2 and replay(initial,a['actions'],goal)
assert not repair(initial,goal,max_steps=1)['success']
assert repair(initial,goal,max_states=1)['stop_reason']=='state_budget'
b=repair(initial,goal,max_steps=2,preferred=['stack the blue block on top of the red block','invalid'])
assert b['success'] and replay(initial,b['actions'],goal)
# Generic renamed three-block tower; shortest length is four from all-table state.
i='the cyan block is on the table, the magenta block is on the table, the lime block is on the table'
g='the cyan block is on top of the magenta block and the magenta block is on top of the lime block'
c=repair(i,g,max_steps=4)
assert c['success'] and len(c['actions'])==4 and replay(i,c['actions'],g)
write(ROOT/'reports/local_repair_checks.json',dict(passed=True,paid_api_calls=0,
 checks=['generic legal plan','insufficient depth','state budget','invalid proposal cannot bypass legality','renamed three-block task']))
print('Local repair checks passed; API calls: 0')
