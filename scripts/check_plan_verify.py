"""Network-disabled validation and retry-budget regression checks."""
from entry import offline_guard
offline_guard()
import json
from common import ROOT,read,write
from plan_verify import verify_plan,solve,goal_order
assert goal_order("the green block is on top of the white block and the white block is on top of the black block")==[("white","black"),("green","white")]
try:
 goal_order("the green block is on top of the white block and the white block is on top of the green block")
 raise AssertionError("Cycle accepted")
except ValueError: pass
initial='the red block is on the table and the blue block is on the table'
goal='the red block is on top of the blue block'
valid=json.dumps({'actions':['pick up the red block','stack the red block on top of the blue block']})
assert verify_plan(valid,initial,goal,6)['success']
invalid=[ 'not JSON', '{}', json.dumps({'actions':'pick up the red block'}),
 json.dumps({'actions':['stack the red block on top of the blue block']}),
 json.dumps({'actions':['pick up the blue block']}),
 json.dumps({'actions':['pick up the red block','put down the red block']}),
 json.dumps({'actions':['pick up the red block']*7})]
for response in invalid: assert not verify_plan(response,initial,goal,6)['success']
cfg=read(ROOT/'configs/plan_verify.json')
calls=[]
def corrected(prompt):
 calls.append(prompt)
 return valid if len(calls)==2 else invalid[3]
result=solve(initial,goal,cfg,corrected)
assert result['success'] and len(calls)==2 and 'Verifier feedback' in calls[1]
calls=[]
def bad(prompt): calls.append(prompt);return 'invalid'
assert not solve(initial,goal,cfg,bad)['success'] and len(calls)==2
calls=[]
def good(prompt): calls.append(prompt);return valid
assert solve(initial,goal,cfg,good)['success'] and len(calls)==1
write(ROOT/'reports/plan_verify_checks.json',dict(passed=True,paid_api_calls=0,
 checks=['valid plan replay','malformed output','illegal action','unmet goal','cycle rejection',
         'depth cap','feedback correction','two-call cap','stop after success']))
print('Plan/verify checks passed; API calls: 0')
