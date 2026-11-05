"""Exercise the real SDK with an in-memory HTTP transport; no network or billing."""
import json
import os
import tempfile
from pathlib import Path
from common import ROOT, write
from entry import offline_guard, api_guard

def main():
    offline_guard()
    import httpx
    from openai import OpenAI
    seen=[]
    def respond(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200,json={'id':'offline-fixture','object':'chat.completion','created':0,
            'model':'offline-fixture','choices':[{'index':0,'finish_reason':'stop',
            'message':{'role':'assistant','content':'good'}}],
            'usage':{'prompt_tokens':20,'completion_tokens':1,'total_tokens':21}})
    with tempfile.TemporaryDirectory() as tmp:
        os.environ.update(LITS_MAX_CALLS='1',LITS_MAX_TOKENS='100',LITS_MAX_OUTPUT='64',
                          LITS_CALL_LEDGER=str(Path(tmp)/'calls.jsonl'))
        api_guard()
        client=OpenAI(api_key='offline-fixture-not-a-key',base_url='https://api.deepseek.com',
            http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        assert client.max_retries==0
        client.chat.completions.create(model='fixture',messages=[{'role':'user','content':'test'}],
            max_tokens=512,extra_body={'chat_template_kwargs':{'enable_thinking':False}})
        assert len(seen)==1 and seen[0]['max_tokens']==64 and 'chat_template_kwargs' not in seen[0]
        assert seen[0]['thinking']=={'type':'disabled'}
        try:
            client.chat.completions.create(model='fixture',messages=[])
            raise AssertionError('Second request should be blocked')
        except RuntimeError as exc:
            assert 'budget exhausted' in str(exc)
        records=[json.loads(s) for s in Path(os.environ['LITS_CALL_LEDGER']).read_text().splitlines()]
        assert records[-1]['usage']['total_tokens']==21 and len(seen)==1
        client.close()
    result={'network_disabled':True,'paid_api_calls':0,'mock_requests':1,
        'checks':['SDK retries disabled','output cap enforced','vLLM-only field removed','DeepSeek thinking explicitly disabled',
                  'all usage recorded','request beyond call budget blocked before transport']}
    write(ROOT/'reports/api_guard_check.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
