"""Run the real upstream CLI with explicit offline or bounded-API execution."""
import json
from datetime import datetime, timezone
import os
import random
import runpy
import socket
import sys
from pathlib import Path

def deny_network(*args, **kwargs):
    raise RuntimeError('OFFLINE CHECK: network access is disabled')

def offline_guard():
    socket.socket.connect = deny_network
    socket.socket.connect_ex = deny_network
    socket.create_connection = deny_network
    socket.getaddrinfo = deny_network
    # Enforce at audit level too, before imported code can create clients.
    def audit(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
            deny_network()
    sys.addaudithook(audit)
    os.environ['PYTHON_DOTENV_DISABLED'] = '1'
    import dotenv
    dotenv.load_dotenv = lambda *a, **kw: False

def api_guard():
    """Cap all OpenAI-compatible calls, including policy/reward/world transitions.

    No provider-specific hidden thinking controls are assumed. Disable SDK retries;
    an errored request remains in the ledger because it may have been billed.
    """
    import openai
    from openai.resources.chat.completions import Completions
    original_init = openai.OpenAI.__init__
    def bounded_init(self, *a, **kw):
        kw['max_retries'] = 0
        kw['timeout'] = 60.0
        original_init(self, *a, **kw)
    openai.OpenAI.__init__ = bounded_init
    original = Completions.create
    calls = 0
    tokens = 0
    ledger = Path(os.environ['LITS_CALL_LEDGER'])
    def bounded_create(self, *a, **kw):
        nonlocal calls, tokens
        if calls >= int(os.environ['LITS_MAX_CALLS']) or tokens >= int(os.environ['LITS_MAX_TOKENS']):
            raise RuntimeError('Experiment API budget exhausted')
        calls += 1
        kw['max_tokens'] = min(kw.get('max_tokens') or 512, int(os.environ['LITS_MAX_OUTPUT']))
        # Upstream injects a vLLM-specific field into all custom endpoints.
        # Remove that field for provider portability, identically in both arms.
        if isinstance(kw.get('extra_body'), dict):
            kw['extra_body'] = {k:v for k,v in kw['extra_body'].items() if k != 'chat_template_kwargs'} or None
        # DeepSeek defaults to thinking mode. This experiment uses non-thinking
        # responses so the 512-token cap is available for the actual action/change.
        from urllib.parse import urlparse
        if urlparse(str(self._client.base_url)).hostname == 'api.deepseek.com':
            kw['extra_body'] = dict(kw.get('extra_body') or {}, thinking={'type':'disabled'})
        record = dict(call=calls, model=kw.get('model'), status='started', timestamp_utc=datetime.now(timezone.utc).isoformat())
        with ledger.open('a') as f:
            f.write(json.dumps(record) + '\n')
        try:
            response = original(self, *a, **kw)
        except Exception as exc:
            with ledger.open('a') as f:
                f.write(json.dumps(dict(call=calls, status='error', error_type=type(exc).__name__)) + '\n')
            raise
        usage = response.usage.model_dump() if response.usage else None
        if usage:
            tokens += usage.get('total_tokens', 0)
        with ledger.open('a') as f:
            f.write(json.dumps(dict(call=calls, status='completed', usage=usage, response_model=response.model, timestamp_utc=datetime.now(timezone.utc).isoformat())) + '\n')
        if usage is None:
            raise RuntimeError('Provider omitted usage; stop because token accounting is unavailable')
        return response
    Completions.create = bounded_create

if __name__ == '__main__':
    mode = sys.argv.pop(1)
    if os.environ.get('LITS_OFFLINE', '1') == '1':
        offline_guard()
    else:
        api_guard()
    seed = int(os.environ.get('LITS_SEED', '42'))
    random.seed(seed)
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    module = {'search':'lits.cli.search', 'eval':'lits.cli.eval_search'}[mode]
    runpy.run_module(module, run_name='__main__')
