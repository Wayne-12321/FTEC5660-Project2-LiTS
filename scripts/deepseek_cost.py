"""Offline estimate from DeepSeek usage; not a provider billing statement.

Rates checked 2026-10-23: https://api-docs.deepseek.com/zh-cn/quick_start/pricing/
Report both off-peak and peak because historical calls lack per-call timestamps.
"""
import argparse
import json
from pathlib import Path
from common import write

def estimate(root):
    experiment = json.loads((root/'experiment.json').read_text())
    if experiment.get('model') != 'openai/deepseek-flash':
        raise ValueError('This rate snapshot applies only to official DeepSeek Flash')
    from urllib.parse import urlparse
    if urlparse(experiment.get('endpoint') or '').hostname != 'api.deepseek.com':
        raise ValueError('Third-party endpoint may charge different prices')
    hits = misses = output = completed = unknown = 0
    for path in root.glob('r*_task*/api_calls.jsonl'):
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        started = {r['call'] for r in records if r['status'] == 'started'}
        accounted = set()
        for r in records:
            if r['status'] != 'completed' or not r.get('usage'):
                continue
            u = r['usage']
            if 'prompt_cache_hit_tokens' not in u or 'prompt_cache_miss_tokens' not in u:
                raise ValueError('Missing DeepSeek cache usage fields; use provider bill')
            if u['prompt_cache_hit_tokens'] + u['prompt_cache_miss_tokens'] != u['prompt_tokens']:
                raise ValueError('Cache token accounting mismatch')
            accounted.add(r['call'])
            completed += 1
            hits += u['prompt_cache_hit_tokens']
            misses += u['prompt_cache_miss_tokens']
            output += u['completion_tokens']
        unknown += len(started - accounted)
    low = (hits * .02 + misses + output * 4) / 1_000_000
    report = dict(model='deepseek-flash', rate_date=experiment.get('pricing_verified_date','2026-10-23'),
        source='https://api-docs.deepseek.com/zh-cn/quick_start/pricing/',
        completed_calls=completed, unaccounted_calls=unknown,
        cache_hit_input_tokens=hits, cache_miss_input_tokens=misses,
        input_tokens=hits+misses, output_tokens=output,
        estimated_cny_off_peak=round(low,8), estimated_cny_peak=round(low*2,8),
        note='Estimate for recorded successful calls only, not confirmed billing. Unknown requests may incur extra charges.')
    write(root/'cost_estimate.json',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('result_dir',type=Path)
    args=p.parse_args()
    print(json.dumps(estimate(args.result_dir),indent=2))
