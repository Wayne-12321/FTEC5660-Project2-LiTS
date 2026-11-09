"""Display saved experiment summaries for a short presentation; no model calls."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(relative):
    return json.loads((ROOT / relative).read_text())

def main():
    report = read('reports/cycle_stability_20261104.json')
    control = 'cycle_control_3'
    pruned = 'cycle_pruned_3'
    print('SAVED RESULTS | same 20 tasks, repeated 3 times per arm')
    print('Round   Success: control / pruned   Calls: control / pruned')
    for row in report['rounds']:
        c, p = row['arms'][control], row['arms'][pruned]
        print(f"  {row['round']}          {c['successes']:2d}/20 / {p['successes']:2d}/20"
              f"                 {c['calls']:4d} / {p['calls']:4d}")
    c, p = report['totals'][control], report['totals'][pruned]
    print(f"Total          {c['successes']}/60 / {p['successes']}/60"
          f"                 {c['calls']} / {p['calls']}")
    print(f"Call reduction: {100 * (1 - p['calls'] / c['calls']):.1f}%")
    print(f"Returned-token reduction: {100 * (1 - p['total_tokens'] / c['total_tokens']):.1f}%")
    print(f"Estimated-fee reduction: {100 * (1 - p['cny_off_peak'] / c['cny_off_peak']):.1f}%")
    print('These are repeated task-runs, not 60 independent test tasks.')
    print('Resource use falls; the accuracy difference changes sign.')
    pilot = read('results/low_score_dev_20261104/low_score_comparison.json')
    c, p = (pilot['arms'][name] for name in ('low_score_control_3', 'low_score_expand_3'))
    print('\nLOW-SCORE PILOT | 2 development tasks per arm')
    print(f"Successes: {c['independent_successes']}/2 vs {p['independent_successes']}/2; "
          f"calls: {c['cost']['completed_calls']} vs {p['cost']['completed_calls']}")
    print('Added candidates were not explored; no effectiveness gain is established.')
    print('\nRead-only summaries: no new search or model request is started.')

if __name__ == '__main__':
    main()
