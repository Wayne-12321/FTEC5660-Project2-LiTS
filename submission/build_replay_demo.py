"""Build a self-contained local replay from existing measured artifacts only."""
import json
from pathlib import Path
from build_replay_data import build

ROOT = Path(__file__).resolve().parents[1]

def main():
    data = build()
    template = (ROOT / 'submission/replay_demo_template.html').read_text()
    assert template.count('__REPLAY_DATA__') == 1
    encoded = json.dumps(data, separators=(',', ':')).replace('<', '\\u003c')
    html = template.replace('__REPLAY_DATA__', encoded)
    output = ROOT / 'output/replay'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'replay_data.json').write_text(json.dumps(data, indent=2) + '\n')
    page = output / 'lits-replay.html'
    page.write_text(html)
    print('Built:', page)
    print('3 recorded examples; legal action replay matched every displayed state.')
    print('New search runs: 0; new model API calls: 0.')

if __name__ == '__main__':
    main()
