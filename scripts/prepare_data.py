"""Freeze two development and twenty test tasks in upstream order."""
import subprocess
from pathlib import Path
from common import ROOT, UPSTREAM, DATA, read, write, sha

def main():
    source = DATA / 'bw_data_step_6.json'
    raw = read(source)
    selection = {'dev': list(range(2)), 'test': list(range(2,22))}
    manifest = {'upstream_commit': subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'], text=True).strip(),
                'source': str(source.relative_to(ROOT)), 'source_sha256':sha(source),
                'selection_rule':'upstream order: first 2 development, next 20 test; not claimed to be paper IDs',
                'source_count':len(raw), 'splits':{}}
    for split, indices in selection.items():
        write(ROOT / 'data' / f'{split}.json', [raw[i] for i in indices])
        manifest['splits'][split] = [dict(index=j, source_index=i,
            task_id=Path(raw[i][0]).stem,
            pddl_sha256=sha(DATA / 'generated_basic' / Path(raw[i][0]).name)) for j,i in enumerate(indices)]
        manifest['splits'][split + '_sha256'] = sha(ROOT / 'data' / f'{split}.json')
    files=[source, DATA/'bw_data_bw_config.yaml', DATA/'bw_data_generated_domain.pddl',
           ROOT/'data/dev.json',ROOT/'data/test.json']
    files += [DATA/'generated_basic'/Path(raw[i][0]).name for indices in selection.values() for i in indices]
    manifest['file_hashes']={str(p.relative_to(ROOT)):sha(p) for p in files}
    write(ROOT / 'data/manifest.json', manifest)
    print(f"Frozen {len(raw)} source tasks -> dev=2, test=20")

if __name__ == '__main__':
    main()
