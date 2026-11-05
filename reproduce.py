"""Reproduce PCF on the two included tasks under either or both protocols."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from run import run
from pcf.io import DATASETS

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',required=True,type=Path)
    p.add_argument('--protocol',choices=['inductive','transductive','both'],default='both')
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=list(DATASETS))
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(2026,2031)))
    p.add_argument('--output-dir',type=Path,default=Path('results/reproduction'))
    a=p.parse_args()
    if len(set(a.seeds))!=len(a.seeds):p.error('Seeds must be unique.')
    if len(set(a.datasets))!=len(a.datasets):p.error('Datasets must be unique.')
    for ds in a.datasets:
        if not (a.data_dir/(ds+'.mat')).is_file():p.error(f'Missing input: {ds}.mat')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for protocol in (['inductive','transductive'] if a.protocol=='both' else [a.protocol]):
        for seed in a.seeds:
            for ds in a.datasets:
                output=a.output_dir/protocol/f'{ds}_{seed}.npz'
                result=run(a.data_dir/(ds+'.mat'),protocol,seed,output,evaluate=True)
                row={'dataset':ds,**result};rows.append(row)
                print(protocol,ds,seed,f'ACC={row["ACC"]:.4f}',flush=True)
                with (a.output_dir/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={}
    for protocol in sorted({r['protocol'] for r in rows}):
        seed_means={}
        for seed in a.seeds:
            selected=[r for r in rows if r['protocol']==protocol and r['seed']==seed]
            seed_means[str(seed)]={key:float(np.mean([r[key] for r in selected])) for key in ['ACC','MacroF1','Brier']}
        summary[protocol]={key:{'mean':float(np.mean([v[key] for v in seed_means.values()])),
                              'sample_sd':float(np.std([v[key] for v in seed_means.values()],ddof=1)) if len(a.seeds)>1 else None}
                           for key in ['ACC','MacroF1','Brier']}
        summary[protocol]['datasets']=a.datasets;summary[protocol]['per_seed_task_means']=seed_means
    (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
