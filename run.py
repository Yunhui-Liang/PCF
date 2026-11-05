"""Run PCF on one probability tensor."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from pcf import PCF
from pcf.io import load_probabilities,load_labels,split_indices,classification_metrics

def run(input_path,protocol,seed,output_path,evaluate=False,n_jobs=1):
    p=load_probabilities(input_path);n,m,k=p.shape
    if protocol=='inductive':train_ids,test_ids=split_indices(n,seed)
    elif protocol=='transductive':train_ids=test_ids=np.arange(n,dtype=np.int64)
    else:raise ValueError(protocol)
    # Match the archived 80/20 runner initialization convention.
    model_seed=seed*10 if protocol=='inductive' else seed
    start=time.perf_counter()
    model=PCF(n_jobs=n_jobs,random_state=model_seed).fit(p[train_ids])
    q=model.predict_proba(p[test_ids])
    if q.shape!=(len(test_ids),k) or not np.isfinite(q).all() or np.any(q<0) or not np.allclose(q.sum(1),1,atol=1e-5):
        raise RuntimeError('Invalid PCF probability output.')
    result={'protocol':protocol,'seed':int(seed),'n_fit':len(train_ids),'n_predict':len(test_ids),
            'n_sources':m,'n_classes':k,'seconds':time.perf_counter()-start,
            'input_sha256':hashlib.sha256(Path(input_path).read_bytes()).hexdigest()}
    output=Path(output_path);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,q=q,pred=q.argmax(1),sample_ids=test_ids,train_ids=train_ids,
                        protocol=np.asarray(protocol),seed=np.asarray(seed))
    # Evaluation labels are read only after fitting and saving the predictions.
    if evaluate:
        y=load_labels(input_path,n,k)
        result.update(classification_metrics(y[test_ids],q))
    output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--protocol',choices=['inductive','transductive'],default='transductive')
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--output',type=Path,default=Path('results/prediction.npz'))
    parser.add_argument('--evaluate',action='store_true')
    parser.add_argument('--n-jobs',type=int,default=1)
    args=parser.parse_args()
    if args.output.suffix.lower()!='.npz':parser.error('--output must end in .npz')
    print(json.dumps(run(args.input,args.protocol,args.seed,args.output,args.evaluate,args.n_jobs),indent=2))

if __name__=='__main__':main()
