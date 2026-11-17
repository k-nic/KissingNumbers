"""Regenerate a configuration from the Golay shell and its compact certificate."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import json
import time
import numpy as np
from common import ROOT, require, integer_array, leech_shell, lexsort, manifest, canonical_hashes, save_points
from geometry import dot_blocks


def certificate(name):
    spec=manifest()['records'][name]
    with np.load(ROOT/spec['certificate'],allow_pickle=False) as z:
        return {k:integer_array(z[k],k,z[k].ndim) for k in z.files}


def conflict_set(v,a,n,d,direct=False,block=256):
    require(block>0,'block size')
    bodies=np.unique(a[:,:24],axis=0)
    bad=np.zeros(len(v),bool)
    for s in range(0,len(bodies),block):
        bad |= np.any(dot_blocks(bodies[s:s+block],v,direct)>64,axis=0)
    require(np.all(d>0),'denominator sign')
    for row,den0 in zip(n,d):
        den=int(den0)
        require(24*8*max(map(abs,map(int,row[:24]))) < 2**63,'rational/shell overflow')
        bad |= v.astype(np.int64)@row[:24] > 64*den
    return np.flatnonzero(bad)


def construct(name,output=None,direct=False,report=None):
    start=time.time();spec=manifest()['records'][name];c=certificate(name)
    v=leech_shell();a=c['added_integer'];n=c['fraction_numerators'];d=c['fraction_denominators']
    dim=spec['dimension'];deleted=c['deleted_leech_indices']
    require(a.ndim==2 and a.shape[1]==dim,'added-point shape')
    require(n.shape==(len(d),dim),'fraction shape')
    require(deleted.ndim==1 and np.all((deleted>=0)&(deleted<len(v))),'deleted indices')
    require(np.array_equal(deleted,np.unique(deleted)),'deleted indices must be sorted and unique')
    require(np.all(np.any(a[:,24:]!=0,axis=1)),'zero-tail integer addition')
    require(np.all(np.any(n[:,24:]!=0,axis=1)),'zero-tail rational addition')
    actual=conflict_set(v,a,n,d,direct)
    require(np.array_equal(actual,deleted),'deletions differ from recomputed conflict set')
    keep=np.ones(len(v),bool);keep[deleted]=False
    bulk=np.zeros((int(keep.sum()),dim),np.int64);bulk[:,:24]=v[keep]
    x=lexsort(np.concatenate([bulk,a]))
    require(len(x)+len(d)==spec['cardinality'],'reconstructed size')
    require(canonical_hashes(x,n,d)==spec['canonical_hashes'],'reconstructed coordinate hash')
    if output:save_points(output,x,n,d)
    result={'status':'PASS','record':name,'retained_leech_points':len(bulk),
            'deleted_leech_points':len(deleted),'added_integer_points':len(a),
            'rational_points':len(d),'cardinality':len(x)+len(d),
            'all_compatible_leech_vectors_retained':True,
            'canonical_hashes':canonical_hashes(x,n,d),'seconds':round(time.time()-start,3)}
    if report:
        from pathlib import Path
        Path(report).parent.mkdir(parents=True,exist_ok=True)
        Path(report).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    return x,n,d


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('record',choices=list(manifest()['records']))
    ap.add_argument('--output');ap.add_argument('--report')
    ap.add_argument('--direct',action='store_true')
    a=ap.parse_args();construct(a.record,a.output,a.direct,a.report)
