"""Verify every unordered pair of a named coordinate certificate."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import json
import math
import platform
import time
import numpy as np
from common import ROOT, load_points, canonical_hashes, manifest, require
from geometry import check_integer, check_rationals


def verify(name, direct=False, block=512, output=None):
    started=time.time()
    spec=manifest()['records'][name]
    x,n,d=load_points(ROOT/spec['coordinates'])
    require(x.shape[1] == spec['dimension'], 'wrong dimension')
    require(len(x)+len(d) == spec['cardinality'], 'wrong cardinality')
    require(canonical_hashes(x,n,d) == spec['canonical_hashes'], 'coordinate identity mismatch')
    print(name+': checking integer pairs',flush=True)
    core=check_integer(x,block=block,direct=direct)
    patch=check_rationals(x,n,d)
    covered=core['unordered_pairs_covered']+patch['rational_core_pairs']+patch['rational_rational_pairs']
    require(covered == math.comb(spec['cardinality'],2), 'total pair coverage')
    report={'status':'PASS','record':name,'dimension':spec['dimension'],
            'cardinality':spec['cardinality'],'exact':True,'tolerance':None,
            'integer_geometry':core,'rational_geometry':patch,
            'total_unordered_pairs_covered':covered,'canonical_hashes':canonical_hashes(x,n,d),
            'python':platform.python_version(),'numpy':np.__version__,
            'seconds':round(time.time()-started,3)}
    if output:
        from pathlib import Path
        Path(output).parent.mkdir(parents=True,exist_ok=True)
        Path(output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('record',choices=list(manifest()['records']))
    ap.add_argument('--direct',action='store_true',help='int64 rather than provably exact binary64 kernel')
    ap.add_argument('--block',type=int,default=512)
    ap.add_argument('--report')
    a=ap.parse_args();verify(a.record,a.direct,a.block,a.report)
