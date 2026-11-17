"""Reconstruct and verify all primary records and the supplied dimension-29 tie."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import json
from pathlib import Path
from common import ROOT,manifest
from construct import construct
from verify import verify
from check_structure import check
from negative_controls import run


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reports',type=Path,default=ROOT/'fresh_reports')
    ap.add_argument('--primary-only',action='store_true')
    ap.add_argument('--direct',action='store_true',help='slow int64 kernel for every large matrix product')
    a=ap.parse_args();a.reports.mkdir(parents=True,exist_ok=True)
    names=[n for n in manifest()['records'] if not (a.primary_only and n.endswith('_alt'))]
    controls=run();(a.reports/'negative_controls.json').write_text(json.dumps(controls,indent=2)+'\n')
    for name in names:
        construct(name,direct=a.direct,report=a.reports/f'construct_{name}.json')
        verify(name,direct=a.direct,output=a.reports/f'verify_{name}.json')
    check(names,a.reports/'structure.json')
    result={'status':'ALL CHECKS PASS','records':names,'exact':True}
    (a.reports/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
