"""Known invalid examples must be rejected by the actual geometry checkers."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
import numpy as np
from common import require
from geometry import check_integer,check_rationals


def run():
    rejected=[]
    tests={
        'duplicate':lambda:check_integer(np.array([[1,2,0],[1,2,0]]),5,2),
        'wrong_norm':lambda:check_integer(np.array([[1,1,0]]),5,2),
        'unpaired_positive_conflict':lambda:check_integer(np.array([[1,2,0],[2,1,0]]),5,2),
        'conflict_with_negative_member_of_pair':lambda:check_integer(np.array([[1,2,0],[-1,-2,0],[-2,-1,0]]),5,2),
        'negative_rational_denominator':lambda:check_rationals(np.array([[10,0]]),np.array([[0,10]]),np.array([-1]),100,50),
        'rational_wrong_norm':lambda:check_rationals(np.array([[10,0]]),np.array([[1,1]]),np.array([1]),100,50),
        'rational_integer_conflict':lambda:check_rationals(np.array([[0,10]]),np.array([[6,8]]),np.array([1]),100,50),
        'rational_rational_conflict':lambda:check_rationals(np.empty((0,2),np.int64),np.array([[6,8],[8,6]]),np.array([1,1]),100,50),
        'equal_rationals_different_denominators':lambda:check_rationals(np.empty((0,2),np.int64),np.array([[6,8],[12,16]]),np.array([1,2]),100,50),
    }
    for name,f in tests.items():
        try:f()
        except ValueError:rejected.append(name)
        else:raise ValueError('negative control accepted: '+name)
    # Explicit positive controls, with and without a complete antipodal pair.
    for direct in (False,True):
        for x in (np.array([[1,1,0],[1,0,1]]),np.array([[1,1,0],[-1,-1,0],[1,0,1]])):
            r=check_integer(x,2,1,block=1,direct=direct)
            require(r['maximum_inner_product']==1,'positive boundary example')
    return {'status':'PASS','invalid_examples_rejected':rejected,'positive_boundary_examples':4}


if __name__=='__main__':print(json.dumps(run(),indent=2))
