"""Check the geometric descriptions and all midpoint endpoint witnesses."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import collections
import itertools
import json
import numpy as np
from common import ROOT,I,J,require,manifest,load_points,leech_shell,layer_counts,lexsort
from construct import certificate
from geometry import check_rationals


def selected_edge_graph(edges):
    edges,counts=np.unique(np.sort(edges,axis=1),axis=0,return_counts=True)
    graph=collections.defaultdict(set)
    for i,j in edges:
        graph[int(i)].add(int(j));graph[int(j)].add(int(i))
    seen=set();components=[]
    for start in graph:
        if start in seen:continue
        seen.add(start);stack=[start];size=0
        while stack:
            v=stack.pop();size+=1
            for w in graph[v]:
                if w not in seen:seen.add(w);stack.append(w)
        components.append(size)
    return {'edges':len(edges),'vertices':len(graph),
        'edge_multiplicities':dict(sorted(collections.Counter(map(int,counts)).items())),
        'degree_histogram':dict(sorted(collections.Counter(map(len,graph.values())).items())),
        'component_size_histogram':dict(sorted(collections.Counter(components).items()))}


def check_shell_structure(v):
    profile=layer_counts(v.astype(np.int64))
    require(profile=={'0,128,0':4320,'32,96,0':122880,'64,64,0':69120,'128,0,0':240},'Leech octad decomposition')
    fv=v[np.sum(v[:,I].astype(np.int64)**2,axis=1)==64]
    w,inv,counts=np.unique(fv[:,J],axis=0,return_inverse=True,return_counts=True)
    require(len(w)==4320 and np.all(counts==16),'16-point fibres')
    frames=[]
    for f in range(len(w)):
        heads=fv[inv==f][:,I].astype(np.int64)
        first=np.argmax(heads!=0,axis=1)
        u=heads[heads[np.arange(len(heads)),first]>0]
        require(u.shape==(8,8) and np.array_equal(u@u.T,64*np.eye(8,dtype=np.int64)), 'orthogonal fibre frame')
        require(np.array_equal(lexsort(heads),lexsort(np.concatenate([u,-u]))),'antipodal fibre frame')
        frames.append(lexsort(u).tobytes())
    require(sorted(collections.Counter(frames).values())==[32]*135,'135 frame classes')
    return {'status':'PASS','shell_points':len(v),'octad_profile':profile,'fibres':4320,'frame_classes':135,'fibres_per_class':32}


def check_cube28(x,v):
    c=json.loads((ROOT/'data/cube28.json').read_text())
    require(c['octad']==I,'cube coordinate convention')
    bases=np.array(c['bases'],np.int64);tails=np.array(c['tails'],np.int64)
    flat=bases.reshape(8,8)
    require(np.array_equal(flat@flat.T,32*np.eye(8,dtype=np.int64)),'cube orthogonality')
    require(np.array_equal(tails@tails.T,32*np.eye(2,dtype=np.int64)),'cube tail orthogonality')
    added=[];removed=[]
    for base in bases:
        for signs in itertools.product((-1,1),repeat=4):
            z=np.zeros(28,np.int64);z[I]=np.array(signs)@base;removed.append(z)
        for omit,tail in enumerate(tails):
            for signs in itertools.product((-1,1),repeat=3):
                for sign in (-1,1):
                    z=np.zeros(28,np.int64);z[I]=np.array(signs)@np.delete(base,omit,axis=0);z[24:]=sign*tail;added.append(z)
    added=np.array(added);removed=np.array(removed)
    aset=set(map(tuple,added));rset=set(map(tuple,removed));xset=set(map(tuple,x))
    require(len(aset)==64 and len(rset)==32,'cube sizes')
    require(aset<=xset and not rset&xset,'cube set membership')
    vset=set(map(tuple,v));require(all(tuple(z[:24]) in vset for z in removed),'cube vertices in Leech shell')
    other=x[np.array([tuple(z) not in aset for z in x])]
    maxima=[]
    for base in bases:
        for omit,tail in enumerate(tails):
            bound=np.abs(other[:,I]@np.delete(base,omit,axis=0).T).sum(1)+np.abs(other[:,24:]@tail)
            require(np.all(bound<=64),'cube external compatibility')
            maxima.append(int(bound.max()))
    # The union after restoring the vertices is the stated 223464-point precursor.
    for s in range(0,len(other),4096):
        require(np.all(removed@other[s:s+4096].T<=64),'cube precursor compatibility')
    gram=removed@removed.T;np.fill_diagonal(gram,-128)
    require(np.all(gram<=64),'cube precursor internal pairs')
    return {'deleted_vertices':32,'inserted_midpoints':64,'precursor_cardinality':len(other)+32,
            'external_inequality_maxima':maxima}


def check_one(name,v):
    spec=manifest()['records'][name];x,n,d=load_points(ROOT/spec['coordinates']);c=certificate(name)
    require(layer_counts(x)==spec['integer_layer_counts'],'layer table mismatch')
    a=c['added_integer'];rows=c['midpoint_rows'];ends=c['midpoint_endpoints'];deleted=c['deleted_leech_indices']
    expected=np.flatnonzero(np.sum(a[:,24:]**2,axis=1)==32)
    require(np.array_equal(rows,expected),'midpoint row indices')
    require(ends.shape==(len(rows),2),'endpoint shape')
    require(np.all((ends>=0)&(ends<len(v))),'endpoint index range')
    p=v[ends[:,0]].astype(np.int64);q=v[ends[:,1]].astype(np.int64)
    require(np.all(p+q==2*a[rows,:24]),'midpoint reconstruction')
    require(np.all(np.sum(p*q,axis=1)==64),'midpoint contact edges')
    require(np.all(np.isin(ends,deleted)),'midpoint endpoint not deleted')
    norms=np.stack([np.sum(a[rows][:,cols]**2,axis=1) for cols in [I,J]],axis=1)
    fibre=np.all(norms==[32,64],axis=1);row=np.all(norms==[64,32],axis=1)
    require(np.all(p[fibre][:,J]==q[fibre][:,J]),'fibre midpoint definition')
    require(np.all(p[row][:,I]==q[row][:,I]),'row midpoint definition')
    bulk=x[np.all(x[:,24:]==0,axis=1)]
    require(np.sum(np.sum(bulk[:,I]**2,axis=1)==32)==122880,'unchanged 122880-point shell layer')
    keep=np.ones(len(v),bool);keep[deleted]=False
    require(np.array_equal(lexsort(bulk[:,:24]),v[keep]),'retained shell equality')
    require(layer_counts(v[deleted].astype(np.int64))==spec['deletion_type_counts'],'deletion table')
    # Complete tail-root system, with no assumed symmetry of the other layers.
    k=spec['dimension']-24;poles=[]
    for i,j in itertools.combinations(range(k),2):
        for si,sj in itertools.product((-8,8),repeat=2):
            z=np.zeros(spec['dimension'],np.int64);z[24+i]=si;z[24+j]=sj;poles.append(z)
    actual=x[np.all(x[:,:24]==0,axis=1)]
    require(np.array_equal(lexsort(actual),lexsort(np.array(poles))),'pole set')
    # Fibre occupancy and deletion histogram.
    fmask=np.sum(v[:,I].astype(np.int64)**2,axis=1)==64
    w,fi=np.unique(v[fmask][:,J],axis=0,return_inverse=True)
    fdict={tuple(z):i for i,z in enumerate(w)}
    aidx=np.array([fdict[tuple(z)] for z in a[rows[fibre]][:,J]])
    acount=np.bincount(aidx,minlength=len(w))
    dcount=np.bincount(fi,weights=(~keep)[fmask],minlength=len(w)).astype(np.int64)
    states,nums=np.unique(np.stack([dcount,acount],axis=1),axis=0,return_counts=True)
    histogram=[{'deleted':int(s[0]),'inserted':int(s[1]),'fibres':int(c)} for s,c in zip(states,nums)]
    stored=json.loads((ROOT/'data'/f'{name}_structure.json').read_text())
    require(histogram==stored['fibre_state_histogram'],'fibre histogram')
    egraph=selected_edge_graph(ends[np.all(norms==[96,0],axis=1)])
    bgraph=selected_edge_graph(ends[np.all(norms==[0,96],axis=1)])
    endpoint_sets={label:set(map(int,np.unique(ends[np.all(norms==typ,axis=1)])))
                   for label,typ in [('fibre',[32,64]),('row',[64,32]),('E8',[96,0]),('BW',[0,96])]}
    row_new=len(endpoint_sets['row']-endpoint_sets['fibre'])
    require(row_new=={'d26':0,'d27':64,'d28':0,'d29':0,'d29_alt':0}[name], 'additional row endpoints')
    result={'status':'PASS','record':name,'integer_layers':layer_counts(x),
            'deletion_types':spec['deletion_type_counts'],'midpoint_points_checked':len(rows),
            'fibre_state_histogram':histogram,'E8_selected_edge_graph':egraph,
            'BW_selected_edge_graph':bgraph,'poles':len(poles),
            'distinct_endpoints_by_layer':{k:len(s) for k,s in endpoint_sets.items()},
            'row_endpoints_outside_fibre_layer':row_new}
    if name in ('d26','d27'):
        aux=a[(np.sum(a[:,I]**2,axis=1)==64)&(np.sum(a[:,J]**2,axis=1)==0)&(np.sum(a[:,24:]**2,axis=1)==64)]
        tails=np.unique(aux[:,24:],axis=0);frames=[]
        for t in tails:
            heads=lexsort(aux[np.all(aux[:,24:]==t,axis=1)][:,I]);gram=heads@heads.T
            require(len(heads)==16 and set(np.unique(gram))=={-64,0,64},'auxiliary eight-frame')
            require(np.all(np.sum(gram==-64,axis=1)==1),'auxiliary antipodes')
            frames.append(heads.tobytes())
        result['E8_auxiliary_distinct_frames']=len(set(frames))
        require(len(set(frames))==(1 if name=='d26' else 3),'auxiliary frame count')
    if name=='d26':
        require(np.flatnonzero(np.any(n!=0,axis=0)).tolist()==I+[24,25],'rational support')
        patch=check_rationals(x,n,d)
        require(patch['maximum_affected_unit_product_numerator']*100000<=49771*patch['maximum_affected_unit_product_denominator'],'strict rational margin')
        endpoint_union=np.unique(ends)
        extra=np.setdiff1d(deleted,endpoint_union)
        require(len(extra)==32 and np.all(np.sum(v[extra][:,I].astype(np.int64)**2,axis=1)==128),'32 additional E8 deletions')
        # These 32 vertices really can be restored if the rational patch is removed.
        for s in range(0,len(x),4096):
            require(np.all(v[extra].astype(np.int64)@x[s:s+4096,:24].T<=64),'integer precursor compatibility')
        gram=v[extra].astype(np.int64)@v[extra].astype(np.int64).T
        np.fill_diagonal(gram,-128)
        require(np.all(gram<=64),'integer precursor restored pairs')
        result.update(rational_patch=patch,additional_E8_deletions=32,integer_precursor_cardinality=len(x)+32)
    if name=='d27':
        require(egraph['component_size_histogram']=={8:16} and egraph['degree_histogram']=={2:128},'E8 eight-cycles')
        require(bgraph['component_size_histogram']=={4:96,8:16} and bgraph['degree_histogram']=={2:512},'BW cycle decomposition')
        require(egraph['edge_multiplicities']=={2:128} and bgraph['edge_multiplicities']=={2:512},'two lifts per edge')
    if name=='d28':result['cube_exchange']=check_cube28(x,v)
    return result


def check(names=None,reports=None):
    v=leech_shell();result={'shell':check_shell_structure(v),'records':{}}
    for name in (names or list(manifest()['records'])):
        result['records'][name]=check_one(name,v)
        print(name+': structure PASS',flush=True)
    if 'd29' in result['records'] and 'd29_alt' in result['records']:
        a=load_points(ROOT/manifest()['records']['d29']['coordinates'])[0]
        b=load_points(ROOT/manifest()['records']['d29_alt']['coordinates'])[0]
        common=len(set(map(tuple,a))&set(map(tuple,b)))
        require(common==228218,'dimension-29 tie intersection')
        result['dimension29_alternatives']={'intersection':common,'exclusive_each':len(a)-common,'isometry_claim':None}
    if reports:
        from pathlib import Path
        Path(reports).parent.mkdir(parents=True,exist_ok=True)
        Path(reports).write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('records',nargs='*');ap.add_argument('--report',default=None)
    a=ap.parse_args();check(a.records,a.report)
