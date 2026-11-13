"""Read only saved arrays; check projections, identities and clocks, no forward."""
from datetime import date,timedelta
import json
import numpy as np
import prepare as p
def main():
    proto=json.loads((p.P/'protocol.json').read_bytes());tasks=proto['tasks'];physical=set();mature=set();versions=0
    om=json.loads((p.O/'manifest.json').read_bytes())['files'];pm=json.loads((p.PP/'manifest.json').read_bytes())['files']
    checked=set();mapping={}
    for t in tasks:
        tid=t['id'];bid=t['base_id'];sources=set(t['source_sites'])
        assert len(sources)==3 and t['donor'] not in sources and t['target'] not in sources and t['target']!=t['donor']
        m=p.arr((p.P/f'models/{bid}.npz').read_bytes());c=p.arr((p.P/f'inputs/{tid}/context.npz').read_bytes());g=p.arr((p.P/f'scores/{tid}.npz').read_bytes())
        orig=p.arr((p.O/f'inputs/{bid}/context.npz').read_bytes());ix=np.flatnonzero(orig['sites']==t['target'])
        for key in c:
            if key!='support':np.testing.assert_array_equal(c[key],orig[key][ix])
        np.testing.assert_array_equal(g['sample_ids'],c['sample_ids'])
        assert m['parameters'].shape==(502,) and m['C'].shape==(54,7) and list(m['source_sites'])==t['source_sites']
        if bid not in checked:
            oldm=p.arr((p.O/f'inputs/{bid}/model.npz').read_bytes());oldp=p.arr((p.PP/f'inputs/{bid}/model.npz').read_bytes())
            np.testing.assert_array_equal(oldm['head'],m['heads'][0]*m['gain_scale'][0])
            np.testing.assert_array_equal(oldm['C'],m['C']);np.testing.assert_array_equal(oldp['expert_parameters'],m['parameters'])
            np.testing.assert_array_equal(oldm['vwc_scale'],m['physical_scales'][1:2]);np.testing.assert_array_equal(oldm['offset'],m['offsets'][1:8])
            checked.add(bid)
        rel=json.loads((p.P/f'inputs/{tid}/releases.json').read_bytes());byh={r['history']:r for r in rel}
        for i,(site,day) in enumerate(zip(c['sites'],c['dates'])):
            physical.add((str(site),str(day)));versions+=1;dt=date.fromisoformat(str(day))
            candidates=[j for j in range(i) if c['sites'][j]==site and c['years'][j]==c['years'][i] and date.fromisoformat(str(c['dates'][j]))+timedelta(days=7)<=dt]
            expected=max(candidates,key=lambda j:c['dates'][j]) if candidates else -1
            assert int(c['support'][i])==expected
            if expected>=0:
                mature.add((str(site),str(day)));r=byh[expected];a=int(c['actions'][expected]);assert r['action']==a
                fb=p.arr((p.P/r['file']).read_bytes());assert str(fb['sample_id'])==c['sample_ids'][expected,a]
            for sid,gain in zip(g['sample_ids'][i],g['gain'][i]):
                if str(sid) in mapping:assert mapping[str(sid)]==gain
                mapping[str(sid)]=gain
    assert len(checked)==30 and len(physical)==64 and len(mature)==59 and versions==768
    # All saved source files remain byte-identical to authenticated ancestry.
    bindings=json.loads((p.P/'provenance.json').read_bytes())['bindings']
    for name,h in bindings.items():
        if '::' not in name:assert p.sha(name)==h
    p.put(p.H/'binding_validation.json',dict(passed=True,models=30,directions=60,versions=versions,physical_cycles=64,
        mature_physical_cycles=59,unified_head_bound=True,all_clocks_checked=True,feedback_capsules_exact=True,
        immutable_expert_parameters=True,source_files_unchanged=True,local_real_forward_calls=0))
    print('BINDING_PASS models=30 directions=60 physical=64 mature=59 versions=768')
if __name__=='__main__':main()
