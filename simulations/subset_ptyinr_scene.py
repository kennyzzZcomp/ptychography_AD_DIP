"""Select a central raster subset without regenerating or renormalizing data."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--grid',type=int,default=16)
    a=p.parse_args()
    if a.output.exists(): p.error('Output already exists; choose a new name')
    with h5py.File(a.source,'r') as src:
        pos=src['positions_px'][()]
        ys,xs=np.unique(pos[:,0]),np.unique(pos[:,1])
        full=len(ys); g=a.grid
        if len(xs)!=full or not np.array_equal(pos,np.array([(y,x) for y in ys for x in xs])):
            raise ValueError('Expected square row-major raster')
        if not 1<=g<=full: raise ValueError('Invalid subset grid')
        start=(full-g)//2
        ids=np.array([r*full+c for r in range(start,start+g) for c in range(start,start+g)])
        selected=pos[ids]
        probe=src['diagnostic_truth/probe_normalized'][()]
        n=probe.shape[0]; shape=tuple(src['object_shape'][()])
        cov=np.zeros(shape,np.float64)
        for y,x in selected: cov[y:y+n,x:x+n]+=np.abs(probe)**2
        mask=cov>.5*cov.max()
        rr,cc=np.where(mask)
        roi=np.array([rr.min(),rr.max()+1,cc.min(),cc.max()+1],dtype=np.int64)
        if min(roi[1]-roi[0],roi[3]-roi[2])<7: raise ValueError('ROI too small for evaluation')
        cfg=json.loads(src.attrs['scene_config_json'])
        cfg.update(grid=g,eval_size=0)
        h=hashlib.sha1()
        # Same constituent order as the original scene fingerprint.
        for v in [src['diagnostic_truth/object'][()],src['diagnostic_truth/probe'][()],selected,
                  src['intensity'][ids],src['diagnostic_truth/clean_intensity'][ids],src['probe_init'][()]]:
            h.update(np.ascontiguousarray(v).tobytes())
        report=dict(source=str(a.source.resolve()),source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
            source_scene_fingerprint=str(src.attrs['scene_fingerprint']),scene_fingerprint=h.hexdigest()[:12],
            grid=g,source_grid=full,source_axis_indices=[start,start+g-1],selected_indices=ids.tolist(),
            positions_min=selected.min(0).tolist(),positions_max=selected.max(0).tolist(),
            roi=roi.tolist(),source_roi=src['roi'][()].tolist(),
            measurement_scale='Unchanged source global normalization; no subset renormalization',
            roi_rule='Bounding box of subset illumination coverage above 50% maximum; evaluation only',
            geometry_note='Absolute positions retained; scene_config alone cannot regenerate this subset')
        a.output.parent.mkdir(parents=True,exist_ok=True)
        with h5py.File(a.output,'x') as dst:
            for k,v in src.attrs.items(): dst.attrs[k]=v
            for key in src:
                if key in ('diffamp','intensity'):
                    dst.create_dataset(key,data=src[key][ids])
                elif key=='positions_px': dst.create_dataset(key,data=selected)
                elif key=='roi': dst.create_dataset(key,data=roi)
                elif key=='diagnostic_truth':
                    group=dst.create_group(key)
                    for k,v in src[key].attrs.items(): group.attrs[k]=v
                    for name in src[key]:
                        if name=='clean_intensity': group.create_dataset(name,data=src[key][name][ids])
                        else: src.copy(src[key][name],group,name=name)
                else: src.copy(src[key],dst,name=key)
            dst.attrs['scene_fingerprint']=report['scene_fingerprint']
            dst.attrs['scene_config_json']=json.dumps(cfg)
            dst.attrs['subset_provenance_json']=json.dumps(report)
    # Verify byte-identical copied fields and exact selection, not just shapes.
    with h5py.File(a.source,'r') as src,h5py.File(a.output,'r') as dst:
        for key in ('diffamp','intensity','diagnostic_truth/clean_intensity','positions_px'):
            np.testing.assert_array_equal(dst[key][()],src[key][ids])
        for key in ('quadratic_phase','probe_init','diagnostic_truth/object','diagnostic_truth/probe_normalized'):
            np.testing.assert_array_equal(dst[key][()],src[key][()])
    report['output_sha256']=hashlib.sha256(a.output.read_bytes()).hexdigest()
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='selected_indices'},indent=2))


if __name__=='__main__': main()
