"""Synthetic geometry only: balanced Fermat groups with disk-overlap constraints.

No diffraction simulation, network training, or reconstruction claim.
Scan point density matches 1/(12 px)^2, NOT an exact 12 px nearest spacing.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree


def disk_overlap(distance, diameter):
    t = np.clip(np.asarray(distance)/diameter, 0, 1)
    return (2/np.pi)*(np.arccos(t)-t*np.sqrt(1-t*t))


def main():
    n, k, diameter, step, threshold = 100, 4, 59.1, 12., .35
    radius = step*np.sqrt(n/np.pi)
    i = np.arange(n)
    r = step/np.sqrt(np.pi)*np.sqrt(i+.5)
    theta = i*np.pi*(3-np.sqrt(5))
    xy = np.column_stack([r*np.cos(theta),r*np.sin(theta)])
    dist = cdist(xy,xy)
    overlap = disk_overlap(dist,diameter)
    groups, left = [], set(range(n))
    while left:
        seed = max(left,key=lambda j:(r[j],j))
        local = [seed]+sorted(left-{seed},key=lambda j:(dist[seed,j],j))[:3]
        groups.append(local)
        left.difference_update(local)
    labels = np.empty(n,int)
    for g,local in enumerate(groups):
        for j,p in enumerate(local):
            labels[p] = (j+g)%k
    initial_labels = labels.copy()
    def group_score(ids):
        d = dist[np.ix_(ids,ids)]
        mst = minimum_spanning_tree(d).tocoo()
        edge_overlap = disk_overlap(mst.data,diameter)
        deficits = np.maximum(threshold-edge_overlap,0)
        # First make connected components join at the required overlap;
        # within-group redundancy then favors distributed coverage.
        penalty = 20*np.count_nonzero(deficits)+500*np.sum(deficits**2)
        redundancy = (overlap[np.ix_(ids,ids)].sum()-len(ids))/(2*n)
        return float(penalty+redundancy)
    scores = [group_score(np.flatnonzero(labels==g)) for g in range(k)]
    best, best_score = labels.copy(), sum(scores)
    rng = np.random.default_rng(0)
    accepted = 0
    for t in range(12000):
        a,b = rng.choice(n,2,replace=False)
        ga,gb = labels[a],labels[b]
        if ga==gb:
            continue
        labels[a],labels[b] = gb,ga
        sa,sb = group_score(np.flatnonzero(labels==ga)),group_score(np.flatnonzero(labels==gb))
        delta = sa+sb-scores[ga]-scores[gb]
        temperature = .025*(.0001/.025)**(t/11999)
        if delta<0 or rng.random()<np.exp(-max(delta,0)/temperature):
            scores[ga],scores[gb] = sa,sb
            accepted += 1
            if sum(scores)<best_score:
                best,best_score = labels.copy(),sum(scores)
        else:
            labels[a],labels[b] = ga,gb
    labels = best
    mesh = np.arange(-radius-diameter/2,radius+diameter/2+.5,.5)
    gx,gy = np.meshgrid(mesh,mesh)
    sample = np.column_stack([gx.ravel(),gy.ravel()])
    support = cdist(xy,sample)<=diameter/2
    scan_disk = np.sum(sample**2,axis=1)<=radius**2
    union = support.any(0)
    def audit(assign):
        result=[]
        for g in range(k):
            ids = np.flatnonzero(assign==g)
            d = dist[np.ix_(ids,ids)]
            mst = minimum_spanning_tree(d).tocoo()
            edge_overlap = disk_overlap(mst.data,diameter)
            mask = support[ids].any(0)
            nearest = (d+np.eye(len(ids))*1e6).min(1)
            result.append(dict(name='ABCD'[g],indices_1based=(ids+1).tolist(),count=len(ids),
                mst_edges=[[int(ids[a]+1),int(ids[b]+1)] for a,b in zip(mst.row,mst.col)],
                mst_min_overlap=float(edge_overlap.min()),mst_mean_overlap=float(edge_overlap.mean()),
                mst_max_distance=float(mst.data.max()),
                components_at_threshold=int(1+np.count_nonzero(edge_overlap<threshold)),
                mean_nearest_distance=float(nearest.mean()),
                coverage_of_scan_disk=float(mask[scan_disk].mean()),
                coverage_of_full_support_union=float(mask[union].mean())))
        return result
    result=dict(description=__doc__,N=n,K=k,probe_diameter_px=diameter,
        scan_radius_px=radius,density_equivalent_step_px=step,area_overlap_threshold=threshold,
        coordinates_px=xy.tolist(),labels=labels.tolist(),initial_labels=initial_labels.tolist(),
        algorithm='outer-first nearest four; cyclic ABCD seed; 12000 balanced pair-swap annealing trials; seed 0',
        accepted_swaps=accepted,score=best_score,grid_spacing_for_coverage_px=.5,
        initial=audit(initial_labels),final=audit(labels))
    assert sorted(np.bincount(labels))==[25]*4
    assert all(g['components_at_threshold']==1 for g in result['final'])
    out=Path('output/fermat_partition_example_20261004')
    out.mkdir(parents=True,exist_ok=True)
    (out/'geometry.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savetxt(out/'scan_groups.csv',np.column_stack([np.arange(1,n+1),xy,labels]),delimiter=',',
        header='measurement_id,x_px,y_px,group_0_to_3',comments='',fmt=['%d','%.8f','%.8f','%d'])
    print(json.dumps({key:result[key] for key in ['initial','final','accepted_swaps','score']},indent=2))


if __name__=='__main__':
    main()
