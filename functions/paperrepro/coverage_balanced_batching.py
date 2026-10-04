"""Probe-intensity coverage-balanced mini-batches: experimental prototype.

This is a proposed objective/implementation, NOT a claim of literature novelty
or a guarantee of improved ptychographic reconstruction.

Requires only NumPy. All point coordinates are (x, y) in the SAME physical units.
Use the initial/calibrated probe intensity, not the simulation ground truth.
"""
from __future__ import annotations
import numpy as np
import time


def disk_window_partition(positions_yx, diameter, *, anchor_step=2.0, seed=0):
    """Training adapter for the geometry demo; no truth or measurement access.

    Four fixed equal-capacity groups, ALL positions retained. Pixel positions
    supplied by the solver are (y,x); translation is removed before quadrature.
    Only a nominal disk is used, NOT the learned/truth probe or fusion masks.
    Unequal groups are rejected: this pilot has fixed input channels and no
    padding/masking policy. The standalone partitioner supports unequal groups.
    """
    started = time.perf_counter()
    points = _points(positions_yx, 'positions_yx')[:, ::-1].copy()
    if len(points) < 4 or len(points) % 4:
        raise ValueError('coverage-balanced training requires the number of measurements divisible by 4 (equal input channels); odd grids are not yet supported')
    if not np.isfinite(diameter) or diameter <= 0:
        raise ValueError('coverage diameter must be finite and positive')
    if not np.isfinite(anchor_step) or anchor_step <= 0:
        raise ValueError('coverage anchor step must be finite and positive')
    points -= points.min(axis=0)
    radius = diameter / 2

    def features(spacing, offset=(0., 0.)):
        lo, hi = points.min(0)-radius-2, points.max(0)+radius+2
        xs = np.arange(lo[0]+offset[0], hi[0], spacing)
        ys = np.arange(lo[1]+offset[1], hi[1], spacing)
        # Fail before allocating unexpectedly huge dense feature matrices.
        if len(points)*len(xs)*len(ys) > 50_000_000:
            raise ValueError('Coverage feature grid too large for this dense pilot')
        xx, yy = np.meshgrid(xs, ys)
        anchors = np.column_stack((xx.ravel(), yy.ravel()))
        result = np.empty((len(points), len(anchors)), dtype=np.float64)
        for i, point in enumerate(points):
            result[i] = np.sum((anchors-point)**2, axis=1) <= radius**2
        return result

    train = features(anchor_step)
    groups, info = partition_features(train, 4, seed=seed, n_restarts=1)
    partition_seconds = time.perf_counter()-started
    evaluation = features(1.0, (.37, .61))
    initial = random_balanced_groups(len(points), 4, seed)
    audit = dict(info, model='nominal uniform disk; geometry proxy only',
                 diameter_px=float(diameter), anchor_step_px=float(anchor_step),
                 seed=int(seed), n_restarts=1,
                 E_percent=100*coverage_error(evaluation, groups),
                 initial_random_E_percent=100*coverage_error(evaluation, initial),
                 eval_anchor_step_px=1.0, eval_anchor_offset_px=[.37, .61],
                 partition_seconds=partition_seconds,
                 setup_with_diagnostics_seconds=time.perf_counter()-started,
                 channel_order='ascending original measurement index within each group',
                 truth_used=False, all_measurements_retained=True)
    return [g.tolist() for g in groups], audit


def _points(a, name):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 2 or len(a) == 0 or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a nonempty finite (N, 2) array in (x,y) order")
    return a


def illumination_features(points, anchors, probe_intensity, *, pixel_size=1.0,
                          spatial_weights=None):
    """Return Phi[i,q] = sqrt(w[q]) * I0(anchors[q] - points[i]).

    Bilinear interpolation with zero extension. The centre of the probe array
    is ((W-1)/2, (H-1)/2). Use pixel_size=(dx,dy) for anisotropic pixel spacing.
    `anchors` are object-plane quadrature locations; weights include cell area.
    Equal weights may be omitted: global scale cancels from the partition.
    """
    p = _points(points, 'points')
    x = _points(anchors, 'anchors')
    I = np.asarray(probe_intensity, dtype=np.float64)
    if I.ndim != 2 or min(I.shape) < 2 or not np.isfinite(I).all() or np.any(I < 0):
        raise ValueError('probe_intensity must be a finite nonnegative 2D array')
    if not np.any(I > 0):
        raise ValueError('probe_intensity must have positive energy')
    pix = np.broadcast_to(np.asarray(pixel_size, dtype=float), (2,))
    if not np.isfinite(pix).all() or np.any(pix <= 0):
        raise ValueError('pixel_size must be positive')
    H, W = I.shape
    # Process one scan at a time to avoid large (N,Q,2) temporaries.
    Phi = np.empty((len(p), len(x)), dtype=np.float64)
    for i, r in enumerate(p):
        uv = (x-r) / pix + np.array([(W-1)/2, (H-1)/2])
        ix = np.floor(uv[:,0]).astype(np.int64)
        iy = np.floor(uv[:,1]).astype(np.int64)
        fx, fy = uv[:,0]-ix, uv[:,1]-iy
        row = np.zeros(len(x))
        for ox, oy, w in ((0,0,(1-fx)*(1-fy)), (1,0,fx*(1-fy)),
                           (0,1,(1-fx)*fy), (1,1,fx*fy)):
            jx, jy = ix+ox, iy+oy
            valid = (jx>=0)&(jx<W)&(jy>=0)&(jy<H)
            row[valid] += w[valid]*I[jy[valid], jx[valid]]
        Phi[i] = row
    if spatial_weights is not None:
        w = np.asarray(spatial_weights, dtype=float)
        if w.shape != (len(x),) or not np.isfinite(w).all() or np.any(w < 0):
            raise ValueError('spatial_weights must be finite, nonnegative, shape (Q,)')
        Phi *= np.sqrt(w)[None,:]
    return Phi


def random_balanced_groups(n_points, n_groups, seed=0):
    """Balanced random partition, no replacements; local RNG only."""
    if not isinstance(n_groups, (int,np.integer)) or not 1 <= n_groups <= n_points:
        raise ValueError('Require 1 <= n_groups <= n_points')
    return [g.astype(np.int64) for g in
            np.array_split(np.random.default_rng(seed).permutation(n_points), n_groups)]


def _validate_groups(groups, N):
    groups = [np.asarray(g,dtype=np.int64) for g in groups]
    if any(g.ndim != 1 or len(g)==0 for g in groups):
        raise ValueError('Groups must be nonempty 1D integer arrays')
    if not np.array_equal(np.sort(np.concatenate(groups)), np.arange(N)):
        raise ValueError('Groups must partition all indices exactly once')
    return groups


def coverage_error(features, groups):
    """Relative RMS discrepancy of batch-mean vs full-mean illumination.

    E^2 = sum_g (m_g/N) ||mean(Phi_g)-mean(Phi)||^2 / ||mean(Phi)||^2.
    This is a GEOMETRY / PROBE PROXY, not reconstruction error.
    """
    P = np.asarray(features, dtype=np.float64)
    groups = _validate_groups(groups, len(P))
    mu = P.mean(axis=0)
    den = np.dot(mu,mu)
    if den <= 0:
        raise ValueError('Features must have a nonzero mean')
    err = sum((len(g)/len(P))*np.sum((P[g].mean(axis=0)-mu)**2) for g in groups)
    return float(np.sqrt(max(0,err)/den))


def partition_features(features, n_groups=4, *, seed=0, n_restarts=1,
                       max_swaps=None, initial_groups=None):
    """Capacity-constrained minimum-discrepancy partition by pairwise swaps.

    Objective J = sum_g ||sum_{i in B_g} Phi_i||^2 / |B_g|.
    With all points partitioned, this differs by constants from weighted MMD^2.
    Each accepted swap strictly decreases J, but only a pair-swap local minimum
    is guaranteed if the iteration cap is not reached. No FPS/Hilbert is used.

    Optional initial_groups permits refinement of an existing balanced ABCD
    partition. For odd grids, raw parity capacities can differ substantially;
    use the default balanced random initialisation in that case.

    Returns (groups, info); group indices are sorted in original input order.
    """
    P = np.asarray(features, dtype=np.float64)
    if P.ndim!=2 or len(P)==0 or not np.isfinite(P).all():
        raise ValueError('features must be a finite nonempty (N,Q) array')
    N = len(P)
    if not isinstance(n_groups,(int,np.integer)) or not 1 <= n_groups <= N:
        raise ValueError('Require 1 <= n_groups <= N')
    if not isinstance(n_restarts,(int,np.integer)) or n_restarts < 1:
        raise ValueError('n_restarts must be a positive integer')
    if max_swaps is None:
        max_swaps = 20*N
    if max_swaps < 0:
        raise ValueError('max_swaps must be nonnegative')
    K = P @ P.T
    scale = float(np.mean(np.diag(K)))
    if scale <= 0:
        raise ValueError('Features contain no positive energy')
    K /= scale  # scale does not change the optimal partition
    diag = np.diag(K)
    rng = np.random.default_rng(seed)
    best = None
    for restart in range(n_restarts):
        if restart == 0 and initial_groups is not None:
            groups = _validate_groups(initial_groups,N)
            if len(groups)!=n_groups:
                raise ValueError('initial_groups count differs from n_groups')
        else:
            groups = list(np.array_split(rng.permutation(N),n_groups))
        labels = np.empty(N,dtype=np.int64)
        for g,idx in enumerate(groups):
            labels[idx]=g
        sizes = np.bincount(labels,minlength=n_groups).astype(float)
        sums = np.array([K[labels==g].sum(axis=0) for g in range(n_groups)])
        history = [float(sum(sums[g,labels==g].sum()/sizes[g]
                             for g in range(n_groups)))]
        local_minimum = False
        for _ in range(max_swaps):
            best_delta, choice = -1e-12, None
            for a in range(n_groups):
                ia = np.flatnonzero(labels==a)
                for b in range(a+1,n_groups):
                    ib = np.flatnonzero(labels==b)
                    # Swap i in a and j in b; capacities remain unchanged.
                    delta = (2*(sums[a,ib][None,:]-sums[a,ia][:,None])/sizes[a]
                             -2*(sums[b,ib][None,:]-sums[b,ia][:,None])/sizes[b]
                             +(diag[ia,None]+diag[None,ib]-2*K[np.ix_(ia,ib)])
                              *(1/sizes[a]+1/sizes[b]))
                    loc = np.unravel_index(np.argmin(delta),delta.shape)
                    d = float(delta[loc])
                    if d < best_delta:
                        best_delta, choice = d,(a,b,int(ia[loc[0]]),int(ib[loc[1]]))
            if choice is None:
                local_minimum = True
                break
            a,b,i,j = choice
            change = K[j]-K[i]
            sums[a] += change
            sums[b] -= change
            labels[i],labels[j] = b,a
            history.append(history[-1]+best_delta)
        objective = float(sum(sums[g,labels==g].sum()/sizes[g] for g in range(n_groups)))
        if best is None or objective < best[0]:
            result = [np.flatnonzero(labels==g).astype(np.int64) for g in range(n_groups)]
            info = {'objective_scaled':objective, 'accepted_swaps':len(history)-1,
                    'pair_swap_local_minimum':local_minimum, 'best_restart':restart,
                    'group_sizes':sizes.astype(int).tolist(), 'objective_history':history}
            best = objective,result,info
    _,groups,info = best
    info['relative_coverage_error'] = coverage_error(P,groups)
    return groups,info


def sparse_sampler_coverage(points, n_groups, probe_intensity, *, pixel_size=1.0,
                            anchor_step=2.0, roi=None, seed=0, n_restarts=1,
                            max_swaps=None):
    """Drop-in style wrapper returning a list of index arrays.

    All coordinates/anchor_step/roi are in the same units as pixel_size.
    roi=(xmin,xmax,ymin,ymax); default covers all translated probe arrays.
    For large scans, replace the dense feature/Gram matrix with tiled or low-rank
    computations: memory O(N*Q+N^2), Gram work O(N^2*Q). Intended here for N~100.
    """
    points = _points(points,'points')
    I = np.asarray(probe_intensity)
    if I.ndim!=2:
        raise ValueError('probe_intensity must be 2D')
    pix = np.broadcast_to(np.asarray(pixel_size,float),(2,))
    if anchor_step <= 0:
        raise ValueError('anchor_step must be positive')
    if roi is None:
        ext = np.array([(I.shape[1]+1)/2,(I.shape[0]+1)/2])*pix
        lo,hi = points.min(axis=0)-ext, points.max(axis=0)+ext
    else:
        r = np.asarray(roi,float)
        if r.shape!=(4,) or not np.isfinite(r).all() or r[1]<=r[0] or r[3]<=r[2]:
            raise ValueError('roi must be (xmin,xmax,ymin,ymax) with positive extent')
        lo,hi = r[[0,2]],r[[1,3]]
    xs = np.arange(lo[0],hi[0]+anchor_step/2,anchor_step)
    ys = np.arange(lo[1],hi[1]+anchor_step/2,anchor_step)
    xx,yy = np.meshgrid(xs,ys)
    anchors = np.column_stack([xx.ravel(),yy.ravel()])
    Phi = illumination_features(points,anchors,I,pixel_size=pixel_size)
    groups,_ = partition_features(Phi,n_groups,seed=seed,n_restarts=n_restarts,
                                  max_swaps=max_swaps)
    return groups
