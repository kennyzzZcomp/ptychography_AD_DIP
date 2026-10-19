"""Isolated real-data adapter; no imports from the simulation implementation."""
from contextlib import contextmanager, nullcontext
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@contextmanager
def preserve_buffers(module):
    saved = [(b, b.detach().clone()) for b in module.buffers()]
    try:
        yield
    finally:
        with torch.no_grad():
            for b, old in saved:
                b.copy_(old)


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.LeakyReLU(.2, True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.LeakyReLU(.2, True))

    def forward(self, x):
        return self.f(x)


def pixel_rms_normalize(features):
    """Normalize channels per pixel; RMS floor 1e-6, no learned parameters.

    Clamp before sqrt so an all-zero feature vector has finite gradients.
    """
    rms = features.square().mean(1, keepdim=True).clamp_min(1e-12).sqrt()
    return features/rms


class UNet(nn.Module):
    """Original concatenating U-Net topology, with optional activation checkpointing."""
    def __init__(self, cin, width, residual=False, checkpoint_blocks=True, feature_norm='none'):
        super().__init__()
        if feature_norm not in ('none', 'pixel-rms'):
            raise ValueError('Unknown feature normalization mode')
        self.feature_norm = feature_norm
        b = width
        self.e1, self.e2, self.e3 = DoubleConv(cin, b), DoubleConv(b, 2*b), DoubleConv(2*b, 4*b)
        self.bot = DoubleConv(4*b, 8*b)
        self.pool = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(8*b, 4*b, 3, 2, 1, output_padding=1)
        self.d3 = DoubleConv(8*b, 4*b)
        self.u2 = nn.ConvTranspose2d(4*b, 2*b, 3, 2, 1, output_padding=1)
        self.d2 = DoubleConv(4*b, 2*b)
        self.u1 = nn.ConvTranspose2d(2*b, b, 3, 2, 1, output_padding=1)
        self.d1 = DoubleConv(2*b, b)
        self.head_amp = nn.Conv2d(b, 1, 3, padding=1)
        self.head_phs = nn.Conv2d(b, 1 if residual else 2, 3, padding=1)
        self.residual, self.checkpoint_blocks = residual, checkpoint_blocks
        self.register_buffer('initial_amp_offset', None)
        with torch.no_grad():
            self.head_amp.weight.zero_(); self.head_phs.weight.zero_()
            self.head_amp.bias.fill_(0 if residual else math.log(math.expm1(1)))
            self.head_phs.bias.zero_()
            if not residual:
                self.head_phs.bias[0] = 1

    def set_initial_object(self, field):
        """Fixed logit offset realizes a positive, zero-phase O0 with zero heads.

        No extra trainable parameters or changes to the residual stage.
        """
        if self.residual or field.ndim != 2 or not torch.all(field.real > 0) or torch.any(field.imag != 0):
            raise ValueError('Initial object must be positive real and stage 1 only')
        self.initial_amp_offset = (torch.log(torch.expm1(field.real))
                                   - self.head_amp.bias.detach()[0]).detach()

    def block(self, module, *args):
        if self.checkpoint_blocks and torch.is_grad_enabled():
            # Recomputing BN must not update running statistics a second time.
            return checkpoint(module, *args, use_reentrant=False,
                              context_fn=lambda: (nullcontext(), preserve_buffers(module)))
        return module(*args)

    def forward(self, x):
        a = self.block(self.e1, x)
        b = self.block(self.e2, self.pool(a))
        c = self.block(self.e3, self.pool(b))
        z = self.block(self.bot, self.pool(c))
        z = self.block(self.d3, torch.cat((self.u3(z), c), 1))
        z = self.block(self.d2, torch.cat((self.u2(z), b), 1))
        z = self.block(self.d1, torch.cat((self.u1(z), a), 1))
        if self.feature_norm == 'pixel-rms':
            z = pixel_rms_normalize(z)
        r, p = self.head_amp(z)[0, 0], self.head_phs(z)[0]
        if self.residual:
            return torch.complex(r, p[0])
        amp = F.softplus(r if self.initial_amp_offset is None else r+self.initial_amp_offset)
        den = torch.sqrt(p[0].square()+p[1].square()+1e-8)
        return torch.complex(amp*p[0]/den, amp*p[1]/den)


def select_center_grid(positions_mm, num_frames=225):
    """Infer rows from slow stored coordinate; retain measured jitter and frame IDs."""
    p = np.asarray(positions_mm, dtype=np.float64).T
    if p.shape != (225, 2) or not np.isfinite(p).all():
        raise ValueError('Expected finite positionx with shape (2,225)')
    rows = np.argsort(p[:, 0], kind='stable').reshape(15, 15)
    grid = np.stack([r[np.argsort(p[r, 1], kind='stable')] for r in rows])
    rp = p[grid]
    row_centers = np.median(rp[:, :, 0], axis=1)
    col_centers = np.median(rp[:, :, 1], axis=0)
    sy, sx = np.median(np.diff(row_centers)), np.median(np.diff(col_centers))
    if min(sy, sx) <= 0 or np.max(np.ptp(rp[:, :, 0], axis=1)) > .4*sy:
        raise ValueError('Coordinates do not form the expected separated 15x15 rows')
    if np.max(np.abs(rp[:, :, 1]-col_centers[None, :])) > .4*sx:
        raise ValueError('Column alignment is not a regular 15x15 raster')
    if num_frames not in (100, 200, 225):
        raise ValueError('Supported selections: 100 (central 10x10), 200 (nearest center), 225 (full 15x15)')
    center = (p.min(0)+p.max(0))/2
    if num_frames == 100:
        # Preserve the original selection and group/channel ordering.
        _, r0, c0 = min((float(np.linalg.norm(p[grid[r:r+10, c:c+10]].mean((0, 1))-center)), r, c)
                        for r in (2, 3) for c in (2, 3))
        rr, cc = np.meshgrid(np.arange(r0, r0+10), np.arange(c0, c0+10), indexing='ij')
        rr, cc = rr.ravel(), cc.ravel()
        group_row, group_col = rr-r0, cc-c0
    elif num_frames == 225:
        rr, cc = np.indices((15, 15))
        rr, cc = rr.ravel(), cc.ravel()
        r0, c0 = 0, 0
        group_row, group_col = rr, cc
    else:
        # Exactly 200 real frames, nearest physical center; deterministic raster ordering.
        distance2 = np.square(rp-center).sum(axis=-1).ravel()
        chosen = np.sort(np.argsort(distance2, kind='stable')[:num_frames])
        rr, cc = np.divmod(chosen, 15)
        r0, c0 = int(rr.min()), int(cc.min())
        group_row, group_col = rr, cc
    ids = grid[rr, cc]
    groups = [np.flatnonzero((group_row%2 == a) & (group_col%2 == b))
              for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))]
    return ids, p[ids], groups, {'row_start_zero_based': r0, 'col_start_zero_based': c0,
                               'grid_rows_zero_based': rr.tolist(), 'grid_cols_zero_based': cc.tolist(),
                               'num_frames': num_frames, 'group_sizes': [len(g) for g in groups],
                               'selection_rule': {100: 'central_10x10', 200: 'nearest_200_to_physical_center',
                                                  225: 'full_15x15'}[num_frames],
                               'row_step_mm': float(sy), 'col_step_mm': float(sx)}


def geometry(positions_mm, n, wavelength_nm, distance_mm, pixel_um, order='yx', flip_y=False, flip_x=False):
    dx_m = wavelength_nm*1e-9 * distance_mm*1e-3 / (n*pixel_um*1e-6)
    pos = positions_mm[:, [0, 1] if order == 'yx' else [1, 0]].copy()*1e-3/dx_m
    pos[:, 0] *= -1 if flip_y else 1
    pos[:, 1] *= -1 if flip_x else 1
    pos -= pos.min(0)
    pos += 2  # Margin for bilinear interpolation, not position rounding.
    side = int(math.ceil((pos.max()+n+2)/8)*8)
    return pos, side, dx_m


def mpie_geometry(positions_mm, n, wavelength_nm, distance_mm, pixel_um,
                  order='yx', flip_y=False, flip_x=False):
    """PtyLab CPM integer slices, expressed in the UNCHANGED network canvas.

    Keep legacy fractional positions solely to replay exactly the same initial
    arrays/RNG stream. Integer reference coordinates are computed independently:
    center meters -> divide by dx -> np.round -> add reference canvas offset.
    Subtract the existing initialization crop origin only AFTER rounding.
    """
    legacy, side, dx = geometry(positions_mm, n, wavelength_nm, distance_mm,
                                pixel_um, order, flip_y, flip_x)
    encoder = np.asarray(positions_mm, np.float64)[:, [0, 1] if order == 'yx' else [1, 0]].copy()*1e-3
    encoder[:, 0] *= -1 if flip_y else 1
    encoder[:, 1] *= -1 if flip_x else 1
    origin = encoder.mean(axis=0)
    centered_px = (encoder-origin)/dx
    rounded = np.round(centered_px).astype(np.int64)
    reference_side = int(np.ptp(rounded, axis=0).max()) + 2*n
    reference_side += reference_side % 2
    reference_positions = rounded + reference_side//2 - n//2
    # This is exactly the old initial_fields() crop; do not recenter O0 anew.
    crop = np.round(reference_side/2-n/2-legacy.mean(axis=0)).astype(np.int64)
    positions = reference_positions-crop
    if np.any(crop < 0) or np.any(crop+side > reference_side):
        raise ValueError('Unchanged initialization canvas does not fit mPIE reference')
    if np.any(positions < 0) or np.any(positions+n > side):
        raise ValueError('mPIE slices do not fit unchanged network canvas')
    metadata = dict(
        reference_object_side=reference_side,
        encoder_origin_m=origin.tolist(),
        reference_positions_px=reference_positions.tolist(),
        network_origin_in_reference_yx=crop.tolist(),
        rounding='np.round((encoder_m-mean(encoder_m))/dx); then add No//2-Np//2',
        canvas_mapping='reference_position = network_position + network_origin_in_reference_yx',
        max_rounding_displacement_px=float(np.abs(rounded-centered_px).max()))
    return positions, side, dx, legacy, metadata


def center_crop_field(field, size):
    """Same-pitch crop, also the adjoint of center_pad_probe for gradients."""
    n = field.shape[-1]
    if field.shape[-2] != n or size < 1 or size > n or (n-size)%2:
        raise ValueError('Centered crop requires square arrays and an even size difference')
    low = (n-size)//2
    return field[..., low:low+size, low:low+size]


def center_pad_probe(probe, size):
    """Embed a compact complex probe without resampling or changing pixel pitch."""
    n = probe.shape[-1]
    if probe.shape[-2] != n or size < n or (size-n)%2:
        raise ValueError('Centered padding requires square arrays and an even size difference')
    pad = (size-n)//2
    return F.pad(probe, (pad,pad,pad,pad))


class Propagator:
    def __init__(self, positions_px, n, sampling='bilinear'):
        self.n = n
        if sampling not in ('bilinear', 'mpie-integer'):
            raise ValueError('Unknown object sampling mode')
        self.sampling = sampling
        positions_px = np.asarray(positions_px)
        if sampling == 'mpie-integer':
            if not np.isfinite(positions_px).all() or np.any(positions_px != np.round(positions_px)):
                raise ValueError('mPIE slicing requires pre-aligned integer positions')
            self.samples = [[(int(y), int(x), 1.)] for y, x in positions_px]
            return
        self.samples = []
        for y, x in positions_px:
            iy, ix = int(math.floor(y)), int(math.floor(x))
            fy, fx = y-iy, x-ix
            self.samples.append([(iy, ix, (1-fy)*(1-fx)), (iy+1, ix, fy*(1-fx)),
                                 (iy, ix+1, (1-fy)*fx), (iy+1, ix+1, fy*fx)])

    def amplitude(self, obj, probe, ids):
        n = self.n
        if self.sampling == 'mpie-integer':
            patches = [obj[y:y+n, x:x+n] for i in ids for y, x, _ in self.samples[int(i)]]
        else:
            patches = [sum(w*obj[y:y+n, x:x+n] for y, x, w in self.samples[int(i)]) for i in ids]
        exit_wave = torch.stack(patches)*probe
        field = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(exit_wave, dim=(-2,-1)),
                                               norm='ortho'), dim=(-2,-1))
        return field.abs()

    def mask(self, ids, side, device):
        out = torch.zeros((side, side), dtype=torch.bool, device=device)
        for i in ids:
            for y, x, w in self.samples[int(i)]:
                if w > 0:
                    out[y:y+self.n, x:x+self.n] = True
        return out

    def coverage(self, probe, side):
        out = probe.real.new_zeros((side, side))
        power = probe.abs().square()
        for samples in self.samples:
            for y, x, w in samples:
                out[y:y+self.n, x:x+self.n] += w*power
        return out/out.max().clamp_min(1e-12)


def chunked_data(obj, probe, ids, measurements, propagator, chunk=1, gradient=True):
    """Global profiled amplitude MSE. Two passes, bounded FFT graph memory.

    The optimum amplitude gain is shared across ALL selected frames, not fitted
    separately per chunk or frame. Holding this optimum fixed in the second
    pass gives the profiled objective's object/probe gradients.
    """
    ids = list(ids)
    if not ids or chunk < 1:
        raise ValueError('Need nonempty frame selection and positive chunk size')
    o = obj.detach().requires_grad_(gradient)
    p = probe.detach().requires_grad_(gradient)
    def target(part):
        return torch.from_numpy(np.array(measurements[part], copy=True)).to(o.device).sqrt()
    num = o.real.new_zeros((), dtype=torch.float64)
    den = num.clone(); energy = num.clone()
    with torch.no_grad():
        for start in range(0, len(ids), chunk):
            part = ids[start:start+chunk]
            a, b = propagator.amplitude(o, p, part), target(part)
            num += (a*b).sum(dtype=torch.float64)
            den += a.square().sum(dtype=torch.float64)
            energy += b.square().sum(dtype=torch.float64)
        gain = (num/den.clamp_min(1e-30)).to(o.real.dtype)
    total = o.real.new_zeros(())
    go = torch.zeros_like(o) if gradient else None
    gp = torch.zeros_like(p) if gradient else None
    with torch.set_grad_enabled(gradient):
        for start in range(0, len(ids), chunk):
            part = ids[start:start+chunk]
            a, b = propagator.amplitude(o, p, part), target(part)
            loss = (gain*a-b).square().sum()/(len(ids)*propagator.n**2)
            if gradient:
                do, dp = torch.autograd.grad(loss, (o, p))
                go += do; gp += dp
            total += loss.detach()
    mean_energy = float(energy)/(len(ids)*propagator.n**2)
    stats = {'amplitude_mse': float(total), 'relative_amplitude_rmse': math.sqrt(float(total)/max(mean_energy, 1e-30)),
             'profile_gain': float(gain), 'gain_mode': 'profiled'}
    if not math.isfinite(stats['amplitude_mse']) or (gradient and not (torch.isfinite(go).all() and torch.isfinite(gp).all())):
        raise FloatingPointError('Non-finite physical loss/gradient')
    return go, gp, stats


def tgv_terms(u, v, mask):
    valid = mask[:-1,:-1] & mask[1:,:-1] & mask[:-1,1:]
    dy, dx = u[1:,:-1]-u[:-1,:-1], u[:-1,1:]-u[:-1,:-1]
    vy, vx = v[0], v[1]
    ry, rx = dy-vy[:-1,:-1], dx-vx[:-1,:-1]
    eyy, exx = vy[1:,:-1]-vy[:-1,:-1], vx[:-1,1:]-vx[:-1,:-1]
    eyx = .5*(vy[:-1,1:]-vy[:-1,:-1]+vx[1:,:-1]-vx[:-1,:-1])
    first = (torch.sqrt(ry.square()+rx.square()+1e-6)-.001)[valid].mean()
    second = (torch.sqrt(eyy.square()+exx.square()+2*eyx.square()+1e-6)-.001)[valid].mean()
    return first+2*second


class TGV:
    def __init__(self, mask, inner=5):
        self.mask, self.inner = mask, inner
        self.v = nn.Parameter(torch.zeros((2, *mask.shape), device=mask.device))
        self.opt = torch.optim.Adam([self.v], lr=.01)

    def __call__(self, obj):
        amp = obj.abs()
        u = amp/amp[self.mask].mean().clamp_min(1e-12)
        for _ in range(self.inner):
            self.opt.zero_grad(set_to_none=True)
            tgv_terms(u.detach(), self.v, self.mask).backward()
            self.opt.step()
        return tgv_terms(u, self.v.detach(), self.mask)


class NLMPrior:
    def __init__(self, base, mask, weight):
        self.mask, self.weight = mask, weight.detach()
        z = self.normalized(base).detach().cpu().numpy()
        hh = (z[:-1,:-1]-z[1:,:-1]-z[:-1,1:]+z[1:,1:])/2
        m = mask.cpu().numpy()
        valid = m[:-1,:-1] & m[1:,:-1] & m[:-1,1:] & m[1:,1:]
        values = hh[valid]
        mad = lambda x: np.median(np.abs(x-np.median(x)))/.67448975
        self.sigma = max(float(np.sqrt((mad(values.real)**2+mad(values.imag)**2)/2)), 1e-4)
        self.refresh(base)

    def normalized(self, obj):
        return obj/obj.abs()[self.mask].mean().clamp_min(1e-12)

    def refresh(self, obj):
        from skimage.restoration import denoise_nl_means
        z = self.normalized(obj).detach().cpu().numpy()
        pair = np.stack((z.real, z.imag), axis=-1)
        d = denoise_nl_means(pair, h=.8*self.sigma, sigma=self.sigma, patch_size=3,
                            patch_distance=7, fast_mode=True, channel_axis=-1, preserve_range=True)
        self.target = torch.from_numpy(np.asarray(d[...,0]+1j*d[...,1], dtype=np.complex64)).to(obj.device)

    def __call__(self, obj):
        return ((self.normalized(obj)-self.target).abs().square()*self.weight)[self.mask].mean()
