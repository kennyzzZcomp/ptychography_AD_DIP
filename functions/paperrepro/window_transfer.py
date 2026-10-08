"""Counterfactual one-step transfer, isolated from the production trajectory.

Rows = source/update group; columns = destination/evaluation group.
Positive loss/error gain = before - after; positive PSNR gain = after - before.
Adam history is retained. A separate zero-current-gradient Adam branch exposes
momentum/weight-decay drift; its contrast is NOT an additive causal decomposition.
"""
import copy

import numpy as np
import torch


def cross_group_transfer(net, optimizer, regularizers, loss_fn, evaluate_fn):
    """Callbacks operate ONLY on copies; callers must freeze external probe/state.

loss_fn(model, source, copied_regularizer) returns the training scalar.
evaluate_fn(model) returns {'groups': [{data_loss, relerr, psnr_amp, ...}], ...}.
Original parameter gradients, buffers, optimizer/TGV states and Torch RNG survive.
"""
    device = next(net.parameters()).device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        baseline_model = copy.deepcopy(net)
        before = evaluate_fn(baseline_model)
        del baseline_model
        after = []
        control = None
        # Each branch starts at the same original checkpoint, not the prior branch.
        for source in [-1] + list(range(len(regularizers))):
            model, opt = copy.deepcopy((net, optimizer))
            opt.zero_grad(set_to_none=True)
            if source == -1:
                for p in model.parameters():
                    if p.requires_grad:
                        p.grad = torch.zeros_like(p)
            else:
                reg = copy.deepcopy(regularizers[source])
                loss = loss_fn(model, source, reg)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite counterfactual loss')
                loss.backward()
                del loss, reg
            opt.step()
            result = evaluate_fn(model)
            if source == -1:
                control = result
            else:
                after.append(result)
            del model, opt

    gains = {}
    for key, sign in [('data_loss', 1), ('relerr', 1), ('psnr_amp', -1)]:
        b = np.asarray([v[key] for v in before['groups']])
        a = np.asarray([[v[key] for v in row['groups']] for row in after])
        c = np.asarray([v[key] for v in control['groups']])
        gains[key] = (sign * (b[None, :] - a)).tolist()
        gains[key + '_vs_zero_gradient_adam'] = (sign * (c[None, :] - a)).tolist()
    return dict(before=before, after_by_source=after, zero_gradient_adam=control,
                gains=gains, row_axis='updated group A,B,C,D',
                column_axis='evaluated group A,B,C,D',
                positive_means='improvement; data_loss/relerr: before-after; psnr_amp: after-before',
                protocol='copied network + retained Adam; copied active TGV; fixed current probe; no truth in update',
                control_note='zero current gradient retains Adam momentum and weight decay; contrast is not a causal decomposition')
