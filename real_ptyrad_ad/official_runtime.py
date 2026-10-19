"""Load unchanged official numeric functions despite Python-3.12 plot syntax.

PtyRAD 1.0.0 declares Python>=3.10, but importing solver.__init__ imports a
plotting file containing PEP701 syntax. On 3.10/3.11 ONLY, compile selected
unaltered FunctionDef nodes from the installed reconstruction.py. No source
file is patched, no arithmetic is rewritten, no alternative solver is supplied.
Only the random grouping/fresh Adam/single-GPU path is supported here.
"""
import ast
import hashlib
import logging
from pathlib import Path
import sys
import types

import numpy as np
import torch
import ptyrad


def load_official_updates():
    if sys.version_info >= (3,12):
        from ptyrad.solver import reconstruction as module
        return module,dict(mode='normal_official_import')
    from ptyrad.runtime.seed import set_random_seed
    from ptyrad.utils.time import parse_sec_to_time_str
    from ptyrad.io.adapter import ndarrays_to_tensors
    source = Path(ptyrad.__file__).parent/'solver/reconstruction.py'
    text = source.read_text(encoding='utf-8')
    tree = ast.parse(text,filename=str(source))
    names = {'create_optimizer','make_batches','recon_step','time_sync',
             'toggle_grad_requires','compute_loss','loss_logger'}
    nodes = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise RuntimeError('Official numeric API changed; refusing compatibility load')
    module = types.ModuleType('ptyrad_official_numeric_reconstruction')
    module.__file__ = str(source)
    module.__dict__.update(np=np,torch=torch,logger=logging.getLogger('ptyrad.solver.reconstruction'),
                           set_random_seed=set_random_seed,parse_sec_to_time_str=parse_sec_to_time_str,
                           ndarrays_to_tensors=ndarrays_to_tensors)
    sys.modules[module.__name__] = module
    # Select complete function definitions, preserving their original AST and
    # source locations. This does not import the package's incompatible plots.
    selected = ast.Module(body=nodes,type_ignores=[])
    exec(compile(selected,str(source),'exec'),module.__dict__)
    return module,dict(mode='unchanged_official_function_AST_load_without_plot_imports',
                       reason='PtyRAD 1.0.0 plotting/basic.py uses Python 3.12 PEP701 syntax',
                       source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                       functions=sorted(names),package_files_modified=False,
                       supported_path='random grouping; fresh Adam; single GPU; no scheduler')
