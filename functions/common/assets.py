# -*- coding: utf-8 -*-
"""素材目录定位。"""

from __future__ import annotations

from pathlib import Path


def asset_dir(cfg: Cfg) -> Path:
    if cfg.assets:
        return Path(cfg.assets)
    here = Path(__file__).resolve().parent
    for c in (here, here.parent, here.parent.parent):
        if (c / "cameraman.bmp").is_file():
            return c
    return here
