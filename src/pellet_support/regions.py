# -*- coding: utf-8 -*-
"""오버행을 찾아 서포터가 차지할 영역을 층별로 계산한다.

슬라이서 내부에서는 이미 계산되어 넘어오는 부분이지만, 이 도구는 슬라이서
없이 돌아가야 하므로 직접 구한다.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from shapely.geometry import Polygon
from shapely.ops import unary_union

from .params import SupportGenParams
from .slicing import clean, drop_small


def build_support_regions(
    slices: Sequence, params: SupportGenParams
) -> Tuple[List, List]:
    """(support, contact) 를 층별 폴리곤 리스트로 반환.

    - support[i] : i 층에서 서포터가 차지하는 XY 영역
    - contact[i] : 그중 모델 아랫면 바로 밑(위에서 contact_layers 이내)인 영역
    """
    n = len(slices)
    ang = math.radians(max(1.0, min(89.0, params.overhang_angle_deg)))
    # 한 층 올라갈 때 자립 가능한 수평 이동량
    step = params.layer_height_mm / math.tan(ang)

    # (1) 아래가 비어 있는 영역 = 오버행
    overhang: List = [Polygon()] * n
    for i in range(1, n):
        cur, below = clean(slices[i]), clean(slices[i - 1])
        if cur.is_empty:
            continue
        grown = below.buffer(step) if not below.is_empty else Polygon()
        oh = cur.difference(grown) if not grown.is_empty else cur
        overhang[i] = drop_small(oh, params.min_island_area_mm2)

    # (2) 위에서 아래로 누적하며 기둥을 만든다
    support: List = [Polygon()] * n
    acc = Polygon()
    gap = max(0, params.contact_z_gap_layers)
    for i in range(n - 1, -1, -1):
        src = i + 1 + gap  # gap 층만큼 띄운 뒤부터 지지 시작
        if src < n:
            acc = clean(unary_union([acc, overhang[src]]))
        if acc.is_empty:
            continue
        model = clean(slices[i])
        blocked = (
            model.buffer(params.xy_clearance_mm) if not model.is_empty else Polygon()
        )
        sup = acc.difference(blocked) if not blocked.is_empty else acc
        support[i] = drop_small(sup, params.min_island_area_mm2 * 0.25)
        # 모델에 막힌 부분은 더 내려가지 않는다(= 모델 윗면에 얹힘)
        acc = support[i]

    if params.support_on_build_plate_only:
        base = clean(support[0])
        for i in range(n):
            cur = clean(support[i])
            if cur.is_empty or base.is_empty:
                support[i] = Polygon()
                continue
            parts = cur.geoms if hasattr(cur, "geoms") else [cur]
            keep = [p for p in parts if p.intersects(base)]
            support[i] = unary_union(keep) if keep else Polygon()

    # (3) contact = 위쪽 contact_layers 이내에서 사라지는 영역
    cl = max(1, params.contact_layers)
    contact: List = [Polygon()] * n
    for i in range(n):
        cur = clean(support[i])
        above = clean(support[i + cl]) if i + cl < n else Polygon()
        contact[i] = (
            cur.difference(above) if not (cur.is_empty or above.is_empty) else cur
        )
    return support, contact
