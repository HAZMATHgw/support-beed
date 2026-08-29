# -*- coding: utf-8 -*-
"""bead 중심 좌표 배치. C++ ``support_bead_generate_centers`` 의 이식본.

원본과 달라진 곳은 두 군데뿐이다.

1. 격자 원점을 그 층 폴리곤의 bbox 가 아니라 오브젝트 고정 원점으로 받는다.
   기준 격자가 층마다 흔들리면 시프트를 정확히 줘도 아래층 hollow 에
   떨어지지 않는다.
2. 층 시프트를 ``% 2`` 대신 ``% stagger_period`` 로 돌려 ABC 순환을 만든다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from shapely.geometry import Point

from .params import SupportBeadParams
from .slicing import clean


@dataclass
class SupportBeadCenter:
    x: float
    y: float
    row: int = 0


def support_bead_generate_centers(
    region_geom,
    params: SupportBeadParams,
    support_layer_id: int,
    grid_origin: Tuple[float, float],
) -> List[SupportBeadCenter]:
    """영역 안에 들어가는 격자점만 골라 bead 중심으로 반환."""
    centers: List[SupportBeadCenter] = []
    region_geom = clean(region_geom)
    if region_geom.is_empty:
        return centers

    pitch_mm = params.pitch_mm()
    if pitch_mm <= 0.0 or params.bead_diameter_mm <= 0.0:
        return centers

    edge_margin = params.bead_diameter_mm * params.edge_margin_ratio
    safe = region_geom.buffer(-edge_margin)
    # 아주 얇은 인터페이스 섬은 shrink 로 사라진다. 그럴 땐 원본으로 되돌리고
    # 중심점 포함 판정에만 의존한다.
    if safe.is_empty:
        safe = region_geom
    safe = clean(safe)

    # 정삼각형 격자: 행 안에서는 pitch, 행과 행 사이는 pitch * sqrt(3)/2
    pitch_x = pitch_mm
    pitch_y = pitch_mm * math.sqrt(3.0) * 0.5

    # 최밀충전: 층 k 는 층 k-1 의 hollow 로 내려앉는다. 위를 향한 격자 삼각형의
    # hollow 는 그 무게중심 (pitch_x/2, pitch_y/3). 이 시프트를 반복하면
    # A -> B -> C 를 돌고 stagger_period 층 만에 A 로 돌아온다.
    layer_shift_x = layer_shift_y = 0.0
    if params.stagger_layers:
        period = max(2, params.stagger_period)
        k = support_layer_id % period
        layer_shift_x = k * (pitch_x / 2.0)
        layer_shift_y = k * (pitch_y / 3.0)

    # 행/열 인덱스는 그 층 안의 카운터가 아니라 오브젝트 전역 격자 인덱스다.
    # 그래야 행 패리티(= 반 칸 시프트)가 층끼리, 영역끼리 어긋나지 않는다.
    ox, oy = grid_origin
    base_x, base_y = ox + layer_shift_x, oy + layer_shift_y
    min_x, min_y, max_x, max_y = safe.bounds

    row_first = int(math.floor((min_y - base_y) / pitch_y))
    row_last = int(math.ceil((max_y - base_y) / pitch_y))

    for row in range(row_first, row_last + 1):
        y = base_y + row * pitch_y
        row_base_x = base_x + (0.0 if row % 2 == 0 else pitch_x / 2.0)
        col_first = int(math.floor((min_x - row_base_x) / pitch_x))
        col_last = int(math.ceil((max_x - row_base_x) / pitch_x))

        row_centers: List[SupportBeadCenter] = []
        for col in range(col_first, col_last + 1):
            x = row_base_x + col * pitch_x
            if safe.contains(Point(x, y)):
                row_centers.append(SupportBeadCenter(x, y, row))

        # Snake order: 짝수 행 좌->우, 홀수 행 우->좌. 이동 거리만 줄인다.
        if params.snake_order and (row % 2 != 0):
            row_centers.reverse()
        centers.extend(row_centers)
    return centers


def bead_angle(params: SupportBeadParams, support_layer_id: int) -> float:
    """짧은 압출 선분의 잔여 이방성을 층마다 돌려서 상쇄한다."""
    if not params.alternate_contact_angle:
        return 0.0
    period = max(2, params.stagger_period) if params.stagger_layers else 2
    return math.pi * (support_layer_id % period) / period
