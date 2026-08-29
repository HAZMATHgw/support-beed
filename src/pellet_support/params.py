# -*- coding: utf-8 -*-
"""충전 기하 파라미터.

이 모듈이 저장소 전체의 유일한 '설계 변수' 정의 지점이다. 구 지름 D 와
겹침량 delta 두 개만 정하면 pitch, 층높이, 접촉원 크기, 압출량, 충전율이
전부 여기서 유도된다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Tuple

EPSILON = 1e-9

#: 모서리 1인 정사면체의 높이 = sqrt(2/3).
#:
#: 서로 맞닿은 세 구가 만드는 오목한 자리(hollow)에 네 번째 구가 앉으면,
#: 그 중심은 아래 세 구의 중심 평면에서 ``pitch * TETRA_HEIGHT`` 만큼 위에
#: 놓인다. 즉 층높이는 선택하는 값이 아니라 pitch 가 정해지는 순간 강제된다.
TETRA_HEIGHT = 0.8164965809277260


class SupportBeadRegion:
    """bead 가 놓이는 위치의 종류."""

    CONTACT = "contact"  # 모델 아랫면과 닿는 껍질
    BODY = "body"        # 서포터 몸통


@dataclass
class SupportBeadParams:
    """C++ ``SupportBeadParams`` 와 1:1 대응."""

    bead_diameter_mm: float = 1.0
    #: delta. 이웃 구에 얼마나 눌려 들어가는지를 지름 대비 비율로.
    #: 0 이면 점접촉(고정력 0), 0.12 를 넘으면 사실상 통짜가 된다.
    lattice_overlap_ratio: float = 0.06
    edge_margin_ratio: float = 0.5
    #: 압출 선분 길이 / 지름. 짧을수록 구에 가깝다.
    segment_ratio: float = 0.15
    #: True = 아래층 hollow 에 안착(배위수 12), False = 수직 기둥(배위수 8)
    stagger_layers: bool = True
    #: 2 = ABAB(hcp), 3 = ABCABC(fcc)
    stagger_period: int = 3
    snake_order: bool = True
    alternate_contact_angle: bool = True
    flow_multiplier: float = 1.0

    # -- 유도값 ------------------------------------------------------------

    def pitch_mm(self) -> float:
        """구 중심 사이 거리."""
        return self.bead_diameter_mm * (1.0 - self.lattice_overlap_ratio)

    def layer_height_mm(self) -> float:
        """이 충전이 성립하기 위해 프로파일이 반드시 써야 하는 층높이."""
        pitch = self.pitch_mm()
        return pitch * TETRA_HEIGHT if self.stagger_layers else pitch

    def contact_disc_radius_mm(self) -> float:
        """눌린 두 구가 만나 생기는 평평한 원판의 반지름."""
        r, d = 0.5 * self.bead_diameter_mm, self.pitch_mm()
        return math.sqrt(max(0.0, r * r - 0.25 * d * d))

    def coordination_number(self) -> int:
        """한 구가 닿는 이웃의 수."""
        return 12 if self.stagger_layers else 8

    def bead_volume_mm3(self) -> float:
        """구 부피에서 이웃이 눌러 없앤 구관(spherical cap)들을 뺀 값."""
        r = 0.5 * self.bead_diameter_mm
        overlap = max(0.0, self.bead_diameter_mm - self.pitch_mm())
        h_cap = 0.5 * overlap
        v_cap = math.pi * h_cap * h_cap * (3.0 * r - h_cap) / 3.0
        return max(
            0.0,
            4.0 / 3.0 * math.pi * r ** 3 - self.coordination_number() * v_cap,
        )

    def mm3_per_mm(self) -> float:
        """압출 선분에 실어야 할 단위길이당 부피.

        선분 길이가 아니라 목표 구 부피에서 역산한다. 그래야 구 크기가
        세그먼트 길이에 끌려다니지 않는다.
        """
        seg = max(EPSILON, self.bead_diameter_mm * self.segment_ratio)
        return self.flow_multiplier * self.bead_volume_mm3() / seg

    def packing_fraction(self) -> float:
        """단위 격자 셀 대비 재료가 차지하는 비율."""
        pitch, height = self.pitch_mm(), self.layer_height_mm()
        cell = math.sqrt(3.0) * 0.5 * pitch * pitch * height
        return self.bead_volume_mm3() / cell if cell > 0 else 0.0


def support_bead_contact_params(nozzle_diameter_mm: float) -> SupportBeadParams:
    """모델 아랫면과 닿는 껍질. 조금 더 눌러 붙여 전단에 견디게 한다."""
    return SupportBeadParams(
        bead_diameter_mm=nozzle_diameter_mm,
        lattice_overlap_ratio=0.08,
        edge_margin_ratio=0.4,
    )


def support_bead_body_params(
    nozzle_diameter_mm: float, bead_ratio: float = 0.97
) -> SupportBeadParams:
    """몸통. 구를 조금 작게 만들어 겹침을 낮추면 펠릿으로 잘 부서진다."""
    return SupportBeadParams(
        bead_diameter_mm=nozzle_diameter_mm * bead_ratio,
        lattice_overlap_ratio=0.04,
        edge_margin_ratio=0.5,
    )


def unify_lattice(
    contact: SupportBeadParams, body: SupportBeadParams
) -> Tuple[SupportBeadParams, SupportBeadParams]:
    """격자는 오브젝트당 하나뿐이라는 제약을 강제한다.

    contact 의 pitch 를 기준으로 삼고, body 는 같은 pitch 위에서 구 지름만
    줄여 겹침을 낮춘다. body 의 delta 를 직접 지정하게 두면 pitch 가 갈라져
    위아래 층이 서로의 hollow 에 안 떨어진다.
    """
    pitch = contact.pitch_mm()
    body_d = body.bead_diameter_mm or contact.bead_diameter_mm
    body = replace(
        body,
        lattice_overlap_ratio=1.0 - pitch / body_d,
        stagger_layers=contact.stagger_layers,
        stagger_period=contact.stagger_period,
    )
    return contact, body


@dataclass
class SupportGenParams:
    """서포터 '영역'을 찾는 단계의 파라미터."""

    nozzle_diameter_mm: float = 1.0
    layer_height_mm: float = 0.77  # bead 기하에서 자동 계산됨
    overhang_angle_deg: float = 45.0
    contact_z_gap_layers: int = 1
    xy_clearance_mm: float = 0.8
    contact_layers: int = 2
    solid_first_layers: int = 1
    min_island_area_mm2: float = 2.0
    support_on_build_plate_only: bool = False
    max_layers: int = 4000
