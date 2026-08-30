# -*- coding: utf-8 -*-
"""슬라이싱부터 메쉬까지 한 번에 묶는 진입점."""

from __future__ import annotations

import math

from dataclasses import replace
from typing import NamedTuple, Optional, Tuple

import trimesh

from .meshing import BeadPlan, plan_beads, plan_to_mesh
from .params import (
    SupportBeadParams,
    SupportGenParams,
    support_bead_body_params,
    support_bead_contact_params,
    unify_lattice,
)
from .regions import build_support_regions
from .slicing import clean, slice_model
from .validation import (
    InvalidParameterError,
    check_model_scale,
    estimate_bead_count,
    guard_bead_count,
    validate_bead_params,
    validate_gen_params,
    validate_mesh,
)


class SupportResult(NamedTuple):
    mesh: trimesh.Trimesh
    plan: Optional[BeadPlan]
    slices: list


def make_params(
    nozzle_diameter_mm: float = 1.0,
    bead_diameter_mm: Optional[float] = None,
    overlap: Optional[float] = None,
    body_bead_ratio: float = 0.97,
    stagger_period: int = 3,
    straight_columns: bool = False,
    segment_ratio: Optional[float] = None,
    edge_margin_ratio: Optional[float] = None,
) -> Tuple[SupportBeadParams, SupportBeadParams]:
    """CLI 와 라이브러리가 공유하는 파라미터 조립 로직.

    ``bead_diameter_mm`` 을 비워두면 노즐 지름의 절반을 기본값으로 쓴다
    (표준 FDM 이 층높이를 노즐 지름의 25~75% 로 쓰는 것과 같은 논리).
    노즐이 굵어도 구를 그만큼 굵게 만들 필요는 없다 — 오히려 굵게 만들면
    gap/interface/edge_margin 이 전부 비드 지름에 비례해 커져서, 굵은
    펠릿 노즐일수록 서포터 위쪽에 큰 사각지대가 생긴다.
    """
    if nozzle_diameter_mm <= 0:
        raise InvalidParameterError(
            f"노즐 지름이 {nozzle_diameter_mm}mm 입니다. 0보다 커야 합니다."
        )
    if bead_diameter_mm is not None and bead_diameter_mm <= 0:
        raise InvalidParameterError(
            f"구슬 지름이 {bead_diameter_mm}mm 입니다. 0보다 커야 합니다. "
            f"자동값(노즐의 절반)을 쓰려면 비워 두세요."
        )
    if not (0 < body_bead_ratio <= 1):
        raise InvalidParameterError(
            f"몸통 구슬 비율이 {body_bead_ratio} 입니다. 0보다 크고 1 이하여야 합니다. "
            f"1을 넘으면 몸통 구슬이 인터페이스보다 커져서 공유 격자가 깨집니다."
        )

    contact = support_bead_contact_params(nozzle_diameter_mm, bead_diameter_mm)
    body = support_bead_body_params(nozzle_diameter_mm, body_bead_ratio, bead_diameter_mm)
    if overlap is not None:
        contact = replace(contact, lattice_overlap_ratio=overlap)

    over = {"stagger_period": max(2, stagger_period)}
    if segment_ratio is not None:
        over["segment_ratio"] = segment_ratio
    if edge_margin_ratio is not None:
        over["edge_margin_ratio"] = edge_margin_ratio
    if straight_columns:
        over["stagger_layers"] = False

    contact = replace(contact, **over)
    body = replace(body, **over)
    contact, body = unify_lattice(contact, body)

    # 여기서 바로 검증한다. 나중에 검증하면 pitch 가 음수인 상태로 화면에
    # 출력되고 나서야 에러가 나서, 사용자가 이상한 숫자를 먼저 보게 된다.
    validate_bead_params(contact, "인터페이스 구슬")
    validate_bead_params(body, "몸통 구슬")
    return contact, body


def generate_support(
    mesh: trimesh.Trimesh,
    gen: SupportGenParams,
    contact_params: SupportBeadParams,
    body_params: SupportBeadParams,
    detail: int = 1,
    verbose: bool = True,
) -> SupportResult:
    # --- 0) 검증: 조용히 틀린 결과를 내느니 명확한 에러로 막는다 -------------
    validate_mesh(mesh)
    validate_bead_params(contact_params, "인터페이스 구슬")
    validate_bead_params(body_params, "몸통 구슬")
    validate_gen_params(gen)
    check_model_scale(mesh, contact_params)
    if not 0 <= detail <= 3:
        raise InvalidParameterError(
            f"구 표면 세분화 단계가 {detail} 입니다. 0~3 사이여야 합니다."
        )

    # --- 1) 탐지: 모델 형상을 제대로 볼 수 있는 얇은 층으로 자른다 -----------
    # 구슬 격자 간격(=layer_height_mm)으로 자르면, 굵은 펠릿을 쓸 때 층이
    # 듬성듬성해져서 그 사이의 오버행을 통째로 놓친다. 모델이 서포터를
    # 필요로 하는지는 모델 형상의 문제이지 펠릿 크기와 무관해야 한다.
    det_h = gen.detection_layer_height_mm or min(0.4, gen.layer_height_mm)
    det_h = max(det_h, float(mesh.extents[2]) / gen.max_detection_layers)
    det_slices, _ = slice_model(mesh, det_h, gen.max_detection_layers)
    if verbose:
        print(f"      탐지 슬라이싱: {len(det_slices)}층 @ {det_h:.3f}mm")

    support, contact = build_support_regions(det_slices, gen, det_h)
    if sum(1 for s in support if not clean(s).is_empty) == 0:
        return SupportResult(trimesh.Trimesh(), None, det_slices)

    # 구슬 수를 미리 어림해서, 메모리가 터져 멈추기 전에 막는다.
    # (탐지 층 기준이라 실제 구슬 층보다 촘촘해 안전한 쪽으로 과대평가된다)
    estimated = estimate_bead_count(support, gen, contact_params)
    guard_bead_count(estimated, gen.max_beads)

    # --- 2) 배치: 구슬 격자 간격으로 위 결과를 다시 샘플링한다 --------------
    z0 = float(mesh.bounds[0][2])
    bead_h = gen.layer_height_mm
    # 맨 아래 구슬은 베드에 얹혀야 한다. 구슬 중심을 층 중앙에 두면 반지름만큼
    # 베드 아래로 파고들어 슬라이서가 잘라내 버리므로, 첫 층 중심을 반지름
    # 높이에 맞추고 그 위로 격자 간격만큼 쌓는다.
    r0 = 0.5 * contact_params.bead_diameter_mm
    z_first = z0 + r0
    n_bead = max(1, int(math.ceil(float(mesh.extents[2]) / bead_h)))
    bead_support, bead_contact = [], []
    for j in range(n_bead):
        z = z_first + j * bead_h  # 그 층 구슬 중심의 높이
        k = min(len(det_slices) - 1, max(0, int((z - z0) / det_h)))
        bead_support.append(support[k])
        bead_contact.append(contact[k])
    if verbose:
        n_used = sum(1 for s in bead_support if not clean(s).is_empty)
        print(f"      구슬 층 {n_bead}개 @ {bead_h:.3f}mm, 서포터 필요 {n_used}개")

    # 격자 원점은 오브젝트 전체에서 딱 한 번만 정해진다.
    grid_origin = (float(mesh.bounds[0][0]), float(mesh.bounds[0][1]))
    plan = plan_beads(
        bead_support, bead_contact, gen, contact_params, body_params,
        grid_origin, z_first - 0.5 * bead_h,  # meshing 이 +h/2 해서 중심을 잡는다
    )
    if verbose:
        print(f"      bead {sum(len(l['beads']) for l in plan.layers)}개")
    return SupportResult(plan_to_mesh(plan, gen, detail), plan, det_slices)
