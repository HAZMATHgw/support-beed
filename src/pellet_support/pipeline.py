# -*- coding: utf-8 -*-
"""슬라이싱부터 메쉬까지 한 번에 묶는 진입점."""

from __future__ import annotations

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


class SupportResult(NamedTuple):
    mesh: trimesh.Trimesh
    plan: Optional[BeadPlan]
    slices: list


def make_params(
    nozzle_diameter_mm: float = 1.0,
    overlap: Optional[float] = None,
    body_bead_ratio: float = 0.97,
    stagger_period: int = 3,
    straight_columns: bool = False,
    segment_ratio: Optional[float] = None,
    edge_margin_ratio: Optional[float] = None,
) -> Tuple[SupportBeadParams, SupportBeadParams]:
    """CLI 와 라이브러리가 공유하는 파라미터 조립 로직."""
    contact = support_bead_contact_params(nozzle_diameter_mm)
    body = support_bead_body_params(nozzle_diameter_mm, body_bead_ratio)
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
    return unify_lattice(contact, body)


def generate_support(
    mesh: trimesh.Trimesh,
    gen: SupportGenParams,
    contact_params: SupportBeadParams,
    body_params: SupportBeadParams,
    detail: int = 1,
    verbose: bool = True,
) -> SupportResult:
    slices, _ = slice_model(mesh, gen.layer_height_mm, gen.max_layers)
    if verbose:
        print(f"      레이어 {len(slices)}개")

    support, contact = build_support_regions(slices, gen)
    n_used = sum(1 for s in support if not clean(s).is_empty)
    if verbose:
        print(f"      서포터가 필요한 레이어 {n_used}개")
    if n_used == 0:
        return SupportResult(trimesh.Trimesh(), None, slices)

    # 격자 원점은 오브젝트 전체에서 딱 한 번만 정해진다.
    grid_origin = (float(mesh.bounds[0][0]), float(mesh.bounds[0][1]))
    plan = plan_beads(
        support,
        contact,
        gen,
        contact_params,
        body_params,
        grid_origin,
        float(mesh.bounds[0][2]),
    )
    if verbose:
        print(f"      bead {sum(len(l['beads']) for l in plan.layers)}개")
    return SupportResult(plan_to_mesh(plan, gen, detail), plan, slices)
