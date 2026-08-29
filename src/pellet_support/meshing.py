# -*- coding: utf-8 -*-
"""층별 배치 계획을 만들고, 그것을 실제 구 메쉬로 바꾼다."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import trimesh

from .lattice import bead_angle, support_bead_generate_centers
from .params import SupportBeadParams, SupportBeadRegion, SupportGenParams
from .slicing import clean


@dataclass
class BeadPlan:
    """층별 bead 배치 결과. 메쉬 생성과 좌표 내보내기가 공용으로 쓴다."""

    layers: List[dict] = field(default_factory=list)


def plan_beads(
    support,
    contact,
    gen: SupportGenParams,
    contact_params: SupportBeadParams,
    body_params: SupportBeadParams,
    grid_origin: Tuple[float, float],
    z0: float,
) -> BeadPlan:
    plan = BeadPlan()
    for i, sup in enumerate(support):
        sup = clean(sup)
        entry = {
            "layer": i,
            "z_bottom": z0 + i * gen.layer_height_mm,
            "solid": None,
            "beads": [],
        }
        if sup.is_empty:
            plan.layers.append(entry)
            continue

        # 첫 층은 접착/구조 때문에 원래대로 꽉 채운다.
        if i < gen.solid_first_layers:
            entry["solid"] = sup
            plan.layers.append(entry)
            continue

        con = clean(contact[i]).intersection(sup)
        body = sup.difference(con) if not con.is_empty else sup
        for region, geom, prm in (
            (SupportBeadRegion.CONTACT, con, contact_params),
            (SupportBeadRegion.BODY, body, body_params),
        ):
            geom = clean(geom)
            if geom.is_empty:
                continue
            ang = bead_angle(prm, i)
            for c in support_bead_generate_centers(geom, prm, i, grid_origin):
                entry["beads"].append(
                    {
                        "x": c.x,
                        "y": c.y,
                        "angle": ang,
                        "region": region,
                        "d": prm.bead_diameter_mm,
                    }
                )
        plan.layers.append(entry)
    return plan


def _unit_sphere(detail: int):
    sphere = trimesh.creation.icosphere(subdivisions=max(0, detail), radius=1.0)
    return np.asarray(sphere.vertices), np.asarray(sphere.faces)


def plan_to_mesh(
    plan: BeadPlan, gen: SupportGenParams, detail: int = 1
) -> trimesh.Trimesh:
    """bead 를 실제 구로 찍는다.

    수만 개를 하나씩 만들면 느리므로 단위구를 한 번만 만들고 반지름이 같은
    구들끼리 묶어 넘파이 브로드캐스팅으로 한 번에 평행이동한다. 겹치는
    구들은 슬라이서가 합집합으로 처리한다.
    """
    unit_v, unit_f = _unit_sphere(detail)
    h = gen.layer_height_mm

    by_radius: Dict[float, List[Tuple[float, float, float]]] = {}
    parts: List[trimesh.Trimesh] = []

    for entry in plan.layers:
        z_center = entry["z_bottom"] + 0.5 * h
        if entry["solid"] is not None:
            geom = entry["solid"]
            for poly in geom.geoms if hasattr(geom, "geoms") else [geom]:
                if poly.is_empty or poly.area < 1e-9:
                    continue
                try:
                    m = trimesh.creation.extrude_polygon(poly, h)
                except ValueError as exc:
                    # trimesh 는 폴리곤을 삼각형으로 쪼갤 엔진을 선택 설치로 둔다.
                    # 없으면 "No available triangulation engine!" 만 던지고 죽는데,
                    # 그 메시지만으로는 무엇을 깔아야 할지 알 수 없다.
                    if "triangulation engine" not in str(exc):
                        raise
                    raise RuntimeError(
                        "폴리곤을 삼각형으로 쪼갤 엔진이 없습니다(첫 층 바닥 생성에 필요). "
                        "터미널에서 다음을 실행하세요:  pip install mapbox-earcut"
                    ) from exc
                m.apply_translation([0, 0, entry["z_bottom"]])
                parts.append(m)
        for b in entry["beads"]:
            by_radius.setdefault(round(0.5 * b["d"], 5), []).append(
                (b["x"], b["y"], z_center)
            )

    for radius, pts in by_radius.items():
        centers = np.asarray(pts, dtype=np.float64)
        verts = (unit_v[None, :, :] * radius + centers[:, None, :]).reshape(-1, 3)
        offset = (np.arange(len(centers)) * len(unit_v))[:, None, None]
        faces = (unit_f[None, :, :] + offset).reshape(-1, 3)
        parts.append(trimesh.Trimesh(vertices=verts, faces=faces, process=False))

    if not parts:
        return trimesh.Trimesh()
    return trimesh.util.concatenate(parts)
