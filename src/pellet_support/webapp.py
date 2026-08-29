# -*- coding: utf-8 -*-
"""로컬 웹 UI.

    pellet-support-web
    또는  python -m pellet_support.webapp

브라우저에서 http://127.0.0.1:5000 을 열고 모델 파일을 끌어다 놓으면
서포터가 포함된 파일을 되돌려준다. 파일은 임시 폴더에만 잠깐 저장되고
서버를 끄면 사라진다.

GitHub Pages 같은 정적 호스팅에서는 동작하지 않는다. trimesh / shapely 가
서버에서 실행되어야 하기 때문에 반드시 로컬(또는 직접 띄운 서버)에서 돌려야 한다.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import threading
import uuid
import webbrowser
from typing import Dict

import trimesh
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_from_directory,
)

from .params import SupportGenParams
from .pipeline import generate_support, make_params
from .report import export_preview, load_mesh, measure_packing

MESH_EXTS = (".stl", ".obj", ".3mf", ".ply", ".off")
MAX_UPLOAD_MB = 200

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

#: job_id -> 임시 폴더 경로
_JOBS: Dict[str, str] = {}
_JOBS_LOCK = threading.Lock()


def safe_filename(name: str) -> str:
    """경로 조작만 막고 한글 등 유니코드 파일명은 살린다.

    werkzeug 의 secure_filename 은 비ASCII 문자를 통째로 버려서
    '배_모델.3mf' 가 '_.3mf' 가 되어 버린다. 여기서는 디렉터리 구분자와
    제어문자, 윈도우 금지문자만 제거한다.
    """
    name = os.path.basename(name.replace("\\", "/"))
    name = re.sub(r'[\x00-\x1f<>:"/\\|?*]', "_", name).strip(" .")
    return name[:120] or "model"


def _form_float(name: str, default: float) -> float:
    raw = request.form.get(name, "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _form_int(name: str, default: int) -> int:
    raw = request.form.get(name, "")
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def _form_bool(name: str) -> bool:
    return request.form.get(name, "").lower() in ("1", "true", "on", "yes")


@app.route("/")
def index():
    return render_template("index.html", max_mb=MAX_UPLOAD_MB)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    upload = request.files.get("model")
    if upload is None or not upload.filename:
        return jsonify(error="모델 파일을 선택하세요."), 400

    filename = safe_filename(upload.filename)
    stem, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext not in MESH_EXTS:
        return jsonify(
            error=f"지원하지 않는 형식입니다: {ext or '확장자 없음'}. "
                  f"{', '.join(MESH_EXTS)} 중 하나를 올려주세요."
        ), 400

    job = uuid.uuid4().hex[:12]
    workdir = tempfile.mkdtemp(prefix=f"pellet-{job}-")
    with _JOBS_LOCK:
        _JOBS[job] = workdir

    in_path = os.path.join(workdir, filename)
    upload.save(in_path)

    out_ext = request.form.get("format", "same")
    if out_ext == "same" or out_ext not in MESH_EXTS:
        out_ext = ext

    try:
        contact, body = make_params(
            nozzle_diameter_mm=_form_float("nozzle", 1.0),
            overlap=_form_float("overlap", 0.08),
            body_bead_ratio=_form_float("body_bead_ratio", 0.97),
            stagger_period=_form_int("stagger_period", 3),
            straight_columns=_form_bool("straight_columns"),
        )
        gen = SupportGenParams(
            nozzle_diameter_mm=_form_float("nozzle", 1.0),
            layer_height_mm=contact.layer_height_mm(),
            overhang_angle_deg=_form_float("overhang_angle", 45.0),
            contact_z_gap_layers=_form_int("z_gap_layers", 1),
            xy_clearance_mm=_form_float("xy_clearance", 0.8),
            contact_layers=_form_int("contact_layers", 2),
            max_layers=_form_int("max_layers", 4000),
        )

        mesh = load_mesh(in_path)
        mesh.apply_translation([0, 0, -mesh.bounds[0][2]])

        result = generate_support(
            mesh, gen, contact, body,
            detail=_form_int("sphere_detail", 1),
            verbose=False,
        )
        if result.mesh.is_empty:
            return jsonify(
                error="이 모델에는 서포터가 필요하지 않습니다. "
                      "오버행 각도를 올려서 다시 시도해 보세요."
            ), 422

        files = []

        support_name = f"{stem}_support{out_ext}"
        result.mesh.export(os.path.join(workdir, support_name))
        files.append({"name": support_name, "label": "서포터만"})

        if _form_bool("with_model"):
            scene = trimesh.Scene()
            scene.add_geometry(mesh, node_name="model")
            scene.add_geometry(result.mesh, node_name="support")
            both_name = f"{stem}_with_support{out_ext}"
            scene.export(os.path.join(workdir, both_name))
            files.append({"name": both_name, "label": "모델 + 서포터"})

        preview_url = None
        try:
            candidates = [l["layer"] for l in result.plan.layers if l["beads"]]
            if candidates:
                export_preview(
                    result.plan, result.slices, candidates[-1],
                    os.path.join(workdir, "preview.png"),
                )
                preview_url = f"/files/{job}/preview.png"
        except Exception:
            pass  # matplotlib 이 없으면 미리보기만 건너뛴다

        stats = None
        try:
            stats = measure_packing(result.plan, gen, contact.pitch_mm())
        except Exception:
            pass  # scipy 가 없으면 검증만 건너뛴다

        n_beads = sum(len(l["beads"]) for l in result.plan.layers)
        return jsonify(
            job=job,
            files=[
                {**f, "url": f"/files/{job}/{f['name']}"} for f in files
            ],
            preview=preview_url,
            layer_height=round(gen.layer_height_mm, 4),
            pitch=round(contact.pitch_mm(), 4),
            layers=len(result.slices),
            beads=n_beads,
            triangles=len(result.mesh.faces),
            volume_cm3=round(abs(result.mesh.volume) / 1000.0, 3),
            measured=stats,
        )
    except MemoryError:
        return jsonify(error="모델이 너무 큽니다. 노즐 지름을 키우거나 모델을 단순화하세요."), 500
    except Exception as exc:  # noqa: BLE001 - 사용자에게 원인을 그대로 보여준다
        return jsonify(error=f"{type(exc).__name__}: {exc}"), 500


@app.route("/files/<job>/<path:name>")
def files(job: str, name: str):
    with _JOBS_LOCK:
        workdir = _JOBS.get(job)
    if workdir is None:
        return jsonify(error="만료된 작업입니다. 다시 생성해 주세요."), 404
    as_attachment = not name.endswith(".png")
    return send_from_directory(workdir, name, as_attachment=as_attachment)


def cleanup() -> None:
    with _JOBS_LOCK:
        for path in _JOBS.values():
            shutil.rmtree(path, ignore_errors=True)
        _JOBS.clear()


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="pellet-support-web", description="펠릿 서포터 생성기 로컬 웹 UI"
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-browser", action="store_true", help="브라우저 자동 실행 안 함")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)

    url = f"http://{args.host}:{args.port}"
    print(f"펠릿 서포터 생성기 실행 중 -> {url}")
    print("종료하려면 Ctrl+C")
    if not args.no_browser and not args.debug:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        app.run(host=args.host, port=args.port, debug=args.debug)
    finally:
        cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
