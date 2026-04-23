# Copyright 2026 Ali Seyhan, Baki Turhan
# Conforms to SWARM_ARCHITECTURE_SPEC.md v2.0.0
"""quinkgl init — scaffold a user peer-script project (§10.7.2, Appendix D.1)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import quinkgl

from .exit_codes import IO_ERROR, SUCCESS, VALIDATION_ERROR

if TYPE_CHECKING:
    from argparse import _SubParsersAction


SPEC_VERSION = "2.0.0"
TEMPLATES_DIR = Path(__file__).with_suffix("").parent / "templates"


def build_parser(sub: _SubParsersAction) -> None:
    parser = sub.add_parser("init", help="Scaffold a user peer-script project")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--framework", choices=["pytorch", "tensorflow", "custom"], default=None)
    parser.add_argument(
        "--template",
        choices=["minimal", "pytorch-vision", "pytorch-tabular", "custom"],
        default="minimal",
    )


class _DefaultingDict(dict):
    """Returns ``{key}`` for missing keys so unrendered placeholders survive
    as regression-test signals."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _build_template_vars(args: argparse.Namespace, output_dir: Path) -> dict[str, str]:
    """Collect template variables per Appendix D.1.3."""
    manifest_path = args.manifest
    manifest_name = "<swarm-name>"
    manifest_filename = "<your-swarm>.qgl"
    swarm_id_short = "<swarm-id>"
    input_shape = "[3, 224, 224]"
    output_shape = "[10]"
    arch_hash = "sha256:<arch-hash>"
    framework = args.framework or "pytorch"

    if manifest_path and Path(manifest_path).exists():
        from quinkgl.manifest import SwarmManifest

        m = SwarmManifest.from_file(manifest_path)
        manifest_name = m.name
        manifest_filename = Path(manifest_path).name
        swarm_id_short = m.manifest_hash()[:12]
        input_shape = json.dumps(m.task.input_shape)
        output_shape = json.dumps(m.task.output_shape)
        arch_hash = m.model.arch_hash
        framework = m.model.framework

    major, minor = quinkgl.__version__.split(".")[:2]
    return _DefaultingDict(
        project_name=output_dir.name,
        framework=framework,
        template=args.template,
        quinkgl_version=quinkgl.__version__,
        quinkgl_pin=f">={major}.{minor},<{major}.{int(minor) + 1}",
        spec_version=SPEC_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        manifest_name=manifest_name,
        manifest_filename=manifest_filename,
        swarm_id_short=swarm_id_short,
        input_shape=input_shape,
        output_shape=output_shape,
        arch_hash=arch_hash,
    )


def _render_template(src: Path, dst: Path, vars: dict[str, str]) -> None:
    text = src.read_text(encoding="utf-8")
    rendered = text.format_map(vars)
    dst.write_text(rendered, encoding="utf-8")


def _scaffold(template_name: str, output_dir: Path, vars: dict[str, str]) -> list[Path]:
    template_dir = TEMPLATES_DIR / template_name
    if not template_dir.exists():
        raise FileNotFoundError(f"Template not found: {template_name}")

    created: list[Path] = []
    for src in template_dir.rglob("*.tmpl"):
        rel = src.relative_to(template_dir)
        # Strip .tmpl suffix; rename gitignore.tmpl -> .gitignore
        parts = list(rel.parts)
        fname = parts[-1]
        if fname == "gitignore.tmpl":
            parts[-1] = ".gitignore"
        elif fname.endswith(".tmpl"):
            parts[-1] = fname[:-5]
        dst = output_dir / Path(*parts)
        dst.parent.mkdir(parents=True, exist_ok=True)
        _render_template(src, dst, vars)
        created.append(dst)

    return created


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        print(f"Output directory already exists: {output_dir}", file=sys.stderr)
        return IO_ERROR

    try:
        vars = _build_template_vars(args, output_dir)
        created = _scaffold(args.template, output_dir, vars)

        # If a manifest path was provided, copy it into the scaffold
        if args.manifest and Path(args.manifest).exists():
            import shutil
            shutil.copy(args.manifest, output_dir / vars["manifest_filename"])

        if args.json:
            print(json.dumps({"created": [str(p.relative_to(output_dir)) for p in created]}))
        else:
            print(f"Scaffolded {args.template} template at {output_dir}")
        return SUCCESS
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return VALIDATION_ERROR
