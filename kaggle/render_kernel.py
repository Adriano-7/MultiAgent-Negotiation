#!/usr/bin/env python3
"""Render a per-(experiment, size) Kaggle kernel staging directory.

Substitutes placeholders into kaggle/kernel.py and kaggle/kernel-metadata.template.json
and writes the pair to a fresh staging dir.

If kaggle/model_registry.yaml contains entries for the experiment's models, those
models are attached via model_sources in the kernel metadata and their local
/kaggle/input/... paths are baked into the kernel via {{MODEL_MAP_JSON}}.
Otherwise the kernel downloads weights from HuggingFace Hub at runtime (original
behaviour, but risks disk exhaustion).

Outputs JSON to stdout describing the rendered staging dir; launch.sh consumes it.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = KAGGLE_DIR / "model_registry.yaml"
EXPERIMENTS_CONFIG = REPO_ROOT / "configs" / "experiments.yaml"


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9-]+", "-", text.lower()).strip("-")
    return re.sub(r"-+", "-", s)


def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return yaml.safe_load(REGISTRY_PATH.read_text()) or {"models": {}}
    return {"models": {}}


def _get_hf_ids(all_configs: dict, experiment: str, size: str) -> list[str]:
    """Return list of HF model IDs for this experiment/size combination."""
    if size and size != "none":
        key = f"models_{size}"
        raw = all_configs.get("_shared", {}).get(key, [])
    else:
        models_cfg = all_configs.get(experiment, {}).get("models", [])
        if isinstance(models_cfg, str):
            raw = all_configs.get("_shared", {}).get(f"models_{models_cfg}", [])
        else:
            raw = models_cfg
    return [m if isinstance(m, str) else m["id"] for m in raw]


def _build_model_sources_and_map(
    hf_ids: list[str], registry: dict
) -> tuple[list[str], dict[str, str]]:
    """Return (model_sources, {hf_id: local_kaggle_path}).

    Warns for any model not yet in the registry (will fall back to HF download).
    Kaggle mounts <owner>/<slug>/transformers/default/<v> at
    /kaggle/input/<slug>/transformers/default/<v>/
    """
    reg_models = registry.get("models", {})
    sources: list[str] = []
    model_map: dict[str, str] = {}

    for hf_id in hf_ids:
        if hf_id not in reg_models:
            print(
                f"  [warn] {hf_id} not in model_registry.yaml — "
                "will download from HuggingFace at runtime",
                file=sys.stderr,
            )
            continue
        kaggle_source = reg_models[hf_id]["kaggle_source"]
        sources.append(kaggle_source)
        # Strip the owner (first path component) to get the Kaggle mount path
        parts = kaggle_source.split("/")  # owner/slug/framework/variation/version
        local_path = "/kaggle/input/" + "/".join(parts[1:])
        model_map[hf_id] = local_path

    return sources, model_map


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--size", default="none", help="model_group name or 'none'")
    p.add_argument("--git-ref", required=True, help="commit hash to check out in-kernel")
    p.add_argument("--git-repo", default="https://github.com/Adriano-7/MultiAgent-Negotiation.git")
    p.add_argument("--user", required=True, help="Kaggle username")
    p.add_argument("--out", required=True, help="staging dir to write into")
    p.add_argument("--gpu-type", default="T4 x2")
    p.add_argument("--kernel-template", default=str(KAGGLE_DIR / "kernel.py"))
    p.add_argument(
        "--metadata-template",
        default=str(KAGGLE_DIR / "kernel-metadata.template.json"),
    )
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    size_for_slug = args.size if args.size and args.size != "none" else "default"
    slug_base = slugify(f"{args.experiment}-{size_for_slug}")
    slug = f"{slug_base}-{args.git_ref[:8]}"
    kernel_id = f"{args.user}/{slug}"
    title = slug.replace("-", " ")

    submitted_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # --- Model registry lookup -----------------------------------------------
    all_configs = yaml.safe_load(EXPERIMENTS_CONFIG.read_text())
    registry = _load_registry()
    hf_ids = _get_hf_ids(all_configs, args.experiment, args.size)
    model_sources, model_map = _build_model_sources_and_map(hf_ids, registry)

    if model_map:
        print(
            f"  [registry] attaching {len(model_map)}/{len(hf_ids)} model(s) "
            "from Kaggle registry",
            file=sys.stderr,
        )

    # --- Render kernel --------------------------------------------------------
    model_map_json = json.dumps(model_map)
    kernel_template = Path(args.kernel_template).read_text()
    rendered = (
        kernel_template
        .replace("{{EXPERIMENT}}", args.experiment)
        .replace("{{SIZE}}", args.size or "none")
        .replace("{{GIT_REF}}", args.git_ref)
        .replace("{{GIT_REPO}}", args.git_repo)
        .replace("{{KAGGLE_GPU_TYPE}}", args.gpu_type)
        .replace("{{SLUG}}", slug)
        .replace("{{KERNEL_ID}}", kernel_id)
        .replace("{{SUBMITTED_AT}}", submitted_at)
        .replace("{{GITHUB_TOKEN}}", os.environ.get("GITHUB_TOKEN", ""))
        .replace("{{HF_TOKEN}}", os.environ.get("HF_TOKEN", ""))
        .replace("{{MODEL_MAP_JSON}}", model_map_json)
    )
    (out_dir / "kernel.py").write_text(rendered)

    # --- Render metadata ------------------------------------------------------
    metadata = json.loads(Path(args.metadata_template).read_text())
    metadata["id"] = kernel_id
    metadata["title"] = title
    metadata["model_sources"] = model_sources
    (out_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(json.dumps({
        "slug": slug,
        "kernel_id": kernel_id,
        "experiment": args.experiment,
        "size": args.size,
        "gpu_type": args.gpu_type,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
