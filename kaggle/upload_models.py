#!/usr/bin/env python3
"""Upload HuggingFace model weights to Kaggle's model registry (one-time setup).

After running this, render_kernel.py will attach the pre-uploaded models to
kernels so they land in /kaggle/input/ instead of being downloaded at runtime,
solving the disk-space exhaustion problem on T4 kernels.

Usage:
    python kaggle/upload_models.py --size very_small
    python kaggle/upload_models.py --size very_small --hf-token hf_xxx
    python kaggle/upload_models.py --model "meta-llama/Llama-3.1-8B-Instruct"

Requirements:
    pip install huggingface_hub pyyaml kaggle
    kaggle.json credentials in ~/.kaggle/kaggle.json (chmod 600)
    ~50 GB free disk space (temp download dir, deleted after upload)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = KAGGLE_DIR / "model_registry.yaml"
CONFIG_PATH = REPO_ROOT / "configs" / "experiments.yaml"

FRAMEWORK = "transformers"
VARIATION = "default"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    name = text.split("/")[-1]
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return re.sub(r"-+", "-", s)[:50]


def _run(args: list[str]) -> tuple[str, str, int]:
    r = subprocess.run(args, capture_output=True, text=True)
    return r.stdout.strip(), r.stderr.strip(), r.returncode


def _kaggle_user() -> str:
    stdout, _, rc = _run(["kaggle", "config", "view"])
    for line in stdout.splitlines():
        if "username:" in line.lower():
            return line.split(":", 1)[1].strip()
    raise RuntimeError(
        "Could not detect Kaggle username. "
        "Run: kaggle config set -n username -v <your_username>"
    )


def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return yaml.safe_load(REGISTRY_PATH.read_text()) or {"models": {}}
    return {"models": {}}


def _save_registry(reg: dict) -> None:
    with open(REGISTRY_PATH, "w") as f:
        yaml.dump(reg, f, default_flow_style=False, sort_keys=False)


def _get_hf_ids_for_size(size: str) -> list[str]:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    key = f"models_{size}"
    raw = cfg.get("_shared", {}).get(key, [])
    if not raw:
        raise ValueError(f"Size group '{size}' not found in _shared config.")
    return [m if isinstance(m, str) else m["id"] for m in raw]


# ---------------------------------------------------------------------------
# Kaggle upload steps (each idempotent)
# ---------------------------------------------------------------------------

def _create_model(owner: str, slug: str, title: str) -> None:
    meta = {
        "ownerSlug": owner,
        "title": title,
        "slug": slug,
        "isPrivate": True,
        "licenses": [{"name": "Apache 2.0"}],
    }
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "model-metadata.json").write_text(json.dumps(meta, indent=2))
        stdout, stderr, rc = _run(["kaggle", "models", "create", "-p", tmp])
    combined = (stdout + stderr).lower()
    if rc != 0 and "already exists" not in combined and "conflict" not in combined:
        raise RuntimeError(f"kaggle models create failed:\n{stderr}\n{stdout}")


def _create_instance(owner: str, slug: str) -> None:
    meta = {
        "ownerSlug": owner,
        "modelSlug": slug,
        "framework": FRAMEWORK,
        "variation": VARIATION,
        "isPrivate": True,
        "licenseName": "Apache 2.0",
        "overview": "",
    }
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "instance-metadata.json").write_text(json.dumps(meta, indent=2))
        stdout, stderr, rc = _run(["kaggle", "models", "instances", "create", "-p", tmp])
    combined = (stdout + stderr).lower()
    if rc != 0 and "already exists" not in combined and "conflict" not in combined:
        raise RuntimeError(f"kaggle models instances create failed:\n{stderr}\n{stdout}")


def _create_version(instance_handle: str, files_dir: str, hf_id: str) -> int:
    stdout, stderr, rc = _run([
        "kaggle", "models", "instances", "versions", "create",
        instance_handle,
        "-p", files_dir,
        "--version-notes", f"Uploaded from HuggingFace Hub: {hf_id}",
    ])
    if rc != 0:
        raise RuntimeError(f"kaggle models instances versions create failed:\n{stderr}\n{stdout}")
    # Try to parse the new version number from CLI output
    for line in stdout.splitlines():
        m = re.search(r"version[:\s]+(\d+)", line, re.I)
        if m:
            return int(m.group(1))
    return 1


# ---------------------------------------------------------------------------
# Main upload function
# ---------------------------------------------------------------------------

def upload_model(
    hf_id: str,
    kaggle_user: str,
    hf_token: str | None,
    tmp_dir: str | None = None,
) -> str:
    """Download from HF Hub and upload to Kaggle Models. Returns kaggle_source.

    Args:
        tmp_dir: Root directory for the temporary model download. Each model is
                 stored in a sub-directory named after its slug and deleted after
                 upload. Defaults to the system temp dir.
    """
    reg = _load_registry()
    if hf_id in reg.get("models", {}):
        existing = reg["models"][hf_id]["kaggle_source"]
        print(f"  [skip] already in registry: {existing}")
        return existing

    slug = _slugify(hf_id)
    title = hf_id.split("/")[-1]
    model_handle = f"{kaggle_user}/{slug}"
    instance_handle = f"{model_handle}/{FRAMEWORK}/{VARIATION}"

    print(f"  Downloading {hf_id} from HuggingFace Hub …")
    with tempfile.TemporaryDirectory(dir=tmp_dir, prefix=f"{slug}_") as tmp:
        local_dir = snapshot_download(
            repo_id=hf_id,
            local_dir=tmp,
            token=hf_token,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
        )

        print(f"  Creating Kaggle model {model_handle} …")
        _create_model(kaggle_user, slug, title)

        print(f"  Creating model instance {instance_handle} …")
        _create_instance(kaggle_user, slug)

        print(f"  Uploading model files (this may take several minutes) …")
        version = _create_version(instance_handle, local_dir, hf_id)

    kaggle_source = f"{instance_handle}/{version}"
    reg.setdefault("models", {})[hf_id] = {"kaggle_source": kaggle_source}
    _save_registry(reg)
    print(f"  Done → {kaggle_source}")
    return kaggle_source


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--size", metavar="SIZE",
                     help="Upload all models for a size group (e.g. very_small, small, medium)")
    src.add_argument("--model", metavar="HF_ID", action="append", dest="models",
                     help="Specific HF model ID to upload (repeatable)")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="HuggingFace token for gated models (default: $HF_TOKEN)")
    p.add_argument("--kaggle-user", default=None,
                   help="Kaggle username (auto-detected from kaggle CLI config if omitted)")
    p.add_argument("--tmp-dir", default=None, metavar="PATH",
                   help="Directory for temporary model downloads (default: system temp). "
                        "Use this to point at a drive with enough free space, e.g. "
                        "--tmp-dir /media/adriano/my-drive. Each model is downloaded "
                        "into a sub-folder and deleted automatically after upload.")
    args = p.parse_args()

    kaggle_user = args.kaggle_user or _kaggle_user()
    hf_ids = args.models if args.models else _get_hf_ids_for_size(args.size)

    if args.tmp_dir:
        Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)
        print(f"Tmp dir     : {args.tmp_dir}")

    print(f"Kaggle user : {kaggle_user}")
    print(f"Models      : {len(hf_ids)}")
    for hf_id in hf_ids:
        print(f"\n── {hf_id}")
        try:
            upload_model(hf_id, kaggle_user, args.hf_token, tmp_dir=args.tmp_dir)
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            return 1

    print("\nDone. kaggle/model_registry.yaml updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
