#!/bin/bash
# ============================================================
# Launch a Kaggle CPU kernel that downloads models from HuggingFace
# Hub and uploads them to your Kaggle Models registry.
#
# After the kernel finishes, pull the updated model_registry.yaml with:
#   git fetch origin model-registry
#   git checkout origin/model-registry -- kaggle/model_registry.yaml
#
# Required secrets in Kaggle (Settings → Add-ons → Secrets):
#   HF_TOKEN        — HuggingFace token (needed for gated models like Llama)
#   GITHUB_TOKEN    — GitHub PAT with repo write access
#   KAGGLE_USERNAME — your Kaggle username
#   KAGGLE_KEY      — your Kaggle API key
#
# Usage:
#   bash kaggle/launch_upload.sh --size very_small
#   bash kaggle/launch_upload.sh --size small
#   DRY_RUN=1 bash kaggle/launch_upload.sh --size very_small
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Load .env ───────────────────────────────────────────────
if [ -f "$REPO_DIR/.env" ]; then
    set -a; source "$REPO_DIR/.env"; set +a
fi

# ── Parse args ──────────────────────────────────────────────
SIZE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --size) SIZE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$SIZE" ]; then
    echo "Usage: bash kaggle/launch_upload.sh --size <very_small|small|medium|big>" >&2
    exit 1
fi

# ── Resolve Kaggle username ─────────────────────────────────
KAGGLE_USER="${KAGGLE_USER:-${KAGGLE_USERNAME:-}}"
if [ -z "${KAGGLE_USER:-}" ] && command -v kaggle >/dev/null 2>&1; then
    KAGGLE_USER="$(kaggle config view 2>/dev/null | awk -F': ' '/^- username:/ {print $2; exit}')"
fi
if [ -z "${KAGGLE_USER:-}" ]; then
    echo "ERROR: set KAGGLE_USERNAME in .env or KAGGLE_USER inline." >&2; exit 1
fi

# ── Git ref / repo ──────────────────────────────────────────
GIT_REF="${GIT_REF:-$(git -C "$REPO_DIR" rev-parse HEAD)}"
GIT_REPO="${GIT_REPO:-https://github.com/Adriano-7/MultiAgent-Negotiation.git}"

# ── Get HF model IDs for this size group ───────────────────
HF_IDS_JSON="$(python3 - <<EOF
import json, yaml
with open("$REPO_DIR/configs/experiments.yaml") as f:
    cfg = yaml.safe_load(f)
raw = cfg.get("_shared", {}).get("models_$SIZE", [])
if not raw:
    raise SystemExit(f"Size group '$SIZE' not found in _shared config.")
ids = [m if isinstance(m, str) else m["id"] for m in raw]
print(json.dumps(ids))
EOF
)"

echo "============================================"
echo "Target     : Kaggle (CPU kernel, no GPU)"
echo "User       : $KAGGLE_USER"
echo "Size group : $SIZE"
echo "Git ref    : $GIT_REF"
echo "Models     : $HF_IDS_JSON"
echo "Dry run    : ${DRY_RUN:-no}"
echo "============================================"

# ── Render the upload kernel ────────────────────────────────
SLUG="upload-models-${SIZE}-${GIT_REF:0:8}"
KERNEL_ID="$KAGGLE_USER/$SLUG"
STAGING_DIR="$SCRIPT_DIR/.staging/upload__${SIZE}__${GIT_REF:0:8}"
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"

# Substitute all placeholders in upload_kernel.py
python3 - <<EOF
import os

template = open("$SCRIPT_DIR/upload_kernel.py").read()
rendered = (template
    .replace("{{HF_TOKEN}}",        os.environ.get("HF_TOKEN", ""))
    .replace("{{GITHUB_TOKEN}}",    os.environ.get("GITHUB_TOKEN", ""))
    .replace("{{KAGGLE_USERNAME}}", os.environ.get("KAGGLE_USERNAME", "$KAGGLE_USER"))
    .replace("{{KAGGLE_KEY}}",      os.environ.get("KAGGLE_KEY", ""))
    .replace("{{HF_IDS_JSON}}",     """$HF_IDS_JSON""")
    .replace("{{KAGGLE_USER}}",     "$KAGGLE_USER")
    .replace("{{GIT_REPO}}",        "$GIT_REPO")
    .replace("{{GIT_REF}}",         "$GIT_REF")
)
open("$STAGING_DIR/kernel.py", "w").write(rendered)
EOF

# Write kernel metadata (CPU kernel — no GPU needed)
python3 - <<EOF
import json
metadata = {
    "id": "$KERNEL_ID",
    "title": "$SLUG",
    "code_file": "kernel.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": [],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": [],
}
open("$STAGING_DIR/kernel-metadata.json", "w").write(json.dumps(metadata, indent=2) + "\n")
EOF

echo "Staging dir: $STAGING_DIR"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[DRY RUN] would push kernel: $KERNEL_ID"
    echo "[DRY RUN] kernel.py preview (first 20 lines):"
    head -20 "$STAGING_DIR/kernel.py"
    exit 0
fi

# ── Push ────────────────────────────────────────────────────
echo "Submitting: $KERNEL_ID"
kaggle kernels push -p "$STAGING_DIR"

echo ""
echo "Kernel submitted. Monitor at:"
echo "  https://www.kaggle.com/code/$KERNEL_ID"
echo ""
echo "When done, pull the updated registry with:"
echo "  git fetch origin model-registry"
echo "  git checkout origin/model-registry -- kaggle/model_registry.yaml"
