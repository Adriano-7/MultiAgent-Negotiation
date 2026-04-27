"""Kaggle CPU kernel: download models from HuggingFace and upload to Kaggle Models registry.

Substituted by launch_upload.sh:
    {{HF_TOKEN}}        HuggingFace token (for gated models like Llama)
    {{GITHUB_TOKEN}}    GitHub PAT (to push updated model_registry.yaml back)
    {{KAGGLE_USERNAME}} Kaggle username  (for kaggle API auth within the kernel)
    {{KAGGLE_KEY}}      Kaggle API key
    {{HF_IDS_JSON}}     JSON list of HF model IDs to upload, e.g. ["meta-llama/..."]
    {{KAGGLE_USER}}     Kaggle username (owner slug for model paths)
    {{GIT_REPO}}        HTTPS URL of the GitHub repo
    {{GIT_REF}}         Commit hash to clone (so we get the right experiments.yaml)
"""
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HF_IDS: list = {{HF_IDS_JSON}}
KAGGLE_USER = "{{KAGGLE_USER}}"
GIT_REPO = "{{GIT_REPO}}"
GIT_REF = "{{GIT_REF}}"

FRAMEWORK = "transformers"
VARIATION = "default"
REGISTRY_BRANCH = "model-registry"
REGISTRY_YAML = "kaggle/model_registry.yaml"
WORK_DIR = Path("/kaggle/working")
MODEL_TMP = WORK_DIR / "model_tmp"
HF_HOME_DIR = WORK_DIR / "hf_cache"   # controlled cache dir — wiped after each model
REPO_DIR = WORK_DIR / "repo"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _disk_free_gb() -> float:
    import shutil as _shutil
    return _shutil.disk_usage(WORK_DIR).free / 1024 ** 3


def setup_secrets() -> None:
    os.environ["HF_TOKEN"] = "{{HF_TOKEN}}"
    os.environ["GITHUB_TOKEN"] = "{{GITHUB_TOKEN}}"
    os.environ["KAGGLE_USERNAME"] = "{{KAGGLE_USERNAME}}"
    os.environ["KAGGLE_KEY"] = "{{KAGGLE_KEY}}"
    # Point HF downloads at a directory we fully control so we can wipe it after
    # each model upload. Without this, snapshot_download also writes to
    # ~/.cache/huggingface/hub and that cache is never cleaned up, exhausting disk.
    os.environ["HF_HOME"] = str(HF_HOME_DIR)

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    creds_file = kaggle_dir / "kaggle.json"
    creds_file.write_text(json.dumps({
        "username": os.environ["KAGGLE_USERNAME"],
        "key": os.environ["KAGGLE_KEY"],
    }))
    creds_file.chmod(0o600)
    print("[setup] secrets configured")


def install_deps() -> None:
    # --upgrade is required: Kaggle kernels ship kaggle 2.0.0 which has a
    # "Key description not found in data" bug when creating models via the API.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade",
         "kaggle", "pyyaml", "huggingface_hub"],
        check=True,
    )
    print("[setup] deps installed")


def clone_repo() -> None:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    cmd = ["git"]
    if token:
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        cmd += ["-c", f"http.extraHeader=Authorization: Basic {basic}",
                "-c", "credential.helper="]
    subprocess.run(cmd + ["clone", GIT_REPO, str(REPO_DIR)], check=True, env=env)
    subprocess.run(["git", "-C", str(REPO_DIR), "checkout", GIT_REF], check=True, env=env)
    print(f"[setup] repo cloned at {GIT_REF[:8]}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def load_registry() -> dict:
    reg_path = REPO_DIR / REGISTRY_YAML
    if reg_path.exists():
        import yaml
        return yaml.safe_load(reg_path.read_text()) or {"models": {}}
    return {"models": {}}


# ---------------------------------------------------------------------------
# Kaggle model upload helpers
# ---------------------------------------------------------------------------

def _run(args: list) -> tuple:
    r = subprocess.run(args, capture_output=True, text=True)
    return r.stdout.strip(), r.stderr.strip(), r.returncode


def _slugify(text: str) -> str:
    name = text.split("/")[-1]
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return re.sub(r"-+", "-", s)[:50]


def _create_model(owner: str, slug: str, title: str) -> None:
    meta = {"ownerSlug": owner, "title": title, "slug": slug,
            "isPrivate": True, "licenses": [{"name": "Apache 2.0"}]}
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "model-metadata.json").write_text(json.dumps(meta))
        stdout, stderr, rc = _run(["kaggle", "models", "create", "-p", tmp])
    combined = (stdout + stderr).lower()
    if rc != 0 and "already exists" not in combined and "conflict" not in combined:
        raise RuntimeError(f"model create failed:\n{stderr}\n{stdout}")


def _create_instance(owner: str, slug: str) -> None:
    meta = {"ownerSlug": owner, "modelSlug": slug, "framework": FRAMEWORK,
            "variation": VARIATION, "isPrivate": True, "licenseName": "Apache 2.0", "overview": ""}
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "instance-metadata.json").write_text(json.dumps(meta))
        stdout, stderr, rc = _run(["kaggle", "models", "instances", "create", "-p", tmp])
    combined = (stdout + stderr).lower()
    if rc != 0 and "already exists" not in combined and "conflict" not in combined:
        raise RuntimeError(f"instance create failed:\n{stderr}\n{stdout}")


def _create_version(instance_handle: str, files_dir: str, hf_id: str) -> int:
    stdout, stderr, rc = _run([
        "kaggle", "models", "instances", "versions", "create",
        instance_handle, "-p", files_dir,
        "--version-notes", f"Uploaded from HuggingFace Hub: {hf_id}",
    ])
    if rc != 0:
        raise RuntimeError(f"version create failed:\n{stderr}\n{stdout}")
    for line in stdout.splitlines():
        m = re.search(r"version[:\s]+(\d+)", line, re.I)
        if m:
            return int(m.group(1))
    return 1


# ---------------------------------------------------------------------------
# Per-model upload (download → upload → delete)
# ---------------------------------------------------------------------------

def upload_model(hf_id: str, registry: dict) -> str:
    """Returns the kaggle_source string and updates registry in place."""
    if hf_id in registry.get("models", {}):
        existing = registry["models"][hf_id]["kaggle_source"]
        print(f"  [skip] already in registry: {existing}")
        return existing

    from huggingface_hub import snapshot_download

    slug = _slugify(hf_id)
    title = hf_id.split("/")[-1]
    instance_handle = f"{KAGGLE_USER}/{slug}/{FRAMEWORK}/{VARIATION}"

    MODEL_TMP.mkdir(parents=True, exist_ok=True)
    print(f"  Disk free before download: {_disk_free_gb():.1f} GB")
    try:
        print(f"  Downloading {hf_id} …")
        local_dir = snapshot_download(
            repo_id=hf_id,
            local_dir=str(MODEL_TMP),
            token=os.environ.get("HF_TOKEN") or None,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
        )

        print(f"  Creating Kaggle model {KAGGLE_USER}/{slug} …")
        _create_model(KAGGLE_USER, slug, title)

        print(f"  Creating model instance …")
        _create_instance(KAGGLE_USER, slug)

        print(f"  Uploading files (this may take a few minutes) …")
        version = _create_version(instance_handle, local_dir, hf_id)

    finally:
        # Wipe both the model files AND the HF hub cache so the next model
        # starts with a clean slate. Without wiping HF_HOME_DIR, the cache
        # accumulates across models and fills the 20 GB working disk.
        print(f"  Cleaning up temp files and HF cache …")
        shutil.rmtree(MODEL_TMP, ignore_errors=True)
        shutil.rmtree(HF_HOME_DIR, ignore_errors=True)
        print(f"  Disk free after cleanup: {_disk_free_gb():.1f} GB")

    kaggle_source = f"{instance_handle}/{version}"
    registry.setdefault("models", {})[hf_id] = {"kaggle_source": kaggle_source}
    print(f"  Done → {kaggle_source}")
    return kaggle_source


# ---------------------------------------------------------------------------
# Push updated registry back to GitHub
# ---------------------------------------------------------------------------

def push_registry(registry: dict) -> None:
    import yaml

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        print("[push] skipped: no GITHUB_TOKEN")
        return

    (REPO_DIR / REGISTRY_YAML).write_text(
        yaml.dump(registry, default_flow_style=False, sort_keys=False)
    )

    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    # -c flags are global git options and must appear BEFORE the subcommand:
    #   git -c key=val fetch   ✓
    #   git fetch -c key=val   ✗  (causes "unknown switch `c'" on some git versions)
    auth = ["-c", f"http.extraHeader=Authorization: Basic {basic}", "-c", "credential.helper="]
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    repo = ["-C", str(REPO_DIR)]

    def git(*args):
        subprocess.run(["git"] + repo + list(args), env=env, check=True)

    def git_auth(*args):
        subprocess.run(["git"] + auth + repo + list(args), env=env, check=True)

    git_auth("fetch", "origin")

    # Checkout or create the registry branch
    r = subprocess.run(
        ["git"] + repo + ["checkout", "-B", REGISTRY_BRANCH, f"origin/{REGISTRY_BRANCH}"],
        env=env, capture_output=True,
    )
    if r.returncode != 0:
        git("checkout", "-B", REGISTRY_BRANCH)

    git("add", REGISTRY_YAML)

    diff = subprocess.run(["git"] + repo + ["diff", "--cached", "--quiet"], env=env)
    if diff.returncode == 0:
        print("[push] no changes to registry — already up to date")
        return

    git("-c", "user.email=kaggle-bot@noreply.local",
        "-c", "user.name=Kaggle Upload Bot",
        "commit", "-m",
        f"[kaggle] update model_registry.yaml\n\nUploaded: {', '.join(HF_IDS)}")

    git_auth("push", "origin", REGISTRY_BRANCH)
    print(f"[push] pushed updated registry to branch '{REGISTRY_BRANCH}'")
    print(f"[push] to apply locally:\n"
          f"  git fetch origin {REGISTRY_BRANCH}\n"
          f"  git checkout origin/{REGISTRY_BRANCH} -- {REGISTRY_YAML}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_secrets()
    install_deps()
    clone_repo()

    registry = load_registry()
    errors = []

    print(f"\nUploading {len(HF_IDS)} model(s) to Kaggle …\n")
    for hf_id in HF_IDS:
        print(f"── {hf_id}")
        try:
            upload_model(hf_id, registry)
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            errors.append((hf_id, exc))
        print()

    import yaml
    print("=== Final model_registry.yaml ===")
    print(yaml.dump(registry, default_flow_style=False, sort_keys=False))
    print("=================================\n")

    push_registry(registry)

    if errors:
        print(f"\n{len(errors)} model(s) failed:", file=sys.stderr)
        for hf_id, exc in errors:
            print(f"  {hf_id}: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
