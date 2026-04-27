"""Kaggle kernel body for one (experiment, size) run of NegotiationArena.

Substituted by kaggle/render_kernel.py:
    {{EXPERIMENT}}    experiment name from configs/experiments.yaml
    {{SIZE}}          model_group ("none" if the experiment defines its own list)
    {{GIT_REF}}       commit hash to check out
    {{GIT_REPO}}      HTTPS git URL
    {{KAGGLE_GPU_TYPE}} requested Kaggle accelerator

Bootstraps the env, runs runner/run_experiment.py exactly as the SLURM path does
(transformers downloads weights from HF Hub on first use), then tars the
experiments/ tree into /kaggle/working/results.tar.gz.
"""
import os
import subprocess
import sys
import base64
import re

REPO_DIR = "/kaggle/working/repo"
HF_HOME = "/kaggle/working/hf_cache"
RESULT_TAR = "/kaggle/working/results.tar.gz"


def validate_runtime() -> None:
    """Fail fast when Kaggle assigns an incompatible GPU/runtime."""
    try:
        import torch
    except Exception as exc:
        print(f"[bootstrap] warning: unable to import torch for runtime check: {exc}")
        return

    if not torch.cuda.is_available():
        print("[bootstrap] CUDA unavailable; proceeding on CPU")
        return

    gpu_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"[bootstrap] GPU: {gpu_name} | compute capability sm_{capability[0]}{capability[1]}")

    # The Kaggle image currently installs a PyTorch build that requires sm_70+.
    if capability[0] < 7:
        requested = "{{KAGGLE_GPU_TYPE}}"
        raise RuntimeError(
            f"Incompatible Kaggle GPU assigned: {gpu_name} (sm_{capability[0]}{capability[1]}). "
            f"This run requires a T4/L4-class GPU because the installed PyTorch build "
            f"supports sm_70+. Re-submit with KAGGLE_GPU_TYPE='{requested}' and ensure "
            f"`kaggle kernels push` passes `--accelerator`."
        )

def export_secrets() -> None:
    """Inject API keys via local template substitution."""
    os.environ["GITHUB_TOKEN"] = "{{GITHUB_TOKEN}}"
    os.environ["HF_TOKEN"] = "{{HF_TOKEN}}"
    print("[bootstrap] secrets injected via template")

def clone_repo() -> None:
    """Clone the repo, using GITHUB_TOKEN for private repos when present.

    GitHub git-over-HTTPS expects Basic auth semantics for PATs. Passing the
    credential via `git -c http.extraHeader=...` avoids persisting secrets to
    `.git/config` after the clone (unlike embedding them in the URL).
    """
    if os.path.isdir(REPO_DIR):
        return
    cmd = ["git"]
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    repo_url = "{{GIT_REPO}}"
    if token:
        basic_auth = base64.b64encode(f"x-access-token:{token}".encode("utf-8")).decode("ascii")
        cmd += [
            "-c",
            f"http.extraHeader=Authorization: Basic {basic_auth}",
            "-c",
            "credential.helper=",
        ]
    elif "github.com" in repo_url:
        # Surface this before git emits "No such device or address" for stdin auth.
        raise RuntimeError(
            "GITHUB_TOKEN not in env. The repo is private; attach the GITHUB_TOKEN "
            "secret to this kernel via Add-ons → Secrets, or enable 'auto-attach to "
            "new notebooks' on the secret in your Kaggle account settings."
        )
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    cmd += ["clone", repo_url, REPO_DIR]
    subprocess.run(cmd, check=True, env=env)
    subprocess.run(["git", "-C", REPO_DIR, "checkout", "{{GIT_REF}}"], check=True, env=env)


def install_deps() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "-r", f"{REPO_DIR}/requirements.txt",
         "transformers", "accelerate", "bitsandbytes",
         "sentencepiece", "protobuf", "pyyaml"],
        check=True,
    )


def run_experiment() -> None:
    os.environ["HF_HOME"] = HF_HOME
    os.makedirs(HF_HOME, exist_ok=True)
    cmd = [
        sys.executable, "runner/run_experiment.py",
        "--config", "configs/experiments.yaml",
        "--experiment", "{{EXPERIMENT}}",
    ]
    if "{{SIZE}}" not in ("", "none"):
        cmd += ["--model_group", "{{SIZE}}"]
    print(f"[bootstrap] running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_DIR, check=True)


def archive_results() -> None:
    subprocess.run(
        ["tar", "-czf", RESULT_TAR, "-C", REPO_DIR, "experiments"],
        check=True,
    )
    size_mb = os.path.getsize(RESULT_TAR) / (1024 * 1024)
    print(f"[bootstrap] wrote {RESULT_TAR} ({size_mb:.1f} MB)")


def main() -> None:
    export_secrets()
    clone_repo()
    install_deps()
    validate_runtime()
    run_experiment()
    archive_results()


if __name__ == "__main__":
    main()
