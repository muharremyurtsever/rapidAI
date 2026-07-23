"""Download Phase 0 models. Run: .venv/bin/python tools/scripts/download_models.py [--skip-bf16] [--only NAME]"""
import argparse
import shutil

from huggingface_hub import snapshot_download

ROOT = "/Volumes/x9/rapidAI/models"
MODELS = {
    "olmoe-1b-7b-4bit": ("mlx-community/OLMoE-1B-7B-0125-Instruct-4bit", None),
    "qwen3-0.6b-4bit": ("mlx-community/Qwen3-0.6B-4bit", None),
    "qwen3-30b-a3b-4bit": ("mlx-community/Qwen3-30B-A3B-4bit", None),
    "qwen3-30b-a3b-bf16": ("Qwen/Qwen3-30B-A3B", ["*.safetensors", "*.json"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bf16", action="store_true")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    free_gb = shutil.disk_usage("/Volumes/x9").free / 1e9
    need = 25 if args.skip_bf16 else 90
    assert free_gb > need, f"Need {need} GB free on /Volumes/x9, have {free_gb:.0f}"
    for name, (repo, patterns) in MODELS.items():
        if args.only and name != args.only:
            continue
        if args.skip_bf16 and name.endswith("bf16"):
            continue
        print(f"→ {repo}")
        snapshot_download(repo, local_dir=f"{ROOT}/{name}", allow_patterns=patterns)
        print(f"✓ {name}")


if __name__ == "__main__":
    main()
