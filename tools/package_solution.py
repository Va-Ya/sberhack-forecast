"""Create a portable handoff archive without training data, environments or unrelated projects."""
from __future__ import annotations

import json
import subprocess
import zipfile
from pathlib import Path

from forecast.data import sha256_file


def main():
    root = Path(__file__).resolve().parents[1]
    required = ["outputs/submission.csv", "model/config.json", "reports/training_summary.json", "README.md"]
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(f"Missing release requirement: {name}")
    destination = root / "release/sberhack-solution.zip"
    destination.parent.mkdir(exist_ok=True)
    files = ["README.md", "HANDOFF.md", "AGENTS.md", "CONTINUE_PROMPT.md", "Dockerfile", ".dockerignore", ".gitignore", ".gitattributes",
             "requirements.txt", "requirements-train.txt", "requirements-dev.txt", "constraints.txt",
             "pyproject.toml", "score.py"]
    for directory in ["forecast", "app", "tests", "tools", "docs", ".agents/skills", "model", "outputs", "reports", "data"]:
        for path in (root / directory).rglob("*"):
            relative = path.relative_to(root)
            if not path.is_file() or any(p in {"__pycache__", "cache", "smoke", "wheels", "wheels-linux", "predictions", "folds",
                                              "model-backup", "refit"} for p in relative.parts):
                continue
            if path.suffix in {".pyc", ".npz"}:
                continue
            files.append(relative.as_posix())
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(files)):
            archive.write(root / name, name)
    manifest = {"archive": destination.name, "sha256": sha256_file(destination),
                "bytes": destination.stat().st_size, "files": len(set(files)),
                "submission_sha256": sha256_file(root / "outputs/submission.csv"),
                "entries_sha256": {name: sha256_file(root / name) for name in sorted(set(files))}}
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    if commit.returncode == 0:
        manifest["source_commit"] = commit.stdout.strip()
    (destination.parent / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "entries_sha256"}), flush=True)


if __name__ == "__main__":
    main()
