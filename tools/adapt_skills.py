"""Keep exact upstream references and expose concise Codex/project-native skill entrypoints."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

SOURCE = "c8dfa617a2ef1523c713ba79e88189d93b0f1173"
REPO = "https://github.com/Shelpess/komanda"
INSTRUCTIONS = {
    "mle-workflow": "Use for reproducible forecasting, leakage checks, temporal validation and shared training/serving code.\n"
    "Define the data/metric contract first. Compare temporal baselines and save all five official metrics.\n"
    "Fit category metadata as of each origin. Record seeds, dependency versions, data hashes and model config.\n"
    "Use native model artifacts, keep the previous working bundle, and update HANDOFF.md after milestones.\n"
    "Consult references/upstream.md for broader ML guidance only when relevant; no feature store or online rollout is needed.",
    "python-patterns": "Use for implementing or reviewing Python.\n"
    "Prefer explicit functions, dataclasses, type annotations, pathlib and context managers.\n"
    "Handle specific failures, retain exception causes, stream large inputs and use NumPy/pandas vectorization.\n"
    "Keep forecast transformations shared with app. Check with ruff and the project's pytest suite.\n"
    "See references/upstream.md for examples; install no unrelated tooling from its examples.",
    "python-testing": "Use for meaningful Python contract and regression tests with pytest.\n"
    "Cover aggregation, cutoffs, zero-safe metrics, deterministic predictions, input errors and CLI/API parity.\n"
    "Use fixtures and parametrization; use tiny synthetic data for ordinary tests and a separate real-data smoke run.\n"
    "Prefer a failing regression before a behavioral fix. Test observable outcomes, not mirrored implementation.\n"
    "Run python -m pytest --basetemp .pytest-tmp. See references/upstream.md for optional testing techniques.",
    "docker-patterns": "Use for packaging and verifying the forecast service.\n"
    "Use a pinned Python slim image, pinned packages, libgomp1 for LightGBM and a non-root runtime user.\n"
    "Copy only forecast, app and the model bundle; exclude training data and archives.\n"
    "Use one uvicorn worker, health checks, bounded CPU threads and no inference-time downloads.\n"
    "Verify with a network-disabled container and input files provided externally.\n"
    "See references/upstream.md for optional container patterns; this project needs no database or Redis.",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, help="Optional local Komanda checkout containing the pinned commit")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = root / ".agents/skills"
    for name, body in INSTRUCTIONS.items():
        folder = target / name
        reference = folder / "references/upstream.md"
        reference.parent.mkdir(exist_ok=True)
        if not reference.exists():
            reference.write_bytes((folder / "SKILL.md").read_bytes())
        (folder / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {body.splitlines()[0]}\nlicense: MIT\n---\n\n"
                                         f"# {name}\n\n{body}\n\nSource: {REPO}/tree/{SOURCE}/resources/ecc/upstream/skills/{name}\n",
                                         encoding="utf-8")
    for name in ["LICENSE", "NOTICE.txt"]:
        destination = target / f"ECC-{name}"
        if args.checkout:
            result = subprocess.run(["git", "-C", str(args.checkout), "show", f"{SOURCE}:resources/ecc/{name}"],
                                    check=True, capture_output=True)
            destination.write_bytes(result.stdout)
        elif not destination.exists():
            raise FileNotFoundError(f"Missing {destination.name}; supply --checkout with the pinned Komanda source")
    (target / "provenance.json").write_text(json.dumps({"repository": REPO, "commit": SOURCE,
                                                        "skills": list(INSTRUCTIONS)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
