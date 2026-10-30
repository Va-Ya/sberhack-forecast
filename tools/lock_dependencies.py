"""Pin the installed runtime dependency closure without unrelated global packages."""
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement


def main():
    root = Path(__file__).resolve().parents[1]
    queue = [Requirement(line).name for line in (root / "requirements.txt").read_text().splitlines()
             if line and not line.startswith("#")]
    found = {}
    while queue:
        name = queue.pop()
        if name.lower().replace("_", "-") in found:
            continue
        distribution = metadata.distribution(name)
        canonical = distribution.metadata["Name"].lower().replace("_", "-")
        found[canonical] = distribution.version
        for text in distribution.requires or []:
            requirement = Requirement(text)
            if requirement.marker is None or requirement.marker.evaluate():
                queue.append(requirement.name)
    lines = [f"{name}=={version}" for name, version in sorted(found.items())]
    (root / "constraints.txt").write_text("# Runtime dependency closure from the verified environment\n" +
                                          "\n".join(lines) + "\n", encoding="utf-8")
    print(f"Pinned {len(lines)} runtime packages", flush=True)


if __name__ == "__main__":
    main()
