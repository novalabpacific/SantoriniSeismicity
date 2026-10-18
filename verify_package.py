"""Check release file integrity using SHA-256, without scientific dependencies.

Run from the package directory:
    python3 verify_package.py
    python3 verify_package.py --manifest checksums.sha256
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("MANIFEST.sha256"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    failures = []
    count = 0
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        parser.error(str(exc))
    for line_number, line in enumerate(lines, 1):
        if not line.strip() or line.startswith("#"):
            continue
        try:
            expected, relative = line.split("  ", 1)
            path = PurePosixPath(relative)
            if (len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected)
                    or path.is_absolute() or ".." in path.parts or "\\" in relative):
                raise ValueError("invalid checksum entry")
        except ValueError as exc:
            failures.append(f"Line {line_number}: {exc}")
            continue
        count += 1
        target = root.joinpath(*path.parts)
        try:
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
        except OSError as exc:
            failures.append(f"{relative}: {exc}")
            continue
        if actual != expected:
            failures.append(f"{relative}: SHA-256 mismatch")
    if count == 0:
        failures.append("No checksum entries were found")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print(f"Verified {count} files against {manifest.name}")


if __name__ == "__main__":
    main()
