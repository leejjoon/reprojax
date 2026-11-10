"""Assert the built distributions contain what the release expects.

Both halves of the plan's packaging check:

- the **sdist** carries the package and its metadata only -- no test suite,
  no FITS fixtures. A test suite without its fixtures cannot even be
  collected, which is worse than shipping none;
- the **wheel** carries no test or FITS file and does include ``py.typed``.

Usage: check_dist_contents.py <dist-dir>
"""

import sys
import tarfile
import zipfile
from pathlib import Path

# Anything matching these in the sdist means MANIFEST.in or the build backend
# started pulling the repository in again.
FORBIDDEN_SDIST = ("tests/", "benchmarks/", "examples/", "docs/")

# The sdist must actually contain the package and the files a consumer needs.
REQUIRED_SDIST = (
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "src/reprojax/__init__.py",
    "src/reprojax/interp.py",
    "src/reprojax/adaptive.py",
    "src/reprojax/transform.py",
    "src/reprojax/core.py",
    "src/reprojax/_astropy_wcs.py",
    "src/reprojax/py.typed",
)

REQUIRED_WHEEL = (
    "reprojax/__init__.py",
    "reprojax/interp.py",
    "reprojax/adaptive.py",
    "reprojax/transform.py",
    "reprojax/core.py",
    "reprojax/_astropy_wcs.py",
    "reprojax/py.typed",
)


def check_sdist(path):
    with tarfile.open(path) as tf:
        names = tf.getnames()
    # Strip the single top-level "name-version/" prefix.
    inner = sorted({n.split("/", 1)[1] for n in names if "/" in n})

    problems = []
    for entry in inner:
        if any(entry.startswith(prefix) for prefix in FORBIDDEN_SDIST):
            problems.append(f"sdist contains {entry!r}")
        if entry.endswith(".fits"):
            problems.append(f"sdist contains FITS data {entry!r}")
    for required in REQUIRED_SDIST:
        if required not in inner:
            problems.append(f"sdist is missing {required!r}")
    return problems, len(inner)


def check_wheel(path):
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()

    problems = []
    for name in names:
        if name.startswith("tests/") or "/tests/" in name:
            problems.append(f"wheel contains a test file {name!r}")
        if name.endswith(".fits"):
            problems.append(f"wheel contains FITS data {name!r}")
    for required in REQUIRED_WHEEL:
        if required not in names:
            problems.append(f"wheel is missing {required!r}")
    return problems, len(names)


def main(argv):
    if len(argv) != 2:
        print("usage: check_dist_contents.py <dist-dir>", file=sys.stderr)
        return 2

    dist = Path(argv[1])
    sdists = sorted(dist.glob("*.tar.gz"))
    wheels = sorted(dist.glob("*.whl"))
    if len(sdists) != 1 or len(wheels) != 1:
        print(
            f"expected exactly one sdist and one wheel in {dist}, "
            f"found {[p.name for p in sdists + wheels]}",
            file=sys.stderr,
        )
        return 1

    problems = []
    for path, checker in ((sdists[0], check_sdist), (wheels[0], check_wheel)):
        found, count = checker(path)
        print(f"{path.name}: {count} entries")
        problems.extend(found)

    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print("distribution contents OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
