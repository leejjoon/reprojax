"""Run the README quick start from an installed package.

The quick start must work with nothing from the repository: no ``tests/``,
no ``tests/data/``, and no source tree on ``sys.path``. This script extracts
the fenced Python block from README.md and executes it in a fresh namespace,
so the README cannot drift away from something that runs.

Usage: check_quickstart.py [path-to-README.md]
"""

import re
import sys
import traceback
from pathlib import Path

DEFAULT_README = Path(__file__).resolve().parent.parent / "README.md"
BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)


def quick_start_source(readme):
    blocks = BLOCK.findall(readme.read_text())
    for block in blocks:
        if "reproject_interp(" in block:
            return block
    raise SystemExit("no Python block calling reproject_interp found in README")


def main(argv):
    readme = Path(argv[1]) if len(argv) > 1 else DEFAULT_README
    source = quick_start_source(readme)
    namespace = {"__name__": "__readme_quickstart__"}
    print(f"executing {len(source.splitlines())} lines from {readme.name}")
    try:
        exec(compile(source, str(readme), "exec"), namespace)
    except Exception:
        traceback.print_exc()
        return 1
    array = namespace["array"]
    footprint = namespace["footprint"]
    assert array.shape == footprint.shape, (array.shape, footprint.shape)
    valid = int(namespace["np"].isfinite(namespace["np"].asarray(array)).sum())
    print(f"quick start OK: {array.shape}, {valid} finite pixels")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
