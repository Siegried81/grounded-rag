"""requirements.lock pins every package that requirements.txt asks for.

The Dockerfile and CI install the lock, so a package added to requirements.txt
but missing from the lock would pass review and then be absent at run time.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _names(path: Path) -> set[str]:
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "-")):
            names.add(re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0].lower().replace("_", "-"))
    return names


def test_lock_covers_every_requirement():
    missing = _names(ROOT / "requirements.txt") - _names(ROOT / "requirements.lock")
    assert not missing, f"not in requirements.lock: {sorted(missing)}"


def test_lock_pins_exact_versions():
    for line in (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith(("#", " ")):
            assert "==" in line, line
