"""config.py reads the repo's own .env and never one from a parent folder.

A bare load_dotenv() walks up the tree, so a worktree with no .env of its own
read the parent folder's file (another project's keys). The test copies
config.py into a throwaway folder and imports it in a fresh interpreter, so the
real environment and the real .env of this checkout stay out of it.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import config

PROBE = "import config, os; print(os.environ.get('GR_DOTENV_PROBE', 'unset'))"


def _run(folder: Path) -> str:
    env = {k: v for k, v in __import__("os").environ.items() if k != "GR_DOTENV_PROBE"}
    out = subprocess.run(
        [sys.executable, "-c", PROBE], cwd=folder, env=env,
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def test_parent_dotenv_is_not_read(tmp_path):
    child = tmp_path / "child"
    child.mkdir()
    shutil.copy(Path(config.__file__), child / "config.py")
    (tmp_path / ".env").write_text("GR_DOTENV_PROBE=from-parent\n")
    assert _run(child) == "unset"


def test_own_dotenv_is_read(tmp_path):
    child = tmp_path / "child"
    child.mkdir()
    shutil.copy(Path(config.__file__), child / "config.py")
    (child / ".env").write_text("GR_DOTENV_PROBE=from-own\n")
    assert _run(child) == "from-own"
