"""Every file CI checks must exist.

The refactor deleted frontend/script.js and llm-integration.js while the
workflow still ran `node --check` on both, so CI failed on a clean tree with
MODULE_NOT_FOUND. Paths in the workflow drift from the tree silently; this
notices.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


@pytest.fixture(scope="module")
def workflow_text():
    if not WORKFLOW.is_file():
        pytest.skip("no CI workflow")
    return WORKFLOW.read_text(encoding="utf-8")


def test_checked_javascript_exists(workflow_text):
    missing = [
        target for target in re.findall(r"node --check (\S+)", workflow_text)
        if not (ROOT / target).is_file()
    ]
    assert not missing, f"CI checks files that are not in the tree: {missing}"


def test_compiled_and_linted_paths_exist(workflow_text):
    targets = set()
    for pattern in (r"python -m compileall[^\n]*", r"python -m pyflakes[^\n]*"):
        for line in re.findall(pattern, workflow_text):
            for token in line.split()[3:]:
                if token.startswith("-") or "*" in token:
                    continue
                targets.add(token)
    missing = sorted(t for t in targets if not (ROOT / t).exists())
    assert not missing, f"CI targets paths that do not exist: {missing}"


def test_scripts_reference_files_that_exist():
    """setup.sh told people to run a driver the refactor removed."""
    missing = []
    for script in (ROOT / "scripts").glob("*.sh"):
        for match in re.findall(r"python3? (\S+\.py)", script.read_text(encoding="utf-8")):
            if not (ROOT / match).is_file():
                missing.append(f"{script.name} -> {match}")
    assert not missing, missing
