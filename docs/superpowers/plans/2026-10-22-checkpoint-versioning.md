# Checkpoint Versioning Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save the project's `major.minor` version in checkpoints and refuse to load checkpoints with a different one (#46).

**Architecture:** `pyproject.toml` is the single version source. A small `src/utils/version.py` reads it through `importlib.metadata` and provides `compat_version()` and `check_checkpoint_version()`. `get_checkpoint()` stores the version, and both checkpoint load sites (`ReProSeg.__init__`, trainer resume) call the check. The legacy-field check from #40 is replaced.

**Tech Stack:** Python 3.10+, torch, pytest (`uv run pytest`).

Spec: `docs/superpowers/specs/2026-10-22-checkpoint-versioning-design.md`

---

## File Structure
- Create `src/utils/version.py`: version helpers (one responsibility: version parsing/compat check).
- Create `test/utils/test_version.py` (add `test/utils/__init__.py` only if sibling test dirs have one).
- Modify `src/train/trainer.py`: save `version`; replace legacy-field check (resume block ~41-48, `get_checkpoint` ~81).
- Modify `src/model/model.py:74`: check before `load_state_dict`.
- Modify `pyproject.toml`: bump to `0.2.0`.
- Modify `README.md`: short "Releases" section.

## Chunk 1: Version helpers

### Task 1: `version.py`

**Files:**
- Create: `src/utils/version.py`
- Test: `test/utils/test_version.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from utils.version import check_checkpoint_version, compat_version, major_minor


def test_major_minor_strips_patch():
    assert major_minor("0.2.5") == "0.2"
    assert major_minor("1.4") == "1.4"


def test_compat_version_ignores_patch(monkeypatch):
    monkeypatch.setattr("utils.version.get_version", lambda: "0.2.7")
    assert compat_version() == "0.2"


@pytest.fixture
def code_version(monkeypatch):
    monkeypatch.setattr("utils.version.get_version", lambda: "0.2.0")


def test_same_version_passes(code_version):
    check_checkpoint_version({"version": "0.2"}, "ckpt.pt")


def test_different_minor_raises(code_version):
    with pytest.raises(ValueError, match=r"0\.1.*0\.2"):
        check_checkpoint_version({"version": "0.1"}, "ckpt.pt")


def test_missing_version_raises(code_version):
    with pytest.raises(ValueError, match="no version"):
        check_checkpoint_version({}, "ckpt.pt")
```

A checkpoint saved by `0.2.5` stores `"0.2"`, so it passes under code `0.2.0`; this is the patch-compatibility case.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest test/utils/test_version.py -v`
Expected: FAIL (`ModuleNotFoundError: utils.version`)

- [ ] **Step 3: Implement**

```python
from collections.abc import Mapping
from importlib.metadata import version


def get_version() -> str:
    return version("reproseg")


def major_minor(full_version: str) -> str:
    return ".".join(full_version.split(".")[:2])


def compat_version() -> str:
    """Checkpoints are compatible when major.minor matches; patch releases never break them."""
    return major_minor(get_version())


def check_checkpoint_version(checkpoint: Mapping, path: object) -> None:
    saved = checkpoint.get("version")
    if saved is None:
        raise ValueError(
            f"Checkpoint {path} has no version (saved before versioning was introduced). "
            "Retrain, or check out the code that produced it."
        )
    current = compat_version()
    if saved != current:
        raise ValueError(
            f"Checkpoint {path} was saved by version {saved}.x but this code is {current}.x. "
            f"Retrain, or run `git checkout v{saved}.<patch>` to load it."
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest test/utils/test_version.py -v`
Expected: 5 passed. On `PackageNotFoundError`, run `uv sync`.

- [ ] **Step 5: Commit**

```bash
git add src/utils/version.py test/utils
git commit -m "Add checkpoint version helpers (#46)"
```

## Chunk 2: Wire into checkpoints

### Task 2: Save and check version

**Files:**
- Modify: `src/train/trainer.py`
- Modify: `src/model/model.py:74`

- [ ] **Step 1: Trainer resume.** Replace the `missing = [...]` / `if missing: raise ValueError(...)` block with `check_checkpoint_version(checkpoint, checkpoint_path)`, and add `from utils.version import check_checkpoint_version, compat_version` to the imports.
- [ ] **Step 2: Trainer save.** In `get_checkpoint(epoch)` add `checkpoint["version"] = compat_version()` after the `model_state_dict` line.
- [ ] **Step 3: Model load.** In `src/model/model.py`, right after the `torch.load(...)` at line 74, add `check_checkpoint_version(checkpoint, cfg.model.checkpoint)` (import from `utils.version`).
- [ ] **Step 4: Verify.** Run `uv run pytest -q`; expected: all pass (130 existing + 5 new). Then `git grep -n "missing" src/train/trainer.py` returns nothing.
- [ ] **Step 5: Commit**

```bash
git add src/train/trainer.py src/model/model.py
git commit -m "Save checkpoint version and check it on load (#46)"
```

## Chunk 3: Release

### Task 3: Bump and document

**Files:** `pyproject.toml`, `uv.lock` (if it records the version), `README.md`

- [ ] **Step 1:** Set `version = "0.2.0"` in `pyproject.toml`; run `uv sync` so installed metadata and the lock update.
- [ ] **Step 2:** Add a README "Releases" section: bump `pyproject.toml`; commit `Release X.Y.Z`; `git tag -a vX.Y.Z -m "X.Y.Z"`; `git push origin vX.Y.Z`; optional `gh release create vX.Y.Z --generate-notes`. State that checkpoints are compatible within the same `major.minor`.
- [ ] **Step 3:** `uv run pytest -q` passes.
- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock README.md
git commit -m "Bump version to 0.2.0 and document releases (#46)"
```

After the stack merges, tag `v0.2.0` on the merge commit (a manual user step, not part of this plan).
