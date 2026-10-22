# Project versioning and checkpoint version check (#46)

## Goal
Fail clearly when resuming a checkpoint produced by an incompatible release. Breaking old checkpoints is accepted.

## Design
- **Version source:** `pyproject.toml` `version` is the single source of truth. `src/utils/version.py` reads it with `importlib.metadata` and exposes `get_version()` and `compat_version()` (`"x.y"`).
- **Compatibility rule:** checkpoints are compatible when `x.y` matches; patch (`z`) differences are compatible. Under `0.y.z`, a minor bump is the breaking one.
- **Saving:** `get_checkpoint()` adds `version: "x.y"`.
- **Loading:** `check_checkpoint_version(ckpt, path)` runs before any state is loaded. A missing or different `x.y` raises an error naming both versions and suggesting `git checkout vX.Y.Z` or retraining.
- **Replaces** the legacy-field check from #40; pre-versioning checkpoints fail with this same error.

## Release process (manual)
1. Bump `version` in `pyproject.toml`; commit `Release X.Y.Z`.
2. `git tag -a vX.Y.Z -m "X.Y.Z"` and `git push origin vX.Y.Z`.
3. Optionally `gh release create vX.Y.Z --generate-notes`, noting breaking changes.

The stack #39/#43/#45 is released as `0.2.0`.

## Tests
Same version passes; patch difference passes; minor difference raises; missing key raises.

## Delivery
Branch off the top of PR #45.
