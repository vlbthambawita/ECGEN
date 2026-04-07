# TODO: Add `ecgen.f.upload_hf` package function

## Goal
Expose HuggingFace checkpoint uploading as a proper importable API at `ecgen.f.upload_hf()`
so users can call it from Python code (Jupyter, scripts, CI) without invoking a CLI script.

---

## Tasks

### 1. Create `src/ecgen/f/` sub-package
- [x] Create `src/ecgen/f/__init__.py` that exports `upload_hf`

### 2. Extract core logic into `src/ecgen/f/upload_hf.py`
- [x] Move `load_config()` → keep as internal helper
- [x] Move `resolve_path()` → resolves relative paths from CWD (not `__file__`)
- [x] Rewrite `upload_checkpoints()` as `upload_hf(config, dry_run=False)` — raises exceptions instead of calling `sys.exit()`
- [x] Add single-checkpoint convenience overload `upload_hf_single(local_path, repo_path, repo_id, ...)`
- [x] Define a `CheckpointEntry` TypedDict for type-safe config entries

### 3. Wire up `src/ecgen/f/__init__.py`
- [x] Import and re-export `upload_hf` and `upload_hf_single`

### 4. Expose at top-level `ecgen.f`
- [x] Add `f` to `src/ecgen/__init__.py` `__all__`

### 5. Add `huggingface_hub` as optional dependency in `pyproject.toml`
- [x] Added under `[project.optional-dependencies]` as `hf` extra: `huggingface_hub>=0.20`
- [x] Install with: `pip install ecgen[hf]`

### 6. Update `scripts/upload_checkpoints_to_hf.py`
- [x] Replaced duplicated logic with a call to `ecgen.f.upload_hf()` — script is now a thin CLI wrapper

### 7. Update docs
- [x] Updated `CLAUDE.md` with Python API and CLI usage examples

## Status: COMPLETE
