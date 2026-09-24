# Plan: move `aind-dynamic-foraging-behavior-video-analysis` to Python 3.11

## Context

The library declares `requires-python = ">=3.9"` and `black target_version = ['py39']`, while the
README badge says `>=3.10` (they disagree). 3.9 is end of life. The goal is to end on
`requires-python = ">=3.11"` **without breaking any consumer at any step**.

The hazard is that the README promises "consuming capsules install it from `main` with no version
pin". The moment `main` says `>=3.11`, any consumer still on 3.9 fails its next image build with
pip's "requires a different Python" error.

### Consumer inventory (found on disk)

| Consumer | Python | How it installs this lib | Risk at floor bump |
|---|---|---|---|
| `kinematics_analysis` | **3.9** (CO jupyterlab py3.9 image) | `-e git+…@main`, unpinned | **Breaks.** Must move first |
| `aind-motion-energy-capsule` | 3.11 (`conda install python=3.11`) | pinned SHA `b21eac0` | None |
| `aind-BEAST-train-test` | 3.11 | pinned SHA `640cdca` | None |
| `tongue-tracking-metadata` | n/a | URL/name string in `processing.json` only | None |
| Batch pipeline / re-encoding capsules (Code Ocean, not on disk) | **unknown** | README says "from `main`, no pin" | **Unknown.** Must inventory |

The library's own imports also matter: `aind_dynamic_foraging_basic_analysis` and
`aind_dynamic_foraging_data_utils` (both installed from `@main`), pynwb, opencv, moviepy
(`moviepy.video.io.VideoFileClip`, which exists in 1.x and 2.x), pandas, numpy, scipy, and seaborn.
`dependencies = []` in `pyproject.toml`, so nothing is declared.

## Strategy: three stages, each shippable on its own

1. **Stage 1, widen:** prove the library works on 3.11 while still allowing 3.9.
2. **Stage 2, migrate consumers** onto 3.11.
3. **Stage 3, raise the floor** to `>=3.11`, with a tagged last-3.9 release as the escape hatch.

---

## To-do list

### Stage 0: inventory (read-only)
- [ ] List every Code Ocean capsule and pipeline that installs this library, including the batch
      tongue-kinematics pipeline and the video re-encoding capsule named in
      `VIDEO_CLIPS_MIGRATION_PLAN.md`. For each, record its Python version and whether it pins a SHA or tracks `@main`.
- [ ] Check `requires-python` on `main` for `aind-dynamic-foraging-basic-analysis` and
      `aind-dynamic-foraging-data-utils`. They are imported by `kinematics/tongue_analysis.py` and
      `kinematics/kinematics_nwb_utils.py`, so they must install on 3.11 too.
- [ ] Ask collaborators whether anyone runs this library from a personal or other-team 3.9
      environment that isn't in a repo.

### Stage 1: this repo, 3.11-ready but still `>=3.9` (one PR)
- [ ] Create a local 3.11 env (`pyenv install 3.11`; only 3.9.21 is installed now) and run
      `coverage run -m unittest discover` plus `flake8`. Fix anything that breaks.
- [ ] Add CI. `.github/workflows/` does not exist, although the README refers to
      `test_and_lint.yml`. Add a `test_and_lint.yml` that runs a matrix over **3.9 and 3.11** with
      unittest + flake8 + interrogate.
      Note: `fail_under = 100` coverage will probably fail. Decide whether to lower it or leave
      coverage out of the CI gate for now.
- [ ] Declare runtime `dependencies` in `pyproject.toml`: pandas, numpy, scipy, matplotlib,
      seaborn, pynwb, opencv-python, moviepy, python-dateutil, and requests (the last is used only
      lazily in `TransferToNWB.py`). The heavy or sibling ones could go in extras (for example
      `[nwb]` and `[video]`). This keeps `video_alignment` "pandas-only" for consumers like BEAST.
      Without declared deps, a 3.11 resolve can pull numpy 2 / pandas 3 unchecked.
- [ ] Check behavior with the newer stack a 3.11 resolve will pick (numpy 2.x, pandas 2.x/3.x):
      look for removed aliases (`np.NaN`, `np.float`), `fillna(method=)`, `DataFrame.append`, and
      `applymap`. A first grep found none, but run the tests to confirm.
- [ ] Fix the README badge (`>=3.10`) so it matches reality.

### Stage 2: migrate consumers
- [ ] **`kinematics_analysis`** (the blocking one), `environment/Dockerfile`:
  - [ ] Choose the base image. The Dockerfile is already hand-edited (not UI-managed), so edit
        `FROM` directly in CO or through the capsule's git repo.
        - **Preferred:** switch `FROM` to a CO JupyterLab starter image that ships Python 3.11.
          Copy the exact tag from CO's Environment UI starter picker. Unpin or rename the
          Ubuntu-20.04-specific apt packages (`build-essential`, `libgit2-dev`, `pandoc`,
          `pkg-config`, `python3-tk`; `libgl1-mesa-glx` → `libgl1` on 24.04).
        - **Fallback** if there's no 3.11 image: keep the py3.9 image and add
          `RUN conda install -y python=3.11 && conda clean -ya` after `FROM`. This pattern already
          works in `aind-motion-energy-capsule`. Re-verify JupyterLab still launches, since BEAST
          had to reinstall it.
        - Do this in a duplicated capsule or on a branch, not the live capsule.
  - [ ] Re-check every pin for 3.11 wheels: `spikeinterface[full]==0.100.0`,
        `scipy==1.13.0`, `wavpack-numcodecs==0.1.5`, `open-ephys-python-tools==0.1.7`,
        `pymupdf==1.24.2`, `ipywidgets==7.7.2` (JupyterLab 3.6 compatibility), `moviepy==1.0.3`,
        `aind-ephys-utils==0.0.15`, `pynwb==3.0.0`, and `hdmf-zarr==0.11.0`.
  - [ ] Replace the `python3-tk=3.8.10…` apt pin. It is the system Python's Tk, not conda's.
  - [ ] Drop the `--ignore-requires-python` workaround for `rachel-analysis-utils`, which needs
        ≥3.10.
  - [ ] Revisit `scanpy==1.10.3`. That pin exists only because of 3.9 (see the comment in the
        Dockerfile).
  - [ ] Update `kinematics_analysis/CLAUDE.md` lines ~20 and ~70 ("Python 3.9 compatible syntax
        only", `requires-python = ">=3.9"`).
  - [ ] Rebuild the image and rerun `run_batch_analysis.py` on 1–2 reference sessions. Diff
        outputs (`tongue_kins.parquet`, `tongue_movs.parquet`, `tongue_quality_stats.json`)
        against the 3.9 outputs. Numeric drift from newer numpy/scipy should be within
        tolerance, and the schemas should be identical.
- [ ] Move each unknown capsule from Stage 0 to 3.11, or pin it to a SHA/tag.
- [ ] Optional hygiene: move the `@main` installs in consumers to a tag. This is what makes
      future floor bumps safe by default.

### Stage 3: raise the floor (separate PR, only after Stage 2 is done)
- [ ] Tag the last 3.9-compatible commit (for example `v0.1.0` / `py39-final`, since only
      `v0.0.0` exists). Record it in the README so 3.9 holdouts can pin
      `@<tag>`.
- [ ] `pyproject.toml`: `requires-python = ">=3.11"`, `target_version = ['py311']`, and add a
      `Programming Language :: Python :: 3.11` classifier.
- [ ] CI matrix: drop 3.9. Optionally add 3.12 so the next upgrade is already covered.
- [ ] Update `VIDEO_CLIPS_MIGRATION_PLAN.md` ("works on Python 3.9").
- [ ] Optional follow-up PR (not in the same one): modernize syntax, for example
      `Optional[X]` → `X | None`. Keep it separate so the floor bump stays easy to revert.
- [ ] Announce to consumers: what changed, the 3.9 tag, and the date.

## Verification
- CI green on 3.9 + 3.11 (Stage 1), then on 3.11 only (Stage 3).
- `pip install git+…@main` succeeds in a fresh 3.11 env and fails cleanly on 3.9 after Stage 3.
- `kinematics_analysis` image builds on 3.11, and its batch outputs match the 3.9 baseline for
  reference sessions.
- Rebuild each unpinned capsule from Stage 0 after Stage 3 merges.

## Rollback
Stage 3 is a single-line revert of `requires-python`. Consumers can pin the 3.9 tag at any time.
