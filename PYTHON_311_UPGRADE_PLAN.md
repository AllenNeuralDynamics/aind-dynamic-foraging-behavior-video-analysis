# Plan: move `aind-dynamic-foraging-behavior-video-analysis` to Python 3.11+

## Context

The library declares `requires-python = ">=3.9"` and `black target_version = ['py39']`, while the
README badge says `>=3.10` (they disagree). 3.9 is end of life. The goal is to end on
`requires-python = ">=3.11"` **without breaking any consumer at any step**.

- **Library minimum: 3.11.** The motion-energy and BEAST capsules run 3.11, so a higher minimum
  would lock them out.
- **`kinematics_analysis` runtime: 3.12**, using AIND's capsule template image
  (`codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04`) with its Python as-is.
- **CI covers 3.11 and 3.12**, so the library works on every consumer's Python.

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

1. **Stage 1, widen:** prove the library works on 3.11 and 3.12 while still allowing 3.9.
2. **Stage 2, migrate consumers:** `kinematics_analysis` moves to 3.12, and other capsules to 3.11
   or newer.
3. **Stage 3, raise the floor** to `>=3.11`, with a tagged last-3.9 release as the escape hatch.

---

## To-do list

### Status
| Date | Stage | Where | Notes |
|---|---|---|---|
| 2026-09-24 | Stage 1 done, in review | PR #5 (`build/python-311-support`) | CI passes on 3.9, 3.11 and 3.12 |
| 2026-09-24 | Stage 2a started | `kinematics_analysis` | Unpushed `kinematics-manuscript` commits backed up |

### Stage 0: inventory (read-only)
- [ ] List every Code Ocean capsule and pipeline that installs this library, including the batch
      tongue-kinematics pipeline and the video re-encoding capsule named in
      `VIDEO_CLIPS_MIGRATION_PLAN.md`. For each, record its Python version and whether it pins a SHA or tracks `@main`.
      (`lcephystonguemovements` was not found in this repo or `kinematics_analysis`. Locate it.)
- [x] Check `requires-python` on `main` for `aind-dynamic-foraging-basic-analysis` and
      `aind-dynamic-foraging-data-utils`. *2026-09-24: both are `>=3.9` and on PyPI (0.4.6 and
      0.1.56), and both install and import on 3.11 and 3.12.*
- [ ] Ask collaborators whether anyone runs this library from a personal or other-team 3.9
      environment that isn't in a repo.
- [ ] Optional: open a blank capsule and note the other base images AIND's CO offers, in case a
      better fit than the template exists.

### Stage 1: this repo, 3.11/3.12-ready but still `>=3.9` (one PR)
- [x] Create local 3.11 and 3.12 envs and run the tests plus `flake8`. *Done with
      `uv venv --python 3.X`. All tests pass on 3.9, 3.11 and 3.12, and no code fixes were needed.*
- [x] Add CI: `.github/workflows/test_and_lint.yml`, a matrix over 3.9, 3.11 and 3.12.
      *Decision: the blocking checks are the tests plus
      `flake8 --select=E9,F63,F7,F82` (syntax errors and undefined names). Coverage (14%),
      interrogate (74%) and full flake8 (916 issues) are reported, not enforced. They have
      never met the 100% thresholds in `pyproject.toml`.*
- [x] Declare runtime dependencies. *Core: `numpy`, `pandas` (enough for `video_alignment` and
      `tongue_lickometer_utils`). Everything else is in the `kinematics` extra: sibling AIND libs,
      scipy, matplotlib, seaborn, opencv-python, moviepy, pynwb, python-dateutil, and requests.*
- [x] Add `tests/test_imports.py`, which imports every module, since most modules have no tests.
- [x] Check behavior with the newer stack. *A 3.11/3.12 resolve picks numpy 2.4–2.5, pandas 3.0,
      pynwb 4.2, and moviepy 2.x (3.9 gets pandas 2.3 and numpy 2.0). A static scan for removed
      numpy/pandas APIs and pandas-3 copy-on-write hazards found nothing. **Tests cover only
      14% of the code**, so numerical equivalence is verified in Stage 2's reference-session diff,
      not here.*
- [x] Fix the README badge (now `>=3.9`) and install instructions (`.[kinematics]`).
- [x] CI green on GitHub for all three versions. *2026-09-24, PR #5: 3.9, 3.11 and 3.12 all
      pass, and `libgl1` is enough for opencv on the runner.*
- [ ] Merge the Stage 1 PR into `main`.
- [ ] Optional: fix the README coverage and interrogate badges, which claim 100%.

### Stage 2: migrate consumers

#### 2a. Safety checklist: before touching any Dockerfile
Everything below is reversible as long as these are done first. The only work that can be lost
is work that exists solely inside a capsule or workstation (uncommitted edits, scratch files,
unsaved `/results`).

- [x] Every local `kinematics_analysis` branch is on GitHub. *2026-09-24: `kinematics-manuscript`
      had 2 local-only commits (`e185087`, `c073d5d`); both are now pushed. No other branch has
      unpushed commits.*
- [ ] In Code Ocean, commit and push anything uncommitted in the `kinematics_analysis` capsule
      and any open workstation. Confirm on GitHub that each branch's tip matches.
- [ ] Save anything valuable in `/results` or workstation scratch as a Code Ocean **data
      asset** (data assets can't be modified, and a Dockerfile change can't touch them).
- [ ] Record the 3.9 baseline in the **original** capsule:
      `pip freeze > environment/py39-freeze.txt`, committed. This makes a rollback exact.
- [ ] Record reference outputs: run `run_batch_analysis.py` on 1–2 reference sessions on 3.9 and
      save `tongue_kins.parquet`, `tongue_movs.parquet`, and `tongue_quality_stats.json` as a data
      asset. This is the comparison target for the new environment.
- [ ] Create branch `env/py312` from `main` in `kinematics_analysis`.
- [ ] Duplicate the capsule in Code Ocean and point the duplicate at `env/py312`. Check that its
      git remote is the same GitHub repo and that its data assets are attached. **All Dockerfile
      work happens in the duplicate. The original capsule stays on 3.9 until 2c.**
- [ ] Note which Code Ocean capsule, if any, runs each other branch (`kinematics-manuscript`,
      `wild`, `local-dev`). Each of those capsules needs the change or a pin before Stage 3.

**Rollback at any point:** before 2c, delete the duplicate capsule and nothing else has
changed. After 2c, `git revert` the Dockerfile commit and rebuild, which gives the exact 3.9
environment recorded in `py39-freeze.txt`.

#### 2b. Migrate the environment in the duplicate capsule (`env/py312`)
- [ ] **`kinematics_analysis`** (the blocking one), `environment/Dockerfile`. The Dockerfile is
      already hand-edited (not UI-managed), so edit it directly in CO or through git.
      A scan of its 18 `.py` files and 53 notebooks found no 3.12 or numpy-2 blockers, so the
      risk is in the environment, not the code.
  - [ ] Change `FROM` to AIND's template image:
        `FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04`.
        Use its Python 3.12 as-is, with no conda/mamba Python swap.
  - [ ] Install JupyterLab explicitly (with `ipywidgets` at a compatible version). The mambaforge
        image has no IDE, and the hand-edited Dockerfile blocks CO's automatic IDE install. This is
        the same problem BEAST hit. Confirm the `postInstall` code-server setup still works.
  - [ ] Update the apt block for Ubuntu 22.04. Drop the 20.04 version pins (`build-essential`,
        `libgit2-dev`, `pandoc`, `pkg-config`). Remove `python3-tk=3.8.10…`, since that's the
        system Python's Tk and not mamba's (use `tk` from conda-forge if needed). Keep `ffmpeg` and
        `libgl1-mesa-glx`.
  - [ ] Re-check every pin for 3.12 wheels or support: `spikeinterface[full]==0.100.0`,
        `wavpack-numcodecs==0.1.5`, `moviepy==1.0.3`, `open-ephys-python-tools==0.1.7`,
        `aind-ephys-utils==0.0.15`, `scipy==1.13.0`, `pymupdf==1.24.2`, `pillow==10.3.0`,
        `pynwb==3.0.0`, `hdmf-zarr==0.11.0`, `zarr==2.18.2`, and `scikit-image==0.24.0`. Bump
        the minimum needed and record why next to each bump. Highest risk: `wavpack-numcodecs`,
        `moviepy`, and `aind-ephys-utils` (may have no 3.12 build), and `spikeinterface`
        0.100 (predates numpy 2; add a `numpy<2` pin if imports break).
        Iterate by switching the base image first, then fixing failing pins, then fixing failing
        imports.
        Fallback if 3.12 is painful: install 3.11 into the same template image with
        `mamba install python=3.11`.
  - [ ] Drop the `--ignore-requires-python` workaround for `rachel-analysis-utils`, which needs
        ≥3.10.
  - [ ] Revisit `scanpy==1.10.3`. That pin exists only because of 3.9 (see the comment in the
        Dockerfile).
  - [ ] Update `kinematics_analysis/CLAUDE.md` lines ~20 and ~70 ("Python 3.9 compatible syntax
        only", `requires-python = ">=3.9"`). Note that the library supports 3.11+, so shared code
        should avoid 3.12-only features.
  - [ ] Rebuild the image and rerun `run_batch_analysis.py` on 1–2 reference sessions. Diff
        outputs (`tongue_kins.parquet`, `tongue_movs.parquet`, `tongue_quality_stats.json`)
        against the 3.9 outputs. Numeric drift from newer numpy/scipy should be within
        tolerance, and the schemas should be identical.
#### 2c. Adopt it
- [ ] Merge `env/py312` into `main`, rebuild the original capsule once, and archive the
      duplicate.
- [ ] **Every active `kinematics_analysis` branch, not just `main`.** As of 2026-09-24, `main`,
      `kinematics-manuscript`, `local-dev` and `wild` each have their own Dockerfile on the py3.9
      image, installing this library from `@main`. For each branch, either merge `main` in after
      the Dockerfile migration, or pin the library to the last-3.9 tag in that branch's
      Dockerfile (suits branches frozen for a manuscript).
- [ ] Branches of *this* library (e.g. `LC_manuscript`) need nothing. They keep their own
      `requires-python` until they merge `main`, and 3.9 code runs fine on 3.11+.
- [ ] Move each unknown capsule from Stage 0 to 3.11 or newer (the AIND template is a good
      default), or pin it to a SHA/tag.
- [ ] Optional hygiene: move the `@main` installs in consumers to a tag. This is what makes
      future floor bumps safe by default.

### Stage 3: raise the floor (separate PR, only after Stage 2 is done)
- [ ] **Gate:** confirm every consumer *and every active branch of each consumer* is on 3.11+
      or pinned to a tag or SHA. Otherwise, rebuilding a stale branch breaks.
- [ ] Tag the last 3.9-compatible commit (for example `v0.1.0` / `py39-final`, since only
      `v0.0.0` exists). Record it in the README so 3.9 holdouts can pin
      `@<tag>`.
- [ ] `pyproject.toml`: `requires-python = ">=3.11"`, `target_version = ['py311']`, and add
      `Programming Language :: Python :: 3.11` / `3.12` classifiers.
- [ ] CI matrix: drop 3.9 and keep 3.11 + 3.12.
- [ ] Update `VIDEO_CLIPS_MIGRATION_PLAN.md` ("works on Python 3.9").
- [ ] Optional follow-up PR (not in the same one): modernize syntax, for example
      `Optional[X]` → `X | None`. Keep it separate so the floor bump stays easy to revert.
- [ ] Announce to consumers: what changed, the 3.9 tag, and the date.

## Verification
- CI green on 3.9 + 3.11 + 3.12 (Stage 1), then on 3.11 + 3.12 (Stage 3).
- `pip install git+…@main` succeeds in fresh 3.11 and 3.12 envs, and fails cleanly on 3.9 after
  Stage 3.
- The `kinematics_analysis` image builds on the AIND 3.12 template, JupyterLab launches, and its
  batch outputs match the 3.9 baseline for reference sessions.
- Rebuild each unpinned capsule from Stage 0 after Stage 3 merges.

## Rollback
Stage 3 is a single-line revert of `requires-python`. Consumers can pin the 3.9 tag at any time.
