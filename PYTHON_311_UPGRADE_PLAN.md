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
| 2026-09-24 | Stage 2a done; 2b drafted | `kinematics_analysis` `env/py312` @ `2510b5e` | Duplicate capsule on `env/py312`; 3.12 Dockerfile + `py39-constraints.txt` pushed. Next: build it on CO |
| 2026-09-25 | 2b: build 1 failed (PyYAML 6.0), fixed | `env/py312` @ `d9ffba3` | Wheel-less compiled deps bumped; AIND libs pinned to baseline. Next: rebuild |
| 2026-09-25 | 2b: build 2 succeeded | duplicate capsule | VS Code launches; reference comparison running |
| 2026-09-25 | 2b: reference comparison passed | duplicate capsule | 3.12 outputs bit-for-bit identical to 3.9 on 2 sessions. Next: spot-check notebooks, then adopt (2c) |
| 2026-09-25 | 2b: spot-checks in progress | duplicate capsule | `eph_09` hit the empty-scratch issue; using the saved scratch asset. Scratch-to-data-asset work logged in `kinematics_analysis` TODO, deferred until after adoption |
| 2026-09-25 | 2b: most spot-checks pass | duplicate capsule | All modules import; eph_01, kin_02, fip_01 run; pymongo C ext OK. Pending: eph_09, kin_03, kin_07 |
| 2026-09-25 | 2b: kernel-hang fix | `env/py312` @ `1024923` | eph_09, kin_03 pass. debugpy 1.6.6 -> 1.8.20, ipykernel 6.29.5. Next: rebuild, rerun kin_07 |
| 2026-09-25 | **2b done**; 2c prepared | `env/py312` @ `e71b64e` | kin_07 passes, all spot-checks done. Adoption changes committed. Next: merge into `wild`, rebuild original capsule |
| 2026-09-25 | 2c: merged into `wild` | `kinematics_analysis` `wild` @ `0a3c9ce` | Next: rebuild original capsule, archive duplicate, then other branches |
| 2026-09-25 | **2c: original capsule on Python 3.12** | `wild` @ `6e36ef2` | Rebuilt; eph_01 runs. Next: archive duplicate; migrate or pin `main`, `kinematics-manuscript`, `local-dev` |
| 2026-09-25 | **2c: branches done** | `kinematics_analysis` | manuscript on 3.12; local-dev retired (`archive/local-dev`); main promoted to wild (`8adbe51`), old main at `archive/main-py39` (pinned, 3.9) |
| 2026-09-25 | `kinematics_analysis` CLAUDE.md updated | `wild` @ `7715fe6` | Branch roles: `wild` = dev with Claude, `main` = verified code (promote by merging `wild`). 3.12 syntax allowed. **Stage 2 done for `kinematics_analysis`.** Remaining gate for Stage 3: the Stage 0 capsules |
| 2026-09-25 | Stage 1 merged; Stage 0 closed; `env/py312` retired | library `main` @ `d94854f`; tag `archive/env-py312` | Only Stage 3 gate left: the video re-encoding capsule (user) |

### Stage 0: inventory (read-only)
- [x] List every Code Ocean capsule and pipeline that installs this library. *2026-09-25:*
      - *The "batch pipeline" in the README **is** `kinematics_analysis`'s Reproducible Run
        (`code/run` → `run_batch_analysis.py` → `run_batch_analysis`). It's on 3.12 and was
        validated bit-for-bit (Stage 2b).*
      - *The video re-encoding capsule is the user's own. It runs **Python 3.10.9**
        (`c1-vscode:4.20.0` base; its other AIND deps are PyPI-pinned) and installs this library
        from `@main`, so **Stage 3 would break its next build.** Fix: pin that one line to
        `@d94854f4f8072bd76823855c6b29ec6a452a2c4a` (library `main` after Stage 1), then rebuild.
        Follow-up: to take any post-Stage-3 library version (notably `video_clips.py` from
        `VIDEO_CLIPS_MIGRATION_PLAN.md`, which targets this capsule), first move it to 3.11+,
        e.g. the AIND template as in Stage 2.*
      - *No other capsules use the library.*
- [x] Ask collaborators whether anyone runs this library from a personal or other-team 3.9
      environment that isn't in a repo. *2026-09-25: none known (user).*
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
- [x] Declare runtime dependencies (`pyarrow` added 2026-09-24: `run_batch_analysis` writes
      parquet). *Core: `numpy`, `pandas` (enough for `video_alignment` and
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
- [x] Merge the Stage 1 PR into `main`. *2026-09-25: PR #5 merged (`d94854f`).*
- [ ] Optional: fix the README coverage and interrogate badges, which claim 100%.

### Stage 2: migrate consumers

#### 2a. Safety checklist: before touching any Dockerfile
Everything below is reversible as long as these are done first. The only work that can be lost
is work that exists solely inside a capsule or workstation (uncommitted edits, scratch files,
unsaved `/results`).

- [x] Every local `kinematics_analysis` branch is on GitHub. *2026-09-24: `kinematics-manuscript`
      had 2 local-only commits (`e185087`, `c073d5d`); both are now pushed. No other branch has
      unpushed commits.*
- [x] In Code Ocean, commit and push anything uncommitted in the `kinematics_analysis` capsule
      and any open workstation. *2026-09-24: done (user).*
- [x] Save anything valuable in `/results` or workstation scratch as a Code Ocean **data
      asset** (data assets can't be modified, and a Dockerfile change can't touch them).
      *2026-09-24: scratch saved (user).*
- [x] Record the 3.9 baseline in the **original** capsule, committed. *2026-09-24: `wild` @
      `7daae78`: `environment/py39-freeze.txt` (227 lines), `py39-conda.txt` (`conda list
      --export`, 250 lines; needed because `@ file:///tmp/build/…` lines are conda-installed and
      can't be reinstalled by pip), `py39-python-version.txt` (3.9.12). The baseline already has
      numpy 2.0.2 working with spikeinterface 0.100.0 and numba 0.60.0, and this library at
      `de27558`.*
      Later, compare environments with `diff <(sort environment/py39-freeze.txt) <(pip freeze | sort)`.
- [x] Record reference outputs: run `code/env_00_reference_sessions.ipynb` (`wild` @ `6af4d2f`)
      in the original 3.9 capsule with `BASELINE_DIR = None`, then save
      `scratch/env_reference/py39/` as a data asset. This is the comparison target. In 2b, the
      same notebook, with `BASELINE_DIR` pointing at that asset, reruns the same sessions and
      compares every output file. *2026-09-24: done, saved as data asset
      **`env_reference_py39`**.*
- [x] Create branch `env/py312` from **`wild`** in `kinematics_analysis`. *2026-09-24: created
      at `6af4d2f` and pushed.* `wild` is the
      working branch (206 commits ahead of `main` on 2026-09-24) and holds the baseline files.
- [x] Duplicate the capsule in Code Ocean and point the duplicate at `env/py312`. Check that its
      git remote is the same GitHub repo and that its data assets are attached. **All Dockerfile
      work happens in the duplicate. The original capsule stays on 3.9 until 2c.**
      *2026-09-24: done (user). The duplicate is linked to GitHub, on `env/py312`.*
- [x] Note which Code Ocean capsule, if any, runs each other branch (`kinematics-manuscript`,
      `wild`, `local-dev`). Each of those capsules needs the change or a pin before Stage 3.
      *2026-09-25: resolved by the branch decisions in 2c: manuscript on 3.12, local-dev retired, main promoted.*

**Rollback at any point:** before 2c, delete the duplicate capsule and nothing else has
changed. After 2c, `git revert` the Dockerfile commit and rebuild, which gives the exact 3.9
environment recorded in `py39-freeze.txt`.

#### 2b. Migrate the environment in the duplicate capsule (`env/py312`)
Approach (2026-09-24): **change only Python, the OS and the Jupyter tooling.** A new
`environment/py39-constraints.txt`, generated from `py39-freeze.txt`, holds every package at its
3.9-baseline version through `pip install -c`. All 187 constrained versions resolved for Linux +
Python 3.12 (checked with `uv pip compile --python-platform x86_64-manylinux_2_35`),
including numpy 2.0.2, pandas 2.3.3, scipy 1.13.0, spikeinterface 0.100.0, numcodecs 0.12.1 and
matplotlib 3.9.4. So any DIFF in the reference comparison is caused by Python 3.12 itself.
pandas 3 and other upgrades become separate, deliberate steps: edit the constraints file.

- [x] **`kinematics_analysis` `environment/Dockerfile`**, drafted on `env/py312` @ `2510b5e`.
      A scan of its 18 `.py` files and 53 notebooks found no 3.12 or numpy-2 blockers.
  - [x] `FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04`, using its
        Python as-is.
  - [x] Install `jupyterlab` and `ipywidgets` explicitly. They resolve to 4.1.6 and 8.0.7; no code
        uses ipywidgets directly.
  - [x] apt block: Ubuntu 20.04 version pins dropped, `python3-tk` removed (conda-forge Python
        ships Tk), and `git`, `curl`, `ca-certificates` added (the minimal base lacks them; `curl`
        is needed by `postInstall`).
  - [x] Pins checked for Linux 3.12: all resolve unchanged. `moviepy==1.0.3` and
        `open-ephys-python-tools==0.1.7` are source-only but pure Python. **`wavpack-numcodecs`
        has never shipped wheels (every release is source-only)**, so it compiles on Linux, as it
        already does on 3.9. It couldn't be test-built here (macOS, no Docker). It's the step most
        likely to fail. If it does, check the build log for a missing system library.
  - [x] `--ignore-requires-python` for `rachel-analysis-utils` removed.
  - [x] `scanpy`: held at 1.10.3 by the constraints file for the migration. It can be bumped
        later.
- [x] **Build the duplicate capsule's environment** on Code Ocean. *2026-09-25: build 2
      (`d9ffba3`) succeeded, and VS Code (code-server) launches. JupyterLab launch not yet
      confirmed.*
      *Build 1 (2026-09-25) failed: `PyYAML==6.0` (baseline) has no 3.12 wheel, and its source
      build breaks under Cython 3. Lesson: "resolves" isn't the same as "has a wheel". Every
      resolved package was then checked for a Linux 3.12 wheel. Fixed on `env/py312` @
      `d9ffba3`: PyYAML 6.0.1, pyzmq 25.1.1, MarkupSafe 2.1.3 (smallest bumps with wheels).
      pymongo stays 4.3.3, pinned exactly by `aind-data-access-api`; its C extensions are
      optional. Source-built on 3.12: asciitree, moviepy, open-ephys-python-tools, zmq (pure
      Python), pymongo, wavpack-numcodecs. The AIND libs are also pinned to their baseline SHAs,
      and `aind-dynamic-foraging-models` to 0.16.0, so the comparison isolates Python.* Check that JupyterLab and
      the `postInstall` code-server setup launch.
      Fallback if 3.12 turns out to be painful: install 3.11 into the same template image
      (`RUN mamba install -y python=3.11` after `FROM`). The constraints file stays the same.
- [x] Run `code/env_00_reference_sessions.ipynb` in the duplicate, with the `env_reference_py39`
      asset attached and `BASELINE_DIR` pointing at its `py39` folder. Review any DIFF rows.
      *2026-09-25: **bit-for-bit identical.** 2 sessions (`behavior_716325_2024-05-31_10-31-14`,
      `behavior_717259_2024-06-28_11-17-19`), 17 parquet files each, plus
      `tongue_quality_stats.json`: no DIFF, no "close" rows.*
- [x] Save `scratch/env_reference/py312/` (including `comparison_vs_baseline.csv`) as a data
      asset, as the record. *2026-09-25: saved (user).*
- [x] Spot-check notebooks the batch pipeline doesn't exercise, in the duplicate:
      `eph_09_structural_axes` (scanpy + MERFISH), an ephys notebook that loads recordings
      (spikeinterface + `wavpack-numcodecs`), one `kin_` and one `fip_`. Pass = runs without
      import errors or crashes.
      *2026-09-25: `eph_09` first failed on `SCRATCH / "combined_unit_tbl.pkl"`. Not a 3.12
      problem: **a duplicated capsule doesn't get the original's `/scratch`.** Fix for the
      spot-check: save the old capsule's scratch as a data asset, attach it, and symlink or copy
      its files into the duplicate's scratch. Notebooks stay unchanged and read byte-identical
      inputs. Don't substitute a different file (e.g. the upstream table under
      `LC-NE_scratch_data_1/combined/combine_unit_tbl/`, which eph_08/09 use only for CCF
      coordinates).*
      *Results 2026-09-25 (duplicate, Python 3.12.4):*
      - [x] *every module in `code/*.py` imports (except `run_*` / `backup_*`, which execute on
        import)*
      - [x] *`eph_01`, `kin_02`, `fip_01` run end to end*
      - [x] *`pymongo.has_c()` is `True`: the source-built C extensions compiled*
      - [x] *`eph_09` (scanpy, trimesh; long-running)*
      - [x] *`kin_03_umap` (numba / llvmlite)*
      - [x] *`kin_07_value_encoding` (live docDB query through
        `aind_analysis_arch_result_access` → pymongo). Passed after the kernel fix below; notebooks
        now start cleanly with "Run All".*
      - *Kernel hang when first running a notebook from VS Code (eph_01, kin_07), at the
        first code cell; imports were fine in a terminal. Cause: `debugpy` was still 1.6.6
        (baseline), which predates 3.12 support; it installed via a generic py2.py3 wheel, so the
        wheel check passed it. Fixed on `env/py312` @ `1024923`: debugpy 1.8.20, ipykernel held at
        6.29.5 (had floated to 7.1). **Lesson: "has a wheel" should mean a compiled cp312 wheel
        when the package ships compiled wheels at all.** A scan for that pattern found only
        pymongo (verified) and pyrsistent (Jupyter-only, pure fallback). Needs a rebuild, then
        confirm notebooks start cleanly with "Run All".*
      - *Not needed: spikeinterface / `wavpack-numcodecs` aren't imported anywhere in the code.*
      - *Deferred, low priority: `pixel_error` (OpenCV), `attach_data` (Code Ocean SDK + PyYAML;
        import cell only, since it attaches assets).*
      - *Housekeeping: close each notebook's kernel when done. Kernels left running in one
        workstation (one used 29 GB) stalled a new kernel's startup.*
- **Out of scope for this migration, tracked separately:** moving all scratch-dependent inputs
  into data assets, including switching the unit table to the official
  `LCrecordings_combined_units/combined_unit_tbl.pkl` (md5 check first). This is logged in
  `kinematics_analysis/TODO.md` (`wild` @ `47632e8`) and gets done *after* adoption, one input
  per commit, so any change in results can be attributed to the data, not to Python.
- [x] At adoption (2c), update `kinematics_analysis/CLAUDE.md` lines ~20 and ~70 ("Python 3.9
      compatible syntax only", `requires-python = ">=3.9"`). Note that the library supports
      3.11+, so shared code should avoid 3.12-only features. *Done on `env/py312` @ `e71b64e`: the
      3.9-syntax rule stays until every active branch has migrated, because code moves between
      branches.*

#### 2c. Adopt it
- [x] Before merging: in `environment/Dockerfile` on `env/py312`, set the three AIND libraries
      back from their baseline SHAs to `@main`, and decide whether to keep
      `aind-dynamic-foraging-models==0.16.0` in `py39-constraints.txt`. *Done @ `e71b64e`: back
      on `@main`. Upstream since the baseline: basic-analysis `compute_side_bias` try/except (NaN
      instead of raising), data-utils `hdmf_zarr<0.14` cap. models kept at 0.16.0. Resolves on
      Linux 3.12; only those two library commits differ from the tested environment.
      `env/py312` -> `wild` merges cleanly.*
- [x] Merge `env/py312` into `wild` (the capsule's working branch). *2026-09-25: merge commit
      `0a3c9ce`. To undo: `git revert -m 1 0a3c9ce` on `wild`, push, rebuild.*
- [x] Rebuild the original capsule once (on `wild`). Check it launches and run one notebook.
      *2026-09-25: rebuilt on Python 3.12, and `eph_01` runs. A sync conflict on
      `env_00_reference_sessions.ipynb` (the capsule's 3.9-run outputs vs. the duplicate's 3.12
      run) was resolved by merging and keeping the 3.12 version. `wild` @ `6e36ef2` has
      byte-identical files to `0a3c9ce`.*
- [x] Archive the duplicate capsule. *2026-09-25: synced, then archived. Its final sync
      (`env/py312` @ `33da61c`) renamed the capsule in `metadata/metadata.yml`, so **never merge
      `env/py312` again**. It stays as history only.*
- [x] Then carry the change to `main` and the other active branches (below). *2026-09-25:*
      - *`kinematics-manuscript`: merged `wild` (`fddaaa2`), so it's on 3.12.*
      - *`local-dev`: retired. Tagged `archive/local-dev` (`7f33cd1`), branch deleted.*
      - *`main`: promoted. The old main got a pin commit (`ebd54b9`: AIND libs pinned to the 3.9
        baseline SHAs, so its 3.9 image stays buildable after Stage 3), tagged
        `archive/main-py39`. Then `main` = merge of that + `wild` (`8adbe51`), with files
        byte-identical to `wild` @ `6e36ef2`. Fast-forward push; nothing rewritten. Old main's
        June edits (two now-archived notebooks; attaching `LCrecordings_combined_units` in
        `.codeocean/datasets.json`) live on in the tag and in main's history.*
      - *Note: main's June code imports some library names that no library commit has (e.g.
        `generate_tongue_dfs_new` was renamed in Oct 2025), so a few old notebooks can't import
        on any pin. The pin guarantees only that the image builds.*
- [x] **Every active `kinematics_analysis` branch, not just `main`.** As of 2026-09-24, `main`,
      `kinematics-manuscript`, `local-dev` and `wild` each have their own Dockerfile on the py3.9
      image, installing this library from `@main`. For each branch, either merge `main` in after
      the Dockerfile migration, or pin the library to the last-3.9 tag in that branch's
      Dockerfile (suits branches frozen for a manuscript).
      *2026-09-25: done, see 2c above.*
- [x] Branches of *this* library (e.g. `LC_manuscript`) need nothing. They keep their own
      `requires-python` until they merge `main`, and 3.9 code runs fine on 3.11+.
      *Confirmed; no action.*
- [ ] Move each unknown capsule from Stage 0 to 3.11 or newer (the AIND template is a good
      default), or pin it to a SHA/tag.
- [ ] Optional hygiene: move the `@main` installs in consumers to a tag. This is what makes
      future floor bumps safe by default.

### Stage 3: raise the floor (separate PR, only after Stage 2 is done)
- [ ] **Gate:** confirm every consumer *and every active branch of each consumer* is on 3.11+.
      *As of 2026-09-25, everything is clear except the video re-encoding capsule (user): it's on
      3.10.9, so it needs its library line pinned to `d94854f` and rebuilt. Clear:
      `kinematics_analysis` (`wild`, `main`, `kinematics-manuscript` on 3.12; `archive/main-py39`
      pinned), `aind-motion-energy-capsule` and `aind-BEAST-train-test` (3.11, pinned SHAs).*
     
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
