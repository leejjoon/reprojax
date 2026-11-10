# reprojax 0.2.0 release plan

Nov 10, 2026 · **revision 3**, folding the owner's decisions D1–D4 and the
revision-2 review. Drafted against `main` at `79ee729` (6 commits ahead of
`origin/main`; no tags, no `.github/`, no `CHANGELOG.md`, no tracked `docs/`).

Status: **plan only.** Nothing implemented, pushed, or configured.

## Decisions (owner, Nov 10)

| | Decision |
| --- | --- |
| **D1** | Exclude tests and FITS data from the sdist via `MANIFEST.in`; add a packaging check that asserts the contents. |
| **D2** | Pin `reproject==0.21.0` in `ci/` constraints; keep `reproject>=0.21` in the `test` extra; name 0.21.0 in `test_real_spherex.py`. |
| **D3** | Keep version `0.2.0`. |
| **D4** | CPU CI on the minimum and latest stacks, plus a recorded local ROCm gate. Keep `matrix_precision="f64"` as the default and document it up front. |

## Verified evidence

Measured on this host on Nov 10, 2026, not taken from the repo's prose.

| Check | Environment | Result |
| --- | --- | --- |
| Full suite, minimum stack | Py 3.11.15, JAX/jaxlib 0.10.2, NumPy 2.4.6, Astropy 7.2.0, reproject 0.21.0, mapprojax 0.3.0 | 100 passed, 2 skipped, 39s |
| Full suite, latest stack | Py 3.14.4 **standard** (`/usr/bin/python3.14`), JAX/jaxlib 0.12.0.1, NumPy 2.5.4, Astropy 8.0.2 | 100 passed, 2 skipped, 31s |
| Full suite, ROCm | Py 3.12.13, JAX/jaxlib 0.11.1 + rocm7 plugins 0.11.1, NumPy 2.5.1, Astropy 8.0.2, `RocmDevice(id=0)` | **102 passed, 0 skipped**, 57s |
| Batched benchmark (ROCm) | as above; `jax.jit`, warm-up excluded, n=8, 2040²→1201² | 0.0261s batched vs 4.4269s looped CPU `reproject_interp` (169.9×) |
| Full suite against the built wheel | Py 3.14, source tree off `sys.path` | 100 passed, 2 skipped, 31s |
| sdist rebuilt from its own unpacked tree | — | wheel builds identically |
| CPU skips | — | exactly the 2 GPU-only tests in `tests/test_batch.py` |

Two environment facts that shape the plan:

- The local `~/projects/reprojax/.venv` is **Python 3.14.4 free-threading**
  (`python3.14t`). It is not the CI stack. `ci/latest.txt` derives from
  standard `cpython-3.14.4-linux-x86_64-gnu` at `/usr/bin/python3.14`, which
  resolves the same package set and passes the suite.
- **Astropy 7.2.0 is the oldest tested floor.** `astropy==6.1.7` fails
  collection in all six test modules with
  `AttributeError: module 'numpy' has no attribute 'in1d'`. `np.in1d` was
  deprecated in NumPy **2.0** and removed in **2.4** — measured directly:
  `2.3.5` has it, `2.4.0` does not. `astropy 7.0.0` fails the same way against
  NumPy 2.4.6; `7.1.0` imports; the full suite was run only on `7.2.0`. So the
  floor is the oldest version with real test evidence, not untested support.

## Gaps a release must close

1. **The sdist ships an unrunnable test suite.** It contains `tests/test_*.py`
   but not `_fixtures.py`, `conftest.py`, or `tests/data/`; four of six modules
   fail to collect. D1 excludes tests. Prototyped: `prune tests / benchmarks /
   examples / docs` gives a 17 KB sdist (`src/`, `README.md`, `LICENSE`,
   `pyproject.toml`, `MANIFEST.in`).
2. **`f32` mode emits spurious float64 warnings with x64 off** — 2 from
   `reproject_interp`, 4 from `reproject_adaptive`, 1 more on the
   generic-provider coverage path.
3. **No CI**, no tags, 6 unpushed commits. `pypi.org/pypi/reprojax/json`
   returns 404: nothing is published at the name. That is absence only — it does
   **not** reserve or guarantee the name, and it does not mean the name is
   unclaimed. Only the pending-publisher setup plus the first successful upload
   establishes it.
4. **The README has no callable example** and never states that the default
   `matrix_precision="f64"` raises unless `jax_enable_x64` is on:
   `ValueError: precision='f64' requires float64. Call jax.config.update('jax_enable_x64', True) ...`
5. **Metadata floors are dead or wrong** (`jax>=0.5`, `numpy>=1.20`,
   `astropy>=6`), `reproject` in the extra is unpinned, and the OSI classifier
   duplicates `license = {file = ...}` and emits build warnings.
6. **No data provenance** for the 16 MB SPHEREx fixture.
7. **No repository or PyPI release settings.** Verified now: `GET
   /repos/leejjoon/reprojax/rulesets` → `[]`; `/environments` →
   `total_count: 0`; `main` → `protected: false`.

## Stage 1 — preflight and metadata

- [ ] The 6 outstanding commits stay **local**. **Gate G1** authorizes pushing
      them together with the finished release work, up to the final SHA S —
      nothing is pushed before that.
- [ ] `MANIFEST.in` per D1 (prune `tests`, `benchmarks`, `examples`, `docs`),
      plus a packaging assertion that the sdist has no `tests/` entry and the
      wheel carries no test or FITS file.
- [ ] `pyproject.toml`:
      - `requires = ["setuptools>=77"]` (needed for the SPDX field; the build
        emits zero deprecation warnings with it),
      - `license = "BSD-3-Clause"` + `license-files = ["LICENSE"]`, drop the
        `License :: OSI Approved` classifier,
      - `numpy>=2.4.6`, `jax>=0.10.2` (match mapprojax),
      - `astropy>=7.2.0` (oldest tested floor, above),
      - `mapprojax>=0.3.0,<0.4` unchanged,
      - `test = ["pytest", "reproject>=0.21"]` (D2).
- [ ] `src/reprojax/py.typed` + `[tool.setuptools.package-data]`
      `reprojax = ["py.typed"]`.
- [ ] Name `reproject==0.21.0` in `tests/test_real_spherex.py` where the
      tolerances are documented, noting that a reference upgrade moves the
      baseline, not just the tool (D2).

## Stage 2 — the float64-in-f32 precision warnings

Root cause: four hardcoded `jnp.float64` sites on paths that otherwise run in
float32.

- [ ] `interp.py` and `adaptive.py` grids: derive the grid dtype from
      `matrix_precision` (`float32` for `"f32"`, else `float64`).
- [ ] `adaptive.py`: `global_v/u_f` are `astype(jnp.float64)`; cast to the
      transform's coordinate dtype (`v0.dtype` / `u0.dtype`).
- [ ] `interp.py` generic-provider `coverage_input`: coverage is a [0, 1]
      weight fraction, so `float32` is exact enough for the threshold test and
      removes the last warning.
- [ ] Fresh-process tests, one file, each asserting **warning count**:
      1. `f64` with x64 on → 0 warnings.
      2. `f64` without x64 → the documented `ValueError` from both entry
         points.
      3. `f32` without x64 → 0 warnings from `reproject_interp`,
         `reproject_adaptive`, **and the generic-provider coverage path**
         (a plain callable with no `pixel_map`, which reaches the
         `coverage_input` branch that the pixel-map path skips).
- [ ] Keep coordinate checks separate from image and footprint checks. The f32
      caps (`_F32_CAP` = `6e-4` / `5e-4` px) bound **coordinate** error on the
      real pair; they say nothing about resampled-image agreement or footprint
      equality. So the new tests assert:
      - coordinate error against the f64 transform, inside the existing caps;
      - image agreement and footprint equality only as the existing real-pair
        tests already do, unchanged.
- [ ] **Preserve the existing numerical and validity contracts** — the same f32
      coordinate caps, the same NaN/validity semantics, the same footprint
      definition (`~isnan(output)`), and the same dtype of result and footprint.
      Prototyped on a throwaway copy: warnings 2 → 0 and 4 → 0, suite still
      100 passed / 2 skipped.

## Stage 3 — CI and the release artifact

CPU-only hosted matrix; every action pinned by SHA.

- [ ] `.github/workflows/ci.yml`, job `cpu`: matrix of Py 3.11.15 +
      `ci/minimum.txt`; Py 3.14.4 **standard** (`freethreaded: false`) +
      `ci/latest.txt`. Each runs the full suite, with the f32 paths also under
      a fail-on-warning check.
- [ ] `ci/minimum.txt`: the verified minimum stack with `reproject==0.21.0`
      (D2) and `mapprojax==0.3.0` pinned.
- [ ] `ci/latest.txt`: from the standard-3.14.4 resolution, same pins.
- [ ] `.github/workflows/ci.yml`, job `packaging` on `main` only — builds the
      wheel and sdist **once** from a clean `git archive`, then:
      1. asserts sdist contents (no `tests/`, no `*.fits`) and wheel contents
         (no tests, `py.typed` present);
      2. installs the wheel into a fresh venv and runs the full suite with the
         source tree off `sys.path`;
      3. installs the sdist into a second fresh venv and runs the README quick
         start;
      4. uploads the single artifact `release-dist` containing the wheel, the
         sdist and `SHA256SUMS`.
- [ ] Green `main` on the final SHA **S**. **Gate G2.**

## Stage 4 — documentation

- [ ] `README.md`: keep the architecture section; replace "Early development"
      with a real Status; state the validated scope (TAN/SIN + SIP;
      `reproject` is a test-time reference, not a dependency).
- [ ] Accuracy claims, **each labelled by algorithm and by whether it is an
      observation or an enforced tolerance**:
      - *interp, real SPHEREx pair* — **observation**: 1,345,444 of 1,345,460
        jointly-valid pixels agree to `rtol<=1e-8`; the other 16 sit at an
        un-flagged data spike. **Enforced**: `rtol=1e-6, atol=0.01` over all
        jointly-valid pixels, plus exact footprint equality.
      - *interp, synthetic TAN/TAN-SIP pairs* — **enforced**:
        `rtol=1e-8, atol=1e-9`.
      - *adaptive, real SPHEREx pair* — **observation**: median disagreement
        ~`4e-8`; 1,334,193 of 1,334,200 jointly-valid pixels form the bulk, and
        the other 7 sit at the same spike with a peak difference ~`0.003`.
        **Enforced**: `rtol=1e-6, atol=0.005` over all jointly-valid pixels,
        plus exact footprint equality.
      - *adaptive, synthetic pairs* — **enforced**: `rtol=1e-6, atol=1e-8`.
      - *`matrix_precision="f32"`* — **enforced coordinate caps**
        `6e-4` / `5e-4` px per direction on the real pair; not image
        agreement, and not machine precision.
- [ ] If the performance figure is used, name backend, JIT status and the
      comparison: batched **jitted** reprojax on **ROCm** (`RocmDevice`,
      `AMD Radeon Graphics`, JAX 0.11.1 + rocm7 plugins 0.11.1) versus
      **looped CPU** `reproject_interp`, warm-up excluded, n=8, 2040²→1201².
      Measured 0.0261s against a 4.43s baseline, and 0.0259s against 7.89s on
      a later run: the reprojax figure is stable and the CPU baseline is what
      varies, so quote a range (~170–300×), not one ratio. It is a
      cross-backend batch-vs-loop number, not a like-for-like backend
      comparison, and no test gates on it.
- [ ] **Self-contained quick start** — no `tests/data`. Synthetic 64² TAN
      header → 40² output, verified running from an installed wheel
      (1,120 of 1,600 output pixels valid, peak 0.73, 0 warnings).
- [ ] Lead with the precision rule: default `"f64"` needs
      `jax.config.update("jax_enable_x64", True)`; `"f32"` needs no x64 and is
      the GPU-throughput path; the same rule applies to `reproject_adaptive`.
- [ ] `CHANGELOG.md`: **dated at publication** (post-G7, in the G8 commit),
      since it is not an sdist input; until then the heading reads
      `## 0.2.0 — unreleased`. Entry content: first public release;
      `PixelTransform` contract;
      `reproject_interp` / `reproject_adaptive` / batched variant; `pixel_map`
      validity capability; both precision modes; `mapprojax>=0.3.0,<0.4`.
- [ ] `tests/data/README.md`: provenance, terms, checksums —
      - `spherex_l2b_v20.fits`, 16,666,560 B, `sha256:36c9d328…56d6e12`;
        SPHEREx L2b IMAGE, 2040², `RA---TAN-SIP`/`DEC--TAN-SIP`,
        `BUNIT = 'MJy / sr'`, `DATE-OBS = 2025-06-02T22:24:40.376`;
      - `galactic_template_header.fits`, 1,700 B, `sha256:d061bb00…5c47f3c7`;
        GLON/GLAT-TAN, 1201², `CRVAL1 = 78.9293`, `CRVAL2 = 0.48367`,
        `LONPOLE = 180`;
      - the FLAGS extension was dropped — the root cause of the 16 interp and
        7 adaptive spike pixels;
      - **if the SPHEREx redistribution terms cannot be identified, Stage 4
        stops** and no release commits are made until they are settled.
- [ ] README attribution for the data, alongside the existing `reproject`
      licence block.

## Stage 5 — settings, gate, tag, publish

### Setup gate — before tagging

- [ ] **G3 — release settings.** Perform all four, then record read-only API
      evidence of each. **2026-11-10: items 1-4 done except the PyPI pending
      publisher, which still needs an interactive login — see the evidence
      block below. G3 remains open until that exists.**
      1. PyPI **pending publisher**: owner `leejjoon`, repository `reprojax`,
         workflow `publish.yml`, environment `pypi`.
      2. **`pypi` environment**: reviewer `leejjoon`, self-review allowed,
         admin bypass **off**, and the `v*` tag deployment policy configured
         **before any tag is pushed** — a tag pushed first would either be
         blocked by a missing policy or, worse, deploy without one.
      3. **`v*` tag ruleset** with exactly the `update` and `deletion` rules.
      4. **`main` ruleset** with exactly `deletion` and `non_fast_forward` —
         **no `update` rule** (we push directly to `main`) and **no required
         status checks**.
      Evidence is read-only API, and it must show the *contents*, not just
      existence:
      - `gh api repos/leejjoon/reprojax/rulesets` to list IDs, then
        `gh api repos/leejjoon/reprojax/rulesets/<id>` for each, verifying
        `enforcement == "active"`, the `ref_name` targeting (`refs/tags/v*` for
        the tag ruleset, `refs/heads/main` for the branch ruleset), and
        `bypass_actors == []` on both. Then the exact rule set, which is the
        part that is easy to get wrong:
        - tag ruleset: `update` and `deletion`, and nothing else;
        - `main` ruleset: `deletion` and `non_fast_forward`, and nothing else
          — in particular **no `update`** and **no `required_status_checks`,
          since main is pushed to directly;
      - `gh api repos/leejjoon/reprojax/environments/pypi`, verifying
        `can_admins_bypass == false`; a `required_reviewers` protection rule
        whose `reviewers` contains `leejjoon` with
        `prevent_self_review == false`; and
        `gh api repos/leejjoon/reprojax/environments/pypi/deployment-branch-policies`
        returning exactly one `{"name": "v*", "type": "tag"}` policy. All three
        fields are in the REST payload — verified against mapprojax's existing
        `pypi` environment, which returns `can_admins_bypass: false`,
        `prevent_self_review: false`, reviewer `leejjoon`, and the single `v*`
        tag policy;
      - `gh api repos/leejjoon/reprojax/branches/main` → `protected: true`, and
        the protection payload shows no required status checks.
      Baseline captured today: `rulesets` `[]`, `environments`
      `total_count: 0`, `branches/main` `protected: false`. The PyPI pending
      publisher has no public read API — record it from the publisher page and
      re-verify the exact owner/repo/workflow/environment tuple immediately
      before G5, because it is only enforced at upload time.

      **G3 evidence — 2026-11-10.** Items 2-4 configured; item 1 outstanding.
      - `environments/pypi`: created 2026-11-10T11:02:29Z.
        `can_admins_bypass: false`; one `required_reviewers` rule with
        reviewer `leejjoon`, `prevent_self_review: false`; one `branch_policy`
        rule backed by `deployment_branch_policies` = exactly
        `[{"name": "v*", "type": "tag"}]` (`total_count: 1`), with
        `protected_branches: false`. The first PUT returned
        `can_admins_bypass: true`, so it was re-put with the field explicit.
      - ruleset `26935196` "main": `target: branch`, `enforcement: active`,
        `ref_name.include: ["~DEFAULT_BRANCH"]`, `bypass_actors: []`, rules
        exactly `["deletion", "non_fast_forward"]`. No `update`, no
        `required_status_checks`.
      - ruleset `26935201` "tags v*": `target: tag`, `enforcement: active`,
        `ref_name.include: ["refs/tags/v*"]`, `bypass_actors: []`, rules
        exactly `["update", "deletion"]`.
      - `branches/main`: `protected: true`, `required_status_checks`
        `{checks: [], contexts: [], enforcement_level: "off"}`.
      - name still unclaimed: `simple/reprojax/` and `pypi/reprojax/json`
        both 404 as of this record.
      - **OUTSTANDING: the PyPI pending publisher (item 1) is not yet
        created.** The PyPI web UI needs an interactive login and this host
        holds no PyPI credentials (no `~/.pypirc`, no `PYPI_TOKEN`). Until it
        exists, `publish.yml` will fail at upload with `invalid-publisher`.
        Create it at <https://pypi.org/manage/account/publishing/> with owner
        `leejjoon`, repository `reprojax`, workflow `publish.yml`, environment
        `pypi`, then re-verify before G5. G3 is **not** complete.

### Gate, tag, publish

- [ ] **G4 — pre-tag local gate.** From the `release-dist` artifact of the
      **single successful `ci.yml` run on `main` for S** — the exact bytes,
      no rebuild: full suite on CPU (minimum + latest stacks) and on ROCm, plus
      the README quick start from a fresh install of the wheel and of the
      sdist. Record run ID, attempt, artifact ID and expiry, plus interpreter,
      JAX/plugin versions and device.
      Also run `twine check --strict` on the **exact downloaded** wheel and
      sdist (not on a local rebuild), so a malformed `METADATA` or a
      `README` that Twine cannot render fails before the tag rather than at
      the upload.
- [ ] **G5 — tag S itself.** `git tag v0.2.0 S` and push the tag. Do not
      re-run or re-dispatch CI on S between G4 and this step (see the rules
      below: `validate` requires exactly one successful run, one artifact).
      No commit is
      made between G4 and the tag, and no commit is added to change the release
      record. S therefore does **not** contain `docs/release/0.2.0.md`; that
      record lands later, and the tag's identity is the byte source, not the
      documentation.
- [ ] `.github/workflows/publish.yml`, on `v*`:
      - **`validate`** (`contents: read`, `actions: read`, no id-token):
        locate the single successful `ci.yml` run on `main` whose `head_sha`
        is the tag's SHA; record run ID, attempt, artifact ID and expiry; stop
        if the artifact is missing, expired, or has been replaced. Verify
        `SHA256SUMS`, and that the wheel's version equals the tag minus `v`.
        Print every hash so the operator can compare them with G4's.
      - **handoff**: `publish` declares `needs: validate` and never proceeds
        if `validate` fails. `validate` declares the original run ID,
        attempt, artifact ID, expiry and the `SHA256SUMS` hash as job
        outputs. There is **no intermediate artifact copy**: `validate` writes
        nothing back to Actions storage.
      - **`publish`** (`environment: pypi`, `needs: validate`, permissions
        `id-token: write`, `actions: read`, `contents: read`): the only job
        with `id-token: write`. **After** environment approval it re-reads the
        original run and artifact by ID, confirms the run is still
        completed/success on the same attempt and the artifact is unexpired
        with an unchanged expiry, downloads **that** artifact by ID — never by
        name — re-verifies `SHA256SUMS` and the staged inventory, then uploads
        **only** the wheel and the sdist. It never rebuilds and never
        re-selects an artifact.
- [ ] **G6 — approve the protected `publish` job.** Before approving, compare
      `validate`'s printed run ID, attempt, artifact ID, expiry and hashes with
      the G4 record; any difference stops the release and returns to G4.
- [ ] **G7 — post-publication verification.** PyPI hashes match
      `SHA256SUMS`; a fresh `pip install reprojax==0.2.0` in a clean venv
      resolves `mapprojax==0.3.0`; runtime dependencies only, no `reproject`;
      and runs the README quick start.
- [ ] **G8 — the release record, written once.** In a single commit **after**
      G7, add `docs/release/0.2.0.md` (S, tag, artifact IDs and hashes, the G4
      matrix, the ROCm result, the PyPI verification) **and** set the
      `CHANGELOG.md` date. Append-only from then on. Both are outside the
      sdist, so neither changes a published byte.
- [ ] Optional, non-blocking: GitHub Release notes for `v0.2.0`; repo topics
      (`astronomy`, `fits`, `jax`, `wcs`, `reprojection`).

### Rules for this release

- **Once the tag exists, any change to a release input means a new commit and
  0.2.1.** Never move the tag; PyPI file names are permanent.
- **Any replacement artifact repeats the full G4 gate.** A re-built wheel is a
  new candidate even if the commit is unchanged.
- **Never re-run CI on S after G4.** The packaging job uploads an artifact
  named `release-dist`; a second run on the same SHA would either fail on the
  duplicate artifact name or leave two candidate artifacts, and `validate`
  (which finds the *single* successful run for the tag's SHA) would then find
  none usable. A CI re-run on S after the gate invalidates the gate: treat it
  as a new candidate and repeat G4, or make a new commit and a new S.
- **Infrastructure re-runs** elsewhere (a flaky job before the gate, an
  artifact re-download) need explicit authorization and never silently change
  which bytes ship.
- **No ported release machinery.** Do not port `release_artifact.py`,
  `check_installed_artifacts.py`, the input digest, the candidate marker, or the
  closure fences. This release is: one CI build, one artifact, one gate, one
  tag, one upload, one record.

## Environment notes for the gate

mapprojax's closure recorded Py 3.12.13 / JAX 0.11.1 / NumPy 2.5.1 /
Astropy 8.0.2 / ROCm plugins 0.11.1 on this Strix Halo host; that set is
reproducible here and is the ROCm gate environment. The CPU gate uses the
`ci/minimum.txt` and `ci/latest.txt` stacks, so the two gates differ in
interpreter on purpose.

## Gate pairing

- **G1 + G2 may run together** (push to S, then green `main` CI on S), the
  second only if the first passes.
- **G7 + G8 may run together** (verify from PyPI, then write the record), the
  second only if the first passes.
- **G3, G5 and G6 each need their own explicit approval** and cannot be folded
  into another gate.

## Commit order

1. Packaging metadata, floors, `py.typed`, `MANIFEST.in` (Stage 1).
2. Float64 dtype fixes with fresh-process tests (Stage 2).
3. CI `cpu` matrix and `packaging` job with `SHA256SUMS` (Stage 3).
4. README quick start and accuracy labels, `CHANGELOG.md`, `tests/data/README.md`
   (Stage 4; stops if the data terms are unresolved).
5. `publish.yml` (Stage 5 settings half).
6. Push up to **S**; green `main` CI on S — **G1** then **G2**, allowed to
   run together.
7. Settings — **G3**; gate on S's `release-dist` — **G4**.
8. Tag **S** — **G5**; approve — **G6**; verify from PyPI — **G7**.
9. `docs/release/0.2.0.md`, once — **G8**, allowed to follow G7 in the same
   run.

## Done when

- **All existing and new tests pass on the `ci/minimum.txt` and
  `ci/latest.txt` stacks and on the ROCm gate, with exactly the 2 GPU-only
  skips on CPU and none on ROCm** — not a fixed pass count.
- `f32` mode runs warning-free with x64 off on all three entry paths; `f64`
  without x64 fails with the documented message; both are covered by
  fresh-process tests.
- The numerical and validity contracts are preserved: same f32 coordinate caps,
  same NaN semantics, same footprint definition.
- The `release-dist` wheel passes the full suite with the source tree off
  `sys.path`; its sdist passes the quick start and contains no tests or FITS
  data.
- README states the precision rule, the validated scope, and each figure with
  its algorithm and observation-vs-tolerance label; the quick start needs
  nothing from `tests/`.
- All four G3 settings are configured with recorded read-only evidence, and the
  pending-publisher tuple is re-verified immediately before tagging.
- `reprojax==0.2.0` is on PyPI with hashes matching `SHA256SUMS`; a fresh
  `pip install reprojax==0.2.0` in a clean venv **confirms `mapprojax==0.3.0`
  was resolved** from PyPI, pulls runtime dependencies only (no `reproject`),
  and runs the README quick start; `docs/release/0.2.0.md` records it — which
  also closes the mapprojax plan's last open item.
