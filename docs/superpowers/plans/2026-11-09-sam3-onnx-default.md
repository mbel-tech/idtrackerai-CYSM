# SAM 3 on the public ONNX weights, by default — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SAM 3 run on the public ONNX weights (`wkentaro/sam3-onnx-models-v0.6.0`, through `osam`) by default, with no token, while `sam3.pt` on PyTorch stays an option.

**Architecture:** A new `sam3_onnx.py` holds an ONNX predictor with the same `predict(frame) -> (masks, scores)` interface as `Sam3Predictor`; a factory `make_sam3_predictor` picks the engine from the weights value, and every caller goes through it. The weights value gains a keyword (`sam3-onnx`, the SAM 3 default) and a folder form; provenance records a label and a hash derived from osam's pinned file hashes.

**Tech Stack:** Python 3.11+, `osam` 0.10.x / `onnxruntime`, PyQt6 (Segmentation App), pytest with stand-in modules.

**Spec:** `docs/superpowers/specs/2026-11-09-sam3-onnx-default-design.md`

## Global Constraints

- Keyword is exactly `sam3-onnx`, compared case-insensitively, constant `SAM3_ONNX` in `segmentation_model.py`.
- Provenance label `sam3-onnx-models-v0.6.0` is derived from osam's blob URLs, never hard-coded.
- Hashes, file names and URLs come only from osam's `Sam3._blobs`; none are copied into this repository.
- `osam` (and `onnxruntime`) is imported only inside functions; idtracker.ai and the Segmentation App must start without it.
- New optional extra: `sam3 = ['osam >= 0.10, < 0.11']`.
- Existing `sam3.pt` path and `huggingface` behaviour is unchanged; Detectron2 still requires `--weights`.
- Tests never download and never need a GPU; `osam` is stood in for via `monkeypatch.setitem(sys.modules, ...)`.
- Run tests with: `QT_QPA_PLATFORM=offscreen python -m pytest <files> -q -p no:cacheprovider`.

## Review Focus

1. Box-cropped masks whose box touches or passes the frame edge — the paste must clip, never raise or wrap (test in Task 3).
2. A folder given with a trailing slash, `~`, or as a relative path — must still be recognised as ONNX (test in Task 1).
3. A folder that already has the files in osam's cache as identical copies — must not re-copy or fail on an existing hard link (test in Task 2).
4. Old contour files made with `sam3.pt` checked against a `.toml` now defaulting to `sam3-onnx` — must be reported stale, not silently reused (test in Task 4).
5. `max_instances = 0` (keep all) must not be passed to osam as `max_annotations=0`, which would return nothing (test in Task 3).

---

### Task 1: The `sam3-onnx` keyword and the SAM 3 default in the model table

**Files:**
- Modify: `src/idtrackerai/extra_tools/detectron2_pipeline/segmentation_model.py` (keywords near line 70; `normalize_model`; `weights_label`)
- Test: `tests/segmentation_model_test.py`

**Interfaces:**
- Produces: `SAM3_ONNX = "sam3-onnx"`; `is_onnx_weights(weights) -> bool` (True for the keyword in any case, or for an existing directory after `expanduser`); `normalize_model` defaults `weights` to `SAM3_ONNX` when `backend == "sam3"`. (`weights_label` learns ONNX in Task 4.)

- [ ] **Step 1: Write the failing tests**

```python
def test_sam3_without_weights_defaults_to_onnx():
    spec = sm.normalize_model({"backend": "sam3", "prompt": "fish"})
    assert spec["weights"] == sm.SAM3_ONNX

def test_onnx_keyword_is_case_insensitive_and_sam3_only():
    assert sm.normalize_model({"backend": "sam3", "prompt": "fish", "weights": "SAM3-ONNX"})["weights"] == sm.SAM3_ONNX
    with pytest.raises(sm.SegmentationModelError, match='"sam3-onnx" only applies to backend = "sam3"'):
        sm.normalize_model({"backend": "detectron2", "weights": "sam3-onnx"})

def test_detectron2_still_needs_weights():
    with pytest.raises(sm.SegmentationModelError, match="weights is missing"):
        sm.normalize_model({"backend": "detectron2"})

@pytest.mark.parametrize("form", ["{d}", "{d}/", "rel"])
def test_a_folder_is_onnx_weights(tmp_path, monkeypatch, form):
    d = tmp_path / "models"; d.mkdir()
    monkeypatch.chdir(tmp_path)
    value = "models" if form == "rel" else form.format(d=d)
    assert sm.is_onnx_weights(value)
    assert not sm.is_onnx_weights(tmp_path / "sam3.pt")
    assert not sm.is_onnx_weights(sm.HUGGINGFACE)
```

- [ ] **Step 2: Run them to see them fail**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/segmentation_model_test.py -q -p no:cacheprovider -k "onnx or detectron2_still"`
Expected: FAIL (`SAM3_ONNX` / `is_onnx_weights` undefined; missing-weights error for sam3).

- [ ] **Step 3: Implement** the constant, `is_onnx_weights`, the default and the keyword check in `normalize_model` (same shape as the existing `HUGGINGFACE` branch, error text `weights = "sam3-onnx" only applies to backend = "sam3"`), and update the missing-weights message to mention only Detectron2. Leave `weights_label` for Task 4.

- [ ] **Step 4: Run the file**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/segmentation_model_test.py -q -p no:cacheprovider`
Expected: PASS (any existing test asserting "sam3 without weights is an error" is updated to the new default, and that change is called out in the commit message).

- [ ] **Step 5: Commit** — `git commit -m "Default SAM 3 weights to the public ONNX conversion in the model table"`

---

### Task 2: `sam3_onnx.py` — osam's pins, folder preparation, identity

**Files:**
- Create: `src/idtrackerai/extra_tools/detectron2_pipeline/sam3_onnx.py`
- Create: `tests/fake_osam.py` (shared stand-in, used by Tasks 2–6)
- Test: `tests/sam3_onnx_test.py`

**Interfaces:**
- Produces in `sam3_onnx.py`:
  - `class Sam3OnnxError(RuntimeError)`
  - `onnx_blobs() -> dict[str, Blob]` — `osam.apis.get_model_type_by_name("sam3")._blobs`; raises `Sam3OnnxError("... pip install osam ...")` when osam is missing.
  - `onnx_label() -> str` — the path segment of the first blob URL after `huggingface.co/<owner>/` (e.g. `sam3-onnx-models-v0.6.0`).
  - `weights_identity() -> tuple[str, str]` — `(onnx_label(), sha256("+".join(blob.hash for key in sorted(blobs))))` hex.
  - `prepare_folder(folder: Path) -> None` — as specified in the spec's `prepare_folder`.
- Produces in `tests/fake_osam.py`:
  - `install_fake_osam(monkeypatch, tmp_path, annotations=None) -> SimpleNamespace` — registers `osam`, `osam.apis`, `osam.types` in `sys.modules`; four fake blobs with `filename` `sam3_<key>.onnx`, `hash` `"sha256:" + sha256(b"<key>")`, `url` `https://huggingface.co/wkentaro/sam3-onnx-models-v0.6.0/resolve/abc/<filename>`, `path` under `tmp_path/"cache"`; a `Sam3` class whose `__init__(device=None)` records `device` and creates any missing cache file (counting downloads in `fake.downloads`), and whose `generate(request)` records the request in `fake.requests` and returns `SimpleNamespace(annotations=annotations or [])`. `fake.write_folder(folder, skip=(), corrupt=())` writes the four files with the right (or wrong) contents.

- [ ] **Step 1: Write the failing tests**

```python
def test_label_and_identity_come_from_osam(monkeypatch, tmp_path):
    install_fake_osam(monkeypatch, tmp_path)
    label, digest = sam3_onnx.weights_identity()
    assert label == "sam3-onnx-models-v0.6.0" and len(digest) == 64

def test_folder_fills_the_cache_without_downloading(monkeypatch, tmp_path):
    fake = install_fake_osam(monkeypatch, tmp_path)
    fake.write_folder(tmp_path / "models")
    sam3_onnx.prepare_folder(tmp_path / "models")
    assert all(Path(b.path).read_bytes() == (tmp_path / "models" / b.filename).read_bytes() for b in fake.blobs.values())
    fake.Sam3(); assert fake.downloads == 0

def test_missing_files_are_all_named(monkeypatch, tmp_path):
    fake = install_fake_osam(monkeypatch, tmp_path)
    fake.write_folder(tmp_path / "models", skip=("decoder", "point_decoder"))
    with pytest.raises(sam3_onnx.Sam3OnnxError, match="sam3_decoder.onnx.*sam3_point_decoder.onnx|hf download"):
        sam3_onnx.prepare_folder(tmp_path / "models")

def test_wrong_file_gives_both_hashes(monkeypatch, tmp_path):
    fake = install_fake_osam(monkeypatch, tmp_path)
    fake.write_folder(tmp_path / "models", corrupt=("decoder",))
    with pytest.raises(sam3_onnx.Sam3OnnxError) as err:
        sam3_onnx.prepare_folder(tmp_path / "models")
    assert fake.blobs["decoder"].hash.split(":")[1] in str(err.value)

def test_identical_cached_file_is_left_alone(monkeypatch, tmp_path):  # Review Focus 3
    fake = install_fake_osam(monkeypatch, tmp_path)
    fake.write_folder(tmp_path / "models")
    sam3_onnx.prepare_folder(tmp_path / "models")
    sam3_onnx.prepare_folder(tmp_path / "models")  # second time: no error

def test_without_osam_it_says_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "osam", None)
    with pytest.raises(sam3_onnx.Sam3OnnxError, match="pip install osam"):
        sam3_onnx.onnx_blobs()
```

- [ ] **Step 2: Run to see them fail**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/sam3_onnx_test.py -q -p no:cacheprovider`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Implement** `tests/fake_osam.py` and the four functions. Import the module the way `sam3_predictor.py` imports `segmentation_model` (package-relative, falling back to path for the Colab bundle). Hash files in 1 MiB chunks; hard-link with `os.link`, falling back to `shutil.copyfile` on `OSError`; skip a target that exists with the same size and hash.

- [ ] **Step 4: Run** the same command. Expected: PASS.

- [ ] **Step 5: Commit** — `git commit -m "Add the osam-backed SAM 3 ONNX weights: pins, folder check, identity"`

---

### Task 3: `Sam3OnnxPredictor` and the engine factory, wired into every caller

**Files:**
- Modify: `sam3_onnx.py` (add the predictor)
- Modify: `src/idtrackerai/extra_tools/detectron2_pipeline/sam3_predictor.py` (add `make_sam3_predictor`)
- Modify: `inference.py:465-478`, `prelabel.py:265-282`, `src/idtrackerai/segmentation_app/widgets/pipeline_threads.py:240-270, 355-390` (call the factory; "not installed" message per engine)
- Modify: `bundle.py:44-48` (ship `sam3_onnx.py` as `tools/sam3_onnx.py`)
- Test: `tests/sam3_onnx_test.py`, `tests/sam3_predictor_test.py`, `tests/bundle_test.py`

**Interfaces:**
- Consumes: `prepare_folder`, `onnx_blobs`, `Sam3OnnxError` (Task 2); `sm.is_onnx_weights`, `sm.SAM3_ONNX` (Task 1).
- Produces: `Sam3OnnxPredictor(weights, prompt, device="cuda", score_threshold=0.5, max_instances=0)` with `.predict(frame: np.ndarray) -> tuple[list[np.ndarray], list[float]]` and `.description == f"sam3-onnx ({prompt!r})"`; `make_sam3_predictor(weights, *, prompt, device, score_threshold, max_instances)` in `sam3_predictor.py`.

- [ ] **Step 1: Write the failing tests**

```python
def ann(box, score, fill=True):
    x0, y0, x1, y1 = box
    return SimpleNamespace(score=score, bounding_box=SimpleNamespace(xmin=x0, ymin=y0, xmax=x1, ymax=y1),
                           mask=np.full((y1 - y0 + 1, x1 - x0 + 1), fill, bool))

def test_masks_are_pasted_into_the_full_frame(monkeypatch, tmp_path):
    install_fake_osam(monkeypatch, tmp_path, annotations=[ann((2, 3, 5, 4), 0.9)])
    masks, scores = sam3_onnx.Sam3OnnxPredictor("sam3-onnx", "fish").predict(np.zeros((10, 12, 3), np.uint8))
    assert masks[0].shape == (10, 12) and masks[0].dtype == np.uint8
    assert masks[0].sum() == 8 and masks[0][3:5, 2:6].all()

def test_box_past_the_edge_is_clipped(monkeypatch, tmp_path):  # Review Focus 1
    install_fake_osam(monkeypatch, tmp_path, annotations=[ann((9, 7, 13, 11), 0.9)])
    masks, _ = sam3_onnx.Sam3OnnxPredictor("sam3-onnx", "fish").predict(np.zeros((10, 12, 3), np.uint8))
    assert masks[0].sum() == 9  # columns 9-11, rows 7-9

def test_threshold_cap_and_order(monkeypatch, tmp_path):
    install_fake_osam(monkeypatch, tmp_path, annotations=[ann((0,0,1,1),0.6), ann((0,0,1,1),0.95), ann((0,0,1,1),0.3)])
    _, scores = sam3_onnx.Sam3OnnxPredictor("sam3-onnx", "fish", score_threshold=0.5, max_instances=1).predict(np.zeros((4,4,3), np.uint8))
    assert scores == [0.95]

def test_prompt_rgb_and_keep_all(monkeypatch, tmp_path):  # Review Focus 5
    fake = install_fake_osam(monkeypatch, tmp_path)
    frame = np.zeros((2, 2, 3), np.uint8); frame[..., 0] = 255  # blue in BGR
    sam3_onnx.Sam3OnnxPredictor("sam3-onnx", "fish", max_instances=0).predict(frame)
    req = fake.requests[0]
    assert req.prompt.texts == ["fish"] and req.image[0, 0, 2] == 255
    assert req.prompt.max_annotations != 0

def test_no_cuda_falls_back_to_cpu(monkeypatch, tmp_path):
    fake = install_fake_osam(monkeypatch, tmp_path, cuda=False)
    sam3_onnx.Sam3OnnxPredictor("sam3-onnx", "fish", device="cuda")
    assert fake.devices == ["cpu"]

@pytest.mark.parametrize("weights,engine", [("sam3-onnx", "Sam3OnnxPredictor"), ("{folder}", "Sam3OnnxPredictor"),
                                            ("huggingface", "Sam3Predictor"), ("{pt}", "Sam3Predictor")])
def test_factory_picks_the_engine(monkeypatch, tmp_path, weights, engine): ...
    # stub both classes to record which was built; assert type name == engine
```

Add to `tests/bundle_test.py`'s expected list: `"tools/sam3_onnx.py"`.

- [ ] **Step 2: Run to see them fail.** Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/sam3_onnx_test.py tests/sam3_predictor_test.py tests/bundle_test.py -q -p no:cacheprovider` — Expected: FAIL.

- [ ] **Step 3: Implement.** The fake gains `cuda=True` and records `devices`; the predictor checks `"CUDAExecutionProvider" in onnxruntime.get_available_providers()` (the fake installs an `onnxruntime` stand-in too) and prints the same CPU-fallback message as `resolve_device`. Pass `max_annotations=max_instances or None`; still sort, threshold and cap in `predict`. Paste with slices clipped to `[0, h)` / `[0, w)` on both the frame and the cropped mask. Replace each `Sam3Predictor(...)` call site with `make_sam3_predictor(...)`. In `pipeline_threads.py`, the ImportError message says `pip install osam` when `sm.is_onnx_weights(self.weights)`, else the current `pip install sam3` text; also catch `Sam3OnnxError` alongside `FileNotFoundError` and emit its text.

- [ ] **Step 4: Run** the three files plus `tests/sam3_prelabel_test.py tests/sam3_panel_test.py`. Expected: PASS.

- [ ] **Step 5: Commit** — `git commit -m "Run SAM 3 through osam when the weights are the ONNX conversion"`

---

### Task 4: Provenance, staleness and CLI defaults

**Files:**
- Modify: `inference.py` (`local_weights` ~116, `weights_sha256` ~109, sidecar `weights=` ~714, `--weights` help ~819 and validation ~940-950)
- Modify: `segmentation_model.py` (`weights_label`)
- Modify: `prelabel.py:222-224` (`--weights` optional, default `sam3-onnx`, help lists the four forms)
- Test: `tests/sam3_onnx_test.py`, `tests/segmentation_model_test.py`, `tests/sam3_prelabel_test.py`, `tests/sam3_huggingface_test.py`

**Interfaces:**
- Consumes: `weights_identity`, `onnx_label`, `prepare_folder` (Task 2); `is_onnx_weights` (Task 1).
- Produces: `local_weights("sam3", w)` returns `w` unchanged for ONNX (after `prepare_folder` when a folder); `weights_sha256(w)` returns `weights_identity()[1]` for ONNX; sidecar `weights` attr is `onnx_label()` for ONNX; `weights_label(spec)` likewise.

- [ ] **Step 1: Write the failing tests**

```python
def test_download_and_folder_record_the_same_weights(monkeypatch, tmp_path):
    fake = install_fake_osam(monkeypatch, tmp_path); fake.write_folder(tmp_path / "m")
    assert inference.weights_sha256("sam3-onnx") == inference.weights_sha256(tmp_path / "m") == sam3_onnx.weights_identity()[1]
    for w in ("sam3-onnx", str(tmp_path / "m")):
        assert sm.weights_label({"weights": w, "backend": "sam3"}) == "sam3-onnx-models-v0.6.0"

def test_old_sam3_pt_contours_are_stale_against_the_onnx_default(monkeypatch, tmp_path):  # Review Focus 4
    install_fake_osam(monkeypatch, tmp_path)
    spec = sm.normalize_model({"backend": "sam3", "prompt": "fish"})
    attrs = {"backend": "sam3", "weights": "sam3.pt", "weights_sha256": "0" * 64, "prompt": "fish"}
    assert any("weights" in p for p in sm.mismatches(attrs, spec, 5, weights_sha256=inference.weights_sha256(spec["weights"])))

def test_export_cli_sam3_needs_no_weights(monkeypatch, tmp_path): ...
    # parse ["--backend","sam3","--prompt","fish","--video",v,"--output",o]; args.weights == "sam3-onnx"

def test_export_cli_detectron2_still_needs_weights(): ...  # SystemExit, "--weights is required"

def test_prelabel_cli_defaults_to_onnx(): ...  # parse without --weights; weights == "sam3-onnx"
```

- [ ] **Step 2: Run to see them fail.** Run the four test files. Expected: FAIL.

- [ ] **Step 3: Implement.** In the export parser, default `--weights` to `sm.SAM3_ONNX` only after `--backend` is known (`if args.weights is None and args.backend == "sam3"`), before the existing `--weights is required` check; add `"sam3-onnx" only applies to --backend sam3` beside the `huggingface` check. Change `--weights` `type=Path` to `type=str` in both CLIs and convert to `Path` where a file is used, so the keyword is not mangled.

- [ ] **Step 4: Run** the four files plus `tests/segmentation_model_ui_test.py`. Expected: PASS.

- [ ] **Step 5: Commit** — `git commit -m "Record the ONNX weights by label and pinned hash; default the SAM 3 CLIs to them"`

---

### Task 5: Segmentation App — default, folder button, GPU check

**Files:**
- Modify: `src/idtrackerai/extra_tools/detectron2_pipeline/gpu.py` (`PROBE`, `GpuReport`)
- Modify: `src/idtrackerai/segmentation_app/widgets/sam3_panel.py` (weights state ~72, buttons ~108-150, `has_gpu` ~253, `uses_huggingface`/`_set_weights`/`choose_weights` ~320-355, command ~500-515, session `.toml` writer)
- Modify: `src/idtrackerai/segmentation_app/tooltips.toml` (keys `sam3_onnx_default`, `sam3_onnx_folder`; reword `sam3_huggingface` as the alternative)
- Test: `tests/sam3_panel_test.py`, `tests/gpu_test.py` if present (else in `tests/sam3_panel_test.py`)

**Interfaces:**
- Consumes: `sm.SAM3_ONNX`, `sm.is_onnx_weights` (Task 1).
- Produces: `PROBE` reports `"osam"` (version via `importlib.metadata`, not imported) and `"onnx_cuda"` (`"CUDAExecutionProvider" in onnxruntime.get_available_providers()`); `GpuReport.osam_version: str | None`, `GpuReport.onnx_cuda: bool`, `GpuReport.sam3_usable_for(weights) -> bool`, `GpuReport.sam3_missing_for(weights) -> list[str]`; `Sam3Panel.use_onnx()`, `Sam3Panel.choose_onnx_folder()`, `Sam3Panel.uses_onnx() -> bool`.

- [ ] **Step 1: Write the failing tests**

```python
def test_panel_opens_on_the_public_onnx_weights(panel):
    assert panel.uses_onnx() and "no token" in panel.weights_label.text()

def test_onnx_folder_then_back(panel, tmp_path, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path))
    panel.choose_onnx_folder(); assert panel.uses_onnx() and str(tmp_path) in panel.weights_label.text()
    panel.use_onnx(); assert str(panel.weights) == sm.SAM3_ONNX

def test_command_and_toml_carry_the_form(panel): ...
    # default: export command contains "--weights sam3-onnx"; session .toml has weights = "sam3-onnx"

def test_gpu_check_follows_the_engine():
    r = GpuReport(checked=True, osam_version="0.10.2", onnx_cuda=True, cuda_available=False, sam3_version=None)
    assert r.sam3_usable_for("sam3-onnx") and not r.sam3_usable_for("/x/sam3.pt")
    assert "pip install osam" in " ".join(GpuReport(checked=True).sam3_missing_for("sam3-onnx"))
```

- [ ] **Step 2: Run to see them fail.** Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/sam3_panel_test.py -q -p no:cacheprovider` — Expected: FAIL.

- [ ] **Step 3: Implement.** Keep `sam3_usable`/`sam3_missing` as the `.pt` answers so other callers are unchanged; `has_gpu` calls `sam3_usable_for(self.weights)`. Labels: default `"SAM 3 ONNX (public weights, no token), fetched at the first run"`, folder `"SAM 3 ONNX from <folder>"`. Buttons: **Use public ONNX**, **Choose ONNX folder**, **Choose sam3.pt**, **Use Hugging Face**. Changing weights re-runs `_refresh_enabled` and the GPU text.

- [ ] **Step 4: Run** `tests/sam3_panel_test.py tests/segmentation_model_ui_test.py`. Expected: PASS.

- [ ] **Step 5: Commit** — `git commit -m "Open the SAM 3 panel on the public ONNX weights, with a folder option"`

---

### Task 6: Packaging extra and the engine comparison command

**Files:**
- Modify: `pyproject.toml` (`[project.optional-dependencies]` add `sam3`; `[project.scripts]` add `idtrackerai_sam3_compare = "idtrackerai.extra_tools.detectron2_pipeline.sam3_compare:main"`)
- Create: `src/idtrackerai/extra_tools/detectron2_pipeline/sam3_compare.py`
- Test: `tests/sam3_compare_test.py`

**Interfaces:**
- Consumes: `make_sam3_predictor` (Task 3).
- Produces: `compare(frame: np.ndarray, a, b) -> list[dict]` — greedy one-to-one match of masks by IoU, each row `{"iou": float, "score_a": float|None, "score_b": float|None}`, unmatched masks listed with `iou` 0; `main()` with `--image`, `--prompt`, `--onnx` (default `sam3-onnx`), `--pt` (required path or `huggingface`), `--device`; prints one row per animal and the mean IoU.

- [ ] **Step 1: Write the failing test**

```python
def test_matching_pairs_by_iou_and_lists_the_unmatched():
    a = Stub([box(0,0,4,4), box(10,10,12,12)], [0.9, 0.8]); b = Stub([box(0,0,4,3)], [0.85])
    rows = sam3_compare.compare(np.zeros((20,20,3),np.uint8), a, b)
    assert rows[0]["iou"] == pytest.approx(20/25) and rows[1] == {"iou": 0.0, "score_a": 0.8, "score_b": None}
```

- [ ] **Step 2: Run to see it fail.** `QT_QPA_PLATFORM=offscreen python -m pytest tests/sam3_compare_test.py -q -p no:cacheprovider` — FAIL.
- [ ] **Step 3: Implement** `compare` and `main`; add the extra and the script entry.
- [ ] **Step 4: Run** it, plus `pip install -e . -q && idtrackerai_sam3_compare --help`. Expected: PASS; help lists the five flags.
- [ ] **Step 5: Commit** — `git commit -m "Add the [sam3] extra and idtrackerai_sam3_compare"`

---

### Task 7: Colab notebook, section 8

**Files:**
- Modify: `src/idtrackerai/extra_tools/detectron2_pipeline/notebooks/colab_detectron2_pipeline.ipynb` (cells 37–44 as of `745ff4f`)
- Test: `tests/sam3_notebook_test.py` (new)

**Interfaces:**
- Consumes: CLI defaults (Task 4), `tools/sam3_onnx.py` in the bundle (Task 3).
- Produces: a variable `SAM3_ENGINE = 'onnx'  # or 'pt'` in the SAM 3 settings cell; `SAM3_WEIGHTS` is `MODEL / 'sam3-onnx'` for `onnx`, `MODEL / 'sam3.pt'` for `pt`.

- [ ] **Step 1: Write the failing test** — load the notebook JSON; assert some code cell contains `SAM3_ENGINE`, `onnxruntime-gpu`, `hf download wkentaro/sam3-onnx-models-v0.6.0` (or `snapshot_download(repo_id="wkentaro/sam3-onnx-models-v0.6.0"`), and that every SAM 3 command cell passes `--weights {SAM3_WEIGHTS}`; and that the `HF_TOKEN` cell's text says it is only for `SAM3_ENGINE = 'pt'`.
- [ ] **Step 2: Run to see it fail.**
- [ ] **Step 3: Edit the notebook with a short Python script that loads/dumps JSON with `indent=1, ensure_ascii=False` plus a trailing newline** (the existing format; a round-trip is byte-identical). Install cell: `pip install -q osam` then `pip uninstall -y -q onnxruntime && pip install -q onnxruntime-gpu` for `onnx`, current `sam3` install for `pt`. Weights cell: for `onnx`, if `SAM3_WEIGHTS` lacks any `sam3_*.onnx`, `huggingface_hub.snapshot_download(repo_id="wkentaro/sam3-onnx-models-v0.6.0", local_dir=SAM3_WEIGHTS, allow_patterns=["*.onnx", "LICENSE"])` (no token); for `pt`, the #15 logic unchanged. Markdown "Weights" section leads with the ONNX route.
- [ ] **Step 4: Run** the new test and `tests/bundle_test.py tests/clip_names_dataset_test.py`. Expected: PASS.
- [ ] **Step 5: Commit** — `git commit -m "Colab: run SAM 3 on the public ONNX weights by default"`

---

### Task 8: Docs and changelog

**Files:**
- Modify: `README.md` (SAM 3 section ~151-181, licence ~278-281), `docs/sam3-pipeline.md` (Installing, Getting the weights, Where it runs), `docs/sam3-session-template.toml` (weights comment ~24), `docs/sam3-toml-handoff.md`, `docs/source/user_guide/changelog.rst`

- [ ] **Step 1:** Rewrite "Getting the weights" to lead with: nothing to do (`sam3-onnx`, fetched once), or a folder from `hf download wkentaro/sam3-onnx-models-v0.6.0 --local-dir models`; then `sam3.pt` (gated; file or `huggingface` + token) as the alternative, with when to prefer it (PyTorch/CUDA, reference implementation). Install section: `pip install osam` (or `idtrackerai[sam3]`), `onnxruntime-gpu` for CUDA. Licence note: the ONNX files are a conversion of Meta's weights under the SAM License. Mention `idtrackerai_sam3_compare`.
- [ ] **Step 2:** Template `.toml`: `weights` may be left out (= `"sam3-onnx"`), or a folder, `"huggingface"`, or a `sam3.pt` path.
- [ ] **Step 3:** Changelog bullet: SAM 3 now runs by default on the public ONNX conversion through osam, no Hugging Face token needed; `sam3.pt` remains an option; contour files record `sam3-onnx-models-v0.6.0`.
- [ ] **Step 4: Run** the whole suite: `QT_QPA_PLATFORM=offscreen python -m pytest tests -q -p no:cacheprovider --ignore=tests/smoke_test.py`. Expected: PASS.
- [ ] **Step 5: Commit** — `git commit -m "Document the no-token SAM 3 route"`
