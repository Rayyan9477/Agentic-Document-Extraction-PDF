# Specialised Medical Input Modes

> **Operator-facing deep-dive.** Canonical taxonomy lives in
> [VERIDOC_MASTER_PLAN.md → Appendix E](./VERIDOC_MASTER_PLAN.md#e-domain-modes--modality-vs-profile).
> This file goes one level deeper — detection heuristics, override
> wiring, and the "adding a new mode" runbook. If the two disagree,
> the master plan wins.

The system tags every document with one or more *modalities* — string
tags from a fixed set — and the image enhancer + extraction prompt
builder branch on those tags. Auto-detection happens during analysis;
callers can override via the API request or the upload UI.

## The 6 modes

| Mode | Detection rule | Image preprocessing | Extraction prompt fragment |
|---|---|---|---|
| `printed` | default; always present | deskew + Non-Local-Means denoise + CLAHE on LAB-luminance | base prompt — no extra rules |
| `handwritten` | analyzer flags `has_handwriting=true` | deskew + **gentle** denoise (h=3 vs 10), CLAHE skipped | "Treat all handwritten values as low-confidence; cap at 0.75 even when reading feels confident" |
| `visual` | low text density + no handwriting + no tables | deskew only — preserve gradients | "Image-dominant page (radiograph / ultrasound / photo). Do not invent structured fields. Only extract printed captions, legends, annotations." |
| `fax` | majority of pages low-contrast + low blur score + low quality score | deskew + **Otsu binarization + 2×2 morphological opening**; CLAHE & Non-Local-Means skipped | "1-bit fax scan. Strokes will be thin and broken; speckle noise around glyphs is normal. Verify each digit individually before committing numeric fields." |
| `table` | analyzer flags `has_tables=true` or `table_count > 0` | as printed | "Respect the table's column boundaries. Do not pull data across rows. Empty cells → null." |
| `form` | analyzer flags `layout_type == "form"` | as printed | "Treat each labelled box / numbered section as an independent label-value pair. Do not read across box boundaries." |

Multiple modes can be active simultaneously. A faxed handwritten form
ends up tagged `{"fax", "handwritten", "form"}`. The image enhancer
gives **`fax` priority** — once a document is 1-bit / CCITT-compressed,
the gentler "handwritten" preprocessing is moot — but the prompt
builder concatenates fragments for **all** active modes so the VLM
sees every relevant rule.

## Where the rules live

* Detection — [src/agents/modality.py](../src/agents/modality.py)
  (`derive_modalities` + `apply_overrides`)
* Image preprocessing branches — [src/preprocessing/image_enhancer.py](../src/preprocessing/image_enhancer.py)
  (`enhance(modes=[...])`)
* Prompt fragments — [src/prompts/extraction.py](../src/prompts/extraction.py)
  (`_MODALITY_PROMPT_FRAGMENTS`, `_build_modality_section`)
* State surface — `ExtractionState["modalities"]` and
  `ExtractionState["modality_override"]` in
  [src/pipeline/state.py](../src/pipeline/state.py)

## Overriding from the caller side

### Via API

```http
POST /api/v1/documents/upload
Content-Type: multipart/form-data
...
modality_override: ["fax", "handwritten"]
```

The override list is **JSON-encoded** in the form-data body. Empty
list / unset field = "auto-detect".

### Via the frontend

`Documents → Upload → Configure → Specialised Mode` shows six toggle
chips. None selected = auto-detect (recommended). The hint string on
each chip describes what the mode does to preprocessing and prompts.

### Via the CLI

The CLI does not expose modality overrides yet (auto-detect only).
This is a planned follow-up; until then, run the API path or the
schema wizard for explicit overrides.

## Detection heuristics

`derive_modalities` is a pure function — no extra VLM calls. It
combines:

1. **Analyzer signals** — `has_handwriting`, `has_tables`,
   `table_count`, `has_signatures`, `layout_type`, `text_density`.
   These come from `_analyze_structure` in
   [src/agents/analyzer.py](../src/agents/analyzer.py).
2. **Per-page image-quality metrics** — produced by
   `ImageEnhancer.analyze_quality` during preprocessing, captured into
   `ExtractionState["image_quality"]`. The `fax` heuristic requires
   *most* pages to be low-contrast + low-blur + low-quality before
   tagging the document; it deliberately doesn't fire on a single
   noisy page.

### Conservative fallback

When the analyzer hasn't run yet (e.g. orchestrator called the
splitter only), `derive_modalities({})` returns `["printed"]` as a
safe default. The prompt builder treats `["printed"]` as "no extra
fragments", so behaviour is identical to the legacy non-modality path.

## Adding a new mode

1. Add the constant + entry in `ALL_MODES` in `src/agents/modality.py`.
2. Add the detection rule in `derive_modalities`.
3. Add the prompt fragment to `_MODALITY_PROMPT_FRAGMENTS` in
   `src/prompts/extraction.py`.
4. (Optional) Add an image-enhancement branch in `ImageEnhancer.enhance`.
5. Add a chip in
   `frontend/src/components/documents/UploadOptions.tsx`'s
   `MODALITY_CHIPS` list.
6. Add a test case in `tests/unit/test_modality.py`.

## Why this is split into preprocessing AND prompt rules

Both layers have to know about modality because they each fix a
*different* class of error:

* **Preprocessing** fixes pixel-level problems the VLM cannot
  recover from (e.g. CLAHE on a fax destroys glyphs; Non-Local-Means
  on handwriting eats pen strokes).
* **Prompts** fix interpretation-level problems (e.g. the VLM will
  happily invent a "patient name" field on a chest X-ray unless told
  not to).

You need both. The image enhancer can't tell the VLM "treat this as
fax-grade input"; the prompt can't repair pixels that have already
been over-processed.
