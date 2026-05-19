# Synthetic demo dataset

This directory contains the synthetic CMS-1500 demo PDFs used by the
Veridoc walkthrough video and the live evaluation harness.

## Files

| File | Modality | Purpose |
|---|---|---|
| `cms1500_clean.pdf` | printed | Baseline: a clean printed CMS-1500 claim. |
| `cms1500_faxed.pdf` | fax | Same content, Otsu-binarised + salt-pepper noise + slight rotation. Exercises the `fax` modality path (see `src/preprocessing/image_enhancer.py`). |
| `cms1500_handwritten.pdf` | handwritten | Same form with patient-name + DOB + signature overlaid in a handwriting font. Exercises the `handwritten` modality path. |
| `synthea_patient.json` | — | The synthetic source data. Hand-crafted Synthea-style identifiers. |

## Provenance & licensing

**All identifiers are synthetic.** No real patient names, DOBs, SSNs, NPIs,
addresses, or phone numbers appear in these files. The data was hand-crafted
to look Synthea-shaped without depending on the 300 MB Synthea sample-data
pack as a build-time dependency.

The CMS-1500 form layout itself is a US federal document published by CMS
in the public domain. The form layout in `scripts/generate_demo_data.py`
is a faithful but simplified rendering — it's clearly synthetic when held
up next to a real claim, which is intentional: reviewers should never
mistake demo material for production output.

| Element | License / source |
|---|---|
| CMS-1500 form layout | Public domain (US Government) |
| Synthetic patient data | CC0 — created for this demo |
| Patient name "Mary Williams" | Synthetic |
| Provider NPI `1234567893` | Valid Luhn, CMS-canonical example NPI |
| CPT 99213, 87880 | Real codes used in valid ways |
| ICD-10 J06.9, R05.9 | Real codes used in valid ways |

## Regeneration

The PDFs are committed to the repo so reviewers can run the demo without
running the generator. They're also deterministic — re-running the
generator produces byte-identical output via a fixed seed.

```bash
python scripts/generate_demo_data.py
```

## End-to-end usage

```bash
# Healthcare mode — emits FHIR R4 Bundle alongside the standard exports.
python main.py extract data/demo/cms1500_clean.pdf --mode healthcare -o output/
python main.py extract data/demo/cms1500_faxed.pdf --mode healthcare -o output/
python main.py extract data/demo/cms1500_handwritten.pdf --mode healthcare -o output/

# General mode — plain JSON / Markdown / Excel; no FHIR emission.
python main.py extract data/demo/cms1500_clean.pdf --mode general -o output/

# Auto-detect (default) — the analyzer picks medical-rcm because the
# document carries the "HEALTH INSURANCE CLAIM FORM" header.
python main.py extract data/demo/cms1500_clean.pdf -o output/
```

The FHIR R4 Bundle lands at `output/<stem>/<stem>.fhir.json`.

## Modality detection

| Variant | Expected modality (from `src/agents/modality.py::derive_modalities`) |
|---|---|
| `cms1500_clean.pdf` | `printed`, `form` |
| `cms1500_faxed.pdf` | `printed`, `form`, `fax` |
| `cms1500_handwritten.pdf` | `printed`, `form`, `handwritten` |

If the analyzer mis-classifies, the operator can force the modality via
the upload UI's modality chips or the CLI `--modality` flag.
