# Project Status

> Shipping reality at the latest merge. For the full design and the
> Phase 9-14 forward plan, see
> [VERIDOC_MASTER_PLAN.md](./VERIDOC_MASTER_PLAN.md) — that is the
> single canonical reference. This file is intentionally narrow: what
> is wired, what is flag-gated, what is *not* shipping yet.

## At a glance

```
Backend           2575 unit + integration tests · all green
Frontend          Next.js 14 · TypeScript clean · schema chooser + modality + PHI live
CI                .github/workflows/ci.yml runs ruff + mypy + pytest matrix on every PR
LangGraph         v3 (langgraph >= 1.0) with Command + interrupt + durable SQLite checkpointer
Phases shipped    0 through 8 (V3 core + Phase 8 enterprise-MVP hardening)
Next phase        9 — backend pivot: vLLM retired, AWS Bedrock in, MI300X out
Default LM        LM Studio @ localhost:1234 (model-id is operator-chosen)
Cloud LM          AWS Bedrock (Phase 9 — landing; not yet default)
```

## Phase ledger

See [master plan §6](./VERIDOC_MASTER_PLAN.md#6-phase-status-ledger)
for the full ledger with test counts. Summary:

| Phase | Subject | State |
|---|---|---|
| 0 | VLM backend abstraction | ✅ shipped |
| 1 | Constrained decoding | ✅ shipped |
| 2 | Heterogeneous dual-VLM | ✅ shipped |
| 3 | Critic agent + bbox round-trip | ✅ shipped |
| 4 | Provenance threading | ✅ shipped |
| 5 | Profiles (Medical-RCM) + fax hardening | ✅ shipped |
| 6 | Eval harness + calibration + observability | ✅ shipped |
| 7 | Production hardening (PHI / audit / multi-tenant) | ✅ shipped |
| 8 | Enterprise-MVP hardening (audit-driven) | ✅ shipped |
| 9 | Backend pivot (vLLM → Bedrock; MI300X retired) | planned |

## Feature ledger

Every row is wired into the running pipeline (not just present in
source). Where any older plan disagrees, this table wins.

### Pipeline & agents

| Feature | State | Anchor |
|---|---|---|
| LangGraph harness | ✅ v3 (Command, interrupt, durable SQLite checkpointer default) | [src/agents/orchestrator.py](../src/agents/orchestrator.py) |
| Splitter agent (multi-document boundary detection) | ✅ default-on | [src/agents/splitter.py](../src/agents/splitter.py) |
| TableDetector agent | ✅ default-on | [src/agents/table_detector.py](../src/agents/table_detector.py) |
| ModelRouter (multi-model dispatch) | ✅ wired into `BaseAgent.send_vision_request` | [src/client/model_router.py](../src/client/model_router.py), [src/agents/base.py](../src/agents/base.py) |
| Constrained decoding (Pydantic-bound structured output) | ✅ Phase 1 | [src/client/constrained.py](../src/client/constrained.py) |
| Heterogeneous dual-VLM (primary + secondary) | ✅ Phase 2 | [src/client/model_router.py](../src/client/model_router.py) |
| Critic agent + bbox round-trip verification | ✅ Phase 3 | [src/agents/](../src/agents/) |
| Provenance threading into all exports | ✅ Phase 4 | [src/pipeline/provenance.py](../src/pipeline/provenance.py) |
| HeterogeneousReconciler (per-field reconciliation) | ✅ | [src/agents/reconciler.py](../src/agents/reconciler.py) |
| Profiles (generic-document + medical-rcm) | ✅ Phase 5 | [src/profiles/](../src/profiles/) |
| ConfidenceCalibrator (Platt / isotonic / partitioned) | ✅ default-on | [src/validation/calibration.py](../src/validation/calibration.py) |
| Hallucination-injection harness | ✅ Phase 6 | tests/eval/inject/ |
| Human-in-the-loop (`interrupt()` / `Command(resume=...)`) | ✅ thread-id tenant isolation | [src/agents/orchestrator.py](../src/agents/orchestrator.py) |
| Cross-field validation | ✅ called from validator | [src/validation/cross_field.py](../src/validation/cross_field.py) |

### Specialised medical input modes (modality)

Detected automatically by the analyzer; user-overridable via
`ProcessRequest.modality_override` or the upload-page chip selector.
Full mode reference in [master plan Appendix E](./VERIDOC_MASTER_PLAN.md#e-domain-modes--modality-vs-profile).

### PHI mode

Opt-in. Off by default. When enabled, every extracted string is
routed through the `openai/privacy-filter` HuggingFace token
classifier (BIOES tagging, 8 PII categories) before storage / export
/ Mem0 / audit log. Regex fallback when the `[phi]` extra is absent.
Production refuses to boot with PHI off unless `PHI_BYPASS_ACK` is
set (Phase 7).

See [PHI_MODE.md](./PHI_MODE.md) for setup and per-request override.

### Observability

One `ObservabilityDispatcher` fans out to two opt-in sinks:

* **Arize Phoenix** — local OpenInference / OpenTelemetry tracing
  via `phoenix.launch_app()`; auto-instruments LangGraph + the
  OpenAI SDK used by the LM Studio client.
* **PostHog** — product analytics (extraction events, VLM call
  counts, decision routing).

Both off by default. See [OBSERVABILITY.md](./OBSERVABILITY.md) and
[master plan §4 / Appendix F](./VERIDOC_MASTER_PLAN.md#4-cross-cutting-concerns).

### Webhooks

* HMAC-SHA256 signing
* SSRF guards (scheme + private-network blocklist, Phase 8)
* Exponential-backoff retry (3 attempts in-line)
* **SQLite-backed Dead Letter Queue** with capped exponential
  backoff (10s → 24h cap), poison-message auto-disable (Phase 8)
* Admin endpoints: `GET /api/v1/webhooks/{id}/dlq` + `POST
  /api/v1/webhooks/{id}/dlq/{entry_id}/redeliver`

### Exports

| Format | Notes |
|---|---|
| JSON · MINIMAL / STANDARD / DETAILED | values, confidence, validation, audit trail |
| JSON · **DATAFRAME_FLAT** | one row per (record × field) for `pandas.read_json` |
| JSON · FHIR_COMPATIBLE | inline DocumentReference (legacy) |
| **FHIR R4 Bundle** | validated when `[fhir]` extra installed; Patient + Coverage + Claim for CMS-1500/UB-04, Patient + ExplanationOfBenefit for EOB |
| Excel | 4 sheets: All Records, Duplicates, Page Summary, Processing Summary |
| Markdown | SIMPLE / DETAILED / SUMMARY / TECHNICAL + **Decision Trail** section |
| **Bbox overlay PNGs** | per-page confidence-coloured rectangles (green ≥ 0.85, amber 0.5-0.85, red < 0.5) |
| RCM signing manifest | signed export bundle (Phase 7/8) — [src/export/rcm_signing.py](../src/export/rcm_signing.py) |

### Security & compliance

| Item | State |
|---|---|
| Dev-token bypass | ✅ removed (backend + frontend) |
| `/health/*` endpoints | ✅ liveness public; detail / security / alerts / dependencies require `system:metrics` |
| Default checkpointer | ✅ SQLite under `.extraction_checkpoints/` (gitignored) |
| Redis defaults | ✅ TLS + AUTH warnings; `result_expires` 1h |
| `mask_phi` enforcement | ✅ wired through consolidated_export + CLI `--mask-phi` |
| Schema-wizard prompt-injection sanitiser | ✅ `_sanitize_schema_text` in `build_field_prompt` |
| NPI Luhn | ✅ |
| RBAC | 7 roles, JWT issuer claim, JTI revocation, key-owner enforcement on revoke (Phase 8) |
| AES-256-GCM encryption | PBKDF2 600k / Scrypt 2^14, key entropy validation |
| PHI masking in logs | 13 regexes via structlog + stdlib filter |
| Audit chain hashing + sidecar anchor (Phase 8) | ✅ `verify_audit_chain_with_anchor()` |
| Multi-tenant | ✅ `TenantResolverMiddleware` (Phase 8); per-tenant FAISS / calibration / audit / checkpoints |
| Production-boot guards | ✅ refuses to start with `auth_enabled=False` (without `AUTH_BYPASS_ACK`) or `phi.enabled=False` (without `PHI_BYPASS_ACK`) |

## Known gaps (truthful list)

Not yet shipping. Listed so docs match reality:

* **vLLM and MI300X are still in the codebase.** Phase 9 removes
  them entirely; until then, the backend abstraction still carries
  unused vLLM glue. The default-shipping path is LM Studio.
* **AWS Bedrock backend is Phase 9.** Not yet wired. The master plan
  Part III, Phase 9 captures the full landing plan.
* **CI mypy step uses `|| true`.** The build doesn't fail on type
  errors yet — drop the pipe once `mypy src --strict` is clean.
* **Streamlit UI** is legacy; the canonical UI is the Next.js app.
* **Docker** images / compose were removed in commit `5a4e521` and
  have not been reintroduced. Container deployment is out of scope
  until requested.
* **Florence-2 second LM Studio instance:** the `ModelRouter`
  supports it (registers two `ModelConfig`s) but operators must
  stand up the second port; default is single-model.
* **EDI 837/835 write path** is not implemented (read-only).
* **Property / contract / performance test directories** aren't
  comprehensive yet; the cross-cutting suites are next.

## How to run things

```bash
# Install
pip install -e ".[dev]"

# Optional extras
pip install -e ".[dev,phi]"            # PHI redaction (transformers + torch)
pip install -e ".[dev,observability]"  # Phoenix + PostHog
pip install -e ".[dev,fhir]"           # validated FHIR R4
pip install -e ".[dev,profiles-rcm]"   # Medical-RCM emitters (C-CDA, X12N 275)

# Test
pytest tests/ -m "not slow"            # 2575 tests

# Run
python main.py extract path/to/doc.pdf
python main.py extract doc.pdf --mask-phi
python main.py                          # web app (backend + frontend)

# Observability (when enabled in settings)
# Phoenix UI: http://localhost:6006
```
