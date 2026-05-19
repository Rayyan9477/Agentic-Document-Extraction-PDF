---
title: Veridoc — Documentation index
audience: developers, operators, reviewers
last_updated: 2026-05-19
---

# Veridoc — Documentation index

Veridoc is an open-source, on-prem-first / cloud-capable document-intelligence
engine. It uses heterogeneous dual-VLM extraction with provenance threading and
a six-layer validation stack, structured around two orthogonal axes — **modality**
(what kind of input) and **profile** (what semantic schema). A generic baseline
runs out of the box; the medical-RCM profile (CMS-1500 / UB-04 / EOB / superbill)
is the first specialization. **LM Studio** and **AWS Bedrock** are parallel
first-class backends. Phase 8 is complete (2,853 passing tests); **Phase 9 — the
Bedrock backend pivot — is next**.

Veridoc is positioned against **Landing AI ADE**, **Pulse**, **Reducto**, and
**LlamaParse** in the closed SaaS tier, and **Docling**, **Marker**, and
**Unstructured** in the open-parser tier. See
[`PRODUCT_OVERVIEW.md`](./PRODUCT_OVERVIEW.md) for the competitive landscape and
the differentiator table.

> [!TIP]
> If you only have 30 seconds: read `STATUS.md` for what works today, then
> `VERIDOC_MASTER_PLAN.md` §1–3 for the product framing.

## Documentation map

```mermaid
%% Doc-tree overview: canonical reference, operational deep-dives, and frozen history.
flowchart LR
    subgraph Canonical
        MP["VERIDOC_MASTER_PLAN.md<br/>Canonical product reference"]
    end
    subgraph Operational
        ST["STATUS.md<br/>Shipping reality"]
        MO["MODES.md<br/>Modality / profile axes"]
        OB["OBSERVABILITY.md<br/>Phoenix / PostHog / audit"]
        PH["PHI_MODE.md<br/>Opt-in PHI redaction"]
    end
    subgraph External["Evaluator-facing"]
        PO["PRODUCT_OVERVIEW.md<br/>Competitive landscape"]
        DS["DEMO_SCRIPT.md<br/>90-second walkthrough"]
        DR["DEMO_READINESS.md<br/>Operational gate snapshot"]
    end
    subgraph Historical
        PR["archive/PRD.md<br/>Origin requirements (frozen)"]
    end

    MP --- ST
    MP --- MO
    MP --- OB
    MP --- PH
    MP --- PO
    PO --- DS
    PO --- DR
    MP -.historical context.-> PR

    click MP "VERIDOC_MASTER_PLAN.md"
    click ST "STATUS.md"
    click MO "MODES.md"
    click OB "OBSERVABILITY.md"
    click PH "PHI_MODE.md"
    click PO "PRODUCT_OVERVIEW.md"
    click DS "DEMO_SCRIPT.md"
    click DR "DEMO_READINESS.md"
    click PR "archive/PRD.md"

    classDef primary fill:#1e40af,stroke:#1e3a8a,color:#fff,stroke-width:2px
    classDef shipped fill:#059669,stroke:#064e3b,color:#fff,stroke-width:2px
    classDef external fill:#0891b2,stroke:#155e75,color:#fff,stroke-width:2px
    classDef future fill:#9ca3af,stroke:#4b5563,color:#000,stroke-width:2px
    class MP primary
    class ST,MO,OB,PH shipped
    class PO,DS,DR external
    class PR future
```

## Where do I start?

```mermaid
%% Persona-based router: pick a starting doc by reader intent.
flowchart TB
    Q{"Who are you?"}
    Q -->|"New — what is Veridoc?"| N["VERIDOC_MASTER_PLAN.md<br/>§1 Vision · §2 Architecture · §3 Axes"]
    Q -->|"Operator deploying it"| O["STATUS.md → then<br/>MODES · OBSERVABILITY · PHI_MODE"]
    Q -->|"Developer building Phase 9+"| D["VERIDOC_MASTER_PLAN.md<br/>Part III — Roadmap"]
    Q -->|"Compliance / security review"| C["PHI_MODE.md +<br/>VERIDOC_MASTER_PLAN.md §4 + Audit appendix"]
    Q -->|"Specific feature deep-dive"| F["VERIDOC_MASTER_PLAN.md<br/>Part IV — Appendices A–I"]
    Q -->|"Is it working today?"| S["STATUS.md"]

    classDef primary fill:#1e40af,stroke:#1e3a8a,color:#fff,stroke-width:2px
    classDef shipped fill:#059669,stroke:#064e3b,color:#fff,stroke-width:2px
    classDef planned fill:#7c3aed,stroke:#5b21b6,color:#fff,stroke-width:2px
    classDef warning fill:#f59e0b,stroke:#b45309,color:#000,stroke-width:2px
    classDef data fill:#475569,stroke:#1e293b,color:#fff,stroke-width:2px
    class Q primary
    class N,F data
    class O,S shipped
    class D planned
    class C warning
```

## Document roles

| Document | Purpose | When to read | Update cadence |
|---|---|---|---|
| [`VERIDOC_MASTER_PLAN.md`](VERIDOC_MASTER_PLAN.md) | Canonical product reference | First read; phase planning | On phase merge |
| [`STATUS.md`](STATUS.md) | Current shipping reality | Before any work | Every PR that lands |
| [`PRODUCT_OVERVIEW.md`](PRODUCT_OVERVIEW.md) | Competitive landscape + differentiators | Evaluator / reviewer first-touch | When competitor positioning changes |
| [`MODES.md`](MODES.md) | Modality + profile detection deep-dive | Adding a mode/profile | When axes change |
| [`OBSERVABILITY.md`](OBSERVABILITY.md) | Telemetry surface ops | Setting up Phoenix/PostHog | When event names change |
| [`PHI_MODE.md`](PHI_MODE.md) | PHI redaction enablement | HIPAA-grade deployments | When PHI flow changes |
| [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) | 90-second demo walkthrough storyboard | Recording a demo video | When the demoable surface shifts |
| [`DEMO_READINESS.md`](DEMO_READINESS.md) | Operational readiness snapshot | Before a live demo | After a demo verification pass |
| [`archive/PRD.md`](archive/PRD.md) | Origin requirements (pre-Phase-9) | Historical context only | Frozen |

## Conventions used in these docs

> [!NOTE]
> **Mermaid diagrams** render natively on GitHub, GitLab, VS Code (with the
> Markdown Mermaid extension), and MkDocs. No client install needed.

> [!IMPORTANT]
> **GitHub admonitions** — these docs use the GitHub flavour:
> `> [!NOTE]`, `> [!TIP]`, `> [!IMPORTANT]`, `> [!WARNING]`, `> [!CAUTION]`.
> They degrade gracefully to blockquotes on renderers that don't support them.

**Phase numbering.** Phases 0–8 are **shipped** (rendered emerald/green in
diagrams). Phases 9–13 are **planned** (purple). Phase 14 is **future** (gray).
The canonical phase ledger lives in `VERIDOC_MASTER_PLAN.md` §6.

### Semantic color palette

| Class | Role | Hex |
|---|---|---|
| `primary` | Headline / entry node | `#1e40af` |
| `validation` | Passing / validated path | `#16a34a` |
| `warning` | Caution / partial | `#f59e0b` |
| `blocker` | Blocking issue | `#dc2626` |
| `data` | Data / reference node | `#475569` |
| `shipped` | Shipped phase / feature | `#059669` |
| `planned` | Planned (next phases) | `#7c3aed` |
| `future` | Future / aspirational | `#9ca3af` |
| `external` | External system / SaaS | `#0891b2` |

```mermaid
%% Visual swatch of the 9 semantic classes used across every diagram in this tree.
flowchart LR
    P["primary<br/>#1e40af"]:::primary
    V["validation<br/>#16a34a"]:::validation
    W["warning<br/>#f59e0b"]:::warning
    B["blocker<br/>#dc2626"]:::blocker
    D["data<br/>#475569"]:::data
    S["shipped<br/>#059669"]:::shipped
    PL["planned<br/>#7c3aed"]:::planned
    F["future<br/>#9ca3af"]:::future
    E["external<br/>#0891b2"]:::external

    classDef primary fill:#1e40af,stroke:#1e3a8a,color:#fff,stroke-width:2px
    classDef validation fill:#16a34a,stroke:#14532d,color:#fff,stroke-width:2px
    classDef warning fill:#f59e0b,stroke:#b45309,color:#000,stroke-width:2px
    classDef blocker fill:#dc2626,stroke:#7f1d1d,color:#fff,stroke-width:2px
    classDef data fill:#475569,stroke:#1e293b,color:#fff,stroke-width:2px
    classDef shipped fill:#059669,stroke:#064e3b,color:#fff,stroke-width:2px
    classDef planned fill:#7c3aed,stroke:#5b21b6,color:#fff,stroke-width:2px
    classDef future fill:#9ca3af,stroke:#4b5563,color:#000,stroke-width:2px
    classDef external fill:#0891b2,stroke:#155e75,color:#fff,stroke-width:2px
```

## Quick links

- Current test status & gap list — [`STATUS.md`](STATUS.md)
- Competitive positioning — [`PRODUCT_OVERVIEW.md`](PRODUCT_OVERVIEW.md)
- Phase status ledger — [`VERIDOC_MASTER_PLAN.md` §6](VERIDOC_MASTER_PLAN.md#6-phase-status-ledger)
- Phase 9 (Bedrock backend pivot) plan — [`VERIDOC_MASTER_PLAN.md` — Phase 9](VERIDOC_MASTER_PLAN.md#phase-9--backend-pivot)
- Architecture appendices (A–I) — [`VERIDOC_MASTER_PLAN.md` Part IV](VERIDOC_MASTER_PLAN.md#part-iv--appendices)
- 90-second demo storyboard — [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md)
- Demo operational readiness — [`DEMO_READINESS.md`](DEMO_READINESS.md)
