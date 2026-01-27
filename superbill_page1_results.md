# Medical Patient List Extraction Report

## Table of Contents

- [Executive Summary](#executive-summary)
- [Processing Metadata](#processing-metadata)
- [Extracted Data](#extracted-data)
- [Validation Results](#validation-results)
  - [Field Validations](#field-validations)
- [Audit Trail](#audit-trail)
  - [Timeline](#timeline)
  - [Processing Details](#processing-details)


*Processing ID: `test_001`*

## Executive Summary

**Status**: Completed

**Overall Confidence**: ✅ 98.0%

**Fields Extracted**: 11

**Validation**: ✅ Passed

## Processing Metadata

- **Processing ID**: `test_001`
- **PDF Path**: `superbill1.pdf`
- **PDF Hash**: `a1b2c3d4e5f6...`
- **Document Type**: Medical Patient List
- **Schema**: VLM Adaptive Schema
- **Status**: Completed
- **Start Time**: 2024-05-15T10:00:00Z
- **End Time**: 2024-05-15T10:01:00Z

## Extracted Data

| Field | Value | Confidence | Status |
|-------|-------|------------|--------|
| Cpt Codes | 45380 - Colonoscopy with biopsy
46500 - Injection to hemorrhoids
G8918 - No preop IV antibiotics needed.
G8907 - No fall, burn, hospital transfer or wrong patient or site. | 85% |  Valid |
| Findings | Normal (Anal Canal)
First degree hemorrhoids (Rectum)
Four quadrants of the analrectal junction were injected with 0.5 ml of tetradecyl sulfate for a total of 2 ml. (Rectum)
Sessile Polyp . Polypectomy done using biopsy forceps (Sigmoid Colon)
Normal (Descending Colon)
Normal (Splenic Flexure)
Normal (Transverse Colon)
Normal (Hepatic Flexure)
Normal (Ascending Colon)
Normal (Ileocecal Valve)
Normal (Cecum)
Not Seen (Ileum) | 85% |  Valid |
| Icd Codes | Polyp of sigmoid colon - D12.5
First degree hemorrhoids - K64.0 | 85% |  Valid |
| Indications | History of colonic polyps - Z86.010 | 85% |  Valid |
| Patient Dob | 05/28/1937 | 85% |  Valid |
| Patient Id | MARESC002 | 85% |  Valid |
| Patient Name | Escamilla, Maria A. | 85% |  Valid |
| Patient Sex | (F) | 85% |  Valid |
| Plan | Wait for pathology report
High fiber diet needed
Follow up colonoscopy in 5 years | 85% |  Valid |
| Referring Physician | Yazji, Monzer, M.D. | 85% |  Valid |
| Sedation | Propofol 160 mg IV | 85% |  Valid |

## Validation Results

**Overall Status**:  Passed

### Field Validations
-  **Cpt Codes** (format)
-  **Findings** (format)
-  **Icd Codes** (format)
-  **Indications** (format)
-  **Patient Dob** (format)
-  **Patient Id** (format)
-  **Patient Name** (format)
-  **Patient Sex** (format)
-  **Plan** (format)
-  **Referring Physician** (format)
-  **Sedation** (format)

## Audit Trail

### Timeline
- **Started**: 2024-05-15T10:00:00Z
- **Completed**: 2024-05-15T10:01:00Z
- **Duration**: 55.15 seconds

### Processing Details
- **VLM Calls**: 5
- **Retry Count**: 0
- **Pages Processed**: 1

*Report generated: 2026-01-27T11:10:29.680131+00:00*