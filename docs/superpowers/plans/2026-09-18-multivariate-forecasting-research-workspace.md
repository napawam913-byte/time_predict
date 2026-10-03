# 多变量时序预测研究工作区 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a traceable learning and literature-research workspace for general multivariate time-series forecasting.

**Architecture:** The project root carries persistent collaboration context, while `related_work/` is a paper-per-directory evidence store. Each paper directory keeps the immutable source material separate from a Chinese reading note and reproduction contract; the cross-paper index is the only comparative view.

**Tech Stack:** Markdown, Bash file checks, PDF metadata tools, official arXiv/CVF/OpenReview paper sources.

**Spec:** `docs/superpowers/specs/2026-09-18-multivariate-forecasting-research-design.md`

## Global Constraints

- Scope is general multivariate long-horizon forecasting only.
- Retain official paper PDFs and LaTeX sources where legally available; never fabricate unavailable sources.
- Use chronological train/validation/test splits and fit preprocessing on training data only.
- Keep reported-paper results and local-reproduction results separate.
- Do not overwrite experiment history; append named configurations and failed runs too.

---

## File Structure

- `AGENTS.md`: project objective, reading order, non-negotiable research rules, session startup checklist.
- `handoff.md`: concise durable state and the next action for a new session.
- `brainstorm.md`: append-only hypotheses and research questions.
- `related_work/README.md`: literature map, comparison table, reading sequence, source policy.
- `related_work/<paper>/paper.pdf`: publisher-hosted paper copy.
- `related_work/<paper>/source/`: paper LaTeX source archive expanded in place.
- `related_work/<paper>/README.md`: Chinese structured note and reproduction contract.
- `related_work/<paper>/provenance.md`: immutable source URLs, access date and checksums.
- `reproduction/`, `experiment/`, `DataSet/`, `dataset.md`: reserved workflow stages with documented entry points.

### Task 1: Initialize the minimal research workflow

**Files:**
- Create: `AGENTS.md`, `handoff.md`, `brainstorm.md`, `dataset.md`
- Create: `experiment/results.md`, `experiment/evaluation.md`, `related_work/README.md`
- Create directories: `related_work/`, `reproduction/`, `experiment/`, `DataSet/`

- [x] **Step 1: Write the project contract**

Place the target problem, mandatory chronological split rule, learning-first reading order and session handoff rule in `AGENTS.md`.

- [x] **Step 2: Create append-only research records**

Initialize `brainstorm.md` with the three research questions from the spec and `experiment/results.md` with a table that keeps run name, commit, dataset split, horizon, metrics and notes.

- [x] **Step 3: Define evaluation before experiments**

Create `experiment/evaluation.md` with fixed horizons, MAE/MSE/RMSE criteria, per-variable reporting requirement and leakage checks.

- [x] **Step 4: Verify the workflow skeleton**

Run:

```bash
test -f AGENTS.md && test -f handoff.md && test -f related_work/README.md && test -f experiment/evaluation.md
```

Expected: exit status `0`.

### Task 2: Acquire source material with provenance

**Files:**
- Create: `related_work/{01_informer_2021,02_autoformer_2021,03_dlinear_2023,04_patchtst_2023,05_timesnet_2023,06_itransformer_2024}/paper.pdf`
- Create: matching `source/` and `provenance.md`

- [x] **Step 1: Use official primary endpoints**

Retrieve each PDF and LaTeX source from its official arXiv identifier; record the exact URL and retrieval date in `provenance.md`.

- [x] **Step 2: Validate each download**

Run:

```bash
pdfinfo related_work/<paper>/paper.pdf | rg '^Pages|^Title'
find related_work/<paper>/source -type f | head
sha256sum related_work/<paper>/paper.pdf
```

Expected: a readable PDF, nonempty source tree, and a checksum recorded in provenance.

- [x] **Step 3: Preserve source caveats**

When an official source endpoint is unavailable, leave `source/README.md` explaining the failure and retain the PDF/provenance rather than inserting reconstructed source.

### Task 3: Write paper-level research notes

**Files:**
- Create: `related_work/00_foundations/README.md`
- Create: `related_work/<paper>/README.md` for each of six papers

- [x] **Step 1: Establish the common note schema**

Each paper note must have: citation and links, problem formulation, central idea, architecture/data flow, claimed evidence, assumptions/limitations, comparison to prior papers, reproduction contract, and questions to verify.

- [x] **Step 2: Apply beginner-oriented analysis**

Explain terms on first use, state what changes relative to the previous work, and distinguish an author claim from an independently verified conclusion.

- [x] **Step 3: Derive reproducible contracts**

For each paper state candidate data sets, input length, forecasting horizons, split policy, reported metric(s), non-negotiable preprocessing details and the smallest implementation milestone.

- [x] **Step 4: Check note completeness**

Run:

```bash
for f in related_work/*/README.md; do rg -q '复现契约' "$f" && rg -q '局限' "$f"; done
```

Expected: exit status `0`.

### Task 4: Synthesize the literature into a learning path

**Files:**
- Modify: `related_work/README.md`, `handoff.md`, `brainstorm.md`

- [x] **Step 1: Build the comparison matrix**

Compare tokenization unit, temporal inductive bias, cross-variable interaction, computational motivation, benchmark protocol and recommended reproduction priority.

- [x] **Step 2: Provide an ordered learning route**

Place foundations and DLinear before Transformer variants, then Autoformer/Informer, PatchTST/TimesNet and iTransformer. Give each item one concrete learning outcome.

- [x] **Step 3: Record next research action**

Set `handoff.md` to download one public benchmark, implement Naive plus DLinear, and log the first leakage-safe baseline before attempting complex models.

### Task 5: Final provenance and workflow audit

**Files:**
- Modify: all `provenance.md`, `related_work/README.md`, `handoff.md`

- [x] **Step 1: Check the source inventory**

Run:

```bash
find related_work -name paper.pdf -print | wc -l
find related_work -name provenance.md -print | wc -l
```

Expected: `6` PDFs and `6` provenance records, plus the foundations entry.

- [x] **Step 2: Search for unfinished research artifacts**

Run:

```bash
rg -n 'TODO|TBD|待补充|假装|fabricat' AGENTS.md handoff.md related_work experiment dataset.md
```

Expected: no matches, except an explicit unavailable-source explanation in a provenance file.

- [x] **Step 3: Confirm the first reproducible next step**

Check that `handoff.md` names a public data set, baseline, split rule and output metrics; no claim of successful model reproduction may appear before a run is logged in `experiment/results.md`.
