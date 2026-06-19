# AI Agent README: Zyro Dynamics HR Help Desk RAG

This document is a practical guide for an AI agent working inside this repository. It explains what the project does, which files matter, how the Kaggle competition flow works, and which constraints must be preserved when making changes.

## Project Purpose

This repository implements a Retrieval-Augmented Generation (RAG) system for the Zyro Dynamics HR Help Desk Kaggle InClass challenge.

The project has three main jobs:

1. Answer HR policy questions from the official Zyro Dynamics documents.
2. Refuse unsupported, non-HR, or sensitive requests safely.
3. Generate the exact encrypted Kaggle competition submission format required by the starter notebook.

The system can run as:

- a Streamlit chatbot
- a generic batch submission generator
- the official encrypted competition submission generator
- an evaluation and tuning pipeline
- a controlled-answer variant builder for leaderboard experiments

## Where The Real Project Root Is

The workspace root contains a wrapper folder:

- workspace root: `d:\kaggle project\langchain-rag-nxtwave`
- actual project root: `d:\kaggle project\langchain-rag-nxtwave\langchain-rag-nxtwave-main`

Most commands should be run from `langchain-rag-nxtwave-main`.

## What Success Looks Like

For this repo, "done" usually means one of these:

- the Streamlit app answers from the official HR corpus only
- the official competition submission CSV is valid and encrypted correctly
- a controlled submission variant changes only the requested answers and preserves all locked fields
- evaluation outputs are written and interpretable
- retrieval or prompting changes improve quality without breaking the competition contract

## Hard Constraints

An AI agent should treat these as project invariants unless the user explicitly asks otherwise.

### Competition contract

- The official submission must contain exactly 15 rows: `Q01` to `Q15`.
- The official Kaggle CSV must contain exactly these columns:
  - `question_id`
  - `question_enc`
  - `answer_enc`
  - `streamlit_link`
  - `langsmith_link`
- `question_enc` values come from the official starter notebook and must not be altered during controlled override work.
- `streamlit_link` and `langsmith_link` are part of the competition contract and are often locked during experiments.
- Answers must be encrypted with the repository's existing Fernet flow.

### Corpus integrity

- `hr_docs/official/` must contain exactly the official 11 PDF files.
- Filenames and SHA-256 hashes are validated.
- Adding, removing, or editing these files breaks corpus validation.

### Official assets that should stay unchanged

- `competition/Starter_Notebook.ipynb`
- `hr_docs/official/*`
- encryption logic in `generate_competition_submission.py`

### Answer quality rules

- Do not invent policy facts that are not in the supplied documents.
- Do not add unsupported details just because they sound plausible.
- For unsupported questions, prefer a clean refusal grounded in the document scope.
- For official competition answers, formatting matters: avoid markdown bullets, tables, or decorative formatting unless the workflow explicitly expects it.

## High-Level Architecture

The main pipeline lives in `hr_rag/pipeline.py`.

At a high level:

1. Load and validate HR policy documents.
2. Split them into chunks.
3. Build embeddings.
4. Store vectors in Chroma or a fallback in-memory store.
5. Build a keyword index.
6. Run guardrails before retrieval.
7. Retrieve with hybrid search:
   - vector similarity
   - MMR diversification
   - BM25-style keyword matching
   - weighted Reciprocal Rank Fusion
8. Optionally apply HyDE for vague questions.
9. Generate a grounded answer from retrieved chunks.
10. Optionally refine low-confidence answers.
11. Append citations for interactive/debug use.

## Key Files And Why They Matter

### Core pipeline

- `hr_rag/pipeline.py`
  - Main RAG implementation.
  - Contains `HRRagConfig`, `HRRagPipeline`, local embeddings fallback, keyword retrieval, in-memory vector store, and helper functions.

- `hr_rag/official_corpus.py`
  - Validates the exact official competition corpus.
  - If this check fails, the official workflow should stop.

### App entry points

- `streamlit_hr_helpdesk.py`
  - Main Streamlit app logic.
  - Builds or loads the pipeline and serves the chatbot UI.

- `app.py`
  - Thin Streamlit Cloud entry point that delegates to `streamlit_hr_helpdesk.py`.

### Submission generation

- `generate_competition_submission.py`
  - Most important file for the official Kaggle challenge.
  - Extracts encrypted questions from the starter notebook.
  - Runs the pipeline.
  - Cleans answers for competition formatting.
  - Re-encrypts answers.
  - Validates final output.

- `generate_submission.py`
  - Generic batch submission generator for non-encrypted CSV workflows.

### Evaluation and tuning

- `evaluate_hr_rag.py`
  - Runs a labeled validation set.
  - Produces summary metrics and detailed per-question reports.

- `tune_hr_rag.py`
  - Sweeps chunking and retrieval settings.
  - Produces a leaderboard plus detailed run outputs.

### Controlled variant builders

The repo contains several `build_*.py` scripts. These are not the main product, but they are important because they reflect a working pattern used in experiments:

- start from an existing valid encrypted CSV
- decrypt with the existing helper
- override only selected clean answers
- re-encrypt only the changed answers
- preserve locked columns and row order
- update the matching `.sources.json`
- validate before saving

Useful examples:

- `build_controlled_variants.py`
- `build_requested_one_change_variants.py`
- `build_hybrid_submission.py`
- `build_guardrail_variants.py`

## Important Directories

- `hr_docs/official/`
  - Exact official competition PDFs.

- `competition/`
  - Starter notebook and competition reference files.

- `submissions/`
  - Generated Kaggle CSVs and matching `.sources.json` debug files.
  - This folder also contains many historical experiments.

- `eval/`
  - Validation datasets, tuning outputs, audits, and result summaries.

- `tests/`
  - Unit tests for pipeline behavior and competition submission rules.

## How Official Submission Generation Works

`generate_competition_submission.py` is the reference workflow for official competition output.

Its flow is:

1. Validate the official PDF corpus.
2. Read `competition/Starter_Notebook.ipynb`.
3. Extract:
   - Fernet key
   - encrypted questions `Q01` to `Q15`
4. Decrypt the questions.
5. Answer them with the HR pipeline.
6. Clean answer formatting for submission scoring.
7. Re-encrypt the answers.
8. Write:
   - `submission.csv`
   - matching `.sources.json`
9. Validate the output format, links, and required fields.

Important helper functions in this file include:

- `extract_competition_questions(...)`
- `clean_answer_for_submission(...)`
- `validate_competition_response(...)`
- `validate_submission(...)`
- `write_outputs(...)`

## How Controlled Submission Experiments Work

This repo has a strong pattern for controlled experiments: do not rebuild the whole pipeline if the user only wants to test wording changes for one or two questions.

Use the existing valid encrypted CSV as the base, then:

1. Load the base CSV.
2. Load the base `.sources.json`.
3. Use `extract_competition_questions(...)` to get the Fernet object and official questions.
4. Decrypt only for validation or inspection.
5. Override only the selected clean answers.
6. Re-encrypt only those changed answers.
7. Preserve:
   - row order
   - `question_enc`
   - `streamlit_link`
   - `langsmith_link`
   - unchanged `answer_enc` values
8. Update the matching `.sources.json`.
9. Run `validate_submission(...)`.
10. Verify that only the intended rows changed.

This pattern is especially useful when the user is optimizing leaderboard score by changing only final wording, not retrieval behavior.

## `.sources.json` Meaning

Each submission CSV usually has a matching `.sources.json` file. This is the readable debug artifact and is often easier to inspect than the encrypted CSV.

Typical fields include:

- `question_id`
- `clean_answer`
- `answer_with_sources`
- `sources`
- `blocked`
- `confidence`
- `critique_rating`
- `refined`
- `controlled_override`
- `unsupported_claims`

When a controlled variant is created, the `.sources.json` must be updated to match the new clean answer.

## Guardrail Behavior

The system is designed to answer only Zyro Dynamics HR policy questions supported by the provided documents.

It should refuse or block:

- non-HR questions
- requests for passwords, secrets, or tokens
- requests for another employee's sensitive data
- unsupported business, revenue, or product questions

Historically in this competition, `Q11` to `Q15` are often treated as out-of-scope or borderline guardrail questions. Some experiments in this repo override those answers in controlled ways, but the default safe stance is document-grounded refusal when the policy does not support an answer.

## Common Commands

### Install dependencies

```bash
pip install -r requirements.txt
```

### Streamlit app

```bash
streamlit run streamlit_hr_helpdesk.py
```

### Official encrypted competition submission

```powershell
python generate_competition_submission.py `
  --streamlit-link "https://YOUR-APP.streamlit.app" `
  --langsmith-link "https://smith.langchain.com/public/YOUR-TRACE/r" `
  --embedding-provider hash `
  --llm-provider groq `
  --chunk-size 1000 `
  --chunk-overlap 200 `
  --retrieval-k 6 `
  --fetch-k 48 `
  --vector-weight 0.65 `
  --disable-self-critique `
  --rebuild
```

### Generic batch submission

```bash
python generate_submission.py --docs-path hr_docs/official --questions test.csv --output submission.csv --rebuild
```

### Evaluation

```bash
python evaluate_hr_rag.py --docs-path hr_docs/official --validation-file eval/hr_validation_sample.jsonl --output-dir eval/results --embedding-provider hash --llm-provider extractive --rebuild
```

### Tuning

```bash
python tune_hr_rag.py --docs-path hr_docs/official --validation-file eval/hr_validation_sample.jsonl --output-dir eval/tuning --embedding-provider hash --llm-provider extractive
```

### Tests

```bash
python -m unittest discover -s tests
```

## Environment And Providers

The pipeline can select providers automatically.

Generation options:

- Groq
- OpenAI
- Ollama
- extractive fallback

Embedding options:

- OpenAI embeddings
- Ollama embeddings
- local hash embeddings

When no external provider is available, the repository can still run in a reduced offline mode using hash embeddings plus extractive answers.

## Safe Edit Zones

These are usually safe places to work when the user wants improvements:

- `hr_rag/pipeline.py`
  - retrieval tuning
  - ranking logic
  - prompt shaping
  - confidence handling
  - source formatting

- `evaluate_hr_rag.py`
  - metrics and reporting

- `tune_hr_rag.py`
  - search grid and leaderboard behavior

- `build_*.py` variant scripts
  - controlled submission experiments

- documentation files

## High-Risk Areas

Be careful in these places:

- `generate_competition_submission.py`
  - changes here can silently break encryption, validation, or score formatting

- `competition/Starter_Notebook.ipynb`
  - should generally remain unchanged

- `hr_docs/official/`
  - modifying corpus files invalidates the official setup

- `app.py`
  - intentionally minimal; changing it is rarely necessary

## Validation Checklist For Submission Work

Before finalizing a competition CSV, confirm:

- exactly 15 rows
- row order is `Q01` through `Q15`
- columns are exactly:
  - `question_id`
  - `question_enc`
  - `answer_enc`
  - `streamlit_link`
  - `langsmith_link`
- no blank `answer_enc`
- output passes `validate_submission(...)`
- for controlled variants, only the intended questions changed
- matching `.sources.json` exists and reflects the new clean answers

For override work, also verify:

- all untouched rows remain byte-for-byte identical to the base file where required
- `question_enc` values are preserved from the base file
- URLs are preserved from the base file

## Common Failure Modes

An AI agent should watch for these:

- accidentally changing the base file instead of writing a new variant
- changing multiple questions when only one should change
- updating the CSV but forgetting `.sources.json`
- introducing markdown formatting that gets cleaned badly for submission scoring
- adding unsupported claims not backed by the policy PDFs
- breaking corpus validation by touching official documents
- using the wrong base submission during a controlled experiment

## Practical Guidance For Another AI Agent

When a user asks for a leaderboard experiment:

1. Identify whether they want:
   - pipeline changes
   - or only final answer overrides
2. If they name a base CSV, use only that file.
3. If they lock certain questions, preserve them exactly.
4. If they ask for one new variant, do not create extras.
5. Validate output before saving.
6. Report the final file path and what changed.

When a user asks for system improvement:

1. Inspect retrieval and validation outputs in `eval/`.
2. Prefer evidence-driven adjustments over broad prompt rewrites.
3. Keep answers grounded in the official corpus.
4. Do not trade correctness for verbosity.

## Short Summary

This repository is not just a chatbot demo. It is a competition-oriented, guardrailed HR RAG system with a strict encrypted submission workflow and a large history of controlled submission experiments.

The safest mental model is:

- `hr_rag/` is the product logic
- `generate_competition_submission.py` is the competition contract
- `submissions/` is the experiment history
- `.sources.json` is the readable truth for encrypted outputs
- `hr_docs/official/` and the starter notebook are effectively immutable

If you preserve those boundaries, you can move fast here without breaking the important parts.
