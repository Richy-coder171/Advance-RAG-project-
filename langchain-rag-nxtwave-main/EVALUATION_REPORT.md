# Project Evaluation Report

This report summarizes the external Kaggle RAG evaluation notes supplied in:

- `build_doc.sh.txt`
- `z.txt`
- `v.txt`

The original report was written for `Handson_lab1.ipynb`, which focused on an API documentation assistant. This project has since evolved into the Zyro Dynamics HR Help Desk RAG system. The findings below map the report's recommendations to the current HR pipeline.

## Report Validation Notes

The supplied document-build script generated a Word report named `Kaggle_RAG_Evaluation_Report.docx` in its original environment.

The validation note in `z.txt` confirms:

| Check | Result |
| --- | --- |
| DOCX file entries | 22 |
| `word/document.xml` present | Yes |
| `[Content_Types].xml` present | Yes |
| Uncompressed size | 228 KB |
| File size on disk | 20 KB |

The note in `v.txt` shows that a separate office validation script was missing:

```text
python3: can't open file '//scripts/office/validate.py'
```

So the generated DOCX structure looked valid, but the optional validator did not run.

## Original Estimated Score

The report estimated the older notebook at:

```text
Current score: 61 / 100
Target top-10 score: 85 / 100
```

The report identified these major gaps:

| Dimension | Estimated Score | Target | Gap |
| --- | ---: | ---: | ---: |
| Recall@5 | 58% | 80%+ | -22 pts |
| Faithfulness | 64% | 85%+ | -21 pts |
| Answer relevance | 71% | 85%+ | -14 pts |
| Exact match | 31% | 70%+ | -39 pts |
| Robustness | 55% | 80%+ | -25 pts |

These values are directional estimates, not measured leaderboard scores for the current HR application.

## Findings Applied To Current Project

| Original Finding | Current Status | Current Implementation |
| --- | --- | --- |
| Context was truncated to 250 characters | Fixed | `format_context()` now allows up to `max_context_chars_per_chunk`, default `1800` |
| Evaluation measured source files only | Partially addressed | Batch run writes `.sources.json`; full answer-level eval set still needed |
| Only 2 ground-truth queries | Still needed | Need 30-50 HR validation questions with reference answers |
| Vector-only retrieval failed on exact terms | Fixed | Current system uses MMR vector retrieval + BM25-style keyword retrieval |
| Guardrails over-blocked valid queries | Adapted | Current HR guardrails block sensitive HR/privacy requests, not normal policy terms |
| Prompt lacked stronger reasoning/citation rules | Adapted | Current prompt enforces grounded answers and citations; self-critique refines low-confidence answers |

## Current Enhancements Beyond The Report

The current HR pipeline already includes several improvements that were either requested later or go beyond the original notebook report:

| Enhancement | Status |
| --- | --- |
| Chroma vector database | Implemented |
| Pure Python vector fallback | Implemented |
| MMR retrieval | Implemented |
| BM25-style lexical retrieval | Implemented |
| Weighted Reciprocal Rank Fusion | Implemented |
| Normalized retrieval confidence | Implemented |
| Detailed citation/source block | Implemented |
| Conditional HyDE query rewriting | Implemented |
| Conditional self-critique/refinement | Implemented |
| Streamlit controls for retrieval tuning | Implemented |
| Batch Kaggle submission generator | Implemented |
| Regression tests for enhanced retrieval behavior | Implemented |

## Remaining High-Value Work

The most important remaining work is not another retrieval trick. It is evaluation data.

### 1. Build An HR Validation Set

Create 30-50 questions with expected answer notes and source files.

Recommended categories:

- Leave policy
- Sick leave
- Payroll
- Benefits
- Insurance
- Reimbursement
- Probation
- Notice period
- Conduct/compliance
- Privacy/guardrail refusal cases

Suggested format:

```json
{
  "question": "What employee benefits are available?",
  "expected_sources": ["Employee-Benefits-and-Perks-1.docx"],
  "reference_answer": "Employees may receive benefits such as health-related benefits, insurance, retirement support, and other company-defined perks depending on eligibility."
}
```

### 2. Add Answer-Level Evaluation

The current `.sources.json` file helps debug retrieval, but final Kaggle quality depends on the generated answer.

Useful metrics:

- Recall@K for source retrieval
- Citation coverage
- Answer groundedness
- ROUGE-L against reference answer
- Embedding similarity against reference answer
- Manual review for sensitive HR questions

### 3. Tune Retrieval Settings Against Validation

Current starting settings:

```text
chunk_size = 900
chunk_overlap = 180
retrieval_k = 6
fetch_k = 24
vector_weight = 0.60
keyword_weight = 0.40
min_confidence = 0.35
critique_confidence_threshold = 0.65
```

Tune:

- `chunk_size`: 700, 900, 1100
- `retrieval_k`: 5, 6, 8
- `fetch_k`: 24, 36, 48
- `vector_weight`: 0.50, 0.60, 0.70
- `min_confidence`: only after inspecting missed answers

## Practical Verdict

The report was worth using, but not literally. Several fixes were already implemented in the current HR pipeline or needed to be adapted from API-documentation language to HR-policy language.

The current system is stronger than the evaluated notebook because it includes hybrid retrieval, MMR, weighted RRF, confidence scoring, citations, HyDE, and self-critique.

An executable HR evaluation runner is now available:

```bash
python evaluate_hr_rag.py \
  --docs-path hr_docs/temp_policies \
  --validation-file eval/hr_validation_sample.jsonl \
  --output-dir eval/results \
  --embedding-provider hash \
  --llm-provider extractive \
  --rebuild
```

The next leaderboard jump will most likely come from:

1. A real HR validation set.
2. Answer-level scoring.
3. Retrieval tuning based on failure cases.
4. Careful guardrail testing.
