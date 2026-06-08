# Zyro Dynamics HR Help Desk RAG

This project is set up for the Kaggle InClass HR Help Desk assignment. It builds a Retrieval-Augmented Generation pipeline that answers employee questions from Zyro Dynamics HR policy documents, blocks out-of-scope or sensitive requests, and can run as either a Streamlit chatbot or a batch submission generator.

## What Is Included

- `hr_rag/pipeline.py`: reusable RAG pipeline with document loading, chunking, Chroma retrieval, lightweight hybrid reranking, HR guardrails, citations, and offline fallback embeddings.
- `streamlit_hr_helpdesk.py`: interactive HR chatbot for local or Streamlit Cloud deployment.
- `generate_submission.py`: batch runner that reads Kaggle questions and writes `submission.csv`.
- `evaluate_hr_rag.py`: validation runner for retrieval recall, answer overlap, confidence, HyDE/refinement usage, and guardrail checks.
- `tune_hr_rag.py`: grid tuner for chunking, hybrid retrieval weights, top-k, and fetch-k.
- `hr_docs/`: place the official Zyro Dynamics HR policy documents here.
- `eval/hr_validation_sample.jsonl`: small starter validation set for local testing.
- `.env.example`: safe template for API keys and tracing settings.
- `TECHNICAL_README.md`: architecture, LangChain components, LLM options, vector database, chain design, and visual diagrams.
- `EVALUATION_REPORT.md`: mapped evaluation findings and remaining competition-readiness work.

## Setup

```bash
pip install -r requirements.txt
```

On Windows, if `pip install` fails because a package file is "being used by another process", use a fresh virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill only the keys you are allowed to use. Do not commit `.env`, and do not put real keys in `.env.example`.

The pipeline auto-selects providers:

- If `OPENAI_API_KEY` exists, OpenAI embeddings are used.
- If `GROQ_API_KEY` exists, Groq is used for generation.
- If no API keys exist, local hash embeddings and extractive answers keep the pipeline runnable offline.

If OpenAI returns a quota error for embeddings, select `hash` or `ollama` in the Streamlit **Embeddings** dropdown and click **Rebuild Index**. The Python pipeline also falls back to local hash embeddings when an embedding provider fails during indexing or retrieval.

The enhanced retrieval path uses MMR vector retrieval, BM25 keyword retrieval, weighted Reciprocal Rank Fusion, normalized fusion confidence, conditional HyDE for vague queries, and conditional self-critique for low-confidence or batch answers.
Top-k retrieval also limits repeated chunks from the same source file so one handbook does not crowd out other relevant HR policy files.

## Add HR Policy Documents

Put the official Zyro Dynamics policy files in `hr_docs/`.

Supported formats: `.md`, `.txt`, `.pdf`, `.docx`, `.csv`, `.json`.

Temporary test policy files are currently stored in `hr_docs/temp_policies/`.

Do not add fabricated policy text for the competition run. The answer prompt is designed to say when the policy documents do not contain an answer.

## Run The Streamlit Chatbot

```bash
streamlit run streamlit_hr_helpdesk.py
```

Use the sidebar to choose the policy folder, rebuild the index, and tune chunking/retrieval settings. For the temporary corpus, set the policy folder to `hr_docs/temp_policies`.

## Generate A Kaggle Submission

Example local command:

```bash
python generate_submission.py ^
  --docs-path hr_docs ^
  --questions test.csv ^
  --sample-submission sample_submission.csv ^
  --output submissions/submission.csv ^
  --rebuild
```

Example inside Kaggle:

```bash
python generate_submission.py \
  --docs-path /kaggle/input/zyro-hr-policies/hr_docs \
  --questions /kaggle/input/zyro-hr-helpdesk/test.csv \
  --sample-submission /kaggle/input/zyro-hr-helpdesk/sample_submission.csv \
  --output /kaggle/working/submission.csv \
  --rebuild
```

If no model API is available:

```bash
python generate_submission.py --docs-path hr_docs --questions test.csv --output submission.csv --embedding-provider hash --llm-provider extractive --rebuild
```

Useful enhanced-pipeline flags:

```bash
python generate_submission.py \
  --docs-path hr_docs \
  --questions test.csv \
  --output submission.csv \
  --chunk-size 700 \
  --chunk-overlap 150 \
  --retrieval-k 10 \
  --fetch-k 48 \
  --vector-weight 0.6 \
  --min-confidence 0.35 \
  --max-chunks-per-source 2 \
  --critique-threshold 0.65 \
  --rebuild
```

Batch answers use self-critique by default. Use `--disable-self-critique`, `--disable-hyde`, or `--no-source-block` for faster experiments.

## Evaluate HR RAG Locally

Run the starter validation set:

```bash
python evaluate_hr_rag.py ^
  --docs-path hr_docs/temp_policies ^
  --validation-file eval/hr_validation_sample.jsonl ^
  --output-dir eval/results ^
  --embedding-provider hash ^
  --llm-provider extractive ^
  --rebuild
```

Outputs:

- `eval/results/summary.json`: aggregate metrics
- `eval/results/details.csv`: per-question metrics with `question`, `expected_source`, `retrieved_sources`, `source_recall`, `confidence`, `clean_answer`, `answer_with_sources`, `reference_answer`, `token_f1`, `rouge_l`, `blocked`, and `error`
- `eval/results/details.json`: full answers and source evidence
- `eval/results/retrieved_chunks.csv`: one row per retrieved chunk with chunk ID, confidence, method, and preview

## Tune Retrieval Settings

Run the required grid over chunking, top-k, fetch-k, and hybrid weights:

```bash
python tune_hr_rag.py ^
  --docs-path hr_docs/temp_policies ^
  --validation-file eval/hr_validation_sample.jsonl ^
  --output-dir eval/tuning ^
  --embedding-provider hash ^
  --llm-provider extractive
```

For a quick smoke test:

```bash
python tune_hr_rag.py --max-runs 3 --embedding-provider hash --llm-provider extractive
```

The tuner writes `eval/tuning/leaderboard.csv`, `eval/tuning/leaderboard.json`, and `eval/tuning/best/` details. The leaderboard is sorted by `avg_rouge_l` first, then `avg_source_recall`.

For the actual competition, replace `eval/hr_validation_sample.jsonl` with a 30-50 question HR validation set using expected source files and reference answers.

## Leaderboard Strategy

Tune against a small validation set before submitting:

- Try `chunk_size` values around `700`, `900`, and `1100`.
- Try `chunk_overlap` values `150`, `180`, and `220`.
- Try `retrieval_k` values `6`, `8`, and `10`.
- Try `fetch_k` values `24`, `36`, and `48`.
- Try vector/BM25 weights of `0.50/0.50`, `0.60/0.40`, and `0.70/0.30`.
- The current tuned default is `chunk_size=700`, `chunk_overlap=150`, `retrieval_k=10`, `fetch_k=48`, and vector/BM25 `0.60/0.40`.
- Inspect the generated `.sources.json` file after every run to confirm the right policies were retrieved.
- Prefer grounded answers over broad explanations. The competition will likely reward answers that match the internal policy text.
- Keep guardrails strict for non-HR, sensitive personal data, credentials, and unsupported questions.

## Legacy Reference Files

The previous API documentation assistant files are kept only as learning/reference material:

- `streamlit_api_assistant.py`
- `Handson_lab1.ipynb`
- `api_docs/`

The production path for this assignment is the HR pipeline: `streamlit_hr_helpdesk.py`, `hr_rag/`, `generate_submission.py`, and `evaluate_hr_rag.py`.
