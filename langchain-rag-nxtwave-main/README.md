# Zyro Dynamics HR Help Desk RAG

This project is set up for the Kaggle InClass HR Help Desk assignment. It builds a Retrieval-Augmented Generation pipeline that answers employee questions from the official 11 HR policy PDFs, blocks out-of-scope or sensitive requests, and can run as either a Streamlit chatbot or the exact encrypted competition submission generator.

## What Is Included

- `hr_rag/pipeline.py`: reusable RAG pipeline with document loading, chunking, Chroma retrieval, lightweight hybrid reranking, HR guardrails, citations, and offline fallback embeddings.
- `app.py`: official Streamlit Cloud entry point.
- `streamlit_hr_helpdesk.py`: interactive HR chatbot implementation.
- `generate_submission.py`: batch runner that reads Kaggle questions and writes `submission.csv`.
- `generate_competition_submission.py`: reads encrypted Q01-Q15 from the official starter notebook and writes the exact five-column competition submission.
- `evaluate_hr_rag.py`: validation runner for retrieval recall, answer overlap, confidence, HyDE/refinement usage, and guardrail checks.
- `tune_hr_rag.py`: grid tuner for chunking, hybrid retrieval weights, top-k, and fetch-k.
- `hr_docs/official/`: the official 11 competition HR policy PDFs.
- `competition/`: unchanged starter notebook, sample submission, and competition instructions.
- `eval/hr_validation_sample.jsonl`: small validation set grounded only in the official PDFs.
- `.env.example`: safe template for API keys and tracing settings.
- `TECHNICAL_README.md`: architecture, LangChain components, LLM options, vector database, chain design, and visual diagrams.

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
- Select `huggingface` to use local `sentence-transformers/all-MiniLM-L6-v2` embeddings.
- If `GROQ_API_KEY` exists, Groq is used for generation.
- If no API keys exist, local hash embeddings and extractive answers keep the pipeline runnable offline.

If OpenAI returns a quota error for embeddings, select `hash` or `ollama` in the Streamlit **Embeddings** dropdown and click **Rebuild Index**. The Python pipeline also falls back to local hash embeddings when an embedding provider fails during indexing or retrieval.

The enhanced retrieval path uses MMR vector retrieval, BM25 keyword retrieval, weighted Reciprocal Rank Fusion, normalized fusion confidence, conditional HyDE for vague queries, and conditional self-critique for low-confidence or batch answers.
Top-k retrieval also limits repeated chunks from the same source file so one handbook does not crowd out other relevant HR policy files.

## Add HR Policy Documents

The official Zyro Dynamics policy files are in `hr_docs/official/`.

Supported formats: `.md`, `.txt`, `.pdf`, `.docx`, `.csv`, `.json`.

The pipeline validates the exact filenames and SHA-256 hashes of all 11 PDFs before loading them. It rejects missing, modified, or additional files.

## Run The Streamlit Chatbot

```bash
streamlit run streamlit_hr_helpdesk.py
```

Use the sidebar to rebuild the index and inspect retrieval. The deployed app defaults to `hr_docs/official`.

For Streamlit Community Cloud, set the app file to:

```text
langchain-rag-nxtwave-main/app.py
```

Add these values in the Streamlit app's Secrets settings:

```toml
GROQ_API_KEY = "your-groq-key"
LANGCHAIN_API_KEY = "your-langsmith-key"
LANGCHAIN_PROJECT = "zyro-rag-challenge"
LANGSMITH_PROJECT = "zyro-rag-challenge"
LANGCHAIN_TRACING_V2 = "true"
LANGSMITH_TRACING = "true"
LLM_PROVIDER = "groq"
EMBEDDING_PROVIDER = "hash"
GROQ_MODEL = "llama-3.3-70b-versatile"
```

## Generate A Kaggle Submission

### Official Encrypted Competition Submission

After deploying the Streamlit app and sharing a LangSmith trace:

```powershell
python generate_competition_submission.py `
  --streamlit-link "https://YOUR-APP.streamlit.app" `
  --langsmith-link "https://smith.langchain.com/public/YOUR-TRACE/r" `
  --embedding-provider hash `
  --llm-provider groq `
  --chunk-size 900 `
  --chunk-overlap 150 `
  --retrieval-k 8 `
  --fetch-k 60 `
  --vector-weight 0.65 `
  --disable-self-critique `
  --rebuild
```

For stronger local semantic embeddings without an API:

```powershell
python generate_competition_submission.py `
  --streamlit-link "https://YOUR-APP.streamlit.app" `
  --langsmith-link "https://smith.langchain.com/public/YOUR-TRACE/r" `
  --embedding-provider huggingface `
  --llm-provider groq `
  --retrieval-k 8 `
  --fetch-k 60 `
  --rebuild
```

This command:

- verifies exactly 11 official PDFs
- decrypts official Q01-Q15 from `competition/Starter_Notebook.ipynb`
- keeps Q01-Q10 in scope and refuses Q11-Q15
- removes UI citations before encrypting scored answers
- creates exactly 15 rows with `question_id`, `question_enc`, `answer_enc`, `streamlit_link`, and `langsmith_link`
- validates all fields and URL patterns before finishing

The final file is written to `submissions/submission.csv`. The local `.sources.json` file contains readable answers and retrieval evidence for inspection.

### Generic Batch Submission

Example local command:

```bash
python generate_submission.py ^
  --docs-path hr_docs/official ^
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
python generate_submission.py --docs-path hr_docs/official --questions test.csv --output submission.csv --embedding-provider hash --llm-provider extractive --rebuild
```

Useful enhanced-pipeline flags:

```bash
python generate_submission.py \
  --docs-path hr_docs/official \
  --questions test.csv \
  --output submission.csv \
  --chunk-size 700 \
  --chunk-overlap 150 \
  --retrieval-k 6 \
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
  --docs-path hr_docs/official ^
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
  --docs-path hr_docs/official ^
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

- Try `chunk_size` values `600`, `700`, `800`, and `900`.
- Try `chunk_overlap` values `150`, `200`, and `250`.
- Try `retrieval_k` values `6`, `8`, and `10`.
- Try `fetch_k` values `48` and `60`.
- Try vector/BM25 weights of `0.55/0.45`, `0.65/0.35`, and `0.70/0.30`.
- The selected proxy-validation configuration is `chunk_size=900`, `chunk_overlap=150`, `retrieval_k=8`, `fetch_k=60`, and vector/BM25 `0.65/0.35`.
- Inspect the generated `.sources.json` file after every run to confirm the right policies were retrieved.
- Prefer grounded answers over broad explanations. The competition will likely reward answers that match the internal policy text.
- Keep guardrails strict for non-HR, sensitive personal data, credentials, and unsupported questions.

The production path for this assignment is the official-corpus HR pipeline: `streamlit_hr_helpdesk.py`, `hr_rag/`, `generate_competition_submission.py`, and `hr_docs/official/`.
