# Zyro Dynamics HR Help Desk RAG

This project is set up for the Kaggle InClass HR Help Desk assignment. It builds a Retrieval-Augmented Generation pipeline that answers employee questions from Zyro Dynamics HR policy documents, blocks out-of-scope or sensitive requests, and can run as either a Streamlit chatbot or a batch submission generator.

## What Is Included

- `hr_rag/pipeline.py`: reusable RAG pipeline with document loading, chunking, Chroma retrieval, lightweight hybrid reranking, HR guardrails, citations, and offline fallback embeddings.
- `streamlit_hr_helpdesk.py`: interactive HR chatbot for local or Streamlit Cloud deployment.
- `generate_submission.py`: batch runner that reads Kaggle questions and writes `submission.csv`.
- `hr_docs/`: place the official Zyro Dynamics HR policy documents here.
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
  --vector-weight 0.6 \
  --min-confidence 0.35 \
  --critique-threshold 0.65 \
  --rebuild
```

Batch answers use self-critique by default. Use `--disable-self-critique`, `--disable-hyde`, or `--no-source-block` for faster experiments.

## Leaderboard Strategy

Tune against a small validation set before submitting:

- Try `chunk_size` values around `700`, `900`, and `1100`.
- Try `retrieval_k` between `5` and `8`.
- Inspect the generated `.sources.json` file after every run to confirm the right policies were retrieved.
- Prefer grounded answers over broad explanations. The competition will likely reward answers that match the internal policy text.
- Keep guardrails strict for non-HR, sensitive personal data, credentials, and unsupported questions.

## Original Lab Files

The previous API documentation assistant is still available as `streamlit_api_assistant.py` and `Handson_lab1.ipynb`.

cd "D:\kaggle project\langchain-rag-nxtwave\langchain-rag-nxtwave-main"
python -m streamlit run streamlit_hr_helpdesk.py
