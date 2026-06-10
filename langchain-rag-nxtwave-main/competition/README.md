# Official Zyro Dynamics Competition Assets

This folder preserves the official competition starter files:

- `Starter_Notebook.ipynb`: contains the encrypted Q01-Q15 evaluation questions, Fernet submission key, required URL validation, and exact submission schema.
- `sample_submission.xlsx`: example five-column submission layout.

The official 11 HR policy PDFs are stored in:

```text
hr_docs/official/
```

Generate the final submission only after deploying Streamlit and sharing a LangSmith trace:

```powershell
python generate_competition_submission.py `
  --streamlit-link "https://YOUR-APP.streamlit.app" `
  --langsmith-link "https://smith.langchain.com/public/YOUR-TRACE/r" `
  --embedding-provider hash `
  --llm-provider groq `
  --rebuild
```

The generator reads and decrypts Q01-Q15 from the unchanged starter notebook, answers them using the official corpus, removes UI citations from the scored answers, encrypts the questions and answers with the official Fernet key, and validates the final 15-row CSV.

Never submit `submission_smoke.csv`; its links are placeholders and its answers use the offline extractive fallback.
