FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit_hr_helpdesk.py .
COPY app.py .
COPY generate_submission.py .
COPY generate_competition_submission.py .
COPY evaluate_hr_rag.py .
COPY hr_rag/ ./hr_rag/
COPY hr_docs/ ./hr_docs/
COPY competition/ ./competition/
COPY eval/ ./eval/
COPY README.md TECHNICAL_README.md EVALUATION_REPORT.md ./

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
