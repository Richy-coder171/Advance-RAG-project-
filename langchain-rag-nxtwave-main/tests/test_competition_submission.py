import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from generate_competition_submission import (
    extract_competition_questions,
    retry_wait_seconds,
    validate_competition_response,
    validate_links,
)
from hr_rag import validate_official_corpus


class CompetitionSubmissionTests(unittest.TestCase):
    def test_official_notebook_contains_q01_to_q15(self):
        _fernet, questions = extract_competition_questions("competition/Starter_Notebook.ipynb")
        self.assertEqual([question_id for question_id, _question in questions], ["Q%02d" % i for i in range(1, 16)])
        self.assertTrue(all(question.strip() for _question_id, question in questions))

    def test_official_corpus_has_exactly_eleven_pdfs(self):
        validate_official_corpus("hr_docs/official")

    def test_official_corpus_rejects_other_data(self):
        with TemporaryDirectory() as temp_dir:
            Path(temp_dir, "unrelated-policy.pdf").write_bytes(b"not official")
            with self.assertRaises(ValueError):
                validate_official_corpus(temp_dir)

    def test_link_validation_matches_competition_contract(self):
        validate_links(
            "https://zyro-hr-helpdesk.streamlit.app",
            "https://smith.langchain.com/public/7d409145-121a-4bd1-b34d-efcbe8fd423e/r",
        )
        with self.assertRaises(ValueError):
            validate_links("http://localhost:8501", "https://smith.langchain.com/public/7d409145-121a-4bd1-b34d-efcbe8fd423e/r")
        with self.assertRaises(ValueError):
            validate_links("https://your-real-app.streamlit.app", "https://smith.langchain.com/public/your-real-trace/r")

    def test_retry_wait_uses_groq_rate_limit_hint(self):
        self.assertAlmostEqual(retry_wait_seconds(Exception("Please try again in 2m47.5s."), 15, 1), 172.5)

    def test_critical_answer_validation_rejects_missing_facts_and_fallback(self):
        response = type("Response", (), {"answer": "", "blocked": False, "critique_rating": None})
        response.answer = "45 days. Excess leave is automatically encashed and credited in the April payroll."
        validate_competition_response("Q02", 2, response)

        response.answer = "Group Medical Insurance covers the employee and spouse."
        with self.assertRaises(ValueError):
            validate_competition_response("Q07", 7, response)

        response.answer = "Note: Zyro Dynamics ensures that no deduction... Promotions at Zyro Dynamics are merit-based."
        with self.assertRaises(ValueError):
            validate_competition_response("Q06", 6, response)

        response.answer = "L4 Senior: Rs. 16.0L to Rs. 26.0L; bonus target: 10% of CTC."
        validate_competition_response("Q06", 6, response)

        response.answer = "L4 Senior: Rs.\u202f16.0\u202fL to Rs.\u202f26.0\u202fL; bonus target: 10% of CTC."
        validate_competition_response("Q06", 6, response)

        response.answer = "L4 Senior: Rs. 16.0L to Rs. 26.0L; bonus target: 10% of CTC. [Document 3]"
        with self.assertRaises(ValueError):
            validate_competition_response("Q06", 6, response)

        response.answer = "A normal answer"
        response.blocked = True
        with self.assertRaises(ValueError):
            validate_competition_response("Q11", 11, response)


if __name__ == "__main__":
    unittest.main()
