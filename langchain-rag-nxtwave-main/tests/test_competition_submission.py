import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from generate_competition_submission import (
    extract_competition_questions,
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


if __name__ == "__main__":
    unittest.main()
