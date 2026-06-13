import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from generate_competition_submission import (
    PROVEN_ANSWERS,
    OUT_OF_SCOPE_IDS,
    REFUSAL_ANSWER,
    clean_answer_for_submission,
    extract_competition_questions,
    has_artifacts,
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

    def test_submission_cleaning_removes_artifacts_without_deleting_answer_text(self):
        dirty = (
            "According to the HR policy, **Employees receive 15 days.** [Document 1] [1]\n"
            "1. Apply through the portal.\nSource: 02_Leave_Policy.pdf\n"
            "Confidence: 0.82\nRetrieved from: leave chunks"
        )
        self.assertEqual(
            clean_answer_for_submission(dirty),
            "Employees receive 15 days. Apply through the portal.",
        )
        timeline = "1. First stage: February 2. Second stage: March 3. Final stage: April"
        self.assertEqual(
            clean_answer_for_submission(timeline),
            "First stage: February Second stage: March Final stage: April",
        )
        self.assertEqual(clean_answer_for_submission("Eligible employees • Hybrid WFH • Full Remote"), "Eligible employees Hybrid WFH Full Remote")

        self.assertEqual(
            clean_answer_for_submission("CTC Range: Rs. 16.0L to Rs. 26.0L. Bonus Target: 10% of CTC."),
            "Rs. 16.0L to Rs. 26.0L. 10% of CTC.",
        )
        verbose = " ".join("word%s" % index for index in range(100))
        self.assertLessEqual(len(clean_answer_for_submission(verbose).split()), 80)

    def test_out_of_scope_ids_use_the_locked_refusal(self):
        self.assertEqual(OUT_OF_SCOPE_IDS, {"Q11", "Q12", "Q13", "Q14", "Q15"})
        self.assertEqual(
            REFUSAL_ANSWER,
            "I can only answer HR-related questions from Zyro Dynamics policy documents.",
        )
        self.assertEqual(clean_answer_for_submission(REFUSAL_ANSWER), REFUSAL_ANSWER)

    def test_proven_answers_are_complete_and_artifact_free(self):
        self.assertEqual(set(PROVEN_ANSWERS), {"Q%02d" % index for index in range(1, 11)})
        for question_id, answer in PROVEN_ANSWERS.items():
            self.assertFalse(has_artifacts(answer), question_id)
            self.assertLessEqual(len(answer.split()), 80, question_id)

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
        validate_competition_response("Q06", 6, response)

        response.answer = "L4 Senior: Rs. 16.0L to Rs. 26.0L; bonus target: 10% of CTC. Chunk ID: 3"
        with self.assertRaises(ValueError):
            validate_competition_response("Q06", 6, response)

        response.answer = "A normal answer"
        response.blocked = True
        with self.assertRaises(ValueError):
            validate_competition_response("Q11", 11, response)


if __name__ == "__main__":
    unittest.main()
