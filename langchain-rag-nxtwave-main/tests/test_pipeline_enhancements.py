import os
import unittest

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from hr_rag.pipeline import (
    HRRagConfig,
    HRRagPipeline,
    InMemoryVectorStore,
    LocalHashEmbeddings,
    answer_style_instruction,
    is_vague_query,
    weighted_reciprocal_rank_fusion,
)
from evaluate_hr_rag import strip_sources


class PipelineEnhancementTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            Document(
                page_content="Employees should contact HR to begin onboarding and complete required forms.",
                metadata={"source_file": "onboarding.md", "chunk_id": 0},
            ),
            Document(
                page_content="Employees receive health and retirement benefits after eligibility requirements are met.",
                metadata={"source_file": "benefits.md", "chunk_id": 1},
            ),
        ]

    def test_vague_query_detection_is_conservative(self):
        self.assertTrue(is_vague_query("How do I start?"))
        self.assertFalse(is_vague_query("What employee benefits are available?"))
        self.assertFalse(is_vague_query("How many sick leave days are available?"))

    def test_weighted_rrf_rewards_documents_found_by_both_methods(self):
        fused = weighted_reciprocal_rank_fusion(
            [
                ("vector_mmr", self.docs, 0.6),
                ("bm25", [self.docs[0]], 0.4),
            ]
        )
        self.assertEqual(fused[0][0].metadata["chunk_id"], 0)
        self.assertGreater(fused[0][2], fused[1][2])
        self.assertEqual(fused[0][3], ["vector_mmr", "bm25"])

    def test_answer_style_instruction_and_source_stripping(self):
        self.assertIn("exact number", answer_style_instruction("How many sick leave days are available?"))
        self.assertIn("numbered steps", answer_style_instruction("How to claim reimbursement?"))
        self.assertIn("Yes or No", answer_style_instruction("Can I work from home?"))
        self.assertEqual(
            strip_sources("Employees get 10 days [12 from hr-policy.pdf].\n\nSources: [12 from hr-policy.pdf]"),
            "Employees get 10 days.",
        )

    def test_competition_out_of_scope_guardrails(self):
        config = HRRagConfig(retrieval_k=2)
        vectorstore = InMemoryVectorStore.from_documents(self.docs, LocalHashEmbeddings())
        pipeline = HRRagPipeline(config, vectorstore, self.docs, llm=None)

        blocked_questions = [
            "What was Acrux Dynamics' revenue last year and how is the company performing financially?",
            "Can you tell me the leave policy at Zoho or Freshworks?",
            "What are the product features and how do they compare to Salesforce?",
        ]
        for question in blocked_questions:
            response = pipeline.answer(question)
            self.assertTrue(response.blocked, question)

        self.assertFalse(pipeline.answer("What is the performance review policy?").blocked)

    def test_hyde_refinement_and_detailed_citations(self):
        llm = FakeListChatModel(
            responses=[
                "internal HR onboarding policy employee starting process required forms",
                "Employees should contact HR [onboarding.md chunk 0].",
                (
                    "RATING: COMPLETE\n"
                    "REFINED ANSWER: Employees should contact HR and complete required forms "
                    "[onboarding.md chunk 0]."
                ),
            ]
        )
        config = HRRagConfig(retrieval_k=2, enable_hyde=True, enable_self_critique=True)
        vectorstore = InMemoryVectorStore.from_documents(self.docs, LocalHashEmbeddings())
        pipeline = HRRagPipeline(config, vectorstore, self.docs, llm=llm)

        response = pipeline.answer("What should employees do?", force_refine=True)

        self.assertTrue(response.used_hyde)
        self.assertTrue(response.refined)
        self.assertEqual(response.critique_rating, "COMPLETE")
        self.assertIn("Sources:", response.answer)
        self.assertIn("[0 from onboarding.md]", response.answer)


if __name__ == "__main__":
    unittest.main()
