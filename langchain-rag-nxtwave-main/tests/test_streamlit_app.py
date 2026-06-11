import os
import unittest

from streamlit.testing.v1 import AppTest


class StreamlitAppTests(unittest.TestCase):
    def test_chat_submission_keeps_app_visible(self):
        os.environ["EMBEDDING_PROVIDER"] = "hash"
        os.environ["LLM_PROVIDER"] = "extractive"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGSMITH_TRACING"] = "false"

        app = AppTest.from_file("app.py", default_timeout=120).run()
        app = app.chat_input[0].set_value(
            "How many Earned Leave days can be carried forward?"
        ).run(timeout=120)

        self.assertEqual(len(app.exception), 0)
        self.assertIn("Zyro Dynamics HR Help Desk", [title.value for title in app.title])
        self.assertEqual(len(app.chat_message), 2)
        self.assertEqual(len(app.chat_input), 1)


if __name__ == "__main__":
    unittest.main()
