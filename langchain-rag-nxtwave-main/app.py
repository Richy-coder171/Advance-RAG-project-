"""Streamlit Cloud entry point for the Zyro Dynamics HR Help Desk."""

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from streamlit_hr_helpdesk import *  # noqa: F401,F403
