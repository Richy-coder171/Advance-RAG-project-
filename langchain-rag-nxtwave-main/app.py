"""Streamlit Cloud entry point for the Zyro Dynamics HR Help Desk."""

import runpy
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

runpy.run_path(str(APP_DIR / "streamlit_hr_helpdesk.py"), run_name="__main__")
