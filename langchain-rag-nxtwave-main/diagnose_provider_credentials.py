from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import Any, Dict


def load_env_candidates() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    candidates = [Path.cwd() / ".env"]
    seen = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        raw = path.read_bytes()
        for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "cp1252", "latin-1"):
            try:
                text = raw.decode(encoding)
            except UnicodeDecodeError:
                continue
            load_dotenv(stream=io.StringIO(text), override=False)
            break


def safe_error(exc: Exception) -> Dict[str, str]:
    return {
        "status": "error",
        "error_type": exc.__class__.__name__,
        "error": str(exc),
    }


def test_openai(check_chat: bool, model: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"configured": bool(os.getenv("OPENAI_API_KEY"))}
    if not result["configured"]:
        result["status"] = "missing_key"
        return result
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        models = client.models.list()
        result["status"] = "models_ok"
        result["model_count"] = len(list(models.data))
        if check_chat:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply with the single word OK."}],
                max_tokens=3,
                temperature=0.0,
            )
            result["chat_status"] = "ok"
            result["chat_preview"] = completion.choices[0].message.content
        return result
    except Exception as exc:
        return {**result, **safe_error(exc)}


def test_groq(check_chat: bool, model: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"configured": bool(os.getenv("GROQ_API_KEY"))}
    if not result["configured"]:
        result["status"] = "missing_key"
        return result
    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
        models = client.models.list()
        result["status"] = "models_ok"
        result["model_count"] = len(list(models.data))
        if check_chat:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply with the single word OK."}],
                max_tokens=3,
                temperature=0.0,
            )
            result["chat_status"] = "ok"
            result["chat_preview"] = completion.choices[0].message.content
        return result
    except Exception as exc:
        return {**result, **safe_error(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose configured hosted-provider credentials for the HR RAG repo.")
    parser.add_argument("--check-chat", action="store_true", help="Also attempt a tiny chat completion, not just model listing.")
    parser.add_argument("--openai-model", default=os.getenv("OPENAI_MODEL", "gpt-4o"))
    parser.add_argument("--groq-model", default=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
    args = parser.parse_args()

    load_env_candidates()
    summary = {
        "env_presence": {
            "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
            "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
            "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
            "GOOGLE_API_KEY": bool(os.getenv("GOOGLE_API_KEY")),
        },
        "openai": test_openai(args.check_chat, args.openai_model),
        "groq": test_groq(args.check_chat, args.groq_model),
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
