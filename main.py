"""AI Knowledge Assistant with Langfuse observability.

Run:  python main.py

Tracing flow (what you will see in Langfuse for every question):

    knowledge-assistant            <- trace (named via propagate_attributes)
    └── assistant-request          <- root span created by @observe
        └── assistant-answer       <- generation auto-captured by the
                                      Langfuse OpenAI wrapper (prompt, output,
                                      model, latency, token usage, errors)
"""

from __future__ import annotations

import logging
import os
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Load .env BEFORE importing Langfuse so the SDK can read its keys
# (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL) from the environment.
load_dotenv()

import openai  # noqa: E402  (only used for its exception classes)
from langfuse import get_client, observe, propagate_attributes  # noqa: E402

# Drop-in replacement for `from openai import OpenAI`. Every call made through this
# client is recorded in Langfuse as a "generation" automatically.
from langfuse.openai import OpenAI  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "logs" / "ai_app.log"

DEFAULT_MODEL = "gpt-4.1-mini"
REQUIRED_ENV_VARS = ["OPENAI_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]
PLACEHOLDER_MARKER = "replace-me"

logger = logging.getLogger("ai_app")

SYSTEM_PROMPT = """You are a helpful knowledge assistant.
Answer the user's question using ONLY the reference context provided.
If the context does not contain the answer, say: "I don't know based on the provided context."
Do not add facts that are not in the context. Keep answers short and clear."""

# Sample reference contexts. In a real app these would come from a search index
# or document store (this is the "R" in RAG). Here they are hardcoded on purpose.
SAMPLE_CONTEXTS: dict[str, dict[str, str]] = {
    "1": {
        "title": "Acme Cloud Storage - Pricing & Limits",
        "text": (
            "Acme Cloud Storage offers three plans. The Free plan includes 5 GB of storage "
            "and a 100 MB maximum file size. The Pro plan costs $8 per month and includes "
            "1 TB of storage with a 10 GB maximum file size. The Team plan costs $15 per "
            "user per month, includes 5 TB of pooled storage, and adds admin controls. "
            "Deleted files are kept in the trash for 30 days on all plans. "
            "Customer support is available by email on all plans and by phone on the Team plan only."
        ),
        "sample_questions": (
            "How much does the Pro plan cost? | "
            "How long are deleted files kept? | "
            "Does the Free plan include phone support?"
        ),
    },
    "2": {
        "title": "Company Leave Policy",
        "text": (
            "Full-time employees receive 20 days of paid annual leave per calendar year. "
            "Up to 5 unused days can be carried over to the next year; any remaining days expire. "
            "Leave requests must be submitted at least 2 weeks in advance through the HR portal. "
            "Employees receive 10 days of paid sick leave per year, and a doctor's note is "
            "required for sick leave longer than 3 consecutive days. "
            "Part-time employees receive leave pro-rated to their contracted hours."
        ),
        "sample_questions": (
            "How many leave days can I carry over? | "
            "When do I need a doctor's note? | "
            "Do I get a bonus for unused leave?"
        ),
    },
    "3": {
        "title": "The Python Programming Language",
        "text": (
            "Python is a high-level, general-purpose programming language created by "
            "Guido van Rossum and first released in 1991. Python emphasizes code readability "
            "and uses significant indentation. Python 3.0 was released in December 2008 and "
            "was not fully backward-compatible with Python 2. Python 2 reached end of life "
            "on January 1, 2020. The reference implementation is called CPython."
        ),
        "sample_questions": (
            "Who created Python? | "
            "When did Python 2 reach end of life? | "
            "Is CPython written in Rust?"
        ),
    },
}


# ---------------------------------------------------------------------------
# Shared helpers (also imported by evaluator.py)
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    """Write timestamped logs to logs/ai_app.log.

    We configure the named "ai_app" logger instead of the root logger so that
    third-party libraries (httpx, openai, ...) do not write request details into our file.
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if logger.handlers:  # already configured
        return
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def check_env() -> None:
    """Exit with a clear message if required environment variables are missing.

    Only variable NAMES are ever printed, never their values.
    """
    missing = [
        name
        for name in REQUIRED_ENV_VARS
        if not os.getenv(name) or PLACEHOLDER_MARKER in os.getenv(name, "")
    ]
    if missing:
        print("\n[CONFIG ERROR] Missing or placeholder environment variables:")
        for name in missing:
            print(f"  - {name}")
        print("\nCopy .env.example to .env and fill in your real keys:")
        print("  cp .env.example .env\n")
        logger.error("Startup aborted: missing env vars %s", missing)
        sys.exit(1)


def get_model(env_var: str = "OPENAI_MODEL") -> str:
    return os.getenv(env_var) or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL


def describe_openai_error(exc: Exception) -> str:
    """Turn an OpenAI exception into a short, secret-free message.

    We deliberately do NOT print str(exc) for auth errors because the API's error
    text can echo part of the key that was sent.
    """
    if isinstance(exc, openai.AuthenticationError):
        return "Authentication failed. Check OPENAI_API_KEY in your .env file."
    if isinstance(exc, openai.PermissionDeniedError):
        return "Permission denied. Your key may not have access to this model."
    if isinstance(exc, openai.NotFoundError):
        return "Model not found. Check OPENAI_MODEL / OPENAI_JUDGE_MODEL in your .env file."
    if isinstance(exc, openai.RateLimitError):
        return "Rate limit or quota exceeded. Wait a moment or check your OpenAI billing."
    if isinstance(exc, openai.APITimeoutError):
        return "The request to OpenAI timed out. Please try again."
    if isinstance(exc, openai.APIConnectionError):
        return "Could not connect to OpenAI. Check your internet connection."
    if isinstance(exc, openai.BadRequestError):
        return "OpenAI rejected the request (bad request). See logs/ai_app.log for details."
    if isinstance(exc, openai.APIStatusError):
        return f"OpenAI API error (HTTP {exc.status_code}). Please try again."
    return f"Unexpected error: {type(exc).__name__}"


def log_openai_error(action: str, exc: Exception) -> None:
    """Log error type, HTTP status and request id - useful for debugging, no secrets."""
    status = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)
    logger.error(
        "%s failed | error=%s | status=%s | request_id=%s",
        action, type(exc).__name__, status, request_id,
    )


def create_openai_client() -> OpenAI:
    # The key is read from the OPENAI_API_KEY environment variable by the SDK.
    # max_retries lets the SDK retry transient errors (429/5xx) with backoff.
    return OpenAI(timeout=60.0, max_retries=2)


# ---------------------------------------------------------------------------
# The assistant
# ---------------------------------------------------------------------------

@observe(name="assistant-request")
def answer_question(
    client: OpenAI, question: str, context_id: str, session_id: str
) -> str | None:
    """Answer a question grounded in a reference context. Returns None on API errors.

    @observe creates the ROOT span of the trace and records the function's
    arguments as input and its return value as output. The OpenAI call below
    becomes a child "generation" of this span automatically.
    """
    context = SAMPLE_CONTEXTS[context_id]
    model = get_model()

    # propagate_attributes sets trace-level attributes (name, session, tags, metadata)
    # on the current span and on every child span created inside this block.
    with propagate_attributes(
        trace_name="knowledge-assistant",
        session_id=session_id,
        tags=["assistant", "openai"],
        metadata={"context_id": context_id, "model": model},
    ):
        # Log sizes, not content: user questions may contain sensitive data.
        logger.info(
            "Request start | model=%s | context_id=%s | question_chars=%d",
            model, context_id, len(question),
        )
        start = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Reference context:\n\"\"\"\n{context['text']}\n\"\"\"\n\n"
                            f"Question: {question}"
                        ),
                    },
                ],
                temperature=0.2,
                # `name` and `metadata` are extra kwargs understood by the Langfuse
                # wrapper. They label the generation and are NOT sent to OpenAI.
                name="assistant-answer",
                metadata={"context_title": context["title"]},
            )
        except openai.OpenAIError as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            log_openai_error("Assistant request", exc)
            logger.info("Request failed after %.0f ms", latency_ms)
            # The wrapper already marked the generation as ERROR in Langfuse.
            print(f"\n[ERROR] {describe_openai_error(exc)}")
            return None

        latency_ms = (time.perf_counter() - start) * 1000
        answer = response.choices[0].message.content or ""
        usage = response.usage
        logger.info(
            "Request complete | latency_ms=%.0f | prompt_tokens=%s | completion_tokens=%s",
            latency_ms,
            usage.prompt_tokens if usage else "n/a",
            usage.completion_tokens if usage else "n/a",
        )
        print(f"\n(latency: {latency_ms:.0f} ms"
              + (f", tokens: {usage.total_tokens}" if usage else "") + ")")
        return answer


def choose_context() -> str | None:
    print("\nChoose a reference context:")
    for key, ctx in SAMPLE_CONTEXTS.items():
        print(f"  [{key}] {ctx['title']}")
    print("  [q] Quit")
    choice = input("> ").strip().lower()
    if choice in ("q", "quit", "exit"):
        return None
    if choice not in SAMPLE_CONTEXTS:
        print("Please choose one of the listed numbers.")
        return choose_context()
    return choice


def main() -> None:
    setup_logging()
    check_env()

    langfuse = get_client()
    client = create_openai_client()
    session_id = f"cli-{uuid.uuid4().hex[:8]}"  # groups this run's traces in Langfuse

    print("=" * 60)
    print("  AI Knowledge Assistant  (OpenAI + Langfuse)")
    print("=" * 60)
    print(f"Model:      {get_model()}")
    print(f"Session id: {session_id}  (search for it in Langfuse > Sessions)")
    logger.info("Assistant started | model=%s | session_id=%s", get_model(), session_id)

    try:
        while True:
            context_id = choose_context()
            if context_id is None:
                break
            ctx = SAMPLE_CONTEXTS[context_id]
            print(f"\nContext: {ctx['title']}")
            print(f"Try:     {ctx['sample_questions']}")
            question = input("\nYour question (blank to go back): ").strip()
            if not question:
                continue

            answer = answer_question(client, question, context_id, session_id)
            if answer is not None:
                print(f"\nAssistant: {answer}")
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        # Langfuse sends data in the background. flush() blocks until everything
        # is delivered - essential for short-lived scripts like this one.
        print("\nSending traces to Langfuse...")
        langfuse.flush()
        logger.info("Assistant stopped | session_id=%s", session_id)
        print("Done. Goodbye!")


if __name__ == "__main__":
    main()
