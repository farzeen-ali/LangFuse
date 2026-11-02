"""Automated hallucination evaluation (LLM-as-a-Judge) + Precision / Recall report.

Run:  python evaluator.py

What happens:
  1. A small labeled dataset of (context, question, answer) examples is loaded.
     Each answer is a fixed test case with a human label:
        1 = hallucination (contains claims NOT supported by the context)
        0 = supported     (every claim is backed by the context)
  2. An OpenAI "judge" model reads each answer + context and returns strict JSON:
        {"is_supported": bool, "support_score": 0.0-1.0, "explanation": str}
  3. The judge's verdicts become hallucination-detector predictions, and we compare
     them with the human labels using scikit-learn (confusion matrix, Precision,
     Recall, F1, False Positives, False Negatives).
  4. The same support scores are re-used with different decision thresholds to
     show the False Positive / False Negative trade-off (no extra API calls).

NOTE: This is an AUTOMATED evaluator. A judge verdict is a signal, not proof that
an answer is factually correct.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# main.py loads .env and wires up the Langfuse OpenAI wrapper on import.
from main import (
    BASE_DIR,
    SAMPLE_CONTEXTS,
    OpenAI,
    check_env,
    create_openai_client,
    describe_openai_error,
    get_model,
    log_openai_error,
    logger,
    setup_logging,
)

import openai
from langfuse import get_client, observe, propagate_attributes

RESULTS_FILE = BASE_DIR / "evaluation_results" / "results.json"

# An answer is flagged as a hallucination when support_score < threshold.
# Higher threshold = stricter detector (flags more answers).
THRESHOLDS: list[float] = [0.3, 0.5, 0.8]

# Errors that will fail for every example - stop instead of repeating them 15 times.
FATAL_ERRORS = (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError)

JUDGE_SYSTEM_PROMPT = """You are a strict fact-checking evaluator.
You will receive a REFERENCE CONTEXT, a QUESTION and an ANSWER.
Decide whether EVERY factual claim in the ANSWER is supported by the REFERENCE CONTEXT.

Rules:
- Use ONLY the reference context. Do not use outside knowledge, even if it is true.
- A claim that adds details not present in the context is NOT supported.
- A claim that contradicts the context is NOT supported.
- Paraphrases of the context ARE supported.
- An answer that says it does not know contains no unsupported claims, so it IS supported.

Respond with ONLY a JSON object, no markdown, in exactly this format:
{"is_supported": true or false, "support_score": number between 0.0 and 1.0, "explanation": "one short sentence"}

support_score is your confidence that ALL claims are supported
(1.0 = certainly all supported, 0.0 = certainly contains an unsupported claim)."""


# ---------------------------------------------------------------------------
# Labeled dataset (label: 1 = hallucination, 0 = supported)
# ---------------------------------------------------------------------------
# Context ids refer to SAMPLE_CONTEXTS in main.py:
#   "1" = Acme Cloud Storage, "2" = Company Leave Policy, "3" = Python
DATASET: list[dict[str, Any]] = [
    # --- Acme Cloud Storage ---
    {"id": "ex01", "context_id": "1", "label": 0,
     "question": "How much does the Pro plan cost?",
     "answer": "The Pro plan costs $8 per month and includes 1 TB of storage."},
    {"id": "ex02", "context_id": "1", "label": 1,
     "question": "How long are deleted files kept?",
     "answer": "Deleted files are kept in the trash for 90 days."},
    {"id": "ex03", "context_id": "1", "label": 0,
     "question": "Does the Free plan include phone support?",
     "answer": "No. Phone support is only available on the Team plan; Free users get email support."},
    {"id": "ex04", "context_id": "1", "label": 1,
     "question": "What does the Team plan include?",
     "answer": "The Team plan costs $15 per user per month, includes 5 TB of pooled storage, "
               "and comes with a 99.9% uptime SLA."},
    {"id": "ex05", "context_id": "1", "label": 0,
     "question": "What do I get on the Free plan?",
     "answer": "You get 5 GB of storage, and individual files can be up to 100 MB."},
    # --- Company Leave Policy ---
    {"id": "ex06", "context_id": "2", "label": 0,
     "question": "How many leave days can I carry over?",
     "answer": "You can carry over up to 5 unused days to next year; the rest expire."},
    {"id": "ex07", "context_id": "2", "label": 0,
     "question": "When do I need a doctor's note?",
     "answer": "A doctor's note is required for sick leave longer than 3 consecutive days."},
    {"id": "ex08", "context_id": "2", "label": 1,
     "question": "Do I get a bonus for unused leave?",
     "answer": "Yes, unused leave days are paid out as a cash bonus at the end of the year."},
    {"id": "ex09", "context_id": "2", "label": 0,
     "question": "How far in advance must I request leave?",
     "answer": "At least 2 weeks in advance, through the HR portal."},
    {"id": "ex10", "context_id": "2", "label": 1,
     "question": "How much annual leave do full-time employees get?",
     "answer": "Full-time employees get 25 days of paid annual leave per year."},
    # --- Python ---
    {"id": "ex11", "context_id": "3", "label": 0,
     "question": "Who created Python?",
     "answer": "Python was created by Guido van Rossum and first released in 1991."},
    {"id": "ex12", "context_id": "3", "label": 1,
     "question": "Is CPython written in Rust?",
     "answer": "Yes, CPython, the reference implementation, is written in Rust."},
    {"id": "ex13", "context_id": "3", "label": 0,
     "question": "When did Python 2 reach end of life?",
     "answer": "Python 2 reached end of life on January 1, 2020."},
    {"id": "ex14", "context_id": "3", "label": 1,
     "question": "Was Python 3 compatible with Python 2?",
     "answer": "Yes, Python 3.0 was released in December 2008 and was fully backward-compatible with Python 2."},
    {"id": "ex15", "context_id": "3", "label": 1,
     "question": "What does Python emphasize?",
     "answer": "Python emphasizes code readability, which is why it is the most popular language in the world."},
]


@dataclass
class JudgeResult:
    is_supported: bool
    support_score: float
    explanation: str


class FatalJudgeError(Exception):
    """Raised when continuing the run is pointless (e.g. invalid API key)."""


# ---------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------

def parse_judge_output(raw: str) -> JudgeResult | None:
    """Safely parse and validate the judge's JSON. Returns None if invalid."""
    text = raw.strip()
    # Some models wrap JSON in ```json fences despite instructions - strip them.
    if text.startswith("```"):
        text = text.strip("`")
        text = text[4:] if text.lower().startswith("json") else text
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Judge returned malformed JSON (%d chars)", len(raw))
        return None

    if not isinstance(data, dict):
        logger.warning("Judge JSON is not an object")
        return None

    is_supported = data.get("is_supported")
    score = data.get("support_score")
    explanation = data.get("explanation", "")

    # bool is a subclass of int in Python, so exclude it explicitly for the score.
    if not isinstance(is_supported, bool):
        logger.warning("Judge JSON: 'is_supported' is not a boolean")
        return None
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0.0 <= score <= 1.0:
        logger.warning("Judge JSON: 'support_score' missing or outside 0.0-1.0")
        return None
    if not isinstance(explanation, str):
        explanation = str(explanation)

    return JudgeResult(is_supported, float(score), explanation.strip()[:300])


def judge_answer(client: OpenAI, context: str, question: str, answer: str) -> JudgeResult | None:
    """Ask the judge model whether `answer` is supported by `context`."""
    try:
        response = client.chat.completions.create(
            model=get_model("OPENAI_JUDGE_MODEL"),
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"REFERENCE CONTEXT:\n\"\"\"\n{context}\n\"\"\"\n\n"
                        f"QUESTION: {question}\n\nANSWER: {answer}"
                    ),
                },
            ],
            temperature=0,  # as deterministic as possible for evaluation
            response_format={"type": "json_object"},  # JSON mode
            name="hallucination-judge",  # Langfuse generation name (not sent to OpenAI)
        )
    except FATAL_ERRORS as exc:
        log_openai_error("Judge request", exc)
        raise FatalJudgeError(describe_openai_error(exc)) from exc
    except openai.OpenAIError as exc:
        log_openai_error("Judge request", exc)
        print(f"    [judge error] {describe_openai_error(exc)}")
        return None

    return parse_judge_output(response.choices[0].message.content or "")


@observe(name="evaluate-example", as_type="evaluator")
def evaluate_example(client: OpenAI, example: dict[str, Any], run_id: str) -> JudgeResult | None:
    """One trace per dataset example: root evaluator span + judge generation + scores."""
    with propagate_attributes(
        trace_name="hallucination-evaluation",
        session_id=run_id,  # all examples of one run are grouped as a session
        tags=["evaluation", "llm-as-a-judge"],
        metadata={"example_id": example["id"], "human_label": str(example["label"])},
    ):
        context = SAMPLE_CONTEXTS[example["context_id"]]["text"]
        result = judge_answer(client, context, example["question"], example["answer"])

        if result is not None:
            # Attach the judge's verdict to the trace as Langfuse scores, so you can
            # filter/chart them in the UI.
            langfuse = get_client()
            langfuse.score_current_trace(
                name="support_score", value=result.support_score,
                data_type="NUMERIC", comment=result.explanation,
            )
            langfuse.score_current_trace(
                name="is_supported", value=1.0 if result.is_supported else 0.0,
                data_type="BOOLEAN",
            )
        return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    """Positive class = 1 = hallucination."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def print_report(title: str, m: dict[str, Any]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print("                      pred: supported   pred: hallucination")
    print(f"  actual: supported      TN = {m['true_negatives']:<3}          FP = {m['false_positives']:<3}")
    print(f"  actual: hallucination  FN = {m['false_negatives']:<3}          TP = {m['true_positives']:<3}")
    print(f"\n  Precision: {m['precision']:.2f}  (of answers flagged, how many were real hallucinations)")
    print(f"  Recall:    {m['recall']:.2f}  (of real hallucinations, how many were caught)")
    print(f"  F1 score:  {m['f1']:.2f}")
    print(f"  False Positives: {m['false_positives']}  (good answers wrongly flagged)")
    print(f"  False Negatives: {m['false_negatives']}  (hallucinations that slipped through)")


def main() -> None:
    setup_logging()
    check_env()

    langfuse = get_client()
    client = create_openai_client()
    run_id = f"eval-{uuid.uuid4().hex[:8]}"
    judge_model = get_model("OPENAI_JUDGE_MODEL")

    print("=" * 60)
    print("  Hallucination Evaluation  (automated LLM-as-a-Judge)")
    print("=" * 60)
    print(f"Judge model: {judge_model}")
    print(f"Examples:    {len(DATASET)}")
    print(f"Run id:      {run_id}  (Langfuse > Sessions)")
    print("Note: the judge is an automated signal, not proof of factual correctness.\n")
    logger.info("Evaluation started | judge_model=%s | run_id=%s | examples=%d",
                judge_model, run_id, len(DATASET))

    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    try:
        for i, example in enumerate(DATASET, 1):
            print(f"[{i:>2}/{len(DATASET)}] {example['id']}: {example['question']}")
            result = evaluate_example(client, example, run_id)
            row: dict[str, Any] = {**example, "judge": asdict(result) if result else None}
            if result is None:
                print("    -> skipped (judge failed or returned invalid JSON)")
            else:
                predicted = 0 if result.is_supported else 1
                row["predicted"] = predicted
                mark = "OK  " if predicted == example["label"] else "MISS"
                print(f"    -> {mark} label={example['label']} predicted={predicted} "
                      f"score={result.support_score:.2f} | {result.explanation}")
            rows.append(row)
    except FatalJudgeError as exc:
        print(f"\n[ERROR] {exc}\nEvaluation stopped.")
        langfuse.flush()
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("\nInterrupted. Sending traces to Langfuse...")
        langfuse.flush()
        raise SystemExit(130)

    elapsed = time.perf_counter() - start
    scored = [r for r in rows if r["judge"] is not None]
    skipped = len(rows) - len(scored)
    logger.info("Evaluation judged %d/%d examples in %.1f s (skipped=%d)",
                len(scored), len(rows), elapsed, skipped)

    if not scored:
        print("\nNo examples could be judged - see logs/ai_app.log.")
        langfuse.flush()
        raise SystemExit(1)
    if skipped:
        print(f"\nWARNING: {skipped} example(s) skipped and excluded from the metrics.")

    y_true = [r["label"] for r in scored]

    # 1) Default detector: the judge's own boolean verdict.
    default_metrics = compute_metrics(y_true, [r["predicted"] for r in scored])
    print_report("Detector report (judge's is_supported verdict)", default_metrics)

    # 2) Threshold detector: flag as hallucination when support_score < threshold.
    #    Same judge outputs, different decision rule - no new API calls.
    print("\nThreshold comparison (hallucination if support_score < threshold)")
    print("-" * 66)
    print(f"{'threshold':>9} | {'precision':>9} | {'recall':>6} | {'f1':>5} | {'FP':>3} | {'FN':>3}")
    threshold_results = []
    for t in THRESHOLDS:
        y_pred = [1 if r["judge"]["support_score"] < t else 0 for r in scored]
        m = compute_metrics(y_true, y_pred)
        threshold_results.append({"threshold": t, **m})
        print(f"{t:>9.2f} | {m['precision']:>9.2f} | {m['recall']:>6.2f} | {m['f1']:>5.2f} "
              f"| {m['false_positives']:>3} | {m['false_negatives']:>3}")
    print("\nLower threshold -> fewer answers flagged -> usually fewer False Positives")
    print("but more False Negatives. The threshold changes the decision rule, not")
    print("the judge's underlying quality. If rows look identical, the judge's")
    print("scores are clustered near 0 and 1 for this dataset.")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "judge_model": judge_model,
        "disclaimer": "Automated LLM-as-a-Judge evaluation; not proof of factual correctness.",
        "label_meaning": {"1": "hallucination", "0": "supported"},
        "examples_total": len(rows),
        "examples_judged": len(scored),
        "examples_skipped": skipped,
        "default_detector": default_metrics,
        "thresholds": threshold_results,
        "examples": rows,
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {RESULTS_FILE.relative_to(BASE_DIR)}")
    logger.info("Evaluation finished | precision=%.2f | recall=%.2f | fp=%d | fn=%d",
                default_metrics["precision"], default_metrics["recall"],
                default_metrics["false_positives"], default_metrics["false_negatives"])

    print("Sending traces to Langfuse...")
    langfuse.flush()
    print("Done.")


if __name__ == "__main__":
    main()
