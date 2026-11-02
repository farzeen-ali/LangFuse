# AI Observability & Evaluation Lab

A small, beginner-friendly project for learning how to **observe** and **evaluate** an LLM application in a production-conscious way.

It contains two scripts:

| Script | What it does |
|---|---|
| `main.py` | An **AI Knowledge Assistant**. It answers your questions in the terminal, using only a reference context. Every request is traced in **Langfuse** and logged to `logs/ai_app.log`. |
| `evaluator.py` | An **automated hallucination detector** that uses an LLM as a judge. It runs over a labeled dataset and reports Precision, Recall, F1, False Positives and False Negatives, then compares several decision thresholds. |

---

## 1. Technologies and why they are used

| Technology | Used for |
|---|---|
| **OpenAI Python SDK** | Calls the chat model, both for the assistant and for the judge. |
| **Langfuse Python SDK (v5)** | LLM observability. `from langfuse.openai import OpenAI` is a drop-in wrapper that records every OpenAI call (prompt, response, model, latency, token usage, cost, errors). `@observe` and `propagate_attributes` group calls into named traces. |
| **python-dotenv** | Loads API keys from a local `.env` file, so they are never hardcoded. |
| **scikit-learn** | Computes the confusion matrix, Precision, Recall and F1. |
| **logging** (standard library) | Writes timestamped application logs to `logs/ai_app.log`. |

## 2. Project structure

```
ai-observability-project/
  main.py                 # Knowledge assistant + shared helpers
  evaluator.py            # LLM-as-a-Judge + metrics + thresholds
  requirements.txt
  .env.example            # Template for your .env (no real secrets)
  .gitignore              # Keeps .env, logs and results out of git
  logs/                   # ai_app.log is written here
  evaluation_results/     # results.json is written here
```

## 3. Installation

You need Python 3.10 or newer.

```bash
cd ai-observability-project
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Configure `.env`

```bash
cp .env.example .env
```

Then edit `.env`:

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | yes | From https://platform.openai.com/api-keys |
| `OPENAI_MODEL` | no | Model for the assistant. Default `gpt-4.1-mini`. |
| `OPENAI_JUDGE_MODEL` | no | Model for the judge. Defaults to `OPENAI_MODEL`. |
| `LANGFUSE_PUBLIC_KEY` | yes | Langfuse > Project Settings > API Keys (`pk-lf-...`) |
| `LANGFUSE_SECRET_KEY` | yes | Same page (`sk-lf-...`) |
| `LANGFUSE_BASE_URL` | no | `https://cloud.langfuse.com` (EU, the default), `https://us.cloud.langfuse.com` (US), or your self-hosted URL. |

If a required variable is missing or still set to `replace-me`, both scripts stop with a clear message. The message lists the variable **names** only, never their values.

> The code passes `temperature`, and the judge also uses JSON mode (`response_format`). Both work with `gpt-4.1-mini`. Some reasoning models reject `temperature`. If you switch models and get a "bad request" error, that is the likely cause.

## 5. Run the assistant

```bash
python main.py
```

1. Pick a reference context (1–3).
2. Ask a question. Example questions are shown for each context.
3. The assistant answers using **only** that context, and prints the latency and token count.
4. Type `q` (or press Ctrl+C) to quit. Traces are flushed to Langfuse before the program exits.

Try asking something the context does *not* cover, such as "Does the Free plan include phone support?". This lets you watch how the model handles missing information.

## 6. Run the evaluation

```bash
python evaluator.py
```

This makes one judge call per example (15 calls) and prints:

- a line per example: the human label, the prediction, the judge's support score and its explanation
- a confusion matrix with Precision, Recall, F1, FP and FN
- a threshold comparison table

Full results are saved to `evaluation_results/results.json`.

### About the dataset

`DATASET` in `evaluator.py` has 15 hand-written answers: 7 hallucinations and 8 supported answers. Several of the hallucinations are subtle, for example a correct answer plus one invented detail ("99.9% uptime SLA").

These answers are **fixed test cases on purpose**. To measure a detector you need ground-truth labels, and you can only label answers that you know in advance. The answers *being evaluated* are fixed. The *judge* that evaluates them is a real OpenAI API call.

## 7. Viewing traces in Langfuse

1. Open your Langfuse project (the URL in `LANGFUSE_BASE_URL`).
2. Go to **Tracing > Traces**.
   - `knowledge-assistant` traces come from `main.py`.
   - `hallucination-evaluation` traces come from `evaluator.py`.
3. Click a trace to see the tree:

   ```
   knowledge-assistant (trace)
   └── assistant-request      span created by @observe (input: question, output: answer)
       └── assistant-answer   generation recorded by the OpenAI wrapper
                              (full prompt, response, model, latency, tokens, cost)
   ```

4. **Sessions**: each run gets an id (`cli-xxxx` or `eval-xxxx`, printed at startup). All traces from that run are grouped under it.
5. **Scores**: evaluation traces carry `support_score` (0–1) and `is_supported` scores. You can filter and chart them in the UI.
6. Failed OpenAI calls appear as generations with level `ERROR`.

If traces don't appear, check the keys and `LANGFUSE_BASE_URL` (EU and US are separate regions). Note that data is sent in the background and can take a few seconds to show up.

## 8. Precision, Recall, FP, FN

The detector's **positive class is "hallucination"**:

- `1` = hallucination (the answer contains claims the context does not support)
- `0` = supported

|                          | Predicted: supported (0) | Predicted: hallucination (1) |
|---|---|---|
| **Actual: supported (0)**     | **TN**: true negative, a good answer correctly passed | **FP**: false positive, a good answer wrongly flagged |
| **Actual: hallucination (1)** | **FN**: false negative, a hallucination that slipped through | **TP**: true positive, a hallucination correctly caught |

- **Precision** = TP / (TP + FP): *when the detector raises an alarm, how often is it right?* Low precision means many false alarms.
- **Recall** = TP / (TP + FN): *of all real hallucinations, how many did it catch?* Low recall means hallucinations reach users.
- **F1** = the harmonic mean of Precision and Recall, as one summary number.

## 9. Thresholds and the FP/FN trade-off

The judge returns a `support_score` from 0.0 to 1.0, which is its confidence that every claim is supported. The evaluator turns this into a decision:

```
hallucination  if  support_score < threshold
```

It compares thresholds `0.3`, `0.5` and `0.8` (set in `THRESHOLDS`):

- **Lower threshold** → fewer answers flagged → **fewer False Positives**, but **more False Negatives**.
- **Higher threshold** → more answers flagged → **fewer False Negatives**, but **more False Positives**.

Important points:

- Changing the threshold **does not make the judge more accurate**. It only moves the cut-off along the same scores. You are choosing which kind of error you would rather have.
- Choose the threshold based on cost. In a medical or legal assistant, a missed hallucination (FN) is expensive, so use a higher threshold. In a casual chatbot, false alarms (FP) are annoying, so use a lower one.
- LLM scores are often clustered near 0 and 1. In that case several thresholds can give identical results. That is a real finding, not a bug.
- With only 15 examples, one example changes a metric by 7–14 percentage points. Use a much larger dataset before making real decisions.
- Tuning thresholds on the same data you report on overfits. In practice, tune on one dataset and report on a separate held-out set.

## 10. Limitations of LLM-as-a-Judge

- **It is an automated signal, not proof.** The judge can be wrong in both directions.
- It only checks whether the answer is *supported by the context*. If the context itself is wrong or outdated, a "supported" answer can still be false.
- The judge can be influenced by wording, answer length or confident phrasing. It can also use outside knowledge despite instructions.
- Results can vary between models and model versions, and slightly between runs, even at `temperature=0`.
- If the judge and the assistant use the same model, they can share blind spots.
- Each evaluation costs API calls and adds latency.
- Validate the judge against human labels, which is what this project does, before you trust it.

## 11. Security precautions

- API keys are read from environment variables (`.env`) and are never hardcoded. `.env` is in `.gitignore`.
- The config check prints variable **names** only, never their values.
- Application logs record metadata only: model, context id, question **length**, latency, token counts, error type, HTTP status and request id. They do not record question text or API error bodies. Error bodies can echo part of a submitted key.
- Authentication errors are shown as a generic message rather than the raw API error text.
- We configure a dedicated `ai_app` logger, so third-party HTTP libraries don't write request details into the log file.
- **Langfuse does store full prompts and responses.** That is the point of observability. Treat your Langfuse project as sensitive data: restrict access, and don't send personal data you are not allowed to store. Langfuse offers masking options for production use.
- Rotate any key that has been accidentally committed or shared.

## 12. Implementation notes

- **Why `load_dotenv()` comes before the Langfuse import:** the Langfuse client reads its keys from the environment, so the `.env` values must be loaded first.
- **Why `name=` and `metadata=` appear in `chat.completions.create(...)`:** these are extra arguments understood by the Langfuse wrapper. They label the generation in Langfuse and are not forwarded to OpenAI.
- **Why `langfuse.flush()`:** Langfuse batches data in a background thread. Short-lived scripts must flush before exiting, or the last traces can be lost.
- **Fatal vs. recoverable errors in the evaluator:** an invalid key or an unknown model stops the run immediately. Timeouts and rate limits skip that example, and skipped examples are excluded from the metrics, with a warning.
