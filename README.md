# Sahaay Voice

> A multilingual voice companion for elders. Safety. Companionship. Gentle reminders.
> **Software prototype.** Not a medical device. Does not diagnose. Hardware signals are simulated.

Built to the spec in [`Sahaay_Voice_PRD.docx`](#) (product) and [`Sahaay_Voice_TRD.docx`](#) (technical). Languages: **English · Hindi · Punjabi** (including code-mixed Hinglish-style speech).

---

## What it does

Three modes the user never needs to name:

1. **Safety Mode** — triggered by a simulated wearable event or strong speech-pattern deviation. Asks the elder a short confirmation prompt in their language. A clear "I am okay" closes the loop. A missing, unclear, or off-target reply notifies a caregiver.
2. **Companion Mode** — warm, dignified conversation. Loneliness and sadness are met with empathy. Self-harm language is intercepted by a hard guardrail before any LLM is involved.
3. **Memory Mode** — delivers a caregiver-approved reminder (medicine, hydration). Awaits acknowledgment. Repeats up to N times, then escalates.

All medical-style questions ("am I having a stroke?") route to a **safe deflection** template — never a diagnosis — and notify the caregiver.

---

## Quick start

```bash
# 1. Install CUDA-enabled PyTorch first (matches your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. Install everything else
pip install -r requirements.txt

# 3. Run the app
streamlit run app/main.py
```

That's it. The first run downloads the Whisper-medium and XLM-R checkpoints (~3 GB total) into your Hugging Face cache. After that, startup is fast.

**No GPU?** Open `config/thresholds.yaml` and set:
```yaml
asr:
  device: "cpu"
  compute_type: "int8"
  model_size: "small"   # or "base"
```

---

## The three views

The Streamlit app has three tabs you can keep open side-by-side:

| Tab | Audience | What's there |
|---|---|---|
| **👵 Elder** | the elderly user | Big mic button, current mode pill, large-text response, autoplay audio reply. |
| **👨‍👩‍👧 Caregiver** | family | Live alerts panel with timestamps, severity, reasons, and the recent transcript. Recent conversation log. |
| **🛠 Operator** | you, demoing | Trigger simulated wearable events (fall, inactivity). Force Safety Mode in any language. Deliver any reminder. Reset baseline. Wipe all data. |

For a recorded demo or a screencast, use the headless runner:
```bash
python scripts/run_demo.py samples/my_clip.wav --json
```

---

## Demo walkthrough → PRD §10 acceptance scenarios

These are the scenarios the PRD calls out, with exactly how to demo each:

| # | Scenario | How to demo |
|---|---|---|
| **A1** | Safety check after simulated fall, clear reply | Operator → "Fall (high)". Switch to Elder. Record "I am okay" or "main theek hoon". Mode returns to idle. |
| **A2** | Safety check after fall, no clear reply | Operator → "Fall (high)". Stay silent / mumble. Caregiver tab gets an urgent alert. |
| **A3** | Reminder acknowledged | Operator → pick a reminder → "Deliver reminder now". Elder records "done" / "ho gaya". Memory closes. |
| **A4** | Reminder repeated then escalated | Deliver reminder, ignore it, record an unrelated reply (twice more). Caregiver gets a warning. |
| **A5** | Companion check-in after sadness | Record "I feel so lonely today" / "main akela hoon". You'll get an empathetic line, not a probing question. |
| **A6** | Clinical question deflection | Record "am I having a stroke?" / "kya mujhe stroke ho raha hai?". Response is the safe-deflection template; caregiver tab gets a warning. |
| **A7** | Speech deviation soft check-in | Collect a few clean recordings to build the baseline (Operator → snapshot to confirm `sample_count ≥ 5`). Then degrade one (`scripts/degrade_audio.py clip.wav --severity severe`) and record it. You should get a soft check-in. |
| **A8** | Self-harm language | Record a sentence containing "I want to die". The response is the gentle self-harm template; a caregiver alert fires regardless of LLM availability. |
| **A9** | Code-mixed (Hinglish) input | Record "main bahut tired hoon today". Detected language tracks naturally; response in the same register. |
| **A10** | Punjabi reminder | Operator → deliver `med-evening-sugar`. Reply in Punjabi. |
| **A11** | Operator data wipe | Operator → "Delete all user data". Confirm baseline + logs + alerts cleared. |
| **A12** | TTS fallback | Disable network temporarily; the cached Safety prompts still play (pre-synthesized at startup). |

---

## Project layout

```
sahaay_voice/
├─ app/                # Streamlit UI (Elder / Caregiver / Operator tabs)
├─ core/               # Pipeline modules — each is independently testable
│  ├─ audio_capture.py # 16kHz mic + VAD utterance segmentation
│  ├─ asr.py           # faster-whisper, multi-language
│  ├─ features.py      # latency, pauses, speech rate, clarity, repetition
│  ├─ nlu.py           # zero-shot XLM-R + optional fine-tuned MuRIL
│  ├─ baseline.py      # Welford running stats in SQLite — no raw samples kept
│  ├─ wearable.py      # simulated wearable event publisher
│  ├─ fusion.py        # rule-based decision: normal / soft / alert / deflect
│  ├─ alerts.py        # caregiver alert store
│  ├─ response.py      # template-first responses + guarded LLM hook
│  ├─ tts.py           # gTTS primary + pyttsx3 fallback + Coqui hook
│  ├─ dialogue.py      # mode state machine; orchestrates a full turn
│  ├─ schemas.py       # dataclasses (single source of typed truth)
│  ├─ logger.py        # JSONL turn logger
│  └─ config.py        # YAML loader + paths
├─ config/             # All tunables live here, no code changes needed
│  ├─ thresholds.yaml  # z-scores, timing, ASR/TTS engine choices
│  ├─ prompts.yaml     # every user-facing string in en/hi/pa
│  ├─ reminders.yaml   # caregiver-approved reminders
│  └─ intents.yaml     # intent labels + example sentences
├─ scripts/
│  ├─ run_demo.py      # headless single-file pipeline runner
│  ├─ evaluate.py      # WER, intent F1, baseline P/R, fusion scenarios
│  ├─ degrade_audio.py # produce simulated degraded speech
│  └─ finetune_muril.py# scaffold to fine-tune MuRIL on labeled data
├─ tests/              # pytest suite — runs in seconds, no GPU needed
├─ data/               # SQLite baseline, JSONL logs, audio cache — local only
└─ requirements.txt
```

---

## Configuration

Everything you'd want to tweak lives under `config/` and is hot-reloadable from the Operator panel.

**`thresholds.yaml`** drives behavior:
- `baseline.z_high / z_low` — when speech features are "far enough" from the user's normal to trigger a soft check-in.
- `safety_mode.repetition_threshold` / `clarity_threshold` — how clear and on-target a Safety-Mode reply needs to be.
- `companion_mode.use_local_llm` — flip to `true` to wire in a local Ollama model (see below).

**`prompts.yaml`** — every user-facing string in `en`, `hi`, `pa`. Edit freely.

**`reminders.yaml`** — caregiver-approved schedule. Edit the file or replace it with output from your caregiver UI.

---

## Models — what we chose and why

| Stage | Default | Why |
|---|---|---|
| ASR | **faster-whisper `medium`**, `float16`, `cuda` | Sweet spot on a 4070; handles English/Hindi/Punjabi out of the box and detects code-mixed reasonably well. Cold-load ~5s; ~5× realtime after. Swap to `large-v3` in config for better Punjabi at ~2× cost. |
| NLU | **zero-shot XLM-R/MNLI** (`joeddav/xlm-roberta-large-xnli`) | Works on en/hi/pa with zero labeled data. A MuRIL fine-tune is a one-flag swap-in once you have data — see [Fine-tuning MuRIL](#fine-tuning-muril). |
| TTS | **gTTS** for all three languages | Pragmatic choice. Local Coqui XTTS-v2 is supported (flip `tts.engine_primary: xtts`) but Punjabi quality is the weakest link in any current local engine; gTTS is the most reliable for demo. Offline `pyttsx3` is the documented fallback per the TRD. |
| Companion LLM | **disabled by default; templates only** | Safer for a demo, no extra infra. To enable: install Ollama, pull a small instruct model, set `companion_mode.use_local_llm: true` in `thresholds.yaml`. The system prompt is hardened with hard rules in `core/response.py`. **Safety Mode is templates only, ever.** |

---

## Fine-tuning MuRIL

When you have labeled examples (real users or expanded synthetic):

```bash
# 1. Build data/intent_train.jsonl with rows like:
#    {"text": "main theek hoon", "intent": "safety_confirmation"}

# 2. Train
python scripts/finetune_muril.py --epochs 3 --batch 16

# 3. Activate
export SAHAAY_FINETUNED=1
streamlit run app/main.py
```

If `data/intent_train.jsonl` doesn't exist, the script seeds itself from `config/intents.yaml` — enough to smoke-test the pipeline, *not* enough to outperform zero-shot. Expand it before relying on it.

---

## Evaluation

```bash
# Full battery — only suites with data on disk run
python scripts/evaluate.py --all

# Just the fusion scenario battery — runs in seconds, no GPU
python scripts/evaluate.py --fusion
```

The harness produces `data/eval_report.json` with:
- **ASR** — WER per language on any folder of `.wav` + sibling `.txt` references.
- **NLU** — accuracy + macro-F1 on a labeled jsonl test set.
- **Baseline detection** — precision/recall of soft-check-in trigger on clean vs. degraded paired samples. Generate degraded samples with `scripts/degrade_audio.py`.
- **Fusion** — pass/fail on a fixed battery covering every TRD §4.7.2 row.

---

## Privacy

- **All data stays on this machine.** No analytics, no telemetry, no cloud calls except the TTS request to Google's gTTS endpoint (which sends *response text*, never user speech). To go fully offline, set `tts.engine_primary: pyttsx3` or `xtts`.
- **No raw audio is stored after a turn closes** unless you point a file path at it. The baseline tracks running mean and variance only (Welford) — no raw feature samples, no recoverable user content.
- **One-button data wipe.** Operator tab → "Delete all user data" clears the SQLite baseline, all JSONL logs, all alerts, and all cached TTS audio.
- **Caregiver visibility is bounded.** The caregiver panel shows alerts and a short recent transcript; it does not stream raw audio.

These choices implement PRD §1.3 (privacy posture), FR-36..FR-39 (data minimization and controls), and TRD §9 (local-only storage, operator delete control).

---

## Running tests

```bash
pytest tests/
```

The default test suite mocks the heavy ASR/NLU models so it runs in a few seconds without GPU. Coverage:

- `test_features.py` — pause stats, clarity bounds, repetition score (exact / partial / unrelated / empty).
- `test_baseline.py` — Welford correctness, outlier z-score detection, abnormal samples excluded, reset.
- `test_fusion.py` — every outcome branch in TRD §4.7.2.
- `test_nlu.py` — keyword fast paths in all three languages + regex slot extraction.
- `test_dialogue.py` — manager-level scenario tests with mocked ASR/NLU for the PRD §10 acceptance flows.

---

## Hardware and out-of-scope reminders

Per PRD §1.2, **hardware wearables are out of scope**. Falls, inactivity, and abnormal motion are simulated through the Operator panel and the `core/wearable.py` publisher API. Wiring a real device would replace `wearable_mod.publish(...)` with a transport that calls the same function from your device-side bridge — no changes to fusion or dialogue.

Per PRD §1.3, **Sahaay Voice never diagnoses anything**. Every clinical question routes to the safe-deflection template, and a caregiver alert is raised. This is enforced in `core/fusion.py` and re-enforced in `core/response.py`; the LLM hook in Companion Mode is hard-disabled in Safety Mode and behind a self-harm guardrail in Companion Mode.

---

## License

Prototype code. Add a license file before any external use.


## PRD acceptance checker

For a fast no-model verification of the Safety, Companion, Memory, clinical-deflection, self-harm, and code-mixed acceptance paths:

```bash
python scripts/acceptance_check.py
```

For objective rule/fusion checks and unit tests:

```bash
pytest tests/
python scripts/evaluate.py --fusion
```

Accuracy targets are configurable in `config/thresholds.yaml` under `accuracy_criteria`.

---

## Phase 1 AI evidence panel

The Operator tab now includes **AI pipeline evidence** for the latest turn:

- ASR backend, model size, device, compute type, detected language, transcript, avg log-probability, approximate confidence.
- Speech processing features: latency, speech duration, internal pauses, voice-activity ratio, speech rate, clarity, Safety repetition score.
- NLU evidence: intent, confidence, emotion, emotion confidence, slots.
- Baseline comparator: max z-score, per-feature z-scores, sufficient-history flag.
- Fusion outcome: normal response, soft check-in, reminder repeat, caregiver alert, or safe deflection, with human-readable reason.

Export recent evidence from logs:

```bash
python scripts/pipeline_report.py --n 25
```

This creates:

```text
data/pipeline_evidence_report.json
data/pipeline_evidence_report.csv
```

---

## Phase 1.5 — Speech/NLP Evidence Cleanup

This build adds a more defensible AI evidence layer for evaluators:

- **Latency is now estimated as prompt end → first voiced frame**, rather than a fixed zero.
- **Pause features are separated** into leading silence, trailing silence, and true internal pauses.
- **Baseline pause tracking uses internal pauses only**, so leaving the recorder open after speaking does not falsely inflate abnormality.
- **Operator → AI pipeline evidence** shows an interpretation table for each speech feature, including whether it is used in the baseline.
- **Loneliness + abnormal speech** now produces a combined companion + gentle wellness check-in response instead of jumping straight to a hard Safety prompt.

Validation commands:

```bash
python -m pytest tests -q
python scripts/acceptance_check.py
python scripts/evaluate.py --fusion
python scripts/pipeline_report.py --n 25
```

---

## Phase 1.6 — ASR/NLU Quality Gate

The app now treats weak speech recognition as an evidence state rather than pretending it understood the elder.

If ordinary Companion/idle speech has low ASR confidence, low clarity, short voiced duration, and low NLU confidence, Sahaay asks the elder to repeat and does **not** update the speech baseline from that noisy turn. Critical safety intents (`clinical_question`, `self_harm_risk`, `emergency_help`, `caregiver_request`, `safety_confirmation`) are never blocked by this gate.

This prevents outputs like a weak transcript (`"I only today"`) being treated as meaningful casual chat. The Operator evidence panel shows the reason, for example: `ASR/NLU confidence weak; asking elder to repeat`.
