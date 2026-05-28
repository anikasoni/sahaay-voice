# Sahaay Voice

### An Explainable Multilingual Speech-NLP Layer for Elder Safety, Companionship, and Memory Support

Sahaay Voice is a software prototype of a multilingual voice companion designed for elderly users. It is not a generic chatbot. The system combines automatic speech recognition, speech-behavior feature extraction, multilingual intent understanding, personal-baseline deviation analysis, simulated wearable signals, and an explainable rule-based fusion layer to decide whether to respond normally, check in gently, repeat a reminder, safely deflect a clinical question, or notify a caregiver.

> **Important:** Sahaay Voice is a software prototype and **not a medical device**. It does not diagnose stroke, dementia, depression, cardiac events, or any medical condition. Clinical-style questions are routed to a safe deflection response and caregiver notification.

---

## Project Overview

Elderly users living alone or semi-independently may face three overlapping problems:

1. **Safety concerns** — falls, confusion, delayed response, unclear speech, or distress may require caregiver attention.
2. **Loneliness and companionship needs** — older adults may want warm, dignified conversation rather than task-only assistant behavior.
3. **Memory support** — medication, hydration, and routine reminders may need acknowledgement and escalation if ignored.

Most general-purpose voice assistants are not designed around multilingual Indian elder-care needs, personal speech baselines, caregiver escalation, or safety-first medical boundaries. Sahaay Voice addresses this gap through a modular speech-NLP and decision-fusion prototype.

---

## Core Innovation

Sahaay Voice is designed as an **explainable elder-care intelligence layer**, not as a black-box conversational agent.

For every user turn, the system produces an auditable evidence object containing:

* ASR transcript
* detected language
* ASR confidence estimate
* speech duration
* pause burden
* voice activity ratio
* speech rate
* clarity score
* NLU intent
* emotion estimate
* extracted slots
* personal-baseline z-score deviation
* fusion decision
* caregiver alert status

This makes caregiver escalation interpretable rather than arbitrary.

---

## Supported Modes

### 1. Safety Mode

Safety Mode is triggered by:

* simulated fall event,
* abnormal speech pattern relative to baseline,
* explicit emergency phrase,
* unclear/off-target safety confirmation.

The system asks a simple confirmation prompt such as:

> “Are you okay? Please say, I am okay.”

If the elder clearly confirms, the loop closes. If the reply is missing, unclear, or off-target, the caregiver panel receives an urgent alert.

---

### 2. Companion Mode

Companion Mode handles routine conversation and loneliness-related expressions. For example:

> “I feel lonely today.”

The system responds with a warm, dignified message. If the speech pattern also deviates from the user’s baseline, the response combines companionship with a gentle wellness check-in instead of immediately alarming the caregiver.

Example:

> “I hear you. Being alone can feel heavy sometimes, and I am right here with you. I also noticed your voice sounded a little different today. Are you feeling okay?”

---

### 3. Memory Support Mode

Memory Mode delivers caregiver-approved reminders such as medication or hydration reminders.

The system waits for acknowledgement:

* “done”
* “I took it”
* “ho gaya”
* “dawai le li”

If the reminder is not acknowledged after configured repeats, it escalates to the caregiver panel.

---

## System Architecture

```text
Microphone Audio
      ↓
Audio Capture / Recording
      ↓
Multilingual ASR
      ↓
Speech Feature Extraction
      ↓
NLU: Intent + Emotion + Slots
      ↓
Personal Baseline Comparator
      ↓
Simulated Wearable Feed
      ↓
Fusion and Escalation Engine
      ↓
Dialogue Manager
      ↓
Response Generator
      ↓
Elder UI / Caregiver Alert Panel / Operator Panel
```

---

## Key Technical Components

| Component           | Implementation                                             |
| ------------------- | ---------------------------------------------------------- |
| User Interface      | Streamlit                                                  |
| ASR                 | faster-whisper medium                                      |
| GPU Runtime         | CUDA / float16 on RTX 4070 when available                  |
| NLU                 | Hybrid deterministic safety rules + multilingual NLU layer |
| Supported languages | English, Hindi, Punjabi, and code-mixed speech             |
| Baseline tracking   | Local speech-feature baseline with z-score deviation       |
| Escalation layer    | Rule-based fusion engine                                   |
| Storage             | Local SQLite + JSONL logs                                  |
| Testing             | Pytest + scripted acceptance and fusion tests              |

---

## Explainable AI Evidence

Each processed turn stores a structured evidence object. Example fields:

```json
{
  "asr": {
    "transcript": "I feel lonely today",
    "detected_language": "en",
    "approx_confidence": 0.64
  },
  "speech_features": {
    "speech_rate_cps": 12.65,
    "clarity_score": 0.78,
    "pause_total_ms": 2320
  },
  "nlu": {
    "intent": "loneliness_expression",
    "emotion": "sad"
  },
  "baseline": {
    "max_z": 7.5,
    "exceed_count": 2,
    "sufficient_history": true
  },
  "fusion": {
    "outcome": "soft_check_in",
    "notify_caregiver": false
  }
}
```

This evidence is useful for debugging, reporting, caregiver review, and academic evaluation.

---

## Evaluation Summary

The current prototype was evaluated using automated tests and controlled scripted scenarios.

| Evaluation Component              |       Result |
| --------------------------------- | -----------: |
| Automated unit tests              | 51/51 passed |
| PRD acceptance scenarios          |   8/8 passed |
| Fusion and escalation scenarios   |   9/9 passed |
| Controlled NLU benchmark size     |  66 examples |
| Number of NLU intent classes      |           11 |
| Controlled NLU benchmark accuracy |        1.000 |
| Controlled NLU macro-F1           |        1.000 |
| Safety-critical intent recall     |        1.000 |

### Safety-Critical Intents Evaluated

* `safety_confirmation`
* `emergency_help`
* `clinical_question`
* `self_harm_risk`

The benchmark is a controlled project-specific validation set designed to verify prototype routing behavior. It should not be interpreted as population-level real-world generalization.

---

## PRD Acceptance Scenarios Covered

| Scenario                                           | Outcome |
| -------------------------------------------------- | ------- |
| Safety check after fall with clear reply           | Passed  |
| Safety check after fall with missing/unclear reply | Passed  |
| Reminder acknowledged                              | Passed  |
| Reminder repeated then escalated                   | Passed  |
| Loneliness expression handled empathetically       | Passed  |
| Clinical-style question safely deflected           | Passed  |
| Self-harm language escalated                       | Passed  |
| Code-mixed input processed                         | Passed  |

---

## Screenshots

Place screenshots inside:

```text
report_assets/screenshots/
```

Recommended screenshot names:

```text
01_prd_acceptance_8_of_8.png
02_unit_tests_51_passed.png
03_acceptance_and_fusion_console.png
04_normal_speech_ai_evidence.png
05_normal_speech_elder_interface.png
06_abnormal_speech_soft_checkin.png
07_abnormal_speech_ai_evidence.png
```

Suggested Markdown:

```markdown
![PRD Acceptance Scenarios](report_assets/screenshots/01_prd_acceptance_8_of_8.png)

![Unit Tests](report_assets/screenshots/02_unit_tests_51_passed.png)

![Normal Speech Evidence](report_assets/screenshots/04_normal_speech_ai_evidence.png)

![Abnormal Speech Soft Check-in](report_assets/screenshots/06_abnormal_speech_soft_checkin.png)
```

---

## Installation

### 1. Create and activate virtual environment

```powershell
py -3.10 -m venv sv
.\sv\Scripts\Activate.ps1
```

### 2. Install CUDA PyTorch

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install requirements

```powershell
python -m pip install -r requirements.txt
```

If `webrtcvad` fails on Windows:

```powershell
(Get-Content requirements.txt) -replace '^webrtcvad>=2.0.10','webrtcvad-wheels>=2.0.10' | Set-Content requirements.win.txt
python -m pip install -r requirements.win.txt
```

---

## Running the App

```powershell
python -m streamlit run app/main.py --server.fileWatcherType none --server.runOnSave false
```

The app contains three views:

| Tab       | Purpose                                                               |
| --------- | --------------------------------------------------------------------- |
| Elder     | Voice interaction interface                                           |
| Caregiver | Alert panel with reasons and severity                                 |
| Operator  | Simulated wearable events, reminders, baseline reset, and AI evidence |

---

## Running Tests and Metrics

### Unit tests

```powershell
python -m pytest tests -q
```

### PRD acceptance checks

```powershell
python scripts/acceptance_check.py
```

### Fusion scenario evaluation

```powershell
python scripts/evaluate.py --fusion
```

### NLU benchmark

```powershell
python scripts/evaluate_nlu_dataset.py
```

### Final project metrics

```powershell
python scripts/final_project_report_metrics.py
```

Outputs are saved under:

```text
data/
report_assets/metrics/
```

---

## Baseline Calibration

The personal baseline is built from normal speech samples.

```powershell
python scripts/calibrate_microphone.py --n 10 --seconds 5
```

After calibration:

* normal calibrated speech should remain below threshold,
* abnormal pause-heavy speech should trigger a soft check-in,
* caregiver alert should not occur unless safety criteria are met.

---

## Safety and Privacy Design

Sahaay Voice follows a conservative safety posture:

* No medical diagnosis.
* Clinical prompts are deflected.
* Self-harm language triggers caregiver escalation.
* Simulated wearable events are used instead of real hardware.
* Baselines and logs remain local.
* Raw cloud deployment and SMS/push notifications are outside prototype scope.

---

## Limitations

This prototype has important limitations:

1. It is not clinically validated.
2. It does not perform real fall detection; wearable events are simulated.
3. ASR WER has not been reported because a labeled `.wav + transcript` benchmark was not created.
4. The NLU benchmark is controlled and project-specific.
5. Punjabi ASR/TTS quality may vary.
6. The current implementation is a local software prototype, not a production caregiver platform.

---

## Future Work

Planned improvements include:

* Larger multilingual speech dataset with elderly speakers.
* ASR WER benchmarking using labeled audio.
* Fine-tuned MuRIL or IndicBERT intent classifier.
* Improved Punjabi ASR/TTS pipeline.
* Integration with real wearable sensors.
* Caregiver mobile app with push notifications.
* Longitudinal baseline adaptation and drift handling.
* Deployment as a local home voice hub.

---

## Repository

Code link:

```text
https://github.com/anikasoni/sahaay-voice
```

---

## Author

**Anika Soni**
Roll Number: **102303912**
Individual Project
Under the Guidance of: **Dr. Jasmeet Singh / Dr. Simran Setia**
Thapar Institute of Engineering and Technology
