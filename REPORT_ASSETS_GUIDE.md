# Sahaay Voice — report asset checklist

Use these commands and screenshots for the final Thapar-style report.

## Commands to run for Results section

```powershell
python -m pytest tests -q
python scripts/acceptance_check.py
python scripts/evaluate.py --fusion
python scripts/evaluate_nlu_dataset.py
python scripts/final_project_report_metrics.py
```

Generated files:

- `data/acceptance_report.json`
- `data/eval_report.json`
- `data/nlu_eval_report.json`
- `data/nlu_eval_summary.csv`
- `data/nlu_eval_predictions.csv`
- `data/final_project_metrics.json`
- `data/final_project_metrics.csv`

## Screenshots to collect

1. Terminal: `51 passed` from pytest.
2. Terminal: `8/8 passed` from acceptance scenarios.
3. Terminal: `9/9 passed` from fusion evaluation.
4. Elder tab: normal calibrated speech → `normal_response`.
5. Elder tab: abnormal/pause-heavy speech → `soft_check_in`, max_z high.
6. Developer evidence: baseline `max_z`, `exceed_count`, `per_feature_z`, fusion reason.
7. Operator tab: wearable simulator and reminder controls.
8. Caregiver tab: urgent safety alert after fall/no reply.
9. Clinical question deflection: non-diagnostic response + caregiver alert.
10. NLU benchmark terminal output: accuracy, macro-F1, critical-intent recall.

## Claims that are safe to make

- The prototype implements a software-only multilingual elder-care voice companion.
- It supports Safety, Companion, and Memory modes.
- It uses faster-whisper ASR, speech feature extraction, NLU intent/emotion classification, baseline deviation, and rule-based fusion.
- It passed the scripted PRD acceptance scenarios and fusion decision battery.
- It does not diagnose medical conditions; clinical-style prompts are deflected and caregiver notification is triggered.
- Wearable signals are simulated, consistent with the project scope.

## Claims not to make

- Do not claim clinical validation.
- Do not claim real wearable fall detection.
- Do not claim ASR WER unless you create a labeled wav+txt test set.
- Do not claim the MuRIL/IndicBERT model was fine-tuned unless you actually run that training.
