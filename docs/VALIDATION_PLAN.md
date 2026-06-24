# Validation Plan

## Important Disclaimer

PrimateScope AI v1.0 provides **AI-assisted pre-labeling** only. It is NOT a
scientifically validated model. All predictions require expert review before
use in any publication, conservation decision, or statistical analysis.

## What v1.0 Provides

- Species-level predictions from SpeciesNet (when installed)
- Detection bounding boxes from MegaDetector
- Human-in-the-loop review workflow
- Audit-tracked corrections
- Export of reviewed data

## What v1.0 Does NOT Provide

- Precision, recall, F1, or mAP metrics
- Species accuracy rates
- Confusion matrices
- Per-station or day/night performance breakdowns
- Validation against ground truth

## Future Validation Metrics

When expert-verified ground truth labels exist, the following metrics can be
calculated:

| Metric | Description |
|---|---|
| Precision | TP / (TP + FP) per class |
| Recall | TP / (TP + FN) per class |
| F1-score | Harmonic mean of precision and recall |
| mAP | Mean average precision across all classes |
| Species accuracy | Correct species / total reviewed |
| Blank-filter FNR | False negative rate for blank filtering |
| Review-time reduction | Time saved vs manual review |
| Confusion matrix | Per-class misclassification |
| Per-species accuracy | Accuracy for each species |
| Per-station performance | Accuracy by camera station |
| Day/night performance | Accuracy by time of day |
| Video clip-level accuracy | Accuracy on video frame predictions |

## Ground Truth Support

The database schema is designed so reviewed labels can become ground truth:

- `review_items.final_label` — the human-verified label
- `review_items.final_species` — the human-verified species
- `review_actions` — full audit trail of corrections

Future steps:
1. Upload a ground-truth CSV
2. Compare AI predictions vs ground truth
3. Calculate metrics automatically
4. Display on a validation dashboard

## Scientific Use

For scientific publication, researchers must:
1. Review every AI prediction manually
2. Document the review process
3. Report model version and confidence thresholds
4. Validate on a held-out ground-truth set
5. Report species-level accuracy and failure modes
