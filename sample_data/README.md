# Sample Data

## Files

### demo_video.mp4
A synthetic 5-second camera trap video used for demo mode and YOLOv8n testing.
Not real wildlife footage.

### speciesnet_sample_output.json
A **test fixture** containing sample SpeciesNet JSON output for unit tests.
Contains 7 prediction entries covering:
- Blank image
- Animal with species classification
- Human detection
- Multiple detections (animal + vehicle)
- Vehicle detection
- Borderline confidence prediction
- Detection failure

**This is test data only. It must NEVER be used as a real inference result.**
Real results come from running `python -m speciesnet.scripts.run_model`.
