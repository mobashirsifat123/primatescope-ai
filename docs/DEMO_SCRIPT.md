# PrimateScope AI - Demo Walkthrough Script

> **Duration:** 7 minutes
> **Audience:** Professor and/or investor
> **Goal:** Show the offline prototype workflow, be honest about what is real vs planned, and close with the pilot ask.

---

## Before You Begin

1. Launch: `cd dist && ./LAUNCH.sh` (or `LAUNCH.bat` on Windows)
2. Wait for: `Local URL: http://localhost:8501`
3. Open browser at http://localhost:8501
4. The persistent footer reads: **Prototype v0.2 | Sample Data | YOLOv8n Baseline | MegaDetector Integration Planned** - point to it if asked about maturity.

---

## OPENING - 30 seconds

> *"This is PrimateScope AI - an offline prototype for AI-assisted primate footage review.*
>
> *The problem: months of camera-trap footage, no efficient way to find the behavior that matters. This prototype shows the target workflow: filter, detect, queue for human review, and export.*
>
> *I'll be clear throughout about what is already running and what is still planned for the pilot."*

---

## PAGE 0 - Overview - 45 seconds

**Sidebar -> Overview (default landing page)**

> *"The hero says it: months of footage, minutes to find what matters. It runs entirely offline - no cloud, no data leaving the site."*

Point to the three value cards: **Local-first**, **Human-in-the-loop**, **Export-ready workflow**.

> *"Below that, the target pipeline: upload, MegaDetector filter, YOLO primate detector, needs-review queue, human correction, CSV export."*

Point to the **Current Demo Coverage** panel:
> *"YOLOv8n baseline is implemented. Upload works. The review queue is simulated. MegaDetector integration and human-correction persistence are planned."*

Point to **Real vs Simulated**:
> *"This column is what runs today. This column is what the pilot validates - MegaDetector V6, a custom primate detector, real field data, and metrics like mAP and recall."*

---

## PAGE 1 - Live AI Analysis (Prototype Analysis View) - 1.5 minutes

**Sidebar -> Live AI Analysis**

> *"The banner at top is honest: this uses YOLOv8n baseline and sample data. Full validation is the next step."*

### Scenario 1 - Grooming (40 seconds)

1. Select **Grooming Event (M03 + F07)**
2. Click **Run Analysis**

> *"Two bounding boxes - M03 and F07. Behavior classification: Grooming at 94% model confidence."*

Point to the **Intelligence Report panel** (right side):
> *"Overall confidence 94.2%, detected entities, and an event log timeline. This is the shape of the output a reviewer would see."*

Point to the **Prototype Observation card**:
> *"A candidate observation - not a validated finding. A primatologist would confirm this before it enters the export."*

### Custom Upload (20 seconds)

> *"The upload path is real. Drop a video and YOLOv8n processes it frame-by-frame on your hardware. On CPU that's roughly 5-10 fps - a GPU would handle real-time."*

---

## PAGE 2 - Review Queue - 1 minute

**Sidebar -> Review Queue**

> *"This is the human-in-the-loop layer. Every borderline detection routes here before it becomes a final label."*

Point to the table columns:
> *"Clip ID, Station, Timestamp, Model Output, Confidence, Queue Reason, Review Status."*

> *"Queue reasons matter: borderline confidence, cross-model disagreement, person or vehicle detected, missing metadata, dense scenes, and random QA samples."*

Filter by **Person/vehicle detected**:
> *"See these flagged rows - a person near a station. That is a privacy and access audit signal, not just wildlife data."*

> *"In v0.2 this queue is an in-memory simulation. The pilot adds a persistent database and reviewer audit log."*

---

## PAGE 3 - Behavior Intelligence - 1 minute

**Sidebar -> Behavior Intelligence**

> *"Three tabs. The 24-hour Gantt shows the activity timeline - Resting, Foraging, Socializing, Traveling across a full day."*

Click **Daily Trends**:
> *"Twelve days of April. The chasing spike around April 6-8 aligns with mating-season onset. This is the kind of pattern a manual reviewer would miss."*

Click **Season Comparison**:
> *"Normal vs Mating vs Dry season. Chasing up 300% in mating season. These are candidate patterns - they need statistical validation before publication."*

---

## PAGE 4 - Individual Profiles - 45 seconds

**Sidebar -> Individual Profiles**

Select **M03**:
> *"M03, the alpha male. Biometrics, health index, troop network. Note the CONFLICT badge on M07 - a challenger male."*

Point to the **social graph**:
> *"The graph is generated from co-detection data. Node size is detection frequency. This is simulated association data in v0.2."*

---

## PAGE 5 - Field Stations - 30 seconds

**Sidebar -> Field Stations**

> *"Six simulated stations. The movement map shows coverage. STN-GAMMA has a critical battery warning - the kind of proactive alert the system would surface."*

---

## PAGE 6 - Candidate Insights & Export - 1 minute

**Sidebar -> Research Insights & Export**

> *"These are candidate insights for expert review - not validated findings. Each has a confidence rating and a draft or export-ready-draft tag."*

Point to insight 1 (Mate Guarding), then scroll to **CSV Export Preview**:
> *"This is the target export schema: project ID, image ID, MegaDetector label and confidence, YOLO label and confidence, review status, final label, reviewer, notes, and model version."*

> *"No file is written to disk in v0.2 - this is a schema preview. The pilot adds real CSV generation with threshold settings and reviewer audit metadata."*

---

## CLOSING - 30 seconds

> *"That is the prototype end-to-end.*
>
> *What runs today: a local app, YOLOv8n baseline, real upload processing, a simulated review queue, and an export schema preview.*
>
> *What the pilot delivers: MegaDetector V6 first-stage filtering, a custom fine-tuned primate detector, a persistent review database with audit logs, real field camera-trap data, and validation metrics - mAP, F1, recall, and review-time reduction.*
>
> *This is not yet a validated scientific model. It is a working prototype of the workflow that gets us there.*
>
> *Thank you."*

---

## Demo Talking Points Summary

| Page | Key Point | Honest Framing |
|---|---|---|
| Overview | Target pipeline + coverage | "YOLO implemented, MegaDetector planned" |
| Live AI | 94% model confidence | "Model output, not validated finding" |
| Review Queue | 6 queue reasons | "Simulated in v0.2, persistent in pilot" |
| Behavior Intel | +300% chasing | "Candidate pattern, needs validation" |
| Export | Full CSV schema | "Preview only, no file written" |
| Footer | Persistent badge | "Prototype v0.2, sample data" |

---

*Demo script - PrimateScope AI v0.2 - For academic and investor demonstrations*
