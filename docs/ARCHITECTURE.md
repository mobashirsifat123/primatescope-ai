# PrimateScope AI — Technical Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PRIMATE SCOPE AI                              │
│                    Field Intelligence System                          │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────────────┐       ┌──────────────────────┐
  │   CAMERA TRAP    │──────▶│   EDGE INFERENCE     │
  │   (Hardware)     │ RTSP  │   (Jetson NX)        │
  └──────────────────┘       └──────────┬───────────┘
                                        │ HTTP/WebSocket
                                        ▼
                              ┌──────────────────────┐
                              │   STREAMLIT DASHBOARD│
                              │   (Local Host)       │
                              │   PrimateScope AI    │
                              └──────────────────────┘
                                        │
                              ┌──────────┴───────────┐
                              ▼                      ▼
                    ┌──────────────┐       ┌──────────────────┐
                    │  NEO4J GRAPH │       │  POSTGRES DB     │
                    │  Social      │       │  Detections,     │
                    │  Networks    │       │  Timestamps      │
                    └──────────────┘       └──────────────────┘
```

---

## Current Demo Architecture

```
app.py (this repository)
├── Data Layer (hardcoded, in-memory)
│   ├── MONKEYS {}         # Individual monkey profiles
│   ├── DEMO_SCENARIOS {}  # 3 pre-built behavior scenarios
│   ├── FIELD_STATIONS {}  # 6 camera trap stations
│   └── INSIGHTS []        # 3 research-grade insights
│
├── Inference Layer
│   ├── YOLOv8n (ultralytics)     # Object detection
│   │   └── @st.cache_resource    # Cached model loading
│   └── CV2 (opencv-python)       # Frame extraction & annotation
│
├── Visualization Layer
│   ├── Streamlit (UI framework)
│   ├── Matplotlib (charts, camera trap view, social graphs)
│   └── NetworkX (social network analysis)
│
└── Rendering
    ├── CSS (Obsidian Canopy design system)
    ├── Google Fonts (Source Serif 4, Geist, JetBrains Mono)
    └── Dark theme throughout (#0A0F0D base)
```

---

## Production Architecture (Planned)

```
                    CAMERA TRAPS (3–6 stations)
                           │ RTSP
                           ▼
              ┌─────────────────────────┐
              │   JETSON NANO/XNX      │
              │   Edge Inference       │
              │   YOLOv8-Primate       │
              │   + Behavior Classifier │
              └────────────┬────────────┘
                           │ MQTT / HTTP
                           ▼
              ┌─────────────────────────┐
              │   CELERY WORKER        │
              │   (Async task queue)   │
              │   + Redis broker       │
              └────────────┬────────────┘
                           │
            ┌──────────────┴──────────────┐
            ▼                              ▼
   ┌──────────────────┐        ┌──────────────────┐
   │   POSTGRES        │        │   NEO4J           │
   │  detections,      │        │   social graph,   │
   │   timestamps,     │        │   centrality,     │
   │   classifications │        │   clustering      │
   └──────────────────┘        └──────────────────┘
            │                              │
            └──────────────┬───────────────┘
                           ▼
              ┌─────────────────────────┐
              │   STREAMLIT DASHBOARD   │
              │   (Researcher UI)       │
              │   + MILVUS vector DB    │
              │   (re-ID embeddings)    │
              └─────────────────────────┘
```

---

## Technology Stack

### Demo (this repository)

| Component | Technology | Version |
|---|---|---|
| UI Framework | Streamlit | 1.x |
| Object Detection | Ultralytics YOLOv8n | 8.x |
| Video Processing | OpenCV | 4.x |
| Charts | Matplotlib | 3.x |
| Social Networks | NetworkX | 3.x |
| Data | Pandas + NumPy | latest |
| Image Processing | Pillow | latest |

### Production (Phase 1–3)

| Component | Technology | Purpose |
|---|---|---|
| Edge Device | NVIDIA Jetson NX | On-site inference |
| Task Queue | Celery + Redis | Async processing |
| Relational DB | PostgreSQL | Detection records |
| Graph DB | Neo4j | Social networks |
| Vector DB | Milvus | Individual re-ID embeddings |
| API | FastAPI | Service layer |
| Deployment | Docker + K3s | Field deployment |

---

## Roadmap

### Phase 0 — Demo (NOW)
- [x] Working Streamlit dashboard
- [x] 3 hardcoded demo scenarios
- [x] YOLOv8n integration (offline-capable)
- [x] Obsidian Canopy design system
- [ ] Synthetic demo video

**Duration:** Complete  
**Status:** ✅ Delivered

---

### Phase 1 — Pilot (Next 12 weeks)
- [ ] Deploy to 3 field stations
- [ ] Calibrate YOLO on study species
- [ ] Collect first behavioral dataset
- [ ] Individual re-ID model fine-tuning
- [ ] First draft of publication

**Duration:** 12 weeks  
**Deliverable:** 1 peer-reviewed publication

---

### Phase 2 — Scale (Months 4–9)
- [ ] Add Milvus vector DB for re-ID
- [ ] Deploy to 10+ stations
- [ ] Automated insight generation
- [ ] Mobile companion app (field notes)

**Duration:** 6 months  
**Deliverable:** Open-source toolkit release

---

### Phase 3 — Production (Months 10–18)
- [ ] Edge inference on Jetson
- [ ] Multi-species support
- [ ] Real-time alert system
- [ ] Commercial licensing tier

**Duration:** 9 months  
**Deliverable:** Commercial product

---

## Data Model

### Individual Monkey Profile
```python
{
    "id": "M03",
    "sex": "Male",
    "age": "Adult",
    "group": "Alpha",
    "zone": "B",
    "detections": 47,
    "primary_assoc": "F07",  # Edge in social graph
    "movement": ["A","B","B","C","B","B","D"]  # Weekly zones
}
```

### Behavior Classification
```python
{
    "timestamp": "2026-04-08 07:42:11",
    "station": "CS-3",
    "zone": "B",
    "individuals": ["M03", "F07"],
    "behaviors": [
        ("Grooming", 0.94),
        ("Resting", 0.04),
        ("Social Proximity", 0.02)
    ],
    "confidence": "High",
    "ai_insight": "Mutual grooming detected..."
}
```

---

## Security & Privacy

- **All processing is local** — no footage leaves the field station
- **No cloud dependency** — works in areas with no connectivity
- **GDPR/ethics compliant** — individual IDs are research codes, not biometric data
- **Ethical review** — methodology approved by [Institutional Animal Care and Use Committee]

---

## Performance Benchmarks (Target)

| Metric | Target |
|---|---|
| Frame processing rate | 30 fps on Jetson NX |
| Detection latency | <100ms per frame |
| Behavior classification | <500ms per clip |
| Memory footprint | <4GB on edge device |
| Battery life | 72h continuous operation |

---

*Architecture document — Phase 0 (Demo). Production architecture subject to revision based on pilot findings.*
