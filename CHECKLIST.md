# PrimateScope AI — Production v1.0 Checklist

## Package Contents

```
dist/
├── app.py                        ✅ Streamlit app (Obsidian Canopy, 8 pages)
├── production_ui.py              ✅ Real Inference mode UI functions
├── requirements.txt              ✅ Production dependencies
├── SETUP.sh / SETUP.bat          ✅ Setup with Python 3.11/3.12 check
├── LAUNCH.sh / LAUNCH.bat        ✅ Launch scripts
├── CHECKLIST.md                  ✅ This file
│
├── database/
│   ├── db.py                     ✅ SQLite connection + schema init
│   ├── models.py                 ✅ Dataclasses + DDL (9 tables)
│   └── repositories.py           ✅ CRUD + stats repositories
│
├── services/
│   ├── speciesnet_runner.py      ✅ SpeciesNet CLI wrapper + env checks
│   ├── video_processor.py        ✅ Frame extraction + clip aggregation
│   ├── result_parser.py          ✅ SpeciesNet JSON parser
│   ├── export_service.py         ✅ CSV export (31 columns)
│   ├── file_storage.py           ✅ Safe file persistence
│   ├── bbox_draw.py              ✅ Bounding box overlay
│   ├── queue_logic.py            ✅ Review queue reason rules
│   └── pipeline.py               ✅ End-to-end orchestration
│
├── utils/
│   ├── constants.py              ✅ Statuses, file types, colors
│   ├── validation.py             ✅ Safe filenames, metadata extraction
│   └── logging_config.py         ✅ Centralized logging
│
├── tests/
│   ├── test_result_parser.py     ✅ 11 parser tests
│   ├── test_database.py          ✅ 6 database tests
│   ├── test_export_service.py    ✅ 3 export tests
│   └── test_video_processor.py   ✅ 3 video tests
│
├── scripts/
│   ├── check_environment.py      ✅ Python/package/CLI check
│   ├── create_sample_project.py  ✅ Create DB project
│   └── reset_local_db.py         ✅ Reset database
│
├── docs/
│   ├── PRODUCTION_UPGRADE.md     ✅ Architecture + changes
│   ├── SPECIESNET_INTEGRATION.md ✅ SpeciesNet CLI docs
│   ├── VALIDATION_PLAN.md        ✅ Validation framework
│   ├── ARCHITECTURE.md           ✅ (existing)
│   ├── DEMO_SCRIPT.md            ✅ (existing)
│   └── PITCH.md                  ✅ (existing)
│
├── sample_data/
│   ├── speciesnet_sample_output.json  ✅ Test fixture
│   └── README.md                      ✅ Sample data guide
│
└── assets/
    └── logo.png                  ✅ Brand logo
```

## Production Verification Commands

```bash
# 1. Environment check
python scripts/check_environment.py

# 2. Run tests
python -m pytest tests/ -v

# 3. Start app (demo mode)
streamlit run app.py --server.headless true

# 4. Verify database auto-creates
ls data/primatescope.db

# 5. Create a sample project
python scripts/create_sample_project.py "Test Project" CHN
```

## Smoke Test

- [ ] App starts without crashing
- [ ] Demo mode works (select scenario → Run Analysis)
- [ ] Real Inference mode shows engine status
- [ ] Create project works
- [ ] Upload images works (or shows clear dependency error)
- [ ] SpeciesNet runs or missing dependency is clearly shown
- [ ] Database records are created
- [ ] Review queue shows real items
- [ ] Approve/correct/reject actions work
- [ ] Notes are saved
- [ ] Review action history is saved
- [ ] CSV export downloads
- [ ] Restart app — data persists
- [ ] No fake validation metrics appear

## What Was Built

Production v1.0 upgrades the demo into a working camera-trap analysis system:

- **Real inference**: SpeciesNet/MegaDetector integration via subprocess
- **SQLite persistence**: 9 tables, auto-created, audit trail
- **Human review**: approve/correct/reject/uncertain with audit log
- **CSV export**: 31-column schema with all metadata
- **Video processing**: frame extraction + clip-level aggregation
- **Bbox overlay**: PIL-based visualization with Obsidian Canopy colors
- **23 unit tests**: parser, database, export, video processor
- **Honesty labels**: simulated pages clearly marked

## Known Limitations

- Behavior detection NOT implemented (requires behavior model)
- Individual ID NOT implemented (requires re-identification)
- Field station map is simulated (requires metadata)
- Validation metrics NOT calculated (requires ground truth)
- Custom primate model is future work
- Python 3.14 not supported (SpeciesNet compatibility)
