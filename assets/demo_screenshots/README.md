# Demo Screenshots

This folder is for storing screenshots captured during demo presentations.

## Suggested Screenshots to Capture

Before your demo meeting, run the app and capture these screenshots:

1. **`page1_grooming_analysis.png`** — Page 1 with Grooming scenario results
   - Show: bounding boxes, behavior bars, AI insight card

2. **`page1_chasing_analysis.png`** — Page 1 with Chasing scenario results
   - Show: two bounding boxes (teal + amber), chasing behavior at 91%

3. **`page1_feeding_analysis.png`** — Page 1 with Feeding scenario results
   - Show: single bounding box, feeding behavior at 89%

4. **`page2_field_stations.png`** — Page 2 overview
   - Show: map, status indicators, metric cards (5/6 active, 637 images)

5. **`page3_timeline.png`** — Page 3 Activity Timeline tab
   - Show: 12-day line chart with chasing spike visible

6. **`page3_comparison.png`** — Page 3 Mating Season Comparison
   - Show: side-by-side bar charts with amber alert box

7. **`page4_social_network.png`** — Page 4 with M03 selected
   - Show: network graph with central M03-F07 cluster

8. **`page5_insights.png`** — Page 5 Research Insights
   - Show: all 3 insight cards with confidence/ready chips

## How to Capture

**macOS:**
- Cmd+Shift+4 → drag to select region
- Or: Cmd+Shift+5 → Capture Entire Screen / Selected Window

**Windows:**
- Win+Shift+S → select region
- Or: Win+PrtSc → full screen (saved to Pictures folder)

**Linux:**
- Shift+PrintScreen → select region

## File Naming Convention

Use lowercase with underscores:
```
page1_[scenario]_analysis.png
page[N]_[tab_name].png
YYYYMMDD_[descriptive_name].png
```

## Placeholder Image (delete before presenting)

The image below is a placeholder — delete this file and replace with actual screenshots:

```
dist/assets/demo_screenshots/
├── README.md         ← this file
└── [your screenshots go here]
```

## After Capturing

1. Compress: `cd dist/assets && tar -czf screenshots.tar.gz demo_screenshots/`
2. Or share individually via email/Drive
3. Delete this README from the folder before presenting (optional)
