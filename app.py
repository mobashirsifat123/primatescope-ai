import os
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches
import networkx as nx
from PIL import Image
import streamlit as st

os.environ["YOLO_VERBOSE"] = "false"

try:
    import cv2
except Exception:
    cv2 = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from database.db import init_db
    from database.repositories import ProjectRepo, StatsRepo
    from production_ui import (
        get_dashboard_stats,
        page_live_analysis_real,
        page_research_insights_real,
        page_review_queue_real,
        render_production_sidebar,
    )
    _PROD_AVAILABLE = True
    _PROD_ERROR = ""
    _PROD_TRACEBACK = ""
except Exception as _prod_err:
    _PROD_AVAILABLE = False
    _PROD_ERROR = str(_prod_err)
    import traceback as _tb
    _PROD_TRACEBACK = _tb.format_exc()

    def init_db():
        pass

    def render_production_sidebar():
        return "Demo Simulation"

    def get_dashboard_stats():
        return None

    def page_live_analysis_real():
        pass

    def page_review_queue_real():
        pass

    def page_research_insights_real():
        pass


# =============================================================================
# OBSIDIAN CANOPY — DESIGN TOKENS
# =============================================================================
BG         = "#0A0F0D"
SURFACE    = "#141C19"
SURFACE_HI = "#1C2623"
BORDER     = "#1E292B"
SLATE      = "#94A3B8"
TEAL       = "#0adec8"
AMBER      = "#FFB84D"
WHITE      = "#FFFFFF"
ERROR_CLR  = "#FFB4AB"
TEXT_SEC   = SLATE

matplotlib.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "savefig.facecolor":BG,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  TEXT_SEC,
    "xtick.color":      TEXT_SEC,
    "ytick.color":      TEXT_SEC,
    "text.color":       WHITE,
    "axes.titlecolor":  WHITE,
    "axes.grid":        False,
    "font.size":        11,
    "font.family":      "sans-serif",
})


# =============================================================================
# DATA MODELS
# =============================================================================
MONKEYS = {
    "M03": {"sex":"Male",   "age":"Adult",   "group":"Alpha", "zone":"B",
            "detections":47, "primary_assoc":"F07",
            "movement":["A","B","B","C","B","B","D"],
            "est_age":"22-25 Yrs", "weight":"185 kg +/- 5%", "health":85,
            "health_note":"Minor laceration on left deltoid (Healing: Day 4).",
            "troop":[
                {"id":"F07","role":"Primary Mate","status":"ok"},
                {"id":"I04","role":"Offspring (2yo)","status":"ok"},
                {"id":"M07","role":"Challenger Male","status":"conflict"},
            ]},
    "F07": {"sex":"Female", "age":"Adult",   "group":"Alpha", "zone":"B",
            "detections":52, "primary_assoc":"M03",
            "movement":["B","B","B","B","C","B","B"],
            "est_age":"18-20 Yrs", "weight":"92 kg +/- 3%", "health":92,
            "health_note":"Excellent condition. No clinical concerns.",
            "troop":[
                {"id":"M03","role":"Primary Mate","status":"ok"},
                {"id":"I04","role":"Offspring (2yo)","status":"ok"},
            ]},
    "M07": {"sex":"Male",   "age":"Subadult","group":"Alpha", "zone":"C",
            "detections":31, "primary_assoc":"M03",
            "movement":["C","C","B","C","C","D","C"],
            "est_age":"9-11 Yrs", "weight":"78 kg +/- 4%", "health":78,
            "health_note":"Elevated stress markers. Displacement behavior noted.",
            "troop":[
                {"id":"M03","role":"Alpha Male","status":"conflict"},
            ]},
    "F02": {"sex":"Female", "age":"Adult",   "group":"Alpha", "zone":"D",
            "detections":38, "primary_assoc":"M03",
            "movement":["D","D","C","D","D","B","D"],
            "est_age":"20-23 Yrs", "weight":"88 kg +/- 3%", "health":88,
            "health_note":"Normal. Dietary shift observed.",
            "troop":[
                {"id":"M01","role":"Associate","status":"ok"},
            ]},
    "M01": {"sex":"Male",   "age":"Adult",   "group":"Beta",  "zone":"A",
            "detections":29, "primary_assoc":"F02",
            "movement":["A","A","B","A","A","A","B"],
            "est_age":"15-17 Yrs", "weight":"165 kg +/- 4%", "health":81,
            "health_note":"Normal locomotion. No respiratory distress.",
            "troop":[
                {"id":"F02","role":"Associate","status":"ok"},
            ]},
    "I04": {"sex":"Unknown","age":"Infant",  "group":"Alpha", "zone":"B",
            "detections":18, "primary_assoc":"F07",
            "movement":["B","B","B","B","B","C","B"],
            "est_age":"2 Yrs", "weight":"14 kg +/- 2%", "health":95,
            "health_note":"Healthy development. Consistent with mother.",
            "troop":[
                {"id":"F07","role":"Mother","status":"ok"},
            ]},
}

DEMO_SCENARIOS = {
    "Grooming": {
        "label":      "Grooming Event (M03 + F07)",
        "behaviors":  [("Grooming",0.94),("Resting",0.04),("Social Proximity",0.02)],
        "individuals":["M03","F07"],
        "insight":    ("Mutual grooming detected between M03 and F07. "
                       "Duration: 4m 12s. Consistent with pair-bond maintenance behavior."),
        "boxes": [
            {"xy":(120,100), "size":(220,320), "color":TEAL, "label":"M03  Grooming  94%"},
            {"xy":(280,140), "size":(200,280), "color":TEAL, "label":"F07  Grooming  91%"},
        ],
        "confidence": 94.2,
        "events": [
            ("00:14:10","Motion Triggered","Sensor A2"),
            ("00:14:15","Subject Enters Frame","Quad 4"),
            ("00:14:22","Positive Identification","Activity: Foraging"),
            ("00:14:45","Subject Exits Frame","Tracking lost"),
        ],
        "entities": [
            ("Pan troglodytes", 0.94, True),
            ("Foliage Disturbance", 0.42, False),
        ],
    },
    "Chasing": {
        "label":      "Chasing Event (M03 + M07)",
        "behaviors":  [("Chasing",0.91),("Aggression",0.07),("Locomotion",0.02)],
        "individuals":["M03","M07"],
        "insight":    ("High-velocity pursuit detected. M03 chasing M07 across Zone B->C. "
                       "Third such event this week."),
        "boxes": [
            {"xy":(60,110), "size":(200,300), "color":TEAL,  "label":"M03  Chasing  91%"},
            {"xy":(420,130),"size":(180,260), "color":AMBER,  "label":"M07  Fleeing  87%"},
        ],
        "confidence": 91.3,
        "events": [
            ("00:08:02","Motion Triggered","Sensor B1"),
            ("00:08:05","Two Subjects Detected","Quad 2-3"),
            ("00:08:12","Chasing Behavior Classified","Velocity: 4.2 m/s"),
            ("00:08:28","Pursuit Ends","Zone C boundary"),
        ],
        "entities": [
            ("Pan troglodytes", 0.91, True),
            ("Pan troglodytes", 0.87, True),
        ],
    },
    "Feeding": {
        "label":      "Feeding Event (M03 + F02)",
        "behaviors":  [("Feeding",0.89),("Foraging",0.08),("Locomotion",0.03)],
        "individuals":["M03","F02"],
        "insight":    ("Bamboo feeding detected near Station 6. Dietary shift from "
                       "fruit-heavy Zone A observed since April 3."),
        "boxes": [
            {"xy":(180,130),"size":(240,310), "color":TEAL,  "label":"M03  Feeding  89%"},
        ],
        "confidence": 89.0,
        "events": [
            ("00:03:15","Motion Triggered","Sensor D2"),
            ("00:03:20","Subject Identified","M03 (Facial ID: 98.4%)"),
            ("00:03:25","Feeding Behavior Classified","Bamboo species"),
            ("00:03:55","Subject Exits Frame","Southward direction"),
        ],
        "entities": [
            ("Pan troglodytes", 0.89, True),
            ("Bamboo sp.", 0.76, False),
        ],
    },
}

SCENARIO_OPTIONS = [
    "Custom Upload",
    "Grooming Event (M03 + F07)",
    "Chasing Event (M03 + M07)",
    "Feeding Event (M03 + F02)",
]
SCENARIO_KEY_BY_LABEL = {
    "Grooming Event (M03 + F07)": "Grooming",
    "Chasing Event (M03 + M07)":  "Chasing",
    "Feeding Event (M03 + F02)":  "Feeding",
}

FIELD_STATIONS = pd.DataFrame({
    "station":     ["CS-1","CS-2","CS-3","CS-4","CS-5","CS-6"],
    "lat":         [25.045,25.048,25.042,25.050,25.038,25.052],
    "lon":         [102.710,102.715,102.708,102.720,102.705,102.722],
    "status":      ["Online","Online","Online","Low Battery","Online","Online"],
    "images_today":[124,89,156,0,67,201],
    "battery":     [84,91,76,12,88,95],
    "signal":      [98,95,92,24,97,99],
    "last_det":    ["12m ago","8m ago","22m ago","4h ago","31m ago","3m ago"],
})

STATION_DETAIL = {
    "CS-1": {"name":"STN-ALPHA", "signal":98, "battery":84, "last":"12m ago", "warning": False},
    "CS-2": {"name":"STN-BETA",  "signal":95, "battery":91, "last":"8m ago",  "warning": False},
    "CS-3": {"name":"STN-DELTA", "signal":92, "battery":76, "last":"22m ago", "warning": False},
    "CS-4": {"name":"STN-GAMMA", "signal":24, "battery":12, "last":"4h ago",  "warning": True},
    "CS-5": {"name":"STN-EPSILON","signal":97,"battery":88, "last":"31m ago", "warning": False},
    "CS-6": {"name":"STN-ZETA",  "signal":99, "battery":95, "last":"3m ago",  "warning": False},
}

RECENT_ACTIVITY = [
    ("Orangutan transit detected", "04:22", TEAL),
    ("Multiple subjects (Macaca)", "02:15", TEAL),
    ("No motion detected", "--:--", SLATE),
]

ALERTS = [
    {"type":"Review Required","color":AMBER,"title":"Anomalous Movement Detected",
     "detail":"Station 12 - Sector Gamma","time":"10:42 AM"},
    {"type":"Detected","color":TEAL,"title":"Gorilla beringei identified (Confidence 94%)",
     "detail":"Station 04 - Sector Alpha","time":"09:15 AM"},
    {"type":"Archived","color":SLATE,"title":"Routine System Check",
     "detail":"All Stations Online","time":"Yesterday"},
]

GLOBAL_STATS = [
    {"label":"Total Detections","value":"24,892","icon":"visibility",
     "trend":"+12.4% vs last period","trend_color":TEAL},
    {"label":"Species Diversity (H')","value":"3.42","icon":"diversity_2",
     "trend":"Stable across sectors","trend_color":SLATE},
    {"label":"Active Stations","value":"42/45","icon":"sensors",
     "trend":"3 stations require maintenance","trend_color":ERROR_CLR},
]

GANTT_TRACKS = [
    {"label":"Resting","color":SLATE,"segments":[
        (0, 5.5, "98% CONF"), (21, 24, "99% CONF")]},
    {"label":"Foraging","color":TEAL,"segments":[
        (6, 10.5, "FRUIT/LEAF"), (15, 18.5, "INSECT")]},
    {"label":"Socializing","color":"#c0c9c3","segments":[
        (11, 13, ""), (19, 20.5, "")]},
    {"label":"Traveling","color":"#8d928f","segments":[
        (5.5, 6, ""), (10.5, 11, ""), (13, 15, "1.2KM"),
        (18.5, 19, ""), (20.5, 21, "")]},
]

GANTT_ANNOTATIONS = [
    {"label":"ANNOTATION_01","color":TEAL,
     "text":"Prolonged mid-day traveling interval (13:00-15:00) correlates with expected territorial patrol patterns observed in sector 7G."},
    {"label":"ANNOTATION_02","color":"#c0c9c3",
     "text":"Socializing events closely precede resting periods, confirming typical group bonding behavior post-foraging."},
    {"label":"DATA_GAP","color":ERROR_CLR,
     "text":"Sensor interference noted between 14:15-14:30. Confidence score dropped below 85% threshold during this span."},
]

INSIGHTS = [
    {"title":"Mating Season Mate Guarding","period":"April 3-12, 2026",
     "content":("Adult male M03 was detected near female F07 on 8 separate occasions "
                "across Camera Stations 2, 3, and 4. Chasing behavior increased by 43% "
                "compared with the previous two weeks (n=12 vs n=8.4 baseline). Feeding "
                "activity shifted from fruit-heavy Zone A to bamboo-rich areas near "
                "Camera Station 6, consistent with range restriction during mate guarding."),
     "conf":"High","ready":True},
    {"title":"Social Network Restructuring","period":"April 1-15, 2026",
     "content":("Betweenness centrality of M03 increased from 0.12 to 0.31 during mating "
                "season. Subadult male M07 was displaced from proximity to F07, with "
                "direct M07-F07 interactions dropping from 6/week to 0/week. Infant I04 "
                "maintained consistent association with F07 (mother) across all observations."),
     "conf":"Medium","ready":True},
    {"title":"Dietary Shift Alert","period":"April 5-10, 2026",
     "content":("Bamboo feeding detections increased 210% compared to March baseline. "
                "Fruit consumption decreased proportionally. This correlates with seasonal "
                "bamboo shoot availability at 1,800m elevation. Recommend ground-truthing "
                "with botanical survey at Station 6."),
     "conf":"High","ready":False},
]

ACTIVITY_DAYS  = [f"Apr {d}" for d in range(1,13)]
ACTIVITY_DATA  = pd.DataFrame({
    "Feeding":        [12,15,10,14,18,16,13,11,17,20,19,15],
    "Grooming":       [8,6,9,7,10,8,6,5,7,9,8,6],
    "Resting":        [22,25,20,23,18,21,24,26,22,19,20,23],
    "Chasing":        [2,1,3,2,5,8,7,6,4,3,2,1],
    "Group_Movement": [9,11,8,10,13,15,12,10,11,9,8,7],
}, index=ACTIVITY_DAYS)

NORMAL_SEASON  = {"Feeding":45,"Grooming":28,"Resting":62,"Chasing":3,"Proximity":15}
MATING_SEASON   = {"Feeding":38,"Grooming":22,"Resting":45,"Chasing":12,"Proximity":34}
DRY_SEASON      = {"Feeding":52,"Grooming":18,"Resting":71,"Chasing":1,"Proximity":8}

EXPORT_FORMATS = [
    {"id":"pdf","name":"PDF Summary","desc":"Executive overview with charts","icon":"picture_as_pdf"},
    {"id":"csv","name":"Raw CSV","desc":"Full detection logs (Time-series)","icon":"table_chart"},
    {"id":"health","name":"Station Health","desc":"Battery, temp, signal diagnostics","icon":"monitor_heart"},
]

SPECIES_OPTIONS = [
    "All Species Detected",
    "Pan troglodytes (Chimpanzee)",
    "Gorilla beringei (Mountain Gorilla)",
    "Pongo abelii (Sumatran Orangutan)",
    "Macaca mulatta (Rhesus Macaque)",
]

NAV_PAGES = [
    "Overview",
    "Research Dashboard",
    "Camera-Trap Analysis Workbench",
    "Review Queue",
    "Behavior Intelligence",
    "Field Stations",
    "Individual Profiles",
    "Research Insights & Export",
]

VALUE_PROPS = [
    {"icon":"wifi_off","title":"Local-first Analysis",
     "desc":"Runs entirely offline on field hardware. No cloud, no data leaving the study site. "
            "Real camera-trap images and videos processed locally by SpeciesNet and MegaDetector."},
    {"icon":"rate_review","title":"Human-in-the-loop Review",
     "desc":"Every prediction routes to a review queue with audit logging. A primatologist "
            "confirms, corrects, or rejects AI output before it becomes a final label. "
            "No unverified automation."},
    {"icon":"file_export","title":"Export-ready Research Workflow",
     "desc":"Reviewed detections export to a structured CSV schema with model version, confidence, "
            "reviewer, and audit metadata, ready for downstream statistical analysis and publication."},
]

ARCHITECTURE_PIPELINE = [
    {"step":"Upload batch","status":"implemented"},
    {"step":"SpeciesNet / MegaDetector inference","status":"implemented"},
    {"step":"SQLite persistence + review queue","status":"implemented"},
    {"step":"Human review + audit log","status":"implemented"},
    {"step":"CSV export","status":"implemented"},
    {"step":"Behavior detection","status":"planned"},
    {"step":"Individual ID (re-identification)","status":"planned"},
    {"step":"Validation metrics","status":"planned"},
    {"step":"Custom primate model","status":"planned"},
]

REAL_VS_SIMULATED = {
    "real": [
        "Local Streamlit app (offline)",
        "SpeciesNet/MegaDetector real inference (when installed)",
        "Image & short-video upload processing",
        "SQLite persistence (media, detections, predictions, reviews)",
        "Human-in-the-loop review with audit log",
        "CSV export of reviewed data",
        "Dashboard UI & navigation",
    ],
    "simulated": [
        "Behavior intelligence (grooming/chasing/feeding) — requires behavior model",
        "Individual primate ID (M03/F07) — requires re-identification model",
        "Field station network map — requires station metadata",
        "Validation metrics (mAP, F1, recall) — requires ground truth",
        "Custom primate detector — future research module",
    ],
}

REVIEW_QUEUE = pd.DataFrame({
    "Image/Clip ID":      ["CLP-8842-A","CLP-8843-B","CLP-8844-C","CLP-8845-D","CLP-8846-E","CLP-8847-F","CLP-8848-G","CLP-8849-H"],
    "Station":            ["CS-1","CS-4","CS-2","CS-6","CS-3","CS-5","CS-1","CS-6"],
    "Timestamp":          ["2026-04-08 07:42","2026-04-08 06:15","2026-04-08 05:30","2026-04-08 04:22","2026-04-07 22:10","2026-04-07 19:45","2026-04-07 14:05","2026-04-07 11:18"],
    "Model Output":       ["Primate (Grooming)","Primate (Feeding)","Person","Primate (Resting)","Primate + Vehicle","Primate (Dense)","Empty / No detection","Primate (Traveling)"],
    "Confidence":         [0.94, 0.62, 0.88, 0.91, 0.71, 0.58, 0.12, 0.83],
    "Queue Reason":       ["Random QA sample","Borderline confidence","Person/vehicle detected","Cross-model disagreement","Person/vehicle detected","Dense scene","Missing metadata","Random QA sample"],
    "Review Status":      ["Pending","Pending","Flagged","Pending","Flagged","Pending","Rejected","Approved"],
})

CSV_EXPORT_PREVIEW = pd.DataFrame({
    "project_id":        ["PSA-2026-Q2","PSA-2026-Q2","PSA-2026-Q2","PSA-2026-Q2"],
    "image_id":          ["CLP-8842-A","CLP-8843-B","CLP-8844-C","CLP-8845-D"],
    "file_name":         ["CS1_20260408_0742.jpg","CS4_20260408_0615.jpg","CS2_20260408_0530.jpg","CS6_20260408_0422.jpg"],
    "captured_at":       ["2026-04-08T07:42:11Z","2026-04-08T06:15:03Z","2026-04-08T05:30:47Z","2026-04-08T04:22:19Z"],
    "station_id":        ["CS-1","CS-4","CS-2","CS-6"],
    "md_label":          ["animal","animal","person","animal"],
    "md_confidence":     [0.91, 0.78, 0.95, 0.88],
    "yolo_label":        ["Primate","Primate","person","Primate"],
    "yolo_confidence":   [0.94, 0.62, 0.88, 0.91],
    "review_status":     ["approved","pending","flagged","pending"],
    "final_label":       ["Grooming", None, "person (non-target)", None],
    "reviewer":          ["A. Mwangi", None, "A. Mwangi", None],
    "notes":             ["Confirmed pair grooming M03+F07", None, "Field staff near station", None],
    "model_version":     ["yolov8n-v0.2","yolov8n-v0.2","yolov8n-v0.2","yolov8n-v0.2"],
})


# =============================================================================
# CSS — OBSIDIAN CANOPY DESIGN SYSTEM
# =============================================================================
FONT_LINKS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600&family=Geist:wght@300;400;500&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet">
"""

CSS = """
<style>
/* === RESET & BASE === */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background-color: #0A0F0D !important;
    color: #FFFFFF;
    font-family: 'Geist', system-ui, sans-serif;
    font-size: 14px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* === MATERIAL SYMBOLS === */
.material-symbols-outlined {
    font-family: 'Material Symbols Outlined';
    font-weight: normal;
    font-style: normal;
    display: inline-block;
    line-height: 1;
    text-transform: none;
    letter-spacing: normal;
    word-wrap: normal;
    white-space: nowrap;
    direction: ltr;
    -webkit-font-feature-settings: 'liga';
    -webkit-font-smoothing: antialiased;
    vertical-align: middle;
}

/* === SCROLLBAR === */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0A0F0D; }
::-webkit-scrollbar-thumb { background: #1E292B; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #0adec8; }

/* === SIDEBAR === */
section[data-testid="stSidebar"] {
    background-color: #141C19 !important;
    border-right: 1px solid #1E292B !important;
    width: 240px !important;
    min-width: 240px !important;
    padding: 0 !important;
}
section[data-testid="stSidebar"] > div {
    padding: 16px 12px !important;
}

/* === TYPOGRAPHY === */
h1, h2, h3, h4, h5 {
    font-family: 'Source Serif 4', Georgia, serif !important;
    font-weight: 600 !important;
    color: #FFFFFF !important;
    line-height: 1.25;
}
p, span, label, div {
    font-family: 'Geist', system-ui, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* === BUTTONS === */
button[kind="primary"], .st-key-main button {
    background-color: #0adec8 !important;
    color: #00201c !important;
    font-family: 'Geist', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 8px 16px !important;
    letter-spacing: 0.02em;
    transition: all 0.15s ease;
}
button[kind="primary"]:hover, .st-key-main button:hover {
    background-color: #0adec8 !important;
    box-shadow: 0 0 12px rgba(10,222,200,0.35) !important;
}

div[data-testid="stHorizontalBlock"] button {
    background-color: transparent !important;
    color: #94A3B8 !important;
    border: 1px solid #94A3B8 !important;
    border-radius: 4px !important;
    font-family: 'Geist', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 7px 14px !important;
    transition: all 0.15s ease;
}
div[data-testid="stHorizontalBlock"] button:hover {
    border-color: #0adec8 !important;
    color: #0adec8 !important;
    box-shadow: 0 0 8px rgba(10,222,200,0.2) !important;
}

/* === CARDS === */
.ps-card {
    background-color: #141C19;
    border: 1px solid #1E292B;
    border-radius: 4px;
    padding: 16px;
    margin-bottom: 8px;
}
.ps-card-hi {
    background-color: #1C2623;
    border: 1px solid #1E292B;
    border-radius: 4px;
    padding: 16px;
    margin-bottom: 8px;
}
.ps-media {
    background-color: #141C19;
    border: 1px solid #1E292B;
    border-radius: 8px;
    overflow: hidden;
    position: relative;
}
.ps-media::after {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at center, transparent 50%, rgba(10,15,13,0.55) 100%);
    pointer-events: none;
}

/* === METRICS === */
div[data-testid="stMetric"] {
    background-color: #141C19 !important;
    border: 1px solid #1E292B !important;
    border-left: 3px solid #0adec8 !important;
    border-radius: 4px !important;
    padding: 12px 14px !important;
}
div[data-testid="stMetricLabel"] {
    font-family: 'Geist', sans-serif !important;
    font-size: 11px !important;
    color: #94A3B8 !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 20px !important;
    font-weight: 600 !important;
    color: #0adec8 !important;
    letter-spacing: 0.05em;
}
div[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
}

/* === PROGRESS BARS === */
div[data-testid="stProgressBar"] > div > div {
    background-color: #1C2623 !important;
    border-radius: 2px !important;
    height: 4px !important;
}
div[data-testid="stProgressBar"] .st-ag {
    background-color: #0adec8 !important;
    border-radius: 2px !important;
}

/* === TABS === */
.st-cc button {
    font-family: 'Source Serif 4', Georgia, serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #94A3B8 !important;
    border-bottom: 2px solid transparent !important;
    padding: 8px 16px !important;
    background: transparent !important;
}
.st-cc button[aria-selected="true"] {
    color: #FFFFFF !important;
    border-bottom: 2px solid #0adec8 !important;
}
.st-cc .tab-content { display: none; }

/* === DIVIDER === */
hr { border: none; border-top: 1px solid #1E292B !important; margin: 16px 0; }

/* === FILE UPLOADER === */
[data-testid="stFileUploader"] > div > div {
    background-color: #050807 !important;
    border: 1px dashed #94A3B8 !important;
    border-radius: 4px !important;
    padding: 16px !important;
}

/* === SELECTBOX / DROPDOWN === */
[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stSelectbox"] [data-baseweb="tag"] {
    background-color: #050807 !important;
    border: 1px solid #94A3B8 !important;
    border-radius: 4px !important;
    color: #FFFFFF !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"]:hover,
[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within {
    border-color: #0adec8 !important;
    box-shadow: 0 0 0 1px rgba(10,222,200,0.3) !important;
}
[data-testid="stSelectbox"] [data-baseweb="menu"] {
    background-color: #141C19 !important;
    border: 1px solid #1E292B !important;
    border-radius: 4px !important;
}

/* === TEXT / TEXT_INPUT === */
input[type="text"], input[type="number"], textarea {
    background-color: #050807 !important;
    border: 1px solid #94A3B8 !important;
    border-radius: 4px !important;
    color: #FFFFFF !important;
    font-family: 'Geist', sans-serif !important;
}
input:focus { border-color: #0adec8 !important; box-shadow: 0 0 0 1px rgba(10,222,200,0.25) !important; }

/* === DATA FRAME === */
[data-testid="stDataFrame"] .dataframe {
    background-color: #141C19 !important;
    border: 1px solid #1E292B !important;
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    color: #FFFFFF !important;
}
[data-testid="stDataFrame"] .dataframe thead th {
    background-color: #1C2623 !important;
    color: #94A3B8 !important;
    font-family: 'Geist', sans-serif !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border-bottom: 1px solid #1E292B !important;
    padding: 8px 12px !important;
}
[data-testid="stDataFrame"] .dataframe tbody tr {
    border-bottom: 1px solid #1E292B !important;
}
[data-testid="stDataFrame"] .dataframe tbody tr:hover {
    background-color: #1C2623 !important;
}
[data-testid="stDataFrame"] .dataframe tbody td {
    padding: 8px 12px !important;
    color: #FFFFFF !important;
}

/* === EXPANDER === */
details {
    background-color: #141C19 !important;
    border: 1px solid #1E292B !important;
    border-radius: 4px !important;
}

/* === CUSTOM COMPONENTS === */
.ps-brand {
    font-family: 'Source Serif 4', Georgia, serif !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #0adec8 !important;
    letter-spacing: 0.01em;
    line-height: 1.2;
    margin-bottom: 4px;
}
.ps-version {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    font-weight: 500 !important;
    color: #94A3B8 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0;
}
.ps-section-title {
    font-family: 'Source Serif 4', Georgia, serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #FFFFFF !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.ps-mono {
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.05em;
}
.ps-data {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    color: #0adec8 !important;
    letter-spacing: 0.05em;
}
.ps-label {
    font-family: 'Geist', sans-serif !important;
    font-size: 11px !important;
    color: #94A3B8 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.ps-text {
    font-family: 'Geist', sans-serif !important;
    font-size: 13px !important;
    color: #94A3B8 !important;
    line-height: 1.6;
}
.ps-text-white {
    font-family: 'Geist', sans-serif !important;
    font-size: 13px !important;
    color: #FFFFFF !important;
    line-height: 1.6;
}
.ps-insight-card {
    background-color: #141C19;
    border: 1px solid #1E292B;
    border-left: 4px solid #FFB84D;
    border-radius: 4px;
    padding: 14px 16px;
    margin: 8px 0;
}
.ps-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 2px 8px;
    border-radius: 2px;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.ps-chip-detected { background-color: rgba(10,222,200,0.08); border: 1px solid #0adec8; color: #0adec8; }
.ps-chip-review   { background-color: rgba(255,184,77,0.08); border: 1px solid #FFB84D; color: #FFB84D; }
.ps-chip-archived { background-color: rgba(148,163,184,0.08); border: 1px solid #94A3B8; color: #94A3B8; }
.ps-chip-ready    { background-color: rgba(10,222,200,0.08); border: 1px solid #0adec8; color: #0adec8; }
.ps-chip-draft    { background-color: rgba(148,163,184,0.08); border: 1px solid #94A3B8; color: #94A3B8; }
.ps-chip-conflict { background-color: rgba(255,180,171,0.08); border: 1px solid #FFB4AB; color: #FFB4AB; }
.ps-status-dot {
    width: 6px; height: 6px; border-radius: 1px; display: inline-block;
}
.ps-banner {
    background-color: rgba(10,222,200,0.07);
    border: 1px solid rgba(10,222,200,0.3);
    border-radius: 4px;
    padding: 8px 14px;
    color: #0adec8;
    font-family: 'Geist', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500;
    margin: 4px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.ps-alert-box {
    background-color: rgba(255,184,77,0.07);
    border: 1px solid rgba(255,184,77,0.4);
    border-radius: 4px;
    padding: 10px 14px;
    color: #FFB84D;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.06em;
    margin-top: 8px;
}
.ps-error-box {
    background-color: rgba(255,180,171,0.07);
    border: 1px solid rgba(255,180,171,0.4);
    border-radius: 4px;
    padding: 10px 14px;
    color: #FFB4AB;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.06em;
    margin-top: 8px;
}
.ps-table-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid #1E292B;
    font-family: 'Geist', sans-serif !important;
    font-size: 13px !important;
}
.ps-table-row:last-child { border-bottom: none; }
.ps-table-key { color: #94A3B8 !important; }
.ps-table-val { color: #FFFFFF !important; font-weight: 500; }
.ps-row-mono { font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; }

/* === HERO / LANDING === */
.ps-hero {
    position: relative;
    min-height: 420px;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #1E292B;
    background: linear-gradient(135deg, #141C19 0%, #0A0F0D 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 48px;
}
.ps-hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 30%, rgba(10,222,200,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.ps-hero-title {
    font-family: 'Source Serif 4', Georgia, serif !important;
    font-size: 42px !important;
    font-weight: 600 !important;
    color: #FFFFFF !important;
    line-height: 1.1;
    letter-spacing: -0.02em;
    text-align: center;
    margin-bottom: 16px;
}
.ps-hero-sub {
    font-family: 'Geist', sans-serif !important;
    font-size: 16px !important;
    color: #94A3B8 !important;
    text-align: center;
    max-width: 600px;
    line-height: 1.6;
    margin: 0 auto 28px auto;
}
.ps-hero-badge {
    display: inline-block;
    background: rgba(10,222,200,0.08);
    border: 1px solid rgba(10,222,200,0.3);
    border-radius: 4px;
    padding: 4px 12px;
    color: #0adec8;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 20px;
}

/* === STAT CARDS (Dashboard) === */
.ps-stat-card {
    background-color: #141C19;
    border: 1px solid #1E292B;
    border-radius: 8px;
    padding: 24px;
    position: relative;
    overflow: hidden;
}
.ps-stat-label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    color: #94A3B8 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 12px;
}
.ps-stat-value {
    font-family: 'Source Serif 4', Georgia, serif !important;
    font-size: 36px !important;
    font-weight: 600 !important;
    color: #FFFFFF !important;
    line-height: 1;
}
.ps-stat-trend {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    margin-top: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* === VALUE PROP CARDS === */
.ps-prop-card {
    background-color: #141C19;
    border: 1px solid #1E292B;
    border-radius: 4px;
    padding: 32px;
    transition: border-color 0.3s ease;
}
.ps-prop-icon {
    color: #0adec8;
    font-size: 36px !important;
    margin-bottom: 20px;
    display: block;
}
.ps-prop-title {
    font-family: 'Source Serif 4', Georgia, serif !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #FFFFFF !important;
    margin-bottom: 12px;
}
.ps-prop-desc {
    font-family: 'Geist', sans-serif !important;
    font-size: 14px !important;
    color: #94A3B8 !important;
    line-height: 1.6;
}

/* === GANTT TRACK === */
.ps-gantt-track {
    display: flex;
    align-items: center;
    margin-bottom: 12px;
}
.ps-gantt-label {
    width: 120px;
    flex-shrink: 0;
    font-family: 'Geist', sans-serif !important;
    font-size: 13px !important;
    color: #FFFFFF !important;
    display: flex;
    align-items: center;
    gap: 6px;
}
.ps-gantt-bar-container {
    flex: 1;
    height: 32px;
    background-color: #0A0F0D;
    border: 1px solid #1E292B;
    border-radius: 4px;
    position: relative;
}
.ps-gantt-segment {
    position: absolute;
    top: 0;
    bottom: 0;
    border-radius: 2px;
    display: flex;
    align-items: center;
    padding: 0 8px;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    overflow: hidden;
    white-space: nowrap;
}

/* === EVENT LOG === */
.ps-event-row {
    display: flex;
    gap: 12px;
    padding: 8px;
    border-radius: 4px;
}
.ps-event-row-active {
    background-color: rgba(10,222,200,0.05);
    border-left: 2px solid #0adec8;
}
.ps-event-time {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    color: #0adec8 !important;
    flex-shrink: 0;
    padding-top: 2px;
}
.ps-event-title {
    font-family: 'Geist', sans-serif !important;
    font-size: 13px !important;
    color: #FFFFFF !important;
}
.ps-event-detail {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    color: #94A3B8 !important;
}

/* === BATTERY BAR === */
.ps-battery-bar {
    width: 100%;
    height: 4px;
    background-color: #1C2623;
    border-radius: 2px;
    overflow: hidden;
    margin: 8px 0;
}
.ps-battery-fill {
    height: 100%;
    border-radius: 2px;
}

/* === CONFIDENCE BAR === */
.ps-conf-bar {
    width: 100%;
    height: 2px;
    background-color: #273647;
    border-radius: 1px;
    overflow: hidden;
}
.ps-conf-fill {
    height: 100%;
    background-color: #0adec8;
}

/* === ENTITY ROW === */
.ps-entity-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: #0A0F0D;
    padding: 8px;
    border: 1px solid #1E292B;
    border-radius: 4px;
    margin-bottom: 8px;
}
.ps-entity-name {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    color: #FFFFFF !important;
}
.ps-entity-conf {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    padding: 2px 8px;
    border-radius: 2px;
}

/* === ALERT ITEM === */
.ps-alert-item {
    border-left: 2px solid;
    padding: 8px 12px;
    margin-bottom: 12px;
}
.ps-alert-type {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    padding: 2px 6px;
    border: 1px solid;
    border-radius: 2px;
    display: inline-block;
    margin-bottom: 4px;
}
.ps-alert-title {
    font-family: 'Geist', sans-serif !important;
    font-size: 13px !important;
    color: #FFFFFF !important;
    font-weight: 500;
    margin: 4px 0;
}
.ps-alert-detail {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    color: #94A3B8 !important;
}

/* === TROOP MEMBER === */
.ps-troop-member {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px;
    border: 1px solid transparent;
    border-radius: 4px;
    margin-bottom: 8px;
    transition: all 0.15s ease;
}
.ps-troop-member:hover {
    background-color: #1C2623;
    border-color: #1E292B;
}
.ps-troop-id {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    color: #FFFFFF !important;
}
.ps-troop-role {
    font-family: 'Geist', sans-serif !important;
    font-size: 11px !important;
    color: #94A3B8 !important;
}

/* === EXPORT FORMAT CARD === */
.ps-export-card {
    background-color: #0A0F0D;
    border: 1px solid #1E292B;
    border-radius: 4px;
    padding: 16px;
    cursor: pointer;
    transition: all 0.15s ease;
}
.ps-export-card:hover {
    border-color: #0adec8;
}
.ps-export-card-selected {
    border: 2px solid #0adec8;
    background-color: #1C2623;
}
.ps-export-name {
    font-family: 'Geist', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #FFFFFF !important;
    margin-bottom: 4px;
}
.ps-export-desc {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
    color: #94A3B8 !important;
}

/* === GLOW on active/focus === */
:focus-visible { outline: 1px solid #0adec8 !important; outline-offset: 2px !important; }

/* === STMAP === */
.stMap { border-radius: 8px !important; }

/* === LINE / BAR CHART containers === */
.js-plotly-plot, .plotly { border-radius: 8px !important; overflow: hidden; }
</style>
"""


# =============================================================================
# HELPERS
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_yolo():
    if YOLO is None:
        return None
    try:
        return YOLO("yolov8n.pt")
    except Exception:
        return None


def _vignette_overlay(fig, ax, pad=0):
    ax_im = ax.imshow(np.zeros((100,100,4), dtype=np.uint8),
                      extent=[0,1,0,1], origin="upper", zorder=10,
                      alpha=0, aspect="auto")
    x = np.linspace(0,1,100)
    y = np.linspace(0,1,100)
    X, Y = np.meshgrid(x,y)
    V = 1 - 0.55 * ((X-0.5)**2 + (Y-0.5)**2) / 0.5**2
    V = np.clip(V, 0, 1)
    vignette = np.zeros((100,100,4), dtype=np.uint8)
    vignette[:,:,3] = ((1-V)*110).astype(np.uint8)
    ax_im.set_array(vignette)
    return ax_im


def render_camera_view(boxes):
    fig, ax = plt.subplots(figsize=(10, 5.625), dpi=120)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    grad = np.linspace(0, 1, 540).reshape(-1, 1)
    grad = np.repeat(grad, 960, axis=1)
    bg = np.zeros((540, 960, 3), dtype=np.uint8)
    bg[:,:,0] = (grad * 18 + 12).astype(np.uint8)
    bg[:,:,1] = (grad * 24 + 16).astype(np.uint8)
    bg[:,:,2] = (grad * 16 + 10).astype(np.uint8)
    yy, xx = np.ogrid[:540, :960]
    vig = 1 - 0.4 * ((xx-480)**2 + (yy-270)**2) / (480**2+270**2)
    vig = np.clip(vig, 0.5, 1.0)
    bg = (bg.astype(np.float32) * vig[..., None]).astype(np.uint8)
    ax.imshow(bg, extent=[0,960,540,0], origin="upper", aspect="auto")
    for b in boxes:
        x, y = b["xy"]
        w, h = b["size"]
        fill_alpha = 0.07
        hc = b["color"].lstrip('#')
        fc = tuple(int(hc[i:i+2], 16) for i in (0, 2, 4))
        fill_color = tuple(c / 255.0 for c in fc) + (fill_alpha,)
        rect = Rectangle((x,y), w, h, linewidth=1.2, edgecolor=b["color"],
                         facecolor=fill_color, zorder=3)
        ax.add_patch(rect)
        ax.text(x+6, y-10, b["label"], color=b["color"], fontsize=9,
                fontweight="bold", zorder=4, family="monospace",
                bbox=dict(facecolor=BG, edgecolor=b["color"], lw=0.8, pad=3, alpha=0.9))
    ax.set_xlim(0, 960)
    ax.set_ylim(540, 0)
    ax.axis("off")
    ax.text(14, 20,
            "CAM-03  \u2022  ZONE B  \u2022  2026-04-08  07:42:11",
            color=SLATE, fontsize=8, family="monospace", alpha=0.85)
    ax.text(946, 20, "\u25CF  REC", color=ERROR_CLR, fontsize=8,
            family="monospace", ha="right", alpha=0.85, fontweight="bold")
    _vignette_overlay(fig, ax)
    plt.tight_layout(pad=0)
    return fig


def render_social_graph(highlight=None):
    G = nx.DiGraph()
    for mid in MONKEYS:
        G.add_node(mid)
    for mid, info in MONKEYS.items():
        tgt = info["primary_assoc"]
        if tgt in MONKEYS:
            G.add_edge(mid, tgt, weight=info["detections"])
    pos = nx.spring_layout(G, seed=42, k=1.8)
    node_colors = []
    for mid in G.nodes():
        sx = MONKEYS[mid]["sex"]
        if sx == "Male":   node_colors.append(TEAL)
        elif sx == "Female": node_colors.append(AMBER)
        else:              node_colors.append(SLATE)
    sizes = [280 + MONKEYS[n]["detections"] * 10 for n in G.nodes()]
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=110)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=BORDER, width=1.4,
                           arrows=True, arrowsize=10, alpha=0.6,
                           connectionstyle="arc3,rad=0.07")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=sizes, edgecolors=BORDER,
                           linewidths=1.0, alpha=0.95)
    if highlight and highlight in G.nodes():
        idx = list(G.nodes()).index(highlight)
        nx.draw_networkx_nodes(G, pos, nodelist=[highlight], ax=ax,
                               node_color=[node_colors[idx]],
                               node_size=sizes[idx]+350,
                               edgecolors=TEAL, linewidths=2.5, alpha=1.0)
    nx.draw_networkx_labels(G, pos, ax=ax, font_color=WHITE,
                            font_size=9, font_weight="bold", font_family="monospace")
    ax.axis("off")
    plt.tight_layout(pad=0.3)
    return fig


def render_season_bar(data, title, color):
    fig, ax = plt.subplots(figsize=(5.5, 4.4), dpi=110)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    keys = list(data.keys())
    vals = list(data.values())
    bars = ax.bar(keys, vals, color=color, edgecolor=BORDER, lw=0.8, alpha=0.9, width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.8, str(v),
                ha="center", va="bottom", color=WHITE, fontsize=9,
                fontweight="bold", family="monospace")
    ax.set_title(title, color=WHITE, fontsize=11, pad=12,
                 fontfamily="Source Serif 4, Georgia, serif", fontweight="600")
    ax.tick_params(axis="x", labelrotation=28, colors=SLATE, labelsize=8.5)
    ax.tick_params(axis="y", colors=SLATE, labelsize=8)
    for sp in ["top","right","left","bottom"]:
        ax.spines[sp].set_color(BORDER)
    ax.set_ylim(0, max(vals)*1.22)
    ax.yaxis.set_tick_params(labelcolor=SLATE)
    ax.xaxis.set_tick_params(labelcolor=SLATE)
    plt.tight_layout(pad=0.5)
    return fig


def render_timeline_chart():
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=110)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    days = list(ACTIVITY_DATA.index)
    x = range(len(days))
    styles = {
        "Feeding":        {"color":TEAL,"ls":"-","lw":1.8,"mk":"o","ms":4},
        "Grooming":       {"color":AMBER,"ls":"-","lw":1.5,"mk":"s","ms":4},
        "Resting":        {"color":SLATE,"ls":"--","lw":1.2,"mk":"^","ms":3},
        "Chasing":        {"color":ERROR_CLR,"ls":"-","lw":2.0,"mk":"D","ms":4},
        "Group_Movement": {"color":WHITE,"ls":":","lw":1.2,"mk":"v","ms":3},
    }
    for col, sty in styles.items():
        ax.plot(x, ACTIVITY_DATA[col], label=col.replace("_"," "),
                color=sty["color"], linestyle=sty["ls"], lw=sty["lw"],
                marker=sty["mk"], markersize=sty["ms"], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(days, color=SLATE, fontsize=8, rotation=30, ha="right")
    ax.tick_params(axis="y", colors=SLATE, labelsize=8)
    for sp in ["top","right","left","bottom"]:
        ax.spines[sp].set_color(BORDER)
    ax.yaxis.set_tick_params(labelcolor=SLATE)
    ax.xaxis.set_tick_params(labelcolor=SLATE)
    ax.legend(framealpha=0.15, facecolor=BG, edgecolor=BORDER,
              labelcolor=WHITE, fontsize=8, loc="upper left",
              ncol=3, columnspacing=0.8, handlelength=1.4)
    plt.tight_layout(pad=0.3)
    return fig


def render_gantt_chart():
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=110)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    track_labels = [t["label"] for t in GANTT_TRACKS]
    y_positions = list(range(len(GANTT_TRACKS) - 1, -1, -1))
    for i, track in enumerate(GANTT_TRACKS):
        y = y_positions[i]
        for start, end, label in track["segments"]:
            width = end - start
            fc = track["color"].lstrip('#')
            rgb = tuple(int(fc[j:j+2], 16) / 255.0 for j in (0, 2, 4))
            fill = rgb + (0.15,)
            edge = rgb + (0.6,)
            ax.barh(y, width, left=start, height=0.5,
                    color=fill, edgecolor=edge, linewidth=1.0)
            if label:
                ax.text(start + width / 2, y, label,
                        ha="center", va="center", color=track["color"],
                        fontsize=8, fontweight="bold", family="monospace")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(track_labels, color=WHITE, fontsize=10, fontfamily="monospace")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 4))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 4)],
                       color=SLATE, fontsize=8, family="monospace")
    for sp in ["top","right","left"]:
        ax.spines[sp].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)
    ax.xaxis.grid(True, color=BORDER, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title("24H Activity Timeline - Subject PAN-T-042",
                 color=WHITE, fontsize=12, pad=12,
                 fontfamily="Source Serif 4, Georgia, serif", fontweight="600")
    plt.tight_layout(pad=0.5)
    return fig


def render_movement_map():
    fig, ax = plt.subplots(figsize=(8, 6), dpi=110)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    xs = [25.045, 25.048, 25.042, 25.050, 25.038, 25.052]
    ys = [102.710, 102.715, 102.708, 102.720, 102.705, 102.722]
    colors_map = [TEAL, TEAL, TEAL, ERROR_CLR, TEAL, TEAL]
    sizes_map = [120, 100, 90, 160, 110, 180]
    ax.scatter(ys, xs, c=colors_map, s=sizes_map, zorder=5,
               edgecolors=WHITE, linewidths=0.8, alpha=0.9)
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            if (i + j) % 2 == 0:
                ax.plot([ys[i], ys[j]], [xs[i], xs[j]],
                        color=TEAL, linestyle="--", linewidth=0.8, alpha=0.25, zorder=1)
    for i, name in enumerate(["ALPHA","BETA","DELTA","GAMMA","EPSILON","ZETA"]):
        clr = ERROR_CLR if name == "GAMMA" else TEAL
        ax.annotate(f"STN-{name}", (ys[i], xs[i]),
                    textcoords="offset points", xytext=(8, 8),
                    color=clr, fontsize=8, family="monospace", fontweight="bold")
    ax.set_xlabel("Longitude", color=SLATE, fontsize=9)
    ax.set_ylabel("Latitude", color=SLATE, fontsize=9)
    ax.tick_params(colors=SLATE, labelsize=8)
    for sp in ["top","right","left","bottom"]:
        ax.spines[sp].set_color(BORDER)
    ax.set_title("Station Movement Map", color=WHITE, fontsize=12, pad=12,
                 fontfamily="Source Serif 4, Georgia, serif", fontweight="600")
    plt.tight_layout(pad=0.5)
    return fig


def _kv(k, v):
    return (f'<div class="ps-table-row">'
            f'<span class="ps-table-key ps-label">{k}</span>'
            f'<span class="ps-table-val">{v}</span>'
            f'</div>')


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="ps-brand">PrimateScope AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="ps-version">Field Intelligence System  v1.0</div>', unsafe_allow_html=True)
        st.markdown('<hr style="margin:12px 0;">', unsafe_allow_html=True)
        page = st.radio("Navigation", NAV_PAGES, label_visibility="collapsed")
        st.markdown('<hr style="margin:12px 0;">', unsafe_allow_html=True)

        yolo_status = "active" if load_yolo() else "standby"
        yolo_color = TEAL if load_yolo() else AMBER

        st.markdown(
            f'<div class="ps-label">System Status</div>'
            f'<div style="margin-top:6px;">'
            f'<span class="ps-chip ps-chip-detected" style="font-size:9px;">'
            f'<span class="ps-status-dot" style="background:{yolo_color};width:5px;height:5px;"></span>'
            f'YOLOv8 &nbsp; {yolo_status.upper()}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="margin-top:8px;" class="ps-label">'
            f'Individuals &nbsp; {len(MONKEYS)} &nbsp;|&nbsp; '
            f'Stations &nbsp; {len(FIELD_STATIONS)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<hr style="margin:12px 0;">', unsafe_allow_html=True)
        st.markdown(
            '<div class="ps-banner" style="font-size:10px;margin:0;">'
            'REAL INFERENCE — AI-ASSISTED, REVIEW REQUIRED'
            '</div>',
            unsafe_allow_html=True,
        )
        
        # Default to Real Analysis mode globally now
        st.session_state["ps_mode"] = "Real Inference"
        
        return page


# =============================================================================
# PAGE 0 — OVERVIEW / LANDING
# =============================================================================
def page_overview():
    st.markdown(
        '<div class="ps-hero">'
        '<div style="position:relative;z-index:1;text-align:center;width:100%;">'
        '<div class="ps-hero-badge">PRODUCTION v1.0 | OFFLINE | SPECIESNET-READY</div>'
        '<h1 class="ps-hero-title">PrimateScope AI</h1>'
        '<p class="ps-hero-sub" style="font-size:20px;color:#FFFFFF;font-weight:500;'
        'max-width:680px;margin-bottom:8px;">'
        'Months of camera-trap footage. Minutes to find the behavior that matters.'
        '</p>'
        '<p class="ps-hero-sub">'
        'An offline prototype for AI-assisted primate footage review, '
        'built for field validation.'
        '</p>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:32px;"></div>', unsafe_allow_html=True)

    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:24px;">'
        + "".join(
            f'<div class="ps-prop-card">'
            f'<span class="material-symbols-outlined ps-prop-icon" style="font-variation-settings:FILL 1;">{p["icon"]}</span>'
            f'<div class="ps-prop-title">{p["title"]}</div>'
            f'<div class="ps-prop-desc">{p["desc"]}</div>'
            f'</div>'
            for p in VALUE_PROPS
        )
        + '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)

    arch_c1, arch_c2 = st.columns([1, 1], gap="large")

    with arch_c1:
        st.markdown('<div class="ps-section-title">Target Pipeline</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="ps-card" style="padding:16px;">'
            '<div style="display:flex;flex-wrap:wrap;align-items:center;gap:6px;'
            'font-family:JetBrains Mono,monospace;font-size:11px;">'
            '<span style="color:#FFFFFF;background:#1C2623;padding:6px 10px;border-radius:3px;">Upload batch</span>'
            '<span style="color:#0adec8;">&#8594;</span>'
            '<span style="color:#FFFFFF;background:#1C2623;padding:6px 10px;border-radius:3px;">MegaDetector filter</span>'
            '<span style="color:#0adec8;">&#8594;</span>'
            '<span style="color:#FFFFFF;background:#1C2623;padding:6px 10px;border-radius:3px;">YOLO / primate detector</span>'
            '<span style="color:#0adec8;">&#8594;</span>'
            '<span style="color:#FFFFFF;background:#1C2623;padding:6px 10px;border-radius:3px;">Needs-review queue</span>'
            '<span style="color:#0adec8;">&#8594;</span>'
            '<span style="color:#FFFFFF;background:#1C2623;padding:6px 10px;border-radius:3px;">Human correction</span>'
            '<span style="color:#0adec8;">&#8594;</span>'
            '<span style="color:#FFFFFF;background:#1C2623;padding:6px 10px;border-radius:3px;">CSV / PDF export</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )

    with arch_c2:
        st.markdown('<div class="ps-section-title">Current Demo Coverage</div>', unsafe_allow_html=True)
        coverage_html = '<div class="ps-card" style="padding:16px;">'
        for step in ARCHITECTURE_PIPELINE:
            status = step["status"]
            if status == "implemented":
                clr, lbl = TEAL, "IMPLEMENTED"
            elif status == "partial":
                clr, lbl = AMBER, "PARTIAL"
            elif status == "simulated":
                clr, lbl = AMBER, "SIMULATED"
            else:
                clr, lbl = SLATE, "PLANNED"
            coverage_html += (
                f'<div class="ps-table-row">'
                f'<span class="ps-text-white" style="font-size:12px;">{step["step"]}</span>'
                f'<span class="ps-chip" style="color:{clr};border:1px solid {clr};'
                f'background:{clr}11;">{lbl}</span>'
                f'</div>'
            )
        coverage_html += '</div>'
        st.markdown(coverage_html, unsafe_allow_html=True)

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)

    real_c1, real_c2 = st.columns(2, gap="large")

    with real_c1:
        st.markdown(
            '<div class="ps-section-title" style="color:#0adec8;">'
            '<span class="material-symbols-outlined" style="font-size:16px;vertical-align:middle;">check_circle</span>'
            ' &nbsp;Real in v0.2</div>',
            unsafe_allow_html=True,
        )
        for item in REAL_VS_SIMULATED["real"]:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;padding:5px 0;'
                f'border-bottom:1px solid #1E292B22;">'
                f'<span style="color:#0adec8;font-size:14px;">&#10003;</span>'
                f'<span class="ps-text-white" style="font-size:13px;">{item}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with real_c2:
        st.markdown(
            '<div class="ps-section-title" style="color:#FFB84D;">'
            '<span class="material-symbols-outlined" style="font-size:16px;vertical-align:middle;">schedule</span>'
            ' &nbsp;Simulated / Next Step</div>',
            unsafe_allow_html=True,
        )
        for item in REAL_VS_SIMULATED["simulated"]:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;padding:5px 0;'
                f'border-bottom:1px solid #1E292B22;">'
                f'<span style="color:#FFB84D;font-size:14px;">&#9711;</span>'
                f'<span class="ps-text" style="font-size:13px;">{item}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )


# =============================================================================
# PAGE 1 — RESEARCH DASHBOARD
# =============================================================================
def page_research_dashboard():
    st.title("Global Research Intelligence")
    mode = st.session_state.get("ps_mode", "Demo Simulation")
    if mode == "Real Inference" and _PROD_AVAILABLE:
        stats = get_dashboard_stats()
        if stats and stats["media_total"] > 0:
            st.markdown(
                f'<div class="ps-text" style="margin-bottom:24px;">'
                f'Project stats from local database — {stats["media_total"]} media files, '
                f'{stats["detections_animal"]} animal detections, '
                f'{stats["review_pending"]} pending review.'
                f'</div>',
                unsafe_allow_html=True,
            )
            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.metric("Media Files", stats["media_total"])
            with sc2:
                st.metric("Animal Detections", stats["detections_animal"])
            with sc3:
                st.metric("Human Detections", stats["detections_human"])
            with sc4:
                st.metric("Pending Review", stats["review_pending"])
            sc5, sc6, sc7, sc8 = st.columns(4)
            with sc5:
                st.metric("Reviewed", stats["review_done"])
            with sc6:
                st.metric("Blanks", stats["blanks"])
            with sc7:
                st.metric("Avg Confidence", f'{stats["avg_confidence"]:.2f}')
            with sc8:
                st.metric("Vehicle Detections", stats["detections_vehicle"])
            if stats["top_species"]:
                st.markdown('<div class="ps-section-title" style="margin-top:24px;">Top Predicted Taxa</div>',
                            unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(stats["top_species"], columns=["Taxon", "Count"]),
                             use_container_width=True, hide_index=True)
            st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
            st.markdown('<hr>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="ps-text" style="margin-bottom:24px;">'
                'No real data yet. Run inference on the Analysis Workbench page.'
                '</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="ps-text" style="margin-bottom:24px;">'
            'Aggregate metrics across 42 active monitoring stations (simulated).'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:24px;margin-bottom:32px;">'
            + "".join(
                f'<div class="ps-stat-card">'
                f'<div class="ps-stat-label">{s["label"]}</div>'
                f'<div class="ps-stat-value">{s["value"]}</div>'
                f'<div class="ps-stat-trend" style="color:{s["trend_color"]};">'
                f'<span class="material-symbols-outlined" style="font-size:16px;">'
                f'{"trending_up" if s["trend_color"] == "#0adec8" else "warning" if s["trend_color"] == "#FFB4AB" else "trending_flat"}'
                f'</span>'
                f'<span>{s["trend"]}</span>'
                f'</div>'
                f'</div>'
                for s in GLOBAL_STATS
            )
            + '</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ps-section-title">Station Activity Heatmap '
        '<span class="ps-chip ps-chip-archived" style="font-size:9px;margin-left:8px;">SIMULATED</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.map(FIELD_STATIONS[["lat","lon"]], zoom=13, size=80)

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
    c_left, c_right = st.columns([2, 1], gap="large")

    with c_left:
        st.markdown(
            '<div class="ps-section-title">Detection Trends - April 1-12 '
            '<span class="ps-chip ps-chip-archived" style="font-size:9px;margin-left:8px;">SIMULATED</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.pyplot(render_timeline_chart(), width="stretch")

    with c_right:
        st.markdown('<div class="ps-section-title">Recent Alerts</div>', unsafe_allow_html=True)
        for alert in ALERTS:
            st.markdown(
                f'<div class="ps-alert-item" style="border-color:{alert["color"]};">'
                f'<div style="display:flex;justify-content:space-between;align-items:start;">'
                f'<span class="ps-alert-type" style="color:{alert["color"]};border-color:{alert["color"]}33;">'
                f'{alert["type"]}</span>'
                f'<span class="ps-data" style="font-size:10px;color:#94A3B8;">{alert["time"]}</span>'
                f'</div>'
                f'<div class="ps-alert-title">{alert["title"]}</div>'
                f'<div class="ps-alert-detail">{alert["detail"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.button("View All Logs", width="stretch")


# =============================================================================
# PAGE 2 — LIVE AI ANALYSIS
# =============================================================================
def page_live_analysis():
    st.title("Camera-Trap Analysis Workbench")
    st.markdown(
        '<div class="ps-text" style="margin-bottom:8px;font-size:13px;color:#94A3B8;">'
        'Upload field images or short clips, run AI-assisted detection, review predictions, '
        'and export research-ready results.'
        '</div>',
        unsafe_allow_html=True,
    )
    tab_real, tab_demo = st.tabs(["Real Analysis", "Demo Simulation"])

    with tab_real:
        if _PROD_AVAILABLE:
            page_live_analysis_real()
        else:
            st.error("⚠️ Production modules failed to load. Real inference is unavailable.")
            st.markdown(
                '<div class="ps-error-box">'
                '<b>Import Error Details</b><br>'
                'The production backend could not be initialized. '
                'This is NOT a demo fallback — the real engine is broken. '
                'Check the traceback below and install missing dependencies.'
                '</div>',
                unsafe_allow_html=True,
            )
            st.code(f"Error: {_PROD_ERROR}\n\nFull Traceback:\n{_PROD_TRACEBACK}", language="python")
            st.markdown(
                "**Common fixes:**\n"
                "- `pip install speciesnet` (macOS: add `--use-pep517`)\n"
                "- `pip install megadetector`\n"
                "- Ensure `database/`, `services/`, `utils/` directories exist\n"
                "- Run `python scripts/check_environment.py`"
            )

    with tab_demo:
        page_live_analysis_demo()

def page_live_analysis_demo():
    st.markdown(
        '<div class="ps-banner" style="margin-bottom:16px;">'
        '<span class="material-symbols-outlined" style="font-size:16px;">info</span>'
        'Demo mode uses YOLOv8n baseline and sample/synthetic data. '
        'Switch to the Real Analysis tab for SpeciesNet/MegaDetector processing.'
        '</div>',
        unsafe_allow_html=True,
    )
    col_l, col_r = st.columns([1, 2], gap="large")

    with col_l:
        st.markdown('<div class="ps-section-title">Demo Scenarios</div>', unsafe_allow_html=True)
        choice = st.selectbox("Scenario", SCENARIO_OPTIONS, label_visibility="collapsed")

        if choice != "Custom Upload":
            clicked = st.button("Run Analysis", type="primary", width="stretch")
            if clicked:
                key = SCENARIO_KEY_BY_LABEL[choice]
                st.session_state["demo_scenario"] = key
        else:
            if "demo_scenario" in st.session_state:
                del st.session_state["demo_scenario"]

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<div class="ps-section-title">Upload Your Own</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Video file", type=["mp4","mov","avi"],
                                    label_visibility="collapsed")

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<div class="ps-section-title">Environment</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="ps-card" style="padding:12px;">'
            '<div style="display:flex;justify-content:space-between;">'
            '<div><div class="ps-label">Temp</div><div class="ps-text-white">24&deg;C</div></div>'
            '<div><div class="ps-label">Humidity</div><div class="ps-text-white">88%</div></div>'
            '<div><div class="ps-label">Light</div><div class="ps-text-white">0.2 lx</div></div>'
            '</div></div>',
            unsafe_allow_html=True,
        )

    with col_r:
        if "demo_scenario" in st.session_state:
            key  = st.session_state["demo_scenario"]
            sc   = DEMO_SCENARIOS[key]
            st.markdown(
                f'<div class="ps-banner">'
                f'<span>Prototype Output</span>'
                f'<span style="margin-left:auto;font-family:JetBrains Mono,monospace;font-size:11px;">'
                f'{sc["label"]}</span></div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="ps-media">', unsafe_allow_html=True)
            st.pyplot(render_camera_view(sc["boxes"]), width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

            ac, bc = st.columns([2, 1], gap="large")

            with ac:
                st.markdown('<hr>', unsafe_allow_html=True)
                st.markdown('<div class="ps-section-title">Behavior Classification</div>', unsafe_allow_html=True)
                for name, prob in sc["behaviors"]:
                    c1, c2 = st.columns([1, 4], gap="small")
                    with c1:
                        st.markdown(
                            f'<div class="ps-data" style="text-align:right;padding-top:2px;">'
                            f'{prob*100:.0f}%</div>',
                            unsafe_allow_html=True,
                        )
                    with c2:
                        st.progress(prob, text=name)

                st.markdown('<hr>', unsafe_allow_html=True)
                st.markdown('<div class="ps-section-title">Detected Individuals</div>', unsafe_allow_html=True)
                icols = st.columns(len(sc["individuals"]), gap="medium")
                for ci, mid in enumerate(sc["individuals"]):
                    m = MONKEYS[mid]
                    with icols[ci]:
                        st.metric(mid, f"{m['sex']} - {m['age']}", f"Zone {m['zone']}")

                st.markdown('<hr>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="ps-insight-card">'
                    f'<div style="font-family:Source Serif 4,Georgia,serif;'
                    f'font-size:13px;font-weight:600;color:#FFB84D;margin-bottom:6px;">'
                    f'Prototype Observation</div>'
                    f'<div class="ps-text">{sc["insight"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with bc:
                st.markdown('<hr>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="ps-card-hi" style="padding:20px;">'
                    '<div style="font-family:Source Serif 4,Georgia,serif;font-size:16px;'
                    'font-weight:600;color:#FFFFFF;border-bottom:1px solid #1E292B;'
                    'padding-bottom:8px;margin-bottom:16px;">Intelligence Report</div>'
                    f'<div class="ps-label">Overall Confidence</div>'
                    f'<div style="font-family:Source Serif 4,Georgia,serif;font-size:32px;'
                    f'font-weight:600;color:#0adec8;line-height:1;margin:4px 0;">{sc["confidence"]:.1f}%</div>'
                    f'<div class="ps-conf-bar"><div class="ps-conf-fill" style="width:{sc["confidence"]:.1f}%;"></div></div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

                st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
                st.markdown('<div class="ps-label" style="margin-bottom:8px;">Detected Entities</div>', unsafe_allow_html=True)
                for name, conf, active in sc["entities"]:
                    conf_color = TEAL if active else SLATE
                    bg_color = "rgba(10,222,200,0.1)" if active else "transparent"
                    st.markdown(
                        f'<div class="ps-entity-row" style="opacity:{1.0 if active else 0.5};">'
                        f'<span class="ps-entity-name">{name}</span>'
                        f'<span class="ps-entity-conf" style="background:{bg_color};color:{conf_color};'
                        f'border:1px solid {conf_color};">{conf:.2f}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
                st.markdown('<div class="ps-label" style="margin-bottom:8px;">Event Log</div>', unsafe_allow_html=True)
                for idx, (time, title, detail) in enumerate(sc["events"]):
                    active_class = "ps-event-row-active" if "Positive Identification" in title or "Chasing" in title else ""
                    st.markdown(
                        f'<div class="ps-event-row {active_class}">'
                        f'<div class="ps-event-time">{time}</div>'
                        f'<div>'
                        f'<div class="ps-event-title" style="{"color:#0adec8;" if active_class else ""}">{title}</div>'
                        f'<div class="ps-event-detail">{detail}</div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

                st.button("Export Data", width="stretch")

        elif choice == "Custom Upload" and uploaded is not None:
            run_custom_upload(uploaded)
        else:
            st.info(
                "Select a demo scenario and click **Run Analysis**, "
                "or upload your own video file for YOLOv8n baseline processing.",
            )


def run_custom_upload(uploaded):
    model = load_yolo()
    if model is None or cv2 is None:
        st.error("YOLO model unavailable - falling back to simulation mode.", icon="!")
        st.markdown(
            '<div class="ps-banner">Simulation: showing synthetic detection output.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="ps-media">', unsafe_allow_html=True)
        st.pyplot(render_camera_view(DEMO_SCENARIOS["Feeding"]["boxes"]), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    tfile = os.path.join("/tmp", uploaded.name)
    with open(tfile, "wb") as f:
        f.write(uploaded.getbuffer())
    cap = cv2.VideoCapture(tfile)
    if not cap.isOpened():
        st.error("Could not open video file.", icon="!")
        return

    frame_placeholder = st.empty()
    mc1, mc2, mc3 = st.columns(3)
    total = processed = detections = 0
    frame_idx = 0
    while processed < 30:
        ret, frame = cap.read()
        if not ret:
            break
        total += 1
        if frame_idx % 3 != 0:
            frame_idx += 1
            continue
        frame_idx += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            res = model(rgb, verbose=False)
        except Exception:
            res = []
        for r in res:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                if conf < 0.25:
                    continue
                cv2.rectangle(rgb, (x1,y1),(x2,y2),(10,222,200),1)
                cv2.putText(rgb, f"Primate {conf:.2f}", (x1, max(0,y1-7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10,222,200), 1)
                detections += 1
        img = Image.fromarray(rgb)
        frame_placeholder.image(img, channels="RGB", width="stretch")
        processed += 1
        with mc1:
            st.metric("Frames", processed)
        with mc2:
            st.metric("Detections", detections)
        with mc3:
            st.metric("Avg Conf", f"{detections/max(1,processed):.2f}")
    cap.release()
    if processed == 0:
        st.warning("No frames could be read from the video.")


# =============================================================================
# PAGE 2b — REVIEW QUEUE
# =============================================================================
def page_review_queue():
    mode = st.session_state.get("ps_mode", "Demo Simulation")
    if mode == "Real Inference" and _PROD_AVAILABLE:
        page_review_queue_real()
        return
    st.markdown(
        '<div class="ps-banner" style="margin-bottom:16px;">'
        '<span class="material-symbols-outlined" style="font-size:16px;">info</span>'
        'Simulated queue for v0.2. Borderline detections, cross-model disagreements, '
        'and person/vehicle flags route here for human confirmation before export.'
        '</div>',
        unsafe_allow_html=True,
    )

    queue_reasons = [
        "All", "Borderline confidence", "Cross-model disagreement",
        "Person/vehicle detected", "Missing metadata", "Dense scene", "Random QA sample",
    ]
    filter_reason = st.selectbox("Filter by Queue Reason", queue_reasons, index=0)

    filtered = REVIEW_QUEUE
    if filter_reason != "All":
        filtered = REVIEW_QUEUE[REVIEW_QUEUE["Queue Reason"] == filter_reason]

    st.markdown(
        f'<div class="ps-label" style="margin-bottom:8px;">'
        f'{len(filtered)} item(s) in queue</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        filtered,
        width="stretch",
        hide_index=True,
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                "Confidence",
                help="Model confidence score",
                format="%.2f",
                min_value=0.0,
                max_value=1.0,
            ),
        },
    )

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="ps-section-title">Queue Reasons Explained</div>', unsafe_allow_html=True)

    reasons = [
        ("Borderline confidence", "Model confidence between 0.5 and 0.7. Human confirms or rejects.", AMBER),
        ("Cross-model disagreement", "MegaDetector and primate detector disagree on label.", AMBER),
        ("Person/vehicle detected", "Non-target subject. Useful for privacy & access audits.", ERROR_CLR),
        ("Missing metadata", "Station ID or timestamp absent. Needs manual attribution.", SLATE),
        ("Dense scene", "Multiple overlapping detections. Reviewer splits or merges boxes.", AMBER),
        ("Random QA sample", "Random sample for quality assurance, regardless of confidence.", TEAL),
    ]
    rc1, rc2, rc3 = st.columns(3, gap="medium")
    for col, (name, desc, clr) in zip([rc1, rc2, rc3], reasons):
        with col:
            st.markdown(
                f'<div class="ps-card" style="padding:14px;">'
                f'<div class="ps-chip" style="color:{clr};border:1px solid {clr};'
                f'background:{clr}11;margin-bottom:8px;">{name.upper()}</div>'
                f'<div class="ps-text" style="font-size:12px;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ps-error-box" style="font-size:10px;">'
        'Human correction & audit log persistence are planned for the pilot. '
        'In v0.2 the queue is an in-memory simulation.'
        '</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# PAGE 3 — BEHAVIOR INTELLIGENCE
# =============================================================================
def page_behavior_intelligence():
    st.title("Behavior Intelligence")
    st.markdown(
        '<div class="ps-alert-box" style="margin-bottom:16px;">'
        'SIMULATED IN v1.0 — Behavior timelines are simulated. Real behavior recognition '
        '(grooming, chasing, feeding) requires manual annotation or a separate behavior model. '
        'SpeciesNet detects species, not behaviors.'
        '</div>',
        unsafe_allow_html=True,
    )
    tab1, tab2, tab3 = st.tabs(["24H Activity Timeline", "Daily Trends", "Season Comparison"])

    with tab1:
        st.markdown(
            '<div class="ps-section-title" style="margin-bottom:16px;">'
            'Subject Activity Analysis - 24H Cycle'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="ps-data" style="margin-bottom:16px;font-size:11px;">'
            'STATION: ALPHA-09 | TIMEZONE: UTC-5 | SUBJECT_ID: PAN-T-042'
            '</div>',
            unsafe_allow_html=True,
        )
        st.pyplot(render_gantt_chart(), width="stretch")

        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="ps-section-title">Technical Annotations</div>', unsafe_allow_html=True)
        ac1, ac2, ac3 = st.columns(3, gap="medium")
        with ac1:
            ann = GANTT_ANNOTATIONS[0]
            st.markdown(
                f'<div class="ps-card" style="border-left:2px solid {ann["color"]};">'
                f'<div class="ps-data" style="color:{ann["color"]};font-size:10px;">{ann["label"]}</div>'
                f'<div class="ps-text" style="font-size:12px;margin-top:6px;">{ann["text"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with ac2:
            ann = GANTT_ANNOTATIONS[1]
            st.markdown(
                f'<div class="ps-card" style="border-left:2px solid {ann["color"]};">'
                f'<div class="ps-data" style="color:{ann["color"]};font-size:10px;">{ann["label"]}</div>'
                f'<div class="ps-text" style="font-size:12px;margin-top:6px;">{ann["text"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with ac3:
            ann = GANTT_ANNOTATIONS[2]
            st.markdown(
                f'<div class="ps-card" style="border-left:2px solid {ann["color"]};">'
                f'<div class="ps-data" style="color:{ann["color"]};font-size:10px;">{ann["label"]}</div>'
                f'<div class="ps-text" style="font-size:12px;margin-top:6px;">{ann["text"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with tab2:
        st.markdown(
            '<div class="ps-section-title" style="margin-bottom:16px;">'
            'Daily Event Counts - April 1-12, 2026</div>',
            unsafe_allow_html=True,
        )
        st.pyplot(render_timeline_chart(), width="stretch")
        st.markdown(
            '<div class="ps-card" style="margin-top:12px;padding:12px 14px;">'
            '<div style="font-family:Source Serif 4,Georgia,serif;font-size:12px;'
            'font-weight:600;color:#FFB84D;margin-bottom:4px;">Spike Detected</div>'
            '<div class="ps-text">Chasing events peaked Apr 6-8 (5->8->7), '
            'aligning with mating-season onset. Resting declined inversely as '
            'group movement rose.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with tab3:
        st.markdown(
            '<div class="ps-section-title" style="margin-bottom:16px;">'
            'Normal vs Mating vs Dry Season</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            st.pyplot(render_season_bar(NORMAL_SEASON, "Normal Season (Mar)", TEAL), width="stretch")
        with c2:
            st.pyplot(render_season_bar(MATING_SEASON, "Mating Season (Apr 3-12)", AMBER), width="stretch")
        with c3:
            st.pyplot(render_season_bar(DRY_SEASON, "Dry Season (Feb)", SLATE), width="stretch")
        st.markdown(
            '<div class="ps-alert-box">'
            'Chasing +300% (Mating) | Feeding +16% (Dry) | Grooming -36% (Dry)'
            '</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# PAGE 4 — FIELD STATIONS
# =============================================================================
def page_field_stations():
    st.title("Field Station Network")
    st.markdown(
        '<div class="ps-alert-box" style="margin-bottom:16px;">'
        'SIMULATED IN v1.0 — Station network view is simulated until station metadata is provided. '
        'Station IDs from filenames (CS-1, STN-ALPHA, etc.) are stored in the database.'
        '</div>',
        unsafe_allow_html=True,
    )
    c_left, c_right = st.columns([2, 1], gap="large")

    with c_left:
        st.markdown('<div class="ps-section-title">Movement Map</div>', unsafe_allow_html=True)
        st.pyplot(render_movement_map(), width="stretch")

        st.markdown(
            '<div class="ps-error-box">'
            'STN-GAMMA reporting power critical. Immediate maintenance advised.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<div class="ps-section-title">Station Status</div>', unsafe_allow_html=True)
        for _, row in FIELD_STATIONS.iterrows():
            online = row["status"] == "Online"
            dot_clr = TEAL if online else AMBER
            stat_clr = TEAL if online else AMBER
            img_count = f"{row['images_today']:,}" if row["images_today"] > 0 else "0"
            st.markdown(
                f'<div class="ps-card" style="display:flex;align-items:center;gap:12px;'
                f'padding:10px 14px;">'
                f'<span class="ps-status-dot" style="background:{dot_clr};width:8px;height:8px;'
                f'flex-shrink:0;"></span>'
                f'<span style="font-family:Source Serif 4,Georgia,serif;font-size:14px;'
                f'font-weight:600;min-width:44px;">{row["station"]}</span>'
                f'<span style="font-family:Geist,sans-serif;font-size:12px;'
                f'color:{stat_clr};">{row["status"]}</span>'
                f'<span style="margin-left:auto;font-family:JetBrains Mono,monospace;'
                f'font-size:11px;color:#94A3B8;">{img_count} imgs</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with c_right:
        st.markdown('<div class="ps-section-title">Network Overview</div>', unsafe_allow_html=True)
        active = int((FIELD_STATIONS["status"]=="Online").sum())
        total_imgs = int(FIELD_STATIONS["images_today"].sum())
        st.metric("Active Stations", f"{active}/6")
        st.metric("Images Today", f"{total_imgs:,}")
        st.metric("Blank Filtered", "8,421")
        st.metric("AI Events", "43")

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<div class="ps-section-title">Station Details</div>', unsafe_allow_html=True)
        for sid, detail in STATION_DETAIL.items():
            warn = detail["warning"]
            clr = ERROR_CLR if warn else TEAL
            st.markdown(
                f'<div class="ps-card" style="padding:12px;margin-bottom:8px;'
                f'{"border-color:"+clr+"55;" if warn else ""}">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
                f'<span class="ps-data" style="color:{clr};">{detail["name"]}</span>'
                f'<span class="ps-mono" style="font-size:10px;color:{clr};">'
                f'{"!" if warn else ""} {detail["signal"]}%</span>'
                f'</div>'
                f'<div class="ps-label">Battery</div>'
                f'<div class="ps-battery-bar">'
                f'<div class="ps-battery-fill" style="width:{detail["battery"]}%;background:{clr};"></div>'
                f'</div>'
                f'<div style="display:flex;justify-content:space-between;">'
                f'<span class="ps-label">Last Detection</span>'
                f'<span class="ps-mono" style="font-size:11px;color:#FFFFFF;">{detail["last"]}</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<div class="ps-section-title">Recent Activity</div>', unsafe_allow_html=True)
        for label, time_str, clr in RECENT_ACTIVITY:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;padding:8px 0;'
                f'border-bottom:1px solid #1E292B33;">'
                f'<span class="ps-status-dot" style="background:{clr};width:6px;height:6px;border-radius:50%;"></span>'
                f'<span class="ps-mono" style="flex:1;font-size:12px;color:#FFFFFF;">{label}</span>'
                f'<span class="ps-mono" style="font-size:11px;color:#94A3B8;">{time_str}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.button("View Full Logs", width="stretch")


# =============================================================================
# PAGE 5 — INDIVIDUAL PROFILES
# =============================================================================
def page_individual_profiles():
    st.title("Individual Primate Profiles")
    st.markdown(
        '<div class="ps-alert-box" style="margin-bottom:16px;">'
        'SIMULATED IN v1.0 — Individual ID profiles (M03, F07, etc.) are simulated. '
        'SpeciesNet detects species, not individuals. Individual identification requires '
        'a re-identification model or manual labeling.'
        '</div>',
        unsafe_allow_html=True,
    )

    mid = st.selectbox("Select Individual", list(MONKEYS.keys()))
    m   = MONKEYS[mid]

    header_c1, header_c2 = st.columns([3, 2], gap="large")
    with header_c1:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">'
            f'<span class="ps-chip ps-chip-detected">DETECTED ACTIVE</span>'
            f'<span class="ps-mono" style="color:#94A3B8;letter-spacing:0.1em;">ID: {mid}</span>'
            f'</div>'
            f'<div style="font-family:Source Serif 4,Georgia,serif;font-size:32px;'
            f'font-weight:600;color:#FFFFFF;">{mid}</div>'
            f'<div class="ps-text" style="margin-top:8px;">'
            f'{m["sex"]} - {m["age"]} - {m["group"]} Group - Zone {m["zone"]}'
            f'</div>',
            unsafe_allow_html=True,
        )
    with header_c2:
        st.button("View Logs", width="stretch")
        st.button("Analyze Behavior", type="primary", width="stretch")

    st.markdown('<hr>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 1.1], gap="large")

    with c1:
        kv_rows = (
            _kv("Sex",        m["sex"]),
            _kv("Age",        m["age"]),
            _kv("Group",      m["group"]),
            _kv("Zone",       m["zone"]),
            _kv("Detections", f'<span style="color:#0adec8;font-weight:600;">{m["detections"]}</span>'),
            _kv("Associate",  f'<span style="color:#0adec8;font-weight:600;">{m["primary_assoc"]}</span>'),
        )
        profile_html = (
            f'<div class="ps-card" style="padding:20px 18px;">'
            f'<div style="font-family:Source Serif 4,Georgia,serif;font-size:28px;'
            f'font-weight:600;color:#0adec8;line-height:1;margin-bottom:16px;">{mid}</div>'
            f'<div style="display:flex;flex-direction:column;gap:6px;">'
            + "".join(kv_rows)
            + '</div></div>'
        )
        st.markdown(profile_html, unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="ps-section-title">Biometrics</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="ps-card" style="padding:16px;">'
            f'<div class="ps-table-row"><span class="ps-table-key ps-label">EST_AGE</span>'
            f'<span class="ps-mono" style="color:#FFFFFF;">{m["est_age"]}</span></div>'
            f'<div class="ps-table-row"><span class="ps-table-key ps-label">WEIGHT_EST</span>'
            f'<span class="ps-mono" style="color:#FFFFFF;">{m["weight"]}</span></div>'
            f'<div class="ps-table-row"><span class="ps-table-key ps-label">HEALTH_INDEX</span>'
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f'<div class="ps-battery-bar" style="width:60px;margin:0;">'
            f'<div class="ps-battery-fill" style="width:{m["health"]}%;background:#0adec8;"></div>'
            f'</div>'
            f'<span class="ps-mono" style="color:#0adec8;font-size:11px;">{m["health"]}/100</span>'
            f'</div></div>'
            f'<div style="margin-top:12px;"><span class="ps-label">CLINICAL_NOTES</span>'
            f'<p class="ps-text" style="margin-top:6px;font-size:12px;">{m["health_note"]}</p></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="ps-section-title">Weekly Movement</div>', unsafe_allow_html=True)
        days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        mv_df = pd.DataFrame({"Day":days, "Zone":m["movement"]})
        st.dataframe(mv_df, width="stretch", hide_index=True,
                     column_config={"Day":{"style":"uppercase"},"Zone":{"style":"mono"}})
        zone_counts = pd.Series(m["movement"]).value_counts()
        st.markdown(
            f'<div style="margin-top:8px;" class="ps-label">Primary zone &nbsp;'
            f'<span class="ps-data">{zone_counts.idxmax()}</span>'
            f'&nbsp;({zone_counts.max()}/7 days)</div>',
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown('<div class="ps-section-title">Troop Network</div>', unsafe_allow_html=True)
        for member in m["troop"]:
            status_cls = "ps-chip-conflict" if member["status"] == "conflict" else ""
            conflict_badge = (
                f'<span class="ps-chip {status_cls}" style="font-size:9px;">CONFLICT</span>'
                if member["status"] == "conflict" else ""
            )
            st.markdown(
                f'<div class="ps-troop-member">'
                f'<div>'
                f'<div class="ps-troop-id">{member["id"]}</div>'
                f'<div class="ps-troop-role">{member["role"]}</div>'
                f'</div>'
                f'{conflict_badge}'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="ps-section-title">Social Network</div>', unsafe_allow_html=True)
        st.pyplot(render_social_graph(highlight=mid), width="stretch")
        st.markdown(
            '<div class="ps-label" style="text-align:center;">'
            '<span style="color:#0adec8;">Male</span>'
            '&nbsp;&nbsp;&nbsp;<span style="color:#FFB84D;">Female</span>'
            '&nbsp;&nbsp;&nbsp;<span style="color:#94A3B8;">Infant</span>'
            '</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# PAGE 6 — RESEARCH INSIGHTS & EXPORT
# =============================================================================
def page_research_insights():
    mode = st.session_state.get("ps_mode", "Demo Simulation")
    if mode == "Real Inference" and _PROD_AVAILABLE:
        page_research_insights_real()
        return
    st.markdown(
        '<div class="ps-text" style="margin-bottom:24px;">'
        'Candidate observations surfaced from sample camera-trap data. '
        'These are not validated findings. Confidence ratings reflect model output, '
        'not scientific certainty. All entries require expert review before export.'
        '</div>',
        unsafe_allow_html=True,
    )

    for idx, ins in enumerate(INSIGHTS, 1):
        conf_cls   = "ps-chip-detected" if ins["conf"]=="High" else "ps-chip-review"
        conf_label = f"CONF: {ins['conf'].upper()}"
        ready_cls  = "ps-chip-ready" if ins["ready"] else "ps-chip-draft"
        ready_txt  = "EXPORT-READY DRAFT" if ins["ready"] else "DRAFT"

        st.markdown(
            f'<div class="ps-card" style="padding:16px 18px;margin-bottom:12px;">'
            f'<div style="display:flex;align-items:flex-start;justify-content:space-between;'
            f'gap:12px;flex-wrap:wrap;">'
            f'<div style="flex:1;">'
            f'<div style="font-family:Source Serif 4,Georgia,serif;font-size:17px;'
            f'font-weight:600;color:#FFFFFF;margin-bottom:6px;line-height:1.3;">'
            f'{idx}. {ins["title"]}</div>'
            f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">'
            f'<span class="ps-chip {conf_cls}">{conf_label}</span>'
            f'<span class="ps-chip {ready_cls}">{ready_txt}</span>'
            f'<span class="ps-data" style="font-size:10px;color:#94A3B8;">'
            f'{ins["period"]}</span>'
            f'</div>'
            f'</div>'
            f'</div>'
            f'<div style="margin-top:12px;" class="ps-text">{ins["content"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        b1, b2 = st.columns([1, 3])
        with b1:
            st.button(f"Export CSV - {ins['title'][:30]}...", key=f"exp_{idx}", width="stretch")
        with b2:
            st.button(f"Annotate - {ins['title'][:30]}...", key=f"ann_{idx}", width="stretch")

        st.markdown('<hr style="margin:8px 0;">', unsafe_allow_html=True)

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="ps-section-title">CSV Export Preview</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ps-text" style="margin-bottom:12px;font-size:12px;">'
        'Sample rows from the target export schema. '
        'Final export will include model version, threshold settings, and reviewer audit metadata.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(CSV_EXPORT_PREVIEW, width="stretch", hide_index=True)

    st.markdown(
        '<div class="ps-banner" style="font-size:11px;">'
        '<span class="material-symbols-outlined" style="font-size:14px;">info</span>'
        'Prototype preview only. No file is written to disk in v0.2. '
        'Persistent storage and real CSV generation are planned for the pilot.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="ps-section-title">Export Configuration</div>', unsafe_allow_html=True)

    export_c1, export_c2 = st.columns([2, 1], gap="large")

    with export_c1:
        st.markdown('<div class="ps-label" style="margin-bottom:12px;">Format Selection</div>', unsafe_allow_html=True)
        selected_format = st.session_state.get("export_format", "pdf")
        fc1, fc2, fc3 = st.columns(3, gap="medium")
        for i, (col, fmt) in enumerate(zip([fc1, fc2, fc3], EXPORT_FORMATS)):
            with col:
                is_selected = selected_format == fmt["id"]
                sel_cls = "ps-export-card-selected" if is_selected else ""
                st.markdown(
                    f'<div class="ps-export-card {sel_cls}">'
                    f'<span class="material-symbols-outlined" style="color:{"#0adec8" if is_selected else "#94A3B8"};'
                    f'font-size:28px;display:block;margin-bottom:8px;">{fmt["icon"]}</span>'
                    f'<div class="ps-export-name">{fmt["name"]}</div>'
                    f'<div class="ps-export-desc">{fmt["desc"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if st.button(f"Select {fmt['name']}", key=f"fmt_{fmt['id']}", width="stretch"):
                    st.session_state["export_format"] = fmt["id"]
                    st.rerun()

        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="ps-label" style="margin-bottom:12px;">Parameters</div>', unsafe_allow_html=True)
        dc1, dc2 = st.columns(2, gap="medium")
        with dc1:
            from datetime import date
            import pandas as pd

            default_start_date = (pd.Timestamp.today() - pd.DateOffset(months=1)).date()

            start_date = st.date_input(
                "Start Date",
                value=default_start_date
            )
        with dc2:
            st.date_input("End Date")
        st.selectbox("Target Species", SPECIES_OPTIONS)

    with export_c2:
        st.markdown('<div class="ps-label" style="margin-bottom:12px;">Schema Fields</div>', unsafe_allow_html=True)
        schema_fields = list(CSV_EXPORT_PREVIEW.columns)
        schema_html = '<div class="ps-card" style="padding:12px;">'
        for field in schema_fields:
            schema_html += (
                f'<div class="ps-row-mono" style="padding:4px 0;border-bottom:1px solid #1E292B33;'
                f'font-size:11px;color:#0adec8;">{field}</div>'
            )
        schema_html += '</div>'
        st.markdown(schema_html, unsafe_allow_html=True)

        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
        st.button("Generate Report", type="primary", width="stretch")
        st.button("Cancel", width="stretch")

        st.markdown(
            '<div class="ps-error-box" style="font-size:10px;margin-top:12px;">'
            'Prototype v0.2 | YOLOv8n Baseline | MegaDetector Integration Planned'
            '</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# MAIN
# =============================================================================
def main():
    st.set_page_config(
        page_title="PrimateScope AI",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    if _PROD_AVAILABLE:
        init_db()
    st.markdown(FONT_LINKS, unsafe_allow_html=True)
    st.markdown(CSS, unsafe_allow_html=True)

    page = render_sidebar()

    if page == "Overview":
        page_overview()
    elif page == "Research Dashboard":
        page_research_dashboard()
    elif page == "Camera-Trap Analysis Workbench":
        page_live_analysis()
    elif page == "Review Queue":
        st.title("Review Queue")
        tab_r, tab_d = st.tabs(["Real Review", "Demo Simulation"])
        with tab_r:
            if _PROD_AVAILABLE:
                page_review_queue_real()
            else:
                st.error("Production modules not loaded.")
        with tab_d:
            page_review_queue()
    elif page == "Behavior Intelligence":
        page_behavior_intelligence()
    elif page == "Field Stations":
        page_field_stations()
    elif page == "Individual Profiles":
        page_individual_profiles()
    elif page == "Research Insights & Export":
        st.title("Research Insights & Export")
        tab_r, tab_d = st.tabs(["Real Export", "Demo Simulation"])
        with tab_r:
            if _PROD_AVAILABLE:
                page_research_insights_real()
            else:
                st.error("Production modules not loaded.")
        with tab_d:
            page_research_insights()

    st.markdown(
        '<hr style="margin-top:24px;border-color:#1E292B;">'
        '<div style="display:flex;justify-content:center;align-items:center;gap:10px;'
        'flex-wrap:wrap;padding:8px 0;">'
        '<span style="background:rgba(255,184,77,0.08);border:1px solid rgba(255,184,77,0.4);'
        'border-radius:3px;padding:4px 12px;font-family:JetBrains Mono,monospace;'
        'font-size:10px;color:#FFB84D;letter-spacing:0.06em;text-transform:uppercase;">'
        'Production v1.0 | SpeciesNet + MegaDetector | SQLite | AI-Assisted — Review Required'
        '</span>'
        '</div>'
        '<div style="text-align:center;font-family:JetBrains Mono,monospace;'
        'font-size:9px;color:#94A3B8;letter-spacing:0.06em;text-transform:uppercase;'
        'margin-top:6px;">'
        'PrimateScope AI - Offline Field Intelligence Prototype - &copy; 2026'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
