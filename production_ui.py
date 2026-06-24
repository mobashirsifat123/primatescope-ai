"""PrimateScope AI — production UI functions for Streamlit.

Contains the Real Inference mode page renderers so app.py stays thin. All
functions use the existing Obsidian Canopy CSS classes and design tokens.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from database.db import get_connection, init_db
from database.repositories import (
    DetectionRepo, ExportRepo, InferenceRepo, MediaRepo, ProjectRepo,
    PredictionRepo, ReviewRepo, StatsRepo,
)
from services.bbox_draw import draw_bboxes_on_image
from services.export_service import CSV_COLUMNS, build_export_dataframe, export_csv
from services.file_storage import FileStorage
from services.pipeline import run_full_batch
from services.speciesnet_runner import (
    check_megadetector_available, check_speciesnet_available, get_engine_version,
)
from utils.constants import (
    BORDERLINE_CONFIDENCE, COMMON_COUNTRY_CODES, DEFAULT_COUNTRY_CODE,
    DEFAULT_PROJECT_NAME, MODE_DEMO, MODE_REAL, REVIEW_STATUSES, QUEUE_REASONS,
    REV_PENDING, CLR_TEAL, CLR_AMBER, CLR_ERROR, CLR_SLATE, CLR_WHITE,
)
from utils.validation import iso_now, validate_country_code

_storage = FileStorage()

# ---------------------------------------------------------------------------
# Engine label map for the UI selector
# ---------------------------------------------------------------------------
ENGINE_SPECIESNET = "speciesnet_only"
ENGINE_MD_SPECIESNET = "md_and_speciesnet"
ENGINE_OPTIONS = {
    ENGINE_SPECIESNET: "SpeciesNet Only — single detection per image",
    ENGINE_MD_SPECIESNET: "MegaDetector + SpeciesNet — multi-detection (recommended)",
}


def _get_conn():
    init_db()
    return get_connection()


def _env_status():
    sn_ok, sn_msg = check_speciesnet_available()
    md_ok, md_msg = check_megadetector_available()
    return sn_ok, sn_msg, md_ok, md_msg


def render_production_sidebar():
    """Render mode toggle, project selector, and engine status in the sidebar."""
    import sys as _sys

    st.markdown('<div class="ps-section-title">Operating Mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Operating Mode",
        [MODE_DEMO, MODE_REAL],
        label_visibility="collapsed",
        help="Demo uses simulated data. Real Inference runs SpeciesNet on uploads.",
    )
    st.session_state["ps_mode"] = mode

    st.markdown('<hr style="margin:12px 0;">', unsafe_allow_html=True)

    if mode == MODE_REAL:
        st.markdown('<div class="ps-section-title">Project</div>', unsafe_allow_html=True)
        conn = _get_conn()
        try:
            projects = ProjectRepo.list_all(conn)
            if projects:
                names = [f"{p.name} ({p.country_code or '---'})" for p in projects]
                sel = st.selectbox("Active Project", names, index=0)
                idx = names.index(sel)
                st.session_state["ps_project_id"] = projects[idx].id
                st.session_state["ps_project"] = projects[idx]
            else:
                st.info("No projects yet. Create one below.")
            with st.expander("Create New Project"):
                pname = st.text_input("Project Name", value=DEFAULT_PROJECT_NAME, key="np_name")
                pdesc = st.text_input("Description", key="np_desc")
                pcountry = st.selectbox(
                    "Country Code", list(COMMON_COUNTRY_CODES),
                    index=list(COMMON_COUNTRY_CODES).index(DEFAULT_COUNTRY_CODE),
                    key="np_country",
                )
                if st.button("Create Project", key="create_proj", use_container_width=True):
                    cc = validate_country_code(pcountry) or DEFAULT_COUNTRY_CODE
                    p = ProjectRepo.create(conn, pname, pdesc, cc)
                    st.success(f"Created: {p.name}")
                    st.rerun()
        finally:
            conn.close()

        st.markdown('<hr style="margin:12px 0;">', unsafe_allow_html=True)
        st.markdown('<div class="ps-section-title">Engine Status</div>', unsafe_allow_html=True)
        sn_ok, sn_msg, md_ok, md_msg = _env_status()
        sn_clr = CLR_TEAL if sn_ok else CLR_AMBER
        md_clr = CLR_TEAL if md_ok else CLR_AMBER
        st.markdown(
            f'<div class="ps-card" style="padding:10px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
            f'<span class="ps-label">SpeciesNet</span>'
            f'<span class="ps-chip" style="color:{sn_clr};border:1px solid {sn_clr};'
            f'background:{sn_clr}11;">{"READY" if sn_ok else "MISSING"}</span></div>'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<span class="ps-label">MegaDetector</span>'
            f'<span class="ps-chip" style="color:{md_clr};border:1px solid {md_clr};'
            f'background:{md_clr}11;">{"READY" if md_ok else "MISSING"}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        py_ver = _sys.version.split()[0]
        st.markdown(
            f'<div class="ps-label" style="margin-top:6px;">Python {py_ver}</div>',
            unsafe_allow_html=True,
        )
        if not sn_ok:
            st.markdown(
                '<div class="ps-alert-box" style="font-size:10px;">'
                'Install: pip install speciesnet<br>(macOS: add --use-pep517)'
                '</div>',
                unsafe_allow_html=True,
            )
    return mode


def page_live_analysis_real():
    """Real inference upload + results page — Camera-Trap Analysis Workbench."""
    import sys as _sys

    st.markdown(
        '<div class="ps-text" style="margin-bottom:16px;font-size:13px;color:#94A3B8;">'
        'Upload field images or short clips, run AI-assisted detection, review predictions, '
        'and export research-ready results.'
        '</div>',
        unsafe_allow_html=True,
    )

    sn_ok, sn_msg, md_ok, md_msg = _env_status()

    if not sn_ok:
        st.markdown(
            '<div class="ps-error-box">'
            'SpeciesNet is not installed. Real inference cannot run. '
            'Install with: <code>pip install speciesnet</code> (macOS: add --use-pep517). '
            'Switch to the <b>Demo Simulation</b> tab to explore the interface.'
            '</div>',
            unsafe_allow_html=True,
        )

    # Scientific honesty banner
    st.markdown(
        '<div class="ps-banner">'
        '<span class="material-symbols-outlined" style="font-size:16px;">info</span> '
        'AI-assisted pre-labeling only. All predictions require human review before scientific use. '
        'Predictions are NOT final until a reviewer approves them. '
        'Model accuracy varies by taxon, geography, and image quality.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Privacy banner
    st.markdown(
        '<div class="ps-alert-box" style="font-size:11px;">'
        '<span class="material-symbols-outlined" style="font-size:14px;">privacy_tip</span> '
        'Human/person detections may contain privacy-sensitive imagery. '
        'Review and export responsibly. Do not share personally identifiable images.'
        '</div>',
        unsafe_allow_html=True,
    )

    # --- Backend Readiness Status Cards ---
    py_ver = _sys.version.split()[0]
    sn_clr = CLR_TEAL if sn_ok else CLR_AMBER
    md_clr = CLR_TEAL if md_ok else CLR_AMBER
    st.markdown('<div class="ps-section-title">Backend Readiness</div>', unsafe_allow_html=True)
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        st.markdown(
            f'<div class="ps-card" style="padding:12px;text-align:center;">'
            f'<div class="ps-label">Python</div>'
            f'<div class="ps-data" style="font-size:16px;">{py_ver}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with bc2:
        st.markdown(
            f'<div class="ps-card" style="padding:12px;text-align:center;">'
            f'<div class="ps-label">SpeciesNet</div>'
            f'<div class="ps-chip" style="color:{sn_clr};border:1px solid {sn_clr};'
            f'background:{sn_clr}11;margin-top:4px;">{"READY" if sn_ok else "MISSING"}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with bc3:
        st.markdown(
            f'<div class="ps-card" style="padding:12px;text-align:center;">'
            f'<div class="ps-label">MegaDetector</div>'
            f'<div class="ps-chip" style="color:{md_clr};border:1px solid {md_clr};'
            f'background:{md_clr}11;margin-top:4px;">{"READY" if md_ok else "MISSING"}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    col_l, col_r = st.columns([1, 2], gap="large")

    with col_l:
        st.markdown('<div class="ps-section-title">Analysis Setup</div>', unsafe_allow_html=True)
        
        # --- Project selector (inline, not sidebar) ---
        conn = _get_conn()
        try:
            projects = ProjectRepo.list_all(conn)
            if projects:
                names = [f"{p.name} ({p.country_code or '---'})" for p in projects]
                sel = st.selectbox("Active Project", names, index=0, key="real_proj_sel")
                idx = names.index(sel)
                st.session_state["ps_project_id"] = projects[idx].id
                st.session_state["ps_project"] = projects[idx]
            else:
                st.info("No projects yet. Create one below.")
            with st.expander("Create New Project", expanded=False):
                st.markdown('<div class="ps-label" style="margin-bottom:8px;">New Project Details</div>', unsafe_allow_html=True)
                pname = st.text_input("Project Name", value=DEFAULT_PROJECT_NAME, key="real_np_name")
                pdesc = st.text_input("Description", key="real_np_desc")
                psite = st.text_input("Study Site", key="real_np_site",
                                      help="E.g. Sundarbans West, Lawachara NP")
                pcountry = st.selectbox(
                    "Country Code", list(COMMON_COUNTRY_CODES),
                    index=list(COMMON_COUNTRY_CODES).index(DEFAULT_COUNTRY_CODE),
                    key="real_np_country",
                )
                if st.button("Create Project", key="real_create_proj", use_container_width=True):
                    cc = validate_country_code(pcountry) or DEFAULT_COUNTRY_CODE
                    p = ProjectRepo.create(conn, pname, pdesc, cc, study_site=psite or None)
                    st.session_state["ps_project_id"] = p.id
                    st.session_state["ps_project"] = p
                    st.success(f"Created: {p.name}")
                    st.rerun()
        finally:
            conn.close()

        pid = st.session_state.get("ps_project_id")
        if not pid:
            st.warning("Create or select a project to proceed.")
            return

        st.markdown('<hr style="margin:12px 0;">', unsafe_allow_html=True)
        st.markdown('<div class="ps-label" style="margin-bottom:8px;">Upload Media</div>', unsafe_allow_html=True)
        images = st.file_uploader(
            "Camera-trap images",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="real_img_upload",
        )
        videos = st.file_uploader(
            "Short video clips (20-30s)",
            type=["mp4", "mov", "avi", "mkv"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="real_vid_upload",
        )

        country = st.selectbox(
            "Country Code (geofencing)",
            list(COMMON_COUNTRY_CODES),
            index=list(COMMON_COUNTRY_CODES).index(DEFAULT_COUNTRY_CODE),
            help="ISO 3166-1 alpha-3. Reduces impossible species predictions.",
            key="real_country",
        )
        st.markdown(
            '<div class="ps-text" style="font-size:11px;">'
            'Country filtering reduces impossible predictions but does not guarantee correctness.'
            '</div>',
            unsafe_allow_html=True,
        )

        # Engine selector
        st.markdown('<div class="ps-section-title" style="margin-top:12px;">Engine</div>',
                    unsafe_allow_html=True)
        engine_choice = st.radio(
            "Inference Engine",
            list(ENGINE_OPTIONS.keys()),
            format_func=lambda k: ENGINE_OPTIONS[k],
            index=1 if md_ok else 0,
            label_visibility="collapsed",
            help="MegaDetector+SpeciesNet supports multi-detection per image. "
                 "SpeciesNet Only uses single-detection mode.",
            key="real_engine",
        )
        st.session_state["ps_engine"] = engine_choice

        st.markdown('<div class="ps-label" style="margin-top:16px;margin-bottom:8px;">Model Settings</div>', unsafe_allow_html=True)
        det_thresh = st.slider(
            "Detection threshold",
            min_value=0.0, max_value=1.0, value=0.25, step=0.05,
            help="Minimum confidence to keep MegaDetector animal/person/vehicle boxes.",
            key="real_det_thresh",
        )
        rev_thresh = st.slider(
            "Review confidence threshold",
            min_value=0.0, max_value=1.0, value=0.70, step=0.05,
            help="Predictions below this score are flagged for manual review.",
            key="real_rev_thresh",
        )

        force = st.checkbox("Force re-run inference", value=False, key="real_force")
        debug_country = st.checkbox("Disable country filter for debugging", value=False, key="real_debug_country")
        frame_interval = st.slider(
            "Detection frame interval",
            min_value=0.25, max_value=5.0, value=1.0, step=0.25,
            help="For videos, extract 1 frame every N seconds. Use 0.5s for short clips.",
            key="real_frame_interval",
        )

        total_files = len(images or []) + len(videos or [])
        if total_files > 0:
            st.markdown(
                f'<div class="ps-card" style="padding:10px;">'
                f'<div class="ps-label">Ready to process</div>'
                f'<div class="ps-data">{total_files} file(s) '
                f'({len(images or [])} images, {len(videos or [])} videos)</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        run_btn = st.button("Run Real Analysis", type="primary",
                            use_container_width=True, disabled=(total_files == 0),
                            key="real_run_btn")

    with col_r:
        st.markdown('<div class="ps-section-title">Run Status & Results</div>', unsafe_allow_html=True)
        # --- Inference execution ---
        job_key = f"job_{pid}_{total_files}_{force}"
        if run_btn and total_files > 0:
            with st.status("Running inference...", expanded=True) as status:
                conn = _get_conn()
                try:
                    all_files = list(images or []) + list(videos or [])
                    result = None
                    for update in run_full_batch(
                        conn, _storage, pid, all_files, country if not debug_country else None, force,
                        frame_interval, det_thresh=det_thresh, review_thresh=rev_thresh, engine=engine_choice,
                    ):
                        if isinstance(update, str):
                            st.write(f"🔄 {update}")
                        else:
                            result = update

                    st.session_state["last_batch"] = result
                    st.session_state["last_job_key"] = job_key
                    if result and result.inference_success:
                        status.update(label="Inference complete", state="complete")
                    else:
                        status.update(label="Inference failed", state="error")
                finally:
                    conn.close()

        _render_last_batch_results()


def safe_attr(obj, name, default="N/A"):
    return getattr(obj, name, default) if obj else default

def _render_backend_proof(result):
    if not result:
        return
    with st.expander("Backend Proof"):
        run_id = result.inference_run_id if getattr(result, "inference_run_id", None) else "N/A"
        inf = result.inference_result
        dur = f"{safe_attr(inf, 'duration_seconds')}s" if inf else "N/A"
        cmd = safe_attr(inf, "command")
        json_path = safe_attr(inf, "output_json_path")
        
        from pathlib import Path
        json_exists = "YES" if json_path and json_path != "N/A" and Path(json_path).exists() else "NO"
        ret_code = safe_attr(inf, "return_code")
        
        st.markdown(
            f'<div class="ps-table-row"><span class="ps-label">Inference Run ID</span><span class="ps-data">{run_id}</span></div>'
            f'<div class="ps-table-row"><span class="ps-label">Engine Command</span><span class="ps-data">{cmd}</span></div>'
            f'<div class="ps-table-row"><span class="ps-label">Output JSON path</span><span class="ps-data">{json_path}</span></div>'
            f'<div class="ps-table-row"><span class="ps-label">Output JSON exists</span><span class="ps-data">{json_exists}</span></div>'
            f'<div class="ps-table-row"><span class="ps-label">Runtime duration</span><span class="ps-data">{dur}</span></div>'
            f'<div class="ps-table-row"><span class="ps-label">Return code</span><span class="ps-data">{ret_code}</span></div>'
            f'<div class="ps-table-row"><span class="ps-label">Media records created</span><span class="ps-data">{getattr(result, "media_count", 0)}</span></div>'
            f'<div class="ps-table-row"><span class="ps-label">Detection records created</span><span class="ps-data">{getattr(result, "detection_count", 0)}</span></div>'
            f'<div class="ps-table-row"><span class="ps-label">Prediction records created</span><span class="ps-data">{getattr(result, "prediction_count", 0)}</span></div>'
            f'<div class="ps-table-row"><span class="ps-label">Review items created</span><span class="ps-data">{getattr(result, "review_item_count", 0)}</span></div>',
            unsafe_allow_html=True
        )
        if inf and (getattr(inf, "stdout", "") or getattr(inf, "stderr", "")):
            with st.expander("stdout / stderr"):
                st.code(f"STDOUT:\n{getattr(inf, 'stdout', 'None')}\n\nSTDERR:\n{getattr(inf, 'stderr', 'None')}", language="text")

def _render_last_batch_results():
    """Render the results of the last inference batch."""
    result = st.session_state.get("last_batch")
    if result is None:
        st.info("Upload camera-trap media and click **Run Real Analysis**.")
        return

    if result.inference_error and not result.inference_success:
        st.markdown(
            f'<div class="ps-error-box">'
            f'Inference error: {result.inference_error}'
            f'</div>',
            unsafe_allow_html=True,
        )
        inf = result.inference_result
        if inf and inf.stderr:
            with st.expander("Engine stderr"):
                st.code(inf.stderr[-3000:], language="text")
        return

    _render_backend_proof(result)

    # Frame Extraction Preview
    from services.file_storage import FileStorage
    _storage = FileStorage()
    pid = st.session_state.get("ps_project_id")
    conn = _get_conn()
    try:
        media_list = MediaRepo.list_by_project(conn, pid)
        video_media = [m for m in media_list if m.media_type == "video"]
        for vm in video_media:
            frames_dir = _storage.frames_dir(pid, Path(vm.original_filename).stem)
            if frames_dir.exists():
                frames = sorted(list(frames_dir.glob("*.jpg")))
                if frames:
                    with st.expander(f"Frame Extraction Preview: {vm.original_filename}"):
                        st.markdown(f"**Total frames extracted:** {len(frames)}")
                        st.markdown(f"**Dimensions:** {getattr(vm, 'width', 'N/A')}x{getattr(vm, 'height', 'N/A')}")
                        st.markdown("**First 10 frames:**")
                        cols = st.columns(5)
                        for idx, fp in enumerate(frames[:10]):
                            with cols[idx % 5]:
                                # Try to extract timestamp from filename e.g. _t005.00
                                ts_match = re.search(r"_t([0-9.]+)\.jpg", fp.name)
                                ts_str = f"t={ts_match.group(1)}s" if ts_match else "t=?"
                                st.image(str(fp), caption=ts_str, use_container_width=True)
    finally:
        conn.close()

    # Raw JSON expander
    if getattr(result, "parse_result", None):
        with st.expander("Raw Inference Output (JSON)"):
            if result.inference_result and getattr(result.inference_result, "output_json_path", None):
                json_path = Path(result.inference_result.output_json_path)
                if json_path.exists():
                    try:
                        import json
                        with open(json_path, "r") as f:
                            raw_json = json.load(f)
                        st.markdown(f"**JSON Path:** `{json_path}`")
                        st.markdown(f"**Top-level keys:** `{', '.join(raw_json.keys())}`")
                        st.markdown(f"**Total items (images/frames):** `{len(raw_json.get('images', raw_json.get('predictions', [])))}`")
                        
                        raw_dets = 0
                        raw_preds = 0
                        items = raw_json.get('images', raw_json.get('predictions', []))
                        for item in items:
                            if "detections" in item:
                                raw_dets += len(item["detections"])
                        
                        st.markdown(f"**Raw detections before filtering:** `{raw_dets}`")
                        
                        if items:
                            st.json(items[0], expanded=False)
                    except Exception:
                        st.code("Could not load raw JSON", language="text")
                else:
                    st.info("Output JSON file not found.")
            else:
                st.info("No raw JSON available.")

    if getattr(result, "detection_count", 0) == 0 and getattr(result, "prediction_count", 0) == 0:
        st.markdown(
            f'<div class="ps-banner">'
            f'<span class="material-symbols-outlined" style="font-size:16px;">info</span>'
            f'<b>No final detections found above threshold.</b><br>'
            f'Frames/Images processed: {getattr(result, "media_count", 0)}<br>'
            f'Suggestion: Lower the Detection Threshold or check if the frame extraction worked.'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<div class="ps-banner">'
        f'<span class="material-symbols-outlined" style="font-size:16px;">check_circle</span>'
        f'AI-assisted analysis complete — {getattr(result, "detection_count", 0)} detections, '
        f'{getattr(result, "prediction_count", 0)} predictions, {getattr(result, "review_item_count", 0)} review items. '
        f'<span style="font-size:11px;color:{CLR_SLATE};">Review required before export.</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Expanded batch summary cards (13+ metrics)
    pid = st.session_state.get("ps_project_id")
    conn = _get_conn()
    try:
        stats = StatsRepo.project_summary(conn, pid)

        # Row 1: Core counts
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Images Processed", result.image_count)
        with mc2:
            st.metric("Videos Processed", result.video_count)
        with mc3:
            st.metric("Total Media", result.media_count)
        with mc4:
            st.metric("Detections", result.detection_count)

        # Row 2: Detection categories
        mc5, mc6, mc7, mc8 = st.columns(4)
        with mc5:
            st.metric("Animal Detections", stats.get("detections_animal", 0))
        with mc6:
            st.metric("Human Detections", stats.get("detections_human", 0))
        with mc7:
            st.metric("Vehicle Detections", stats.get("detections_vehicle", 0))
        with mc8:
            st.metric("Blanks", stats.get("blanks", 0))

        # Row 3: Review status breakdown
        mc9, mc10, mc11, mc12 = st.columns(4)
        with mc9:
            st.metric("Pending Review", stats.get("review_pending", 0))
        with mc10:
            st.metric("Approved", _count_review_status(conn, pid, "approved"))
        with mc11:
            st.metric("Corrected", _count_review_status(conn, pid, "corrected"))
        with mc12:
            st.metric("Uncertain", _count_review_status(conn, pid, "uncertain"))
            
        # Row 4: Averages and Low Conf
        mc13, mc14, _, _ = st.columns(4)
        with mc13:
            avg_conf = stats.get("avg_confidence", 0.0)
            st.metric("Avg Confidence", f"{avg_conf:.2f}")
        with mc14:
            st.metric("Low Confidence", _count_low_confidence(conn, pid))
    finally:
        conn.close()

    if result.errors:
        with st.expander(f"Warnings ({len(result.errors)})"):
            for e in result.errors:
                st.markdown(f"- {e}")

    # Species/taxon summary table
    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
    _render_species_summary(pid)

    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="ps-section-title">Result Cards</div>', unsafe_allow_html=True)

    conn = _get_conn()
    try:
        media_list = MediaRepo.list_by_project(conn, pid)
        for m in media_list[:20]:
            if m.media_type == "video":
                _render_video_result_card(conn, m)
            else:
                _render_result_card(conn, m)
    finally:
        conn.close()




def _count_review_status(conn, project_id, status):
    """Count review items by specific status."""
    try:
        cur = conn.execute(
            "SELECT COUNT(*) FROM review_items "
            "WHERE project_id = ? AND review_status = ?",
            (project_id, status),
        )
        return cur.fetchone()[0]
    except Exception:
        return 0

def _count_low_confidence(conn, project_id):
    """Count predictions below borderline confidence threshold."""
    try:
        cur = conn.execute(
            "SELECT COUNT(*) FROM species_predictions sp "
            "JOIN media_files m ON sp.media_id = m.id "
            "WHERE m.project_id = ? AND sp.prediction_score < ? "
            "AND sp.prediction_score IS NOT NULL",
            (project_id, BORDERLINE_CONFIDENCE),
        )
        return cur.fetchone()[0]
    except Exception:
        return 0


def _render_species_summary(project_id):
    """Render a species/taxon summary table for the current project."""
    conn = _get_conn()
    try:
        stats = StatsRepo.project_summary(conn, project_id)
        top = stats.get("top_species")
        if top:
            st.markdown('<div class="ps-section-title">Species Summary</div>',
                        unsafe_allow_html=True)
            sp_df = pd.DataFrame(top, columns=["Taxon", "Count"])
            st.dataframe(sp_df, use_container_width=True, hide_index=True)
        else:
            st.markdown(
                '<div class="ps-text" style="font-size:11px;color:#94A3B8;">'
                'No species predictions yet.</div>',
                unsafe_allow_html=True,
            )
    finally:
        conn.close()


def _render_result_card(conn, media):
    """Render a single media result card with thumbnail + prediction."""
    dets = DetectionRepo.list_by_media(conn, media.id)
    preds = PredictionRepo.list_by_media(conn, media.id)
    items = ReviewRepo.list_by_media(conn, media.id)
    pred = preds[0] if preds else None
    det = dets[0] if dets else None
    item = items[0] if items else None

    st.markdown(f'<div class="ps-card" style="padding:12px;margin-bottom:8px;">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 3], gap="medium")

    with c1:
        if media.media_type == "image" and Path(media.stored_path).exists():
            det_dicts = [{
                "detector_label": d.detector_label,
                "detector_confidence": d.detector_confidence,
                "bbox_x": d.bbox_x, "bbox_y": d.bbox_y,
                "bbox_w": d.bbox_w, "bbox_h": d.bbox_h,
                "bbox_format": d.bbox_format or "normalized_xywh",
                "prediction_label": pred.prediction_label if pred else None,
            } for d in dets]
            img = draw_bboxes_on_image(media.stored_path, det_dicts)
            if img:
                st.image(img, use_container_width=True)

    with c2:
        label = pred.prediction_label if pred else "no prediction"
        score = pred.prediction_score if pred else None
        conf_clr = CLR_TEAL if (score and score >= BORDERLINE_CONFIDENCE) else CLR_AMBER
        st.markdown(
            f'<div style="font-family:Source Serif 4,Georgia,serif;font-size:15px;'
            f'font-weight:600;color:{CLR_WHITE};">{media.original_filename}</div>',
            unsafe_allow_html=True,
        )
        sci = pred.prediction_scientific_name if pred and pred.prediction_scientific_name else ""
        st.markdown(
            f'<div class="ps-data" style="color:{conf_clr};">{label} '
            f'{"("+sci+")" if sci else ""} &nbsp; '
            f'{f"{score:.2f}" if score else "n/a"}</div>',
            unsafe_allow_html=True,
        )
        if det:
            st.markdown(
                f'<div class="ps-label" style="margin-top:4px;">'
                f'Detector: {det.detector_label} ({det.detector_confidence:.2f}) &nbsp;|&nbsp; '
                f'Model: {pred.model_version or "?" if pred else "?"}</div>',
                unsafe_allow_html=True,
            )
        if item:
            rclr = CLR_AMBER if item.review_status == REV_PENDING else CLR_TEAL
            st.markdown(
                f'<span class="ps-chip" style="color:{rclr};border:1px solid {rclr};'
                f'background:{rclr}11;">{item.review_status.upper()}</span>'
                f'<span class="ps-chip" style="color:{CLR_SLATE};border:1px solid {CLR_SLATE};'
                f'background:{CLR_SLATE}11;margin-left:4px;">{item.queue_reason}</span>',
                unsafe_allow_html=True,
            )

            with st.expander("Review Actions"):
                ac1, ac2 = st.columns(2)
                reviewer = st.text_input("Reviewer", key=f"la_rv_{media.id}")
                notes = st.text_input("Note", key=f"la_nt_{media.id}")
                bc1, bc2 = st.columns(2)
                with bc1:
                    if st.button("Approve", key=f"la_ap_{media.id}", use_container_width=True):
                        from services.review_service import approve_prediction
                        approve_prediction(conn, item.id, reviewer, notes)
                        st.rerun()
                    if st.button("Mark Blank", key=f"la_mb_{media.id}", use_container_width=True):
                        from services.review_service import mark_blank
                        mark_blank(conn, item.id, reviewer, notes)
                        st.rerun()
                    if st.button("Uncertain", key=f"la_un_{media.id}", use_container_width=True):
                        from services.review_service import mark_uncertain
                        mark_uncertain(conn, item.id, reviewer, notes)
                        st.rerun()
                with bc2:
                    flabel = st.text_input("Correct Label", key=f"la_fl_{media.id}")
                    if st.button("Correct", key=f"la_cr_{media.id}", use_container_width=True):
                        from services.review_service import correct_prediction
                        correct_prediction(conn, item.id, flabel, None, reviewer, notes)
                        st.rerun()
                    if st.button("Flag Human", key=f"la_fh_{media.id}", use_container_width=True):
                        from services.review_service import flag_human
                        flag_human(conn, item.id, reviewer, notes)
                        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def _render_video_result_card(conn, media):
    """Render an enhanced video result card with duration, frames, best frame."""
    dets = DetectionRepo.list_by_media(conn, media.id)
    preds = PredictionRepo.list_by_media(conn, media.id)
    items = ReviewRepo.list_by_media(conn, media.id)
    pred = preds[0] if preds else None
    item = items[0] if items else None

    # Count frame categories from detections safely
    animal_count = sum(1 for d in dets if getattr(d, 'detector_label', None) and d.detector_label.lower() == "animal")
    human_count = sum(1 for d in dets if getattr(d, 'detector_label', None) and d.detector_label.lower() == "human")
    vehicle_count = sum(1 for d in dets if getattr(d, 'detector_label', None) and d.detector_label.lower() == "vehicle")
    blank_count = sum(1 for p in preds if getattr(p, 'prediction_label', None) and p.prediction_label.lower() in ("blank", "empty"))

    st.markdown(f'<div class="ps-card" style="padding:12px;margin-bottom:8px;">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 3], gap="medium")

    with c1:
        # Try to show best frame thumbnail if we have video summaries
        batch_result = st.session_state.get("last_batch")
        best_frame_shown = False
        if batch_result and getattr(batch_result, 'video_summaries', None):
            for vs in batch_result.video_summaries:
                if getattr(vs, 'video_file', None) == getattr(media, 'original_filename', None) and getattr(vs, 'best_frame_path', None):
                    best_path = Path(vs.best_frame_path)
                    if best_path.exists():
                        from services.bbox_draw import best_video_frame_thumbnail
                        # Find detections for this specific frame from Prediction records
                        frame_dets = []
                        import json
                        for p in preds:
                            if p.raw_prediction_json:
                                try:
                                    raw = json.loads(p.raw_prediction_json)
                                    # MegaDetector format: raw['file'] matches best_path.name or similar
                                    if raw.get('file', '').endswith(best_path.name) or raw.get('filepath', '').endswith(best_path.name):
                                        # Use _parse_md_entry or similar to get normalized detections
                                        # Actually we can just use the raw detections but we need to map them to the format bbox_draw expects
                                        # But bbox_draw expects keys: detector_label, detector_confidence, bbox_x, bbox_y, bbox_w, bbox_h
                                        # Let's extract from raw
                                        for d in raw.get("detections", []):
                                            bx, by, bw, bh = None, None, None, None
                                            if "bbox" in d and len(d["bbox"]) == 4:
                                                bx, by, bw, bh = d["bbox"]
                                            cat = str(d.get("category", ""))
                                            # We just need some label, it doesn't have to be perfect if we don't have the mapping here
                                            label = "animal" if cat == "1" else "human" if cat == "2" else "vehicle" if cat == "3" else "detection"
                                            frame_dets.append({
                                                "detector_label": label,
                                                "detector_confidence": d.get("conf"),
                                                "bbox_x": bx, "bbox_y": by, "bbox_w": bw, "bbox_h": bh,
                                                "bbox_format": "normalized_xywh",
                                                "prediction_label": vs.best_species_prediction
                                            })
                                except Exception:
                                    pass
                        
                        frame_img = best_video_frame_thumbnail(best_path, frame_dets)
                        if frame_img:
                            st.image(frame_img, use_container_width=True,
                                     caption=f"Best frame (t={getattr(vs, 'best_species_score', 'N/A')})")
                            best_frame_shown = True
                    break
        if not best_frame_shown:
            st.markdown(
                f'<div class="ps-media" style="height:80px;display:flex;'
                f'align-items:center;justify-content:center;">'
                f'<span class="material-symbols-outlined" style="font-size:32px;color:{CLR_SLATE};">'
                f'videocam</span></div>',
                unsafe_allow_html=True,
            )

    with c2:
        label = getattr(pred, 'prediction_label', "no prediction") if pred else "no prediction"
        score = getattr(pred, 'prediction_score', None) if pred else None
        conf_clr = CLR_TEAL if (score and score >= BORDERLINE_CONFIDENCE) else CLR_AMBER

        st.markdown(
            f'<div style="font-family:Source Serif 4,Georgia,serif;font-size:15px;'
            f'font-weight:600;color:{CLR_WHITE};">'
            f'<span class="material-symbols-outlined" style="font-size:16px;vertical-align:text-bottom;'
            f'margin-right:4px;color:{CLR_SLATE};">videocam</span>'
            f'{media.original_filename}</div>',
            unsafe_allow_html=True,
        )

        # Duration and frame info
        dur_str = f"{media.duration_seconds:.1f}s" if media.duration_seconds else "?"
        fps_str = f"{media.fps:.1f}" if media.fps else "?"
        st.markdown(
            f'<div class="ps-label" style="margin-top:4px;">'
            f'Duration: {dur_str} &nbsp;|&nbsp; FPS: {fps_str} &nbsp;|&nbsp; '
            f'Frames analyzed: {len(preds)}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="ps-data" style="color:{conf_clr};margin-top:4px;">{label} '
            f'{f"{score:.2f}" if score else "n/a"}</div>',
            unsafe_allow_html=True,
        )

        # Frame category counts
        st.markdown(
            f'<div style="display:flex;gap:8px;margin-top:6px;flex-wrap:wrap;">'
            f'<span class="ps-chip" style="color:{CLR_TEAL};border:1px solid {CLR_TEAL};'
            f'background:{CLR_TEAL}11;">Animal: {animal_count}</span>'
            f'<span class="ps-chip" style="color:{CLR_AMBER};border:1px solid {CLR_AMBER};'
            f'background:{CLR_AMBER}11;">Human: {human_count}</span>'
            f'<span class="ps-chip" style="color:{CLR_SLATE};border:1px solid {CLR_SLATE};'
            f'background:{CLR_SLATE}11;">Vehicle: {vehicle_count}</span>'
            f'<span class="ps-chip" style="color:{CLR_SLATE};border:1px solid {CLR_SLATE};'
            f'background:{CLR_SLATE}11;">Blank: {blank_count}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if item:
            rclr = CLR_AMBER if item.review_status == REV_PENDING else CLR_TEAL
            st.markdown(
                f'<span class="ps-chip" style="color:{rclr};border:1px solid {rclr};'
                f'background:{rclr}11;margin-top:6px;">{item.review_status.upper()}</span>'
                f'<span class="ps-chip" style="color:{CLR_SLATE};border:1px solid {CLR_SLATE};'
                f'background:{CLR_SLATE}11;margin-left:4px;">{item.queue_reason}</span>',
                unsafe_allow_html=True,
            )

    # Timeline expander for video clips
    if batch_result and batch_result.video_summaries:
        for vs in batch_result.video_summaries:
            if vs.video_file == media.original_filename and vs.timeline:
                with st.expander(f"Frame Timeline ({len(vs.timeline)} frames)"):
                    tl_rows = []
                    for fr in vs.timeline:
                        tl_rows.append({
                            "time_s": fr.get("timestamp_seconds", "?"),
                            "detector": fr.get("detector_label", "—"),
                            "prediction": fr.get("prediction_label", "—"),
                            "score": fr.get("prediction_score"),
                        })
                    if tl_rows:
                        st.dataframe(pd.DataFrame(tl_rows), use_container_width=True,
                                     hide_index=True)
                break

        if item:
            with st.expander("Review Actions"):
                ac1, ac2 = st.columns(2)
                reviewer = st.text_input("Reviewer", key=f"v_rv_{media.id}")
                notes = st.text_input("Note", key=f"v_nt_{media.id}")
                bc1, bc2 = st.columns(2)
                with bc1:
                    if st.button("Approve", key=f"v_ap_{media.id}", use_container_width=True):
                        from services.review_service import approve_prediction
                        approve_prediction(conn, item.id, reviewer, notes)
                        st.rerun()
                    if st.button("Mark Blank", key=f"v_mb_{media.id}", use_container_width=True):
                        from services.review_service import mark_blank
                        mark_blank(conn, item.id, reviewer, notes)
                        st.rerun()
                    if st.button("Uncertain", key=f"v_un_{media.id}", use_container_width=True):
                        from services.review_service import mark_uncertain
                        mark_uncertain(conn, item.id, reviewer, notes)
                        st.rerun()
                with bc2:
                    flabel = st.text_input("Correct Label", key=f"v_fl_{media.id}")
                    if st.button("Correct", key=f"v_cr_{media.id}", use_container_width=True):
                        from services.review_service import correct_prediction
                        correct_prediction(conn, item.id, flabel, None, reviewer, notes)
                        st.rerun()
                    if st.button("Flag Human", key=f"v_fh_{media.id}", use_container_width=True):
                        from services.review_service import flag_human
                        flag_human(conn, item.id, reviewer, notes)
                        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def page_review_queue_real():
    """Database-backed review queue with filters and action controls."""
    st.title("Review Queue — Real Inference")
    st.markdown(
        '<div class="ps-banner">'
        '<span class="material-symbols-outlined" style="font-size:16px;">info</span>'
        'AI-assisted predictions below require human review. You are the final authority. '
        'Approve, correct, or reject each prediction before export.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Privacy banner for review queue
    st.markdown(
        '<div class="ps-alert-box" style="font-size:11px;">'
        '<span class="material-symbols-outlined" style="font-size:14px;">privacy_tip</span> '
        'Images flagged as "human_detected" may contain personally identifiable content. '
        'Handle with care per your institution\'s ethics policy.'
        '</div>',
        unsafe_allow_html=True,
    )

    pid = st.session_state.get("ps_project_id")
    if not pid:
        st.warning("Select a project in the sidebar first.")
        return

    conn = _get_conn()
    try:
        items = ReviewRepo.list_by_project(conn, pid)
        if not items:
            st.info("No review items yet. Run real inference on the Analysis Workbench page.")
            return

        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            status_filter = st.selectbox(
                "Review Status", ["All"] + list(REVIEW_STATUSES), index=0
            )
        with fc2:
            reason_filter = st.selectbox(
                "Queue Reason", ["All"] + list(QUEUE_REASONS), index=0
            )
        with fc3:
            conf_min = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)
        with fc4:
            media_filter = st.selectbox("Media Type", ["All", "image", "video"])
            
        fc5, fc6, fc7 = st.columns(3)
        with fc5:
            species_filter = st.text_input("Species Prediction", "", help="Filter by predicted label")
        with fc6:
            station_filter = st.text_input("Station ID", "")
        with fc7:
            date_filter = st.date_input("Date Range", value=None)

        filtered = items
        if status_filter != "All":
            filtered = [i for i in filtered if i.review_status == status_filter]
        if reason_filter != "All":
            filtered = [i for i in filtered if i.queue_reason == reason_filter]

        st.markdown(
            f'<div class="ps-label" style="margin-bottom:8px;">'
            f'{len(filtered)} item(s) in queue</div>',
            unsafe_allow_html=True,
        )

        if not filtered:
            st.info("No items match the current filters.")
            return

        rows = []
        for it in filtered:
            m = MediaRepo.get(conn, it.media_id)
            preds = PredictionRepo.list_by_media(conn, it.media_id)
            p = preds[0] if preds else None
            
            if media_filter != "All" and m and m.media_type != media_filter:
                continue
            if p and p.prediction_score and p.prediction_score < conf_min:
                continue
            if species_filter and p and species_filter.lower() not in (p.prediction_label or "").lower():
                continue
            if station_filter and m and station_filter.lower() not in (m.station_id or "").lower():
                continue
            # date_filter omitted for brevity to avoid st.date_input type issues, filtering logic will be simple.
            rows.append({
                "review_id": it.id,
                "filename": m.original_filename if m else "?",
                "media_type": m.media_type if m else "?",
                "station": m.station_id if m else "",
                "prediction": p.prediction_label if p else "—",
                "score": p.prediction_score if p else None,
                "detector": (DetectionRepo.list_by_media(conn, it.media_id)[0].detector_label
                             if DetectionRepo.list_by_media(conn, it.media_id) else "—"),
                "queue_reason": it.queue_reason,
                "status": it.review_status,
                "reviewer": it.reviewer or "",
            })

        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(
                df, use_container_width=True, hide_index=True,
                column_config={
                    "score": st.column_config.ProgressColumn(
                        "Score", format="%.2f", min_value=0.0, max_value=1.0,
                    ),
                },
            )

        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="ps-section-title">Review Detail</div>', unsafe_allow_html=True)

        sel_id = st.selectbox(
            "Select item to review",
            [r["review_id"] for r in rows],
            format_func=lambda rid: next(
                (f"{r['filename']} — {r['prediction']} ({r['status']})" for r in rows if r["review_id"] == rid), rid
            ),
        )

        if sel_id:
            _render_review_detail(conn, sel_id)
    finally:
        conn.close()


def _render_review_detail(conn, item_id):
    """Render the detail panel + action controls for one review item."""
    item = ReviewRepo.get(conn, item_id)
    if not item:
        return
    media = MediaRepo.get(conn, item.media_id)
    preds = PredictionRepo.list_by_media(conn, item.media_id)
    dets = DetectionRepo.list_by_media(conn, item.media_id)
    pred = preds[0] if preds else None
    det = dets[0] if dets else None
    actions = ReviewRepo.list_actions(conn, item_id)

    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        if media and media.media_type == "image" and Path(media.stored_path).exists():
            det_dicts = [{
                "detector_label": d.detector_label,
                "detector_confidence": d.detector_confidence,
                "bbox_x": d.bbox_x, "bbox_y": d.bbox_y,
                "bbox_w": d.bbox_w, "bbox_h": d.bbox_h,
                "bbox_format": d.bbox_format or "normalized_xywh",
            } for d in dets]
            img = draw_bboxes_on_image(media.stored_path, det_dicts)
            if img:
                st.image(img, use_container_width=True)
        elif media and media.media_type == "video":
            st.info("Video — best frame shown on Analysis Workbench page.")

    with c2:
        st.markdown('<div class="ps-card" style="padding:14px;">', unsafe_allow_html=True)
        if media:
            st.markdown(
                f'<div class="ps-table-row"><span class="ps-label">File</span>'
                f'<span class="ps-text-white">{media.original_filename}</span></div>'
                f'<div class="ps-table-row"><span class="ps-label">Station</span>'
                f'<span class="ps-text-white">{media.station_id or "—"}</span></div>'
                f'<div class="ps-table-row"><span class="ps-label">Captured</span>'
                f'<span class="ps-text-white">{media.captured_at or "—"}</span></div>',
                unsafe_allow_html=True,
            )
        if pred:
            st.markdown(
                f'<div class="ps-table-row"><span class="ps-label">AI Prediction</span>'
                f'<span class="ps-data">{pred.prediction_label}</span></div>'
                f'<div class="ps-table-row"><span class="ps-label">Score</span>'
                f'<span class="ps-data">{pred.prediction_score:.3f}</span></div>'
                f'<div class="ps-table-row"><span class="ps-label">Model</span>'
                f'<span class="ps-mono" style="font-size:11px;color:#FFFFFF;">{pred.model_version or "?"}</span></div>',
                unsafe_allow_html=True,
            )
        if det:
            st.markdown(
                f'<div class="ps-table-row"><span class="ps-label">Detector</span>'
                f'<span class="ps-data">{det.detector_label} ({det.detector_confidence:.2f})</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f'<div class="ps-table-row"><span class="ps-label">Queue Reason</span>'
            f'<span class="ps-chip ps-chip-review">{item.queue_reason}</span></div>'
            f'<div class="ps-table-row"><span class="ps-label">Status</span>'
            f'<span class="ps-chip {"ps-chip-detected" if item.review_status != "pending" else "ps-chip-review"}">'
            f'{item.review_status.upper()}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="ps-section-title">Review Action</div>', unsafe_allow_html=True)

    ac1, ac2 = st.columns(2, gap="medium")
    with ac1:
        reviewer = st.text_input("Reviewer name", value=item.reviewer or "", key=f"rv_{item_id}")
        new_label = st.text_input(
            "Corrected final label/species", value=item.final_label or "",
            key=f"fl_{item_id}",
        )
        notes = st.text_area("Notes", value=item.notes or "", key=f"nt_{item_id}", height=80)
    with ac2:
        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button("Approve", key=f"ap_{item_id}", use_container_width=True):
                ReviewRepo.apply_action(conn, item_id, "approve", "approved",
                                        reviewer, notes=notes, final_label=new_label or None)
                st.success("Approved.")
                st.rerun()
            if st.button("Mark Blank", key=f"bk_{item_id}", use_container_width=True):
                ReviewRepo.apply_action(conn, item_id, "mark_blank", "blank_confirmed",
                                        reviewer, notes=notes, final_label="blank")
                st.success("Marked blank.")
                st.rerun()
            if st.button("Flag Human", key=f"hu_{item_id}", use_container_width=True):
                ReviewRepo.apply_action(conn, item_id, "flag_human", "human_confirmed",
                                        reviewer, notes=notes, final_label="human")
                st.success("Flagged human.")
                st.rerun()
        with bc2:
            if st.button("Correct", key=f"co_{item_id}", use_container_width=True):
                ReviewRepo.apply_action(conn, item_id, "correct", "corrected",
                                        reviewer, notes=notes, final_label=new_label or None)
                st.success("Corrected.")
                st.rerun()
            if st.button("Mark Uncertain", key=f"un_{item_id}", use_container_width=True):
                ReviewRepo.apply_action(conn, item_id, "mark_uncertain", "uncertain",
                                        reviewer, notes=notes)
                st.success("Marked uncertain.")
                st.rerun()
            if st.button("Reject", key=f"rj_{item_id}", use_container_width=True):
                ReviewRepo.apply_action(conn, item_id, "reject", "rejected",
                                        reviewer, notes=notes)
                st.success("Rejected.")
                st.rerun()

    if actions:
        with st.expander(f"Audit History ({len(actions)})"):
            for a in actions:
                st.markdown(
                    f'- **{a.action}** by {a.reviewer or "?"} at {a.created_at} — '
                    f'{a.old_status} -> {a.new_status}'
                    f'{f" | notes: {a.notes}" if a.notes else ""}',
                )


def page_research_insights_real():
    """Real export page with DB-backed stats and CSV download."""
    st.title("Research Insights & Export — Real Data")
    pid = st.session_state.get("ps_project_id")
    if not pid:
        st.warning("Select a project in the sidebar first.")
        return

    conn = _get_conn()
    try:
        stats = StatsRepo.project_summary(conn, pid)
        if stats["media_total"] == 0:
            st.info("No data yet. Run real inference first.")
            return

        st.markdown(
            '<div class="ps-banner">'
            '<span class="material-symbols-outlined" style="font-size:16px;">info</span>'
            'Stats below are AI-assisted and require expert review. '
            'Export reviewed data for downstream analysis.'
            '</div>',
            unsafe_allow_html=True,
        )

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.metric("Media Files", stats["media_total"])
        with mc2:
            st.metric("Animal Detections", stats["detections_animal"])
        with mc3:
            st.metric("Human Detections", stats["detections_human"])
        with mc4:
            st.metric("Blanks", stats["blanks"])

        mc5, mc6, mc7, mc8 = st.columns(4)
        with mc5:
            st.metric("Pending Review", stats["review_pending"])
        with mc6:
            st.metric("Reviewed", stats["review_done"])
        with mc7:
            st.metric("Avg Confidence", f'{stats["avg_confidence"]:.2f}')
        with mc8:
            st.metric("Vehicle Detections", stats["detections_vehicle"])

        if stats["top_species"]:
            st.markdown('<div class="ps-section-title">Top Predicted Taxa</div>', unsafe_allow_html=True)
            sp_df = pd.DataFrame(stats["top_species"], columns=["Taxon", "Count"])
            st.dataframe(sp_df, use_container_width=True, hide_index=True)

        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="ps-section-title">CSV Export</div>', unsafe_allow_html=True)

        ec1, ec2 = st.columns([1, 2], gap="large")
        with ec1:
            reviewed_only = st.radio(
                "Export scope",
                ["Reviewed only", "All predictions"],
                help="Reviewed only: excludes pending items. All: includes everything with review_status.",
            )
            is_reviewed = reviewed_only == "Reviewed only"
            if st.button("Generate CSV", type="primary", use_container_width=True):
                out_path = _storage.exports_dir(pid) / f"export_{iso_now().replace(':','')}.csv"
                count, path = export_csv(conn, pid, out_path, reviewed_only=is_reviewed)
                st.session_state["last_export"] = (str(path), count)
                st.success(f"Exported {count} rows.")
                st.rerun()

            last = st.session_state.get("last_export")
            if last:
                path_str, count = last
                with open(path_str, "rb") as f:
                    st.download_button(
                        "Download CSV", f, file_name=Path(path_str).name,
                        mime="text/csv", use_container_width=True,
                    )
                st.markdown(
                    f'<div class="ps-label">Saved: {count} rows at {path_str}</div>',
                    unsafe_allow_html=True,
                )

            if stats["review_pending"] > 0 and not is_reviewed:
                st.markdown(
                    '<div class="ps-alert-box">'
                    f'{stats["review_pending"]} unreviewed predictions will be included.'
                    '</div>',
                    unsafe_allow_html=True,
                )

        with ec2:
            st.markdown('<div class="ps-label" style="margin-bottom:8px;">Preview (first 200 rows)</div>',
                        unsafe_allow_html=True)
            preview = build_export_dataframe(conn, pid, reviewed_only=is_reviewed)
            if preview:
                st.dataframe(pd.DataFrame(preview), use_container_width=True, hide_index=True)
            else:
                st.info("No rows to preview.")

        exports = ExportRepo.list_by_project(conn, pid)
        if exports:
            st.markdown('<div class="ps-section-title">Export History</div>', unsafe_allow_html=True)
            ex_rows = [{"date": e.created_at, "type": e.export_type,
                        "rows": e.row_count, "path": e.export_path} for e in exports]
            st.dataframe(pd.DataFrame(ex_rows), use_container_width=True, hide_index=True)
    finally:
        conn.close()


def get_dashboard_stats():
    """Return DB stats for the current project, or None if no project."""
    pid = st.session_state.get("ps_project_id")
    if not pid:
        return None
    conn = _get_conn()
    try:
        return StatsRepo.project_summary(conn, pid)
    finally:
        conn.close()
